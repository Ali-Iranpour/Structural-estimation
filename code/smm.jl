#!/usr/bin/env julia
# =============================================================================
# smm.jl — simulated method of moments for the parent-child block.
#
#     cd code && julia --project=.. smm.jl [--quick] [--starts N] [--evals N]
#
# Matches 12 simulated moments to targets by minimising a weighted relative
# distance, then writes the estimated parameters and the moment fit to
# output/data/ and output/reports/.
#
# -----------------------------------------------------------------------------
# WHY IT IS STRUCTURED THIS WAY
#
# A naive SMM loop re-solves everything per parameter draw. Most of that work is
# wasted, because the parameters split into three tiers by what they touch:
#
#   Tier 0  the child's work and college value functions.  ~15 s.
#           Of the estimated set, only `college_cost` enters them. Cached on it
#           ROUNDED TO 0.1, so the whole search costs ~18 of these rather than
#           one per draw. `college_cost` has to be estimated because it sets
#           a_req[1] and is the only direct lever on the college share.
#
#   Tier 1  the child's transfer stage and terminal-value spline.  ~2 s.
#           Adds (omega, kappa_terminal), which enter obj_transfer_* and
#           terminal_value and nothing else. Cached on all three.
#
#   Tier 2  the parent solve, the parent simulation, and the family/college
#           simulation.  ~3 s.  Depends on everything, so it always runs.
#
# Without the tiering every draw would pay Tier 0 + Tier 1 + Tier 2 (~20 s); with
# it the typical draw pays only Tier 2, which is the ~6x that makes a few
# thousand evaluations affordable.
#
# COMMON RANDOM NUMBERS. Every model is built with the same `seed`, so the
# initial draws and the shock paths are identical across evaluations. Without
# this the objective is a step function of noise and no derivative-free method
# will converge.
#
# UNITS. Consumption and assets are reported in BOTH model units and display
# units (x10 = thousands of dollars, matching tables.jl's ASSET_RESCALE). The
# targets below are given in model units; the report shows both.
# =============================================================================

using Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames, Statistics, Dates
using ProgressMeter, Distributions, StatsBase, QuantEcon, FastGaussQuadrature
using Parameters, Dierckx
import TOML

const SRC = joinpath(@__DIR__, "src")
include(joinpath(SRC, "paths.jl"))
include(joinpath(SRC, "manifest.jl"))
include(joinpath(SRC, "diagnostics.jl"))
include(joinpath(SRC, "child_lifecycle.jl"))
include(joinpath(SRC, "parent_family.jl"))

const QUICK  = "--quick" in ARGS
const SEED   = 1234
argval(flag, dflt) = (i = findfirst(==(flag), ARGS);
                      i === nothing ? dflt : parse(Int, ARGS[i+1]))
const N_STARTS = argval("--starts", QUICK ? 2 : 6)
const N_EVALS  = argval("--evals",  QUICK ? 60 : 400)

# search grids (small); the winner is re-verified at production grids at the end
const C_NA, C_NK, C_NT = QUICK ? (20, 20, 4) : (30, 30, 6)
const P_NA, P_NHC, P_NP = QUICK ? (12, 12, 5) : (16, 16, 5)
const SIMN = QUICK ? 500 : 2000

# =============================================================================
# 1. Moments
# =============================================================================
"""
    Moment(name, target, weight, display_scale)

`target` is in MODEL units. `display_scale` converts to the units the user reads
(10 for money, 1 for time shares and rates), and is used only for reporting.
"""
struct Moment
    name::String
    target::Float64
    weight::Float64
    scale::Float64
end

# Targets are midpoints of the ranges asked for. Money targets are the requested
# display numbers divided by 10.
const MOMENTS = [
    Moment("e_p, t=1",              0.25, 1.0, 10.0)   # 0-5    display
    Moment("e_p, t=17",             0.75, 1.0, 10.0)   # 5-10
    Moment("tau_p, t=1",            0.475, 2.0, 1.0)   # 0.45-0.50
    Moment("tau_p, t=17",           0.275, 2.0, 1.0)   # 0.30-0.25
    Moment("i_c, first bargaining", 0.05, 1.0, 1.0)    # 0-0.1
    Moment("i_c, t=17",             0.15, 1.0, 1.0)    # 0.1-0.2
    Moment("h_p, t=1",              0.35, 1.5, 1.0)    # 0.3-0.4
    Moment("h_p, t=17",             0.35, 1.5, 1.0)    # 0.3-0.4
    Moment("assets, t=17",         25.0,  2.0, 10.0)   # ~250
    Moment("c_p, t=1",              3.75, 1.5, 10.0)   # 35-40
    Moment("c_p, t=17",             5.25, 1.5, 10.0)   # 50-55
    Moment("college share",         0.25, 3.0, 100.0)  # 20-30%
]

# =============================================================================
# 2. Parameter space
# =============================================================================
"""
    Par(name, lo, hi, x0, link)

`link` is `:id` for parameters already on an unbounded scale (the sigma
intercepts and slopes are log-elasticities), `:log` for strictly positive
levels, `:logit` for shares in (0,1). Searching on the linked scale keeps the
optimiser inside the economically meaningful region without hard rejections.
"""
struct Par
    name::String
    lo::Float64
    hi::Float64
    x0::Float64
    link::Symbol
end

const PARS = [
    Par("sigma_1_0",     -2.5,  0.6, -0.90, :id)     # parental-time elasticity, level
    Par("sigma_1_1",     -0.10, 0.02, -0.02, :id)    #                            slope
    Par("sigma_2_0",     -3.0,  0.3, -1.80, :id)     # education-spending elasticity
    Par("sigma_2_1",     -0.02, 0.10, 0.02, :id)
    Par("sigma_3_0",     -1.2, -0.15, -0.36, :id)    # self-productivity (kept flat)
    Par("sigma_4_0",     -5.5, -1.5, -4.50, :id)     # child-study elasticity
    # widened: i_c has to RISE with age, and it is fighting the child's leisure weight
    # (1-mu_t)*lambda_1, which also rises. sigma_4_1 needs room to win that race.
    Par("sigma_4_1",     -0.02, 0.30, 0.02, :id)
    Par("phi_2_0",        0.05, 4.0,  0.50, :log)    # leisure weight  -> h_p
    Par("phi_3_0",        0.02, 6.0,  1.00, :log)    # weight on log HC (= lambda_2_0)
    Par("R_0",            0.6,  4.0,  1.60, :log)    # HC technology TFP
    Par("omega",          0.02, 0.95, 0.30, :logit)  # altruism -> transfer -> college
    Par("kappa_terminal", 0.5, 40.0,  5.00, :log)    # taste for retained assets
    # college_cost sets a_req[1] and hence the college threshold, so it is the only lever
    # that moves the college share directly. It is a Tier-0 parameter -- it enters
    # solve_model_college! and compute_min_assets -- so the Tier-0 cache is keyed on it,
    # ROUNDED to 0.1. That bounds the cache at ~18 entries over the box instead of one
    # solve per draw, which is what makes it affordable to estimate.
    Par("college_cost",   0.30, 2.00, 1.20, :log)
    # The Euler equation fixes the consumption SLOPE: c_{t+1}/c_t = (beta*(1+r))^(1/rho).
    # BETA IS HELD AT 0.98 by instruction -- it should never approach 1 -- so `r` is the
    # only remaining lever on that slope, and it is estimated. At beta = 0.98:
    #
    #     r = 0.015  ->  c_p 37.5 falls to 35.4
    #     r = 0.020  ->  c_p 37.5 stays  37.3   (flat)
    #     r = 0.030  ->  c_p 37.5 rises to 41.4
    #     r = 0.050  ->  c_p 37.5 rises to 50.9   (hits the target)
    #
    # A LOWER r and a RISING consumption profile are therefore in direct conflict; the
    # search resolves it against the other 11 moments. `r` enters the child's lifecycle
    # budget too, so the Tier-0 cache is keyed on it as well, rounded to 0.005.
    Par("r",              0.010, 0.055, 0.025, :id)
]

const BETA_0 = 0.98   # fixed by instruction, not estimated

link_fwd(p::Par, v) = p.link === :id ? v : p.link === :log ? log(v) : log(v/(1-v))
link_inv(p::Par, x) = p.link === :id ? x : p.link === :log ? exp(x) : 1/(1+exp(-x))

const XLO = [link_fwd(p, p.lo) for p in PARS]
const XHI = [link_fwd(p, p.hi) for p in PARS]
"""
Start vector. `--warm` resumes from the previous run's estimates instead of the
constructor defaults, so a long search can be continued in stages.
"""
function start_vector()
    f = joinpath(datadir(), "smm_estimates.toml")
    if "--warm" in ARGS && isfile(f)
        got = TOML.parsefile(f)["parameters"]
        x = [link_fwd(p, get(got, p.name, p.x0)) for p in PARS]
        println("warm start from ", f)
        return clamp.(x, XLO, XHI)
    end
    return [link_fwd(p, p.x0) for p in PARS]
end
const X0 = start_vector()
to_theta(x) = [link_inv(PARS[i], x[i]) for i in eachindex(PARS)]

# =============================================================================
# 3. Cached model solves
# =============================================================================
# Tier 0: the child's work and college value functions, cached on college_cost rounded to
# 0.1. Nothing else in the estimated set enters them.
const TIER0 = Dict{NTuple{2,Float64},Any}()
ccost_key(cc) = round(cc, digits = 1)
r_key(rr)     = round(rr / 0.005) * 0.005
function tier0(cc, rr)
    key = (ccost_key(cc), r_key(rr))
    get!(TIER0, key) do
        t0 = time()
        ch = ConSavLaborCollege_AR1(simN = SIMN, Na = C_NA, Nk = C_NK, Nt = C_NT,
                                    sigma_eps = 0.5, rho = 1.5, a_max = 100.0, w = 20.0,
                                    seed = SEED, college_cost = key[1], r = key[2])
        solve_model_work!(ch); solve_model_college!(ch)
        @printf("    [tier0] college_cost %.1f, r %.3f solved in %.1fs (cache %d)\n",
                key[1], key[2], time() - t0, length(TIER0) + 1); flush(stdout)
        ch
    end
end

# Tier 1: transfer stage + terminal spline, cached on (college_cost, omega, kappa_terminal).
const TIER1 = Dict{NTuple{4,Float64},Any}()
function child_for(cc, rr, omega, psi, kap)
    key = (ccost_key(cc), r_key(rr), round(omega, digits = 5), round(kap, digits = 5))
    get!(TIER1, key) do
        base = tier0(cc, rr)
        ch = ConSavLaborCollege_AR1(simN = SIMN, Na = C_NA, Nk = C_NK, Nt = C_NT,
                                    sigma_eps = 0.5, rho = 1.5, a_max = 100.0, w = 20.0,
                                    seed = SEED, omega = omega, college_cost = ccost_key(cc),
                                    r = r_key(rr), psi_terminal = psi, kappa_terminal = kap)
        # share the Tier-0 solution; the transfer stage only reads these
        ch.sol_c_work = base.sol_c_work;       ch.sol_h_work = base.sol_h_work
        ch.sol_v_work = base.sol_v_work
        ch.sol_c_college = base.sol_c_college; ch.sol_h_college = base.sol_h_college
        ch.sol_v_college = base.sol_v_college
        optimal_transfer_work!(ch); optimal_transfer_college!(ch)
        (child = ch, V = terminal_value_spline(ch; s = 10.0))
    end
end

# =============================================================================
# 4. theta -> simulated moments
# =============================================================================
const PSI_TERM = 4.0   # held fixed; it only shifts both branches by a constant

"""
    simulate_moments(theta) -> Vector or nothing

`nothing` signals a solver failure, which the objective converts to a penalty
rather than an exception, so one bad corner cannot abort a whole search.
"""
function simulate_moments(theta)
    s10, s11, s20, s21, s30, s40, s41, p2, p3, R0, om, kap, cc, rr = theta
    t1 = child_for(cc, rr, om, PSI_TERM, kap)
    m = Parent_child_interaction_age_specific_AR1(
            Na = P_NA, Nk = 2, Nhc = P_NHC, Np = P_NP, w = 12.5, simN = SIMN, seed = SEED,
            sigma_1_0 = s10, sigma_1_1 = s11, sigma_2_0 = s20, sigma_2_1 = s21,
            sigma_3_0 = s30, sigma_3_1 = 0.0, sigma_4_0 = s40, sigma_4_1 = s41,
            phi_2_0 = p2, phi_3_0 = p3, lambda_2_0 = p3, R_0 = R0, R_1 = 0.0,
            beta_0 = BETA_0, r = rr)
    m.V_child_interp = t1.V
    ch = t1.child
    try
        redirect_stdout(devnull) do
            solve_model!(m; verbose = false)
            simulate_model!(m)
        end
        avg(x, t) = mean(filter(isfinite, x[:, t]))
        term = filter(isfinite, m.sim_a[:, m.T + 1])
        isempty(term) && return nothing
        pc = redirect_stdout(devnull) do
            ch.sim_a_init .= m.sim_a[:, m.T + 1]
            ch.sim_k_init .= m.sim_hc[:, m.T + 1]
            _, path, _ = simulate_model_family!(ch)
            path
        end
        return [avg(m.sim_e, 1), avg(m.sim_e, m.T),
                avg(m.sim_t, 1), avg(m.sim_t, m.T),
                avg(m.sim_i, T_CHILD_VOICE), avg(m.sim_i, m.T),
                avg(m.sim_h, 1), avg(m.sim_h, m.T),
                mean(term),
                avg(m.sim_c, 1), avg(m.sim_c, m.T),
                count(==(:college), pc) / length(pc)]
    catch
        return nothing
    end
end

# =============================================================================
# 5. Objective
# =============================================================================
const TARGETS = [mm.target for mm in MOMENTS]
const WEIGHTS = [mm.weight for mm in MOMENTS]
const SCALES  = [max(abs(mm.target), 0.05) for mm in MOMENTS]   # relative error
const PENALTY = 1.0e4

# ---- progress tracking -------------------------------------------------------
# The search is a few thousand model solves. Without this it is a silent hour.
const BUDGET   = N_STARTS * N_EVALS       # upper bound; starts can stop early
const N_EVAL   = Ref(0)
const N_FAIL   = Ref(0)
const BEST_Q   = Ref(Inf)
const CUR_START = Ref(1)
const T_ZERO   = Ref(time())
const REPORT_EVERY = 25

fmt_hms(sec) = sec < 0 || !isfinite(sec) ? "  --  " :
    (h = floor(Int, sec/3600); m = floor(Int, (sec % 3600)/60); ss = floor(Int, sec % 60);
     h > 0 ? @sprintf("%dh%02dm", h, m) : @sprintf("%2dm%02ds", m, ss))

function tick!(Q)
    N_EVAL[] += 1
    Q >= PENALTY && (N_FAIL[] += 1)
    Q < BEST_Q[] && (BEST_Q[] = Q)
    if N_EVAL[] % REPORT_EVERY == 0
        el   = time() - T_ZERO[]
        per  = el / N_EVAL[]
        left = max(BUDGET - N_EVAL[], 0) * per
        pct  = 100 * N_EVAL[] / BUDGET
        bar  = repeat("#", clamp(round(Int, pct/2.5), 0, 40))
        @printf("  [%-40s] %5.1f%%  %5d/%d  start %d  elapsed %s  ETA %s  best Q %9.4f  fails %d\n",
                bar, pct, N_EVAL[], BUDGET, CUR_START[], fmt_hms(el), fmt_hms(left),
                BEST_Q[], N_FAIL[])
        flush(stdout)
    end
end

"Weighted relative distance. Returns PENALTY on a failed solve."
function objective(x::Vector{Float64})
    ms = simulate_moments(to_theta(x))
    Q = (ms === nothing || any(!isfinite, ms)) ? PENALTY :
        sum(WEIGHTS .* ((ms .- TARGETS) ./ SCALES) .^ 2)
    tick!(Q)
    return Q
end

# =============================================================================
# 6. Search
# =============================================================================
"""
    estimate() -> (x, Q, evals)

BOBYQA from several starts. BOBYQA rather than Nelder-Mead because it builds a
quadratic model and handles box bounds natively, which matters when the boundary
is where the solver fails. Start 1 is the current calibration, so the search can
never return something worse than where it began.
"""
function estimate()
    rng = MersenneTwister(SEED)
    bestx, bestQ, total = copy(X0), Inf, 0
    for s in 1:N_STARTS
        CUR_START[] = s
        x0 = s == 1 ? copy(X0) : XLO .+ rand(rng, length(PARS)) .* (XHI .- XLO)
        opt = Opt(:LN_BOBYQA, length(PARS))
        lower_bounds!(opt, XLO); upper_bounds!(opt, XHI)
        maxeval!(opt, N_EVALS); xtol_rel!(opt, 1e-4); ftol_rel!(opt, 1e-6)
        nev = 0
        min_objective!(opt, (x, g) -> (nev += 1; objective(x)))
        local Q, xo
        try
            (Q, xo, _) = optimize(opt, x0)
        catch e
            @printf("  start %d: aborted (%s)\n", s, first(sprint(showerror, e), 60)); continue
        end
        total += nev
        @printf("  start %d: Q = %10.4f after %4d evals%s\n", s, Q, nev,
                Q < bestQ ? "   <- best" : "")
        Q < bestQ && (bestQ = Q; bestx = copy(xo))
    end
    return bestx, bestQ, total
end

# =============================================================================
# 7. Report
# =============================================================================
function report(x, Q)
    theta = to_theta(x)
    ms = simulate_moments(theta)
    println("\n", "="^86)
    println("MOMENT FIT      (display units: money x10, college share in %)")
    println("="^86)
    @printf("  %-26s %12s %12s %10s   %s\n", "moment", "target", "simulated", "rel.err", "")
    for (i, mm) in enumerate(MOMENTS)
        rel = (ms[i] - mm.target) / SCALES[i]
        flag = abs(rel) < 0.10 ? "ok" : abs(rel) < 0.30 ? "~" : "MISS"
        @printf("  %-26s %12.4f %12.4f %9.1f%%   %s\n",
                mm.name, mm.target * mm.scale, ms[i] * mm.scale, 100rel, flag)
    end
    @printf("\n  weighted objective Q = %.4f\n", Q)

    println("\n", "="^86)
    println("ESTIMATED PARAMETERS")
    println("="^86)
    @printf("  %-18s %12s %12s %10s\n", "parameter", "start", "estimate", "change")
    for (i, p) in enumerate(PARS)
        ch = p.link === :id ? theta[i] - p.x0 : 100 * (theta[i] / p.x0 - 1)
        @printf("  %-18s %12.4f %12.4f %9.2f%s\n", p.name, p.x0, theta[i], ch,
                p.link === :id ? "" : "%")
    end
    @printf("  %-18s %12.4f %12.4f %9s\n", "beta_0 (fixed)", BETA_0, BETA_0, "-")
    rr = theta[findfirst(p -> p.name == "r", PARS)]
    @printf("\n  Euler check: (beta(1+r))^(1/rho) = %.5f per period -> c_p x%.3f over 16 periods\n",
            (BETA_0*(1+rr))^(1/1.5), ((BETA_0*(1+rr))^(1/1.5))^16)
    return theta, ms
end

function save_results(theta, ms, Q)
    dir = datapath()
    open(joinpath(dir, "smm_estimates.toml"), "w") do io
        println(io, "# SMM estimates — written by code/smm.jl. Do not edit by hand.")
        println(io, "[run]")
        @printf(io, "timestamp  = \"%s\"\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        @printf(io, "git_commit = \"%s\"\n", git_sha())
        @printf(io, "objective  = %.6f\n", Q)
        @printf(io, "starts     = %d\nmax_evals  = %d\nseed       = %d\n",
                N_STARTS, N_EVALS, SEED)
        println(io, "\n[parameters]")
        for (i, p) in enumerate(PARS)
            @printf(io, "%-16s = %.6f\n", p.name, theta[i])
        end
        println(io, "\n[moments]")
        for (i, mm) in enumerate(MOMENTS)
            @printf(io, "# %-26s target %10.4f   simulated %10.4f\n",
                    mm.name, mm.target, ms[i])
        end
    end
    @printf("\nwrote %s\n", joinpath(dir, "smm_estimates.toml"))
end

# =============================================================================
banner(s) = (println("\n", "="^86); println(s); println("="^86))
t_start = time()
banner("SMM" * (QUICK ? "  [QUICK]" : "") *
       @sprintf("   %d parameters, %d moments, %d starts x %d evals",
                length(PARS), length(MOMENTS), N_STARTS, N_EVALS))
@printf("child grid %dx%dx%d   parent grid %dx%dx%d   simN %d   seed %d\n",
        C_NA, C_NK, C_NT, P_NA, 2, P_NHC, SIMN, SEED)

@printf("beta fixed at %.2f;  r is estimated (it is the only lever left on the consumption slope)\n", BETA_0)
@printf("\nbaseline objective at the current calibration: Q = %.4f\n", objective(X0))
println("\nprogress: one line per $(REPORT_EVERY) evaluations")
T_ZERO[] = time(); N_EVAL[] = 0; N_FAIL[] = 0; BEST_Q[] = Inf
flush(stdout)
banner("Searching")
bestx, bestQ, nev = estimate()
@printf("\n%d evaluations (%d failed), %d Tier-0 and %d Tier-1 solves cached, %.1f minutes\n",
        nev, N_FAIL[], length(TIER0), length(TIER1), (time() - t_start) / 60)
theta, ms = report(bestx, bestQ)
save_results(theta, ms, bestQ)
banner(@sprintf("DONE in %.1f minutes", (time() - t_start) / 60))
