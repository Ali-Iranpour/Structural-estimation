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
include(joinpath(SRC, "tiktak.jl"))

const QUICK  = "--quick" in ARGS
const SEED   = 1234
argval(flag, dflt) = (i = findfirst(==(flag), ARGS);
                      i === nothing ? dflt : parse(Int, ARGS[i+1]))
# TikTak budget.
#
# MEASURED: Nelder-Mead at ftol 1e-3 in 14 dimensions converges in ~295 function
# evaluations. At ~1.3 s per evaluation that is ~6.4 min per restart, so the cost
# is dominated by the LOCAL stage, not by the Sobol stage:
#
#     N = 1000 Sobol             1000 evals   ~22 min
#     N* = 50 restarts x 295    14750 evals   ~5.3 h      <- dominates
#
# N* = 25 with the local searches capped at 200 evaluations brings the total to
# ~6000 evaluations, about 2.2 h. N* = 25 is 2.5% of N, still inside the paper's
# 1-10% guidance. Raise --restarts for a longer, more reliable run.
const N_SOBOL    = argval("--sobol",    QUICK ? 120 : 1000)
const N_RESTARTS = argval("--restarts", QUICK ?  10 :   25)
const LOCAL_MAX  = argval("--localmax", QUICK ?  60 :  200)
# The polish is a single search but an expensive one -- measured at ~344 evaluations
# uncapped on a 3-restart smoke run, which at 1.3 s each is not free. Capped.
const POLISH_MAX = argval("--polishmax", QUICK ? 120 : 500)
# Local-stage batch width. Wider = faster but degrades TikTak toward plain
# multistart; see the measured table in src/tiktak.jl. Default follows the
# reference repo's #cores <= sqrt(#restarts).
const BATCH = argval("--batch", max(1, min(Threads.nthreads(), floor(Int, sqrt(N_RESTARTS)))))
# Threads are OFF by default and must stay off here: NLopt.jl is not safe under
# concurrent optimize calls, and this objective is thousands of them. Measured --
# `-t 8` with parallel=true kills the process silently (exit 0, no output); `-t 1`
# completes. Use Distributed/pmap for many cores; see the header of src/tiktak.jl.
const PARALLEL = "--parallel" in ARGS

# Two grid settings, and the distinction matters.
#
# SEARCH grids are deliberately coarse: at ~1.3 s per evaluation a few thousand
# evaluations is ~2 h, where the production grids cost ~5 s and the same budget
# would be ~7 h. That is affordable only because the earlier refinement study
# showed the state grids are already converged -- DOUBLING any of them moved the
# college share by at most 0.15pp, and parent Nhc 30 -> 60 moved it by 0.00.
#
# VERIFY grids are the production ones. The winner is re-evaluated there at the
# end, and BOTH moment vectors are reported, so any moment that is an artefact
# of the coarse search surface is visible rather than hidden.
struct Grids
    c_na::Int; c_nk::Int; c_nt::Int
    p_na::Int; p_nhc::Int; p_np::Int
    simN::Int
end
const SEARCH_GRIDS = Grids(20, 20, 4, 12, 12, 5, 500)
const VERIFY_GRIDS = QUICK ? SEARCH_GRIDS : Grids(30, 30, 6, 30, 30, 7, 5000)
const GRIDS = Ref(SEARCH_GRIDS)

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
# THREAD SAFETY AND MEMORY. Two things force the design here.
#
#   (1) `simulate_moments` MUTATES the child it is handed -- it writes sim_a_init,
#       sim_k_init, and then simulate_model_family! fills every sim_* array. A cache
#       that hands the same object to two threads is a data race. Tier 0 is only ever
#       READ (its solution arrays are shared by reference), so it can be shared under a
#       lock; Tier 1 hands out a mutable child, so it is THREAD-LOCAL.
#
#   (2) One cached child holds six 5-D solution arrays: 19.6 MB at the search grids,
#       66 MB at the production grids. An earlier unbounded run reached 283 Tier-1
#       entries, which is 5.5 GB on an 8 GB machine. Both caches are therefore CAPPED
#       with simple FIFO eviction.
const TIER0_CAP = 32                      # ~630 MB at search grids
const TIER0 = Dict{NTuple{2,Float64},Any}()
const TIER0_ORDER = NTuple{2,Float64}[]
const TIER0_LOCK = ReentrantLock()

ccost_key(cc) = round(cc, digits = 1)
r_key(rr)     = round(rr / 0.005) * 0.005

function tier0(cc, rr)
    key = (ccost_key(cc), r_key(rr))
    lock(TIER0_LOCK) do
        haskey(TIER0, key) && return TIER0[key]
        g = GRIDS[]
        ch = ConSavLaborCollege_AR1(simN = g.simN, Na = g.c_na, Nk = g.c_nk, Nt = g.c_nt,
                                    sigma_eps = 0.5, rho = 1.5, a_max = 100.0, w = 20.0,
                                    seed = SEED, college_cost = key[1], r = key[2])
        solve_model_work!(ch); solve_model_college!(ch)
        TIER0[key] = ch; push!(TIER0_ORDER, key)
        while length(TIER0_ORDER) > TIER0_CAP
            delete!(TIER0, popfirst!(TIER0_ORDER))
        end
        return ch
    end
end

# Tier 1: the transfer solution and the terminal spline. SHARED and lock-protected,
# and it stores ONLY immutable arrays -- never a child model. An earlier version kept
# per-thread caches of child objects indexed by Threads.threadid(); that is unsafe on
# Julia >= 1.9, where tasks MIGRATE between threads, so threadid() is not stable for the
# life of a task. Storing only immutable data removes the problem at the root: there is
# no per-thread state left to get wrong.
const TIER1 = Dict{NTuple{4,Float64},Any}()
const TIER1_ORDER = NTuple{4,Float64}[]
const TIER1_LOCK = ReentrantLock()
const TIER1_CAP_N = 64          # small: 4 transfer arrays + a spline, not a model

function transfer_for(cc, rr, omega, psi, kap)
    key = (ccost_key(cc), r_key(rr), round(omega, digits = 5), round(kap, digits = 5))
    lock(TIER1_LOCK) do
        haskey(TIER1, key) && return TIER1[key]
        base = tier0(cc, rr)
        g = GRIDS[]
        ch = ConSavLaborCollege_AR1(simN = g.simN, Na = g.c_na, Nk = g.c_nk, Nt = g.c_nt,
                                    sigma_eps = 0.5, rho = 1.5, a_max = 100.0, w = 20.0,
                                    seed = SEED, omega = omega, college_cost = key[1],
                                    r = key[2], psi_terminal = psi, kappa_terminal = kap)
        ch.sol_c_work = base.sol_c_work;       ch.sol_h_work = base.sol_h_work
        ch.sol_v_work = base.sol_v_work
        ch.sol_c_grad = base.sol_c_grad;       ch.sol_h_grad = base.sol_h_grad
        ch.sol_v_grad = base.sol_v_grad
        ch.sol_c_college = base.sol_c_college; ch.sol_h_college = base.sol_h_college
        ch.sol_v_college = base.sol_v_college
        optimal_transfer_work!(ch); optimal_transfer_college!(ch)
        val = (tr_c = ch.sol_tr_college, tr_w = ch.sol_tr_work,
               trv_c = ch.sol_tr_v_college, trv_w = ch.sol_tr_v_work,
               V = terminal_value_spline(ch; s = 10.0))
        TIER1[key] = val; push!(TIER1_ORDER, key)
        while length(TIER1_ORDER) > TIER1_CAP_N
            delete!(TIER1, popfirst!(TIER1_ORDER))
        end
        return val
    end
end

"""
    child_for(...) -> (child, V)

A FRESH child per call, sharing the cached solution and transfer arrays by reference
and owning only its own simulation arrays. Fresh because `simulate_model_family!`
writes into `sim_*`, so a shared object cannot be handed to concurrent callers.
"""
function child_for(cc, rr, omega, psi, kap)
    base = tier0(cc, rr)
    t1   = transfer_for(cc, rr, omega, psi, kap)
    g    = GRIDS[]
    ch = ConSavLaborCollege_AR1(simN = g.simN, Na = g.c_na, Nk = g.c_nk, Nt = g.c_nt,
                                sigma_eps = 0.5, rho = 1.5, a_max = 100.0, w = 20.0,
                                seed = SEED, omega = omega, college_cost = ccost_key(cc),
                                r = r_key(rr), psi_terminal = psi, kappa_terminal = kap)
    ch.sol_c_work = base.sol_c_work;       ch.sol_h_work = base.sol_h_work
    ch.sol_v_work = base.sol_v_work
    ch.sol_c_grad = base.sol_c_grad;       ch.sol_h_grad = base.sol_h_grad
    ch.sol_v_grad = base.sol_v_grad
    ch.sol_c_college = base.sol_c_college; ch.sol_h_college = base.sol_h_college
    ch.sol_v_college = base.sol_v_college
    ch.sol_tr_college = t1.tr_c;           ch.sol_tr_work = t1.tr_w
    ch.sol_tr_v_college = t1.trv_c;        ch.sol_tr_v_work = t1.trv_w
    return (child = ch, V = t1.V)
end

n_tier1() = length(TIER1)

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
    g = GRIDS[]
    m = Parent_child_interaction_age_specific_AR1(
            Na = g.p_na, Nk = 2, Nhc = g.p_nhc, Np = g.p_np, simN = g.simN, seed = SEED,
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
# TikTak has two stages with different cost profiles, so they get separate
# reporting: the Sobol stage is a known number of single evaluations, while a
# local search is a variable number, estimated from the restarts done so far.
const N_EVAL  = Ref(0)
const N_FAIL  = Ref(0)
const T_ZERO  = Ref(time())
const SOBOL_EVERY = 25

fmt_hms(sec) = (sec < 0 || !isfinite(sec)) ? "  --  " :
    (h = floor(Int, sec/3600); m = floor(Int, (sec % 3600)/60); ss = floor(Int, sec % 60);
     h > 0 ? @sprintf("%dh%02dm", h, m) : @sprintf("%2dm%02ds", m, ss))
bar(pct) = (n = clamp(round(Int, pct/2.5), 0, 40); repeat("#", n) * repeat("-", 40-n))

function on_sobol(i, ntot, fx, best)
    (i % SOBOL_EVERY == 0 || i == ntot) || return
    el = time() - T_ZERO[]; eta = el/i * (ntot - i)
    @printf("  sobol  [%s] %5.1f%%  %5d/%d   elapsed %s  ETA %s   best Q %9.4f   fails %d\n",
            bar(100i/ntot), 100i/ntot, i, ntot, fmt_hms(el), fmt_hms(eta), best, N_FAIL[])
    flush(stdout)
end

const T_LOCAL0 = Ref(0.0)
function on_local(j, nstar, theta, f_local, best)
    j == 1 && (T_LOCAL0[] = time())
    el = time() - T_LOCAL0[]; eta = j == 0 ? 0.0 : el/j * (nstar - j)
    @printf("  local  [%s] %5.1f%%  restart %3d/%d  theta %.3f  this %9.4f   best Q %9.4f   elapsed %s  ETA %s\n",
            bar(100j/nstar), 100j/nstar, j, nstar, theta, f_local, best,
            fmt_hms(el), fmt_hms(eta))
    flush(stdout)
end

"Weighted relative distance. Returns PENALTY on a failed solve."
function objective(x::Vector{Float64})
    ms = simulate_moments(to_theta(x))
    N_EVAL[] += 1
    if ms === nothing || any(!isfinite, ms)
        N_FAIL[] += 1
        return PENALTY
    end
    return sum(WEIGHTS .* ((ms .- TARGETS) ./ SCALES) .^ 2)
end


# =============================================================================
# 6. Search
# =============================================================================
"""
    estimate() -> TikTakResult

TikTak (Arnoud, Guvenen & Kleineberg 2022). See `src/tiktak.jl` for the
algorithm and for the decisions the paper leaves open.

The incumbent calibration is injected via `extra_seeds`. That is a deliberate
deviation from the published algorithm, which seeds only from Sobol points: it
guarantees the estimate is never worse than the calibration we started from,
which matters here because a uniform random point in 14 dimensions usually fails
to solve at all.
"""
function estimate()
    tiktak(objective, XLO, XHI;
           N = N_SOBOL, Nstar = N_RESTARTS,
           theta_p = 0.5, theta_lo = 0.1, theta_hi = 0.995,   # settled 2026-08-07
           local_alg = :LN_NELDERMEAD, local_tol = 1e-3,      # TikTak-nm3
           local_maxeval = LOCAL_MAX,
           polish_alg = :LN_BOBYQA, polish_tol = 1e-10, polish_maxeval = POLISH_MAX,
           extra_seeds = [copy(X0)],
           parallel = PARALLEL, batch = BATCH,
           on_sobol = on_sobol, on_local = on_local)
end

# =============================================================================
# 7. Report
# =============================================================================
qval(ms) = ms === nothing ? NaN : sum(WEIGHTS .* ((ms .- TARGETS) ./ SCALES) .^ 2)

function report(x, Q)
    theta = to_theta(x)
    ms = simulate_moments(theta)                       # at the search grids
    println("\nre-evaluating the winner at the production grids ...")
    use_grids!(VERIFY_GRIDS)
    mv = simulate_moments(theta)
    Qv = qval(mv)
    use_grids!(SEARCH_GRIDS)

    println("\n", "="^86)
    println("MOMENT FIT      (display units: money x10, college share in %)")
    println("="^86)
    @printf("  %-26s %10s | %10s %8s | %10s %8s  %s\n",
            "moment", "target", "search", "err", "PRODUCTION", "err", "")
    for (i, mm) in enumerate(MOMENTS)
        rel  = (ms[i] - mm.target) / SCALES[i]
        relv = mv === nothing ? NaN : (mv[i] - mm.target) / SCALES[i]
        flag = !isfinite(relv) ? "?" : abs(relv) < 0.10 ? "ok" : abs(relv) < 0.30 ? "~" : "MISS"
        @printf("  %-26s %10.4f | %10.4f %7.1f%% | %10.4f %7.1f%%  %s\n",
                mm.name, mm.target * mm.scale, ms[i] * mm.scale, 100rel,
                mv === nothing ? NaN : mv[i] * mm.scale, 100relv, flag)
    end
    @printf("\n  weighted objective Q:  search grids %.4f   PRODUCTION grids %.4f\n", Q, Qv)
    abs(Qv - Q) > 0.5 * max(Q, 1e-9) &&
        println("  WARNING: Q moves by more than 50% between grids -- the search surface is\n" *
                "           not a faithful stand-in for the production model here.")

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
    return theta, (mv === nothing ? ms : mv)
end

function save_results(theta, ms, Q)
    dir = datapath()
    open(joinpath(dir, "smm_estimates.toml"), "w") do io
        println(io, "# SMM estimates — written by code/smm.jl. Do not edit by hand.")
        println(io, "[run]")
        @printf(io, "timestamp  = \"%s\"\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        @printf(io, "git_commit = \"%s\"\n", git_sha())
        @printf(io, "objective  = %.6f\n", Q)
        @printf(io, "sobol_N    = %d\nrestarts   = %d\nseed       = %d\n",
                N_SOBOL, N_RESTARTS, SEED)
        @printf(io, "optimizer  = \"TikTak (Arnoud-Guvenen-Kleineberg 2022)\"\n")
        @printf(io, "beta_0     = %.4f   # fixed, not estimated\n", BETA_0)
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
"""
    use_grids!(g)

Switch grid setting and drop both caches, which hold solves at the old grid.
"""
function use_grids!(g::Grids)
    GRIDS[] = g
    lock(TIER0_LOCK) do
        empty!(TIER0); empty!(TIER0_ORDER)
    end
    lock(TIER1_LOCK) do
        empty!(TIER1); empty!(TIER1_ORDER)
    end
end

banner(s) = (println("\n", "="^86); println(s); println("="^86))
t_start = time()
banner("SMM by TikTak" * (QUICK ? "  [QUICK]" : "") *
       @sprintf("   %d parameters, %d moments,  N = %d Sobol,  N* = %d restarts",
                length(PARS), length(MOMENTS), N_SOBOL, N_RESTARTS))
@printf("search grids: child %dx%dx%d  parent %dx%dx%d  Np %d  simN %d\n",
        SEARCH_GRIDS.c_na, SEARCH_GRIDS.c_nk, SEARCH_GRIDS.c_nt,
        SEARCH_GRIDS.p_na, 2, SEARCH_GRIDS.p_nhc, SEARCH_GRIDS.p_np, SEARCH_GRIDS.simN)
@printf("verify grids: child %dx%dx%d  parent %dx%dx%d  Np %d  simN %d   seed %d\n",
        VERIFY_GRIDS.c_na, VERIFY_GRIDS.c_nk, VERIFY_GRIDS.c_nt,
        VERIFY_GRIDS.p_na, 2, VERIFY_GRIDS.p_nhc, VERIFY_GRIDS.p_np, VERIFY_GRIDS.simN, SEED)
@printf("beta fixed at %.2f;  r is estimated (it is the only lever left on the consumption slope)\n",
        BETA_0)
@printf("theta_j = clamp((j/N*)^%.2f, %.2f, %.3f);  local Nelder-Mead ftol 1e-3, max %d evals;  polish BOBYQA ftol 1e-10\n",
        0.5, 0.1, 0.995, LOCAL_MAX)
@printf("budget: %d Sobol + up to %d x %d local + %d polish = %d evaluations (~%.1f h at 1.3 s each)\n",
        N_SOBOL, N_RESTARTS, LOCAL_MAX, POLISH_MAX,
        N_SOBOL + N_RESTARTS*LOCAL_MAX + POLISH_MAX,
        1.3*(N_SOBOL + N_RESTARTS*LOCAL_MAX + POLISH_MAX)/3600)
@printf("the incumbent calibration is forced into the seed pool (documented deviation)\n")
@printf("threads: %d, parallel = %s%s\n", Threads.nthreads(), PARALLEL,
        PARALLEL ? @sprintf("  (local batch %d)", BATCH) :
                   "   -- NLopt is not thread-safe here; use Distributed for many cores")

@printf("\nbaseline objective at the current calibration: Q = %.4f\n", objective(X0))
T_ZERO[] = time(); N_EVAL[] = 0; N_FAIL[] = 0
flush(stdout)

banner("Searching")
res = estimate()
@printf("\n%d evaluations (%d failed), %d Tier-0 and %d Tier-1 solves cached, %.1f minutes\n",
        res.n_eval, N_FAIL[], length(TIER0), n_tier1(), (time() - t_start) / 60)
@printf("Q: best Sobol point %.4f  ->  after local stage %.4f  ->  after polish %.4f\n",
        res.f_sobol_best, res.f_prepolish, res.f)
@printf("restarts that improved the incumbent: %d of %d\n",
        count(t -> t.improved, res.trace), length(res.trace))

theta, ms = report(res.x, res.f)
save_results(theta, ms, res.f)
banner(@sprintf("DONE in %.1f minutes", (time() - t_start) / 60))
