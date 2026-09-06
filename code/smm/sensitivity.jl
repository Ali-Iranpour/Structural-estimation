#!/usr/bin/env julia
# =============================================================================
# sensitivity.jl -- how every estimated parameter responds to every target moment.
#
# The exercise adopted from the supplied paper: perturb ONE target moment, hold the others
# fixed, RE-ESTIMATE ALL NINE PARAMETERS JOINTLY, and plot each parameter against the
# perturbed target. Nine parameters x ten moments = 90 curves.
#
#     cd code/smm
#     # pilot -- a few hours, enough to validate the machinery end to end
#     julia +1.11 --project=../.. sensitivity.jl --pilot
#
#     # the full exercise. CENTRE IT ON A FITTED BASELINE, not on the calibration:
#     julia +1.11 --project=../.. sensitivity.jl \
#         --at ../../output/smm_runs/<fitted-run>/estimates.toml \
#         --moments all --offsets -2,-1,1,2 --restarts 8 --sobol 200 --procs 20
#
#     # resume a killed run (the directory is reused and the log appended)
#     julia +1.11 --project=../.. sensitivity.jl --resume ../../output/sensitivity/<dir>
#
# WHY A RE-ESTIMATION AND NOT A DERIVATIVE
# ----------------------------------------
# dtheta/dmhat from the implicit function theorem is a local linearisation of the same
# Jacobian jacobian.jl already saves. This exercise is different and more informative
# precisely because it is NOT that: it re-solves the whole minimisation at each perturbed
# target, so a curve that is flat, kinked, or runs into a bound shows those things, and a
# parameter that is only weakly separated shows up as an unstable or non-monotone response
# rather than as a clean slope. The cost is that it needs one full estimation per point.
#
# WHAT IS HELD FIXED, AND WHY IT HAS TO BE
# ----------------------------------------
# Everything except the one perturbed target: the moment SCALES (`moment_scale` is computed
# ONCE at the baseline targets and reused, so moving a target does not also move its own
# weight -- otherwise the curve would confound the two), the equal weights, the random
# draws (`seed`, `simN`), the grids, the bounds, the links, and the solver settings. The
# child solve is built once per process and reused, exactly as in run_smm.jl.
#
# PERTURBATION SIZE
# -----------------
# In STANDARD ERRORS OF THE SAMPLE MOMENT, taken from [moment_cov] in the targets file --
# not in per-observation SDs, which are 12-48x larger and are not the sampling variability
# of a mean. `--offsets -2,-1,1,2` therefore means "two standard errors below the estimated
# moment", which is a range the data cannot rule out. If [moment_cov] is missing the script
# refuses rather than silently substituting an SD.
#
# STARTING POINTS
# ---------------
# Each perturbed problem starts from the baseline solution and, along an offset ladder,
# from the neighbouring offset's solution -- the problems are continuous in the target, so
# a nearby optimum is a good start and this is where most of the cost is saved. That is
# also a risk: a chain of warm starts can walk a whole ladder into one basin. So every
# point ALSO gets an independent check from a fresh multistart (`--sobol`/`--restarts`), and
# the two are compared. `alt_gap` in the output is how much worse the warm start was; a
# large value there means the curve at that point is not trustworthy.
#
# WHAT THE OUTPUT IS NOT
# ----------------------
# Not identification evidence, not standard errors, and not a robustness claim. It is a
# picture of how the argmin moves when a target moves. Read it beside jacobian.jl (local
# separation) and standard_errors.jl (sampling uncertainty); none of the three substitutes
# for the others.
# =============================================================================

using Distributed, Printf, Dates, LinearAlgebra, TOML

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

function argstr(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("$flag needs a value")
    return ARGS[i + 1]
end
argval(flag, default) = (v = argstr(flag, nothing); v === nothing ? default : parse(Int, v))

const PILOT   = "--pilot" in ARGS
const ATFILE  = argstr("--at", "")
const RESUME  = argstr("--resume", "")
const RESUMING = !isempty(RESUME)
const NPROC   = argval("--procs", PILOT ? 20 : 20)
const GRID    = argval("--grid", PILOT ? 20 : 30)
const GRID_RPT= argval("--report-grid", 30)
const SIM_N   = argval("--simN", 2000)
const SEED    = argval("--seed", 1234)
const N_SOBOL = argval("--sobol",    PILOT ? 40 : 200)
const N_RESTART = argval("--restarts", PILOT ? 2 : 8)
const LOCAL_MAXEVAL = argval("--local-evals", PILOT ? 150 : 600)
const POLISH_MAXEVAL = argval("--polish-evals", PILOT ? 150 : 600)
const OFFSETS = parse.(Float64, split(argstr("--offsets", PILOT ? "-1,1" : "-2,-1,1,2"), ','))

const RUN_DIR = RESUMING ? RESUME :
    argstr("--outdir", joinpath(REPO, "output", "sensitivity",
        (PILOT ? "pilot_" : "full_") * Dates.format(now(), "yyyy-mm-dd_HHMMSS")))
mkpath(RUN_DIR)
const LOG = open(joinpath(RUN_DIR, "run.log"), RESUMING ? "a" : "w")
say(a...)    = (println(a...); println(LOG, a...); flush(LOG); flush(stdout))
sayf(f, a...) = (s = Printf.format(Printf.Format(f), a...);
                 print(s); print(LOG, s); flush(LOG); flush(stdout))
banner(s) = (say(); say("="^78); say(s); say("="^78))

banner("Target-moment sensitivity" * (PILOT ? "   [PILOT -- machinery validation, NOT the exercise]" : ""))
sayf("started  %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
sayf("outdir   %s\n", relpath(RUN_DIR, REPO))

if NPROC > 0 && (nprocs() - 1) < NPROC
    addprocs(NPROC - (nprocs() - 1); exeflags = `--project=$REPO`)
end
@everywhere using LinearAlgebra
@everywhere LinearAlgebra.BLAS.set_num_threads(1)

@everywhere begin
    using Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames
    using Statistics, Dates, ProgressMeter, Distributions, StatsBase
    using QuantEcon, FastGaussQuadrature, Parameters, Dierckx, TOML
    const REPO_ = normpath(joinpath(@__DIR__, "..", ".."))
    const SRC   = joinpath(REPO_, "code", "src")
    include(joinpath(SRC, "paths.jl"));       include(joinpath(SRC, "manifest.jl"))
    include(joinpath(SRC, "diagnostics.jl")); include(joinpath(SRC, "child_lifecycle.jl"))
    include(joinpath(SRC, "parent_family.jl")); include(joinpath(SRC, "tiktak.jl"))
    include(joinpath(REPO_, "code", "smm", "moments.jl"))
end

@everywhere const SENS_GRID = $GRID
@everywhere const SENS_SIMN = $SIM_N
@everywhere const SENS_SEED = $SEED
@everywhere const BASE_TARGETS = load_targets(joinpath(REPO_, "Input", "smm_targets_baseline.toml"))

# MOMENT SCALES ARE FROZEN AT THE BASELINE, once, here.
#
# `moment_scale` divides a level moment by its own target. If the scale were recomputed at
# the perturbed target, moving a target would change both the residual AND its weight, and
# the curve would show the sum of the two. Freezing it isolates the thing being varied.
@everywhere const FROZEN_SCALE =
    Dict(k => moment_scale(k, BASE_TARGETS[k].mean) for k in SMM_MOMENTS)

@everywhere function build_child_value()
    ch = ConSavLaborCollege_AR1(; Na = 30, Nk = 30, Nt = 5, rho = 1.5, psi_terminal = 0.0,
                                  kappa_terminal = 5.0, omega = 0.3, a_max = 100.0, w = 20.0,
                                  simN = 500, seed = 1234)
    redirect_stdout(devnull) do; redirect_stderr(devnull) do
        solve_model_work!(ch); solve_model_college!(ch)
        optimal_transfer_work!(ch); optimal_transfer_college!(ch)
    end end
    return terminal_value_spline(ch; s = 10.0)
end
print("solving the child value function on every process ... "); flush(stdout)
let t = time(); @everywhere const V_CHILD = build_child_value(); sayf("%.1fs\n", time() - t) end

# The objective at a PERTURBED target vector, with everything else frozen.
@everywhere function sens_objective(z, shifted::Dict{String,Float64}; Na, Nhc, simN, seed)
    kw = unpack(z)
    smm_feasible(kw) || return (_penalize!(:infeasible_sigma_2); SMM_PENALTY)
    try
        p = Parent_child_interaction_age_specific_AR1(; Na = Na, Nk = 2, Nhc = Nhc,
                                                        simN = simN, seed = seed, kw...)
        p.V_child_interp = V_CHILD
        redirect_stdout(devnull) do
            solve_model!(p; verbose = false); simulate_model!(p)
        end
        m = model_moments(p)
        v = simulation_violations(p)
        v.total > 0 && return (_penalize!(:invalid_sim); SMM_PENALTY)
        q = 0.0
        for k in SMM_MOMENTS
            mj = getfield(m, Symbol(k))
            isfinite(mj) || return SMM_PENALTY
            q += ((mj - shifted[k]) / FROZEN_SCALE[k])^2     # frozen scale, shifted target
        end
        return q
    catch err
        cause = _root_cause(err)
        if cause isa ErrorException || cause isa DomainError ||
           cause isa AssertionError || cause isa InexactError
            return (_penalize!(nameof(typeof(cause))); SMM_PENALTY)
        end
        rethrow()
    end
end

# ---- the moment standard errors the offsets are expressed in -----------------
const TGT_RAW = TOML.parsefile(joinpath(REPO, "Input", "smm_targets_baseline.toml"))
haskey(TGT_RAW, "moment_cov") || error("""
    Input/smm_targets_baseline.toml has no [moment_cov] block, so perturbations cannot be
    expressed in standard errors of the sample moments. Regenerate it:
        uv run --with pandas --with numpy python tools/make_smm_targets.py
    Do NOT substitute the per-observation `sd`: it is 12-48x larger and is not the sampling
    variability of a mean.""")
const MC   = TGT_RAW["moment_cov"]
const MSE  = Dict(String(n) => Float64(s) for (n, s) in zip(MC["names"], MC["se"]))

const WHICH = let w = argstr("--moments", PILOT ? "mean_h_p" : "all")
    w == "all" ? collect(SMM_MOMENTS) : String.(split(w, ','))
end
for k in WHICH
    k in SMM_MOMENTS || error("--moments: $k is not a targeted moment")
end

# ---- baseline point ----------------------------------------------------------
const Z0 = if isempty(ATFILE)
    say("centre    PARENT_DEFAULTS (the incumbent calibration)")
    say("          !! The exercise SHOULD be centred on a fitted baseline. Pass --at once one")
    say("             exists; a sensitivity curve around a calibration describes the")
    say("             calibration's neighbourhood, not the estimator's.")
    incumbent()
else
    raw = TOML.parsefile(ATFILE)
    pars = raw["parameters"]
    sayf("centre    %s\n", relpath(ATFILE, REPO))
    [to_search(Float64(pars[String(q.name)]), q) for q in SMM_PARAMS]
end

const LO, HI = search_bounds()
sayf("moments   %s\n", join(WHICH, ", "))
sayf("offsets   %s  (standard errors of the sample moment)\n", join(OFFSETS, ", "))
sayf("budget    %d sobol + %d restarts per point, grid %d, simN %d, seed %d\n",
     N_SOBOL, N_RESTART, GRID, SIM_N, SEED)
sayf("points    %d  (%d moments x %d offsets)\n", length(WHICH)*length(OFFSETS),
     length(WHICH), length(OFFSETS))

# ---- one estimation at one perturbed target ----------------------------------
const RESULTS_F = joinpath(RUN_DIR, "curves.csv")
if !RESUMING || !isfile(RESULTS_F)
    open(RESULTS_F, "w") do io
        println(io, "moment,offset_se,target,", join((String(q.name) for q in SMM_PARAMS), ","),
                ",Q,Q_warm,Q_alt,alt_gap,n_eval,ret,on_bound")
    end
end
# CHECKPOINT/RESUME is just "which rows are already in curves.csv". The rows are
# independent estimations, so a completed row never has to be redone and the file IS the
# checkpoint -- no separate state to keep consistent with it.
const DONE = Set{Tuple{String,Float64}}()
if RESUMING
    for (i, ln) in enumerate(eachline(RESULTS_F))
        i == 1 && continue
        f = split(ln, ',')
        push!(DONE, (String(f[1]), parse(Float64, f[2])))
    end
    sayf("resuming  %d of %d points already done\n", length(DONE), length(WHICH)*length(OFFSETS))
end

# The perturbed target vector lives in ONE mutable global per process, and `obj_w` is
# defined ONCE. Re-defining a method inside the sweep would invalidate the method cache on
# every point and pay a recompilation for each -- and `@everywhere` inside a loop body is
# how that mistake usually gets made. Only the CONTENTS of OBJ_SHIFT change.
@everywhere const OBJ_SHIFT = Dict{String,Float64}()
@everywhere function set_shift!(d::Dict{String,Float64})
    empty!(OBJ_SHIFT); merge!(OBJ_SHIFT, d); return nothing
end
@everywhere obj_w(z) = sens_objective(z, OBJ_SHIFT; Na = SENS_GRID, Nhc = SENS_GRID,
                                      simN = SENS_SIMN, seed = SENS_SEED)

function estimate_at(shifted::Dict{String,Float64}, z_start::Vector{Float64})
    for w in procs(); remotecall_fetch(set_shift!, w, shifted); end
    obj = obj_w
    # WARM: a short local polish from the neighbouring solution.
    wopt = Opt(:LN_NELDERMEAD, length(LO))
    lower_bounds!(wopt, LO); upper_bounds!(wopt, HI)
    ftol_rel!(wopt, 1e-4); ftol_abs!(wopt, 1e-10); xtol_rel!(wopt, 1e-7)
    maxeval!(wopt, LOCAL_MAXEVAL)
    nw = Ref(0)
    min_objective!(wopt, (z, g) -> (nw[] += 1; obj(z)))
    (q_warm, z_warm, ret_warm) = optimize(wopt, copy(z_start))

    # ALT: an independent multistart, so the warm chain cannot quietly decide the answer.
    alt = tiktak(obj_w, LO, HI; N = N_SOBOL, Nstar = N_RESTART,
                 extra_seeds = [copy(z_start)],
                 map_fn = nprocs() > 1 ? pmap : map,
                 local_maxeval = LOCAL_MAXEVAL, polish_maxeval = POLISH_MAXEVAL)

    if alt.f < q_warm
        return (z = alt.x, q = alt.f, q_warm = q_warm, q_alt = alt.f,
                n_eval = nw[] + alt.n_eval, ret = alt.polish_ret)
    else
        return (z = z_warm, q = q_warm, q_warm = q_warm, q_alt = alt.f,
                n_eval = nw[] + alt.n_eval, ret = ret_warm)
    end
end

on_bound(z) = any(i -> (p = (z[i]-LO[i])/(HI[i]-LO[i]); p < 0.02 || p > 0.98), eachindex(z))

# ---- the sweep ---------------------------------------------------------------
const T_START = time()
const N_POINTS = length(WHICH) * length(OFFSETS)
n_done = Ref(length(DONE))
for k in WHICH
    se_k = MSE[k]
    # Walk outward from the baseline in each direction, so each point starts from the
    # nearest one already solved rather than from the baseline every time.
    for dir in (1, -1)
        offs = sort(filter(o -> sign(o) == dir, OFFSETS); by = abs)
        z_prev = copy(Z0)
        for o in offs
            if (k, o) in DONE
                say("  skip $k $(o)se -- already in curves.csv"); continue
            end
            shifted = Dict{String,Float64}(m => BASE_TARGETS[m].mean for m in SMM_MOMENTS)
            shifted[k] = BASE_TARGETS[k].mean + o * se_k
            t0 = time()
            r = estimate_at(shifted, z_prev)
            z_prev = copy(r.z)
            est = unpack(r.z)
            n_done[] += 1
            open(RESULTS_F, "a") do io
                println(io, k, ",", o, ",", @sprintf("%.10g", shifted[k]), ",",
                        join((@sprintf("%.10g", getfield(est, q.name)) for q in SMM_PARAMS), ","),
                        ",", @sprintf("%.10g", r.q), ",", @sprintf("%.10g", r.q_warm),
                        ",", @sprintf("%.10g", r.q_alt), ",",
                        @sprintf("%.10g", r.q_warm - r.q_alt), ",",
                        r.n_eval, ",", r.ret, ",", on_bound(r.z))
            end
            sayf("  %-16s %+.0f se  target %8.4f  Q %10.5g  (warm %10.5g / alt %10.5g)  %5.1f min  [%d/%d]\n",
                 k, o, shifted[k], r.q, r.q_warm, r.q_alt, (time()-t0)/60, n_done[], N_POINTS)
            if r.q_warm - r.q_alt > 0.01 * max(abs(r.q_alt), 1e-12)
                sayf("     !! the warm start was %.3g worse than an independent multistart --\n",
                     r.q_warm - r.q_alt)
                say("        this point's curve value came from the multistart, and the warm")
                say("        chain is walking into a different basin. Treat neighbours with care.")
            end
        end
    end
end

# ---- summary -----------------------------------------------------------------
banner(@sprintf("Done in %.1f min -- %d points", (time()-T_START)/60, n_done[]))
sayf("curves written to %s\n", relpath(RESULTS_F, REPO))
open(joinpath(RUN_DIR, "meta.toml"), "w") do io
    println(io, "# GENERATED by code/smm/sensitivity.jl.")
    println(io, "generated  = \"", Dates.format(now(), "yyyy-mm-dd HH:MM"), "\"")
    println(io, "git_commit = \"", git_sha(), "\"")
    println(io, "pilot      = ", PILOT, "   # true = machinery validation, NOT the exercise")
    println(io, "centre     = \"", isempty(ATFILE) ? "PARENT_DEFAULTS" : relpath(ATFILE, REPO), "\"")
    println(io, "moments    = [", join(("\"$m\"" for m in WHICH), ", "), "]")
    println(io, "offsets_se = [", join(OFFSETS, ", "), "]")
    println(io, "grid       = ", GRID)
    println(io, "simN       = ", SIM_N)
    println(io, "seed       = ", SEED, "   # common random numbers, fixed across every point")
    println(io, "n_sobol    = ", N_SOBOL)
    println(io, "n_restarts = ", N_RESTART)
    println(io, "scales     = \"frozen at the baseline targets -- moving a target does not move its weight\"")
    println(io, "perturbation_unit = \"standard error of the sample moment, [moment_cov]\"")
end
say("""
Read curves.csv beside the caveats in this script's header. `alt_gap` > 0 means the warm
start was beaten by an independent multistart at that point; `on_bound` means the argmin
hit its box and the response there is censored, not a slope.""")
close(LOG)
