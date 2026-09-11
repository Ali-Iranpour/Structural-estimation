#!/usr/bin/env julia
# =============================================================================
# profile_param.jl -- is a boundary a binding box, or a ridge?
#
#     cd code/smm
#     julia +1.11 --project=../.. profile_param.jl \
#         --param sigma_2_1 --values -0.05,-0.075,-0.10,-0.125,-0.15,-0.20 \
#         --at ../../output/smm_runs/2026-09-07_114138/candidate.toml \
#         --procs 20 --grid 30 --local-evals 800
#
# THE QUESTION THIS ANSWERS
# -------------------------
# A parameter sitting on its bound has two very different explanations, and the estimate
# itself cannot tell them apart:
#
#   (a) THE BOX BINDS. The objective genuinely keeps improving past the wall. Widen it and
#       the estimate comes to rest interior.
#   (b) THE OBJECTIVE IS FLAT there -- a ridge with some other parameter. The optimizer
#       slides to whatever wall it is given. Widening only moves the wall, and the answer
#       is a specification change or another moment, not a bigger box.
#
# sigma_2_1 has now pinned at -0.05 and again at -0.10, which is what (b) looks like from
# the outside. The discriminator is the SHAPE OF Q along the parameter, and that requires
# re-optimizing everything else at each point -- a single-coordinate slice through a
# correlated pair (sigma_2_0/sigma_2_1 cosine 0.869) measures the ridge, not the profile.
#
# WHAT IT DOES
# ------------
# For each value v of the chosen parameter: FIX it at v, remove it from the estimated set,
# and jointly re-optimize THE OTHER EIGHT from the supplied starting point -- plus an
# independent multistart check, because a warm chain along a ladder can walk itself into
# one basin. Everything else is held exactly as run_smm.jl holds it: moment scales, equal
# weights, common random numbers, grids, bounds, links, solver settings.
#
# HOW TO READ IT
# --------------
#   Q falls materially and then flattens INTERIOR  -> (a): widen the box to there.
#   Q is flat across the whole ladder              -> (b): the parameter is not identified
#                                                     in this direction. Do not widen.
#   Q falls monotonically to the last value tried  -> inconclusive; extend the ladder.
#
# `dQ_per_0.01` in the output is the local slope, which is the number to look at: a slope
# indistinguishable from zero IS the flat case, whatever the level of Q.
#
# This does not change the acceptance gate and does not promote anything.
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

const PARAM   = Symbol(argstr("--param", "sigma_2_1"))
const ATFILE  = argstr("--at", "")
const NPROC   = argval("--procs", 20)
const GRID    = argval("--grid", 30)
const SIM_N   = argval("--simN", 2000)
const SEED    = argval("--seed", 1234)
const N_SOBOL = argval("--sobol", 120)
const N_RESTART = argval("--restarts", 2)
const LOCAL_MAXEVAL  = argval("--local-evals", 800)
const POLISH_MAXEVAL = argval("--polish-evals", 400)
const VALUES = parse.(Float64, split(argstr("--values", "-0.05,-0.075,-0.10,-0.125,-0.15,-0.20"), ','))

const RUN_DIR = argstr("--outdir",
    joinpath(REPO, "output", "profiles",
             string(PARAM) * "_" * Dates.format(now(), "yyyy-mm-dd_HHMMSS")))
mkpath(RUN_DIR)
let link = joinpath(dirname(RUN_DIR), "latest")
    try; (islink(link) || ispath(link)) && rm(link; force = true); symlink(basename(RUN_DIR), link); catch; end
end
const LOG = open(joinpath(RUN_DIR, "run.log"), "w")
say(a...) = (println(a...); println(LOG, a...); flush(LOG); flush(stdout))
sayf(f, a...) = (s = Printf.format(Printf.Format(f), a...);
                 print(s); print(LOG, s); flush(LOG); flush(stdout))
banner(s) = (say(); say("="^78); say(s); say("="^78))

banner("Profile of $PARAM -- joint re-optimization of every other parameter")
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

@everywhere const PGRID = $GRID
@everywhere const PSIMN = $SIM_N
@everywhere const PSEED = $SEED
@everywhere const PPARAM = $(QuoteNode(PARAM))
# TARGETS COME FROM THE SNAPSHOT, NOT FROM Input/.
#
# Input/ holds source data only since c1849c6; every run freezes its own targets.toml
# beside its results. `smm_targets_file` resolves, in order: an explicit --targets, then
# the targets.toml sitting NEXT TO the --at estimate, then the newest snapshot.
#
# The --at path is the one that matters here: a profile around a candidate has to use the
# SAME targets that candidate was estimated against, or the profile is measuring a change
# of targets as though it were a change of parameter.
const TARGETS_FILE = smm_targets_file(argstr("--targets", ""); at = ATFILE)
sayf("targets  %s\n", relpath(TARGETS_FILE, REPO))
@everywhere const TARGETS = load_targets($TARGETS_FILE)

# The FREE set is every estimated parameter except the profiled one.
@everywhere const FREE = [q for q in SMM_PARAMS if q.name !== PPARAM]
length(FREE) == length(SMM_PARAMS) - 1 ||
    error("--param $PARAM is not one of the estimated parameters: " *
          join((String(q.name) for q in SMM_PARAMS), ", "))
sayf("profiling %s; %d parameters re-optimized at each point: %s\n",
     PARAM, length(FREE), join((String(q.name) for q in FREE), ", "))

@everywhere free_bounds() = ([to_search(q.lo, q) for q in FREE],
                             [to_search(q.hi, q) for q in FREE])
@everywhere function free_unpack(z, fixed_value)
    vals = Dict{Symbol,Float64}(PPARAM => fixed_value)
    for (i, q) in enumerate(FREE)
        vals[q.name] = clamp(from_search(z[i], q), q.lo, q.hi)
    end
    return NamedTuple{Tuple(keys(vals))}(Tuple(values(vals)))
end

# The objective with PARAM FIXED. Identical to smm_objective in every other respect --
# same scales, same equal weights, same seed, same feasibility and domain gates.
@everywhere function profile_objective(z, fixed_value)
    kw = free_unpack(z, fixed_value)
    smm_feasible(kw) || return SMM_PENALTY
    try
        # The SHARED pipeline. `fixed_value` may now be a CHILD parameter -- profiling
        # kappa_theta or kappa_terminal is exactly the use this tool was built for, since
        # both are suspected of sliding along a ridge -- so the child block has to be
        # rebuilt at each ladder rung rather than held fixed.
        r = run_pipeline(kw, TARGETS; Na = PGRID, Nk = 2, Nhc = PGRID,
                         simN = PSIMN, seed = PSEED,
                         child_grid = (Na = 30, Nk = 30, Nt = 5), demo_sim = false)
        m = model_moments(r, TARGETS)
        simulation_violations(r.parent).total > 0 && return SMM_PENALTY
        m.n_nonfinite > 0 && return SMM_PENALTY
        w = moment_weights(TARGETS)
        q = 0.0
        for (j, k) in enumerate(SMM_MOMENTS)
            mj = getfield(m, Symbol(k))
            isfinite(mj) || return SMM_PENALTY
            q += w[j] * (mj - TARGETS[k].mean)^2
        end
        return q
    catch err
        cause = _root_cause(err)
        is_model_failure(cause) && return SMM_PENALTY
        rethrow()
    end
end

# THE CHILD SOLVE IS NO LONGER A CONSTANT (2026-09-10). Four of the fourteen estimated
# parameters are child parameters, so a fixed `V_CHILD` would answer for a model this
# tool never solved -- and, for the Jacobian specifically, would produce four columns of
# exact zeros in precisely the directions that were just added. The shared pipeline in
# moments.jl rebuilds the child per evaluation and caches only the two stages that read
# none of the four; that cache is warmed here so the first evaluation is not paying for it.
print("warming the child solve on every process ... "); flush(stdout)
let t0 = time()
    @everywhere let cfg = child_config(TARGETS; Na = 30, Nk = 30, Nt = 5,
                                                simN = $SIM_N, seed = $SEED)
        child_base(cfg)
    end
    sayf("%.1fs\n", time() - t0)
end
check_psychic_centring(target_m_psychic(TARGETS))

# ---- the starting point ------------------------------------------------------
const START = begin
    base = Dict{Symbol,Float64}(q.name => param_default(q.name) for q in SMM_PARAMS)
    if !isempty(ATFILE)
        pars = TOML.parsefile(ATFILE)["parameters"]
        for (k, v) in pars
            sym = Symbol(k); haskey(base, sym) && (base[sym] = Float64(v))
        end
        sayf("centre   %s\n", relpath(ATFILE, REPO))
    else
        say("centre   PARENT_DEFAULTS")
    end
    base
end
const LO, HI = free_bounds()
sayf("values   %s\n", join(VALUES, ", "))
sayf("budget   %d sobol + %d restarts per point, local %d, polish %d, grid %d\n",
     N_SOBOL, N_RESTART, LOCAL_MAXEVAL, POLISH_MAXEVAL, GRID)

const RESULTS_F = joinpath(RUN_DIR, "profile.csv")
open(RESULTS_F, "w") do io
    println(io, "value,Q,Q_warm,Q_alt,", join((String(q.name) for q in FREE), ","),
            ",free_on_bound,ret,n_eval,minutes")
end

@everywhere OBJ_FIX = Ref(NaN)
@everywhere obj_p(z) = profile_objective(z, OBJ_FIX[])

# Walk the ladder from the centre outward, warm-starting from the neighbour, with an
# independent multistart at every point so the chain cannot quietly decide the answer.
#
# IN A FUNCTION, deliberately: a top-level `for` that assigns `z_prev` creates a NEW LOCAL
# each iteration instead of carrying the warm start forward -- the soft-scope rule
# CLAUDE.md warns about, and it bit here on the first run.
function sweep(values, z_start, lo, hi)
    rows = NamedTuple[]
    z_prev = copy(z_start)
    t_all = time()
    for v in values
        for w in procs(); remotecall_fetch(vv -> (Main.OBJ_FIX[] = vv; nothing), w, v); end
        t0 = time()
        opt = Opt(:LN_NELDERMEAD, length(lo))
        lower_bounds!(opt, lo); upper_bounds!(opt, hi)
        ftol_rel!(opt, 1e-4); ftol_abs!(opt, 1e-10); xtol_rel!(opt, 1e-7)
        maxeval!(opt, LOCAL_MAXEVAL)
        nw = Ref(0)
        min_objective!(opt, (z, g) -> (nw[] += 1; obj_p(z)))
        (q_warm, z_warm, ret_warm) = optimize(opt, copy(z_prev))

        alt = tiktak(obj_p, lo, hi; N = N_SOBOL, Nstar = N_RESTART,
                     extra_seeds = [copy(z_prev)],
                     map_fn = nprocs() > 1 ? pmap : map,
                     local_maxeval = LOCAL_MAXEVAL, polish_maxeval = POLISH_MAXEVAL)

        q_best, z_best, ret = alt.f < q_warm ? (alt.f, alt.x, alt.winner_ret) :
                                               (q_warm, z_warm, ret_warm)
        z_prev = copy(z_best)
        vals = free_unpack(z_best, v)
        ob = [String(FREE[i].name) for i in eachindex(z_best)
              if (pp = (z_best[i]-lo[i])/(hi[i]-lo[i]); pp < 0.02 || pp > 0.98)]
        push!(rows, (v = v, q = q_best, ob = ob, ret = ret))
        open(RESULTS_F, "a") do io
            println(io, v, ",", q_best, ",", q_warm, ",", alt.f, ",",
                    join((@sprintf("%.10g", getfield(vals, q.name)) for q in FREE), ","),
                    ",", isempty(ob) ? "" : join(ob, "|"), ",", ret, ",",
                    nw[] + alt.n_eval, ",", round((time()-t0)/60, digits = 1))
        end
        sayf("  %-10s = %+8.4f   Q %12.6g   (warm %10.4g / alt %10.4g)  %5.1f min  %s%s\n",
             PARAM, v, q_best, q_warm, alt.f, (time()-t0)/60, ret,
             isempty(ob) ? "" : "   OTHERS ON BOUND: " * join(ob, ", "))
    end
    return rows, time() - t_all
end

const Z_START = [to_search(START[q.name], q) for q in FREE]
const ROWS, T_TOTAL = sweep(VALUES, Z_START, LO, HI)

# ---- the verdict -------------------------------------------------------------
banner(@sprintf("Profile complete -- %.1f min", T_TOTAL/60))
say("value          Q          dQ per 0.01 of the parameter")
say("-"^62)
for (i, r) in enumerate(ROWS)
    slope = i == 1 ? NaN :
            (r.q - ROWS[i-1].q) / ((r.v - ROWS[i-1].v) / 0.01)
    sayf("%+8.4f  %12.6g   %s%s\n", r.v, r.q,
         isnan(slope) ? "  --" : @sprintf("%+11.6f", slope),
         isempty(r.ob) ? "" : "   (others pinned: " * join(r.ob, ", ") * ")")
end
let qs = [r.q for r in ROWS], span = maximum(qs) - minimum(qs), best = argmin(qs)
    sayf("\nQ ranges %.6g over the ladder; minimum at %s = %+.4f\n",
         span, PARAM, ROWS[best].v)
    rel = span / max(minimum(qs), eps())
    sayf("relative spread %.1f%% of the best Q\n", 100 * rel)
    say("")
    if rel < 0.05
        say("VERDICT: FLAT. Q barely moves along this parameter once the others re-optimize,")
        say("which is the ridge case. Widening the box will move the wall, not the answer --")
        say("the parameter is not identified in this direction by these moments. Fix it, or")
        say("add a moment that separates it from its correlated partner.")
    elseif best == length(ROWS)
        say("VERDICT: INCONCLUSIVE. Q is still falling at the last value tried, so the")
        say("ladder does not yet contain the minimum. Extend it before widening the box.")
    elseif best == 1
        say("VERDICT: the minimum is at the FIRST value tried -- extend the ladder the other way.")
    else
        sayf("VERDICT: BINDING BOX with an interior minimum near %+.4f. Widening to include\n",
             ROWS[best].v)
        say("that value is justified; re-run the estimation with the widened box, fresh.")
    end
end
say("\nRead this beside the caveats in the header. A profile is one ladder at one")
say("starting point under one grid; it is not identification and not inference.")

open(joinpath(RUN_DIR, "meta.toml"), "w") do io
    println(io, "# GENERATED by code/smm/profile_param.jl")
    println(io, "generated  = \"", Dates.format(now(), "yyyy-mm-dd HH:MM"), "\"")
    println(io, "git_commit = \"", git_sha(), "\"")
    println(io, "param      = \"", PARAM, "\"")
    println(io, "values     = [", join(VALUES, ", "), "]")
    println(io, "centre     = \"", isempty(ATFILE) ? "PARENT_DEFAULTS" : relpath(ATFILE, REPO), "\"")
    println(io, "reoptimized= [", join(("\"$(q.name)\"" for q in FREE), ", "), "]")
    println(io, "grid       = ", GRID)
    println(io, "simN       = ", SIM_N)
    println(io, "seed       = ", SEED)
    println(io, "n_sobol    = ", N_SOBOL)
    println(io, "n_restarts = ", N_RESTART)
    println(io, "local_maxeval  = ", LOCAL_MAXEVAL)
    println(io, "polish_maxeval = ", POLISH_MAXEVAL)
end
sayf("\nwrote %s\n", relpath(RESULTS_F, REPO))
close(LOG)
