#!/usr/bin/env julia
# =============================================================================
# grid_sensitivity.jl -- does the asset grid change the answer?
#
#     cd code/smm
#     julia +1.11 --project=../.. grid_sensitivity.jl \
#         --at ../../output/smm_runs/2026-09-07_114138/candidate.toml \
#         --a-max 100,150,200,300,400 --procs 20
#
# WHY
# ---
# Policies are interpolated with Flat() extrapolation, so a simulated household above the
# top asset node silently reuses the policy there. At the current a_max = 100 the candidate
# run had 2 households of 2,000 above the ceiling, peaking at 254 -- a thin tail, but "thin"
# is a claim about the POLICY error, not about the share, and nobody has measured it.
#
# THE TRADE-OFF THAT MAKES THIS NON-OBVIOUS
# -----------------------------------------
# CLAUDE.md caps the asset grid at 30 NODES by instruction. So raising a_max does not add
# resolution, it SPREADS THE SAME 30 NODES over a wider range: coverage improves and
# resolution falls, in the region where almost all the mass actually is. A wider grid is
# therefore not automatically better, and this script measures both sides rather than
# assuming the coverage side wins.
#
# The grid is focused (create_focused_grid puts 30% of nodes below a_focus = a_min + 3),
# so the loss is concentrated in the upper-middle range, not at the bottom.
#
# WHAT IT REPORTS, PER a_max
#   * every targeted moment and Q, at a FIXED parameter vector
#   * the change in each moment against the baseline a_max, in the units the objective
#     sees -- a moment that moves less than its own residual scale cannot matter
#   * coverage: households ever above, at t=1, at the handoff, and the simulated maximum
#   * terminal assets and the saving rate, which are what a_max is most likely to distort
#
# HOW TO CHOOSE
# -------------
# Pick the smallest a_max at which the targeted moments have STOPPED MOVING. That is a
# convergence criterion, not a coverage one: coverage always improves with a wider grid,
# so "no household off-grid" is not a stopping rule and would push a_max to infinity at
# the cost of resolution everywhere else.
#
# Then USE THAT GRID FOR EVERY COMPARISON. Q is not comparable across grids -- the
# objective surface itself moves -- so a candidate estimated at one a_max cannot be ranked
# against one estimated at another.
#
# --reoptimize re-runs a short local search at each a_max instead of evaluating at a fixed
# point. That is the stronger test (the argmin can move even when the moments at a fixed
# point do not) and costs a local optimization per grid; off by default.
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

const ATFILE   = argstr("--at", "")
const NPROC    = argval("--procs", 20)
const GRID     = argval("--grid", 30)          # Na = Nhc; the 30-node cap is an instruction
const SIM_N    = argval("--simN", 2000)
const SEED     = argval("--seed", 1234)
const REOPT    = "--reoptimize" in ARGS
const REOPT_EVALS = argval("--reopt-evals", 300)
const A_MAXES  = parse.(Float64, split(argstr("--a-max", "100,150,200,300,400"), ','))

const RUN_DIR = argstr("--outdir",
    joinpath(REPO, "output", "grid_sensitivity",
             Dates.format(now(), "yyyy-mm-dd_HHMMSS")))
mkpath(RUN_DIR)
let link = joinpath(dirname(RUN_DIR), "latest")
    try; (islink(link) || ispath(link)) && rm(link; force = true); symlink(basename(RUN_DIR), link); catch; end
end
const LOG = open(joinpath(RUN_DIR, "run.log"), "w")
say(a...) = (println(a...); println(LOG, a...); flush(LOG); flush(stdout))
sayf(f, a...) = (s = Printf.format(Printf.Format(f), a...);
                 print(s); print(LOG, s); flush(LOG); flush(stdout))
banner(s) = (say(); say("="^78); say(s); say("="^78))

banner("Asset-grid sensitivity" * (REOPT ? "   [with re-optimization]" : "   [at a fixed point]"))
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
@everywhere const GS_GRID, GS_SIMN, GS_SEED = $GRID, $SIM_N, $SEED

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

# NOTE ON THE CHILD SOLVE. It keeps a_max = 100 throughout, deliberately: the child block's
# own grid is a separate object and changing it would confound the parent-grid question
# with a child-grid one. What crosses the handoff is sim_a[:, T+1], and the child's
# terminal-value spline is evaluated by interpolation, so a parent household above 100 is
# already extrapolating there -- which is part of what this measures.

# The SHARED pipeline, at a given parent asset ceiling. The child block is rebuilt per
# draw: `kw` can now contain child parameters, and the child's terminal-value spline is
# what the parent's a_max interacts with, so holding the child fixed while varying a_max
# would measure the wrong thing.
@everywhere function solve_at(kw, a_max)
    # `a_max` REACHES THE PARENT CONSTRUCTOR. It did not between 2026-09-10 and this fix:
    # the refactor onto the shared pipeline dropped it, so every rung of the sweep solved
    # at the DEFAULT ceiling and the tool reported "the asset grid does not move the
    # moments" no matter what ceilings it was given. That is the one conclusion this tool
    # exists to reach, and it was reaching it vacuously.
    #
    # `parent_extra` is the pipeline's declared channel for non-estimated parent settings;
    # it errors if it would collide with an estimated parameter.
    return run_pipeline(kw, TARGETS; Na = GS_GRID, Nk = 2, Nhc = GS_GRID,
                        simN = GS_SIMN, seed = GS_SEED,
                        child_grid = (Na = 30, Nk = 30, Nt = 5),
                        parent_extra = (a_max = a_max,), demo_sim = false)
end

# RENAMED from `evaluate_at` (2026-09-10): moments.jl now exports an `evaluate_at` of its
# own, and two same-named functions in Main differing only in argument types is a trap --
# a call that should have been an error would silently dispatch to the other one.
@everywhere function gs_evaluate_at(kw, a_max)
    try
        r = solve_at(kw, a_max)
        p = r.parent
        m = model_moments(r, TARGETS); d = moment_diagnostics(p); v = simulation_violations(p)
        w = moment_weights(TARGETS)
        q = 0.0
        for (j, k) in enumerate(SMM_MOMENTS)
            mj = getfield(m, Symbol(k))
            isfinite(mj) || return nothing
            q += w[j] * (mj - TARGETS[k].mean)^2
        end
        # The ceiling actually used, read back off the solved model. Reported alongside
        # the requested one so a future refactor that drops `a_max` again shows up as a
        # visible mismatch in the output rather than as a null result.
        isapprox(p.a_max, a_max; rtol = 1e-12) || error(
            "grid sweep asked for a_max = $a_max but the parent was built with " *
            "$(p.a_max) -- the ceiling is not reaching the constructor.")
        return (q = q, m = m, d = d, viol = v.total, a_max_used = p.a_max,
                node_spacing = (a_max - p.a_grid[1]) / (GS_GRID - 1))
    catch err
        is_model_failure(_root_cause(err)) && return nothing
        rethrow()
    end
end

# ---- the point ---------------------------------------------------------------
const THETA = begin
    base = Dict{Symbol,Float64}(q.name => param_default(q.name) for q in SMM_PARAMS)
    if !isempty(ATFILE)
        for (k, v) in TOML.parsefile(ATFILE)["parameters"]
            sym = Symbol(k); haskey(base, sym) && (base[sym] = Float64(v))
        end
        sayf("point    %s\n", relpath(ATFILE, REPO))
    else
        say("point    PARENT_DEFAULTS")
    end
    NamedTuple{Tuple(keys(base))}(Tuple(values(base)))
end
sayf("a_max    %s   (Na = %d nodes throughout -- the cap is an instruction, see CLAUDE.md)\n",
     join(A_MAXES, ", "), GRID)

const RESULTS_F = joinpath(RUN_DIR, "grid_sensitivity.csv")
open(RESULTS_F, "w") do io
    println(io, "a_max,node_spacing,Q,", join(SMM_MOMENTS, ","),
            ",terminal_assets,saving_rate,hh_ever_above,hh_at_t1,hh_at_handoff,a_max_sim,invalid")
end

const OUT = []
for am in A_MAXES
    t0 = time()
    r = gs_evaluate_at(THETA, am)
    if r === nothing
        sayf("  a_max %6.0f   MODEL FAILURE -- skipped\n", am); continue
    end
    push!(OUT, (a_max = am, r...))
    d = r.d
    open(RESULTS_F, "a") do io
        println(io, am, ",", @sprintf("%.4f", r.node_spacing), ",", r.q, ",",
                join((@sprintf("%.10g", getfield(r.m, Symbol(k))) for k in SMM_MOMENTS), ","),
                ",", d.terminal_assets, ",", d.saving_rate, ",",
                d.a_hh_ever_above, ",", d.a_hh_above_t1, ",", d.a_hh_above_handoff, ",",
                d.a_max_sim, ",", r.viol)
    end
    sayf("  a_max %6.0f  spacing %6.3f   Q %10.6g   term.assets %7.3f   above: %d ever / %d handoff   max %6.1f   %4.1f min\n",
         am, r.node_spacing, r.q, d.terminal_assets,
         round(Int, d.a_hh_ever_above * d.n_sim), round(Int, d.a_hh_above_handoff * d.n_sim),
         d.a_max_sim, (time()-t0)/60)
end

# ---- how much did the moments actually move? ---------------------------------
banner("Movement against the baseline a_max = $(A_MAXES[1]), in units of each moment's own residual scale")
say("A moment that moves less than ~0.01 of its scale cannot change the objective")
say("meaningfully; that is the convergence criterion, not the off-grid share.")
say("")
const BASE = OUT[1]
sayf("%-16s %10s", "moment", "scale")
for o in OUT; sayf(" %11s", @sprintf("%.0f", o.a_max)); end
say("")
say("-"^(27 + 12*length(OUT)))
for k in SMM_MOMENTS
    sc = 1.0 / sqrt(moment_weights(TARGETS)[findfirst(==(k), collect(SMM_MOMENTS))])
    sayf("%-16s %10.4f", k, sc)
    for o in OUT
        Δ = (getfield(o.m, Symbol(k)) - getfield(BASE.m, Symbol(k))) / sc
        sayf(" %11.5f", Δ)
    end
    say("")
end
sayf("%-16s %10s", "Q", "")
for o in OUT; sayf(" %11.6g", o.q); end
say("")
sayf("%-16s %10s", "terminal assets", "")
for o in OUT; sayf(" %11.3f", o.d.terminal_assets); end
say("")

# IN RESIDUAL UNITS, i.e. DIVIDED by each moment's standard error -- the same scale the
# objective works in. This read `/ 1.0 / sqrt(w)` between 2026-09-10 and this fix, and
# since sqrt(w) = 1/se that MULTIPLIED by the standard error instead of dividing. For a
# moment that moved 0.001 with an SE of 0.01 it reported 0.00001 residual units instead of
# 0.1 -- four orders of magnitude too small, always below the 0.01 threshold below, and so
# always concluding that the asset grid does not matter.
let W = moment_weights(TARGETS),
    worst = maximum(maximum(abs(getfield(o.m, Symbol(k)) - getfield(BASE.m, Symbol(k))) *
                            sqrt(W[findfirst(==(k), collect(SMM_MOMENTS))]) for k in SMM_MOMENTS)
                    for o in OUT)
    sayf("\nlargest moment movement across the whole sweep: %.5f residual units\n", worst)
    say(worst < 0.01 ?
        "The asset grid does NOT move the targeted moments. a_max = $(A_MAXES[1]) is adequate;\nkeep it and spend the nodes on resolution." :
        "The asset grid DOES move the moments. Choose the smallest a_max at which they stop\nmoving, and use that grid for every comparison from here on -- Q is not comparable\nacross grids.")
end

if REOPT
    banner("Re-optimization at each a_max -- does the ARGMIN move, not just the moments?")
    say("A fixed-point sweep can miss a grid effect that the optimizer absorbs into the")
    say("parameters. This is the stronger test.")
    lo, hi = search_bounds()
    z0 = [to_search(getfield(THETA, q.name), q) for q in SMM_PARAMS]
    open(joinpath(RUN_DIR, "reoptimized.csv"), "w") do io
        println(io, "a_max,Q,", join((String(q.name) for q in SMM_PARAMS), ","), ",ret,n_eval")
    end
    for am in A_MAXES
        # REMOVED: `remotecall_fetch(() -> (Main.GS_AMAX = $am; nothing), w)`.
        #
        # A leftover from an older design that pushed the ceiling to the workers through a
        # global. `$am` is an interpolation outside a quote, which is a LOWERING error, and
        # Julia lowers a top-level `if` block whether or not the branch is taken -- so this
        # file aborted at this line on EVERY run, `--reoptimize` or not, and had done since
        # before the 2026-09-10 refactor. Nothing reads `GS_AMAX`; the ceiling now travels
        # as an argument to `gs_evaluate_at` and reaches the parent through `parent_extra`.
        obj = z -> (r = gs_evaluate_at(unpack(z), am); r === nothing ? SMM_PENALTY : r.q)
        opt = Opt(:LN_NELDERMEAD, length(lo))
        lower_bounds!(opt, lo); upper_bounds!(opt, hi)
        ftol_rel!(opt, 1e-4); ftol_abs!(opt, 1e-10); xtol_rel!(opt, 1e-7)
        maxeval!(opt, REOPT_EVALS)
        n = Ref(0); min_objective!(opt, (z, g) -> (n[] += 1; obj(z)))
        t0 = time(); (qq, zz, rr) = optimize(opt, copy(z0))
        e = unpack(zz)
        open(joinpath(RUN_DIR, "reoptimized.csv"), "a") do io
            println(io, am, ",", qq, ",",
                    join((@sprintf("%.10g", getfield(e, q.name)) for q in SMM_PARAMS), ","),
                    ",", rr, ",", n[])
        end
        sayf("  a_max %6.0f   Q %10.6g   ret %s   %d evals   %.1f min\n",
             am, qq, rr, n[], (time()-t0)/60)
    end
end

open(joinpath(RUN_DIR, "meta.toml"), "w") do io
    println(io, "# GENERATED by code/smm/grid_sensitivity.jl")
    println(io, "generated  = \"", Dates.format(now(), "yyyy-mm-dd HH:MM"), "\"")
    println(io, "git_commit = \"", git_sha(), "\"")
    println(io, "point      = \"", isempty(ATFILE) ? "PARENT_DEFAULTS" : relpath(ATFILE, REPO), "\"")
    println(io, "a_max      = [", join(A_MAXES, ", "), "]")
    println(io, "Na         = ", GRID, "   # node cap is an instruction, see CLAUDE.md")
    println(io, "simN       = ", SIM_N)
    println(io, "seed       = ", SEED)
    println(io, "reoptimized= ", REOPT)
end
sayf("\nwrote %s\n", relpath(RESULTS_F, REPO))
close(LOG)
