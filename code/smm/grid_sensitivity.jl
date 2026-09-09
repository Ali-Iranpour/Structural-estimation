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

# NOTE ON THE CHILD SOLVE. It keeps a_max = 100 throughout, deliberately: the child block's
# own grid is a separate object and changing it would confound the parent-grid question
# with a child-grid one. What crosses the handoff is sim_a[:, T+1], and the child's
# terminal-value spline is evaluated by interpolation, so a parent household above 100 is
# already extrapolating there -- which is part of what this measures.

@everywhere function solve_at(kw, a_max)
    p = Parent_child_interaction_age_specific_AR1(; Na = GS_GRID, Nk = 2, Nhc = GS_GRID,
                                                    a_max = a_max, simN = GS_SIMN,
                                                    seed = GS_SEED, school_time = target_school_time(TARGETS), kw...)
    p.V_child_interp = V_CHILD
    redirect_stdout(devnull) do
        solve_model!(p; verbose = false); simulate_model!(p)
    end
    return p
end

@everywhere function evaluate_at(kw, a_max)
    try
        p = solve_at(kw, a_max)
        m = model_moments(p); d = moment_diagnostics(p); v = simulation_violations(p)
        q = 0.0
        for k in SMM_MOMENTS
            mj = getfield(m, Symbol(k))
            isfinite(mj) || return nothing
            q += ((mj - TARGETS[k].mean) / moment_scale(k, TARGETS[k].mean))^2
        end
        return (q = q, m = m, d = d, viol = v.total,
                node_spacing = (a_max - p.a_grid[1]) / (GS_GRID - 1))
    catch err
        is_model_failure(_root_cause(err)) && return nothing
        rethrow()
    end
end

# ---- the point ---------------------------------------------------------------
const THETA = begin
    base = Dict{Symbol,Float64}(q.name => getfield(PARENT_DEFAULTS, q.name) for q in SMM_PARAMS)
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
    r = evaluate_at(THETA, am)
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
    sc = moment_scale(k, TARGETS[k].mean)
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

let worst = maximum(maximum(abs((getfield(o.m, Symbol(k)) - getfield(BASE.m, Symbol(k))) /
                                moment_scale(k, TARGETS[k].mean)) for k in SMM_MOMENTS)
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
        for w in procs(); remotecall_fetch(() -> (Main.GS_AMAX = $am; nothing), w); end
        obj = z -> (r = evaluate_at(unpack(z), am); r === nothing ? SMM_PENALTY : r.q)
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
