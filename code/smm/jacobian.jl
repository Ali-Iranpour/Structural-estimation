#!/usr/bin/env julia
# =============================================================================
# jacobian.jl -- the residual Jacobian, SAVED, with everything needed to reproduce it.
#
#     cd code/smm
#     julia +1.11 --project=../.. jacobian.jl                      # at the incumbent
#     julia +1.11 --project=../.. jacobian.jl --at ../../output/smm_runs/<d>/estimates.toml
#     julia +1.11 --project=../.. jacobian.jl --extend sigma_4_1   # the 10-column variant
#     julia +1.11 --project=../.. jacobian.jl --extend sigma_4_1,mu_1
#
# WHY THIS FILE EXISTS
# --------------------
# The condition numbers and pairwise cosines quoted in moments.jl, code/smm/README.md and
# docs/SMM.md were computed in a review conversation and never saved: no matrix, no
# evaluation point, no step size, no script. They could not be reproduced, checked at a
# different point, or compared like for like against a ten-column variant -- which is
# exactly what deciding whether to free sigma_4_1 requires. This script produces them as
# an artefact instead of a claim.
#
# WHAT IS SAVED (jacobian_<stamp>/)
#   jacobian.toml    every number below, plus the full metadata needed to redo it
#   J_h<step>.csv    one residual Jacobian per finite-difference step, plain CSV
#   README.txt       what the columns and rows are
#
# WHAT IT DOES NOT ESTABLISH
# --------------------------
# A local Jacobian at one point under one scaling. Full column rank here is not global
# identification, a small condition number is not precision, and none of it is inference --
# for that see standard_errors.jl. Pairwise cosines are invariant to column rescaling;
# condition numbers are NOT, so the scaling is recorded with them and any comparison has to
# hold it fixed.
# =============================================================================

using Distributed, Printf, Dates, LinearAlgebra, TOML
include(joinpath(@__DIR__, "finite_differences.jl"))

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

function argstr(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("$flag needs a value")
    return ARGS[i + 1]
end
argval(flag, default) = (v = argstr(flag, nothing); v === nothing ? default : parse(Int, v))
argflt(flag, default) = (v = argstr(flag, nothing); v === nothing ? default : parse(Float64, v))

const AT_FILE  = argstr("--at", "")
const EXTEND   = filter(!isempty, split(argstr("--extend", ""), ','))
const GRID     = argval("--grid", 30)
const SIM_N    = argval("--simN", 2000)
const SEED     = argval("--seed", 1234)
const NPROC    = argval("--procs", 20)
const OUTDIR   = argstr("--outdir",
                    joinpath(REPO, "output", "identification",
                             "jacobian_" * Dates.format(now(), "yyyy-mm-dd_HHMMSS")))
# Several steps, deliberately. One step size cannot tell a real small singular value from
# numerical derivative noise; three can. Expressed as a fraction of the parameter's own BOX
# WIDTH so every column is differentiated on the same relative scale.
const STEPS = let v = argstr("--steps", "")
    isempty(v) ? [0.005, 0.01, 0.02] : parse.(Float64, split(v, ','))
end

mkpath(OUTDIR)
const LOG = open(joinpath(OUTDIR, "run.log"), "w")
say(a...)  = (println(a...); println(LOG, a...); flush(LOG); flush(stdout))
sayf(f, a...) = (s = Printf.format(Printf.Format(f), a...);
                 print(s); print(LOG, s); flush(LOG); flush(stdout))

say("="^78); say("Residual Jacobian and identification audit"); say("="^78)
sayf("started  %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
sayf("outdir   %s\n", relpath(OUTDIR, REPO))

if NPROC > 0 && (nprocs() - 1) < NPROC
    addprocs(NPROC - (nprocs() - 1); exeflags = `--project=$REPO`)
end
@everywhere using LinearAlgebra
@everywhere LinearAlgebra.BLAS.set_num_threads(1)
sayf("workers  %d\n", max(0, nprocs() - 1))

@everywhere begin
    using Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames
    using Statistics, Dates, ProgressMeter, Distributions, StatsBase
    using QuantEcon, FastGaussQuadrature, Parameters, Dierckx, TOML
    const REPO_ = normpath(joinpath(@__DIR__, "..", ".."))
    const SRC   = joinpath(REPO_, "code", "src")
    include(joinpath(SRC, "paths.jl"));      include(joinpath(SRC, "manifest.jl"))
    include(joinpath(SRC, "diagnostics.jl")); include(joinpath(SRC, "child_lifecycle.jl"))
    include(joinpath(SRC, "parent_family.jl"))
    include(joinpath(REPO_, "code", "smm", "moments.jl"))
end

@everywhere const TARGETS = load_targets(joinpath(REPO_, "Input", "smm_targets_baseline.toml"))
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

# -----------------------------------------------------------------------------
# The columns: the nine estimated parameters, plus any candidate extension
# -----------------------------------------------------------------------------
# A candidate column needs a BOX to be scaled by, exactly as an estimated one does -- the
# "full-box move" scaling is what makes columns comparable at all. These are the boxes that
# would be used if the parameter were freed, stated here rather than improvised.
# Diagnostic candidates only, not extra estimated parameters. The mu_1 box
# keeps mu_t = 1 + mu_1*(t-5) in (0,1) for ages 6..17.
const CANDIDATES = Dict(
    :sigma_4_1 => (lo = -0.05, hi = 0.15, link = :level),
    :mu_1      => (lo = -0.08, hi = -0.005, link = :level),
    :phi_1     => (lo = 0.2,  hi = 5.0,  link = :log),
    :lambda_1  => (lo = 0.2,  hi = 5.0,  link = :log),
)

struct Column
    name::Symbol
    lo::Float64
    hi::Float64
    link::Symbol
    estimated::Bool
end
const COLUMNS = vcat(
    [Column(q.name, q.lo, q.hi, q.link, true) for q in SMM_PARAMS],
    [begin
        sym = Symbol(nm)
        haskey(CANDIDATES, sym) || error("--extend: no box on record for $nm. Add it to CANDIDATES.")
        c = CANDIDATES[sym]
        hasproperty(PARENT_DEFAULTS, sym) ||
            error("--extend $nm is not a parent-block parameter; the child solve could not be reused.")
        Column(sym, c.lo, c.hi, c.link, false)
    end for nm in EXTEND])

to_s(v, c::Column)   = c.link === :log ? log(v) : v
from_s(z, c::Column) = c.link === :log ? exp(z) : z

# -----------------------------------------------------------------------------
# The evaluation point
# -----------------------------------------------------------------------------
# "Evaluate near the fitted baseline, not just at the incumbent calibration": --at takes an
# estimates.toml (or checkpoint.toml) and reads its [parameters] block. Any column not
# named there falls back to PARENT_DEFAULTS, which is what makes a candidate extension
# evaluable at a fitted nine-parameter point.
const THETA0 = begin
    base = Dict{Symbol,Float64}(c.name => getfield(PARENT_DEFAULTS, c.name) for c in COLUMNS)
    if !isempty(AT_FILE)
        raw = TOML.parsefile(AT_FILE)
        pars = get(raw, "parameters", Dict{String,Any}())
        for (k, v) in pars
            sym = Symbol(k)
            haskey(base, sym) && (base[sym] = Float64(v))
        end
        sayf("point    %s\n", relpath(AT_FILE, REPO))
    else
        say("point    PARENT_DEFAULTS (the incumbent calibration)")
    end
    base
end

# Any parameter NOT a column is held at its default; the model is built from a full kwarg
# set so nothing is implicit.
@everywhere function residuals_at(vals::Dict{Symbol,Float64}; Na, Nhc, simN, seed)
    kw = NamedTuple{Tuple(keys(vals))}(Tuple(values(vals)))
    p = Parent_child_interaction_age_specific_AR1(; Na = Na, Nk = 2, Nhc = Nhc,
                                                    simN = simN, seed = seed, kw...)
    p.V_child_interp = V_CHILD
    redirect_stdout(devnull) do
        solve_model!(p; verbose = false); simulate_model!(p)
    end
    m = model_moments(p)
    v = simulation_violations(p)
    r = [ (getfield(m, Symbol(k)) - TARGETS[k].mean) / moment_scale(k, TARGETS[k].mean)
          for k in SMM_MOMENTS ]
    return (r = r, nviol = v.total, nbad = m.n_nonfinite)
end

const NP_ = length(COLUMNS)
const NM_ = length(SMM_MOMENTS)
sayf("columns  %d  (%d estimated%s)\n", NP_, count(c -> c.estimated, COLUMNS),
     isempty(EXTEND) ? "" : ", extended by " * join(EXTEND, ", "))
sayf("moments  %d\n", NM_)
sayf("grids    Na = Nhc = %d, simN = %d, seed = %d (common random numbers)\n", GRID, SIM_N, SEED)
sayf("steps    %s  (fraction of each parameter's box width; one-sided near bounds)\n",
     join(STEPS, ", "))

# -----------------------------------------------------------------------------
# Build the Jacobian at each step size
# -----------------------------------------------------------------------------
# COLUMN SCALING. Column j is d r / d z_j times the box width in search coordinates, i.e.
# the residual change from moving parameter j across its WHOLE admissible box. Without a
# common scale the condition number is a statement about units. Pairwise cosines are
# invariant to it; the condition number is not, which is why both are reported and the
# scaling is written into the output.
function jacobian_at(step::Float64)
    zs = [to_s(THETA0[c.name], c) for c in COLUMNS]
    stencils = [bounded_stencil(zs[j], to_s(c.lo,c), to_s(c.hi,c), step)
                for (j,c) in enumerate(COLUMNS)]
    jobs = NamedTuple[]
    for (j,stencil) in enumerate(stencils), (z,weight) in zip(stencil.points,stencil.weights)
        vals = copy(THETA0)
        vals[COLUMNS[j].name] = from_s(z, COLUMNS[j])
        push!(jobs, (j=j, weight=weight, vals=vals))
    end
    out = pmap(jb -> residuals_at(jb.vals; Na=GRID, Nhc=GRID, simN=SIM_N, seed=SEED), jobs)
    J = zeros(NM_, NP_)
    for (job,r) in zip(jobs,out)
        r.nviol == 0 && r.nbad == 0 && all(isfinite,r.r) || error(
            "Invalid finite-difference evaluation for $(COLUMNS[job.j].name); refusing Jacobian")
        c = COLUMNS[job.j]
        J[:,job.j] .+= job.weight .* r.r .* (to_s(c.hi,c)-to_s(c.lo,c))
    end
    return J, 0, stencils, length(jobs)
end

const RESULTS = Dict{Float64,Any}()
for step in STEPS
    t = time()
    J, nbad, stencils, n_evaluations = jacobian_at(step)
    F = svd(J)
    cond_ = F.S[1] / F.S[end]
    RESULTS[step] = (J=J, S=F.S, V=F.V, cond=cond_, nbad=nbad, stencils=stencils, n_evaluations=n_evaluations)
    # MORE COLUMNS THAN MOMENTS is a guaranteed null direction, and a thin SVD does not
    # show it: for a 10x11 matrix `svd` returns 10 singular values, all of which can be
    # positive, while the parameter space still has a direction the moments cannot see.
    # Say so rather than let ten positive values read as full rank.
    if NP_ > NM_
        sayf("  !! %d columns against %d moments: at least %d parameter direction(s) are\n",
             NP_, NM_, NP_ - NM_)
        say("     UNIDENTIFIED by construction. The singular values below are the thin SVD's")
        say("     and do not include them; equal counts would not fix identification either.")
    end
    sayf("\nstep %.4f  (%d evaluations, %.1f min)%s\n", step, n_evaluations, (time()-t)/60,
         nbad > 0 ? @sprintf("  !! %d invalid/non-finite cells at perturbed points", nbad) : "")
    sayf("  singular values: %s\n", join((@sprintf("%.4g", s) for s in F.S), "  "))
    sayf("  condition number %.1f   smallest singular value %.4g\n", cond_, F.S[end])
    # The weak direction, in parameters: the right singular vector of the smallest value.
    # The weakest SEEN direction. With NP_ > NM_ the truly unseen directions are the null
    # space, which the thin SVD omits entirely -- reported above.
    w = F.V[:, end]
    ord = sortperm(abs.(w); rev = true)
    sayf("  weakest direction: %s\n",
         join((@sprintf("%s %+.3f", COLUMNS[k].name, w[k]) for k in ord[1:min(4, NP_)]), ", "))
    # Save the matrix itself.
    open(joinpath(OUTDIR, @sprintf("J_h%.4f.csv", step)), "w") do io
        println(io, "moment," * join((String(c.name) for c in COLUMNS), ","))
        for (i, k) in enumerate(SMM_MOMENTS)
            println(io, k, ",", join((@sprintf("%.17g", J[i, j]) for j in 1:NP_), ","))
        end
    end
end

# -----------------------------------------------------------------------------
# Is the smallest singular value resolved above derivative noise?
# -----------------------------------------------------------------------------
# The triage's question, stated as a number. If sigma_min moves by more across step sizes
# than it is worth, it is a measurement of the finite-difference error and not of the
# model.
let smin = [RESULTS[s].S[end] for s in STEPS], conds = [RESULTS[s].cond for s in STEPS]
    spread = maximum(smin) - minimum(smin)
    say("\nresolution of the smallest singular value across steps")
    for (s, v, c) in zip(STEPS, smin, conds)
        sayf("  step %.4f   sigma_min %.5g   cond %.1f\n", s, v, c)
    end
    sayf("  spread %.3g against level %.3g  ->  ratio %.3f\n",
         spread, minimum(smin), spread / max(minimum(smin), eps()))
    say(spread < 0.1 * minimum(smin) ?
        "  RESOLVED: sigma_min is stable across steps, so it is the model's, not the difference's." :
        "  NOT RESOLVED: sigma_min moves with the step. Treat the condition number as an upper\n" *
        "  bound on conditioning quality and do not read the weak direction as structural.")
end

# -----------------------------------------------------------------------------
# Pairwise cosines -- scale-invariant, so these are the comparable numbers
# -----------------------------------------------------------------------------
const REF = RESULTS[STEPS[1]]
say("\npairwise |cos| between scaled Jacobian columns (step $(STEPS[1]))")
say("  a cosine near 1 means two parameters move the moments in nearly the same direction;")
say("  it is a warning about local separation, NOT proof of observational equivalence.")
const COSINES = Tuple{Symbol,Symbol,Float64}[]
for a in 1:NP_, b in (a+1):NP_
    ca, cb = REF.J[:, a], REF.J[:, b]
    push!(COSINES, (COLUMNS[a].name, COLUMNS[b].name,
                    abs(dot(ca, cb)) / (norm(ca) * norm(cb))))
end
for (a, b, c) in first(sort(COSINES; by = last, rev = true), 8)
    sayf("  %-12s %-12s %.4f\n", a, b, c)
end

# -----------------------------------------------------------------------------
# Persist
# -----------------------------------------------------------------------------
open(joinpath(OUTDIR, "jacobian.toml"), "w") do io
    println(io, "# Residual Jacobian and identification audit. GENERATED by code/smm/jacobian.jl.")
    println(io, "# Everything needed to reproduce this is in this file; the matrices are the CSVs beside it.")
    println(io, "generated   = \"", Dates.format(now(), "yyyy-mm-dd HH:MM"), "\"")
    println(io, "git_commit  = \"", git_sha(), "\"")
    println(io, "point_file  = \"", isempty(AT_FILE) ? "PARENT_DEFAULTS" : relpath(AT_FILE, REPO), "\"")
    println(io, "targets     = \"Input/smm_targets_baseline.toml\"")
    println(io, "grid_Na     = ", GRID)
    println(io, "grid_Nhc    = ", GRID)
    println(io, "grid_Nk     = 2")
    println(io, "simN        = ", SIM_N)
    println(io, "seed        = ", SEED, "   # common random numbers across every evaluation")
    println(io, "steps       = [", join(STEPS, ", "), "]   # fraction of each box width")

    println(io, "difference  = \"central or second-order one-sided in search coordinates\"")
    println(io, "column_scale= \"d r / d z_j, times the box width in SEARCH coordinates\"")
    println(io, "moments     = [", join(("\"$m\"" for m in SMM_MOMENTS), ", "), "]")
    println(io, "\n[point]")
    for c in COLUMNS; @printf(io, "%-10s = %.10f\n", c.name, THETA0[c.name]); end
    println(io, "\n[columns]")
    println(io, "names     = [", join(("\"$(c.name)\"" for c in COLUMNS), ", "), "]")
    println(io, "estimated = [", join((c.estimated for c in COLUMNS), ", "), "]")
    println(io, "lo        = [", join((c.lo for c in COLUMNS), ", "), "]")
    println(io, "hi        = [", join((c.hi for c in COLUMNS), ", "), "]")
    println(io, "link      = [", join(("\"$(c.link)\"" for c in COLUMNS), ", "), "]")
    println(io, "\n[moment_scales]")
    for k in SMM_MOMENTS
        @printf(io, "%-16s = %.10f\n", k, moment_scale(k, TARGETS[k].mean))
    end
    for step in STEPS
        r = RESULTS[step]
        @printf(io, "\n[step_%s]\n", replace(@sprintf("%.4f", step), "." => "_"))
        println(io, "matrix_file        = \"", @sprintf("J_h%.4f.csv", step), "\"")
        println(io, "condition_number   = ", r.cond)
        println(io, "singular_values    = [", join((@sprintf("%.10g", s) for s in r.S), ", "), "]")
        println(io, "smallest_sv        = ", r.S[end])
        println(io, "rank_tol           = ", maximum(size(r.J)) * eps() * r.S[1])
        println(io, "numerical_rank     = ", count(>(maximum(size(r.J)) * eps() * r.S[1]), r.S))
        println(io, "invalid_cells      = ", r.nbad)
        println(io, "n_evaluations      = ", r.n_evaluations)
        TOML.print(io, Dict(
            "stencil_schemes" => [st.scheme for st in r.stencils],
            "stencil_search_points" => [st.points for st in r.stencils],
            "stencil_weights" => [st.weights for st in r.stencils]))
        println(io, "weakest_direction  = [", join((@sprintf("%.10g", v) for v in r.V[:, end]), ", "),
                "]   # right singular vector of the smallest singular value")
    end
    println(io, "\n[cosines]")
    for (a, b, c) in sort(COSINES; by = last, rev = true)
        @printf(io, "\"%s|%s\" = %.6f\n", a, b, c)
    end
end
open(joinpath(OUTDIR, "README.txt"), "w") do io
    println(io, """
    Residual Jacobian, produced by code/smm/jacobian.jl.

    J_h<step>.csv   rows = the $(NM_) targeted moments, in SMM_MOMENTS order.
                    cols = $(join((String(c.name) for c in COLUMNS), ", ")).
                    Entry (i,j) is the change in residual i -- (model - data)/scale --
                    from moving parameter j across its ENTIRE box, estimated by central differences
                    or second-order one-sided differences near bounds. Actual search
                    points, weights and schemes are recorded in jacobian.toml.

    jacobian.toml   singular values, condition numbers, weakest directions, pairwise
                    cosines, and the full metadata: evaluation point, boxes, links,
                    moment scales, grids, seed and steps.

    Read with the qualifications in the script header: this is ONE local Jacobian at ONE
    point under ONE scaling. It is not global identification and it is not inference.""")
end
sayf("\nwrote %s\n", relpath(joinpath(OUTDIR, "jacobian.toml"), REPO))
close(LOG)
