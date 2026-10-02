# =============================================================================
# localsearch.jl -- the Sobol design, the mixing schedule, one bounded NLopt local solve,
# and the refinement of a point on another objective.
# =============================================================================

"""
    sobol_points(lo, hi, N; skip_first = true) -> Vector{Vector{Float64}}

`N` Sobol' points in the box. The first point of a Sobol' sequence sits at a
corner or centre depending on the generator; it is skipped so the sample is not
anchored to a degenerate point.
"""
function sobol_points(lo::Vector{Float64}, hi::Vector{Float64}, N::Int; skip_first::Bool = true)
    s = Sobol.SobolSeq(lo, hi)
    skip_first && Sobol.next!(s)
    return [copy(Sobol.next!(s)) for _ in 1:N]
end

"""
    theta_at(j, K, p, lo, hi) -> Float64

The mixing weight of restart j out of K: 0 for the first restart (it starts at the best
seed itself), then clamp((j/K)^p, lo, hi). With the defaults (p = 0.5, lo = 0.1,
hi = 0.995) this is min(max(0.1, sqrt(j/K)), 0.995), AGK (2022) Appendix A.6, footnote 27.
K is the run's schedule denominator, fixed for its whole life (plan step 2.4).
"""
theta_at(j::Int, K::Int, p::Float64, lo::Float64, hi::Float64) = j == 1 ? 0.0 : clamp((j / K)^p, lo, hi)
theta_at(cfg::TikTakConfig, j::Int, K::Int) = theta_at(j, K, cfg.theta_p, cfg.theta_lo, cfg.theta_hi)

"""
    make_start!(x0, seed, Z, theta, lo, hi) -> x0

The start of a restart, written into the scratch buffer `x0`:
x0 = clamp((1 - theta) seed + theta Z, lo, hi), one fused loop with no temporaries (plan J3).
Bit-identical to the broadcast the local stage always used. The buffer may not alias its
inputs; a caller copies the result into the job it dispatches before reusing the buffer.
"""
function make_start!(x0::Vector{Float64}, seed::Vector{Float64}, Z::Vector{Float64}, theta::Float64,
                     lo::Vector{Float64}, hi::Vector{Float64})
    (x0 === seed || x0 === Z) && throw(ArgumentError("make_start!: the scratch buffer aliases an input"))
    @. x0 = clamp((1 - theta) * seed + theta * Z, lo, hi)
    return x0
end

# -----------------------------------------------------------------------------
# Progress from inside a solve: nothing by default; a callback on the master; a coalesced
# message to the master from a worker (step 6).
# -----------------------------------------------------------------------------
abstract type Progress end
"No progress reporting."
struct NoProgress <: Progress end
report!(::NoProgress, job::RestartJob, n::Int, v::Float64, best::Float64) = nothing

"""
    CallbackProgress(cb, every)

Calls `cb(stage, j, attempt, n_eval, f_last, f_best)` at most once per `every` seconds
(`every = 0`: after every evaluation), on the process running the solve.
"""
mutable struct CallbackProgress{C} <: Progress
    cb::C
    every::Float64
    last::Float64
end
CallbackProgress(cb, every::Real) = CallbackProgress(cb, Float64(every), 0.0)
function report!(p::CallbackProgress, job::RestartJob, n::Int, v::Float64, best::Float64)
    t = time()
    if n == 1 || t - p.last >= p.every
        p.last = t
        p.cb(job.stage, job.j, job.attempt, n, v, best)
    end
    return nothing
end

"Search coordinates to box-normalized ones, u = (x - lo) / (hi - lo), clamped to [0, 1]."
to_unit(x::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64}) = clamp.((x .- lo) ./ (hi .- lo), 0.0, 1.0)
"Box-normalized coordinates back to search coordinates, clamped into the box (a round trip is exact to rounding)."
from_unit(u::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64}) = clamp.(lo .+ u .* (hi .- lo), lo, hi)

"""
    run_local(f, job, progress = NoProgress()) -> RestartResult

One bounded local solve, with an `Opt` THAT BELONGS TO IT (built here, on whichever process
runs the job; never shared). For a restart the start is evaluated first, as the local stage
always did, and then the solver runs from it; the evaluation count includes that start.

`f` is a type parameter, so the kernel is compiled for the actual objective (plan J1). An
exception is re-thrown under `on_error = :rethrow`; under `:discard` it is returned as
:SEED_EXCEPTION or :EXCEPTION with its message (the start point kept), never swallowed.
"""
function run_local(f::F, job::RestartJob, progress::P = NoProgress()) where {F,P<:Progress}
    t0 = time()
    s = job.settings
    x0 = copy(job.x0)
    nev = Ref(0)
    best = Ref(Inf)
    call = function (x)
        v = Float64(f(x))
        nev[] += 1
        v < best[] && (best[] = v)
        report!(progress, job, nev[], v, best[])
        return v
    end
    fstart = job.f_start_known
    if job.eval_start
        # The start evaluation is guarded too. It was not before 2026-08, so one throw here
        # killed the whole run at whichever restart hit it.
        try
            fstart = call(copy(x0))          # a copy: the solver's start is not the objective's to modify
        catch e
            job.on_error === :rethrow && rethrow()
            return RestartResult(job.run_id, job.stage, job.j, job.attempt, Inf, x0, Inf, nev[],
                                 :SEED_EXCEPTION, sprint(showerror, e), Distributed.myid(), time() - t0)
        end
    end
    n = length(x0)
    opt = NLopt.Opt(s.alg, n)
    w = job.hi .- job.lo
    if job.normalize
        # the solver works in u = (x - lo) / (hi - lo); the objective always sees x
        NLopt.lower_bounds!(opt, zeros(n)); NLopt.upper_bounds!(opt, ones(n))
    else
        NLopt.lower_bounds!(opt, job.lo); NLopt.upper_bounds!(opt, job.hi)
    end
    NLopt.ftol_rel!(opt, s.ftol_rel); NLopt.maxeval!(opt, s.maxeval)
    # ftol_abs and xtol_rel back up ftol_rel near a zero optimum, where |df| <= ftol_rel*|f|
    # is hard to satisfy; neither is scale-free (ftol_abs is in objective units, xtol_rel is
    # relative to |x|), so both are set far below the ftol_rel that normally stops a search.
    NLopt.ftol_abs!(opt, s.ftol_abs); NLopt.xtol_rel!(opt, s.xtol_rel)
    if s.initial_step > 0                           # else NLopt's own heuristic (the baseline)
        frac = s.initial_step * job.step_scale
        NLopt.initial_step!(opt, job.normalize ? fill(frac, n) : frac .* w)
    end
    xbuf = similar(x0)                              # job-local scratch for the normalized map
    if job.normalize
        NLopt.min_objective!(opt, (u, g) -> (@. xbuf = clamp(job.lo + u * w, job.lo, job.hi); call(xbuf)))
    else
        NLopt.min_objective!(opt, (x, g) -> call(x))
    end
    try
        start = job.normalize ? to_unit(x0, job.lo, job.hi) : x0
        (floc, zloc, ret) = NLopt.optimize(opt, start)
        xloc = job.normalize ? from_unit(zloc, job.lo, job.hi) : zloc
        return RestartResult(job.run_id, job.stage, job.j, job.attempt, fstart, xloc, Float64(floc),
                             nev[], ret, "", Distributed.myid(), time() - t0)
    catch e
        job.on_error === :rethrow && rethrow()
        # Information, not a fatal error -- but not invisible either.
        return RestartResult(job.run_id, job.stage, job.j, job.attempt, fstart, x0, fstart, nev[],
                             :EXCEPTION, sprint(showerror, e), Distributed.myid(), time() - t0)
    end
end

"""
    refine(f, x_start, lo, hi; settings, cfg, objective_id) -> RefineOutcome

A local solve of the retained point on ANOTHER objective -- the runner's full-grid
refinement after a coarse-grid search. The start is re-evaluated on `f` first; that value
is the reported point's value if the solve finds nothing better, and it carries no
convergence evidence of its own (origin :reevaluated). The solve then merges exactly as a
restart does (`merge_candidate`): a strict improvement becomes the reported point with the
solve's own return code -- so a refinement that stopped on MAXEVAL_REACHED is reported as
budget-limited, however much it improved (finding 2) -- and a converged solve that returns
the start itself verifies it.

Any exception in the solve is caught and reported as `:failed` with the start point kept,
as the runner always did; the caller decides what a failed stage means for acceptance.
"""
function refine(f::F, x_start::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64};
                settings::SolverSettings, cfg::TikTakConfig, objective_id::String) where {F}
    f0 = Float64(f(x_start))
    inc = Incumbent(copy(x_start), f0, 1,
                    CandidateOrigin(:reevaluated, 0, 1, 0, :REEVALUATED, objective_id), NO_VERIFICATION)
    n = Ref(0)
    try
        opt = NLopt.Opt(settings.alg, length(lo))
        NLopt.lower_bounds!(opt, lo); NLopt.upper_bounds!(opt, hi)
        NLopt.ftol_rel!(opt, settings.ftol_rel); NLopt.ftol_abs!(opt, settings.ftol_abs)
        NLopt.xtol_rel!(opt, settings.xtol_rel); NLopt.maxeval!(opt, settings.maxeval)
        NLopt.min_objective!(opt, (z, g) -> (n[] += 1; f(z)))
        (qr, zr, retr) = NLopt.optimize(opt, x_start)
        action, inc2 = merge_candidate(inc, zr, Float64(qr), retr, :refine, 0, 1, lo, hi, cfg,
                                       objective_id, solver_summary(settings))
        return RefineOutcome(action === :improved ? :improved : :no_improvement, retr, n[], f0, inc2, "")
    catch e
        return RefineOutcome(:failed, :EXCEPTION, n[], f0, inc, sprint(showerror, e))
    end
end

"The outcome of a refinement that was not needed: the search already used the reporting objective."
refine_skipped(inc::Incumbent) = RefineOutcome(:skipped, :NOT_RUN, 0, inc.f, inc, "")
