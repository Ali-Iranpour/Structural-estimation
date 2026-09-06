# =============================================================================
# tiktak.jl — the TikTak global optimizer.
#
# Arnoud, Guvenen & Kleineberg (2022), "Benchmarking Global Optimizers",
# Section 2.1 and Appendix A.2.1. Reference implementation:
# https://github.com/serdarozkan/TikTak
#
# TikTak is a multistart algorithm in two stages:
#
#   GLOBAL (pre-testing)   Evaluate f at N Sobol' points covering the box, sort
#                          ascending, and keep the best N* as "seed" points
#                          s_1, ..., s_N* with f(s_1) <= ... <= f(s_N*). The
#                          rest are discarded. The paper puts N* at 1-10% of N.
#
#   LOCAL                  Run N* local searches. The j-th starts not at s_j but
#                          at a convex combination of s_j and the best minimiser
#                          found so far Z*:
#
#                              s~_j = (1 - theta_j) * s_j + theta_j * Z*
#
#                          theta_j rises with j, so early restarts explore and
#                          later ones exploit. Then one final "polishing" search
#                          from the winner with a stringent tolerance.
#
# -----------------------------------------------------------------------------
# DECISIONS THE PAPER LEAVES OPEN, and what this implementation does.
# The paper specifies theta_j only as "very small, possibly zero" early and
# "gradually increased"; it gives no formula, and neither does the repo README.
# These were settled explicitly (2026-08-07) rather than silently defaulted:
#
#   theta_j        clamp((j / N*)^0.5, 0.1, 0.995)
#                  Square-root growth, the form used in the circulated TikTak
#                  implementations: 0.32 at 10% through the local stage, 0.71 at
#                  50%. The 0.995 cap keeps the seed point from vanishing
#                  entirely, so late restarts still carry some new information.
#                  Exposed as (theta_p, theta_lo, theta_hi) so the schedule can
#                  be changed without touching the algorithm.
#
#   local search   Nelder-Mead at ftol 1e-3 — the paper's TikTak-nm3 variant.
#                  DFNLS (its "d" variants) has no maintained Julia binding, so
#                  the nm variants are the faithful ones available. Footnote 14
#                  defines the Nelder-Mead tolerance as the spread of function
#                  values across the simplex; NLopt's `ftol_rel` is the closest
#                  available stopping rule.
#
#   polishing      BOBYQA at ftol 1e-10. The paper applies a polishing search
#                  with a stringent criterion after every global optimisation
#                  (Section 3.3) and names DFNLS and/or BOBYQA for it.
#
#   extra seeds    DEVIATION FROM THE PUBLISHED ALGORITHM, taken deliberately.
#                  Pure TikTak seeds only from Sobol points. `extra_seeds` lets
#                  the caller inject known-good points (here: the incumbent
#                  calibration) into the pre-testing pool, so the result can
#                  never be worse than where the search started. They compete on
#                  function value like any Sobol point.
#
#   early stop     The paper suggests stopping when the last two DISTINCT values
#                  of Z* are close. Implemented as `stop_tol`, DEFAULT 0.0 =
#                  disabled, so the full budget runs unless asked otherwise.
#
# -----------------------------------------------------------------------------
# PARALLELISM. Run Julia with `-t auto` (or `-t N`) and both stages use threads.
#
#   Global stage    Embarrassingly parallel -- N independent evaluations.
#
#   Local stage     Sequential in the published algorithm: restart j starts from
#                   Z* as updated by restarts 1..j-1. Running it in parallel means
#                   a batch of `batch` restarts all read the Z* frozen at the START
#                   of their batch, so within a batch they do not see each other's
#                   improvements. That is the ASYNCHRONOUS variant the reference
#                   implementation uses to scale, and it is why that repo suggests
#                   #cores <= sqrt(N): too wide a batch and the theta mixing stops
#                   transmitting information, degrading TikTak toward plain
#                   multistart. `batch = 1` reproduces the sequential algorithm
#                   exactly and is the default when Julia has one thread.
#
#                   MEASURED cost of batching (Rastrigin d=6, N=1500, N*=40):
#                       batch  1 (sequential)   f = 2.985   <- published algorithm
#                       batch  2               f = 3.980
#                       batch  4               f = 6.965
#                       batch  8               f = 5.970
#                   So width is not free. The default follows the reference repo's
#                   heuristic, #cores <= sqrt(#restarts), which for N* = 25 is 5.
#                   The GLOBAL stage has no such constraint and always uses every
#                   thread, which is where most of the speedup comes from anyway.
#
# `f` MUST be thread-safe when batch > 1: no shared mutable state between calls.
#
# !! `parallel` DEFAULTS TO FALSE, and for this project it must stay false. !!
#
# Measured 2026-08-07: with `parallel = true` and 8 threads the SMM objective kills
# the process silently -- exit status 0, no error, no output past the first few
# evaluations. The same run at `-t 1` completes normally. The objective calls
# `solve_model!`, which issues thousands of NLopt `optimize` calls, and NLopt.jl's
# callback machinery is not safe under concurrent optimization from several threads.
# The fault is in the OBJECTIVE, not in this algorithm: the staging here is correct
# and works for any thread-safe `f` (the self-test runs fine at `-t 8`).
#
# TO USE MANY CORES, USE PROCESSES, NOT THREADS. Each worker process gets its own
# NLopt state, which sidesteps the problem entirely and is also what scales across
# nodes on a cluster:
#
#     using Distributed; addprocs(N)
#     @everywhere include("smm.jl")
#     fs = pmap(f, cands)            # in place of the Threads.@threads loop below
#
# The Sobol stage is embarrassingly parallel and is where nearly all the speedup is
# (N evaluations against N* local searches of ~300 each, but the local stage cannot
# widen far without degrading the algorithm -- see the batching table above).
# =============================================================================

using Sobol

"""
    TikTakResult

`x`/`f` are the best point and value after polishing. `trace` records one row
per local search: the seed index, theta, the start value, and the value the
local search reached — enough to see whether the search was still improving when
the budget ran out.
"""
struct TikTakResult
    x::Vector{Float64}
    f::Float64
    n_eval::Int
    f_sobol_best::Float64        # best of the pre-testing stage, before any local search
    f_prepolish::Float64         # best after the local stage, before polishing
    trace::Vector{NamedTuple{(:j, :theta, :f_start, :f_local, :improved, :ret),
                             Tuple{Int,Float64,Float64,Float64,Bool,Symbol}}}
    n_exception::Int             # local searches that threw -- always a bug in `f`
    # HOW THE FINAL POINT WAS ARRIVED AT, kept separate from WHAT it is.
    #
    # A run that stops because every restart exhausted `maxeval` is not the same result as
    # one where they converged, and "a finite objective" certifies neither. NLopt's return
    # code is the only thing that distinguishes them, so it is carried out of the algorithm
    # rather than discarded inside it: per restart in `trace.ret`, and for the polish here.
    polish_ret::Symbol           # :FTOL_REACHED, :MAXEVAL_REACHED, :EXCEPTION, :SKIPPED...
    polish_improved::Bool        # did the polish actually move the incumbent?
    n_eval_polish::Int           # evaluations spent in the polish alone
end

"""
    ret_class(ret) -> Symbol

Bucket an NLopt return code into `:converged`, `:limit` (a budget stopped it, not a
criterion), or `:other` (including failures and exceptions).

`:MAXEVAL_REACHED` and `:MAXTIME_REACHED` mean the search was cut off with its stopping
test unsatisfied. That is a legitimate answer to report, but it is not convergence, and a
run made mostly of those has a budget problem however good its objective looks.
"""
ret_class(ret::Symbol) =
    ret in (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED, :STOPVAL_REACHED) ? :converged :
    ret in (:MAXEVAL_REACHED, :MAXTIME_REACHED)                       ? :limit     :
    :other

"""
    ret_tally(result) -> Dict{Symbol,Int}

How many local searches ended in each NLopt return code. Reported next to the objective so
"converged" is a statement about the searches and not about the number they produced.
"""
ret_tally(r::TikTakResult) = begin
    d = Dict{Symbol,Int}()
    for row in r.trace; d[row.ret] = get(d, row.ret, 0) + 1; end
    d
end

"""
    sobol_points(lo, hi, N; skip_first = true) -> Vector{Vector{Float64}}

`N` Sobol' points in the box. The first point of a Sobol' sequence sits at a
corner or centre depending on the generator; it is skipped so the sample is not
anchored to a degenerate point.
"""
function sobol_points(lo::Vector{Float64}, hi::Vector{Float64}, N::Int; skip_first::Bool = true)
    s = SobolSeq(lo, hi)
    skip_first && Sobol.next!(s)
    return [copy(Sobol.next!(s)) for _ in 1:N]
end

"""
    _local_search(f, x0, lo, hi, alg, tol, maxeval, ftol_abs, xtol_rel, fstart)

One local search, with an `Opt` THAT BELONGS TO IT.

This is a free function, not a closure inside `tiktak`, and that is the whole point.
Julia's scoping rule is that assigning a name inside a nested function refers to the
enclosing local of the same name if one exists. `tiktak` assigns `opt` again for the
polishing search, so an `opt` created inside a closure there is not a fresh local at
all -- it is the enclosing binding, captured in a `Core.Box` and SHARED by every
concurrent restart. Verified in lowered code. With `batch > 1` the restarts then
configure and drive one another's optimizer, which is a data race in NLopt's C state
and a sufficient explanation for the silent exit-0 crash recorded in the header.

Returns the return code and any exception rather than discarding them: an exception
reaching here is a bug in `f` (the SMM objective already converts genuine model
failures into a finite penalty), so it is reported, not swallowed.
"""
function _local_search(f, x0::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64},
                       alg::Symbol, tol::Float64, maxeval::Int,
                       ftol_abs_::Float64, xtol_rel_::Float64, fstart::Float64)
    opt = Opt(alg, length(lo))
    lower_bounds!(opt, lo); upper_bounds!(opt, hi)
    ftol_rel!(opt, tol); maxeval!(opt, maxeval)
    # See the note on local_ftol_abs in the tiktak signature: without these two a
    # search whose optimum is 0 can never satisfy ftol_rel and always runs the full
    # maxeval.
    ftol_abs!(opt, ftol_abs_); xtol_rel!(opt, xtol_rel_)
    nev = Ref(0)
    min_objective!(opt, (x, g) -> (nev[] += 1; f(x)))
    try
        (floc, xloc, ret) = optimize(opt, x0)
        return (floc, xloc, nev[], ret, nothing)
    catch e
        # Information, not a fatal error -- but not invisible either.
        return (fstart, x0, nev[], :EXCEPTION, e)
    end
end

"""
    tiktak(f, lo, hi; kwargs...) -> TikTakResult

Minimise `f` over the box `[lo, hi]`.

`f` must return a finite `Float64`; signal an infeasible point with a large
finite penalty rather than `Inf` or an exception, so the local searches can still
form a descent direction away from it.

Keyword arguments
  N, Nstar            pre-testing points and local searches (paper: N* is 1-10% of N)
  theta_p/lo/hi       mixing-weight schedule, clamp((j/Nstar)^p, lo, hi)
  local_alg/tol/maxeval    local stage (default Nelder-Mead, 1e-3)
  local_ftol_abs/xtol_rel  absolute stopping rules for the local stage. Needed
                           because ftol_rel cannot stop a search whose optimum is
                           0 -- see the note in the signature. Do not set to 0.
  polish_alg/tol/maxeval   final polish (default BOBYQA, 1e-10)
  polish_ftol_abs          same, for the polish
  extra_seeds         points forced into the pre-testing pool
  stop_tol            early stop on |Z* - Z*_prev|; 0.0 disables
  on_sobol/on_local   callbacks for progress reporting. on_local also receives the
                      incumbent minimiser, so a caller can checkpoint after every restart.
"""
function tiktak(f, lo::Vector{Float64}, hi::Vector{Float64};
                N::Int = 1000, Nstar::Int = 50,
                theta_p::Float64 = 0.5, theta_lo::Float64 = 0.1, theta_hi::Float64 = 0.995,
                local_alg::Symbol = :LN_NELDERMEAD, local_tol::Float64 = 1e-3,
                local_maxeval::Int = 2000,
                # ftol_rel CANNOT STOP A SEARCH WHOSE OPTIMUM IS ZERO. It tests
                # |df| <= ftol_rel * |f|, so as f -> 0 the threshold goes to 0 with
                # it and the test stops being satisfiable; the search then runs to
                # maxeval every time. That is not hypothetical here: a just-identified
                # SMM (3 moments, 3 parameters) can drive Q to ~0, and measured
                # 2026-08-27 restart 1 reached Q ~ 0 at evaluation 61 and was still
                # going at 290 with the value unchanged, bound for all 2000.
                # At 15.5 s an evaluation that is 8 h per restart instead of 15 min.
                #
                # ftol_abs is scale-free and fires exactly where ftol_rel dies; xtol_rel
                # is the backstop that works at ANY value of f, because a collapsed
                # simplex means converged regardless of what f is worth there. Both are
                # far tighter than local_tol, so on a problem with a non-zero optimum
                # ftol_rel still stops the search first and behaviour is unchanged.
                local_ftol_abs::Float64 = 1e-10, local_xtol_rel::Float64 = 1e-8,
                polish_alg::Symbol = :LN_BOBYQA, polish_tol::Float64 = 1e-10,
                polish_maxeval::Int = 4000, polish_ftol_abs::Float64 = 1e-14,
                extra_seeds::Vector{Vector{Float64}} = Vector{Vector{Float64}}(),
                stop_tol::Float64 = 0.0,
                # RESUME. Pass `(seeds, f_sobol_best, Z, fZ, j_start)` to skip the global
                # stage entirely and re-enter the local stage at restart `j_start` with
                # `Z`/`fZ` as the incumbent. This is exact continuation, not a warm start:
                # the seeds are the ones the original pre-testing selected, so restart j
                # sees precisely the mixture it would have seen. Sobol points are cheap to
                # regenerate but their EVALUATIONS are not, which is why the seeds travel
                # in the checkpoint rather than being recomputed.
                resume::Union{Nothing,NamedTuple} = nothing,
                # Fires once, right after pre-testing, with the selected seeds. A caller
                # checkpoints them so a killed run can be resumed.
                on_seeds = (seeds, f_sobol_best) -> nothing,
                parallel::Bool = false,     # see the NLopt warning in the header
                # BATCH IS GATED ON `parallel`. It used to default off Threads.nthreads()
                # regardless, so `parallel = false` silently still ran the local stage on
                # Threads.@threads whenever Julia had more than one thread -- reproduced:
                # `julia -t 4` with parallel = false evaluated the objective on threads
                # 1..4. That is the one place this project must never be threaded.
                batch::Int = parallel ?
                             max(1, min(Threads.nthreads(), floor(Int, sqrt(Nstar)))) : 1,
                # PROCESS-level parallelism for the pre-testing stage. Pass `pmap`
                # (with workers added and the objective defined @everywhere) to spread
                # the N Sobol evaluations across worker PROCESSES. This is the safe way
                # to parallelise here: `parallel = true` above uses THREADS, and
                # NLopt.jl is not thread-safe in this project. Each worker process
                # carries its own NLopt state, so the hazard cannot arise.
                # Default `map` is plain serial and changes nothing.
                map_fn = map,
                on_sobol = (i, N, fx, best) -> nothing,
                # `best_x` is the incumbent MINIMISER, not just its value: without it a
                # caller cannot checkpoint, and a 20-hour run that dies has nothing to
                # resume from.
                on_local = (j, Nstar, theta, f_local, best, best_x) -> nothing)

    length(lo) == length(hi) || error("lo and hi must have the same length")
    all(lo .< hi) || error("every lo must be strictly below its hi")
    resume !== nothing || 1 <= Nstar <= N + length(extra_seeds) ||
        error("need 1 <= Nstar <= N + #extra_seeds")
    parallel || batch <= 1 || error("""
        batch = $batch was requested with parallel = false. The local stage would run on
        Threads.@threads while the caller believes threading is off. Pass parallel = true
        if that is what you want, or leave batch at 1.""")
    n_eval = 0

    # ---- global stage: pre-testing ------------------------------------------
    if resume !== nothing
        length(resume.seeds) == Nstar || error(
            "resume has $(length(resume.seeds)) seeds but Nstar = $Nstar")
        all(length(sd) == length(lo) for sd in resume.seeds) || error(
            "resume seeds have the wrong dimension for this box")
        1 <= resume.j_start <= Nstar + 1 || error(
            "resume.j_start = $(resume.j_start) is outside 1..$(Nstar + 1)")
    end
    cands = resume === nothing ? sobol_points(lo, hi, N) : Vector{Vector{Float64}}()
    resume === nothing && append!(cands, [clamp.(s, lo, hi) for s in extra_seeds])
    fs = Vector{Float64}(undef, length(cands))
    if resume !== nothing
        # nothing to do: the seeds already are the surviving pre-tested points
    elseif parallel
        done = Threads.Atomic{Int}(0)
        plock = ReentrantLock()
        best_so_far = Ref(Inf)
        Threads.@threads for i in eachindex(cands)
            v = f(cands[i]); fs[i] = v
            k = Threads.atomic_add!(done, 1) + 1
            lock(plock) do
                v < best_so_far[] && (best_so_far[] = v)
                on_sobol(k, length(cands), v, best_so_far[])
            end
        end
        n_eval += length(cands)
    elseif map_fn !== map
        # Distributed (or any user-supplied map). Evaluated as one batch, so the
        # progress callback fires afterwards rather than during -- a worker process
        # cannot write into this process's closure.
        fs .= map_fn(f, cands)
        n_eval += length(cands)
        best_so_far = Inf
        for (i, v) in pairs(fs)
            v < best_so_far && (best_so_far = v)
            on_sobol(i, length(cands), v, best_so_far)
        end
    else
        best_so_far = Inf
        for (i, x) in pairs(cands)
            fs[i] = f(x); n_eval += 1
            fs[i] < best_so_far && (best_so_far = fs[i])
            on_sobol(i, length(cands), fs[i], best_so_far)
        end
    end
    local seeds::Vector{Vector{Float64}}, f_sobol_best::Float64
    if resume === nothing
        order = sortperm(fs)                  # ascending: f(s_1) <= ... <= f(s_N*)
        seeds = [cands[k] for k in order[1:Nstar]]
        f_sobol_best = fs[order[1]]
        on_seeds(seeds, f_sobol_best)
    else
        seeds = resume.seeds
        f_sobol_best = resume.f_sobol_best
    end

    # ---- local stage --------------------------------------------------------
    Z, fZ = resume === nothing ? (copy(seeds[1]), f_sobol_best) :
                                 (copy(resume.Z), resume.fZ)
    trace = NamedTuple{(:j, :theta, :f_start, :f_local, :improved, :ret),
                       Tuple{Int,Float64,Float64,Float64,Bool,Symbol}}[]
    fZ_prev_distinct = Inf
    n_exception = 0

    # Batched: within a batch every restart reads the same Z*, so they can run
    # concurrently. batch == 1 is the sequential published algorithm.
    nb = max(1, batch)
    j = resume === nothing ? 1 : resume.j_start
    stop = false
    while j <= Nstar && !stop
        idx = j:min(j + nb - 1, Nstar)
        Zsnap, fZsnap = copy(Z), fZ            # frozen for the whole batch
        results = Vector{Any}(undef, length(idx))

        run_one = function (m)
            jj = idx[m]
            theta = jj == 1 ? 0.0 : clamp((jj / Nstar)^theta_p, theta_lo, theta_hi)
            x0 = clamp.((1 - theta) .* seeds[jj] .+ theta .* Zsnap, lo, hi)
            # The seed evaluation is guarded too. It was not before, so one throw here
            # killed the whole run at whichever restart hit it -- on a 20-hour estimation
            # that is the entire budget lost to one bad point.
            fstart, seed_err = try
                (f(x0), nothing)
            catch e
                (Inf, e)
            end
            if seed_err !== nothing
                results[m] = (jj = jj, theta = theta, fstart = Inf, floc = Inf,
                              xloc = x0, nev = 1, ret = :SEED_EXCEPTION, err = seed_err)
                return
            end
            floc, xloc, nev, ret, err = _local_search(f, x0, lo, hi, local_alg, local_tol,
                                                      local_maxeval, local_ftol_abs,
                                                      local_xtol_rel, fstart)
            results[m] = (jj = jj, theta = theta, fstart = fstart, floc = floc,
                          xloc = xloc, nev = nev + 1, ret = ret, err = err)
        end

        if parallel && nb > 1 && length(idx) > 1
            Threads.@threads for m in eachindex(idx); run_one(m); end
        else
            for m in eachindex(idx); run_one(m); end
        end

        for res in results                      # merge in seed order, deterministically
            n_eval += res.nev
            improved = isfinite(res.floc) && res.floc < fZ
            if improved
                fZ_prev_distinct = fZ
                Z, fZ = copy(res.xloc), res.floc
            end
            if res.err !== nothing
                # Reaching here means `f` threw something it did not classify. The SMM
                # objective already turns genuine model failures into a finite penalty and
                # re-throws real bugs, so this is a bug -- report it the moment it happens
                # rather than letting a 20-hour run finish and look converged.
                n_exception += 1
                @warn "local search $(res.jj) threw; the point was discarded" exception = res.err
            end
            push!(trace, (j = res.jj, theta = res.theta, f_start = res.fstart,
                          f_local = res.floc, improved = improved, ret = res.ret))
            on_local(res.jj, Nstar, res.theta, res.floc, fZ, Z)
            if stop_tol > 0 && improved && isfinite(fZ_prev_distinct) &&
               abs(fZ - fZ_prev_distinct) < stop_tol
                stop = true
            end
        end
        j += length(idx)
    end
    f_prepolish = fZ

    # ---- polishing ----------------------------------------------------------
    # `polish_opt`, not `opt`: a bare `opt` here is what boxed the local stage's
    # optimizer for years. Keep these names distinct.
    polish_opt = Opt(polish_alg, length(lo))
    lower_bounds!(polish_opt, lo); upper_bounds!(polish_opt, hi)
    ftol_rel!(polish_opt, polish_tol); xtol_rel!(polish_opt, polish_tol)
    maxeval!(polish_opt, polish_maxeval)
    ftol_abs!(polish_opt, polish_ftol_abs)   # as the local stage, one order tighter
    n_polish = Ref(0)
    min_objective!(polish_opt, (x, g) -> (n_eval += 1; n_polish[] += 1; f(x)))
    polish_ret = :NOT_RUN
    polish_improved = false
    try
        (fp, xp, retp) = optimize(polish_opt, Z)
        polish_ret = retp
        if isfinite(fp) && fp < fZ
            Z, fZ = copy(xp), fp
            polish_improved = true
        end
    catch e
        n_exception += 1
        polish_ret = :EXCEPTION
        @warn "polishing search threw; the pre-polish point was kept" exception = e
    end

    return TikTakResult(Z, fZ, n_eval, f_sobol_best, f_prepolish, trace, n_exception,
                        polish_ret, polish_improved, n_polish[])
end

"""
    tiktak_selftest(; verbose = true) -> Bool

Three checks, chosen so that each can only pass if a different part of the
algorithm is right:

  1. Sphere in 10d must hit the minimum to machine precision. Catches bad box
     scaling, a broken local stage, or a mishandled return value.
  2. Rastrigin in 3d, whose local minima form a dense lattice, must be solved
     exactly at a budget where that is achievable (N = 2000). Catches a global
     stage that is not actually exploring.
  3. TikTak must BEAT plain multistart on Rastrigin in 4d at an identical
     budget, where "plain multistart" is this same code with theta pinned to 0.
     This is the only check that tests the distinguishing feature -- the mixing
     of each seed with the incumbent best. Measured: 0.995 against 2.985.

Budgets matter more than they look. Rastrigin in 6d has on the order of 11^6
local minima in the box, so a small budget failing there is the function being
hard, not the optimizer being wrong.
"""
function tiktak_selftest(; verbose::Bool = true)
    rastrigin(x) = 10length(x) + sum(xi^2 - 10cos(2π*xi) for xi in x)
    sphere(x) = sum(x .^ 2)
    pass = true

    r1 = tiktak(sphere, fill(-10.0, 10), fill(10.0, 10); N = 200, Nstar = 10)
    ok1 = r1.f < 1e-12; pass &= ok1
    verbose && @printf("  sphere d=10          f = %.2e            [%s]\n",
                       r1.f, ok1 ? "PASS" : "FAIL")

    r2 = tiktak(rastrigin, fill(-5.12, 3), fill(5.12, 3); N = 2000, Nstar = 100)
    ok2 = r2.f < 1e-6; pass &= ok2
    verbose && @printf("  rastrigin d=3        f = %.2e            [%s]\n",
                       r2.f, ok2 ? "PASS" : "FAIL")

    rt = tiktak(rastrigin, fill(-5.12, 4), fill(5.12, 4); N = 1000, Nstar = 50)
    rm = tiktak(rastrigin, fill(-5.12, 4), fill(5.12, 4); N = 1000, Nstar = 50,
                theta_lo = 0.0, theta_hi = 0.0)          # theta == 0 => plain multistart
    ok3 = rt.f <= rm.f; pass &= ok3
    verbose && @printf("  rastrigin d=4        TikTak %.4f vs multistart %.4f  [%s]\n",
                       rt.f, rm.f, ok3 ? "PASS" : "FAIL")

    verbose && @printf("  tiktak_selftest: %s\n", pass ? "ALL PASS" : "FAILURES ABOVE")
    return pass
end
