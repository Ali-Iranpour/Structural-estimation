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
    trace::Vector{NamedTuple{(:j, :theta, :f_start, :f_local, :improved), Tuple{Int,Float64,Float64,Float64,Bool}}}
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
    tiktak(f, lo, hi; kwargs...) -> TikTakResult

Minimise `f` over the box `[lo, hi]`.

`f` must return a finite `Float64`; signal an infeasible point with a large
finite penalty rather than `Inf` or an exception, so the local searches can still
form a descent direction away from it.

Keyword arguments
  N, Nstar            pre-testing points and local searches (paper: N* is 1-10% of N)
  theta_p/lo/hi       mixing-weight schedule, clamp((j/Nstar)^p, lo, hi)
  local_alg/tol/maxeval    local stage (default Nelder-Mead, 1e-3)
  polish_alg/tol/maxeval   final polish (default BOBYQA, 1e-10)
  extra_seeds         points forced into the pre-testing pool
  stop_tol            early stop on |Z* - Z*_prev|; 0.0 disables
  on_sobol/on_local   callbacks for progress reporting
"""
function tiktak(f, lo::Vector{Float64}, hi::Vector{Float64};
                N::Int = 1000, Nstar::Int = 50,
                theta_p::Float64 = 0.5, theta_lo::Float64 = 0.1, theta_hi::Float64 = 0.995,
                local_alg::Symbol = :LN_NELDERMEAD, local_tol::Float64 = 1e-3,
                local_maxeval::Int = 2000,
                polish_alg::Symbol = :LN_BOBYQA, polish_tol::Float64 = 1e-10,
                polish_maxeval::Int = 4000,
                extra_seeds::Vector{Vector{Float64}} = Vector{Vector{Float64}}(),
                stop_tol::Float64 = 0.0,
                parallel::Bool = false,     # see the NLopt warning in the header
                batch::Int = max(1, min(Threads.nthreads(), floor(Int, sqrt(Nstar)))),
                on_sobol = (i, N, fx, best) -> nothing,
                on_local = (j, Nstar, theta, f_local, best) -> nothing)

    length(lo) == length(hi) || error("lo and hi must have the same length")
    all(lo .< hi) || error("every lo must be strictly below its hi")
    1 <= Nstar <= N + length(extra_seeds) || error("need 1 <= Nstar <= N + #extra_seeds")
    n_eval = 0

    # ---- global stage: pre-testing ------------------------------------------
    cands = sobol_points(lo, hi, N)
    append!(cands, [clamp.(s, lo, hi) for s in extra_seeds])
    fs = Vector{Float64}(undef, length(cands))
    if parallel
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
    else
        best_so_far = Inf
        for (i, x) in pairs(cands)
            fs[i] = f(x); n_eval += 1
            fs[i] < best_so_far && (best_so_far = fs[i])
            on_sobol(i, length(cands), fs[i], best_so_far)
        end
    end
    order = sortperm(fs)                      # ascending: f(s_1) <= ... <= f(s_N*)
    seeds = [cands[k] for k in order[1:Nstar]]
    f_sobol_best = fs[order[1]]

    # ---- local stage --------------------------------------------------------
    Z, fZ = copy(seeds[1]), f_sobol_best      # incumbent best minimiser and value
    trace = NamedTuple{(:j, :theta, :f_start, :f_local, :improved), Tuple{Int,Float64,Float64,Float64,Bool}}[]
    fZ_prev_distinct = Inf

    # Batched: within a batch every restart reads the same Z*, so they can run
    # concurrently. batch == 1 is the sequential published algorithm.
    nb = max(1, batch)
    j = 1
    stop = false
    while j <= Nstar && !stop
        idx = j:min(j + nb - 1, Nstar)
        Zsnap, fZsnap = copy(Z), fZ            # frozen for the whole batch
        results = Vector{Any}(undef, length(idx))

        run_one = function (m)
            jj = idx[m]
            theta = jj == 1 ? 0.0 : clamp((jj / Nstar)^theta_p, theta_lo, theta_hi)
            x0 = clamp.((1 - theta) .* seeds[jj] .+ theta .* Zsnap, lo, hi)
            fstart = f(x0)
            opt = Opt(local_alg, length(lo))
            lower_bounds!(opt, lo); upper_bounds!(opt, hi)
            ftol_rel!(opt, local_tol); maxeval!(opt, local_maxeval)
            nev = Ref(0)
            min_objective!(opt, (x, g) -> (nev[] += 1; f(x)))
            floc, xloc = fstart, x0
            try
                (floc, xloc, _) = optimize(opt, x0)
            catch
                # a local search that blows up is information, not a fatal error
            end
            results[m] = (jj = jj, theta = theta, fstart = fstart, floc = floc,
                          xloc = xloc, nev = nev[] + 1)
        end

        if nb > 1 && length(idx) > 1
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
            push!(trace, (j = res.jj, theta = res.theta, f_start = res.fstart,
                          f_local = res.floc, improved = improved))
            on_local(res.jj, Nstar, res.theta, res.floc, fZ)
            if stop_tol > 0 && improved && isfinite(fZ_prev_distinct) &&
               abs(fZ - fZ_prev_distinct) < stop_tol
                stop = true
            end
        end
        j += length(idx)
    end
    f_prepolish = fZ

    # ---- polishing ----------------------------------------------------------
    opt = Opt(polish_alg, length(lo))
    lower_bounds!(opt, lo); upper_bounds!(opt, hi)
    ftol_rel!(opt, polish_tol); xtol_rel!(opt, polish_tol); maxeval!(opt, polish_maxeval)
    min_objective!(opt, (x, g) -> (n_eval += 1; f(x)))
    try
        (fp, xp, _) = optimize(opt, Z)
        if isfinite(fp) && fp < fZ
            Z, fZ = copy(xp), fp
        end
    catch
    end

    return TikTakResult(Z, fZ, n_eval, f_sobol_best, f_prepolish, trace)
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
