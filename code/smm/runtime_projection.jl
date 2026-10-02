# =============================================================================
# runtime_projection.jl -- how long a run will take, as SCENARIOS (tiktak_fix_plan.md R1.2,
# tiktak_problems.md finding 19).
#
# Pure arithmetic, no model: run_smm.jl calls it after timing one evaluation, and
# tools/test_runtime_projection.jl tests it. It replaces a fixed "165 evaluations per restart"
# (measured 2026-08-27 on an older, smaller problem) that made a 1,000-evaluation cap look like
# a few hours: mu08's restarts used 268-500 evaluations, and E1-P allows 1,000 at ~42 s each.
#
# Two scenarios, never one number:
#   cap        every restart and the polish use their whole evaluation budget: the most the
#              run can take (retried work after a lost worker comes on top)
#   empirical  per-restart evaluation counts observed in a compatible earlier run, given
#              explicitly (run_smm.jl --runtime-evals), cut at this run's cap
# A projection is not a deadline: a driver's timeout is the enforced limit.
#
# THE LOCAL STAGE is simulated the way the scheduler hands out work: restart j goes to the
# first free worker, in order. With restarts of equal length T on P workers that gives
#   serial (P = 1)       K T
#   first_alone          T + ceil((K-1)/P) T     restart 1 alone, the others wait for it
#   immediate_mixed      ceil(K/P) T             the pool fills at once
# so starting at once saves a whole round only when K fills the rounds: K = 21 on 20 workers
# takes 2T either way, K = 20 takes T instead of 2T.
# =============================================================================

"""
    local_wall(lengths, P, bootstrap) -> Float64

Wall time of the local stage when restart j takes `lengths[j]` (any time unit) on `P`
workers, each restart handed to the first free worker in order. `:first_alone` holds every
worker until restart 1 is done; `:immediate_mixed` does not. A serial stage is P = 1.
"""
function local_wall(lengths::AbstractVector{<:Real}, P::Int, bootstrap::Symbol)
    P >= 1 || throw(ArgumentError("local_wall: P = $P workers"))
    bootstrap in (:first_alone, :immediate_mixed) || throw(ArgumentError("local_wall: unknown bootstrap :$bootstrap"))
    isempty(lengths) && return 0.0
    free = zeros(P)                        # when each worker is next free
    first = 1
    if bootstrap === :first_alone
        free .= lengths[1]                 # restart 1 alone: every worker is free only when it is done
        first = 2
    end
    for j in first:length(lengths)
        k = argmin(free)                   # the first free worker (the lowest index on a tie)
        free[k] += lengths[j]
    end
    return maximum(free)
end

"""
    restart_evals(K, cap; observed = Int[]) -> Vector{Int}

Evaluations per restart for K restarts under an NLopt cap of `cap` (each restart also
evaluates its start once: at most cap + 1). Without `observed`: the cap scenario. With it:
restart j takes observed[mod1(j, n)], cut at cap + 1 -- the counts of a compatible earlier
run, reused in order.
"""
function restart_evals(K::Int, cap::Int; observed::AbstractVector{<:Integer} = Int[])
    K >= 0 && cap >= 1 || throw(ArgumentError("restart_evals: K = $K, cap = $cap"))
    all(>=(1), observed) || throw(ArgumentError("restart_evals: observed counts must be >= 1"))
    isempty(observed) && return fill(cap + 1, K)
    return [min(observed[mod1(j, length(observed))], cap + 1) for j in 1:K]
end

"""
    project_runtime(; t_eval, n_pretest, pretest_workers, evals, local_workers, bootstrap,
                    polish_evals, refine_evals, n_procs) -> NamedTuple

Seconds per stage for one scenario. `t_eval` is seconds per evaluation; `evals` the
evaluations of each restart (`restart_evals`); `local_workers` the P of the local stage (1 when
serial); `polish_evals` and `refine_evals` 0 when that stage does not run; `n_procs` the
processes the run holds (workers + master).

Returns `pretest`, `local_`, `polish`, `refine`, `wall` (their sum) and two CPU totals:
`busy` -- every evaluation once, the CPU actually used -- and `reserved` = wall x n_procs, the
cores held whether computing or idle. An idle worker holds a core but does not load the machine.
"""
function project_runtime(; t_eval::Real, n_pretest::Integer, pretest_workers::Integer,
                         evals::AbstractVector{<:Integer}, local_workers::Integer, bootstrap::Symbol,
                         polish_evals::Integer, refine_evals::Integer, n_procs::Integer)
    t_eval >= 0 && isfinite(t_eval) || throw(ArgumentError("project_runtime: t_eval = $t_eval"))
    min(n_pretest, polish_evals, refine_evals) >= 0 || throw(ArgumentError("project_runtime: negative evaluation count"))
    pre = cld(n_pretest, max(pretest_workers, 1)) * t_eval
    loc = local_wall(evals .* float(t_eval), Int(local_workers), bootstrap)
    pol = polish_evals * t_eval
    ref = refine_evals * t_eval
    wall = pre + loc + pol + ref
    busy = (n_pretest + sum(evals; init = 0) + polish_evals + refine_evals) * t_eval
    return (pretest = pre, local_ = loc, polish = pol, refine = ref, wall = wall, busy = busy, reserved = wall * n_procs)
end

"""
    round_advice(K, P, bootstrap) -> Union{Nothing,NamedTuple}

`nothing` when the restarts fill whole rounds of P workers; otherwise how many workers the
last round uses (`last`) and the nearest restart counts that fill the rounds: `more` (same
wall time, more restarts) and `fewer` (one round less; 0 if none).
"""
function round_advice(K::Int, P::Int, bootstrap::Symbol)
    P <= 1 && return nothing
    rest = bootstrap === :first_alone ? K - 1 : K         # the restarts dealt out in rounds of P
    rest <= 0 && return nothing
    r = rest % P
    r == 0 && return nothing
    base = bootstrap === :first_alone ? 1 : 0
    return (last = r, more = base + cld(rest, P) * P, fewer = base + fld(rest, P) * P)
end

"A duration for the log: minutes up to two hours, then hours."
fmt_duration(s::Real) = s < 7200 ? string(round(s / 60; digits = 1), " min") : string(round(s / 3600; digits = 1), " h")
