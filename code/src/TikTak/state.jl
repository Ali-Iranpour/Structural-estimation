# =============================================================================
# state.jl -- the authoritative state of a run after seed selection.
#
# ONE mutable RunState, changed only through the functions in this file and only by the
# process that owns the run (the master). The serial path and the asynchronous scheduler
# share them, so both merge, count and checkpoint identically (plan J1).
#
# 2026-09-28 (follow-ups 1 and 4): the state is VALIDATED when it is restored and before a
# stage is declared complete (`state_violations`), solver settings live in epochs
# (SettingsEpoch) so a record keeps the settings it ran with, and every attempt's work is
# accounted once under its AttemptKey.
# =============================================================================

"""
    RunState

Everything a checkpoint must hold to continue the search exactly (plan step 4): the seeds
and the schedule denominator K, the incumbent with its origin and verification, every
committed restart, the next restart index, the lifetime counters, the polish, and the
identities the run was started under.
"""
mutable struct RunState
    run_id::String
    lo::Vector{Float64}
    hi::Vector{Float64}
    cfg::TikTakConfig                   # the CURRENT configuration (its solver settings = the last epoch's)
    objective_id::String
    objective_fields::Dict{String,Any}
    optimizer_id::String
    optimizer_fields::Dict{String,Any}
    nstar_requested::Int
    K::Int                              # effective restarts = the schedule denominator
    seeds::Vector{Vector{Float64}}
    seed_f::Vector{Float64}
    seed_origin::Vector{Symbol}         # :sobol | :supplied | :legacy_import
    seed_candidate::Vector{Int}
    f_sobol_best::Float64
    pretest::PretestSummary
    inc::Incumbent
    records::Vector{RestartRecord}
    inflight::Dict{Int,InFlight}        # dispatched, not committed (asynchronous stage, or a serial replay)
    next_j::Int                         # the next restart index not yet dispatched
    commit_seq::Int
    dispatch_seq::Int
    checkpoint_seq::Int
    fZ_prev_distinct::Float64           # stop_tol state
    stopped_early::Bool
    n_exception::Int
    polish::PolishRecord
    f_prepolish::Float64                # the incumbent value when the local stage completed (NaN before)
    evals_pretest::Int                  # lifetime counters of COMMITTED work
    evals_local::Int
    evals_polish::Int
    evals_abandoned::Int                # known evaluations of attempts that executed but were never committed
    jobs_lost::Int                      # attempts lost with a worker: their evaluations are UNKNOWN
    attempts_unknown::Int               # attempts that failed, were interrupted or still ran at a stop: UNKNOWN
    pretest_lost::Int                   # pre-testing evaluations dispatched whose value never arrived
    duplicates::Int                     # repeated deliveries of an attempt already accounted (zero work)
    retries::Int                        # re-dispatches of a job after its worker was lost
    evals_segment::Int                  # committed in this process
    counts_complete::Bool               # false after a legacy import (earlier counts unknown)
    stage::Symbol                       # :local | :local_complete | :complete
    status::Symbol                      # :running | :paused | :complete | :failed
    resume_semantics::Symbol
    purpose::String                     # the preset label: smoke | integration | pilot | production | custom
    epochs::Vector{SettingsEpoch}       # solver settings over the run's life; the last one is in force
    accounted::Dict{AttemptKey,Symbol}  # every attempt whose work is accounted, and how (follow-up 4)
    legacy_through::Int                 # restarts 1..legacy_through ran before a legacy import (0: none)
    segments::Vector{Segment}
    events::Vector{RunEvent}
end

_now() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")

function add_event!(st::RunState, kind::Symbol, detail::AbstractString = "")
    push!(st.events, RunEvent(length(st.events) + 1, _now(), kind, String(detail)))
    return st
end

function start_segment!(st::RunState, mode::Symbol, workers::Int, stop_after::Int, note::AbstractString)
    push!(st.segments, Segment(length(st.segments) + 1, _now(), mode, workers, stop_after, st.next_j, String(note)))
    return st
end

"A fresh run id: time stamp plus a random suffix (unique across runs started the same second)."
new_run_id() = Dates.format(Dates.now(), "yyyymmdd_HHMMSS") * "_" * string(rand(UInt32); base = 16, pad = 8)

"The settings epoch in force now (new dispatches use it)."
current_epoch(st::RunState) = length(st.epochs)

"The first settings epoch of a run: the configuration it started with."
first_epoch(cfg::TikTakConfig, opt_id::String) =
    SettingsEpoch(1, cfg.local_, cfg.polish, cfg.normalize, cfg.skip_polish, opt_id, 1, "start")

"""
    restart_job(st, j, attempt, x0buf; objective_key, master, progress_every) -> RestartJob

The job for restart j, mixed with the CURRENT incumbent under the CURRENT settings epoch: x0
is computed in the scratch buffer and COPIED into the job, so the buffer can be reused
without touching a dispatched start (plan J3).
"""
function restart_job(st::RunState, j::Int, attempt::Int, x0buf::Vector{Float64};
                     objective_key::Symbol = :default, master::Int = 1, progress_every::Float64 = Inf)
    theta = theta_at(st.cfg, j, st.K)
    make_start!(x0buf, st.seeds[j], st.inc.x, theta, st.lo, st.hi)
    s = st.cfg.local_
    scale = s.step_schedule === :theta_shrink ? max(s.step_min, 1 - theta) : 1.0
    return RestartJob(st.run_id, :local, j, attempt, theta, copy(x0buf), st.inc.version, true, NaN,
                      s, st.lo, st.hi, st.cfg.on_error, objective_key, master, progress_every,
                      scale, st.cfg.normalize, current_epoch(st))
end

"""
    next_attempt(st, stage, j) -> Int

The attempt number of the next dispatch of restart j (`stage = :local`) or of the polish
(`:polish`, j = 0): one more than any attempt of it already accounted -- a restart re-run
after a failure is a new attempt, never a second account of the failed one.
"""
next_attempt(st::RunState, stage::Symbol, j::Int) =
    1 + maximum((k.attempt for k in keys(st.accounted) if k.stage === stage && k.j == j); init = 0)

"The polish job: BOBYQA from the retained point, whose value is already known."
polish_job(st::RunState; objective_key::Symbol = :default, master::Int = 1, progress_every::Float64 = Inf) =
    RestartJob(st.run_id, :polish, 0, next_attempt(st, :polish, 0), 1.0, copy(st.inc.x), st.inc.version, false,
               st.inc.f, st.cfg.polish, st.lo, st.hi, st.cfg.on_error, objective_key, master, progress_every,
               1.0, st.cfg.normalize, current_epoch(st))

"""
    redispatch(job, master, objective_key, progress_every) -> RestartJob

The same job again as the next attempt: same restart, theta, x0, incumbent version, solver
settings and settings epoch. A replayed or retried job is never rebuilt from a newer
incumbent or under newer settings.
"""
redispatch(job::RestartJob, master::Int, objective_key::Symbol, progress_every::Float64) =
    RestartJob(job.run_id, job.stage, job.j, job.attempt + 1, job.theta, copy(job.x0), job.incumbent_version,
               job.eval_start, job.f_start_known, job.settings, job.lo, job.hi, job.on_error,
               objective_key, master, progress_every, job.step_scale, job.normalize, job.epoch)

"The accounting key of a job's attempt."
attempt_key(job::RestartJob) = AttemptKey(job.run_id, job.stage, job.j, job.attempt)

"""
    commit_restart!(st, job, r) -> RestartRecord

Merge one finished restart into the state: count its evaluations, apply the merge rule,
append the record. `job` is the job as dispatched; `r` must answer it (same run, restart
and attempt) -- a result for any other job, or an attempt whose work is already accounted,
is a programming error here: the scheduler filters duplicates and stale attempts first.
"""
function commit_restart!(st::RunState, job::RestartJob, r::RestartResult; dispatch_seq::Int = 0,
                         commits_at_dispatch::Int = st.commit_seq)
    (r.run_id == st.run_id && r.j == job.j && r.attempt == job.attempt && r.stage === :local) ||
        error("commit_restart!: result (run $(r.run_id), j $(r.j), attempt $(r.attempt)) does not answer " *
              "job (run $(st.run_id), j $(job.j), attempt $(job.attempt))")
    any(rec -> rec.j == job.j, st.records) && error("commit_restart!: restart $(job.j) is already committed")
    key = attempt_key(job)
    haskey(st.accounted, key) && error("commit_restart!: the work of $(key) is already accounted ($(st.accounted[key]))")
    st.accounted[key] = :committed
    st.evals_local += r.n_eval
    st.evals_segment += r.n_eval
    f_before = st.inc.f
    action, st.inc = merge_candidate(st.inc, r.x, r.f, r.ret, :local, r.j, r.attempt, st.lo, st.hi,
                                     st.cfg, st.objective_id, solver_summary(job.settings))
    if action === :improved
        st.fZ_prev_distinct = f_before
        if st.cfg.stop_tol > 0 && isfinite(f_before) && abs(st.inc.f - f_before) < st.cfg.stop_tol
            st.stopped_early = true
        end
    end
    if r.ret in (:EXCEPTION, :SEED_EXCEPTION)
        # Reaching here means `f` threw something it did not classify under on_error =
        # :discard. The SMM objective turns genuine model failures into a finite penalty and
        # re-throws real bugs, so this is a bug -- reported the moment it happens.
        st.n_exception += 1
        @warn "local search $(r.j) threw; the point was discarded" error = r.error
    end
    st.commit_seq += 1
    rec = RestartRecord(job.j, job.attempt, job.theta, job.x0, job.incumbent_version, r.f_start,
                        copy(r.x), r.f, r.n_eval, r.ret, action, st.inc.version, r.worker, r.elapsed,
                        st.commit_seq, dispatch_seq, commits_at_dispatch, r.error, false, job.epoch)
    push!(st.records, rec)
    return rec
end

"Merge the polish into the state."
function commit_polish!(st::RunState, job::RestartJob, r::RestartResult)
    key = attempt_key(job)
    haskey(st.accounted, key) && error("commit_polish!: the work of $(key) is already accounted")
    st.accounted[key] = :committed
    st.evals_polish += r.n_eval
    st.evals_segment += r.n_eval
    improved = false
    if r.ret === :EXCEPTION
        st.n_exception += 1
        @warn "polishing search threw; the pre-polish point was kept" error = r.error
    else
        action, st.inc = merge_candidate(st.inc, r.x, r.f, r.ret, :polish, 0, r.attempt, st.lo, st.hi,
                                         st.cfg, st.objective_id, solver_summary(job.settings))
        improved = action === :improved
    end
    st.polish = PolishRecord(true, r.ret, improved, r.n_eval, job.epoch)
    return st
end

"""
    account_uncommitted!(st, kind, j, attempt, result; stage = :local) -> Symbol

The accounting of a message about an attempt that is NOT the one in flight -- a second
delivery, a superseded attempt, a result of another run (follow-up 4). Returns what was done:

  :duplicate   the attempt's work is already accounted: zero added, `duplicates` counted
  :abandoned   an attempt that EXECUTED but was superseded, first seen now: its known
               evaluations are added once to `evals_abandoned`
  :other_run   a result of another run: nothing added
  :no_result   a failure message without a result: nothing to add
"""
function account_uncommitted!(st::RunState, j::Int, attempt::Int, r::Union{Nothing,RestartResult};
                              stage::Symbol = :local)
    r === nothing && return :no_result
    r.run_id == st.run_id || return :other_run
    key = AttemptKey(st.run_id, stage, j, attempt)
    if haskey(st.accounted, key)
        st.duplicates += 1
        return :duplicate
    end
    st.accounted[key] = :abandoned
    st.evals_abandoned += r.n_eval
    return :abandoned
end

"Record an attempt whose evaluations are UNKNOWN (`how`: :lost, :failed, :interrupted, :running_at_stop)."
function account_unknown!(st::RunState, key::AttemptKey, how::Symbol)
    haskey(st.accounted, key) && return false
    st.accounted[key] = how
    how === :lost ? (st.jobs_lost += 1) : (st.attempts_unknown += 1)
    return true
end

"The per-restart trace row the result has always carried, from a committed record."
trace_row(rec::RestartRecord) = (j = rec.j, theta = rec.theta, f_start = rec.f_start, f_local = rec.f_local,
                                 improved = rec.action === :improved, ret = rec.ret)

"Planned restarts that are committed (a committed restart is never re-run)."
n_committed(st::RunState) = length(st.records)

# -----------------------------------------------------------------------------
# Consistency (follow-up 1.2)
# -----------------------------------------------------------------------------
"""
    state_violations(st; complete = <the state says so>) -> Vector{String}

Every invariant the state violates, one readable line each (empty: consistent). Checked when
a checkpoint is restored (a violation refuses the resume) and before the local stage or the
run is declared complete (a violation fails the run). `complete = true` adds the completion
invariants: nothing in flight, and every planned restart committed -- unless the run stopped
early under its recorded stop_tol policy, which is intended and not missing work.

  * committed restart IDs are unique and in 1..K; no restart is both committed and in flight
  * the seeds, their values and origins have K entries of the box's dimension
  * the incumbent is a point of the box with a valid value
  * the dispatch cursor: every restart below next_j is committed or in flight (or ran before
    a legacy import), and nothing at or beyond it is
  * every record and job names an existing settings epoch
"""
function state_violations(st::RunState; complete::Bool = st.status === :complete || st.stage in (:local_complete, :complete))
    v = String[]
    n = length(st.lo)
    K = st.K
    length(st.hi) == n || push!(v, "the box has $(n) lower and $(length(st.hi)) upper bounds")
    1 <= K <= max(st.nstar_requested, 1) || push!(v, "K = $K is outside 1..nstar_requested = $(st.nstar_requested)")
    (length(st.seeds) == K && length(st.seed_f) == K && length(st.seed_origin) == K && length(st.seed_candidate) == K) ||
        push!(v, "the seed arrays do not all have K = $K entries")
    all(s -> length(s) == n, st.seeds) || push!(v, "a seed does not have the box's dimension $n")
    (length(st.inc.x) == n && all(isfinite, st.inc.x) && all(st.lo .<= st.inc.x .<= st.hi)) ||
        push!(v, "the incumbent is not a point of the box")
    valid_value(st.inc.f, st.cfg.invalid_value) || push!(v, "the incumbent value $(st.inc.f) is not a valid objective value")
    js = Int[r.j for r in st.records]
    dup = sort(unique(j for j in js if count(==(j), js) > 1))
    isempty(dup) || push!(v, "restart(s) $(join(dup, ", ")) committed more than once")
    bad = sort(unique(j for j in js if !(1 <= j <= K)))
    isempty(bad) || push!(v, "committed restart(s) $(join(bad, ", ")) outside 1..$K")
    committed = Set(js)
    for (j, fl) in st.inflight
        fl.job.j == j || push!(v, "the in-flight entry for restart $j holds restart $(fl.job.j)")
        1 <= j <= K || push!(v, "restart $j in flight is outside 1..$K")
        j in committed && push!(v, "restart $j is both committed and in flight")
    end
    1 <= st.next_j <= K + 1 || push!(v, "the dispatch cursor next_j = $(st.next_j) is outside 1..$(K + 1)")
    passed = [j for j in 1:min(st.next_j - 1, K) if !(j in committed) && !haskey(st.inflight, j) && j > st.legacy_through]
    isempty(passed) || push!(v, "restart(s) $(join(passed, ", ")) were passed by the dispatch cursor (next_j = " *
                             "$(st.next_j)) but are neither committed nor in flight")
    ahead = sort([j for j in union(committed, keys(st.inflight)) if j >= st.next_j])
    isempty(ahead) || push!(v, "restart(s) $(join(ahead, ", ")) are committed or in flight at or beyond the cursor next_j = $(st.next_j)")
    st.commit_seq == length(st.records) || push!(v, "commit_seq = $(st.commit_seq) but $(length(st.records)) records")
    ne = length(st.epochs)
    ne >= 1 || push!(v, "no settings epoch")
    all(r -> 1 <= r.epoch <= ne, st.records) || push!(v, "a record names a settings epoch that does not exist")
    all(fl -> 1 <= fl.job.epoch <= ne, values(st.inflight)) || push!(v, "an in-flight job names a settings epoch that does not exist")
    st.stopped_early && !(st.cfg.stop_tol > 0) && push!(v, "the run is marked stopped early but has no stop_tol policy")
    if complete
        isempty(st.inflight) || push!(v, "the local stage is complete with restart(s) " *
                                      "$(join(sort(collect(keys(st.inflight))), ", ")) still in flight")
        if !st.stopped_early
            missing_ = [j for j in 1:K if !(j in committed) && j > st.legacy_through]
            isempty(missing_) || push!(v, "the local stage is complete but restart(s) $(join(missing_, ", ")) of " *
                                          "1..$K were never committed")
        end
    end
    return v
end

"Raised when the run state breaks an invariant at a point where it must hold (a bug, never an input problem)."
struct StateInvariantError <: Exception
    msg::String
end
Base.showerror(io::IO, e::StateInvariantError) = print(io, "TikTak state invariant violated: ", e.msg)

"""
    continuation_semantics(st) -> Symbol

How the run as a whole relates to an uninterrupted run (follow-up 1.3), from its entire
history: a legacy import, an explicitly allowed optimizer change, a segment with several
local workers, or a replayed or retried job each make it something other than the exact
sequential path. One segment is :fresh.
"""
function continuation_semantics(st::RunState)
    any(e -> e.kind === :legacy_import, st.events) && return :legacy_import
    length(st.segments) <= 1 && return :fresh
    any(e -> e.kind === :optimizer_change, st.events) && return :changed_optimizer
    (any(s -> s.mode === :async_process && s.workers > 1, st.segments) ||
     any(e -> e.kind in (:replay, :retry), st.events)) && return :async_continuation
    return :serial_exact
end

"Refresh `st.resume_semantics` from the history (an already-complete resume keeps its label)."
refresh_semantics!(st::RunState) =
    (st.resume_semantics === :already_complete || (st.resume_semantics = continuation_semantics(st)); st)

"""
    search_budget_complete(result) -> Bool

Did the planned search run to its end, VALIDATED from the state rather than read from the
status symbol (follow-up 1.6): status :complete or :stopped_early (a recorded stop_tol
policy), nothing in flight, and no violated invariant -- in particular every planned
restart committed.
"""
search_budget_complete(r) = r.status in (:complete, :stopped_early) && isempty(r.inflight) && isempty(r.violations)

"""
    work_known(result) -> Bool

Is the total of evaluations actually executed known? False after a legacy import, or when
any attempt was lost, failed, interrupted, left running, or a pre-testing value never arrived.
"""
work_known(r) = r.n_eval_complete && r.accounting.jobs_lost == 0 && r.accounting.attempts_unknown == 0 &&
                r.accounting.pretest_lost == 0

"""
    build_result(st; state_path) -> TikTakResult

The result of a run in whatever state it is -- complete, stopped early or paused. For a
paused run `polish_ret` is :NOT_RUN and `x`/`f` are the incumbent so far.
"""
function build_result(st::RunState; state_path::String = "")
    refresh_semantics!(st)
    o = st.inc.origin
    trace = TraceRow[trace_row(rec) for rec in sort(st.records; by = rec -> rec.j)]
    polish_ret = st.polish.done ? st.polish.ret : (st.cfg.skip_polish && st.stage === :complete ? :SKIPPED : :NOT_RUN)
    fpp = st.stage === :local ? st.inc.f : st.f_prepolish
    status = st.status === :paused ? :paused : st.status === :failed ? :failed :
             st.stopped_early ? :stopped_early : :complete
    n_eval = st.evals_pretest + st.evals_local + st.evals_polish
    viol = state_violations(st; complete = status in (:complete, :stopped_early))
    return TikTakResult(copy(st.inc.x), st.inc.f, n_eval, st.f_sobol_best, fpp, trace, st.n_exception,
                        polish_ret, st.polish.improved, st.polish.n_eval,
                        o.stage, o.restart, o.ret,
                        st.nstar_requested, st.K, st.K, st.pretest, st.cfg,
                        st.inc, st.objective_id, status,
                        copy(st.records), st.evals_segment, st.counts_complete, st.resume_semantics,
                        st.run_id, state_path, st.purpose,
                        (pretest = st.evals_pretest, local_ = st.evals_local, polish = st.evals_polish,
                         abandoned_known = st.evals_abandoned, jobs_lost = st.jobs_lost,
                         attempts_unknown = st.attempts_unknown, pretest_lost = st.pretest_lost,
                         duplicates_ignored = st.duplicates, retries = st.retries, reused = st.pretest.reused),
                        sort(collect(keys(st.inflight))), viol, copy(st.epochs))
end
