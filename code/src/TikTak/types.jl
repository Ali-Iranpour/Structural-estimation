# =============================================================================
# types.jl -- the result types of the TikTak module.
# =============================================================================

"One row of the per-restart trace: seed index, theta, start value, local value, whether
it replaced the incumbent, and that search's own NLopt return code."
const TraceRow = NamedTuple{(:j, :theta, :f_start, :f_local, :improved, :ret),
                            Tuple{Int,Float64,Float64,Float64,Bool,Symbol}}

"""
    CandidateOrigin

WHERE the retained point came from (plan step 3): the stage and search that first supplied
it, that search's own return code, and the objective its value was computed on.

  stage     :sobol | :supplied (pre-testing), :local, :polish, :refine,
            :reevaluated (a point carried to another objective and evaluated there),
            :legacy_import (resumed from a checkpoint that did not record its origin)
  restart   the restart index j for :local, else 0
  attempt   1, or the re-dispatch number of an asynchronous job
  candidate the pre-testing candidate index for :sobol / :supplied, else 0
  ret       the NLopt code of THAT search; :SOBOL_ONLY / :SUPPLIED for a pre-tested point,
            :REEVALUATED, and :UNKNOWN when a legacy checkpoint did not record it
"""
struct CandidateOrigin
    stage::Symbol
    restart::Int
    attempt::Int
    candidate::Int
    ret::Symbol
    objective_id::String
end

"""
    Verification

A LATER solve that confirmed the retained point without improving it: it returned the same
point (box-normalized max-norm distance <= `coord_tol`) with a consistent value and a
converged return code, on the objective `objective_id`. Equal Q at a DIFFERENT point is not
a verification: a weakly identified model can have many such points.
"""
struct Verification
    status::Symbol          # :none | :verified
    stage::Symbol
    restart::Int
    attempt::Int
    ret::Symbol
    x_tested::Vector{Float64}
    f_tested::Float64
    distance::Float64
    coord_tol::Float64
    objective_id::String
    solver::String
end
const NO_VERIFICATION = Verification(:none, :none, 0, 0, :NONE, Float64[], NaN, NaN, NaN, "", "")

"""
    Incumbent

The retained point, its value, its origin and any verification. `version` counts
replacements of the point (a verification does not change it), so an asynchronous job can
record which incumbent it was mixed with.
"""
struct Incumbent
    x::Vector{Float64}
    f::Float64
    version::Int
    origin::CandidateOrigin
    verification::Verification
end

"""
    PretestSummary

What the pre-testing stage evaluated, counted by kind, so "N = 1000" is never mistaken for
1000 useful exploratory points (plan step 2.4). Caller-supplied points (`extra_seeds`) are
counted apart from the Sobol' coverage.

  attempted   Sobol' draws evaluated for the pool (the design's attempted count)
  valid       of those: finite and below `invalid_value`
  invalid     finite but >= `invalid_value` (the SMM penalty convention)
  nonfinite   NaN or +-Inf (never a seed)
  errors      threw and were discarded (`on_error = :discard` only)
  supplied    caller-supplied points; supplied_valid of them valid
  selected    seeds kept = the effective number of restarts; selected_supplied of them supplied
  reused      values taken from a pre-testing cache instead of evaluated (step 5)
  overshoot   draws evaluated beyond the pool of a valid-draw target (step 5)
  target      requested valid draws (0 = the fixed-attempt design)
  stop_reason :planned, :valid_target, :attempt_cap or :time_cap
"""
struct PretestSummary
    attempted::Int
    valid::Int
    invalid::Int
    nonfinite::Int
    errors::Int
    supplied::Int
    supplied_valid::Int
    selected::Int
    selected_supplied::Int
    reused::Int
    overshoot::Int
    target::Int
    target_reached::Bool
    stop_reason::Symbol
end

"""
    SettingsEpoch

The solver settings in force from one point of a run on (2026-09-28, follow-up 1.5). Epoch 1
is the run's start. A resume that changes solver settings -- allowed only explicitly, with
`allow_optimizer_change = true` -- opens a new epoch instead of rewriting the old one: every
committed restart, every job in flight and the polish record the epoch they were dispatched
under, so earlier work keeps the settings it actually ran with, and a replayed job keeps the
settings of its original dispatch. `optimizer_id` is the optimizer identity of the epoch.
"""
struct SettingsEpoch
    index::Int
    local_::SolverSettings
    polish::SolverSettings
    normalize::Bool
    skip_polish::Bool
    optimizer_id::String
    from_segment::Int
    note::String
end

"""
    AttemptKey

One DISPATCHED ATTEMPT of a job: the run, the stage (:local or :polish), the restart index
(0 for the polish) and the attempt number (2026-09-28, follow-up 4). The work of an attempt
is accounted once, under this key: a second delivery of the same attempt adds nothing.
"""
struct AttemptKey
    run_id::String
    stage::Symbol
    j::Int
    attempt::Int
end

"""
    RestartJob

One bounded local solve, complete in itself: the exact start `x0` (already mixed and
clamped, owned by the job), the solver settings and the box. It is what a checkpoint
records at dispatch and what a worker process receives, so it holds coordinates, settings
and IDs only -- never the objective, targets or a solved model (plan J2).

  stage              :local (a restart) or :polish
  j, attempt         restart index and dispatch number (a replayed or retried job keeps j)
  incumbent_version  the version of the incumbent mixed into x0
  eval_start         evaluate f(x0) before the solver (restarts: yes, as always; polish: no)
  epoch              the settings epoch `settings` belong to (a replay keeps its epoch)
"""
struct RestartJob
    run_id::String
    stage::Symbol
    j::Int
    attempt::Int
    theta::Float64
    x0::Vector{Float64}
    incumbent_version::Int
    eval_start::Bool
    f_start_known::Float64
    settings::SolverSettings
    lo::Vector{Float64}
    hi::Vector{Float64}
    on_error::Symbol
    objective_key::Symbol
    master::Int
    progress_every::Float64
    step_scale::Float64          # multiplies settings.initial_step (the :theta_shrink schedule); 1 otherwise
    normalize::Bool              # solve in box-normalized coordinates
    epoch::Int                   # the settings epoch of `settings` and `normalize`
end

"""
    RestartResult

What one finished solve returns: endpoint, value, start value, evaluations, the NLopt code
(or :EXCEPTION / :SEED_EXCEPTION under `on_error = :discard`), the process that ran it and
its wall time. A concrete type replacing the `Vector{Any}` of NamedTuples the batch loop
used to fill (plan J1). An exception crosses only as its message (`error`), the slow path.
"""
struct RestartResult
    run_id::String
    stage::Symbol
    j::Int
    attempt::Int
    f_start::Float64
    x::Vector{Float64}
    f::Float64
    n_eval::Int                  # evaluations, including the separate start evaluation
    ret::Symbol
    error::String
    worker::Int
    elapsed::Float64             # seconds
end

"""
    RestartRecord

A COMMITTED restart: the job as dispatched and its result as merged. Kept for every restart
(plan: preserve more than the best objective) and checkpointed, so a resumed run knows
every start, endpoint, evaluation count and return code. A record imported from a legacy
restarts.csv has empty `x0`/`x` (they were never saved) and `legacy = true`.

  action   :improved (replaced the incumbent), :verified (verified it), :none
  incumbent_version   the version mixed into x0 (asynchronous jobs may see different ones)
  epoch    the settings epoch the restart ran under (its solver settings)
"""
struct RestartRecord
    j::Int
    attempt::Int
    theta::Float64
    x0::Vector{Float64}
    incumbent_version::Int
    f_start::Float64
    x::Vector{Float64}
    f_local::Float64
    n_eval::Int
    ret::Symbol
    action::Symbol
    version_after::Int
    worker::Int
    elapsed::Float64
    commit_seq::Int
    dispatch_seq::Int            # when it was dispatched, in the run's dispatch order
    commits_at_dispatch::Int     # restarts committed when it was dispatched: two jobs overlapped
                                 # iff each was dispatched before the other was committed
    error::String
    legacy::Bool
    epoch::Int
end

"""
    InFlight

A DISPATCHED restart that has not been committed: the exact job (start, attempt, incumbent
version) and where it went. Persisted in the checkpoint BEFORE the job is submitted, so a run
that stops with jobs in flight replays them from their recorded starts (plan steps 6-7).
"""
struct InFlight
    job::RestartJob
    worker::Int
    dispatched::Float64          # time()
    dispatch_seq::Int
    commits_at_dispatch::Int
end

"A run event (start, resume, pause, retry, failure...), kept in the checkpoint as an audit trail."
struct RunEvent
    seq::Int
    time::String
    kind::Symbol
    detail::String
end

"How one process lifetime of a run executed (a run resumed twice has three segments)."
struct Segment
    index::Int
    started::String
    mode::Symbol
    workers::Int
    stop_after::Int
    next_j::Int
    note::String
end

"The polish stage as the checkpoint records it (`epoch`: the settings it ran under; 0 = not run)."
struct PolishRecord
    done::Bool
    ret::Symbol
    improved::Bool
    n_eval::Int
    epoch::Int
end
const POLISH_PENDING = PolishRecord(false, :NOT_RUN, false, 0, 0)

"""
The evaluation accounting of a result (lifetime counts). COMMITTED work: `pretest`, `local_`,
`polish`. Other KNOWN work: `abandoned_known`, evaluations of attempts that executed but were
never merged (superseded). UNKNOWN work, counted in attempts: `jobs_lost` (lost with their
worker process), `attempts_unknown` (failed with an error, interrupted, or still running when
the stage stopped), `pretest_lost` (pre-testing evaluations dispatched whose value never
arrived). `duplicates_ignored`: repeated deliveries of an attempt already accounted -- they
add no work. `retries`: re-dispatches after a lost worker. `reused`: pre-testing values from
a cache. See `work_known` for whether the actual-work total is complete.
"""
const Accounting = NamedTuple{(:pretest, :local_, :polish, :abandoned_known, :jobs_lost, :attempts_unknown,
                               :pretest_lost, :duplicates_ignored, :retries, :reused), NTuple{10,Int}}

"""
    TikTakResult

`x`/`f` are the best point and value after polishing. `trace` records one row
per local search: the seed index, theta, the start value, and the value the
local search reached — enough to see whether the search was still improving when
the budget ran out.

`nstar_requested` is what the caller asked for; `nstar_effective` is how many restarts the
pre-testing pool could seed and the run actually planned; `schedule_denominator` is the K of
theta_j = clamp((j/K)^p, lo, hi), fixed for the life of a run including its resumptions.
"""
struct TikTakResult
    x::Vector{Float64}
    f::Float64
    n_eval::Int
    f_sobol_best::Float64        # best of the pre-testing stage, before any local search
    f_prepolish::Float64         # best after the local stage, before polishing
    trace::Vector{TraceRow}
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
    # WHICH SEARCH PRODUCED THE POINT IN `x`, and how THAT search ended.
    #
    # Acceptance is a statement about the RETAINED WINNER, not about the population of
    # restarts. "Some restart converged" says nothing about the point actually returned:
    # the winner may have come from a restart that exhausted `maxeval`, or from the polish,
    # or -- if every local search failed to improve -- straight from the Sobol stage with
    # no local refinement at all. Reporting `winner_ret` makes that checkable.
    winner_stage::Symbol         # :sobol, :local or :polish
    winner_j::Int                # which restart, when winner_stage === :local; else 0
    winner_ret::Symbol           # the return code of THAT search
    # EFFECTIVE BUDGETS (2026-09-27, plan step 2.4)
    nstar_requested::Int
    nstar_effective::Int
    schedule_denominator::Int
    pretest::PretestSummary
    config::TikTakConfig
    # ORIGIN AND VERIFICATION OF THE RETURNED POINT, AND HOW THE RUN ENDED (plan step 3).
    # `winner_stage/_j/_ret` above are the origin's fields, kept for existing callers.
    incumbent::Incumbent
    objective_id::String         # the objective the search minimised ("unspecified" if not given)
    status::Symbol               # :complete, :stopped_early (stop_tol) or :paused (step 4)
    # THE RUN AS CHECKPOINTED (plan step 4). `n_eval` above is the LIFETIME count across
    # resumptions; `n_eval_segment` is this process's share. A legacy import does not know
    # its earlier evaluations: `n_eval_complete = false` and `n_eval` counts only what is known.
    records::Vector{RestartRecord}
    n_eval_segment::Int
    n_eval_complete::Bool
    # HOW THIS RESULT RELATES TO AN UNINTERRUPTED RUN (2026-09-28, follow-up 1.3), from the whole
    # history, not the last segment: :fresh (one segment), :serial_exact (every segment serial or
    # single-worker, nothing replayed: the sequential path exactly), :async_continuation (some
    # segment ran several workers, or a job was replayed or retried), :changed_optimizer (solver
    # settings or optimizer software changed on a resume, allowed explicitly), :legacy_import,
    # :already_complete (nothing was run by this call).
    resume_semantics::Symbol
    run_id::String
    state_path::String           # "" when the run was not checkpointed
    purpose::String              # the preset label recorded with the run (plan step 8)
    accounting::Accounting       # where the evaluations went (plan step 7.5, follow-up 4)
    # VALIDATED COMPLETENESS (follow-up 1.6): restarts still in flight, and every violated state
    # invariant (empty for a consistent state). `search_budget_complete(result)` reads these,
    # never the status symbol alone.
    inflight::Vector{Int}
    violations::Vector{String}
    epochs::Vector{SettingsEpoch}
end

"""
    RefineOutcome

A final solve of the retained point on the REPORTING objective (the runner's full-grid
refinement). `incumbent` is the reported point with its evidence on that objective: a
certificate earned on the search grid does not transfer to another grid.

  status   :skipped (the search already used the reporting objective), :improved,
           :no_improvement, or :failed (the solve threw; the start point is kept)
"""
struct RefineOutcome
    status::Symbol
    ret::Symbol
    evals::Int                   # solver evaluations, not counting the re-evaluation of the start
    f_start::Float64             # the start point's value on the reporting objective
    incumbent::Incumbent
    error::String
end
