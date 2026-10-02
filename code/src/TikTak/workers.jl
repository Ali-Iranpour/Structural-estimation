# =============================================================================
# workers.jl -- what runs on a worker process, and the progress that comes back from it.
#
# A worker process loads this module and the model ONCE and registers its objective under a
# key (register_objective!). A job then carries only coordinates, settings and IDs; the
# worker looks its objective up locally and builds its own NLopt optimizer. Nothing large --
# targets, a solved model, a runner closure -- crosses the process boundary per restart
# (plan J2), and no NLopt handle or mutable model object is ever shared between processes.
#
# The simulator's random numbers are the objective's own business: a fixed parameter vector
# is evaluated with the same draws on every worker (common random numbers). Nothing here
# seeds anything by worker or restart ID (plan J4).
#
# WORKER LIFECYCLE (2026-09-28, tiktak_fix_plan.md follow-up 3 and 7.1-7.2):
#   leases     every remote call the master submits is LEASED: its worker is recorded as running
#              that job until the call SETTLES (returns or fails), whether or not anyone still
#              waits for the answer. Only the task waiting on the call releases the lease. No stage
#              dispatches to a leased worker: a worker still solving after an abort's drain deadline
#              is quarantined -- or retired, if the caller owns it and says so -- never handed a
#              second job while the first still owns its model and optimizer (finding 16).
#   guard      each worker also refuses a second TikTak job or evaluation while one runs
#              (WorkerBusyError): a caller that bypasses the master's leases cannot interleave two.
#   provenance an exception the worker itself DELIVERED (a RemoteException) is an application error,
#              whatever its type -- an objective that reads a truncated file throws EOFError from a
#              healthy process. Only a transport failure or a process that no longer exists is a
#              lost worker (finding 12).
#   threads    owned workers are started with an explicit Julia thread count (`start_workers`), and
#              `worker_resources` reports what each actually runs with.
# =============================================================================

"Objectives registered on THIS process, by key."
const OBJECTIVES = Dict{Symbol,Any}()

"""
    register_objective!(key, f)

Make `f` the objective that jobs with `objective_key = key` evaluate on this process. Call it
on every worker (`@everywhere TikTak.register_objective!(:search, objective)`) after the
model is loaded, and on the master for serial use.
"""
register_objective!(key::Symbol, f) = (OBJECTIVES[key] = f; nothing)

function registered(key::Symbol)
    f = get(OBJECTIVES, key, nothing)
    f === nothing && error("TikTak: objective :$key is not registered on process $(Distributed.myid()); " *
                           "call TikTak.register_objective!(:$key, f) there first")
    return f
end

"""
    die_with_parent!() -> Bool

Ask the kernel to SIGKILL this process when its parent -- the master that started it --
dies (Linux `prctl(PR_SET_PDEATHSIG)`; a no-op returning false elsewhere). A worker busy in a
long solve does not notice that its master is gone until the solve ends: `tmux kill-session`
left 20 workers of a cancelled run at full CPU on 2026-09-27 (VERSION.md step 20). The
runners call this on every worker right after starting them. Best effort: if the master died
before the call, nothing happens.
"""
function die_with_parent!()
    Sys.islinux() || return false
    try
        return ccall(:prctl, Cint, (Cint, Culong, Culong, Culong, Culong), 1, 9, 0, 0, 0) == 0   # PR_SET_PDEATHSIG, SIGKILL
    catch
        return false
    end
end

"The job this process is running (nothing when idle) -- for diagnostics and tests."
const CURRENT_JOB = Ref{Union{Nothing,RestartJob}}(nothing)
current_job() = CURRENT_JOB[]

# ---- the worker's own guard: one TikTak job or evaluation at a time ----------------------
"What this process runs for TikTak now (nothing when idle): the single-occupancy guard."
const ACTIVE_WORK = Ref{Union{Nothing,String}}(nothing)
const ACTIVE_LOCK = ReentrantLock()

"A job or evaluation refused because this worker is still running another one."
struct WorkerBusyError <: Exception
    worker::Int
    running::String
    refused::String
end
Base.showerror(io::IO, e::WorkerBusyError) =
    print(io, "TikTak worker $(e.worker) is still running $(e.running); refused $(e.refused)")

function claim_worker!(what::String)
    lock(ACTIVE_LOCK) do
        ACTIVE_WORK[] === nothing || throw(WorkerBusyError(Distributed.myid(), ACTIVE_WORK[], what))
        ACTIVE_WORK[] = what
    end
    return nothing
end
release_worker!() = (lock(() -> (ACTIVE_WORK[] = nothing), ACTIVE_LOCK); nothing)
"What this process is running for TikTak (nothing when idle)."
active_work() = ACTIVE_WORK[]

# ---- the master's leases: which worker is running which submitted call -------------------
"""
    Lease

A remote call submitted from this process that has not settled: its worker, the job (run,
stage, restart or candidate index, attempt), when it was submitted, and the task waiting on it.
"""
struct Lease
    worker::Int
    run_id::String
    stage::Symbol
    j::Int
    attempt::Int
    since::Float64
    task::Task
end
const LEASES = Dict{Int,Lease}()
const LEASE_LOCK = ReentrantLock()

describe(l::Lease) = "worker $(l.worker) runs $(l.stage) $(l.j) attempt $(l.attempt) of run $(l.run_id) " *
                     "($(round(time() - l.since; digits = 1)) s)"
"Workers with a submitted call that has not settled."
busy_workers() = lock(() -> sort(collect(keys(LEASES))), LEASE_LOCK)
"The unsettled calls, as NamedTuples (the cleanup obligation a caller can inspect)."
worker_leases() = lock(LEASE_LOCK) do
    [(worker = l.worker, run_id = l.run_id, stage = l.stage, j = l.j, attempt = l.attempt, seconds = time() - l.since)
     for l in sort(collect(values(LEASES)); by = l -> l.worker)]
end

"""
    leased_call(body, w, run_id, stage, j, attempt) -> Task

Run `body()` -- which makes the remote call to worker `w` and must never throw -- in a task
that holds a lease on `w` until the call settles. A worker that already has a lease is refused:
never two jobs on one worker.
"""
function leased_call(body, w::Int, run_id::String, stage::Symbol, j::Int, attempt::Int)
    t = Task() do
        try
            body()
        finally
            lock(LEASE_LOCK) do
                l = get(LEASES, w, nothing)
                (l !== nothing && l.task === current_task()) && delete!(LEASES, w)
            end
        end
    end
    lock(LEASE_LOCK) do
        haskey(LEASES, w) && error("TikTak: not submitted -- $(describe(LEASES[w]))")
        LEASES[w] = Lease(w, run_id, stage, j, attempt, time(), t)
    end
    schedule(t)
    return t
end

"""
    wait_idle(ws; timeout = 0) -> Vector{Int}

Wait up to `timeout` seconds until none of the workers `ws` has an unsettled call; returns the
ones still busy.
"""
function wait_idle(ws; timeout::Real = 0.0)
    busy() = [w for w in ws if w in busy_workers()]
    (isempty(busy()) || timeout <= 0) && return busy()
    timedwait(() -> isempty(busy()), Float64(timeout); pollint = 0.05)
    return busy()
end

"""
    usable_workers(ws, busy_wait; log) -> Vector{Int}

The workers of `ws` a stage may use: alive, and with no unsettled call after waiting up to
`busy_wait` seconds. A worker still running a call of an earlier (aborted) run is QUARANTINED --
left out, and reported through `log(kind, message)` -- never given a second job.
"""
function usable_workers(ws::Vector{Int}, busy_wait::Real; log = (kind, msg) -> nothing)
    live = [w for w in ws if w in Distributed.workers()]
    length(live) < length(ws) && log(:workers_missing, "worker(s) $(join(setdiff(ws, live), ", ")) no longer exist")
    still = wait_idle(live; timeout = busy_wait)
    if !isempty(still)
        ls = lock(() -> [LEASES[w] for w in still if haskey(LEASES, w)], LEASE_LOCK)
        log(:workers_quarantined, "not used, a call of an earlier run has not settled: " * join(describe.(ls), "; "))
    end
    return [w for w in live if !(w in still)]
end

"""
    retire_workers!(ws) -> Vector{Int}

Remove OWNED worker processes that still run an unsettled call (the caller's explicit policy,
`retire_after_abort = true`). Never called on workers the caller did not declare its own.
"""
function retire_workers!(ws)
    gone = [w for w in ws if w in Distributed.workers()]
    isempty(gone) || try Distributed.rmprocs(gone; waitfor = 0) catch end
    return gone
end

# ---- owned workers with an explicit thread budget (7.1) ----------------------------------
"""
    start_workers(n; project, threads = 1, exeflags = ``) -> Vector{Int}

Start `n` OWNED worker processes with an explicit Julia thread count. Workers inherit the
master's environment -- a JULIA_NUM_THREADS there sizes them -- but not its `--threads` flag, so
the count is set on their command line, which wins over the environment.
"""
start_workers(n::Int; project::AbstractString, threads::Int = 1, exeflags::Cmd = ``) =
    Distributed.addprocs(n; exeflags = `--project=$project --threads=$threads $exeflags`)

"""
    resources() -> NamedTuple

This process as it actually runs: id, pid, Julia threads (default and interactive pools), BLAS
threads (-1 if LinearAlgebra is not loaded), and the JULIA_NUM_THREADS it inherited.
"""
function resources()
    la = get(Base.loaded_modules, Base.PkgId(Base.UUID("37e2e46d-f89d-539d-b4ee-838fcccc9c8e"), "LinearAlgebra"), nothing)
    blas = la === nothing ? -1 : Int(Base.invokelatest(la.BLAS.get_num_threads))
    return (id = Distributed.myid(), pid = getpid(), julia_threads = Threads.nthreads(),
            interactive_threads = Threads.nthreads(:interactive), blas_threads = blas,
            env_julia_num_threads = get(ENV, "JULIA_NUM_THREADS", ""))
end
"`resources()` of each worker in `ws` (and of the master for `ws = [1]`)."
worker_resources(ws) = [w == Distributed.myid() ? resources() : Distributed.remotecall_fetch(resources, w) for w in ws]

# ---- progress: coalesced, rate-limited, never able to block a worker --------------------
"""
    ProgressEvent

The latest state of one running solve, sent from its worker to the master: best and last
value after `n_eval` evaluations. Telemetry only -- never needed for a result to arrive.
"""
struct ProgressEvent
    run_id::String
    stage::Symbol
    j::Int
    attempt::Int
    n_eval::Int
    f_last::Float64
    f_best::Float64
    worker::Int
    time::Float64
end

"Minimum seconds between two progress messages from one job (a flood of fast evaluations is coalesced)."
const MIN_REMOTE_PROGRESS = 0.2

"""
    RemoteProgress(master, every)

Reports to the master with `remote_do` -- fire-and-forget, so a busy or slow master never
blocks a worker -- at most once per `every` seconds (never more often than
MIN_REMOTE_PROGRESS). The master keeps only the latest event per worker.
"""
mutable struct RemoteProgress <: Progress
    master::Int
    every::Float64
    last::Float64
end
function report!(p::RemoteProgress, job::RestartJob, n::Int, v::Float64, best::Float64)
    t = time()
    if n == 1 || t - p.last >= p.every
        p.last = t
        try
            Distributed.remote_do(progress_sink!, p.master,
                                  ProgressEvent(job.run_id, job.stage, job.j, job.attempt, n, v, best,
                                                Distributed.myid(), t))
        catch
        end
    end
    return nothing
end

"On the master: the latest progress event per worker, and how many arrived."
const PROGRESS_STORE = Dict{Int,ProgressEvent}()
const PROGRESS_RECEIVED = Ref(0)
function progress_sink!(ev::ProgressEvent)
    PROGRESS_STORE[ev.worker] = ev
    PROGRESS_RECEIVED[] += 1
    return nothing
end

# ---- the entry points a worker runs ---------------------------------------------------
"""
    worker_run_job(job) -> RestartResult

Run one local solve on this process with the objective registered under
`job.objective_key`. The function barrier `_run_registered` compiles the solve for the
objective's concrete type (plan J1).
"""
function worker_run_job(job::RestartJob)
    f = registered(job.objective_key)
    progress = (isfinite(job.progress_every) && job.master != Distributed.myid()) ?
               RemoteProgress(job.master, max(job.progress_every, MIN_REMOTE_PROGRESS), 0.0) : NoProgress()
    claim_worker!("$(job.stage) $(job.j) attempt $(job.attempt) of run $(job.run_id)")
    CURRENT_JOB[] = job
    try
        return _run_registered(f, job, progress)
    finally
        CURRENT_JOB[] = nothing
        release_worker!()
    end
end
_run_registered(f::F, job::RestartJob, progress::P) where {F,P} = run_local(f, job, progress)

"""
    worker_eval(key, x, on_error) -> (value, errored, worker, seconds)

One pre-testing evaluation on this process. Under `on_error = :discard` a throw scores Inf and
is flagged; under `:rethrow` it propagates to the master as a RemoteException.
"""
function worker_eval(key::Symbol, x::Vector{Float64}, on_error::Symbol)
    f = registered(key)
    claim_worker!("a pre-testing evaluation")
    try
        t0 = time()
        v, e = _eval_registered(f, x, on_error)
        return (v, e, Distributed.myid(), time() - t0)
    finally
        release_worker!()
    end
end
_eval_registered(f::F, x, on_error::Symbol) where {F} =
    on_error === :discard ? (try (Float64(f(x)), false) catch; (Inf, true) end) : (Float64(f(x)), false)

"""
    classify_remote_error(e; worker = 0) -> (kind, message)

What a failed remote call means, from WHERE the exception came from (finding 12):

  :worker_lost   the worker process is gone: a ProcessExitedException raised on this side, a
                 transport IOError/EOFError that the worker did NOT deliver, or `worker` no longer
                 exists. Retried, bounded.
  :interrupted   an InterruptException (the master interrupted the job while stopping)
  :busy          the worker refused the job: it still runs another (WorkerBusyError)
  :error         any exception the live worker DELIVERED (a RemoteException) -- an EOFError or
                 IOError from an objective reading a truncated file included: an error in the
                 objective or in this code, never retried

Measured with Julia 1.11 Distributed: an application exception arrives as RemoteException >
CapturedException > the exception, with the worker still alive; a worker that exits or is
killed mid-call arrives as a bare ProcessExitedException and is gone from `workers()`.
"""
function classify_remote_error(e; worker::Int = 0)
    delivered = false
    inner = e
    while true
        if inner isa Distributed.RemoteException
            delivered = true
            inner = inner.captured
        elseif inner isa CapturedException
            inner = inner.ex
        elseif inner isa TaskFailedException
            inner = inner.task.exception
        else
            break
        end
    end
    msg = sprint(showerror, inner)
    (inner isa Distributed.ProcessExitedException && !delivered) && return (:worker_lost, msg)
    (worker > 0 && !(worker in Distributed.procs())) && return (:worker_lost, "worker $worker no longer exists: " * msg)
    (!delivered && (inner isa Base.IOError || inner isa EOFError)) && return (:worker_lost, msg)
    inner isa InterruptException && return (:interrupted, "interrupted")
    inner isa WorkerBusyError && return (:busy, msg)
    return (:error, msg)
end
