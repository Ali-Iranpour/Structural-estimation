# =============================================================================
# scheduler.jl -- the asynchronous, process-parallel stages (plan steps 6-7, J2).
#
# The MASTER owns the run: the incumbent, the restart queue, the checkpoint. Worker PROCESSES
# run bounded local solves (one active solve per worker, each with its own NLopt optimizer
# and its own copy of the model). On the master one task per active job waits on its
# remote call and forwards the completion -- result or failure -- to a typed Channel; ONE
# scheduler loop consumes completions IN THE ORDER THEY FINISH, merges each exactly once,
# checkpoints, and immediately gives the idle worker the next restart, mixed with the
# incumbent as it is at that moment. A slow restart never holds up a faster one.
#
# This is TikTak's asynchronous variant, not the sequential algorithm: restart j is mixed
# with the best COMMITTED result when j is dispatched, which may not yet include restarts
# still running. With one local worker it is exactly the sequential algorithm.
#
#   bootstrap   :first_alone (default) -- restart 1 runs alone and is committed before the pool
#               is filled; :immediate_mixed -- the pool fills at once, restarts j >= 2 mixed with
#               the best pre-tested point until a result is committed (config.jl). Declared
#               optimizer settings, neither a copy of the reference code's start-up
#   dispatch    the job (start, attempt, incumbent version) is checkpointed BEFORE it is sent,
#               and its worker is LEASED until the remote call settles (workers.jl)
#   worker lost the job is re-dispatched from its RECORDED start (bounded retries), never
#               rebuilt from a newer incumbent
#   error       an error in the objective (or this code) is never retried: the run checkpoints,
#               interrupts the other jobs, keeps what completes within `drain_timeout`, and stops
#   pause       a pause file or the --stop-after-restarts launch limit stops dispatching;
#               running jobs are drained and committed; nothing is polished
#
# SHUTDOWN (2026-09-28, follow-up 3, finding 16). Any stop -- an objective error, a master-side
# exception (a checkpoint write, a caller's callback), a lost pool -- takes the same ordered way
# out: stop dispatching; keep committing what completes until the drain deadline; commit results
# already delivered; record every call still running as UNSETTLED (its evaluations unknown); close
# the channel; checkpoint; throw. An unsettled call keeps its worker leased, so neither this nor
# a later run in the same process dispatches to that worker until the call has settled -- or, with
# `retire_after_abort` (owned workers only), the worker is removed.
#
# Threads are never used for SMM work: tasks only wait. Telemetry (progress) is coalesced and
# never needed for a result to arrive.
# =============================================================================

"A finished remote job, as the waiting task forwards it to the scheduler."
struct Completion
    kind::Symbol                 # :ok | :worker_lost | :interrupted | :busy | :error
    worker::Int
    j::Int
    attempt::Int
    result::Union{Nothing,RestartResult}
    message::String
end

"""
How the asynchronous local stage is executed (not part of the optimizer identity).

`fault` is FAULT INJECTION FOR THE TESTS ONLY (tools/test_tiktak.jl): :none in every real
run; :duplicate_results (:triplicate_results) makes every waiting task deliver its completion
twice (three times), which the scheduler must merge -- and count -- once.
"""
struct AsyncExec
    workers::Vector{Int}
    objective_key::Symbol
    progress_every::Float64
    stop_after::Int
    pause_file::String
    max_retries::Int
    drain_timeout::Float64
    interrupt_on_abort::Bool     # SIGINT the busy workers when stopping (may end those processes)
    fault::Symbol
    busy_wait::Float64           # seconds to wait for a worker still running an earlier call
    retire_after_abort::Bool     # remove (owned) workers whose call is still running at the deadline
end
AsyncExec(ws, key, every, stop_after, pause_file, max_retries, drain_timeout) =
    AsyncExec(ws, key, every, stop_after, pause_file, max_retries, drain_timeout, false, :none, 30.0, false)

"Deliveries per completion under the test fault (1 in every real run)."
fault_copies(fault::Symbol) = fault === :duplicate_results ? 2 : fault === :triplicate_results ? 3 : 1

"""
    RemoteJobError

A single remote job (the polish) that failed: how (`kind`, as `classify_remote_error`), on which
worker, and which attempt.
"""
struct RemoteJobError <: Exception
    kind::Symbol
    worker::Int
    stage::Symbol
    j::Int
    attempt::Int
    message::String
end
Base.showerror(io::IO, e::RemoteJobError) =
    print(io, "TikTak: the $(e.stage) job (attempt $(e.attempt)) on worker $(e.worker): $(e.kind) -- $(e.message)")

"""
    active_progress(st) -> Vector{ProgressEvent}

The latest progress event of each job that is IN FLIGHT now (same restart and attempt), by
restart index. A late event from a finished or superseded job is dropped here: delayed
telemetry never reports work that is over.
"""
active_progress(st::RunState) =
    sort(ProgressEvent[ev for ev in values(PROGRESS_STORE)
                       if ev.run_id == st.run_id && haskey(st.inflight, ev.j) && st.inflight[ev.j].job.attempt == ev.attempt];
         by = ev -> ev.j)

"Take the next completion, waiting at most `timeout` seconds; `nothing` on timeout."
function take_completion(ch::Channel, timeout::Float64)
    isready(ch) && return take!(ch)
    timedwait(() -> isready(ch), max(timeout, 0.01); pollint = 0.02) === :ok || return nothing
    return take!(ch)
end

"""
    submit_job(w, job, deliver) -> Task

Send `job` to worker `w` under a lease (workers.jl). The waiting task turns the call's outcome
into a Completion -- a failure classified by where it came from -- hands it to `deliver` and
returns it. It never throws.
"""
submit_job(w::Int, job::RestartJob, deliver) =
    leased_call(w, job.run_id, job.stage, job.j, job.attempt) do
        c = try
            Completion(:ok, w, job.j, job.attempt, Distributed.remotecall_fetch(worker_run_job, w, job), "")
        catch e
            kind, msg = classify_remote_error(e; worker = w)
            Completion(kind, w, job.j, job.attempt, nothing, msg)
        end
        try deliver(c) catch end                          # (the scheduler may have closed its channel)
        c
    end

"""
    local_stage_async!(st, cfg, ex, save!, on_local, on_progress) -> st

The asynchronous local stage over the worker processes `ex.workers`. Returns with
`st.status === :paused` after a pause, or with the local stage done (every planned restart
committed, or stop_tol reached and the running jobs drained). Throws after checkpointing
when a job fails with an error, when every worker is gone, or on a master-side exception.
"""
function local_stage_async!(st::RunState, cfg::TikTakConfig, ex::AsyncExec, save!, on_local, on_progress)
    master = Distributed.myid()
    # live workers not still running a call of an earlier (aborted) run -- a busy one is
    # waited for up to busy_wait seconds, then quarantined (follow-up 3)
    pool = usable_workers(ex.workers, ex.busy_wait; log = (k, m) -> add_event!(st, k, m))
    isempty(pool) && throw(ArgumentError("TikTak: the asynchronous local stage has no usable worker process (none " *
        "alive and idle; unsettled calls: $(join(string.(worker_leases()), "; ")))"))
    losses = Dict{Int,Int}()                            # worker losses per restart in THIS segment
    idle = copy(pool)
    # worker => the job (restart, attempt) it runs: ONE job per worker, always. A worker is
    # freed only by the completion of ITS job -- a late duplicate of an earlier job must not
    # hand a busy worker a second job.
    busy = Dict{Int,Tuple{Int,Int}}()
    free!(w, j, attempt) = (get(busy, w, nothing) == (j, attempt) && w in pool) &&
                           (delete!(busy, w); push!(idle, w); true)
    tasks = Dict{Tuple{Int,Int},Tuple{Int,Task}}()     # (j, attempt) => (worker, the task waiting on it)
    requeue = RestartJob[]                              # replays and retries, dispatched first
    copies = fault_copies(ex.fault)
    ch = Channel{Completion}(copies * length(pool) + 1)
    x0buf = Vector{Float64}(undef, length(st.lo))
    pausing = Ref(false); aborting = Ref(false); fatal = Ref{Union{Nothing,Completion}}(nothing)
    master_error = Ref{Any}(nothing)
    empty!(PROGRESS_STORE)
    last_progress = Ref(time())
    abort_deadline = Ref(Inf)

    # a checkpoint may hold jobs that were in flight when the run stopped: replay them
    for fl in sort(collect(values(st.inflight)); by = fl -> fl.job.j)
        push!(requeue, redispatch(fl.job, master, ex.objective_key, ex.progress_every))
        add_event!(st, :replay, "restart $(fl.job.j) was in flight at the checkpoint: replayed from its recorded start " *
                   "(attempt $(fl.job.attempt + 1), settings epoch $(fl.job.epoch))")
    end
    committed(j) = any(r -> r.j == j, st.records)
    # THE BOOTSTRAP (cfg.bootstrap). :first_alone -- while restart 1 is dispatched and not
    # committed, nothing else starts (only restart 1 itself: a legacy import that resumes later
    # never waits on it). :immediate_mixed -- nothing waits: restart_job mixes each new start
    # with st.inc as it is at dispatch, which before the first commit is the best pre-tested point.
    bootstrapping() = cfg.bootstrap === :first_alone && haskey(st.inflight, 1) && !committed(1)
    launched() = st.next_j - 1                          # restarts ever allocated

    function next_job()
        isempty(requeue) || return popfirst!(requeue)
        (pausing[] || aborting[] || st.stopped_early) && return nothing
        st.next_j > st.K && return nothing
        launched() >= ex.stop_after && return nothing   # the launch limit: never beyond the prefix
        bootstrapping() && return nothing
        j = st.next_j
        st.next_j += 1
        return restart_job(st, j, next_attempt(st, :local, j), x0buf; objective_key = ex.objective_key, master = master,
                           progress_every = ex.progress_every)
    end

    function dispatch!(w::Int, job::RestartJob)
        st.dispatch_seq += 1
        busy[w] = (job.j, job.attempt)
        st.inflight[job.j] = InFlight(job, w, time(), st.dispatch_seq, st.commit_seq)
        save!(st)                                        # the dispatch record is on disk first
        tasks[(job.j, job.attempt)] = (w, submit_job(w, job, c -> for _ in 1:copies; put!(ch, c); end))
        return nothing
    end

    function stop_all!(why::String)
        aborting[] = true
        if ex.interrupt_on_abort
            # (not `busy =`: assigning a captured name inside a closure REBINDS the outer variable)
            running = [fl.worker for fl in values(st.inflight) if fl.worker in pool]
            for w in unique(running)
                try Distributed.interrupt(w) catch end
            end
        end
        add_event!(st, :abort, why)
        return nothing
    end
    begin_stop!(why::String) = aborting[] || (stop_all!(why); abort_deadline[] = time() + ex.drain_timeout)

    "Merge one completion (or account for it) -- the ONE place a message changes the state."
    function handle!(c::Completion)
        entry = pop!(tasks, (c.j, c.attempt), nothing)
        entry === nothing || wait(entry[2])              # that call has settled: its lease is released
        fl = get(st.inflight, c.j, nothing)
        if fl === nothing || fl.job.attempt != c.attempt
            # not the attempt in flight -- a second delivery or a superseded attempt: never merged,
            # and its work is accounted ONCE per attempt (follow-up 4, finding 13): a duplicate adds
            # nothing; an attempt that executed but was superseded adds its known evaluations once
            how = account_uncommitted!(st, c.j, c.attempt, c.result)
            add_event!(st, how === :duplicate ? :duplicate_result : :stale_result,
                       "ignored a $(c.kind) message for restart $(c.j) attempt $(c.attempt) (" *
                       (how === :duplicate ? "already accounted: counted zero" :
                        how === :abandoned ? "executed but superseded: $(c.result.n_eval) evaluations counted as known abandoned work" :
                        how === :other_run ? "a result of another run: counted zero" : "no result") * ")")
            free!(c.worker, c.j, c.attempt)
            return nothing
        end
        key = attempt_key(fl.job)
        if c.kind === :ok
            delete!(st.inflight, c.j)
            rec = commit_restart!(st, fl.job, c.result; dispatch_seq = fl.dispatch_seq,
                                  commits_at_dispatch = fl.commits_at_dispatch)
            free!(c.worker, c.j, c.attempt)
            save!(st)
            master_error[] === nothing && on_local(rec.j, st.K, rec.theta, rec.f_local, st.inc.f, st.inc.x, callback_row(rec))
        elseif c.kind === :worker_lost || c.kind === :busy
            # the worker cannot take jobs: gone, or (:busy) still running a call this scheduler did
            # not submit. Either way it leaves the pool; the job is re-dispatched from its start.
            filter!(!=(c.worker), pool); filter!(!=(c.worker), idle); delete!(busy, c.worker)
            if c.kind === :worker_lost
                account_unknown!(st, key, :lost)
                add_event!(st, :worker_lost, "worker $(c.worker) lost during restart $(c.j) attempt $(c.attempt) " *
                           "($(c.message)); its evaluations are unknown")
            else
                st.accounted[key] = :refused             # never started: no work
                add_event!(st, :workers_quarantined, "worker $(c.worker) refused restart $(c.j) attempt $(c.attempt) " *
                           "($(c.message)); not used again in this stage")
            end
            if aborting[] || pausing[]
                # left in flight: a resume replays it from its recorded start
            elseif c.kind === :busy ? !isempty(pool) :
                   ((losses[c.j] = get(losses, c.j, 0) + 1) <= ex.max_retries && !isempty(pool))
                c.kind === :worker_lost && (st.retries += 1)
                pushfirst!(requeue, redispatch(fl.job, master, ex.objective_key, ex.progress_every))
                add_event!(st, :retry, "restart $(c.j) re-dispatched from its recorded start (attempt $(c.attempt + 1))")
            else
                fatal[] = c
                begin_stop!(isempty(pool) ? "no usable worker process left" :
                            "restart $(c.j) lost its worker $(losses[c.j]) time(s) in this segment; retries exhausted")
            end
            save!(st)
        elseif c.kind === :interrupted
            # interrupted while stopping: it stays in flight and is replayed on resume
            account_unknown!(st, key, :interrupted)
            free!(c.worker, c.j, c.attempt)
        else
            account_unknown!(st, key, :failed)
            fatal[] === nothing && (fatal[] = c)
            begin_stop!("restart $(c.j) failed on worker $(c.worker): $(c.message)")
            free!(c.worker, c.j, c.attempt)
            save!(st)
        end
        return nothing
    end

    # The loop. A master-side exception inside it -- a checkpoint write, a caller's callback --
    # is caught HERE and turns the loop into the same orderly drain as a failed job: no new
    # dispatch, completions merged until the deadline, then the exception is rethrown below.
    while true
        try
            if !aborting[]
                while !isempty(idle)
                    job = next_job()
                    job === nothing && break
                    dispatch!(popfirst!(idle), job)
                end
            end
            isempty(tasks) && break
            aborting[] && time() > abort_deadline[] && break
            c = take_completion(ch, aborting[] ? 0.2 : isfinite(ex.progress_every) ? ex.progress_every : 1.0)
            if !pausing[] && !isempty(ex.pause_file) && isfile(ex.pause_file)
                pausing[] = true
                add_event!(st, :pause_requested, "pause file $(ex.pause_file) found; draining $(length(tasks)) job(s)")
            end
            if isfinite(ex.progress_every) && time() - last_progress[] >= ex.progress_every && master_error[] === nothing
                last_progress[] = time()
                on_progress(active_progress(st))
            end
            c === nothing || handle!(c)
        catch e
            master_error[] === nothing ? (master_error[] = e) :
                add_event!(st, :master_error, "a further master-side error while draining: " * sprint(showerror, e))
            begin_stop!("master-side error: " * sprint(showerror, e))
        end
    end
    # completions already delivered are merged (a result that arrived as the deadline passed is
    # known work, not lost work), then every call still running is recorded as UNSETTLED
    while isready(ch)
        try
            handle!(take!(ch))
        catch e
            master_error[] === nothing && (master_error[] = e)
        end
    end
    if !isempty(tasks)
        for ((j, a), (w, _)) in sort(collect(tasks); by = first)
            account_unknown!(st, AttemptKey(st.run_id, :local, j, a), :running_at_stop)
            add_event!(st, :unsettled, "restart $j attempt $a was still running on worker $w at the drain deadline; " *
                       (ex.retire_after_abort ? "the worker is retired (retire_after_abort)" :
                        "the worker stays leased -- quarantined -- until that call settles (TikTak.worker_leases())"))
        end
        ex.retire_after_abort && retire_workers!(unique(w for (w, _) in values(tasks)))
    end
    close(ch)
    if master_error[] !== nothing
        fail!(st, "master-side error; $(length(st.inflight)) job(s) left in flight for replay, $(length(tasks)) still " *
                  "running: " * sprint(showerror, master_error[]))
        try save!(st) catch end
        throw(master_error[])
    end
    if fatal[] !== nothing
        c = fatal[]
        fail!(st, "the asynchronous local stage stopped: restart $(c.j) ($(c.kind)): $(c.message); " *
                  "$(length(st.inflight)) job(s) left in flight for replay")
        save!(st)
        error("TikTak: restart $(c.j) on worker $(c.worker): $(c.kind) -- $(c.message)")
    end
    if pausing[] || (st.next_j <= st.K && launched() >= ex.stop_after && !st.stopped_early) || !isempty(st.inflight)
        pause!(st, pausing[] ? "pause file $(ex.pause_file)" :
                   isempty(st.inflight) ? "stop_after_restarts = $(ex.stop_after) reached ($(length(st.records)) of $(st.K) committed)" :
                   "$(length(st.inflight)) job(s) left in flight")
        save!(st)
    end
    return st
end

"""
    run_job_on_pool(pool, job; max_retries, busy_wait, on_lost, log) -> (RestartResult, job)

One job (the polish) on the first usable worker, under a lease; re-dispatched -- as the next
attempt, from the same start -- on another worker if its worker is lost or refuses (still busy),
never retried after an error. Returns the result and the attempt that produced it; throws a
RemoteJobError otherwise. If the caller is interrupted while waiting, the remote call keeps its
lease until it settles.
"""
function run_job_on_pool(pool::Vector{Int}, job::RestartJob; max_retries::Int = 1, busy_wait::Real = 30.0,
                         on_lost = (job, msg) -> nothing, log = (k, m) -> nothing)
    retries = 0
    ws = usable_workers(pool, busy_wait; log = log)
    while true
        isempty(ws) && error("TikTak: no usable worker process left for the $(job.stage) job")
        w = first(ws)
        c = fetch(submit_job(w, job, c -> nothing))
        c.kind === :ok && return c.result, job
        (c.kind in (:worker_lost, :busy) && retries < max_retries) ||
            throw(RemoteJobError(c.kind, w, job.stage, job.j, job.attempt, c.message))
        c.kind === :worker_lost && on_lost(job, c.message)
        c.kind === :busy && log(:workers_quarantined, "worker $w refused the $(job.stage) job: $(c.message)")
        retries += 1
        filter!(!=(w), ws)
        job = redispatch(job, job.master, job.objective_key, job.progress_every)
    end
end

"""
    pretest_async!(pt, cfg, workers, key; chunk, cache_path, time_cap, on_sobol, max_retries,
                   busy_wait, drain_timeout, retire_after_abort, log) -> (i_end, stop_reason)

The pre-testing stage over worker processes, without a batch barrier: every idle worker gets
the next needed candidate at once; values are stored as they arrive and the cache is written
every `chunk` completions. The POOL is still defined by the values in candidate-index order
(PoolCursor), so the selection does not depend on which worker finished first. With a valid-
draw target, dispatching stops once the confirmed prefix plus the draws in flight could
reach it; draws completed beyond the pool are overshoot. A lost worker's candidate is
re-dispatched (bounded); an error stops the stage after the running evaluations finish or the
drain deadline passes. Every evaluation runs under a lease, and a master-side exception takes
the same orderly way out; evaluations whose value never arrives are counted in `pt.lost`.
"""
function pretest_async!(pt::Pretest, cfg::TikTakConfig, workers::Vector{Int}, key::Symbol;
                        chunk::Int = 32, cache_path::String = "", time_cap::Float64 = Inf,
                        on_sobol = (i, n, fx, best) -> nothing, max_retries::Int = 1,
                        busy_wait::Real = 30.0, drain_timeout::Real = 60.0, retire_after_abort::Bool = false,
                        log = (k, m) -> nothing)
    pool = usable_workers(workers, busy_wait; log = log)
    isempty(pool) && throw(ArgumentError("TikTak: process-parallel pre-testing has no usable worker process"))
    t0 = time()
    planned = cfg.n_sobol + length(pt.supplied)
    idle = copy(pool)
    # (kind, candidate, supplied, worker, value, errored, message)
    ch = Channel{Tuple{Symbol,Int,Bool,Int,Float64,Bool,String}}(length(pool) + 1)
    attempts = Dict{Tuple{Int,Bool},Int}()               # dispatches per candidate
    inflight = Set{Tuple{Int,Bool}}()
    tasks = Dict{Tuple{Int,Bool},Tuple{Int,Task}}()      # (candidate, supplied) => (worker, waiting task)
    queue = Tuple{Int,Bool}[(k, true) for k in findall(!, pt.sdone)]   # supplied points first
    best = Ref(Inf)
    for v in Iterators.flatten((pt.values[pt.done], pt.svalues[pt.sdone]))
        valid_value(v, cfg.invalid_value) && v < best[] && (best[] = v)
    end
    ndone = Ref(count(pt.done) + count(pt.sdone)); since = Ref(0)
    save!() = (isempty(cache_path) || write_cache(cache_path, pt); since[] = 0)
    cur = PoolCursor(0, 0)
    last_k = Ref(0)
    stopped = Ref(false)                                  # no new Sobol draws (pool complete or time cap)
    time_capped = Ref(false)
    err = Ref{Union{Nothing,String}}(nothing)
    master_error = Ref{Any}(nothing)
    deadline = Ref(Inf)
    stopping() = err[] !== nothing || master_error[] !== nothing
    begin_stop!() = deadline[] == Inf && (deadline[] = time() + drain_timeout)

    function dispatch!(w::Int, k::Int, supplied::Bool)
        x = supplied ? pt.supplied[k] : point!(pt, k)
        attempts[(k, supplied)] = get(attempts, (k, supplied), 0) + 1
        push!(inflight, (k, supplied))
        t = leased_call(w, "pretest $(pt.design_id)", :pretest, supplied ? -k : k, attempts[(k, supplied)]) do
            c = try
                v, e, _, _ = Distributed.remotecall_fetch(worker_eval, w, key, x, cfg.on_error)
                (:ok, k, supplied, w, v, e, "")
            catch ex
                kind, msg = classify_remote_error(ex; worker = w)
                (kind, k, supplied, w, NaN, false, msg)
            end
            try put!(ch, c) catch end
            c
        end
        tasks[(k, supplied)] = (w, t)
        return nothing
    end
    "The next candidate to evaluate, or nothing (none needed now)."
    function next_candidate()
        isempty(queue) || return popfirst!(queue)
        stopped[] && return nothing
        i_end, status = advance!(cur, pt, cfg)
        if status !== :need_more
            stopped[] = true
            return nothing
        end
        if time() - t0 >= time_cap
            stopped[] = true; time_capped[] = true
            return nothing
        end
        # with a valid-draw target, do not dispatch when the draws already in flight could
        # complete it: this bounds the overshoot by the number of workers
        n_sob = count(p -> !p[2], inflight)
        cfg.n_valid_target > 0 && cur.valid + n_sob >= cfg.n_valid_target && return nothing
        k = max(last_k[], i_end) + 1
        while k <= cfg.n_sobol && ((k <= length(pt.done) && pt.done[k]) || (k, false) in inflight)
            k += 1
        end
        k > cfg.n_sobol && return nothing
        last_k[] = k
        return (k, false)
    end
    function handle!(c)
        kind, k, supplied, w, v, e, msg = c
        entry = pop!(tasks, (k, supplied), nothing)
        entry === nothing || wait(entry[2])
        delete!(inflight, (k, supplied))
        if kind === :ok
            if supplied
                pt.svalues[k] = v; pt.sdone[k] = true; pt.serrored[k] = e
            else
                ensure_length!(pt, k); pt.values[k] = v; pt.done[k] = true; pt.errored[k] = e
            end
            pt.evals_segment += 1; ndone[] += 1; since[] += 1
            w in pool && push!(idle, w)
            valid_value(v, cfg.invalid_value) && v < best[] && (best[] = v)
            on_sobol(min(ndone[], planned - 1), planned, v, best[])
            since[] >= chunk && save!()
        elseif kind === :worker_lost || kind === :busy
            filter!(!=(w), pool); filter!(!=(w), idle)
            if kind === :worker_lost
                pt.lost += 1                               # its evaluation's outcome is unknown
            else
                attempts[(k, supplied)] -= 1               # refused, never started: not an attempt
                log(:workers_quarantined, "worker $w refused pre-testing candidate $k: $msg")
            end
            if !stopping() && attempts[(k, supplied)] <= max_retries && !isempty(pool)
                pushfirst!(queue, (k, supplied))           # the same candidate, on another worker
            elseif !stopping()
                err[] = "pre-testing candidate $k lost its worker; " * (isempty(pool) ? "no worker process left" : "retries exhausted")
                begin_stop!()
            end
        else
            pt.lost += 1
            if err[] === nothing
                err[] = "pre-testing candidate $k on worker $w: $msg"
                begin_stop!()
            end
            w in pool && push!(idle, w)
        end
        return nothing
    end

    while true
        try
            if !stopping()
                while !isempty(idle)
                    c = next_candidate()
                    c === nothing && break
                    dispatch!(popfirst!(idle), c[1], c[2])
                end
            end
            isempty(inflight) && break                     # nothing running and nothing more to start
            stopping() && time() > deadline[] && break
            c = take_completion(ch, stopping() ? 0.2 : 1.0)
            c === nothing || handle!(c)
        catch e
            master_error[] === nothing && (master_error[] = e)
            begin_stop!()
        end
    end
    while isready(ch)                                      # values already delivered are kept
        try handle!(take!(ch)) catch e; master_error[] === nothing && (master_error[] = e) end
    end
    if !isempty(tasks)
        pt.lost += length(tasks)
        log(:unsettled, "$(length(tasks)) pre-testing evaluation(s) still running at the drain deadline on worker(s) " *
            join(sort(unique(w for (w, _) in values(tasks))), ", ") *
            (retire_after_abort ? "; retired" : "; quarantined until they settle"))
        retire_after_abort && retire_workers!(unique(w for (w, _) in values(tasks)))
    end
    close(ch)
    try
        save!()
    catch e
        master_error[] === nothing && (master_error[] = e)
    end
    master_error[] === nothing || throw(master_error[])
    err[] === nothing || error("TikTak: " * err[])
    i_end, status = advance!(cur, pt, cfg)
    status === :need_more && !time_capped[] && error("TikTak: pre-testing ended with its pool incomplete " *
                                                     "(no worker process left?)")
    stop_reason = time_capped[] && status === :need_more ? :time_cap :
                  status === :reached ? :valid_target : status === :cap ? :attempt_cap : :planned
    on_sobol(planned, planned, NaN, best[])
    return i_end, stop_reason
end
