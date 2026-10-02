# =============================================================================
# driver.jl -- the TikTak algorithm: pre-testing, the local stage, the polish.
#
# Arnoud, Guvenen & Kleineberg (2022), "Benchmarking Global Optimizers": Section 2.1 (the
# algorithm), Section 3.3 (polishing), Appendix A.6 with footnotes 26-27 (the settings used
# in the paper). Reference implementation: https://github.com/serdarozkan/TikTak (read at
# commit 16ac0d9, 2026-09-27).
#
# TikTak is a multistart algorithm in two stages:
#
#   GLOBAL (pre-testing)   Evaluate f at N Sobol' points covering the box, sort
#                          ascending, and keep the best N* VALID ones as "seed" points
#                          s_1, ..., s_N* with f(s_1) <= ... <= f(s_N*). The paper's
#                          benchmark puts N* at 0.1 N (Section 2.1 discusses 1-10%).
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
# SETTINGS, AND WHERE THEY COME FROM
#
#   theta_j        min(max(0.1, sqrt(j / N*)), 0.995), restart 1 at 0: the formula of
#                  AGK (2022) Appendix A.6, footnote 27 (the paper DOES give it; an
#                  earlier comment here said otherwise). Exposed as (theta_p, theta_lo,
#                  theta_hi). The reference repository at 16ac0d9 caps at 0.95 instead --
#                  a version difference, not an error of either.
#
#   local search   NLopt Nelder-Mead with ftol_rel 1e-3 -- a TikTak-nm3-style search. NLopt's
#                  Nelder-Mead tests the spread of f over its simplex (f_worst - f_best),
#                  which is the paper's footnote-14 criterion; its other stopping rules and
#                  implementation details mean this is not a bit-for-bit reproduction of the
#                  benchmark solver. DFNLS (the "d" variants) has no maintained Julia binding.
#
#   polishing      BOBYQA at ftol_rel 1e-10 (Section 3.3 names DFNLS and/or BOBYQA). BOBYQA
#                  fits a local quadratic model, so a rough objective can limit it.
#
#   extra seeds    DEVIATION FROM THE PUBLISHED ALGORITHM, taken deliberately. `extra_seeds`
#                  puts known points (the incumbent calibration) into the pre-testing pool,
#                  where they compete on value like any Sobol' point; a retained supplied
#                  point is VERIFIED if a converged solve returns it unchanged.
#
#   early stop     The paper suggests stopping when the last two distinct values of Z* are
#                  close. `stop_tol`, default 0.0 = disabled: the full budget runs.
#
# -----------------------------------------------------------------------------
# EXECUTION (2026-09-27, tiktak_fix_plan.md steps 5-7). Processes, never threads, for work.
#
#   Pre-testing    serial, a caller-supplied batch map (pmap), or `pretest_workers`: worker
#                  processes fed one candidate at a time as each frees up (scheduler.jl).
#                  `parallel = true` evaluates it on threads, for a thread-safe `f` only.
#
#   Local stage    local_mode = :serial (default here): the published sequential algorithm --
#                  restart j is built from the incumbent that restarts 1..j-1 left.
#                  local_mode = :async_process: TikTak's asynchronous variant on worker
#                  processes (scheduler.jl). Restart 1 runs alone (bootstrap = :first_alone, the
#                  default) or the pool fills at once (:immediate_mixed, config.jl); then each
#                  idle worker gets the next restart at once, mixed with the best COMMITTED point
#                  (before any commit: the best pre-tested point), while other restarts are still running. Completion order can change later starts, so
#                  the path is not the sequential one; with one worker it is exactly the same.
#                  The reference repository scales this way and suggests about
#                  sqrt(restarts) workers -- an empirical heuristic, not a bound.
#
#                  (Until 2026-09-27 a `batch` option ran SYNCHRONOUS batches on threads: all
#                  restarts of a batch read one frozen Z*, and the batch waited for its slowest
#                  member. That is not the reference's asynchronous scheme; it was removed.)
#
#   Threads        Measured 2026-08-07: with 8 threads the SMM objective killed the process
#                  silently (exit 0). A later inspection found a sufficient cause in this
#                  file's old code -- an `opt` binding captured in a Core.Box and SHARED by
#                  concurrent restarts, so they drove one another's NLopt state. That binding
#                  is gone (each solve builds its own optimizer), but thread safety of the
#                  whole SMM objective was never established, so the project uses worker
#                  PROCESSES: each has its own NLopt state and its own copy of the model.
#
#   Tolerances     NLopt's relative test |df| < ftol_rel (|f_new| + |f_old|)/2 also accepts
#                  equal values, including 0 (NLopt 2.10); ftol_abs (objective units -- NOT
#                  scale-free) and xtol_rel (relative to |x|) are backstops set far below
#                  ftol_rel. For the SMM, Q is a sum of inverse-variance-weighted squared
#                  errors, so ftol_abs means different things at different Q.
# =============================================================================

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
  local_ftol_abs/xtol_rel  secondary stopping rules for the local stage, far below
                           local_tol; see the note in the signature. Do not set to 0.
  polish_alg/tol/maxeval   final polish (default BOBYQA, 1e-10)
  polish_ftol_abs          same, for the polish
  skip_polish         true bypasses the polish stage entirely; `polish_ret` is :SKIPPED
                      and the pre-polish incumbent is returned. The explicit switch,
                      because `polish_maxeval = 0` means NO LIMIT to NLopt.
  extra_seeds         points forced into the pre-testing pool
  stop_tol            early stop on |Z* - Z*_prev|; 0.0 disables
  on_sobol/on_local   callbacks for progress reporting. on_local also receives the
                      incumbent minimiser, so a caller can checkpoint after every restart.

Checkpointing and resumption (2026-09-27, plan step 4)
  state_path          a file for the versioned run state (checkpoint.jl). Written right after
                      seed selection, after every committed restart, at a pause and after the
                      polish. "" (default) keeps the run in memory only.
  resume              false: a fresh run (refused if `state_path` already holds a checkpoint);
                      true: continue from `state_path` (refused if its objective or optimizer
                      identity differs); :auto: continue if a checkpoint exists, else start.
                      A NamedTuple (seeds, f_sobol_best, Z, fZ, j_start[, history]) is the
                      LEGACY import of an old-format checkpoint: its origin and earlier
                      evaluation counts are unknown and reported as such.
  stop_after_restarts pause once this many restarts are committed (0: right after seed
                      selection; = the effective K: after the local stage, before the polish).
                      The schedule denominator stays K; resume continues at the next restart.
  objective_fields    the caller's description of the objective, recorded in the checkpoint and
                      compared field by field on resume (`objective_id` is its hash)
  allow_optimizer_change  resume despite changed SOLVER settings or optimizer SOFTWARE (identity.jl):
                      recorded as an event and a new settings epoch, and the run is labelled
                      :changed_optimizer. A change of the search PLAN -- box, dimension, pre-testing
                      design, supplied seeds, K, schedule -- is refused even so (start a new run).

A restored checkpoint must be internally consistent (`state_violations`), or the resume is
refused before anything is evaluated. Jobs a checkpoint holds IN FLIGHT are replayed from
their recorded starts and settings in either local mode, so every planned restart is
committed exactly once whichever mode the run resumes in (2026-09-28, finding 11).
"""
function tiktak(f, lo::Vector{Float64}, hi::Vector{Float64};
                N::Int = 1000, Nstar::Int = 50,
                theta_p::Float64 = 0.5, theta_lo::Float64 = 0.1, theta_hi::Float64 = 0.995,
                local_alg::Symbol = :LN_NELDERMEAD, local_tol::Float64 = 1e-3,
                local_maxeval::Int = 2000,
                # ftol_rel STRUGGLES NEAR A ZERO OPTIMUM. It tests |df| <= ftol_rel * |f|,
                # so as f -> 0 the threshold shrinks with it. Measured 2026-08-27: a
                # just-identified SMM restart reached Q ~ 0 at evaluation 61 and was still
                # going at 290 with the value unchanged. ftol_abs (in objective units) and
                # xtol_rel (relative to |x|) back it up; both are far tighter than local_tol,
                # so on a problem with a non-zero optimum ftol_rel still stops the search first.
                local_ftol_abs::Float64 = 1e-10, local_xtol_rel::Float64 = 1e-8,
                polish_alg::Symbol = :LN_BOBYQA, polish_tol::Float64 = 1e-10,
                polish_maxeval::Int = 4000, polish_ftol_abs::Float64 = 1e-14,
                skip_polish::Bool = false,
                extra_seeds::Vector{Vector{Float64}} = Vector{Vector{Float64}}(),
                stop_tol::Float64 = 0.0,
                # A4. What to do when `f` THROWS.
                #
                # `:rethrow` (the default) stops the run and shows the error. That is right
                # for this project: smm_objective already converts every EXPECTED model
                # failure into a finite penalty and re-throws only genuine bugs, so an
                # exception arriving here is a coding error, and continuing past it would
                # silently discard restarts and finish looking converged.
                #
                # `:discard` restores the old behaviour -- count it, warn, drop the point
                # and carry on -- for a caller whose objective cannot make that guarantee.
                on_error::Symbol = :rethrow,
                resume::Union{Nothing,Bool,Symbol,NamedTuple} = false,   # nothing = false (older callers)
                # Fires once, right after pre-testing, with the selected seeds.
                on_seeds = (seeds, f_sobol_best) -> nothing,
                # THREADS evaluate the PRE-TESTING stage only (a thread-safe `f` is required).
                # The synchronous threaded batches of local searches were removed on
                # 2026-09-27; `batch > 1` is refused.
                parallel::Bool = false,
                batch::Int = 1,
                # PROCESS-level parallelism for the pre-testing stage: pass `pmap` (with
                # workers added and the objective defined @everywhere). Default `map` is serial.
                map_fn = map,
                on_sobol = (i, N, fx, best) -> nothing,
                # Called after every committed restart with the incumbent value and minimiser
                # and the restart's trace row.
                on_local = (j, Nstar, theta, f_local, best, best_x, row) -> nothing,
                # INVALID CANDIDATES NEVER SEED A RESTART (2026-09-20 follow-up B). The
                # objective signals an invalid point with a large FINITE value (SMM_PENALTY).
                # Values >= invalid_value are excluded from seeding; with fewer valid points
                # than N* the restarts are reduced (warned) unless allow_fewer_restarts =
                # false, and with none the run stops. Inf keeps the old behaviour.
                invalid_value::Float64 = Inf,
                allow_fewer_restarts::Bool = true,
                skip_first::Bool = true,
                # WHICH OBJECTIVE this is (plan step 3): recorded with every origin and
                # verification, so evidence earned on one objective (a coarse search grid)
                # is never read as evidence on another (the reporting grid).
                objective_id::String = "unspecified",
                objective_fields::AbstractDict = Dict{String,Any}(),
                # See TikTakConfig: a converged solve verifies the retained point when it
                # returns that point within verify_xtol (box-normalized) and a consistent value.
                verify_xtol::Float64 = 1e-9, verify_ftol_rel::Float64 = 1e-8,
                verify_ftol_abs::Float64 = 1e-12,
                state_path::String = "",
                stop_after_restarts::Int = typemax(Int),
                allow_optimizer_change::Bool = false,
                run_id::String = "",
                # Called (read-only!) with the RunState right after each checkpoint write, so a
                # caller can derive its own files (a CSV, a summary) from the authoritative state.
                on_checkpoint = st -> nothing,
                # PRE-TESTING DESIGN AND RECOVERY (plan step 5). n_valid_target > 0 continues
                # the Sobol' sequence until n_valid_target valid draws (N is then the hard
                # attempt cap); require_valid_target makes an exhausted cap a failed gate.
                # pretest_cache: this run's chunked cache (resumed under resume = true/:auto);
                # pretest_reuse: another run's cache of the SAME objective and design, whose
                # values are taken instead of evaluated (reported as reused); pretest_chunk:
                # values between cache writes; pretest_time_cap: seconds (timing-dependent).
                n_valid_target::Int = 0,
                require_valid_target::Bool = false,
                pretest_cache::String = "",
                pretest_reuse::String = "",
                pretest_chunk::Int = 32,
                pretest_time_cap::Float64 = Inf,
                # PROCESS-PARALLEL EXECUTION (plan steps 6-7). local_mode = :async_process runs
                # the restarts asynchronously on the worker processes `local_workers` (the
                # objective registered there under `objective_key`); local_count = 0 picks
                # min(workers, max(1, floor(sqrt(K)))), capped by the restarts left.
                # pretest_workers evaluates the pre-testing stage on worker processes (no
                # batch barrier). Serial mode is the reference path and needs no worker.
                local_mode::Symbol = :serial,
                local_workers::Vector{Int} = Int[],
                local_count::Int = 0,
                pretest_workers::Vector{Int} = Int[],
                objective_key::Symbol = :default,
                # :first_alone (default) | :immediate_mixed -- the asynchronous start-up (config.jl)
                bootstrap::Symbol = :first_alone,
                # progress of running solves: on_progress(events) at most every
                # progress_every seconds (Inf: none); coalesced, never needed for a result
                progress_every::Float64 = Inf,
                on_progress = evs -> nothing,
                # a graceful pause: if this file exists, no new restart starts, the running
                # ones are committed, and the run pauses (asynchronous mode)
                pause_file::String = "",
                max_retries::Int = 1,
                drain_timeout::Float64 = 60.0,
                # when a job fails, SIGINT the other busy workers (best effort; it may end those
                # processes, so only for callers that discard their workers afterwards)
                interrupt_on_abort::Bool = false,
                # WORKER LIFECYCLE (2026-09-28, follow-up 3; workers.jl). A worker still running a
                # call of an earlier, aborted run is waited for up to busy_wait seconds and then left
                # out (quarantined) -- never given a second job. retire_after_abort = true REMOVES
                # the workers whose calls are still running at the drain deadline: only for a caller
                # that owns them (the default keeps them, leased, until their calls settle).
                busy_wait::Float64 = 30.0,
                retire_after_abort::Bool = false,
                # WHAT THE RUN IS FOR (plan step 8): a preset name recorded with the run; see presets.jl
                purpose::String = "custom",
                # SEARCH GEOMETRY AND STEPS (plan step 9; the defaults are the baseline): see
                # TikTakConfig.normalize and SolverSettings. local_initial_step and
                # polish_initial_step are fractions of the box width (0 = NLopt's heuristic).
                normalize::Bool = false,
                local_initial_step::Float64 = 0.0,
                local_step_schedule::Symbol = :fixed,
                local_step_min::Float64 = 0.1,
                polish_initial_step::Float64 = 0.0,
                _fault::Symbol = :none,           # FAULT INJECTION, tools/test_tiktak.jl only
                # Validate the configuration and, when resuming, load and verify the checkpoint
                # -- exactly as a real run would -- then return `nothing` without evaluating f.
                # A caller runs this before its expensive setup so a bad resume fails first.
                preflight_only::Bool = false)

    resume === nothing && (resume = false)
    cfg = TikTakConfig(N, n_valid_target, require_valid_target, skip_first, invalid_value, Nstar, allow_fewer_restarts,
                       theta_p, theta_lo, theta_hi,
                       SolverSettings(local_alg, local_tol, local_ftol_abs, local_xtol_rel, local_maxeval,
                                      local_initial_step, local_step_schedule, local_step_min),
                       SolverSettings(polish_alg, polish_tol, polish_ftol_abs, polish_tol, polish_maxeval,
                                      polish_initial_step, :fixed, 0.0),
                       skip_polish, stop_tol, on_error, verify_xtol, verify_ftol_rel, verify_ftol_abs, bootstrap,
                       normalize)
    # EVERYTHING IS CHECKED BEFORE THE FIRST EVALUATION (plan step 2.3).
    validate(cfg, lo, hi; n_supplied = length(extra_seeds), resuming = resume isa NamedTuple)
    batch <= 1 || throw(ArgumentError(
        "TikTak: batch = $batch -- the synchronous threaded batches of local searches were removed " *
        "on 2026-09-27. Use the process-parallel asynchronous local stage instead."))
    stop_after_restarts >= 0 || throw(ArgumentError("TikTak: stop_after_restarts must be >= 0"))
    local_mode in (:serial, :async_process) ||
        throw(ArgumentError("TikTak: local_mode must be :serial or :async_process, got :$local_mode"))
    local_mode === :async_process && isempty(local_workers) &&
        throw(ArgumentError("TikTak: local_mode = :async_process needs local_workers (worker process ids)"))
    Distributed.myid() in local_workers && throw(ArgumentError(
        "TikTak: the master coordinates; it cannot also be one of the local_workers"))
    local_count >= 0 && max_retries >= 0 && drain_timeout >= 0 && busy_wait >= 0 ||
        throw(ArgumentError("TikTak: negative execution setting"))
    resume isa Symbol && resume !== :auto && throw(ArgumentError("TikTak: resume must be true, false, :auto or a NamedTuple"))
    (resume === true && isempty(state_path) && isempty(pretest_cache)) &&
        throw(ArgumentError("TikTak: resume = true needs a state_path or a pretest_cache"))
    supplied = [checked_point(s, lo, hi; what = "extra seed $i") for (i, s) in enumerate(extra_seeds)]
    obj_fields = Dict{String,Any}(objective_fields)
    opt_fields = optimizer_fields(cfg, lo, hi, supplied)
    opt_id = fields_id(opt_fields)
    ckpt = !isempty(state_path)
    save!(st) = (refresh_semantics!(st); ckpt && (write_state(state_path, st); on_checkpoint(st); true))

    # ---- where the run starts ------------------------------------------------
    exists = ckpt && checkpoint_exists(state_path)
    cache_exists = !isempty(pretest_cache) && checkpoint_exists(pretest_cache)
    resuming = resume === true || resume === :auto
    st = if resume isa NamedTuple
        legacy_import_state(resume, cfg, lo, hi, objective_id, obj_fields, opt_id, opt_fields, run_id)
    elseif exists && resuming
        restore_state(state_path, cfg, lo, hi, objective_id, obj_fields, opt_id, opt_fields, allow_optimizer_change)
    elseif resume === true && !cache_exists
        throw(ArgumentError("TikTak: resume = true, but there is no checkpoint at $(isempty(state_path) ? pretest_cache : state_path)"))
    elseif exists || (cache_exists && !resuming)
        throw(ArgumentError("TikTak: $(exists ? state_path : pretest_cache) already holds a checkpoint. Pass resume = " *
                            "true to continue that run, or choose other paths for a new one."))
    else
        nothing                                   # a fresh local stage (pre-testing may resume from its cache)
    end
    # the pre-testing stage set up now, so a preflight also verifies a cache it would resume
    pt = st === nothing ? Pretest(lo, hi, skip_first, supplied, objective_id, obj_fields) : nothing
    if pt !== nothing
        isempty(pretest_reuse) || load_cache!(pt, pretest_reuse; imported = true)
        cache_exists && load_cache!(pt, pretest_cache)
    end
    preflight_only && return nothing

    if st === nothing
        # ---- global stage: pre-testing ----------------------------------------
        # `on_error` means the same thing at EVERY stage: under `:rethrow` (the default,
        # and what this project uses) `f` is called bare and a coding error stops the run.
        pre_events = Tuple{Symbol,String}[]               # the run state does not exist yet
        i_end, stop_reason = isempty(pretest_workers) ?
            run_pretest!(pt, f, cfg; parallel = parallel, map_fn = map_fn, chunk = pretest_chunk,
                         cache_path = pretest_cache, time_cap = pretest_time_cap, on_sobol = on_sobol) :
            pretest_async!(pt, cfg, pretest_workers, objective_key; chunk = pretest_chunk,
                           cache_path = pretest_cache, time_cap = pretest_time_cap, on_sobol = on_sobol,
                           max_retries = max_retries, busy_wait = busy_wait, drain_timeout = drain_timeout,
                           retire_after_abort = retire_after_abort,
                           log = (k, m) -> push!(pre_events, (k, String(m))))
        if cfg.n_valid_target > 0 && stop_reason !== :valid_target
            msg = "the valid-draw target was not reached: $(count(k -> valid_value(pt.values[k], invalid_value), 1:i_end)) " *
                  "valid of $(cfg.n_valid_target) wanted in $i_end draws (stopped by $stop_reason)"
            cfg.require_valid_target && throw(PretestGateError(msg))
            @warn "TikTak: " * msg
        end
        cands = Vector{Float64}[copy(point!(pt, k)) for k in 1:i_end]
        append!(cands, pt.supplied)
        fs = vcat(pt.values[1:i_end], pt.svalues)
        errored = vcat(pt.errored[1:i_end], pt.serrored)
        valid_order, K = select_seeds(fs, cfg, Nstar; n_errors = count(errored))
        pretest = pretest_summary(pt, i_end, stop_reason, cfg, valid_order, K)
        st = new_state(cfg, lo, hi, cands, fs, valid_order, K, i_end, pretest, objective_id, obj_fields,
                       opt_id, opt_fields, isempty(run_id) ? new_run_id() : run_id)
        # lifetime: every value this run evaluated (overshoot included), not the imported ones
        st.evals_pretest = count(pt.done) + count(pt.sdone) - pt.imported
        st.evals_segment = pt.evals_segment
        st.pretest_lost = pt.lost                        # dispatched evaluations whose value never arrived
        add_event!(st, :start, "K = $K of $Nstar requested; pre-testing pool $i_end draws ($stop_reason)" *
                   (pt.reused > 0 ? ", $(pt.reused) values from a cache" : "") *
                   (pt.lost > 0 ? "; $(pt.lost) pre-testing evaluation(s) lost (value never arrived)" : ""))
        for (k, m) in pre_events; add_event!(st, k, "pre-testing: " * m); end
        on_seeds(st.seeds, st.f_sobol_best)
    end
    # the local workers actually used: the automatic count is min(pool, sqrt(K)), capped by the
    # restarts still to run -- a conservative default, not a mathematical limit
    remaining = max(1, st.K - length(st.records))
    nloc = local_mode === :serial ? 0 :
           min(length(local_workers), remaining, local_count > 0 ? local_count : max(1, floor(Int, sqrt(st.K))))
    ws = local_workers[1:nloc]
    resolve_preset(purpose)                          # an unknown label is an error
    if st.purpose != purpose
        st.resume_semantics === :fresh || add_event!(st, :purpose, "purpose label $(st.purpose) -> $purpose")
        st.purpose = purpose
    end
    start_segment!(st, local_mode, nloc, stop_after_restarts,
                   st.resume_semantics === :fresh && isempty(st.segments) ? "start" : "resume")
    if st.stage === :complete
        # (restore_state has verified the saved state, completion invariants included)
        add_event!(st, :resume, "the checkpointed run was already complete; nothing re-run")
        st.resume_semantics = :already_complete
        return build_result(st; state_path = state_path)
    end
    # the resume label comes from the WHOLE history (continuation_semantics): an asynchronous
    # segment with several workers, a replayed job or a changed optimizer is never "serial exact"
    refresh_semantics!(st)
    st.status = :running
    save!(st)                          # the stage-local checkpoint right after seed selection

    # ---- local stage --------------------------------------------------------
    if local_mode === :serial
        local_stage_serial!(st, f, cfg, stop_after_restarts, save!, on_local)
    else
        local_stage_async!(st, cfg, AsyncExec(ws, objective_key, progress_every, stop_after_restarts, pause_file,
                                              max_retries, drain_timeout, interrupt_on_abort, _fault,
                                              busy_wait, retire_after_abort),
                           save!, on_local, on_progress)
    end
    st.status === :paused && return build_result(st; state_path = state_path)
    if st.stage === :local
        # COMPLETENESS IS VALIDATED, not inferred from the cursor (follow-up 1.2): every planned
        # restart committed exactly once and nothing in flight, or a recorded stop_tol stop
        assert_consistent!(st, save!, "the end of the local stage"; complete = true)
        st.stage = :local_complete
        st.f_prepolish = st.inc.f
        save!(st)
    end

    # ---- polishing ----------------------------------------------------------
    if n_committed(st) >= stop_after_restarts && !skip_polish && !st.polish.done
        pause!(st, "stop_after_restarts = $stop_after_restarts reached before the polish")
        save!(st)
        return build_result(st; state_path = state_path)
    end
    if !skip_polish && !st.polish.done
        pj = local_mode === :serial ? polish_job(st) :
             polish_job(st; objective_key = objective_key, master = Distributed.myid())
        r, pj = try
            local_mode === :serial ? (run_local(f, pj), pj) :
                run_job_on_pool(ws, pj; max_retries = max_retries, busy_wait = busy_wait,
                                on_lost = (j_, msg) -> (account_unknown!(st, attempt_key(j_), :lost); st.retries += 1;
                                                        add_event!(st, :worker_lost, "the polish (attempt $(j_.attempt)) lost " *
                                                                   "its worker ($msg); re-dispatched from the same point")),
                                log = (k, m) -> add_event!(st, k, "polish: " * m))
        catch e
            # the failed attempt's work is unknown; an interrupted wait leaves the call running (leased)
            k_ = e isa RemoteJobError ? AttemptKey(st.run_id, :polish, 0, e.attempt) : attempt_key(pj)
            account_unknown!(st, k_, e isa RemoteJobError && e.kind === :worker_lost ? :lost :
                                     e isa InterruptException ? :running_at_stop : :failed)
            fail!(st, "the polish threw: " * sprint(showerror, e)); save!(st)
            rethrow()
        end
        commit_polish!(st, pj, r)
    end
    st.stage = :complete
    st.status = :complete
    assert_consistent!(st, save!, "the end of the run"; complete = true)
    add_event!(st, :complete, st.stopped_early ? "stopped early (stop_tol)" : "planned restarts complete")
    save!(st)
    return build_result(st; state_path = state_path)
end

"""
    assert_consistent!(st, save!, where; complete)

Check the state invariants at a point where they must hold. A violation is a bug in this
module, never an input problem: the run is marked failed, checkpointed, and stops with every
violation named -- it is never reported complete.
"""
function assert_consistent!(st::RunState, save!, where::AbstractString; complete::Bool)
    v = state_violations(st; complete = complete)
    isempty(v) && return st
    fail!(st, "state invariant(s) violated at $where: " * join(v, "; "))
    try save!(st) catch end
    throw(StateInvariantError("at $where: " * join(v, "; ")))
end

"Mark the run paused (a clean, resumable stop at a restart boundary)."
function pause!(st::RunState, why::AbstractString)
    st.status = :paused
    add_event!(st, :pause, why)
    return st
end

"Mark the run failed; the committed state is kept and a resume continues from it."
function fail!(st::RunState, why::AbstractString)
    st.status = :failed
    add_event!(st, :failure, why)
    return st
end

"""
    local_stage_serial!(st, f, cfg, stop_after, save!, on_local)

The published sequential local stage: restart j starts from its seed mixed with the
incumbent that restarts 1..j-1 left, runs on this process, and is committed and
checkpointed before restart j+1 is built. A throw from `f` (under `on_error = :rethrow`)
records a failure event and saves the last committed state before propagating, so a resume
re-runs that restart from the same incumbent -- and therefore from the same start.

JOBS IN FLIGHT FIRST (2026-09-28, finding 11). A checkpoint written by the asynchronous stage
may hold dispatched, uncommitted restarts below the cursor `next_j`. They are replayed here,
in restart order, from their RECORDED start, theta, incumbent version and settings epoch --
never rebuilt, never skipped -- before any new restart is built. (Until 2026-09-28 the serial
loop resumed at `next_j` and never looked at them: a run could finish "complete" with a
restart missing.)
"""
function local_stage_serial!(st::RunState, f::F, cfg::TikTakConfig, stop_after::Int, save!, on_local) where {F}
    me = Distributed.myid()
    for fl in sort(collect(values(st.inflight)); by = fl -> fl.job.j)
        job = redispatch(fl.job, me, fl.job.objective_key, Inf)
        st.dispatch_seq += 1
        # still in flight, now as the replay attempt: a crash during the replay leaves it
        # recorded for the next resume
        st.inflight[job.j] = InFlight(job, me, time(), st.dispatch_seq, st.commit_seq)
        add_event!(st, :replay, "restart $(job.j) was in flight at the checkpoint: replayed serially from its " *
                   "recorded start (attempt $(job.attempt), settings epoch $(job.epoch))")
        save!(st)
        r = try
            run_local(f, job)
        catch e
            account_unknown!(st, attempt_key(job), :failed)
            fail!(st, "the replay of restart $(job.j) threw: " * sprint(showerror, e)); save!(st)
            rethrow()
        end
        fl2 = st.inflight[job.j]
        delete!(st.inflight, job.j)
        rec = commit_restart!(st, job, r; dispatch_seq = fl2.dispatch_seq, commits_at_dispatch = fl2.commits_at_dispatch)
        save!(st)
        on_local(rec.j, st.K, rec.theta, rec.f_local, st.inc.f, st.inc.x, trace_row(rec))
    end
    x0buf = Vector{Float64}(undef, length(st.lo))
    while st.next_j <= st.K && !st.stopped_early
        if n_committed(st) >= stop_after
            pause!(st, "stop_after_restarts = $stop_after reached at restart $(st.next_j - 1) of $(st.K)")
            save!(st)
            return st
        end
        j = st.next_j
        job = restart_job(st, j, next_attempt(st, :local, j), x0buf)
        st.dispatch_seq += 1
        r = try
            run_local(f, job)
        catch e
            account_unknown!(st, attempt_key(job), :failed)
            fail!(st, "restart $j threw: " * sprint(showerror, e)); save!(st)
            rethrow()
        end
        rec = commit_restart!(st, job, r; dispatch_seq = st.dispatch_seq)
        st.next_j = j + 1
        save!(st)
        on_local(j, st.K, rec.theta, rec.f_local, st.inc.f, st.inc.x, trace_row(rec))
    end
    return st
end

"""
    new_state(...) -> RunState

The state right after seed selection: seeds and their values, the incumbent (the best valid
candidate, with its origin), K, and the pre-testing summary.
"""
function new_state(cfg::TikTakConfig, lo, hi, cands, fs, valid_order, K, n_sobol, pretest,
                   objective_id, obj_fields, opt_id, opt_fields, run_id)
    sel = valid_order[1:K]
    seeds = [copy(cands[k]) for k in sel]
    origin_of(k) = k <= n_sobol ? :sobol : :supplied
    k1 = valid_order[1]
    o = k1 <= n_sobol ? CandidateOrigin(:sobol, 0, 1, k1, :SOBOL_ONLY, objective_id) :
                        CandidateOrigin(:supplied, 0, 1, k1 - n_sobol, :SUPPLIED, objective_id)
    # The incumbent and its value come from the SAME index (finding 9).
    inc = Incumbent(copy(cands[k1]), fs[k1], 1, o, NO_VERIFICATION)
    records = RestartRecord[]
    sizehint!(records, K)
    return RunState(run_id, copy(lo), copy(hi), cfg, objective_id, obj_fields, opt_id, opt_fields,
                    cfg.nstar, K, seeds, fs[sel], Symbol[origin_of(k) for k in sel],
                    Int[k <= n_sobol ? k : k - n_sobol for k in sel], fs[k1], pretest,
                    inc, records, Dict{Int,InFlight}(), 1, 0, 0, 0, Inf, false, 0, POLISH_PENDING, NaN,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, true, :local, :running, :fresh, "custom",
                    [first_epoch(cfg, opt_id)], Dict{AttemptKey,Symbol}(), 0, Segment[], RunEvent[])
end

"""
    restore_state(path, cfg, lo, hi, objective_id, obj_fields, opt_id, opt_fields, allow_change) -> RunState

Load the newest valid checkpoint generation and refuse it unless it describes THIS run, in
this order, before anything is evaluated (the preflight runs exactly this):

  1. a schema this code reads;
  2. the same objective (identity hash, compared field by field for the message);
  3. the same search PLAN -- box and dimension, pre-testing design, supplied seeds, K, mixing
     schedule, validity threshold, stop_tol, verification, bootstrap. Refused even with
     `allow_change`: the saved seeds, incumbent and records belong to the saved plan, and
     continuing them under another one returns points of the wrong box (finding 14);
  4. the same solver settings and optimizer software, unless `allow_change` -- then a new
     settings epoch is opened, so earlier records and in-flight jobs keep what they ran with;
  5. an internally consistent state (`state_violations`): a checkpoint that claims completion
     with restarts missing or in flight, commits a restart twice, or holds an incumbent
     outside its box is refused, never continued or reported.
"""
function restore_state(path, cfg, lo, hi, objective_id, obj_fields, opt_id, opt_fields, allow_change::Bool)
    d, which = read_state_dict(path)
    check_schema(d)
    which === :current || @warn "TikTak: the newest checkpoint generation did not verify; resuming from the $which one" path
    saved_obj = d["objective"]
    String(saved_obj["id"]) == objective_id || throw(ResumeRefused(
        "the objective changed (saved id $(saved_obj["id"]), now $objective_id):\n  " *
        join(identity_differences(saved_obj["fields"], obj_fields), "\n  ")))
    saved_opt = d["optimizer"]
    chg = identity_change(saved_opt["fields"], opt_fields)
    isempty(chg.plan) || throw(ResumeRefused(
        "the requested run is a different SEARCH PLAN from the saved one -- refused even with " *
        "allow_optimizer_change, because the saved seeds, incumbent and records belong to the saved plan:\n  " *
        join(chg.plan, "\n  ") *
        "\n  To build on that run, start a NEW run: pretest_reuse = <its pre-testing cache> reuses its " *
        "pre-testing values when the objective and candidate design match, and extra_seeds = [<its best point>] " *
        "warm-starts from it (the point is re-evaluated; no convergence evidence carries over). Never clamp old " *
        "points into a new box."))
    changed = vcat(chg.software, chg.solver)
    (isempty(changed) || allow_change) || throw(ResumeRefused(
        "the optimizer changed (a different search, not a continuation):\n  " * join(changed, "\n  ") *
        "\n  Pass allow_optimizer_change = true (--allow-optimizer-change) to continue under the new " *
        (isempty(chg.solver) ? "software" : "settings") * ": recorded as a new settings epoch; committed restarts " *
        "and jobs in flight keep the settings they ran with, and the run is labelled changed_optimizer."))
    st = state_from_dict(d, cfg)
    (st.lo == lo && st.hi == hi) || throw(ResumeRefused("the saved box differs from the requested one"))   # (the plan check covers it)
    v = state_violations(st)
    isempty(v) || throw(ResumeRefused("the checkpoint is internally inconsistent, so it can be neither " *
                                      "continued nor reported:\n  " * join(v, "\n  ")))
    if !isempty(changed)
        push!(st.epochs, SettingsEpoch(length(st.epochs) + 1, cfg.local_, cfg.polish, cfg.normalize, cfg.skip_polish,
                                       opt_id, length(st.segments) + 1, "resume with allow_optimizer_change: " *
                                       join(changed, "; ")))
        add_event!(st, :optimizer_change, "resumed under a changed optimizer (allowed explicitly); settings epoch " *
                   "$(length(st.epochs)) from here: " * join(changed, "; "))
        st.optimizer_id, st.optimizer_fields = opt_id, opt_fields
    end
    which === :current || add_event!(st, :recovered, "loaded the $which checkpoint generation")
    add_event!(st, :resume, "resumed at restart $(st.next_j) of $(st.K) with $(length(st.inflight)) job(s) in " *
               "flight, status was $(st.status)")
    return st
end

"""
    legacy_import_state(r, cfg, lo, hi, ...) -> RunState

An old-format checkpoint (seeds, f_sobol_best, Z, fZ, j_start[, history]): the seeds and
incumbent are kept, their number is the schedule denominator (finding 4), the incumbent's
origin is `:legacy_import` with return code :UNKNOWN, and the earlier evaluation counts are
UNKNOWN (`counts_complete = false`) -- nothing is invented. `history` rows (from an old
restarts.csv) become legacy records.
"""
function legacy_import_state(r::NamedTuple, cfg, lo, hi, objective_id, obj_fields, opt_id, opt_fields, run_id)
    seeds = [checked_point(sd, lo, hi; what = "resumed seed $i") for (i, sd) in enumerate(r.seeds)]
    K = length(seeds)
    1 <= K <= cfg.nstar || error("resume has $K seeds but at most Nstar = $(cfg.nstar) can be resumed")
    1 <= r.j_start <= K + 1 || error("resume.j_start = $(r.j_start) is outside 1..$(K + 1)")
    Z = checked_point(r.Z, lo, hi; what = "resumed incumbent")
    valid_value(Float64(r.fZ), cfg.invalid_value) || error("resumed incumbent value $(r.fZ) is not a valid objective value")
    inc = Incumbent(Z, Float64(r.fZ), 1, CandidateOrigin(:legacy_import, 0, 1, 0, :UNKNOWN, objective_id), NO_VERIFICATION)
    # one legacy record per restart below j_start: an old restarts.csv that lists a restart twice
    # (an old-style re-run) keeps its LAST row, and the import says so
    rows = Dict{Int,Any}()
    ndup = 0
    for h in get(r, :history, NamedTuple[])
        h.j < r.j_start || continue
        haskey(rows, h.j) && (ndup += 1)
        rows[h.j] = h
    end
    records = RestartRecord[]
    for jh in sort(collect(keys(rows)))
        h = rows[jh]
        push!(records, RestartRecord(h.j, 1, h.theta, Float64[], 0, h.f_start, Float64[], h.f_local, -1, h.ret,
                                     h.improved ? :improved : :none, 0, 0, NaN, length(records) + 1, 0,
                                     length(records), "", true, 1))
    end
    pretest = PretestSummary(0, 0, 0, 0, 0, 0, 0, K, 0, 0, 0, 0, true, :resumed)
    rid = isempty(run_id) ? new_run_id() : run_id
    # restarts 1..j_start-1 ran before the import: recorded when the old history has them, and
    # never re-run either way (`legacy_through`)
    accounted = Dict{AttemptKey,Symbol}(AttemptKey(rid, :local, rec.j, 1) => :legacy for rec in records)
    st = RunState(rid, copy(lo), copy(hi), cfg, objective_id, obj_fields,
                  opt_id, opt_fields, cfg.nstar, K, seeds, fill(NaN, K), fill(:legacy_import, K), zeros(Int, K),
                  Float64(r.f_sobol_best), pretest, inc, records, Dict{Int,InFlight}(), r.j_start, length(records), 0, 0,
                  Inf, false, 0, POLISH_PENDING, NaN, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, :local, :running, :legacy_import,
                  "custom", [first_epoch(cfg, opt_id)], accounted, r.j_start - 1, Segment[], RunEvent[])
    add_event!(st, :legacy_import, "imported an old-format checkpoint at restart $(r.j_start) of $K; " *
               "incumbent origin and earlier evaluation counts unknown" *
               (ndup > 0 ? "; $ndup repeated history row(s) dropped (the last row of each restart kept)" : ""))
    return st
end
