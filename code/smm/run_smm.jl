#!/usr/bin/env julia
# =============================================================================
# run_smm.jl -- estimate four parent parameters on four data means.
#
#     cd code/smm && julia --project=../.. run_smm.jl --quick     # 2 min smoke test
#     cd code/smm && julia --project=../.. run_smm.jl             # the real run
#
# See README.md in this folder for flags, runtimes and how to read the output.
# See moments.jl for what each moment is and how model units map to dollars/hours.
#
# Everything this run produces goes to output/smm_runs/<timestamp>/ :
#     run.log         the full console transcript, exactly as it appeared
#     estimates.toml  the estimated parameters, the fit, and the budget that made it
#
# -----------------------------------------------------------------------------
# PARALLELISM, AND WHY ONLY HALF THE RUN IS PARALLEL
# -----------------------------------------------------------------------------
# Worker PROCESSES (Distributed.jl), never threads. NLopt.jl is not thread-safe
# in this project -- with threads the objective killed the process with exit 0
# and no error message. Each worker process owns its NLopt state, so the hazard
# cannot arise. See the header of ../src/tiktak.jl.
#
# TikTak has two stages and they do NOT parallelise the same way:
#
#   Sobol stage    N independent evaluations. Embarrassingly parallel: this is
#                  the part that gets faster when you add workers.
#
#   Local stage    N* Nelder-Mead searches, SEQUENTIAL BY CONSTRUCTION. Restart j
#                  starts from a mix of its own seed and the best point found by
#                  restarts 1..j-1, so it cannot start before j-1 finishes. No
#                  number of workers speeds this up.
#
# On a wide machine the local stage therefore dominates the wall clock. The run
# prints the two halves of the runtime estimate SEPARATELY, before the search
# starts, so you can see that before committing to a budget. If you want to spend
# a big machine on this problem, spend it on --sobol (more, better seeds), not on
# --restarts.
#
# BLAS is pinned to one thread per worker. Without that, 20 worker processes each
# open a BLAS pool sized to all 112 cores and the machine thrashes -- which on a
# SHARED server is everyone else's problem too, not just a slow run.
# =============================================================================

using Distributed, Printf, Dates, LinearAlgebra

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------
function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("$flag needs a value")
    return parse(Int, ARGS[i + 1])
end
function argstr(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("$flag needs a value")
    return ARGS[i + 1]
end

const QUICK       = "--quick"       in ARGS
const SERIAL      = "--serial"      in ARGS
const REPORT_ONLY = "--report-only" in ARGS
const N_SOBOL     = argval("--sobol",    QUICK ? 12 : 200)
const N_RESTART   = argval("--restarts", QUICK ?  2 :  10)
const EVERY_SEC   = float(argval("--every", 2))   # progress line throttle, seconds

# -----------------------------------------------------------------------------
# Search grid vs report grid
# -----------------------------------------------------------------------------
# 98% of an evaluation is solve_model!, and its cost scales with the parent's
# Na x Nhc. MEASURED at full grids, 2026-08-27:
#
#     Na=Nhc=30  solve 11.62s  simulate 0.24s     c_p 3.0153  l_p 0.4703  e_p 2.1884
#     Na=Nhc=20  solve  4.78s  simulate 0.03s     c_p 3.0149  l_p 0.4701  e_p 2.1924
#     Na=Nhc=15  solve  2.71s  simulate 0.01s     c_p 3.0279  l_p 0.4693  e_p 2.2179
#
# Dropping the parent grid 30 -> 20 makes an evaluation 2.5x cheaper and moves the
# three targeted moments by 0.01%, 0.04% and 0.2% -- against gaps of 3%, 11% and
# 461% that the estimation is trying to close. The search simply does not need the
# resolution; only the answer does.
#
# So --grid sets the grid the SEARCH runs on, and the final fit is ALWAYS reported
# at the full grid, re-solved from the estimated parameters. Never report a Q that
# was minimised on a coarse grid -- run the search cheap, quote the answer exact.
#
# Simulation is NOT the place to economise: it is 2% of the cost, and cutting simN
# to 500 moved c_p more (3.0275) than halving the grid did. simN stays at 2000.
const GRID_FULL   = QUICK ? 12 : 30
const GRID_SEARCH = argval("--grid", GRID_FULL)
const SIM_N       = QUICK ? 300 : 2000

# -----------------------------------------------------------------------------
# How many workers
# -----------------------------------------------------------------------------
# THIS IS A SHARED MACHINE. Sys.CPU_THREADS is what the box HAS (112 here), not
# what this job may take. WORKER_BUDGET is the house rule -- the number agreed as
# a fair share -- and it binds before either hardware limit does. Raise it only
# by agreement with whoever else is on the machine, not because the cores look idle.
#
# The two hardware caps still apply underneath, so this file also runs sensibly on
# a laptop: N_CORES-1 leaves a core for the master, and the RAM cap reflects that
# every worker is a full Julia process holding its own copy of the solved child
# model. Whichever of the three is smallest wins, and the run prints which one bound.
const WORKER_BUDGET = 20
const N_CORES = Sys.CPU_THREADS
const RAM_GB  = Sys.total_memory() / 2^30
const RAM_CAP = max(1, floor(Int, RAM_GB / 2.0))     # ~2 GB per worker process
const SAFE_MAX = max(1, min(WORKER_BUDGET, N_CORES - 1, RAM_CAP))
const NPROC    = argval("--procs", SERIAL ? 0 : SAFE_MAX)

bound_by() = SAFE_MAX == WORKER_BUDGET ? "shared-server budget" :
             SAFE_MAX == N_CORES - 1    ? "cores"               : "RAM"

# -----------------------------------------------------------------------------
# Run directory and logging
# -----------------------------------------------------------------------------
const STAMP   = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
const RUN_DIR = argstr("--outdir", joinpath(REPO, "output", "smm_runs", STAMP))
mkpath(RUN_DIR)
const LOG = open(joinpath(RUN_DIR, "run.log"), "w")

# Paths are printed relative to the repo root so the log stays readable (and the
# same whoever ran it). `relpath`, not string surgery: normpath leaves a trailing
# slash on REPO, so stripping REPO * "/" by hand silently does nothing.
short(p) = relpath(p, REPO)

# One lock for all output. The progress watcher below runs as a separate task and
# would otherwise interleave half-written lines with the main task's.
const OUTLOCK = ReentrantLock()

"""
Print to the console and to the run log at once, so a finished run is readable
after the fact.

BOTH streams are flushed on every line. stdout is only line-buffered when it is a
terminal -- the moment the run is detached (`nohup ... > out.txt`, which is how any
multi-hour run should be started) it becomes block-buffered, and without the flush
progress sits invisibly in a 4 KB buffer while the run is in fact working fine.
Measured: the log showed restart 1 at eval 130 while the console still read
"timing one objective evaluation".
"""
function say(args...)
    lock(OUTLOCK) do
        println(args...); println(LOG, args...)
        flush(stdout); flush(LOG)
    end
end
function sayf(fmt, args...)
    s = Printf.format(Printf.Format(fmt), args...)
    lock(OUTLOCK) do
        print(s); print(LOG, s)
        flush(stdout); flush(LOG)
    end
end
banner(s) = (say(); say("="^76); say(s); say("="^76))

banner("SMM: parent-block moments" * (QUICK ? "   [QUICK -- smoke test, not an estimate]" : ""))
sayf("started    %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
sayf("host       %s\n", gethostname())
sayf("machine    %d cores, %.0f GB RAM, load average %.1f\n",
     N_CORES, RAM_GB, Sys.loadavg()[1])
if SERIAL
    sayf("workers    0 (--serial: everything on the master, for debugging)\n")
else
    sayf("workers    %d of %d cores (%.0f%%) -- capped by %s\n",
         NPROC, N_CORES, 100 * NPROC / N_CORES, bound_by())
end
sayf("budget     %d Sobol points, %d restarts\n", N_SOBOL, N_RESTART)
if GRID_SEARCH != GRID_FULL
    sayf("grids      search at Na=Nhc=%d, fit REPORTED at Na=Nhc=%d\n", GRID_SEARCH, GRID_FULL)
else
    sayf("grids      Na=Nhc=%d for both the search and the report\n", GRID_FULL)
end
sayf("writing to %s\n", short(RUN_DIR))

# -----------------------------------------------------------------------------
# Start the workers
# -----------------------------------------------------------------------------
# Each worker inherits --project so it resolves the same Manifest.toml, and gets
# a single BLAS thread (see the header).
if !SERIAL && NPROC > 0 && (nprocs() - 1) < NPROC
    print("starting $NPROC worker processes ... "); flush(stdout)
    t = time()
    addprocs(NPROC - (nprocs() - 1); exeflags = `--project=$REPO`)
    sayf("%.1fs\n", time() - t)
end
@everywhere using LinearAlgebra
@everywhere LinearAlgebra.BLAS.set_num_threads(1)
sayf("running on %d process(es): 1 master + %d worker(s), 1 BLAS thread each\n",
     nprocs(), max(0, nprocs() - 1))

# -----------------------------------------------------------------------------
# Load the model on every process
# -----------------------------------------------------------------------------
print("loading the model on every process ... "); flush(stdout)
t = time()
@everywhere begin
    using Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames
    using Statistics, Dates, ProgressMeter, Distributions, StatsBase
    using QuantEcon, FastGaussQuadrature, Parameters, Dierckx, TOML
    const REPO_ = normpath(joinpath(@__DIR__, "..", ".."))
    const SRC   = joinpath(REPO_, "code", "src")
    include(joinpath(SRC, "paths.jl"))
    include(joinpath(SRC, "manifest.jl"))       # git_sha(), used when writing estimates.toml
    include(joinpath(SRC, "diagnostics.jl"))
    include(joinpath(SRC, "child_lifecycle.jl"))
    include(joinpath(SRC, "parent_family.jl"))
    include(joinpath(SRC, "tiktak.jl"))
    include(joinpath(REPO_, "code", "smm", "moments.jl"))
end
sayf("%.1fs\n", time() - t)

@everywhere const TARGETS  = load_targets(joinpath(REPO_, "Input", "smm_targets_baseline.toml"))
@everywhere const CHILD_G_ = $(QUICK ? (Na = 12, Nk = 12, Nt = 3) : (Na = 30, Nk = 30, Nt = 5))
# G_ is what every objective evaluation uses; G_FULL_ is what the reported fit uses.
# They are the same unless --grid was passed. The child grid is untouched either way.
@everywhere const G_      = (Na = $GRID_SEARCH, Nk = 2, Nhc = $GRID_SEARCH, simN = $SIM_N)
@everywhere const G_FULL_ = (Na = $GRID_FULL,   Nk = 2, Nhc = $GRID_FULL,   simN = $SIM_N)

# -----------------------------------------------------------------------------
# Child solve: once per process, reused by every evaluation
# -----------------------------------------------------------------------------
# The estimated parameters are ALL parent-block, so the child lifecycle, its
# transfer stage and the terminal value spline do not depend on them. Solving
# once per process and reusing is EXACT, not an approximation, and it is what
# makes each evaluation cheap enough to run thousands of.
@everywhere function build_child_value()
    ch = ConSavLaborCollege_AR1(; Na = CHILD_G_.Na, Nk = CHILD_G_.Nk, Nt = CHILD_G_.Nt,
                                  rho = 1.5, psi_terminal = 4.0, kappa_terminal = 5.0,
                                  omega = 0.3, a_max = 100.0, w = 20.0,
                                  simN = 500, seed = 1234)
    # ProgressMeter writes its bar to STDERR, so redirect_stdout alone leaves a
    # half-drawn "Solving working model... 72%" across our own startup line.
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            solve_model_work!(ch); solve_model_college!(ch)
            optimal_transfer_work!(ch); optimal_transfer_college!(ch)
        end
    end
    return terminal_value_spline(ch; s = 10.0)
end

print("solving the child value function on every process ... "); flush(stdout)
t = time()
@everywhere const V_CHILD = build_child_value()
sayf("%.1fs (all processes at once)\n", time() - t)

@everywhere objective(z) = smm_objective(z, TARGETS, V_CHILD;
                                         Na = G_.Na, Nk = G_.Nk, Nhc = G_.Nhc,
                                         simN = G_.simN)

# -----------------------------------------------------------------------------
# Live progress
# -----------------------------------------------------------------------------
# The problem this solves: `pmap` hands the whole Sobol batch to the workers and
# returns only when ALL of it is done, so tiktak's on_sobol callback fires N times
# in a burst at the end. A 40-minute stage would print nothing and then 200 lines.
#
# So every evaluation reports the moment it finishes:
#   on a WORKER   push the value into a RemoteChannel; a task on the master drains
#                 it and prints. pmap yields while waiting, so the drain task runs.
#   on the MASTER (the local stage, which runs here) print directly -- an NLopt
#                 search does not yield, so a channel would buffer until it ended.
#
# Both paths funnel into tick!, which throttles to one line per --every seconds.
const PROGRESS = RemoteChannel(() -> Channel{Float64}(1 << 15))

mutable struct Tracker
    stage::Symbol      # :sobol or :local
    done::Int          # evaluations finished in the current stage
    total::Int         # expected evaluations, :sobol only (:local has no known total)
    best::Float64
    restart::Int
    nrestart::Int
    t0::Float64        # start of the CURRENT stage, for the Sobol ETA
    trun::Float64      # start of the whole search, so every line agrees on the clock
    tlast::Float64
end
const TRACKER = Tracker(:sobol, 0, 0, Inf, 1, N_RESTART, time(), time(), time())

function stage!(s::Symbol, total::Int = 0)
    TRACKER.stage = s; TRACKER.done = 0; TRACKER.total = total
    TRACKER.t0 = time(); TRACKER.tlast = time()
end

function tick!(q::Float64)
    T = TRACKER
    T.done += 1
    isfinite(q) && q < T.best && (T.best = q)
    now_ = time()
    last = T.stage === :sobol && T.done == T.total      # always print the final line
    (last || now_ - T.tlast >= EVERY_SEC) || return
    T.tlast = now_
    if T.stage === :sobol
        el   = (now_ - T.t0) / 60
        frac = T.done / max(T.total, 1)
        eta  = frac > 0 ? el * (1 - frac) / frac : 0.0
        sayf("  sobol    %5d/%-5d %3.0f%%   best Q %11.4g   %5.1f min elapsed, ~%.0f min left\n",
             T.done, T.total, 100frac, T.best, el, eta)
    else
        sayf("  restart %3d/%-3d  eval %5d   this Q %11.4g   best Q %11.4g   %5.1f min\n",
             T.restart, T.nrestart, T.done, q, T.best, (now_ - T.trun) / 60)
    end
    return
end

# Defined on every process: workers take the channel branch, the master prints.
@everywhere const PROGRESS_ = $PROGRESS
@everywhere function objective_tracked(z)
    q = objective(z)
    if myid() == 1
        Main.tick!(q)
    else
        try; put!(PROGRESS_, q); catch; end     # progress must never break the run
    end
    return q
end

"""Drain the progress channel until `total` values have arrived. Returns the task."""
function watch_progress(total::Int)
    return @async begin
        try
            n = 0
            while n < total
                tick!(take!(PROGRESS)); n += 1
            end
        catch err
            say("  (progress watcher stopped: $err)")
        end
    end
end

# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------
banner("Targets  (Input/smm_targets_baseline.toml)")
sayf("%d moments, %d parameters -- %s\n", length(SMM_MOMENTS), length(SMM_PARAMS),
     length(SMM_MOMENTS) == length(SMM_PARAMS) ? "just-identified, Q can reach 0" :
     length(SMM_MOMENTS) >  length(SMM_PARAMS) ? "OVER-identified, Q cannot reach 0 and weights matter" :
                                                 "UNDER-identified: more parameters than moments")
sayf("parameters: %s\n", join((String(q.name) for q in SMM_PARAMS), ", "))

sayf("%-10s %10s %10s %8s   %s\n", "moment", "mean", "sd", "N", "source")
for k in SMM_MOMENTS
    tg = TARGETS[k]
    sayf("%-10s %10.4f %10.4f %8d   %s\n", k, tg.mean, tg.sd, tg.n, tg.source)
end
say("\nSDs are shown but NOT targeted: the model's cross-sectional dispersion comes")
say("from a 5-node wage shock plus initial draws and cannot reach the data's")
say("(leisure SD is 7.4x too small). Targeting them would distort the means.")

"""
Solve once at `z`, print the fit report to BOTH the console and the log.

Always at G_FULL_, never at the search grid: the number that leaves this run has
to be the fit at full resolution, whatever grid the optimizer happened to use to
find the parameters.
"""
function say_report(z)
    buf = IOBuffer()
    r = report_fit(z, TARGETS, V_CHILD; Na = G_FULL_.Na, Nk = G_FULL_.Nk, Nhc = G_FULL_.Nhc,
                   simN = G_FULL_.simN, out = buf)
    s = String(take!(buf))
    lock(OUTLOCK) do
        print(s); print(LOG, s); flush(LOG)
    end
    return r
end

# -----------------------------------------------------------------------------
# Time one evaluation, then predict the run
# -----------------------------------------------------------------------------
x0 = incumbent()
print("timing one objective evaluation ... "); flush(stdout)
t = time(); q0 = objective(x0); T_EVAL = time() - t
sayf("%.1fs\n", T_EVAL)

const NW = max(1, nprocs() - 1)
# The two stages are estimated separately because they scale differently: the
# Sobol stage divides by the worker count, the local stage does not divide at all.
#
# 165 evaluations per restart is MEASURED, not a guess. The 2026-08-27 verification
# run spent 373 evaluations on 40 Sobol points + 2 restarts + the polish, i.e. ~166
# per restart with the polish amortised in. The earlier prior of 60 understated the
# run by 2.7x, which is the difference between "this finishes over lunch" and "this
# finishes this evening" -- so it is worth keeping honest.
const N_SOBOL_EVAL = N_SOBOL + 1                 # +1: the incumbent is seeded in
const N_LOCAL_EVAL = N_RESTART * 165
const MIN_SOBOL = N_SOBOL_EVAL * T_EVAL / NW / 60
const MIN_LOCAL = N_LOCAL_EVAL * T_EVAL / 60
sayf("\nprojected runtime\n")
sayf("  sobol stage   %6d evals / %2d workers  = %6.1f min   (parallel)\n",
     N_SOBOL_EVAL, NW, MIN_SOBOL)
sayf("  local stage   %6d evals, sequential    = %6.1f min   (cannot be parallelised)\n",
     N_LOCAL_EVAL, MIN_LOCAL)
sayf("  total                                    %6.1f min\n", MIN_SOBOL + MIN_LOCAL)
say("\nThe local stage dominates because TikTak's restarts are sequential by")
say("construction. More workers shorten the first line only -- to use a wide")
say("machine well, raise --sobol (better seeds), not --restarts.")

banner("Incumbent calibration")
sayf("Q = %.6f\n", q0)
say_report(x0)

if REPORT_ONLY
    banner("--report-only: stopping before the search")
    sayf("wrote %s\n", short(joinpath(RUN_DIR, "run.log")))
    close(LOG); exit(0)
end

# -----------------------------------------------------------------------------
# Estimate
# -----------------------------------------------------------------------------
banner("TikTak search")
lo, hi = search_bounds()
t_start = time()
elapsed() = (time() - t_start) / 60

const USE_PMAP = !(SERIAL || nprocs() == 1)
TRACKER.trun = time()
stage!(:sobol, N_SOBOL_EVAL)
watcher = USE_PMAP ? watch_progress(N_SOBOL_EVAL) : nothing

result = tiktak(objective_tracked, lo, hi;
                N = N_SOBOL, Nstar = N_RESTART,
                extra_seeds = [x0],             # the incumbent competes like any Sobol point
                map_fn = USE_PMAP ? pmap : map,
                # tick! does the per-evaluation reporting, so on_sobol would only
                # double-print. It is used for one thing: i == n is the moment the
                # Sobol stage ends and the local stage begins, which is where the
                # progress format has to change over. Draining the watcher first
                # keeps a straggler Sobol line from being labelled as a restart.
                on_sobol = function (i, n, fx, best)
                    i == n || return
                    # Let the watcher finish draining so a straggler Sobol line is not
                    # printed in the restart format. BOUNDED, never `wait(watcher)`:
                    # the put! above is wrapped in a try, so if a value were ever
                    # dropped the watcher would block on take! forever and an
                    # unbounded wait here would hang the whole estimation on the
                    # progress plumbing. Progress must never be able to do that.
                    if watcher !== nothing
                        t_drain = time()
                        while !istaskdone(watcher) && time() - t_drain < 5.0
                            sleep(0.05)
                        end
                    end
                    sayf("  sobol    complete: %d evaluations, best Q %.4g, %.1f min\n",
                         n, best, elapsed())
                    stage!(:local)
                end,
                on_local = function (j, ns, th, fl, best)
                    eta = elapsed() / max(j, 1) * (ns - j)
                    sayf("  restart %3d/%-3d DONE   this %11.4g   best Q %11.4g   %5.1f min, ~%.0f min left\n",
                         j, ns, fl, best, elapsed(), eta)
                    TRACKER.restart = min(j + 1, ns)
                    TRACKER.done = 0                 # eval counter restarts with the search
                end)

banner(@sprintf("Finished in %.1f min -- %d evaluations", elapsed(), result.n_eval))
sayf("Q: sobol-best %.6g  ->  pre-polish %.6g  ->  final %.6g\n",
     result.f_sobol_best, result.f_prepolish, result.f)
sayf("incumbent Q was %.6g  (improvement %.1f%%)\n", q0, 100 * (q0 - result.f) / q0)

# ---- how much of the box the model could not live in -----------------------
# A penalised draw is a real answer, but the RATE is diagnostic: a few percent is
# a box with soft edges, half is a box that is mostly outside the model.
function gather_penalties()
    total = Dict{Symbol,Int}()
    for w in procs()
        d = w == myid() ? SMM_PENALTY_LOG : remotecall_fetch(() -> SMM_PENALTY_LOG, w)
        for (k, v) in d
            total[k] = get(total, k, 0) + v
        end
    end
    return total
end
const PENALTIES = gather_penalties()
const N_PENALIZED = sum(values(PENALTIES); init = 0)
if N_PENALIZED > 0
    sayf("\npenalised evaluations: %d of %d (%.1f%%)\n",
         N_PENALIZED, result.n_eval, 100 * N_PENALIZED / max(result.n_eval, 1))
    for (k, v) in sort(collect(PENALTIES); by = last, rev = true)
        sayf("  %-24s %6d\n", k, v)
    end
    say("(a penalised draw is a parameter vector the model cannot be solved at --")
    say(" scored 1e6 rather than crashed on. A high rate means the SEARCH BOX is")
    say(" too wide, not that the model is wrong.)")
else
    say("\nno penalised evaluations -- every draw in the box solved")
end

banner("Estimated calibration")
say_report(result.x)

# -----------------------------------------------------------------------------
# Persist
# -----------------------------------------------------------------------------
est = unpack(result.x)
open(joinpath(RUN_DIR, "estimates.toml"), "w") do io
    println(io, "# SMM, three parent moments. GENERATED by code/smm/run_smm.jl.")
    println(io, "generated  = \"", Dates.format(now(), "yyyy-mm-dd HH:MM"), "\"")
    println(io, "git_commit = \"", git_sha(), "\"")
    println(io, "targets    = \"Input/smm_targets_baseline.toml\"")
    println(io, "quick      = ", QUICK)
    println(io, "n_sobol    = ", N_SOBOL)
    println(io, "n_restarts = ", N_RESTART)
    println(io, "n_eval     = ", result.n_eval)
    println(io, "grid_search= ", GRID_SEARCH, "   # parent Na = Nhc used by the optimizer")
    println(io, "grid_report= ", GRID_FULL, "   # parent Na = Nhc the reported fit was re-solved at")
    println(io, "workers    = ", max(0, nprocs() - 1))
    println(io, "n_penalized= ", N_PENALIZED, "   # draws the model could not be solved at")
    println(io, "minutes    = ", round(elapsed(), digits = 1))
    println(io, "Q_final    = ", result.f)
    println(io, "Q_incumbent= ", q0)
    println(io, "\n[parameters]")
    for q in SMM_PARAMS
        @printf(io, "%-10s = %.8f   # was %.8f\n", q.name, getfield(est, q.name),
                getfield(PARENT_DEFAULTS, q.name))
    end
end
say("")
sayf("wrote %s\n", short(joinpath(RUN_DIR, "estimates.toml")))
sayf("wrote %s\n", short(joinpath(RUN_DIR, "run.log")))
close(LOG)
