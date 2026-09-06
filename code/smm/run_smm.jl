#!/usr/bin/env julia
# =============================================================================
# run_smm.jl -- estimate NINE parent parameters against TEN data moments.
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
# Arnoud-Guvenen-Kleineberg use N* = 0.1N. N = 1000 / N* = 100 is that standard.
# The Sobol stage is parallel (~8 min at 20 workers); the 100 restarts are SEQUENTIAL
# and dominate -- budget roughly 25 h. Cut --restarts, not --sobol, if that is too long.
const N_SOBOL     = argval("--sobol",    QUICK ? 12 : 1000)
const N_RESTART   = argval("--restarts", QUICK ?  2 : 100)
const EVERY_SEC   = float(argval("--every", 2))   # progress line throttle, seconds
# Evaluations for the full-grid refinement that follows a coarse search. Small on purpose:
# it starts from the coarse argmin, which is already close, so this is a polish and not a
# second search. At ~12 s per full-grid evaluation, 200 is about 40 minutes.
const REFINE_MAXEVAL = argval("--refine", QUICK ? 10 : 200)

# Evaluation caps for the local searches and the final polish. `--quick` used to leave
# these at their full 2000/4000, so a "2 minute smoke test" could sit in a single restart
# for a quarter of an hour -- the flag reduced the NUMBER of restarts and not their
# length. Exposed so a budget can be set from measured restart traces rather than assumed.
const LOCAL_MAXEVAL  = argval("--local-evals",  QUICK ?  60 : 2000)
const POLISH_MAXEVAL = argval("--polish-evals", QUICK ? 120 : 4000)

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
# --resume DIR continues a killed run from its checkpoint. The run directory is REUSED,
# so the checkpoint keeps advancing in place and the log is appended rather than
# truncated -- losing the first half of a two-day run's transcript to a resume would
# defeat the point of having one.
const RESUME_DIR = argstr("--resume", "")
const RESUMING   = !isempty(RESUME_DIR)
RESUMING && !isdir(RESUME_DIR) && error("--resume: no such directory: $RESUME_DIR")
const RUN_DIR = RESUMING ? RESUME_DIR :
                argstr("--outdir", joinpath(REPO, "output", "smm_runs", STAMP))
mkpath(RUN_DIR)
const LOG = open(joinpath(RUN_DIR, "run.log"), RESUMING ? "a" : "w")

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
sayf("budget     %d Sobol points, %d restarts (<= %d evals each, polish %d)\n",
     N_SOBOL, N_RESTART, LOCAL_MAXEVAL, POLISH_MAXEVAL)
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
# A5. The targets file's content hash. A run record that names the file is not enough --
# the file is regenerated by tools/make_smm_targets.py and its contents decide the answer.
using SHA
const TARGETS_FILE = joinpath(REPO, "Input", "smm_targets_baseline.toml")
const TARGETS_SHA  = bytes2hex(SHA.sha256(read(TARGETS_FILE)))[1:16]
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
                                  rho = 1.5, psi_terminal = 0.0, kappa_terminal = 5.0,
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
RESUMING ? say("(resuming: fit table skipped, see the original run's log)") : say_report(x0)

if REPORT_ONLY
    banner("--report-only: stopping before the search")
    sayf("wrote %s\n", short(joinpath(RUN_DIR, "run.log")))
    close(LOG); exit(0)
end

# -----------------------------------------------------------------------------
# Estimate
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------
# Written after EVERY restart, not once at the end. The local stage is ~99% of the wall
# clock and runs for the better part of a day; before this, a tmux disconnect, a
# wall-clock limit or a pre-emption lost the entire run, because parameters were only
# persisted after the final report -- which itself can fail.
#
# The file is rewritten in full each time (it is a few hundred bytes) via a temporary
# file and an atomic rename, so a crash midway through a write cannot leave a truncated
# checkpoint behind. It is deliberately readable by the same TOML parser as
# estimates.toml, so a killed run's best point can be fed straight back in.
const CKPT = joinpath(RUN_DIR, "checkpoint.toml")

const SEEDS_F = joinpath(RUN_DIR, "seeds.toml")

"""
    save_seeds!(seeds, f_sobol_best)

Persist the pre-testing survivors ONCE, right after the global stage.

Sobol points are deterministic and cost nothing to regenerate; their EVALUATIONS are the
expensive part (1000 solves). Keeping the selected seeds means a resumed run re-enters
the local stage with exactly the mixture the original would have used, so continuation is
exact rather than a warm restart.
"""
function save_seeds!(seeds::Vector{Vector{Float64}}, f_sobol_best::Float64)
    tmp = SEEDS_F * ".tmp"
    open(tmp, "w") do io
        println(io, "# Pre-testing survivors. Written once; read by --resume.")
        println(io, "f_sobol_best = ", f_sobol_best)
        println(io, "nstar        = ", length(seeds))
        println(io, "seeds = [")
        for sd in seeds
            println(io, "  [", join((@sprintf("%.17g", v) for v in sd), ", "), "],")
        end
        println(io, "]")
    end
    mv(tmp, SEEDS_F; force = true)
    return nothing
end

"""
    checkpoint!(j, best, best_x; stage, grid)

`stage` and `grid` are NOT decoration. After the full-grid refinement this is called with
an objective computed at GRID_FULL, while the local stage's calls carry GRID_SEARCH
values. Recording only GRID_SEARCH -- as the first version did -- labelled a grid-30
objective as grid-20, which is exactly the confusion the separate Q_final / Q_search
fields in estimates.toml exist to prevent.
"""
function checkpoint!(j::Int, best::Float64, best_x::Vector{Float64};
                     stage::String = "local", grid::Int = GRID_SEARCH)
    est = unpack(best_x)
    tmp = CKPT * ".tmp"
    open(tmp, "w") do io
        println(io, "# Written after each restart by code/smm/run_smm.jl. Safe to read")
        println(io, "# while the run is going; rewritten atomically.")
        println(io, "# Resume with:  julia --project=../.. run_smm.jl --resume <this dir>")
        println(io, "stage         = \"", stage, "\"")
        println(io, "restarts_done = ", j)
        # A3. IDENTITY OF THE PROBLEM, so a resume can refuse rather than guess.
        # Parameter NAMES and BOXES in particular: the R_0 box changed on 2026-09-06, and
        # resuming an older run into the new box would mix two different problems in one
        # sequence of restarts -- the saved incumbent and seeds would be points in a box
        # that no longer exists.
        println(io, "param_names   = [", join(("\"$(q.name)\"" for q in SMM_PARAMS), ", "), "]")
        println(io, "param_lo      = [", join((q.lo for q in SMM_PARAMS), ", "), "]")
        println(io, "param_hi      = [", join((q.hi for q in SMM_PARAMS), ", "), "]")
        println(io, "param_link    = [", join(("\"$(q.link)\"" for q in SMM_PARAMS), ", "), "]")
        println(io, "targets_sha   = \"", TARGETS_SHA, "\"")
        println(io, "seed          = ", SIM_N > 0 ? 1234 : 1234)
        println(io, "sim_n         = ", SIM_N)
        println(io, "restarts_total= ", N_RESTART)
        println(io, "Q_best        = ", best)
        println(io, "objective_grid= ", grid, "   # the grid Q_best was computed at")
        println(io, "Q_incumbent   = ", q0, "   # at grid_search")
        println(io, "grid_search   = ", GRID_SEARCH)
        println(io, "grid_report   = ", GRID_FULL)
        println(io, "minutes       = ", round(elapsed(), digits = 1))
        println(io, "updated       = \"", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\"")
        println(io, "\n[search_vector]")
        println(io, "z = [", join((@sprintf("%.17g", v) for v in best_x), ", "), "]")
        println(io, "\n[parameters]")
        for q in SMM_PARAMS
            @printf(io, "%-10s = %.8f\n", q.name, getfield(est, q.name))
        end
    end
    mv(tmp, CKPT; force = true)
    return nothing
end

"""
    load_resume(dir) -> NamedTuple

Rebuild tiktak's `resume` argument from a run directory's checkpoint and seeds.
Refuses rather than guesses when the saved run does not match this one.
"""
function load_resume(dir::AbstractString)
    ck_p, sd_p = joinpath(dir, "checkpoint.toml"), joinpath(dir, "seeds.toml")
    isfile(ck_p) || error("--resume: $ck_p not found")
    isfile(sd_p) || error("""
        --resume: $sd_p not found. The interrupted run stopped before its pre-testing
        stage finished, so there are no seeds to continue from. Start it fresh.""")
    ck, sd = TOML.parsefile(ck_p), TOML.parsefile(sd_p)
    seeds = [Float64.(v) for v in sd["seeds"]]
    n = length(SMM_PARAMS)
    refuse(msg) = error("--resume refuses to continue $(short(dir)):\n    " * msg *
                        "\n  Start a fresh run instead. Resuming across a changed problem " *
                        "would mix two different\n  estimations in one sequence of restarts.")

    all(length(x) == n for x in seeds) || refuse(
        "seeds have dimension $(length(first(seeds))) but SMM_PARAMS has $n -- " *
        "the parameter set changed.")
    Int(ck["restarts_total"]) == N_RESTART || refuse(
        "that run had --restarts $(ck["restarts_total"]), this one has $N_RESTART.")
    Int(ck["grid_search"]) == GRID_SEARCH || refuse(
        "that run searched at grid $(ck["grid_search"]), this one at $GRID_SEARCH -- " *
        "the objectives differ.")

    # A3. THE SAVED STAGE AND OBJECTIVE GRID DECIDE WHETHER A RESUME IS EVEN MEANINGFUL.
    #
    # A "refined" or "final" checkpoint holds a FULL-GRID objective. Feeding it back as the
    # local stage's incumbent would compare a grid-30 value against grid-20 values for the
    # rest of the run, and every subsequent restart would be measured against a number it
    # cannot beat. That is a silent corruption of the search, not an inconvenience.
    stage = get(ck, "stage", "local")
    ogrid = Int(get(ck, "objective_grid", GRID_SEARCH))
    stage == "local" || refuse(
        "that checkpoint is at stage \"$stage\", not \"local\" -- it holds a FINISHED " *
        "run's winner, not\n    an interrupted search. There is nothing to continue.")
    ogrid == GRID_SEARCH || refuse(
        "that checkpoint's objective was computed at grid $ogrid, but this run searches " *
        "at $GRID_SEARCH.")

    # Parameter names, boxes and links must be identical: the saved seeds and incumbent are
    # points in a specific box, in search coordinates.
    if haskey(ck, "param_names")
        names_now = [String(q.name) for q in SMM_PARAMS]
        String.(ck["param_names"]) == names_now || refuse(
            "parameter set changed:\n      saved $(join(ck["param_names"], ", "))" *
            "\n      now   $(join(names_now, ", "))")
        for (fld, now) in (("param_lo",   [q.lo for q in SMM_PARAMS]),
                           ("param_hi",   [q.hi for q in SMM_PARAMS]))
            saved = Float64.(ck[fld])
            saved == now || refuse(
                "$fld changed:\n      saved $saved\n      now   $now" *
                "\n    (the R_0 box changed on 2026-09-06 -- old runs cannot be resumed.)")
        end
        String.(get(ck, "param_link", ["?"])) == [String(q.link) for q in SMM_PARAMS] ||
            refuse("parameter links changed; search coordinates are not comparable.")
    else
        refuse("that checkpoint predates the bounds/targets identity fields (A3) and " *
               "cannot be\n    verified against this run's box.")
    end
    haskey(ck, "targets_sha") && ck["targets_sha"] != TARGETS_SHA && refuse(
        "the targets file changed since that run (sha $(ck["targets_sha"]) -> $TARGETS_SHA).")

    j_done = Int(ck["restarts_done"])
    # A3. RESTART HISTORY. The completed restarts' trace is reloaded so the run record
    # covers the whole estimation and not only the part after the interruption.
    hist_p = joinpath(dir, "restarts.csv")
    history = NamedTuple[]
    if isfile(hist_p)
        for (i, ln) in enumerate(eachline(hist_p))
            i == 1 && continue
            f = split(strip(ln), ',')
            length(f) >= 6 || continue
            push!(history, (j = parse(Int, f[1]), theta = parse(Float64, f[2]),
                            f_start = parse(Float64, f[3]), f_local = parse(Float64, f[4]),
                            improved = parse(Bool, f[5]), ret = Symbol(f[6])))
        end
    end
    return (seeds = seeds, f_sobol_best = Float64(sd["f_sobol_best"]),
            Z = Float64.(ck["search_vector"]["z"]), fZ = Float64(ck["Q_best"]),
            j_start = j_done + 1, history = history)
end

const RESTARTS_F = joinpath(RUN_DIR, "restarts.csv")
const RESUME_STATE = RESUMING ? load_resume(RESUME_DIR) : nothing
if !RESUMING || !isfile(RESTARTS_F)
    open(RESTARTS_F, "w") do io
        println(io, "restart,theta,f_start,f_local,improved,ret")
    end
end
if RESUMING
    banner("RESUMING")
    sayf("continuing %s from restart %d of %d\n", short(RESUME_DIR),
         RESUME_STATE.j_start, N_RESTART)
    sayf("incumbent from the checkpoint: Q = %.6g (at grid %d)\n",
         RESUME_STATE.fZ, GRID_SEARCH)
    say("The pre-testing stage is skipped -- its surviving seeds are reloaded, so the")
    say("restarts see exactly the mixture they would have seen in the original run.")
end

banner("TikTak search")
lo, hi = search_bounds()
t_start = time()
elapsed() = (time() - t_start) / 60

const USE_PMAP = !(SERIAL || nprocs() == 1)
TRACKER.trun = time()
# On a resume there is no Sobol stage, so the tracker must START in :local -- the stage
# transition normally happens in the on_sobol callback, which never fires. Without this
# the restarts were reported under the Sobol format ("sobol 31/13  238%").
if RESUMING
    stage!(:local)
    TRACKER.restart = RESUME_STATE.j_start
    # SEED THE TRACKER'S BEST WITH THE RESUMED INCUMBENT. Without this it starts at Inf,
    # so the first evaluation after a resume sets it and the progress line reports a "best
    # Q" WORSE than the point the run is actually carrying -- e.g. "best Q 0.8317" while
    # the checkpoint held 0.5351. The search itself was unaffected (tiktak tracks its own
    # incumbent) but the log said something untrue, which on a two-day run is how a
    # perfectly good resume gets killed and restarted by hand.
    TRACKER.best = RESUME_STATE.fZ
else
    stage!(:sobol, N_SOBOL_EVAL)
end
watcher = (USE_PMAP && !RESUMING) ? watch_progress(N_SOBOL_EVAL) : nothing

result = tiktak(objective_tracked, lo, hi;
                N = N_SOBOL, Nstar = N_RESTART,
                extra_seeds = [x0],             # the incumbent competes like any Sobol point
                map_fn = USE_PMAP ? pmap : map,
                local_maxeval = LOCAL_MAXEVAL, polish_maxeval = POLISH_MAXEVAL,
                resume = RESUME_STATE,
                on_seeds = save_seeds!,
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
                on_local = function (j, ns, th, fl, best, best_x, row)
                    checkpoint!(j, best, best_x; stage = "local", grid = GRID_SEARCH)
                    # A5/A3. One row per restart, written as it finishes: the run record
                    # covers every restart, and a resumed run reloads these so the history
                    # spans the interruption instead of restarting at the resume point.
                    open(RESTARTS_F, "a") do io
                        println(io, row.j, ",", row.theta, ",", row.f_start, ",",
                                row.f_local, ",", row.improved, ",", row.ret)
                    end
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

# -----------------------------------------------------------------------------
# How the search TERMINATED -- not how good its number is
# -----------------------------------------------------------------------------
# A finite objective certifies nothing on its own. A restart that stopped because it hit
# `maxeval` did not satisfy any convergence test, and a run made mostly of those has a
# budget problem however good the objective looks; a restart that threw is a bug in the
# objective, not a bad parameter draw (`smm_objective` already converts genuine model
# failures into a finite penalty and RE-THROWS everything else). All three used to be
# invisible: the return codes lived in `result.trace` and were never read.
const RET_TALLY = ret_tally(result)
const N_CONVERGED = sum(v for (k, v) in RET_TALLY if ret_class(k) === :converged; init = 0)
const N_LIMIT     = sum(v for (k, v) in RET_TALLY if ret_class(k) === :limit;     init = 0)
const N_OTHER     = sum(v for (k, v) in RET_TALLY if ret_class(k) === :other;     init = 0)

say("\nhow the local searches ended")
for (k, v) in sort(collect(RET_TALLY); by = last, rev = true)
    sayf("  %-22s %5d   (%s)\n", k, v, ret_class(k))
end
sayf("  %-22s %5d converged / %d hit a budget / %d other\n", "", N_CONVERGED, N_LIMIT, N_OTHER)
sayf("polish: ret %s (%s), %d evaluations, %s\n", result.polish_ret,
     ret_class(result.polish_ret), result.n_eval_polish,
     result.polish_improved ? "improved the incumbent" : "did not improve the incumbent")
if result.n_exception > 0
    sayf("\n!! %d local search(es) THREW. That is a bug in the objective, not a bad draw --\n",
         result.n_exception)
    say("   smm_objective scores genuine model failures and re-throws everything else.")
    say("   The affected restarts were discarded; treat this run as suspect.")
end
if N_LIMIT > N_CONVERGED
    sayf("\n!! %d of %d restarts stopped on a BUDGET, not a convergence test.\n",
         N_LIMIT, length(result.trace))
    say("   Raise --local-evals, or read the trace before quoting this as a minimum.")
end

# -----------------------------------------------------------------------------
# Full-grid refinement
# -----------------------------------------------------------------------------
# The search minimises Q on GRID_SEARCH. Re-EVALUATING that winner at the full grid is
# not the same as OPTIMISING at the full grid: the coarse and fine objectives have
# slightly different minimisers, so the coarse argmin is a good starting point and not an
# answer. Q(20) = 2.5273 against Q(30) = 2.5485 at the incumbent, so the surfaces differ
# by ~1% -- small, but the whole point of the estimate is where the minimum SITS.
#
# So: a short BOBYQA polish on the full-grid objective, started from the coarse winner.
# Skipped entirely when the search already ran at the full grid, which is the default.
const Z_SEARCH = copy(result.x)
const Q_SEARCH = result.f

# IN A FUNCTION, DELIBERATELY. `try` is a soft scope at top level, so assigning Z_FINAL /
# Q_FINAL inside one binds a LOCAL and silently leaves the global at its old value -- the
# same rule that bites top-level `for` loops (see CLAUDE.md). The first version of this
# block did exactly that and reported the UNREFINED point while printing the refined one.
# Returning the pair makes the data flow explicit and the trap unreachable.
# WHAT THE REFINEMENT DID, kept separate from WHETHER IT RAN.
#
# `refined = GRID_SEARCH != GRID_FULL` -- the old field -- says only that a stage was
# SELECTED. It read `true` after an exception and after a refinement that found nothing,
# which are three different outcomes with the same label. These four are distinguishable:
#
#   :skipped         the search already ran at the report grid, so there was nothing to do
#   :improved        BOBYQA converged (or stopped) at a strictly better full-grid point
#   :no_improvement  it ran and returned nothing better than the coarse winner
#   :failed          it threw; the coarse winner was kept
const REFINE_SKIPPED = (status = :skipped, ret = :NOT_RUN, evals = 0)

function refine_at_full_grid(z_search::Vector{Float64}, lo, hi)
    banner(@sprintf("Full-grid refinement (grid %d -> %d)", GRID_SEARCH, GRID_FULL))
    q_coarse_winner = objective_full(z_search)
    sayf("Q at the coarse winner, re-evaluated at grid %d: %.6g\n", GRID_FULL, q_coarse_winner)
    t_ref = time()
    ropt = Opt(:LN_BOBYQA, length(lo))
    lower_bounds!(ropt, lo); upper_bounds!(ropt, hi)
    ftol_rel!(ropt, 1e-6); ftol_abs!(ropt, 1e-10); xtol_rel!(ropt, 1e-6)
    maxeval!(ropt, REFINE_MAXEVAL)
    n_ref = Ref(0)
    min_objective!(ropt, (z, g) -> (n_ref[] += 1; objective_full(z)))
    try
        (qr, zr, retr) = optimize(ropt, z_search)
        sayf("refined: %d evaluations, %.1f min, ret %s (%s)\n",
             n_ref[], (time()-t_ref)/60, retr, ret_class(retr))
        if isfinite(qr) && qr < q_coarse_winner
            sayf("Q(grid %d): %.6g at the coarse winner  ->  %.6g refined  (%.2f%% better)\n",
                 GRID_FULL, q_coarse_winner, qr,
                 100*(q_coarse_winner - qr)/max(abs(q_coarse_winner), eps()))
            return (copy(zr), qr, (status = :improved, ret = retr, evals = n_ref[]))
        end
        sayf("Q(grid %d): %.6g at the coarse winner; refinement did not improve on it\n",
             GRID_FULL, q_coarse_winner)
        return (copy(z_search), q_coarse_winner,
                (status = :no_improvement, ret = retr, evals = n_ref[]))
    catch e
        @warn "full-grid refinement failed; keeping the coarse winner" exception = e
        sayf("refinement FAILED after %d evaluations: %s\n", n_ref[], sprint(showerror, e))
        return (copy(z_search), q_coarse_winner,
                (status = :failed, ret = :EXCEPTION, evals = n_ref[]))
    end
end

@everywhere objective_full(z) = smm_objective(z, TARGETS, V_CHILD;
                                              Na = G_FULL_.Na, Nk = G_FULL_.Nk,
                                              Nhc = G_FULL_.Nhc, simN = G_FULL_.simN)
const (Z_FINAL, Q_FINAL, REFINE) = if GRID_SEARCH != GRID_FULL
    refine_at_full_grid(Z_SEARCH, lo, hi)
else
    say("\nsearch ran at the full grid -- no refinement stage needed")
    (copy(Z_SEARCH), Q_SEARCH, REFINE_SKIPPED)
end

# THE FINAL WINNER IS ALWAYS CHECKPOINTED, refinement or not.
#
# This used to be guarded by `GRID_SEARCH != GRID_FULL`, and both grids default to 30 --
# so on a DEFAULT run the last checkpoint written was the pre-polish incumbent after the
# final restart, and the polish's improvement existed only in estimates.toml. If the report
# stage then died, the best point on disk was not the best point found.
checkpoint!(N_RESTART, Q_FINAL, Z_FINAL;
            stage = GRID_SEARCH == GRID_FULL ? "final" : "refined", grid = GRID_FULL)

# ---- how much of the box the model could not live in -----------------------
# A penalised draw is a real answer, but the RATE is diagnostic: a few percent is
# a box with soft edges, half is a box that is mostly outside the model.
function gather_penalties()
    total = Dict{Symbol,Int}()
    for w in procs()
        # `Main.SMM_PENALTY_LOG`, not the bare name: a closure over the bare name
        # SERIALISES this process's copy and asks the worker to install it, which each
        # worker then refuses -- "Cannot transfer global variable" on every gather. Going
        # through Main makes it a global lookup performed on the worker.
        d = w == myid() ? SMM_PENALTY_LOG : remotecall_fetch(() -> copy(Main.SMM_PENALTY_LOG), w)
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
const FINAL_REPORT = say_report(Z_FINAL)

# -----------------------------------------------------------------------------
# Acceptance: is this point fit to be used, separate from how good its Q is
# -----------------------------------------------------------------------------
# Three conditions, and all three have to hold. A lower finite objective is not one of
# them: a point can be the best one found and still sit on economically impossible paths,
# or be the output of searches that all ran out of budget.
const N_INVALID_FINAL = FINAL_REPORT.violations.total

# A2. ACCEPTANCE IS A STATEMENT ABOUT THE RETAINED WINNER, NOT ABOUT THE POPULATION.
#
# The previous test was `N_CONVERGED > 0` -- "some restart converged". That says nothing
# about the point actually returned: the winner can come from a restart that exhausted
# maxeval while a different restart converged to something worse, and the run would still
# report `accepted = true`. The winner's own provenance and return code decide it.
#
# The final point can come from four places, and all four have to be right:
#   1. the search's winner        result.winner_stage / winner_ret
#   2. the polish                 (winner_stage === :polish, so 1 covers it)
#   3. the full-grid refinement   REFINE.status / REFINE.ret -- a FAILED refinement is
#                                 disqualifying even though the coarse winner was kept,
#                                 because the reported Q then came from a stage that threw
#   4. no local search at all     winner_stage === :sobol, i.e. every restart failed to
#                                 improve on a raw Sobol point. Never acceptable.
const WINNER_CONVERGED = ret_class(result.winner_ret) === :converged
const REFINE_OK        = REFINE.status in (:skipped, :improved, :no_improvement) &&
                         (REFINE.status === :skipped || ret_class(REFINE.ret) !== :other)
const ACCEPTED = WINNER_CONVERGED && REFINE_OK &&
                 (result.n_exception == 0) && (N_INVALID_FINAL == 0) &&
                 isfinite(Q_FINAL) && (Q_FINAL < SMM_PENALTY)

say("\nacceptance -- about the RETAINED WINNER, not about the other restarts")
sayf("  winner came from        %s%s\n", result.winner_stage,
     result.winner_stage === :local ? " restart $(result.winner_j)" : "")
sayf("  its own return code     %s  (%s)  %s\n", result.winner_ret,
     ret_class(result.winner_ret), WINNER_CONVERGED ? "yes" : "NO")
sayf("  refinement              %s (ret %s, %d evals)  %s\n",
     REFINE.status, REFINE.ret, REFINE.evals, REFINE_OK ? "yes" : "NO")
sayf("  no objective exceptions %s  (%d)\n", result.n_exception == 0 ? "yes" : "NO ",
     result.n_exception)
sayf("  final sim in domain     %s  (%d invalid cells)\n",
     N_INVALID_FINAL == 0 ? "yes" : "NO ", N_INVALID_FINAL)
sayf("  objective finite        %s  (Q = %.6g)\n",
     (isfinite(Q_FINAL) && Q_FINAL < SMM_PENALTY) ? "yes" : "NO ", Q_FINAL)
sayf("  (for context: %d of %d restarts converged, %d hit a budget, %d other)\n",
     N_CONVERGED, length(result.trace), N_LIMIT, N_OTHER)
if ACCEPTED
    say("  ACCEPTED -- this point may be quoted, with the caveats in docs/SMM.md")
else
    say("  NOT ACCEPTED -- do not quote this point. Fix the failing condition above first;")
    say("  a finite Q is not a certification, and neither is another restart's convergence.")
end

# -----------------------------------------------------------------------------
# Persist
# -----------------------------------------------------------------------------
est = unpack(Z_FINAL)
open(joinpath(RUN_DIR, "estimates.toml"), "w") do io
    println(io, "# SMM: ", length(SMM_MOMENTS), " parent moments, ", length(SMM_PARAMS),
                 " parameters. GENERATED by code/smm/run_smm.jl.")
    println(io, "generated  = \"", Dates.format(now(), "yyyy-mm-dd HH:MM"), "\"")
    println(io, "git_commit = \"", git_sha(), "\"")
    println(io, "targets    = \"Input/smm_targets_baseline.toml\"")
    println(io, "quick      = ", QUICK)
    println(io, "n_sobol    = ", N_SOBOL)
    println(io, "n_restarts = ", N_RESTART)
    println(io, "n_eval     = ", result.n_eval, "   # TikTak only: sobol + restarts + polish")
    println(io, "n_eval_total= ", result.n_eval + REFINE.evals,
            "   # including the full-grid refinement")
    println(io, "n_eval_polish= ", result.n_eval_polish)
    println(io, "n_eval_refine= ", REFINE.evals)
    println(io, "grid_search= ", GRID_SEARCH, "   # parent Na = Nhc used by the optimizer")
    println(io, "grid_report= ", GRID_FULL, "   # parent Na = Nhc the reported fit was re-solved at")
    println(io, "workers    = ", max(0, nprocs() - 1))
    println(io, "n_penalized= ", N_PENALIZED, "   # draws the model could not be solved at")
    println(io, "minutes    = ", round(elapsed(), digits = 1))
    # BOTH objectives, on the grids they were computed at. Storing only the search-grid
    # value under the name "Q_final" was how a coarse-grid number ended up being quoted
    # next to a full-grid fit table.
    println(io, "Q_final    = ", Q_FINAL, "   # at grid_report, after refinement")
    println(io, "Q_search   = ", Q_SEARCH, "   # at grid_search, what the search minimised")
    println(io, "Q_incumbent= ", q0, "   # at grid_search")
    # ACCEPTANCE. How the point was reached, beside what it is worth.
    println(io, "refine_status = \"", REFINE.status, "\"   # skipped|improved|no_improvement|failed")
    println(io, "refine_ret    = \"", REFINE.ret, "\"")
    println(io, "polish_ret    = \"", result.polish_ret, "\"")
    # A2. WHERE THE REPORTED POINT CAME FROM, beside what it is worth.
    println(io, "winner_stage  = \"", result.winner_stage, "\"   # sobol|local|polish")
    println(io, "winner_restart= ", result.winner_j, "   # 0 unless winner_stage = local")
    println(io, "winner_ret    = \"", result.winner_ret, "\"   # the return code of THAT search")
    println(io, "winner_converged = ", WINNER_CONVERGED)
    println(io, "refine_ok     = ", REFINE_OK)
    println(io, "polish_improved = ", result.polish_improved)
    println(io, "n_converged   = ", N_CONVERGED, "   # local searches that met a stopping test")
    println(io, "n_hit_budget  = ", N_LIMIT, "   # stopped on maxeval/maxtime, NOT converged")
    println(io, "n_ret_other   = ", N_OTHER)
    println(io, "n_exception   = ", result.n_exception, "   # >0 means a BUG in the objective")
    println(io, "n_invalid_final = ", N_INVALID_FINAL, "   # off-domain cells at the final point")
    println(io, "accepted      = ", ACCEPTED,
            "   # converged restarts, no exceptions, and a feasible final simulation")
    print(io, "ret_tally  = {")
    print(io, join(("$(k) = $(v)" for (k, v) in sort(collect(RET_TALLY); by = first)), ", "))
    println(io, "}")
    println(io, "\n[parameters]")
    for q in SMM_PARAMS
        @printf(io, "%-10s = %.8f   # was %.8f\n", q.name, getfield(est, q.name),
                getfield(PARENT_DEFAULTS, q.name))
    end
end

# -----------------------------------------------------------------------------
# A5. The reproducible run record
# -----------------------------------------------------------------------------
# estimates.toml answers "what came out". This answers "what exactly produced it", so the
# run can be repeated or audited without the conversation that surrounded it. Everything
# that can change the answer goes here: the code, the targets BY CONTENT not by name, the
# parameter boxes and links, the seed, the grids, and every solver tolerance and budget.
# The per-restart results are in restarts.csv beside it.
open(joinpath(RUN_DIR, "run_record.toml"), "w") do io
    println(io, "# Reproducible run record. GENERATED by code/smm/run_smm.jl (A5).")
    println(io, "# Everything that can change the answer. Per-restart results: restarts.csv")
    println(io, "generated    = \"", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\"")
    println(io, "host         = \"", gethostname(), "\"")
    println(io, "julia        = \"", VERSION, "\"")
    println(io, "\n[code]")
    println(io, "git_commit   = \"", git_sha(), "\"")
    println(io, "dirty        = ", occursin("dirty", git_sha()),
            "   # true means uncommitted changes were in the working tree")
    println(io, "\n[targets]")
    println(io, "file         = \"", short(TARGETS_FILE), "\"")
    println(io, "sha256_16    = \"", TARGETS_SHA, "\"   # CONTENT hash; the file is regenerated")
    println(io, "moments      = [", join(("\"$m\"" for m in SMM_MOMENTS), ", "), "]")
    for k in SMM_MOMENTS
        @printf(io, "%-16s = %.10g\n", "target_" * k, TARGETS[k].mean)
    end
    println(io, "\n[parameters]")
    println(io, "names        = [", join(("\"$(q.name)\"" for q in SMM_PARAMS), ", "), "]")
    println(io, "lo           = [", join((q.lo for q in SMM_PARAMS), ", "), "]")
    println(io, "hi           = [", join((q.hi for q in SMM_PARAMS), ", "), "]")
    println(io, "link         = [", join(("\"$(q.link)\"" for q in SMM_PARAMS), ", "), "]")
    println(io, "start        = [", join((@sprintf("%.17g", v) for v in x0), ", "),
            "]   # search coords; the incumbent, forced into the Sobol pool")
    println(io, "fixed_note   = \"sigma_4_1 and mu_1 are NOT estimated; they hold at PARENT_DEFAULTS\"")
    println(io, "sigma_4_1    = ", PARENT_DEFAULTS.sigma_4_1)
    println(io, "mu_1         = ", PARENT_DEFAULTS.mu_1)
    println(io, "\n[numerical]")
    println(io, "seed         = 1234   # common random numbers, identical across evaluations")
    println(io, "simN         = ", SIM_N)
    println(io, "grid_search  = ", GRID_SEARCH, "   # parent Na = Nhc during the search")
    println(io, "grid_report  = ", GRID_FULL, "   # parent Na = Nhc for the reported fit")
    println(io, "child_grid   = ", CHILD_G_)
    println(io, "n_sobol      = ", N_SOBOL)
    println(io, "n_restarts   = ", N_RESTART)
    println(io, "local_alg    = \"LN_NELDERMEAD\"")
    println(io, "local_ftol_rel = 1e-3")
    println(io, "local_ftol_abs = 1e-10")
    println(io, "local_xtol_rel = 1e-8")
    println(io, "local_maxeval  = ", LOCAL_MAXEVAL)
    println(io, "polish_alg   = \"LN_BOBYQA\"")
    println(io, "polish_tol   = 1e-10")
    println(io, "polish_maxeval = ", POLISH_MAXEVAL)
    println(io, "refine_maxeval = ", REFINE_MAXEVAL)
    println(io, "theta_schedule = \"clamp((j/Nstar)^0.5, 0.1, 0.995), restart 1 pinned at 0\"")
    println(io, "penalty      = ", SMM_PENALTY)
    println(io, "workers      = ", max(0, nprocs() - 1))
    println(io, "resumed      = ", RESUMING)
    println(io, "\n[result]")
    println(io, "Q_final      = ", Q_FINAL)
    println(io, "Q_search     = ", Q_SEARCH)
    println(io, "Q_incumbent  = ", q0)
    println(io, "winner_stage = \"", result.winner_stage, "\"")
    println(io, "winner_ret   = \"", result.winner_ret, "\"")
    println(io, "accepted     = ", ACCEPTED)
    println(io, "n_eval_total = ", result.n_eval + REFINE.evals)
    println(io, "minutes      = ", round(elapsed(), digits = 1))
end

say("")
sayf("wrote %s\n", short(joinpath(RUN_DIR, "run_record.toml")))
sayf("wrote %s\n", short(RESTARTS_F))
sayf("wrote %s\n", short(joinpath(RUN_DIR, "estimates.toml")))
sayf("wrote %s\n", short(joinpath(RUN_DIR, "run.log")))
close(LOG)
