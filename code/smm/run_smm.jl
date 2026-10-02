#!/usr/bin/env julia
# =============================================================================
# run_smm.jl -- estimate the 20 memo-18/19 parameters against the 67 memo-19 moments.
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
# PARALLELISM (rewritten 2026-09-27, tiktak_fix_plan.md step 6)
# -----------------------------------------------------------------------------
# Worker PROCESSES (Distributed.jl), never threads. NLopt.jl is not thread-safe
# in this project -- with threads the objective killed the process with exit 0
# and no error message. Each worker process owns its NLopt state and its own copy
# of the model, so the hazard cannot arise. ONE pool of workers serves both stages.
#
#   Sobol stage    N independent evaluations, dispatched to every worker as each
#                  one frees up (no batch barrier), values cached chunk by chunk.
#
#   Local stage    --local-mode async (the default when there are workers): TikTak's
#                  ASYNCHRONOUS variant. A fresh run fills the pool at once (--bootstrap
#                  immediate_mixed, the default since 2026-10-01: restarts mix with the best
#                  pre-tested point until a result is committed); --bootstrap first_alone runs
#                  restart 1 alone first; a resume keeps its saved policy. Then every idle worker
#                  immediately gets the next restart, mixed with the best point
#                  COMMITTED so far -- restarts still running are not waited for. The
#                  master only schedules, merges and checkpoints. The number of local
#                  workers is min(pool, floor(sqrt(restarts))) unless --local-procs
#                  says otherwise (a conservative default, not a mathematical bound).
#                  --local-mode serial is the published sequential algorithm on the
#                  master: restart j starts only after j-1 is committed; reproducible
#                  bit for bit, and with one local worker async is exactly the same.
#
# The two stages scale differently, and the run prints both projections before the
# search starts.
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

# EVERY FLAG IS CHECKED. An unknown flag used to be silently ignored, so `--workers 1`
# started the full 20 processes and `--polish-eval 0` polished for 4,000 evaluations. A
# flag the runner does not know is now an error before anything is written.
const KNOWN_FLAGS = ("--quick", "--serial", "--report-only", "--seed", "--sobol", "--restarts",
                     "--every", "--refine", "--local-evals", "--polish-evals", "--grid",
                     "--procs", "--resume", "--temp", "--outdir", "--targets",
                     "--init-from", "--skip-polish", "--parent-extra", "--child-extra",
                     "--print-identity", "--stop-after-restarts",
                     "--allow-optimizer-change", "--sobol-valid", "--pretest-chunk", "--reuse-pretest",
                     "--local-mode", "--local-procs", "--max-retries", "--preset",
                     "--allow-fewer-restarts", "--require-valid-target",
                     "--normalize", "--local-init-step", "--local-step-schedule", "--local-step-min", "--polish-init-step",
                     "--bootstrap", "--local-ftol", "--expect-start-q", "--runtime-evals")
let unknown = [a for a in ARGS if startswith(a, "--") && !(a in KNOWN_FLAGS)]
    isempty(unknown) || error("unknown flag(s): " * join(unknown, ", ") *
                              "\n    known: " * join(KNOWN_FLAGS, " "))
end

const QUICK       = "--quick"       in ARGS
# THE OPTIMIZER MODULE, loaded here on the master (no worker, no model: plan J1) so a preset
# can supply defaults before the budgets below are fixed. Workers load it with the model.
include(joinpath(REPO, "code", "src", "tiktak.jl"))
# --preset smoke|integration|pilot|production (2026-09-27, plan step 8): WHAT THE RUN IS FOR,
# recorded with it. It supplies default budgets and gates (code/src/TikTak/presets.jl); an
# explicit flag always wins. Without --preset the run is "custom" and every default is the
# one this runner always had. Five restarts are a development test; a final estimation may
# use 100-1000 restarts with a matching pre-testing pool.
const PRESET = TikTak.resolve_preset(let i = findfirst(==("--preset"), ARGS)
    i === nothing ? "" : (i < length(ARGS) ? ARGS[i + 1] : error("--preset needs a value"))
end)
const CUSTOM = PRESET.name == "custom"
hasflag(f) = f in ARGS
# SMM_TEST_FIXTURES=1 (moments.jl; Ali 2026-10-02): stand-in composition and wage loading, for TESTS only
get(ENV, "SMM_TEST_FIXTURES", "") == "1" && PRESET.name in ("pilot", "production") &&
    error("SMM_TEST_FIXTURES=1 with --preset $(PRESET.name): the stand-in inputs are for tests, never for an estimate")
PRESET.name == "production" && QUICK &&
    error("--preset production with --quick: --quick changes the grids and simN, so its Q is not a production Q")
# --skip-polish is an EXPLICIT bypass of the final BOBYQA polish, for pilot runs whose
# budget is the local stage. It is not `--polish-evals 0`: NLopt reads maxeval = 0 as
# "no limit", which would have run the default 4,000-evaluation polish under a flag that
# said the opposite. tiktak receives `skip_polish = true` and records :SKIPPED.
const SKIP_POLISH = "--skip-polish" in ARGS || (PRESET.skip_polish && !hasflag("--polish-evals"))
# --init-from FILE loads the incumbent from a previous run's checkpoint.toml or estimates.toml.
# EXACT when the file has a [search_vector] whose param_names and param_link equal this
# specification's (every checkpoint.toml; estimates.toml since 2026-09-29): the vector is used
# bit for bit. Otherwise BY NAME from [parameters], which files print to 8 decimals -- a ROUNDED
# start, recorded as such (tiktak_problems.md finding 18: E1-P's gate checked the exact mu08
# point, Q 916.3000261432542, and the run started from the rounded one, Q 916.3000248634202).
# By name, a parameter the file lacks starts at its SMM start; one outside its box is an error.
const INIT_FROM   = argstr("--init-from", "")
# --expect-start-q Q: a GATE in the run itself. After the timing evaluation at the start point
# -- the vector this run actually loaded -- the run stops before the search unless Q equals the
# given value exactly (write it as printed by "start Q (full precision)", e.g. 916.3000261432542).
# A driver no longer needs a separate check through another runner and a sentinel file.
const EXPECT_START_Q = let s = argstr("--expect-start-q", nothing)
    s === nothing ? nothing : (v = tryparse(Float64, s); v === nothing || !isfinite(v) ?
        error("--expect-start-q needs a finite number, got $(repr(s))") : v)
end
# --parent-extra / --child-extra (2026-09-24, the mu = 0.7 run): NON-ESTIMATED constructor settings of the run -- the
# grid (e.g. a_max=300,...; ap_max=300,...,Nap=120) and numerical options (work_transfer_sim=direct) -- passed to every
# evaluation, the report and the refinement, written to run_record.toml and checkpoint.toml, and verified on --resume.
# Unset = the constructor defaults, exactly as every earlier run.
function parse_extra_arg(s)
    isempty(s) && return (;)
    ks = Symbol[]; vs = Any[]
    for kv in split(s, ',')
        k, v = split(kv, '='); push!(ks, Symbol(strip(k))); v = strip(v); isN = startswith(String(ks[end]), "N")
        push!(vs, isN && tryparse(Int, v) !== nothing ? parse(Int, v) : (tryparse(Float64, v) !== nothing ? parse(Float64, v) : Symbol(v)))
    end
    NamedTuple{Tuple(ks)}(Tuple(vs))
end
const PE_STR = argstr("--parent-extra", ""); const CE_STR = argstr("--child-extra", "")
const PE_RUN = parse_extra_arg(PE_STR); const CE_RUN = parse_extra_arg(CE_STR)
const GRID_EXTRA = "parent: " * PE_STR * " | child: " * CE_STR
const SERIAL      = "--serial"      in ARGS
const REPORT_ONLY = "--report-only" in ARGS
# --print-identity prints the objective identity a checkpoint records (objective_identity()
# below) as TOML and exits before the child warm-up and any evaluation. Tests build their
# resume fixtures from it, so they follow the configuration instead of hard-coding it.
const PRINT_IDENTITY = "--print-identity" in ARGS
# --stop-after-restarts M PAUSES the run once M restarts are committed (2026-09-27, plan step
# 8): the schedule denominator stays --restarts, the full seed list is kept, the polish is not
# run, and `--resume <dir>` continues at restart M+1. 0 pauses right after seed selection;
# M = --restarts pauses after the local stage, before the polish. Not --restarts M, which
# changes the schedule itself.
const STOP_AFTER = argval("--stop-after-restarts", typemax(Int))
STOP_AFTER >= 0 || error("--stop-after-restarts must be >= 0")
# No --legacy-import in this repository (Ali, 2026-10-02): a run directory written before the TikTak port
# (checkpoint.toml / seeds.toml, no tiktak_state.toml) is refused on --resume; warm-start a new run from it with
# --init-from. The import code below is kept unchanged from apps/Structural-estimation-v2 and is unreachable here.
const LEGACY_IMPORT = false
# --allow-optimizer-change: resume although a SOLVER setting (--local-evals, --polish-evals,
# --skip-polish ...) or the optimizer SOFTWARE (code/src/TikTak/*.jl, NLopt) changed. Recorded as
# an event and a new settings epoch -- committed restarts and replayed jobs keep the settings they
# ran with -- and the run is labelled changed_optimizer. A change of the search PLAN (box, restart
# count, pre-testing design, supplied start) is refused even with this flag (2026-09-28, finding 14).
const ALLOW_OPT_CHANGE = "--allow-optimizer-change" in ARGS
# Arnoud-Guvenen-Kleineberg use N* = 0.1N. N = 1000 / N* = 100 is that standard.
# Since 2026-09-27 the restarts run asynchronously on the worker pool (--local-mode async,
# min(workers, floor(sqrt(restarts))) of them by default); --local-mode serial runs them one
# after another on the master, as before. The run prints both stages' projections first.
# THE SIMULATION SEED, named rather than left as an implicit keyword default.
#
# It was `1234` in four places -- the objective's default argument, the child warm-up, a
# line in run_record.toml and a self-cancelling `SIM_N > 0 ? 1234 : 1234` in the
# checkpoint -- so nothing could compare it and a resume could not check it. Common random
# numbers are the whole reason the objective is a smooth function of the parameters, so
# two seeds are two different objectives and a checkpoint from one must not resume into
# the other.
const SEED_       = argval("--seed",     1234)
# PRE-TESTING BUDGET (2026-09-27, plan step 5). --sobol N is the number of ATTEMPTED Sobol
# draws. --sobol-valid V instead continues the Sobol sequence until V draws are VALID (a
# penalised draw does not count), and --sobol is then the hard cap on attempts (default 10V).
# Supplied points (the incumbent) are counted apart from either.
const N_VALID_TARGET = argval("--sobol-valid", CUSTOM || hasflag("--sobol") ? 0 : PRESET.n_valid_target)
const N_SOBOL     = argval("--sobol", N_VALID_TARGET > 0 ? (CUSTOM || hasflag("--sobol-valid") ? 10 * N_VALID_TARGET : PRESET.n_sobol) :
                                      (CUSTOM ? (QUICK ? 12 : 1000) : PRESET.n_sobol))
N_VALID_TARGET == 0 || N_SOBOL >= N_VALID_TARGET ||
    error("--sobol $N_SOBOL (the attempt cap) is below --sobol-valid $N_VALID_TARGET")
# --reuse-pretest FILE: take the values of another run's pretest_cache.toml (same objective
# and candidate design, verified) instead of evaluating them again -- e.g. a production-prefix
# test reusing a pool that was paid for once. Reported as reused, never as this run's evaluations.
const REUSE_PRETEST = argstr("--reuse-pretest", "")
isempty(REUSE_PRETEST) || isfile(REUSE_PRETEST) || error("--reuse-pretest: no such file: $REUSE_PRETEST")
const N_RESTART   = argval("--restarts", CUSTOM ? (QUICK ? 2 : 100) : PRESET.nstar)
# A pool with fewer valid draws than restarts: custom and development presets run fewer
# restarts (reported); pilot and production presets stop with a failed pre-testing gate
# unless --allow-fewer-restarts. --require-valid-target makes a missed --sobol-valid a gate too.
const ALLOW_FEWER = hasflag("--allow-fewer-restarts") || PRESET.allow_fewer_restarts
const REQUIRE_VALID = hasflag("--require-valid-target") || PRESET.require_valid_target
const EVERY_SEC   = float(argval("--every", 2))   # progress line throttle, seconds
# Evaluations for the full-grid refinement that follows a coarse search. Small on purpose:
# it starts from the coarse argmin, which is already close, so this is a polish and not a
# second search. At ~12 s per full-grid evaluation, 200 is about 40 minutes.
const REFINE_MAXEVAL = argval("--refine", QUICK ? 10 : 200)

# Evaluation caps for the local searches and the final polish. `--quick` used to leave
# these at their full 2000/4000, so a "2 minute smoke test" could sit in a single restart
# for a quarter of an hour -- the flag reduced the NUMBER of restarts and not their
# length. Exposed so a budget can be set from measured restart traces rather than assumed.
const LOCAL_MAXEVAL  = argval("--local-evals",  CUSTOM ? (QUICK ?  60 : 2000) : PRESET.local_maxeval)
const POLISH_MAXEVAL = argval("--polish-evals", CUSTOM ? (QUICK ? 120 : 4000) : PRESET.polish_maxeval)
# SEARCH GEOMETRY (2026-09-29; the module's step-9 options, exposed here for the settings pilot E3). Every default is
# the baseline -- NLopt's own initial simplex, no normalization -- so a run without these flags has the optimizer
# identity it always had and its checkpoints resume as before. They are optimizer settings: part of the checkpoint's
# optimizer identity (a resume with other values is refused unless --allow-optimizer-change, which then records a
# settings epoch) and written to run_record.toml as run.
#   --normalize                 solve each local search in box-normalized coordinates u = (x - lo)/(hi - lo)
#   --local-init-step F         Nelder-Mead's initial simplex step as a fraction F of each box width (0 = NLopt's default)
#   --local-step-schedule S     fixed (default) | theta_shrink: restart j's step = F * max(step_min, 1 - theta_j)
#   --local-step-min F          the floor of theta_shrink (default 0.1)
#   --polish-init-step F        BOBYQA's initial trust radius as a fraction of the box width (0 = NLopt's default)
function argfloat(flag, default)
    v = argstr(flag, nothing)
    v === nothing && return default
    x = tryparse(Float64, v)
    x === nothing && error("$flag needs a number, got $(repr(v))")
    return x
end
const GEOM_NORMALIZE = "--normalize" in ARGS
const GEOM_LOCAL_STEP = argfloat("--local-init-step", 0.0)
const GEOM_STEP_SCHEDULE = let s = argstr("--local-step-schedule", "fixed")
    s in ("fixed", "theta_shrink") || error("--local-step-schedule must be fixed or theta_shrink, got $s")
    Symbol(s)
end
const GEOM_STEP_MIN = argfloat("--local-step-min", 0.1)
const GEOM_POLISH_STEP = argfloat("--polish-init-step", 0.0)
# refused here, before any model is loaded (the module checks the same again before its first evaluation)
for (f, v) in (("--local-init-step", GEOM_LOCAL_STEP), ("--local-step-min", GEOM_STEP_MIN), ("--polish-init-step", GEOM_POLISH_STEP))
    (isfinite(v) && 0 <= v <= 1) || error("$f = $v must be a fraction of the box width in [0, 1]")
end
GEOM_STEP_SCHEDULE === :theta_shrink && GEOM_LOCAL_STEP == 0 &&
    error("--local-step-schedule theta_shrink needs --local-init-step > 0 (it scales that step)")
# --local-ftol F (2026-09-29, tiktak_problems.md finding 26): Nelder-Mead's ftol_rel. It stops when
# 2|f_worst - f_best| / (|f_worst| + |f_best|) < F across the simplex. Default 1e-3 (at Q ~ 916: a
# spread of ~0.9 in Q); the reference's amoeba uses the same test at 1e-4. A solver setting: part
# of the optimizer identity, changeable on a resume only with --allow-optimizer-change (an epoch).
const LOCAL_FTOL = argfloat("--local-ftol", 1e-3)
(isfinite(LOCAL_FTOL) && 0 < LOCAL_FTOL < 1) ||
    error("--local-ftol $LOCAL_FTOL must be in (0, 1); 0 would switch the test off (every restart runs to its cap)")
# --bootstrap first_alone|immediate_mixed (2026-09-29, fix plan R2; code/src/TikTak/config.jl): how the
# asynchronous local stage starts. first_alone runs restart 1 alone; immediate_mixed fills the pool at once.
# Part of the SEARCH PLAN: a resume never changes it. DEFAULT (Ali, 2026-10-01, after the optimizer experiments:
# immediate_mixed reached the lowest Q in the shortest time): a FRESH run uses immediate_mixed; a RESUMED run
# without the flag keeps the policy its checkpoint was written with (BOOTSTRAP below, once RESUME_DIR is known).
# The TikTak module's own default is unchanged (first_alone), so its source -- part of every checkpoint's optimizer
# identity -- is unchanged too.
const BOOTSTRAP_FLAG = let s = argstr("--bootstrap", "")
    isempty(s) || s in ("first_alone", "immediate_mixed") ||
        error("--bootstrap must be first_alone or immediate_mixed, got $s")
    s
end
# --runtime-evals N1,N2,... (2026-09-29, finding 19): evaluation counts per restart observed in a
# COMPATIBLE earlier run, for the empirical runtime scenario (restart j takes N[mod1(j, n)], cut at
# this run's cap). Name their source in the driver, e.g. mu08: 500,500,500,423,268. Display only.
const RUNTIME_EVALS = let s = argstr("--runtime-evals", "")
    isempty(s) ? Int[] : [let v = tryparse(Int, strip(t)); (v === nothing || v < 1) ?
        error("--runtime-evals needs positive integers separated by commas, got $(repr(s))") : v end
        for t in split(s, ',')]
end
# A ZERO CAP IS REFUSED HERE, before workers or a model load: NLopt reads maxeval = 0 as
# NO LIMIT, so `--refine 0` would have run an unbounded full-grid refinement (2026-09-27).
for (flag, v) in (("--local-evals", LOCAL_MAXEVAL), ("--refine", REFINE_MAXEVAL))
    v >= 1 || error("$flag $v: an evaluation cap must be >= 1 (NLopt reads 0 as no limit)")
end
("--skip-polish" in ARGS) || POLISH_MAXEVAL >= 1 ||
    error("--polish-evals $POLISH_MAXEVAL: NLopt reads 0 as no limit; use --skip-polish to bypass the polish")

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
# THE LOCAL STAGE (2026-09-27, plan step 6). async: restarts run asynchronously on the worker
# pool; serial: the reference path, on the master. --local-procs N fixes the number of local
# workers (0 = automatic, min(pool, floor(sqrt(restarts)))). A worker lost mid-restart is
# replaced by a re-dispatch of the same job at most --max-retries times.
const LOCAL_MODE = let m = argstr("--local-mode", (SERIAL || NPROC == 0) ? "serial" : "async")
    m in ("serial", "async") || error("--local-mode must be serial or async, got $m")
    (m == "async" && (SERIAL || NPROC == 0)) && error("--local-mode async needs worker processes (--procs >= 1, not --serial)")
    Symbol(m)
end
const LOCAL_PROCS = argval("--local-procs", 0)
const MAX_RETRIES = argval("--max-retries", 1)
LOCAL_PROCS >= 0 && MAX_RETRIES >= 0 || error("--local-procs and --max-retries must be >= 0")

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
# The start-up policy (see --bootstrap above): the flag if given; else, on a resume, the policy the checkpoint was written
# with (a checkpoint without the field, or a run directory without tiktak_state.toml, predates the option: first_alone,
# then the only policy); else immediate_mixed. An unreadable checkpoint is not refused here: verify_state_identity
# refuses it below with the runner's own message.
const BOOTSTRAP = !isempty(BOOTSTRAP_FLAG) ? Symbol(BOOTSTRAP_FLAG) :
    !RESUMING ? :immediate_mixed :
    let f = joinpath(RESUME_DIR, "tiktak_state.toml")
        s = isfile(f) ? (try get(first(TikTak.read_state_dict(f))["optimizer"]["fields"], "bootstrap", "first_alone")
                         catch; "first_alone" end) : "first_alone"
        Symbol(String(s))
    end

# --temp [label] labels a THROWAWAY run in output/smm_runs/<timestamp>_temp[_label]/.
# Same contents as a real run -- run.log, run_record.toml with every
# setting, restarts.csv, estimates.toml -- because a smoke test whose settings you cannot
# reconstruct is not worth keeping either.
#
# TIMESTAMPED, ALWAYS. `--outdir ../../temp/six_smoke` was the old way and it produced
# nine hand-named folders that silently overwrote each other on a re-run and carried no
# record of what they were run with. A stamp cannot collide, and it sorts.
#
# New run directories are gitignored, so nothing here is tracked. Delete the whole folder whenever.
const TEMP_LABEL = let i = findfirst(==("--temp"), ARGS)
    i === nothing ? nothing :
    (i < length(ARGS) && !startswith(ARGS[i + 1], "--")) ? ARGS[i + 1] : ""
end
const RUN_DIR = RESUMING ? RESUME_DIR :
    TEMP_LABEL !== nothing ?
        joinpath(REPO, "output", "smm_runs", STAMP * "_temp" * (isempty(TEMP_LABEL) ? "" : "_" * TEMP_LABEL)) :
        argstr("--outdir", joinpath(REPO, "output", "smm_runs", STAMP))
mkpath(RUN_DIR)

# A STABLE NAME FOR "THE RUN THAT IS GOING ON RIGHT NOW".
#
# `output/smm_runs/latest` is re-pointed at every run, so you never have to read a
# timestamp out of the startup output and paste it into a tail command:
#
#     tail -f output/smm_runs/latest/run.log
#
# The link target is RELATIVE (just the folder name), so it keeps working if the whole
# output tree is moved or copied to another machine. A symlink is a convenience and never
# a reason to fail a run, so every failure here is swallowed -- some filesystems do not
# support them, and losing the estimation over that would be absurd.
let link = joinpath(dirname(RUN_DIR), "latest")
    try
        (islink(link) || ispath(link)) && rm(link; force = true, recursive = false)
        symlink(basename(RUN_DIR), link)
    catch
    end
end

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

# Printed before moments.jl is loaded, so a literal; asserted against the module below.
banner("SMM: 20 parameters (15 parent + 5 child) against 67 moments (2 P + 59 S + 5 T + 1 W)" *
       (QUICK ? "   [QUICK -- smoke test, not an estimate]" : ""))
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
sayf("purpose    %s -- %s\n", PRESET.name, PRESET.note)
sayf("budget     %s, %d restarts (<= %d evals each, polish %s)\n",
     N_VALID_TARGET > 0 ? "$N_VALID_TARGET VALID Sobol draws (at most $N_SOBOL attempts)" : "$N_SOBOL Sobol draws",
     N_RESTART, LOCAL_MAXEVAL, SKIP_POLISH ? "SKIPPED" : string(POLISH_MAXEVAL))
PRESET.name == "production" && !hasflag("--local-evals") &&
    say("           (the production cap of $LOCAL_MAXEVAL evals per restart is a default: set --local-evals from pilot traces)")
isempty(INIT_FROM) || sayf("init-from  %s\n", INIT_FROM)
if GRID_SEARCH != GRID_FULL
    sayf("grids      search at Na=Nhc=%d, fit REPORTED at Na=Nhc=%d\n", GRID_SEARCH, GRID_FULL)
else
    sayf("grids      Na=Nhc=%d for both the search and the report\n", GRID_FULL)
end
sayf("writing to %s\n", short(RUN_DIR))
sayf("           (also reachable as %s/latest -- tail that, no timestamp needed)\n",
     short(dirname(RUN_DIR)))

# -----------------------------------------------------------------------------
# Start the workers
# -----------------------------------------------------------------------------
# Each worker inherits --project so it resolves the same Manifest.toml, and gets
# a single BLAS thread (see the header).
#
# THE RUNNER OWNS ITS POOL (2026-09-28, tiktak_fix_plan.md 7.1). It starts in a fresh process and
# never adopts a worker it did not start; each worker gets an EXPLICIT Julia thread count
# (--threads=1): workers inherit the master's environment -- a JULIA_NUM_THREADS there would size
# them -- but not its --threads flag. What every process actually runs with is reported below.
nprocs() == 1 || error("run_smm.jl starts and owns its worker pool; this process already has $(nprocs() - 1) " *
                       "worker(s) it did not start. Run it in a fresh process.")
const POOL = if !SERIAL && NPROC > 0
    print("starting $NPROC worker processes ... "); flush(stdout)
    t = time()
    ws_ = addprocs(NPROC; exeflags = `--project=$REPO --threads=1`)
    sayf("%.1fs\n", time() - t)
    ws_
else
    Int[]
end
@everywhere using LinearAlgebra
@everywhere LinearAlgebra.BLAS.set_num_threads(1)
# a worker dies with the master (Linux PR_SET_PDEATHSIG): killing the run -- `tmux kill-session`
# included -- can no longer leave workers computing at full CPU (VERSION.md step 20)
nprocs() > 1 && @everywhere workers() begin
    include(joinpath($REPO, "code", "src", "tiktak.jl"))
    TikTak.die_with_parent!()
end
include(joinpath(REPO, "code", "src", "tiktak.jl"))       # (idempotent: loaded again with the model below)
const RESOURCES = TikTak.worker_resources(vcat(1, POOL))
let jt = unique(r.julia_threads for r in RESOURCES), bt = unique(r.blas_threads for r in RESOURCES)
    sayf("running on %d process(es): 1 master + %d worker(s); Julia threads per process %s, BLAS threads %s%s\n",
         nprocs(), length(POOL), join(jt, "/"), join(bt, "/"),
         isempty(get(ENV, "JULIA_NUM_THREADS", "")) ? "" : " (JULIA_NUM_THREADS=$(ENV["JULIA_NUM_THREADS"]) in the environment)")
    any(r -> r.id != 1 && r.julia_threads != 1, RESOURCES) && @warn "a worker runs more than one Julia thread" RESOURCES
end

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

const TARGETS_FILE = freeze_smm_targets(RUN_DIR; source=argstr("--targets", ""), resume=RESUMING)
@everywhere const TARGETS = load_targets($TARGETS_FILE)
# Fail at startup, not at the first evaluation, while an input of the wage loading is
# missing: child_wage_config refuses until sd(log AFQT) is supplied (moments.jl).
child_wage_config()
check_half_period_weight(target_mu_half(TARGETS))
SMM_TEST_FIXTURES && banner("SMM_TEST_FIXTURES=1 -- composition: " * TARGETS["_spec"].composition_source *
                            "; wage loading: PLACEHOLDER. A TEST, NOT AN ESTIMATE.")
# A5. The targets file's content hash. A run record that names the file is not enough --
# the file is regenerated by tools/make_smm_targets.py and its contents decide the answer.
using SHA
const TARGETS_SHA  = bytes2hex(SHA.sha256(read(TARGETS_FILE)))[1:16]

# SPECIFICATION IDENTITY. The targets hash alone no longer identifies the problem: four
# child parameters are estimated, so the MODEL SOURCE is part of the objective in a way it
# was not when the child block was a fixed spline. A checkpoint written before an edit to
# child_lifecycle.jl now describes a different objective, and resuming into it would mix
# two models in one sequence of restarts.
const SOURCE_FILES = ["code/src/child_lifecycle.jl", "code/src/parent_family.jl",
                      "code/smm/moments.jl"]
const SOURCE_SHA = bytes2hex(SHA.sha256(
    reduce(vcat, read(joinpath(REPO_, f)) for f in SOURCE_FILES)))[1:16]

# Bumped whenever the moment set, the parameter set or the parameter MEANING changes. The
# name/box checks below catch most of it; this catches the rest -- the 2026-09-10 psychic
# centring changed what kappa_0 MEANS without changing any name or bound.
# WHEN THE MOMENTS OR THE ESTIMATED PARAMETERS CHANGE, update these counts, the banner above and SPEC_VERSION below
# together (the check stops the run if they disagree with moments.jl).
(length(SMM_PARAMS), length(SMM_PARENT_PARAMS), length(SMM_CHILD_PARAMS), length(SMM_MOMENTS),
 length(SMM_P_MOMENTS), length(SMM_S_MOMENTS), length(SMM_T_MOMENTS), length(SMM_W_MOMENTS)) ==
    (20, 15, 5, 67, 2, 59, 5, 1) ||
    error("the banner says 20 = 15 + 5 parameters against 67 = 2 P + 59 S + 5 T + 1 W moments; moments.jl disagrees")
# 2026-10-02: memo-18 technology (sigma_j on age, logistic TFP, no shock), memo-19 moment vector.
const SPEC_VERSION = "smm20_memo19_v1" * (SMM_TEST_FIXTURES ? "_TESTFIX" : "")   # a fixture run is its own spec

# -----------------------------------------------------------------------------
# The initial point: the block starts, or a previous run's estimates by name
# -----------------------------------------------------------------------------
"""
    initial_point() -> (x0, natural)

The search-space incumbent and its natural-unit values. Without --init-from it is
`incumbent()` (the SMM starts). With it, every parameter the file names is taken from it
and every parameter it lacks starts at `smm_start` -- reported per parameter so a run
record never has to be reverse-engineered. A loaded value outside its box is refused:
silently clamping the incumbent of a sixteen-parameter search onto a wall is how a
"warm start" turns into a boundary artefact.
"""
function initial_point()
    nat = Dict{Symbol,Float64}(q.name => smm_start(q.name) for q in SMM_PARAMS)
    src = Dict{Symbol,String}(q.name => "smm_start" for q in SMM_PARAMS)
    if !isempty(INIT_FROM)
        isfile(INIT_FROM) || error("--init-from: no such file: $INIT_FROM")
        raw = TOML.parsefile(INIT_FROM)
        exact = exact_search_vector(raw)
        if exact.z !== nothing
            z = exact.z
            for (i, q) in enumerate(SMM_PARAMS)
                nat[q.name] = from_search(z[i], q); src[q.name] = "init-from, exact search vector"
            end
            return z, nat, src, "exact"
        end
        isempty(exact.note) || @warn "--init-from: " * exact.note
        haskey(raw, "parameters") || error("--init-from: $INIT_FROM has no [parameters] table")
        for q in SMM_PARAMS
            haskey(raw["parameters"], String(q.name)) || continue
            v = Float64(raw["parameters"][String(q.name)])
            q.lo <= v <= q.hi || error(@sprintf(
                "--init-from: %s = %.6g from %s is outside its box [%g, %g]", q.name, v, INIT_FROM, q.lo, q.hi))
            nat[q.name] = v; src[q.name] = "init-from"
        end
        extra = [k for k in keys(raw["parameters"]) if !any(q -> String(q.name) == k, SMM_PARAMS)]
        isempty(extra) || @warn "--init-from: ignoring parameters not in this specification" extra
        return [to_search(nat[q.name], q) for q in SMM_PARAMS], nat, src, "rounded"
    end
    return [to_search(nat[q.name], q) for q in SMM_PARAMS], nat, src, "smm_start"
end

"""
    exact_search_vector(raw) -> (z, note)

The exact start in a TOML file, if it has one: a `[search_vector]` `z` whose top-level
`param_names` and `param_link` equal this specification's, finite, inside the search box (a
coordinate within 1e-9 of its box width outside is snapped onto the bound, as TikTak does with
any supplied point). `z = nothing` when the file has no usable exact vector, with `note` saying
why when it has one that does not fit (then the caller falls back to [parameters] by name). An
exact vector that fits the names but is not a valid point is an ERROR, never a silent fallback.
"""
function exact_search_vector(raw::AbstractDict)
    sv = get(raw, "search_vector", nothing)
    (sv isa AbstractDict && haskey(sv, "z")) || return (z = nothing, note = "")
    names_now = [String(q.name) for q in SMM_PARAMS]; links_now = [String(q.link) for q in SMM_PARAMS]
    names = get(raw, "param_names", nothing); links = get(raw, "param_link", nothing)
    (names isa AbstractVector && links isa AbstractVector) ||
        return (z = nothing, note = "[search_vector] ignored: the file has no param_names / param_link to check it against; " *
                                    "loading [parameters] by name (ROUNDED)")
    (String.(names) == names_now && String.(links) == links_now) ||
        return (z = nothing, note = "[search_vector] ignored: its parameters or links differ from this specification " *
                                    "($(join(String.(names), ", "))); loading [parameters] by name (ROUNDED)")
    zraw = sv["z"]
    (zraw isa AbstractVector && length(zraw) == length(SMM_PARAMS) && all(x -> x isa Real, zraw)) ||
        error("--init-from: [search_vector] z must hold $(length(SMM_PARAMS)) numbers")
    z = Float64.(zraw)
    lo, hi = search_bounds()
    for (i, q) in enumerate(SMM_PARAMS)
        isfinite(z[i]) || error("--init-from: [search_vector] $(q.name) = $(z[i]) is not finite")
        w = hi[i] - lo[i]
        lo[i] - 1e-9 * w <= z[i] <= hi[i] + 1e-9 * w || error(@sprintf(
            "--init-from: %s = %.17g (search coordinates) is outside its box [%g, %g]", q.name, z[i], lo[i], hi[i]))
        z[i] = clamp(z[i], lo[i], hi[i])
    end
    return (z = z, note = "")
end

const X0, X0_NAT, X0_SRC, X0_PRECISION = initial_point()
# a short content hash of the start vector: what a gate or a later run can compare against
const X0_SHA = bytes2hex(SHA.sha256(reinterpret(UInt8, X0)))[1:16]

# -----------------------------------------------------------------------------
# A5. The reproducible run record -- written TWICE
# -----------------------------------------------------------------------------
# Once here, before anything runs, and again at the end with the results appended.
#
# Writing it only at the end was a mistake in the same family as checkpointing only at the
# end: a run that is killed, that stops at --report-only, or that dies in the report has no
# record of WHAT IT WAS RUN WITH, which is exactly what you want when you are trying to
# work out why it died. The settings are all known before the first evaluation, so there is
# no reason to wait. The second write rewrites the file in full and adds [result].
#
# `result = nothing` means the run has not finished; the file says so in `status`.
function write_run_record(result = nothing, q_final = NaN, q_search = NaN,
                          refine = nothing, accepted = false; status = nothing)
    open(joinpath(RUN_DIR, "run_record.toml"), "w") do io
        println(io, "# Reproducible run record. GENERATED by code/smm/run_smm.jl (A5).")
        println(io, "# Written at STARTUP and rewritten at the end. Per-restart results:")
        println(io, "# restarts.csv. Everything that can change the answer is here.")
        # "started" is the honest state until the run finishes: if the file still says
        # that, the run did not reach its end -- killed, crashed, or still going.
        println(io, "status       = \"",
                status !== nothing ? status : (result === nothing ? "started" : "finished"),
                "\"")
        println(io, "generated    = \"", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\"")
        println(io, "started      = \"", STAMP, "\"")
        println(io, "host         = \"", gethostname(), "\"")
        println(io, "julia        = \"", VERSION, "\"")
        println(io, "command      = \"run_smm.jl ", join(ARGS, " "), "\"")
        println(io, "spec_version = \"", SPEC_VERSION, "\"")
        println(io, "test_fixtures = ", SMM_TEST_FIXTURES, "   # SMM_TEST_FIXTURES=1: stand-in composition and wage loading, NOT an estimate")
        println(io, "grid_extra   = \"", GRID_EXTRA, "\"   # --parent-extra / --child-extra")
        println(io, "run_dir      = \"", short(RUN_DIR), "\"")
        println(io, "temp_run     = ", TEMP_LABEL !== nothing,
                "   # true = throwaway under temp/, not a result")
        println(io, "\n[code]")
        println(io, "git_commit   = \"", git_sha(), "\"")
        println(io, "dirty        = ", occursin("dirty", git_sha()),
                "   # true means uncommitted changes were in the working tree")
        println(io, "\n[targets]")
        println(io, "file         = \"", short(TARGETS_FILE), "\"")
        println(io, "sha256_16    = \"", TARGETS_SHA, "\"   # CONTENT hash; the file is regenerated")
        println(io, "moments      = [", join(("\"$m\"" for m in SMM_MOMENTS), ", "), "]")
        for k in SMM_MOMENTS
            @printf(io, "%-22s = %.10g\n", "target_" * k, TARGETS[k].mean)
        end
        println(io, "\n[parameters]")
        println(io, "names        = [", join(("\"$(q.name)\"" for q in SMM_PARAMS), ", "), "]")
        println(io, "lo           = [", join((q.lo for q in SMM_PARAMS), ", "), "]")
        println(io, "hi           = [", join((q.hi for q in SMM_PARAMS), ", "), "]")
        println(io, "link         = [", join(("\"$(q.link)\"" for q in SMM_PARAMS), ", "), "]")
        println(io, "start        = [",
                join((@sprintf("%.17g", v) for v in X0), ", "),
                "]   # search coords; the incumbent, forced into the Sobol pool")
        println(io, "init_from    = \"", INIT_FROM, "\"   # empty = the SMM starts")
        println(io, "start_precision = \"", X0_PRECISION, "\"   # exact = a [search_vector] bit for bit; rounded = [parameters] by name (8 decimals); smm_start")
        println(io, "start_sha16  = \"", X0_SHA, "\"   # sha256 of the start vector's bytes (search coordinates)")
        println(io, "expect_start_q = ", EXPECT_START_Q === nothing ? "\"\"" : repr(EXPECT_START_Q), "   # --expect-start-q: the run stops unless Q at the start equals it exactly")
        for q in SMM_PARAMS
            @printf(io, "%-18s = %.10g   # starting value, natural units (%s)\n",
                    "start_" * String(q.name), X0_NAT[q.name], X0_SRC[q.name])
        end
        # memo 19 (2026-10-02): mu_0/mu_1 and R_0/R_1 no longer exist -- the child's weight is the data's mu_t by age
        # and the TFP is the estimated logistic d_0..d_3 (in [parameters]); Ali's wording, 2026-10-02
        println(io, "fixed_note   = \"calibrated, not estimated: mu_t by age and mu_half from the target file; ",
                "sigma_eta = 0 (DFVW); TFP is the estimated logistic d_0..d_3\"")
        println(io, "mu_by_age    = [", join(target_mu_by_age(TARGETS), ", "), "]   # the child's weight at ages 6..17 (target file)")
        println(io, "mu_half      = ", target_mu_half(TARGETS), "   # the child's weight at the half period (target file)")
        println(io, "sigma_eta    = ", PARENT_DEFAULTS.sigma_eta, "   # the HC shock, fixed: DFVW's technology is deterministic")
        println(io, "spec_version = \"", SPEC_VERSION, "\"")
        println(io, "source_sha   = \"", SOURCE_SHA, "\"")
        println(io, "\n[numerical]")
        println(io, "seed         = ", SEED_, "   # common random numbers, identical across evaluations")
        println(io, "simN         = ", SIM_N)
        println(io, "grid_search  = ", GRID_SEARCH, "   # parent Na = Nhc during the search")
        println(io, "grid_report  = ", GRID_FULL, "   # parent Na = Nhc for the reported fit")
        println(io, "child_grid   = \"", CHILD_G_, "\"")
        println(io, "n_sobol      = ", N_SOBOL, N_VALID_TARGET > 0 ? "   # the ATTEMPT CAP (--sobol-valid is set)" : "   # attempted draws")
        println(io, "n_sobol_valid_target = ", N_VALID_TARGET, "   # 0 = the fixed-attempt design")
        println(io, "reuse_pretest = \"", REUSE_PRETEST, "\"")
        println(io, "n_restarts   = ", N_RESTART)
        println(io, "local_alg    = \"LN_NELDERMEAD\"")
        println(io, "local_ftol_rel = ", LOCAL_FTOL, "   # --local-ftol (default 1e-3)")
        println(io, "local_ftol_abs = 1e-10")
        println(io, "local_xtol_rel = 1e-8")
        println(io, "local_maxeval  = ", LOCAL_MAXEVAL)
        println(io, "polish_alg   = \"LN_BOBYQA\"")
        println(io, "polish_tol   = 1e-10")
        println(io, "polish_maxeval = ", SKIP_POLISH ? 0 : POLISH_MAXEVAL,
                SKIP_POLISH ? "   # --skip-polish: the polish stage is bypassed" : "")
        println(io, "skip_polish  = ", SKIP_POLISH)
        println(io, "normalize    = ", GEOM_NORMALIZE, "   # --normalize: local searches in box-normalized coordinates")
        println(io, "local_initial_step = ", GEOM_LOCAL_STEP, "   # fraction of each box width; 0 = NLopt's default simplex")
        println(io, "local_step_schedule = \"", GEOM_STEP_SCHEDULE, "\"   # fixed | theta_shrink")
        println(io, "local_step_min = ", GEOM_STEP_MIN, "   # floor of theta_shrink")
        println(io, "polish_initial_step = ", GEOM_POLISH_STEP, "   # fraction of each box width; 0 = NLopt's default")
        println(io, "refine_maxeval = ", REFINE_MAXEVAL)
        println(io, "Neta         = 5   # Gauss-Hermite nodes for the HC shock (parent constructor default)")
        println(io, "theta_schedule = \"clamp((j/Nstar)^0.5, 0.1, 0.995), restart 1 pinned at 0\"")
        println(io, "penalty      = ", SMM_PENALTY)
        println(io, "workers      = ", max(0, nprocs() - 1))
        # what each process ACTUALLY ran with (TikTak.worker_resources; master first), 2026-09-28
        println(io, "process_ids  = [", join((r.id for r in RESOURCES), ", "), "]")
        println(io, "julia_threads = [", join((r.julia_threads for r in RESOURCES), ", "), "]   # per process; workers get --threads=1")
        println(io, "blas_threads = [", join((r.blas_threads for r in RESOURCES), ", "), "]")
        println(io, "env_julia_num_threads = \"", get(ENV, "JULIA_NUM_THREADS", ""), "\"")
        println(io, "quick        = ", QUICK)
        println(io, "report_only  = ", REPORT_ONLY)
        println(io, "resumed      = ", RESUMING)
        println(io, "resume_mode  = \"", RESUMING ? String(RESUME_MODE) : "fresh", "\"   # fresh | state | pretest (legacy imports are not available in this repository)")
        println(io, "stop_after_restarts = ", STOP_AFTER == typemax(Int) ? -1 : STOP_AFTER, "   # -1 = no pause")
        println(io, "preset       = \"", PRESET.name, "\"   # what the run is for (TikTak presets); custom = none")
        println(io, "allow_fewer_restarts = ", ALLOW_FEWER)
        println(io, "require_valid_target = ", REQUIRE_VALID)
        println(io, "local_mode   = \"", LOCAL_MODE, "\"   # async = asynchronous restarts on the worker pool; serial = on the master")
        println(io, "bootstrap    = \"", BOOTSTRAP, "\"   # first_alone = restart 1 alone first; immediate_mixed = the pool fills at once")
        println(io, "runtime_evals = [", join(RUNTIME_EVALS, ", "), "]   # --runtime-evals: observed counts for the empirical runtime scenario (display only)")
        println(io, "local_procs  = ", LOCAL_PROCS, "   # 0 = automatic: min(workers, floor(sqrt(restarts)))")
        println(io, "max_retries  = ", MAX_RETRIES, "   # re-dispatches of a restart whose worker was lost")
        println(io, "pretest_chunk = ", TIKTAK_KW.pretest_chunk, "   # pre-testing values between cache writes")
        println(io, "state_file   = \"tiktak_state.toml\"   # the authoritative checkpoint (TikTak module)")
        if result !== nothing
            println(io, "\n[result]")
            println(io, "Q_final      = ", q_final)
            println(io, "Q_search     = ", q_search)
            println(io, "Q_incumbent  = ", q0)
            println(io, "winner_stage = \"", result.winner_stage, "\"")
            println(io, "winner_ret   = \"", result.winner_ret, "\"")
            println(io, "refine_status= \"", refine.status, "\"")
            println(io, "point_origin = \"", refine.incumbent.origin.stage, "\"   # the reported point's origin")
            println(io, "verification = \"", refine.incumbent.verification.status, "\"")
            println(io, "local_converged = ", TikTak.local_converged(refine.incumbent, OBJ_ID_REPORT),
                    "   # on the reporting objective")
            println(io, "run_status   = \"", result.status, "\"")
            # the solver settings the search RAN with (the last settings epoch; earlier epochs are in tiktak_state.toml)
            println(io, "effective_local  = \"", TikTak.solver_summary(result.config.local_), "\"")
            println(io, "effective_polish = \"", TikTak.solver_summary(result.config.polish), "\"")
            println(io, "effective_normalize = ", result.config.normalize)
            println(io, "settings_epochs = ", length(result.epochs), "   # > 1: settings changed on a resume (--allow-optimizer-change)")
            println(io, "accepted     = ", accepted)
            println(io, "n_eval_total = ", result.n_eval + refine.evals)
            println(io, "minutes      = ", round(elapsed(), digits = 1))
        end
    end
    return nothing
end
@everywhere const CHILD_G_ = $(QUICK ? (Na = 12, Nk = 12, Nt = 3) : (Na = 30, Nk = 30, Nt = 5))
# G_ is what every objective evaluation uses; G_FULL_ is what the reported fit uses.
# They are the same unless --grid was passed. The child grid is untouched either way.
@everywhere const G_      = (Na = $GRID_SEARCH, Nk = 2, Nhc = $GRID_SEARCH, simN = $SIM_N)
@everywhere const G_FULL_ = (Na = $GRID_FULL,   Nk = 2, Nhc = $GRID_FULL,   simN = $SIM_N)

"""
    objective_identity() -> Vector{Pair{String,Any}}

Everything that identifies the objective this run minimises, in the order a checkpoint
records it: the parameter set and box, the targets by content, the model source, the
specification, the moment set, the centring, and the numerical problem (grids, simulated
households, seed). ONE definition, used by `checkpoint!` and by `--print-identity`, so a
test fixture cannot drift from what the runner writes and verifies.
"""
objective_identity() = Pair{String,Any}[
    "param_names"  => [String(q.name) for q in SMM_PARAMS],
    "param_lo"     => [q.lo for q in SMM_PARAMS],
    "param_hi"     => [q.hi for q in SMM_PARAMS],
    "param_link"   => [String(q.link) for q in SMM_PARAMS],
    "targets_sha"  => TARGETS_SHA,
    "source_sha"   => SOURCE_SHA,
    "spec_version" => SPEC_VERSION,
    "grid_extra"   => GRID_EXTRA,
    "m_psychic"    => target_m_psychic(TARGETS),
    "moment_names" => collect(String, SMM_MOMENTS),
    "child_grid"   => string(CHILD_G_.Na, "x", CHILD_G_.Nk, "x", CHILD_G_.Nt),
    "seed"         => SEED_,
    "sim_n"        => SIM_N,
    "grid_search"  => GRID_SEARCH,
    "grid_report"  => GRID_FULL,
]
if PRINT_IDENTITY
    println("\n# --print-identity: the objective identity a checkpoint of this run records")
    TOML.print(stdout, Dict{String,Any}(objective_identity()))
    close(LOG); exit(0)
end

"""
    objective_id(grid) -> String

A short content hash of the objective evaluated at parent grid `grid`: objective_identity()
without its two grid fields, plus the grid. The search objective and the reporting
objective get different ids unless the search already ran at the reporting grid, so a
convergence certificate earned on the coarse grid is never read as one on the full grid.
"""
function objective_id(grid::Int)
    d = Dict{String,Any}(k => v for (k, v) in objective_identity() if !(k in ("grid_search", "grid_report")))
    d["objective_grid"] = grid
    io = IOBuffer(); TOML.print(io, d; sorted = true)
    return bytes2hex(SHA.sha256(take!(io)))[1:16]
end
const OBJ_ID_SEARCH = objective_id(GRID_SEARCH)
const OBJ_ID_REPORT = objective_id(GRID_FULL)

# -----------------------------------------------------------------------------
# The optimizer settings, and --resume: validated HERE, before the child warm-up
# -----------------------------------------------------------------------------
# THE RESUME IS VALIDATED BEFORE THE CHILD WARM-UP, THE TIMING EVALUATION, THE REPORT AND
# `--report-only`. It used to be built at the search stage, so a `--report-only --resume`
# performed no compatibility checking; since 2026-09-27 it also runs before any evaluation,
# so a resume that is going to be refused costs a model load and nothing else.
#
# ONE set of optimizer settings serves the preflight below and the search itself, so the
# optimizer identity a checkpoint is checked against is the one the search then uses.
const STATE_F = joinpath(RUN_DIR, "tiktak_state.toml")   # the AUTHORITATIVE checkpoint (TikTak module)
const PRETEST_F = joinpath(RUN_DIR, "pretest_cache.toml") # every pre-testing value, chunk by chunk
# touch <run dir>/PAUSE to pause an asynchronous run gracefully: no new restart starts, the
# running ones are committed, the state is checkpointed; resume with --resume <run dir>.
const PAUSE_F = joinpath(RUN_DIR, "PAUSE")
const LO_S, HI_S = search_bounds()
const USE_PMAP = !(SERIAL || nprocs() == 1)
const TIKTAK_KW = (N = N_SOBOL, Nstar = N_RESTART,
                   extra_seeds = [X0],             # the incumbent competes like any Sobol point
                   invalid_value = SMM_PENALTY,    # a penalised draw never seeds a restart (ported from v2, 2026-09-27)
                   local_maxeval = LOCAL_MAXEVAL, polish_maxeval = POLISH_MAXEVAL,
                   local_tol = LOCAL_FTOL,         # --local-ftol (finding 26)
                   bootstrap = BOOTSTRAP,          # --bootstrap (fix plan R2)
                   skip_polish = SKIP_POLISH,
                   objective_id = OBJ_ID_SEARCH,
                   objective_fields = Dict{String,Any}(objective_identity()),
                   state_path = STATE_F,
                   n_valid_target = N_VALID_TARGET,
                   pretest_cache = PRETEST_F, pretest_reuse = REUSE_PRETEST,
                   # values between cache writes: a few rounds of the worker pool
                   pretest_chunk = argval("--pretest-chunk", max(8, 4 * max(nprocs() - 1, 1))),
                   stop_after_restarts = STOP_AFTER,
                   allow_optimizer_change = ALLOW_OPT_CHANGE,
                   # one pool of worker processes for both stages; the objective is registered
                   # on each of them under :search (below), and jobs carry only coordinates
                   local_mode = LOCAL_MODE === :async ? :async_process : :serial,
                   local_workers = POOL, local_count = LOCAL_PROCS,
                   pretest_workers = USE_PMAP ? POOL : Int[],
                   # the pool is this runner's own and ends with it: a worker still solving after an
                   # abort's drain deadline is removed rather than left running (follow-up 3)
                   retire_after_abort = true,
                   objective_key = :search, progress_every = EVERY_SEC,
                   pause_file = PAUSE_F, max_retries = MAX_RETRIES,
                   allow_fewer_restarts = ALLOW_FEWER, require_valid_target = REQUIRE_VALID,
                   purpose = PRESET.name,
                   # search geometry (2026-09-29; defaults = the baseline, see the flags above)
                   normalize = GEOM_NORMALIZE, local_initial_step = GEOM_LOCAL_STEP,
                   local_step_schedule = GEOM_STEP_SCHEDULE, local_step_min = GEOM_STEP_MIN,
                   polish_initial_step = GEOM_POLISH_STEP)

refuse_resume(dir, msg) = error("--resume refuses to continue $(short(dir)):\n    " * msg *
                                "\n  Start a fresh run instead. Resuming across a changed problem " *
                                "would mix two different\n  estimations in one sequence of restarts.")

"The refusal for one changed identity field, in the words the checks have always used."
function identity_refusal(k, a, b)
    k == "param_names"  && return "parameter set changed:\n      saved $(join(a, ", "))\n      now   $(join(b, ", "))"
    k in ("param_lo", "param_hi") && return "$k changed:\n      saved $a\n      now   $b"
    k == "param_link"   && return "parameter links changed; search coordinates are not comparable."
    k == "targets_sha"  && return "the targets file changed since that run (sha $a -> $b)."
    k == "source_sha"   && return "the model source changed since that run (sha $a -> $b).\n    " *
                                  "Files hashed: " * join(SOURCE_FILES, ", ")
    k == "spec_version" && return "specification changed: saved \"$a\", now \"$b\"."
    k == "grid_extra"   && return "grid / numerical overrides changed: saved \"$a\", now \"$b\"."
    k == "m_psychic"    && return "the psychic-cost centring changed ($a -> $b); kappa_0 is on a different scale."
    k == "moment_names" && return "moment set changed:\n      saved $(join(a, ", "))\n      now   $(join(b, ", "))"
    k == "child_grid"   && return "child grid changed: saved $a, now $b. The child block is part of the objective."
    k == "sim_n"        && return "simulated households changed: saved $a, now $b. Q moves with simN at fixed parameters."
    k == "seed"         && return "seed changed: saved $a, now $b. Common random numbers: two seeds are two objectives."
    k == "grid_search"  && return "that run searched at grid $a, this one at $b -- the objectives differ."
    k == "grid_report"  && return "report grid changed: saved $a, now $b. The reported fits would not be comparable."
    return "$k changed: saved $(repr(a)), now $(repr(b))."
end

"""
    verify_state_identity(dir) -> NamedTuple

The runner's check of a CURRENT-format checkpoint (tiktak_state.toml): every field of
objective_identity(), by name, with a message that says what changed, and the requested
restart count. The TikTak module then checks the objective hash and the optimizer identity.
"""
function verify_state_identity(dir::AbstractString)
    d, which = try
        TikTak.read_state_dict(joinpath(dir, "tiktak_state.toml"))
    catch e
        refuse_resume(dir, "its tiktak_state.toml cannot be read: " * sprint(showerror, e))
    end
    saved = get(get(d, "objective", Dict{String,Any}()), "fields", Dict{String,Any}())
    for (k, now_) in objective_identity()
        haskey(saved, k) || refuse_resume(dir, "that checkpoint has no `$k` field, so this run cannot verify " *
                                               "that it describes the same objective.")
        TikTak._same(saved[k], now_) || refuse_resume(dir, identity_refusal(k, saved[k], now_))
    end
    nreq = Int(d["plan"]["nstar_requested"])
    nreq == N_RESTART || refuse_resume(dir, "that run had --restarts $nreq, this one has $N_RESTART. A different " *
        "restart count changes the mixing schedule: that is a new search (warm-start it with --init-from).")
    return (next_j = Int(d["progress"]["next_j"]), fZ = Float64(d["incumbent"]["f"]),
            K = Int(d["plan"]["schedule_denominator"]), stage = String(d["stage"]),
            status = String(d["status"]), which = which)
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

    # A5 (2026-09-10). THE FOURTEEN-PARAMETER SPECIFICATION IS NOT RESUME-COMPATIBLE WITH
    # ANYTHING WRITTEN BEFORE IT. A pre-2026-09-10 checkpoint has ten parameter names and
    # is already refused above -- but these three catch what the name check cannot:
    #
    #   spec_version  the psychic cost was RECENTRED. kappa_0 keeps its name and its box
    #                 and means something different, so a saved incumbent is a valid point
    #                 in an invalid parameterisation. Nothing else would notice.
    #   source_sha    four child parameters are estimated, so child_lifecycle.jl is part of
    #                 the objective now. An edit to it between runs changes Q at a fixed z.
    #   moment_names  seventeen moments, and the covariance is indexed by their ORDER.
    #
    # A missing field means the checkpoint predates the field, which is itself grounds to
    # refuse: it cannot be verified, and "cannot verify" is not "compatible".
    # REQUIRED, NOT OPTIONAL. Until this fix these read `haskey(ck, f) && <mismatch> &&
    # refuse(...)`, which ACCEPTS a checkpoint that simply lacks the field -- exactly the
    # case the comment above says must be refused. Only `spec_version` was genuinely
    # required. `require` makes "cannot verify" mean "refuse".
    require(f) = haskey(ck, f) || refuse(
        "that checkpoint has no `$f` field, so this run cannot verify that it describes " *
        "the same\n    objective. It predates the current specification. " *
        "Start a fresh run.")

    for f in ("spec_version", "source_sha", "moment_names", "m_psychic",
              "child_grid", "sim_n", "seed", "grid_report")
        require(f)
    end

    get(ck, "grid_extra", "parent:  | child: ") == GRID_EXTRA || refuse(
        "grid / numerical overrides changed: saved \"$(get(ck, "grid_extra", "(none)"))\", now \"$GRID_EXTRA\".")
    ck["spec_version"] == SPEC_VERSION || refuse(
        "specification changed: saved \"$(ck["spec_version"])\", now \"$SPEC_VERSION\".\n" *
        "    Parameter names and boxes can be identical across a specification change and " *
        "still\n    describe different models -- the 2026-09-10 psychic recentring did " *
        "exactly that.")
    ck["source_sha"] == SOURCE_SHA || refuse(
        "the model source changed since that run (sha $(ck["source_sha"]) -> $SOURCE_SHA).\n" *
        "    Four CHILD parameters are estimated, so child_lifecycle.jl is part of the\n" *
        "    objective: the same z would score differently. Files hashed: " *
        join(SOURCE_FILES, ", "))
    String.(ck["moment_names"]) == collect(SMM_MOMENTS) || refuse(
        "moment set changed:\n      saved $(join(ck["moment_names"], ", "))" *
        "\n      now   $(join(SMM_MOMENTS, ", "))")
    isapprox(Float64(ck["m_psychic"]), target_m_psychic(TARGETS); atol = 1e-9) || refuse(
        "the psychic-cost centring changed ($(ck["m_psychic"]) -> " *
        "$(target_m_psychic(TARGETS))); kappa_0 is on a different scale.")

    # THE NUMERICAL PROBLEM, not just the specification.
    #
    # `Q_best` and the saved seeds are loaded back and compared against values this run
    # computes. That comparison is only meaningful if both sides solve the same numerical
    # problem, and grid_search alone does not establish it: the CHILD grid, the number of
    # simulated households and the RNG seed all move Q at fixed parameters. Resuming a
    # simN = 2000 checkpoint into a simN = 500 run kept the old, better-resolved `Q_best`
    # as an incumbent that the new run's noisier evaluations could not beat, so every
    # subsequent restart was measured against a number from a different problem.
    child_now = string(CHILD_G_.Na, "x", CHILD_G_.Nk, "x", CHILD_G_.Nt)
    String(ck["child_grid"]) == child_now || refuse(
        "child grid changed: saved $(ck["child_grid"]), now $child_now. The child block " *
        "is part of\n    the objective now, so Q is not comparable across its grid.")
    Int(ck["sim_n"]) == SIM_N || refuse(
        "simulated households changed: saved $(ck["sim_n"]), now $SIM_N. Q moves with " *
        "simN at fixed\n    parameters, so the saved Q_best is not comparable.")
    Int(ck["seed"]) == SEED_ || refuse(
        "seed changed: saved $(ck["seed"]), now $SEED_. Common random numbers are what " *
        "make Q a\n    smooth function of the parameters; two seeds are two objectives.")
    Int(ck["grid_report"]) == GRID_FULL || refuse(
        "report grid changed: saved $(ck["grid_report"]), now $GRID_FULL. The refinement " *
        "stage and\n    the reported fit would not be comparable with that run's.")

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

"""
    load_legacy_seeds_only(dir) -> NamedTuple

A run written before 2026-09-27 that was stopped after its pre-testing but before its first
restart committed has seeds.toml and no checkpoint.toml. Its identity is checked against its
run_record.toml; the local stage then starts at restart 1 from the best seed.
"""
function load_legacy_seeds_only(dir::AbstractString)
    rr_p, sd_p = joinpath(dir, "run_record.toml"), joinpath(dir, "seeds.toml")
    isfile(rr_p) || refuse_resume(dir, "no checkpoint.toml and no run_record.toml to verify the seeds against.")
    rr, sd = TOML.parsefile(rr_p), TOML.parsefile(sd_p)
    pr, nu = rr["parameters"], rr["numerical"]
    checks = ("param_names" => (String.(pr["names"]), [String(q.name) for q in SMM_PARAMS]),
              "param_lo" => (Float64.(pr["lo"]), [q.lo for q in SMM_PARAMS]),
              "param_hi" => (Float64.(pr["hi"]), [q.hi for q in SMM_PARAMS]),
              "param_link" => (String.(pr["link"]), [String(q.link) for q in SMM_PARAMS]),
              "targets_sha" => (rr["targets"]["sha256_16"], TARGETS_SHA),
              "source_sha" => (pr["source_sha"], SOURCE_SHA), "spec_version" => (pr["spec_version"], SPEC_VERSION),
              "grid_extra" => (rr["grid_extra"], GRID_EXTRA), "seed" => (nu["seed"], SEED_),
              "sim_n" => (nu["simN"], SIM_N), "grid_search" => (nu["grid_search"], GRID_SEARCH),
              "grid_report" => (nu["grid_report"], GRID_FULL), "child_grid" => (nu["child_grid"], string(CHILD_G_)))
    for (k, (a, b)) in checks
        a == b || refuse_resume(dir, identity_refusal(k, a, b))
    end
    Int(nu["n_restarts"]) == N_RESTART || refuse_resume(dir, "that run had --restarts $(nu["n_restarts"]), this one has $N_RESTART.")
    seeds = [Float64.(v) for v in sd["seeds"]]
    return (seeds = seeds, f_sobol_best = Float64(sd["f_sobol_best"]), Z = copy(seeds[1]),
            fZ = Float64(sd["f_sobol_best"]), j_start = 1, history = NamedTuple[])
end

const RESUME_MODE = if !RESUMING
    :fresh
elseif TikTak.checkpoint_exists(STATE_F)
    :state
elseif TikTak.checkpoint_exists(PRETEST_F)
    :pretest                               # killed during pre-testing: its completed values are reused
elseif isfile(joinpath(RESUME_DIR, "checkpoint.toml")) || isfile(joinpath(RESUME_DIR, "seeds.toml"))
    LEGACY_IMPORT || refuse_resume(RESUME_DIR, "it is a LEGACY checkpoint (checkpoint.toml / seeds.toml, written " *
        "before the TikTak port\n    of 2026-10-02; no tiktak_state.toml). It cannot be resumed in this repository. " *
        "Warm-start a new run with\n    --init-from $(short(joinpath(RESUME_DIR, "checkpoint.toml"))).")
    :legacy
else
    error("--resume: $(short(RESUME_DIR)) holds no checkpoint (no tiktak_state.toml, checkpoint.toml or seeds.toml)")
end
const RESUME_INFO = RESUME_MODE === :state ? verify_state_identity(RESUME_DIR) : nothing
const RESUME_STATE = RESUME_MODE === :legacy ?
    (isfile(joinpath(RESUME_DIR, "checkpoint.toml")) ? load_resume(RESUME_DIR) : load_legacy_seeds_only(RESUME_DIR)) :
    nothing
const RESUME_ARG = RESUME_MODE in (:state, :pretest) ? true : RESUME_MODE === :legacy ? RESUME_STATE : false
# The TikTak module's own check -- the objective hash and the optimizer identity -- through
# the code path the search itself uses; nothing is evaluated.
if RESUMING
    try
        tiktak(x -> error("the preflight evaluates nothing"), LO_S, HI_S; TIKTAK_KW..., resume = RESUME_ARG,
               preflight_only = true)
    catch e
        e isa TikTak.ResumeRefused ? refuse_resume(RESUME_DIR, e.msg) : rethrow()
    end
end
# what is being resumed, said before anything expensive (and in --report-only runs too)
if RESUMING
    banner("RESUMING")
    if RESUME_MODE === :state
        sayf("continuing %s from restart %d of %d (checkpoint status %s, stage %s%s)\n", short(RESUME_DIR),
             RESUME_INFO.next_j, RESUME_INFO.K, RESUME_INFO.status, RESUME_INFO.stage,
             RESUME_INFO.which === :current ? "" : "; the newest generation did not verify, using the $(RESUME_INFO.which) one")
        sayf("incumbent from the checkpoint: Q = %.6g (at grid %d)\n", RESUME_INFO.fZ, GRID_SEARCH)
        say("The pre-testing stage is skipped: the seeds, the incumbent with its origin and every")
        say("committed restart are reloaded from tiktak_state.toml; restarts still in flight are replayed")
        say("from their recorded starts. How the result relates to an uninterrupted run is recorded as")
        say("resume_semantics (serial_exact only if every segment was serial and nothing was replayed).")
    elseif RESUME_MODE === :pretest
        sayf("continuing the PRE-TESTING stage of %s: its completed values are reused from pretest_cache.toml\n",
             short(RESUME_DIR))
    else
        sayf("LEGACY IMPORT of %s at restart %d of %d (--legacy-import)\n", short(RESUME_DIR),
             RESUME_STATE.j_start, length(RESUME_STATE.seeds))
        sayf("incumbent from the checkpoint: Q = %.6g (at grid %d); its origin and the earlier evaluation\n",
             RESUME_STATE.fZ, GRID_SEARCH)
        say("counts were never recorded and are reported as UNKNOWN.")
    end
end


# -----------------------------------------------------------------------------
# Child solve: rebuilt per evaluation, from a complete dependency key
# -----------------------------------------------------------------------------
# THIS USED TO BE `@everywhere const V_CHILD = build_child_value()` -- one child solve per
# process, reused by every evaluation. That was exact while all ten estimated parameters
# were parent-block parameters, and it is NOT exact now: kappa_0, kappa_theta, kappa_ParEd
# and kappa_terminal are estimated, and every one of them changes the child solve. Keeping
# the old line would have produced converged runs for a model that was never solved.
#
# The rebuild lives in `build_child_solution` (moments.jl), which reuses only the two
# stages that provably read none of the four -- the high-school path and the graduate's
# working life -- and redoes the rest. That is bit-identical to a full re-solve and about
# 16.9x faster, so an evaluation costs ~1.2 s more than it did rather than ~12.9 s more.
#
# The per-process cache is warmed here so the first Sobol batch is not paying for it while
# the progress meter reports nothing.
print("warming the child solve on every process ... "); flush(stdout)
t = time()
# child_wage: memo 19's child_config requires the wage loading (2026-10-02 merge fix: the memo-19 runner called it
# without, which would stop at this line once child_wage_config accepts its inputs)
@everywhere let cfg = child_config(TARGETS; Na = CHILD_G_.Na, Nk = CHILD_G_.Nk,
                                            Nt = CHILD_G_.Nt, simN = G_.simN, seed = $SEED_,
                                            child_wage = child_wage_config())
    child_base(cfg)
end
sayf("%.1fs (all processes at once)\n", time() - t)

# The centring constant is frozen in the target file and CHILD_DEFAULTS.kappa_0 has to be
# the legacy psychic cost re-expressed at it. Checked here, once, before any evaluation:
# a mismatch would leave the starting value 0.21 off on a parameter whose whole box is 7
# wide, and it would look like a bad fit rather than a bookkeeping error.
check_psychic_centring(target_m_psychic(TARGETS))

@everywhere objective(z) = smm_objective(z, TARGETS;
                                         Na = G_.Na, Nk = G_.Nk, Nhc = G_.Nhc,
                                         simN = G_.simN, seed = $SEED_,
                                         child_grid = CHILD_G_,
                                         # The demonstration child simulation is skipped
                                         # inside the search: it supplies nothing (its
                                         # outputs are erased before the parent block runs)
                                         # and it costs a simulation per evaluation. It IS
                                         # run in the reported fit, where the full
                                         # specified sequence is what is being reported.
                                         demo_sim = false,
                                         child_extra = $CE_RUN, parent_extra = $PE_RUN)
# Registered ONCE per process: a local-search job names the objective by key, and each
# worker evaluates its own copy (model caches and NLopt state stay private to the process).
@everywhere TikTak.register_objective!(:search, objective)

# -----------------------------------------------------------------------------
# Live progress
# -----------------------------------------------------------------------------
# Everything funnels into tick!, which throttles to one line per --every seconds:
#   serial         the master evaluates, and objective_tracked ticks after each value
#   worker pool    the TikTak module reports each pre-testing value as it arrives (on_sobol)
#                  and the running restarts' latest state every --every seconds (on_progress,
#                  coalesced messages from the workers). Telemetry never blocks a worker and a
#                  result never waits for it.
# (Until 2026-09-27 workers pushed every value into a bounded RemoteChannel that only the
# Sobol stage drained; a parallel local stage would have filled it and blocked the workers.)
mutable struct Tracker
    stage::Symbol      # :sobol or :local
    done::Int          # evaluations finished in the current stage
    total::Int         # expected evaluations, :sobol only (:local has no known total); the attempt CAP with --sobol-valid
    valid::Int         # :sobol: VALID values received so far (not penalised)
    target::Int        # :sobol with --sobol-valid: the valid values that end the stage; 0 = a fixed number of attempts
    best::Float64      # the best VALID Q so far (a penalised value is not a Q)
    restart::Int
    nrestart::Int
    t0::Float64        # start of the CURRENT stage, for the Sobol ETA
    trun::Float64      # start of the whole search, so every line agrees on the clock
    tlast::Float64
end
const TRACKER = Tracker(:sobol, 0, 0, 0, 0, Inf, 1, N_RESTART, time(), time(), time())

function stage!(s::Symbol, total::Int = 0; target::Int = 0)
    TRACKER.stage = s; TRACKER.done = 0; TRACKER.total = total
    TRACKER.valid = 0; TRACKER.target = target
    TRACKER.t0 = time(); TRACKER.tlast = time()
end

"The best valid Q for a progress line, or a plain statement that there is none yet (it used to print the penalty, 1e+12)."
bstr(q) = isfinite(q) ? @sprintf("%11.4g", q) : " (none valid)"

function tick!(q::Float64)
    T = TRACKER
    T.done += 1
    ok = isfinite(q) && q < SMM_PENALTY                  # a penalised value is neither valid nor a "best Q"
    ok && q < T.best && (T.best = q)
    T.stage === :sobol && ok && (T.valid += 1)
    now_ = time()
    last = T.stage === :sobol && T.done == T.total      # always print the final line
    (last || now_ - T.tlast >= EVERY_SEC) || return
    T.tlast = now_
    if T.stage === :sobol && T.target > 0
        # --sobol-valid (fixed 2026-10-02, Ali: "the way you print the sobol stage is wrong"): the stage
        # ends when T.target values are VALID, so progress and the ETA count valid values; the attempts
        # are shown against their cap. It printed attempts / valid target ("1236/151  819%") and a
        # negative time left.
        el   = (now_ - T.t0) / 60
        frac = min(T.valid / T.target, 1.0)
        eta  = T.valid >= T.target ? "target reached" :
               T.valid == 0 ? "no ETA before the first valid draw" :
               @sprintf("~%.0f min left", el * (T.target - T.valid) / T.valid)
        sayf("  sobol    valid %4d/%-4d %3.0f%%   attempts %5d (cap %d)   best Q %s   %5.1f min elapsed, %s\n",
             T.valid, T.target, 100frac, T.done, T.total, bstr(T.best), el, eta)
    elseif T.stage === :sobol
        el   = (now_ - T.t0) / 60
        frac = T.done / max(T.total, 1)
        eta  = frac > 0 ? el * (1 - frac) / frac : 0.0
        sayf("  sobol    %5d/%-5d %3.0f%%   valid %4d   best Q %s   %5.1f min elapsed, ~%.0f min left\n",
             T.done, T.total, 100frac, T.valid, bstr(T.best), el, eta)
    else
        sayf("  restart %3d/%-3d  eval %5d   this Q %s   best Q %s   %5.1f min\n",
             T.restart, T.nrestart, T.done, qstr(q), qstr(T.best), (now_ - T.trun) / 60)
    end
    return
end

"The master's objective in serial mode: evaluate, then report."
objective_tracked(z) = (q = objective(z); tick!(q); q)

"Q with four decimals, so a small improvement shows (916.3000 -> 916.2731); huge (penalty) values in e-notation."
qstr(q) = isfinite(q) && abs(q) < 1e7 ? @sprintf("%12.4f", q) : @sprintf("%12.4e", q)

# ONE LINE PER FINISHED EVALUATION (2026-09-29). on_progress is polled every --every seconds with
# each running job's latest state; a job is printed only when its evaluation count has moved
# since it was last printed, so a 35 s evaluation gives one line, not 15 identical ones.
# Keyed by (stage, restart, attempt): a re-dispatched attempt starts its count again.
const SHOWN_EVALS = Dict{Tuple{Symbol,Int,Int},Int}()

"Each running job that finished another evaluation since the last poll (asynchronous local stage and polish)."
function print_local_progress(evs)
    for ev in evs
        key = (ev.stage, ev.j, ev.attempt)
        ev.n_eval > get(SHOWN_EVALS, key, 0) || continue
        SHOWN_EVALS[key] = ev.n_eval
        who = ev.stage === :polish ? "polish         " : @sprintf("restart %3d/%-3d", ev.j, TRACKER.nrestart)
        sayf("  %s  eval %5d   this Q %s   best in this search %s   incumbent %s   %5.1f min   [%d running]\n",
             who, ev.n_eval, qstr(ev.f_last), qstr(ev.f_best), qstr(TRACKER.best),
             (time() - TRACKER.trun) / 60, length(evs))
    end
end

# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------
banner("Targets  ($(short(TARGETS_FILE)))")
# COUNTS ARE NOT IDENTIFICATION (2026-09-28, tiktak_problems.md finding 10). Moments >= parameters is
# only the ORDER condition: it does not establish identification (that needs the rank of the moment
# Jacobian at the estimate -- tools/check_jacobian_rank.jl), equal counts do not guarantee that Q can
# reach 0, and more moments than parameters does not imply that Q stays above 0. Whether the optimizer
# failed or the specification cannot match the targets is read from the fit and those diagnostics.
sayf("%d moments, %d parameters -- %s\n", length(SMM_MOMENTS), length(SMM_PARAMS),
     length(SMM_MOMENTS) == length(SMM_PARAMS) ? "as many moments as parameters" :
     length(SMM_MOMENTS) >  length(SMM_PARAMS) ? "$(length(SMM_MOMENTS) - length(SMM_PARAMS)) more moments than parameters (the weights matter at a misfit)" :
                                                 "FEWER moments than parameters: the order condition fails")
say("(a count is not identification, and says nothing about the attainable Q: see the fit table and",
    " tools/check_jacobian_rank.jl)")
sayf("parameters: %s\n", join((String(q.name) for q in SMM_PARAMS), ", "))
isempty(PE_STR * CE_STR) || sayf("grid / numerical overrides: %s\n", GRID_EXTRA)

# memo 19: the target entries carry the bootstrap SE (the weight is 1/se^2), not a cross-sectional SD
# (2026-10-02 merge fix: the memo-19 runner still printed tg.sd, which load_targets no longer provides)
sayf("%-28s %12s %12s %8s   %s\n", "moment", "mean", "se", "N", "source")
for k in SMM_MOMENTS
    tg = TARGETS[k]
    sayf("%-28s %12.4f %12.4f %8d   %s\n", k, tg.mean, tg.se, tg.n, tg.source)
end

# SETTINGS ON DISK BEFORE ANYTHING RUNS. A killed run, a --report-only run and a run that
# dies in the report all leave a record of what they were run with -- which is precisely
# when you want it. Rewritten with [result] at the end.
write_run_record()
sayf("\nsettings written to %s  (status = started)\n",
     short(joinpath(RUN_DIR, "run_record.toml")))

"""
Solve once at `z`, print the fit report to BOTH the console and the log.

Always at G_FULL_, never at the search grid: the number that leaves this run has
to be the fit at full resolution, whatever grid the optimizer happened to use to
find the parameters.
"""
function say_report(z)
    buf = IOBuffer()
    r = report_fit(z, TARGETS; Na = G_FULL_.Na, Nk = G_FULL_.Nk, Nhc = G_FULL_.Nhc,
                   simN = G_FULL_.simN, seed = SEED_, child_grid = CHILD_G_, out = buf,
                   child_extra = CE_RUN, parent_extra = PE_RUN)
    s = String(take!(buf))
    lock(OUTLOCK) do
        print(s); print(LOG, s); flush(LOG)
    end
    return r
end

# -----------------------------------------------------------------------------
# Time one evaluation, then predict the run
# -----------------------------------------------------------------------------
x0 = X0
say("initial point")
for q in SMM_PARAMS
    sayf("  %-14s %12.6g   (%s)\n", q.name, X0_NAT[q.name], X0_SRC[q.name])
end
sayf("  %-14s %12.6g   (FIXED, not estimated: DFVW's technology is deterministic)\n", "sigma_eta", PARENT_DEFAULTS.sigma_eta)
print("timing one objective evaluation ... "); flush(stdout)
t = time(); q0 = objective(x0); T_EVAL = time() - t
sayf("%.1fs\n", T_EVAL)
# THE START, AT FULL PRECISION (2026-09-29, finding 18): what a later gate or run compares against.
say("start Q (full precision) = ", repr(q0), "   [", X0_PRECISION, " start, sha ", X0_SHA, "]")
if EXPECT_START_Q !== nothing
    if q0 == EXPECT_START_Q
        say("start gate passed: Q at the start this run loaded equals --expect-start-q bit for bit")
    else
        write_run_record(; status = "start_gate_failed")
        error("--expect-start-q: Q at the start this run loaded is $(repr(q0)), expected $(repr(EXPECT_START_Q)) " *
              "(difference $(q0 - EXPECT_START_Q)). The start ($X0_PRECISION, sha $X0_SHA) or the objective is not the " *
              "one the expected value was computed for. Nothing was searched.")
    end
end

const NW = max(1, nprocs() - 1)
# THE RUNTIME PROJECTION (2026-09-29, fix plan R1.2; code/smm/runtime_projection.jl). Scenarios, not
# one number: until 2026-09-29 this assumed 165 evaluations per restart (measured 2026-08-27 on an
# older, smaller problem) and left out the polish -- mu08's restarts used 268-500 evaluations and
# E1-P allows 1,000, so the printed total could be several times too short. The Sobol stage divides
# by the worker count; the asynchronous local stage runs in rounds of NL local workers after its
# start-up rule (--bootstrap); a serial local stage runs one restart at a time on the master.
include(joinpath(REPO, "code", "smm", "runtime_projection.jl"))
const N_SOBOL_EVAL = (N_VALID_TARGET > 0 ? N_VALID_TARGET : N_SOBOL) + 1   # +1: the supplied start; a valid
                                                 # target needs AT LEAST that many attempts
const NL = LOCAL_MODE === :async ? min(NW, N_RESTART, LOCAL_PROCS > 0 ? LOCAL_PROCS : max(1, floor(Int, sqrt(N_RESTART)))) : 1
const N_POLISH_EVAL = SKIP_POLISH ? 0 : POLISH_MAXEVAL
const N_REFINE_EVAL = GRID_SEARCH == GRID_FULL ? 0 : REFINE_MAXEVAL
proj_(evals) = project_runtime(; t_eval = T_EVAL, n_pretest = N_SOBOL_EVAL, pretest_workers = USE_PMAP ? NW : 1,
                                evals, local_workers = NL, bootstrap = BOOTSTRAP, polish_evals = N_POLISH_EVAL,
                                refine_evals = N_REFINE_EVAL, n_procs = nprocs())
const PROJ_CAP = proj_(restart_evals(N_RESTART, LOCAL_MAXEVAL))
const PROJ_EMP = isempty(RUNTIME_EVALS) ? nothing : proj_(restart_evals(N_RESTART, LOCAL_MAXEVAL; observed = RUNTIME_EVALS))
let e = PROJ_EMP, emp(f) = e === nothing ? "--" : fmt_duration(f(e)),
    row(label, f, note) = sayf("  %-13s %11s %11s   %s\n", label, fmt_duration(f(PROJ_CAP)), emp(f), note),
    startup = LOCAL_MODE !== :async ? "serial, on the master" :
              BOOTSTRAP === :first_alone ? "restart 1 alone, then $NL at a time" : "$NL at a time from the start",
    t_restart = (LOCAL_MAXEVAL + 1) * T_EVAL
    sayf("\nprojected runtime -- scenarios, not a deadline (a driver's timeout is the enforced limit)\n")
    sayf("  one evaluation took %.1f s (the first, on the master: it includes compilation, so later ones are often faster)\n", T_EVAL)
    sayf("  %-13s %11s %11s\n", "stage", "cap", "empirical")
    row("pre-testing", p -> p.pretest, @sprintf("%d evals on %d worker(s)%s", N_SOBOL_EVAL, USE_PMAP ? NW : 1,
                                                 N_VALID_TARGET > 0 ? ", at least (valid-draw target)" : ""))
    row("local stage", p -> p.local_, @sprintf("%d restarts, %s; at the cap %.0f round(s) of %s", N_RESTART, startup,
                                                PROJ_CAP.local_ / t_restart, fmt_duration(t_restart)))
    row("polish", p -> p.polish, SKIP_POLISH ? "skipped" : "$N_POLISH_EVAL evals on 1 worker")
    N_REFINE_EVAL > 0 && row("refinement", p -> p.refine, "$N_REFINE_EVAL evals at the report grid (each costlier than shown)")
    row("total", p -> p.wall, "")
    sayf("  %-13s %11s %11s   %s\n", "core-hours", @sprintf("%.0f/%.0f", PROJ_CAP.busy / 3600, PROJ_CAP.reserved / 3600),
         e === nothing ? "--" : @sprintf("%.0f/%.0f", e.busy / 3600, e.reserved / 3600),
         "busy / held: an idle worker holds a core without loading the machine")
    sayf("  cap: every restart uses its %d evaluations (+1 for its start) and the polish its %d\n", LOCAL_MAXEVAL, N_POLISH_EVAL)
    if e === nothing
        say("  empirical: none given (--runtime-evals with the per-restart counts of a compatible run, e.g. mu08: 500,500,500,423,268)")
    else
        ncut = count(>(LOCAL_MAXEVAL + 1), RUNTIME_EVALS)
        sayf("  empirical: --runtime-evals %s (mean %.0f), reused in order%s; the polish at its cap\n",
             join(RUNTIME_EVALS, ","), mean(RUNTIME_EVALS), ncut > 0 ? ", $ncut cut at this cap" : "")
    end
    isempty(REUSE_PRETEST) || say("  (--reuse-pretest: cached pre-testing values cost nothing; shown as if evaluated)")
    RESUMING && say("  (resuming: part of this is already done)")
    adv = LOCAL_MODE === :async ? round_advice(N_RESTART, NL, BOOTSTRAP) : nothing
    adv === nothing || sayf("  note: the last round of restarts uses %d of %d local workers; --restarts %d fills it " *
                            "(same wall time)%s\n", adv.last, NL, adv.more,
                            adv.fewer > 0 ? ", --restarts $(adv.fewer) saves a round" : "")
end
LOCAL_MODE === :async || NW <= 1 ||
    say("\n--local-mode serial: the restarts run one after another on the master while the workers idle.")


banner("Incumbent calibration")
sayf("Q = %.6f\n", q0)
RESUMING ? say("(resuming: fit table skipped, see the original run's log)") : say_report(x0)

if REPORT_ONLY
    banner("--report-only: stopping before the search")
    write_run_record(; status = "report_only")   # it finished what it was asked to do
    sayf("wrote %s\n", short(joinpath(RUN_DIR, "run_record.toml")))
    sayf("wrote %s\n", short(joinpath(RUN_DIR, "run.log")))
    close(LOG); exit(0)
end

# -----------------------------------------------------------------------------
# Estimate
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------
# THE AUTHORITATIVE CHECKPOINT IS tiktak_state.toml (2026-09-27, tiktak_fix_plan.md step 4),
# written by the TikTak module right after seed selection, after every committed restart, at
# a pause and after the polish: versioned, checksummed, with the previous generation kept as
# .prev. It holds the seeds and K, the incumbent with its origin and verification, every
# committed restart (start, endpoint, evaluations, return code) and the lifetime counters.
#
# checkpoint.toml, seeds.toml and restarts.csv are DERIVED from it after every write
# (write_derived), so a crash between two writes can leave them one step behind but never
# inconsistent: restarts.csv is regenerated in full and cannot duplicate or omit a restart.
# checkpoint.toml keeps its [parameters] table for --init-from; --resume reads only
# tiktak_state.toml.
const CKPT = joinpath(RUN_DIR, "checkpoint.toml")
const SEEDS_F = joinpath(RUN_DIR, "seeds.toml")
const RESTARTS_F = joinpath(RUN_DIR, "restarts.csv")

"Write `text` to `path` atomically (tmp file + rename)."
function atomic_write(path::AbstractString, text::AbstractString)
    tmp = path * ".tmp"
    write(tmp, text)
    mv(tmp, path; force = true)
    return nothing
end

"""
    save_seeds!(seeds, f_sobol_best)

The pre-testing survivors, for reading (the resume takes them from tiktak_state.toml).
"""
function save_seeds!(seeds::Vector{Vector{Float64}}, f_sobol_best::Float64)
    # The EFFECTIVE restart count is the number of seeds (the valid pool may have reduced
    # it); progress lines use it, not the requested --restarts (finding 4).
    TRACKER.nrestart = length(seeds)
    io = IOBuffer()
    println(io, "# Pre-testing survivors. DERIVED from tiktak_state.toml (the authoritative checkpoint).")
    println(io, "f_sobol_best = ", f_sobol_best)
    println(io, "nstar        = ", length(seeds), "   # effective restarts (the schedule denominator)")
    println(io, "nstar_requested = ", N_RESTART)
    println(io, "seeds = [")
    for sd in seeds
        println(io, "  [", join((@sprintf("%.17g", v) for v in sd), ", "), "],")
    end
    println(io, "]")
    atomic_write(SEEDS_F, String(take!(io)))
    return nothing
end

"""
    checkpoint!(j, best, best_x; stage, grid)

A readable summary of the current best point (DERIVED; not what --resume reads).

`stage` and `grid` are NOT decoration. After the full-grid refinement this is called with
an objective computed at GRID_FULL, while the local stage's calls carry GRID_SEARCH
values. Recording only GRID_SEARCH -- as the first version did -- labelled a grid-30
objective as grid-20, which is exactly the confusion the separate Q_final / Q_search
fields in estimates.toml exist to prevent.
"""
function checkpoint!(j::Int, best::Float64, best_x::Vector{Float64};
                     stage::String = "local", grid::Int = GRID_SEARCH)
    est = unpack(best_x)
    io = IOBuffer()
    println(io, "# A summary of the current best point, DERIVED from tiktak_state.toml by code/smm/run_smm.jl.")
    println(io, "# --resume <this dir> reads tiktak_state.toml; --init-from uses [search_vector] exactly ([parameters] is rounded).")
    println(io, "stage         = \"", stage, "\"")
    println(io, "restarts_done = ", j)
    for kv in objective_identity()
        TOML.print(io, Dict{String,Any}(kv))
    end
    println(io, "restarts_total= ", N_RESTART)
    println(io, "Q_best        = ", best)
    println(io, "objective_grid= ", grid, "   # the grid Q_best was computed at")
    println(io, "Q_incumbent   = ", q0, "   # at grid_search")
    println(io, "minutes       = ", round(elapsed(), digits = 1))
    println(io, "updated       = \"", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\"")
    println(io, "\n[search_vector]")
    println(io, "z = [", join((@sprintf("%.17g", v) for v in best_x), ", "), "]")
    println(io, "\n[parameters]")
    for q in SMM_PARAMS
        @printf(io, "%-10s = %.8f\n", q.name, getfield(est, q.name))
    end
    atomic_write(CKPT, String(take!(io)))
    return nothing
end

"""
    write_derived(st)

Regenerate restarts.csv, seeds.toml and checkpoint.toml from the TikTak run state. Called by
the module right after each authoritative checkpoint write; reads the state, never changes it.
"""
function write_derived(st)
    io = IOBuffer()
    println(io, "restart,theta,f_start,f_local,improved,ret,attempt,n_eval,action,worker,elapsed_s,incumbent_version,commit_seq")
    for r in sort(st.records; by = x -> x.j)
        println(io, r.j, ",", r.theta, ",", r.f_start, ",", r.f_local, ",", r.action === :improved, ",", r.ret, ",",
                r.attempt, ",", r.n_eval, ",", r.action, ",", r.worker, ",", round(r.elapsed, digits = 2), ",",
                r.incumbent_version, ",", r.commit_seq)
    end
    atomic_write(RESTARTS_F, String(take!(io)))
    isfile(SEEDS_F) || save_seeds!(st.seeds, st.f_sobol_best)
    checkpoint!(length(st.records), st.inc.f, st.inc.x; stage = String(st.stage), grid = GRID_SEARCH)
    return nothing
end

banner("TikTak search")
lo, hi = LO_S, HI_S
t_start = time()
elapsed() = (time() - t_start) / 60

TRACKER.trun = time()
# On a resume there is no Sobol stage, so the tracker must START in :local -- the stage
# transition normally happens in the on_sobol callback, which never fires. Without this
# the restarts were reported under the Sobol format ("sobol 31/13  238%").
if RESUMING && RESUME_MODE !== :pretest
    stage!(:local)
    TRACKER.restart = RESUME_MODE === :state ? RESUME_INFO.next_j : RESUME_STATE.j_start
    # SEED THE TRACKER'S BEST WITH THE RESUMED INCUMBENT. Without this it starts at Inf,
    # so the first evaluation after a resume sets it and the progress line reports a "best
    # Q" WORSE than the point the run is actually carrying -- e.g. "best Q 0.8317" while
    # the checkpoint held 0.5351. The search itself was unaffected (tiktak tracks its own
    # incumbent) but the log said something untrue, which on a two-day run is how a
    # perfectly good resume gets killed and restarted by hand.
    TRACKER.best = RESUME_MODE === :state ? RESUME_INFO.fZ : RESUME_STATE.fZ
elseif N_VALID_TARGET > 0
    # the progress line counts every valid value it receives, the supplied start's too, so the
    # target it shows is the module's (valid SOBOL draws) plus the start when the start is valid
    let start_valid = isfinite(q0) && q0 < SMM_PENALTY
        stage!(:sobol, N_SOBOL + 1; target = N_VALID_TARGET + (start_valid ? 1 : 0))
        sayf("pre-testing: until %d Sobol' draws are valid (at most %d attempts); the progress line counts the supplied start too (%s)%s\n",
             N_VALID_TARGET, N_SOBOL, start_valid ? "valid" : "penalised",
             RESUMING ? "; values evaluated before this resume are not in its counts" : "")
    end
else
    stage!(:sobol, N_SOBOL_EVAL)
end
# a PAUSE file left by the segment that paused would pause this one at once: retire it
if RESUMING && isfile(PAUSE_F)
    mv(PAUSE_F, PAUSE_F * ".consumed_" * STAMP; force = true)
    say("(retired the PAUSE file of the previous segment: ", short(PAUSE_F * ".consumed_" * STAMP), ")")
end
const TICK_FROM_SOBOL = !isempty(TIKTAK_KW.pretest_workers)   # the master is not evaluating

result = tiktak(objective_tracked, lo, hi; TIKTAK_KW...,
                resume = RESUME_ARG,
                on_progress = print_local_progress,
                on_seeds = save_seeds!,
                on_checkpoint = write_derived,
                # With a worker pool the module reports each pre-testing value as it arrives;
                # serially objective_tracked has already ticked. i == n is the end of the
                # Sobol stage, where the progress format changes over.
                on_sobol = function (i, n, fx, best)
                    if i < n
                        TICK_FROM_SOBOL && (TRACKER.done = i - 1; tick!(fx))
                        return
                    end
                    sayf("  sobol    complete: %d values, %d valid (the supplied start included), best Q %s, %.1f min\n",
                         TRACKER.done, TRACKER.valid, strip(qstr(best)), elapsed())
                    stage!(:local)
                end,
                on_local = function (j, ns, th, fl, best, best_x, row)
                    # restarts.csv and checkpoint.toml are regenerated by write_derived
                    # after the module's checkpoint write; this only reports progress.
                    # row.ret is why the search stopped: FTOL_REACHED / XTOL_REACHED (converged by
                    # the tolerance) or MAXEVAL_REACHED (the --local-evals budget ran out).
                    # The ETA is only meaningful serially: asynchronously restart 1 runs alone.
                    eta = LOCAL_MODE === :async ? "" :
                          @sprintf(", ~%.0f min left", elapsed() / max(j, 1) * (ns - j))
                    sayf("  restart %3d/%-3d DONE  %-16s start Q %s   end Q %s   incumbent %s   %5.1f min%s\n",
                         j, ns, row.ret, qstr(row.f_start), qstr(fl), qstr(best), elapsed(), eta)
                    TRACKER.restart = min(j + 1, ns)
                    TRACKER.done = 0                 # eval counter restarts with the search
                    TRACKER.best = min(TRACKER.best, best)
                end)

if result.status === :paused
    banner(@sprintf("PAUSED after %d of %d restarts (--stop-after-restarts %d) -- %.1f min",
                    length(result.records), result.nstar_effective, STOP_AFTER, elapsed()))
    sayf("incumbent Q %.6g (search grid %d); the polish, the refinement and the report are NOT run.\n",
         result.f, GRID_SEARCH)
    sayf("continue with the same command plus:  --resume %s\n", short(RUN_DIR))
    say("(drop --stop-after-restarts, or raise it, to go further; the schedule denominator stays",
        " $(result.nstar_effective))")
    write_run_record(; status = "paused")
    sayf("wrote %s\n", short(STATE_F))
    close(LOG); exit(0)
end
banner(@sprintf("Finished in %.1f min -- %d evaluations", elapsed(), result.n_eval))
sayf("Q: sobol-best %.6g  ->  pre-polish %.6g  ->  final %.6g\n",
     result.f_sobol_best, result.f_prepolish, result.f)
let ps = result.pretest
    ps.stop_reason === :resumed ||
        sayf("pre-testing: %d Sobol draws in the pool -- %d valid, %d penalised, %d non-finite, %d discarded; %d supplied (%d valid)\n" *
             "             stop: %s%s; %d values reused from a cache, %d draws evaluated past the pool\n",
             ps.attempted, ps.valid, ps.invalid, ps.nonfinite, ps.errors, ps.supplied, ps.supplied_valid,
             ps.stop_reason, ps.target > 0 ? " (target $(ps.target) valid: $(ps.target_reached ? "reached" : "NOT reached"))" : "",
             ps.reused, ps.overshoot)
    sayf("restarts: %d requested, %d effective (the schedule denominator)%s\n",
         result.nstar_requested, result.nstar_effective,
         result.nstar_effective < result.nstar_requested ? "  -- REDUCED: too few valid pre-testing values" : "")
end
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
if result.polish_ret === :SKIPPED
    say("polish: SKIPPED by --skip-polish (0 evaluations); the pre-polish point is final")
else
    sayf("polish: ret %s (%s), %d evaluations, %s\n", result.polish_ret,
         ret_class(result.polish_ret), result.n_eval_polish,
         result.polish_improved ? "improved the incumbent" : "did not improve the incumbent")
end
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
# 2026-09-27 (finding 2): the solve is TikTak.refine, which merges exactly as a restart
# does. The reported point carries its OWN evidence on the full-grid objective: a
# refinement that improved but stopped on MAXEVAL_REACHED is budget-limited, and the
# coarse winner's certificate never transfers to the full grid.
const REFINE_SETTINGS = TikTak.SolverSettings(:LN_BOBYQA, 1e-6, 1e-10, 1e-6, max(REFINE_MAXEVAL, 1))

function refine_at_full_grid(z_search::Vector{Float64}, lo, hi)
    banner(@sprintf("Full-grid refinement (grid %d -> %d)", GRID_SEARCH, GRID_FULL))
    t_ref = time()
    out = TikTak.refine(objective_full, z_search, lo, hi; settings = REFINE_SETTINGS,
                        cfg = result.config, objective_id = OBJ_ID_REPORT)
    sayf("Q at the coarse winner, re-evaluated at grid %d: %.6g\n", GRID_FULL, out.f_start)
    if out.status === :failed
        @warn "full-grid refinement failed; keeping the coarse winner" exception = out.error
        sayf("refinement FAILED after %d evaluations: %s\n", out.evals, out.error)
        return out
    end
    sayf("refined: %d evaluations, %.1f min, ret %s (%s)\n",
         out.evals, (time()-t_ref)/60, out.ret, ret_class(out.ret))
    if out.status === :improved
        sayf("Q(grid %d): %.6g at the coarse winner  ->  %.6g refined  (%.2f%% better)\n",
             GRID_FULL, out.f_start, out.incumbent.f,
             100*(out.f_start - out.incumbent.f)/max(abs(out.f_start), eps()))
    else
        sayf("Q(grid %d): %.6g at the coarse winner; refinement did not improve on it%s\n",
             GRID_FULL, out.f_start,
             out.incumbent.verification.status === :verified ? " (it returned that point: verified)" : "")
    end
    return out
end

@everywhere objective_full(z) = smm_objective(z, TARGETS;
                                              Na = G_FULL_.Na, Nk = G_FULL_.Nk,
                                              Nhc = G_FULL_.Nhc, simN = G_FULL_.simN,
                                              seed = $SEED_,
                                              child_grid = CHILD_G_, demo_sim = false,
                                              child_extra = $CE_RUN, parent_extra = $PE_RUN)
const REFINE = if GRID_SEARCH != GRID_FULL
    refine_at_full_grid(Z_SEARCH, lo, hi)
else
    say("\nsearch ran at the full grid -- no refinement stage needed")
    TikTak.refine_skipped(result.incumbent)
end
const Z_FINAL = copy(REFINE.incumbent.x)
const Q_FINAL = REFINE.incumbent.f

# THE FINAL WINNER IS ALWAYS CHECKPOINTED, refinement or not.
#
# This used to be guarded by `GRID_SEARCH != GRID_FULL`, and both grids default to 30 --
# so on a DEFAULT run the last checkpoint written was the pre-polish incumbent after the
# final restart, and the polish's improvement existed only in estimates.toml. If the report
# stage then died, the best point on disk was not the best point found.
checkpoint!(length(result.records), Q_FINAL, Z_FINAL;
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
    say(" scored SMM_PENALTY = $(SMM_PENALTY) rather than crashed on. A high rate means the SEARCH BOX is")
    say(" too wide, not that the model is wrong.)")
    # the exception sites behind the scored model failures (ported from v2, 2026-09-27): the
    # broad classes (DomainError, InexactError) are scored only from the solver's own files,
    # and every site is listed here so the classification can be checked, not trusted
    let sites = Dict{String,Int}()
        for w in procs()
            d = w == myid() ? SMM_FAILURE_SITES : remotecall_fetch(() -> copy(Main.SMM_FAILURE_SITES), w)
            for (k, v) in d; sites[k] = get(sites, k, 0) + v; end
        end
        isempty(sites) || say("exception sites scored as model failures (file:function:line):")
        for (k, v) in sort(collect(sites); by = last, rev = true); sayf("  %-60s %6d\n", k, v); end
    end
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

# ---- did anything land on a box edge? ---------------------------------------
# The 2% search-coordinate threshold is a conservative review flag, not a test
# of optimizer convergence or proof that the target is unattainable. A bounded
# optimizer can converge at an edge. Keep this flag in acceptance until boundary
# profiles and numerical sensitivity have been reviewed; do not silently relax it.
# The original evidence in saved run files is retained unchanged.
const PINNED = let (lo_s, hi_s) = search_bounds(), est_ = unpack(Z_FINAL)
    out = NamedTuple[]
    for (i, q) in enumerate(SMM_PARAMS)
        pos = (Z_FINAL[i] - lo_s[i]) / (hi_s[i] - lo_s[i])
        (pos < 0.02 || pos > 0.98) &&
            push!(out, (name = q.name, value = getfield(est_, q.name),
                        which = pos < 0.02 ? "LOWER" : "UPPER",
                        pos = pos, lo = q.lo, hi = q.hi))
    end
    out
end
# NEAR a bound but not on it. Reported, NOT gated on -- the acceptance test stays exactly
# where it is. lambda_2 came back at 97.09% of its box in the 2026-09-07 candidate: inside
# the 98% line, so silent, while being one short step from a wall. A parameter drifting
# toward its box across successive runs is worth seeing BEFORE it pins, and that is a
# different thing from failing acceptance.
const NEAR_BOUND = let (lo_s, hi_s) = search_bounds(), est_ = unpack(Z_FINAL)
    out = NamedTuple[]
    for (i, q) in enumerate(SMM_PARAMS)
        pos = (Z_FINAL[i] - lo_s[i]) / (hi_s[i] - lo_s[i])
        (0.02 <= pos < 0.05 || 0.95 < pos <= 0.98) &&
            push!(out, (name = q.name, value = getfield(est_, q.name),
                        which = pos < 0.5 ? "lower" : "upper", pos = pos))
    end
    out
end
if !isempty(NEAR_BOUND)
    say("\n!  APPROACHING A BOUND (within 5%) -- not a failure, but watch it across runs:")
    for p_ in NEAR_BOUND
        sayf("  %-10s = %10.4f  %.1f%% of its box, near the %s bound\n",
             p_.name, p_.value, 100 * p_.pos, p_.which)
    end
end
if isempty(PINNED)
    say("\nall parameters interior to their boxes")
else
    say("\n!! PARAMETERS NEAR A SEARCH BOUND -- boundary review required:")
    for p_ in PINNED
        sayf("  %-10s = %10.4f  near its %s bound [%.3f, %.3f]  (%.1f%% into the box)\n",
             p_.name, p_.value, p_.which, p_.lo, p_.hi, 100 * p_.pos)
    end
    say("This proximity flag is separate from the optimizer stopping status. Check a")
    say("boundary profile and numerical sensitivity before promoting the result; proximity")
    say("alone does not prove that the box is too narrow or the target is unattainable.")
end

# A2. ACCEPTANCE IS A STATEMENT ABOUT THE REPORTED POINT, NOT ABOUT THE POPULATION.
#
# The previous test was `N_CONVERGED > 0` -- "some restart converged". That says nothing
# about the point actually returned: the winner can come from a restart that exhausted
# maxeval while a different restart converged to something worse, and the run would still
# report `accepted = true`. The reported point's own evidence decides it.
#
# 2026-09-27 (tiktak_fix_plan.md step 3): WHERE the point came from and WHETHER a later
# solve verified it are kept apart (TikTak.CandidateOrigin / Verification), and the
# evidence must be on the REPORTING objective:
#   * no refinement (search grid = report grid): the search's own evidence -- the search
#     that produced the point converged, or a later converged solve (a restart, the polish)
#     returned that same point. An unchanged optimum is no longer reported as unconverged.
#   * refinement: the refinement's evidence on the full grid only. A refinement that
#     improved but stopped on MAXEVAL_REACHED is budget-limited (it used to pass); a
#     converged refinement certifies the point even after a budget-limited coarse search.
# The rule is TikTak.acceptance, which tools/test_tiktak.jl exercises directly.
const EVIDENCE = REFINE.incumbent
const WINNER_CONVERGED = ret_class(result.winner_ret) === :converged   # the search winner's own code (context)
const REFINE_OK = REFINE.status !== :failed && (REFINE.status === :skipped || ret_class(REFINE.ret) !== :other)
const ACCEPT = TikTak.acceptance(;
    execution_ok = result.n_exception == 0 && REFINE.status !== :failed,
    candidate_valid = TikTak.candidate_valid(Z_FINAL, Q_FINAL, lo, hi, SMM_PENALTY) && N_INVALID_FINAL == 0,
    local_converged = TikTak.local_converged(EVIDENCE, OBJ_ID_REPORT),
    # VALIDATED from the state (2026-09-28, finding 11), not read from the status symbol: every
    # planned restart committed once, nothing in flight, no violated invariant
    search_budget_complete = TikTak.search_budget_complete(result),
    pinned = String[String(p_.name) for p_ in PINNED])
const ACCEPTED = ACCEPT.accepted

say("\nacceptance -- about the REPORTED POINT, on the reporting objective (grid $GRID_FULL)")
let o = EVIDENCE.origin, v = EVIDENCE.verification
    sayf("  point came from         %s%s  (its search: %s, %s)\n", o.stage,
         o.stage === :local ? " restart $(o.restart)" : "", o.ret, ret_class(o.ret))
    sayf("  verified afterwards     %s\n", v.status === :verified ?
         @sprintf("yes -- %s returned it (%s, distance %.1e)", v.stage, v.ret, v.distance) : "no")
    sayf("  refinement              %s (ret %s, %d evals)\n", REFINE.status, REFINE.ret, REFINE.evals)
end
sayf("  execution ok            %s  (%d objective exceptions%s)\n", ACCEPT.execution_ok ? "yes" : "NO ",
     result.n_exception, REFINE.status === :failed ? "; refinement FAILED" : "")
sayf("  candidate valid         %s  (Q = %.6g, %d invalid cells in the final simulation)\n",
     ACCEPT.candidate_valid ? "yes" : "NO ", Q_FINAL, N_INVALID_FINAL)
sayf("  locally converged       %s  (evidence on the reporting objective)\n", ACCEPT.local_converged ? "yes" : "NO ")
sayf("  search budget complete  %s  (%s; %d of %d restarts committed, %d in flight; resume: %s)\n",
     ACCEPT.search_budget_complete ? "yes" : "NO ", result.status, length(result.trace), result.nstar_effective,
     length(result.inflight), result.resume_semantics)
for v_ in result.violations
    sayf("    state invariant violated: %s\n", v_)
end
sayf("  no parameter on a bound %s  (%s)\n", ACCEPT.interior ? "yes" : "NO ",
     isempty(PINNED) ? "all interior" : join((String(p_.name) for p_ in PINNED), ", "))
sayf("  (for context: %d of %d restarts converged, %d hit a budget, %d other)\n",
     N_CONVERGED, length(result.trace), N_LIMIT, N_OTHER)
if ACCEPTED
    say("  ACCEPTED -- this point may be quoted, with the caveats in docs/SMM.md")
else
    say("  NOT ACCEPTED -- do not quote this point:")
    for r_ in ACCEPT.reasons; say("    - ", r_); end
    say("  a finite Q is not a certification, and neither is another restart's convergence.")
end
# EXECUTION IS NOT ESTIMATION (plan step 8): a smoke run passes on clean execution alone.
const VERDICT = TikTak.execution_verdict(PRESET.name, ACCEPT)
say("\nverdict: ", VERDICT)

# -----------------------------------------------------------------------------
# Persist
# -----------------------------------------------------------------------------
est = unpack(Z_FINAL)
open(joinpath(RUN_DIR, "estimates.toml"), "w") do io
    println(io, "# SMM: ", length(SMM_MOMENTS), " moments (",
                 length(SMM_P_MOMENTS), " P + ", length(SMM_S_MOMENTS), " S + ",
                 length(SMM_T_MOMENTS), " T + ", length(SMM_W_MOMENTS),
                 " W), ", length(SMM_PARAMS), " parameters (",
                 length(SMM_PARENT_PARAMS), " parent + ", length(SMM_CHILD_PARAMS),
                 " child). GENERATED by code/smm/run_smm.jl.")
    println(io, "spec_version = \"", SPEC_VERSION, "\"")
    println(io, "source_sha   = \"", SOURCE_SHA, "\"   # ", join(SOURCE_FILES, " + "))
    println(io, "m_psychic    = ", target_m_psychic(TARGETS),
                 "   # kappa_0 is the psychic cost AT THIS log-ability, not at log theta = 0")
    println(io, "child_grid   = \"", CHILD_G_.Na, "x", CHILD_G_.Nk, "x", CHILD_G_.Nt, "\"")
    println(io, "child_params = [", join(("\"$n\"" for n in SMM_CHILD_PARAMS), ", "), "]")
    println(io, "parent_params = [", join(("\"$n\"" for n in SMM_PARENT_PARAMS), ", "), "]")
    println(io, "sigma_eta_fixed = ", PARENT_DEFAULTS.sigma_eta, "   # NOT estimated: DFVW's technology is deterministic")
    println(io, "mu_half      = ", target_mu_half(TARGETS), "   # the child's weight at the half period (target file)")
    println(io, "grid_extra   = \"", GRID_EXTRA, "\"")
    println(io, "init_from    = \"", INIT_FROM, "\"")
    println(io, "skip_polish  = ", SKIP_POLISH)
    println(io, "weighting    = \"diagonal inverse-variance on the joint clustered covariance\"")
    println(io, "generated  = \"", Dates.format(now(), "yyyy-mm-dd HH:MM"), "\"")
    println(io, "git_commit = \"", git_sha(), "\"")
    println(io, "targets    = \"", short(TARGETS_FILE), "\"")
    println(io, "quick      = ", QUICK)
    println(io, "n_sobol    = ", N_SOBOL, "   # attempted Sobol draws")
    println(io, "n_restarts = ", N_RESTART, "   # requested")
    println(io, "n_restarts_effective = ", result.nstar_effective, "   # planned after pre-testing (the schedule denominator)")
    println(io, "n_sobol_valid = ", result.pretest.valid, "   # of the attempted draws; supplied points counted apart")
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
    # A2. WHERE THE SEARCH WINNER CAME FROM, beside what it is worth (on the search grid).
    println(io, "winner_stage  = \"", result.winner_stage, "\"   # sobol|supplied|local|polish|legacy_import")
    println(io, "winner_restart= ", result.winner_j, "   # 0 unless winner_stage = local")
    println(io, "winner_ret    = \"", result.winner_ret, "\"   # the return code of THAT search")
    println(io, "winner_converged = ", WINNER_CONVERGED, "   # that search's own code; context, not the acceptance input")
    println(io, "refine_ok     = ", REFINE_OK, "   # the refinement did not throw; context, not the acceptance input")
    # THE REPORTED POINT'S EVIDENCE ON THE REPORTING OBJECTIVE (2026-09-27, plan step 3).
    let o = EVIDENCE.origin, v = EVIDENCE.verification
        println(io, "point_origin  = \"", o.stage, "\"   # the stage that supplied the reported point")
        println(io, "point_origin_restart = ", o.restart)
        println(io, "point_origin_ret = \"", o.ret, "\"")
        println(io, "verification  = \"", v.status, "\"   # verified = a later converged solve returned this point")
        println(io, "verification_stage = \"", v.stage, "\"")
        println(io, "verification_ret = \"", v.ret, "\"")
        println(io, "objective_id_search = \"", OBJ_ID_SEARCH, "\"")
        println(io, "objective_id_report = \"", OBJ_ID_REPORT, "\"   # the evidence refers to this objective")
    end
    println(io, "run_status    = \"", result.status, "\"   # complete|stopped_early")
    println(io, "resume_semantics = \"", result.resume_semantics,
            "\"   # fresh|serial_exact|async_continuation|changed_optimizer|legacy_import|already_complete")
    println(io, "restarts_in_flight = [", join(result.inflight, ", "), "]   # must be empty for a complete search")
    println(io, "state_violations = [", join(("\"" * replace(v_, "\"" => "'") * "\"" for v_ in result.violations), ", "), "]")
    println(io, "execution_ok  = ", ACCEPT.execution_ok)
    println(io, "candidate_valid = ", ACCEPT.candidate_valid)
    println(io, "local_converged = ", ACCEPT.local_converged, "   # on the reporting objective")
    println(io, "search_budget_complete = ", ACCEPT.search_budget_complete)
    println(io, "acceptance_reasons = [", join(("\"" * replace(r_, "\"" => "'") * "\"" for r_ in ACCEPT.reasons), ", "), "]")
    println(io, "preset        = \"", PRESET.name, "\"")
    println(io, "verdict       = \"", replace(VERDICT, "\"" => "'"), "\"   # execution vs estimation, by purpose")
    println(io, "polish_improved = ", result.polish_improved)
    println(io, "n_converged   = ", N_CONVERGED, "   # local searches that met a stopping test")
    println(io, "n_hit_budget  = ", N_LIMIT, "   # stopped on maxeval/maxtime, NOT converged")
    println(io, "n_ret_other   = ", N_OTHER)
    println(io, "n_exception   = ", result.n_exception, "   # >0 means a BUG in the objective")
    println(io, "n_invalid_final = ", N_INVALID_FINAL, "   # off-domain cells at the final point")
    println(io, "params_on_bound = [",
            join(("\"$(p_.name)\"" for p_ in PINNED), ", "),
            "]   # within 2% of a search-box edge; requires boundary review")
    println(io, "params_near_bound = [",
            join(("\"$(p_.name)\"" for p_ in NEAR_BOUND), ", "),
            "]   # within 5% of an edge; reported only, does NOT affect acceptance")
    println(io, "accepted      = ", ACCEPTED,
            "   # all of execution_ok, candidate_valid, local_converged, search_budget_complete, no bound")
    print(io, "ret_tally  = {")
    print(io, join(("$(k) = $(v)" for (k, v) in sort(collect(RET_TALLY); by = first)), ", "))
    println(io, "}")
    # THE EXACT POINT (2026-09-29, finding 18): [parameters] below is rounded to 8 decimals for
    # reading; --init-from uses this vector bit for bit when names and links match.
    println(io, "param_names   = [", join(("\"$(q.name)\"" for q in SMM_PARAMS), ", "), "]")
    println(io, "param_link    = [", join(("\"$(q.link)\"" for q in SMM_PARAMS), ", "), "]")
    println(io, "\n[search_vector]   # the final point in search coordinates, full precision")
    println(io, "z = [", join((@sprintf("%.17g", v) for v in Z_FINAL), ", "), "]")
    println(io, "\n[parameters]")
    for q in SMM_PARAMS
        @printf(io, "%-14s = %.8f   # %s block; was %.8f\n", q.name, getfield(est, q.name),
                q.owner, param_default(q.name))
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
write_run_record(result, Q_FINAL, Q_SEARCH, REFINE, ACCEPTED)

say("")
sayf("wrote %s\n", short(joinpath(RUN_DIR, "run_record.toml")))
sayf("wrote %s\n", short(RESTARTS_F))
sayf("wrote %s\n", short(joinpath(RUN_DIR, "estimates.toml")))
sayf("wrote %s\n", short(joinpath(RUN_DIR, "run.log")))
close(LOG)
