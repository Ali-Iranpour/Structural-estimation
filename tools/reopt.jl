#!/usr/bin/env julia
# =============================================================================
# reopt.jl -- bounded re-optimisation from given points (ported on 2026-10-02 from apps/Structural-estimation-v2,
# where it is tools/reopt_v5e.jl, written for the v5e audit of 2026-09-20; the logic is unchanged, the defaults
# that named v2's files are gone: --targets and --start are required here)
#
#   julia --project=. tools/reopt.jl --outdir <dir> --targets <file> --start <toml> [--label <name>]
#        [--fix omega=0.45,mu=0.95,kappa_terminal=35,R_1=0]
#        [--start <checkpoint|estimates toml>[,<toml>...]]
#        [--local-evals 300] [--sobol 0] [--restarts 2] [--polish-evals 0]
#        [--procs 1] [--extra-moments mean_c_p --extra-targets <targets.toml>]
#        [--seed 1234] [--simN 2000] [--grid 30]
#        [--bounds mu=0.2:0.8,kappa_terminal=0.5:200]
#
# --bounds RUN-SPECIFIC boxes (Part B, 2026-09-21; tools/run_bounds.jl): the named entries of
#          SMM_PARAMS are replaced in this process (and its workers) before anything reads the
#          box, so the optimiser's bounds, --fix's admissibility, unpack's clamp and the
#          near-bound report all use the run's box. Recorded in results.toml / best_estimates.toml
#          ([run_bounds]). The source constants and production defaults are untouched.
# --start  is VALIDATED, never clamped (2026-09-22, tools/start_loader.jl): a non-finite value or a
#          free coordinate outside the box in force by more than 1e-9 of its width refuses the run
#          before any evaluation; a start recorded under another box is a documented WARM START.
# --resume additionally verifies the checkpoint's [config] (bounds, fixed, targets, extra rows,
#          seed, simN, grid, overrides, budget, init step, ftol, source hash) against the run.
#
# --fix    names in SMM_PARAMS leave the free vector and are held at the value (in their
#          search coordinates, so `unpack` returns exactly the value); `omega`, which is not
#          estimated, is passed through `child_extra`. Nothing is promoted: SMM_PARAMS and
#          the constructor defaults are untouched, and the run's own record says what was
#          fixed. omega is not estimated: it is only ever fixed here -- EXCEPT under --free (below).
# --free   omega=LO:HI (the mu = 0.7 plan, Ali 2026-09-24): omega becomes ONE extra search coordinate
#          (level link, box [LO, HI]) appended after the free SMM_PARAMS, passed per evaluation through
#          child_extra (part of both child cache keys). Allowed only for omega and only while mu is NOT searched
#          (in this repository mu is never in SMM_PARAMS: it is fixed in CHILD_DEFAULTS; v2 requires --fix mu=...)
#          -- mu and omega are not separately identified. SMM_PARAMS is untouched; every record
#          writes [parameters].omega and [free_extra.omega] {lo, hi}; a start must carry [parameters].omega.
# TikTak mode checkpoints (since 2026-09-27, tiktak_fix_plan.md steps 4-8): the TikTak module's versioned
#          state (tiktak_state.toml: seeds, K, incumbent with origin and verification, every restart,
#          counters) and its chunked pre-testing cache (pretest_cache.toml); --resume continues after the
#          module verifies the objective (the [config] objective fields) and the optimizer settings. With
#          --procs > 1 the pre-testing AND the restarts run on the worker processes (asynchronous restarts;
#          --local-procs N, 0 = min(workers, floor(sqrt(restarts)))). --stop-after-restarts M pauses after M.
#          --preset smoke|integration|pilot|production records the purpose and supplies default budgets.
# OBJECTIVE IDENTITY (2026-09-28, tiktak_fix_plan.md follow-up 2; tools/reopt_identity.jl): the targets and
#          extra-targets files BY CONTENT, the resolved extra rows (target, weight), the moment order, the free
#          and fixed parameters with their box and links, grid/simN/seed, the overrides, the SMM_* switches and the
#          model + adapter source (the objective code is tools/reopt_objective.jl). Checked against a checkpoint
#          or pre-testing cache BEFORE the child warm-up; any difference refuses the resume, naming the field.
#          A TikTak checkpoint or cache written before 2026-09-28 (the path-only identity), and any old-format
#          one (tiktak_seeds.toml, --legacy-import), is refused as a continuation: start a new --outdir with
#          --start <its best_estimates.toml> (re-evaluated; nothing else carries over).
# --sobol 0 (default) pure local re-optimisation: Nelder-Mead in the free box from every
#          start, at most --local-evals evaluations each; starts run in parallel over
#          --procs worker processes (one NLopt state per process, never threads).
# --sobol N a bounded TikTak pilot: N Sobol' points, --restarts local searches of
#          --local-evals each, BOBYQA polish only when --polish-evals > 0, every start
#          forced into the pre-testing pool; the Sobol' stage is spread over --procs.
# --extra-moments rows appended to the objective (experiment C): target mean from the
#          targets, weight 1/se^2 from --extra-targets's [moment_cov] block. Q is reported
#          split into the original rows (Q_original) and the extra rows (Q_extra).
#
# The objective is `smm_objective` -- the audited one, with its named penalties -- at the
# full grids, the given targets and common random numbers (seed 1234, simN 2000).
# Written: run.log, results.toml (per start: Q at the start, best Q, evaluations, NLopt
# return, parameters by name, free parameters within 2% of a bound), best_estimates.toml
# ([parameters] by name, for --init-from / --at), table.txt (comparison-table rows of the
# start and the best point, experiment_table.jl format).
# =============================================================================
using Distributed, Printf, Dates, LinearAlgebra, TOML

const REPO = normpath(joinpath(@__DIR__, ".."))

function argstr(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("$flag needs a value")
    return ARGS[i + 1]
end
argval(flag, default) = (v = argstr(flag, nothing); v === nothing ? default : parse(Int, v))
const KNOWN = ("--outdir", "--label", "--fix", "--start", "--local-evals", "--sobol", "--restarts",
               "--polish-evals", "--procs", "--extra-moments", "--extra-targets", "--targets",
               "--seed", "--simN", "--grid", "--parent-extra", "--child-extra", "--checkpoint-every",
               "--init-step", "--resume", "--ftol-rel", "--bounds", "--free", "--local-alg",
               "--preset", "--local-procs", "--stop-after-restarts", "--legacy-import", "--sobol-valid")
let unknown = [a for a in ARGS if startswith(a, "--") && !(a in KNOWN)]
    isempty(unknown) || error("unknown flag(s): " * join(unknown, " "))
end

const OUTDIR  = argstr("--outdir", ""); isempty(OUTDIR) && error("--outdir is required")
const LABEL   = argstr("--label", basename(OUTDIR))
const NPROC   = argval("--procs", 1)
const GRID    = argval("--grid", 30)
const SIM_N   = argval("--simN", 2000)
const SEED    = argval("--seed", 1234)
const N_SOBOL = argval("--sobol", 0)
const N_RESTART = argval("--restarts", 2)
const N_VALID_TARGET = argval("--sobol-valid", 0)      # TikTak mode: continue the Sobol' sequence to this many VALID draws
const LOCAL_PROCS = argval("--local-procs", 0)          # TikTak mode, --procs > 1: local workers (0 = automatic)
const STOP_AFTER = argval("--stop-after-restarts", typemax(Int))
const LEGACY_IMPORT = "--legacy-import" in ARGS
const PRESET_NAME = argstr("--preset", "custom")        # recorded; budgets stay as given by the flags here
const LOCAL_MAXEVAL  = argval("--local-evals", 300)
const POLISH_MAXEVAL = argval("--polish-evals", 0)
const TARGETS_FILE = let f = argstr("--targets", ""); isempty(f) && error("--targets is required"); abspath(f) end
const EXTRA_NAMES = [Symbol(s) for s in split(argstr("--extra-moments", ""), ',') if !isempty(s)]
const EXTRA_TFILE = argstr("--extra-targets", TARGETS_FILE)
const STARTS = [String(s) for s in split(argstr("--start", ""), ',') if !isempty(s)]
isempty(STARTS) && error("--start is required (a checkpoint, estimates or best_estimates toml; repeat it on --resume)")
# follow-up additions (2026-09-20 evening): numerical overrides for the chosen grid, checkpoints
# of the incumbent every --checkpoint-every evaluations (checkpoint_best.toml + progress.toml),
# restart from that checkpoint with --resume (a RESTART of Nelder-Mead at the saved best point
# with the remaining budget -- NLopt's simplex state is not serialisable, so this is not an
# exact optimizer-state resume, and results.toml says so), and the initial simplex step as a
# fraction of the box width (--init-step; 0 = NLopt's default heuristic).
function parse_nt(str)
    isempty(str) && return (;)
    ks = Symbol[]; vs = Any[]
    for kv in split(str, ',')
        k, v = split(kv, '='); push!(ks, Symbol(strip(k))); v = strip(v)
        # node counts (N*) are Ints, every other numeric setting is a Float64 (the constructors are typed)
        isN = startswith(String(ks[end]), "N")
        push!(vs, isN && tryparse(Int, v) !== nothing ? parse(Int, v) : (tryparse(Float64, v) !== nothing ? parse(Float64, v) : Symbol(v)))
    end
    NamedTuple{Tuple(ks)}(Tuple(vs))
end
const PE_ARG = parse_nt(argstr("--parent-extra", ""))
const CE_ARG = parse_nt(argstr("--child-extra", ""))
const CKPT_EVERY = argval("--checkpoint-every", 25)
const RESUME = "--resume" in ARGS
# SOLVER FLAGS (2026-09-28, follow-up 5; tools/reopt_settings.jl). Pure-local mode: --ftol-rel (default 1e-4),
# --init-step (default 0 = NLopt's heuristic) and --local-alg neldermead|bobyqa (2026-09-24: bobyqa for the
# polish of the TikTak winner). TikTak mode: --ftol-rel and --init-step are FORWARDED to its local stage when
# given (TikTak's own defaults otherwise, local_tol 1e-3); --local-alg is refused there. results.toml records
# the settings the optimizer actually ran with.
include(joinpath(REPO, "tools", "reopt_settings.jl"))
const FTOL_ARG = let s = argstr("--ftol-rel", nothing); s === nothing ? nothing : parse(Float64, s) end
const STEP_ARG = let s = argstr("--init-step", nothing); s === nothing ? nothing : parse(Float64, s) end
const ALG_ARG = argstr("--local-alg", nothing)
const LOCAL_SET = reopt_local_settings(; ftol_rel = FTOL_ARG, init_step = STEP_ARG, local_alg = N_SOBOL > 0 ? nothing : ALG_ARG)
const TK_SOLVER_KW = N_SOBOL > 0 ? reopt_tiktak_kw(; ftol_rel = FTOL_ARG, init_step = STEP_ARG, local_alg = ALG_ARG) : (;)
const INIT_STEP = LOCAL_SET.init_step        # the pure-local mode's settings (TikTak mode: see TK_SOLVER_KW)
const FTOL_REL = LOCAL_SET.ftol_rel
const LOCAL_ALG = LOCAL_SET.alg
const BOUNDS_STR = argstr("--bounds", "")
const FIXED = let d = Dict{Symbol,Float64}()
    for s in split(argstr("--fix", ""), ',')
        isempty(s) && continue
        k, v = split(s, '='); d[Symbol(strip(k))] = parse(Float64, strip(v))
    end
    d
end
# --free omega=LO:HI (see the header)
const FREE_EXTRA = let s = argstr("--free", "")
    if isempty(s)
        nothing
    else
        k, v = split(s, '='); strip(k) == "omega" || error("--free: only omega can be freed (got $k)")
        lo, hi = parse.(Float64, split(v, ':'))
        isfinite(lo) && isfinite(hi) && lo < hi || error("--free omega=$v: need finite LO < HI")
        (lo = lo, hi = hi)
    end
end
FREE_EXTRA === nothing || !haskey(FIXED, :omega) || error("--free omega and --fix omega together")

mkpath(OUTDIR)
const LOG = open(joinpath(OUTDIR, "run.log"), "a")
say(a...) = (println(a...); println(LOG, a...); flush(LOG); flush(stdout))
sayf(f, a...) = (s = Printf.format(Printf.Format(f), a...); print(s); print(LOG, s); flush(LOG); flush(stdout))
say("="^78); say("reopt.jl  ", LABEL, "   ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS")); say("="^78)
say("command   julia tools/reopt.jl ", join(ARGS, " "))
say("outdir    ", OUTDIR)

# the script owns its pool (2026-09-28, tiktak_fix_plan.md 7.1): a fresh process, workers with an
# EXPLICIT Julia thread count (they inherit a JULIA_NUM_THREADS from the environment, not --threads)
nprocs() == 1 || error("reopt.jl starts and owns its worker pool; this process already has $(nprocs() - 1) worker(s)")
const POOL = NPROC > 1 ? addprocs(NPROC; exeflags = `--project=$REPO --threads=1`) : Int[]
@everywhere using LinearAlgebra
@everywhere LinearAlgebra.BLAS.set_num_threads(1)
# workers die with this process (Linux PR_SET_PDEATHSIG; see code/src/TikTak/workers.jl)
nprocs() > 1 && @everywhere workers() (include(joinpath($REPO, "code", "src", "tiktak.jl")); TikTak.die_with_parent!())
include(joinpath(REPO, "code", "src", "tiktak.jl"))
const RESOURCES = TikTak.worker_resources(vcat(1, POOL))
say("processes ", length(RESOURCES), " (master + ", length(POOL), " workers); Julia threads ",
    join(unique(r.julia_threads for r in RESOURCES), "/"), ", BLAS threads ", join(unique(r.blas_threads for r in RESOURCES), "/"))
@everywhere begin
    using Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames
    using Statistics, Dates, ProgressMeter, Distributions, StatsBase
    using QuantEcon, FastGaussQuadrature, Parameters, Dierckx, TOML
    const REPO_ = normpath(joinpath(@__DIR__, ".."))
    const SRC   = joinpath(REPO_, "code", "src")
    include(joinpath(SRC, "paths.jl"));       include(joinpath(SRC, "manifest.jl"))
    include(joinpath(SRC, "diagnostics.jl")); include(joinpath(SRC, "child_lifecycle.jl"))
    include(joinpath(SRC, "parent_family.jl")); include(joinpath(SRC, "tiktak.jl"))
    include(joinpath(REPO_, "code", "smm", "moments.jl"))
end
include(joinpath(REPO, "tools", "experiment_table.jl"))
# run-specific boxes: applied on every process BEFORE FREE/LO/HI/ZFIX are built and before any
# evaluation (unpack clamps into SMM_PARAMS' box)
@everywhere include(joinpath(REPO_, "tools", "run_bounds.jl"))
# start / checkpoint loading and validation (2026-09-22): refuses non-finite or out-of-box
# starts instead of clamping them; verifies a checkpoint's [config] on --resume
@everywhere include(joinpath(REPO_, "tools", "start_loader.jl"))
@everywhere const BOUNDS_ = parse_run_bounds($BOUNDS_STR)
@everywhere const BOUNDS_LINES = apply_run_bounds!(BOUNDS_)
isempty(BOUNDS_LINES) || say("bounds    RUN-SPECIFIC: ", join(BOUNDS_LINES, "; "))

@everywhere const TARGETS = load_targets($TARGETS_FILE)
const SPEC_NOTE = "v1 spec, $(length(SMM_PARAMS)) parameters / $(length(SMM_MOMENTS)) moments"
@everywhere const PGRID = $GRID; @everywhere const PSIMN = $SIM_N; @everywhere const PSEED = $SEED
@everywhere const FIXED_ = $FIXED
@everywhere const FREE_EXTRA_ = $FREE_EXTRA
@everywhere const LOCAL_ALG_ = $(QuoteNode(LOCAL_ALG))

# ---- what is fixed, what is free ---------------------------------------------
for k in keys(FIXED)
    k === :omega || any(q -> q.name === k, SMM_PARAMS) ||
        error("--fix $k: not an estimated parameter and not omega")
end
# mu and omega are not separately identified: omega may be freed only while mu is held (here mu is never searched)
FREE_EXTRA === nothing || !any(q -> q.name === :mu, SMM_PARAMS) || haskey(FIXED, :mu) ||
    error("--free omega requires mu to be held (--fix mu=...): mu and omega are not separately identified")
@everywhere const CE_ARG_ = $CE_ARG
@everywhere const PE_ARG_ = $PE_ARG
@everywhere const CE = merge(CE_ARG_, haskey(FIXED_, :omega) ? (omega = FIXED_[:omega],) : (;))
FREE_EXTRA === nothing || !haskey(CE_ARG, :omega) || error("--free omega and --child-extra omega together")
@everywhere const FREE_IDX = [i for (i, q) in enumerate(SMM_PARAMS) if !haskey(FIXED_, q.name)]
@everywhere const FREE = SMM_PARAMS[FREE_IDX]
@everywhere const ZFIX = let z = fill(NaN, length(SMM_PARAMS))
    for (i, q) in enumerate(SMM_PARAMS)
        haskey(FIXED_, q.name) || continue
        q.lo <= FIXED_[q.name] <= q.hi || error("--fix $(q.name) = $(FIXED_[q.name]) is outside its box [$(q.lo), $(q.hi)]")
        z[i] = to_search(FIXED_[q.name], q)
    end
    z
end
# the SMM part of a free vector; under --free the last coordinate is omega (level link). `embed`,
# `ce_of` and the objective `obj` are in tools/reopt_objective.jl (included below, once EXTRA_ exists)
@everywhere const NF = length(FREE_IDX)
@everywhere const LO = vcat([to_search(q.lo, q) for q in FREE], FREE_EXTRA_ === nothing ? Float64[] : [FREE_EXTRA_.lo])
@everywhere const HI = vcat([to_search(q.hi, q) for q in FREE], FREE_EXTRA_ === nothing ? Float64[] : [FREE_EXTRA_.hi])
const FREE_NAMES = vcat([String(q.name) for q in FREE], FREE_EXTRA === nothing ? String[] : ["omega"])
const FREE_EXTRA_STR = FREE_EXTRA === nothing ? "" : "omega=$(FREE_EXTRA.lo):$(FREE_EXTRA.hi)"
free_extra_record(io) = FREE_EXTRA === nothing ||
    println(io, "\n[free_extra.omega]\nlo = ", FREE_EXTRA.lo, "\nhi = ", FREE_EXTRA.hi, "\nlink = \"level\"\nnote = \"estimated (--free); mu fixed\"")

# ---- extra targeted rows (experiment C) ----------------------------------------
# the target mean from the targets, the weight 1/se^2 from --extra-targets's [moment_cov]
include(joinpath(REPO, "tools", "reopt_identity.jl"))
extra_rows(names, tfile, targets) = extra_rows_from(names, TOML.parsefile(tfile),
    k -> (haskey(targets, k) || error("--extra-moments $k: no [$k] table in the targets"); targets[k].mean))
const EXTRA = extra_rows(EXTRA_NAMES, EXTRA_TFILE, TARGETS)
# provenance: the model-source hash (as the driver records it) and the tools hash (as partB_gate1.sh records it)
using SHA
sha16(files) = bytes2hex(sha256(vcat([read(joinpath(REPO, f)) for f in files]...)))[1:16]
const SOURCE_SHA16 = sha16(("code/src/child_lifecycle.jl", "code/src/parent_family.jl", "code/smm/moments.jl", "code/src/tiktak.jl"))
const TOOLS_SHA16 = sha16(("tools/reopt.jl", "tools/run_bounds.jl", "tools/start_loader.jl", "tools/experiment_table.jl",
                           "tools/reopt_identity.jl", "tools/reopt_objective.jl", "tools/reopt_settings.jl"))
say("source    code sha16 ", SOURCE_SHA16, "; tools sha16 ", TOOLS_SHA16)
@everywhere const EXTRA_ = $EXTRA

# ---- the objective: smm_objective, the audited one (tools/reopt_objective.jl) ------
@everywhere include(joinpath(REPO_, "tools", "reopt_objective.jl"))
# ---- ITS IDENTITY (2026-09-28, follow-up 2; tools/reopt_identity.jl) -------------
# Built from the resolved run, before any evaluation: target and extra-target CONTENT, the
# resolved extra rows, moment order, parameters and box, numerics, SMM_* switches, model and
# adapter source. Every checkpoint and pre-testing cache of this run records it; a resume
# checks it field by field below, before the child warm-up.
const OBJ_FIELDS = reopt_objective_fields(; repo = REPO, targets_file = TARGETS_FILE, extra_targets_file = EXTRA_TFILE,
    extra_rows = EXTRA, moment_names = collect(String, SMM_MOMENTS), free_names = FREE_NAMES,
    free_links = vcat([String(q.link) for q in FREE], FREE_EXTRA === nothing ? String[] : ["level"]),
    lo = LO, hi = HI, fixed = FIXED,
    fixed_search = Dict(String(q.name) => ZFIX[i] for (i, q) in enumerate(SMM_PARAMS) if haskey(FIXED, q.name)),
    free_extra = FREE_EXTRA_STR, run_bounds = BOUNDS_STR, seed = SEED, sim_n = SIM_N, grid = GRID,
    child_grid = "30x30x5", parent_extra = string(PE_ARG), child_extra = string(CE))
const OBJ_ID = TikTak.fields_id(OBJ_FIELDS)
say("objective id ", OBJ_ID, " (targets sha ", OBJ_FIELDS["targets_sha"], ", model ", OBJ_FIELDS["model_sha"],
    ", adapter ", OBJ_FIELDS["adapter_sha"], isempty(EXTRA) ? "" : ", extra targets sha " * OBJ_FIELDS["extra_targets_sha"], ")")
say("grid      parent_extra = ", PE_ARG, "; child_extra = ", CE, "; init-step ", INIT_STEP, "; checkpoint every ", CKPT_EVERY, "; resume ", RESUME)

say("targets   ", relpath(TARGETS_FILE, REPO))
say("fixed     ", isempty(FIXED) ? "(nothing)" : join(("$k = $v" for (k, v) in FIXED), ", "))
say("free      ", length(FREE_NAMES), " parameters: ", join(FREE_NAMES, ", "), FREE_EXTRA === nothing ? "" : "  (omega on [$(FREE_EXTRA.lo), $(FREE_EXTRA.hi)], level)")
isempty(EXTRA) || say("extra     ", join((@sprintf("%s (target %.4f, se %.4f)", k, m, 1 / sqrt(w)) for (k, m, w) in EXTRA), ", "))
sayf("budget    %s; local-evals %d; grid %d; simN %d; seed %d; procs %d\n",
     N_SOBOL > 0 ? "TikTak $N_SOBOL sobol + $N_RESTART restarts, polish $(POLISH_MAXEVAL)" : "local $(LOCAL_ALG) from each start",
     LOCAL_MAXEVAL, GRID, SIM_N, SEED, NPROC)
say("solver    ", N_SOBOL > 0 ?
    "TikTak local stage: " * (isempty(TK_SOLVER_KW) ? "its defaults" : join(("$k = $v" for (k, v) in pairs(TK_SOLVER_KW)), ", ")) *
    " (the settings it ran with are written to results.toml [effective_settings])" :
    "pure local: $(LOCAL_ALG), ftol_rel $(FTOL_REL), initial step $(INIT_STEP) of the box width")

check_psychic_centring(target_m_psychic(TARGETS))
# (the child warm-up runs further down, AFTER every resume check: a refused resume costs no model solve)

# ---- starting points -------------------------------------------------------------
# (the former load_start + clamp of this block was replaced on 2026-09-22 by tools/start_loader.jl:
#  load_start_point validates both loading paths and refuses instead of clamping)
const CKPT_FILE = joinpath(OUTDIR, "checkpoint_best.toml")
# the pure-local RESTART-from-best resume (checkpoint_best.toml exists only in local mode; TikTak
# mode resumes from its own tiktak_state.toml below)
const RESUMED = RESUME && N_SOBOL == 0 && isfile(CKPT_FILE)
const EVALS_DONE = RESUMED ? Int(TOML.parsefile(CKPT_FILE)["n_eval"]) : 0
# what a checkpoint records and a resume must match (tools/start_loader.jl)
const CKPT_CONFIG = checkpoint_config(run_bounds = BOUNDS_STR, fixed = FIXED, targets = relpath(TARGETS_FILE, REPO), extra_moments = EXTRA_NAMES,
                                      seed = SEED, simN = SIM_N, grid = GRID, parent_extra = string(PE_ARG), child_extra = string(CE),
                                      local_maxeval = LOCAL_MAXEVAL, init_step = INIT_STEP, ftol_rel = FTOL_REL,
                                      source_sha16 = SOURCE_SHA16, tools_sha16 = TOOLS_SHA16, free_extra = FREE_EXTRA_STR,
                                      mode = N_SOBOL > 0 ? "tiktak sobol=$N_SOBOL restarts=$N_RESTART polish=$POLISH_MAXEVAL" :
                                             (LOCAL_ALG === :LN_NELDERMEAD ? "local" : "local bobyqa"),
                                      objective_id = OBJ_ID)
@everywhere const CKPT_CONFIG_ = $CKPT_CONFIG
const RESUME_VERIFICATION = Ref(RESUMED ? "pending" : "not a resume")
const START_Z = map(RESUMED ? [CKPT_FILE] : STARTS) do f
    # refuses (error, before any evaluation) a non-finite or out-of-box start; snaps a value within
    # 1e-9 of the box width onto the bound; names a warm start from another box and a replaced fixed coordinate
    z, how, notes = load_start_point(f; free_idx = FREE_IDX, fixed = FIXED)
    if RESUMED
        st, vn = verify_checkpoint(TOML.parsefile(f), CKPT_CONFIG; free_idx = FREE_IDX, fixed = FIXED)
        RESUME_VERIFICATION[] = st; append!(notes, vn)
    end
    say("start     ", relpath(f, REPO), " (", how, ")",
        RESUMED ? "  [RESTART from the checkpointed best after $EVALS_DONE evaluations; not an exact optimizer-state resume; configuration " * RESUME_VERIFICATION[] * "]" : "")
    for nt in notes; say("          note: ", nt); end
    if FREE_EXTRA === nothing
        z[FREE_IDX]
    else
        w, wnote = load_start_omega(f, FREE_EXTRA.lo, FREE_EXTRA.hi); say("          omega ", w, wnote)
        vcat(z[FREE_IDX], w)
    end
end

on_bound(zf) = [FREE_NAMES[i] for i in eachindex(zf) if (p = (zf[i] - LO[i]) / (HI[i] - LO[i]); p < 0.02 || p > 0.98)]
named(zf) = (kw = unpack(embed(zf)); d = Dict{String,Any}(String(q.name) => getfield(kw, q.name) for q in SMM_PARAMS);
             FREE_EXTRA === nothing || (d["omega"] = zf[NF + 1]); d)

# ---- local mode ------------------------------------------------------------------
@everywhere function write_checkpoint(path, zbest, qbest, n, n_prev, t0, label, i)
    kw = unpack(embed(zbest))
    open(path, "w") do io
        println(io, "# checkpoint of the incumbent (tools/reopt.jl); resume = RESTART of Nelder-Mead from this point")
        println(io, "label = \"", label, "\"\nstart = ", i, "\nQ = ", qbest, "\nn_eval = ", n + n_prev, "\nn_eval_this_run = ", n,
                "\nelapsed_min = ", round((time() - t0) / 60, digits = 1), "\nwritten = \"", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\"",
                "\nresume_mode = \"restart_from_best\"")
        println(io, "\n[parameters]")
        for q in SMM_PARAMS; println(io, q.name, " = ", @sprintf("%.17g", getfield(kw, q.name))); end
        FREE_EXTRA_ === nothing || println(io, "omega = ", @sprintf("%.17g", zbest[NF + 1]))
        run_bounds_record(io)                       # the box this point belongs to
        FREE_EXTRA_ === nothing || println(io, "\n[free_extra.omega]\nlo = ", FREE_EXTRA_.lo, "\nhi = ", FREE_EXTRA_.hi, "\nlink = \"level\"")
        println(io); TOML.print(io, Dict("config" => CKPT_CONFIG_))   # what a --resume must match (2026-09-22)
    end
end
@everywhere function run_local(i, z0, maxeval, label, outdir, ckpt_every, init_step, n_prev, ftol_rel, nstarts = 1)
    t0 = time(); n = Ref(0); best = Ref(Inf); q0 = Ref(NaN); zbest = Ref(copy(z0))
    trace = Tuple{Int,Float64}[]
    opt = Opt(LOCAL_ALG_, length(z0))
    lower_bounds!(opt, LO); upper_bounds!(opt, HI)
    ftol_rel!(opt, ftol_rel); ftol_abs!(opt, 1e-10); xtol_rel!(opt, 1e-7); maxeval!(opt, maxeval)
    init_step > 0 && initial_step!(opt, init_step .* (HI .- LO))
    sfx = nstarts > 1 ? "_start$i" : ""
    ckpt = joinpath(outdir, "checkpoint_best$sfx.toml"); prog = joinpath(outdir, "progress$sfx.toml")
    min_objective!(opt, (z, g) -> begin
        n[] += 1; v = obj(z)
        n[] == 1 && (q0[] = v)
        if v < best[]; best[] = v; zbest[] = copy(z); push!(trace, (n[], v)); end
        (n[] % 10 == 0 || n[] == 1) && (println(@sprintf("  [%s start %d] eval %4d  Q %12.4f  best %12.4f  %.1f min", label, i, n[], v, best[], (time() - t0) / 60)); flush(stdout))
        if ckpt_every > 0 && n[] % ckpt_every == 0
            write_checkpoint(ckpt, zbest[], best[], n[], n_prev, t0, label, i)
            open(prog, "w") do io
                println(io, "label = \"", label, "\"\nn_eval = ", n[] + n_prev, "\nbudget = ", maxeval + n_prev, "\nbest_Q = ", best[], "\nlast_Q = ", v,
                        "\nelapsed_min = ", round((time() - t0) / 60, digits = 1), "\nupdated = \"", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\"")
            end
        end
        v
    end)
    (q, z, ret) = optimize(opt, copy(z0))
    write_checkpoint(ckpt, zbest[], best[], n[], n_prev, t0, label, i)
    return (i = i, q0 = q0[], q = q, z = z, ret = ret, n_eval = n[], minutes = (time() - t0) / 60, trace = trace)
end

# ---- TikTak mode: what a resume continues, checked BEFORE the child warm-up (follow-up 2) --------
# The TikTak module's checkpoint (tiktak_state.toml) and pre-testing cache record OBJ_FIELDS; a
# resume is checked against them field by field here, and then through the module's own preflight
# (objective id, optimizer identity, state consistency) -- all before any model is solved.
const TK_STATE = joinpath(OUTDIR, "tiktak_state.toml"); const TK_CACHE = joinpath(OUTDIR, "pretest_cache.toml")
const TK_SEEDS = joinpath(OUTDIR, "tiktak_seeds.toml")
const TK_RESUME = N_SOBOL > 0 && RESUME && (TikTak.checkpoint_exists(TK_STATE) || TikTak.checkpoint_exists(TK_CACHE))
# which checkpoint the resume continues (results.toml): the run state, or only the pre-testing cache
const TK_ROUTE = !TK_RESUME ? "none" : TikTak.checkpoint_exists(TK_STATE) ? "tiktak_state" : "pretest_cache"
if N_SOBOL > 0 && RESUME && !TK_RESUME && isfile(TK_SEEDS)
    error("resume refused: $OUTDIR holds an OLD-FORMAT TikTak checkpoint (tiktak_seeds.toml, before 2026-09-27). It " *
          "recorded the targets by path only, so its values and seed ranking cannot be verified against the current " *
          "objective (2026-09-28, follow-up 2), and it is not continued (--legacy-import is no longer accepted here). " *
          "Start a new --outdir with --start <its best_estimates.toml> (re-evaluated; nothing else carries over).")
end
LEGACY_IMPORT && error("--legacy-import is no longer accepted by reopt.jl (2026-09-28): an old-format TikTak " *
                       "checkpoint cannot be verified against the objective (see the header)")
if TK_RESUME
    for nt in check_reopt_resume(((TK_STATE, :state), (TK_CACHE, :cache)), OBJ_FIELDS)
        say("resume    ", nt)
    end
end
const ASYNC = N_SOBOL > 0 && NPROC > 1
# every TikTak setting in ONE place: the preflight below and the search use the same identity
const TK_KW = merge(TK_SOLVER_KW, (N = N_SOBOL, Nstar = N_RESTART, n_valid_target = N_VALID_TARGET,
               extra_seeds = [copy(z) for z in START_Z], invalid_value = SMM_PENALTY,
               local_maxeval = LOCAL_MAXEVAL, polish_maxeval = max(POLISH_MAXEVAL, 1), skip_polish = POLISH_MAXEVAL == 0,
               objective_id = OBJ_ID, objective_fields = OBJ_FIELDS, state_path = TK_STATE, pretest_cache = TK_CACHE,
               pretest_chunk = max(8, 4 * max(NPROC, 1)), stop_after_restarts = STOP_AFTER,
               local_mode = ASYNC ? :async_process : :serial, local_workers = ASYNC ? POOL : Int[],
               local_count = LOCAL_PROCS, pretest_workers = ASYNC ? POOL : Int[], objective_key = :search,
               retire_after_abort = true,                # the pool is this script's own (follow-up 3)
               purpose = PRESET_NAME))
if TK_RESUME
    try
        tiktak(obj, LO, HI; TK_KW..., resume = true, preflight_only = true)
    catch e
        e isa TikTak.ResumeRefused ? error("resume refused: " * e.msg) : rethrow()
    end
    say("resume    TikTak preflight passed: objective, optimizer identity and checkpoint consistency verified")
end

print("warming the child cache on every process ... "); flush(stdout)
let t0 = time()
    # child_wage: memo 19's child_config requires the wage loading (as the runner's warm-up; 2026-10-02 merge fix)
    @everywhere let cfg = child_config(TARGETS; Na = 30, Nk = 30, Nt = 5, simN = PSIMN, seed = PSEED,
                                       child_wage = child_wage_config())
        child_base(cfg)
    end
    sayf("%.1fs\n", time() - t0)
end

results = NamedTuple[]
t_all = time()
if N_SOBOL == 0
    mapper = nprocs() > 1 ? pmap : map
    remaining = max(LOCAL_MAXEVAL - EVALS_DONE, 1)
    RESUMED && say("resume    ", EVALS_DONE, " evaluations already spent; ", remaining, " remain")
    res = mapper(i -> run_local(i, START_Z[i], remaining, LABEL, OUTDIR, CKPT_EVERY, INIT_STEP, EVALS_DONE, FTOL_REL, length(START_Z)), 1:length(START_Z))
    append!(results, res)
else
    # ---- TikTak through the module's checkpoints (2026-09-27; identity 2026-09-28) ---------------
    ASYNC && @everywhere TikTak.register_objective!(:search, obj)
    tk_progress(j, K, theta, f_local, best_, best_x, row) =
        sayf("  restart %d/%d theta %.3f: start %12.4f -> local %12.4f (%s)  incumbent %12.4f  %.1f min\n",
             j, K, theta, row.f_start, f_local, row.ret, best_, (time() - t_all) / 60)
    tk = tiktak(obj, LO, HI; TK_KW..., resume = TK_RESUME, on_local = tk_progress,
                on_sobol = (i, n, fx, best_) -> (i == n || i % 25 == 0) &&
                    sayf("  sobol %5d/%d  best %12.4f  %.1f min\n", i, n, best_, (time() - t_all) / 60))
    push!(results, (i = 0, q0 = tk.f_sobol_best, q = tk.f, z = tk.x, ret = tk.winner_ret, n_eval = tk.n_eval,
                    minutes = (time() - t_all) / 60, trace = Tuple{Int,Float64}[], tiktak = tk))
end
const MINUTES = (time() - t_all) / 60

# ---- records -----------------------------------------------------------------------
best = results[argmin([r.q for r in results])]
open(joinpath(OUTDIR, "results.toml"), "w") do io
    println(io, "# GENERATED by tools/reopt.jl")
    println(io, "label = \"", LABEL, "\"")
    println(io, "generated = \"", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\"")
    println(io, "command = \"", join(ARGS, " "), "\"")
    println(io, "targets = \"", relpath(TARGETS_FILE, REPO), "\"")
    println(io, "mode = \"", N_SOBOL > 0 ? "tiktak" : "local", "\"")
    println(io, "n_sobol = ", N_SOBOL, "\nn_restarts = ", N_RESTART, "\nlocal_maxeval = ", LOCAL_MAXEVAL, "\npolish_maxeval = ", POLISH_MAXEVAL)
    println(io, "grid = ", GRID, "\nsimN = ", SIM_N, "\nseed = ", SEED, "\nprocs = ", NPROC)
    println(io, "parent_extra = \"", PE_ARG, "\"\nchild_extra = \"", CE, "\"")
    # the solver settings AS RUN (2026-09-28, follow-up 5): TikTak mode reads them from the configuration
    # the optimizer ran with, never from the command line; the full set is in [effective_settings]
    tkc = N_SOBOL > 0 ? results[1].tiktak.config : nothing
    println(io, "init_step = ", tkc === nothing ? INIT_STEP : tkc.local_.initial_step, "   # fraction of the box width; 0 = NLopt's heuristic")
    println(io, "ftol_rel = ", tkc === nothing ? FTOL_REL : tkc.local_.ftol_rel, "   # of the ", tkc === nothing ? "pure-local solver" : "TikTak local stage")
    for l in reopt_resume_lines(; tiktak = N_SOBOL > 0, local_resumed = RESUMED, evals_done = EVALS_DONE,
                                local_verification = RESUME_VERIFICATION[], tk_route = TK_ROUTE,
                                tk_semantics = tkc === nothing ? :fresh : results[1].tiktak.resume_semantics)
        println(io, l)
    end
    println(io, "objective_id = \"", OBJ_ID, "\"   # tools/reopt_identity.jl: target contents, extra rows, switches, model + adapter source")
    println(io, "source_sha16 = \"", SOURCE_SHA16, "\"\ntools_sha16 = \"", TOOLS_SHA16, "\"\nstart_loader = \"tools/start_loader.jl (refuses out-of-box / non-finite starts; no clamp)\"")
    println(io, "minutes_total = ", round(MINUTES, digits = 1))
    println(io, "free = [", join(("\"$n\"" for n in FREE_NAMES), ", "), "]")
    println(io, "starts = [", join(("\"$(relpath(f, REPO))\"" for f in STARTS), ", "), "]")
    println(io, "\n[effective_settings]")
    if tkc === nothing
        println(io, "local_alg = \"", LOCAL_ALG, "\"\nlocal_ftol_rel = ", FTOL_REL, "\nlocal_ftol_abs = 1e-10\nlocal_xtol_rel = 1e-7\nlocal_maxeval = ",
                max(LOCAL_MAXEVAL - EVALS_DONE, 1), "\nlocal_initial_step = ", INIT_STEP)
    else
        for l in effective_settings_lines(tkc); println(io, l); end
    end
    println(io, "\n[fixed]")
    for (k, v) in FIXED; println(io, k, " = ", v); end
    run_bounds_record(io)
    println(io, "\n[extra_moments]")
    for (k, m, w) in EXTRA; println(io, k, " = { target = ", m, ", weight = ", w, " }"); end
    for r in results
        println(io, "\n[[result]]")
        println(io, "start = ", r.i, "\nQ_start = ", r.q0, "\nQ = ", r.q, "\nret = \"", r.ret, "\"\nn_eval = ", r.n_eval, "\nn_eval_total = ", r.n_eval + EVALS_DONE, "\nminutes = ", round(r.minutes, digits = 1))
        println(io, "free_near_bound = [", join(("\"$s\"" for s in on_bound(r.z)), ", "), "]")
        if haskey(r, :tiktak)
            tk = r.tiktak; o = tk.incumbent.origin; v = tk.incumbent.verification
            println(io, "run_status = \"", tk.status, "\"   # complete|stopped_early|paused")
            println(io, "purpose = \"", tk.purpose, "\"\nnstar_requested = ", tk.nstar_requested, "\nnstar_effective = ", tk.nstar_effective)
            println(io, "pretest_attempted = ", tk.pretest.attempted, "\npretest_valid = ", tk.pretest.valid, "\npretest_reused = ", tk.pretest.reused)
            println(io, "point_origin = \"", o.stage, "\"\npoint_origin_ret = \"", o.ret, "\"\nverification = \"", v.status, "\"")
            println(io, "local_converged = ", TikTak.local_converged(tk), "   # on this run's objective")
            println(io, "n_eval_complete = ", tk.n_eval_complete, "\nresume_semantics = \"", tk.resume_semantics, "\"")
            println(io, "search_budget_complete = ", TikTak.search_budget_complete(tk),
                    "   # validated: every planned restart committed once, none in flight")
            println(io, "restarts_in_flight = [", join(tk.inflight, ", "), "]")
            println(io, "work_known = ", TikTak.work_known(tk), "   # false: some attempt's evaluations are unknown (lost, failed, unsettled)")
            a = tk.accounting
            println(io, "jobs_lost = ", a.jobs_lost, "\nattempts_unknown = ", a.attempts_unknown, "\npretest_lost = ", a.pretest_lost,
                    "\nabandoned_known = ", a.abandoned_known, "\nduplicates_ignored = ", a.duplicates_ignored, "\nretries = ", a.retries)
        end
        println(io, "z_free = [", join((@sprintf("%.17g", x) for x in r.z), ", "), "]")
        println(io, "trace = [", join(("[$(t[1]), $(t[2])]" for t in r.trace), ", "), "]")
        println(io, "[result.parameters]")
        for (k, v) in sort(collect(named(r.z))); println(io, k, " = ", @sprintf("%.17g", v)); end
    end
end
open(joinpath(OUTDIR, "best_estimates.toml"), "w") do io
    println(io, "# GENERATED by tools/reopt.jl -- the best point of run \"", LABEL, "\"")
    println(io, "# NOT a promoted estimate. Fixed: ", isempty(FIXED) ? "(nothing)" : join(("$k = $v" for (k, v) in FIXED), ", "))
    println(io, "label = \"", LABEL, "\"\nQ = ", best.q, "\nomega = ", FREE_EXTRA === nothing ? get(FIXED, :omega, CHILD_DEFAULTS.omega) : best.z[NF + 1],
            FREE_EXTRA === nothing ? "" : "   # ESTIMATED (--free omega=$(FREE_EXTRA_STR[7:end]))", "\nrun_bounds_arg = \"", BOUNDS_STR, "\"")
    println(io, "source_sha16 = \"", SOURCE_SHA16, "\"\ntools_sha16 = \"", TOOLS_SHA16, "\"")
    println(io, "spec_note = \"", SPEC_NOTE, "; ", isempty(FIXED) ? "nothing fixed" : "fixed " * join(("$k=$v" for (k, v) in FIXED), ","),
            isempty(EXTRA) ? "" : "; extra rows " * join(String.(EXTRA_NAMES), ","), "\"")
    println(io, "\n[parameters]")
    for (k, v) in sort(collect(named(best.z))); println(io, k, " = ", @sprintf("%.17g", v)); end
    run_bounds_record(io)
    free_extra_record(io)
    println(io, "\n[fixed]")
    for (k, v) in FIXED; println(io, k, " = ", v); end
end

# ---- the comparison-table rows: every start and the best point -------------------------
say(""); say("comparison-table rows (experiment_table.jl format):")
open(joinpath(OUTDIR, "table.txt"), "w") do io
    println(io, TABLE_HEADER); say(TABLE_HEADER)
    pts = [("start $i", z) for (i, z) in enumerate(START_Z)]
    N_SOBOL == 0 && append!(pts, [("best from start $(r.i)", r.z) for r in results])
    N_SOBOL > 0 && push!(pts, ("best (tiktak)", best.z))
    for (lab, z) in pts
        kw = unpack(embed(z)); point = Dict{Symbol,Float64}(q.name => getfield(kw, q.name) for q in SMM_PARAMS)
        rec = case_record(LABEL * " " * lab, point, TARGETS; child_extra = ce_of(z), parent_extra = PE_ARG, simN = SIM_N, seed = SEED, Na = GRID, Nhc = GRID,
                          extra = EXTRA)
        write_case(joinpath(OUTDIR, "case_" * replace(lab, r"[^A-Za-z0-9_.=-]" => "_") * ".toml"), rec)
        line = table_line(rec); println(io, line); say(line)
        cl = contributions_line(rec); println(io, cl); say(cl)
        if haskey(rec, "Q_extra")
            el = @sprintf("    Q_original %.4f  Q_extra %.4f  total %.4f   %s", rec["Q"], rec["Q_extra"], rec["Q"] + rec["Q_extra"],
                          join((@sprintf("%s %+.1f", k, v) for (k, v) in rec["t_extra"]), "  "))
            println(io, el); say(el)
        end
    end
end
sayf("\nbest Q %.6f (start %d, %s, %d evaluations); total %.1f min\n", best.q, best.i, best.ret, best.n_eval, MINUTES)
say("DONE"); close(LOG)
