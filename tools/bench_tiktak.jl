#!/usr/bin/env julia
# =============================================================================
# bench_tiktak.jl -- measurements behind the TikTak changes of 2026-09-27 (plan J5, step 9)
#
#   julia --project=. --threads=1 tools/bench_tiktak.jl synthetic --out <dir>
#   julia --project=. --threads=1 tools/bench_tiktak.jl model --targets <targets.toml> --out <dir> [--procs 2]
#        [--grid 12 --simN 300 --child-grid 12x12x3 --seed 1234 --parent-extra ... --child-extra ...]
#        [--warm-evals 6 --mem-evals 12 --job-evals 8 --profile-evals 0]
#
# The model part's NUMERICAL SETTINGS are explicit (2026-09-28, tiktak_fix_plan.md 7.3): the defaults
# are the --quick grids; `--grid 30 --simN 2000 --child-grid 30x30x5` plus the run's --parent-extra /
# --child-extra and SMM_* switches reproduce a production evaluation -- run_smm.jl's objective exactly.
# Every evaluation count is bounded by a flag: a benchmark never launches a search.
#
# synthetic  (no economic model; at most 4 worker processes)
#   A. step-9 settings on the SAME seeds: baseline NLopt steps vs normalized coordinates vs
#      explicit / theta-shrinking initial steps; final f, evaluations, convergence share,
#      endpoint dispersion, time -- over shifted instances of three test functions
#   B. scheduling on a CPU-bound synthetic objective (busy-wait per evaluation): the frozen
#      pre-2026-09-27 code, the new serial path, async with 1/2/4 workers; wall time, f,
#      evaluations, cold (first) vs warm (second) runs
#   C. orchestration costs: checkpoint write time at K = 20 and K = 1000, allocations of the
#      mixing kernel after warm-up
# model      (the real SMM objective under the SMM_* switches in force, at the grids given)
#   worker start-up, model load, child warm-up, cold and warm evaluation, the same Q at the
#   same point on fresh and warmed workers and on the master (J4), process memory over many
#   evaluations, bytes per job/result message, and @code_warntype of the local-solve kernel
#   on the objective's actual type (J1). Since 2026-09-28 (7.3) also: the resources each
#   process actually runs with, per-evaluation allocations and GC time ON A WARMED WORKER,
#   the round trip of a job message, and (--profile-evals P > 0) a sampling profile of P
#   evaluations on a warmed worker: profile_worker_flat.txt, profile_worker_tree.txt and a
#   self-time breakdown by source file (where the time of an evaluation actually goes).
#
# Writes <out>/bench_<part>.toml and <out>/bench_<part>.md. Numbers are measurements on this
# machine at the time of the run, not promises: the shared server's load varies.
# =============================================================================

using Distributed, Printf, TOML, Statistics, Serialization, Dates, InteractiveUtils
const REPO = normpath(joinpath(@__DIR__, ".."))
argstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i + 1])
const PART = isempty(ARGS) ? "synthetic" : ARGS[1]
# (ported from apps/Structural-estimation-v2 on 2026-10-02; scratch in temp/ unless --out is given)
const OUT = mkpath(argstr("--out", joinpath(REPO, "temp", "bench_tiktak_" * string(round(Int, time())))))
const T0 = time()
include(joinpath(REPO, "code", "src", "tiktak.jl"))
md = IOBuffer(); res = Dict{String,Any}("generated" => string(now()), "host" => gethostname(), "julia" => string(VERSION),
                                         "cpu_threads" => Sys.CPU_THREADS, "loadavg_start" => Sys.loadavg()[1])
say(a...) = (println(a...); println(md, a...); flush(stdout))
rss_mb() = parse(Int, match(r"VmRSS:\s+(\d+)", read("/proc/self/status", String)).captures[1]) / 1024

# the frozen pre-2026-09-27 implementation (the "before" of part B)
module LegacyTikTak
using NLopt, Printf
include(joinpath(@__DIR__, "testdata", "tiktak_legacy_20260927.jl"))
end

if PART == "synthetic"
    rast(x) = 10length(x) + sum(xi^2 - 10cos(2π * xi) for xi in x)
    rosen(x) = sum(100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2 for i in 1:length(x)-1)
    say("# TikTak benchmark -- synthetic ($(now()))\n")

    # ---- A. step-9 settings ------------------------------------------------------------
    say("## A. Search geometry and initial steps (plan step 9), same seeds\n")
    say("Median over 5 shifted instances per function; N = 400 Sobol', K = 12 restarts, local cap 400, polish 150.\n")
    problems = [("rastrigin d=4", d -> rast, fill(-5.12, 4), fill(5.12, 4)),
                ("rosenbrock d=5", d -> rosen, fill(-2.0, 5), fill(2.0, 5)),
                ("scaled quadratic d=6", d -> (x -> sum(abs2, (x .- d) ./ [1e-3, 1e-2, 1.0, 10.0, 1e2, 1e3])),
                 -[1e-3, 1e-2, 1.0, 10.0, 1e2, 1e3], [1e-3, 1e-2, 1.0, 10.0, 1e2, 1e3])]
    settings = [("baseline (NLopt default step)", (;)),
                ("normalized coordinates", (normalize = true,)),
                ("initial step 0.10 of the box", (local_initial_step = 0.10,)),
                ("initial step 0.05 of the box", (local_initial_step = 0.05,)),
                ("theta-shrink step 0.2, min 0.05", (local_initial_step = 0.2, local_step_schedule = :theta_shrink, local_step_min = 0.05)),
                ("normalized + step 0.10", (normalize = true, local_initial_step = 0.10))]
    tabA = Dict{String,Any}[]
    say("| function | setting | median final f | median evaluations | converged share | endpoint spread | seconds |")
    say("|---|---|---:|---:|---:|---:|---:|")
    for (pname, mk, lo, hi) in problems
        for (sname, kw) in settings
            fs = Float64[]; ne = Int[]; conv = Float64[]; spread = Float64[]; secs = Float64[]
            for inst in 1:5
                shift = 0.1 .* (hi .- lo) .* sin.(inst .* (1:length(lo)))      # deterministic shifts
                f = let g = mk(shift), sh = shift; pname == "scaled quadratic d=6" ? g : (x -> g(x .- sh)) end
                t = @elapsed r = tiktak(f, lo, hi; N = 400, Nstar = 12, local_maxeval = 400, polish_maxeval = 150, kw...)
                push!(fs, r.f); push!(ne, r.n_eval); push!(secs, t)
                push!(conv, count(t_ -> TikTak.ret_class(t_.ret) === :converged, r.trace) / length(r.trace))
                ends = [rec.x for rec in r.records]
                push!(spread, mean(TikTak.boxdist(a, b, lo, hi) for a in ends, b in ends))
            end
            row = Dict{String,Any}("function" => pname, "setting" => sname, "f_median" => median(fs), "f_all" => fs,
                                   "evals_median" => median(ne), "converged_share" => mean(conv),
                                   "endpoint_spread" => median(spread), "seconds_median" => median(secs))
            push!(tabA, row)
            say(@sprintf("| %s | %s | %.3g | %d | %.2f | %.3f | %.3f |", pname, sname, median(fs), round(Int, median(ne)),
                         mean(conv), median(spread), median(secs)))
        end
    end
    res["A_settings"] = tabA

    # ---- B. scheduling, before and after -----------------------------------------------------
    say("\n## B. Local-stage scheduling on a CPU-bound objective (plan J5 before/after)\n")
    busy_ms = 2.0
    say("Rastrigin d=4 with a $(busy_ms) ms busy-wait per evaluation; N = 200, K = 16, local cap 120, no polish.")
    say("The `pre-2026-09-27` row is the frozen old code (serial local stage on the master); the Sobol' stage is serial in every row.\n")
    ws = addprocs(4; exeflags = `--project=$REPO --threads=1 --startup-file=no`)
    t_ws = @elapsed begin
        @everywhere ws include(joinpath($REPO, "code", "src", "tiktak.jl"))
        @everywhere ws begin
            rast(x) = 10length(x) + sum(xi^2 - 10cos(2π * xi) for xi in x)
            function busy_rast(x)
                t = time_ns(); while time_ns() - t < 2_000_000; end
                return rast(x)
            end
            TikTak.register_objective!(:busy, busy_rast)
        end
    end
    function busy_rast(x)
        t = time_ns(); while time_ns() - t < 2_000_000; end
        return rast(x)
    end
    lo4, hi4 = fill(-5.12, 4), fill(5.12, 4)
    kwb = (N = 200, Nstar = 16, local_maxeval = 120, skip_polish = true)
    rowsB = Dict{String,Any}[]
    say("| path | run | wall s | local-stage s | final f | evaluations | local workers |")
    say("|---|---|---:|---:|---:|---:|---:|")
    for (label, runit, nw) in (("pre-2026-09-27 serial", () -> LegacyTikTak.tiktak(busy_rast, lo4, hi4; kwb...), 0),
                                ("new serial", () -> tiktak(busy_rast, lo4, hi4; kwb...), 0),
                                ("async 1 worker", () -> tiktak(busy_rast, lo4, hi4; kwb..., local_mode = :async_process, local_workers = ws, local_count = 1, objective_key = :busy), 1),
                                ("async 2 workers", () -> tiktak(busy_rast, lo4, hi4; kwb..., local_mode = :async_process, local_workers = ws, local_count = 2, objective_key = :busy), 2),
                                ("async 4 workers", () -> tiktak(busy_rast, lo4, hi4; kwb..., local_mode = :async_process, local_workers = ws, local_count = 4, objective_key = :busy), 4))
        for run_ in ("cold", "warm")
            t = @elapsed r = runit()
            t_sobol = (kwb.N + 0) * busy_ms / 1000          # the serial Sobol' stage, by construction
            push!(rowsB, Dict{String,Any}("path" => label, "run" => run_, "wall_s" => t, "local_s_approx" => t - t_sobol,
                                          "f" => r.f, "n_eval" => r.n_eval, "local_workers" => nw))
            say(@sprintf("| %s | %s | %.2f | %.2f | %.4f | %d | %d |", label, run_, t, t - t_sobol, r.f, r.n_eval, nw))
        end
    end
    res["B_scheduling"] = rowsB
    res["B_worker_setup_s"] = t_ws
    rmprocs(ws; waitfor = 30)

    # ---- C. orchestration costs ------------------------------------------------------------
    say("\n## C. Orchestration costs\n")
    r = tiktak(rast, lo4, hi4; N = 60, Nstar = 5, skip_polish = true, local_maxeval = 30)
    for K in (20, 1000)
        st = TikTak.new_state(r.config, lo4, hi4, [rand(4) for _ in 1:K], rand(K), collect(1:K), K, K, r.pretest,
                              "obj", Dict{String,Any}(), "opt", Dict{String,Any}(), "bench")
        for j in 1:K
            job = TikTak.restart_job(st, j, 1, zeros(4))
            TikTak.commit_restart!(st, job, TikTak.RestartResult(st.run_id, :local, j, 1, 1.0, rand(4), rand(), 100, :FTOL_REACHED, "", 1, 1.0))
        end
        p = joinpath(mktempdir(), "s.toml")
        TikTak.write_state(p, st)
        tw = minimum(@elapsed(TikTak.write_state(p, st)) for _ in 1:5)
        tr = minimum(@elapsed(TikTak.state_from_dict(first(TikTak.read_state_dict(p)), r.config)) for _ in 1:5)
        res["C_checkpoint_K$K"] = Dict("write_s" => tw, "read_s" => tr, "bytes" => filesize(p))
        say(@sprintf("checkpoint with %4d committed restarts: %8.1f KB, write %.3f s, read+convert %.3f s", K, filesize(p) / 1024, tw, tr))
    end
    buf = zeros(4); s1, z1 = rand(4), rand(4)
    TikTak.make_start!(buf, s1, z1, 0.3, lo4, hi4)
    a_mix = @allocated TikTak.make_start!(buf, s1, z1, 0.3, lo4, hi4)
    say("make_start! after warm-up: $a_mix bytes allocated per call")
    res["C_make_start_alloc_bytes"] = a_mix

elseif PART == "model"
    const TFILE = argstr("--targets", "")
    isempty(TFILE) && error("--targets <targets.toml> is required")
    const NPW = parse(Int, argstr("--procs", "2"))
    # the numerical problem, explicit (7.3); the defaults are run_smm.jl --quick
    const BGRID = parse(Int, argstr("--grid", "12"))
    const BSIMN = parse(Int, argstr("--simN", "300"))
    const BCHILD = let p = parse.(Int, split(argstr("--child-grid", "12x12x3"), 'x')); (Na = p[1], Nk = p[2], Nt = p[3]) end
    const BSEED = parse(Int, argstr("--seed", "1234"))
    const PE_STR = argstr("--parent-extra", ""); const CE_STR = argstr("--child-extra", "")
    const N_WARM = parse(Int, argstr("--warm-evals", "6"))
    const N_MEM = parse(Int, argstr("--mem-evals", "12"))
    const N_JOB = parse(Int, argstr("--job-evals", "8"))
    const N_PROF = parse(Int, argstr("--profile-evals", "0"))
    "run_smm.jl's parse_extra_arg: k=v,... as a NamedTuple (N* Ints, numbers Float64, else Symbols)."
    function parse_extra(s)
        isempty(s) && return (;)
        ks = Symbol[]; vs = Any[]
        for kv in split(s, ',')
            k, v = split(kv, '='); push!(ks, Symbol(strip(k))); v = strip(v); isN = startswith(String(ks[end]), "N")
            push!(vs, isN && tryparse(Int, v) !== nothing ? parse(Int, v) : (tryparse(Float64, v) !== nothing ? parse(Float64, v) : Symbol(v)))
        end
        NamedTuple{Tuple(ks)}(Tuple(vs))
    end
    const PE_RUN = parse_extra(PE_STR); const CE_RUN = parse_extra(CE_STR)
    say("# TikTak benchmark -- the real SMM objective ($(now()))\n")
    say("Targets `$(relpath(TFILE, REPO))`; SMM switches: ", join(("$k=$(ENV[k])" for k in sort(collect(keys(ENV))) if startswith(k, "SMM_")), " "), "\n")
    say(@sprintf("Numerical problem: parent Na = Nhc = %d, simN = %d, child grid %dx%dx%d, seed %d; parent-extra `%s`; child-extra `%s`%s\n",
                 BGRID, BSIMN, BCHILD.Na, BCHILD.Nk, BCHILD.Nt, BSEED, PE_STR, CE_STR,
                 (BGRID, BSIMN, BCHILD) == (12, 300, (Na = 12, Nk = 12, Nt = 3)) ? " (the --quick grids)" : ""))
    setup = quote
        using Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames
        using Statistics, Dates, ProgressMeter, Distributions, StatsBase
        using QuantEcon, FastGaussQuadrature, Parameters, Dierckx, TOML
        LinearAlgebra.BLAS.set_num_threads(1)
        const REPO_ = $REPO; const SRC = joinpath(REPO_, "code", "src")
        include(joinpath(SRC, "paths.jl")); include(joinpath(SRC, "manifest.jl"))
        include(joinpath(SRC, "diagnostics.jl")); include(joinpath(SRC, "child_lifecycle.jl"))
        include(joinpath(SRC, "parent_family.jl")); include(joinpath(SRC, "tiktak.jl"))
        include(joinpath(REPO_, "code", "smm", "moments.jl"))
        const TARGETS = load_targets($TFILE)
        # exactly run_smm.jl's search objective, at the grids given
        objective(z) = smm_objective(z, TARGETS; Na = $BGRID, Nk = 2, Nhc = $BGRID, simN = $BSIMN, seed = $BSEED,
                                     child_grid = $BCHILD, demo_sim = false,
                                     child_extra = $CE_RUN, parent_extra = $PE_RUN)
        TikTak.register_objective!(:search, objective)
        # one evaluation measured where it runs: wall time, bytes allocated, GC time (7.3)
        timed_eval(z) = (s = @timed objective(z); (q = s.value, seconds = s.time, bytes = s.bytes, gc_seconds = s.gctime))
        # a sampling profile of the evaluations of `pts` on THIS process (7.3): flat and tree reports,
        # and the SELF time (samples where the frame is the leaf) summed by source file
        # (Profile is loaded on every process BEFORE this block: a block is macro-expanded as a whole)
        function profile_evals(pts; delay = 0.005)
            Profile.clear(); Profile.init(n = 20_000_000, delay = delay)
            Profile.@profile for p in pts; objective(p); end
            ctx(io) = IOContext(io, :displaysize => (100_000, 400))
            flat = sprint(io -> Profile.print(ctx(io); format = :flat, sortedby = :count, mincount = 5, C = false))
            tree = sprint(io -> Profile.print(ctx(io); format = :tree, maxdepth = 40, mincount = 20, noisefloor = 2.0, C = false))
            byfile = Dict{String,Int}(); total = 0
            for l in split(flat, '\n')
                m = match(r"^\s*(\d+)\s+(\d+)\s+(\S+)\s+(-?\d+)\s", l)
                m === nothing && continue
                self = parse(Int, m.captures[2]); total += self
                f = basename(String(m.captures[3]))
                byfile[f] = get(byfile, f, 0) + self
            end
            return (flat = flat, tree = tree, byfile = byfile, total_self = total)
        end
    end
    t_add = @elapsed ws = TikTak.start_workers(NPW; project = REPO, exeflags = `--startup-file=no`)
    @everywhere using Profile                 # a standard library, on every project's load path
    t_load_master = @elapsed Core.eval(Main, setup)
    t_load_workers = @elapsed @everywhere ws $setup
    say(@sprintf("start %d workers: %.1f s; load the model: %.1f s on the master, %.1f s on the workers (in parallel)", NPW, t_add, t_load_master, t_load_workers))
    resources = TikTak.worker_resources(vcat(1, ws))
    say("processes as they run (TikTak.worker_resources): ",
        join((@sprintf("%d: %d Julia thread(s), %d BLAS", r.id, r.julia_threads, r.blas_threads) for r in resources), "; "))
    lo, hi = Main.search_bounds(); x0 = Main.incumbent()
    t_cold_master = @elapsed q0 = Main.objective(x0)
    t_cold_w = @elapsed q_w1 = remotecall_fetch(z -> Main.objective(z), ws[1], x0)
    say(@sprintf("first evaluation (compilation + child cache): master %.1f s, worker %.1f s", t_cold_master, t_cold_w))
    rng = Random.MersenneTwister(7)
    pts = [lo .+ rand(rng, length(lo)) .* (hi .- lo) for _ in 1:N_WARM]
    warm = Float64[]
    for p in pts; push!(warm, @elapsed Main.objective(p)); end
    say(@sprintf("warm evaluations on the master: median %.2f s (min %.2f, max %.2f) over %d random points", median(warm), minimum(warm), maximum(warm), length(pts)))
    # J4: the same point gives the same Q on the master, a fresh worker and a warmed worker
    q_fresh = remotecall_fetch(z -> Main.objective(z), ws[end], x0)
    for p in pts; remotecall_fetch(z -> Main.objective(z), ws[1], p); end      # warm worker 1 elsewhere
    q_warm = remotecall_fetch(z -> Main.objective(z), ws[1], x0)
    q_back = Main.objective(x0)
    same = q0 == q_w1 == q_fresh == q_warm == q_back
    say(@sprintf("J4 common random numbers: Q(x0) master %.10g | worker (cold) %.10g | fresh worker %.10g | warmed worker %.10g | master again %.10g -> %s",
                 q0, q_w1, q_fresh, q_warm, q_back, same ? "bit-identical" : "DIFFERENT"))
    # memory over many evaluations on one warmed worker, each evaluation timed THERE: wall time,
    # bytes allocated and GC time per evaluation (7.3), not a master-side round trip
    rss_w() = remotecall_fetch(() -> parse(Int, match(r"VmRSS:\s+(\d+)", read("/proc/self/status", String)).captures[1]) / 1024, ws[1])
    m0 = rss_w()
    mems = Float64[]; wt = NamedTuple[]
    for k in 1:N_MEM
        push!(wt, remotecall_fetch(z -> Main.timed_eval(z), ws[1], lo .+ rand(rng, length(lo)) .* (hi .- lo)))
        (k % max(1, N_MEM ÷ 3) == 0 || k == N_MEM) && push!(mems, rss_w())
    end
    say(@sprintf("worker memory (RSS): %.0f MB after warm-up, then %s MB over %d more evaluations", m0,
                 join((@sprintf("%.0f", m) for m in mems), " / "), N_MEM))
    w_s = [x.seconds for x in wt]; w_mb = [x.bytes / 2^20 for x in wt]; w_gc = [x.gc_seconds for x in wt]
    say(@sprintf("warm evaluations ON THE WORKER: median %.2f s (min %.2f, max %.2f); allocated median %.0f MB per evaluation; GC %.1f%% of the time (median %.2f s)",
                 median(w_s), minimum(w_s), maximum(w_s), median(w_mb), 100 * sum(w_gc) / sum(w_s), median(w_gc)))
    # one local job on a worker, and its messages
    st_cfg = tiktak(x -> 0.0, lo, hi; N = 2, Nstar = 1, skip_polish = true, local_maxeval = 1).config
    job = TikTak.RestartJob("bench", :local, 1, 1, 0.0, copy(x0), 1, true, NaN,
                            TikTak.SolverSettings(:LN_NELDERMEAD, 1e-3, 1e-10, 1e-8, N_JOB), lo, hi, :rethrow, :search, 1, Inf, 1.0, false, 1)
    t_job = @elapsed r_job = remotecall_fetch(TikTak.worker_run_job, ws[1], job)
    io = IOBuffer(); serialize(io, job); b_job = position(io)
    io = IOBuffer(); serialize(io, r_job); b_res = position(io)
    rt = [@elapsed(remotecall_fetch(identity, ws[1], job)) for _ in 1:20]    # a job's round trip, no solve
    say(@sprintf("one local job (Nelder-Mead, cap %d) on a warm worker: %.1f s, %d evaluations; message sizes: job %d bytes, result %d bytes; a job message's round trip: median %.2f ms",
                 N_JOB, t_job, r_job.n_eval, b_job, b_res, 1000 * median(rt)))
    # where the time of an evaluation goes: a sampling profile on the warmed worker (7.3)
    prof = nothing
    if N_PROF > 0
        ppts = [lo .+ rand(rng, length(lo)) .* (hi .- lo) for _ in 1:N_PROF]
        t_prof = @elapsed prof = remotecall_fetch(p -> Main.profile_evals(p), ws[1], ppts)
        write(joinpath(OUT, "profile_worker_flat.txt"), prof.flat)
        write(joinpath(OUT, "profile_worker_tree.txt"), prof.tree)
        top = sort(collect(prof.byfile); by = p -> -p[2])
        say(@sprintf("\nsampling profile of %d evaluations on worker %d (%.1f s, %d leaf samples): self time by source file", N_PROF, ws[1],
                     t_prof, prof.total_self))
        say("| file | share of self time |\n|---|---:|")
        for (f, n) in top[1:min(end, 15)]
            say(@sprintf("| %s | %.1f%% |", f, 100 * n / max(prof.total_self, 1)))
        end
        say("(profile_worker_flat.txt, profile_worker_tree.txt)")
    end
    open(joinpath(OUT, "code_warntype_run_local_smm.txt"), "w") do io
        code_warntype(io, TikTak.run_local, (typeof(Main.objective), TikTak.RestartJob, TikTak.NoProgress))
    end
    open(joinpath(OUT, "code_warntype_worker_run_job.txt"), "w") do io
        code_warntype(io, TikTak._run_registered, (typeof(Main.objective), TikTak.RestartJob, TikTak.RemoteProgress))
    end
    say("code_warntype of run_local / _run_registered on the SMM objective's type: code_warntype_*_smm.txt, code_warntype_worker_run_job.txt")
    res["model"] = Dict("workers" => NPW, "start_s" => t_add, "load_master_s" => t_load_master, "load_workers_s" => t_load_workers,
                        "cold_master_s" => t_cold_master, "cold_worker_s" => t_cold_w, "warm_median_s" => median(warm),
                        "crn_bit_identical" => same, "q_x0" => q0, "rss_after_warmup_mb" => m0, "rss_series_mb" => mems,
                        "job_s" => t_job, "job_evals" => r_job.n_eval, "job_bytes" => b_job, "result_bytes" => b_res,
                        # 7.3 (2026-09-28)
                        "grid" => BGRID, "simN" => BSIMN, "child_grid" => "$(BCHILD.Na)x$(BCHILD.Nk)x$(BCHILD.Nt)",
                        "parent_extra" => PE_STR, "child_extra" => CE_STR,
                        "julia_threads" => [r.julia_threads for r in resources], "blas_threads" => [r.blas_threads for r in resources],
                        "worker_eval_s" => w_s, "worker_eval_alloc_mb" => w_mb, "worker_eval_gc_s" => w_gc,
                        "job_roundtrip_ms_median" => 1000 * median(rt),
                        "profile_evals" => N_PROF, "profile_self_by_file" => prof === nothing ? Dict{String,Int}() : prof.byfile)
    rmprocs(ws; waitfor = 30)
else
    error("usage: bench_tiktak.jl synthetic|model ...")
end
res["seconds_total"] = time() - T0
res["loadavg_end"] = Sys.loadavg()[1]
open(io -> TOML.print(io, res), joinpath(OUT, "bench_$PART.toml"), "w")
write(joinpath(OUT, "bench_$PART.md"), String(take!(md)))
println("\nwrote ", joinpath(OUT, "bench_$PART.toml"), " and .md  (", round(time() - T0, digits = 1), " s)")
