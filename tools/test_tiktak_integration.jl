#!/usr/bin/env julia
# =============================================================================
# test_tiktak_integration.jl -- the TikTak changes on the REAL SMM objective (plan steps 8, 10)
#
#     julia --project=. tools/test_tiktak_integration.jl <targets.toml> [--out <dir>] [--only a,b,...]
#
# Runs code/smm/run_smm.jl at --quick grids (ported from v2 on 2026-10-02) and checks
# what each run wrote. Development tests: budget-limited answers are EXPECTED and are not
# failures; exceptions, invalid point/value pairs, lost or duplicated restarts, wrong
# provenance and falsely claimed convergence are.
#
#   serial   the plan's cheap serial smoke (64 Sobol', 5 restarts, 60 evals, no polish)
#   pmap     the same with the pre-testing on 2 worker processes and the restarts serial:
#            must reproduce `serial` exactly (the objective is the same function on every
#            process -- common random numbers, plan J4)
#   async    the same with the restarts asynchronous on 2 workers: every restart committed
#            once, on the workers, overlapping
#   resume   `serial` paused after 2 restarts and resumed: must reproduce `serial` exactly
#   aresume  asynchronous, paused after 2 restarts and resumed: completes, each restart once
#   refine   a coarse search grid with a full-grid refinement: the reported point's evidence
#            is on the full-grid objective
#
# Independent runs go 3 at a time (at most 7 processes). Nothing is written under
# output/smm_runs/ (that would re-point output/smm_runs/latest).
# =============================================================================

using TOML, Printf, Test

const REPO = normpath(joinpath(@__DIR__, ".."))
const TFILE = length(ARGS) >= 1 && !startswith(ARGS[1], "--") ? abspath(ARGS[1]) :
              error("usage: test_tiktak_integration.jl <targets.toml> [--out dir] [--only names]")
argstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i + 1])
const OUT = mkpath(argstr("--out", mktempdir(; prefix = "tiktak_integration_", cleanup = false)))
const ONLY = let o = argstr("--only", ""); isempty(o) ? nothing : Set(split(o, ',')) end
want(n) = ONLY === nothing || n in ONLY
const RUNNER = joinpath(REPO, "code", "smm", "run_smm.jl")
# memo 19 (Ali, 2026-10-02): most draws are penalised at the --quick grids (3 valid of 64), so the smoke fixture draws
# until it has 5 VALID Sobol' points (--sobol-valid 5, at most 400 attempts) instead of 64 plain draws; checks unchanged.
const SMOKE = ["--quick", "--sobol", "400", "--sobol-valid", "5", "--restarts", "5", "--local-evals", "60", "--skip-polish", "--targets", TFILE]
# v1 (2026-10-02): --sobol-valid 4 (at most 200 attempts) -- most random draws are penalised at the --quick grids,
# and 16 plain draws gave too few seeds for 4 restarts. The checks are unchanged.
const SMALL = ["--quick", "--sobol", "200", "--sobol-valid", "4", "--restarts", "4", "--local-evals", "15", "--skip-polish", "--targets", TFILE]

"Run the runner; the log goes to <OUT>/<name>.log. Returns (ok, seconds)."
function runsmm(name, args)
    log = joinpath(OUT, name * ".log")
    cmd = Cmd(vcat(collect(Base.julia_cmd().exec), ["--project=$REPO", "--threads=1", "--startup-file=no", RUNNER], args))
    t = @elapsed ok = try
        run(pipeline(cmd; stdout = log, stderr = log, append = true)); true
    catch
        false
    end
    println(@sprintf("  %-10s %s in %.1f min", name, ok ? "finished" : "FAILED", t / 60)); flush(stdout)
    return ok, t
end
dir(name) = joinpath(OUT, "run_" * name)
logtext(name) = read(joinpath(OUT, name * ".log"), String)
estimates(name) = TOML.parsefile(joinpath(dir(name), "estimates.toml"))
state(name) = TOML.parse(first(TikTak_read(joinpath(dir(name), "tiktak_state.toml"))))
include(joinpath(REPO, "code", "src", "tiktak.jl"))
TikTak_read(p) = TikTak.read_checksummed(p)
"restarts.csv rows without the worker and timing columns (they legitimately differ)."
function restart_rows(name)
    lines = readlines(joinpath(dir(name), "restarts.csv"))
    hdr = split(lines[1], ',')
    keep = [i for (i, h) in enumerate(hdr) if !(h in ("worker", "elapsed_s", "commit_seq"))]
    return [join(split(l, ',')[keep], ',') for l in lines[2:end]]
end

println("integration runs -> $OUT")
results = Dict{String,Any}()
jobs = Pair{String,Vector{String}}[]
want("serial") && push!(jobs, "serial" => vcat(SMOKE, ["--serial", "--outdir", dir("serial")]))
want("pmap")   && push!(jobs, "pmap" => vcat(SMOKE, ["--procs", "2", "--local-mode", "serial", "--outdir", dir("pmap")]))
# first_alone explicitly: this run checks that restart 1 runs alone exactly as in the serial run (the runner's default
# start-up has been immediate_mixed since 2026-10-01; tools/test_runner_start.jl covers that policy)
want("async")  && push!(jobs, "async" => vcat(SMOKE, ["--procs", "2", "--bootstrap", "first_alone", "--outdir", dir("async")]))
want("refine") && push!(jobs, "refine" => ["--quick", "--serial", "--grid", "10", "--sobol", "8", "--restarts", "2",
                                           "--local-evals", "12", "--polish-evals", "12", "--refine", "30",
                                           "--targets", TFILE, "--outdir", dir("refine")])
# the pause/resume pairs run their two halves in order
function pair_resume(name, args)
    ok1, t1 = runsmm(name * "_a", vcat(args, ["--stop-after-restarts", "2", "--outdir", dir(name)]))
    ok2, t2 = runsmm(name * "_b", vcat(args, ["--resume", dir(name)]))
    return ok1 && ok2, t1 + t2
end
tasks = Dict{String,Task}()
sem = Base.Semaphore(3)
for (name, args) in jobs
    tasks[name] = @async Base.acquire(() -> runsmm(name, args), sem)
end
want("resume")  && (tasks["resume"]  = @async Base.acquire(() -> pair_resume("resume", vcat(SMOKE, ["--serial"])), sem))
want("aresume") && (tasks["aresume"] = @async Base.acquire(() -> pair_resume("aresume", vcat(SMALL, ["--procs", "2"])), sem))
for (k, t) in tasks; results[k] = fetch(t); end

@testset "TikTak on the real SMM objective" begin
    if haskey(results, "serial")
        @testset "serial smoke" begin
            @test results["serial"][1]
            e = estimates("serial"); st = state("serial")
            @test e["run_status"] == "complete" && e["execution_ok"] && e["candidate_valid"]
            @test length(restart_rows("serial")) == 5 && st["status"] == "complete" && isempty(st["inflight"])
            @test e["n_restarts_effective"] == 5 && isfinite(e["Q_final"])
            @test occursin("verdict:", logtext("serial"))
        end
    end
    if haskey(results, "pmap") && haskey(results, "serial")
        @testset "2-process pre-testing reproduces the serial run" begin
            @test results["pmap"][1]
            @test restart_rows("pmap") == restart_rows("serial")
            @test estimates("pmap")["Q_final"] == estimates("serial")["Q_final"]
            @test state("pmap")["seeds"]["x"] == state("serial")["seeds"]["x"]
        end
    end
    if haskey(results, "async")
        @testset "2-process asynchronous restarts" begin
            @test results["async"][1]
            st = state("async"); recs = st["records"]
            @test sort([r["j"] for r in recs]) == collect(1:5) && isempty(st["inflight"])
            @test all(r -> r["worker"] != 1, recs) && length(unique(r["worker"] for r in recs)) == 2
            overlap = any(a["commits_at_dispatch"] < b["commit_seq"] && b["commits_at_dispatch"] < a["commit_seq"]
                          for a in recs, b in recs if a["j"] < b["j"])
            @test overlap
            e = estimates("async")
            @test e["run_status"] == "complete" && e["execution_ok"]
            if haskey(results, "serial")
                @test state("async")["seeds"]["x"] == state("serial")["seeds"]["x"]   # same pool
                @test restart_rows("async")[1] == restart_rows("serial")[1]           # restart 1 runs alone
            end
        end
    end
    if haskey(results, "resume") && haskey(results, "serial")
        @testset "serial pause after 2 + resume = the uninterrupted run" begin
            @test results["resume"][1]
            @test occursin("PAUSED after 2 of 5 restarts", logtext("resume_a"))
            @test restart_rows("resume") == restart_rows("serial")
            @test estimates("resume")["Q_final"] == estimates("serial")["Q_final"]
            st = state("resume")
            @test st["resume_semantics"] == "serial_exact" && length(st["segments"]) == 2
            @test st["counters"]["evals_pretest"] + st["counters"]["evals_local"] ==
                  state("serial")["counters"]["evals_pretest"] + state("serial")["counters"]["evals_local"]
        end
    end
    if haskey(results, "aresume")
        @testset "asynchronous pause after 2 + resume" begin
            @test results["aresume"][1]
            st = state("aresume")
            @test sort([r["j"] for r in st["records"]]) == collect(1:4) && isempty(st["inflight"])
            @test st["resume_semantics"] == "async_continuation" && estimates("aresume")["run_status"] == "complete"
        end
    end
    if haskey(results, "refine")
        @testset "coarse search + full-grid refinement" begin
            @test results["refine"][1]
            e = estimates("refine")
            @test e["objective_id_search"] != e["objective_id_report"]
            @test e["refine_status"] in ("improved", "no_improvement")
            @test e["point_origin"] in ("refine", "reevaluated")        # never a coarse-grid certificate
            @test e["local_converged"] == (e["point_origin"] == "refine" && TikTak.ret_class(Symbol(e["refine_ret"])) === :converged ||
                                           e["verification"] == "verified")
        end
    end
end
open(joinpath(OUT, "integration_summary.toml"), "w") do io
    TOML.print(io, Dict(k => Dict("ok" => v[1], "seconds" => v[2]) for (k, v) in results))
end
println("\nintegration logs and runs: $OUT")
