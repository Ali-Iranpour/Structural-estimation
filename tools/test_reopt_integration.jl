#!/usr/bin/env julia
# =============================================================================
# test_reopt_integration.jl -- tools/reopt.jl end to end on the REAL objective
# (2026-09-28, tiktak_fix_plan.md follow-ups 2, 3 and 5)
#
#   julia --project=. tools/test_reopt_integration.jl <targets.toml> <start.toml> [--out dir]   (ported from v2, 2026-10-02)
#
# Small grids (--grid 10 --simN 500) and tiny budgets: an integration test, not an estimation.
# The pure identity/settings helpers are tested without the model in tools/test_reopt_identity.jl.
#
#   a  TikTak mode, serial, --ftol-rel 1e-4 --init-step 0.05, paused after restart 1
#   b  --resume of a: continues from tiktak_state.toml (there is no checkpoint_best.toml); results.toml
#      reports THAT route, the settings that actually ran and a validated complete search
#   c  a target mean changed IN THE SAME FILE, then --resume: refused, naming targets_sha, before the
#      child warm-up (nothing is evaluated)
#   d  --local-alg in TikTak mode: refused before any evaluation
#   e  TikTak mode on 2 worker processes: one Julia thread each, asynchronous restarts committed once
# =============================================================================
using Test, TOML, Printf

const REPO = normpath(joinpath(@__DIR__, ".."))
length(ARGS) >= 2 || error("usage: test_reopt_integration.jl <targets.toml> <start.toml> [--out dir]")
const TFILE = abspath(ARGS[1]); const START = abspath(ARGS[2])
argstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i + 1])
const OUT = mkpath(argstr("--out", mktempdir(; prefix = "reopt_integration_", cleanup = false)))
const COMMON = ["--grid", "10", "--simN", "500", "--local-evals", "3", "--polish-evals", "0", "--start", START]

"Run reopt.jl; its output goes to <OUT>/<name>.log. Returns (exit code, log text)."
function reopt(name, args)
    log = joinpath(OUT, name * ".log")
    cmd = Cmd(vcat(collect(Base.julia_cmd().exec), ["--project=$REPO", "--threads=1", "--startup-file=no",
                                                     joinpath(REPO, "tools", "reopt.jl")], args))
    t = @elapsed p = run(pipeline(ignorestatus(cmd); stdout = log, stderr = log))
    println(@sprintf("  %-4s exit %d in %.1f min", name, p.exitcode, t / 60)); flush(stdout)
    return p.exitcode, read(log, String)
end
results(dir) = TOML.parsefile(joinpath(dir, "results.toml"))
state(dir) = TOML.parse(first(TikTak.read_checksummed(joinpath(dir, "tiktak_state.toml"))))
include(joinpath(REPO, "code", "src", "tiktak.jl"))

println("reopt integration runs -> $OUT")
da = joinpath(OUT, "run_ab")
ca, la = reopt("a", vcat(COMMON, ["--targets", TFILE, "--sobol", "6", "--restarts", "3", "--procs", "1",
                                  "--ftol-rel", "1e-4", "--init-step", "0.05", "--stop-after-restarts", "1", "--outdir", da]))
const STATE_A = ca == 0 ? state(da) : Dict{String,Any}()     # b resumes the same folder: read a's state NOW
cb, lb = reopt("b", vcat(COMMON, ["--targets", TFILE, "--sobol", "6", "--restarts", "3", "--procs", "1",
                                  "--ftol-rel", "1e-4", "--init-step", "0.05", "--resume", "--outdir", da]))
# c: a private copy of the targets, a run on it, then a changed mean in that same file
dc = joinpath(OUT, "run_c"); mkpath(dc)
tcopy = joinpath(dc, "targets_copy.toml"); cp(TFILE, tcopy; force = true)
cc1, lc1 = reopt("c1", vcat(COMMON, ["--targets", tcopy, "--sobol", "4", "--restarts", "2", "--procs", "1",
                                     "--stop-after-restarts", "0", "--outdir", dc]))
let raw = TOML.parsefile(tcopy)
    raw["mean_h_p"]["mean"] = raw["mean_h_p"]["mean"] * 1.01
    open(io -> TOML.print(io, raw), tcopy, "w")
end
cc2, lc2 = reopt("c2", vcat(COMMON, ["--targets", tcopy, "--sobol", "4", "--restarts", "2", "--procs", "1",
                                     "--resume", "--outdir", dc]))
cd_, ld = reopt("d", vcat(COMMON, ["--targets", TFILE, "--sobol", "4", "--restarts", "2", "--procs", "1",
                                   "--local-alg", "bobyqa", "--outdir", joinpath(OUT, "run_d")]))
de = joinpath(OUT, "run_e")
ce, le = reopt("e", vcat(COMMON, ["--targets", TFILE, "--sobol", "8", "--restarts", "4", "--procs", "2", "--outdir", de]))

@testset "reopt.jl on the real objective" begin
    @testset "a: paused after restart 1" begin
        @test ca == 0 && occursin("objective id", la)
        @test get(STATE_A, "status", "") == "paused" && length(get(STATE_A, "records", [])) == 1
    end
    @testset "b: resumed from tiktak_state.toml; the record says what ran" begin
        @test cb == 0 && occursin("TikTak preflight passed", lb)
        @test !isfile(joinpath(da, "checkpoint_best.toml"))
        r = results(da); res = r["result"][1]
        @test r["resumed"] == true && startswith(r["resume_mode"], "tiktak_state") && !haskey(r, "evals_before_resume")
        @test !occursin("restart_from_best", r["resume_mode"]) && startswith(r["resume_verification"], "verified")
        @test r["resume_semantics"] == "serial_exact" && res["resume_semantics"] == "serial_exact"
        @test r["ftol_rel"] == 1e-4 && r["init_step"] == 0.05                     # the flags reached TikTak ...
        @test r["effective_settings"]["local_ftol_rel"] == 1e-4 && r["effective_settings"]["local_initial_step"] == 0.05
        st = state(da)
        @test st["optimizer"]["fields"]["local"]["ftol_rel"] == 1e-4               # ... and its optimizer identity
        @test res["search_budget_complete"] == true && res["run_status"] == "complete"
        @test sort([x["j"] for x in st["records"]]) == [1, 2, 3] && isempty(st["inflight"])
        @test r["objective_id"] == st["objective"]["id"] && st["objective"]["fields"]["identity_version"] == 2
    end
    @testset "c: a changed target mean in the same file is refused before the warm-up" begin
        @test cc1 == 0
        @test cc2 != 0 && occursin("resume refused", lc2) && occursin("targets_sha", lc2)
        @test !occursin("warming the child cache", lc2)
    end
    @testset "d: --local-alg is refused in TikTak mode" begin
        @test cd_ != 0 && occursin("--local-alg applies to the pure-local mode", ld) && !occursin("warming the child cache", ld)
    end
    @testset "e: two worker processes" begin
        @test ce == 0 && occursin("Julia threads 1,", le)
        st = state(de); recs = st["records"]
        @test sort([x["j"] for x in recs]) == [1, 2, 3, 4] && isempty(st["inflight"]) && all(x -> x["worker"] != 1, recs)
        @test results(de)["result"][1]["search_budget_complete"] == true
    end
end
println("\nreopt integration logs and runs: $OUT")
