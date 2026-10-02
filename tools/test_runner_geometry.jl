#!/usr/bin/env julia
# =============================================================================
# test_runner_geometry.jl -- the search-geometry flags of code/smm/run_smm.jl (2026-09-29, for the settings pilot E3)
#
#   julia --project=. tools/test_runner_geometry.jl <targets.toml> [--out <dir>]
#
# Ported from apps/Structural-estimation-v2 on 2026-10-02 with the TikTak module. v2's check d (an old checkpoint,
# written before these flags existed, resumes without them) is not part of this copy: no checkpoint written before the
# port can be resumed here (Ali, 2026-10-02), and v2 skips d for the same reason.
#
# Runs the runner at --quick, serially, with tiny budgets (≈ 2 min per run). Checks:
#   a  the flags reach the optimizer: tiktak_state.toml's optimizer identity and run_record.toml carry them
#   b  a resume with a DIFFERENT geometry is refused, naming the setting; the same geometry resumes; a different one
#      with --allow-optimizer-change resumes and is recorded as a settings epoch
#   c  invalid combinations are refused before anything is evaluated
#   e  a complete run (θ-shrinking steps) writes the settings it RAN with at the end of run_record.toml
# Nothing is written under output/smm_runs/; scratch goes to <app>/temp/ unless --out is given.
# =============================================================================
using Test, TOML
const REPO = normpath(joinpath(@__DIR__, ".."))
const TFILE = abspath(ARGS[1])
argstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i + 1])
# scratch inside the application (Ali, 2026-09-29), kept after the run: <app>/temp/runner_geometry_<stamp>/ unless --out
const OUT = mkpath(argstr("--out", joinpath(REPO, "temp", "runner_geometry_" * string(round(Int, time())))))
include(joinpath(REPO, "code", "src", "tiktak.jl"))
const RUNNER = joinpath(REPO, "code", "smm", "run_smm.jl")
function runner(name, args)
    log = joinpath(OUT, name * ".log")
    cmd = Cmd(vcat(collect(Base.julia_cmd().exec), ["--project=$REPO", "--threads=1", "--startup-file=no", RUNNER], args))
    t = @elapsed p = run(pipeline(ignorestatus(cmd); stdout = log, stderr = log))
    println(rpad(name, 22), " exit ", p.exitcode, " in ", round(t / 60, digits = 1), " min"); flush(stdout)
    return p.exitcode, read(log, String)
end
const SMALL = ["--quick", "--serial", "--sobol", "4", "--restarts", "2", "--local-evals", "3", "--skip-polish", "--targets", TFILE]
dir = joinpath(OUT, "run")
state() = first(TikTak.read_state_dict(joinpath(dir, "tiktak_state.toml")))

ca, la = runner("a_fresh", vcat(SMALL, ["--normalize", "--local-init-step", "0.05", "--stop-after-restarts", "0", "--outdir", dir]))
st_a = ca == 0 ? state() : Dict{String,Any}()
rr = ca == 0 ? read(joinpath(dir, "run_record.toml"), String) : ""
cb1, lb1 = runner("b_changed_refused", vcat(SMALL, ["--normalize", "--local-init-step", "0.1", "--resume", dir, "--report-only"]))
cb2, lb2 = runner("b_same_accepted", vcat(SMALL, ["--normalize", "--local-init-step", "0.05", "--resume", dir, "--report-only"]))
cb3, lb3 = runner("b_changed_allowed", vcat(SMALL, ["--normalize", "--local-init-step", "0.1", "--resume", dir,
                                                    "--allow-optimizer-change", "--stop-after-restarts", "1"]))
cc1, lc1 = runner("c_theta_no_step", vcat(SMALL, ["--local-step-schedule", "theta_shrink", "--outdir", joinpath(OUT, "c1")]))
cc2, lc2 = runner("c_step_too_big", vcat(SMALL, ["--local-init-step", "1.5", "--outdir", joinpath(OUT, "c2")]))
# e: a COMPLETE run -- the end-of-run record (the settings the search ran with) is written only when a run finishes
de = joinpath(OUT, "run_e")
ce, le = runner("e_complete_run", vcat(SMALL, ["--local-init-step", "0.2", "--local-step-schedule", "theta_shrink",
                                               "--local-step-min", "0.05", "--outdir", de]))
rre = ce == 0 ? read(joinpath(de, "run_record.toml"), String) : ""

@testset "run_smm.jl search-geometry flags" begin
    @testset "a: the flags reach the optimizer and the record" begin
        @test ca == 0
        o = get(get(st_a, "optimizer", Dict()), "fields", Dict())
        @test get(o, "normalize", nothing) == true && get(get(o, "local", Dict()), "initial_step", nothing) == 0.05
        @test get(get(o, "local", Dict()), "step_schedule", nothing) == "fixed"
        @test occursin("normalize    = true", rr) && occursin("local_initial_step = 0.05", rr)
    end
    @testset "b: a changed geometry is a changed optimizer" begin
        @test cb1 != 0 && occursin("refuses to continue", lb1) && occursin("local.initial_step", lb1)
        @test cb2 == 0 && occursin("--report-only: stopping before the search", lb2)
        @test cb3 == 0
        s = state()
        @test length(s["epochs"]) == 2 && s["epochs"][1]["local"]["initial_step"] == 0.05 && s["epochs"][2]["local"]["initial_step"] == 0.1
        @test s["resume_semantics"] == "changed_optimizer" && any(e -> e["kind"] == "optimizer_change", s["events"])
    end
    @testset "c: invalid combinations refused before any model is loaded" begin
        @test cc1 != 0 && occursin("theta_shrink needs --local-init-step", lc1) && !occursin("loading the model", lc1)
        @test cc2 != 0 && occursin("must be a fraction of the box width", lc2) && !occursin("loading the model", lc2)
    end
    @testset "e: a complete run records the settings it ran with" begin
        @test ce == 0 && occursin("run_status   = \"complete\"", rre)
        @test occursin("initial_step=0.2 theta_shrink", rre) && occursin("settings_epochs = 1", rre)
        @test occursin("effective_normalize = false", rre) && occursin("local_step_min = 0.05", rre)
    end
end
println("\nlogs: $OUT")
