#!/usr/bin/env julia
# =============================================================================
# test_smm_resume.jl -- does --resume actually REFUSE an incompatible checkpoint?
#
#     julia --project=.. tools/test_smm_resume.jl <targets.toml>
#
# WHY THIS IS A SEPARATE, SHELLING-OUT TEST. The compatibility logic lives inside
# `resume_from` in run_smm.jl, which is a script, not a module: it cannot be imported and
# called. So the only honest test is to build a checkpoint directory, invoke the runner
# against it, and check that it refuses. Every case below was ACCEPTED before 2026-09-10:
# the checks read `haskey(ck, f) && <mismatch> && refuse(...)`, which passes silently when
# the field is simply absent, and `child_grid` / `sim_n` / `seed` were never compared at
# all -- so a checkpoint from a different numerical problem could hand its `Q_best` to a
# run that could not reproduce it.
#
# Each case must fail EARLY, before any solving, so this costs seconds rather than hours.
# =============================================================================

using Printf, TOML, Test, SHA

const REPO = normpath(joinpath(@__DIR__, ".."))
const TFILE = length(ARGS) >= 1 ? ARGS[1] : error("usage: test_smm_resume.jl <targets.toml>")
const RUNNER = joinpath(REPO, "code", "smm", "run_smm.jl")

# The two content hashes the runner computes for itself. The fixture has to carry the REAL
# values, or every case is refused on the hash before it reaches the check under test --
# which would make this suite pass for the wrong reason.
const SOURCE_FILES_T = ["code/src/child_lifecycle.jl", "code/src/parent_family.jl",
                        "code/smm/moments.jl"]
const SOURCE_SHA_T = bytes2hex(SHA.sha256(
    reduce(vcat, read(joinpath(REPO, f)) for f in SOURCE_FILES_T)))[1:16]
const TARGETS_SHA_T = bytes2hex(SHA.sha256(read(TFILE)))[1:16]

# --quick fixes the grids, and a resume must match them: grid 12, child 12x12x3, simN 300,
# 2 restarts. These come from run_smm.jl's QUICK branch.
function good_checkpoint()
    Dict{String,Any}(
        "stage" => "local", "restarts_done" => 1, "restarts_total" => 2,
        # The SIXTEEN-parameter, seven-TAS-moment specification of 2026-09-11.
        "param_names" => ["phi_2","phi_3","lambda_2","R_0","sigma_1_0","sigma_1_1",
                          "sigma_2_0","sigma_2_1","sigma_4_0","sigma_4_1",
                          "kappa_0","kappa_theta","kappa_ParEd","kappa_terminal",
                          "sigma_eta","sigma_eps"],
        # The CURRENT boxes (2026-09-12: kappa_0 [-3, 1], kappa_ParEd [-1, 0.5], sigma_4_1 top 0.30);
        # the control case must be accepted, so the fixture has to carry them verbatim.
        "param_lo" => [0.01,0.05,0.05,0.5,-4.0,-0.2,-5.0,-0.3,-10.0,-0.05,-3.0,-10.0,-1.0,0.5,0.0,0.1],
        "param_hi" => [20.0,20.0,100.0,100.0,-0.1,0.05,-0.5,0.05,-1.0,0.30,1.0,0.0,0.5,40.0,0.08,2.0],
        "param_link" => ["log","log","log","log","level","level","level","level","level",
                         "level","level","level","level","log","level","log"],
        "targets_sha" => TARGETS_SHA_T,
        "source_sha" => SOURCE_SHA_T,
        "spec_version" => "smm16_tas7_gap_v1",
        "m_psychic" => 6.263396877461691,
        "moment_names" => ["mean_c_p","mean_h_p","mean_t_p_early","mean_t_p_late",
                           "mean_e_p_early","mean_e_p_late","mean_i_c_early","mean_i_c_late",
                           "mean_hc_early","mean_hc_late","k0_complete","kth_ga17_gap",
                           "kpe_g0_c","kpe_g1_c","kterm_x_strict_w99","kse_w_gap","sd_ga17"],
        "child_grid" => "12x12x3", "sim_n" => 300, "seed" => 1234,
        "Q_best" => 1.0, "objective_grid" => 12, "Q_incumbent" => 2.0,
        "grid_search" => 12, "grid_report" => 12, "minutes" => 1.0,
        "search_vector" => Dict("z" => zeros(16)),
    )
end

function write_run(dir, ck)
    mkpath(dir)
    open(joinpath(dir, "checkpoint.toml"), "w") do io
        z = ck["search_vector"]["z"]
        for (k, v) in ck
            k == "search_vector" && continue
            println(io, k, " = ", v isa AbstractString ? "\"$v\"" :
                                  v isa AbstractVector ? "[" * join((x isa AbstractString ? "\"$x\"" : x for x in v), ", ") * "]" : v)
        end
        println(io, "\n[search_vector]")
        println(io, "z = [", join(z, ", "), "]")
    end
    open(joinpath(dir, "seeds.toml"), "w") do io
        println(io, "f_sobol_best = 2.0")
        println(io, "seeds = [[", join(zeros(length(ck["search_vector"]["z"])), ", "), "]]")
    end
    # A resume REFUSES to run without the original frozen target snapshot in the run
    # directory (paths.jl:136), and that refusal fires before any compatibility check. The
    # fixture has to look like a real interrupted run, not just carry a checkpoint.
    cp(TFILE, joinpath(dir, "targets.toml"); force = true)
    return dir
end

"""Run the runner against `dir` and return (accepted::Bool, output)."""
function try_resume(dir)
    # `--serial` runs everything on the master. The flag is `--procs`, not `--workers`;
    # passing an unknown flag is silently ignored, so an earlier version of this test
    # started twenty worker processes per case for nothing.
    cmd = `julia --project=$REPO --threads=1 $RUNNER --resume $dir --report-only --quick --serial --targets $TFILE`
    out = IOBuffer()
    ok = try
        run(pipeline(cmd; stdout = out, stderr = out)); true
    catch
        false
    end
    return ok, String(take!(out))
end

const TMP = mktempdir()
refused(name, mutate!) = begin
    ck = good_checkpoint(); mutate!(ck)
    dir = write_run(joinpath(TMP, replace(name, r"[^a-z0-9]" => "_")), ck)
    ok, out = try_resume(dir)
    if ok || !occursin("refuses to continue", out)
        println("\n---- case $name: ok=$ok ----")
        println(last(out, 1200))
    end
    !ok && occursin("refuses to continue", out)
end

@testset "--resume refuses incompatible checkpoints" begin
    # CONTROL. If the untouched fixture were refused for some unrelated reason, every case
    # below would "pass" while testing nothing. This is the check that the suite is
    # actually exercising the compatibility logic.
    let dir = write_run(joinpath(TMP, "control"), good_checkpoint())
        ok, out = try_resume(dir)
        accepted = ok || !occursin("refuses to continue", out)
        accepted || println("\n---- CONTROL WAS REFUSED ----\n", last(out, 1500))
        @test accepted
    end

    # A missing identity field cannot be verified, and "cannot verify" is not "compatible".
    for f in ("source_sha", "moment_names", "m_psychic", "child_grid", "sim_n",
              "seed", "grid_report", "spec_version")
        @test refused("missing_$f", ck -> delete!(ck, f))
    end
    # A field that is present but describes a different problem.
    @test refused("source", ck -> ck["source_sha"] = "0000000000000000")
    @test refused("spec",   ck -> ck["spec_version"] = "smm10_parent_v1")
    @test refused("spec14", ck -> ck["spec_version"] = "smm14_tas7_centred_v1")   # the previous one
    @test refused("params14", ck -> (for k in ("param_names","param_lo","param_hi","param_link")
                                        ck[k] = ck[k][1:14] end; ck["search_vector"] = Dict("z" => zeros(14))))
    @test refused("centre", ck -> ck["m_psychic"] = 0.0)
    @test refused("moments", ck -> ck["moment_names"] = ck["moment_names"][1:10])
    # The NUMERICAL problem, which was never compared at all: a saved Q_best from one
    # grid, simN or seed is not comparable with values this run computes.
    @test refused("child_grid", ck -> ck["child_grid"] = "30x30x5")
    @test refused("sim_n",      ck -> ck["sim_n"] = 2000)
    @test refused("seed",       ck -> ck["seed"] = 999)
    @test refused("grid_report",ck -> ck["grid_report"] = 30)
end

# ---- the 2026-09-11 flags -------------------------------------------------------
# An unknown flag must be an ERROR before anything is written (it used to be ignored), and
# --init-from must refuse a file whose value sits outside the current box.
@testset "flag validation and --init-from" begin
    # `--temp`: these are throwaway invocations and must not create real run folders.
    common = ["--quick", "--serial", "--report-only", "--temp", "flagtest", "--targets", TFILE]
    runcmd(args...) = begin
        out = IOBuffer()
        cmd = Cmd(["julia", "--project=$REPO", "--threads=1", RUNNER, args..., common...])
        ok = try run(pipeline(cmd; stdout = out, stderr = out)); true catch; false end
        ok, String(take!(out))
    end
    ok, out = runcmd("--workers", "1")
    @test !ok && occursin("unknown flag", out)
    bad = joinpath(TMP, "bad_init.toml")
    write(bad, "[parameters]\nsigma_eta = 0.5\n")               # box top is 0.08
    ok, out = runcmd("--init-from", bad)
    @test !ok && occursin("outside its box", out)
    ok, out = runcmd("--init-from", joinpath(TMP, "nonexistent.toml"))
    @test !ok && occursin("no such file", out)
    # The temp folders those three created are removed; they carry nothing.
    for d in filter(n -> occursin("_temp_flagtest", n), readdir(joinpath(REPO, "output", "smm_runs")))
        rm(joinpath(REPO, "output", "smm_runs", d); recursive = true, force = true)
    end
end
println("\nresume-rejection checks completed")
