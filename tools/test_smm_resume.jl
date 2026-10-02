#!/usr/bin/env julia
# =============================================================================
# test_smm_resume.jl -- does --resume continue what it should and REFUSE what it must?
#
#     julia --project=. tools/test_smm_resume.jl <targets.toml>
#
# (Ported from apps/Structural-estimation-v2 on 2026-10-02 with the TikTak module. This repository has no
# specification switches and no --legacy-import.) About 15 minutes: one small real run, then one runner invocation per
# case (a refusal costs a model load and nothing more; an accepted resume also warms the child).
#
# WHY THIS IS A SHELLING-OUT TEST. The compatibility logic lives in run_smm.jl, a script: the
# only honest test is to build a run directory, invoke the runner against it, and read what it
# does. Each refusal must name the check under test -- a refusal for an unrelated reason fails
# the case -- and every group has a CONTROL that must be accepted, so the suite cannot pass by
# refusing everything.
#
# THE FIXTURES COME FROM THE RUNNER (2026-09-27, tiktak_fix_plan.md steps 1.5 and 4). They
# used to hard-code the parameter names, boxes, links, moments and spec string of one
# historical specification, so under any later specification the control was refused and the
# suite tested nothing. Now:
#   * a CURRENT-format checkpoint (tiktak_state.toml) is produced by a real, tiny run of the
#     runner itself, paused after its first restart (--stop-after-restarts 1);
#   * the LEGACY layouts (checkpoint.toml + seeds.toml; seeds.toml alone) are that run's own
#     derived files with tiktak_state.toml removed -- always REFUSED here (Ali, 2026-10-02);
#   * identity values come from `run_smm.jl --print-identity`.
#
# Nothing here writes under output/smm_runs/ (a run there would re-point
# output/smm_runs/latest, which may belong to a running estimation).
# =============================================================================

using Printf, TOML, Test

const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "code", "src", "tiktak.jl"))       # the checksum writer, to re-sign mutated fixtures
const TFILE = length(ARGS) >= 1 ? abspath(ARGS[1]) : error("usage: test_smm_resume.jl <targets.toml>")
const RUNNER = joinpath(REPO, "code", "smm", "run_smm.jl")
const TMP = mktempdir(; prefix = "smm_resume_test_")
const JULIA = Base.julia_cmd()
# the base run's settings: every resume below must repeat them (the optimizer identity)
# v1 (2026-10-02): on this model most random draws are penalised at the --quick grids (4 valid of 60), so the base run
# draws until it has 2 VALID Sobol' points (--sobol-valid 2, at most 80 attempts) -- with v2's 4 plain draws it had
# one usable seed and could not pause after restart 1 of 2. The checks below are unchanged.
const BUDGET = ["--quick", "--serial", "--sobol", "80", "--sobol-valid", "2", "--restarts", "2", "--local-evals", "2",
                "--skip-polish", "--targets", TFILE]

"Run the runner with `args`; return (exit_ok::Bool, output::String)."
function runner(args::Vector{String})
    out = IOBuffer()
    cmd = Cmd(vcat(collect(JULIA.exec), ["--project=$REPO", "--threads=1", "--startup-file=no", RUNNER], args))
    ok = try
        run(pipeline(cmd; stdout = out, stderr = out)); true
    catch
        false
    end
    return ok, String(take!(out))
end

const IDENTITY = let (ok, out) = runner(["--print-identity", "--quick", "--serial", "--targets", TFILE,
                                         "--outdir", joinpath(TMP, "identity")])
    ok || error("run_smm.jl --print-identity failed:\n" * last(out, 2000))
    i = findfirst("# --print-identity", out)
    i === nothing && error("no identity block in the runner output:\n" * last(out, 2000))
    TOML.parse(out[last(i)+1:end] |> s -> s[findfirst('\n', s)+1:end])
end

# ---- the base run: a real, tiny search paused after restart 1 ------------------------------
const BASE = joinpath(TMP, "base")
let (ok, out) = runner(vcat(BUDGET, ["--stop-after-restarts", "1", "--outdir", BASE]))
    ok && occursin("PAUSED after 1 of 2 restarts", out) || error("the base run did not pause as planned:\n" * last(out, 3000))
    for f in ("tiktak_state.toml", "pretest_cache.toml", "checkpoint.toml", "seeds.toml", "restarts.csv", "run_record.toml")
        isfile(joinpath(BASE, f)) || error("the base run wrote no $f")
    end
end
"A copy of the base run directory, to mutate."
function fresh_copy(name)
    dir = joinpath(TMP, replace(name, r"[^A-Za-z0-9]" => "_"))
    rm(dir; recursive = true, force = true)
    cp(BASE, dir)
    return dir
end
"Rewrite tiktak_state.toml after `mutate!(dict)`, with a VALID checksum (only the change under test differs)."
function mutate_state!(dir, mutate!)
    path = joinpath(dir, "tiktak_state.toml")
    d = TOML.parse(first(TikTak.read_checksummed(path)))
    mutate!(d)
    io = IOBuffer(); TOML.print(io, d; sorted = true)
    TikTak.write_checksummed(path, String(take!(io)); keep_previous = false)
    rm(path * ".prev"; force = true)
    return dir
end
"BUDGET with `flag` set to `value` (replaced, not duplicated: the runner reads the first occurrence)."
function with_flag(flag::String, value::String)
    b = copy(BUDGET)
    i = findfirst(==(flag), b)
    i === nothing ? append!(b, [flag, value]) : (b[i + 1] = value)
    return b
end
resume_run(dir; budget = BUDGET, extra = String[]) = runner(vcat(budget, extra, ["--resume", dir, "--report-only"]))

"The resume is refused, and the refusal names `expect` (the check under test)."
function refused(name, expect, setup; budget = BUDGET, extra = String[])
    dir = setup(fresh_copy(name))
    ok, out = resume_run(dir; budget = budget, extra = extra)
    right = !ok && occursin("refuses to continue", out) && occursin(expect, out)
    right || println("\n---- case $name: ok=$ok, expected \"$expect\" ----\n", last(out, 1500))
    return right
end
"The resume is accepted (it reaches --report-only's exit), and its output contains `expect_text`."
function accepted(name, setup; budget = BUDGET, extra = String[], expect_text = "")
    dir = setup(fresh_copy(name))
    ok, out = resume_run(dir; budget = budget, extra = extra)
    right = ok && !occursin("refuses to continue", out) && occursin("--report-only: stopping before the search", out) &&
            (isempty(expect_text) || occursin(expect_text, out))
    right || println("\n---- ACCEPT case $name: ok=$ok ----\n", last(out, 2000))
    return right
end

const FAILED = String[]
"Run a named testset; a failure is recorded and the next testset still runs."
function tset(body, name)
    try
        @testset "$name" begin
            body()
        end
    catch e
        e isa Test.TestSetException || rethrow()
        push!(FAILED, name)
    end
end

tset("current-format checkpoint (tiktak_state.toml)") do
    @test accepted("control", identity; expect_text = "continuing")
    # a missing identity field cannot be verified, and "cannot verify" is not "compatible"
    for f in ("source_sha", "moment_names", "m_psychic", "child_grid", "sim_n", "seed", "grid_report",
              "spec_version", "targets_sha", "param_names")
        @test refused("missing_$f", "no `$f` field", dir -> mutate_state!(dir, d -> delete!(d["objective"]["fields"], f)))
    end
    for (name, expect, mutate) in (
            ("source", "model source changed", fs -> fs["source_sha"] = "0000000000000000"),
            ("spec", "specification changed", fs -> fs["spec_version"] = "smm10_parent_v1"),
            ("targets", "targets file changed", fs -> fs["targets_sha"] = "0000000000000000"),
            ("params", "parameter set changed", fs -> fs["param_names"] = reverse(fs["param_names"])),
            ("param_lo", "param_lo changed", fs -> fs["param_lo"][1] -= 0.01),
            ("param_hi", "param_hi changed", fs -> fs["param_hi"][end] += 0.01),
            ("links", "parameter links changed", fs -> fs["param_link"][1] = fs["param_link"][1] == "log" ? "level" : "log"),
            ("grid_extra", "overrides changed", fs -> fs["grid_extra"] = "parent: Na=99 | child: "),
            ("centre", "centring changed", fs -> fs["m_psychic"] = 0.0),
            ("moments", "moment set changed", fs -> fs["moment_names"] = fs["moment_names"][1:end-1]),
            ("child_grid", "child grid changed", fs -> fs["child_grid"] = "30x30x5"),
            ("sim_n", "simulated households changed", fs -> fs["sim_n"] += 1),
            ("seed", "seed changed", fs -> fs["seed"] += 1),
            ("grid_report", "report grid changed", fs -> fs["grid_report"] += 1),
            ("grid_search", "searched at grid", fs -> fs["grid_search"] += 1))
        @test refused(name, expect, dir -> mutate_state!(dir, d -> mutate(d["objective"]["fields"])))
    end
    # a different restart count is a new search (it changes the mixing denominator)
    @test refused("restarts", "--restarts", identity; budget = with_flag("--restarts", "3"))
    # a different optimizer setting is refused by the TikTak module, naming the setting ...
    @test refused("local_evals", "local.maxeval", identity; budget = with_flag("--local-evals", "3"))
    # ... unless explicitly allowed, which is then recorded
    @test accepted("local_evals_allowed", identity; budget = with_flag("--local-evals", "3"),
                   extra = ["--allow-optimizer-change"])
    # a damaged newest generation: the previous one is used, and the log says so
    @test accepted("truncated_current", dir -> (p = joinpath(dir, "tiktak_state.toml");
                                               write(p, read(p, String)[1:200]); dir);
                   expect_text = "previous")
    # nothing valid left: a clear refusal, not a crash in the middle
    @test refused("no_valid_generation", "cannot be read",
                  dir -> (for f in ("tiktak_state.toml", "tiktak_state.toml.prev")
                              p = joinpath(dir, f); isfile(p) && write(p, "garbage")
                          end; dir))
end

tset("legacy checkpoints (before the 2026-10-02 port): always refused here") do
    legacy(dir) = (for f in readdir(dir)
                       (startswith(f, "tiktak_state.toml") || startswith(f, "pretest_cache.toml")) && rm(joinpath(dir, f))
                   end; dir)
    # no --legacy-import in this repository (Ali, 2026-10-02): an old run warm-starts a new one with --init-from
    @test refused("legacy_plain", "cannot be resumed in this repository", legacy)
    seeds_only(dir) = (legacy(dir); rm(joinpath(dir, "checkpoint.toml")); dir)
    @test refused("legacy_seeds_only", "cannot be resumed in this repository", seeds_only)
    # the import flag does not exist: an unknown flag, refused before anything is written
    let dir = legacy(fresh_copy("legacy_flag"))
        ok, out = resume_run(dir; extra = ["--legacy-import"])
        @test !ok && occursin("unknown flag", out)
    end
end

# ---- flag validation ------------------------------------------------------------------
# An unknown flag must be an ERROR before anything is written, and --init-from must refuse
# a file whose value sits outside the current box.
tset("flag validation and --init-from") do
    common(tag) = ["--quick", "--serial", "--report-only", "--targets", TFILE, "--outdir", joinpath(TMP, "flags_" * tag)]
    ok, out = runner(vcat(["--workers", "1"], common("unknown")))
    @test !ok && occursin("unknown flag", out)
    bad = joinpath(TMP, "bad_init.toml")
    let i = findfirst(==("level"), IDENTITY["param_link"]), name = IDENTITY["param_names"][i]
        write(bad, "[parameters]\n$name = $(IDENTITY["param_hi"][i] + 1.0)\n")   # above its box
    end
    ok, out = runner(vcat(["--init-from", bad], common("box")))
    @test !ok && occursin("outside its box", out)
    ok, out = runner(vcat(["--init-from", joinpath(TMP, "nonexistent.toml")], common("missing")))
    @test !ok && occursin("no such file", out)
    ok, out = runner(vcat(["--local-evals", "0"], common("zero")))
    @test !ok && occursin("must be >= 1", out)
    ok, out = runner(vcat(["--preset", "final"], common("preset")))
    @test !ok && occursin("unknown preset", out)
end
println("\nresume checks completed (fixtures from the runner itself; scratch $TMP)")
isempty(FAILED) ? println("ALL TESTSETS PASSED") : (println("FAILED TESTSETS: ", join(FAILED, ", ")); exit(1))
