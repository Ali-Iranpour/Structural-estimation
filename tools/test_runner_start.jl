#!/usr/bin/env julia
# =============================================================================
# test_runner_start.jl -- the 2026-09-29 runner changes of tiktak_fix_plan.md R1/R2 (code/smm/run_smm.jl)
#
#   julia --project=. tools/test_runner_start.jl <targets.toml> [--out <dir>]   (ported from v2, 2026-10-02)
#
# Runs the runner at --quick with tiny budgets (about 1-3 min per run; one run uses 3 workers). Checks:
#   f  a complete run writes an exact [search_vector] (with param_names/param_link) into estimates.toml,
#      the full-precision start Q and the runtime scenarios into its log, the new settings into run_record.toml
#   g  --init-from that estimates.toml starts EXACTLY there (bit for bit); from a file with only [parameters]
#      or with a [search_vector] of other parameter names, it starts from the ROUNDED values and says so
#   h  --expect-start-q: the exact value passes; any other value stops the run before the search
#   j  --bootstrap / --local-ftol / --runtime-evals reach the optimizer identity, the record and the log
#   k  invalid values are refused before any model is loaded
#   l  a resume that changes --bootstrap is refused even with --allow-optimizer-change (search plan);
#      one that changes --local-ftol needs --allow-optimizer-change and becomes a settings epoch
#   m  --bootstrap immediate_mixed on 3 workers: the first three restarts start before any is committed
#   n  the default start-up: fresh = immediate_mixed (since 2026-10-01); a resumed first_alone checkpoint keeps first_alone
# Nothing is written under output/smm_runs/; scratch goes to <app>/temp/ unless --out is given.
# =============================================================================
using Test, TOML
const REPO = normpath(joinpath(@__DIR__, ".."))
const TFILE = abspath(ARGS[1])
argstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i + 1])
const OUT = mkpath(argstr("--out", joinpath(REPO, "temp", "runner_start_" * string(round(Int, time())))))
include(joinpath(REPO, "code", "src", "tiktak.jl"))
const RUNNER = joinpath(REPO, "code", "smm", "run_smm.jl")
function runner(name, args)
    log = joinpath(OUT, name * ".log")
    cmd = Cmd(vcat(collect(Base.julia_cmd().exec), ["--project=$REPO", "--threads=1", "--startup-file=no", RUNNER], args))
    t = @elapsed p = run(pipeline(ignorestatus(cmd); stdout = log, stderr = log))
    println(rpad(name, 24), " exit ", p.exitcode, " in ", round(t / 60, digits = 1), " min"); flush(stdout)
    return p.exitcode, read(log, String)
end
const SMALL = ["--quick", "--serial", "--sobol", "4", "--restarts", "2", "--local-evals", "3", "--skip-polish", "--targets", TFILE]
state(dir) = first(TikTak.read_state_dict(joinpath(dir, "tiktak_state.toml")))
"The value of `key = value` in a run_record.toml (a flat read; the file is TOML)."
record(dir) = TOML.parsefile(joinpath(dir, "run_record.toml"))
"The first value stored under `key` anywhere in a nested Dict (nothing if absent)."
function findkey(d, key)
    d isa AbstractDict || return nothing
    haskey(d, key) && return d[key]
    for v in values(d)
        r = findkey(v, key); r === nothing || return r
    end
    return nothing
end
"The full-precision start Q the runner printed."
start_q(log) = (m = match(r"start Q \(full precision\) = (\S+)", log); m === nothing ? NaN : parse(Float64, m.captures[1]))

# f -- a complete run
df = joinpath(OUT, "f")
cf, lf = runner("f_complete", vcat(SMALL, ["--outdir", df]))
est = cf == 0 ? TOML.parsefile(joinpath(df, "estimates.toml")) : Dict{String,Any}()
zf = Float64.(get(get(est, "search_vector", Dict()), "z", Float64[]))

# g -- exact and rounded starts
dg = joinpath(OUT, "g")
cg, lg = runner("g_exact", vcat(SMALL, ["--init-from", joinpath(df, "estimates.toml"), "--stop-after-restarts", "0", "--outdir", dg]))
params_only = joinpath(OUT, "params_only.toml")
cf == 0 && open(io -> TOML.print(io, Dict("parameters" => est["parameters"])), params_only, "w")
other_names = joinpath(OUT, "other_names.toml")
cf == 0 && open(io -> TOML.print(io, Dict("param_names" => reverse(est["param_names"]), "param_link" => reverse(est["param_link"]),
                                          "search_vector" => Dict("z" => zf), "parameters" => est["parameters"])), other_names, "w")
dg2, dg3 = joinpath(OUT, "g2"), joinpath(OUT, "g3")
cg2, lg2 = runner("g_rounded", vcat(SMALL, ["--init-from", params_only, "--stop-after-restarts", "0", "--outdir", dg2]))
cg3, lg3 = runner("g_other_names", vcat(SMALL, ["--init-from", other_names, "--stop-after-restarts", "0", "--outdir", dg3]))
# a start no default and no 8-decimal rounding can reproduce: the final point moved by i * 1e-10 in coordinate i
# (toward the box's interior), and one moved far outside the box
zp = [zf[i] + (zf[i] > 0 ? -1 : 1) * i * 1e-10 for i in eachindex(zf)]
perturbed, outside = joinpath(OUT, "perturbed.toml"), joinpath(OUT, "outside.toml")
cf == 0 && for (f, z) in ((perturbed, zp), (outside, [i == 1 ? 1e6 : zf[i] for i in eachindex(zf)]))
    open(io -> TOML.print(io, Dict("param_names" => est["param_names"], "param_link" => est["param_link"],
                                   "search_vector" => Dict("z" => z))), f, "w")
end
dg4 = joinpath(OUT, "g4")
cg4, lg4 = runner("g_exact_perturbed", vcat(SMALL, ["--init-from", perturbed, "--stop-after-restarts", "0", "--outdir", dg4]))
cg5, lg5 = runner("g_outside_box", vcat(SMALL, ["--init-from", outside, "--outdir", joinpath(OUT, "g5")]))

# h -- the start gate
qg = start_q(lg)
ch1, lh1 = runner("h_gate_pass", vcat(SMALL, ["--init-from", joinpath(df, "estimates.toml"), "--expect-start-q", repr(qg),
                                              "--stop-after-restarts", "0", "--outdir", joinpath(OUT, "h1")]))
ch2, lh2 = runner("h_gate_fail", vcat(SMALL, ["--init-from", joinpath(df, "estimates.toml"), "--expect-start-q", repr(nextfloat(qg)),
                                              "--stop-after-restarts", "0", "--outdir", joinpath(OUT, "h2")]))

# j -- the new settings
dj = joinpath(OUT, "j")
cj, lj = runner("j_settings", vcat(SMALL, ["--bootstrap", "immediate_mixed", "--local-ftol", "1e-4", "--runtime-evals", "500,423",
                                           "--stop-after-restarts", "0", "--outdir", dj]))
# snapshot now: group l below resumes this same directory and rewrites its record and current settings
const ST_J = cj == 0 ? state(dj) : Dict{String,Any}()
const RR_J = cj == 0 ? record(dj) : Dict{String,Any}()

# k -- refused before the model loads
ck = [runner("k_" * n, vcat(SMALL, a, ["--outdir", joinpath(OUT, "k_" * n)])) for (n, a) in
      (("bootstrap", ["--bootstrap", "first"]), ("ftol0", ["--local-ftol", "0"]), ("evals", ["--runtime-evals", "5,x"]),
       ("expect", ["--expect-start-q", "abc"]))]

# l -- resume rules (from j, paused right after seed selection)
cl1, ll1 = runner("l_bootstrap_changed", vcat(SMALL, ["--bootstrap", "first_alone", "--local-ftol", "1e-4", "--resume", dj,
                                                      "--allow-optimizer-change", "--report-only"]))
cl2, ll2 = runner("l_ftol_changed", vcat(SMALL, ["--bootstrap", "immediate_mixed", "--local-ftol", "1e-3", "--resume", dj, "--report-only"]))
cl3, ll3 = runner("l_ftol_allowed", vcat(SMALL, ["--bootstrap", "immediate_mixed", "--local-ftol", "1e-3", "--resume", dj,
                                                 "--allow-optimizer-change", "--stop-after-restarts", "1"]))

# n -- the default start-up (2026-10-01): a fresh run is immediate_mixed; a checkpoint written under first_alone and resumed
# WITHOUT --bootstrap keeps first_alone (no plan change, no override needed)
dn = joinpath(OUT, "n")
cn1, ln1 = runner("n_first_alone", vcat(SMALL, ["--bootstrap", "first_alone", "--stop-after-restarts", "0", "--outdir", dn]))
cn2, ln2 = runner("n_resume_default", vcat(SMALL, ["--resume", dn, "--stop-after-restarts", "1"]))

# m -- immediate_mixed on the real objective, 3 workers
dm = joinpath(OUT, "m")
# v1 (2026-10-02): most random draws are penalised at the --quick grids, so the run draws until it has 3 VALID
# Sobol' points (--sobol-valid 3, at most 150 attempts); with 6 plain draws it had too few seeds for 3 restarts.
cm, lm = runner("m_immediate_async", ["--quick", "--procs", "3", "--local-procs", "3", "--sobol", "150", "--sobol-valid", "3", "--restarts", "3",
                                      "--local-evals", "3", "--skip-polish", "--targets", TFILE, "--bootstrap", "immediate_mixed",
                                      "--outdir", dm])

@testset "run_smm.jl starts, gates, runtime scenarios and start-up (2026-09-29)" begin
    @testset "f: a complete run" begin
        @test cf == 0
        @test length(zf) == length(get(est, "param_names", [])) == length(get(est, "param_link", [])) > 0
        @test haskey(est, "parameters")
        ck_ = TOML.parsefile(joinpath(df, "checkpoint.toml"))
        @test Float64.(ck_["search_vector"]["z"]) == zf                       # the final point, both files, exactly
        r = record(df)
        @test r["parameters"]["start_precision"] == "smm_start" && r["numerical"]["local_ftol_rel"] == 1e-3
        @test r["numerical"]["bootstrap"] == "immediate_mixed" && r["numerical"]["runtime_evals"] == []   # fresh-run default since 2026-10-01
        @test isfinite(start_q(lf))
        @test occursin("projected runtime -- scenarios", lf) && occursin("empirical: none given", lf)
    end
    @testset "g: exact and rounded starts" begin
        @test cg == 0
        rg = record(dg)
        @test rg["parameters"]["start_precision"] == "exact"
        @test Float64.(rg["parameters"]["start"]) == zf                        # bit for bit
        @test Float64.(findkey(TOML.parsefile(joinpath(dg, "pretest_cache.toml")), "supplied")[1]) == zf   # what TikTak got
        @test cg2 == 0 && record(dg2)["parameters"]["start_precision"] == "rounded"
        @test Float64.(record(dg2)["parameters"]["start"]) != zf               # 8 decimals are not the point
        @test maximum(abs.(Float64.(record(dg2)["parameters"]["start"]) .- zf)) < 1e-7
        @test cg3 == 0 && record(dg3)["parameters"]["start_precision"] == "rounded"
        @test occursin("[search_vector] ignored", lg3)
        @test cg4 == 0 && record(dg4)["parameters"]["start_precision"] == "exact"
        @test zp != zf && Float64.(record(dg4)["parameters"]["start"]) == zp          # bit for bit, not the default
        @test Float64.(findkey(TOML.parsefile(joinpath(dg4, "pretest_cache.toml")), "supplied")[1]) == zp
        @test cg5 != 0 && occursin("outside its box", lg5) && !isfile(joinpath(OUT, "g5", "tiktak_state.toml"))
    end
    @testset "h: the start gate" begin
        @test ch1 == 0 && occursin("start gate passed", lh1)
        @test ch2 != 0 && occursin("--expect-start-q", lh2) && occursin("Nothing was searched", lh2)
        @test record(joinpath(OUT, "h2"))["status"] == "start_gate_failed"
        @test !isfile(joinpath(OUT, "h2", "tiktak_state.toml"))
    end
    @testset "j: settings reach the identity, the record and the log" begin
        @test cj == 0
        o = ST_J["optimizer"]["fields"]
        @test o["bootstrap"] == "immediate_mixed" && o["local"]["ftol_rel"] == 1e-4
        rj = RR_J
        @test rj["numerical"]["bootstrap"] == "immediate_mixed" && rj["numerical"]["local_ftol_rel"] == 1e-4
        @test rj["numerical"]["runtime_evals"] == [500, 423]
        @test occursin("empirical: --runtime-evals 500,423", lj)
    end
    @testset "k: invalid values refused before the model loads" begin
        for (c, l) in ck
            @test c != 0 && !occursin("loading the model", l)
        end
        @test occursin("--bootstrap must be", ck[1][2]) && occursin("--local-ftol", ck[2][2])
        @test occursin("--runtime-evals", ck[3][2]) && occursin("--expect-start-q", ck[4][2])
    end
    @testset "l: resume rules" begin
        @test cl1 != 0 && occursin("bootstrap", ll1) && occursin("refuses to continue", ll1)
        @test cl2 != 0 && occursin("local.ftol_rel", ll2)
        @test cl3 == 0
        s = state(dj)
        @test length(s["epochs"]) == 2 && s["epochs"][1]["local"]["ftol_rel"] == 1e-4 && s["epochs"][2]["local"]["ftol_rel"] == 1e-3
    end
    @testset "n: a resume keeps the saved start-up policy" begin
        @test cn1 == 0 && cn2 == 0 && !occursin("refuses to continue", ln2)
        @test state(dn)["optimizer"]["fields"]["bootstrap"] == "first_alone"
        @test record(dn)["numerical"]["bootstrap"] == "first_alone"
    end
    @testset "m: immediate_mixed on 3 workers" begin
        @test cm == 0
        recs = state(dm)["records"]
        @test sort([r["j"] for r in recs]) == [1, 2, 3]
        @test all(r -> r["commits_at_dispatch"] == 0, recs)                   # all three started before any commit
        @test all(r -> r["incumbent_version"] == 1, recs)
        @test occursin("3 at a time from the start", lm)
    end
end
