#!/usr/bin/env julia
# =============================================================================
# test_reopt_identity.jl -- the objective identity of tools/reopt.jl (2026-09-28,
# tiktak_fix_plan.md follow-up 2; finding 17 of tiktak_problems.md)
#
#     julia --project=. --threads=1 tools/test_reopt_identity.jl
#
# No economic model: the identity builder and the resume check are pure (tools/reopt_identity.jl),
# and the "objective" is a counting synthetic function whose checkpoints carry the identity, so
# each case shows that a changed economic input is refused BEFORE any evaluation and that an
# unchanged one resumes. Temporary target files are written and mutated in place (same path).
# =============================================================================
using Test, TOML

const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "code", "src", "tiktak.jl"))
include(joinpath(REPO, "tools", "reopt_identity.jl"))
include(joinpath(REPO, "tools", "reopt_settings.jl"))

# ---- follow-up 5 (finding 15): the solver flags reach the optimizer, and the record says what ran ----
@testset "reopt solver settings and resume metadata" begin
    # pure-local mode: the historical defaults, and the flags as given
    @test reopt_local_settings() == (alg = :LN_NELDERMEAD, ftol_rel = 1e-4, init_step = 0.0)
    @test reopt_local_settings(; ftol_rel = 1e-6, init_step = 0.05, local_alg = "bobyqa") == (alg = :LN_BOBYQA, ftol_rel = 1e-6, init_step = 0.05)
    @test_throws ErrorException reopt_local_settings(; init_step = 1.5)
    # TikTak mode: given flags are FORWARDED; absent ones leave TikTak's defaults; --local-alg is refused
    @test reopt_tiktak_kw() == (;)
    @test reopt_tiktak_kw(; ftol_rel = 1e-4, init_step = 0.1) == (local_tol = 1e-4, local_initial_step = 0.1)
    @test_throws ErrorException reopt_tiktak_kw(; local_alg = "bobyqa")
    # ... and they reach NLopt: the configuration TikTak runs with, which results.toml records
    f(x) = sum(abs2, x .- 0.2)
    base = tiktak(f, [-1.0, -1.0], [1.0, 1.0]; N = 10, Nstar = 2, skip_polish = true, reopt_tiktak_kw()...)
    given = tiktak(f, [-1.0, -1.0], [1.0, 1.0]; N = 10, Nstar = 2, skip_polish = true,
                   reopt_tiktak_kw(; ftol_rel = 1e-4, init_step = 0.1)...)
    @test base.config.local_.ftol_rel == 1e-3 && base.config.local_.initial_step == 0.0     # defaults unchanged
    @test given.config.local_.ftol_rel == 1e-4 && given.config.local_.initial_step == 0.1
    eff = TOML.parse(join(effective_settings_lines(given.config), "\n"))
    @test eff["local_ftol_rel"] == 1e-4 && eff["local_initial_step"] == 0.1 && eff["local_alg"] == "LN_NELDERMEAD"
    # resume metadata from the route actually taken: no pure-local field in TikTak mode
    t = TOML.parse(join(reopt_resume_lines(; tiktak = true, tk_route = "tiktak_state", tk_semantics = :async_continuation), "\n"))
    @test t["resumed"] && startswith(t["resume_mode"], "tiktak_state") && t["resume_semantics"] == "async_continuation"
    @test !occursin("restart_from_best", t["resume_mode"]) && !haskey(t, "evals_before_resume")
    fresh = TOML.parse(join(reopt_resume_lines(; tiktak = true), "\n"))
    @test !fresh["resumed"] && fresh["resume_mode"] == "fresh" && fresh["resume_verification"] == "not a resume"
    loc = TOML.parse(join(reopt_resume_lines(; tiktak = false, local_resumed = true, evals_done = 25, local_verification = "verified"), "\n"))
    @test loc["resumed"] && startswith(loc["resume_mode"], "restart_from_best") && loc["evals_before_resume"] == 25
end

const TMP = mktempdir(; prefix = "reopt_identity_")

"A minimal targets file: [name] tables with a mean, and a [moment_cov] block with names and se."
function write_targets(path; means = Dict("m1" => 1.0, "m2" => 2.0, "x1" => 5.0), se = Dict("m1" => 0.1, "m2" => 0.2, "x1" => 0.5))
    d = Dict{String,Any}(k => Dict("mean" => v) for (k, v) in means)
    names = sort(collect(keys(se)))
    d["moment_cov"] = Dict("names" => names, "se" => [se[k] for k in names])
    open(io -> TOML.print(io, d; sorted = true), path, "w")
    return path
end

# the run as reopt.jl resolves it (fixed values stand in for SMM_PARAMS etc.)
const TFILE = write_targets(joinpath(TMP, "targets.toml"))
const XFILE = write_targets(joinpath(TMP, "extra_targets.toml"))
function fields(; tfile = TFILE, xfile = XFILE, extra = [:x1], moments = ["m1", "m2"], grid = 30, sim_n = 2000,
                spec = Dict{String,Any}(k => "" for k in REOPT_SPEC_ENV_KEYS), child_extra = "(Nap = 90,)")
    traw = TOML.parsefile(tfile)
    rows = extra_rows_from(extra, TOML.parsefile(xfile), k -> traw[k]["mean"])
    return reopt_objective_fields(; repo = REPO, targets_file = tfile, extra_targets_file = xfile, extra_rows = rows,
        moment_names = moments, free_names = ["a", "b"], free_links = ["level", "log"], lo = [-1.0, -1.0], hi = [1.0, 1.0],
        fixed = Dict(:mu => 0.8), fixed_search = Dict("mu" => 0.8), free_extra = "", run_bounds = "",
        seed = 1234, sim_n = sim_n, grid = grid, child_grid = "30x30x5", parent_extra = "(a_max = 300.0,)",
        child_extra = child_extra, spec = spec)
end

const CALLS = Ref(0)
counted(x) = (CALLS[] += 1; sum(abs2, x .- 0.3))
kw(fl) = (N = 24, Nstar = 4, local_maxeval = 20, skip_polish = true, objective_fields = fl, objective_id = TikTak.fields_id(fl))

"A paused run of the counting objective under `fl`, with its checkpoint and pre-testing cache."
function paused_run(fl)
    dir = mktempdir(TMP)
    st, cache = joinpath(dir, "tiktak_state.toml"), joinpath(dir, "pretest_cache.toml")
    tiktak(counted, [-1.0, -1.0], [1.0, 1.0]; kw(fl)..., state_path = st, pretest_cache = cache, stop_after_restarts = 1)
    return st, cache
end
refusal(f) = try f(); "" catch e; sprint(showerror, e) end

@testset "reopt objective identity" begin
    base = fields()
    @testset "the builder" begin
        @test base["identity_version"] == REOPT_IDENTITY_VERSION
        @test base["targets_sha"] == file_sha16(TFILE) && base["extra_targets_sha"] == file_sha16(XFILE)
        @test base["extra_names"] == ["x1"] && base["extra_targets"] == [5.0] && base["extra_weights"] == [1 / 0.5^2]
        @test fields() == base                                                    # deterministic
        @test TikTak.fields_id(fields()) == TikTak.fields_id(base)
        @test fields(; extra = Symbol[])["extra_targets_sha"] == ""              # no extra rows: the file does not matter
    end

    # each economic change is refused on resume BEFORE any evaluation, naming the changed field;
    # the module's own check refuses the checkpoint and the cache too (objective id)
    cases = [
        ("a target mean changed IN THE SAME FILE", () -> write_targets(TFILE; means = Dict("m1" => 1.5, "m2" => 2.0, "x1" => 5.0)),
         () -> write_targets(TFILE), "targets_sha"),
        ("an extra-target weight changed (same names, new se)", () -> write_targets(XFILE; se = Dict("m1" => 0.1, "m2" => 0.2, "x1" => 0.25)),
         () -> write_targets(XFILE), "extra_weights"),
    ]
    for (what, mutate!, restore!, needle) in cases
        st, cache = paused_run(base)
        mutate!()
        now = fields()
        CALLS[] = 0
        m = refusal(() -> check_reopt_resume(((st, :state), (cache, :cache)), now))
        m2 = refusal(() -> tiktak(counted, [-1.0, -1.0], [1.0, 1.0]; kw(now)..., state_path = st, pretest_cache = cache, resume = true))
        @testset "refused: $what" begin
            @test occursin("resume refused", m) && occursin(needle, m)
            @test occursin("objective changed", m2)
            @test CALLS[] == 0
        end
        restore!()
    end
    for (what, now, needle) in (("the moment order", fields(; moments = ["m2", "m1"]), "moment_names"),
                                ("the parent grid", fields(; grid = 20), "grid"),
                                ("the simulated households", fields(; sim_n = 4000), "sim_n"),
                                ("a child override", fields(; child_extra = "(Nap = 60,)"), "child_extra"),
                                ("an SMM_* switch", fields(; spec = merge(Dict{String,Any}(k => "" for k in REOPT_SPEC_ENV_KEYS),
                                                                          Dict("SMM_MU_FIXED" => "0.7"))), "spec_env.SMM_MU_FIXED"))
        st, cache = paused_run(base)
        CALLS[] = 0
        m = refusal(() -> check_reopt_resume(((st, :state), (cache, :cache)), now))
        @testset "refused: $what" begin
            @test occursin("resume refused", m) && occursin(needle, m)
            @test CALLS[] == 0
        end
    end

    @testset "unchanged inputs resume" begin
        st, cache = paused_run(base)
        notes = check_reopt_resume(((st, :state), (cache, :cache)), fields())
        @test length(notes) == 2 && all(n -> occursin("verified", n), notes)
        r = tiktak(counted, [-1.0, -1.0], [1.0, 1.0]; kw(fields())..., state_path = st, pretest_cache = cache, resume = true)
        @test r.status === :complete && TikTak.search_budget_complete(r)
    end

    @testset "a checkpoint of the pre-2026-09-28 identity is not continued" begin
        # the identity reopt.jl used to record: the target file by PATH, extra rows by NAME
        old = Dict{String,Any}("targets" => "targets.toml", "extra_moments" => ["x1"], "seed" => 1234, "simN" => 2000,
                               "grid" => 30, "run_bounds" => "", "fixed" => Dict("mu" => 0.8), "parent_extra" => "",
                               "child_extra" => "", "source_sha16" => "abc", "free_extra" => "")
        st, cache = paused_run(old)
        CALLS[] = 0
        m = refusal(() -> check_reopt_resume(((st, :state), (cache, :cache)), base))
        @test occursin("INCOMPLETE objective identity", m) && occursin("--start", m) && CALLS[] == 0
    end
end
