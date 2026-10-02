#!/usr/bin/env julia
# =============================================================================
# test_tiktak.jl -- regression tests for the TikTak optimizer (2026-09-27)
#
#     julia --project=. --threads=1 tools/test_tiktak.jl            # everything
#     julia --project=. --threads=1 tools/test_tiktak.jl baseline   # one or more groups
#
# Synthetic, deterministic objectives only: no economic model is solved. The groups
# that start worker processes start at most two, remove them at the end and fail if
# any is left behind. tools/test_tiktak_integration.jl covers the real SMM objective.
#
# Groups: baseline, findings (one test per confirmed finding in tiktak_problems.md),
# validity, status, checkpoint, pretest, async, recovery, transitions (follow-up 1, findings
# 11 and 14), lifecycle (follow-up 3, findings 12 and 16, plan 7.1), presets, geometry, julia,
# orphans.
# =============================================================================

using Test, Printf, Distributed, TOML, NLopt, Sobol, InteractiveUtils

const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "code", "src", "tiktak.jl"))

# The implementation that was in code/src/tiktak.jl until 2026-09-27, frozen verbatim
# (sha256 61126ee4e076d72d). The serial path of the module must reproduce it bit for bit
# wherever a fix does not deliberately change the answer.
module LegacyTikTak
using NLopt, Printf
include(joinpath(@__DIR__, "testdata", "tiktak_legacy_20260927.jl"))
end

const GROUPS = isempty(ARGS) ? nothing : Set(ARGS)
want(g) = GROUPS === nothing || g in GROUPS
const SCRATCH = mktempdir(; prefix = "tiktak_test_")
const FAILED_GROUPS = String[]
"Run one named group of tests; a failing group is recorded and the next group still runs."
function group(body, name)
    want(name) || return
    try
        @testset "$name" begin
            body()
        end
    catch e
        e isa Test.TestSetException || rethrow()
        push!(FAILED_GROUPS, name)
    end
end

rastrigin(x) = 10length(x) + sum(xi^2 - 10cos(2π * xi) for xi in x)
sphere(x) = sum(abs2, x)
shifted(x) = sum(abs2, x .- 0.3)

"Everything two runs report about the search path, for a bit-for-bit comparison."
path_of(r) = (x = r.x, f = r.f, n_eval = r.n_eval, trace = [Tuple(t) for t in r.trace],
              polish = (r.polish_ret, r.polish_improved, r.n_eval_polish), f_sobol_best = r.f_sobol_best,
              f_prepolish = r.f_prepolish)
"The search path without the evaluation counts (compared bit for bit; the counts are checked exactly apart)."
path_core(r) = (x = r.x, f = r.f, trace = [Tuple(t) for t in r.trace], polish = (r.polish_ret, r.polish_improved),
                f_sobol_best = r.f_sobol_best, f_prepolish = r.f_prepolish)
"""
Evaluations the v1 2026-10-02 rule "a known start value is not recomputed" saves against the legacy
code: one per restart (the solver's first call is at its start, evaluated just before), one more for
restart 1 (its start is its seed, whose pre-tested value is known) and one for the polish (its start is
the incumbent). Measured on every baseline case before this check was written: exactly this, with the
path otherwise bit-identical.
"""
known_start_saving(r) = length(r.trace) + 1 + (r.polish_ret === :SKIPPED || r.n_eval_polish == 0 ? 0 : 1)

"Is the returned point's convergence established (by its own search or by a later verification)?"
converged_evidence(r) = isdefined(TikTak, :local_converged) ? TikTak.local_converged(r) :
                        ret_class(r.winner_ret) === :converged

# =============================================================================
group("baseline") do
    @test tiktak_selftest(; verbose = false)

    # Importing the optimizer starts no worker and loads no model (plan J1).
    let out = read(`$(Base.julia_cmd()) --project=$REPO --startup-file=no --threads=1 -e '
            include(joinpath(ARGS[1], "code", "src", "tiktak.jl"))
            using Distributed
            println(nprocs(), " ", isdefined(Main, :smm_objective), " ", isdefined(Main, :TikTak))' $REPO`, String)
        @test strip(out) == "1 false true"
    end

    # The serial path equals the legacy implementation where no fix changes the answer.
    for (name, f, d, kw) in (("rastrigin d=4", rastrigin, 4, (N = 300, Nstar = 12)),
                             ("shifted sphere + seed", shifted, 3, (N = 50, Nstar = 4, extra_seeds = [fill(0.1, 3)])),
                             ("skip polish", rastrigin, 2, (N = 40, Nstar = 5, skip_polish = true)),
                             ("tight budgets", rastrigin, 3, (N = 60, Nstar = 5, local_maxeval = 7, polish_maxeval = 9)),
                             ("early stop", rastrigin, 2, (N = 80, Nstar = 10, stop_tol = 0.5)),
                             ("invalid region", x -> x[1] > 0 ? 1e12 : sphere(x .+ 1), 2, (N = 30, Nstar = 5, invalid_value = 1e12)))
        a = tiktak(f, fill(-5.12, d), fill(5.12, d); kw...)
        b = LegacyTikTak.tiktak(f, fill(-5.12, d), fill(5.12, d); kw...)
        @testset "legacy equivalence: $name" begin
            # (2026-10-02) the path bit for bit; the counts lower by exactly the known-start saving
            @test path_core(a) == path_core(b)
            @test a.n_eval == b.n_eval - known_start_saving(a)
            @test a.n_eval_polish == b.n_eval_polish - (a.n_eval_polish == 0 ? 0 : 1)
        end
    end

    # The retained objective never gets worse (monotone incumbent), restart by restart.
    let r = tiktak(rastrigin, fill(-5.12, 3), fill(5.12, 3); N = 100, Nstar = 10)
        best = r.f_sobol_best
        for t in r.trace
            @test t.improved == (t.f_local < best)
            best = min(best, t.f_local)
        end
        @test r.f_prepolish == best && r.f <= r.f_prepolish
    end
end

# =============================================================================
# One test per confirmed finding of tiktak_problems.md. Each asserts the CORRECT
# behaviour; against the 2026-09-27 code each fails (see the step-1 log).
# =============================================================================
group("findings") do
    @testset "F1 local restarts run on several worker processes" begin
        ws = addprocs(2; exeflags = `--project=$REPO --threads=1 --startup-file=no`)
        try
            @everywhere ws include(joinpath($REPO, "code", "src", "tiktak.jl"))
            @everywhere ws begin
                const EVAL_PIDS = Int[]
                pid_sphere(x) = (push!(EVAL_PIDS, myid()); sleep(0.002); sum(abs2, x))
                TikTak.register_objective!(:pid_sphere, pid_sphere)
            end
            local_pids = Set{Int}()
            r = tiktak(x -> sum(abs2, x), fill(-2.0, 3), fill(2.0, 3); N = 40, Nstar = 6,
                       skip_polish = true, local_mode = :async_process, local_workers = ws,
                       objective_key = :pid_sphere)
            for rec in r.records; push!(local_pids, rec.worker); end
            @test length(local_pids) == 2 && !(1 in local_pids)
        finally
            rmprocs(ws; waitfor = 30)
        end
    end

    @testset "F2 a budget-limited refinement is not convergence" begin
        # A converged coarse search, then a refinement on the reporting objective that
        # improves but stops on its evaluation cap: (status = :improved, ret =
        # :MAXEVAL_REACHED) passed the old predicate.
        r = tiktak(x -> sum(abs2, x .- 0.1), fill(-1.0, 4), fill(1.0, 4); N = 20, Nstar = 2, objective_id = "coarse")
        out = TikTak.refine(x -> sum(abs2, x .- 0.1) + 0.3 * sum(x), r.x, fill(-1.0, 4), fill(1.0, 4);
                            settings = TikTak.SolverSettings(:LN_BOBYQA, 1e-6, 1e-10, 1e-6, 12),
                            cfg = r.config, objective_id = "fine")
        @test TikTak.local_converged(r)
        @test out.status === :improved && out.ret === :MAXEVAL_REACHED
        acc = TikTak.acceptance(; execution_ok = true, candidate_valid = true, search_budget_complete = true,
                                local_converged = TikTak.local_converged(out.incumbent, "fine"))
        @test !acc.accepted
    end

    @testset "F3 resume keeps provenance and cumulative accounting" begin
        dir = mktempdir(SCRATCH)
        kw = (N = 30, Nstar = 3, extra_seeds = [[0.25, -0.25]])
        full = tiktak(sphere, [-1.0, -1.0], [1.0, 1.0]; kw...)
        part = tiktak(sphere, [-1.0, -1.0], [1.0, 1.0]; kw..., state_path = joinpath(dir, "s.toml"),
                      stop_after_restarts = 3)                        # paused just before the polish
        res = tiktak(sphere, [-1.0, -1.0], [1.0, 1.0]; kw..., state_path = joinpath(dir, "s.toml"), resume = true)
        @test res.x == full.x && res.f == full.f
        @test res.winner_ret == full.winner_ret && res.winner_stage == full.winner_stage
        @test res.n_eval == full.n_eval
        @test converged_evidence(res) == converged_evidence(full)
    end

    @testset "F4 resume after the restart count was reduced" begin
        dir = mktempdir(SCRATCH)
        f = x -> x[1] > -0.5 ? 1e12 : sphere(x)                     # three quarters of the box invalid
        kw = (N = 16, Nstar = 8, invalid_value = 1e12, skip_polish = true, local_maxeval = 30)
        full = tiktak(f, [-1.0, -1.0], [1.0, 1.0]; kw...)
        tiktak(f, [-1.0, -1.0], [1.0, 1.0]; kw..., state_path = joinpath(dir, "s.toml"), stop_after_restarts = 0)
        res = tiktak(f, [-1.0, -1.0], [1.0, 1.0]; kw..., state_path = joinpath(dir, "s.toml"), resume = true)
        @test res.nstar_requested == 8 && res.nstar_effective == full.nstar_effective < 8
        @test res.x == full.x && res.f == full.f
    end

    @testset "F5 an unchanged optimum is verified" begin
        r = tiktak(x -> sum(abs2, x), [-1.0, -1.0], [1.0, 1.0]; N = 8, Nstar = 2, extra_seeds = [[0.0, 0.0]])
        @test r.f == 0.0
        @test converged_evidence(r)
    end

    @testset "F6 pre-testing can target a number of VALID draws" begin
        f = x -> x[1] > 0.0 ? 1e12 : sphere(x)                      # half the box is invalid
        r = tiktak(f, [-1.0, -1.0], [1.0, 1.0]; N = 400, n_valid_target = 50, Nstar = 5,
                   invalid_value = 1e12, skip_polish = true, local_maxeval = 20)
        @test r.pretest.valid >= 50 && r.pretest.attempted > 50
    end

    @testset "F7 completed pre-testing values survive an interruption" begin
        dir = mktempdir(SCRATCH)
        calls = Ref(0)
        boom = x -> (calls[] += 1; calls[] == 25 && error("simulated crash"); sphere(x))
        @test_throws Exception tiktak(boom, [-1.0, -1.0], [1.0, 1.0]; N = 40, Nstar = 3,
                                      pretest_cache = joinpath(dir, "cache.toml"), pretest_chunk = 8)
        calls[] = 0
        r = tiktak(x -> (calls[] += 1; sphere(x)), [-1.0, -1.0], [1.0, 1.0]; N = 40, Nstar = 3, skip_polish = true,
                   local_maxeval = 1, pretest_cache = joinpath(dir, "cache.toml"), pretest_chunk = 8, resume = :auto)
        @test r.pretest.reused >= 16
    end

    @testset "F8 search geometry and initial steps are explicit and recorded" begin
        r = tiktak(sphere, [-1.0, -1.0], [3.0, 3.0]; N = 20, Nstar = 2, normalize = true,
                   local_initial_step = 0.1, skip_polish = true)
        @test r.config.normalize && r.config.local_.initial_step == 0.1
    end

    @testset "F9 the best value always belongs to the returned point" begin
        f = x -> x[1] < -0.9 ? -Inf : sphere(x .- 0.5)
        r = tiktak(f, [-1.0, -1.0], [1.0, 1.0]; N = 64, Nstar = 3, skip_polish = true, local_maxeval = 30)
        @test isfinite(r.f) && r.f == f(r.x) && isfinite(r.f_sobol_best)
    end

    @testset "F10 source comments no longer assert what the implementation does not do" begin
        src = join((read(joinpath(REPO, "code", "src", "TikTak", fn), String)
                    for fn in readdir(joinpath(REPO, "code", "src", "TikTak")) if endswith(fn, ".jl")), "\n")
        runner = read(joinpath(REPO, "code", "smm", "run_smm.jl"), String)
        for (phrase, text) in (("it gives no formula", src), ("That is the ASYNCHRONOUS variant", src),
                               ("ftol_abs is scale-free", src), ("cannot be parallelised", runner),
                               # 2026-09-28, finding 10: moment counts are not identification and do not decide Q
                               ("just-identified, Q can reach 0", runner), ("Q cannot reach 0 and weights matter", runner))
            stale = occursin(phrase, text)
            stale && println("  F10: stale claim still present: \"", phrase, "\"")
            @test !stale
        end
    end
end

# =============================================================================
# Step 2: candidate validity and effective budgets
# =============================================================================
group("validity") do
    calls = Ref(0)
    counted(x) = (calls[] += 1; sphere(x))
    lo2, hi2 = [-1.0, -1.0], [1.0, 1.0]
    # An invalid configuration is refused before the first evaluation.
    for (what, args, kw) in (("infinite bound", ([-Inf, -1.0], hi2), (;)),
                             ("lo >= hi", ([1.0, -1.0], hi2), (;)),
                             ("dimension mismatch", ([-1.0], hi2), (;)),
                             ("Nstar = 0", (lo2, hi2), (Nstar = 0,)),
                             ("Nstar > N + #seeds", (lo2, hi2), (N = 3, Nstar = 5)),
                             ("local_maxeval = 0", (lo2, hi2), (local_maxeval = 0,)),
                             ("polish_maxeval = 0", (lo2, hi2), (polish_maxeval = 0,)),
                             ("theta_lo > theta_hi", (lo2, hi2), (theta_lo = 0.5, theta_hi = 0.2)),
                             ("theta_p <= 0", (lo2, hi2), (theta_p = 0.0,)),
                             ("theta_hi > 1", (lo2, hi2), (theta_hi = 1.5,)),
                             ("unknown algorithm", (lo2, hi2), (local_alg = :LN_NOSUCH,)),
                             ("NaN invalid_value", (lo2, hi2), (invalid_value = NaN,)),
                             ("extra seed of the wrong dimension", (lo2, hi2), (extra_seeds = [[0.0]],)),
                             ("non-finite extra seed", (lo2, hi2), (extra_seeds = [[NaN, 0.0]],)),
                             ("extra seed outside the box", (lo2, hi2), (extra_seeds = [[0.0, 1.5]],)))
        calls[] = 0
        @testset "refused: $what" begin
            @test_throws Exception tiktak(counted, args...; merge((N = 10, Nstar = 2), kw)...)
            @test calls[] == 0
        end
    end
    # polish_maxeval = 0 is fine when the polish is explicitly skipped
    @test tiktak(sphere, lo2, hi2; N = 10, Nstar = 2, polish_maxeval = 0, skip_polish = true).polish_ret === :SKIPPED
    # a seed within 1e-9 of the box width outside is snapped onto the bound, not refused
    let r = tiktak(sphere, lo2, hi2; N = 10, Nstar = 2, extra_seeds = [[1.0 + 1e-12, 0.0]], skip_polish = true)
        @test r.pretest.supplied == 1
    end

    # Valid values mixed with NaN, +-Inf and penalties: only valid points seed, the counts
    # add up, and the returned value belongs to the returned point.
    let pts = TikTak.sobol_points(lo2, hi2, 64)
        kind(x) = x[1] < -0.8 ? -Inf : x[1] > 0.8 ? NaN : x[2] > 0.6 ? Inf : x[2] < -0.6 ? 1e12 : sphere(x .- 0.2)
        r = tiktak(kind, lo2, hi2; N = 64, Nstar = 4, invalid_value = 1e12, skip_polish = true, local_maxeval = 25)
        ps = r.pretest
        vals = map(kind, pts)
        @test ps.attempted == 64
        @test ps.valid == count(v -> isfinite(v) && v < 1e12, vals)
        @test ps.invalid == count(==(1e12), vals)
        @test ps.nonfinite == count(!isfinite, vals)
        @test ps.valid + ps.invalid + ps.nonfinite + ps.errors == ps.attempted
        @test isfinite(r.f_sobol_best) && r.f_sobol_best == minimum(v for v in vals if isfinite(v) && v < 1e12)
        @test all(t -> isfinite(t.f_start) && t.f_start < 1e12, r.trace)
        @test r.f == kind(r.x)
    end

    # Nothing valid: the run stops before the local stage.
    calls[] = 0
    @test_throws ErrorException tiktak(x -> (calls[] += 1; 1e12), lo2, hi2; N = 16, Nstar = 2, invalid_value = 1e12)
    @test calls[] == 16

    # Too few valid values: reduced and reported, or a failed gate that costs no local search.
    let f = x -> x[1] > -0.5 ? 1e12 : sphere(x)
        r = @test_logs (:warn, r"fewer VALID") match_mode = :any tiktak(f, lo2, hi2; N = 16, Nstar = 8,
                invalid_value = 1e12, skip_polish = true, local_maxeval = 10)
        @test r.nstar_requested == 8 && r.nstar_effective == r.pretest.valid < 8
        @test r.schedule_denominator == r.nstar_effective && length(r.trace) == r.nstar_effective
        calls[] = 0
        err = try
            tiktak(x -> (calls[] += 1; f(x)), lo2, hi2; N = 16, Nstar = 8, invalid_value = 1e12,
                   allow_fewer_restarts = false, skip_polish = true)
            nothing
        catch e
            e
        end
        @test err isa TikTak.PretestGateError
        @test calls[] == 16                                   # the pre-testing only, no local search
    end

    # Legacy resume state that is not a point of this box is refused.
    let seeds = [[0.0, 0.0], [0.5, 0.5]]
        good = (seeds = seeds, f_sobol_best = 0.0, Z = [0.0, 0.0], fZ = 0.0, j_start = 2)
        @test tiktak(sphere, lo2, hi2; N = 10, Nstar = 2, resume = good, skip_polish = true).f == 0.0
        @test_throws Exception tiktak(sphere, lo2, hi2; N = 10, Nstar = 2, resume = merge(good, (Z = [0.0, 3.0],)))
        @test_throws Exception tiktak(sphere, lo2, hi2; N = 10, Nstar = 2, resume = merge(good, (Z = [0.0],)))
        @test_throws Exception tiktak(sphere, lo2, hi2; N = 10, Nstar = 2, resume = merge(good, (seeds = [[0.0], [0.5]],)))
        @test_throws Exception tiktak(sphere, lo2, hi2; N = 10, Nstar = 2, resume = merge(good, (fZ = -Inf,)))
    end

    # Finding 4 through the legacy resume: five seeds saved for eight requested restarts
    # resume with the requested count and take the effective K from the seeds.
    let f = x -> x[1] > -0.5 ? 1e12 : sphere(x), saved = Ref{Any}(), kw = (N = 16, Nstar = 8, invalid_value = 1e12, skip_polish = true)
        full = tiktak(f, lo2, hi2; kw..., on_seeds = (s, fb) -> (saved[] = (s, fb)))
        res = tiktak(f, lo2, hi2; kw..., resume = (seeds = saved[][1], f_sobol_best = saved[][2],
                                                  Z = copy(saved[][1][1]), fZ = saved[][2], j_start = 1))
        @test res.nstar_effective == full.nstar_effective < 8 && res.nstar_requested == 8
        @test res.x == full.x && res.f == full.f && [Tuple(t) for t in res.trace] == [Tuple(t) for t in full.trace]
    end
end

# =============================================================================
# Step 3: candidate origin, verification and acceptance
# =============================================================================
group("status") do
    lo2, hi2 = [-1.0, -1.0], [1.0, 1.0]
    cfg = tiktak(sphere, lo2, hi2; N = 4, Nstar = 1, skip_polish = true, local_maxeval = 1).config
    origin(stage, ret; obj = "A") = TikTak.CandidateOrigin(stage, 0, 1, 0, ret, obj)
    inc0 = TikTak.Incumbent([0.2, 0.2], 1.0, 3, origin(:local, :MAXEVAL_REACHED), TikTak.NO_VERIFICATION)

    # the merge rule, directly
    a, i1 = TikTak.merge_candidate(inc0, [0.2, 0.2], 1.0, :FTOL_REACHED, :polish, 0, 1, lo2, hi2, cfg, "A", "s")
    @test a === :verified && i1.x == inc0.x && i1.version == 3 && i1.origin == inc0.origin
    @test i1.verification.stage === :polish && i1.verification.distance == 0.0
    @test !TikTak.local_converged(inc0, "A") && TikTak.local_converged(i1, "A")
    @test !TikTak.local_converged(i1, "B")                     # evidence belongs to objective A
    a, _ = TikTak.merge_candidate(inc0, [-0.6, 0.4], 1.0, :FTOL_REACHED, :local, 2, 1, lo2, hi2, cfg, "A", "s")
    @test a === :none                                           # equal Q at a DIFFERENT point
    a, _ = TikTak.merge_candidate(inc0, [0.2, 0.2], 1.0, :MAXEVAL_REACHED, :local, 2, 1, lo2, hi2, cfg, "A", "s")
    @test a === :none                                           # same point, but not converged
    a, _ = TikTak.merge_candidate(inc0, [0.2, 0.2], 1.0, :FTOL_REACHED, :local, 2, 1, lo2, hi2, cfg, "B", "s")
    @test a === :none                                           # a solve of ANOTHER objective
    a, _ = TikTak.merge_candidate(inc0, [0.2, 0.2], 1.5, :FTOL_REACHED, :local, 2, 1, lo2, hi2, cfg, "A", "s")
    @test a === :none                                           # same point, inconsistent value
    a, i2 = TikTak.merge_candidate(i1, [0.1, 0.1], 0.5, :MAXEVAL_REACHED, :local, 4, 1, lo2, hi2, cfg, "A", "s")
    @test a === :improved && i2.version == 4 && i2.origin.restart == 4 && i2.origin.ret === :MAXEVAL_REACHED
    @test i2.verification.status === :none && !TikTak.local_converged(i2, "A")   # a new point loses the old verification
    for (x, v) in (([0.1, 0.1], -Inf), ([0.1, 0.1], NaN), ([0.1, 1.5], 0.5))
        a, _ = TikTak.merge_candidate(inc0, x, v, :FTOL_REACHED, :local, 5, 1, lo2, hi2, cfg, "A", "s")
        @test a === :none                                       # an invalid candidate never replaces the point
    end

    # an exactly supplied optimum: verified by the restart that returns it
    r = tiktak(sphere, lo2, hi2; N = 8, Nstar = 2, extra_seeds = [[0.0, 0.0]])
    @test r.winner_stage === :supplied && r.incumbent.verification.status === :verified && TikTak.local_converged(r)

    # budget-limited evidence that a later solve verifies without improving: the restart
    # from the optimum stops on MAXEVAL; the polish returns the same point with XTOL
    rs = tiktak(sphere, lo2, hi2; N = 8, Nstar = 1, extra_seeds = [[0.0, 0.0]], local_maxeval = 3, skip_polish = true)
    rp = tiktak(sphere, lo2, hi2; N = 8, Nstar = 1, extra_seeds = [[0.0, 0.0]], local_maxeval = 3)
    @test rs.trace[1].ret === :MAXEVAL_REACHED && !TikTak.local_converged(rs) && rs.polish_ret === :SKIPPED
    @test rp.incumbent.verification.stage === :polish && TikTak.local_converged(rp) && rp.f == rs.f

    # refinement on a second objective
    fine(x) = sum(abs2, x .- 0.1) + 0.3 * sum(x)
    budget = tiktak(x -> sum(abs2, x .- 0.1), fill(-1.0, 4), fill(1.0, 4); N = 20, Nstar = 2, local_maxeval = 4,
                    polish_maxeval = 4, objective_id = "coarse")
    @test !TikTak.local_converged(budget)                       # the coarse search is budget-limited
    good = TikTak.refine(fine, budget.x, fill(-1.0, 4), fill(1.0, 4);
                         settings = TikTak.SolverSettings(:LN_BOBYQA, 1e-10, 1e-14, 1e-10, 2000),
                         cfg = budget.config, objective_id = "fine")
    @test good.status === :improved && TikTak.ret_class(good.ret) === :converged
    @test TikTak.local_converged(good.incumbent, "fine") && !TikTak.local_converged(good.incumbent, "coarse")
    # a refinement started AT the exact minimum returns that point: verified, not improved
    # (from a converged but inexact point BOBYQA may still find a tiny strict improvement)
    same = TikTak.refine(sphere, zeros(4), fill(-1.0, 4), fill(1.0, 4);
                         settings = TikTak.SolverSettings(:LN_BOBYQA, 1e-10, 1e-14, 1e-10, 2000),
                         cfg = budget.config, objective_id = "fine")
    @test same.status === :no_improvement && same.incumbent.origin.stage === :reevaluated
    @test same.incumbent.verification.status === :verified && TikTak.local_converged(same.incumbent, "fine")
    boom_calls = Ref(0)
    failed = TikTak.refine(x -> (boom_calls[] += 1; boom_calls[] > 3 && error("refinement crash"); fine(x)),
                           budget.x, fill(-1.0, 4), fill(1.0, 4);
                           settings = TikTak.SolverSettings(:LN_BOBYQA, 1e-10, 1e-14, 1e-10, 200),
                           cfg = budget.config, objective_id = "fine")
    @test failed.status === :failed && failed.incumbent.x == budget.x && !TikTak.local_converged(failed.incumbent, "fine")
    @test occursin("refinement crash", failed.error)
    acc_failed = TikTak.acceptance(; execution_ok = false, candidate_valid = true, search_budget_complete = true,
                                   local_converged = TikTak.local_converged(failed.incumbent, "fine"))
    @test !acc_failed.accepted && length(acc_failed.reasons) == 2
    acc_good = TikTak.acceptance(; execution_ok = true, candidate_valid = true, search_budget_complete = true,
                                 local_converged = TikTak.local_converged(good.incumbent, "fine"))
    @test acc_good.accepted && isempty(acc_good.reasons)
    @test !TikTak.acceptance(; execution_ok = true, candidate_valid = true, search_budget_complete = true,
                             local_converged = true, pinned = ["mu"]).accepted
    @test TikTak.refine_skipped(r.incumbent).incumbent === r.incumbent

    # a deliberately skipped polish is part of the plan, not a failure
    sk = tiktak(sphere, lo2, hi2; N = 20, Nstar = 3, skip_polish = true)
    @test sk.polish_ret === :SKIPPED && sk.status === :complete && sk.n_eval_polish == 0
end

# =============================================================================
# Step 4: versioned checkpoints and exact serial resumption
# =============================================================================
"What must agree between an uninterrupted run and an interrupted-and-resumed one."
function resume_fingerprint(r)
    recs = sort(r.records; by = x -> x.j)
    return (x = r.x, f = r.f, n_eval = r.n_eval, trace = [Tuple(t) for t in r.trace],
            starts = [x.x0 for x in recs], ends = [x.x for x in recs], ids = [x.j for x in recs],
            origin = r.incumbent.origin, verification = (r.incumbent.verification.status, r.incumbent.verification.stage),
            polish = (r.polish_ret, r.polish_improved, r.n_eval_polish), status = r.status, K = r.nstar_effective)
end
"resume_fingerprint without the total count (for a checkpoint whose restarts ran under older software)."
resume_fingerprint_core(r) = Base.structdiff(resume_fingerprint(r), NamedTuple{(:n_eval,)})

group("checkpoint") do
    lo3, hi3 = fill(-5.12, 3), fill(5.12, 3)
    kw = (N = 60, Nstar = 6, extra_seeds = [fill(0.7, 3)], local_maxeval = 80, polish_maxeval = 60)
    full = tiktak(rastrigin, lo3, hi3; kw...)
    @test full.status === :complete && full.n_eval_complete && full.resume_semantics === :fresh

    # interrupted after seed selection, after restart 2, and just before the polish
    for M in (0, 2, kw.Nstar)
        dir = mktempdir(SCRATCH); path = joinpath(dir, "state.toml")
        part = tiktak(rastrigin, lo3, hi3; kw..., state_path = path, stop_after_restarts = M)
        @test part.status === :paused && length(part.records) == M && part.polish_ret === :NOT_RUN
        res = tiktak(rastrigin, lo3, hi3; kw..., state_path = path, resume = true)
        @testset "resume at M = $M" begin
            @test resume_fingerprint(res) == resume_fingerprint(full)
            @test res.resume_semantics === :serial_exact && res.n_eval_complete
            @test res.n_eval == full.n_eval                         # lifetime accounting is exact
            @test res.n_eval_segment == full.n_eval - part.n_eval   # this process's share
        end
    end
    # several pauses in a row
    let dir = mktempdir(SCRATCH), path = joinpath(dir, "state.toml")
        tiktak(rastrigin, lo3, hi3; kw..., state_path = path, stop_after_restarts = 1)
        tiktak(rastrigin, lo3, hi3; kw..., state_path = path, resume = true, stop_after_restarts = 4)
        res = tiktak(rastrigin, lo3, hi3; kw..., state_path = path, resume = true)
        @test resume_fingerprint(res) == resume_fingerprint(full)
        d = TOML.parsefile(path)
        @test length(d["segments"]) == 3 && count(e -> e["kind"] == "pause", d["events"]) == 2
    end

    # a crash inside restart 4: the committed state survives, the restart is re-run from the
    # same incumbent, and the answer is the uninterrupted one
    let dir = mktempdir(SCRATCH), path = joinpath(dir, "state.toml"), n = Ref(0), crash_at = Ref(0)
        # evaluations up to the middle of restart 4 in the uninterrupted run
        crash_at[] = full.pretest.attempted + full.pretest.supplied + sum(r.n_eval for r in full.records if r.j <= 3) + 5
        boom(x) = (n[] += 1; n[] == crash_at[] && error("simulated crash in restart 4"); rastrigin(x))
        @test_throws Exception tiktak(boom, lo3, hi3; kw..., state_path = path)   # NLopt wraps it (CapturedException)
        d = TOML.parsefile(path)
        @test d["status"] == "failed" && d["progress"]["next_j"] == 4 && length(d["records"]) == 3
        res = tiktak(rastrigin, lo3, hi3; kw..., state_path = path, resume = true)
        @test resume_fingerprint(res) == resume_fingerprint(full)
    end

    # the checkpoint is readable TOML with its identities, K and every committed record
    let dir = mktempdir(SCRATCH), path = joinpath(dir, "state.toml")
        tiktak(rastrigin, lo3, hi3; kw..., state_path = path, stop_after_restarts = 3,
               objective_id = "obj-A", objective_fields = Dict("targets_sha" => "abc", "grid" => 30))
        d = TOML.parsefile(path)
        @test d["schema_version"] == 2 && d["kind"] == "tiktak_state" && d["plan"]["schedule_denominator"] == 6
        @test length(d["epochs"]) == 1 && all(r["epoch"] == 1 for r in d["records"])         # schema 2 (2026-09-28)
        @test sort([a["j"] for a in d["accounted"] if a["how"] == "committed"]) == [1, 2, 3]
        @test d["objective"]["id"] == "obj-A" && d["objective"]["fields"]["grid"] == 30
        @test d["optimizer"]["fields"]["local"]["maxeval"] == 80 && length(d["records"]) == 3
        @test all(length(r["x0"]) == 3 for r in d["records"]) && d["seeds"]["origin"][1] in ("sobol", "supplied")

        # a different objective or optimizer is refused, naming what changed
        for (what, extra, needle) in (("objective", (objective_id = "obj-B", objective_fields = Dict("targets_sha" => "xyz", "grid" => 30)), "targets_sha"),
                                      ("local budget", (objective_id = "obj-A", objective_fields = Dict("targets_sha" => "abc", "grid" => 30), local_maxeval = 81), "local.maxeval"),
                                      ("restart count", (objective_id = "obj-A", objective_fields = Dict("targets_sha" => "abc", "grid" => 30), Nstar = 7), "nstar_requested"),
                                      ("box", (objective_id = "obj-A", objective_fields = Dict("targets_sha" => "abc", "grid" => 30)), "lo"))
            err = try
                box = what == "box" ? (fill(-5.0, 3), hi3) : (lo3, hi3)
                tiktak(rastrigin, box...; merge(kw, extra)..., state_path = path, resume = true)
                nothing
            catch e
                e
            end
            @testset "refused: changed $what" begin
                @test err isa TikTak.ResumeRefused
                @test err isa TikTak.ResumeRefused && occursin(needle, err.msg)
            end
        end
        # ... unless an optimizer change is explicitly allowed, which is recorded
        r = tiktak(rastrigin, lo3, hi3; kw..., local_maxeval = 81, state_path = path, resume = true,
                   objective_id = "obj-A", objective_fields = Dict("targets_sha" => "abc", "grid" => 30),
                   allow_optimizer_change = true)
        @test r.status === :complete
        @test any(e -> e["kind"] == "optimizer_change", TOML.parsefile(path)["events"])
    end

    # corrupted or truncated generations
    let dir = mktempdir(SCRATCH), path = joinpath(dir, "state.toml")
        tiktak(rastrigin, lo3, hi3; kw..., state_path = path, stop_after_restarts = 2)
        @test isfile(path) && isfile(path * ".prev")
        txt = read(path, String)
        write(path, txt[1:div(length(txt), 2)])                  # truncated newest generation
        res = @test_logs (:warn, r"did not verify") match_mode = :any tiktak(rastrigin, lo3, hi3; kw..., state_path = path, resume = true)
        @test resume_fingerprint(res) == resume_fingerprint(full)   # the previous generation is exact too
        write(path, replace(read(path, String), "\"complete\"" => "\"running\""))   # an edit breaks the checksum
        write(path * ".prev", "garbage")
        @test_throws ErrorException tiktak(rastrigin, lo3, hi3; kw..., state_path = path, resume = true)
    end
    # a file of another schema is refused by name
    let dir = mktempdir(SCRATCH), path = joinpath(dir, "state.toml")
        TikTak.write_checksummed(path, "schema_version = 99\nkind = \"tiktak_state\"\n")
        err = try tiktak(rastrigin, lo3, hi3; kw..., state_path = path, resume = true); nothing catch e; e end
        @test err isa TikTak.ResumeRefused || (err isa KeyError)
    end

    # resume modes
    let dir = mktempdir(SCRATCH), path = joinpath(dir, "state.toml")
        @test_throws ArgumentError tiktak(rastrigin, lo3, hi3; kw..., state_path = path, resume = true)
        a = tiktak(rastrigin, lo3, hi3; kw..., state_path = path, resume = :auto)      # nothing there: fresh
        @test_throws ArgumentError tiktak(rastrigin, lo3, hi3; kw..., state_path = path)   # would overwrite
        n = Ref(0)
        b = tiktak(x -> (n[] += 1; rastrigin(x)), lo3, hi3; kw..., state_path = path, resume = :auto)
        @test n[] == 0 && b.x == a.x && b.f == a.f && b.resume_semantics === :already_complete
    end

    # a legacy (old-format) import is labelled, and invents nothing
    let saved = Ref{Any}(), last = Ref{Any}()
        tiktak(rastrigin, lo3, hi3; kw..., on_seeds = (s, fb) -> (saved[] = (s, fb)),
               on_local = (j, K, th, fl, b, bx, row) -> (j == 3 && (last[] = (b, copy(bx)))))
        r = tiktak(rastrigin, lo3, hi3; kw..., resume = (seeds = saved[][1], f_sobol_best = saved[][2],
                                                        Z = last[][2], fZ = last[][1], j_start = 4))
        @test r.resume_semantics === :legacy_import && !r.n_eval_complete
        @test r.x == full.x && r.f == full.f                     # same numerical path from restart 4
        @test r.winner_stage !== :legacy_import || !TikTak.local_converged(r)
    end
end

# =============================================================================
# Step 5: recoverable pre-testing and a valid-draw target
# =============================================================================
group("pretest") do
    lo2, hi2 = [-1.0, -1.0], [1.0, 1.0]
    half(x) = x[1] > 0.0 ? 1e12 : sum(abs2, x .+ 0.3)          # half the box invalid
    kwp = (Nstar = 4, invalid_value = 1e12, local_maxeval = 40, polish_maxeval = 30)
    seeds_of(r) = [rec.x0 for rec in sort(r.records; by = x -> x.j)][1:1]   # restart 1 starts at the best seed

    # an interrupted stage resumes from its cache and ends exactly where an uninterrupted one does
    full = tiktak(half, lo2, hi2; kwp..., N = 64)
    let dir = mktempdir(SCRATCH), cache = joinpath(dir, "cache.toml"), n = Ref(0)
        boom(x) = (n[] += 1; n[] == 30 && error("simulated crash"); half(x))
        @test_throws Exception tiktak(boom, lo2, hi2; kwp..., N = 64, pretest_cache = cache, pretest_chunk = 10)
        d = TOML.parsefile(cache)
        @test d["n_done"] == 29                                   # every completed value, saved as the error propagated
        m = Ref(0)
        res = tiktak(x -> (m[] += 1; half(x)), lo2, hi2; kwp..., N = 64, pretest_cache = cache, pretest_chunk = 10, resume = :auto)
        @test res.x == full.x && res.f == full.f && [Tuple(t) for t in res.trace] == [Tuple(t) for t in full.trace]
        @test res.pretest.reused == 29 && res.n_eval == full.n_eval   # lifetime: the cached values were this run's own
        @test m[] == full.n_eval - 29                             # evaluated in the resumed process only
        @test res.pretest.attempted == 64 && res.pretest.valid == full.pretest.valid
    end

    # ties: the selection follows candidate order, whatever the evaluation order or batching
    let flat(x) = round(2x[1]) + 3.0                               # large blocks of equal values
        a = tiktak(flat, lo2, hi2; N = 40, Nstar = 5, skip_polish = true, local_maxeval = 5)
        b = LegacyTikTak.tiktak(flat, lo2, hi2; N = 40, Nstar = 5, skip_polish = true, local_maxeval = 5)
        c = tiktak(flat, lo2, hi2; N = 40, Nstar = 5, skip_polish = true, local_maxeval = 5, map_fn = (g, xs) -> reverse(map(g, reverse(xs))),
                   pretest_chunk = 7)
        @test [Tuple(t) for t in a.trace] == [Tuple(t) for t in b.trace] == [Tuple(t) for t in c.trace]
        @test a.x == b.x == c.x
    end

    # a valid-draw target: the shortest prefix with the target number of valid draws
    let r = tiktak(half, lo2, hi2; kwp..., N = 400, n_valid_target = 30)
        pts = TikTak.sobol_points(lo2, hi2, 400)
        vals = map(half, pts)
        istar = findfirst(i -> count(v -> v < 1e12, vals[1:i]) == 30, 1:400)
        @test r.pretest.attempted == istar && r.pretest.valid == 30 && r.pretest.target_reached
        @test r.pretest.stop_reason === :valid_target && r.pretest.overshoot == 0 && r.pretest.invalid == istar - 30
        # batched evaluation overshoots but selects the same pool
        rb = tiktak(half, lo2, hi2; kwp..., N = 400, n_valid_target = 30, map_fn = (g, xs) -> map(g, xs), pretest_chunk = 16)
        @test rb.pretest.attempted == istar && rb.pretest.overshoot > 0 && rb.x == r.x && rb.f == r.f
        @test rb.n_eval == r.n_eval + rb.pretest.overshoot          # overshoot was evaluated, and is counted
    end
    # an all-valid objective: the valid target reproduces the fixed design of the same size
    let a = tiktak(sphere, lo2, hi2; N = 50, Nstar = 4, local_maxeval = 30, polish_maxeval = 20),
        b = tiktak(sphere, lo2, hi2; N = 500, n_valid_target = 50, Nstar = 4, local_maxeval = 30, polish_maxeval = 20)
        @test path_of(a) == path_of(b) && b.pretest.attempted == 50
    end
    # an exhausted attempt cap: a reported shortfall, or a failed gate when required
    let r = @test_logs (:warn, r"valid-draw target was not reached") match_mode = :any tiktak(half, lo2, hi2; kwp..., N = 20, n_valid_target = 19)
        @test r.pretest.stop_reason === :attempt_cap && !r.pretest.target_reached && r.pretest.attempted == 20
        @test_throws TikTak.PretestGateError tiktak(half, lo2, hi2; kwp..., N = 20, n_valid_target = 19, require_valid_target = true)
    end
    # a time cap stops at the longest completed prefix
    let slow(x) = (sleep(0.01); sphere(x)), r = tiktak(slow, lo2, hi2; N = 500, Nstar = 2, skip_polish = true, local_maxeval = 3, pretest_time_cap = 0.3)
        @test r.pretest.stop_reason === :time_cap && 2 <= r.pretest.attempted < 500
    end

    # a cache from another objective or another design is refused
    let dir = mktempdir(SCRATCH), cache = joinpath(dir, "cache.toml")
        tiktak(half, lo2, hi2; kwp..., N = 20, pretest_cache = cache, objective_id = "A", skip_polish = true,
               pretest_chunk = 8)                                   # several generations: 8, 16, 20
        for (what, kw) in (("objective", (objective_id = "B",)), ("box", (objective_id = "A",)), ("seeds", (objective_id = "A", extra_seeds = [[0.0, 0.0]])))
            box = what == "box" ? ([-1.0, -2.0], hi2) : (lo2, hi2)
            err = try tiktak(half, box...; kwp..., N = 20, pretest_cache = cache, resume = true, kw...); nothing catch e; e end
            @testset "cache refused: $what" begin
                @test err isa TikTak.ResumeRefused
            end
        end
        # another run may REUSE it (same objective and design): the values are not re-evaluated
        n = Ref(0)
        r = tiktak(x -> (n[] += 1; half(x)), lo2, hi2; kwp..., N = 20, pretest_reuse = cache, objective_id = "A", skip_polish = true)
        @test r.pretest.reused == 20 && n[] == r.n_eval_segment && r.n_eval == r.n_eval_segment  # imported values are not this run's
        # a truncated newest cache generation falls back to the previous one
        write(cache, "values = [1.0")
        r2 = @test_logs (:warn, r"did not verify") match_mode = :any tiktak(half, lo2, hi2; kwp..., N = 20, pretest_cache = cache, objective_id = "A", skip_polish = true, resume = :auto)
        @test r2.pretest.reused > 0
    end
end

# =============================================================================
# Steps 6-7: process-parallel asynchronous local restarts, recovery and accounting
# =============================================================================
include(joinpath(REPO, "tools", "testdata", "tiktak_test_objectives.jl"))    # the master's copies

"Is process `pid` gone (or a zombie)? Checked after the workers are removed."
proc_gone(pid) = !isdir("/proc/$pid") || occursin(r"State:\s+Z", read("/proc/$pid/status", String))

"""
    with_workers(body, n; timeout)

Start `n` worker processes with the module and the test objectives and run `body(ws)` in THIS
task (so its @test results count). A watchdog removes the workers after `timeout` seconds,
which makes any pending remote call fail instead of hanging. Afterwards the workers are
removed and the test fails if any of their processes is left behind.
"""
function with_workers(body, n::Int; timeout::Real = 600)
    ws = addprocs(n; exeflags = `--project=$REPO --threads=1 --startup-file=no`)
    pids = Int[remotecall_fetch(getpid, w) for w in ws]
    done = Ref(false)
    watchdog = @async begin
        t0 = time()
        while !done[] && time() - t0 < timeout
            sleep(0.5)
        end
        if !done[]
            @warn "with_workers: timeout after $timeout s -- removing the workers"
            try rmprocs(intersect(ws, workers()); waitfor = 10) catch end
        end
    end
    try
        @everywhere ws include(joinpath($REPO, "code", "src", "tiktak.jl"))
        @everywhere ws include(joinpath($REPO, "tools", "testdata", "tiktak_test_objectives.jl"))
        body(ws)
    finally
        done[] = true
        wait(watchdog)
        live = intersect(ws, workers())
        isempty(live) || rmprocs(live; waitfor = 30)
        t0 = time()
        while !all(proc_gone, pids) && time() - t0 < 20; sleep(0.2); end
        @test all(proc_gone, pids)                               # no orphan worker processes
        @test nprocs() == 1
    end
end

"The incumbent version current after commit number c (c = 0: before any commit)."
version_after_commit(recs, c) = c == 0 ? 1 : only(r.version_after for r in recs if r.commit_seq == c)

group("async") do
    lo3, hi3 = fill(-5.12, 3), fill(5.12, 3)
    kwa = (N = 60, Nstar = 8, local_maxeval = 40, polish_maxeval = 30)
    with_workers(2) do ws
        # one local worker IS the sequential algorithm
        ser = tiktak(tt_rast, lo3, hi3; kwa...)
        one = tiktak(tt_rast, lo3, hi3; kwa..., local_mode = :async_process, local_workers = ws,
                     local_count = 1, objective_key = :rast)
        @test resume_fingerprint(one) == resume_fingerprint(ser)
        @test all(r -> r.worker == ws[1], one.records)

        # two workers: restart 2 is slow, so later restarts overtake it
        seen = TikTak.ProgressEvent[]
        two = tiktak(tt_slow, lo3, hi3; kwa..., skip_polish = true, local_mode = :async_process,
                     local_workers = ws, objective_key = :slow, progress_every = 0.3,
                     on_progress = evs -> append!(seen, evs))
        recs = two.records
        @test sort([r.j for r in recs]) == collect(1:8)             # every restart once, none twice
        @test all(r -> all(TikTak.checked_point(r.x0, lo3, hi3; tol = 0.0) .== r.x0), recs)   # in-box starts
        @test Set(r.worker for r in recs) == Set(ws)                # both workers, never the master
        r1 = only(r for r in recs if r.j == 1)
        @test r1.commits_at_dispatch == 0 && all(r -> r.j == 1 || r.commits_at_dispatch >= 1, recs)   # bootstrap
        overlap = any(a.commits_at_dispatch < b.commit_seq && b.commits_at_dispatch < a.commit_seq
                      for a in recs, b in recs if a.j < b.j)
        @test overlap                                               # jobs really ran at the same time
        r2 = only(r for r in recs if r.j == 2)
        overtook = [r for r in recs if r.j > 2 && r.commits_at_dispatch < r2.commit_seq && r.commit_seq < r2.commit_seq]
        @test !isempty(overtook)                                    # a later restart started AND finished first
        @test issorted([r.j for r in sort(recs; by = r -> r.commit_seq)]) == false   # completion order != restart order
        # every job was mixed with the incumbent current when it was dispatched
        @test all(r -> r.incumbent_version == version_after_commit(recs, r.commits_at_dispatch), recs)
        @test !isempty(seen) && all(ev -> ev.run_id == two.run_id, seen)
        @test length(TikTak.PROGRESS_STORE) <= length(ws)           # coalesced: one entry per worker

        # the automatic local count: min(workers, floor(sqrt(K)))
        k3 = tiktak(tt_rast, lo3, hi3; N = 30, Nstar = 3, local_maxeval = 20, skip_polish = true,
                    local_mode = :async_process, local_workers = ws, objective_key = :rast, state_path = joinpath(mktempdir(SCRATCH), "s.toml"))
        @test all(r -> r.worker == ws[1], k3.records)               # floor(sqrt(3)) = 1 worker

        # a flood of fast evaluations is coalesced, never blocks a worker, and results still arrive
        n0 = TikTak.PROGRESS_RECEIVED[]
        t0 = time()
        fl = tiktak(tt_flood, lo3, hi3; N = 40, Nstar = 4, local_maxeval = 3000, skip_polish = true,
                    local_mode = :async_process, local_workers = ws, local_count = 2, objective_key = :flood,
                    progress_every = 0.0)
        el = time() - t0
        msgs = TikTak.PROGRESS_RECEIVED[] - n0
        @test length(fl.records) == 4
        @test msgs <= 4 * (el / TikTak.MIN_REMOTE_PROGRESS + 2)    # at most one per 0.2 s per job (+ first)

        # process-parallel pre-testing selects exactly what serial pre-testing selects
        pp = tiktak(tt_rast, lo3, hi3; kwa..., pretest_workers = ws, objective_key = :rast)
        @test path_of(pp) == path_of(ser)
        half(x) = x[1] > 0.0 ? 1e12 : tt_rast(x)
        @everywhere ws tt_half(x) = x[1] > 0.0 ? 1e12 : tt_rast(x)
        @everywhere ws TikTak.register_objective!(:half, tt_half)
        sv = tiktak(half, lo3, hi3; N = 400, n_valid_target = 30, Nstar = 5, invalid_value = 1e12, skip_polish = true, local_maxeval = 20)
        pv = tiktak(half, lo3, hi3; N = 400, n_valid_target = 30, Nstar = 5, invalid_value = 1e12, skip_polish = true, local_maxeval = 20,
                    pretest_workers = ws, objective_key = :half, pretest_chunk = 4)
        @test pv.pretest.attempted == sv.pretest.attempted && pv.x == sv.x && pv.f == sv.f
        @test pv.pretest.overshoot <= length(ws)                    # bounded by the workers in flight
    end
end

# =============================================================================
# 2026-09-29 (fix plan R2): the asynchronous start-up policy, bootstrap = :immediate_mixed
# =============================================================================
group("bootstrap") do
    lo3, hi3 = fill(-5.12, 3), fill(5.12, 3)
    kwa = (N = 60, Nstar = 8, local_maxeval = 40, polish_maxeval = 30)
    # an unknown policy is refused before the first evaluation
    let calls = Ref(0)
        @test_throws ArgumentError tiktak(x -> (calls[] += 1; sphere(x)), [-1.0, -1.0], [1.0, 1.0];
                                          N = 10, Nstar = 2, bootstrap = :no_such)
        @test calls[] == 0
    end
    # serial local stage: the start-up has no effect (restarts are sequential under either value)
    ser = tiktak(tt_rast, lo3, hi3; kwa...)
    @test path_of(tiktak(tt_rast, lo3, hi3; kwa..., bootstrap = :immediate_mixed)) == path_of(ser)
    @test tiktak(tt_rast, lo3, hi3; kwa..., bootstrap = :immediate_mixed).config.bootstrap === :immediate_mixed
    with_workers(3) do ws
        # one local worker is still exactly the sequential algorithm
        one = tiktak(tt_rast, lo3, hi3; kwa..., local_mode = :async_process, local_workers = ws, local_count = 1,
                     objective_key = :rast, bootstrap = :immediate_mixed)
        @test resume_fingerprint(one) == resume_fingerprint(ser)

        @everywhere ws TT_SLOW_J[] = 1
        # the default first (it also compiles the job path on all three workers, so wall-clock order below
        # is decided by restart 1's slowness, not by JIT): nothing else starts before restart 1 is committed
        fa = tiktak(tt_slow, lo3, hi3; kwa..., skip_polish = true, local_mode = :async_process, local_workers = ws,
                    local_count = 3, objective_key = :slow)
        @test only(r for r in fa.records if r.j == 1).commits_at_dispatch == 0
        @test all(r -> r.j == 1 || r.commits_at_dispatch >= 1, fa.records)
        # three workers, restart 1 slow: the pool fills at once, and later restarts overtake restart 1
        path = joinpath(mktempdir(SCRATCH), "s.toml")
        im = tiktak(tt_slow, lo3, hi3; kwa..., skip_polish = true, local_mode = :async_process, local_workers = ws,
                    local_count = 3, objective_key = :slow, bootstrap = :immediate_mixed, state_path = path)
        recs = im.records
        seeds = [Float64.(s) for s in TOML.parsefile(path)["seeds"]["x"]]
        @test sort([r.j for r in recs]) == collect(1:8)                        # every restart once
        early = sort([r for r in recs if r.commits_at_dispatch == 0]; by = r -> r.j)
        @test [r.j for r in early] == [1, 2, 3]                                # three started before any commit
        @test all(r -> r.incumbent_version == 1, early)                         # ... mixed with the best pre-tested point
        @test early[1].theta == 0.0 && early[1].x0 == seeds[1]                  # restart 1: the best seed, unmixed
        for r in early[2:end]                                                   # restart j: seed j mixed with seed 1
            @test r.theta == TikTak.theta_at(r.j, 8, 0.5, 0.1, 0.995)
            @test r.x0 == clamp.((1 - r.theta) .* seeds[r.j] .+ r.theta .* seeds[1], lo3, hi3)
        end
        r1 = only(r for r in recs if r.j == 1)
        @test any(r -> r.j > 1 && r.commit_seq < r1.commit_seq, recs)          # restart 1 was overtaken
        # every later job was mixed with the incumbent committed when it was dispatched
        @test all(r -> r.incumbent_version == version_after_commit(recs, r.commits_at_dispatch), recs)
        @test all(r -> all(TikTak.checked_point(r.x0, lo3, hi3; tol = 0.0) .== r.x0), recs)

        @everywhere ws TT_SLOW_J[] = 2

        # K = 1, and fewer restarts than workers
        for K in (1, 2)
            rk = tiktak(tt_rast, lo3, hi3; N = 20, Nstar = K, local_maxeval = 20, skip_polish = true,
                        local_mode = :async_process, local_workers = ws, local_count = 3, objective_key = :rast,
                        bootstrap = :immediate_mixed)
            @test sort([r.j for r in rk.records]) == collect(1:K) && all(r -> r.commits_at_dispatch == 0, rk.records)
        end

        # a pause after four launches, then a resume: each restart committed once, starts kept
        kwp = (kwa..., skip_polish = true, local_mode = :async_process, local_workers = ws, local_count = 3,
               objective_key = :slow_all, bootstrap = :immediate_mixed)
        p2 = joinpath(mktempdir(SCRATCH), "s.toml")
        a = tiktak(tt_slow_all, lo3, hi3; kwp..., state_path = p2, stop_after_restarts = 4)
        @test a.status === :paused && sort([r.j for r in a.records]) == [1, 2, 3, 4]
        b = tiktak(tt_slow_all, lo3, hi3; kwp..., state_path = p2, resume = true)
        @test b.status === :complete && sort([r.j for r in b.records]) == collect(1:8)
        @test [r.x0 for r in sort(b.records; by = r -> r.j)][1:4] == [r.x0 for r in sort(a.records; by = r -> r.j)]

        # the start-up belongs to the SEARCH PLAN: a resume that changes it is refused, even with the override
        p3 = joinpath(mktempdir(SCRATCH), "s.toml")
        tiktak(tt_slow_all, lo3, hi3; kwp..., state_path = p3, stop_after_restarts = 2)
        err = try tiktak(tt_slow_all, lo3, hi3; kwp..., bootstrap = :first_alone, state_path = p3, resume = true,
                         allow_optimizer_change = true); nothing catch e; e end
        @test err isa TikTak.ResumeRefused && occursin("SEARCH PLAN", err.msg) && occursin("bootstrap", err.msg)
    end
end

"The incumbent's point at `version`, rebuilt from the records (1 = the best seed)."
incumbent_at(recs, seeds, v) = v == 1 ? seeds[1] :
    only(r.x for r in recs if r.action === :improved && r.version_after == v)

group("recovery") do
    lo3, hi3 = fill(-5.12, 3), fill(5.12, 3)
    kwr = (N = 60, Nstar = 8, local_maxeval = 40, skip_polish = true)
    akw(ws, key; n = 2) = (local_mode = :async_process, local_workers = ws, local_count = n, objective_key = key)

    # a worker exits in the middle of restart 3: that job is re-dispatched from its RECORDED
    # start on the other worker, and the run completes
    let dir = mktempdir(SCRATCH), path = joinpath(dir, "s.toml")
        with_workers(2) do ws
            @everywhere ws TT_KILL_J[] = 3
            r = tiktak(tt_rast, lo3, hi3; kwr..., akw(ws, :kill)..., state_path = path)
            @test r.status === :complete && sort([x.j for x in r.records]) == collect(1:8)
            rec3 = only(x for x in r.records if x.j == 3)
            @test rec3.attempt == 2 && r.accounting.jobs_lost == 1 && r.accounting.retries == 1
            @test !TikTak.work_known(r)                     # the lost attempt's evaluations stay UNKNOWN
            d = TOML.parsefile(path)
            @test any(e -> e["kind"] == "worker_lost", d["events"]) && any(e -> e["kind"] == "retry", d["events"])
            seeds = [Float64.(x) for x in d["seeds"]["x"]]
            z = incumbent_at(r.records, seeds, rec3.incumbent_version)
            @test rec3.x0 == clamp.((1 - rec3.theta) .* seeds[3] .+ rec3.theta .* z, lo3, hi3)   # the recorded start
        end
    end

    # every attempt of restart 2 kills its worker: retries exhausted, the run stops with the
    # job IN FLIGHT in the checkpoint; a resume (new workers) replays it from its start
    let dir = mktempdir(SCRATCH), path = joinpath(dir, "s.toml")
        with_workers(2) do ws
            @everywhere ws TT_KILL_J[] = 2
            err = try tiktak(tt_rast, lo3, hi3; kwr..., akw(ws, :kill_always)..., state_path = path, max_retries = 1); nothing catch e; e end
            @test err !== nothing
        end
        d = TOML.parsefile(path)
        @test d["status"] == "failed" && any(x -> x["j"] == 2, d["inflight"]) && d["counters"]["jobs_lost"] == 2
        with_workers(2) do ws
            r = tiktak(tt_rast, lo3, hi3; kwr..., akw(ws, :rast)..., state_path = path, resume = true)
            @test r.status === :complete && sort([x.j for x in r.records]) == collect(1:8)
            @test r.resume_semantics === :async_continuation && only(x for x in r.records if x.j == 2).attempt == 3
        end
    end

    # an error in the objective is never retried: checkpoint and stop; resume replays the job
    let dir = mktempdir(SCRATCH), path = joinpath(dir, "s.toml")
        with_workers(2) do ws
            @everywhere ws TT_FAIL_J[] = 3
            err = try tiktak(tt_rast, lo3, hi3; kwr..., akw(ws, :fail)..., state_path = path, drain_timeout = 20.0); nothing catch e; e end
            @test err isa ErrorException && occursin("deliberate coding error", sprint(showerror, err))
            d = TOML.parsefile(path)
            @test d["status"] == "failed" && d["counters"]["retries"] == 0 && any(x -> x["j"] == 3, d["inflight"])
            r = tiktak(tt_rast, lo3, hi3; kwr..., akw(ws, :rast)..., state_path = path, resume = true)
            @test r.status === :complete && sort([x.j for x in r.records]) == collect(1:8)
        end
    end

    # the master fails right after restart 5's dispatch record is on disk, before it is sent:
    # the orderly shutdown drains, and the resume runs restart 5 exactly once
    let dir = mktempdir(SCRATCH), path = joinpath(dir, "s.toml"), crashed = Ref(false)
        cb = st -> (haskey(st.inflight, 5) && !crashed[] && (crashed[] = true; error("simulated master crash after dispatching restart 5")))
        with_workers(2) do ws
            err = try tiktak(tt_slow_all, lo3, hi3; kwr..., akw(ws, :slow_all)..., state_path = path,
                             on_checkpoint = cb, drain_timeout = 20.0); nothing catch e; e end
            @test err !== nothing && occursin("simulated master crash", sprint(showerror, err))
            d = TOML.parsefile(path)
            @test d["status"] == "failed" && any(x -> x["j"] == 5, d["inflight"])
            r = tiktak(tt_slow_all, lo3, hi3; kwr..., akw(ws, :slow_all)..., state_path = path, resume = true)
            @test sort([x.j for x in r.records]) == collect(1:8)       # each restart exactly once
            @test only(x for x in r.records if x.j == 5).attempt == 2
        end
    end

    # A RESULT DELIVERED TWICE (or three times) IS MERGED ONCE AND COUNTED ONCE (2026-09-28, finding 13).
    # Until then the test here asserted that the second copy added all local evaluations to
    # abandoned_known -- work that was never done. Checked against an instrumented call counter.
    with_workers(1) do ws
        @everywhere ws begin
            const TT_CALLS = Ref(0)
            tt_count(x) = (TT_CALLS[] += 1; tt_rast(x))
            TikTak.register_objective!(:count, tt_count)
        end
        ser = tiktak(tt_rast, lo3, hi3; kwr...)
        for (fault, copies) in ((:duplicate_results, 2), (:triplicate_results, 3))
            @everywhere ws TT_CALLS[] = 0
            dup = tiktak(tt_rast, lo3, hi3; kwr..., akw(ws, :count; n = 1)..., _fault = fault)
            calls = sum(remotecall_fetch(() -> Main.TT_CALLS[], w) for w in ws)
            @testset "$(copies) deliveries of every result" begin
                @test resume_fingerprint(dup) == resume_fingerprint(ser)
                @test dup.accounting.abandoned_known == 0 && dup.accounting.local_ == ser.accounting.local_
                @test dup.accounting.duplicates_ignored == (copies - 1) * length(dup.records)
                @test calls == dup.accounting.local_                    # each executed evaluation counted once
                @test TikTak.work_known(dup)
            end
        end
    end
    # the ledger itself: a late copy adds nothing; an attempt that EXECUTED but was superseded
    # adds its known evaluations exactly once; another run's result adds nothing; the ledger
    # survives the checkpoint, so a late copy of a pre-resume attempt still adds nothing
    let dir = mktempdir(SCRATCH), path = joinpath(dir, "s.toml")
        r = tiktak(tt_rast, lo3, hi3; kwr..., state_path = path, stop_after_restarts = 3)
        st = TikTak.state_from_dict(first(TikTak.read_state_dict(path)), r.config)
        res(run, j, a, n) = TikTak.RestartResult(run, :local, j, a, 1.0, zeros(3), 1.0, n, :FTOL_REACHED, "", 2, 0.1)
        rec2 = only(x for x in st.records if x.j == 2)
        @test TikTak.account_uncommitted!(st, 2, rec2.attempt, res(st.run_id, 2, rec2.attempt, rec2.n_eval)) === :duplicate
        @test st.evals_abandoned == 0 && st.duplicates == 1 && st.evals_local == r.accounting.local_
        @test TikTak.account_uncommitted!(st, 3, 7, res(st.run_id, 3, 7, 37)) === :abandoned && st.evals_abandoned == 37
        @test TikTak.account_uncommitted!(st, 3, 7, res(st.run_id, 3, 7, 37)) === :duplicate && st.evals_abandoned == 37
        @test TikTak.account_uncommitted!(st, 3, 8, res("another run", 3, 8, 50)) === :other_run && st.evals_abandoned == 37
        @test TikTak.account_uncommitted!(st, 4, 1, nothing) === :no_result
        TikTak.write_state(path, st)
        st2 = TikTak.state_from_dict(first(TikTak.read_state_dict(path)), r.config)
        @test st2.accounted == st.accounted && st2.evals_abandoned == 37 && st2.duplicates == 2
        @test TikTak.account_uncommitted!(st2, 1, 1, res(st.run_id, 1, 1, 99)) === :duplicate && st2.evals_abandoned == 37
        @test TikTak.account_uncommitted!(st2, 3, 7, res(st.run_id, 3, 7, 37)) === :duplicate
        # the resumed run completes; the committed totals are the uninterrupted ones
        full = tiktak(tt_rast, lo3, hi3; kwr...)
        rr = tiktak(tt_rast, lo3, hi3; kwr..., state_path = path, resume = true)
        @test rr.accounting.local_ == full.accounting.local_ && rr.accounting.abandoned_known == 37
        @test rr.accounting.duplicates_ignored == 2                     # as checkpointed (st, not st2)
    end

    # delayed telemetry: a late event from a finished or superseded job is never reported
    let dir = mktempdir(SCRATCH), path = joinpath(dir, "s.toml")
        r = tiktak(tt_rast, lo3, hi3; kwr..., state_path = path, stop_after_restarts = 2)
        st = TikTak.state_from_dict(first(TikTak.read_state_dict(path)), r.config)
        job = TikTak.restart_job(st, 3, 2, zeros(3))
        st.inflight[3] = TikTak.InFlight(job, 7, time(), 3, 2)
        empty!(TikTak.PROGRESS_STORE)
        for ev in (TikTak.ProgressEvent(st.run_id, :local, 3, 1, 10, 1.0, 1.0, 6, time()),    # superseded attempt
                   TikTak.ProgressEvent(st.run_id, :local, 3, 2, 12, 2.0, 1.5, 7, time()),    # the live one
                   TikTak.ProgressEvent(st.run_id, :local, 2, 1, 40, 3.0, 2.5, 8, time()),    # restart 2 is committed
                   TikTak.ProgressEvent("another run", :local, 3, 2, 5, 4.0, 4.0, 9, time()))
            TikTak.progress_sink!(ev)
        end
        act = TikTak.active_progress(st)
        @test length(act) == 1 && act[1].worker == 7 && act[1].attempt == 2
        empty!(TikTak.PROGRESS_STORE)
    end

    # a graceful pause with several restarts in flight: nothing new starts, the running ones are
    # committed, the run pauses; the resume completes the plan
    let dir = mktempdir(SCRATCH), path = joinpath(dir, "s.toml"), pause = joinpath(dir, "PAUSE")
        with_workers(2) do ws
            kwp = (N = 60, Nstar = 12, local_maxeval = 60, skip_polish = true)
            toucher = @async (sleep(2.0); touch(pause))
            r = tiktak(tt_slow_all, lo3, hi3; kwp..., akw(ws, :slow_all)..., state_path = path, pause_file = pause,
                       progress_every = 0.25)
            wait(toucher)
            d = TOML.parsefile(path)
            @test r.status === :paused && 1 <= length(r.records) < 12
            @test isempty(d["inflight"]) && any(e -> e["kind"] == "pause_requested", d["events"])
            @test sort([x.j for x in r.records]) == collect(1:length(r.records))   # dispatched = committed
            rm(pause)
            r2 = tiktak(tt_slow_all, lo3, hi3; kwp..., akw(ws, :slow_all)..., state_path = path, pause_file = pause, resume = true)
            @test r2.status === :complete && sort([x.j for x in r2.records]) == collect(1:12)
        end
    end

    # the asynchronous launch limit: a prefix of 5 never launches restart 6
    let dir = mktempdir(SCRATCH), path = joinpath(dir, "s.toml")
        with_workers(2) do ws
            r = tiktak(tt_rast, lo3, hi3; N = 60, Nstar = 12, local_maxeval = 40, skip_polish = true, akw(ws, :rast)...,
                       state_path = path, stop_after_restarts = 5)
            d = TOML.parsefile(path)
            @test r.status === :paused && sort([x.j for x in r.records]) == collect(1:5) && d["progress"]["next_j"] == 6
            @test isempty(d["inflight"])
            r2 = tiktak(tt_rast, lo3, hi3; N = 60, Nstar = 12, local_maxeval = 40, skip_polish = true, akw(ws, :rast)...,
                        state_path = path, resume = true)
            @test r2.status === :complete && sort([x.j for x in r2.records]) == collect(1:12)
            @test all(x -> x.theta == TikTak.theta_at(r2.config, x.j, 12), r2.records)   # K stayed 12
        end
    end
end

# =============================================================================
# Follow-up 1 (2026-09-28): recovery transitions -- findings 11 and 14
# =============================================================================
"Rewrite a checkpoint file after editing its Dict (the checksum is recomputed: a deliberate, valid edit)."
function rewrite_state(path, edit!)
    d = TOML.parsefile(path)
    edit!(d)
    io = IOBuffer(); TOML.print(io, d; sorted = true)
    TikTak.write_checksummed(path, String(take!(io)); keep_previous = false)
    rm(path * ".prev"; force = true)
end

"An asynchronous run aborted on the master right after restart 2's dispatch record is on disk, before it is sent."
function inflight_state(ws, path; kw...)
    crashed = Ref(false)
    cb = st -> (haskey(st.inflight, 2) && !crashed[] && (crashed[] = true; error("abort before restart 2 is sent")))
    err = try
        tiktak(tt_rast, fill(-5.12, 2), fill(5.12, 2); kw..., local_mode = :async_process, local_workers = ws,
               objective_key = :rast, state_path = path, on_checkpoint = cb)
        nothing
    catch e
        e
    end
    return err
end

"Each planned restart committed exactly once, nothing in flight, no violated invariant."
once_each(r, K) = sort([x.j for x in r.records]) == collect(1:K) && isempty(r.inflight) && isempty(r.violations)

group("transitions") do
    kwa = (N = 30, Nstar = 4, local_maxeval = 20, skip_polish = true)
    lo2, hi2 = fill(-5.12, 2), fill(5.12, 2)
    ref = tiktak(tt_rast, lo2, hi2; kwa...)                          # the uninterrupted serial run

    with_workers(2) do ws
        # ---- finding 11: a job in flight is replayed, in EITHER local mode --------------------
        for mode in (:serial, :async_process)
            dir = mktempdir(SCRATCH); path = joinpath(dir, "s.toml")
            @test inflight_state(ws, path; kwa...) !== nothing
            d = TOML.parsefile(path)
            @test d["progress"]["next_j"] == 3 && [x["j"] for x in d["inflight"]] == [2] && d["status"] == "failed"
            x0_saved = Float64.(d["inflight"][1]["x0"])
            mkw = mode === :serial ? (;) : (local_mode = :async_process, local_workers = ws, objective_key = :rast)
            r = tiktak(tt_rast, lo2, hi2; kwa..., mkw..., state_path = path, resume = true)
            @testset "in-flight job resumed in $mode mode" begin
                @test r.status === :complete && once_each(r, 4) && TikTak.search_budget_complete(r)
                rec2 = only(x for x in r.records if x.j == 2)
                @test rec2.attempt == 2 && rec2.x0 == x0_saved             # replayed from its RECORDED start
                @test r.resume_semantics === :async_continuation           # never "serial exact"
                ev = [e["kind"] for e in TOML.parsefile(path)["events"]]
                @test "replay" in ev
            end
        end

        # a drained asynchronous pause resumed serially is still an asynchronous history
        let dir = mktempdir(SCRATCH), path = joinpath(dir, "s.toml")
            p = tiktak(tt_rast, lo2, hi2; kwa..., local_mode = :async_process, local_workers = ws, objective_key = :rast,
                       state_path = path, stop_after_restarts = 2)
            @test p.status === :paused && isempty(p.inflight) && !TikTak.search_budget_complete(p)
            r = tiktak(tt_rast, lo2, hi2; kwa..., state_path = path, resume = true)
            @test once_each(r, 4) && r.resume_semantics === :async_continuation
        end
        # a serial pause resumed on ONE local worker is the sequential path exactly; on two it is not
        for (n, sem) in ((1, :serial_exact), (2, :async_continuation))
            dir = mktempdir(SCRATCH); path = joinpath(dir, "s.toml")
            tiktak(tt_rast, lo2, hi2; kwa..., state_path = path, stop_after_restarts = 1)
            r = tiktak(tt_rast, lo2, hi2; kwa..., local_mode = :async_process, local_workers = ws, local_count = n,
                       objective_key = :rast, state_path = path, resume = true)
            @test once_each(r, 4) && r.resume_semantics === sem
            n == 1 && @test resume_fingerprint(r) == resume_fingerprint(ref)
        end
        # an already complete state, in either mode: nothing is evaluated
        let dir = mktempdir(SCRATCH), path = joinpath(dir, "s.toml")
            tiktak(tt_rast, lo2, hi2; kwa..., state_path = path)
            for mkw in ((;), (local_mode = :async_process, local_workers = ws, objective_key = :rast))
                n = Ref(0)
                r = tiktak(x -> (n[] += 1; tt_rast(x)), lo2, hi2; kwa..., mkw..., state_path = path, resume = true)
                @test n[] == 0 && r.resume_semantics === :already_complete && once_each(r, 4)
            end
        end

        # ---- an inconsistent checkpoint is refused, never continued or reported ------------------
        let dir = mktempdir(SCRATCH), path = joinpath(dir, "s.toml")
            tiktak(tt_rast, lo2, hi2; kwa..., state_path = path)
            good = read(path, String)
            for (what, edit!, needle) in (
                    ("complete with a restart missing", d -> filter!(x -> x["j"] != 2, d["records"]), "never committed"),
                    ("a restart committed twice", d -> push!(d["records"], d["records"][1]), "more than once"),
                    ("complete with a restart in flight",
                     d -> (d["inflight"] = [Dict("j" => 3, "attempt" => 1, "stage" => "local", "theta" => 0.5,
                                                 "x0" => [0.0, 0.0], "incumbent_version" => 1, "worker" => 2,
                                                 "dispatched" => 0.0, "dispatch_seq" => 9, "commits_at_dispatch" => 2,
                                                 "step_scale" => 1.0, "epoch" => 1)]), "in flight"),
                    ("an incumbent outside the box", d -> (d["incumbent"]["x"] = [9.0, 0.0]), "not a point of the box"),
                    ("the audit's finding-11 state: complete, 2 skipped and in flight",
                     d -> (x = filter(r -> r["j"] == 2, d["records"])[1]; filter!(r -> r["j"] != 2, d["records"]);
                           d["progress"]["commit_seq"] = 3;
                           d["inflight"] = [Dict("j" => 2, "attempt" => 1, "stage" => "local", "theta" => x["theta"],
                                                 "x0" => x["x0"], "incumbent_version" => 1, "worker" => 2, "dispatched" => 0.0,
                                                 "dispatch_seq" => 2, "commits_at_dispatch" => 1, "step_scale" => 1.0,
                                                 "epoch" => 1)]), "in flight"))
                write(path, good); rm(path * ".prev"; force = true)
                rewrite_state(path, edit!)
                n = Ref(0)
                err = try tiktak(x -> (n[] += 1; tt_rast(x)), lo2, hi2; kwa..., state_path = path, resume = true); nothing catch e; e end
                @testset "refused: $what" begin
                    @test err isa TikTak.ResumeRefused && occursin(needle, err.msg)
                    @test n[] == 0
                end
            end
        end
    end

    # ---- finding 14: what a resume may change ---------------------------------------------
    f8(x) = sum(abs2, x .- 0.8)
    kwb = (N = 20, Nstar = 4, local_maxeval = 30, skip_polish = true)
    base = mktempdir(SCRATCH)
    mkpaused() = (p = joinpath(mktempdir(base), "s.toml");
                  tiktak(f8, [-1.0, -1.0], [1.0, 1.0]; kwb..., state_path = p, stop_after_restarts = 1); p)
    # the audit's reproduction: a changed box with the override is refused, in the preflight too
    let p = mkpaused(), n = Ref(0)
        for pf in (true, false)
            err = try tiktak(x -> (n[] += 1; f8(x)), [-0.1, -0.1], [0.1, 0.1]; kwb..., state_path = p, resume = true,
                             allow_optimizer_change = true, preflight_only = pf); nothing catch e; e end
            @test err isa TikTak.ResumeRefused && occursin("SEARCH PLAN", err.msg) && occursin("lo:", err.msg)
        end
        @test n[] == 0
    end
    for (what, box, kw, needle) in (("dimension", (fill(-1.0, 3), fill(1.0, 3)), (;), "lo:"),
                                    ("restart count K", ([-1.0, -1.0], [1.0, 1.0]), (Nstar = 5,), "nstar_requested"),
                                    ("supplied seeds", ([-1.0, -1.0], [1.0, 1.0]), (extra_seeds = [[0.1, 0.1]],), "supplied"),
                                    ("mixing schedule", ([-1.0, -1.0], [1.0, 1.0]), (theta_hi = 0.9,), "theta"),
                                    ("pre-testing design", ([-1.0, -1.0], [1.0, 1.0]), (N = 25,), "n_sobol"),
                                    ("stop_tol", ([-1.0, -1.0], [1.0, 1.0]), (stop_tol = 0.1,), "stop_tol"))
        p = mkpaused()
        err = try tiktak(f8, box...; merge(kwb, kw)..., state_path = p, resume = true, allow_optimizer_change = true); nothing catch e; e end
        @testset "plan change refused even with the override: $what" begin
            @test err isa TikTak.ResumeRefused && occursin("SEARCH PLAN", err.msg) && occursin(needle, err.msg)
        end
    end
    # a SOLVER change: refused without the override; with it, a new settings epoch
    let p = mkpaused()
        err = try tiktak(f8, [-1.0, -1.0], [1.0, 1.0]; kwb..., local_maxeval = 45, state_path = p, resume = true); nothing catch e; e end
        @test err isa TikTak.ResumeRefused && occursin("local.maxeval", err.msg) && occursin("allow_optimizer_change", err.msg)
        r = tiktak(f8, [-1.0, -1.0], [1.0, 1.0]; kwb..., local_maxeval = 45, state_path = p, resume = true,
                   allow_optimizer_change = true)
        @test once_each(r, 4) && r.resume_semantics === :changed_optimizer
        @test length(r.epochs) == 2 && r.epochs[1].local_.maxeval == 30 && r.epochs[2].local_.maxeval == 45
        @test [x.epoch for x in sort(r.records; by = x -> x.j)] == [1, 2, 2, 2]
        @test all(x -> x.n_eval <= 31, filter(x -> x.epoch == 1, r.records))   # the start + 30
    end
    # a SOFTWARE change only (a module bug fix): same settings, the same numerical path
    let p = mkpaused()
        rewrite_state(p, d -> (d["optimizer"]["fields"]["tiktak_source_sha"] = "0000000000000000"; d["optimizer"]["id"] = "old"))
        err = try tiktak(f8, [-1.0, -1.0], [1.0, 1.0]; kwb..., state_path = p, resume = true); nothing catch e; e end
        @test err isa TikTak.ResumeRefused && occursin("tiktak_source_sha", err.msg)
        r = tiktak(f8, [-1.0, -1.0], [1.0, 1.0]; kwb..., state_path = p, resume = true, allow_optimizer_change = true)
        full = tiktak(f8, [-1.0, -1.0], [1.0, 1.0]; kwb...)
        @test resume_fingerprint(r) == resume_fingerprint(full) && r.resume_semantics === :changed_optimizer
    end
    # a job IN FLIGHT across a solver change keeps the settings of its dispatch
    with_workers(1) do ws
        dir = mktempdir(SCRATCH); path = joinpath(dir, "s.toml")
        inflight_state(ws, path; kwa...)
        r = tiktak(tt_rast, lo2, hi2; kwa..., local_maxeval = 26, state_path = path, resume = true, allow_optimizer_change = true)
        rec = Dict(x.j => x for x in r.records)
        @test once_each(r, 4) && rec[2].epoch == 1 && rec[3].epoch == 2 && rec[4].epoch == 2
        @test rec[2].n_eval <= 21 && r.epochs[1].local_.maxeval == 20
    end

    # ---- schema-1 checkpoints of the 2026-09-27 code (frozen fixtures) ---------------------
    let p = joinpath(mktempdir(SCRATCH), "s.toml")
        cp(joinpath(REPO, "tools", "testdata", "tiktak_state_schema1_paused.toml"), p)
        lo3, hi3 = fill(-5.12, 3), fill(5.12, 3)
        kwp = (N = 60, Nstar = 6, extra_seeds = [fill(0.7, 3)], local_maxeval = 80, polish_maxeval = 60)
        err = try tiktak(tt_rast, lo3, hi3; kwp..., state_path = p, resume = true); nothing catch e; e end
        @test err isa TikTak.ResumeRefused && occursin("tiktak_source_sha", err.msg)   # other software: explicit only
        r = tiktak(tt_rast, lo3, hi3; kwp..., state_path = p, resume = true, allow_optimizer_change = true)
        full = tiktak(tt_rast, lo3, hi3; kwp...)
        # (2026-10-02) the same path; the restarts the fixture committed under the 2026-09-27 code keep
        # their counts, each higher by its known-start saving (restart 1: 2, any other: 1)
        fx = TOML.parsefile(joinpath(REPO, "tools", "testdata", "tiktak_state_schema1_paused.toml"))
        old_js = [Int(x["j"]) for x in fx["records"]]
        @test resume_fingerprint_core(r) == resume_fingerprint_core(full)
        @test r.n_eval == full.n_eval + sum(j == 1 ? 2 : 1 for j in old_js)
        d = TOML.parsefile(p)
        @test d["schema_version"] == 2 && any(e -> e["kind"] == "schema_migration", d["events"])
    end
    let p = joinpath(mktempdir(SCRATCH), "s.toml")
        src = joinpath(REPO, "tools", "testdata", "tiktak_state_schema1_inflight.toml")
        cp(src, p)
        r = tiktak(tt_rast, lo2, hi2; kwa..., state_path = p, resume = true, allow_optimizer_change = true)
        rec2 = only(x for x in r.records if x.j == 2)
        @test once_each(r, 4) && rec2.attempt == 2 && rec2.x0 == Float64.(TOML.parsefile(src)["inflight"][1]["x0"])
    end
end

# =============================================================================
# Follow-up 3 (2026-09-28): objective errors vs lost workers, and the worker lifecycle --
# findings 12 and 16, plan 7.1-7.2
# =============================================================================
"A master-side failure raised from the checkpoint callback once restart `j` is recorded in flight."
fail_when_inflight(j) = (crashed = Ref(false);
                         st -> (haskey(st.inflight, j) && !crashed[] && (crashed[] = true; error("master failure while restart 2 runs"))))

group("lifecycle") do
    lo3, hi3 = fill(-5.12, 3), fill(5.12, 3)
    lo2, hi2 = [-1.0, -1.0], [1.0, 1.0]
    kwl = (N = 40, Nstar = 6, local_maxeval = 30, skip_polish = true)
    akw(ws, key; n = 2) = (local_mode = :async_process, local_workers = ws, local_count = n, objective_key = key)

    with_workers(2) do ws
        # ---- finding 12: where an exception came from decides what it means ------------------
        w = ws[1]
        for (what, thrower, kind) in (("EOFError", () -> throw(EOFError()), :error),
                                      ("IOError", () -> throw(Base.IOError("broken input", -5)), :error),
                                      ("ErrorException", () -> error("boom"), :error),
                                      ("WorkerBusyError", () -> throw(TikTak.WorkerBusyError(0, "a job", "another")), :busy))
            e = try remotecall_fetch(thrower, w); nothing catch e_; e_ end
            @testset "delivered by a live worker: $what -> $kind" begin
                @test TikTak.classify_remote_error(e; worker = w)[1] === kind
                @test w in workers()
            end
        end
        @test TikTak.classify_remote_error(EOFError(); worker = w)[1] === :worker_lost          # transport, not delivered
        @test TikTak.classify_remote_error(Base.IOError("closed", -32))[1] === :worker_lost
        @test TikTak.classify_remote_error(ProcessExitedException(99))[1] === :worker_lost
        @test TikTak.classify_remote_error(RemoteException(w, CapturedException(ProcessExitedException(99), backtrace()));
                                           worker = w)[1] === :error                          # the objective's own nested call
        @test TikTak.classify_remote_error(RemoteException(w, CapturedException(EOFError(), backtrace()));
                                           worker = 12345)[1] === :worker_lost                # a worker that no longer exists

        # the audit's case: an objective reading a truncated file in restart 3 -- an ERROR, not retried
        let dir = mktempdir(SCRATCH), path = joinpath(dir, "s.toml")
            @everywhere ws TT_FAIL_J[] = 3
            err = try tiktak(tt_rast, lo3, hi3; kwl..., akw(ws, :eof)..., state_path = path, drain_timeout = 20.0); nothing catch e; e end
            @test err !== nothing && occursin("EOFError", sprint(showerror, err))
            d = TOML.parsefile(path)
            @test d["counters"]["retries"] == 0 && d["counters"]["jobs_lost"] == 0 && d["counters"]["attempts_unknown"] >= 1
            @test !any(e -> e["kind"] == "worker_lost", d["events"])
            @test all(in(workers()), ws) && isempty(TikTak.wait_idle(ws; timeout = 20))   # both still in service
            r = tiktak(tt_rast, lo3, hi3; kwl..., akw(ws, :rast)..., state_path = path, resume = true)
            @test once_each(r, 6) && only(x for x in r.records if x.j == 3).attempt == 2
        end
        # pre-testing: an IOError delivered by a live worker stops the stage, never retried
        let err = try tiktak(tt_rast, lo3, hi3; N = 40, Nstar = 3, skip_polish = true, local_maxeval = 5,
                             pretest_workers = ws, objective_key = :ioerr_pretest, drain_timeout = 20.0); nothing catch e; e end
            @test err !== nothing && occursin("IOError", sprint(showerror, err)) && !occursin("lost its worker", sprint(showerror, err))
            @test all(in(workers()), ws) && isempty(TikTak.wait_idle(ws; timeout = 20))
        end
        # the polish: an EOFError from a live worker is the objective's error
        let err = try tiktak(tt_rast, lo3, hi3; N = 30, Nstar = 2, local_maxeval = 10, polish_maxeval = 20,
                             akw(ws, :eof_polish; n = 1)...); nothing catch e; e end
            @test err isa TikTak.RemoteJobError && err.kind === :error && err.attempt == 1
            @test all(in(workers()), ws)
        end

        # ---- the worker's own guard: a second job is refused, never interleaved ---------------
        let w2 = ws[2]
            @everywhere [w2] TikTak.register_objective!(:sleep15, x -> (sleep(1.5); sum(abs2, x)))
            t = @async remotecall_fetch(TikTak.worker_eval, w2, :sleep15, [0.0, 0.0], :rethrow)
            sleep(0.5)
            @test remotecall_fetch(TikTak.active_work, w2) !== nothing
            e = try remotecall_fetch(TikTak.worker_eval, w2, :sleep15, [0.0, 0.0], :rethrow); nothing catch e_; e_ end
            @test TikTak.classify_remote_error(e; worker = w2)[1] === :busy
            @test fetch(t)[1] == 0.0                                    # the first one finished undisturbed
            @test remotecall_fetch(TikTak.active_work, w2) === nothing
        end
    end

    # ---- finding 16: a job that outlives the abort's drain deadline ------------------------
    kws = (N = 20, Nstar = 4, local_maxeval = 40, skip_polish = true)   # restart 2 alone takes ~8 s
    with_workers(2) do ws
        dir = mktempdir(SCRATCH); path = joinpath(dir, "s.toml")
        err = try tiktak(tt_rast, lo2, hi2; kws..., akw(ws, :sleepy)..., state_path = path,
                         on_checkpoint = fail_when_inflight(3), drain_timeout = 0.05); nothing catch e; e end
        @test err !== nothing && occursin("master failure", sprint(showerror, err))
        busy = TikTak.busy_workers()
        @test length(busy) == 1 && busy[1] in ws                      # the master KNOWS the worker still runs
        @test only(TikTak.worker_leases()).j == 2
        @test remotecall_fetch(TikTak.current_job, busy[1]) !== nothing
        d = TOML.parsefile(path)
        @test any(e -> e["kind"] == "unsettled", d["events"]) && d["counters"]["attempts_unknown"] >= 1
        @test sort([x["j"] for x in d["inflight"]]) == [2, 3]          # both replayed on resume
        # IMMEDIATE reuse of the same pool: the busy worker is quarantined, never given a second job
        other = only(setdiff(ws, busy))
        p2 = joinpath(dir, "s2.toml")
        r = tiktak(tt_rast, lo2, hi2; kws..., akw(ws, :rast)..., state_path = p2, busy_wait = 0.0)
        @test r.status === :complete && all(x -> x.worker == other, r.records)
        @test any(e -> e["kind"] == "workers_quarantined", TOML.parsefile(p2)["events"])
        if !isempty(TikTak.busy_workers())                              # (still solving: a pool of it alone is refused)
            @test_throws ArgumentError tiktak(tt_rast, lo2, hi2; kws..., akw(busy, :rast; n = 1)..., busy_wait = 0.0)
        end
        # the lease ends when the call settles; the worker is then usable and the aborted run resumes
        @test isempty(TikTak.wait_idle(ws; timeout = 60))
        @test remotecall_fetch(TikTak.active_work, busy[1]) === nothing
        r2 = tiktak(tt_rast, lo2, hi2; kws..., akw(ws, :rast)..., state_path = path, resume = true)
        @test once_each(r2, 4)
    end
    # an OWNED pool: retire_after_abort removes the worker still solving, keeps the other
    with_workers(2) do ws
        err = try tiktak(tt_rast, lo2, hi2; kws..., akw(ws, :sleepy)..., state_path = joinpath(mktempdir(SCRATCH), "s.toml"),
                         on_checkpoint = fail_when_inflight(3), drain_timeout = 0.05, retire_after_abort = true); nothing catch e; e end
        @test err !== nothing
        t0 = time()
        while length(intersect(ws, workers())) == 2 && time() - t0 < 30; sleep(0.2); end
        @test length(intersect(ws, workers())) == 1
        @test isempty(TikTak.wait_idle(ws; timeout = 30))              # its call settled (the process is gone)
    end
    # pre-testing: a master-side exception (a callback) stops the stage in order; values survive
    with_workers(2) do ws
        dir = mktempdir(SCRATCH); cache = joinpath(dir, "cache.toml")
        k = Ref(0)
        cb = (i, n, fx, best) -> (i < n && (k[] += 1) == 10 && error("callback failure"))
        err = try tiktak(tt_rast, lo3, hi3; N = 40, Nstar = 3, skip_polish = true, local_maxeval = 5, pretest_workers = ws,
                         objective_key = :slow_all, pretest_cache = cache, pretest_chunk = 4, on_sobol = cb); nothing catch e; e end
        @test err !== nothing && occursin("callback failure", sprint(showerror, err))
        @test isempty(TikTak.wait_idle(ws; timeout = 30))
        n = Ref(0)
        r = tiktak(x -> (n[] += 1; tt_rast(x)), lo3, hi3; N = 40, Nstar = 3, skip_polish = true, local_maxeval = 5,
                   pretest_cache = cache, resume = :auto)
        @test r.pretest.reused >= 10 && n[] == 40 - r.pretest.reused + sum(x.n_eval for x in r.records)
    end

    # ---- 7.1: an explicit thread budget beats an inherited JULIA_NUM_THREADS -----------------
    let script = joinpath(mktempdir(SCRATCH), "threads.jl")
        write(script, """
            using Distributed
            include(joinpath("$REPO", "code", "src", "tiktak.jl"))
            a = TikTak.start_workers(1; project = "$REPO")
            b = addprocs(1; exeflags = `--project=$REPO`)                 # no explicit thread count
            @everywhere include(joinpath("$REPO", "code", "src", "tiktak.jl"))
            @everywhere using LinearAlgebra
            @everywhere LinearAlgebra.BLAS.set_num_threads(1)
            for r in TikTak.worker_resources(vcat(a, b)); println("RES ", r.id, " ", r.julia_threads, " ", r.blas_threads, " ", r.env_julia_num_threads); end
            for w in vcat(a, b); println("NT ", remotecall_fetch(() -> Threads.nthreads(), w)); end
            rmprocs(workers())
            """)
        out = read(addenv(`$(Base.julia_cmd()) --project=$REPO --startup-file=no --threads=1 $script`, "JULIA_NUM_THREADS" => "3"), String)
        res = [split(l)[2:end] for l in split(out, '\n') if startswith(l, "RES ")]
        @test length(res) == 2 && res[1][2] == "1" && res[2][2] == "3"   # explicit 1; inherited 3
        @test all(r -> r[3] == "1" && r[4] == "3", res)
        @test [parse(Int, split(l)[2]) for l in split(out, '\n') if startswith(l, "NT ")] == [1, 3]   # report = execution
    end
end

# =============================================================================
# Step 8: development presets and production-prefix runs
# =============================================================================
group("presets") do
    lo3, hi3 = fill(-5.12, 3), fill(5.12, 3)
    # a standalone five-restart test walks the WHOLE schedule, explore to exploit
    r5 = tiktak(tt_rast, lo3, hi3; N = 50, Nstar = 5, local_maxeval = 20, skip_polish = true, purpose = "smoke")
    @test [round(t.theta; digits = 3) for t in r5.trace] == [0.0, 0.632, 0.775, 0.894, 0.995]
    @test r5.purpose == "smoke"

    # the first five restarts of a PLANNED 1000-restart run are a different thing: theta 0.1
    dir = mktempdir(SCRATCH); path = joinpath(dir, "state.toml"); cache = joinpath(dir, "cache.toml")
    kwk = (N = 10_000, Nstar = 1000, local_maxeval = 30, skip_polish = true, purpose = "production")
    p5 = tiktak(tt_rast, lo3, hi3; kwk..., state_path = path, pretest_cache = cache, stop_after_restarts = 5)
    @test p5.status === :paused && p5.nstar_effective == 1000 && length(p5.records) == 5
    @test [t.theta for t in p5.trace] == [0.0, 0.1, 0.1, 0.1, 0.1]
    d = TOML.parsefile(path)
    @test d["progress"]["next_j"] == 6 && length(d["seeds"]["x"]) == 1000 && d["purpose"] == "production"
    # the prefix continues at restart 6 with the same seeds and K ...
    p8 = tiktak(tt_rast, lo3, hi3; kwk..., state_path = path, pretest_cache = cache, resume = true, stop_after_restarts = 8)
    @test [x.j for x in sort(p8.records; by = r -> r.j)] == collect(1:8) && p8.status === :paused
    # ... exactly as a run paused at 8 in one go (the pre-testing pool reused, not re-evaluated)
    dir2 = mktempdir(SCRATCH); n = Ref(0)
    q8 = tiktak(x -> (n[] += 1; tt_rast(x)), lo3, hi3; kwk..., state_path = joinpath(dir2, "state.toml"),
                pretest_reuse = cache, stop_after_restarts = 8)
    @test q8.pretest.reused == 10_000 && n[] == sum(r.n_eval for r in q8.records)
    @test [(r.j, r.theta, r.x0, r.x, r.f_local) for r in sort(p8.records; by = r -> r.j)] ==
          [(r.j, r.theta, r.x0, r.x, r.f_local) for r in sort(q8.records; by = r -> r.j)]
    @test all(r -> r.theta == 0.1 || r.j == 1, q8.records)          # still the K = 1000 schedule

    # preset resolution
    @test TikTak.resolve_preset("smoke").skip_polish && !TikTak.resolve_preset("smoke").requires_convergence
    @test !TikTak.resolve_preset("production").allow_fewer_restarts && TikTak.resolve_preset("production").nstar == 1000
    @test TikTak.resolve_preset("").name == "custom"
    @test_throws ArgumentError TikTak.resolve_preset("final")
    @test_throws ArgumentError tiktak(tt_rast, lo3, hi3; N = 10, Nstar = 2, purpose = "final")
    # execution vs estimation: a smoke run that executed cleanly passes without convergence
    notconv = TikTak.acceptance(; execution_ok = true, candidate_valid = true, local_converged = false,
                                search_budget_complete = true)
    @test startswith(TikTak.execution_verdict("smoke", notconv), "smoke: execution PASSED")
    @test startswith(TikTak.execution_verdict("production", notconv), "production: NOT ACCEPTED")
    crashed = TikTak.acceptance(; execution_ok = false, candidate_valid = true, local_converged = false,
                                search_budget_complete = true)
    @test startswith(TikTak.execution_verdict("smoke", crashed), "smoke: execution FAILED")
end

# =============================================================================
# Step 9: search geometry and initial steps (options; the defaults are the baseline)
# =============================================================================
group("geometry") do
    lo, hi = [-1.0, 0.0, -300.0], [3.0, 1e-3, 700.0]               # very different widths
    # the normalized map is reversible, and mixing in u is mixing in x
    for x in ([0.3, 5e-4, 12.0], lo, hi, [2.9999999, 1e-9, -299.5])
        u = TikTak.to_unit(x, lo, hi)
        @test all(0 .<= u .<= 1) && isapprox(TikTak.from_unit(u, lo, hi), x; rtol = 4eps(), atol = 1e-12)
    end
    s, z, θ = [0.1, 2e-4, 100.0], [2.0, 9e-4, -50.0], 0.37
    xm = TikTak.make_start!(zeros(3), s, z, θ, lo, hi)
    um = (1 - θ) .* TikTak.to_unit(s, lo, hi) .+ θ .* TikTak.to_unit(z, lo, hi)
    @test isapprox(TikTak.from_unit(um, lo, hi), xm; rtol = 1e-12)

    # the default path is untouched; a normalized run starts from the same points and converges
    f(x) = sum(abs2, (x .- [1.0, 5e-4, 200.0]) ./ (hi .- lo))
    a = tiktak(f, lo, hi; N = 40, Nstar = 4)
    b = tiktak(f, lo, hi; N = 40, Nstar = 4, normalize = true)
    @test a.records[1].x0 == b.records[1].x0                    # the start is a search-coordinate point
    @test b.f < 1e-12 && b.config.normalize && !a.config.normalize
    @test all(r -> all(lo .<= r.x .<= hi), b.records)           # endpoints mapped back into the box

    # an explicit initial step reaches NLopt: the first simplex vertex is x0 + step * width * e_1
    for (norm, frac) in ((false, 0.05), (true, 0.05), (false, 0.2))
        pts = Vector{Float64}[]
        g(x) = (push!(pts, copy(x)); f(x))
        r = tiktak(g, lo, hi; N = 20, Nstar = 1, skip_polish = true, local_maxeval = 5, normalize = norm,
                   local_initial_step = frac)
        x0 = r.records[1].x0
        # pts: 20 Sobol', then the start evaluation, then NLopt's first vertices
        k = findfirst(i -> i > 21 && pts[i] != x0, eachindex(pts))
        step = pts[k] .- x0
        d = findfirst(!iszero, step)
        @test isapprox(abs(step[d]), min(frac * (hi[d] - lo[d]), hi[d] - x0[d]); rtol = 1e-9) ||
              isapprox(abs(step[d]), frac * (hi[d] - lo[d]); rtol = 1e-9)
    end
    # the theta-shrink schedule: restart j's step is initial_step * max(step_min, 1 - theta_j)
    st_scale = Float64[]
    r = tiktak(f, lo, hi; N = 30, Nstar = 5, skip_polish = true, local_maxeval = 5, local_initial_step = 0.2,
               local_step_schedule = :theta_shrink, local_step_min = 0.1, state_path = joinpath(mktempdir(SCRATCH), "s.toml"))
    for rec in r.records
        push!(st_scale, max(0.1, 1 - rec.theta))
    end
    @test st_scale[1] == 1.0 && st_scale[end] == 0.1              # wide first, narrow last
    @test_throws ArgumentError tiktak(f, lo, hi; N = 10, Nstar = 2, local_step_schedule = :theta_shrink)   # needs a step
    @test_throws ArgumentError tiktak(f, lo, hi; N = 10, Nstar = 2, local_initial_step = 1.5)
    # the settings are part of the optimizer identity: a resume with another step is refused
    let dir = mktempdir(SCRATCH), path = joinpath(dir, "s.toml")
        tiktak(f, lo, hi; N = 30, Nstar = 4, skip_polish = true, local_initial_step = 0.1, state_path = path, stop_after_restarts = 2)
        err = try tiktak(f, lo, hi; N = 30, Nstar = 4, skip_polish = true, local_initial_step = 0.2, state_path = path, resume = true); nothing catch e; e end
        @test err isa TikTak.ResumeRefused && occursin("initial_step", err.msg)
    end
end

# =============================================================================
# J1 / J3: concrete inferred types on the hot kernels, and no aliasing of scratch storage
# =============================================================================
group("julia") do
    lo, hi = fill(-5.12, 3), fill(5.12, 3)
    r = tiktak(tt_rast, lo, hi; N = 20, Nstar = 2, skip_polish = true, local_maxeval = 10)
    cfg = r.config
    st = TikTak.new_state(cfg, lo, hi, [fill(0.1, 3), fill(0.2, 3)], [1.0, 2.0], [1, 2], 2, 2, r.pretest,
                          "obj", Dict{String,Any}(), "opt", Dict{String,Any}(), "run")
    buf = zeros(3)
    # inference on the actual argument types (plan J1: not trivial substitutes)
    @inferred TikTak.theta_at(cfg, 3, 10)
    @inferred TikTak.make_start!(buf, st.seeds[1], st.inc.x, 0.3, lo, hi)
    job = @inferred TikTak.restart_job(st, 1, 1, buf)
    res = @inferred TikTak.run_local(tt_rast, job)
    @test res isa TikTak.RestartResult
    @inferred TikTak.merge_candidate(st.inc, res.x, res.f, res.ret, :local, 1, 1, lo, hi, cfg, "obj", "s")
    @inferred TikTak.boxdist(res.x, st.inc.x, lo, hi)
    @inferred TikTak.local_converged(st.inc, "obj")
    @inferred TikTak.valid_value(1.0, Inf)
    rec = @inferred TikTak.commit_restart!(st, job, res)
    @test rec isa TikTak.RestartRecord
    open(joinpath(SCRATCH, "code_warntype_run_local.txt"), "w") do io
        code_warntype(io, TikTak.run_local, (typeof(tt_rast), TikTak.RestartJob, TikTak.NoProgress))
    end
    # the mixing kernel allocates nothing once compiled (measured after warm-up)
    TikTak.make_start!(buf, st.seeds[2], st.inc.x, 0.5, lo, hi)
    @test (@allocated TikTak.make_start!(buf, st.seeds[2], st.inc.x, 0.5, lo, hi)) == 0

    # scratch reuse never reaches a seed or a dispatched start (plan J3)
    seeds_before = deepcopy(st.seeds)
    j2 = TikTak.restart_job(st, 2, 1, buf)
    x0_before = copy(j2.x0)
    fill!(buf, 99.0)                                             # reuse the scratch buffer
    TikTak.make_start!(buf, st.seeds[1], st.inc.x, 0.9, lo, hi)
    @test j2.x0 == x0_before && st.seeds == seeds_before && j2.x0 !== buf
    @test_throws ArgumentError TikTak.make_start!(st.seeds[1], st.seeds[1], st.inc.x, 0.5, lo, hi)
    # an objective that overwrites its argument cannot corrupt the recorded seeds or starts
    vandal(x) = (v = tt_rast(x); x .= 0.0; v)
    dir = mktempdir(SCRATCH); path = joinpath(dir, "s.toml")
    rv = tiktak(vandal, lo, hi; N = 30, Nstar = 4, skip_polish = true, local_maxeval = 15, state_path = path)
    d = TOML.parsefile(path)
    seeds = [Float64.(x) for x in d["seeds"]["x"]]
    @test all(s -> any(!iszero, s), seeds)                       # nothing zeroed in the checkpoint
    @test all(rec -> rec.x0 == clamp.((1 - rec.theta) .* seeds[rec.j] .+ rec.theta .*
                                      incumbent_at(rv.records, seeds, rec.incumbent_version), lo, hi), rv.records)
end

# a worker must not outlive a killed master (the orphan workers of 2026-09-27, VERSION.md step 20)
group("orphans") do
    script = joinpath(mktempdir(SCRATCH), "master.jl")
    write(script, """
        using Distributed
        ws = addprocs(1; exeflags = `--project=$REPO --threads=1 --startup-file=no`)
        @everywhere ws include(joinpath("$REPO", "code", "src", "tiktak.jl"))
        ok = remotecall_fetch(() -> TikTak.die_with_parent!(), ws[1])
        println("WORKER ", remotecall_fetch(getpid, ws[1]), " ", ok); flush(stdout)
        remote_do(() -> (while true; x = sum(rand(1000)); end), ws[1])    # a busy worker, never yielding
        sleep(600)
        """)
    io = Pipe()
    master = run(pipeline(`$(Base.julia_cmd()) --startup-file=no --threads=1 $script`; stdout = io, stderr = devnull); wait = false)
    close(io.in)
    line = readline(io)
    m = match(r"WORKER (\d+) (true|false)", line)
    @test m !== nothing && m.captures[2] == "true"
    if m !== nothing
        pid = parse(Int, m.captures[1])
        sleep(1.0)
        @test !proc_gone(pid)                                    # alive and busy while the master lives
        kill(master, Base.SIGKILL)                               # the master is killed, as by tmux kill-session
        t0 = time()
        while !proc_gone(pid) && time() - t0 < 15; sleep(0.2); end
        @test proc_gone(pid)                                     # the busy worker died with it
        proc_gone(pid) || ccall(:kill, Cint, (Cint, Cint), pid, 9)
    end
end

GC.gc()
println("\ntest_tiktak.jl: finished (scratch ", SCRATCH, ")")
# =============================================================================
# v1, 2026-10-02 (Ali, after the memo-19 pilot): a known start value is not recomputed, and a
# penalised MIXED start falls back to the restart's own seed (localsearch.jl, run_local).
# =============================================================================
group("start_rules") do
    lo, hi = fill(-5.0, 2), fill(5.0, 2)
    s = TikTak.SolverSettings(:LN_NELDERMEAD, 1e-6, 1e-12, 1e-10, 60)
    # two wells separated by a penalised band |x1| < 4: mixing across the band is penalised, and a
    # Nelder-Mead simplex started inside it sees only the penalty
    pen = x -> abs(x[1]) < 4 ? 1e12 : min(sum(abs2, x .- [4.5, 0.0]), sum(abs2, x .+ [4.5, 0.0]) + 0.5)
    # a job that evaluates its start: the start is evaluated ONCE (the solver's first call reuses it)
    let x0 = [1.3, -0.7], at = Ref(0), calls = Ref(0)
        f = x -> (calls[] += 1; x == x0 && (at[] += 1); sum(abs2, x .- 0.5))
        job = TikTak.RestartJob("t", :local, 2, 1, 0.5, copy(x0), 1, true, NaN, s, lo, hi, :rethrow, :default, 1, Inf, 1.0, false, 1)
        r = TikTak.run_local(f, job)
        @test at[] == 1 && r.n_eval == calls[] && r.f_start == sum(abs2, x0 .- 0.5) && !r.start_fallback
    end
    # a job whose start value is known (restart 1's seed, the polish's incumbent): not evaluated there at all
    let x0 = [1.3, -0.7], at = Ref(0)
        f = x -> (x == x0 && (at[] += 1); sum(abs2, x .- 0.5))
        job = TikTak.RestartJob("t", :local, 1, 1, 0.0, copy(x0), 1, false, sum(abs2, x0 .- 0.5), s, lo, hi,
                                :rethrow, :default, 1, Inf, 1.0, false, 1)
        r = TikTak.run_local(f, job)
        @test at[] == 0 && r.f_start == sum(abs2, x0 .- 0.5)
    end
    # a penalised mixed start falls back to its seed (value known); without the fallback -- the
    # behaviour before -- the restart stalls on the flat penalty and is lost
    let seed = [-4.5, 1.0], x0 = [0.0, 0.5]
        job_fb = TikTak.RestartJob("t", :local, 3, 1, 0.5, copy(x0), 1, true, NaN, s, lo, hi, :rethrow, :default, 1, Inf,
                                   1.0, false, 1, copy(seed), pen(seed), 1e12)
        r = TikTak.run_local(pen, job_fb)
        @test r.start_fallback && r.f_start == pen(seed) && r.f < pen(seed)
        job_old = TikTak.RestartJob("t", :local, 3, 1, 0.5, copy(x0), 1, true, NaN, s, lo, hi, :rethrow, :default, 1, Inf, 1.0, false, 1)
        r0 = TikTak.run_local(pen, job_old)
        @test !r0.start_fallback && r0.f >= 1e12
    end
    # end to end, serially: a restart falls back EXACTLY when its mixed start (the dispatched x0) is
    # penalised, and then starts at a valid value; no restart ends penalised. (In two dimensions the
    # old behaviour sometimes escapes the band by its first simplex, so the stall itself is shown by
    # the job above, not here.)
    let kw = (N = 200, Nstar = 8, invalid_value = 1e12, skip_polish = true, local_maxeval = 80)
        r = tiktak(pen, lo, hi; kw...)
        @test any(x -> x.start_fallback, r.records)
        @test all(x -> x.start_fallback == (pen(x.x0) >= 1e12), r.records)
        @test all(x -> x.f_local < 1e12 && x.f_start < 1e12, r.records)
    end
    # checkpoints: a job saved with the rule replays with it; a job saved before it replays as it ran
    let cfgr = tiktak(x -> sum(abs2, x), lo, hi; N = 4, Nstar = 1, skip_polish = true, local_maxeval = 2).config
        ep = [TikTak.SettingsEpoch(1, cfgr.local_, cfgr.polish, false, true, "id", 1, "t")]
        seeds = [[1.0, 1.0], [-3.0, 2.0]]; seed_f = [2.0, 13.0]
        new = Dict{String,Any}("j" => 2, "attempt" => 1, "stage" => "local", "theta" => 0.4, "x0" => [0.5, 0.5],
                               "incumbent_version" => 1, "worker" => 2, "dispatched" => 0.0, "dispatch_seq" => 2,
                               "commits_at_dispatch" => 0, "step_scale" => 1.0, "epoch" => 1,
                               "eval_start" => true, "f_start_known" => NaN, "fallback" => true)
        fl = TikTak.inflight_from(new, "t", cfgr, ep, lo, hi, seeds, seed_f)
        @test fl.job.fallback_x == seeds[2] && fl.job.fallback_f == 13.0 && fl.job.eval_start
        @test TikTak.inflight_dict(fl)["fallback"] == true
        old = Dict{String,Any}(k => v for (k, v) in new if !(k in ("eval_start", "f_start_known", "fallback")))
        fo = TikTak.inflight_from(old, "t", cfgr, ep, lo, hi, seeds, seed_f)
        @test isempty(fo.job.fallback_x) && fo.job.eval_start && isnan(fo.job.f_start_known)
        @test TikTak.inflight_dict(fo)["fallback"] == false
    end
end

if isempty(FAILED_GROUPS)
    println("ALL GROUPS PASSED")
else
    println("FAILED GROUPS: ", join(FAILED_GROUPS, ", ")); exit(1)
end
