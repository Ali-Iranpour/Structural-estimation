#!/usr/bin/env julia
# =============================================================================
# test_penalties.jl -- the objective's validity rules that the optimizer relies on
#
#     julia --project=. tools/test_penalties.jl [targets.toml]
#
# Ported on 2026-10-02 from apps/Structural-estimation-v2 (2026-09-20 penalty audit) with the TikTak
# module. Only the sections that test rules this repository shares with v2 are here -- the rules
# ported on 2026-09-27 (penalty 1e12, named failure sites, the origin rule, TikTak's invalid_value):
#   9   a deliberately injected programming error is NOT a model failure
#   10  a budget / domain violation is counted by simulation_violations; a float-sized
#       deviation (1e-12) is not, a substantive one (1e-3) is
#   17  exception origin: DomainError / InexactError are model failures only from the
#       solver's own files; TikTak never seeds a restart from a penalised point
# v2's sections 1-8 and 11-16 test its v5 transfer and college-weight moment functions
# (work-path transfers, the escrow flow, log-space weights), which this model does not have.
# The numbering follows v2's file so the two can be compared.
# =============================================================================
using Test, Printf, Random, NLopt, LinearAlgebra, Interpolations, Statistics, Distributions,
      QuantEcon, FastGaussQuadrature, Parameters, Dierckx, ProgressMeter, TOML, DataFrames, StatsBase, Dates
BLAS.set_num_threads(1)
const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "code", "src", "paths.jl"))
include(joinpath(REPO, "code", "src", "child_lifecycle.jl"))
include(joinpath(REPO, "code", "src", "parent_family.jl"))
include(joinpath(REPO, "code", "smm", "moments.jl"))
include(joinpath(REPO, "code", "src", "tiktak.jl"))

const TFILE = length(ARGS) >= 1 ? ARGS[1] : smm_targets_file()
const T = load_targets(TFILE)
# the --quick grids of run_smm.jl, at the SMM starting values (a valid point there)
const G  = (Na = 12, Nk = 2, Nhc = 12); const CG = (Na = 12, Nk = 12, Nt = 3); const N = 300
base() = Dict{Symbol,Float64}(q.name => param_default(q.name) for q in SMM_PARAMS)
r0 = run_pipeline(named_point(base()), T; Na = G.Na, Nk = G.Nk, Nhc = G.Nhc, simN = N, seed = 1234,
                  child_grid = CG, demo_sim = false)
m0 = model_moments(r0, T)
@printf("base point: n_nonfinite %d, simulation violations %d\n", m0.n_nonfinite, simulation_violations(r0.parent).total)

@testset "penalty and validity rules shared with v2" begin
    @test m0.n_nonfinite == 0

    # 9. a programming error is not a model failure
    @test !is_model_failure(AssertionError("length(belief_type) == simN"))
    @test !is_model_failure(ErrorException("some new bug"))
    @test !is_model_failure(MethodError(+, (1, "a")))
    @test is_model_failure(AssertionError("util_total: box bounds violated (c=NaN, i_c=NaN)"))
    @test is_model_failure(ErrorException("Period 3: only 80.0% of 100 grid points converged (floor 95.0%). Refusing to return a solution built on failed optimizations."))

    # 10. domain violations are counted with the documented tolerance
    pa = deepcopy(r0.parent)
    v0 = simulation_violations(pa).total
    pa.sim_a[1, 5] = pa.a_min - 1e-12
    @test simulation_violations(pa).total == v0
    pa.sim_a[1, 5] = pa.a_min - 1e-3
    @test simulation_violations(pa).assets_below_min == 1

    # 17. exception origin and seed selection
    @test is_model_failure(DomainError(-1.0, "log"), "parent_family.jl:util_parent:805")
    @test is_model_failure(InexactError(:Int, Int, NaN), "child_lifecycle.jl:solve_model_work!:900")
    @test !is_model_failure(DomainError(-1.0, "log"), "moments.jl:tas_moments:800")
    @test !is_model_failure(DomainError(-1.0, "log"))                      # unknown origin: visible
    @test !is_model_failure(InexactError(:Int, Int, NaN), "unknown")
    # the site of a captured callback exception is read from its stored backtrace
    let e = try; error("Period 5: only 80.0% of 100 grid points converged (floor 95.0%). Refusing to return a solution built on failed optimizations."); catch err; CapturedException(err, catch_backtrace()); end
        @test is_model_failure(_root_cause(e), failure_site(e))
    end
    @test SMM_PENALTY >= 1.0e12
    # TikTak with invalid_value: penalised candidates never seed a restart, even when they
    # are the majority; with fewer valid points than restarts the restarts are reduced
    let f = z -> (z[1] > 0.5 ? SMM_PENALTY : (z[1] - 0.2)^2 + (z[2] - 0.3)^2)
        res = tiktak(f, [0.0, 0.0], [1.0, 1.0]; N = 40, Nstar = 3, invalid_value = SMM_PENALTY,
                     local_maxeval = 30, skip_polish = true)
        @test res.f < SMM_PENALTY && res.x[1] <= 0.5
        @test all(r -> r.f_start < SMM_PENALTY, res.trace)
        res2 = @test_logs (:warn, r"fewer VALID") match_mode=:any tiktak(f, [0.0, 0.0], [1.0, 1.0]; N = 8, Nstar = 8, invalid_value = SMM_PENALTY,
                     local_maxeval = 20, skip_polish = true)
        @test length(res2.trace) < 8 && all(r -> r.f_start < SMM_PENALTY, res2.trace)
    end
end
println("PENALTY RULES: ", Test.get_testset_depth() == 0 ? "done" : "")
