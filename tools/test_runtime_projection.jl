# =============================================================================
# test_runtime_projection.jl -- the runtime scenarios run_smm.jl prints (code/smm/runtime_projection.jl;
# tiktak_fix_plan.md R1.2). Pure arithmetic: no model, no workers, a few seconds.
#
#     julia --project=. tools/test_runtime_projection.jl
# =============================================================================
using Test
include(joinpath(@__DIR__, "..", "code", "smm", "runtime_projection.jl"))

@testset "runtime projection" begin
    T = 1.0
    @testset "local stage: the closed forms for equal restarts" begin
        for K in (1, 2, 5, 20, 21, 40, 41, 1000), P in (1, 4, 20)
            L = fill(T, K)
            @test local_wall(L, P, :first_alone) == T + cld(K - 1, P) * T
            @test local_wall(L, P, :immediate_mixed) == cld(K, P) * T
        end
        @test local_wall(fill(T, 21), 20, :first_alone) == local_wall(fill(T, 21), 20, :immediate_mixed) == 2T   # E1-P
        @test local_wall(fill(T, 20), 20, :first_alone) == 2T && local_wall(fill(T, 20), 20, :immediate_mixed) == T
        @test local_wall(fill(T, 7), 1, :first_alone) == local_wall(fill(T, 7), 1, :immediate_mixed) == 7T   # serial
        @test local_wall(Float64[], 4, :first_alone) == 0.0
        @test_throws ArgumentError local_wall([1.0], 0, :first_alone)
        @test_throws ArgumentError local_wall([1.0], 2, :no_such)
    end
    @testset "local stage: unequal restarts go to the first free worker" begin
        # 2 workers, first_alone: r1 (3) alone; then r2 (5) and r3 (1) at t=3; r4 (1) at 4 on r3's worker
        @test local_wall([3.0, 5.0, 1.0, 1.0], 2, :first_alone) == 8.0
        # immediate: r1 and r2 at 0; r3 at 3 (r1's worker), r4 at 4 -> ends 5 = r2's end
        @test local_wall([3.0, 5.0, 1.0, 1.0], 2, :immediate_mixed) == 5.0
    end
    @testset "evaluations per restart" begin
        @test restart_evals(3, 1000) == [1001, 1001, 1001]                      # cap + the start evaluation
        @test restart_evals(7, 400; observed = [500, 500, 500, 423, 268]) == [401, 401, 401, 401, 268, 401, 401]
        @test restart_evals(0, 10) == Int[]
        @test_throws ArgumentError restart_evals(2, 0)
        @test_throws ArgumentError restart_evals(2, 10; observed = [0])
    end
    @testset "a whole run: E1-P's configuration" begin
        # 1,000 Sobol draws + the supplied point on 20 workers; K = 21 on 20 local workers; polish 200
        p = project_runtime(; t_eval = 42.0, n_pretest = 1001, pretest_workers = 20, evals = restart_evals(21, 1000),
                            local_workers = 20, bootstrap = :first_alone, polish_evals = 200, refine_evals = 0, n_procs = 21)
        @test p.pretest == 51 * 42.0
        @test p.local_ == 2 * 1001 * 42.0                                         # two full rounds
        @test p.polish == 200 * 42.0 && p.refine == 0.0
        @test p.wall == p.pretest + p.local_ + p.polish
        @test p.busy == (1001 + 21 * 1001 + 200) * 42.0
        @test p.reserved == p.wall * 21 && p.reserved > p.busy                    # idle workers hold cores
        @test 25 < p.wall / 3600 < 27                                            # ~ the hand estimate of 27 h
        q = project_runtime(; t_eval = 42.0, n_pretest = 1001, pretest_workers = 20, evals = restart_evals(20, 1000),
                            local_workers = 20, bootstrap = :immediate_mixed, polish_evals = 200, refine_evals = 0, n_procs = 21)
        @test q.local_ == 1001 * 42.0                                             # K = 20, at once: one round
        @test_throws ArgumentError project_runtime(; t_eval = -1.0, n_pretest = 1, pretest_workers = 1, evals = [1],
                                                   local_workers = 1, bootstrap = :first_alone, polish_evals = 0,
                                                   refine_evals = 0, n_procs = 1)
    end
    @testset "round advice" begin
        @test round_advice(21, 20, :first_alone) === nothing                      # 1 + 20: full
        @test round_advice(21, 20, :immediate_mixed) == (last = 1, more = 40, fewer = 20)
        @test round_advice(20, 20, :immediate_mixed) === nothing
        @test round_advice(20, 20, :first_alone) == (last = 19, more = 21, fewer = 1)
        @test round_advice(100, 1, :first_alone) === nothing                      # serial: no rounds
        @test round_advice(1, 20, :first_alone) === nothing
        @test round_advice(5, 20, :immediate_mixed) == (last = 5, more = 20, fewer = 0)
    end
    @test fmt_duration(90.0) == "1.5 min" && fmt_duration(7200.0) == "2.0 h"
end
