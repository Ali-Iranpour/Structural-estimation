# Own study / fixed school regression: julia --threads=1 --project=. tools/test_smm_own_study.jl
using Test, Random, NLopt, LinearAlgebra, Interpolations, Statistics, Dates, Printf
using ProgressMeter, Distributions, QuantEcon, FastGaussQuadrature, Parameters, Dierckx
const REPO = normpath(joinpath(@__DIR__, ".."))
for f in ("paths.jl", "child_lifecycle.jl", "parent_family.jl")
    include(joinpath(REPO, "code/src", f))
end
include(joinpath(REPO, "code/smm/moments.jl"))
targets = load_targets(smm_targets_file())
school = target_school_time(targets)
p = Parent_child_interaction_age_specific_AR1(Na=12, Nhc=12, simN=200, school_time=school)
@testset "Own study specification and frozen targets" begin
    # Was `== 10 == 10` for the square parent-only design. The specification is now
    # fourteen parameters against seventeen moments, so this pins BOTH counts AND the
    # split, which is the part that can go wrong silently: a child parameter that
    # accidentally reports itself as parent-owned would still give 14 and 17.
    @test length(SMM_PARAMS) == 14
    @test length(SMM_MOMENTS) == 17
    @test length(SMM_PARENT_PARAMS) == 10 && length(SMM_CHILD_PARAMS) == 4
    @test length(SMM_PARENT_MOMENTS) == 10 && length(SMM_TAS_MOMENTS) == 7
    @test collect(SMM_MOMENTS) == vcat(collect(SMM_PARENT_MOMENTS), collect(SMM_TAS_MOMENTS))
    @test :sigma_4_1 in getfield.(SMM_PARAMS, :name)
    @test Set(SMM_CHILD_PARAMS) == Set((:kappa_0, :kappa_theta, :kappa_ParEd, :kappa_terminal))
    @test SMM_AGE_HC_LATE_LO == 12
    @test all(iszero, school[1:5])
    @test all(0.3 .< school[6:17] .< 0.4)
    @test targets["mean_i_c_early"].mean ≈ 0.0393 atol=1e-4
    @test targets["mean_i_c_late"].mean ≈ 0.0496 atol=1e-4
    @test targets["mean_hc_late"].mean ≈ 6.2589 atol=1e-4
    @test child_leisure(p, 0.2, 0.05, 12) ≈ 0.75-school[12]
    @test_throws ArgumentError Parent_child_interaction_age_specific_AR1(school_time=fill(NaN,17))
    @test_throws ArgumentError Parent_child_interaction_age_specific_AR1(school_time=fill(0.3,17))
    @test_throws ArgumentError Parent_child_interaction_age_specific_AR1(school_time=zeros(16))
    # Excluding ages 10/11 is verified with distinct values, not a constant trajectory.
    for t in 1:18; p.sim_hc[:,t] .= exp(t/10); end
    for f in (:sim_c,:sim_e,:sim_h,:sim_t,:sim_i); getfield(p,f) .= 0.05; end
    @test model_moments(p).mean_hc_late ≈ mean(1.2:0.1:1.7)
    # Reject legacy target semantics even when all moment names still exist.
    old = TOML.parsefile(smm_targets_file()); delete!(old,"child_time_spec")
    mktemp() do path, io
        TOML.print(io,old); close(io)
        @test_throws ErrorException load_targets(path)
    end
    # School changes the time cost, not the production function at fixed own study.
    # `legacy` must now ask for zeros EXPLICITLY: the default is the real schedule.
    legacy = Parent_child_interaction_age_specific_AR1(Na=12,Nhc=12,simN=200,school_time=zeros(17))
    @test HC_technology_full(p,0.2,0.3,500.0,0.05,12) ==
          HC_technology_full(legacy,0.2,0.3,500.0,0.05,12)
    @test child_leisure(legacy,0.2,0.05,12) ≈ 0.75

    # The DEFAULT is the real median-school schedule, and it is the same schedule the
    # frozen targets carry. A default of zeros silently solved PARENT_DEFAULTS -- which
    # were fitted WITH school -- against a budget that has none, and no test caught it.
    defaulted = Parent_child_interaction_age_specific_AR1(Na=12,Nhc=12,simN=200)
    @test defaulted.school_time == SCHOOL_TIME_BY_AGE
    @test defaulted.school_time ≈ school
    @test all(iszero, SCHOOL_TIME_BY_AGE[1:T_CHILD_VOICE-1])
    @test all(0.3 .< SCHOOL_TIME_BY_AGE[T_CHILD_VOICE:end] .< 0.4)
    @test default_school_time(17) == SCHOOL_TIME_BY_AGE
    @test length(default_school_time(20)) == 20            # longer horizon repeats age 17
    @test default_school_time(20)[18:20] == fill(SCHOOL_TIME_BY_AGE[end], 3)
end
println("Solving child and school-time parent for gradient/feasibility checks"); flush(stdout)
ch=ConSavLaborCollege_AR1(Na=30,Nk=30,Nt=5,rho=1.5,psi_terminal=0.0,
    kappa_terminal=5.0,omega=0.3,a_max=100.0,w=20.0,simN=200)
redirect_stdout(devnull) do
    solve_model_work!(ch); solve_model_college!(ch)
    optimal_transfer_work!(ch); optimal_transfer_college!(ch)
end
p.V_child_interp=terminal_value_spline(ch;s=10.0)
redirect_stdout(devnull) do
    solve_model!(p); simulate_model!(p)
end
@testset "School-time gradients and simulations" begin
    @test simulation_violations(p).total == 0
    @test all(p.sim_i[:,1:5] .== 0)
    @test maximum(p.sim_i .+ p.sim_t .+ school') <= 1+SIM_FEAS_TOL
    function fd(f,x)
        [begin
            xp=copy(x); xm=copy(x); xp[j]+=1e-6; xm[j]-=1e-6
            (f(xp,Float64[])-f(xm,Float64[]))/(2e-6)
        end for j in eachindex(x)]
    end
    for t in (6,16,17)
        x=[2.0,0.05,0.2,0.3,0.2]; g=zeros(5)
        E=t==17 ? nothing : expected_interp(p,create_interp(p,p.sol_v,t+1))
        f=(x,g)-> t==17 ? obj_last_period_full(p,x[1],x[2],x[3],x[4],x[5],
            20.0,500.0,0.0,t,1.0,p.V_child_interp,g) :
            obj_work_period_full(p,x[1],x[2],x[3],x[4],x[5],20.0,500.0,0.0,t,1.0,3,E,g)
        f(x,g)
        @test g ≈ fd(f,x) rtol=2e-5 atol=1e-6
        constraint_child_time(x,g,school[t])
        @test g ≈ fd((x,g)->constraint_child_time(x,g,school[t]),x) atol=1e-8
    end
    # The diagnostic must detect school crowding out leisure.
    bad=deepcopy(p); bad.sim_t[1,12]=0.4; bad.sim_i[1,12]=0.4
    @test simulation_violations(bad).child_leisure_negative > 0
    # Heterogeneous simulation uses the same constrained policy path.
    ph=deepcopy(p)
    redirect_stdout(devnull) do
        simulate_model_hetero!([ph],ones(Int,ph.simN);verbose=false)
    end
    @test simulation_violations(ph).total == 0
    @test ph.sim_i ≈ p.sim_i
end
println("Own study checks passed")
