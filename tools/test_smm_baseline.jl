# Full-grid regression: julia --threads=1 --project=. tools/test_smm_baseline.jl
using Test, TOML, SHA, Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames
using Statistics, Dates, ProgressMeter, Distributions, StatsBase
using QuantEcon, FastGaussQuadrature, Parameters, Dierckx
BLAS.set_num_threads(1)
const REPO=normpath(joinpath(@__DIR__,".."))
for f in ("paths.jl","manifest.jl","diagnostics.jl","child_lifecycle.jl","parent_family.jl")
    include(joinpath(REPO,"code/src",f))
end
include(joinpath(REPO,"code/smm/moments.jl"))
const SNAP=TOML.parsefile(joinpath(REPO,"Input/parent_baseline_9param.toml"))
@testset "Frozen nine-parameter baseline and future search bounds" begin
    @test length(SMM_PARAMS)==9
    for (name,value) in SNAP["parameters"]
        @test getproperty(PARENT_DEFAULTS,Symbol(name))==value
    end
    for (name,value) in SNAP["fixed"]
        @test getproperty(PARENT_DEFAULTS,Symbol(name))==value
    end
    lo,hi=search_bounds(); z=incumbent()
    @test all(lo .< z .< hi)
    @test smm_feasible(unpack(z))
    @test !smm_feasible((sigma_1_0=-0.1,sigma_1_1=0.05))
    @test !smm_feasible((sigma_2_0=-0.5,sigma_2_1=0.05))
    for (name,expected) in SNAP["source_sha256"]
        path=name=="targets" ? joinpath(REPO,"Input/smm_targets_baseline.toml") : joinpath(REPO,SNAP["source_run"],name)
        @test bytes2hex(sha256(read(path)))==expected
    end
end
println("Checking fitted baseline at grid 30 / simN 2000");flush(stdout)
ch=ConSavLaborCollege_AR1(;Na=30,Nk=30,Nt=5,rho=1.5,psi_terminal=0.0,kappa_terminal=5.0,
    omega=0.3,a_max=100.0,w=20.0,simN=500,seed=1234)
redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        solve_model_work!(ch);solve_model_college!(ch)
        optimal_transfer_work!(ch);optimal_transfer_college!(ch)
    end
end
V=terminal_value_spline(ch;s=10.0)
p=Parent_child_interaction_age_specific_AR1(;Na=30,Nk=2,Nhc=30,simN=2000,seed=1234)
p.V_child_interp=V
redirect_stdout(devnull) do
    solve_model!(p;verbose=false);simulate_model!(p)
end
m=model_moments(p);tg=load_targets(joinpath(REPO,"Input/smm_targets_baseline.toml"))
Q=sum(((getproperty(m,Symbol(k))-tg[k].mean)/moment_scale(k,tg[k].mean))^2 for k in SMM_MOMENTS)
@testset "Full-grid fit and handoff" begin
    @test Q ≈ SNAP["Q_final"] atol=1e-9 rtol=0
    @test simulation_violations(p).total==0
    @test all(isfinite,p.sim_hc[:,end]) && all(>(0),p.sim_hc[:,end])
    @test count(>(p.a_max),p.sim_a[:,end])==2
    @test continuation_selftest(verbose=false)
end
println("Reproduced Q=",Q,"; invalid cells=",simulation_violations(p).total)
