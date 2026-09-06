# Reproduce the saved fit and probe nearby model choices without re-estimation.
# julia --threads=1 --project=. output/smm_diagnostics/2026-09-06_183119/inspect_run.jl
using Distributed, TOML, SHA, Printf, Dates
const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
const RUN = joinpath(REPO,"output/smm_runs/2026-09-06_183119")
const OUT = @__DIR__
const CK = TOML.parsefile(joinpath(RUN,"checkpoint.toml"))
const EST = TOML.parsefile(joinpath(RUN,"estimates.toml"))
@assert bytes2hex(sha256(read(joinpath(REPO,"Input/smm_targets_baseline.toml"))))[1:16] == CK["targets_sha"]
@assert strip(read(`git -C $REPO rev-parse --short HEAD`,String)) == EST["git_commit"] "Run this diagnostic at the estimation's code revision"
const BASE = Dict(Symbol(n)=>(link=="log" ? exp(z) : z) for (n,link,z) in zip(CK["param_names"], CK["param_link"], CK["search_vector"]["z"]))
addprocs(4; exeflags=`--threads=1 --project=$REPO`)
@everywhere const REPO_ = $REPO
@everywhere begin
    using Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames
    using Statistics, Dates, ProgressMeter, Distributions, StatsBase
    using QuantEcon, FastGaussQuadrature, Parameters, Dierckx, TOML
    BLAS.set_num_threads(1)
    for file in ("paths.jl","manifest.jl","diagnostics.jl","child_lifecycle.jl","parent_family.jl")
        include(joinpath(REPO_,"code/src",file))
    end
    include(joinpath(REPO_,"code/smm/moments.jl"))
    const TARGETS_ = load_targets(joinpath(REPO_,"Input/smm_targets_baseline.toml"))
    const BASE_ = $BASE
    function child_value_()
        ch = ConSavLaborCollege_AR1(;Na=30,Nk=30,Nt=5,rho=1.5,psi_terminal=0.0,
            kappa_terminal=5.0,omega=0.3,a_max=100.0,w=20.0,simN=500,seed=1234)
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                solve_model_work!(ch); solve_model_college!(ch)
                optimal_transfer_work!(ch); optimal_transfer_college!(ch)
            end
        end
        terminal_value_spline(ch;s=10.0)
    end
    const V_ = child_value_()
    function evaluate_(job)
        kw=merge(BASE_,job.changes)
        p=Parent_child_interaction_age_specific_AR1(;Na=30,Nk=2,Nhc=30,simN=2000,seed=1234,kw...)
        p.V_child_interp=V_
        t0=time()
        try
            redirect_stdout(devnull) do
                solve_model!(p;verbose=false); simulate_model!(p)
            end
            m=model_moments(p); v=simulation_violations(p); d=moment_diagnostics(p)
            residual=[(getfield(m,Symbol(k))-TARGETS_[k].mean)/moment_scale(k,TARGETS_[k].mean) for k in SMM_MOMENTS]
            a=p.sim_a[:,p.T+1]; hc=p.sim_hc[:,p.T+1]
            return (label=job.label,changes=job.changes,seconds=time()-t0,error="",Q=sum(abs2,residual),
                residual=residual,moments=[getfield(m,Symbol(k)) for k in SMM_MOMENTS],
                violations=v.total,diagnostics=d,handoff=(assets_min=minimum(a),assets_mean=mean(a),assets_max=maximum(a),
                hc_min=minimum(hc),hc_mean=mean(hc),hc_max=maximum(hc),
                bc_share=mean(p.sim_k[:,1]),bc_constant=all(p.sim_k .== p.sim_k[:,1])))
        catch err
            return (label=job.label,changes=job.changes,seconds=time()-t0,error=sprint(showerror,err))
        end
    end
end
println("Reproducing saved winner"); flush(stdout)
base=evaluate_((label="baseline",changes=Dict{Symbol,Float64}()))
@assert isempty(base.error) base.error
@assert isapprox(base.Q,EST["Q_final"];atol=1e-8,rtol=0)
@assert base.violations==0
println("Baseline Q=",base.Q,"; handoff=",base.handoff);flush(stdout)
open(joinpath(OUT,"fit_moments.csv"),"w") do io
    println(io,"moment,target,simulated,raw_residual,scale,scaled_residual,Q_contribution,Q_share")
    for (i,k) in enumerate(SMM_MOMENTS)
        target=TARGETS_[k].mean; sim=base.moments[i]; r=base.residual[i]
        println(io,join((k,target,sim,sim-target,moment_scale(k,target),r,r^2,r^2/base.Q),','))
    end
end
open(joinpath(OUT,"fit_diagnostics.toml"),"w") do io
    TOML.print(io,Dict("run"=>RUN,"git_commit"=>EST["git_commit"],"generated"=>string(now()),"Q_reproduced"=>base.Q,
        "n_invalid"=>base.violations,"handoff"=>Dict(string(k)=>v for (k,v) in pairs(base.handoff)),
        "grid_coverage"=>Dict(string(k)=>v for (k,v) in pairs(base.diagnostics))))
end
jobs=NamedTuple[]
for (name,vals) in ((:R_1,[-1.0,-0.5,0.5,1.0]),(:sigma_4_1,[0.0,0.01,0.03,0.04,0.08,0.12]),
                  (:mu_1,[-0.06,-0.05,-0.03,-0.02]),(:sigma_1_0,[-0.2,-0.15]),
                  (:sigma_2_1,[-0.05,-0.075,-0.10]),(:sigma_4_0,[-6.0,-6.2,-6.5]))
    for val in vals
        push!(jobs,(label="$(name)=$(val)",changes=Dict(name=>val)))
    end
end
# Illustrative study-elasticity pivots. Keep elasticity at child age 11 fixed while changing its slope.
# The changed intercept lies outside the estimation box: explicitly a bound/specification diagnostic.
for slope in (0.04,0.08,0.12)
    push!(jobs,(label="study_pivot_slope=$(slope)",changes=Dict(:sigma_4_1=>slope,:sigma_4_0=>BASE[:sigma_4_0]-6*(slope-0.02))))
end
println("Evaluating ",length(jobs)," controlled probes on four processes; no parameters are optimized");flush(stdout)
rows=pmap(evaluate_,jobs)
open(joinpath(OUT,"probes.csv"),"w") do io
    println(io,"case,Q,delta_Q,seconds,invalid,",join(SMM_MOMENTS,','),",error")
    for row in rows
        if isempty(row.error)
            println(io,join((row.label,row.Q,row.Q-base.Q,row.seconds,row.violations,row.moments... ,""),','))
            println(row.label," Q=",row.Q," delta=",row.Q-base.Q," invalid=",row.violations)
        else
            println(io,row.label,",NaN,NaN,",row.seconds,",NaN,",repeat(",",length(SMM_MOMENTS)),repr(row.error))
            println(row.label," FAILED ",row.error)
        end
    end
end
open(joinpath(OUT,"probe_settings.toml"),"w") do io
    TOML.print(io,Dict("note"=>"Controlled slices, not re-estimated models; baseline nine parameters fixed except listed changes. No identification or global optimality claim.",
        "grid"=>30,"simN"=>2000,"seed"=>1234,"baseline"=>Dict(string(k)=>v for (k,v) in BASE),
        "probes"=>[Dict("label"=>job.label,"changes"=>Dict(string(k)=>v for (k,v) in job.changes)) for job in jobs]))
end
rmprocs(workers())
println("DONE")
