# Reproduce the provisional fit and conditional probes; no re-estimation.
# julia --threads=1 --project=. output/smm_diagnostics/2026-09-07_114138/inspect_run.jl
using TOML, SHA, Printf, Dates, Random, NLopt, LinearAlgebra, Interpolations, DataFrames
using Statistics, ProgressMeter, Distributions, StatsBase, QuantEcon
using FastGaussQuadrature, Parameters, Dierckx
BLAS.set_num_threads(1)
const REPO=normpath(joinpath(@__DIR__,"../../.."))
const RUN=joinpath(REPO,"output/smm_runs/2026-09-07_114138")
const CK=TOML.parsefile(joinpath(RUN,"checkpoint.toml"))
const EST=TOML.parsefile(joinpath(RUN,"estimates.toml"))
const TARGET_PATH=joinpath(RUN,"targets.toml")
@assert bytes2hex(sha256(read(TARGET_PATH)))[1:16]==CK["targets_sha"]
for f in ("paths.jl","manifest.jl","diagnostics.jl","child_lifecycle.jl","parent_family.jl")
    include(joinpath(REPO,"code/src",f))
end
include(joinpath(REPO,"code/smm/moments.jl"))
const TG=load_targets(TARGET_PATH)
const BASE=Dict(Symbol(n)=>(link=="log" ? exp(z) : z) for (n,link,z) in zip(CK["param_names"],CK["param_link"],CK["search_vector"]["z"]))
function child_value(amax)
    ch=ConSavLaborCollege_AR1(;Na=30,Nk=30,Nt=5,rho=1.5,psi_terminal=0.0,kappa_terminal=5.0,
        omega=0.3,a_max=amax,w=20.0,simN=500,seed=1234)
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            solve_model_work!(ch);solve_model_college!(ch)
            optimal_transfer_work!(ch);optimal_transfer_college!(ch)
        end
    end
    terminal_value_spline(ch;s=10.0)
end
println("Solving baseline child grid");flush(stdout)
const V=child_value(100.0)
function evaluate(label,changes;amax=100.0,value=V)
    kw=merge(BASE,changes)
    p=Parent_child_interaction_age_specific_AR1(;Na=30,Nk=2,Nhc=30,simN=2000,seed=1234,a_max=amax,kw...)
    p.V_child_interp=value
    started=time()
    redirect_stdout(devnull) do
        solve_model!(p;verbose=false);simulate_model!(p)
    end
    m=model_moments(p);v=simulation_violations(p);d=moment_diagnostics(p)
    residual=[(getproperty(m,Symbol(k))-TG[k].mean)/moment_scale(k,TG[k].mean) for k in SMM_MOMENTS]
    a=p.sim_a[:,end];hc=p.sim_hc[:,end]
    row=(label=label,Q=sum(abs2,residual),invalid=v.total,seconds=time()-started,
         moments=m,residual=residual,diagnostics=d,
         handoff=(a_min=minimum(a),a_mean=mean(a),a_max=maximum(a),
                  hc_min=minimum(hc),hc_mean=mean(hc),hc_max=maximum(hc),
                  bc_share=mean(p.sim_k[:,1]),bc_constant=all(p.sim_k .== p.sim_k[:,1])))
    println(label," Q=",row.Q," invalid=",row.invalid);flush(stdout)
    row
end
base=evaluate("fitted",Dict{Symbol,Float64}())
@assert isapprox(base.Q,EST["Q_final"];atol=1e-10,rtol=0)
@assert base.invalid==0
open(joinpath(@__DIR__,"fit_moments.csv"),"w") do io
    println(io,"moment,target,simulated,raw_residual,scale,scaled_residual,Q_contribution,Q_share")
    for (i,k) in enumerate(SMM_MOMENTS)
        target=TG[k].mean;sim=getproperty(base.moments,Symbol(k));r=base.residual[i]
        println(io,join((k,target,sim,sim-target,moment_scale(k,target),r,r^2,r^2/base.Q),','))
    end
end
open(joinpath(@__DIR__,"fit_diagnostics.toml"),"w") do io
    TOML.print(io,Dict("Q_reproduced"=>base.Q,"invalid"=>base.invalid,
        "handoff"=>Dict(string(k)=>v for (k,v) in pairs(base.handoff)),
        "grid_coverage"=>Dict(string(k)=>v for (k,v) in pairs(base.diagnostics))))
end
jobs=NamedTuple[]
for (name,vals) in ((:sigma_2_1,[-0.11,-0.125,-0.15]),(:sigma_4_1,[0.03,0.04,0.06]),
                   (:R_1,[-0.5,0.5]),(:mu_1,[-0.05,-0.03]))
    for val in vals
        push!(jobs,(label="$(name)=$(val)",changes=Dict(name=>val)))
    end
end
for slope in (-0.125,-0.15)
    push!(jobs,(label="money_pivot=$(slope)",changes=Dict(:sigma_2_1=>slope,
        :sigma_2_0=>BASE[:sigma_2_0]-8*(slope-BASE[:sigma_2_1]))))
end
open(joinpath(@__DIR__,"probe_settings.toml"),"w") do io
    TOML.print(io,Dict("note"=>"Conditional slices, not joint fits or identification tests; asset sensitivity also changes grid spacing at fixed node count.",
        "grid"=>30,"simN"=>2000,"seed"=>1234,"fixed"=>Dict("R_1"=>PARENT_DEFAULTS.R_1,"sigma_4_1"=>PARENT_DEFAULTS.sigma_4_1,"mu_1"=>PARENT_DEFAULTS.mu_1),
        "baseline"=>Dict(string(k)=>v for (k,v) in BASE),
        "probes"=>[Dict("label"=>j.label,"changes"=>Dict(string(k)=>v for (k,v) in j.changes)) for j in jobs]))
end
open(joinpath(@__DIR__,"probes.csv"),"w") do io
    println(io,"case,Q,delta_Q,invalid,seconds,",join(SMM_MOMENTS,','),",asset_hh_above,hc_hh_above,error")
    function save(row)
        d=row.diagnostics
        println(io,join((row.label,row.Q,row.Q-base.Q,row.invalid,row.seconds,
            (getproperty(row.moments,Symbol(k)) for k in SMM_MOMENTS)...,
            d.a_hh_ever_above,d.hc_hh_ever_above,""),','));flush(io)
    end
    save(base)
    for job in jobs
        try
            save(evaluate(job.label,job.changes))
        catch err
            println("FAILED ",job.label," ",sprint(showerror,err));flush(stdout)
            rethrow()
        end
    end
    println("Solving asset-grid sensitivity at a_max=300 with 30 nodes");flush(stdout)
    alt=evaluate("asset_max=300",Dict{Symbol,Float64}();amax=300.0,value=child_value(300.0))
    save(alt)
end
println("DONE")
