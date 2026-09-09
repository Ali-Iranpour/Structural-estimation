# Reproduce the baseline policy/terminal diagnostics cited in transfer_CRRA_wage.ipynb.
# julia --threads=1 --project=. tools/check_transfer_baseline.jl
const REPO = normpath(joinpath(@__DIR__, ".."))
using Printf, Random, NLopt, LinearAlgebra, Interpolations, Statistics, ProgressMeter,
      Distributions, QuantEcon, FastGaussQuadrature, Parameters, Dierckx, Serialization
include(joinpath(REPO, "code/src/child_lifecycle.jl"))
include(joinpath(REPO, "code/src/parent_family.jl"))
child = ConSavLaborCollege_AR1(Na=30, Nk=30, Nt=5, sigma_eps=0.5, rho=1.5,
    psi_terminal=0.0, kappa_terminal=5.0, omega=0.3, a_max=100.0, w=20.0)
solve_model_work!(child); solve_model_college!(child)
optimal_transfer_college!(child); optimal_transfer_work!(child)
p = Parent_child_interaction_age_specific_AR1(Na=30, Nk=2, Nhc=30)
p.V_child_interp = terminal_value_spline(child; s=10.0)
solve_model!(p); simulate_model!(p)
println("age,c,e,h,t,i,income,wage,hc")
for t in 1:17
    println(join([t, [mean(getfield(p,f)[:,t]) for f in (:sim_c,:sim_e,:sim_h,:sim_t,:sim_i,:sim_income,:sim_wage,:sim_hc)]...], ","))
end
println("DONE")

include(joinpath(REPO,"code/smm/moments.jl"))
println("violations: ",simulation_violations(p))
println("selected state: ", (p.k_grid[1],p.hc_grid[15],p.p_grid[1]))
for t in (6,16,17)
    println("slice age ",t)
    for f in (:sol_c,:sol_e,:sol_h,:sol_t,:sol_i)
        println(f," ",getfield(p,f)[t,:,1,15,1])
    end
end
for t in (16,17)
    E = t == 16 ? expected_interp(p,create_interp(p,p.sol_v,17)) : nothing
    vals = Float64[]
    for ia in 1:30, ih in 1:30, ip in 1:p.Np
        a,hc,k,shock=p.a_grid[ia],p.hc_grid[ih],0.0,p.p_grid[ip]
        c,i,e,h,tp=[getfield(p,f)[t,ia,1,ih,ip] for f in (:sol_c,:sol_i,:sol_e,:sol_h,:sol_t)]
        ap=(1+p.r)*a+p.tax_lambda*(wage_func(p,k,t,shock)*h)^(1-p.tau)+p.y-c-e
        hn=HC_technology_full(p,tp,e,hc,i,t)
        dh=t==17 ? eval_child_value(p.V_child_interp,ap,hn,k,true)[3] : value_and_gradient(E[ip],ap,k,hn)[4]
        push!(vals,hn*dh)
    end
    println("t=",t," continuation marginal wrt log HC: median ",median(vals)," p10,p90 ",quantile(vals,[0.1,0.9]))
end

println("ia,a,a_next,HC_next,dV_da,HC_next*dV_dHC,e")
for ia in [1,5,9,10,15,29,30]
    t=17; k=0.0; hc=p.hc_grid[15]; shock=p.p_grid[1]; a=p.a_grid[ia]
    c,i,e,h,tp=[getfield(p,f)[t,ia,1,15,1] for f in (:sol_c,:sol_i,:sol_e,:sol_h,:sol_t)]
    ap=(1+p.r)*a+p.tax_lambda*(wage_func(p,k,t,shock)*h)^(1-p.tau)+p.y-c-e
    hn=HC_technology_full(p,tp,e,hc,i,t)
    v,da,dh=eval_child_value(p.V_child_interp,ap,hn,k,true)
    println(join([ia,a,ap,hn,da,hn*dh,e],","))
end
println("Reoptimization: ia,old_e,new_e,objective_gain,return_code")
for ia in (1,5,9,10,15,29,30)
    t=17; k=0.0; hc=p.hc_grid[15]; shock=p.p_grid[1]; a=p.a_grid[ia]
    oldx=[getfield(p,f)[t,ia,1,15,1] for f in (:sol_c,:sol_i,:sol_e,:sol_h,:sol_t)]
    f=(x,g)->obj_last_period_full(p,x[1],x[2],x[3],x[4],x[5],a,hc,k,t,shock,p.V_child_interp,g)
    opt=Opt(:LD_SLSQP,5); b=budget_ceiling(p,a,k,t,shock)
    lower_bounds!(opt,[1e-4,TIME_FLOOR,1e-4,TIME_FLOOR,TIME_FLOOR]); upper_bounds!(opt,[b,1.,b,1.,1.])
    max_objective!(opt,f)
    inequality_constraint!(opt,constraint_min_leisure_full,TOL_CONSTR)
    inequality_constraint!(opt,constraint_child_time,TOL_CONSTR)
    inequality_constraint!(opt,(x,g)->asset_constraint_full(x,g,p,k,t,a,shock),TOL_CONSTR)
    inequality_constraint!(opt,(x,g)->asset_constraint_max(x,g,p,k,t,a,shock),TOL_CONSTR)
    xtol_rel!(opt,1e-8); maxeval!(opt,5000)
    val,x,ret=optimize(opt,oldx)
    println(join([ia,oldx[3],x[3],val-f(oldx,Float64[]),ret],","))
end
