# From repository root: julia --threads=1 --project=. output/smm_diagnostics/2026-09-07_school_time/probe.jl
# Conditional full-grid slices; other eight parameters fixed at the preserved fit.
include(joinpath(@__DIR__,"../../../tools/test_smm_baseline.jl"))
const CURRENT=load_targets(joinpath(REPO,"output/smm_runs/2026-09-07_114138/targets.toml"))
function score_current(model, intercept)
    mm=model_moments(model)
    qq=sum(((getproperty(mm,Symbol(k))-CURRENT[k].mean)/moment_scale(k,CURRENT[k].mean))^2 for k in SMM_MOMENTS)
    println("sigma_4_0=",intercept," Q_current=",qq," i_early=",mm.mean_i_c_early," i_late=",mm.mean_i_c_late," invalid=",simulation_violations(model).total);flush(stdout)
end
score_current(p, PARENT_DEFAULTS.sigma_4_0)
for intercept in (-6.0,-3.0,-2.0,-1.0)
    candidate=Parent_child_interaction_age_specific_AR1(;Na=30,Nk=2,Nhc=30,simN=2000,seed=1234,sigma_4_0=intercept)
    candidate.V_child_interp=V
    redirect_stdout(devnull) do
        solve_model!(candidate;verbose=false);simulate_model!(candidate)
    end
    score_current(candidate, intercept)
end
