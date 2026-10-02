# dispersion_check.jl -- memo 18 section 4 risk (Ali, 2026-10-02, item 2): can persistence near 1 restore skill
# dispersion? At the recovery test's theta0 (tools/test_param_recovery.jl), test grids, fixture composition and
# placeholder wage (tools/smm_test_fixtures.jl), persistence s_3t = exp(sigma_3_0 + sigma_3_1 t) is set to
# sigma_3_1 = 0 and sigma_3_0 in {-0.05, -0.02, -0.01}; everything else at theta0. A few evaluations, no search.
# Run from the repository root:  julia --project=. --threads=1 output/diagnostics/2026-10-02_dispersion/dispersion_check.jl
using Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames, Statistics, Dates
using ProgressMeter, Distributions, StatsBase, QuantEcon, FastGaussQuadrature, Parameters, Dierckx, TOML
const REPO = pwd(); const SRC = joinpath(REPO, "code", "src")
include(joinpath(SRC, "paths.jl")); include(joinpath(SRC, "manifest.jl")); include(joinpath(SRC, "diagnostics.jl"))
include(joinpath(SRC, "child_lifecycle.jl")); include(joinpath(SRC, "parent_family.jl")); include(joinpath(SRC, "tiktak.jl"))
include(joinpath(REPO, "code", "smm", "moments.jl")); include(joinpath(REPO, "tools", "smm_test_fixtures.jl"))
# theta0 copied from tools/test_param_recovery.jl (THETA0)
const THETA0 = Dict{Symbol,Float64}(
    :phi_2 => 0.196278, :phi_3 => 0.100000, :lambda_2 => 1.577178,
    :sigma_1_0 => -0.630853, :sigma_1_1 => -0.115329, :sigma_2_0 => -7.154000, :sigma_2_1 => 0.072000,
    :sigma_3_0 => -0.244781, :sigma_3_1 => 0.005000, :sigma_4_0 => -6.598000, :sigma_4_1 => 0.271000,
    :d_0 => 4.109547, :d_1 => 4.532005, :d_2 => 2.000000, :d_3 => 2.878411,
    :kappa_0 => 0.413682, :kappa_theta => -0.182563, :kappa_ParEd => -0.108108,
    :kappa_terminal => 8.786782, :sigma_eps => 1.142235)
const GRID = (Na = 20, Nk = 2, Nhc = 20, simN = 1000, seed = 1234, child_grid = (Na = 20, Nk = 20, Nt = 5))
T0 = load_targets(joinpath(REPO, "output/smm_runs/2026-10-01_225333_185835_targets/targets.toml"); require_composition = false)
TF = with_composition(T0, fixture_composition(T0))
se = Dict(k => 1 / sqrt(w) for (k, w) in zip(SMM_MOMENTS, moment_weights(TF)))
# LEVEL-MATCHED variants (added after the first pass): with persistence near 1 and theta0's TFP, mean ln k reaches
# 10.7-11.4 at 17, beyond the skill grid's top (HC_LN_MAX = 10), where the policies stop following skill -- the late
# collapse there is a grid artifact. Here TFP (d_0 and d_1 multiplied by one factor c) is bisected so that mean ln k at
# 17 equals theta0's, keeping the children inside the grid. Persistence as asked: sigma_3_1 = 0, sigma_3_0 as given.
const LEVEL_ONLY = get(ENV, "LEVEL_MATCH", "0") == "1"
function mean17(th)
    z = [to_search(th[q.name], q) for q in SMM_PARAMS]
    r = run_pipeline(unpack(z), TF; Na = GRID.Na, Nk = GRID.Nk, Nhc = GRID.Nhc, simN = GRID.simN, seed = GRID.seed,
                     child_grid = GRID.child_grid, demo_sim = false, child_wage = PLACEHOLDER_CHILD_WAGE)
    mean(filter(isfinite, log.(r.parent.sim_hc[:, 17])))
end
cases = [("theta0", THETA0)]
const TARGET17 = LEVEL_ONLY ? mean17(THETA0) : NaN
for a30 in (-0.05, -0.02, -0.01)
    th = merge(THETA0, Dict(:sigma_3_0 => a30, :sigma_3_1 => 0.0))
    lab = @sprintf("s3=%.3f", exp(a30))
    if LEVEL_ONLY
        lo, hi = 0.05, 1.0                      # c: TFP multiplier; mean ln k at 17 rises with c
        for _ in 1:9
            c = sqrt(lo * hi)
            mean17(merge(th, Dict(:d_0 => c * THETA0[:d_0], :d_1 => c * THETA0[:d_1]))) > TARGET17 ? (hi = c) : (lo = c)
        end
        c = sqrt(lo * hi); th = merge(th, Dict(:d_0 => c * THETA0[:d_0], :d_1 => c * THETA0[:d_1]))
        @printf("%s: TFP factor c = %.4f (d_0 %.4f, d_1 %.4f)\n", lab, c, th[:d_0], th[:d_1])
        lab *= @sprintf(" R*%.3f", c)
    end
    push!(cases, (lab, th))
end
show_m = [k for k in SMM_MOMENTS if startswith(k, "S3_sd_LW") || startswith(k, "S5_corr") || k in ("kth_lw17_gap", "m_eps", "k0_complete")]
res = Dict{String,Any}()
for (lab, th) in cases
    z = [to_search(th[q.name], q) for q in SMM_PARAMS]
    kw = unpack(z)
    feas = smm_feasible(kw)
    t0 = time()
    r = run_pipeline(kw, TF; Na = GRID.Na, Nk = GRID.Nk, Nhc = GRID.Nhc, simN = GRID.simN, seed = GRID.seed,
                     child_grid = GRID.child_grid, demo_sim = false, child_wage = PLACEHOLDER_CHILD_WAGE)
    m = model_moments(r, TF)
    q = smm_objective(z, TF; Na = GRID.Na, Nk = GRID.Nk, Nhc = GRID.Nhc, simN = GRID.simN, seed = GRID.seed,
                      child_grid = GRID.child_grid, demo_sim = false, child_wage = PLACEHOLDER_CHILD_WAGE)
    lnk = log.(r.parent.sim_hc)
    sd_age = [std(filter(isfinite, lnk[:, t])) for t in 1:size(lnk, 2)]
    mean_age = [mean(filter(isfinite, lnk[:, t])) for t in 1:size(lnk, 2)]
    res[lab] = (feas = feas, q = q, sd = sd_age, mean = mean_age, m = Dict(k => getfield(m, Symbol(k)) for k in show_m),
                nbad = m.n_nonfinite, viol = simulation_violations(r.parent).total, secs = time() - t0)
    @printf("%-20s feasible %s  Q %.1f  nonfinite %d  violations %d  (%.0f s)\n", lab, feas, q, m.n_nonfinite, res[lab].viol, res[lab].secs)
end
labs = first.(cases)
println("\nSD of ln k by child age (column t = age t; column 18 = the handoff)   [DFVW latent SD at 17: $(round(SD_LNK17, digits = 3))]")
@printf("%-5s", "age"); for l in labs; @printf(" %20s", l); end; println()
for t in 1:length(res["theta0"].sd)
    @printf("%-5d", t); for l in labs; @printf(" %20.3f", res[l].sd[t]); end; println()
end
println("\nmean of ln k by age 1, 9, 17:")
for l in labs; @printf("  %-20s %7.3f %7.3f %7.3f\n", l, res[l].mean[1], res[l].mean[9], res[l].mean[17]); end
println("\nMoments: model (t-stat against the data mean, data SE)")
@printf("%-28s %10s %8s", "moment", "data", "se"); for l in labs; @printf(" %20s", l); end; println()
for k in show_m
    @printf("%-28s %10.4f %8.4f", k, TF[k].mean, se[k])
    for l in labs; v = res[l].m[k]; @printf(" %11.4f (%+6.1f)", v, (v - TF[k].mean) / se[k]); end; println()
end
