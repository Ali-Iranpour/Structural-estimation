#!/usr/bin/env julia
# =============================================================================
# test_param_recovery.jl -- does the memo-19 SMM recover the parameters that generated its
# moments? Run BEFORE any estimation on data (task 2026-10-01).
#
#     julia --project=. --threads=1 tools/test_param_recovery.jl TARGETS.toml tech  OUTDIR
#     julia --project=. --threads=1 tools/test_param_recovery.jl TARGETS.toml all   OUTDIR
#
#   tech   the 12 memo-18 technology parameters free, the other 8 held at the truth
#   all    all 20 estimated parameters free
#
# THE TEST. Fix a parameter point theta0 ("the truth"), simulate the 67 moments there, and
# use them as the data, with the real data's standard errors as weights. Start the
# production objective (`smm_objective`, same unpacking, clamping, penalties) from a point
# displaced from theta0 by 5% of each box in every free direction at once, and minimise.
#
# WHY 5% AND NOT 15%. MEASURED 2026-10-02: a 15% displacement in the 12 technology
# directions is a degenerate economy -- own-study productivity 0.86 at 17, ln k FALLING
# after age 9, nobody in college -- whose college moments are undefined and penalised.
# Persistence compounds every elasticity over 16 years, so 5% already moves ln k at 17 from
# 6.74 to 6.17 (tech) or 8.25 (all) and the college share to 0.01 or 0.96. Reaching the
# basin from further out is the job of the Sobol stage of TikTak, not of this test, which
# asks whether the objective is locally invertible. A start that is still invalid is pulled
# halfway back toward theta0 until it is not. Pass = Q falls to ~0 and every
# parameter returns to theta0 within a small share of its box. With common random numbers
# Q(theta0) is EXACTLY zero, which the script checks first.
#
# WHAT IS SYNTHETIC, AND WHY THAT IS LEGITIMATE HERE. Two inputs are NOT PROVIDED yet: the
# data's age composition of the pooled S frames and sd(log AFQT) for the wage loading. The
# test uses `fixture_composition` and `PLACEHOLDER_CHILD_WAGE` (tools/smm_test_fixtures.jl).
# Recovery asks whether the estimator inverts the model at SOME fixed specification; it
# does not need those inputs to be right, only to be held fixed. Its result does not carry
# over automatically to the real composition and wage loading -- rerun it once they exist.
# Since 2026-10-02 the script uses each when it exists: the target file's [composition] (from
# the 19:32 targets on) and the anchored wage loading once sd(log AFQT) is set (0.20,
# provisional); the log and the result file name which inputs a run used.
#
# theta0 is a pilot point, NOT an estimate: the DFVW Table-7 starts, with TFP, persistence
# sigma_3_0, phi_3, lambda_2 and kappa_0 moved by 315 Sobplx steps toward the real data means
# (fixture composition, placeholder wage), then d_2 and phi_3 pulled off their box edges.
# At it the scaled Jacobian has rank 20/20, condition number 1.35e4; the weakest direction
# is sigma_eps with d_2.
#
# Grids are the TEST grids (parent 20/2/20, child 20/20/5, simN 1000), ~2.7 s per
# evaluation, not production. Single thread: NLopt is not thread-safe in this project.
# =============================================================================

using Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames, Statistics, Dates
using ProgressMeter, Distributions, StatsBase, QuantEcon, FastGaussQuadrature, Parameters, Dierckx, TOML

const REPO = normpath(joinpath(@__DIR__, ".."))
const SRC  = joinpath(REPO, "code", "src")
include(joinpath(SRC, "paths.jl")); include(joinpath(SRC, "manifest.jl"))
include(joinpath(SRC, "diagnostics.jl"))
include(joinpath(SRC, "child_lifecycle.jl")); include(joinpath(SRC, "parent_family.jl"))
include(joinpath(SRC, "tiktak.jl"))
include(joinpath(REPO, "code", "smm", "moments.jl"))
include(joinpath(REPO, "tools", "smm_test_fixtures.jl"))

length(ARGS) == 3 && ARGS[2] in ("tech", "all") ||
    error("usage: test_param_recovery.jl TARGETS.toml (tech|all) OUTDIR")
const TARGET_PATH, MODE, OUTDIR = abspath(ARGS[1]), ARGS[2], abspath(ARGS[3])
# the code's own wage loading once sd(log AFQT) is set (provisional 0.20 since 2026-10-02), the placeholder
# otherwise; and the commit read at START, the code this process loaded (it used to be read at the end)
const CHILD_WAGE, CHILD_WAGE_LABEL = isfinite(CHILD_DEFAULTS.sd_log_afqt) ?
    (child_wage_config(), "ANCHORED (sd(log AFQT) = $(CHILD_DEFAULTS.sd_log_afqt), provisional; lnw0 = $(LNW0_ANCHORED))") :
    (PLACEHOLDER_CHILD_WAGE, "PLACEHOLDER (alpha_theta = 0.2/sd_lnk17)")
const GIT_AT_START = git_sha()
mkpath(OUTDIR)
const LOG = open(joinpath(OUTDIR, "recovery_$(MODE).log"), "w")
logln(s...) = (println(stdout, s...); println(LOG, s...); flush(LOG); flush(stdout))

const GRID = (Na = 20, Nk = 2, Nhc = 20, simN = 1000, seed = 1234,
              child_grid = (Na = 20, Nk = 20, Nt = 5))

const THETA0 = Dict{Symbol,Float64}(
    :phi_2 => 0.196278, :phi_3 => 0.100000, :lambda_2 => 1.577178,
    :sigma_1_0 => -0.630853, :sigma_1_1 => -0.115329, :sigma_2_0 => -7.154000, :sigma_2_1 => 0.072000,
    :sigma_3_0 => -0.244781, :sigma_3_1 => 0.005000, :sigma_4_0 => -6.598000, :sigma_4_1 => 0.271000,
    :d_0 => 4.109547, :d_1 => 4.532005, :d_2 => 2.000000, :d_3 => 2.878411,
    :kappa_0 => 0.413682, :kappa_theta => -0.182563, :kappa_ParEd => -0.108108,
    :kappa_terminal => 8.786782, :sigma_eps => 1.142235)
const TECH = [:sigma_1_0, :sigma_1_1, :sigma_2_0, :sigma_2_1, :sigma_3_0, :sigma_3_1, :sigma_4_0, :sigma_4_1, :d_0, :d_1, :d_2, :d_3]

T0 = load_targets(TARGET_PATH; require_composition = false)
# the data's composition when the target file carries it (from 2026-10-02 19:32 on), the fixture otherwise
const COMP_LABEL = T0["_spec"].composition === nothing ? "FIXTURE (tools/smm_test_fixtures.jl)" : "DATA (target file)"
TF = T0["_spec"].composition === nothing ? with_composition(T0, fixture_composition(T0)) : T0
names_ = [q.name for q in SMM_PARAMS]
z_true = [to_search(THETA0[q.name], q) for q in SMM_PARAMS]
lb, ub = search_bounds()
free = MODE == "tech" ? [findfirst(==(n), names_) for n in TECH] : collect(1:length(names_))

obj_full(z) = smm_objective(z, TP; Na = GRID.Na, Nk = GRID.Nk, Nhc = GRID.Nhc, simN = GRID.simN,
                            seed = GRID.seed, child_grid = GRID.child_grid, demo_sim = false,
                            child_wage = CHILD_WAGE)

# ---- the pseudo-data: the model's own moments at theta0 -----------------------
logln("recovery test ($MODE): $(length(free)) free parameters, started ", now())
logln("targets: $TARGET_PATH  (composition: $COMP_LABEL; wage loading: $CHILD_WAGE_LABEL; code $GIT_AT_START)")
r0 = run_pipeline(unpack(z_true), TF; Na = GRID.Na, Nk = GRID.Nk, Nhc = GRID.Nhc, simN = GRID.simN,
                  seed = GRID.seed, child_grid = GRID.child_grid, demo_sim = false,
                  child_wage = CHILD_WAGE)
m0 = model_moments(r0, TF)
m0.n_nonfinite == 0 && simulation_violations(r0.parent).total == 0 ||
    error("theta0 does not produce a valid simulation")
TP = copy(TF)
for k in SMM_MOMENTS
    e = TF[k]
    TP[k] = merge(e, (mean = getfield(m0, Symbol(k)),))
end
q_true = obj_full(z_true)
logln(@sprintf("Q at the truth = %.3e  (must be 0: common random numbers)", q_true))
q_true < 1e-10 || error("Q(theta0) = $q_true is not zero -- the objective is not deterministic")

# ---- the displaced start: +-5% of each box, fixed signs -----------------------
const DISPLACE = 0.05
rng = MersenneTwister(20261002)
z_start = copy(z_true)
for i in free
    s = rand(rng, Bool) ? 1.0 : -1.0
    z_start[i] = clamp(z_true[i] + s * DISPLACE * (ub[i] - lb[i]),
                       lb[i] + 0.01 * (ub[i] - lb[i]), ub[i] - 0.01 * (ub[i] - lb[i]))
end
# A displaced point can be inadmissible (an elasticity >= 1) or invalid (a college group
# empty, so a moment undefined); pull it halfway back toward theta0 until it is neither.
for _ in 1:10
    smm_feasible(unpack(z_start)) && obj_full(z_start) < SMM_PENALTY && break
    z_start[free] .= 0.5 .* (z_start[free] .+ z_true[free])
end
smm_feasible(unpack(z_start)) && obj_full(z_start) < SMM_PENALTY ||
    error("could not find a valid displaced start")

n_eval = Ref(0); best = Ref(Inf); t_start = time()
function f_free(x::Vector, g::Vector)
    z = copy(z_true); z[free] .= x
    q = obj_full(z)
    n_eval[] += 1
    q < best[] && (best[] = q)
    n_eval[] % 50 == 0 && logln(@sprintf("  eval %5d  best Q %.6g  (%.1f min)", n_eval[], best[], (time() - t_start) / 60))
    return q
end
q_start = f_free(z_start[free], Float64[])
logln(@sprintf("Q at the displaced start = %.6g", q_start))

function run_stage(alg, x0, maxev)
    opt = Opt(alg, length(free))
    lower_bounds!(opt, lb[free]); upper_bounds!(opt, ub[free])
    min_objective!(opt, f_free); maxeval!(opt, maxev)
    ftol_abs!(opt, 1e-10); xtol_rel!(opt, 1e-7)
    alg === :LN_BOBYQA && initial_step!(opt, 0.05 .* (ub[free] .- lb[free]))
    q, x, ret = NLopt.optimize(opt, x0)
    logln(@sprintf("stage %s: %s, Q %.6g after %d evaluations in total", alg, ret, q, n_eval[]))
    return q, x
end
# BOBYQA first (model-based, fast on smooth stretches), then Nelder-Mead -- the production
# local method -- from where it stopped, which is robust to the objective's small steps.
q1, x1 = run_stage(:LN_BOBYQA, z_start[free], MODE == "tech" ? 1500 : 2500)
q2, x2 = run_stage(:LN_NELDERMEAD, x1, MODE == "tech" ? 1000 : 1500)
x_hat, q_hat = q2 <= q1 ? (x2, q2) : (x1, q1)

# ---- report ---------------------------------------------------------------------
z_hat = copy(z_true); z_hat[free] .= x_hat
logln("\nparameter        truth      start   estimate   |error| % of box")
worst = 0.0
rows = String[]
for i in free
    q = SMM_PARAMS[i]
    err = 100 * abs(z_hat[i] - z_true[i]) / (ub[i] - lb[i])
    global worst = max(worst, err)
    line = @sprintf("%-14s %9.4f  %9.4f  %9.4f   %6.2f", q.name, from_search(z_true[i], q),
                    from_search(z_start[i], q), from_search(z_hat[i], q), err)
    logln(line); push!(rows, line)
end
pass = q_hat < 1e-2 && worst < 2.0
logln(@sprintf("\nQ: start %.6g -> end %.6g after %d evaluations, %.1f min. Worst parameter error %.2f%% of its box.",
               q_start, q_hat, n_eval[], (time() - t_start) / 60, worst))
logln(pass ? "RESULT: RECOVERED (Q < 1e-2 and every parameter within 2% of its box)" :
             "RESULT: NOT RECOVERED by that criterion -- read the table above")
open(joinpath(OUTDIR, "recovery_$(MODE).toml"), "w") do io
    TOML.print(io, Dict("mode" => MODE, "targets" => TARGET_PATH, "git_commit" => GIT_AT_START,
                        "composition" => COMP_LABEL,
                        "child_wage" => CHILD_WAGE_LABEL,
                        "grid" => "parent 20/2/20, child 20/20/5, simN 1000, seed 1234",
                        "q_start" => q_start, "q_end" => q_hat, "evaluations" => n_eval[],
                        "worst_error_pct_of_box" => worst, "recovered" => pass,
                        "truth" => Dict(String(SMM_PARAMS[i].name) => from_search(z_true[i], SMM_PARAMS[i]) for i in free),
                        "start" => Dict(String(SMM_PARAMS[i].name) => from_search(z_start[i], SMM_PARAMS[i]) for i in free),
                        "estimate" => Dict(String(SMM_PARAMS[i].name) => from_search(z_hat[i], SMM_PARAMS[i]) for i in free),
                        "penalties" => Dict(String(k) => v for (k, v) in SMM_PENALTY_LOG)))
end
close(LOG)
