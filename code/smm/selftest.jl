#!/usr/bin/env julia
# =============================================================================
# selftest.jl -- prove the guards fire, by breaking things on purpose.
#
#     cd code/smm && julia +1.11 --project=../.. selftest.jl
#
# Every check here INJECTS the failure it is testing. A guard that has never been
# shown to fire is a comment, not a check: the A1 human-capital gap, the A4
# programming-error gap and the A2 acceptance flaw were all live in code that
# looked correct and had reassuring comments above it.
#
# What each check establishes:
#
#   A1  a non-finite or non-positive human capital at the age-18 handoff (column
#       T+1) is REFUSED, not scored. That column becomes the child's initial k.
#   A4  an expected model failure is scored as a penalty; an unexpected coding
#       error (MethodError, UndefVarError, a bare error("typo")) is RE-THROWN and
#       is visible.
#   A2  acceptance follows the RETAINED WINNER's return code, not the population
#       of restarts.
#   A3  a resume across changed bounds, a changed grid, a changed target file or
#       a finished ("refined") checkpoint is REFUSED rather than silently mixed.
#
# It runs in about a minute on small grids and needs no worker processes.
# =============================================================================

using Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames, Logging
using Statistics, Dates, ProgressMeter, Distributions, StatsBase
using QuantEcon, FastGaussQuadrature, Parameters, Dierckx, TOML

const REPO_ = normpath(joinpath(@__DIR__, "..", ".."))
const SRC   = joinpath(REPO_, "code", "src")
include(joinpath(SRC, "paths.jl"));       include(joinpath(SRC, "manifest.jl"))
include(joinpath(SRC, "diagnostics.jl")); include(joinpath(SRC, "child_lifecycle.jl"))
include(joinpath(SRC, "parent_family.jl")); include(joinpath(SRC, "tiktak.jl"))
include(joinpath(REPO_, "code", "smm", "moments.jl"))

const PASS = Ref(true)
function check(label, ok::Bool, detail = "")
    PASS[] &= ok
    @printf("  %-58s %s%s\n", label, ok ? "PASS" : "FAIL",
            isempty(detail) ? "" : "   " * detail)
    return ok
end
banner(s) = (println(); println("="^78); println(s); println("="^78))

# -----------------------------------------------------------------------------
banner("Setting up a small solved baseline")
const TARGETS = load_targets(smm_targets_file())
ch = ConSavLaborCollege_AR1(; Na = 12, Nk = 12, Nt = 3, rho = 1.5, psi_terminal = 0.0,
                              kappa_terminal = 5.0, omega = 0.3, a_max = 100.0, w = 20.0,
                              simN = 200, seed = 1234)
redirect_stdout(devnull) do; redirect_stderr(devnull) do
    solve_model_work!(ch); solve_model_college!(ch)
    optimal_transfer_work!(ch); optimal_transfer_college!(ch)
end end
const V_CHILD = terminal_value_spline(ch; s = 10.0)

function solved_parent(; Na = 10, Nhc = 10, simN = 200)
    p = Parent_child_interaction_age_specific_AR1(; Na = Na, Nk = 2, Nhc = Nhc,
                                                    simN = simN, seed = 1234, school_time = target_school_time(TARGETS))
    p.V_child_interp = V_CHILD
    redirect_stdout(devnull) do
        solve_model!(p; verbose = false); simulate_model!(p)
    end
    return p
end
p0 = solved_parent()
@printf("  baseline solved: %d households, T = %d, violations = %d\n",
        size(p0.sim_c, 1), p0.T, simulation_violations(p0).total)

# -----------------------------------------------------------------------------
banner("A1 -- terminal human capital at the age-18 handoff (column T+1)")
# The handoff column is the one that becomes the child's sim_k_init. Before this fix,
# simulation_violations looked at columns 1..T only, so every one of these was accepted.
check("clean baseline has zero violations", simulation_violations(p0).total == 0)

for (label, val, field) in (("HC = NaN  at the handoff",  NaN,  :nonfinite),
                            ("HC = 0.0  at the handoff",  0.0,  :hc_nonpositive),
                            ("HC = -1.0 at the handoff", -1.0,  :hc_nonpositive),
                            ("HC = Inf  at the handoff",  Inf,  :nonfinite))
    p = solved_parent()
    p.sim_hc[1, p.T + 1] = val
    v = simulation_violations(p)
    check(label * " is refused", v.total > 0 && getfield(v, field) > 0,
          @sprintf("total=%d %s=%d", v.total, field, getfield(v, field)))
end

# and it must still be caught INSIDE the objective, not merely by the checker
let p = solved_parent()
    p.sim_hc[1, p.T + 1] = -1.0
    v = simulation_violations(p)
    check("the objective's own gate would reject that draw", v.total > 0)
end

# a mid-stage HC failure must still be caught (the old behaviour must not regress)
let p = solved_parent()
    p.sim_hc[3, 7] = -2.0
    check("mid-stage HC <= 0 is still refused", simulation_violations(p).hc_nonpositive > 0)
end

# -----------------------------------------------------------------------------
banner("A4 -- expected model failures are scored, coding errors are re-thrown")
check("solver convergence refusal is a model failure",
      is_model_failure(ErrorException(
          "Period 5: only 80.0% of 100 grid points converged (floor 95.0%). " *
          "maxeval=3, other=0 Dict(). Refusing to return a solution built on failed optimizations.")))
check("DomainError from the solver's own code is a model failure (v2 2026-09-20 follow-up B; ported 2026-09-27)",
      is_model_failure(DomainError(-1.0, "log"), "parent_family.jl:util_parent:805") &&
      is_model_failure(DomainError(-1.0, "log"), "child_lifecycle.jl:util_work:738"))
check("DomainError from anywhere else, or of unknown origin, is NOT a model failure",
      !is_model_failure(DomainError(-1.0, "log")) && !is_model_failure(DomainError(-1.0, "log"), "moments.jl:tas_moments:800") &&
      !is_model_failure(DomainError(-1.0, "log"), "unknown"))
check("the SLSQP-callback assertion (box bounds violated) is a model failure",
      is_model_failure(AssertionError("HC_technology_full: box bounds violated (t_p=NaN, e_p=NaN)")))
check("any OTHER AssertionError is a programming error (2026-09-20 audit)",
      !is_model_failure(AssertionError("t_p > 0")) && !is_model_failure(AssertionError("simT exceeds T")))
check("InexactError follows the same origin rule",
      is_model_failure(InexactError(:Int, Int, NaN), "parent_family.jl:solve_model!:1500") &&
      !is_model_failure(InexactError(:Int, Int, NaN)))

check("MethodError is NOT a model failure",    !is_model_failure(MethodError(+, (1, "a"))))
check("UndefVarError is NOT a model failure",  !is_model_failure(UndefVarError(:typo)))
check("BoundsError is NOT a model failure",    !is_model_failure(BoundsError([1], 5)))
check("a bare error(\"typo\") is NOT a model failure",
      !is_model_failure(ErrorException("typo")))
check("a DIFFERENT error() message is NOT a model failure",
      !is_model_failure(ErrorException("something else went wrong")))
check("wrapped causes are unwrapped",
      is_model_failure(_root_cause(CapturedException(AssertionError("util_total: box bounds violated (c=NaN, i_c=NaN)"), backtrace()))))
check("the sigma restriction is named by the share that fails",
      smm_infeasible_which((sigma_1_0 = 0.5, sigma_1_1 = 0.0)) === :sigma_1 &&
      smm_infeasible_which((sigma_2_0 = 0.5, sigma_2_1 = 0.0)) === :sigma_2 &&
      smm_infeasible_which((sigma_1_0 = -0.6, sigma_1_1 = -0.1, sigma_2_0 = -3.5, sigma_2_1 = -0.1)) === :none)

# tiktak must STOP on a coding error rather than discarding the restart
let thrown = Ref(false)
    boom(x) = (thrown[] = true; throw(MethodError(+, (1, "a"))))
    ok = try
        tiktak(boom, [0.0], [1.0]; N = 4, Nstar = 1)
        false                      # reaching here means it swallowed the error
    catch e
        _root_cause(e) isa MethodError
    end
    check("tiktak RE-THROWS a coding error (on_error = :rethrow)", ok)
end
# `:discard` is SUPPOSED to warn -- that is the behaviour being tested. But it warns with
# `exception =`, so Julia prints a full stack trace per discarded search, and this one
# check was emitting ~100 lines of alarming-looking output for a test that PASSES. A
# self-test whose passing output looks like a crash teaches people to stop reading it, so
# the logger is silenced for the duration and the COUNT is checked instead.
let boom2(x) = sum(x) < 0.5 ? throw(MethodError(+, (1, "a"))) : sum(x .^ 2)
    r = Logging.with_logger(Logging.NullLogger()) do
        tiktak(boom2, [0.0], [1.0]; N = 8, Nstar = 2, on_error = :discard,
               local_maxeval = 20, polish_maxeval = 20)
    end
    check("on_error = :discard discards instead of throwing", r.f isa Float64)
    check("  ... and counts what it discarded", r.n_exception >= 1,
          "n_exception = $(r.n_exception)")
end

# -----------------------------------------------------------------------------
banner("A2 -- acceptance follows the REPORTED POINT's own evidence")
# Sphere: every search converges, and the winner is the polish.
let r = tiktak(x -> sum(x .^ 2), fill(-5.0, 3), fill(5.0, 3); N = 60, Nstar = 4)
    check("winner_stage is recorded", r.winner_stage in (:sobol, :supplied, :local, :polish),
          "got :$(r.winner_stage)")
    check("winner_ret is a real return code", r.winner_ret !== :SOBOL_ONLY || r.f == r.f_sobol_best,
          "$(r.winner_ret)")
    check("the returned point has convergence evidence (TikTak.local_converged)",
          TikTak.local_converged(r), "$(r.winner_ret), verification $(r.incumbent.verification.status)")
end
# A budget so small that every local search stops on maxeval. The population contains no
# converged search, so acceptance must be false however good the objective looks.
let r = tiktak(x -> sum(abs.(x) .^ 1.5), fill(-5.0, 6), fill(5.0, 6);
               N = 40, Nstar = 3, local_maxeval = 5, polish_maxeval = 5)
    tally = ret_tally(r)
    n_conv = sum(v for (k, v) in tally if ret_class(k) === :converged; init = 0)
    check("budget-starved run: winner did NOT converge",
          !TikTak.local_converged(r) || n_conv > 0,
          "winner_ret=$(r.winner_ret) converged_restarts=$n_conv")
    check("ret_class buckets MAXEVAL_REACHED as :limit",
          ret_class(:MAXEVAL_REACHED) === :limit)
    check("ret_class buckets FTOL_REACHED as :converged",
          ret_class(:FTOL_REACHED) === :converged)
end
# 2026-09-27 (finding 5): a supplied point that is already optimal is VERIFIED by the
# searches that return it, instead of being reported as never converged.
let r = tiktak(x -> sum(abs2, x), [-1.0, -1.0], [1.0, 1.0]; N = 8, Nstar = 2, extra_seeds = [[0.0, 0.0]])
    check("an unchanged supplied optimum is verified, not reported unconverged",
          r.f == 0.0 && r.winner_stage === :supplied && TikTak.local_converged(r),
          "origin $(r.winner_ret), verification $(r.incumbent.verification.status)")
end
# 2026-09-27 (finding 2): a refinement on another objective that IMPROVES but stops on its
# evaluation cap is budget-limited there, whatever the coarse search's certificate said.
let r = tiktak(x -> sum(abs2, x .- 0.1), fill(-1.0, 4), fill(1.0, 4); N = 20, Nstar = 2, objective_id = "coarse"),
    out = TikTak.refine(x -> sum(abs2, x .- 0.1) + 0.3 * sum(x), r.x, fill(-1.0, 4), fill(1.0, 4);
                        settings = TikTak.SolverSettings(:LN_BOBYQA, 1e-6, 1e-10, 1e-6, 12),
                        cfg = r.config, objective_id = "fine")
    acc(ev) = TikTak.acceptance(; execution_ok = true, candidate_valid = true, search_budget_complete = true,
                                  local_converged = TikTak.local_converged(ev, "fine"))
    check("the coarse search converged (the precondition of this check)", TikTak.local_converged(r))
    check("an improving refinement stopped on MAXEVAL is NOT accepted",
          out.status === :improved && out.ret === :MAXEVAL_REACHED && !acc(out.incumbent).accepted,
          "status $(out.status), ret $(out.ret)")
end

# -----------------------------------------------------------------------------
banner("A3 -- resume refuses an incompatible run (the TikTak checkpoint)")
# 2026-09-27: the checkpoint is the TikTak module's versioned tiktak_state.toml, and its identity
# fields come from THIS configuration (SMM_PARAMS, SMM_MOMENTS), never from a hard-coded list --
# the old block asserted one historical R_1 box and failed under any specification switch.
# The runner-level refusals (every identity field, the legacy formats) are tools/test_smm_resume.jl.
mktempdir() do dir
    lo_s, hi_s = search_bounds()
    fields = Dict{String,Any}("param_names" => [String(q.name) for q in SMM_PARAMS],
                              "param_lo" => [q.lo for q in SMM_PARAMS], "param_hi" => [q.hi for q in SMM_PARAMS],
                              "param_link" => [String(q.link) for q in SMM_PARAMS],
                              "moment_names" => collect(String, SMM_MOMENTS))
    oid = TikTak.fields_id(fields)
    quad(z) = sum(abs2, (z .- incumbent()) ./ (hi_s .- lo_s))          # a cheap stand-in objective
    path = joinpath(dir, "tiktak_state.toml")
    run1 = tiktak(quad, lo_s, hi_s; N = 30, Nstar = 3, local_maxeval = 20, skip_polish = true,
                  state_path = path, objective_id = oid, objective_fields = fields, stop_after_restarts = 1)
    d = TOML.parsefile(path)
    check("a checkpoint records the current parameter names, boxes and links",
          d["objective"]["fields"]["param_names"] == fields["param_names"] &&
          Float64.(d["objective"]["fields"]["param_lo"]) == fields["param_lo"] &&
          d["optimizer"]["fields"]["lo"] == lo_s)
    check("the checkpoint is versioned, checksummed and paused where it was asked to",
          d["schema_version"] == TikTak.STATE_SCHEMA && run1.status === :paused && d["progress"]["next_j"] == 2 &&
          occursin("# sha256 ", read(path, String)))
    base = (N = 30, Nstar = 3, local_maxeval = 20, skip_polish = true, state_path = path, resume = true,
            preflight_only = true)
    refused(kw) = try
        tiktak(quad, lo_s, hi_s; merge(base, kw)...)       # merge: a later field replaces, never repeats
        false
    catch e
        e isa TikTak.ResumeRefused
    end
    check("the same objective and optimizer resume", !refused((objective_id = oid, objective_fields = fields)))
    let f2 = deepcopy(fields)
        f2["param_hi"][1] += 1.0
        check("a changed parameter box is refused", refused((objective_id = TikTak.fields_id(f2), objective_fields = f2)))
    end
    let f3 = deepcopy(fields)
        f3["moment_names"] = f3["moment_names"][1:end-1]
        check("a changed moment set is refused", refused((objective_id = TikTak.fields_id(f3), objective_fields = f3)))
    end
    check("a changed restart count is refused (a new search, not a continuation)",
          refused((objective_id = oid, objective_fields = fields, Nstar = 4)))
    check("a changed local budget is refused", refused((objective_id = oid, objective_fields = fields, local_maxeval = 21)))
end

# -----------------------------------------------------------------------------
banner("Specification is frozen as instructed")
# 2026-09-10: was "ten and ten". Those two checks necessarily FAILED under the
# fourteen-parameter specification, so this self-test could not pass at all -- and its
# closing banner is "do not run the estimation", which would have been the standing advice.
# 2026-09-11: sixteen parameters (the two shock scales added), seventeen moments (the
# three ability tertiles replaced by the mean gap; the wealth gap and the age-17 SD added).
# 2026-09-27: fifteen parameters and sixteen moments -- kse_w_gap dropped as a wrong moment
# (~11-year timing gap) and sigma_eps, which it identified, fixed at 2.0.
check("fifteen estimated parameters", length(SMM_PARAMS) == 15, "$(length(SMM_PARAMS))")
check("sixteen targeted moments", length(SMM_MOMENTS) == 16, "$(length(SMM_MOMENTS))")
check("eleven parent + four child parameters",
      length(SMM_PARENT_PARAMS) == 11 && length(SMM_CHILD_PARAMS) == 4,
      "$(length(SMM_PARENT_PARAMS)) + $(length(SMM_CHILD_PARAMS))")
check("eleven parent + five TAS moments (mean_a_p_late in, kterm_x_strict_w99 out)",
      length(SMM_PARENT_MOMENTS) == 11 && length(SMM_TAS_MOMENTS) == 5 &&
      SMM_PARENT_MOMENTS[end] == "mean_a_p_late",
      "$(length(SMM_PARENT_MOMENTS)) + $(length(SMM_TAS_MOMENTS))")
check("the four child parameters are the kappas",
      Set(SMM_CHILD_PARAMS) == Set((:kappa_0, :kappa_theta, :kappa_ParEd, :kappa_terminal)))
check("sigma_eta is a parent parameter, box [0, 0.08] level, starts at the fitted baseline",
      (q = SMM_PARAMS[findfirst(x -> x.name === :sigma_eta, SMM_PARAMS)];
       q.owner === :parent && q.lo == 0.0 && q.hi == 0.08 && q.link === :level &&
       smm_start(:sigma_eta) == PARENT_DEFAULTS.sigma_eta && 0 < PARENT_DEFAULTS.sigma_eta < 0.08))
check("sigma_eps is NOT estimated and holds at 2.0 (fixed 2026-09-27; completed via CHILD_ESTIMATED)",
      !any(q -> q.name === :sigma_eps, SMM_PARAMS) && CHILD_DEFAULTS.sigma_eps == 2.0 &&
      :sigma_eps in CHILD_ESTIMATED)
check("2026-09-27 child settings: mu 0.8, omega 0.2, y 0.144, passed by child_config",
      CHILD_DEFAULTS.mu == 0.8 && CHILD_DEFAULTS.omega == 0.2 && CHILD_DEFAULTS.y == 0.144 &&
      CHILD_DEFAULTS.college_cost == 0.6 &&
      (cfg = child_config(TARGETS; Na = 30, Nk = 30, Nt = 5, simN = 10, seed = 1);
       cfg.mu == 0.8 && cfg.omega == 0.2 && cfg.y == 0.144 && cfg.college_cost == 0.6))
check("2026-09-27 parent y 0.1632 reaches the constructor default",
      PARENT_DEFAULTS.y == 0.1632 &&
      Parent_child_interaction_age_specific_AR1(; Na = 5, Nk = 2, Nhc = 5, simN = 10).y == 0.1632)
check("the 2026-09-12 boxes: kappa_0 [-3, 1], kappa_ParEd [-1, 0.5], sigma_4_1 [-0.05, 0.30]",
      (box(n) = (q = SMM_PARAMS[findfirst(x -> x.name === n, SMM_PARAMS)]; (q.lo, q.hi));
       box(:kappa_0) == (-3.0, 1.0) && box(:kappa_ParEd) == (-1.0, 0.5) && box(:sigma_4_1) == (-0.05, 0.30)))
check("every search start is inside its box",
      all(q -> q.lo <= smm_start(q.name) <= q.hi, SMM_PARAMS))
check("the baseline kappas and the target centring agree (CHILD_DEFAULTS.m_psychic)",
      isfinite(CHILD_DEFAULTS.m_psychic) && CHILD_DEFAULTS.m_psychic > 6.0)
check("R_1 is NOT estimated and holds at 0",
      !any(q -> q.name === :R_1, SMM_PARAMS) && PARENT_DEFAULTS.R_1 == 0.0)
check("the five TAS targets, in order (kse_w_gap and kterm_x_strict_w99 out since 2026-09-27)",
      collect(SMM_TAS_MOMENTS) == ["k0_complete", "kth_ga17_gap", "kpe_g0_c", "kpe_g1_c", "sd_ga17"])
check("moment order is parent block then TAS block",
      collect(SMM_MOMENTS) == vcat(collect(SMM_PARENT_MOMENTS), collect(SMM_TAS_MOMENTS)))
check("every estimated parameter routes to exactly one block",
      all(q -> (q.owner === :parent) == hasproperty(PARENT_DEFAULTS, q.name) &&
               (q.owner === :child)  == hasproperty(CHILD_DEFAULTS, q.name), SMM_PARAMS))
check("sigma_4_1 is estimated", any(q -> q.name === :sigma_4_1, SMM_PARAMS))
check("mu_1 is NOT estimated and holds at -0.04",
      !any(q -> q.name === :mu_1, SMM_PARAMS) && PARENT_DEFAULTS.mu_1 == -0.04)
let q = SMM_PARAMS[findfirst(x -> x.name === :R_0, SMM_PARAMS)]
    check("R_0 box is [0.5, 100.0], searched in logs",
          q.lo == 0.5 && q.hi == 100.0 && q.link === :log,
          "[$(q.lo), $(q.hi)] :$(q.link)")
end

banner(PASS[] ? "ALL CHECKS PASS" : "FAILURES ABOVE -- do not run the estimation")
exit(PASS[] ? 0 : 1)
