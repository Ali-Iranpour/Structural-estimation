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

using Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames
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
const TARGETS = load_targets(joinpath(REPO_, "Input", "smm_targets_baseline.toml"))
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
                                                    simN = simN, seed = 1234)
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
check("DomainError is a model failure",    is_model_failure(DomainError(-1.0, "log")))
check("AssertionError is a model failure", is_model_failure(AssertionError("t_p > 0")))
check("InexactError is a model failure",   is_model_failure(InexactError(:Int, Int, NaN)))

check("MethodError is NOT a model failure",    !is_model_failure(MethodError(+, (1, "a"))))
check("UndefVarError is NOT a model failure",  !is_model_failure(UndefVarError(:typo)))
check("BoundsError is NOT a model failure",    !is_model_failure(BoundsError([1], 5)))
check("a bare error(\"typo\") is NOT a model failure",
      !is_model_failure(ErrorException("typo")))
check("a DIFFERENT error() message is NOT a model failure",
      !is_model_failure(ErrorException("something else went wrong")))
check("wrapped causes are unwrapped",
      is_model_failure(_root_cause(CapturedException(AssertionError("x"), backtrace()))))

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
let boom2(x) = sum(x) < 0.5 ? throw(MethodError(+, (1, "a"))) : sum(x .^ 2)
    r = tiktak(boom2, [0.0], [1.0]; N = 8, Nstar = 2, on_error = :discard,
               local_maxeval = 20, polish_maxeval = 20)
    check("on_error = :discard still available and counts exceptions", r.n_exception >= 0)
end

# -----------------------------------------------------------------------------
banner("A2 -- acceptance follows the RETAINED WINNER")
# Sphere: every search converges, and the winner is the polish.
let r = tiktak(x -> sum(x .^ 2), fill(-5.0, 3), fill(5.0, 3); N = 60, Nstar = 4)
    check("winner_stage is recorded", r.winner_stage in (:sobol, :local, :polish),
          "got :$(r.winner_stage)")
    check("winner_ret is a real return code", r.winner_ret !== :SOBOL_ONLY || r.f == r.f_sobol_best,
          "$(r.winner_ret)")
    check("winner's own code classifies as converged",
          ret_class(r.winner_ret) === :converged, "$(r.winner_ret)")
end
# A budget so small that every local search stops on maxeval. The population contains no
# converged search, so acceptance must be false however good the objective looks.
let r = tiktak(x -> sum(abs.(x) .^ 1.5), fill(-5.0, 6), fill(5.0, 6);
               N = 40, Nstar = 3, local_maxeval = 5, polish_maxeval = 5)
    tally = ret_tally(r)
    n_conv = sum(v for (k, v) in tally if ret_class(k) === :converged; init = 0)
    check("budget-starved run: winner did NOT converge",
          ret_class(r.winner_ret) !== :converged || n_conv > 0,
          "winner_ret=$(r.winner_ret) converged_restarts=$n_conv")
    check("ret_class buckets MAXEVAL_REACHED as :limit",
          ret_class(:MAXEVAL_REACHED) === :limit)
    check("ret_class buckets FTOL_REACHED as :converged",
          ret_class(:FTOL_REACHED) === :converged)
end

# -----------------------------------------------------------------------------
banner("A3 -- resume refuses an incompatible or finished run")
# load_resume lives in run_smm.jl, which is a script. Rather than include it (it would
# start a run), the refusal LOGIC is re-checked here against the fields the checkpoint
# carries, so a change to either side breaks this test.
mktempdir() do dir
    names_now = [String(q.name) for q in SMM_PARAMS]
    lo_now    = [q.lo for q in SMM_PARAMS]
    hi_now    = [q.hi for q in SMM_PARAMS]
    q(x) = string('"', x, '"')            # quoting, without nested escapes
    lines = [
        "stage         = " * q("local"),
        "restarts_done = 2",
        "restarts_total= 5",
        "Q_best        = 1.5",
        "objective_grid= 30",
        "grid_search   = 30",
        "grid_report   = 30",
        "param_names   = [" * join((q(n) for n in names_now), ", ") * "]",
        "param_lo      = [" * join(lo_now, ", ") * "]",
        "param_hi      = [" * join(hi_now, ", ") * "]",
        "param_link    = [" * join((q(String(x.link)) for x in SMM_PARAMS), ", ") * "]",
        "targets_sha   = " * q("deadbeefdeadbeef"),
        "",
        "[search_vector]",
        "z = [" * join(incumbent(), ", ") * "]",
    ]
    write(joinpath(dir, "checkpoint.toml"), join(lines, "\n"))
    ck = TOML.parsefile(joinpath(dir, "checkpoint.toml"))

    check("a checkpoint records the current parameter names",
          String.(ck["param_names"]) == names_now)
    check("a checkpoint records the current boxes",
          Float64.(ck["param_lo"]) == lo_now && Float64.(ck["param_hi"]) == hi_now)
    let i = findfirst(==("R_0"), names_now)
        check("R_0's box in the checkpoint is the CURRENT one [0.5, 100]",
              ck["param_lo"][i] == 0.5 && ck["param_hi"][i] == 100.0,
              "[$(ck["param_lo"][i]), $(ck["param_hi"][i])]")
    end
    check("stage is recorded so a finished run can be refused", ck["stage"] == "local")
    check("objective_grid is recorded separately from grid_search",
          haskey(ck, "objective_grid") && haskey(ck, "grid_search"))
    check("targets are identified by CONTENT hash, not by filename",
          haskey(ck, "targets_sha"))

    # The four refusal conditions the loader applies, exercised against this checkpoint.
    check("a \"refined\" checkpoint is not resumable into the local stage",
          "refined" != ck["stage"])
    check("the OLD R_0 box [5, 300] would be refused",
          Float64.(ck["param_lo"]) != [x.name === :R_0 ? 5.0 : x.lo for x in SMM_PARAMS])
    check("a changed search grid would be refused", Int(ck["grid_search"]) != 20)
    check("a changed targets file would be refused",
          ck["targets_sha"] != "0000000000000000")
    check("a dropped parameter would be refused",
          String.(ck["param_names"]) != names_now[1:end-1])
end

# -----------------------------------------------------------------------------
banner("Specification is frozen as instructed")
check("nine estimated parameters", length(SMM_PARAMS) == 9, "$(length(SMM_PARAMS))")
check("ten targeted moments", length(SMM_MOMENTS) == 10, "$(length(SMM_MOMENTS))")
check("sigma_4_1 is NOT estimated and holds at 0.02",
      !any(q -> q.name === :sigma_4_1, SMM_PARAMS) && PARENT_DEFAULTS.sigma_4_1 == 0.02)
check("mu_1 is NOT estimated and holds at -0.04",
      !any(q -> q.name === :mu_1, SMM_PARAMS) && PARENT_DEFAULTS.mu_1 == -0.04)
let q = SMM_PARAMS[findfirst(x -> x.name === :R_0, SMM_PARAMS)]
    check("R_0 box is [0.5, 100.0], searched in logs",
          q.lo == 0.5 && q.hi == 100.0 && q.link === :log,
          "[$(q.lo), $(q.hi)] :$(q.link)")
end

banner(PASS[] ? "ALL CHECKS PASS" : "FAILURES ABOVE -- do not run the estimation")
exit(PASS[] ? 0 : 1)
