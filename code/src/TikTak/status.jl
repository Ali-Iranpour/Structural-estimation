# =============================================================================
# status.jl -- return-code classes and the acceptance rule.
#
# Pure functions, no I/O. The runner (code/smm/run_smm.jl) and the regression tests call
# the SAME functions, so a test of the acceptance rule tests what the runner does rather
# than a copy of it (plan step 1.4).
# =============================================================================

"""
    ret_class(ret) -> Symbol

Bucket an NLopt return code into `:converged`, `:limit` (a budget stopped it, not a
criterion), or `:other` (including failures and exceptions).

`:MAXEVAL_REACHED` and `:MAXTIME_REACHED` mean the search was cut off with its stopping
test unsatisfied. That is a legitimate answer to report, but it is not convergence, and a
run made mostly of those has a budget problem however good its objective looks.
"""
ret_class(ret::Symbol) =
    ret in (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED, :STOPVAL_REACHED) ? :converged :
    ret in (:MAXEVAL_REACHED, :MAXTIME_REACHED)                       ? :limit     :
    :other

"""
    ret_tally(result) -> Dict{Symbol,Int}

How many local searches ended in each NLopt return code. Reported next to the objective so
"converged" is a statement about the searches and not about the number they produced.
"""
ret_tally(r::TikTakResult) = begin
    d = Dict{Symbol,Int}()
    for row in r.trace; d[row.ret] = get(d, row.ret, 0) + 1; end
    d
end

"Box-normalized max-norm distance: max_i |x_i - y_i| / (hi_i - lo_i)."
function boxdist(x::Vector{Float64}, y::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64})
    d = 0.0
    for i in eachindex(x, y, lo, hi)
        d = max(d, abs(x[i] - y[i]) / (hi[i] - lo[i]))
    end
    return d
end

"One line naming the solver and its stopping rules, recorded with a verification."
solver_summary(s::SolverSettings) =
    @sprintf("%s ftol_rel=%g ftol_abs=%g xtol_rel=%g maxeval=%d", s.alg, s.ftol_rel, s.ftol_abs, s.xtol_rel, s.maxeval) *
    (s.initial_step > 0 ? @sprintf(" initial_step=%g %s", s.initial_step, s.step_schedule) : "")

"""
    candidate_valid(x, f, lo, hi, invalid_value) -> Bool

A point the run may return: the box's dimension, finite coordinates inside the box, and a
finite value below the invalid-value threshold.
"""
candidate_valid(x::Vector{Float64}, f::Float64, lo::Vector{Float64}, hi::Vector{Float64}, invalid_value::Float64) =
    length(x) == length(lo) && all(isfinite, x) && all(lo .<= x .<= hi) && valid_value(f, invalid_value)

"""
    merge_candidate(inc, x, f, ret, stage, restart, attempt, lo, hi, cfg, objective_id, solver)
        -> (action, incumbent)

THE merge rule, shared by the local stage, the polish and the runner's refinement.

  :improved  `f` is valid and STRICTLY below the retained value: the point is replaced and
             its origin is this search (restart, attempt, return code, objective). Any
             verification of the old point is gone with it.
  :verified  not an improvement, but a converged solve on the same objective returned the
             retained point itself (within cfg.verify_xtol, box-normalized) with a consistent
             value: the retained point keeps its origin and gains a verification.
  :none      anything else. Equal Q at a different x verifies nothing.

Monotone retention is unchanged from the published algorithm: only a strict improvement
moves the incumbent.
"""
function merge_candidate(inc::Incumbent, x::Vector{Float64}, f::Float64, ret::Symbol,
                         stage::Symbol, restart::Int, attempt::Int,
                         lo::Vector{Float64}, hi::Vector{Float64}, cfg::TikTakConfig,
                         objective_id::String, solver::String)
    if candidate_valid(x, f, lo, hi, cfg.invalid_value) && f < inc.f
        origin = CandidateOrigin(stage, restart, attempt, 0, ret, objective_id)
        return (:improved, Incumbent(copy(x), f, inc.version + 1, origin, NO_VERIFICATION))
    end
    if inc.verification.status !== :verified && ret_class(ret) === :converged &&
       objective_id == inc.origin.objective_id && length(x) == length(inc.x) && isfinite(f)
        d = boxdist(x, inc.x, lo, hi)
        if d <= cfg.verify_xtol && abs(f - inc.f) <= max(cfg.verify_ftol_abs, cfg.verify_ftol_rel * abs(inc.f))
            ver = Verification(:verified, stage, restart, attempt, ret, copy(x), f, d, cfg.verify_xtol,
                               objective_id, solver)
            return (:verified, Incumbent(inc.x, inc.f, inc.version, inc.origin, ver))
        end
    end
    return (:none, inc)
end

"""
    local_converged(inc, objective_id) -> Bool
    local_converged(result::TikTakResult) -> Bool

Is there convergence evidence for THIS point on THIS objective? Either the search that
produced it stopped on a convergence test, or a later converged solve verified it. A
certificate earned on another objective (the coarse search grid) does not count.
"""
local_converged(inc::Incumbent, objective_id::AbstractString) =
    (inc.origin.objective_id == objective_id && ret_class(inc.origin.ret) === :converged) ||
    (inc.verification.status === :verified && inc.verification.objective_id == objective_id)
local_converged(r::TikTakResult) = local_converged(r.incumbent, r.objective_id)

"""
    Acceptance

The final verdict and each condition behind it, reported separately (plan step 3.5):

  execution_ok            no stage threw (objective exceptions, a failed refinement)
  candidate_valid         the reported point is a valid point with a valid, in-domain value
  local_converged         convergence evidence for the reported point on the REPORTING objective
  search_budget_complete  the planned restarts ran (an early stop by stop_tol counts); not paused
  interior                no parameter within the pinned margin of its bound (the runner's policy)

None of these is a claim of global optimality.
"""
struct Acceptance
    execution_ok::Bool
    candidate_valid::Bool
    local_converged::Bool
    search_budget_complete::Bool
    interior::Bool
    accepted::Bool
    reasons::Vector{String}
end

"""
    acceptance(; execution_ok, candidate_valid, local_converged, search_budget_complete,
                 pinned = String[]) -> Acceptance

The acceptance rule of code/smm/run_smm.jl (2026-09-27). The runner and the tests call this
function, so a test of the rule is a test of what the runner does (plan step 1.4).
"""
function acceptance(; execution_ok::Bool, candidate_valid::Bool, local_converged::Bool,
                    search_budget_complete::Bool, pinned::AbstractVector = String[])
    pinned = String[string(p) for p in pinned]       # an empty comprehension may arrive as Vector{Any}
    reasons = String[]
    execution_ok || push!(reasons, "a stage threw (objective exception or failed refinement)")
    candidate_valid || push!(reasons, "the reported point or its value is not valid")
    local_converged || push!(reasons, "no convergence evidence for the reported point on the reporting objective")
    search_budget_complete || push!(reasons, "the planned search did not complete")
    isempty(pinned) || push!(reasons, "parameter(s) near a search bound: " * join(pinned, ", "))
    return Acceptance(execution_ok, candidate_valid, local_converged, search_budget_complete,
                      isempty(pinned), isempty(reasons), reasons)
end
