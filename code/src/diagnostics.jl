# =============================================================================
# diagnostics.jl — the minimal accuracy checks (X1).
#
# These exist so that solver and domain failures announce themselves instead of
# being eyeballed. They are cheap: run them after every solve and simulation.
#
#   check_solution(child)          NaN/Inf in every solution array
#   check_simulation(child)        simulated states outside their grids
#   check_boundary(child)          share of policies pinned at a bound
#   check_gradient(f, x)           analytic gradient vs central differences
#   run_all_checks(child)          all of the above, one summary
#
# Every check returns a NamedTuple and, unless `throw_on_fail = false`, errors on
# the first violation. Requires paths.jl only for nothing; standalone otherwise.
# =============================================================================

"""
    nan_inf_report(arrays::Pair...; allow_nan = String[], throw_on_fail = true)

Count non-finite entries per array. `allow_nan` names arrays where NaN is a legitimate
"economically infeasible" marker rather than a numerical failure — under the current
design that is the college arrays. `Inf` is never allowed anywhere: `-Inf` belongs only
at the point of a discrete comparison, never in a stored array.
"""
function nan_inf_report(arrays::Pair...; allow_nan::Vector{String} = String[],
                        throw_on_fail::Bool = true, verbose::Bool = true)
    rows = NamedTuple[]
    bad  = String[]
    for (name, a) in arrays
        v = vec(a)
        n = count(isnan, v); i = count(isinf, v)
        pn, pi = 100n/length(v), 100i/length(v)
        push!(rows, (array = name, pct_nan = pn, pct_inf = pi))
        i > 0 && push!(bad, "$name has $(round(pi, digits=2))% Inf (never allowed)")
        n > 0 && !(name in allow_nan) &&
            push!(bad, "$name has $(round(pn, digits=2))% NaN (not an allowed-NaN array)")
    end
    if verbose
        @printf("%-24s %8s %8s\n", "array", "%NaN", "%Inf")
        for r in rows
            @printf("%-24s %8.2f %8.2f%s\n", r.array, r.pct_nan, r.pct_inf,
                    r.array in allow_nan ? "   (NaN = infeasible)" : "")
        end
    end
    if !isempty(bad)
        msg = "Non-finite values in solution arrays:\n  " * join(bad, "\n  ")
        throw_on_fail ? error(msg) : @warn msg
    end
    return (rows = rows, violations = bad)
end

"""
    check_solution(m; kwargs...)

NaN/Inf audit of every child solution array. College arrays are allowed NaN, which marks
states from which college cannot be completed.
"""
function check_solution(m; throw_on_fail::Bool = true, verbose::Bool = true)
    nan_inf_report(
        "sol_v_work"        => m.sol_v_work,
        "sol_c_work"        => m.sol_c_work,
        "sol_h_work"        => m.sol_h_work,
        "sol_tr_work"       => m.sol_tr_work,
        "sol_tr_v_work"     => m.sol_tr_v_work,
        "sol_v_college"     => m.sol_v_college,
        "sol_c_college"     => m.sol_c_college,
        "sol_tr_college"    => m.sol_tr_college,
        "sol_tr_v_college"  => m.sol_tr_v_college,
        "sol_exp_college"   => m.sol_exp_college,
        "sol_exp_v_college" => m.sol_exp_v_college;
        allow_nan = ["sol_v_college", "sol_c_college", "sol_h_college",
                     "sol_tr_college", "sol_tr_v_college",
                     "sol_exp_college", "sol_exp_v_college"],
        throw_on_fail = throw_on_fail, verbose = verbose)
end

"""
    check_simulation(m; tol_share = 0.01)

Share of simulated states outside their solution grid. Beyond the grid the policy is a
flat extrapolation, so a material share means the model is being simulated where it was
never solved. Fails above `tol_share`.
"""
function check_simulation(m; tol_share::Float64 = 0.01, throw_on_fail::Bool = true,
                          verbose::Bool = true)
    a, k = vec(m.sim_a), vec(m.sim_k)
    a, k = filter(isfinite, a), filter(isfinite, k)
    below_a = count(<(m.a_min), a) / max(length(a), 1)
    above_a = count(>(m.a_max), a) / max(length(a), 1)
    above_k = count(>(m.k_max), k) / max(length(k), 1)
    below_k = count(<(m.k_grid[1]), k) / max(length(k), 1)
    res = (below_a = below_a, above_a = above_a, below_k = below_k, above_k = above_k)
    if verbose
        @printf("simulated states outside the grid:\n")
        @printf("  assets  below a_min %.2f%%   above a_max %.2f%%\n", 100below_a, 100above_a)
        @printf("  HC      below k_min %.2f%%   above k_max %.2f%%\n", 100below_k, 100above_k)
    end
    worst = maximum(values(res))
    if worst > tol_share
        msg = "Simulated states leave the solution domain: $(round(100worst, digits=2))% " *
              "(tolerance $(round(100tol_share, digits=2))%). $res. Widen the grids."
        throw_on_fail ? error(msg) : @warn msg
    end
    return res
end

"""
    check_boundary(m; tol_share = 0.05)

Share of interior-period labor choices pinned at a bound. A large share means the box
constraint, not the economics, is setting the policy.
"""
function check_boundary(m; tol_share::Float64 = 0.05, throw_on_fail::Bool = false,
                        verbose::Bool = true)
    h = filter(isfinite, vec(m.sol_h_work))
    lo = count(x -> x <= 1e-3 + 1e-9, h) / max(length(h), 1)
    hi = count(x -> x >= 1.0 - 1e-9,  h) / max(length(h), 1)
    verbose && @printf("labor at bounds: lower %.2f%%   upper %.2f%%\n", 100lo, 100hi)
    if max(lo, hi) > tol_share
        msg = "Labor policy pinned at a bound for $(round(100max(lo,hi), digits=2))% of states."
        throw_on_fail ? error(msg) : @warn msg
    end
    return (at_lower = lo, at_upper = hi)
end

"""
    check_gradient(f!, x; h = 1e-6, rtol = 1e-4)

Compare an analytic gradient against central differences. `f!(x, grad)` must follow the
NLopt convention: fill `grad` when non-empty and return the objective.

This is the check that would have caught P1 — a leftover `∂V/∂k` term in the labor FOC
whose objective and gradient described different functions.
"""
function check_gradient(f!, x::Vector{Float64}; h::Float64 = 1e-6, rtol::Float64 = 1e-4,
                        throw_on_fail::Bool = true, verbose::Bool = true)
    g = zeros(length(x)); f!(copy(x), g)
    fd = similar(g)
    for i in eachindex(x)
        xp = copy(x); xm = copy(x)
        xp[i] += h; xm[i] -= h
        fd[i] = (f!(xp, Float64[]) - f!(xm, Float64[])) / (2h)
    end
    err = [abs(g[i] - fd[i]) / max(abs(fd[i]), 1e-8) for i in eachindex(x)]
    if verbose
        @printf("%4s %14s %14s %10s\n", "i", "analytic", "finite-diff", "rel.err")
        for i in eachindex(x)
            @printf("%4d %14.6f %14.6f %10.2e\n", i, g[i], fd[i], err[i])
        end
    end
    worst = maximum(err)
    if worst > rtol
        msg = "Analytic gradient disagrees with finite differences: max rel. error " *
              "$(round(worst, sigdigits=3)) at x = $x. Objective and gradient describe " *
              "different functions."
        throw_on_fail ? error(msg) : @warn msg
    end
    return (analytic = g, finite_diff = fd, rel_err = err, worst = worst)
end

"""
    run_all_checks(m; kwargs...)

Solution, simulation and boundary checks in one call.
"""
function run_all_checks(m; throw_on_fail::Bool = true, verbose::Bool = true)
    verbose && println("--- solution ---")
    sol = check_solution(m; throw_on_fail = throw_on_fail, verbose = verbose)
    verbose && println("\n--- simulation ---")
    sim = all(isnan, m.sim_a) ? nothing :
          check_simulation(m; throw_on_fail = throw_on_fail, verbose = verbose)
    verbose && println("\n--- boundaries ---")
    bnd = check_boundary(m; throw_on_fail = false, verbose = verbose)
    verbose && println("\nall checks passed")
    return (solution = sol, simulation = sim, boundary = bnd)
end
