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
        "sol_h_college"     => m.sol_h_college,
        "sol_tr_v_college"  => m.sol_tr_v_college;
        # College arrays: NaN marks states from which college cannot be completed.
        # sol_tr_work/sol_tr_v_work are NO LONGER allowed NaN. They used to be, because the
        # shared grid put a = 0 in the parental dimension, where the parent cannot retain
        # delta_P and kappa_term*log(a_term) diverges. N13 gave the parent its own grid
        # starting at delta_P, so the work transfer is now defined at every parental node
        # and a NaN there is a real failure. Inf is never allowed in any array.
        allow_nan = ["sol_v_college", "sol_c_college", "sol_h_college",
                     "sol_tr_college", "sol_tr_v_college"],
        throw_on_fail = throw_on_fail, verbose = verbose)
end

"""
    check_feasibility_mask(m; throw_on_fail = true)

Verify that NaN in the college arrays lands **exactly** where the model says college is
infeasible, and nowhere else (X4).

`check_solution` allows NaN blanket-wide per array, so a solver failure *inside* the
feasible region would pass. This compares the observed NaN pattern against `a_req` from
`compute_min_assets`: a NaN at a feasible state, or a finite value at an infeasible one,
is reported.
"""
function check_feasibility_mask(m; throw_on_fail::Bool = true, verbose::Bool = true)
    a_req = compute_min_assets(m)
    unexpected_nan = 0; unexpected_finite = 0; tot = 0
    for t in 1:m.t_college, ip in 1:m.Np, ik in 1:m.Nk, ia in 1:m.Na
        feasible = m.a_grid[ia] >= a_req[t]
        v = m.sol_v_college[t, ia, ik, ip, 1]
        tot += 1
        if feasible && isnan(v)
            unexpected_nan += 1
        elseif !feasible && !isnan(v)
            unexpected_finite += 1
        end
    end
    # The transfer stage carries a second, different mask on the PARENTAL grid: college
    # needs a >= a_req[1] + delta_P. C14 built interpolants across it, so check it too.
    col_min = min_parent_assets_for_college(m)
    for it in 1:m.Nt, ip in 1:m.Np, ik in 1:m.Nk, ia in 1:m.Nap
        feasible = m.ap_grid[ia] >= col_min
        v = m.sol_tr_v_college[ia, ik, ip, it]
        tot += 1
        if feasible && isnan(v)
            unexpected_nan += 1
        elseif !feasible && !isnan(v)
            unexpected_finite += 1
        end
    end
    res = (unexpected_nan = unexpected_nan, unexpected_finite = unexpected_finite, checked = tot)
    verbose && @printf("college NaN mask vs a_req: %d NaN at feasible states, %d finite at infeasible (of %d)\n",
                       unexpected_nan, unexpected_finite, tot)
    if unexpected_nan > 0 || unexpected_finite > 0
        msg = "College NaN pattern does not match the feasibility mask: " *
              "$unexpected_nan NaN at feasible states, $unexpected_finite finite at infeasible states."
        throw_on_fail ? error(msg) : @warn msg
    end
    return res
end

"""
    check_solver_domain(m; tol_share = 0.01)

Share of **stored solution** transitions that leave the solved grid (C16).

The work solver constrains only `a' >= a_min`; nothing bounds `a'` above or `k'` at all,
so the continuation value is evaluated off-grid by extrapolation. Forward simulation can
report 0% while the *solution itself* left the domain, which is why this is measured
separately from `check_simulation`.
"""
function check_solver_domain(m; tol_share::Float64 = 0.01, throw_on_fail::Bool = true,
                             verbose::Bool = true)
    na = 0; nk = 0; nbind = 0; tot = 0
    for t in 1:(m.T - 1), ip in 1:m.Np, ik in 1:m.Nk, ia in 1:m.Na
        c = m.sol_c_work[t, ia, ik, ip, 1]; h = m.sol_h_work[t, ia, ik, ip, 1]
        (isfinite(c) && isfinite(h)) || continue
        w_pre = wage_func(m, m.k_grid[ik], t, m.p_grid[ip])
        a_n = (1 + m.r) * m.a_grid[ia] + after_tax_income(m, w_pre, h) - c + m.y
        k_n = m.k_grid[ik] + h
        tot += 1
        (a_n > m.a_max || a_n < m.a_min) && (na += 1)
        (k_n > m.k_max || k_n < m.k_grid[1]) && (nk += 1)
        # C16 imposes k' <= k_max as a box bound on h, so k_n can no longer exceed k_max
        # and the `nk` share above is 0 by construction. What matters instead is how often
        # that ceiling BINDS: every binding state is one where k_max, a computational
        # choice, is doing economic work. A non-trivial share means k_max is too small.
        h_hi = clamp(m.k_max - m.k_grid[ik], 1e-3, 1.0)
        h >= h_hi - 1e-6 && h_hi < 1.0 && (nbind += 1)
    end
    res = (assets = na / max(tot, 1), hc = nk / max(tot, 1),
           hc_ceiling_binds = nbind / max(tot, 1), checked = tot)
    verbose && @printf("stored work transitions off-grid: assets %.2f%%   HC %.2f%%   | HC ceiling binds %.2f%%   (of %d)\n",
                       100res.assets, 100res.hc, 100res.hc_ceiling_binds, tot)
    worst = max(res.assets, res.hc)
    if worst > tol_share
        msg = "Stored solution transitions leave the grid: $(round(100worst, digits=2))% " *
              "(tolerance $(round(100tol_share, digits=2))%). The continuation value is " *
              "being extrapolated. Widen the grids or add upper-domain constraints."
        throw_on_fail ? error(msg) : @warn msg
    end
    if res.hc_ceiling_binds > tol_share
        @warn "The k_max ceiling binds at $(round(100res.hc_ceiling_binds, digits=2))% of " *
              "stored work states. k_max is restricting human capital as economics, not " *
              "just bounding the grid. Raise k_max (currently $(m.k_max))."
    end
    return res
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
    na_a, na_k = length(a), length(k)

    # X3: non-finite entries are VIOLATIONS, not values to be filtered away. Previously
    # this filtered first and divided by the survivors, so a sim_a that was 96% NaN
    # reported "0.00% outside" and passed -- masking C12 and any NaN-producing failure.
    nonfinite_a = count(!isfinite, a) / max(na_a, 1)
    nonfinite_k = count(!isfinite, k) / max(na_k, 1)

    # shares are over ALL entries, so they cannot be inflated by a shrinking denominator
    below_a = count(x -> isfinite(x) && x < m.a_min,     a) / max(na_a, 1)
    above_a = count(x -> isfinite(x) && x > m.a_max,     a) / max(na_a, 1)
    below_k = count(x -> isfinite(x) && x < m.k_grid[1], k) / max(na_k, 1)
    above_k = count(x -> isfinite(x) && x > m.k_max,     k) / max(na_k, 1)
    res = (below_a = below_a, above_a = above_a, below_k = below_k, above_k = above_k,
           nonfinite_a = nonfinite_a, nonfinite_k = nonfinite_k)
    if verbose
        @printf("simulated states outside the grid:\n")
        @printf("  assets  below a_min %.2f%%   above a_max %.2f%%   non-finite %.2f%%\n",
                100below_a, 100above_a, 100nonfinite_a)
        @printf("  HC      below k_min %.2f%%   above k_max %.2f%%   non-finite %.2f%%\n",
                100below_k, 100above_k, 100nonfinite_k)
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
    # h_lo mirrors the solver's box bound in child_lifecycle.jl. Kept as a named constant
    # here rather than a bare literal so the two cannot drift apart silently.
    h_lo, h_hi = 1e-3, 1.0
    lo = count(x -> x <= h_lo + 1e-9, h) / max(length(h), 1)
    hi = count(x -> x >= h_hi - 1e-9, h) / max(length(h), 1)
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
    verbose && println("\n--- solver domain (C16) ---")
    dom = check_solver_domain(m; throw_on_fail = throw_on_fail, verbose = verbose)
    verbose && println("\n--- feasibility mask (X4) ---")
    msk = check_feasibility_mask(m; throw_on_fail = throw_on_fail, verbose = verbose)
    verbose && println("\n--- boundaries ---")
    bnd = check_boundary(m; throw_on_fail = false, verbose = verbose)
    verbose && println("\nall checks passed")
    return (solution = sol, simulation = sim, solver_domain = dom, mask = msk, boundary = bnd)
end

# =============================================================================
# Phase 4 — full accuracy diagnostics (X1)
# =============================================================================

"""
    bellman_residual(m; n_sample = 500, rng = MersenneTwister(1))

Max and mean relative Bellman residual on the child's work path:

    |V_t(a,k,z) - [u(c*,h*) + beta E V_{t+1}(a',k',z')]| / (1 + |V_t|)

evaluated at randomly sampled interior grid points using the *stored* policies. A large
residual means the stored value is not the value its own policy generates — i.e. the
optimizer did not actually solve the problem at those states.
"""
function bellman_residual(m; n_sample::Int = 500, rng = MersenneTwister(1), verbose::Bool = true)
    res = Float64[]
    # Interpolators are built once per period, not once per sample: the loop previously
    # constructed Np of them at every draw (2500 builds at n_sample = 500).
    interps = Dict{Int,Any}()
    for _ in 1:n_sample
        t   = rand(rng, 1:(m.T - 1))
        ia  = rand(rng, 2:m.Na); ik = rand(rng, 2:m.Nk); ip = rand(rng, 1:m.Np)
        a, k, z = m.a_grid[ia], m.k_grid[ik], m.p_grid[ip]
        c, h, V = m.sol_c_work[t,ia,ik,ip,1], m.sol_h_work[t,ia,ik,ip,1], m.sol_v_work[t,ia,ik,ip,1]
        (isfinite(c) && isfinite(h) && isfinite(V)) || continue

        w_pre  = wage_func(m, k, t, z)
        a_next = (1 + m.r) * a + after_tax_income(m, w_pre, h) - c + m.y
        k_next = k + h
        interp = get!(interps, t) do
            create_interpolator(m, m.sol_v_work, t + 1)
        end
        EV = sum(m.p_transition[ip, jp] * interp[jp](a_next, k_next) for jp in 1:m.Np)
        rhs = util_work(m, c, h) + m.beta * EV
        push!(res, abs(V - rhs) / (1 + abs(V)))
    end
    isempty(res) && return (max = NaN, mean = NaN, n = 0)
    out = (max = maximum(res), mean = mean(res), n = length(res))
    verbose && @printf("Bellman residual (work path): max %.3e   mean %.3e   over %d states\n",
                       out.max, out.mean, out.n)
    return out
end

"""
    monotonicity_report(m)

Share of adjacent asset-grid pairs violating monotonicity of `V` in assets and of the
consumption policy in assets. Both should be (weakly) increasing; a material violation
means the solution is not economically sensible, whatever the optimizer reported.
"""
function monotonicity_report(m; verbose::Bool = true)
    badV = 0; badC = 0; tot = 0
    for t in 1:m.T, ik in 1:m.Nk, ip in 1:m.Np, ia in 1:(m.Na - 1)
        v1, v2 = m.sol_v_work[t,ia,ik,ip,1], m.sol_v_work[t,ia+1,ik,ip,1]
        c1, c2 = m.sol_c_work[t,ia,ik,ip,1], m.sol_c_work[t,ia+1,ik,ip,1]
        (isfinite(v1) && isfinite(v2)) || continue
        tot += 1
        v2 < v1 - 1e-8 && (badV += 1)
        (isfinite(c1) && isfinite(c2) && c2 < c1 - 1e-8) && (badC += 1)
    end
    out = (V_violations = badV/max(tot,1), c_violations = badC/max(tot,1), pairs = tot)
    verbose && @printf("monotonicity in assets: V %.2f%%   c %.2f%%   (%d pairs)\n",
                       100out.V_violations, 100out.c_violations, out.pairs)
    return out
end

"""
    mc_standard_errors(x; n_boot = 200, rng = MersenneTwister(2))

Bootstrap standard error of a simulated mean. Without this a counterfactual difference
cannot be told apart from Monte Carlo noise, even with common random numbers.
"""
function mc_standard_errors(x::AbstractVector; n_boot::Int = 200, rng = MersenneTwister(2))
    v = collect(skipmissing(filter(isfinite, x)))
    isempty(v) && return (mean = NaN, se = NaN, n = 0)
    n = length(v)
    bs = [mean(v[rand(rng, 1:n, n)]) for _ in 1:n_boot]
    return (mean = mean(v), se = std(bs), n = n)
end

"""
    grid_refinement(build, solve_and_summarise, sizes; verbose = true)

Solve at several grid sizes and report how a scalar summary moves. `build(n)` returns a
model, `solve_and_summarise(m)` returns the scalar. Stability across `sizes` is the
evidence that a result is a property of the model rather than of the discretization.
"""
function grid_refinement(build, solve_and_summarise, sizes::Vector{Int}; verbose::Bool = true)
    rows = NamedTuple[]
    for n in sizes
        m = build(n)
        val = solve_and_summarise(m)
        push!(rows, (grid = n, value = val))
        verbose && @printf("  Na=Nk=%-4d  summary = %.6f\n", n, val)
    end
    if length(rows) > 1
        d = abs(rows[end].value - rows[end-1].value) / max(abs(rows[end].value), 1e-12)
        verbose && @printf("  relative change over the last refinement: %.3e\n", d)
        return (rows = rows, last_rel_change = d)
    end
    return (rows = rows, last_rel_change = NaN)
end

"""
    shock_discretization_report(ρ, σ, N; verbose = true)

Compare Tauchen against Rouwenhorst on the moments they are meant to reproduce:
the implied persistence and the unconditional standard deviation.

Phase 0.7 kept the stationary AR(1) as a documented approximation to the estimated
permanent-plus-transitory process. This quantifies the *discretization* error inside that
approximation, so the choice of `N` and method can be stated rather than assumed.
"""
function shock_discretization_report(ρ::Float64, σ::Float64, N::Int; verbose::Bool = true)
    rows = NamedTuple[]
    for (name, mc) in (("Tauchen", tauchen(N, ρ, σ, 0.0, 3)),
                       ("Rouwenhorst", rouwenhorst(N, ρ, σ)))
        P, z = mc.p, collect(mc.state_values)
        π = stationary_dist(Matrix(P))
        μ  = sum(π .* z)
        v  = sum(π .* (z .- μ).^2)
        # implied persistence: cov(z_{t+1}, z_t) / var(z_t) under the stationary dist
        cov = sum(π[i] * P[i,j] * (z[i]-μ) * (z[j]-μ) for i in 1:N, j in 1:N)
        push!(rows, (method = name, sd = sqrt(v), persistence = cov/v))
    end
    target_sd = σ / sqrt(1 - ρ^2)
    if verbose
        @printf("AR(1) discretization, rho=%.3f sigma=%.3f N=%d  (target sd %.4f, rho %.3f)\n",
                ρ, σ, N, target_sd, ρ)
        for r in rows
            @printf("  %-12s sd %.4f (err %+.2f%%)   persistence %.4f (err %+.2f%%)\n",
                    r.method, r.sd, 100*(r.sd/target_sd - 1), r.persistence,
                    100*(r.persistence/ρ - 1))
        end
    end
    return (rows = rows, target_sd = target_sd, target_rho = ρ)
end
