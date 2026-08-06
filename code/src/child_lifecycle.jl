# -------------------------------
# Utility: Nonlinear Grid Creator
# -------------------------------
function nonlinspace(start::Float64, stop::Float64, num::Int, curv::Float64)
    lin_vals = range(0, stop=1, length=num)
    curved_vals = lin_vals .^ curv
    return start .+ (stop - start) .* curved_vals
end

function create_focused_grid(a_min::Float64, a_focus::Float64, a_max::Float64, Na::Int, focus_share::Float64, curv::Float64)
    Na_focus = ceil(Int, Na * focus_share)
    Na_rest = Na - Na_focus
    grid_focus = nonlinspace(a_min, a_focus, Na_focus, curv)
    grid_rest = nonlinspace(a_focus, a_max, Na_rest + 1, curv)[2:end]
    return vcat(grid_focus, grid_rest)
end

# =============================================================================
# Child lifecycle: college decision, work, AR(1) wage shock, progressive tax.
#
# Derived from child_lifecycle_ret.jl by removing the retirement stage, which
# model.txt does not contain ("The model has no retirement stage and ends as
# the child becomes 68 years old"). Everything else — the HSV/Benabou
# progressive tax, the transfer problem, the belief machinery — is unchanged.
#
# See docs/ERRORS.md for the open findings that still apply to this file.
# =============================================================================

# =============================================================================
# Dynamic Labor Model with College Decision and AR1 Shock
# =============================================================================
mutable struct ConSavLaborCollege_AR1
    T::Int; t_college::Int; rho::Float64; beta::Float64; phi::Float64
    eta::Float64; alpha::Float64; y::Float64; w::Float64; tau::Float64
    r::Float64; a_max::Float64; a_min::Float64; Na::Int; k_max::Float64
    Nk::Int; simT::Int; simN::Int
    a_grid::Vector{Float64}; k_grid::Vector{Float64}

    psi_terminal::Float64       # Terminal value weight on human capital
    kappa_terminal::Float64     # Weight on parent's retained assets
    omega::Float64              # Weight on child's life-cycle utility
    mu::Float64                 # Weight on Parent's utility

    # Transitory shocks
    Nt::Int                     # Number of grid points for t
    t_grid::Vector{Float64}     # Grid for transitory shock t
    sigma_eps::Float64          # Std dev of transitory shock
    t_weight::Vector{Float64}   # Weights for t quadrature

    # --- Stochastic Shock Parameters (AR1 only) ---
    Np::Int; p_grid::Vector{Float64}; p_transition::Matrix{Float64}
    p_ar1::Float64; sigma_p::Float64

    # --- Solution arrays (5D: T, Na, Nk, Np, Nt) ---
    sol_c_work::Array{Float64, 5}; sol_h_work::Array{Float64, 5}; sol_v_work::Array{Float64, 5}
    sol_c_college::Array{Float64, 5}; sol_h_college::Array{Float64, 5}; sol_v_college::Array{Float64, 5}

    # --- Solution arrays (4D: Na, Nk, Np, Nt) ---
    sol_tr_college::Array{Float64, 4}; sol_tr_work::Array{Float64, 4}
    sol_tr_v_college::Array{Float64, 4}; sol_tr_v_work::Array{Float64, 4}

    # --- Simulation arrays ---
    sim_c::Matrix{Float64}; sim_h::Matrix{Float64}; sim_a::Matrix{Float64}
    sim_k::Matrix{Float64}; sim_p_idx::Matrix{Int}
    sim_a_init::Vector{Float64}; sim_k_init::Vector{Float64}; sim_p_init_idx::Vector{Int}
    sim_income::Matrix{Float64}; sim_wage::Matrix{Float64}
    draws_uniform_p::Matrix{Float64}
    w_vec::Vector{Float64}; college_cost::Float64; college_boost::Float64
    kappa::Float64 # parameter for psychic cost
    tax_lambda::Float64          # Progressive tax level parameter (HSV/Benabou)
    c_floor::Float64             # Consumption floor; MUST equal the optimizer's lower bound on c
    delta_P::Float64             # Minimum retained parental asset (a_bar^P) at the transfer stage
end

# =============================================================================
# Constructor for ConSavLaborCollege_AR1 with AR1 Shock
# =============================================================================
function ConSavLaborCollege_AR1(;
                # T = 51: ages 18..68 inclusive, per model.txt. Was 52 in
                # child_lifecycle_ret.jl, which implied a terminal age of 69.
                T::Int=51, t_college::Int=4, beta::Float64=0.97, rho::Float64=1.0,
                r::Float64=0.03, a_max::Float64=100.0, Na::Int=30, y::Float64=0.6,
                # a_min = 0: the work branch admits tr = 0, so 0 must be IN the child's asset
                # grid or the first child period is evaluated by extrapolation. Was 0.01.
                simN::Int=5000, a_min::Float64=0.0, k_max::Float64=30.0, Nk::Int=30,
                w::Float64=12.5, tau::Float64=0.18, eta::Float64=2.0, alpha::Float64=0.08,
                phi::Float64=18.0, seed::Int=1234, college_cost::Float64=1.2,
                college_boost::Float64=2.0, kappa::Float64=5.0,
                # Shock parameters (AR1 only)
                p_ar1::Float64=0.95, sigma_p::Float64=0.2, Np::Int=5,
                # Preference shock parameters
                Nt=11, sigma_eps=0.5,
                # --- Terminal value parameters ---
                psi_terminal::Float64=1.0, kappa_terminal::Float64=10.0, omega::Float64=0.5,
                    # --- Bargaining parameter ---
                mu = 0.5,
                tax_lambda::Float64=0.82,
                # Consumption floor used BOTH as the optimizer's lower bound on c and in
                # the college feasibility recursion. These were 0.01 and 0.3 respectively,
                # so the feasibility test excluded states the optimizer could in fact solve.
                c_floor::Float64=0.01,
                # a_bar^P: minimum asset the parent retains at the transfer stage. Set
                # equal to c_floor by decision (2026-08-05): the parent must retain at
                # least one period's minimum consumption. Was an implicit 1e-9, which let
                # kappa_terminal*log(a_term) reach about -186. See N12 in docs/ERRORS.md.
                delta_P::Float64=0.01
                )

    simT = T
    a_grid = create_focused_grid(a_min, 2.0, a_max, Na, 0.2, 1.3)
    k_grid = nonlinspace(0.001, k_max, Nk, 1.5)

    # Gauss-Hermite quadrature for transitory shocks
    nodes, weights = gausshermite(Nt)
    t_grid = sqrt(2) * sigma_eps .* nodes
    t_weight = weights / sqrt(pi)

    # --- Setup Persistent AR1 Shock ---
    mc = tauchen(Np, p_ar1, sigma_p, 0.0, 3)
    p_grid = exp.(mc.state_values)
    p_transition = mc.p

    # --- Initialize solution arrays (5D) ---
    sol_shape = (T, Na, Nk, Np, Nt)
    sol_c_work = fill(NaN, sol_shape); sol_h_work = fill(NaN, sol_shape); sol_v_work = fill(NaN, sol_shape)
    sol_c_college = fill(NaN, sol_shape); sol_h_college = fill(NaN, sol_shape); sol_v_college = fill(NaN, sol_shape)

    # --- Initialize half period solution arrays (5D) ---
    tr_shape = (Na, Nk, Np, Nt)
    sol_tr_college = fill(NaN, tr_shape); sol_tr_work = fill(NaN, tr_shape)
    sol_tr_v_college = fill(NaN, tr_shape); sol_tr_v_work = fill(NaN, tr_shape)

    # --- Initialize simulation arrays ---
    sim_shape = (simN, T)
    sim_c = fill(NaN, sim_shape); sim_h = fill(NaN, sim_shape)
    sim_a = fill(NaN, (simN, T+1)); sim_k = fill(NaN, sim_shape)
    sim_p_idx = fill(0, sim_shape)
    sim_income = fill(NaN, sim_shape); sim_wage = fill(NaN, sim_shape)

    rng = MersenneTwister(seed)
    sim_a_init = rand(rng, simN) .* 20
    sim_k_init = rand(rng, simN) .* 5
    # C6 (Phase 0.5c): draw the initial child shock from the STATIONARY distribution --
    # the same distribution the transfer problem integrates over. Starting everyone at the
    # median state meant the transfer was optimal against a distribution no simulated child
    # was drawn from.
    sim_p_init_idx = let pi_p = stationary_dist(p_transition), u = rand(rng, simN)
        [discrete_draw(pi_p, u[i]) for i in 1:simN]
    end

    draws_uniform_p = rand(rng, sim_shape...)
    w_vec = fill(w, T)

    return ConSavLaborCollege_AR1(
        T, t_college, rho, beta, phi, eta, alpha, y, w, tau, r,
        a_max, a_min, Na, k_max, Nk, simT, simN, a_grid, k_grid,
        psi_terminal, kappa_terminal, omega, mu,
        Nt, t_grid, sigma_eps, t_weight,
        Np, p_grid, p_transition, p_ar1, sigma_p,
        sol_c_work, sol_h_work, sol_v_work, sol_c_college, sol_h_college, sol_v_college,
        sol_tr_college, sol_tr_work, sol_tr_v_college, sol_tr_v_work,
        sim_c, sim_h, sim_a, sim_k, sim_p_idx,
        sim_a_init, sim_k_init, sim_p_init_idx, sim_income, sim_wage,
        draws_uniform_p, w_vec, college_cost, college_boost, kappa, tax_lambda,
        c_floor, delta_P
    )
end


const WAGE_SCALING_FACTOR = 0.584

# ---------------------------------------------------------------------------
# Extrapolation convention (C11). One rule, applied everywhere:
#
#   VALUE functions  -> Line()  : preserve the boundary slope. Flat() would assert
#                                 dV/da = 0 off-grid, which is economically wrong and
#                                 makes saving look worthless.
#   POLICY functions -> Flat()  : hold the boundary policy. Line() would extrapolate a
#                                 policy into territory where it was never optimal.
#
# Previously the transfer solve used Line() while the simulators used Flat() on the same
# objects, so the value that selected a transfer was not the value its simulated policy
# generated. With a_min = 0 and tr bounded into [0 or a_req[1], a - delta_P], evaluation
# should stay inside the grid; `check_simulation` reports any excursion.
# ---------------------------------------------------------------------------

# ================================
# Progressive tax helpers
# ================================
# Pre-tax hourly wage (no taxes here)
@inline function wage_func(model::ConSavLaborCollege_AR1, k::Float64, t::Int, p_shock::Float64)
    base_wage = model.w_vec[t] * (1 + model.alpha * k)
    return base_wage * p_shock * WAGE_SCALING_FACTOR
end

# After-tax labor income: λ * (w*h)^(1 - τ)
@inline function after_tax_income(model::ConSavLaborCollege_AR1, w_pre::Float64, h::Float64)
    return model.tax_lambda * (w_pre * h)^(1.0 - model.tau)
end

# d/dh of after-tax labor income
# = λ (1-τ) * w * (w*h)^(-τ)
@inline function d_after_tax_dh(model::ConSavLaborCollege_AR1, w_pre::Float64, h::Float64)
    return model.tax_lambda * (1.0 - model.tau) * w_pre * (w_pre * h)^(-model.tau)
end

# ================================
# Solver result validation
# ================================
"""
    check_nlopt!(ret, minf, x, where)

Reject an optimizer result instead of storing it. NLopt does not throw on
`:INVALID_ARGS` or `:FAILURE`; it returns NaN, and silently storing that lets the NaN
propagate through `init` and the interpolated continuation value. Economic
infeasibility is handled separately, by the feasibility masks -- this function only
signals *numerical* failure.
"""
@inline function check_nlopt!(ret, minf, x, where::AbstractString)
    if ret in (:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :FORCED_STOP)
        error("NLopt returned $ret at $where")
    end
    if !isfinite(minf) || any(!isfinite, x)
        error("NLopt returned a non-finite result at $where: minf=$minf, x=$x")
    end
    return nothing
end

# ================================
# Work-path solver (progressive tax, no retirement)
# ================================
function solve_model_work!(model::ConSavLaborCollege_AR1)
    @unpack T, Na, Nk, Np, a_grid, k_grid, p_grid = model
    @unpack sol_c_work, sol_h_work, sol_v_work = model
    @unpack y = model

    # ---- Final period (t = T): work, consume everything, no bequest ----
    for i_p in 1:Np, i_k in 1:Nk, i_a in 1:Na
        assets, capital = a_grid[i_a], k_grid[i_k]
        p_shock = p_grid[i_p]

        function obj_wrapper(h_vec::Vector, grad::Vector)
            f = obj_last_period(model, h_vec, assets, capital, T, p_shock, grad)
            if length(grad) > 0
                grad[:] = -grad[:]
            end
            return -f
        end

        opt = Opt(:LD_SLSQP, 1)
        lower_bounds!(opt, [1e-3])
        upper_bounds!(opt, [1.0])
        ftol_rel!(opt, 1e-8)
        maxeval!(opt, 1000)
        min_objective!(opt, obj_wrapper)
        (minf, h_vec, ret) = optimize(opt, [0.3])
        check_nlopt!(ret, minf, h_vec, "work terminal ia=$i_a ik=$i_k ip=$i_p")

        h_opt = h_vec[1]
        w_pre = wage_func(model, capital, T, p_shock)
        cons  = assets + after_tax_income(model, w_pre, h_opt) + y

        sol_h_work[T, i_a, i_k, i_p, :] .= h_opt
        sol_c_work[T, i_a, i_k, i_p, :] .= cons
        sol_v_work[T, i_a, i_k, i_p, :] .= -minf
    end

    # ---- Working ages (t = T-1 down to 1) ----
    @showprogress 1 "Solving working model..." for t in (T-1):-1:1
        interp = create_interpolator(model, sol_v_work, t + 1)
        for i_p in 1:Np, i_k in 1:Nk, i_a in 1:Na
            assets, capital = a_grid[i_a], k_grid[i_k]
            p_shock = p_grid[i_p]

            function obj_wrapper(x::Vector, grad::Vector)
                f = obj_work_period(model, x, assets, capital, t, p_shock, i_p, interp, grad)
                if length(grad) > 0
                    grad[:] = -grad[:]  # negate for minimization
                end
                return -f
            end

            # Consumption box bound = the actual period budget at h = 1, rather
            # than an arbitrary constant. The asset constraint below already
            # enforces a_next >= a_min, so this bound never binds; it only has
            # to be loose enough to contain the optimum and the initial guess.
            # With a fixed cap of 50.0 the initial guess taken from t+1 could
            # exceed it (terminal consumption reaches ~63 at a_max), and NLopt
            # then returns :INVALID_ARGS with NaN, which propagates backwards.
            w_pre_t = wage_func(model, capital, t, p_shock)
            c_hi = max((1.0 + model.r) * assets + after_tax_income(model, w_pre_t, 1.0) + y, 0.02)

            opt = Opt(:LD_SLSQP, 2)
            lower_bounds!(opt, [0.01, 1e-3])
            upper_bounds!(opt, [c_hi, 1.0])
            ftol_rel!(opt, 1e-8)
            maxeval!(opt, 1000)
            inequality_constraint!(opt, (x, grad) -> asset_constraint_work(x, grad, model, assets, capital, t, p_shock), 1e-6)
            min_objective!(opt, obj_wrapper)

            init = [clamp(sol_c_work[t + 1, i_a, i_k, i_p, 1], 0.01, c_hi), 0.4]
            (minf, x_opt, ret) = optimize(opt, init)
            check_nlopt!(ret, minf, x_opt, "work t=$t ia=$i_a ik=$i_k ip=$i_p")
            sol_c_work[t, i_a, i_k, i_p, :] .= x_opt[1]
            sol_h_work[t, i_a, i_k, i_p, :] .= x_opt[2]
            sol_v_work[t, i_a, i_k, i_p, :] .= -minf
        end
    end
end

# ================================
# College-path solver (unchanged)
# ================================
function solve_model_college!(model::ConSavLaborCollege_AR1)
    @unpack T, t_college, Na, Nk, Np, a_grid, k_grid, p_grid, c_floor = model
    @unpack sol_c_college, sol_h_college, sol_v_college, sol_v_work = model

    a_req = compute_min_assets(model)
    if a_req[1] > a_grid[end]
        error("College is infeasible at every asset grid point: a_req[1] = $(a_req[1]) > " *
              "a_max = $(a_grid[end]). Widen the grid or revisit college_cost / y / c_floor.")
    end

    @showprogress 1 "Solving college model..." for t in T:-1:1
        if t > t_college
            sol_c_college[t, :, :, :, :] .= model.sol_c_work[t, :, :, :, :]
            sol_h_college[t, :, :, :, :] .= model.sol_h_work[t, :, :, :, :]
            sol_v_college[t, :, :, :, :] .= model.sol_v_work[t, :, :, :, :]

        else
            # Continuation: for t < t_college the next period is a college year and its
            # value is only defined on the feasible slice; at t == t_college the next
            # period is the work path, defined everywhere.
            interp = t + 1 > t_college ?
                create_interpolator(model, model.sol_v_college, t + 1) :
                create_interpolator_college(model, model.sol_v_college, t + 1, a_req)

            for i_p in 1:Np, i_k in 1:Nk, i_a in 1:Na
                assets, capital = a_grid[i_a], k_grid[i_k]

                # Economically infeasible: cannot finish college from here. Leave NaN as
                # a "not computed" marker. These cells are never interpolated -- college
                # interpolants are built over the feasible slice only -- and never
                # compared, because the discrete choice consults `college_feasible`.
                if assets < a_req[t]
                    sol_c_college[t, i_a, i_k, i_p, :] .= NaN
                    sol_h_college[t, i_a, i_k, i_p, :] .= NaN
                    sol_v_college[t, i_a, i_k, i_p, :] .= NaN
                    continue
                end

                # Consumption upper bound = the actual budget, so the box never binds and
                # the initial guess is always inside it.
                c_hi = max((1.0 + model.r) * assets - model.college_cost + model.y - a_req[t + 1],
                           c_floor + 1e-8)

                nodes = t == 1 ? (1:model.Nt) : (1:1)
                for i_t in nodes
                    eps_it = t == 1 ? model.t_grid[i_t] : 0.0

                    opt = Opt(:LD_SLSQP, 1)
                    lower_bounds!(opt, c_floor)
                    upper_bounds!(opt, c_hi)
                    ftol_rel!(opt, 1e-8)
                    maxeval!(opt, 1000)
                    min_objective!(opt, (c_vec, grad) -> begin
                        f = obj_college_period_general(model, c_vec, assets, capital, t,
                                                       i_p, interp, eps_it, grad)
                        if length(grad) > 0; grad[:] = -grad[:] end
                        return -f
                    end)
                    inequality_constraint!(opt,
                        (x, grad) -> asset_constraint_college(x, grad, model, assets, t, a_req[t + 1]),
                        1e-6)

                    (minf, c_vec, ret) = optimize(opt, [clamp(0.13, c_floor, c_hi)])
                    check_nlopt!(ret, minf, c_vec, "college t=$t ia=$i_a ik=$i_k ip=$i_p it=$i_t")

                    if t == 1
                        sol_c_college[t, i_a, i_k, i_p, i_t] = c_vec[1]
                        sol_h_college[t, i_a, i_k, i_p, i_t] = 0.0
                        sol_v_college[t, i_a, i_k, i_p, i_t] = -minf
                    else
                        sol_c_college[t, i_a, i_k, i_p, :] .= c_vec[1]
                        sol_h_college[t, i_a, i_k, i_p, :] .= 0.0
                        sol_v_college[t, i_a, i_k, i_p, :] .= -minf
                    end
                end
            end
        end
    end
    return a_req
end

# ================================
# Objectives & constraints
# ================================

# --- Final period objective: work, consume everything (progressive tax) ---
@inline function obj_last_period(model::ConSavLaborCollege_AR1, h_vec::Vector, assets::Float64,
    capital::Float64, t::Int, p_shock::Float64, grad::Vector)
    h     = h_vec[1]
    w_pre = wage_func(model, capital, t, p_shock)
    c     = assets + after_tax_income(model, w_pre, h) + model.y

    u = util_work(model, c, h)
    if length(grad) > 0
        # du/dh = u'(c) * d(after-tax income)/dh  -  phi * h^eta
        grad[1] = c^(-model.rho) * d_after_tax_dh(model, w_pre, h) - model.phi * h^model.eta
    end
    return u
end

# --- Work period objective (progressive-tax income) ---
@inline function obj_work_period(model::ConSavLaborCollege_AR1, x::Vector, assets::Float64, capital::Float64,
    t::Int, p_shock::Float64, i_p::Int, interp, grad::Vector)
    c, h = x[1], x[2]
    w_pre  = wage_func(model, capital, t, p_shock)
    y_lab  = after_tax_income(model, w_pre, h)               # λ (w h)^(1-τ)
    dy_dh  = d_after_tax_dh(model, w_pre, h)                 # λ (1-τ) w (w h)^(-τ)

    a_next = (1.0 + model.r) * assets + y_lab - c + model.y
    k_next = capital + h

    # Expectation over future persistent shocks
    V_next = 0.0
    gradV_c = 0.0
    gradV_h = 0.0
    for j_p in 1:model.Np
        p_trans_prob = model.p_transition[i_p, j_p]
        if p_trans_prob > 1e-12
            interp_jp = interp[j_p]
            Vj = interp_jp(a_next, k_next)
            gradV = Interpolations.gradient(interp_jp, a_next, k_next)
            dV_da = gradV[1]
            dV_dk = gradV[2]

            V_next  += p_trans_prob * Vj
            gradV_c += p_trans_prob * (-dV_da)                      # ∂a_next/∂c = -1
            gradV_h += p_trans_prob * (dy_dh * dV_da + dV_dk)       # ∂a_next/∂h = dy_dh, ∂k_next/∂h = 1
        end
    end

    util_now = util_work(model, c, h)
    dutil_dc = c^(-model.rho)
    dutil_dh = -model.phi * h^model.eta

    V = util_now + model.beta * V_next

    if length(grad) > 0
        grad[1] = dutil_dc + model.beta * gradV_c
        grad[2] = dutil_dh + model.beta * gradV_h
    end
    return V
end

# --- College period (unchanged functional form) ---
@inline function obj_college_period_general(
    model::ConSavLaborCollege_AR1, c_vec::Vector, assets::Float64, capital::Float64,
    t::Int, i_p::Int, interp, ε::Float64, grad::Vector
)
    c = c_vec[1]
    a_next = (1 + model.r) * assets - c - model.college_cost + model.y
    k_next = capital + model.college_boost

    V_next = 0.0
    gradV_c = 0.0
    for j_p in 1:model.Np
        p_prob = model.p_transition[i_p, j_p]
        if p_prob > 1e-12
            interp_jp = interp[j_p]
            Vj = interp_jp(a_next, k_next)
            gradV = Interpolations.gradient(interp_jp, a_next, k_next)
            dV_da = gradV[1]
            V_next  += p_prob * Vj
            gradV_c += p_prob * (-dV_da)
        end
    end

    V = util_college(model, c, capital) + (t==1 ? ε : 0.0) + model.beta * V_next

    if length(grad) > 0
        grad[1] = c^(-model.rho) + model.beta * gradV_c
    end
    return V
end

# ================================
# Constraints
# ================================
@inline function asset_constraint_work(x::Vector, grad::Vector, model::ConSavLaborCollege_AR1,
    assets::Float64, capital::Float64, t::Int, p_shock::Float64)
    c, h = x[1], x[2]
    w_pre = wage_func(model, capital, t, p_shock)
    y_lab = after_tax_income(model, w_pre, h)
    a_next = (1.0 + model.r) * assets + y_lab - c + model.y
    g = model.a_min - a_next
    if length(grad) > 0
        grad[1] = 1.0                                   # ∂g/∂c = 1
        grad[2] = -d_after_tax_dh(model, w_pre, h)      # ∂g/∂h = -∂a_next/∂h
    end
    return g
end

# a_next must be enough to FINISH college, not merely to stay above a_min. `a_req_next`
# is a_req[t+1] from compute_min_assets. Using a_min here (the old behaviour) let the
# optimizer choose states from which the remaining college years were infeasible, which
# is what made the -Inf region reachable.
@inline function asset_constraint_college(c_vec::Vector, grad::Vector, model::ConSavLaborCollege_AR1,
    assets::Float64, t::Int, a_req_next::Float64)
    c = c_vec[1]
    a_next = (1.0 + model.r) * assets - c - model.college_cost + model.y
    g = a_req_next - a_next
    if length(grad) > 0
        grad[1] = 1.0
    end
    return g
end

# ================================
# Utilities
# ================================
@inline function util_work(model::ConSavLaborCollege_AR1, c, h)
    if model.rho == 1.0
        cons_utility = log(c)
    else
        cons_utility = (c^(1.0 - model.rho)) / (1.0 - model.rho)
    end
    labor_disutility = model.phi * (h^(1.0 + model.eta)) / (1.0 + model.eta)
    return cons_utility - labor_disutility
end

@inline function util_college(model::ConSavLaborCollege_AR1, c::Float64, k::Float64)
    if model.rho == 1.0
        cons_utility = log(c)
    else
        cons_utility = (c^(1.0 - model.rho)) / (1.0 - model.rho)
    end
    psychic_cost = model.kappa / (k + 1.0)^4
    return cons_utility - psychic_cost
end

# ================================
# Interpolators
# ================================
function create_interpolator(model::ConSavLaborCollege_AR1, sol_v::Array, t::Int)
    return [
        extrapolate(
            interpolate((model.a_grid, model.k_grid), sol_v[t, :, :, i_p, 1], Gridded(Linear())),
            Line()
        )
        for i_p in 1:model.Np
    ]
end

# ================================
# College feasibility
# ================================
# Required assets to enter college year t and still be able to FINISH. Recursion,
# with the same consumption floor the optimizer enforces:
#
#     a_req[t_college+1] = a_min          (the work-path minimum on graduation)
#     a_req[t]           = (a_req[t+1] + c_floor + college_cost - y) / (1 + r)
#
# Returned with length t_college+1 so index t+1 is always defined.
function compute_min_assets(model::ConSavLaborCollege_AR1)
    @unpack t_college, r, y, college_cost, a_min, c_floor = model
    a_req = zeros(t_college + 1)
    a_req[t_college + 1] = a_min
    for t in t_college:-1:1
        a_req[t] = (a_req[t + 1] + c_floor + college_cost - y) / (1 + r)
    end
    return a_req
end

"""
    first_feasible_a(model, a_req, t)

Index of the first asset-grid point from which college year `t` is feasible, or
`nothing` if none is. College interpolants are built over `first_feasible_a(...):Na`
only, so infeasible cells never enter an interpolation.
"""
function first_feasible_a(model::ConSavLaborCollege_AR1, a_req::Vector{Float64}, t::Int)
    thr = t <= length(a_req) ? a_req[t] : model.a_min
    return findfirst(a -> a >= thr, model.a_grid)
end

# Continuation-value interpolator for the college path, restricted to the feasible
# slice of the asset grid at period `t`. Infeasible cells hold NaN and are excluded.
function create_interpolator_college(model::ConSavLaborCollege_AR1, sol_v::Array,
                                     t::Int, a_req::Vector{Float64})
    i0 = first_feasible_a(model, a_req, t)
    i0 === nothing && return nothing
    ag = model.a_grid[i0:end]
    return [
        extrapolate(interpolate((ag, model.k_grid), sol_v[t, i0:end, :, i_p, 1],
                                Gridded(Linear())), Line())
        for i_p in 1:model.Np
    ]
end

# --------------------------
# Simulation domain guard
# --------------------------
"""
    snap(x, lo, hi; tol = 1e-10)

Clamp ONLY floating-point-sized violations. A state that genuinely leaves `[lo, hi]` is
returned unchanged so that `check_simulation` can see and report it -- silently clipping an
economically meaningful out-of-grid state would rewrite the transition law.
"""
@inline function snap(x::Float64, lo::Float64, hi::Float64; tol::Float64 = 1e-10)
    x < lo && x > lo - tol && return lo
    x > hi && x < hi + tol && return hi
    return x
end

# --------------------------
# Helper for Simulation
# --------------------------
function discrete_draw(probs::AbstractVector{Float64}, draw::Float64)
    cdf = cumsum(probs)
    return findfirst(x -> x >= draw, cdf)
end

# --------------------------
# Simulation (AR1 Shock Only + Shock-free Retirement)
# --------------------------
function simulate_model_child!(model::ConSavLaborCollege_AR1)
    @unpack simN, T, t_college, r, college_cost, college_boost, a_min = model
    @unpack a_grid, k_grid, p_grid, p_transition = model
    @unpack sim_a, sim_k, sim_c, sim_h, sim_income, sim_wage = model
    @unpack sim_p_idx, sim_a_init, sim_k_init, sim_p_init_idx, draws_uniform_p, y = model
    @unpack Nt, t_weight = model

    # -- 1. Precompute interpolators --

    # College policy cells below the feasibility threshold hold NaN, so each college-year
    # interpolant is built over that year's feasible asset slice -- NaN never enters an
    # interpolation.
    a_req_sim = compute_min_assets(model)
    csl(t) = begin
        i0 = t <= t_college ? first_feasible_a(model, a_req_sim, t) : 1
        i0 === nothing ? (1:length(a_grid)) : (i0:length(a_grid))
    end

    # College (same as before)
    interp_c_college = [
        [LinearInterpolation((a_grid[csl(t)], k_grid), model.sol_c_college[t, csl(t), :, i_p, i_t]; extrapolation_bc=Flat())
            for t in 1:T, i_p in 1:model.Np]
        for i_t in 1:Nt
    ]
    interp_h_college = [
        [LinearInterpolation((a_grid[csl(t)], k_grid), model.sol_h_college[t, csl(t), :, i_p, i_t]; extrapolation_bc=Flat())
            for t in 1:T, i_p in 1:model.Np]
        for i_t in 1:Nt
    ]
    interp_v_college = [
        [LinearInterpolation((a_grid[csl(1)], k_grid), model.sol_v_college[1, csl(1), :, i_p, i_t]; extrapolation_bc=Flat())
            for i_p in 1:model.Np]
        for i_t in 1:Nt
    ]

    # Work (per shock)
    interp_c_work = [
        LinearInterpolation((a_grid, k_grid), model.sol_c_work[t, :, :, i_p, 1]; extrapolation_bc=Flat())
        for t in 1:T, i_p in 1:model.Np
    ]
    interp_h_work = [
        LinearInterpolation((a_grid, k_grid), model.sol_h_work[t, :, :, i_p, 1]; extrapolation_bc=Flat())
        for t in 1:T, i_p in 1:model.Np
    ]
    interp_v_work = [
        LinearInterpolation((a_grid, k_grid), model.sol_v_work[1, :, :, i_p, 1]; extrapolation_bc=Flat())
        for i_p in 1:model.Np
    ]

    # -- 2. Assign a taste shock node to each agent for t == 1 --
    cum_weights = cumsum(t_weight)
    rng = MersenneTwister(123)  # Reproducible
    eps_indices = [findfirst(w -> w ≥ rand(rng), cum_weights) for _ in 1:simN]

    # -- 3. Initial path choice (stochastic via taste node) --
    path_choice = Vector{Symbol}(undef, simN)
    for i in 1:simN
        a0, k0, p0_idx = sim_a_init[i], sim_k_init[i], sim_p_init_idx[i]
        i_t = eps_indices[i]
        # -Inf enters ONLY here, at the discrete comparison.
        EV_college = a0 >= a_req_sim[1] ? interp_v_college[i_t][p0_idx](a0, k0) : -Inf
        EV_work    = interp_v_work[p0_idx](a0, k0)
        path_choice[i] = EV_college > EV_work ? :college : :work
    end

    # -- 4. Initialize simulation arrays --
    sim_a[:, 1] .= sim_a_init
    sim_k[:, 1] .= sim_k_init
    sim_p_idx[:, 1] .= sim_p_init_idx

    # -- 5. Simulate forward --
    for t in 1:T
        for i in 1:simN
            a = sim_a[i, t]
            k = sim_k[i, t]
            p_idx = sim_p_idx[i, t]
            i_t = eps_indices[i]

            if path_choice[i] == :college && t <= t_college
                # ----- In college -----
                if t == 1
                    c = interp_c_college[i_t][t, p_idx](a, k)
                    h = interp_h_college[i_t][t, p_idx](a, k)
                else
                    c = interp_c_college[1][t, p_idx](a, k)   # or average across i_t if you prefer
                    h = interp_h_college[1][t, p_idx](a, k)
                end
                sim_income[i, t] = 0.0
                sim_wage[i, t]   = 0.0

            else
                # ----- Working, progressive tax -----
                c = interp_c_work[t, p_idx](a, k)
                h = interp_h_work[t, p_idx](a, k)
                p_shock = p_grid[p_idx]
                w_pre = wage_func(model, k, t, p_shock)     # pre-tax hourly wage
                sim_wage[i, t] = w_pre / WAGE_SCALING_FACTOR
                sim_income[i, t] = after_tax_income(model, w_pre, h)
            end

            sim_c[i, t] = c
            sim_h[i, t] = h

            # ----- State transitions -----
            if t < T
                if path_choice[i] == :college && t <= t_college
                    a_next = (1 + r) * a - c - college_cost + y
                    k_next = k + college_boost

                else
                    a_next = (1 + r) * a + sim_income[i, t] - c + y
                    k_next = k + h
                end

                # snap() fixes float-sized overshoot only; genuine excursions are left
                # visible for check_simulation rather than silently clipped.
                sim_a[i, t+1] = snap(a_next, a_min, model.a_max)
                sim_k[i, t+1] = snap(k_next, k_grid[1], model.k_max)

                p_draw = draws_uniform_p[i, t]
                p_trans_probs = p_transition[p_idx, :]
                sim_p_idx[i, t+1] = discrete_draw(p_trans_probs, p_draw)
            end
        end
    end

    # -- 6. Report results --
    num_college = sum(path_choice .== :college)
    println("\n--- Simulation Results for Child optimization ---")
    println("Number choosing college: $num_college ($(round(100*num_college/simN, digits=1))%)")
    println("Number choosing work:    $(simN - num_college)")

    return model, path_choice, eps_indices
end

# ========== Parent's terminal value ==========

"""
    terminal_value_surface(m) -> Matrix (Na x Nk)

The parent's period-`T` continuation value, assembled with the timing frozen in Phase 0.5:

    E_{eps_0} [ max_{d,tr} E_{z_0} [ W_d(tr; eps_0, z_0) ] ]

`eps_0` is observed at the half period and `z_0` is not, so enrolment and the transfer
condition on `eps_0` but not on realized `z_0`. The nesting is the substance -- this is
NOT `max_{d,tr} E_{eps_0,z_0}[W_d]`, which would select the transfer before the preference
shock is seen.

Building blocks already exist:
  * `sol_tr_v_college[:, :, ip, it]` = max_tr E_{z_0}[W_E | eps_it]   (eps-specific)
  * `sol_tr_v_work[:, :, ip, 1]`     = max_tr E_{z_0}[W_W]            (no eps)
`max.` is the max over `d`; the `t_weight` sum is `E_{eps_0}`.

`-Inf` is applied only here, at the discrete comparison, for states where college is
infeasible -- never stored in an interpolated array.
"""
function terminal_value_surface(m::ConSavLaborCollege_AR1; ip::Int = 1)
    a_col_min = min_parent_assets_for_college(m)
    out = zeros(m.Na, m.Nk)
    for it in 1:m.Nt
        w = m.t_weight[it]
        for ik in 1:m.Nk, ia in 1:m.Na
            vw = m.sol_tr_v_work[ia, ik, ip, 1]
            vc = m.a_grid[ia] >= a_col_min ? m.sol_tr_v_college[ia, ik, ip, it] : -Inf
            isnan(vc) && (vc = -Inf)          # NaN marks infeasible; -Inf only at the max
            if !isfinite(vw) && !isfinite(vc)
                # Neither branch is defined here. At a <= delta_P the parent cannot retain
                # its floor, a_term -> 0, and kappa_term*log(a_term) diverges: a genuine
                # singularity of the model, not a numerical artefact. Marked NaN so
                # terminal_value_spline can drop the row rather than fit through it.
                out[ia, ik] = NaN
            else
                out[ia, ik] += w * max(vc, vw)
            end
        end
    end
    return out
end

"""
    terminal_value_spline(m; s = 10.0, ip = 1)

The parent's terminal continuation value as a `Spline2D` over `(a, HC)`, built with the
Phase-0.5 timing (see `terminal_value_surface`).

The spline is fitted over the **valid** rows of the asset grid only. At parental assets
at or below `delta_P` the parent cannot retain its floor, `a_term -> 0`, and
`kappa_term * log(a_term)` diverges: that is a genuine singularity of the model, not a
numerical artefact, and the grid should not carry it. Those rows are excluded rather than
filled with a sentinel.

Returns the spline; `valid_rows(m)` gives the indices used.
"""
function terminal_value_spline(m::ConSavLaborCollege_AR1; s::Float64 = 10.0, ip::Int = 1)
    V  = terminal_value_surface(m; ip = ip)
    ok = [all(isfinite, view(V, ia, :)) for ia in 1:m.Na]
    ia0 = findfirst(ok)
    ia0 === nothing && error("Terminal value is non-finite at every asset grid point")
    all(ok[ia0:end]) || error("Terminal value has interior non-finite rows: $(findall(.!ok))")
    return Spline2D(m.a_grid[ia0:end], m.k_grid, V[ia0:end, :]; s = s)
end

"""
    valid_rows(m) -> UnitRange

Asset-grid rows over which the parent's terminal value is finite (assets above `delta_P`).
"""
function valid_rows(m::ConSavLaborCollege_AR1)
    V  = terminal_value_surface(m)
    ok = [all(isfinite, view(V, ia, :)) for ia in 1:m.Na]
    ia0 = findfirst(ok)
    return ia0 === nothing ? (1:0) : (ia0:m.Na)
end

# ========== Transfer-stage helpers ==========

"""
    maximize_1d(f, lo, hi; n_grid=48, n_refine=3)

Bounded, derivative-free maximization of a scalar `f` on `[lo, hi]`: coarse grid sweep,
then two local refinements around the incumbent.

The transfer problem is one-dimensional and its objective is built from piecewise-linear
interpolants and a feasibility boundary, so it is C0 but not C1. SLSQP assumes a smooth
objective and was previously started from different points and tolerances in the two
branches, which meant the college/work comparison was between two differently-conditioned
local optima. This is slower per state but branch-symmetric and robust at the boundary.

Non-finite objective values are skipped rather than stored.
"""
function maximize_1d(f, lo::Float64, hi::Float64; n_grid::Int=48, n_refine::Int=3)
    hi <= lo && return (lo, f(lo))
    best_x, best_f = lo, -Inf
    a, b = lo, hi
    for _ in 1:n_refine
        xs = range(a, b, length=n_grid)
        for x in xs
            v = f(x)
            if isfinite(v) && v > best_f
                best_f = v; best_x = x
            end
        end
        step = (b - a) / (n_grid - 1)
        a = max(lo, best_x - step)
        b = min(hi, best_x + step)
        b <= a && break
    end
    return best_x, best_f
end

"""
    min_parent_assets_for_college(model)

Smallest parental asset level at which the college branch is feasible: the child must
receive at least `a_req[1]` while the parent retains `delta_P`. Below this the college
optimizer is never called and the branch is declared infeasible.
"""
min_parent_assets_for_college(model::ConSavLaborCollege_AR1) =
    compute_min_assets(model)[1] + model.delta_P

"""
    first_feasible_parent_a(model)

First asset-grid index at which the college transfer branch is feasible. College transfer
interpolants are built from this index up, so infeasible cells (held as NaN) never enter
an interpolation.
"""
function first_feasible_parent_a(model::ConSavLaborCollege_AR1)
    thr = min_parent_assets_for_college(model)
    return findfirst(a -> a >= thr, model.a_grid)
end

# ========== Main Solvers ==========

function optimal_transfer_work!(model::ConSavLaborCollege_AR1)
    @unpack Na, Nk, a_grid, k_grid, p_transition, Np, delta_P = model
    π_p = stationary_dist(p_transition)

    V1_work = [extrapolate(interpolate((a_grid, k_grid), model.sol_v_work[1, :, :, ip, 1],
                                       Gridded(Linear())), Line()) for ip in 1:Np]

    for ia in 1:Na, ik in 1:Nk
        assets = a_grid[ia]
        HC     = k_grid[ik]

        # Work path admits a zero transfer, so the domain is [0, a - delta_P]. It is
        # never infeasible: if the parent holds less than delta_P the domain collapses
        # to {0} and the parent simply transfers nothing.
        tr_lo, tr_hi = 0.0, max(0.0, assets - delta_P)

        f(tr) = obj_transfer_work(model, tr, assets, HC, Float64[], V1_work, Np, π_p)
        (tr_opt, v_opt) = maximize_1d(f, tr_lo, tr_hi)

        model.sol_tr_work[ia, ik, :, :]   .= tr_opt
        model.sol_tr_v_work[ia, ik, :, :] .= v_opt
    end
    return nothing
end

function optimal_transfer_college!(model::ConSavLaborCollege_AR1)
    @unpack Na, Nk, Nt, a_grid, k_grid, p_transition, Np, delta_P = model
    π_p   = stationary_dist(p_transition)
    a_req = compute_min_assets(model)

    # sol_v_college holds NaN where college is infeasible, so build over the feasible
    # slice only. The transfer lower bound is a_req[1], so the interpolant is never
    # evaluated below it.
    i1 = first_feasible_a(model, a_req, 1)
    i1 === nothing && error("College infeasible at every asset grid point")
    ag1 = a_grid[i1:end]
    V1_college = [[extrapolate(interpolate((ag1, k_grid), model.sol_v_college[1, i1:end, :, ip, it],
                                           Gridded(Linear())), Line()) for it in 1:Nt] for ip in 1:Np]

    for ia in 1:Na, ik in 1:Nk, it in 1:Nt
        assets = a_grid[ia]
        HC     = k_grid[ik]

        # College requires the child to start with at least a_req[1] -- enough to finish
        # all t_college years -- while the parent retains delta_P.
        tr_lo, tr_hi = a_req[1], assets - delta_P
        if tr_hi < tr_lo
            # Economically infeasible. Do NOT call the optimizer, and do NOT write -Inf
            # into an array that gets interpolated: NaN marks "not computed", and the
            # discrete choice applies -Inf at the point of comparison instead.
            model.sol_tr_college[ia, ik, :, it]   .= NaN
            model.sol_tr_v_college[ia, ik, :, it] .= NaN
            continue
        end

        f(tr) = obj_transfer_college(model, tr, assets, HC, Float64[], V1_college, it, Np, π_p)
        (tr_opt, v_opt) = maximize_1d(f, tr_lo, tr_hi)   # same grid/tolerances as work

        model.sol_tr_college[ia, ik, :, it]   .= tr_opt
        model.sol_tr_v_college[ia, ik, :, it] .= v_opt
    end
    return nothing
end

# ======== Objective: Work Path ==========
function obj_transfer_work(
    model::ConSavLaborCollege_AR1, tr::Float64, assets::Float64, HC::Float64,
    grad::Vector, V1_work::Vector, Np::Int, π_p::Vector
)
    a_terminal = assets - tr

    if a_terminal <= 0.0
        if length(grad) > 0
            grad[1] = 0.0
        end
        # NaN marks infeasible, consistently with the rest of the module. -1e12 was a
        # finite sentinel that survived every finiteness check and leaked into the
        # parent's terminal value as a real number.
        return NaN
    end

    V_parent = terminal_value(model, HC, a_terminal)
    # Expectation over AR1 shock
    V_child = 0.0
    dV_child_dtr = 0.0
    for ip in 1:Np
        interp = V1_work[ip]
        Vj = interp(tr, HC)
        gradV = Interpolations.gradient(interp, tr, HC)
        dV_child_dtr += π_p[ip] * gradV[1]
        V_child      += π_p[ip] * Vj
    end

    coef = (1-model.mu) + model.mu*model.omega
    f = coef * V_child + model.mu * V_parent

    if length(grad) > 0
        dV_parent_dtr = -model.kappa_terminal / (a_terminal)
        grad[1] = coef * dV_child_dtr + model.mu * dV_parent_dtr
    end
    return f
end

# ========== Objective: College Path ==========
function obj_transfer_college(
    model::ConSavLaborCollege_AR1, tr::Float64, assets::Float64, HC::Float64,
    grad::Vector, V1_college::Vector, it::Int, Np::Int, π_p::Vector
)
    a_terminal = assets - tr

    if a_terminal <= 0.0
        if length(grad) > 0
            grad[1] = 0.0
        end
        # NaN marks infeasible, consistently with the rest of the module. -1e12 was a
        # finite sentinel that survived every finiteness check and leaked into the
        # parent's terminal value as a real number.
        return NaN
    end

    V_parent = terminal_value(model, HC, a_terminal)
    # Expectation over AR1 shock
    V_child = 0.0
    dV_child_dtr = 0.0
    for ip in 1:Np
        interp = V1_college[ip][it]
        Vj = interp(tr, HC)
        gradV = Interpolations.gradient(interp, tr, HC)
        dV_child_dtr += π_p[ip] * gradV[1]
        V_child      += π_p[ip] * Vj
    end

    coef = (1-model.mu) + model.mu*model.omega
    f = coef * V_child + model.mu * V_parent

    if length(grad) > 0
        dV_parent_dtr = -model.kappa_terminal / (a_terminal)
        grad[1] = coef * dV_child_dtr + model.mu * dV_parent_dtr
    end
    return f
end

# ========== Terminal Value ==========
@inline function terminal_value(model::ConSavLaborCollege_AR1, k::Float64, a_terminal::Float64)
    @unpack psi_terminal, kappa_terminal = model
    return psi_terminal * log(k) + kappa_terminal * log(a_terminal)
end
# ==


# ========== Stationary Distribution Helper ==========
function stationary_dist(P)
    vals, vecs = eigen(P')
    π = vec(real(vecs[:, argmax(real(vals))]))
    π .*= sign(π[1])
    π ./= sum(π)
    return π
end





# --------------------------
# Simulation (AR1 Shock Only + Family Optimization)
# --------------------------
function simulate_model_family!(model::ConSavLaborCollege_AR1)
    @unpack simN, T, t_college, r, college_cost, college_boost, a_min = model
    @unpack a_grid, k_grid, p_grid, p_transition, Np = model
    @unpack sim_a, sim_k, sim_c, sim_h, sim_income, sim_wage = model
    @unpack sim_p_idx, sim_a_init, sim_k_init, sim_p_init_idx, draws_uniform_p, y = model
    @unpack Nt, t_weight = model

    # -- 1. Precompute interpolators for policies and transfer values --
    # College policy cells below the feasibility threshold hold NaN, so each college-year
    # interpolant is built over that year's feasible asset slice -- NaN never enters an
    # interpolation.
    a_req_sim = compute_min_assets(model)
    csl(t) = begin
        i0 = t <= t_college ? first_feasible_a(model, a_req_sim, t) : 1
        i0 === nothing ? (1:length(a_grid)) : (i0:length(a_grid))
    end

    # College policy interpolators
    interp_c_college = [
        [LinearInterpolation((a_grid[csl(t)], k_grid), model.sol_c_college[t, csl(t), :, i_p, i_t]; extrapolation_bc=Flat())
            for t in 1:T, i_p in 1:Np]
        for i_t in 1:Nt
    ]
    interp_h_college = [
        [LinearInterpolation((a_grid[csl(t)], k_grid), model.sol_h_college[t, csl(t), :, i_p, i_t]; extrapolation_bc=Flat())
            for t in 1:T, i_p in 1:Np]
        for i_t in 1:Nt
    ]
    # Work policy interpolators
    interp_c_work = [
        LinearInterpolation((a_grid, k_grid), model.sol_c_work[t, :, :, i_p, 1]; extrapolation_bc=Flat())
        for t in 1:T, i_p in 1:Np
    ]
    interp_h_work = [
        LinearInterpolation((a_grid, k_grid), model.sol_h_work[t, :, :, i_p, 1]; extrapolation_bc=Flat())
        for t in 1:T, i_p in 1:Np
    ]
    # Transfer value interpolators
    sol_tr_v_college_interp = [
        [LinearInterpolation((a_grid, k_grid), model.sol_tr_v_college[:, :, ip, it]; extrapolation_bc=Line())
            for ip in 1:Np]
        for it in 1:Nt
    ]
    sol_tr_v_work_interp = [
        LinearInterpolation((a_grid, k_grid), model.sol_tr_v_work[:, :, ip, 1]; extrapolation_bc=Line())
        for ip in 1:Np
    ]
    # Transfer amount interpolators
    sol_tr_college_interp = [
        [LinearInterpolation((a_grid, k_grid), model.sol_tr_college[:, :, ip, it]; extrapolation_bc=Flat())
            for ip in 1:Np]
        for it in 1:Nt
    ]
    sol_tr_work_interp = [
        LinearInterpolation((a_grid, k_grid), model.sol_tr_work[:, :, ip, 1]; extrapolation_bc=Flat())
        for ip in 1:Np
    ]

    # -- 2. Assign a taste shock node to each agent for t == 1 --
    cum_weights = cumsum(t_weight)
    rng = MersenneTwister(123)  # Reproducible
    eps_indices = [findfirst(w -> w ≥ rand(rng), cum_weights) for _ in 1:simN]

    # -- 3. Initial path choice based on parent's transfer decision --
    path_choice = Vector{Symbol}(undef, simN)
    tr_initial = Vector{Float64}(undef, simN)
    for i in 1:simN
        parent_assets = sim_a_init[i]  # Parent's assets
        HC = sim_k_init[i]             # Child's initial human capital
        ip = sim_p_init_idx[i]         # Persistent shock index
        it = eps_indices[i]            # Taste shock index

        # Compute parent's value for each path
        f_college = sol_tr_v_college_interp[it][ip](parent_assets, HC)
        f_work = sol_tr_v_work_interp[ip](parent_assets, HC)

        # Choose path and set transfer
        if f_college > f_work
            path_choice[i] = :college
            tr = sol_tr_college_interp[it][ip](parent_assets, HC)
        else
            path_choice[i] = :work
            tr = sol_tr_work_interp[ip](parent_assets, HC)
        end
        tr_initial[i] = tr  # Ensure non-negative transfer
    end

    # -- 4. Initialize simulation arrays with transfer as initial asset --
    sim_a[:, 1] .= tr_initial
    sim_k[:, 1] .= sim_k_init
    sim_p_idx[:, 1] .= sim_p_init_idx

    # -- 5. Simulate forward --
    @showprogress "Simulating..." for t in 1:T
        for i in 1:simN
            a = sim_a[i, t]
            k = sim_k[i, t]
            p_idx = sim_p_idx[i, t]
            i_t = eps_indices[i]

            if path_choice[i] == :college && t <= t_college
                # ----- In college -----
                if t == 1
                    c = interp_c_college[i_t][t, p_idx](a, k)
                    h = interp_h_college[i_t][t, p_idx](a, k)
                else
                    c = interp_c_college[1][t, p_idx](a, k)  # Use mean taste shock
                    h = interp_h_college[1][t, p_idx](a, k)
                end
                sim_income[i, t] = 0.0
                sim_wage[i, t] = 0.0
            else
                # ----- Working -----
                c = interp_c_work[t, p_idx](a, k)
                h = interp_h_work[t, p_idx](a, k)
                p_shock = p_grid[p_idx]
                w_pre = wage_func(model, k, t, p_shock)  # Pre-tax hourly wage
                sim_wage[i, t] = w_pre / WAGE_SCALING_FACTOR
                sim_income[i, t] = after_tax_income(model, w_pre, h)
            end

            sim_c[i, t] = c
            sim_h[i, t] = h

            # ----- State transitions -----
            if t < T
                if path_choice[i] == :college && t <= t_college
                    a_next = (1 + r) * a - c - college_cost + y
                    k_next = k + college_boost
                else
                    a_next = (1 + r) * a + sim_income[i, t] - c + y
                    k_next = k + h
                end
                # snap() fixes float-sized overshoot only; genuine excursions are left
                # visible for check_simulation rather than silently clipped.
                sim_a[i, t+1] = snap(a_next, a_min, model.a_max)
                sim_k[i, t+1] = snap(k_next, k_grid[1], model.k_max)

                # Transition for persistent shock
                p_draw = draws_uniform_p[i, t]
                p_trans_probs = p_transition[p_idx, :]
                sim_p_idx[i, t+1] = discrete_draw(p_trans_probs, p_draw)
            end
        end
    end

    # -- 6. Report results --
    num_college = sum(path_choice .== :college)
    println("\n--- Simulation Results for Family optimization ---")
    println("Number choosing college: $num_college ($(round(100*num_college/simN, digits=1))%)")
    println("Number choosing work:    $(simN - num_college)")

    return model, path_choice, eps_indices
end




