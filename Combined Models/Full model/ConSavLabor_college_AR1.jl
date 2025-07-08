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
# Dynamic Labor Model with College Decision and Stochastic Shocks
# =============================================================================
mutable struct ConSavLaborCollege_AR1
    T::Int; t_college::Int; rho::Float64; beta::Float64; phi::Float64
    eta::Float64; alpha::Float64; y::Float64; w::Float64; tau::Float64
    r::Float64; a_max::Float64; a_min::Float64; Na::Int; k_max::Float64
    Nk::Int; simT::Int; simN::Int
    a_grid::Vector{Float64}; k_grid::Vector{Float64}

    # --- Stochastic Shock Parameters ---
    Nt::Int; t_grid::Vector{Float64}; sigma_eps::Float64; t_weight::Vector{Float64}
    Np::Int; p_grid::Vector{Float64}; p_transition::Matrix{Float64}
    p_ar1::Float64; sigma_p::Float64

    # --- Solution arrays (now 5D to include shocks) ---
    sol_c_work::Array{Float64, 5}; sol_h_work::Array{Float64, 5}; sol_v_work::Array{Float64, 5}
    sol_c_college::Array{Float64, 5}; sol_h_college::Array{Float64, 5}; sol_v_college::Array{Float64, 5}

    # --- Simulation arrays ---
    sim_c::Matrix{Float64}; sim_h::Matrix{Float64}; sim_a::Matrix{Float64}
    sim_k::Matrix{Float64}; sim_t_idx::Matrix{Int}; sim_p_idx::Matrix{Int}
    sim_a_init::Vector{Float64}; sim_k_init::Vector{Float64}; sim_p_init_idx::Vector{Int}
    sim_income::Matrix{Float64}; sim_wage::Matrix{Float64}
    draws_uniform_t::Matrix{Float64}; draws_uniform_p::Matrix{Float64}
    w_vec::Vector{Float64}; college_cost::Float64; college_boost::Float64
    kappa::Float64 # parameter for psychic cost
end

# =============================================================================
# Constructor for ConSavLaborCollege_AR1 with Shocks
# =============================================================================
function ConSavLaborCollege_AR1(;
    T::Int=50, t_college::Int=4, beta::Float64=0.97, rho::Float64=1.0,
    r::Float64=0.03, a_max::Float64=20.0, Na::Int=30, y::Float64=0.6,
    simN::Int=5000, a_min::Float64=0.0, k_max::Float64=30.0, Nk::Int=30,
    w::Float64=12.5, tau::Float64=0.25, eta::Float64=2.0, alpha::Float64=0.08,
    phi::Float64=20.0, seed::Int=1234, college_cost::Float64=1.2,
    college_boost::Float64=2.0, kappa::Float64=0.5,
    # Shock parameters
    sigma_eps::Float64=0.3, Nt::Int=3,
    p_ar1::Float64=0.9, sigma_p::Float64=0.1, Np::Int=5)

    simT = T
    a_grid = create_focused_grid(a_min, 8.0, a_max, Na, 0.7, 1.1)
    k_grid = nonlinspace(0.0, k_max, Nk, 1.5)

    # --- Setup Shocks ---
    nodes, weights = gausshermite(Nt)
    mu_t = -0.5 * sigma_eps^2
    t_grid = exp.(mu_t .+ sigma_eps .* nodes)
    t_weight = weights / sum(weights)

    mc = tauchen(Np, p_ar1, sigma_p, 0.0, 3)
    p_grid = exp.(mc.state_values)
    p_transition = mc.p

    # --- Initialize solution arrays (5D) ---
    sol_shape = (T, Na, Nk, Nt, Np)
    sol_c_work = fill(NaN, sol_shape); sol_h_work = fill(NaN, sol_shape); sol_v_work = fill(NaN, sol_shape)
    sol_c_college = fill(NaN, sol_shape); sol_h_college = fill(NaN, sol_shape); sol_v_college = fill(NaN, sol_shape)

    # --- Initialize simulation arrays ---
    sim_shape = (simN, T)
    sim_c = fill(NaN, sim_shape); sim_h = fill(NaN, sim_shape)
    sim_a = fill(NaN, sim_shape); sim_k = fill(NaN, sim_shape)
    sim_t_idx = fill(0, sim_shape); sim_p_idx = fill(0, sim_shape)
    sim_income = fill(NaN, sim_shape); sim_wage = fill(NaN, sim_shape)

    rng = MersenneTwister(seed)
    sim_a_init = rand(rng, simN) .* 10
    sim_k_init = zeros(Float64, simN)
    sim_p_init_idx = fill(ceil(Int, Np/2), simN) # Start at median persistent shock

    draws_uniform_t = rand(rng, sim_shape...); draws_uniform_p = rand(rng, sim_shape...)
    w_vec = fill(w, T)

    return ConSavLaborCollege_AR1(
        T, t_college, rho, beta, phi, eta, alpha, y, w, tau, r,
        a_max, a_min, Na, k_max, Nk, simT, simN, a_grid, k_grid,
        Nt, t_grid, sigma_eps, t_weight, Np, p_grid, p_transition, p_ar1, sigma_p,
        sol_c_work, sol_h_work, sol_v_work, sol_c_college, sol_h_college, sol_v_college,
        sim_c, sim_h, sim_a, sim_k, sim_t_idx, sim_p_idx,
        sim_a_init, sim_k_init, sim_p_init_idx, sim_income, sim_wage,
        draws_uniform_t, draws_uniform_p, w_vec, college_cost, college_boost, kappa
    )
end



# ---------------------------------
# Model Solver for "Work" Path
# ---------------------------------
function solve_model_work!(model::ConSavLaborCollege_AR1)
    @unpack T, Na, Nk, Nt, Np, a_grid, k_grid, t_grid, p_grid = model
    @unpack sol_c_work, sol_h_work, sol_v_work = model

    # --- Final period (t = T) ---
    for i_p in 1:Np, i_t in 1:Nt, i_k in 1:Nk, i_a in 1:Na
        assets, capital = a_grid[i_a], k_grid[i_k]
        t_shock, p_shock = t_grid[i_t], p_grid[i_p]

        function obj_wrapper(h_vec::Vector, grad::Vector)
            f = obj_last_period(model, h_vec, assets, capital, T, t_shock, p_shock, grad)
            if length(grad) > 0
                grad[:] = -grad[:] # Negate for minimization
            end
            return -f # Minimize negative utility
        end
        opt = Opt(:LD_SLSQP, 1)
        lower_bounds!(opt, [1e-3])
        upper_bounds!(opt, [1.0])
        ftol_rel!(opt, 1e-8)
        min_objective!(opt, obj_wrapper)
        init = [0.3]
        (minf, h_vec, ret) = optimize(opt, init)

        h_opt = h_vec[1]
        cons = assets + wage_func(model, capital, T, t_shock, p_shock) * h_opt + model.y
        sol_h_work[T, i_a, i_k, i_t, i_p] = h_opt
        sol_c_work[T, i_a, i_k, i_t, i_p] = cons
        sol_v_work[T, i_a, i_k, i_t, i_p] = -minf
    end

    # --- Earlier periods (t = T-1 to 1) ---
    @showprogress 1 "Solving working model..." for t in (T-1):-1:1
        interp = create_interpolator(model, sol_v_work, t + 1)
        for i_p in 1:Np, i_t in 1:Nt, i_k in 1:Nk, i_a in 1:Na
            assets, capital = a_grid[i_a], k_grid[i_k]
            t_shock, p_shock = t_grid[i_t], p_grid[i_p]

            function obj_wrapper(x::Vector, grad::Vector)
                f = obj_work_period(model, x, assets, capital, t, t_shock, p_shock, i_p, interp, grad)
                if length(grad) > 0
                    grad[:] = -grad[:] # Negate for minimization
                end
                return -f # Minimize negative value function
            end            
            opt = Opt(:LD_SLSQP, 2)
            lower_bounds!(opt, [0.01, 1e-3])
            upper_bounds!(opt, [30, 1.0])
            ftol_rel!(opt, 1e-8)
            maxeval!(opt, 1000)
            inequality_constraint!(opt, (x, grad) -> asset_constraint_work(x, grad, model, assets, capital, t, t_shock, p_shock), 0.0)
            min_objective!(opt, obj_wrapper)
            # A good initial guess can be the solution from the previous period
            init = [max(sol_c_work[t + 1, i_a, i_k, i_t, i_p], 1e-6), sol_h_work[t + 1, i_a, i_k, i_t, i_p]]
            (minf, x_opt, ret) = optimize(opt, init)
            sol_c_work[t, i_a, i_k, i_t, i_p] = x_opt[1]
            sol_h_work[t, i_a, i_k, i_t, i_p] = x_opt[2]
            sol_v_work[t, i_a, i_k, i_t, i_p] = -minf
        end
    end
end

# ---------------------------------
# Model Solver for "College" Path
# ---------------------------------
function solve_model_college!(model::ConSavLaborCollege_AR1)
    @unpack T, t_college, Na, Nk, Np, Nt, a_grid, k_grid, p_grid, t_grid = model
    @unpack sol_c_college, sol_h_college, sol_v_college, sol_v_work = model
     # 1. Pre-calculate the minimum required assets for each college year
    a_min_t = compute_min_assets(model)

    @showprogress 1 "Solving college model..." for t in T:-1:1
        if t == T
            for i_p in 1:Np, i_t in 1:Nt, i_k in 1:Nk, i_a in 1:Na
                assets, capital = a_grid[i_a], k_grid[i_k]
                t_shock, p_shock = t_grid[i_t], p_grid[i_p]

                function obj_wrapper(h_vec::Vector, grad::Vector)
                    f = obj_last_period(model, h_vec, assets, capital, T, t_shock, p_shock, grad)
                    if length(grad) > 0
                        grad[:] = -grad[:] # Negate for minimization
                    end
                    return -f # Minimize negative utility
                end
                opt = Opt(:LD_SLSQP, 1)
                lower_bounds!(opt, [1e-3])
                upper_bounds!(opt, [1.0])
                ftol_rel!(opt, 1e-8)
                min_objective!(opt, obj_wrapper)
                init = [0.3]
                (minf, h_vec, ret) = optimize(opt, init)

                h_opt = h_vec[1]
                cons = assets + wage_func(model, capital, T, t_shock, p_shock) * h_opt + model.y
                sol_h_college[T, i_a, i_k, i_t, i_p] = h_opt
                sol_c_college[T, i_a, i_k, i_t, i_p] = cons
                sol_v_college[T, i_a, i_k, i_t, i_p] = -minf
            end
        
        elseif t > t_college
            interp = create_interpolator(model, sol_v_work, t + 1)
            for i_p in 1:Np, i_t in 1:Nt, i_k in 1:Nk, i_a in 1:Na
                assets, capital = a_grid[i_a], k_grid[i_k]
                t_shock, p_shock = t_grid[i_t], p_grid[i_p]

                function obj_wrapper(x::Vector, grad::Vector)
                    f = obj_work_period(model, x, assets, capital, t, t_shock, p_shock, i_p, interp, grad)
                    if length(grad) > 0
                        grad[:] = -grad[:] # Negate for minimization
                    end
                    return -f # Minimize negative value function
                end            
                opt = Opt(:LD_SLSQP, 2)
                lower_bounds!(opt, [0.01, 1e-3])
                upper_bounds!(opt, [30, 1.0])
                ftol_rel!(opt, 1e-8)
                maxeval!(opt, 1000)
                inequality_constraint!(opt, (x, grad) -> asset_constraint_work(x, grad, model, assets, capital, t, t_shock, p_shock), 0.0)
                min_objective!(opt, obj_wrapper)
                # A good initial guess can be the solution from the previous period
                init = [max(sol_c_college[t + 1, i_a, i_k, i_t, i_p], 1e-6), sol_h_college[t + 1, i_a, i_k, i_t, i_p]]
                (minf, x_opt, ret) = optimize(opt, init)
                sol_c_college[t, i_a, i_k, i_t, i_p] = x_opt[1]
                sol_h_college[t, i_a, i_k, i_t, i_p] = x_opt[2]
                sol_v_college[t, i_a, i_k, i_t, i_p] = -minf
            end

        else # During college periods
            # Continuation value comes from college path, except for last year
            V_cont = (t == t_college) ? sol_v_work : sol_v_college
            interp = create_interpolator(model, V_cont, t + 1)

            for i_p in 1:Np, i_k in 1:Nk, i_a in 1:Na
                assets, capital = a_grid[i_a], k_grid[i_k]

                # --- ROBUST FEASIBILITY CHECK ---
                # Check if current assets are sufficient for the entire remaining college duration.
                if assets < a_min_t[t]
                    # This state is infeasible. Set a very low value and skip optimization.
                    for i_t in 1:model.Nt
                        sol_c_college[t, i_a, i_k, i_t, i_p] = NaN
                        sol_h_college[t, i_a, i_k, i_t, i_p] = NaN
                        sol_v_college[t, i_a, i_k, i_t, i_p] = -1e10
                    end
                    continue # Move to the next state
                end

                # --- OPTIMIZATION (only for feasible states) ---
                function obj_wrapper(c_vec::Vector, grad::Vector)
                    f = obj_college_period(model, c_vec, assets, capital, t, i_p, interp, grad)
                    if length(grad) > 0
                        grad[:] = -grad[:]
                    end
                    return -f
                end
                # The upper bound for consumption is now dictated by the feasibility check
                #max_c = (1.0 + model.r) * assets + model.y - model.college_cost - model.a_min
                init = [0.13]

                opt = Opt(:LD_SLSQP, 1)
                lower_bounds!(opt, 0.01)
                upper_bounds!(opt, 20.0)
                ftol_rel!(opt, 1e-8)
                maxeval!(opt, 1000)
                min_objective!(opt, obj_wrapper)
                inequality_constraint!(opt, (x, grad) -> asset_constraint_college(x, grad, model, assets, t), 1e-6)
                (minf, c_vec, ret) = optimize(opt, init)

                    for i_t in 1:model.Nt
                        sol_c_college[t, i_a, i_k, i_t, i_p] = c_vec[1]
                        sol_h_college[t, i_a, i_k, i_t, i_p] = 0.0
                        sol_v_college[t, i_a, i_k, i_t, i_p] = -minf
                    end
            end
        end
    end
end
# ------------------------------------------------
# Objective and Constraint Functions
# ------------------------------------------------

# --- Last Period (Working) ---
@inline function obj_last_period(model::ConSavLaborCollege_AR1,h_vec::Vector,assets::Float64,
    capital::Float64, t::Int, t_shock::Float64, p_shock::Float64, grad::Vector
)
    h = h_vec[1]
    w_eff = wage_func(model, capital, t, t_shock, p_shock)
    c = assets + w_eff * h + model.y

    u = util_work(model, c, h)
    du_dc = c^(-model.rho)
    du_dh = -model.phi * h^model.eta
    du_dh_total = w_eff * du_dc + du_dh

    if length(grad) > 0
        grad[1] = du_dh_total
    end
    return u
end



# --- Work Period ---
@inline function obj_work_period(model::ConSavLaborCollege_AR1, x::Vector, assets::Float64, capital::Float64,
    t::Int, t_shock::Float64, p_shock::Float64, i_p::Int, interp, grad::Vector
)
    c, h = x[1], x[2]
    w_eff = wage_func(model, capital, t, t_shock, p_shock)
    a_next = (1.0 + model.r) * assets + w_eff * h - c + model.y
    k_next = capital + h

    # Expected value over future shocks (transitory and persistent)
    V_next = 0.0
    gradV_c = 0.0
    gradV_h = 0.0

    # Loop over next period's persistent shocks
    for j_p in 1:model.Np
        p_trans_prob = model.p_transition[i_p, j_p]
        if p_trans_prob > 1e-12 # Optimization for sparse transition matrices
            EV_t = 0.0
            gradEV_t_c = 0.0
            gradEV_t_h = 0.0
            # Loop over next period's transitory shocks
            for j_t in 1:model.Nt
                interp_jt_jp = interp[j_t, j_p]
                t_weight = model.t_weight[j_t]
                Vj = interp_jt_jp(a_next, k_next)
                gradV = Interpolations.gradient(interp_jt_jp, a_next, k_next)
                dV_da = gradV[1]
                dV_dk = gradV[2]

                EV_t += t_weight * Vj
                gradEV_t_c += t_weight * (-dV_da)
                gradEV_t_h += t_weight * (w_eff * dV_da + dV_dk)
            end
            V_next += p_trans_prob * EV_t
            gradV_c += p_trans_prob * gradEV_t_c
            gradV_h += p_trans_prob * gradEV_t_h
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


# --- College Period ---
@inline function obj_college_period(model::ConSavLaborCollege_AR1, c_vec::Vector, assets::Float64,
    capital::Float64, t::Int, i_p::Int, interp, grad::Vector)

    c = c_vec[1]
    a_next = (1 + model.r) * assets - c - model.college_cost + model.y
    k_next = capital + model.college_boost

    # Expected value over future shocks (persistent and transitory)
    V_next = 0.0
    gradV_c = 0.0
    for j_p in 1:model.Np
        p_prob = model.p_transition[i_p, j_p]
        if p_prob > 1e-12
            EV_t, gradEV_t_c = 0.0, 0.0
            for j_t in 1:model.Nt
                interp_jt_jp = interp[j_t, j_p]
                t_w = model.t_weight[j_t]
                gradV = Interpolations.gradient(interp_jt_jp, a_next, k_next)
                dV_da = gradV[1]
                EV_t += t_w * interp_jt_jp(a_next, k_next)
                gradEV_t_c += t_w * (-dV_da)
            end
            V_next += p_prob * EV_t
            gradV_c += p_prob * gradEV_t_c
        end
    end

    V = util_college(model, c, capital) + model.beta * V_next
    if length(grad) > 0
        grad[1] = c^(-model.rho) + model.beta * gradV_c
    end
    return V
end

# --- Constraints ---
@inline function asset_constraint_work(x::Vector, grad::Vector, model::ConSavLaborCollege_AR1,
     assets::Float64, capital::Float64, t::Int, t_shock::Float64, p_shock::Float64)
    c, h = x[1], x[2]
    w = wage_func(model, capital, t, t_shock, p_shock)
    a_next = (1.0 + model.r) * assets + w * h - c + model.y
    g = model.a_min - a_next  # g <= 0 ensures a_next >= a_min
    if length(grad) > 0
        grad[1] = 1.0  # ∂g/∂c
        grad[2] = -w   # ∂g/∂h
    end
    return g
end

@inline function asset_constraint_college(c_vec::Vector, grad::Vector, model::ConSavLaborCollege_AR1,
     assets::Float64, t::Int)
    c = c_vec[1]
    a_next = (1.0 + model.r) * assets - c - model.college_cost + model.y
    g = model.a_min - a_next  # g <= 0 ensures a_next >= a_min
    if length(grad) > 0
        grad[1] = 1.0  # ∂g/∂c
    end
    return g
end

# ------------------------------------------------
# Supporting Functions
# ------------------------------------------------
@inline function util_work(model::ConSavLaborCollege_AR1, c, h)
    if model.rho == 1.0
        cons_utility = log(c)
    else
        cons_utility = (c^(1.0 - model.rho)) / (1.0 - model.rho)
    end
    labor_disutility = model.phi * (h^(1.0 + model.eta)) / (1.0 + model.eta)
    return cons_utility - labor_disutility
end

# --- Utility during College ---
@inline function util_college(model::ConSavLaborCollege_AR1, c::Float64, k::Float64)
    if model.rho == 1.0
        cons_utility = log(c)
    else
        cons_utility = (c^(1.0 - model.rho)) / (1.0 - model.rho)
    end
    
    
    psychic_cost = model.kappa * log(k + 1.0)
    
    return cons_utility - psychic_cost
end

@inline function wage_func(model::ConSavLaborCollege_AR1, k::Float64, t::Int, t_shock::Float64, p_shock::Float64)
    base_wage = model.w_vec[t] * (1 + model.alpha * k)
    return (1 - model.tau) * base_wage * p_shock * t_shock * 0.584
end

function create_interpolator(model::ConSavLaborCollege_AR1, sol_v::Array, t::Int)
    return [
        LinearInterpolation((model.a_grid, model.k_grid), sol_v[t, :, :, i_t, i_p], extrapolation_bc=Line())
        for i_t in 1:model.Nt, i_p in 1:model.Np
    ]
end


"""
This function calculates the minimum asset level required at the start of each
college year (t) to be able to afford college costs and minimum consumption
for all remaining college years without violating the borrowing constraint.
It solves backwards from the last year of college.
"""
function compute_min_assets(model::ConSavLaborCollege_AR1)
    @unpack t_college, r, y, college_cost, a_min = model
    c_min = 0.15  # Minimum consumption threshold

    a_min_t = zeros(t_college)

    # For the last college period (t = t_college), you need enough assets to
    # pay for costs and min consumption, and still end up with a_min.
    a_min_t[t_college] = (a_min + c_min + college_cost - y) / (1 + r)

    # For all earlier periods, you need enough assets to pay for today's costs
    # and end up with the minimum required for the *next* period.
    for t in (t_college-1):-1:1
        a_min_t[t] = (a_min_t[t+1] + c_min + college_cost - y) / (1 + r)
    end

    return a_min_t
end


# --------------------------
# Helper for Simulation
# --------------------------
"""
Draws a discrete outcome from a vector of probabilities.
"""
function discrete_draw(probs::AbstractVector{Float64}, draw::Float64)
    cdf = cumsum(probs)
    return findfirst(x -> x >= draw, cdf)
end

# --------------------------
# Simulation (Stochastic)
# --------------------------
function simulate_model!(model::ConSavLaborCollege_AR1)
    # ------------------------------------------------
    # 1) Basic setup and parameter extraction
    # ------------------------------------------------
    @unpack simN, T, t_college, r, college_cost, college_boost, a_min = model
    @unpack a_grid, k_grid, t_grid, p_grid, t_weight, p_transition = model
    @unpack sim_a, sim_k, sim_c, sim_h, sim_income, sim_wage = model
    @unpack sim_t_idx, sim_p_idx, sim_a_init, sim_k_init, sim_p_init_idx = model
    @unpack draws_uniform_t, draws_uniform_p, y = model

    # ------------------------------------------------
    # 2) Precompute policy and value function interpolators
    # ------------------------------------------------
    # Create multi-dimensional interpolators: [Time, Transitory Idx, Persistent Idx]
    interp_c_work = [
        LinearInterpolation((a_grid, k_grid), model.sol_c_work[t, :, :, i_t, i_p]; extrapolation_bc=Flat())
        for t in 1:T, i_t in 1:model.Nt, i_p in 1:model.Np
    ]
    interp_h_work = [LinearInterpolation((a_grid, k_grid), model.sol_h_work[t, :, :, i_t, i_p]; extrapolation_bc=Flat())
        for t in 1:T, i_t in 1:model.Nt, i_p in 1:model.Np
    ]
    interp_c_college = [LinearInterpolation((a_grid, k_grid), model.sol_c_college[t, :, :, i_t, i_p]; extrapolation_bc=Flat())
        for t in 1:T, i_t in 1:model.Nt, i_p in 1:model.Np
    ]

    # Value functions are only needed for the initial decision at t=1
    interp_v_work = [LinearInterpolation((a_grid, k_grid), model.sol_v_work[1, :, :, i_t, i_p]; extrapolation_bc=Flat())
        for i_t in 1:model.Nt, i_p in 1:model.Np
    ]
    interp_v_college = [LinearInterpolation((a_grid, k_grid), model.sol_v_college[1, :, :, i_t, i_p]; extrapolation_bc=Flat())
        for i_t in 1:model.Nt, i_p in 1:model.Np
    ]

    # ------------------------------------------------
    # 3) Compute initial values & choose college vs. work
    # ------------------------------------------------
    path_choice = Vector{Symbol}(undef, simN)
    for i in 1:simN
        a0, k0, p0_idx = sim_a_init[i], sim_k_init[i], sim_p_init_idx[i]

        # Calculate the EXPECTED value of each path over the initial transitory shock distribution
        EV_college = sum(t_weight[i_t] * interp_v_college[i_t, p0_idx](a0, k0) for i_t in 1:model.Nt)
        EV_work    = sum(t_weight[i_t] * interp_v_work[i_t, p0_idx](a0, k0) for i_t in 1:model.Nt)
        
        path_choice[i] = EV_college > EV_work ? :college : :work
    end

    # ------------------------------------------------
    # 4) Initialize the simulation arrays
    # ------------------------------------------------
    sim_a[:, 1] .= sim_a_init
    sim_k[:, 1] .= sim_k_init
    sim_p_idx[:, 1] .= sim_p_init_idx

    # ------------------------------------------------
    # 5) Simulate forward for each t = 1,...,T
    # ------------------------------------------------
    @showprogress "Simulating..." for t in 1:T
        for i in 1:simN
            # Retrieve current states
            a = sim_a[i, t]
            k = sim_k[i, t]
            p_idx = sim_p_idx[i, t]

            # Draw current transitory shock
            t_draw = draws_uniform_t[i, t]
            t_idx = discrete_draw(t_weight, t_draw)
            sim_t_idx[i, t] = t_idx

            # Choose consumption & hours via shock-dependent policy
            if path_choice[i] == :college && t <= t_college
                # In college, policy only depends on (a,k). `sol_c_college` is constant across t_idx.
                c = interp_c_college[t, t_idx, p_idx](a, k)
                h = 0.0
            else # Working path (or post-college)
                c = interp_c_work[t, t_idx, p_idx](a, k)
                h = interp_h_work[t, t_idx, p_idx](a, k)
            end

            sim_c[i, t] = c
            sim_h[i, t] = h

            # Compute income and wage based on path and realized shocks
            if path_choice[i] == :college && t <= t_college
                sim_income[i, t] = y # Non-labor income only
                sim_wage[i, t] = 0
            else
                t_shock = t_grid[t_idx]
                p_shock = p_grid[p_idx]
                wage = wage_func(model, k, t, t_shock, p_shock)
                sim_wage[i, t] = wage / 0.584
                sim_income[i, t] = wage * h # Labor + non-labor income
            end

            # If not in the final period, update next period's states
            if t < T
                # State transition for assets and human capital
                if path_choice[i] == :college && t <= t_college
                    a_next = (1 + r)*a - c - college_cost + y
                    k_next = k + college_boost
                else # Working path
                    a_next = (1 + r)*a + sim_income[i,t] - c + y # (1+r)a + labor_income - c + y
                    k_next = k + h
                end
                
                sim_a[i, t+1] = max(a_next, a_min)
                sim_k[i, t+1] = k_next

                # State transition for persistent shock
                p_draw = draws_uniform_p[i, t]
                p_trans_probs = p_transition[p_idx, :]
                sim_p_idx[i, t+1] = discrete_draw(p_trans_probs, p_draw)
            end
        end
    end

    # ------------------------------------------------
    # 6) Report results
    # ------------------------------------------------
    num_college = sum(path_choice .== :college)
    println("\n--- Simulation Results ---")
    println("Number choosing college: $num_college ($(round(100*num_college/simN, digits=1))%)")
    println("Number choosing work:    $(simN - num_college)")

    return model, path_choice
end