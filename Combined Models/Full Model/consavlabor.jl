# -------------------------------
# Utility: Nonlinear Grid Creator
# -------------------------------
function nonlinspace(start::Float64, stop::Float64, num::Int, curv::Float64)
    lin_vals = range(0, stop=1, length=num)
    curved_vals = lin_vals .^ curv
    return start .+ (stop - start) .* curved_vals
end


function create_focused_grid(a_min::Float64, a_focus::Float64, a_max::Float64, Na::Int, focus_share::Float64, curv::Float64)
    # Number of points in the focus region (0 to 250,000)
    Na_focus = ceil(Int, Na * focus_share)
    # Number of points in the remaining region (250,000 to 1,000,000)
    Na_rest = Na - Na_focus
    
    # Create grid for the focus region
    grid_focus = nonlinspace(a_min, a_focus, Na_focus, curv)
    # Create grid for the upper region, excluding the duplicate point at a_focus
    grid_rest = nonlinspace(a_focus, a_max, Na_rest + 1, curv)[2:end]
    
    # Combine the two grids
    return vcat(grid_focus, grid_rest)
end


# -------------------------------
# Dynamic Labor Model Definition
# -------------------------------
mutable struct ConSavLabor
    T::Int                        # Time periods
    rho::Float64                 # Risk aversion (CRRA)
    beta::Float64                # Discount factor
    phi::Float64                 # Weight on labor disutility
    eta::Float64                 # Frisch elasticity parameter
    alpha::Float64               # (possibly productivity or returns to labor)
    w::Float64                   # Wage rate
    y::Float64                   # unearned income
    tau::Float64                 # Labor income tax
    r::Float64                   # Interest rate
    a_max::Float64               # Max asset level
    a_min::Float64               # Min asset level
    Na::Int                      # Number of asset grid points
    k_max::Float64               # Max labor effort
    Nk::Int                      # Number of labor grid points
    simT::Int                    # Simulation time periods
    simN::Int                    # Number of simulated agents
    a_grid::Vector{Float64}      # Asset grid
    k_grid::Vector{Float64}      # Labor grid
    sol_c::Array{Float64,3}      # Optimal consumption [T, Na, Nk]
    sol_h::Array{Float64,3}      # Optimal labor effort [T, Na, Nk]
    sol_v::Array{Float64,3}      # Value function [T, Na, Nk]
    sim_c::Array{Float64,2}      # Simulated consumption [simN, simT]
    sim_h::Array{Float64,2}      # Simulated labor [simN, simT]
    sim_a::Array{Float64,2}      # Simulated assets [simN, simT]
    sim_k::Array{Float64,2}      # Simulated labor choice [simN, simT]
    sim_income::Matrix{Float64}  # Income over time (not explicitly defined in the original code)
    sim_wage::Matrix{Float64}  # Wage over time (not explicitly defined in the original code)

    sim_a_init::Vector{Float64}  # Initial assets
    sim_k_init::Vector{Float64}  # Initial labor effort
    

    draws_uniform::Array{Float64,2}  # Uniform draws for simulation [simN, simT]
    w_vec::Vector{Float64}       # Time-varying wage vector [T]
end

# -------------------------------
# Constructor for ConSavLabor
# -------------------------------
function ConSavLabor(; T::Int=50, beta::Float64=0.97, rho::Float64=1.0, y::Float64=0.6,
                        r::Float64=0.03, a_max::Float64=20.0, Na::Int=30, simN::Int=5000,
                        a_min::Float64=0.0, k_max::Float64=30.0, Nk::Int=30,
                        w::Float64=12.5, tau::Float64=0.25,
                        eta::Float64=2.0, alpha::Float64=0.08,
                        phi::Float64=20.0, seed::Int=1234)


    # --- Time horizon and simulation settings ---
    simT = T

    # --- Grids for state variables and decisions ---
    #a_grid = nonlinspace(a_min, a_max, Na, 1.5)
    a_grid = create_focused_grid(a_min, 5.0, a_max, Na, 0.7, 1.1)
    k_grid = nonlinspace(0.0, k_max, Nk, 1.5)


    # --- Storage for solution (policy + value functions) ---
    # Dimensions: (T, Na, Nk) 
    sol_shape = (T, Na, Nk)
    sol_c = fill(NaN, sol_shape)   # Optimal consumption
    sol_h = fill(NaN, sol_shape)   # Optimal labor effort
    sol_v = fill(NaN, sol_shape)   # Value function

    # --- Simulation storage ---
    sim_shape = (simN, simT)
    sim_c = fill(NaN, sim_shape)
    sim_h = fill(NaN, sim_shape)
    sim_a = fill(NaN, sim_shape)
    sim_k = fill(NaN, sim_shape)

    # --- Random draws for simulation ---
    rng = MersenneTwister(seed)
    draws_uniform = rand(rng, sim_shape...)

    # --- Initial conditions for simulation ---
    sim_a_init = zeros(Float64, simN)
    sim_k_init = zeros(Float64, simN)
    sim_income = fill(NaN, (simN, T));  # Initialize income array for simulation
    sim_wage = fill(NaN, (simN, T));   # Initialize wage array for simulation

    # --- Wage vector (can vary by time) ---
    w_vec = fill(w, T)

    # --- Return constructed model ---
    return ConSavLabor(T, rho, beta, phi, eta, alpha, w, y, tau, r,
                        a_max, a_min, Na, k_max, Nk, simT, simN,
                        a_grid, k_grid,
                        sol_c, sol_h, sol_v,
                        sim_c, sim_h, sim_a, sim_k, sim_income, sim_wage,
                        sim_a_init, sim_k_init, draws_uniform, w_vec)
end


# ------------------------------------------------
# Objective and Constraint Functions
# ------------------------------------------------

@inline function obj_last_period(model::ConSavLabor, h_vec::Vector, assets::Float64, capital::Float64, t::Int, grad::Vector)
    h = h_vec[1]
    w = wage_func(model, capital, t)
    income = w * h
    c = assets + income + model.y

    u = util(model, c, h)             # Your utility function
    du_dc = c^(-model.rho)
    du_dh = -model.phi * h^model.eta
    du_dh_total = w * du_dc + du_dh

    if length(grad) > 0
        grad[1] = du_dh_total
    end
    return u
end

@inline function obj_work_period(model::ConSavLabor, x::Vector, assets::Float64, capital::Float64, t::Int, interp, grad::Vector)
    c, h = x[1], x[2]
    w = wage_func(model, capital, t)
    income = w * h
    a_next = (1.0 + model.r) * assets + income - c + model.y
    k_next = capital + h
    V_next = interp(a_next, k_next)
    util_now = util(model, c, h)
    dutil_dc = c^(-model.rho)
    dutil_dh = -model.phi * h^model.eta

    V = util_now + model.beta * V_next
    if length(grad) > 0
        grad_V_next = Interpolations.gradient(interp, a_next, k_next)
        dV_next_da = grad_V_next[1]
        dV_next_dk = grad_V_next[2]
        dV_dc = dutil_dc - model.beta * dV_next_da
        dV_dh = dutil_dh + model.beta * (w * dV_next_da + dV_next_dk)
        grad[1] = dV_dc
        grad[2] = dV_dh
    end
    return V
end

@inline function asset_constraint(x::Vector, grad::Vector, model::ConSavLabor, assets::Float64, capital::Float64, t::Int)
    c, h = x[1], x[2]
    w = wage_func(model, capital, t)
    a_next = (1.0 + model.r) * assets + w * h + model.y - c
    g = model.a_min - a_next  # g <= 0 ensures a_next >= a_min
    if length(grad) > 0
        grad[1] = 1.0  # ∂g/∂c = -∂a_next/∂c = 1
        grad[2] = -w   # ∂g/∂h = -∂a_next/∂h = -w
    end
    return g
end

# --------------------------
# Model Solver
# --------------------------

function solve_model!(model::ConSavLabor; max_iter::Int=500)
    T, Na, Nk = model.T, model.Na, model.Nk
    a_grid, k_grid = model.a_grid, model.k_grid
    sol_c, sol_h, sol_v = model.sol_c, model.sol_h, model.sol_v

    # --- Final period (t = T) ---
    for i_a in 1:Na
        for i_k in 1:Nk
            assets = a_grid[i_a]
            capital = k_grid[i_k]
            function obj_wrapper(h_vec::Vector, grad::Vector)
                f = obj_last_period(model, h_vec, assets, capital, T, grad)
                if length(grad) > 0
                    grad[:] = -grad[:]  # Negate for minimization
                end
                return -f  # Minimize negative utility
            end
            opt = Opt(:LD_SLSQP, 1)
            lower_bounds!(opt, [0.0])
            upper_bounds!(opt, [1.0])
            ftol_rel!(opt, 1e-8)
            min_objective!(opt, obj_wrapper)
            init = [0.3]
            (minf, h_vec, ret) = optimize(opt, init)
            h_opt = h_vec[1]
            cons = assets + wage_func(model, capital, T) * h_opt + model.y
            sol_h[T, i_a, i_k] = h_opt
            sol_c[T, i_a, i_k] = cons
            sol_v[T, i_a, i_k] = -minf
        end
    end

    # --- Earlier periods (t = T-1 to 1) ---
    @showprogress 1 "Solving working model..." for t in (T-1):-1:1
        interp = create_interp(model, sol_v, t + 1)
        for i_a in 1:Na
            for i_k in 1:Nk
                assets = a_grid[i_a]
                capital = k_grid[i_k]
                function obj_wrapper(x::Vector, grad::Vector)
                    f = obj_work_period(model, x, assets, capital, t, interp, grad)
                    if length(grad) > 0
                        grad[:] = -grad[:]  # Negate for minimization
                    end
                    return -f  # Minimize negative value function
                end
                function constraint_wrapper(x::Vector, grad::Vector)
                    return asset_constraint(x, grad, model, assets, capital, t)
                end
                opt = Opt(:LD_SLSQP, 2)
                lower_bounds!(opt, [1e-8, 0.0])
                upper_bounds!(opt, [30, 1.0])
                ftol_rel!(opt, 1e-8)
                min_objective!(opt, obj_wrapper)
                inequality_constraint!(opt, constraint_wrapper, 0.0)
                init = [sol_c[t + 1, i_a, i_k], sol_h[t + 1, i_a, i_k]]
                (minf, x_opt, ret) = optimize(opt, init)
                sol_c[t, i_a, i_k] = x_opt[1]
                sol_h[t, i_a, i_k] = x_opt[2]
                sol_v[t, i_a, i_k] = -minf
            end
        end
    end
end

# ------------------------------------------------
# Supporting Functions
# ------------------------------------------------

@inline function util(model::ConSavLabor, c, h)
    if model.rho == 1.0
        cons_utility = log(c)
    else
        cons_utility = (c^(1.0 - model.rho)) / (1.0 - model.rho)
    end
    labor_disutility = model.phi * (h^(1.0 + model.eta)) / (1.0 + model.eta)
    return cons_utility - labor_disutility
end

function create_interp(model::ConSavLabor, sol_v::Array{Float64, 3}, t::Int)
    return LinearInterpolation((model.a_grid, model.k_grid), sol_v[t, :, :], extrapolation_bc=Line())
end

@inline function wage_func(model::ConSavLabor, capital::Float64, t::Int)
    return (1 - model.tau) * model.w_vec[t] * (1.0 + model.alpha * capital) * 0.584
end

