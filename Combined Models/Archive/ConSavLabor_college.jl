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
# Dynamic Labor Model Definition with College Decision
#
# This mutable struct defines the model parameters, state grids, solution arrays,
# and simulation arrays for a dynamic consumption-saving model with a college decision.
#
# Key components:
# - Model parameters (e.g., T, beta, r, w, etc.)
# - State variable grids (for assets 'a_grid' and human capital 'k_grid')
# - Solution arrays for both work and college choices (consumption, labor, and value functions)
# - Simulation arrays for consumption, assets, labor, and capital over simulated life paths
# - Other auxiliary arrays, such as initial conditions, random draws, and wage vector (w_vec)
# - Parameters for the college decision (college_cost and college_boost)
# =============================================================================
mutable struct ConSavLaborCollege
    T::Int                     # Total number of periods
    t_college::Int             # Number of college periods
    rho::Float64               # Risk aversion parameter (or relative risk aversion)
    beta::Float64              # Discount factor
    phi::Float64               # Parameter (e.g., disutility of labor or related to college)
    eta::Float64               # Another parameter (e.g., related to preferences)
    alpha::Float64             # Wage increase per year of human capital
    y::Float64                 # Unearned income 
    w::Float64                 # Base wage level
    tau::Float64               # Tax rate or other scaling parameter on wage income
    r::Float64                 # Interest rate on assets
    a_max::Float64             # Maximum asset level in the grid
    a_min::Float64             # Minimum asset level (borrowing constraint)
    Na::Int                    # Number of grid points for assets
    k_max::Float64             # Maximum human capital level in the grid
    Nk::Int                    # Number of grid points for human capital
    simT::Int                  # Number of simulation periods (typically equals T)
    simN::Int                  # Number of simulation agents
    a_grid::Vector{Float64}    # Grid for assets (state variable)
    k_grid::Vector{Float64}    # Grid for human capital (state variable)

    # Solution arrays for the working alternative
    sol_c_work::Array{Float64, 3}  # Optimal consumption when working
    sol_h_work::Array{Float64, 3}  # Optimal labor supply when working
    sol_v_work::Array{Float64, 3}  # Value function when working

    # Solution arrays for the college alternative
    sol_c_college::Array{Float64, 3}  # Optimal consumption when in college
    sol_h_college::Array{Float64, 3}  # Optimal labor supply when in college (typically zero)
    sol_v_college::Array{Float64, 3}  # Value function when in college

    # Simulation arrays (each row corresponds to one simulated agent over time)
    sim_c::Matrix{Float64}      # Consumption over time
    sim_h::Matrix{Float64}      # Labor supply over time
    sim_a::Matrix{Float64}      # Asset holdings over time
    sim_k::Matrix{Float64}      # Human capital over time
    sim_a_init::Vector{Float64} # Initial asset levels for simulated agents
    sim_k_init::Vector{Float64} # Initial human capital for simulated agents
    sim_income::Matrix{Float64}  # Income over time (not explicitly defined in the original code)
    sim_wage::Matrix{Float64}  # Wage over time (not explicitly defined in the original code)
    
    draws_uniform::Matrix{Float64}  # Matrix of uniform random draws for simulation purposes
    w_vec::Vector{Float64}          # Wage vector over time (can vary across periods)
    college_cost::Float64           # Cost of attending college per period (or overall cost structure)
    college_boost::Float64          # Increase in human capital (or productivity boost) from college
end

# =============================================================================
# Constructor for ConSavLaborCollege
#
# This constructor initializes the model with default parameters or user-specified ones.
# It sets up:
# - The time horizon (T) and the number of college periods (t_college)
# - The asset and human capital grids using the nonlinspace utility function
# - Empty solution arrays (filled with NaN) for both work and college alternatives
# - Simulation arrays with appropriate dimensions, including initial conditions
# - The wage vector (w_vec) and random draws for simulation purposes
# - The parameters specific to the college decision (college_cost and college_boost)
# =============================================================================
function ConSavLaborCollege(; 
    T::Int=50, t_college::Int=4, beta::Float64=0.97, rho::Float64=1.0, 
    r::Float64=0.03, a_max::Float64=20.0, Na::Int=50, y::Float64=0.6,
    simN::Int=5000, a_min::Float64=0.0, k_max::Float64=30.0, Nk::Int=30, 
    w::Float64=12.5, tau::Float64=0.25, eta::Float64=2.0, alpha::Float64=0.1, 
    phi::Float64=20.0, seed::Int=1234, college_cost::Float64=1.2, 
    college_boost::Float64=2.0)
    

    
    simT = T  # Simulation time equals model time horizon
    
    # --- Grids for state variables ---
    #a_grid = create_focused_grid(a_min, 500000.0, a_max, Na, 0.5, 1.1);    # Nonlinear grid for assets
    a_grid = create_focused_grid(a_min, 7.0, a_max, Na, 0.8, 1.1)
    k_grid = create_focused_grid(0.0, 5.0, k_max, Nk, 0.8, 1.1)
    #a_grid  = range(a_min, a_max, length=Na)
    #k_grid  = range(0.0, k_max, length=Nk)      # Nonlinear grid for human capital
    
    # --- Initialize solution arrays (3D arrays for each period, asset, and human capital grid point) ---
    sol_c_work = fill(NaN, (T, Na, Nk));
    sol_h_work = fill(NaN, (T, Na, Nk));
    sol_v_work = fill(NaN, (T, Na, Nk));

    sol_c_college = fill(NaN, (T, Na, Nk));
    sol_h_college = fill(NaN, (T, Na, Nk));
    sol_v_college = fill(NaN, (T, Na, Nk));

    # --- Initialize simulation arrays (rows: simulated agents, columns: periods) ---
    sim_c = fill(NaN, (simN, T));
    sim_h = fill(NaN, (simN, T));
    sim_a = fill(NaN, (simN, T));
    sim_k = fill(NaN, (simN, T));

    # --- Set initial conditions for simulation ---
    rng = MersenneTwister(seed);
    sim_a_init = rand(rng, simN) .* 10;  # Initial assets drawn from a uniform distribution
    sim_k_init = zeros(Float64, simN);         # Initial human capital is set to zero
    sim_income = fill(NaN, (simN, T));  # Initialize income array for simulation
    sim_wage = fill(NaN, (simN, T));   # Initialize wage array for simulation

    # --- Initialize wage vector and random draws for simulation ---
    w_vec = fill(w, T);                        # Wage remains constant over time by default
    draws_uniform = rand(rng, simN, T);          # Uniform random draws for simulation purposes
    
    # --- Return an instance of ConSavLaborCollege with all fields initialized ---
    return ConSavLaborCollege(
        T, t_college, rho, beta, phi, eta, alpha, y, w, tau, r,
        a_max, a_min, Na, k_max, Nk, simT, simN, a_grid, k_grid,
        sol_c_work, sol_h_work, sol_v_work, sol_c_college, sol_h_college, sol_v_college,
        sim_c, sim_h, sim_a, sim_k,
        sim_a_init, sim_k_init, sim_income, sim_wage,
        draws_uniform, w_vec, college_cost, college_boost
    )
end

# --------------------------
# Model Solver for College Periods
# --------------------------
function solve_model_college!(model::ConSavLaborCollege)
    T, Na, Nk = model.T, model.Na, model.Nk
    a_grid, k_grid = model.a_grid, model.k_grid
    sol_c, sol_h, sol_v = model.sol_c_college, model.sol_h_college, model.sol_v_college

    a_min_t = compute_min_assets(model)
    c_min = 1e-6  # Minimum consumption

    @showprogress 1 "Solving model..." for t in T:-1:1
        if t == T
            # Final period
            for i_a in 1:Na, i_k in 1:Nk
                assets = a_grid[i_a]
                capital = k_grid[i_k]
                function obj_wrapper(h_vec::Vector, grad::Vector)
                    f = obj_last_period(model, h_vec, assets, capital, T, grad)
                    if length(grad) > 0
                        grad[:] = -grad[:]
                    end
                    return -f
                end
                opt = Opt(:LD_SLSQP, 1)
                lower_bounds!(opt, [0.0])
                upper_bounds!(opt, [1.0])
                ftol_rel!(opt, 1e-8)
                min_objective!(opt, obj_wrapper)
                init = [0.5]
                (minf, h_vec, ret) = optimize(opt, init)
                h_opt = h_vec[1]
                cons = assets + wage_func(model, capital, T) * h_opt + model.y
                sol_h[T, i_a, i_k] = h_opt
                sol_c[T, i_a, i_k] = cons
                sol_v[T, i_a, i_k] = -minf
            end
        elseif t > model.t_college
            # Post-college work periods
            interp = create_interp(model, sol_v, t + 1)
            for i_a in 1:Na, i_k in 1:Nk
                assets = a_grid[i_a]
                capital = k_grid[i_k]
                function obj_wrapper(x::Vector, grad::Vector)
                    f = obj_work_period(model, x, assets, capital, t, interp, grad)
                    if length(grad) > 0
                        grad[:] = -grad[:]
                    end
                    return -f
                end
                function constraint_wrapper(x::Vector, grad::Vector)
                    return asset_constraint_work(x, grad, model, assets, capital, t)
                end
                opt = Opt(:LD_SLSQP, 2)
                lower_bounds!(opt, [1e-6, 0.0])
                upper_bounds!(opt, [30.0, 1.0])
                ftol_rel!(opt, 1e-8)
                min_objective!(opt, obj_wrapper)
                inequality_constraint!(opt, constraint_wrapper, 1e-6)
                init = [1.0, 0.5]  # or use previous guess if you want
                (minf, x_opt, ret) = optimize(opt, init)
                sol_c[t, i_a, i_k] = x_opt[1]
                sol_h[t, i_a, i_k] = x_opt[2]
                sol_v[t, i_a, i_k] = -minf
            end
        else
            # College periods (with feasibility check)
            interp = create_interp(model, sol_v, t + 1)
            for i_a in 1:Na
                assets = a_grid[i_a]
                if assets < a_min_t[t]
                    for i_k in 1:Nk
                        sol_v[t, i_a, i_k] = -20  # Use -Inf for infeasible
                        sol_c[t, i_a, i_k] = 0.0
                        #sol_h[t, i_a, i_k] = NaN
                    end
                else
                    for i_k in 1:Nk
                        capital = k_grid[i_k]
                        function obj_wrapper(c_vec::Vector, grad::Vector)
                            f = obj_college_period(model, c_vec, assets, capital, t, interp, grad)
                            if length(grad) > 0
                                grad[:] = -grad[:]
                            end
                            return -f
                        end
                        function constraint_wrapper(c_vec::Vector, grad::Vector)
                            return asset_constraint_college(c_vec, grad, model, assets, t)
                        end
                        opt = Opt(:LD_SLSQP, 1)
                        lower_bounds!(opt, [c_min])
                        upper_bounds!(opt, [30.0])
                        ftol_rel!(opt, 1e-6)
                        maxeval!(opt, 10000)
                        min_objective!(opt, obj_wrapper)
                        inequality_constraint!(opt, constraint_wrapper, 1e-6)
                        init = [1.0]  # or use a better guess
                        (minf, c_vec, ret) = optimize(opt, init)
                        sol_c[t, i_a, i_k] = c_vec[1]
                        sol_v[t, i_a, i_k] = -minf
                        sol_h[t, i_a, i_k] = 0.0
                    end
                end
            end
        end
    end
    return model
end
# --------------------------
# Model Solver for Work Periods
# --------------------------

function solve_model_work!(model::ConSavLaborCollege)
    T, Na, Nk = model.T, model.Na, model.Nk
    a_grid, k_grid = model.a_grid, model.k_grid
    sol_c, sol_h, sol_v = model.sol_c_work, model.sol_h_work, model.sol_v_work

    # --- Final period (t = T) ---
    for i_a in 1:Na
        for i_k in 1:Nk
            assets = a_grid[i_a]
            capital = k_grid[i_k]
            function obj_wrapper(h_vec::Vector, grad::Vector)
                f = obj_last_period(model, h_vec, assets, capital, T, grad)
                if length(grad) > 0
                    grad[:] = -grad[:]
                end
                return -f
            end
            opt = Opt(:LD_SLSQP, 1)
            lower_bounds!(opt, [0.0])
            upper_bounds!(opt, [1.0])
            ftol_rel!(opt, 1e-8)
            min_objective!(opt, obj_wrapper)
            init = [0.5]
            (minf, h_vec, ret) = optimize(opt, init)
            h_opt = h_vec[1]
            cons = assets + wage_func(model, capital, T) * h_opt + model.y  # Corrected to include model.y
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
                        grad[:] = -grad[:]
                    end
                    return -f
                end
                function constraint_wrapper(x::Vector, grad::Vector)
                    return asset_constraint_work(x, grad, model, assets, capital, t)
                end
                opt = Opt(:LD_SLSQP, 2)
                lower_bounds!(opt, [1e-6, 0.0])
                upper_bounds!(opt, [30.0, 1.0])  # Adjust c upper bound as needed
                ftol_rel!(opt, 1e-8)
                min_objective!(opt, obj_wrapper)
                inequality_constraint!(opt, constraint_wrapper, 1e-6)
                init = [sol_c[t + 1, i_a, i_k], sol_h[t + 1, i_a, i_k]]
                (minf, x_opt, ret) = optimize(opt, init)
                sol_c[t, i_a, i_k] = x_opt[1]
                sol_h[t, i_a, i_k] = x_opt[2]
                sol_v[t, i_a, i_k] = -minf
            end
        end
    end

    return model
end

# ------------------------------------------------
# Supporting functions 
# ------------------------------------------------

@inline function obj_college_period(model::ConSavLaborCollege, c_vec::Vector, assets::Float64, capital::Float64, t::Int, interp, grad::Vector)
    c = c_vec[1]
    a_next = (1.0 + model.r) * assets - c - model.college_cost + model.y
    k_next = capital + model.college_boost
    V_next = interp(a_next, k_next)
    util_now = util_college(model, c)
    V = util_now + model.beta * V_next
    if length(grad) > 0
        grad_V_next = Interpolations.gradient(interp, a_next, k_next)
        dV_next_da = grad_V_next[1]
        du_dc = c^(-model.rho) # util_college is log(c)
        dV_dc = du_dc - model.beta * dV_next_da
        grad[1] = dV_dc
    end
    return V
end

@inline function obj_work_period(model::ConSavLaborCollege, x::Vector, assets::Float64, capital::Float64, t::Int, interp, grad::Vector)
    c, h = x[1], x[2]
    w = wage_func(model, capital, t)
    income = w * h
    a_next = (1.0 + model.r) * assets + income - c + model.y
    k_next = capital + h
    V_next = interp(a_next, k_next)
    util_now = util_work(model, c, h)
    V = util_now + model.beta * V_next
    if length(grad) > 0
        grad_V_next = Interpolations.gradient(interp, a_next, k_next)
        dV_next_da = grad_V_next[1]
        dV_next_dk = grad_V_next[2]
        du_dc = c^(-model.rho)
        du_dh = -model.phi * h^model.eta
        dV_dc = du_dc - model.beta * dV_next_da
        dV_dh = du_dh + model.beta * (w * dV_next_da + dV_next_dk)
        grad[1] = dV_dc
        grad[2] = dV_dh
    end
    return V
end

@inline function util_work(model::ConSavLaborCollege, c, h)
    if model.rho == 1.0
        cons_utility = log(c)
    else
        cons_utility = (c^(1.0 - model.rho)) / (1.0 - model.rho)
    end
    labor_disutility = model.phi * (h^(1.0 + model.eta)) / (1.0 + model.eta)
    return cons_utility - labor_disutility
end

# Utility function for college periods with No labor disutility
@inline function util_college(model::ConSavLaborCollege, c::Float64)
    if model.rho == 1.0
        cons_utility = log(c)
    else
        cons_utility = (c^(1.0 - model.rho)) / (1.0 - model.rho)
    end
    return cons_utility # - pychic cost
end

# Wage function
@inline function wage_func(model::ConSavLaborCollege, k::Float64, t::Int)
    return (1.0 - model.tau) * model.w_vec[t] * (1.0 + model.alpha * k) * 0.584
end

# Objective function for the last period

@inline function obj_last_period(model::ConSavLaborCollege, h_vec::Vector, assets::Float64, capital::Float64, t::Int, grad::Vector)
    h = h_vec[1]
    w = wage_func(model, capital, t)
    income = w * h
    c = assets + income + model.y
    u = util_work(model, c, h)
    if length(grad) > 0
        du_dc = c^(-model.rho)
        du_dh = -model.phi * h^model.eta
        grad[1] = w * du_dc + du_dh  # du/dh
    end
    return u
end


@inline function asset_constraint_work(x::Vector, grad::Vector, model::ConSavLaborCollege, assets::Float64, capital::Float64, t::Int)
    c, h = x[1], x[2]
    w = wage_func(model, capital, t)
    a_next = (1.0 + model.r) * assets + w * h - c + model.y
    g = model.a_min - a_next  # g <= 0 ensures a_next >= a_min
    if length(grad) > 0
        grad[1] = 1.0  # ∂g/∂c
        grad[2] = -w   # ∂g/∂h
    end
    return g
end

@inline function asset_constraint_college(c_vec::Vector, grad::Vector, model::ConSavLaborCollege, assets::Float64, t::Int)
    c = c_vec[1]
    a_next = (1.0 + model.r) * assets - c - model.college_cost + model.y
    g = model.a_min - a_next  # g <= 0 ensures a_next >= a_min
    if length(grad) > 0
        grad[1] = 1.0  # ∂g/∂c
    end
    return g
end

# Interpolation helper function
function create_interp(model::ConSavLaborCollege, sol_v::Array{Float64, 3}, t::Int)
    return LinearInterpolation((model.a_grid, model.k_grid), sol_v[t, :, :], extrapolation_bc=Line())
end

function compute_min_assets(model::ConSavLaborCollege)
    t_college = model.t_college
    r, y, college_cost, a_min = model.r, model.y, model.college_cost, model.a_min
    c_min = 1e-6  # Minimum consumption threshold
    a_min_t = zeros(t_college)
    
    # Last college period
    a_min_t[t_college] = (a_min + c_min + college_cost - y) / (1 + r)
    
    # Backward from t_college-1 to 1
    for t in (t_college-1):-1:1
        a_min_t[t] = (a_min_t[t+1] + c_min + college_cost - y) / (1 + r)
    end
    
    return a_min_t
end