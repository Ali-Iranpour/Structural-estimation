# =============================================================================
# parent_family.jl
#
# Parent-child family problem (childhood + adolescence, t = 1..17).
# Extracted verbatim from transfer_CRRA_wage.ipynb so the model can be
# version-controlled, diffed, and reused without notebook execution-order
# effects. An untouched copy of the pre-extraction notebook is kept at
#   archive/Combined Models/Full model/transfer_CRRA_wage_ORIGINAL.ipynb
#
# Requires (loaded by the notebook before this file):
#   Random, NLopt, LinearAlgebra, Interpolations, DataFrames, Statistics,
#   ProgressMeter, Distributions, StatsBase, QuantEcon, FastGaussQuadrature,
#   Parameters, Dierckx
# and the child lifecycle module: include("ConSavLabor_college_ret.jl")
#
# NOTE: no logic has been changed in this extraction. Known issues are
# documented in README.md ("Known issues").
# =============================================================================



# -----------------------------------------------------------------------------
# Helper macro: suppress solver output
#   (was notebook cell 12)
# -----------------------------------------------------------------------------

macro suppress_output(ex)
    quote
        open("/dev/null", "w") do devnull
            redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    $(esc(ex))
                end
            end
        end
    end
end


# -----------------------------------------------------------------------------
# Helper: safe maximum over (possibly NaN / -Inf) value pairs
#   (was notebook cell 8)
# -----------------------------------------------------------------------------

function safe_maximum(x, y)
    # Both x and y are scalars (floats)
    if isnan(x) || x == -Inf
        if isnan(y) || y == -Inf
            return NaN
        else
            return y
        end
    elseif isnan(y) || y == -Inf
        return x
    else
        return max(x, y)
    end
end


# -----------------------------------------------------------------------------
# Model definition: struct + constructor
#   (was notebook cell 14)
# -----------------------------------------------------------------------------

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

# -------------------------------
# Dynamic Family Model Definition
# -------------------------------
using Random

mutable struct Parent_child_interaction_age_specific_AR1
    # --- Model Parameters ---
    T::Int                        # Number of periods
    beta_vector::Vector{Float64}                 # Discount factor
    phi_1_vector::Vector{Float64}                # Parent's disutility of labor
    phi_2_vector::Vector{Float64}                # Parent's utility from human capital
    phi_3_vector::Vector{Float64}                # Parent's utility from consumption
    R_vector::Vector{Float64}                    # Human capital technology parameter
    sigma_1_vector::Vector{Float64}              # Elasticity: HC w.r.t. child care time
    sigma_2_vector::Vector{Float64}              # Elasticity: HC w.r.t. investment
    sigma_3_vector::Vector{Float64}              # Elasticity: HC w.r.t. current human capital
    sigma_4_vector::Vector{Float64}              # Elasticity: HC w.r.t. child's own study time
    lambda_1_vector::Vector{Float64}             # Child's utility of labor
    lambda_2_vector::Vector{Float64}             # Child's utility from human capital
    mu_vector::Vector{Float64}                   # Parameter for bargining inside the family
    rho::Float64                  # risk aversion
    eta::Float64                  # Frisch elasticity
    tax_lambda::Float64           # Tax progressivity
    tau::Float64                  # Labor income tax
    r::Float64                    # Interest rate
    y::Float64                    # Unearned income
    w::Float64                   # Wage rate
    a_max::Float64                # Max asset level
    a_min::Float64                # Min asset level
    Na::Int                       # Asset grid size
    k_max::Float64                # Max physical capital
    k_min::Float64                # Min physical capital
    Nk::Int                       # Physical capital grid size
    hc_max::Float64               # Max human capital (for child/parent)
    hc_min::Float64               # Min human capital
    Nhc::Int                      # Human capital grid size
    alpha::Float64                # Parameter in wage function

    # --- Stochastic Shock Parameters (AR1 only) ---
    Np::Int; p_grid::Vector{Float64}; p_transition::Matrix{Float64}
    p_ar1::Float64; sigma_p::Float64

    

    # --- Grids ---
    a_grid::Vector{Float64}       # Asset grid
    k_grid::Vector{Float64}       # Parent Human capital grid
    hc_grid::Vector{Float64}      # Human capital grid

    # --- Solution Arrays (for value function iteration) ---
    sol_c::Array{Float64, 5}       # Parental consumption    [T, Na, Nk, Nhc]
    sol_i::Array{Float64, 5}       # Child own study time    [T, Na, Nk, Nhc]
    sol_h::Array{Float64, 5}       # Parental labor supply   [T, Na, Nk, Nhc]
    sol_t::Array{Float64, 5}       # Child care time         [T, Na, Nk, Nhc]
    sol_e::Array{Float64, 5}       # Education expenditure   [T, Na, Nk, Nhc]
    sol_v::Array{Float64, 5}       # Value function          [T, Na, Nk, Nhc]
    sol_tr::Array{Float64, 5}      # Transfers               [T, Na, Nk, Nhc]
    sol_t_asset::Array{Float64, 5} 

    # --- Simulation Storage ---
    simN::Int                     # Number of simulated agents
    simT::Int                     # Number of simulation periods
    sim_c::Array{Float64,2}       # Simulated consumption   [simN, simT]
    sim_h::Array{Float64,2}       # Simulated labor         [simN, simT]
    sim_t::Array{Float64,2}       # Simulated child care    [simN, simT]
    sim_e::Array{Float64,2}       # Simulated education exp [simN, simT]
    sim_a::Array{Float64,2}       # Simulated assets        [simN, simT]
    sim_i::Array{Float64,2}       # Simulated child study   [simN, simT]
    sim_k::Array{Float64,2}       # Simulated capital       [simN, simT]
    sim_hc::Array{Float64,2}      # Simulated human capital [simN, simT]
    sim_wage::Array{Float64,2}     # Simulated wage         [simN, simT]
    sim_income::Array{Float64,2}   # Simulated income       [simN, simT]
    sim_tr::Array{Float64,2}       # Simulated transfers     [simN, simT]
    sim_t_asset::Array{Float64, 2}
    sim_p::Array{Int,2}       # Simulated AR1 shocks    [simN, simT]


    # --- Initial conditions ---
    sim_a_init::Vector{Float64}   # Initial assets          [simN]
    sim_k_init::Vector{Float64}   # Initial capital         [simN]
    sim_hc_init::Vector{Float64}  # Initial human capital   [simN]
    sim_p_init::Vector{Int}    # Initial AR1 shocks      [Np, simN]

    # --- Wage vector (for each period) ---
    w_vec::Vector{Float64}        # Wage per period         [T]
    V_child_interp::Any  # Added field for V_child_interp

    # --- Regression coefficients (from Stata output) ---
    β0::Float64                # _cons
    β_bothcollege::Float64     # Both_College
    β_age::Float64             # Age
    β_age2::Float64            # Age_2
    β_age2_capital::Float64    # Age_2_Edu   (interaction of Age^2 with Both_College)
    β_age_capital::Float64     # Age_Edu     (interaction of Age with Both_College)
    
end

function Parent_child_interaction_age_specific_AR1(;
        # --- Scalar Defaults for Non-varying Parameters ---
        T::Int=17, rho::Float64=1.5, eta::Float64=2.0,
        tau::Float64=0.18, r::Float64=0.03, w::Float64=12.5,
        y::Float64=0.6, alpha::Float64=0.08, tax_lambda::Float64=0.82,
        # --- grid info ---
        a_max::Float64=50.0, a_min::Float64=0.0, Na::Int=30,
        k_max::Float64=1.0, k_min::Float64=0.0, Nk::Int=2,
        hc_max::Float64=6.0, hc_min::Float64=0.001, Nhc::Int=30 ,
        # --- simulation details ----
        simN::Int=5000, simT::Int=T, seed::Int=1234,

        # --- Slope/Intercept parameters for ALL age-specific variables ---
        beta_0 = 0.96,     beta_1 = 0.0,
        phi_1_0 = 1.0,     phi_1_1 = 0.0,
        phi_2_0 = 20.0,     phi_2_1 = 0.0,
        phi_3_0 = 0.03,     phi_3_1 = 0.0,
        R_0 = 2.0,         R_1 = 0.06,
        sigma_1_0 = -1.8,  sigma_1_1 = -0.02,
        sigma_2_0 = -1.8,  sigma_2_1 = 0.02,
        sigma_3_0 = -2.4,  sigma_3_1 = 0.06,
        sigma_4_0 = -3.5,  sigma_4_1 = 0.02,
        lambda_1_0 = 0.7,  lambda_1_1 = 0.0,
        lambda_2_0 = 0.3,  lambda_2_1 = 0.0,
        # --- Bargaining parameter ---
        mu_0 = 1.0,        mu_1 = -0.04,
        # Shock parameters (AR1 only)
        p_ar1::Float64=0.9, sigma_p::Float64=0.1, Np::Int=3,
        β0 = 2.798937,
        β_bothcollege = 0.3077394,
        β_age = 0.0230108,
        β_age2 = -0.0004319,
        β_age2_capital = -0.0004296,
        β_age_capital = 0.0173774,
        )  


    # Grids (custom grid functions)
    a_grid = create_focused_grid(a_min, a_min + 3.0, a_max, Na, 0.3, 1.2)
    k_grid = range(k_min, k_max, length=Nk)
    hc_grid = create_focused_grid(hc_min, hc_min + 3.0, hc_max, Nhc, 0.8, 1.2)

    #a_grid  = range(a_min, a_max, length=Na)
    #k_grid  = range(k_min, k_max, length=Nk)
    #hc_grid = range(hc_min, hc_max, length=Nhc)

    # --- Setup Persistent AR1 Shock ---
    mc = tauchen(Np, p_ar1, sigma_p, 0.0, 3)
    p_grid = exp.(mc.state_values)
    p_transition = mc.p

    # --- Age-specific parameter vectors ---
    beta_vector    = [beta_0 + beta_1 * (t-1) for t in 1:T]
    phi_1_vector   = [phi_1_0 + phi_1_1 * (t-1) for t in 1:T]
    phi_2_vector   = [phi_2_0 + phi_2_1 * (t-1) for t in 1:T]
    phi_3_vector   = [phi_3_0 + phi_3_1 * (t-1) for t in 1:T]
    R_vector       = [R_0 + R_1 * (t-1) for t in 1:T]
    #R_vector       = [t <= 7 ? 2.0 : 2.5 + 0.1 * (t-1) for t in 1:T]
    mu_vector      = [t <= 7 ? 1.0 : mu_0 + mu_1 * (t-7) for t in 1:T]

    sigma_1_vector = [exp(sigma_1_0 + sigma_1_1 * (t-1)) for t in 1:T]
    sigma_2_vector = [exp(sigma_2_0 + sigma_2_1 * (t-1)) for t in 1:T]
    sigma_3_vector = [exp(sigma_3_0 + sigma_3_1 * (t-1)) for t in 1:T]
    sigma_4_vector = [t <= 7 ? 0.0 : exp(sigma_4_0 + sigma_4_1 * (t-7)) for t in 1:T]

    
    #sigma_1_vector = [0.10 for t in 1:T]  # very small, but constant
    #sigma_2_vector = [0.10 + 0.01*(t-1) for t in 1:T] # slowly rising
    #sigma_3_vector = [0.30 for t in 1:T]  # moderate persistence
    #sigma_4_vector = [t <= 7 ? 0.0 : 0.10 + 0.01*(t-8) for t in 1:T]

    lambda_1_vector = [lambda_1_0 + lambda_1_1 * (t-1) for t in 1:T]
    lambda_2_vector = [lambda_2_0 + lambda_2_1 * (t-1) for t in 1:T]
    

    # Solution arrays (4D: [T, Na, Nk, Nhc])
    sol_shape = (T, Na, Nk, Nhc, Np)
    sol_c = fill(NaN, sol_shape)
    sol_i = fill(NaN, sol_shape)
    sol_h = fill(NaN, sol_shape)
    sol_t = fill(NaN, sol_shape)
    sol_e = fill(NaN, sol_shape)
    sol_v = fill(NaN, sol_shape)
    sol_tr = fill(NaN, sol_shape)
    sol_t_asset  = fill(NaN, sol_shape)

    # Simulation arrays (2D: [simN, simT])
    sim_shape = (simN, simT)
    sim_c = fill(NaN, sim_shape)
    sim_h = fill(NaN, sim_shape)
    sim_t = fill(NaN, sim_shape)
    sim_e = fill(NaN, sim_shape)
    sim_i = fill(NaN, sim_shape)
    sim_wage = fill(NaN, sim_shape)
    sim_income = fill(NaN, sim_shape)
    sim_tr = fill(NaN, sim_shape)
    sim_t_asset = fill(NaN, sim_shape)
    sim_p = zeros(Int, sim_shape)

    # STATE variables: need simT+1 columns!
    sim_a = fill(NaN, (simN, simT+1))
    sim_k = fill(NaN, (simN, simT+1))
    sim_hc = fill(NaN, (simN, simT+1))

    # Initial conditions

    rng_a  = MersenneTwister(1234)
    rng_k  = MersenneTwister(5678)
    rng_hc = MersenneTwister(9012)
    rng_p  = MersenneTwister(3456)
    sim_a_init = rand(rng_a, LogNormal(0.2962227, 1.401793), simN)
    sim_k_init = Float64.(rand(rng_k, Bernoulli(0.3), simN))  # 70% zeros, 30% ones
    sim_hc_init = rand(rng_hc, simN) .* 1;
    sim_p_init = fill(ceil(Int, Np/2), simN)



    # Wage vector
    w_vec = fill(w, T)
    

    return Parent_child_interaction_age_specific_AR1(
    T,
    beta_vector, phi_1_vector, phi_2_vector, phi_3_vector, R_vector,
    sigma_1_vector, sigma_2_vector, sigma_3_vector, sigma_4_vector,
    lambda_1_vector, lambda_2_vector, mu_vector,
    rho, eta, tax_lambda, tau, r, y, w, a_max, a_min, Na,
    k_max, k_min, Nk,
    hc_max, hc_min, Nhc,
    alpha,
    Np, p_grid, p_transition, p_ar1, sigma_p,
    a_grid, k_grid, hc_grid,
    sol_c, sol_i, sol_h, sol_t, sol_e, sol_v, sol_tr, sol_t_asset,
    simN, simT,
    sim_c, sim_h, sim_t, sim_e, sim_a, sim_i, sim_k, sim_hc, sim_wage, sim_income, sim_tr, sim_t_asset, sim_p,
    sim_a_init, sim_k_init, sim_hc_init, sim_p_init,
    w_vec, nothing, β0, β_bothcollege, β_age, β_age2, β_age2_capital, β_age_capital)
end


# -----------------------------------------------------------------------------
# Solver: backward induction, objectives, utilities, constraints
#   (was notebook cell 15)
# -----------------------------------------------------------------------------

# === Put near the top of your file ===
const TOL_CONSTR = 1e-8
const WAGE_SCALING_FACTOR = 0.584 # e.g., Adjustment for hours worked per year
const AMIN = 0.0    # Minimum asset level


# --------------------------
# Model Solver
# --------------------------
function solve_model!(model::Parent_child_interaction_age_specific_AR1)
    T, Na, Nk, Nhc, Np = model.T, model.Na, model.Nk, model.Nhc, model.Np
    a_grid, k_grid, hc_grid, p_grid = model.a_grid, model.k_grid, model.hc_grid, model.p_grid



    # ----- Terminal period (t = T) -----
    t = T
    println("Solving period $t ... (full model, separate)")
    converge_count = 0
    maxeval_count = 0
    other_dict = Dict{Symbol, Int}()
    itercounts = Int[]
    total = 0
    #interp = create_interp2(model, model.sol_v, t+1)
    for i_a in 1:Na, i_k in 1:Nk, i_hc in 1:Nhc, i_p in 1:Np
        assets = a_grid[i_a]
        capital = k_grid[i_k]
        HC = hc_grid[i_hc]
        p_shock = p_grid[i_p]


        function obj_wrapper(x::Vector, grad::Vector)
            c_p, i_c, e_p, h_p, t_p = x
            f = obj_last_period_full(model, c_p, i_c, e_p, h_p, t_p, assets, HC, capital, t, p_shock,
                                    model.V_child_interp, grad)
            if length(grad) > 0
                grad[:] = -grad[:]
            end
            return -f
        end

        opt = Opt(:LD_SLSQP, 5)
        lower_bounds!(opt, [1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
        upper_bounds!(opt, [100, 1.0, 100, 1.0, 1.0])
        min_objective!(opt, obj_wrapper)
        inequality_constraint!(opt, constraint_min_leisure_full, TOL_CONSTR)
        inequality_constraint!(opt, constraint_child_time, TOL_CONSTR)
        inequality_constraint!(opt, (x, grad) -> asset_constraint_full(x, grad, model, capital, t, assets, p_shock), TOL_CONSTR)
        inequality_constraint!(opt, (x, grad) -> asset_constraint_max(x, grad, model, capital, t, assets, p_shock), TOL_CONSTR)


        init = [1.0, 0.7, 1.0, 0.7, 0.2]
        xtol_rel!(opt, 1e-4)
        maxeval!(opt, 5000)
        (minf, x_opt, ret) = optimize(opt, init)

        push!(itercounts, opt.numevals)
        rt = result_type_name(ret)
        if rt == "converged"
            converge_count += 1
        elseif rt == "maxeval"
            maxeval_count += 1
        elseif any(isnan, x_opt)
            println("Warning: NaN in solution at t=$t, i_a=$i_a, i_k=$i_k, i_hc=$i_hc")
            x_opt = init
        end
        total += 1

        model.sol_c[t, i_a, i_k, i_hc, i_p] = x_opt[1]
        model.sol_i[t, i_a, i_k, i_hc, i_p] = x_opt[2]
        model.sol_e[t, i_a, i_k, i_hc, i_p] = x_opt[3]
        model.sol_h[t, i_a, i_k, i_hc, i_p] = x_opt[4]
        model.sol_t[t, i_a, i_k, i_hc, i_p] = x_opt[5]
        model.sol_tr[t, i_a, i_k, i_hc, i_p] = 0.0
        model.sol_t_asset[t, i_a, i_k, i_hc, i_p] = 0.0
        model.sol_v[t, i_a, i_k, i_hc, i_p] = -minf
    end
    print_period_stats(t, converge_count, maxeval_count, other_dict, itercounts, total)

    # ----- Earlier periods (t = T-1 down to 8) -----
    for t in (T-1):-1:8
        println("Solving period $t ... (full model)")
        converge_count = 0
        maxeval_count = 0
        other_dict = Dict{Symbol, Int}()
        itercounts = Int[]
        total = 0
        interp = create_interp(model, model.sol_v, t+1)
        for i_a in 1:Na, i_k in 1:Nk, i_hc in 1:Nhc, i_p in 1:Np
            assets = a_grid[i_a]
            capital = k_grid[i_k]
            HC = hc_grid[i_hc]
            p_shock = p_grid[i_p]

            function obj_wrapper(x::Vector, grad::Vector)
                c_p, i_c, e_p, h_p, t_p= x[1], x[2], x[3], x[4], x[5]
                f = obj_work_period_full(model, c_p, i_c, e_p, h_p, t_p, assets, HC, capital, t, p_shock, i_p, interp, grad)
                if length(grad) > 0
                    grad[:] = -grad[:]
                end
                return -f
            end

            opt = Opt(:LD_SLSQP, 5)
            lower_bounds!(opt, [0.01, 1e-4, 0.01, 1e-4, 1e-4])
            upper_bounds!(opt, [100, 1.0, 100, 1.0, 1.0])
            inequality_constraint!(opt, constraint_min_leisure_full, TOL_CONSTR)
            inequality_constraint!(opt, constraint_child_time, TOL_CONSTR)
            inequality_constraint!(opt, (x, grad) -> asset_constraint_full(x, grad, model, capital, t, assets, p_shock), TOL_CONSTR)
            inequality_constraint!(opt, (x, grad) -> asset_constraint_max(x, grad, model, capital, t, assets, p_shock), TOL_CONSTR)

            min_objective!(opt, obj_wrapper)
            init = [
                model.sol_c[t+1, i_a, i_k, i_hc, i_p],
                model.sol_i[t+1, i_a, i_k, i_hc, i_p],
                model.sol_e[t+1, i_a, i_k, i_hc, i_p],
                model.sol_h[t+1, i_a, i_k, i_hc, i_p],
                model.sol_t[t+1, i_a, i_k, i_hc, i_p],
            ]
            xtol_rel!(opt, 1e-4)
            maxeval!(opt, 1000)
            (minf, x_opt, ret) = optimize(opt, init)
            
            push!(itercounts, opt.numevals)
            rt = result_type_name(ret)
            if rt == "converged"
                converge_count += 1
            elseif rt == "maxeval"
                maxeval_count += 1
            end
            total += 1

            model.sol_c[t, i_a, i_k, i_hc, i_p] = x_opt[1]
            model.sol_i[t, i_a, i_k, i_hc, i_p] = x_opt[2]
            model.sol_e[t, i_a, i_k, i_hc, i_p] = x_opt[3]
            model.sol_h[t, i_a, i_k, i_hc, i_p] = x_opt[4]
            model.sol_t[t, i_a, i_k, i_hc, i_p] = x_opt[5]
            model.sol_tr[t, i_a, i_k, i_hc, i_p] = 0.0
            model.sol_t_asset[t, i_a, i_k, i_hc, i_p] = 0.0
            model.sol_v[t, i_a, i_k, i_hc, i_p] = -minf
        end
        print_period_stats(t, converge_count, maxeval_count, other_dict, itercounts, total)
    end

    # ----- Parent-only periods (t = 7 down to 1) -----
    for t in 7:-1:1
        println("Solving period $t ... (parent only)")
        converge_count = 0
        maxeval_count = 0
        other_dict = Dict{Symbol, Int}()
        itercounts = Int[]
        total = 0
        interp = create_interp(model, model.sol_v, t+1)
        for i_a in 1:Na, i_k in 1:Nk, i_hc in 1:Nhc, i_p in 1:Np
            assets = a_grid[i_a]
            capital = k_grid[i_k]
            HC = hc_grid[i_hc]
            p_shock = p_grid[i_p]

            function obj_wrapper(x::Vector, grad::Vector)
                c_p, e_p, h_p, t_p = x[1], x[2], x[3], x[4]
                f = obj_work_period_parentonly(model, c_p, e_p, h_p, t_p, assets, HC, capital, t, p_shock, i_p, interp, grad)
                if length(grad) > 0
                    grad[:] = -grad[:]
                end
                return -f
            end

            opt = Opt(:LD_SLSQP, 4)
            lower_bounds!(opt, [0.01, 0.01, 1e-6, 1e-6])
            upper_bounds!(opt, [100.0, 100.0, 1.0, 1.0])
            inequality_constraint!(opt, constraint_min_leisure_parentonly, TOL_CONSTR)
            inequality_constraint!(opt, (x, grad) -> asset_constraint_parentonly(x, grad, model, capital, t, assets, p_shock), TOL_CONSTR)
            inequality_constraint!(opt, (x, grad) -> asset_constraint_max_parentonly(x, grad, model, capital, t, assets, p_shock), TOL_CONSTR)
            min_objective!(opt, obj_wrapper)
            init = [
                model.sol_c[t+1, i_a, i_k, i_hc, i_p],
                model.sol_e[t+1, i_a, i_k, i_hc, i_p],
                model.sol_h[t+1, i_a, i_k, i_hc, i_p],
                model.sol_t[t+1, i_a, i_k, i_hc, i_p],
            ]
            xtol_rel!(opt, 1e-4)
            maxeval!(opt, 1000)
            (minf, x_opt, ret) = optimize(opt, init)

            push!(itercounts, opt.numevals)
            rt = result_type_name(ret)
            if rt == "converged"
                converge_count += 1
            elseif rt == "maxeval"
                maxeval_count += 1
            else
                other_dict[ret] = get(other_dict, ret, 0) + 1
            end
            total += 1

            model.sol_c[t, i_a, i_k, i_hc, i_p] = x_opt[1]
            model.sol_e[t, i_a, i_k, i_hc, i_p] = x_opt[2]
            model.sol_h[t, i_a, i_k, i_hc, i_p] = x_opt[3]
            model.sol_t[t, i_a, i_k, i_hc, i_p] = x_opt[4]
            model.sol_i[t, i_a, i_k, i_hc, i_p] = 0.0
            model.sol_tr[t, i_a, i_k, i_hc, i_p] = 0.0
            model.sol_t_asset[t, i_a, i_k, i_hc, i_p] = 0.0
            model.sol_v[t, i_a, i_k, i_hc, i_p] = -minf
        end
        print_period_stats(t, converge_count, maxeval_count, other_dict, itercounts, total)
    end
end

# ------------------------------------------------
# Supporting functions 
# ------------------------------------------------
function obj_last_period_full(model::Parent_child_interaction_age_specific_AR1, c_p, i_c, e_p, h_p, t_p,
                             assets, HC, capital, t, p_shock::Float64, V_child_interp, grad)
    # Calculate leisure
    leisure_p = 1.0 - h_p - t_p
    leisure_c = 1.0 - t_p - i_c

    # Wage and next period states
    w = wage_func(model, capital, t, p_shock)
    labor_pre = w * h_p
    after_tax = model.tax_lambda * labor_pre ^ (1 - model.tau)
    a_next = (1.0 + model.r) * assets + after_tax + model.y - c_p - e_p
    HC_next = HC_technology_full(model, t_p, e_p, HC, i_c, t)

    # Objective function
    util_now = util_total(model, c_p, h_p, t_p, i_c, HC, t)
    V_next = V_child_interp(a_next, HC_next)
    f = util_now + model.beta_vector[t] * V_next

    # Gradient calculations
    if length(grad) > 0
        # Compute derivatives at the correct point
        dV_da = Dierckx.derivative(V_child_interp, a_next, HC_next, 1, 0)
        dV_dHC = Dierckx.derivative(V_child_interp, a_next, HC_next, 0, 1)

        # Partial derivatives of utility
        dutil_dc_p = model.phi_1_vector[t] * (c_p ^ (-model.rho))
        dutil_di_c = (1 - model.mu_vector[t]) * model.lambda_1_vector[t] / leisure_c * (-1)
        dutil_de_p = 0.0
        dutil_dh_p = - model.phi_2_vector[t] * (h_p ^ model.eta)
        dutil_dt_p = (1 - model.mu_vector[t]) * model.lambda_1_vector[t] / leisure_c * (-1)

        # Partial derivatives of HC_next
        dHC_next_dt_p = HC_next * model.sigma_1_vector[t] / t_p
        dHC_next_de_p = HC_next * model.sigma_2_vector[t] / e_p
        dHC_next_di_c = HC_next * model.sigma_4_vector[t] / i_c

        # Marginal wage for tax
        marginal = model.tax_lambda * (1 - model.tau) * labor_pre ^ (- model.tau) * w

        # Gradients
        grad[1] = dutil_dc_p + model.beta_vector[t] * dV_da * (-1)  # ∂f/∂c_p
        grad[2] = dutil_di_c + model.beta_vector[t] * dV_dHC * dHC_next_di_c  # ∂f/∂i_c
        grad[3] = dutil_de_p + model.beta_vector[t] * (-dV_da + dV_dHC * dHC_next_de_p)  # ∂f/∂e_p
        grad[4] = dutil_dh_p + model.beta_vector[t] * (dV_da * marginal)  # ∂f/∂h_p
        grad[5] = dutil_dt_p + model.beta_vector[t] * dV_dHC * dHC_next_dt_p  # ∂f/∂t_p

        # Handle NaN in gradients
        if any(isnan, grad)
            grad .= -1e12
            return -1e12
        end
    end


    return f
end

@inline function obj_work_period_full(
    model::Parent_child_interaction_age_specific_AR1, c_p::Float64, i_c::Float64, e_p::Float64, h_p::Float64, t_p::Float64,
    assets::Float64, HC::Float64, capital::Float64, t::Int, p_shock::Float64, i_p::Int, interp::Vector, grad::Vector
)
    w = wage_func(model, capital, t, p_shock)
    labor_pre = w * h_p
    after_tax = model.tax_lambda * labor_pre ^ (1 - model.tau)
    a_next = (1.0 + model.r) * assets + after_tax + model.y - c_p - e_p
    k_next = capital
    leisure_p = 1.0 - h_p - t_p
    leisure_c = 1.0 - t_p - i_c
    HC_next = HC_technology_full(model, t_p, e_p, HC, i_c, t)
    util_now = util_total(model, c_p, h_p, t_p, i_c, HC, t)
    
    # Compute expected value and gradients over next shock states
    V_next = 0.0
    dV_da_sum = 0.0
    dV_dk_sum = 0.0
    dV_dHC_sum = 0.0
    for j_p in 1:model.Np
        p_trans_prob = model.p_transition[i_p, j_p]
        if p_trans_prob > 1e-12
            interp_jp = interp[j_p]
            Vj = interp_jp(a_next, k_next, HC_next)
            ∇V_jp = Interpolations.gradient(interp_jp, a_next, k_next, HC_next)
            dV_da_jp, dV_dk_jp, dV_dHC_jp = ∇V_jp
            V_next += p_trans_prob * Vj
            dV_da_sum += p_trans_prob * dV_da_jp
            dV_dk_sum += p_trans_prob * dV_dk_jp
            dV_dHC_sum += p_trans_prob * dV_dHC_jp
        end
    end
    f = util_now + model.beta_vector[t] * V_next

    if length(grad) > 0
        dutil_dc_p = model.phi_1_vector[t] * (c_p ^ (-model.rho))
        dutil_dh_p = - model.phi_2_vector[t] * (h_p ^ model.eta)
        term_leisure_c = (1 - model.mu_vector[t]) * model.lambda_1_vector[t] / leisure_c * (-1)
        marginal = model.tax_lambda * (1 - model.tau) * labor_pre ^ (- model.tau) * w
        grad[1] = dutil_dc_p + model.beta_vector[t] * dV_da_sum * (-1)
        grad[2] = term_leisure_c + model.beta_vector[t] * dV_dHC_sum * (HC_next * model.sigma_4_vector[t] / i_c)
        grad[3] = model.beta_vector[t] * (-dV_da_sum + dV_dHC_sum * (HC_next * model.sigma_2_vector[t] / e_p))
        grad[4] = dutil_dh_p + model.beta_vector[t] * (dV_da_sum * marginal + dV_dk_sum)
        grad[5] = term_leisure_c + model.beta_vector[t] * dV_dHC_sum * (HC_next * model.sigma_1_vector[t] / t_p)
    end
    return f
end

@inline function obj_work_period_parentonly(
    model::Parent_child_interaction_age_specific_AR1, c_p::Float64, e_p::Float64, h_p::Float64, t_p::Float64,
    assets::Float64, HC::Float64, capital::Float64, t::Int, p_shock::Float64, i_p::Int, interp::Vector, grad::Vector
)
    w = wage_func(model, capital, t, p_shock)
    labor_pre = w * h_p
    after_tax = model.tax_lambda * labor_pre ^ (1 - model.tau)
    a_next = (1.0 + model.r) * assets + after_tax + model.y - c_p - e_p
    k_next = capital
    HC_next = HC_technology_parentonly(model, t_p, e_p, HC, t)
    leisure = 1.0 - h_p - t_p
    util_now = util_parent(model, c_p, h_p, t_p, HC, t)
    
    # Compute expected value and gradients over next shock states
    V_next = 0.0
    dV_da_sum = 0.0
    dV_dk_sum = 0.0
    dV_dHC_sum = 0.0
    for j_p in 1:model.Np
        p_trans_prob = model.p_transition[i_p, j_p]
        if p_trans_prob > 1e-12
            interp_jp = interp[j_p]
            Vj = interp_jp(a_next, k_next, HC_next)
            ∇V_jp = Interpolations.gradient(interp_jp, a_next, k_next, HC_next)
            dV_da_jp, dV_dk_jp, dV_dHC_jp = ∇V_jp
            V_next += p_trans_prob * Vj
            dV_da_sum += p_trans_prob * dV_da_jp
            dV_dk_sum += p_trans_prob * dV_dk_jp
            dV_dHC_sum += p_trans_prob * dV_dHC_jp
        end
    end

    if length(grad) > 0
        marginal = model.tax_lambda * (1 - model.tau) * labor_pre ^ (- model.tau) * w
        grad[1] = model.phi_1_vector[t] * (c_p ^ (-model.rho)) - model.beta_vector[t] * dV_da_sum
        grad[2] = model.beta_vector[t] * (-dV_da_sum + dV_dHC_sum * (HC_next * model.sigma_2_vector[t] / e_p))
        grad[3] = -model.phi_2_vector[t] * (h_p ^ model.eta) + model.beta_vector[t] * (marginal * dV_da_sum + dV_dk_sum)
        grad[4] = model.beta_vector[t] * dV_dHC_sum * (HC_next * model.sigma_1_vector[t] / t_p)
    end
    return util_now + model.beta_vector[t] * V_next
end

# ------------------------------------------------
# Utility Functions
# ------------------------------------------------

@inline function util_total(model::Parent_child_interaction_age_specific_AR1, c::Float64, h_p::Float64,
                            t_p::Float64, i_c::Float64, HC::Float64, t::Int)
    leisure_c = 1.0 - t_p - i_c
    if c <= 0.0 || i_c <= 0.0 || leisure_c <= 0.0
        return -1e8
    end
    rho = model.rho
    eta = model.eta
    u_cons = model.phi_1_vector[t] * (c ^ (1.0 - rho) / (1.0 - rho))
    disutil_h = - model.phi_2_vector[t] * (h_p ^ (1.0 + eta) / (1.0 + eta))
    u_parent = u_cons + disutil_h
    u_child  = model.mu_vector[t] * model.phi_3_vector[t] * log(HC) + 
            (1 - model.mu_vector[t]) * (model.lambda_1_vector[t] * log(leisure_c) + model.lambda_2_vector[t] * log(HC))
    return u_parent + u_child
end

@inline function util_parent(model::Parent_child_interaction_age_specific_AR1, c::Float64, h_p::Float64, t_p::Float64, HC::Float64, t::Int)
    if c <= 0.0
        return -1e8
    end
    rho = model.rho
    eta = model.eta
    u_cons = model.phi_1_vector[t] * (c ^ (1.0 - rho) / (1.0 - rho))
    disutil_h = - model.phi_2_vector[t] * (h_p ^ (1.0 + eta) / (1.0 + eta))
    return u_cons + disutil_h + model.phi_3_vector[t] * log(HC)
end



# ------------------------------------------------
# Human Capital Functions
# ------------------------------------------------

@inline function HC_technology_full(model::Parent_child_interaction_age_specific_AR1, t_p::Float64, e_p::Float64, HC::Float64, i_c::Float64, t::Int)
    if t_p <= 0.0 || e_p <= 0.0
        return -1e8
    end
    return exp(log(model.R_vector[t]) +
        model.sigma_1_vector[t] * log(t_p) +
        model.sigma_2_vector[t] * log(e_p) +
        model.sigma_3_vector[t] * log(HC)  +
        model.sigma_4_vector[t] * log(i_c))
end

@inline function HC_technology_parentonly(model::Parent_child_interaction_age_specific_AR1, t_p::Float64, e_p::Float64, HC::Float64, t::Int)
    if t_p <= 0.0 || e_p <= 0.0
        return -1e8 
    end
    return exp(log(model.R_vector[t]) +
            model.sigma_1_vector[t] * log(t_p) +
            model.sigma_2_vector[t] * log(e_p) + 
            model.sigma_3_vector[t] * log(HC))
end

# ------------------------------------------------
# Budget constraints
# ------------------------------------------------


@inline function wage_func(model::Parent_child_interaction_age_specific_AR1, capital::Float64, t::Int, p_shock::Float64)
    log_wage =
        model.β0 +
        model.β_bothcollege * capital +
        model.β_age * t +
        model.β_age2 * (t^2) +
        model.β_age2_capital * ((t^2) * capital) +
        model.β_age_capital * (t * capital)

    return 2 * exp(log_wage) * p_shock * WAGE_SCALING_FACTOR   # <-- use the const
end

@inline function asset_constraint_full(x::Vector, grad::Vector, model::Parent_child_interaction_age_specific_AR1,
                                    capital::Float64, t::Int, assets::Float64, p_shock::Float64)
    c_p, i_c, e_p, h_p, t_p = x
    w = wage_func(model, capital, t, p_shock)
    labor_pre = w * h_p
    after_tax = model.tax_lambda * labor_pre ^ (1 - model.tau)
    a_next = (1.0 + model.r) * assets + after_tax + model.y - c_p - e_p
    if length(grad) > 0
        marginal = model.tax_lambda * (1 - model.tau) * labor_pre ^ (- model.tau) * w
        grad[1] = 1.0
        grad[2] = 0.0
        grad[3] = 1.0
        grad[4] = -marginal
        grad[5] = 0.0
    end
    return 1e-6 - a_next
end


@inline function asset_constraint_max(x::Vector, grad::Vector, model::Parent_child_interaction_age_specific_AR1,
                                    capital::Float64, t::Int, assets::Float64, p_shock::Float64)
    c_p, i_c, e_p, h_p, t_p = x
    w = wage_func(model, capital, t, p_shock)
    labor_pre = w * h_p
    after_tax = model.tax_lambda * labor_pre ^ (1 - model.tau)
    a_next = (1.0 + model.r) * assets + after_tax + model.y - c_p - e_p
    if length(grad) > 0
        marginal = model.tax_lambda * (1 - model.tau) * labor_pre ^ (- model.tau) * w
        grad[1] = -1.0
        grad[2] = 0.0
        grad[3] = -1.0
        grad[4] = marginal
        grad[5] = 0.0
    end
    return a_next - model.a_max
end

@inline function asset_constraint_parentonly(x::Vector, grad::Vector, model::Parent_child_interaction_age_specific_AR1,
                                             capital::Float64, t::Int, assets::Float64, p_shock::Float64)
    c_p, e_p, h_p, t_p = x
    w = wage_func(model, capital, t, p_shock)
    labor_pre = w * h_p
    after_tax = model.tax_lambda * labor_pre ^ (1 - model.tau)
    a_next = (1.0 + model.r) * assets + after_tax + model.y - c_p - e_p
    if length(grad) > 0
        marginal = model.tax_lambda * (1 - model.tau) * labor_pre ^ (- model.tau) * w
        grad[1] = 1.0
        grad[2] = 1.0
        grad[3] = -marginal
        grad[4] = 0.0
    end
    return 1e-6 - a_next
end


@inline function asset_constraint_max_parentonly(x::Vector, grad::Vector, model::Parent_child_interaction_age_specific_AR1,
                                             capital::Float64, t::Int, assets::Float64, p_shock::Float64)
    c_p, e_p, h_p, t_p = x
    w = wage_func(model, capital, t, p_shock)
    labor_pre = w * h_p
    after_tax = model.tax_lambda * labor_pre ^ (1 - model.tau)
    a_next = (1.0 + model.r) * assets + after_tax + model.y - c_p - e_p
    if length(grad) > 0
        marginal = model.tax_lambda * (1 - model.tau) * labor_pre ^ (- model.tau) * w
        grad[1] = -1.0
        grad[2] = -1.0
        grad[3] = marginal
        grad[4] = 0.0
    end
    return a_next - model.a_max
end

# ------------------------------------------------
# Time constraints
# ------------------------------------------------

@inline function constraint_min_leisure_full(x::Vector, grad::Vector)
    h_p = x[4]
    t_p = x[5]
    if length(grad) > 0
        grad .= 0.0
        grad[4] = 1.0
        grad[5] = 1.0
    end
    return (h_p + t_p) - 1.0
end

@inline function constraint_min_leisure_parentonly(x::Vector, grad::Vector)
    h_p = x[3]
    t_p = x[4]
    if length(grad) > 0
        grad .= 0.0
        grad[3] = 1.0
        grad[4] = 1.0
    end
    return (h_p + t_p) - 1.0
end

@inline function constraint_child_time(x::Vector, grad::Vector = [])
    i_c = x[2]
    t_p = x[5]
    if !isempty(grad)
        grad .= 0.0
        grad[2] = 1.0
        grad[5] = 1.0
    end
    return i_c + t_p - 1.0
end


# ------------------------------------------------
# Interpolation Functions
# ------------------------------------------------

function create_interp(model::Parent_child_interaction_age_specific_AR1, sol_v, t)
    Np = model.Np
    interp_vec = Vector{Any}(undef, Np)
    @inbounds @simd for i_p in 1:Np
        vview = @view sol_v[t, :, :, :, i_p]   # avoid copy
        itp = interpolate((model.a_grid, model.k_grid, model.hc_grid), vview, Gridded(Linear()))
        interp_vec[i_p] = extrapolate(itp, Line())
    end
    return interp_vec
end

# ------------------------------------------------
# Debug Functions
# ------------------------------------------------

function result_type_name(ret)
    if ret == :FTOL_REACHED || ret == :XTOL_REACHED
        return "converged"
    elseif ret == :MAXEVAL_REACHED
        return "maxeval"
    else
        return "other"
    end
end

function print_period_stats(t, converge_count, maxeval_count, other_dict, itercounts, total)
    avg_iter = round(mean(itercounts), digits=2)
    println("Period $t: Converged: $(round(converge_count/total*100, digits=1))%, Maxeval: $(round(maxeval_count/total*100, digits=1))%, Other: $(round(sum(values(other_dict))/total*100, digits=1))%, Avg iters: $avg_iter")
    if sum(values(other_dict)) > 0
        println("    Other status codes:")
        for (code, count) in other_dict
            println("        $code : $count times ($(round(count/total*100, digits=1))%)")
        end
    end
end


# -----------------------------------------------------------------------------
# Simulation: forward simulation of the parent problem
#   (was notebook cell 16)
# -----------------------------------------------------------------------------



function simulate_model!(model::Parent_child_interaction_age_specific_AR1)
    # Unpack
    simN, simT = model.simN, model.T
    sim_a, sim_k, sim_hc = model.sim_a, model.sim_k, model.sim_hc
    sim_c, sim_i, sim_e, sim_h, sim_t, sim_tr = model.sim_c, model.sim_i, model.sim_e, model.sim_h, model.sim_t, model.sim_tr
    sim_wage, sim_income = model.sim_wage, model.sim_income
    sim_p = model.sim_p  # Simulated AR1 shock states (simN, simT)

    # Safety checks for solution arrays (updated to 5D with Np)
    for solname in (:sol_c, :sol_i, :sol_e, :sol_h, :sol_t)
        sol = getfield(model, solname)
        if !(typeof(sol) <: Array)
            error("$(solname) must be a 5D Array (T, Na, Nk, Nhc, Np), but got ", typeof(sol))
        end
        if size(sol, 1) < simT
            error("Simulation time horizon (simT=$(simT)) exceeds available solution periods ($(solname) has T=$(size(sol,1))).")
        end
    end

    # Simulate AR1 shock paths
    for i in 1:simN
        sim_p[i, 1] = model.sim_p_init[i]  # Initial shock state
        for t in 1:simT-1
            current_state = sim_p[i, t]
            next_state = sample(1:model.Np, Weights(model.p_transition[current_state, :]))
            sim_p[i, t+1] = next_state
        end
    end

    # Build interpolators for each period, shock state, and variable
    interp_dict = Dict{Tuple{Int, Int, Symbol}, Any}()
    hc_grid = model.hc_grid  # Assuming hc_grid is in levels; adjust if in logs
    for t in 1:simT, i_p in 1:model.Np
        interp_dict[(t, i_p, :c)] = extrapolate(interpolate((model.a_grid, model.k_grid, hc_grid), model.sol_c[t, :, :, :, i_p], Gridded(Linear())), Flat())
        interp_dict[(t, i_p, :i)] = extrapolate(interpolate((model.a_grid, model.k_grid, hc_grid), model.sol_i[t, :, :, :, i_p], Gridded(Linear())), Flat())
        interp_dict[(t, i_p, :e)] = extrapolate(interpolate((model.a_grid, model.k_grid, hc_grid), model.sol_e[t, :, :, :, i_p], Gridded(Linear())), Flat())
        interp_dict[(t, i_p, :h)] = extrapolate(interpolate((model.a_grid, model.k_grid, hc_grid), model.sol_h[t, :, :, :, i_p], Gridded(Linear())), Flat())
        interp_dict[(t, i_p, :t)] = extrapolate(interpolate((model.a_grid, model.k_grid, hc_grid), model.sol_t[t, :, :, :, i_p], Gridded(Linear())), Flat())
    end

    # Initialize initial states
    for i in 1:simN
        sim_a[i, 1]  = model.sim_a_init[i]
        sim_k[i, 1]  = model.sim_k_init[i]
        sim_hc[i, 1] = model.sim_hc_init[i]
    end

    # Main simulation loop
    for i in 1:simN
        for t in 1:simT
            a  = sim_a[i, t]
            k  = sim_k[i, t]
            hc = sim_hc[i, t]
            p_state = sim_p[i, t]          # Current shock state index
            p_shock = model.p_grid[p_state] # Current shock value

            # Policy choices using interpolants for current t and shock state
            sim_c[i, t] = interp_dict[(t, p_state, :c)](a, k, hc)
            sim_i[i, t] = interp_dict[(t, p_state, :i)](a, k, hc)
            sim_e[i, t] = interp_dict[(t, p_state, :e)](a, k, hc)
            sim_h[i, t] = interp_dict[(t, p_state, :h)](a, k, hc)
            sim_t[i, t] = interp_dict[(t, p_state, :t)](a, k, hc)

            # Wage and income with AR1 shock
            wage = wage_func(model, k, t, p_shock)
            sim_wage[i, t] = wage / WAGE_SCALING_FACTOR  # Store true wage (not scaled)
            labor_pre = wage * sim_h[i, t]
            after_tax = model.tax_lambda * labor_pre ^ (1 - model.tau)
            sim_income[i, t] = after_tax  # Store after-tax income
 
            # Transition equations
            sim_a[i, t+1] = (1.0 + model.r) * a + sim_income[i, t] + model.y - sim_c[i, t] - sim_e[i, t]
            sim_k[i, t+1] = k 
            if t < 8
                hc_next = exp(log(model.R_vector[t]) +
                              model.sigma_1_vector[t] * log(max(sim_t[i, t], 1e-8)) +
                              model.sigma_2_vector[t] * log(max(sim_e[i, t], 1e-8)) +
                              model.sigma_3_vector[t] * log(max(hc, 1e-8)))
            else
                hc_next = exp(log(model.R_vector[t]) +
                              model.sigma_1_vector[t] * log(max(sim_t[i, t], 1e-8)) +
                              model.sigma_2_vector[t] * log(max(sim_e[i, t], 1e-8)) +
                              model.sigma_3_vector[t] * log(max(hc, 1e-8)) +
                              model.sigma_4_vector[t] * log(max(sim_i[i, t], 1e-8)))
            end
            sim_hc[i, t+1] = hc_next
        end
    end

        final_assets = sim_a[:, simT+1]
    final_hc     = sim_hc[:, simT+1]

    println("Mean final asset: ", mean(final_assets))
    println("Mean final human capital: ", mean(final_hc))
    println("Std. dev. of final human capital: ", std(final_hc))

    return final_assets, final_hc
end


# -----------------------------------------------------------------------------
# Simulation: heterogeneous beliefs (parent side)
#   (was notebook cell 44)
# -----------------------------------------------------------------------------

# Step 5: Simulation function with heterogeneous beliefs
function simulate_model_hetero!(
    parent_models::Vector{Parent_child_interaction_age_specific_AR1},
    belief_type::Vector{Int};
    verbose::Bool = true
)
    # ======= Shared setup from the first (base) model =======
    model = parent_models[1]  # assumes shared sim arrays and grids live here
    simN, simT = model.simN, model.T

    a_grid, k_grid, hc_grid = model.a_grid, model.k_grid, model.hc_grid
    sim_a, sim_k, sim_hc = model.sim_a, model.sim_k, model.sim_hc
    sim_c, sim_i, sim_e   = model.sim_c, model.sim_i, model.sim_e
    sim_h, sim_t, sim_tr  = model.sim_h, model.sim_t, model.sim_tr
    sim_wage, sim_income  = model.sim_wage, model.sim_income
    sim_p                 = model.sim_p  # (simN, simT) integer indices of shock states

    # ======= Safety checks: require 5D policy arrays =======
    for pm in parent_models
        for solname in (:sol_c, :sol_i, :sol_e, :sol_h, :sol_t)
            sol = getfield(pm, solname)
            @assert sol isa Array && ndims(sol) == 5 "$(solname) must be 5D (T,Na,Nk,Nhc,Np). Got $(typeof(sol)) with ndims=$(ndims(sol))."
            @assert size(sol, 1) >= simT "simT=$(simT) exceeds $(solname)'s T=$(size(sol,1))."
        end
    end

    # ======= Build interpolators by (model m, time t, shock i_p, var) =======
    # NOTE: include the 5th index (i_p)! e.g., pm.sol_c[t, :, :, :, i_p]
    interp = Dict{Tuple{Int,Int,Int,Symbol}, Any}()
    for m in 1:length(parent_models)
        pm = parent_models[m]
        for t in 1:simT, i_p in 1:pm.Np
            interp[(m,t,i_p,:c)] = extrapolate(
                interpolate((pm.a_grid, pm.k_grid, pm.hc_grid), @view(pm.sol_c[t, :, :, :, i_p]), Gridded(Linear())),
                Flat()
            )
            interp[(m,t,i_p,:i)] = extrapolate(
                interpolate((pm.a_grid, pm.k_grid, pm.hc_grid), @view(pm.sol_i[t, :, :, :, i_p]), Gridded(Linear())),
                Flat()
            )
            interp[(m,t,i_p,:e)] = extrapolate(
                interpolate((pm.a_grid, pm.k_grid, pm.hc_grid), @view(pm.sol_e[t, :, :, :, i_p]), Gridded(Linear())),
                Flat()
            )
            interp[(m,t,i_p,:h)] = extrapolate(
                interpolate((pm.a_grid, pm.k_grid, pm.hc_grid), @view(pm.sol_h[t, :, :, :, i_p]), Gridded(Linear())),
                Flat()
            )
            interp[(m,t,i_p,:t)] = extrapolate(
                interpolate((pm.a_grid, pm.k_grid, pm.hc_grid), @view(pm.sol_t[t, :, :, :, i_p]), Gridded(Linear())),
                Flat()
            )
        end
    end

    # ======= Initialize levels at t=1 =======
    @inbounds for i in 1:simN
        sim_a[i, 1]  = model.sim_a_init[i]
        sim_k[i, 1]  = model.sim_k_init[i]
        sim_hc[i, 1] = model.sim_hc_init[i]
    end

    # ======= Simulate AR(1) belief-specific shock paths =======
    @inbounds for i in 1:simN
        m  = belief_type[i]
        pm = parent_models[m]
        sim_p[i, 1] = (hasfield(typeof(pm), :sim_p_init) && length(pm.sim_p_init) >= i) ? pm.sim_p_init[i] : model.sim_p_init[i]
        for t in 1:simT-1
            current = sim_p[i, t]
            sim_p[i, t+1] = sample(1:pm.Np, Weights(pm.p_transition[current, :]))
        end
    end

    # ======= Main simulation =======
    @inbounds for i in 1:simN
        m  = belief_type[i]
        pm = parent_models[m]
        for t in 1:simT
            a  = sim_a[i, t]
            k  = sim_k[i, t]
            hc = sim_hc[i, t]

            p_state = sim_p[i, t]
            p_shock = pm.p_grid[p_state]

            sim_c[i, t] = interp[(m, t, p_state, :c)](a, k, hc)
            sim_i[i, t] = interp[(m, t, p_state, :i)](a, k, hc)
            sim_e[i, t] = interp[(m, t, p_state, :e)](a, k, hc)
            sim_h[i, t] = interp[(m, t, p_state, :h)](a, k, hc)
            sim_t[i, t] = interp[(m, t, p_state, :t)](a, k, hc)

            wage = wage_func(model, k, t, p_shock)
            sim_wage[i, t] = wage / WAGE_SCALING_FACTOR  # Store true wage (not scaled)
            labor_pre = wage * sim_h[i, t]
            after_tax = model.tax_lambda * labor_pre ^ (1 - model.tau)
            sim_income[i, t] = after_tax  # Store after-tax income

            sim_a[i, t+1] = (1.0 + pm.r) * a + sim_income[i, t] + pm.y - sim_c[i, t] - sim_e[i, t]
            sim_k[i, t+1] = k

            if t < 8
                sim_hc[i, t+1] = exp(
                    log(pm.R_vector[t]) +
                    pm.sigma_1_vector[t] * log(max(sim_t[i, t], 1e-8)) +
                    pm.sigma_2_vector[t] * log(max(sim_e[i, t], 1e-8)) +
                    pm.sigma_3_vector[t] * log(max(hc, 1e-8))
                )
            else
                sim_hc[i, t+1] = exp(
                    log(pm.R_vector[t]) +
                    pm.sigma_1_vector[t] * log(max(sim_t[i, t], 1e-8)) +
                    pm.sigma_2_vector[t] * log(max(sim_e[i, t], 1e-8)) +
                    pm.sigma_3_vector[t] * log(max(hc, 1e-8)) +
                    pm.sigma_4_vector[t] * log(max(sim_i[i, t], 1e-8))
                )
            end
        end
    end

    # ======= Outputs =======
    final_assets = sim_a[:, simT+1]
    final_hc     = sim_hc[:, simT+1]

    if verbose
        println("Mean final asset: ", mean(final_assets))
        println("Mean final human capital: ", mean(final_hc))
        println("Std. dev. of final human capital: ", std(final_hc))
    end

    return final_assets, final_hc, belief_type
end


# -----------------------------------------------------------------------------
# Simulation: heterogeneous beliefs (family / child side)
#   (was notebook cell 49)
# -----------------------------------------------------------------------------

function simulate_model_family_hetero!(
    base_child::ConSavLaborCollege_AR1,
    child_models::Vector{ConSavLaborCollege_AR1},
    belief_type::Vector{Int},
    verbose::Bool = true
)
    # -- Unpack grids/dims --
    @unpack simN, T, t_college, r, college_cost, college_boost, a_min, t_retire = base_child
    @unpack a_grid, k_grid, p_grid, p_transition, Np = base_child
    @unpack sim_a, sim_k, sim_c, sim_h, sim_income, sim_wage = base_child
    @unpack sim_p_idx, sim_a_init, sim_k_init, sim_p_init_idx, draws_uniform_p, y = base_child
    @unpack Nt, t_weight, t_retire = base_child

    num_bins = length(child_models)
    belief_values = [child_models[m].college_boost for m in 1:num_bins]
    college_boost_true = base_child.college_boost

    # -- Work interpolators --
    interp_c_work = [
        LinearInterpolation((a_grid, k_grid), base_child.sol_c_work[t, :, :, ip, 1]; extrapolation_bc=Flat())
        for t in 1:T, ip in 1:Np
    ]
    interp_h_work = [
        LinearInterpolation((a_grid, k_grid), base_child.sol_h_work[t, :, :, ip, 1]; extrapolation_bc=Flat())
        for t in 1:T, ip in 1:Np
    ]
    sol_tr_v_work_interp = [
        LinearInterpolation((a_grid, k_grid), base_child.sol_tr_v_work[:, :, ip, 1]; extrapolation_bc=Flat())
        for ip in 1:Np
    ]
    sol_tr_work_interp = [
        LinearInterpolation((a_grid, k_grid), base_child.sol_tr_work[:, :, ip, 1]; extrapolation_bc=Flat())
        for ip in 1:Np
    ]

    # -- Retirement interpolators (shock-free, i_p=1, i_t=1) --
    interp_c_retire = [
        LinearInterpolation((a_grid, k_grid), base_child.sol_c_work[t, :, :, 1, 1]; extrapolation_bc=Flat())
        for t in 1:T
    ]

    # -- College interpolators (belief-specific) --
    interp_c_college_belief = [
        LinearInterpolation((a_grid, k_grid), child_models[m].sol_c_college[t, :, :, ip, it]; extrapolation_bc=Flat())
        for m in 1:num_bins, it in 1:Nt, t in 1:T, ip in 1:Np
    ]
    interp_h_college_belief = [
        LinearInterpolation((a_grid, k_grid), child_models[m].sol_h_college[t, :, :, ip, it]; extrapolation_bc=Flat())
        for m in 1:num_bins, it in 1:Nt, t in 1:T, ip in 1:Np
    ]
    sol_tr_v_college_interp_belief = [
        LinearInterpolation((a_grid, k_grid), child_models[m].sol_tr_v_college[:, :, ip, it]; extrapolation_bc=Flat())
        for m in 1:num_bins, it in 1:Nt, ip in 1:Np
    ]
    sol_tr_college_interp_belief = [
        LinearInterpolation((a_grid, k_grid), child_models[m].sol_tr_college[:, :, ip, it]; extrapolation_bc=Flat())
        for m in 1:num_bins, it in 1:Nt, ip in 1:Np
    ]

    # -- Draw taste shock nodes --
    cum_weights = cumsum(t_weight)
    rng = MersenneTwister(2222)
    eps_indices = [clamp(findfirst(w -> w ≥ rand(rng), cum_weights), 1, Nt) for _ in 1:simN]

    # -- Assign initial path and transfer --
    path_choice = Vector{Symbol}(undef, simN)
    tr_initial = Vector{Float64}(undef, simN)
    for i in 1:simN
        m = belief_type[i]
        it = eps_indices[i]
        ip = sim_p_init_idx[i]
        parent_assets = sim_a_init[i]
        HC = sim_k_init[i]
        f_college = sol_tr_v_college_interp_belief[m, it, ip](parent_assets, HC)
        f_work = sol_tr_v_work_interp[ip](parent_assets, HC)
        if f_college > f_work
            path_choice[i] = :college
            tr_initial[i] = max(sol_tr_college_interp_belief[m, it, ip](parent_assets, HC), 1e-6)
        else
            path_choice[i] = :work
            tr_initial[i] = max(sol_tr_work_interp[ip](parent_assets, HC), 1e-6)
        end
    end

    sim_a[:, 1] .= tr_initial
    sim_k[:, 1] .= sim_k_init
    sim_p_idx[:, 1] .= sim_p_init_idx

    # -- Main simulation loop --
    @showprogress "Simulating..." for t in 1:T
        for i in 1:simN
            m = belief_type[i]
            it = eps_indices[i]
            p_idx = sim_p_idx[i, t]
            a = sim_a[i, t]
            k = sim_k[i, t]

            # Indices for college interpolators
            idx_c = (m, it, t, p_idx)
            idx_h = (m, it, t, p_idx)

            if t >= t_retire
                # ----- Retirement (shock-free) -----
                c = interp_c_retire[t](a, k)
                h = 0.0
                pen = pension_amount(base_child, k, t)
                w_pre = wage_func(base_child, k, t, 1.0)
                sim_wage[i, t] = w_pre / WAGE_SCALING_FACTOR
                sim_income[i, t] = pen
            elseif path_choice[i] == :college && t <= t_college
                # ----- In college -----
                if t == 1
                    c = interp_c_college_belief[idx_c...](a, k)
                    h = interp_h_college_belief[idx_h...](a, k)
                else
                    idx_c = (m, 1, t, p_idx)
                    idx_h = (m, 1, t, p_idx)
                    c = interp_c_college_belief[idx_c...](a, k)
                    h = interp_h_college_belief[idx_h...](a, k)
                end
                sim_income[i, t] = 0.0
                sim_wage[i, t] = 0.0
            else
                # ----- Working (pre-retirement) -----
                c = interp_c_work[t, p_idx](a, k)
                h = interp_h_work[t, p_idx](a, k)
                p_shock = p_grid[p_idx]
                w_pre = wage_func(base_child, k, t, p_shock)
                sim_wage[i, t] = w_pre / WAGE_SCALING_FACTOR
                sim_income[i, t] = after_tax_income(base_child, w_pre, h)
            end

            # Store
            sim_c[i, t] = c
            sim_h[i, t] = h

            # Update states for next period (if t < T)
            if t < T
                if t >= t_retire
                    pen = pension_amount(base_child, k, t)
                    a_next = (1 + r) * a - c + pen + y
                    k_next = k
                elseif path_choice[i] == :college
                    if t < t_college
                        a_next = (1 + r) * a - c - college_cost + y
                        k_next = k + belief_values[m]
                    elseif t == t_college
                        a_next = (1 + r) * a - c - college_cost + y
                        k_next = k + college_boost_true + 3 * (college_boost_true - belief_values[m])
                    else
                        a_next = (1 + r) * a + sim_income[i, t] - c + y
                        k_next = k + h
                    end
                else
                    a_next = (1 + r) * a + sim_income[i, t] - c + y
                    k_next = k + h
                end
                sim_a[i, t+1] = max(a_next, a_min)
                sim_k[i, t+1] = k_next

                # Persistent shock update
                if t < t_retire
                    p_draw = draws_uniform_p[i, t]
                    p_trans_probs = p_transition[p_idx, :]
                    sim_p_idx[i, t+1] = discrete_draw(p_trans_probs, p_draw)
                else
                    sim_p_idx[i, t+1] = 1  # Fixed p_shock = 1.0 in retirement
                end
            end
        end
    end

    # -- Report results --
    if verbose
        num_college = sum(path_choice .== :college)
        println("\n--- Simulation Results with Heterogeneous Beliefs ---")
        println("Number choosing college: $num_college ($(round(100*num_college/simN, digits=1))%)")
        println("Number choosing work:    $(simN - num_college)")
    end

    return path_choice, tr_initial
end

