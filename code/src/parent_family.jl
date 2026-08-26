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
    eta::Float64                  # curvature of the parent's leisure CRRA (was: Frisch)
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
    draws_uniform_p::Matrix{Float64}  # Pre-drawn uniforms for the AR(1) path [simN, simT]
    seed::Int                         # Seed actually used (the kwarg was previously ignored)

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
        # a_max = 100, not 50. Simulated parental assets reached 281.5 against a grid
        # ending at 50, with 0.43% of states off-grid; at 100 that falls to 0.10% and the
        # moments barely move (mean terminal assets 22.07 -> 22.13).
        a_max::Float64=100.0, a_min::Float64=0.0, Na::Int=30,
        k_max::Float64=1.0, k_min::Float64=0.0, Nk::Int=2,
        hc_max::Float64=6.0, hc_min::Float64=0.001, Nhc::Int=30 ,
        # --- simulation details ----
        simN::Int=5000, simT::Int=T, seed::Int=1234,

        # --- Slope/Intercept parameters for ALL age-specific variables ---
        beta_0 = 0.97,     beta_1 = 0.0,
        phi_1_0 = 1.0,     phi_1_1 = 0.0,
        # P10: 0.8, not 20.0. phi_2 used to scale a Frisch labor disutility
        # -phi_2*h^(1+eta)/(1+eta); it now weights the parent's leisure CRRA
        # phi_2*l_p^(1-eta)/(1-eta), a completely different scale. 0.8 reproduces the old
        # simulated labor supply almost exactly -- mean h_p 0.2860 against 0.2848 -- which
        # is the one moment that can be held fixed while the leisure term is restored.
        # NOTE: tau_p is NOT pinned by phi_2. It comes out at 0.011-0.023 for every phi_2
        # from 0.05 to 3.0, because the FOC phi_2*l^(-eta) = beta*dV/dHC*HC_next*sigma_1/tau_p
        # scales with phi_2 on both sides. tau_p is set by sigma_1 and the value of the
        # child's HC. See docs/ERRORS.md P10 for the calibration tension that creates.
        phi_2_0 = 0.5,      phi_2_1 = 0.0,
        # ---- HC block, recalibrated together (see the note below) ----
        phi_3_0 = 1.0,      phi_3_1 = 0.0,
        R_0 = 1.6,         R_1 = 0.0,
        sigma_1_0 = -0.90, sigma_1_1 = -0.02,
        sigma_2_0 = -1.8,  sigma_2_1 = 0.02,
        sigma_3_0 = -0.36, sigma_3_1 = 0.0,
        sigma_4_0 = -4.5,  sigma_4_1 = 0.02,
        lambda_1_0 = 0.7,  lambda_1_1 = 0.0,
        lambda_2_0 = 1.0,  lambda_2_1 = 0.0,
        # --- Bargaining parameter ---
        mu_0 = 1.0,        mu_1 = -0.04,
        # Shock parameters (AR1 only)
        # Np = 7, not 3. At Np = 3 the parent's shock grid was the binding approximation in
        # the whole model: raising it to 7 moved the college share 17.85% -> 22.40% and mean
        # terminal parental assets +8.9%, while DOUBLING any state grid moved the college
        # share by at most 0.15pp (parent Nhc 30 -> 60 moved it by 0.00). It converges by
        # 5-7: Np = 5 gives 22.00%, 7 gives 22.40%, 9 gives 22.30%, 13 gives 21.80%.
        p_ar1::Float64=0.9, sigma_p::Float64=0.1, Np::Int=7,
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
    # Rouwenhorst, not Tauchen. For a Gaussian-innovation AR(1) it matches the unconditional
    # mean, the unconditional variance and the first-order autocorrelation EXACTLY at every
    # N -- not asymptotically. Tauchen is built on the unconditional distribution and
    # degrades as rho -> 1, which is where this model sits. Measured here:
    #
    #   rho=0.90 sigma=0.10  N=3 : Tauchen sd +21.5%, persistence +10.8%   Rouwenhorst exact
    #   rho=0.90 sigma=0.10  N=7 : Tauchen sd +17.1%, persistence  +0.2%   Rouwenhorst exact
    #   rho=0.95 sigma=0.20  N=5 : Tauchen sd +31.4%, persistence  +4.0%   Rouwenhorst exact
    #
    # Standard reference: Kopecky & Suen (2010, RED). Trade-off: Rouwenhorst fixes the grid
    # half-width at sqrt(N-1)*sigma_z, so there is no `m` knob, and it matches the first two
    # moments but not higher ones -- the invariant distribution is binomial, normal only as
    # N grows. Neither matters for the moments this model targets.
    mc = rouwenhorst(Np, p_ar1, sigma_p)
    p_grid = exp.(mc.state_values)
    p_transition = mc.p

    # --- Age-specific parameter vectors ---
    beta_vector    = [beta_0 + beta_1 * (t-1) for t in 1:T]
    phi_1_vector   = [phi_1_0 + phi_1_1 * (t-1) for t in 1:T]
    phi_2_vector   = [phi_2_0 + phi_2_1 * (t-1) for t in 1:T]
    phi_3_vector   = [phi_3_0 + phi_3_1 * (t-1) for t in 1:T]
    R_vector       = [R_0 + R_1 * (t-1) for t in 1:T]
    #R_vector       = [t < T_CHILD_VOICE ? 2.0 : 2.5 + 0.1 * (t-1) for t in 1:T]
    mu_vector      = [t < T_CHILD_VOICE ? 1.0 : mu_0 + mu_1 * (t - (T_CHILD_VOICE - 1))
                      for t in 1:T]

    sigma_1_vector = [exp(sigma_1_0 + sigma_1_1 * (t-1)) for t in 1:T]
    sigma_2_vector = [exp(sigma_2_0 + sigma_2_1 * (t-1)) for t in 1:T]
    sigma_3_vector = [exp(sigma_3_0 + sigma_3_1 * (t-1)) for t in 1:T]
    sigma_4_vector = [t < T_CHILD_VOICE ? 0.0 :
                      exp(sigma_4_0 + sigma_4_1 * (t - (T_CHILD_VOICE - 1))) for t in 1:T]

    
    #sigma_1_vector = [0.10 for t in 1:T]  # very small, but constant
    #sigma_2_vector = [0.10 + 0.01*(t-1) for t in 1:T] # slowly rising
    #sigma_3_vector = [0.30 for t in 1:T]  # moderate persistence
    #sigma_4_vector = [t < T_CHILD_VOICE ? 0.0 : 0.10 + 0.01*(t-T_CHILD_VOICE) for t in 1:T]

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

    # All randomness is seeded off the `seed` kwarg, which was previously accepted and
    # then ignored (these seeds were hardcoded). Arms constructed with the same seed share
    # initial conditions AND shock paths -- common random numbers -- so a difference
    # between arms is a treatment effect rather than a reshuffle.
    rng_a  = MersenneTwister(seed)
    rng_k  = MersenneTwister(seed + 1)
    rng_hc = MersenneTwister(seed + 2)
    rng_p  = MersenneTwister(seed + 3)
    sim_a_init = rand(rng_a, LogNormal(0.2962227, 1.401793), simN)
    sim_k_init = Float64.(rand(rng_k, Bernoulli(0.3), simN))  # 70% zeros, 30% ones
    sim_hc_init = rand(rng_hc, simN) .* 1;
    sim_p_init = fill(ceil(Int, Np/2), simN)
    # Pre-drawn uniforms for the AR(1) transition: reproducible, and identical across arms.
    # Previously `sample(...)` was called against the GLOBAL RNG.
    draws_uniform_p = rand(rng_p, simN, simT)



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
    sim_a_init, sim_k_init, sim_hc_init, sim_p_init, draws_uniform_p, seed,
    w_vec, nothing, β0, β_bothcollege, β_age, β_age2, β_age2_capital, β_age_capital)
end


# -----------------------------------------------------------------------------
# Solver: backward induction, objectives, utilities, constraints
#   (was notebook cell 15)
# -----------------------------------------------------------------------------

# === Put near the top of your file ===
const TOL_CONSTR = 1e-8
"""
    T_CHILD_VOICE

First period in which the child is a decision maker. Periods `1 .. T_CHILD_VOICE-1` are
childhood: the parents choose alone over `(c_p, e_p, h_p, tau_p)` and the child's study
time is absent from HC production. From `T_CHILD_VOICE` the child bargains, `tau_c` enters
the choice set, and the welfare weight starts falling from 1.

`T_CHILD_VOICE = 6` means the parent-only periods are `t = 1..5` and the child's own study
time enters from `t = 6`, which is child age 6 under the model's `t <-> age` indexing
(`t = 17` is age 17, the last period before separation at 18).

Six things key off this boundary: the two backward-induction loops, `mu_vector`,
`sigma_4_vector`, and the two simulators' HC-technology branch. They are derived from this
constant rather than written out, because they were six scattered literals and changing the
boundary meant finding all of them.
"""
#     The human-capital block, recalibrated 2026-08-07
#
# `sigma_3_1` was **+0.06**, so self-productivity ROSE with age: 0.09 at t=1 to 0.24 at t=17.
# The persistence chain from an early investment to T is the product of ~16 such terms --
# about `0.15^16 = 1e-14` -- so early human capital had no memory at all and there was no
# reason to front-load. Parental time came out RISING, 0.004 to 0.058, when it should fall.
#
# Fixing the sign alone was not enough: the LEVEL had to come up too, and raising it exposed
# that everything else in the block was scaled to a technology that destroyed skill (HC fell
# from 0.491 to 0.434 between t=1 and t=2, because every input is far below 1 and Cobb-Douglas
# maps `R = 2.54` down to 0.855). So the block moved together:
#
#     sigma_3_0  -2.40 -> -0.36    self-productivity 0.09 -> 0.70, now FLAT (sigma_3_1 = 0)
#     sigma_1_0  -1.80 -> -0.90    parental-time elasticity 0.165 -> 0.41
#     sigma_4_0  -3.50 -> -4.50    child-study elasticity 0.030 -> 0.011
#     R_0         2.00 ->  1.60    TFP, retuned to keep HC inside hc_grid; R_1 0.06 -> 0
#     phi_3_0     0.03 ->  1.00    parent's weight on log HC
#     lambda_2_0  0.30 ->  1.00    child's weight on log HC
#     phi_2_0     0.80 ->  0.50    leisure weight, to hold labor supply at its target
#     psi_terminal 1.0 ->  4.00    in the CHILD module -- see below
#
# `psi_terminal` had to move with `phi_3`/`lambda_2`. Left at 1.0 while the flow weight went
# to 1.0, the last period valued skill far less than every earlier one and tau_p collapsed at
# t = 17 (0.059 against 0.155 once psi rose).
#
# Resulting profile, against the 0.40 -> 0.20 target:
#
#     t          1      5      9     13     17
#     tau_p  0.392  0.352  0.310  0.268  0.191
#     i_c    ~0.14           h_p 0.290 (was 0.285)      HC 1.76 -> 1.95
#
# CAVEAT, and it is the honest cost of hitting 0.40: sigma_1 = 0.41 makes the input
# elasticities sum to 1.29 with sigma_3 = 0.70, i.e. increasing returns. Cunha-Heckman-
# Schennach-style technologies put self-productivity at 0.85-0.95 and investment elasticities
# at 0.05-0.20, summing to about 1. At the optimum `sigma_1 * (value share of HC) = tau_p *
# (price of time)`, so asking the parent to spend 40% of their time on the child REQUIRES
# either a large sigma_1 or a large skill valuation -- it is a property of the model, not of
# the solver. If tau_p nearer 0.15 -> 0.08 is acceptable, sigma_1 can sit at ~0.15, squarely
# in the literature range.
#
const T_CHILD_VOICE = 6

"""
    TIME_FLOOR

Lower bound on the time shares `tau_p`, `tau_c` and `h_p` in every parent optimization.

The Cobb-Douglas production logs give `d(HC_next)/dx = HC_next * sigma_j / x`, unbounded as
`x -> 0`. At a floor of 1e-4 that reached |grad| = 2e4 while SLSQP was exploring the corner,
which is enough to wreck the BFGS model and return a NaN iterate -- the same failure mode as
the leisure cliff, from the production logs rather than the utility ones.

1e-3 is **strictly slack at every solved state**, so it cannot change a solution: over 61,200
states the stored minima are `tau_p` 0.0152, `i_c` 0.0055, `h_p` 0.1711, and none sits at or
below 1e-3. It just caps the 1/x factor 10x lower. A floor of 1e-2 would NOT be slack --
`i_c` is below it at 1.08% of states.

This is a bound on exploration, not economics: `sigma_j * log(x)` makes `x` a good, so
`df/dx -> +inf` as `x -> 0` and the true optimum is always interior.
"""
const TIME_FLOOR = 1e-3

const WAGE_SCALING_FACTOR = 0.584 # e.g., Adjustment for hours worked per year

# P4: child leisure is the ONLY quantity SLSQP can drive non-positive -- c, i_c, e_p, t_p
# and h_p are all held positive by box bounds, whereas leisure_c = 1 - t_p - i_c is a
# NONLINEAR constraint, which SLSQP is free to violate at trial points.
#
# The old code returned a flat -1e8 there while still computing the gradient from the
# smooth formula, so objective and gradient described different functions and the line
# search accepted steps that worsened the objective. Instead the objective is floored and
# the gradient was floored to match: below LEISURE_FLOOR the objective was constant in
# leisure, so its derivative was exactly zero.
#
# That was consistent but not enough. `d/dl [lambda*log(max(l, L))]` is `lambda/l` above L
# and 0 below, so the derivative CLIFFS by lambda/L at the floor. At L = 1e-8 that cliff is
# about 1.3e7, and SLSQP builds a BFGS quadratic model out of it: one step across the cliff
# and the iterate comes back NaN. Measured directly -- the parent solve died with
# |grad| = 1.28e7 at leisure_c = 0 (t_p = 0.3135, i_c = 0.6865, which sum to exactly 1).
#
# So the log is LINEARIZED below the floor instead of flattened:
#
#     l >= L :  log(l)                    d/dl = 1/l
#     l <  L :  log(L) + (l - L)/L        d/dl = 1/L
#
# Value and slope both match at l = L, so the pair is C1 rather than merely continuous, and
# the derivative is bounded by 1/L everywhere. L is raised to 1e-4, capping the leisure term
# of the gradient at lambda*1e4. The floor is a numerical guard, not economics: it only
# applies at trial points that violate the nonlinear leisure constraint, and the optimum
# keeps leisure far above 1e-4 because lambda*log(l) -> -Inf as l -> 0.
#
# P10: the floor is 1e-2, not 1e-4, because the PARENT's leisure now enters as CRRA with
# curvature `eta`. The bounded derivative below the floor is L^(-eta); at eta = 2 that is
# 1e8 with L = 1e-4 but 1e4 with L = 1e-2. Verified slack -- see the check in the docstring
# of `util_total`.
const LEISURE_FLOOR = 1e-2

"""
    crra_leisure(l, nu)   /   d_crra_leisure(l, nu)

`l^(1-nu)/(1-nu)`, or `log(l)` at `nu = 1`, LINEARIZED below `LEISURE_FLOOR`:

    l >= L :  l^(1-nu)/(1-nu)                        d/dl = l^(-nu)
    l <  L :  L^(1-nu)/(1-nu) + L^(-nu) * (l - L)    d/dl = L^(-nu)

Value and slope both match at `L`, so the pair is C1 rather than merely continuous, and the
derivative is bounded by `L^(-nu)` everywhere. Flattening below the floor instead -- the
earlier version -- left a derivative CLIFF of `1/L` there, and one SLSQP step across it came
back NaN.
"""
@inline function crra_leisure(l::Float64, nu::Float64)
    L = LEISURE_FLOOR
    base(x) = nu == 1.0 ? log(x) : x ^ (1.0 - nu) / (1.0 - nu)
    return l >= L ? base(l) : base(L) + L ^ (-nu) * (l - L)
end
@inline d_crra_leisure(l::Float64, nu::Float64) = (l >= LEISURE_FLOOR ? l : LEISURE_FLOOR) ^ (-nu)

# The CHILD's leisure stays logarithmic, per model.txt.
@inline log_leisure(l::Float64)   = crra_leisure(l, 1.0)
@inline d_log_leisure(l::Float64) = d_crra_leisure(l, 1.0)


"""
    budget_ceiling(model, assets, capital, t, p_shock)

The most the parent could possibly spend this period: all wealth, all labor income at
`h = 1`, plus transfers. `c` and `e` are bounded by this rather than by a constant.

A fixed box of `[0, 100]` on each let SLSQP evaluate `c = e = 100` at a state holding
`a = 0.25`, giving `a_next = -174` against a grid starting at 0. The continuation is then
`Line()`-extrapolated 174 units past its edge, which returned `dV/dHC = -45` -- wrong sign,
wrong magnitude -- and the resulting gradient of 6e4 destroyed the BFGS model and came back
as a NaN iterate. The asset constraint is nonlinear, so SLSQP is free to violate it; the
box is what has to keep the iterates near the feasible set.

Same fix, and the same reasoning, as `c_hi` in the child's `solve_model_work!`.
"""
@inline function budget_ceiling(model::Parent_child_interaction_age_specific_AR1,
                                assets::Float64, capital::Float64, t::Int, p_shock::Float64)
    w = wage_func(model, capital, t, p_shock)
    return max((1.0 + model.r) * assets + model.tax_lambda * w ^ (1 - model.tau) + model.y, 0.02)
end

"""
    snap_parent(x, lo, hi; tol = 1e-10)

Clamp only floating-point-sized violations of `[lo, hi]`; leave genuine excursions visible
so the diagnostics can report them. See `snap` in child_lifecycle.jl.
"""
@inline function snap_parent(x::Float64, lo::Float64, hi::Float64; tol::Float64 = 1e-10)
    x < lo && x > lo - tol && return lo
    x > hi && x < hi + tol && return hi
    return x
end

# --------------------------
# Model Solver
# --------------------------
"""
    solve_model!(model; min_converged = 0.95, verbose = true)

Backward induction for the parent problem.

Returns a `Vector{NamedTuple}` of per-period solver diagnostics and **throws** if the
converged share in any period falls below `min_converged`. Printing alone was not enough:
the notebook wraps every counterfactual in `@suppress_output`, which sent
`print_period_stats` to /dev/null, so 30+ models were solved with no idea how many grid
points converged. A returned value plus a hard floor cannot be suppressed.
"""
function solve_model!(model::Parent_child_interaction_age_specific_AR1;
                      min_converged::Float64 = 0.95, verbose::Bool = true)
    diagnostics = NamedTuple[]
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
        bmax = budget_ceiling(model, assets, capital, t, p_shock)
        lower_bounds!(opt, [1e-4, TIME_FLOOR, 1e-4, TIME_FLOOR, TIME_FLOOR])
        upper_bounds!(opt, [bmax, 1.0, bmax, 1.0, 1.0])
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
        # Validity is checked FIRST. Previously this sat in an elseif after the
        # converged/maxeval branches, so a result reporting :FTOL_REACHED with a NaN
        # iterate slipped through; and the fallback replaced x_opt without recomputing
        # minf, storing a value that did not match the stored policy.
        if any(!isfinite, x_opt) || !isfinite(minf)
            error("Non-finite solver result at t=$t, i_a=$i_a, i_k=$i_k, i_hc=$i_hc, " *
                  "i_p=$i_p (ret=$ret, minf=$minf, x=$x_opt)")
        end
        if rt == "converged"
            converge_count += 1
        elseif rt == "maxeval"
            maxeval_count += 1
        else
            other_dict[ret] = get(other_dict, ret, 0) + 1
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
    push!(diagnostics, record_period!(t, converge_count, maxeval_count, other_dict,
                                      itercounts, total, min_converged, verbose))

    # ----- Earlier periods (t = T-1 down to 8) -----
    for t in (T-1):-1:T_CHILD_VOICE
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
            bmax = budget_ceiling(model, assets, capital, t, p_shock)
            lower_bounds!(opt, [0.01, TIME_FLOOR, 0.01, TIME_FLOOR, TIME_FLOOR])
            upper_bounds!(opt, [bmax, 1.0, bmax, 1.0, 1.0])
            inequality_constraint!(opt, constraint_min_leisure_full, TOL_CONSTR)
            inequality_constraint!(opt, constraint_child_time, TOL_CONSTR)
            inequality_constraint!(opt, (x, grad) -> asset_constraint_full(x, grad, model, capital, t, assets, p_shock), TOL_CONSTR)
            inequality_constraint!(opt, (x, grad) -> asset_constraint_max(x, grad, model, capital, t, assets, p_shock), TOL_CONSTR)

            min_objective!(opt, obj_wrapper)
            # Warm start from t+1, CLAMPED into this period's box. An out-of-box initial
            # guess makes NLopt return :INVALID_ARGS with NaN, which then propagates
            # backwards through every earlier period -- the same failure already fixed on
            # the child side, where terminal consumption exceeded a hardcoded cap.
            lo = [0.01, TIME_FLOOR, 0.01, TIME_FLOOR, TIME_FLOOR]
            hi = [bmax, 1.0, bmax, 1.0, 1.0]
            init = clamp.([
                model.sol_c[t+1, i_a, i_k, i_hc, i_p],
                model.sol_i[t+1, i_a, i_k, i_hc, i_p],
                model.sol_e[t+1, i_a, i_k, i_hc, i_p],
                model.sol_h[t+1, i_a, i_k, i_hc, i_p],
                model.sol_t[t+1, i_a, i_k, i_hc, i_p],
            ], lo, hi)
            xtol_rel!(opt, 1e-4)
            maxeval!(opt, 1000)
            (minf, x_opt, ret) = optimize(opt, init)
            
            push!(itercounts, opt.numevals)
            if any(!isfinite, x_opt) || !isfinite(minf)
                error("Non-finite solver result at t=$t, i_a=$i_a, i_k=$i_k, i_hc=$i_hc, " *
                      "i_p=$i_p (ret=$ret, minf=$minf, x=$x_opt)")
            end
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
            model.sol_i[t, i_a, i_k, i_hc, i_p] = x_opt[2]
            model.sol_e[t, i_a, i_k, i_hc, i_p] = x_opt[3]
            model.sol_h[t, i_a, i_k, i_hc, i_p] = x_opt[4]
            model.sol_t[t, i_a, i_k, i_hc, i_p] = x_opt[5]
            model.sol_tr[t, i_a, i_k, i_hc, i_p] = 0.0
            model.sol_t_asset[t, i_a, i_k, i_hc, i_p] = 0.0
            model.sol_v[t, i_a, i_k, i_hc, i_p] = -minf
        end
        push!(diagnostics, record_period!(t, converge_count, maxeval_count, other_dict,
                                          itercounts, total, min_converged, verbose))
    end

    # ----- Parent-only periods (t = 7 down to 1) -----
    for t in (T_CHILD_VOICE - 1):-1:1
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
            # T9: the time floors were 1e-6 here against 1e-4 in the full-model loop, for
            # the same variables. grad[4] carries HC_next * sigma_1 / t_p, so a floor of
            # 1e-6 lets that term reach 1e6 -- the same gradient blow-up that killed the
            # full-model loop through the leisure floor, 100x worse. Both loops now use
            # 1e-4. The floor is slack at the optimum: Cobb-Douglas HC production sends
            # HC_next -> 0 as t_p -> 0, and log(HC) is in utility, so the optimum keeps
            # t_p far away from it.
            bmax = budget_ceiling(model, assets, capital, t, p_shock)
            lower_bounds!(opt, [0.01, 0.01, TIME_FLOOR, TIME_FLOOR])
            upper_bounds!(opt, [bmax, bmax, 1.0, 1.0])
            inequality_constraint!(opt, constraint_min_leisure_parentonly, TOL_CONSTR)
            inequality_constraint!(opt, (x, grad) -> asset_constraint_parentonly(x, grad, model, capital, t, assets, p_shock), TOL_CONSTR)
            inequality_constraint!(opt, (x, grad) -> asset_constraint_max_parentonly(x, grad, model, capital, t, assets, p_shock), TOL_CONSTR)
            min_objective!(opt, obj_wrapper)
            # Clamped for the same reason as the full-model loop above.
            init = clamp.([
                model.sol_c[t+1, i_a, i_k, i_hc, i_p],
                model.sol_e[t+1, i_a, i_k, i_hc, i_p],
                model.sol_h[t+1, i_a, i_k, i_hc, i_p],
                model.sol_t[t+1, i_a, i_k, i_hc, i_p],
            ], [0.01, 0.01, TIME_FLOOR, TIME_FLOOR], [bmax, bmax, 1.0, 1.0])
            xtol_rel!(opt, 1e-4)
            maxeval!(opt, 1000)
            (minf, x_opt, ret) = optimize(opt, init)

            push!(itercounts, opt.numevals)
            # P6: this loop stored results unchecked; only the terminal and adolescence
            # loops were guarded, so P6 was only partially fixed.
            if any(!isfinite, x_opt) || !isfinite(minf)
                error("Non-finite solver result at t=$t, i_a=$i_a, i_k=$i_k, i_hc=$i_hc, " *
                      "i_p=$i_p (ret=$ret, minf=$minf, x=$x_opt)")
            end
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
        push!(diagnostics, record_period!(t, converge_count, maxeval_count, other_dict,
                                          itercounts, total, min_converged, verbose))
    end

    return diagnostics
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
        dutil_di_c = -(1 - model.mu_vector[t]) * model.lambda_1_vector[t] * d_log_leisure(leisure_c)
        dutil_de_p = 0.0
        dutil_dl_p = - model.phi_2_vector[t] * d_crra_leisure(leisure_p, model.eta)   # P10
        dutil_dh_p = dutil_dl_p
        dutil_dt_p = -(1 - model.mu_vector[t]) * model.lambda_1_vector[t] * d_log_leisure(leisure_c) + dutil_dl_p

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

        # P4: no finite sentinel. Returning -1e12 with a -1e12 gradient produced a value
        # every downstream finiteness check accepts, laundering a NaN gradient into a
        # "valid" result. A NaN here means the objective was evaluated where it is not
        # defined -- a bug to surface, not a state to price.
        if any(isnan, grad)
            error("obj_last_period_full non-finite gradient at c_p=$c_p, i_c=$i_c, " *
                  "e_p=$e_p, h_p=$h_p, t_p=$t_p (grad=$grad)")
        end
    end

    # Checked unconditionally: NLopt may request a value without a gradient.
    if !isfinite(f)
        error("obj_last_period_full non-finite value at c_p=$c_p, i_c=$i_c, e_p=$e_p, " *
              "h_p=$h_p, t_p=$t_p (f=$f)")
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
        dutil_dl_p = - model.phi_2_vector[t] * d_crra_leisure(leisure_p, model.eta)   # P10
        term_leisure_c = -(1 - model.mu_vector[t]) * model.lambda_1_vector[t] * d_log_leisure(leisure_c)
        marginal = model.tax_lambda * (1 - model.tau) * labor_pre ^ (- model.tau) * w
        grad[1] = dutil_dc_p + model.beta_vector[t] * dV_da_sum * (-1)
        grad[2] = term_leisure_c + model.beta_vector[t] * dV_dHC_sum * (HC_next * model.sigma_4_vector[t] / i_c)
        grad[3] = model.beta_vector[t] * (-dV_da_sum + dV_dHC_sum * (HC_next * model.sigma_2_vector[t] / e_p))
        # P1: no dV_dk_sum. `k` is the fixed BothCollege indicator (k_next = capital), so
        # d k_next / d h_p = 0. The term was a leftover from when k was parental human
        # capital accumulating by learning-by-doing; it told the optimizer that working
        # more makes you college-educated.
        grad[4] = dutil_dl_p + model.beta_vector[t] * (dV_da_sum * marginal)
        grad[5] = term_leisure_c + dutil_dl_p +
                  model.beta_vector[t] * dV_dHC_sum * (HC_next * model.sigma_1_vector[t] / t_p)
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
        # P1: no dV_dk_sum -- see obj_work_period_full.
        dutil_dl_p = -model.phi_2_vector[t] * d_crra_leisure(leisure, model.eta)   # P10
        grad[3] = dutil_dl_p + model.beta_vector[t] * (marginal * dV_da_sum)
        grad[4] = dutil_dl_p + model.beta_vector[t] * dV_dHC_sum * (HC_next * model.sigma_1_vector[t] / t_p)
    end
    return util_now + model.beta_vector[t] * V_next
end

# ------------------------------------------------
# Utility Functions
# ------------------------------------------------

@inline function util_total(model::Parent_child_interaction_age_specific_AR1, c::Float64, h_p::Float64,
                            t_p::Float64, i_c::Float64, HC::Float64, t::Int)
    leisure_c = 1.0 - t_p - i_c
    # c and i_c are held positive by box bounds; leisure_c is not (see LEISURE_FLOOR).
    @assert c > 0.0 && i_c > 0.0 "util_total: box bounds violated (c=$c, i_c=$i_c)"
    rho = model.rho
    eta = model.eta
    u_cons = model.phi_1_vector[t] * (c ^ (1.0 - rho) / (1.0 - rho))
    # P10: the parent's own leisure, restored. It was replaced by a Frisch labor disutility
    # -phi_2*h^(1+eta)/(1+eta), which has no tau_p in it at all -- so parental time with the
    # child was free, and util_parent returned the identical value at tau_p = 0.05 and 0.90.
    u_leisure = model.phi_2_vector[t] * crra_leisure(1.0 - h_p - t_p, eta)
    u_parent = u_cons + u_leisure
    u_child  = model.mu_vector[t] * model.phi_3_vector[t] * log(HC) +
            (1 - model.mu_vector[t]) * (model.lambda_1_vector[t] * log_leisure(leisure_c) +
                                        model.lambda_2_vector[t] * log(HC))
    return u_parent + u_child
end

@inline function util_parent(model::Parent_child_interaction_age_specific_AR1, c::Float64, h_p::Float64, t_p::Float64, HC::Float64, t::Int)
    @assert c > 0.0 "util_parent: box bound violated (c=$c)"
    rho = model.rho
    eta = model.eta
    u_cons = model.phi_1_vector[t] * (c ^ (1.0 - rho) / (1.0 - rho))
    u_leisure = model.phi_2_vector[t] * crra_leisure(1.0 - h_p - t_p, eta)   # P10
    return u_cons + u_leisure + model.phi_3_vector[t] * log(HC)
end



# ------------------------------------------------
# Human Capital Functions
# ------------------------------------------------

@inline function HC_technology_full(model::Parent_child_interaction_age_specific_AR1, t_p::Float64, e_p::Float64, HC::Float64, i_c::Float64, t::Int)
    # P4: returning -1e8 as a human-capital LEVEL was doubly wrong -- it is not a value,
    # and it was then multiplied by sigma/t_p inside the gradient. Bounds keep t_p, e_p > 0.
    @assert t_p > 0.0 && e_p > 0.0 "HC_technology_full: box bounds violated (t_p=$t_p, e_p=$e_p)"
    return exp(log(model.R_vector[t]) +
        model.sigma_1_vector[t] * log(t_p) +
        model.sigma_2_vector[t] * log(e_p) +
        model.sigma_3_vector[t] * log(HC)  +
        model.sigma_4_vector[t] * log(i_c))
end

@inline function HC_technology_parentonly(model::Parent_child_interaction_age_specific_AR1, t_p::Float64, e_p::Float64, HC::Float64, t::Int)
    # P4: see HC_technology_full.
    @assert t_p > 0.0 && e_p > 0.0 "HC_technology_parentonly: box bounds violated (t_p=$t_p, e_p=$e_p)"
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

"""
    PchipContinuation

The parent's continuation value: **C1 in `hc` without overshooting**, linear in `a`, and
linearly blended across the two `k` nodes.

Why `hc` specifically, and why shape-preserving.

`Gridded(Linear())` makes `dV/dhc` a STEP function -- constant inside each cell, jumping at
every node. `tau_p`'s first-order condition depends on `dV/dHC` directly, since that is the
only thing paying for parental time, so the optimal `tau_p` inherits the steps and adjacent
asset nodes whose `HC_next` lands in different cells get discretely different `tau_p`. That
is the ragged policy plot. Total variation of `tau_p` over the asset grid, everything else
fixed (TV/range is 1.0 for a monotone policy, so it measures pure wiggle):

    t                 16       15       14       10
    linear TV     0.0868   0.0490   0.0463   0.0432
    PCHIP  TV     0.0415   0.0113   0.0024   0.0028
    linear TV/rng   2.71     3.88     4.35     2.60
    PCHIP  TV/rng   1.37     1.48     1.46     2.74

Total variation falls by 2x to 19x. Count the direction reversals instead and the numbers
look unchanged -- but that is the wrong metric here: once the spurious variation is gone the
policy's whole range is ~1e-3, so what is left reverses at the solver's own `xtol_rel`
resolution. The amplitude is what moved.

Two other explanations were tested and ruled out first. Tightening the solver (`xtol_rel`
1e-4 -> 1e-10, `ftol_rel` 1e-13, `maxeval` 40x) changed the counts by at most 1, so it is
not solver noise. Refining `Nhc` 20 -> 40 -> 80 made it WORSE (5 -> 10 -> 15), which is what
a cell-boundary artefact does: more cells, more steps.

An interpolating **cubic** also fixes the raggedness, but it overshoots: it pushed `dV/dhc`
to 13.1 at the low-`hc` edge, and combined with the `HC_next * sigma_1 / tau_p` factor that
produced gradients around 2e4 and broke the `sigma_3_1 x 1.5` counterfactual with a NaN
iterate. PCHIP (Fritsch-Carlson monotone cubic Hermite) is C1 like the cubic but its node
slopes are bounded by the neighbouring secants, so it cannot amplify. It also interpolates
exactly, so the Bellman consistency residual is preserved, and it is faster than both the
cubic and the linear version because the solver wastes fewer iterations.

`a` stays linear: consumption's FOC is dominated by `u'(c)`, which is smooth and steep, and
the consumption policies were never the ragged ones.
"""
struct PchipContinuation
    ag::Vector{Float64}; kg::Vector{Float64}; hg::Vector{Float64}
    V::Array{Float64,3}          # (Na, Nk, Nhc)
    D::Array{Float64,3}          # dV/dhc at the nodes, Fritsch-Carlson limited
end

"""
    _pchip_slopes(x, y) -> d

Monotone cubic Hermite node slopes (Fritsch-Carlson). Where the data turn, the slope is set
to zero; elsewhere it is a weighted harmonic mean of the neighbouring secants, which is what
bounds it by them and rules out overshoot.
"""
function _pchip_slopes(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    n = length(x); d = zeros(n)
    n == 1 && return d
    h = diff(x); del = diff(y) ./ h
    if n == 2
        d .= del[1]; return d
    end
    for i in 2:(n-1)
        if del[i-1] * del[i] <= 0
            d[i] = 0.0
        else
            w1 = 2h[i] + h[i-1]; w2 = h[i] + 2h[i-1]
            d[i] = (w1 + w2) / (w1 / del[i-1] + w2 / del[i])
        end
    end
    # one-sided ends, clipped so they cannot exceed the adjacent secant
    d[1] = ((2h[1] + h[2]) * del[1] - h[1] * del[2]) / (h[1] + h[2])
    (d[1] * del[1] <= 0) ? (d[1] = 0.0) :
        ((del[1] * del[2] <= 0 && abs(d[1]) > abs(3del[1])) ? (d[1] = 3del[1]) : nothing)
    d[n] = ((2h[n-1] + h[n-2]) * del[n-1] - h[n-1] * del[n-2]) / (h[n-1] + h[n-2])
    (d[n] * del[n-1] <= 0) ? (d[n] = 0.0) :
        ((del[n-1] * del[n-2] <= 0 && abs(d[n]) > abs(3del[n-1])) ? (d[n] = 3del[n-1]) : nothing)
    return d
end

# Hermite value and slope on one hc cell; outside the grid, continue linearly from the
# boundary node so value and gradient come from the same line.
@inline function _herm(P::PchipContinuation, ia::Int, ik::Int, hc::Float64)
    hg = P.hg; n = length(hg)
    if hc <= hg[1]
        return (P.V[ia,ik,1] + P.D[ia,ik,1] * (hc - hg[1]), P.D[ia,ik,1])
    elseif hc >= hg[n]
        return (P.V[ia,ik,n] + P.D[ia,ik,n] * (hc - hg[n]), P.D[ia,ik,n])
    end
    i = clamp(searchsortedlast(hg, hc), 1, n-1)
    h = hg[i+1] - hg[i]; t = (hc - hg[i]) / h
    y0, y1 = P.V[ia,ik,i], P.V[ia,ik,i+1]
    d0, d1 = P.D[ia,ik,i], P.D[ia,ik,i+1]
    t2 = t*t; t3 = t2*t
    v  = (2t3 - 3t2 + 1)*y0 + (t3 - 2t2 + t)*h*d0 + (-2t3 + 3t2)*y1 + (t3 - t2)*h*d1
    dv = ((6t2 - 6t)*y0 + (3t2 - 4t + 1)*h*d0 + (-6t2 + 6t)*y1 + (3t2 - 2t)*h*d1) / h
    return (v, dv)
end

@inline function _cell(g::Vector{Float64}, x::Float64)
    n = length(g)
    n == 1 && return (1, 1, 0.0)
    i = clamp(searchsortedlast(g, x), 1, n-1)
    return (i, i+1, (x - g[i]) / (g[i+1] - g[i]))
end

function (P::PchipContinuation)(a::Float64, k::Float64, hc::Float64)
    ia, ja, wa = _cell(P.ag, a); ik, jk, wk = _cell(P.kg, k)
    v00, _ = _herm(P, ia, ik, hc); v10, _ = _herm(P, ja, ik, hc)
    v01, _ = _herm(P, ia, jk, hc); v11, _ = _herm(P, ja, jk, hc)
    return (1-wk)*((1-wa)*v00 + wa*v10) + wk*((1-wa)*v01 + wa*v11)
end

function Interpolations.gradient(P::PchipContinuation, a::Float64, k::Float64, hc::Float64)
    ia, ja, wa = _cell(P.ag, a); ik, jk, wk = _cell(P.kg, k)
    v00, d00 = _herm(P, ia, ik, hc); v10, d10 = _herm(P, ja, ik, hc)
    v01, d01 = _herm(P, ia, jk, hc); v11, d11 = _herm(P, ja, jk, hc)
    ha = P.ag[ja] - P.ag[ia]
    da = ha == 0 ? 0.0 : ((1-wk)*(v10 - v00) + wk*(v11 - v01)) / ha
    hk = P.kg[jk] - P.kg[ik]
    dk = hk == 0 ? 0.0 : (((1-wa)*v01 + wa*v11) - ((1-wa)*v00 + wa*v10)) / hk
    dh = (1-wk)*((1-wa)*d00 + wa*d10) + wk*((1-wa)*d01 + wa*d11)
    return (da, dk, dh)
end

function create_interp(model::Parent_child_interaction_age_specific_AR1, sol_v, t)
    ag, kg, hg = model.a_grid, collect(model.k_grid), model.hc_grid
    return [begin
        V = Array{Float64,3}(undef, model.Na, model.Nk, model.Nhc)
        D = similar(V)
        @inbounds for ik in 1:model.Nk, ia in 1:model.Na
            @views V[ia, ik, :] .= sol_v[t, ia, ik, :, i_p]
            @views D[ia, ik, :] .= _pchip_slopes(hg, V[ia, ik, :])
        end
        PchipContinuation(ag, kg, hg, V, D)
    end for i_p in 1:model.Np]
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

"""
    record_period!(...)

Build the per-period diagnostic record, print it when `verbose`, and throw if the
converged share is below `min_converged`.
"""
function record_period!(t, converge_count, maxeval_count, other_dict, itercounts, total,
                        min_converged::Float64, verbose::Bool)
    share = total == 0 ? 1.0 : converge_count / total
    rec = (period = t, total = total, converged = converge_count,
           converged_share = share, maxeval = maxeval_count,
           other = sum(values(other_dict); init = 0),
           other_codes = copy(other_dict),
           mean_iters = isempty(itercounts) ? 0.0 : mean(itercounts))
    verbose && print_period_stats(t, converge_count, maxeval_count, other_dict, itercounts, total)
    if share < min_converged
        error("Period $t: only $(round(100*share, digits=1))% of $(total) grid points " *
              "converged (floor $(round(100*min_converged, digits=1))%). " *
              "maxeval=$maxeval_count, other=$(rec.other) $(other_dict). " *
              "Refusing to return a solution built on failed optimizations.")
    end
    return rec
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
            sim_p[i, t+1] = discrete_draw(model.p_transition[current_state, :],
                                          model.draws_uniform_p[i, t])
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
            a_next_p = (1.0 + model.r) * a + sim_income[i, t] + model.y - sim_c[i, t] - sim_e[i, t]
            # Float-sized violations are snapped; genuine excursions are left visible so
            # check_simulation can report them. Clipping a real out-of-grid state would
            # silently rewrite the transition law.
            sim_a[i, t+1] = snap_parent(a_next_p, model.a_min, model.a_max)
            sim_k[i, t+1] = k 
            if t < T_CHILD_VOICE
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
    # The loops below are @inbounds, so an out-of-range index reads garbage memory and
    # crashes with a bus error rather than a BoundsError. Validate the caller-supplied
    # arrays first -- cheap, and it turns a silent memory fault into a clear message.
    @assert length(belief_type) == simN
        "belief_type has $(length(belief_type)) entries but the model simulates simN=$simN agents"
    @assert all(1 .<= belief_type .<= length(parent_models)) "belief_type holds indices outside 1:$(length(parent_models))"
    for (bi, pm) in enumerate(parent_models)
        @assert pm.simN >= simN "parent_models[$bi].simN = $(pm.simN) < simN = $simN"
        @assert length(pm.sim_p_init) >= simN "parent_models[$bi].sim_p_init is too short"
    end


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
            sim_p[i, t+1] = discrete_draw(pm.p_transition[current, :],
                                          model.draws_uniform_p[i, t])
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

            # P9: wage and tax come from the belief-specific `pm`, not the base model.
            # Policies were already taken from `pm`; mixing them is harmless only while
            # every belief model shares wage and tax parameters.
            wage = wage_func(pm, k, t, p_shock)
            sim_wage[i, t] = wage / WAGE_SCALING_FACTOR  # Store true wage (not scaled)
            labor_pre = wage * sim_h[i, t]
            after_tax = pm.tax_lambda * labor_pre ^ (1 - pm.tau)
            sim_income[i, t] = after_tax  # Store after-tax income

            sim_a[i, t+1] = snap_parent((1.0 + pm.r) * a + sim_income[i, t] + pm.y -
                                        sim_c[i, t] - sim_e[i, t], pm.a_min, pm.a_max)
            sim_k[i, t+1] = k

            if t < T_CHILD_VOICE
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

"""
    simulate_model_family_hetero!(base_child, child_models, belief_type, ...)

Heterogeneous beliefs about the return to college.

⚠️ CALLER CHANGE: `child_models[m]` must now differ in **`beta_E`**, the log college
wage premium, not in `college_boost`. `college_boost` no longer affects anything --
college buys `beta_E` in the wage rather than an increment to the human-capital
stock. Construct the bins with e.g. `ConSavLaborCollege_AR1(; beta_E = b_m, ...)`.
"""
function simulate_model_family_hetero!(
    base_child::ConSavLaborCollege_AR1,
    child_models::Vector{ConSavLaborCollege_AR1},
    belief_type::Vector{Int},
    verbose::Bool = true
)
    # -- Unpack grids/dims --
    @unpack simN, T, t_college, r, college_cost, a_min = base_child
    @unpack a_grid, k_grid, p_grid, p_transition, Np = base_child
    @unpack ap_grid = base_child                  # N13: transfer arrays live on this grid
    @unpack sim_a, sim_k, sim_c, sim_h, sim_income, sim_wage = base_child
    @unpack sim_p_idx, sim_a_init, sim_k_init, sim_p_init_idx, draws_uniform_p, y = base_child
    @unpack Nt, t_weight = base_child

    num_bins = length(child_models)
    # Beliefs are now about beta_E, the log college wage premium, rather than about
    # an increment to the human-capital stock. This is closer to Bleemer (2018), which
    # measures beliefs about the EARNINGS return to college, and it is what the wage
    # equation in child_lifecycle.jl actually contains. Decisions are still taken under
    # the biased number: the family enrols and transfers believing beta_E^m, the child
    # consumes through college believing it, and the truth arrives as a one-time
    # surprise at labour-market entry. What is no longer needed is the reconciliation
    # term k + b* + (T_E-1)(b* - b_m), which existed only to reconcile a perceived
    # STOCK with the true one over four years. With no stock there is no drift.
    # The belief now acts entirely through child_models[m], each solved at its own
    # beta_E: those supply interp_*_college_belief and sol_tr_*_college_belief, which
    # are what the enrolment and transfer decisions are read from. There is no longer
    # a belief term in any law of motion, so no belief vector is needed here.

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
        LinearInterpolation((ap_grid, k_grid), base_child.sol_tr_v_work[:, :, ip, 1]; extrapolation_bc=Flat())
        for ip in 1:Np
    ]
    sol_tr_work_interp = [
        LinearInterpolation((ap_grid, k_grid), base_child.sol_tr_work[:, :, ip, 1]; extrapolation_bc=Flat())
        for ip in 1:Np
    ]


    # -- College interpolators (belief-specific) --
    # C14: these were built over the whole asset grid, including the rows the college
    # solver left as NaN. Two separate feasibility masks apply, and each belief model
    # carries its own:
    #   * the child's college policies are NaN below a_req[t]        -> csl(m, t)
    #   * the transfer arrays are NaN below a_req[1] + delta_P       -> ip0[m]
    a_req_b = [compute_min_assets(child_models[m]) for m in 1:num_bins]
    csl = function (m, t)
        i0 = t <= child_models[m].t_college ? first_feasible_a(child_models[m], a_req_b[m], t) : 1
        return i0 === nothing ? (1:length(a_grid)) : (i0:length(a_grid))
    end
    col_min = [min_parent_assets_for_college(child_models[m]) for m in 1:num_bins]
    ip0 = Vector{Int}(undef, num_bins)
    for m in 1:num_bins
        i = first_feasible_parent_a(child_models[m])
        i === nothing && error("Belief bin $m: college transfer infeasible at every asset " *
                               "grid point (needs a >= $(col_min[m]))")
        i > length(ap_grid) - 1 && error("Belief bin $m: college transfer feasible at only " *
                                         "$(length(ap_grid) - i + 1) parental asset node(s); need 2")
        ip0[m] = i
    end

    interp_c_college_belief = [
        LinearInterpolation((a_grid[csl(m, t)], k_grid), child_models[m].sol_c_college[t, csl(m, t), :, ip, it]; extrapolation_bc=Flat())
        for m in 1:num_bins, it in 1:Nt, t in 1:t_college, ip in 1:Np
    ]
    interp_h_college_belief = [
        LinearInterpolation((a_grid[csl(m, t)], k_grid), child_models[m].sol_h_college[t, csl(m, t), :, ip, it]; extrapolation_bc=Flat())
        for m in 1:num_bins, it in 1:Nt, t in 1:t_college, ip in 1:Np
    ]
    # The graduate's working life is belief-specific too: each bin was solved at its own
    # beta_E, so it has its own post-graduation policies. eps-free, hence no `it`.
    interp_c_grad_belief = [
        LinearInterpolation((a_grid, k_grid), child_models[m].sol_c_grad[t, :, :, ip, 1]; extrapolation_bc=Flat())
        for m in 1:num_bins, t in 1:T, ip in 1:Np
    ]
    interp_h_grad_belief = [
        LinearInterpolation((a_grid, k_grid), child_models[m].sol_h_grad[t, :, :, ip, 1]; extrapolation_bc=Flat())
        for m in 1:num_bins, t in 1:T, ip in 1:Np
    ]

    sol_tr_v_college_interp_belief = [
        LinearInterpolation((child_models[m].ap_grid[ip0[m]:end], k_grid), child_models[m].sol_tr_v_college[ip0[m]:end, :, ip, it]; extrapolation_bc=Flat())
        for m in 1:num_bins, it in 1:Nt, ip in 1:Np
    ]
    sol_tr_college_interp_belief = [
        LinearInterpolation((child_models[m].ap_grid[ip0[m]:end], k_grid), child_models[m].sol_tr_college[ip0[m]:end, :, ip, it]; extrapolation_bc=Flat())
        for m in 1:num_bins, it in 1:Nt, ip in 1:Np
    ]

    # -- Draw taste shock nodes --
    # N15: was MersenneTwister(2222) against the child simulators' 123, so a homogeneous
    # and a heterogeneous run of the same model assigned different agents to different
    # taste-shock nodes. Both now read base_child's one stored draw set.
    cum_weights = cumsum(t_weight); cum_weights ./= cum_weights[end]
    # C7: clamp(nothing, ...) is a MethodError, so the old guard did not guard.
    eps_indices = [clamp(something(findfirst(w -> w ≥ base_child.draws_uniform_t[i], cum_weights), Nt), 1, Nt)
                   for i in 1:simN]

    # -- Assign initial path and transfer --
    path_choice = Vector{Symbol}(undef, simN)
    tr_initial = Vector{Float64}(undef, simN)
    for i in 1:simN
        m = belief_type[i]
        it = eps_indices[i]
        ip = sim_p_init_idx[i]
        parent_assets = sim_a_init[i]
        HC = sim_k_init[i]
        # C14: below col_min[m] the college branch was never solved -- -Inf, as in
        # discrete_college_choice, rather than an extrapolation of the feasible slice.
        # The kappa_ParEd term in the psychic cost is additive and constant across the
        # college years, so it enters here as a closed-form value offset rather than as
        # an extra state. base_child carries the true kappa_ParEd; the belief concerns
        # beta_E only, not the psychic cost.
        f_college = parent_assets >= col_min[m] ?
                    sol_tr_v_college_interp_belief[m, it, ip](parent_assets, HC) +
                        pared_value_offset(base_child, base_child.sim_bc_init[i]) : -Inf
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

            if path_choice[i] == :college && t <= t_college
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
                # ----- Working -----
                # A graduate's working life is solved with E = 1 into the college
                # arrays, so read it from there.
                if path_choice[i] == :college
                    c = interp_c_grad_belief[m, t, p_idx](a, k)
                    h = interp_h_grad_belief[m, t, p_idx](a, k)
                else
                    c = interp_c_work[t, p_idx](a, k)
                    h = interp_h_work[t, p_idx](a, k)
                end
                p_shock = p_grid[p_idx]
                # base_child carries the TRUE beta_E, so a graduate's realized wage is
                # the correct one: the belief only ever governed the decision.
                E_i = path_choice[i] == :college ? 1.0 : 0.0
                w_pre = wage_func(base_child, k, t, E_i, p_shock)
                sim_wage[i, t] = w_pre / WAGE_SCALING_FACTOR
                sim_income[i, t] = after_tax_income(base_child, w_pre, h)
            end

            # Store
            sim_c[i, t] = c
            sim_h[i, t] = h

            # Update states for next period (if t < T)
            if t < T
                if path_choice[i] == :college && t <= t_college
                    a_next = (1 + r) * a - c - college_cost + y
                else
                    a_next = (1 + r) * a + sim_income[i, t] - c + y
                end
                # Human capital is fixed at theta for life, so there is no perceived
                # stock to drift from the true one and no correction to apply.
                k_next = k
                # C15: snap only float-sized violations, matching both child simulators.
                # `max(a_next, a_min)` rewrote the budget law by replacing a genuinely
                # negative asset with a_min, and left the upper domain unguarded.
                sim_a[i, t+1] = snap_parent(a_next, a_min, base_child.a_max)
                sim_k[i, t+1] = snap_parent(k_next, base_child.k_grid[1], base_child.k_max)

                # Persistent shock update
                p_draw = draws_uniform_p[i, t]
                p_trans_probs = p_transition[p_idx, :]
                sim_p_idx[i, t+1] = discrete_draw(p_trans_probs, p_draw)
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

