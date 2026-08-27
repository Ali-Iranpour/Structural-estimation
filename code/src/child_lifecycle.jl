# -------------------------------
# Utility: Nonlinear Grid Creator
# -------------------------------

# ===========================================================================
# Grid helpers
# ===========================================================================
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

# ===========================================================================
# Constants
# ===========================================================================


const WAGE_SCALING_FACTOR = 0.584

# ================================
# Progressive tax helpers
# ================================
# Model period t = 1 is biological age 18, so age = 17 + t. The age profile is
# calibrated in biological age (Daruich & Fernandez Table B3), not in model time.
const AGE_AT_T1 = 18
@inline model_age(t::Int) = AGE_AT_T1 + t - 1

# ===========================================================================
# Model: struct and constructor
# ===========================================================================

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
    eta::Float64; y::Float64; w::Float64; tau::Float64
    r::Float64; a_max::Float64; a_min::Float64; Na::Int; k_max::Float64
    Nk::Int; simT::Int; simN::Int
    a_grid::Vector{Float64}; k_grid::Vector{Float64}

    # N13: the transfer arrays are indexed on PARENTAL assets, which are a different
    # object from the child's assets even though both are wealth. Sharing one grid forced
    # a_min = 0 into the parental dimension, where the parent cannot retain delta_P,
    # a_term -> 0, and kappa_terminal*log(a_term) diverges.
    ap_grid::Vector{Float64}; Nap::Int; ap_min::Float64; ap_max::Float64

    psi_terminal::Float64       # Terminal value weight on human capital
    kappa_terminal::Float64     # Weight on parent's retained assets
    omega::Float64              # Weight on child's life-cycle utility
    mu::Float64                 # Weight on Parent's utility

    # Transitory shocks
    Nt::Int                     # Number of grid points for t
    t_grid::Vector{Float64}     # Grid for transitory shock t
    t_weight::Vector{Float64}   # Weights for t quadrature

    # --- Stochastic Shock Parameters (AR1 only) ---
    Np::Int; p_grid::Vector{Float64}; p_transition::Matrix{Float64}
    p_ar1::Float64; sigma_p::Float64

    # --- Solution arrays (5D: T, Na, Nk, Np, Nt) ---
    sol_c_work::Array{Float64, 5}; sol_h_work::Array{Float64, 5}; sol_v_work::Array{Float64, 5}
    # The GRADUATE's working life (E = 1). Held separately from the college arrays
    # because it is ep-free, like the work arrays: carrying the taste-shock dimension
    # on it wasted Nt-1 of every Nt entries.
    sol_c_grad::Array{Float64, 5}; sol_h_grad::Array{Float64, 5}; sol_v_grad::Array{Float64, 5}
    # The STUDY years only, t = 1..t_college. This is the one object that genuinely
    # varies in the taste shock, and only at t = 1, where eps_0 enters.
    sol_c_college::Array{Float64, 5}; sol_h_college::Array{Float64, 5}; sol_v_college::Array{Float64, 5}

    # --- Solution arrays (4D: Na, Nk, Np, Nt) ---
    sol_tr_college::Array{Float64, 4}; sol_tr_work::Array{Float64, 4}
    sol_tr_v_college::Array{Float64, 4}; sol_tr_v_work::Array{Float64, 4}

    # --- Simulation arrays ---
    sim_c::Matrix{Float64}; sim_h::Matrix{Float64}; sim_a::Matrix{Float64}
    sim_k::Matrix{Float64}; sim_p_idx::Matrix{Int}
    sim_a_init::Vector{Float64}; sim_k_init::Vector{Float64}; sim_p_init_idx::Vector{Int}
    # Parent's BothCollege indicator, for the kappa_ParEd term in the psychic cost.
    # Zeros unless the caller fills it (run_all.jl sets it from the parent block).
    sim_bc_init::Vector{Float64}
    sim_income::Matrix{Float64}; sim_wage::Matrix{Float64}
    draws_uniform_p::Matrix{Float64}
    # N15: ONE stored set of taste-shock draws, seeded from `seed`. The three simulators
    # used to draw their own -- MersenneTwister(123) in the two child simulators and
    # MersenneTwister(2222) in the heterogeneous parent one -- so arms that should have
    # differed only in the parameter under study also differed in who drew which eps.
    draws_uniform_t::Vector{Float64}
    college_cost::Float64
    # --- Psychic cost of college: kappa_0 + kappa_theta*log(theta)
    #                                       + kappa_ParEd*BothCollege
    kappa_0::Float64          # level
    kappa_theta::Float64      # ability gradient (NEGATIVE: ability lowers the cost)
    kappa_ParEd::Float64      # parental-education shift (NEGATIVE)

    # --- Wage process -----------------------------------------------------
    # ln w_t = lnw0 + beta_E*E + (alpha_theta + alpha_thetaE*E)*theta_tilde
    #                + (gamma1 + gamma1E*E)*age + (gamma2 + gamma2E*E)*age^2 + ln z
    # replacing w0*(1 + alpha*HC). See docs/WAGE_PROCESS.md.
    lnw0::Float64             # log base wage
    beta_E::Float64           # college intercept shift
    alpha_theta::Float64      # return to 1 SD of childhood HC, high school
    alpha_thetaE::Float64     # additional return for graduates
    gamma1::Float64           # age slope, high school
    gamma1E::Float64          # additional age slope for graduates
    gamma2::Float64           # age curvature, high school
    gamma2E::Float64          # additional curvature for graduates
    m_theta::Float64          # centring constant for log(theta); pure level shift
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
                # k_max = 8: HC no longer accumulates, so k only ever holds theta,
                # which comes from the parent block. Was 30, which wasted most of the
                # grid on states no agent visits. 8 rather than 6 because the higher
                # return to childhood HC raises parental investment: theta now reaches
                # 6.53 at production grids, past the parent's own hc_max of 6.0. See
                # docs/WAGE_PROCESS_IMPLEMENTED.md.
                #
                # k_max = 10 (was 8.0), MATCHED to the parent's hc_max. These two grids
                # hold the SAME object -- the child's human capital -- on either side of
                # the age-18 handoff (parent.sim_hc[:, T+1] -> child.sim_k_init), so a
                # mismatch silently clips the transfer and, worse, makes HC above the
                # child's ceiling worth ZERO to the parent's terminal problem, because
                # the terminal value spline is only defined over k_grid. At the current
                # sigma_3 = 0.407 the arriving distribution maxes at 4.8, so 10 clips
                # nobody; the match is what keeps the two blocks consistent under the
                # counterfactual arms, which push HC well above the baseline.
                simN::Int=5000, a_min::Float64=0.0, k_max::Float64=10.0, Nk::Int=30,
                w::Float64=12.5, tau::Float64=0.18, eta::Float64=2.0,
                phi::Float64=18.0, seed::Int=1234, college_cost::Float64=1.2,
                # --- Wage process ------------------------------------------------
                # Age profile and the college intercept: Daruich & Fernandez (2023)
                # Appendix Table B3, PSID 1968-2016, quadratic in BIOLOGICAL age,
                # estimated separately by education with a Heckman selection
                # correction. High school (0.0234, -0.000199, const 2.247) and
                # college (0.0552, -0.000513, const 1.953); the E terms are the
                # college-minus-high-school differences. The implied premium runs
                # 0.25 at age 22 to 0.51 at age 50.
                #
                # alpha_theta is an ELASTICITY of the wage with respect to childhood
                # human capital, which is what both papers estimate: they regress log
                # wages on LOG ability. Daruich & Fernandez Table B4 (PSID + NLSY, the
                # closer data to ours): 0.654 high school, 0.976 college, so
                # alpha_thetaE = 0.322 and the ratio is 1.49. Colas Table 2 gives
                # 0.31 / 0.47, ratio 1.52 -- same ratio, half the level.
                #
                # NOT standardized per SD. An earlier attempt divided by sd(log theta),
                # which made the investment incentive alpha_theta / s_theta; at the
                # dispersion the parent block actually produces (sd(log theta) ~ 0.2)
                # that is an elasticity near 0.9, above both papers, and parental
                # investment diverges -- mean theta went 2.11 -> 3.91 in one iteration
                # with no fixed point. As an elasticity there is no circularity at all:
                # m_theta is a pure level shift that lnw0 absorbs.
                # lnw0 is a pure NORMALIZATION, not a paper value: it fixes the units
                # of the wage. Set so the mean simulated wage matches the previous
                # w0*(1+alpha*HC) specification at the same grids, which keeps assets,
                # consumption and the tax base on their existing scale so the rest of
                # the calibration still applies. Re-derive if the profile parameters
                # move: shift lnw0 by log(target mean wage / simulated mean wage).
                lnw0::Float64=log(w) - 0.4144,
                beta_E::Float64=-0.294,
                alpha_theta::Float64=0.654,
                alpha_thetaE::Float64=0.322,
                gamma1::Float64=0.0234,
                gamma1E::Float64=0.0318,
                gamma2::Float64=-0.000199,
                gamma2E::Float64=-0.000314,
                # Centring constant only: the wage carries alpha_theta*(log theta -
                # m_theta), so m_theta shifts the level and nothing else. Set near
                # E[log theta] from the parent block so lnw0 reads as the log wage of a
                # child of average childhood human capital. Changing it is exactly
                # offset by lnw0 and has no behavioural content.
                m_theta::Float64=0.724,
                # --- Psychic cost of college -------------------------------------
                # Colas Table 2: kappa_0 + kappa_theta*log(theta) + kappa_fem*female
                #                + kappa_ParEd*ParEdu, with (starred entries are
                # 10,000x the parameter) kappa_0 = 0.44, kappa_theta = -8.3e-4,
                # kappa_ParEd = -1.7e-4. Signs and the RATIO transport; the LEVELS
                # cannot -- their utility is c^(1-gamma)/(1-gamma) at gamma = 1.9 with
                # consumption in dollars, ours is rho = 1.5 with c of order 0.1-5, so
                # a kappa of 4e-4 is meaningless here. The level is therefore set to
                # reproduce the total discounted psychic cost of the old
                # kappa/(HC+1)^4 form over the four college years, across the range of
                # theta the model actually visits; the ratio
                # kappa_ParEd / kappa_theta = 0.205 is Colas's.
                kappa_0::Float64=0.0462,
                kappa_theta::Float64=-0.0342,
                kappa_ParEd::Float64=-0.0070,
                # Shock parameters (AR1 only)
                p_ar1::Float64=0.95, sigma_p::Float64=0.2, Np::Int=5,
                # Preference shock parameters
                # Nt = 5 Gauss-Hermite nodes for the taste shock eps (was 11). GH with n
                # nodes is exact for degree 2n-1, and measured: Nt 10 -> 5 moves the
                # college share 51.5 -> 52.1 and nothing else at all.
                Nt=5, sigma_eps=0.5,
                # --- Terminal value parameters ---
                psi_terminal::Float64=4.0, kappa_terminal::Float64=10.0, omega::Float64=0.5,
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
                delta_P::Float64=0.01,
                # N13: parental asset grid. ap_min defaults to delta_P -- the smallest
                # parental asset at which the retained balance is still strictly positive,
                # so the terminal value is finite on every row and valid_rows is 1:Nap.
                Nap::Int=Na, ap_min::Float64=delta_P, ap_max::Float64=a_max
                )

    simT = T
    a_grid = create_focused_grid(a_min, 2.0, a_max, Na, 0.2, 1.3)
    k_grid = nonlinspace(0.001, k_max, Nk, 1.5)

    # N13/C14: parental asset grid. Starting at delta_P removes the singular row; putting
    # an exact node at the college transfer threshold removes the dead band between that
    # threshold and the first grid point above it (measured at 2.39 wide on the shared
    # grid), where a state was economically feasible but had no solved cell to interpolate.
    ap_min >= ap_max && error("ap_min ($ap_min) must be below ap_max ($ap_max)")
    ap_min < delta_P && error("ap_min ($ap_min) is below delta_P ($delta_P); the parent " *
                              "cannot retain its floor and the terminal value diverges")
    ap_grid = create_focused_grid(ap_min, ap_min + 2.0, ap_max, Nap, 0.2, 1.3)
    # Same recursion as compute_min_assets, so the node matches bit-for-bit.
    let a_req_1 = a_min
        for _ in 1:t_college
            a_req_1 = (a_req_1 + c_floor + college_cost - y) / (1 + r)
        end
        col_thr = a_req_1 + delta_P
        if ap_min < col_thr < ap_max
            ap_grid[argmin(abs.(ap_grid .- col_thr))] = col_thr
            sort!(ap_grid)
        end
    end

    # Gauss-Hermite quadrature for transitory shocks
    nodes, weights = gausshermite(Nt)
    t_grid = sqrt(2) * sigma_eps .* nodes
    t_weight = weights / sqrt(pi)

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

    # --- Initialize solution arrays (5D) ---
    # The taste shock eps_0 enters only at t = 1, on the college branch. Every other
    # solution slice is eps-free, so carrying Nt on all six arrays wasted about 88% of
    # them -- 306 MB of 350 MB at production grids. Three shapes instead of one:
    #   work / grad : eps-free working lives, trailing dimension 1
    #   college     : the study years only, where eps genuinely varies
    shape_wk = (T, Na, Nk, Np, 1)
    shape_cl = (t_college, Na, Nk, Np, Nt)
    sol_c_work = fill(NaN, shape_wk); sol_h_work = fill(NaN, shape_wk); sol_v_work = fill(NaN, shape_wk)
    sol_c_grad = fill(NaN, shape_wk); sol_h_grad = fill(NaN, shape_wk); sol_v_grad = fill(NaN, shape_wk)
    sol_c_college = fill(NaN, shape_cl); sol_h_college = fill(NaN, shape_cl); sol_v_college = fill(NaN, shape_cl)

    # --- Initialize half period solution arrays (5D) ---
    tr_shape = (Nap, Nk, Np, Nt)   # N13: parental asset dimension, not the child's
    sol_tr_college = fill(NaN, tr_shape); sol_tr_work = fill(NaN, tr_shape)
    sol_tr_v_college = fill(NaN, tr_shape); sol_tr_v_work = fill(NaN, tr_shape)

    # --- Initialize simulation arrays ---
    sim_shape = (simN, T)
    sim_c = fill(NaN, sim_shape); sim_h = fill(NaN, sim_shape)
    sim_a = fill(NaN, (simN, T+1)); sim_k = fill(NaN, sim_shape)
    sim_p_idx = fill(0, sim_shape)
    sim_income = fill(NaN, sim_shape); sim_wage = fill(NaN, sim_shape)

    rng = MersenneTwister(seed)
    # N13: sim_a_init holds PARENTAL assets, so it is drawn on the parental grid's domain.
    # Draws below ap_min previously landed on rows where the terminal value diverges.
    sim_a_init = ap_min .+ rand(rng, simN) .* (min(20.0, ap_max) - ap_min)
    sim_k_init = rand(rng, simN) .* 5
    sim_bc_init = zeros(simN)
    # C6 (Phase 0.5c): draw the initial child shock from the STATIONARY distribution --
    # the same distribution the transfer problem integrates over. Starting everyone at the
    # median state meant the transfer was optimal against a distribution no simulated child
    # was drawn from.
    sim_p_init_idx = let pi_p = stationary_dist(p_transition), u = rand(rng, simN)
        [discrete_draw(pi_p, u[i]) for i in 1:simN]
    end

    draws_uniform_p = rand(rng, sim_shape...)
    draws_uniform_t = rand(rng, simN)      # N15: shared across every simulator

    return ConSavLaborCollege_AR1(
        T, t_college, rho, beta, phi, eta, y, w, tau, r,
        a_max, a_min, Na, k_max, Nk, simT, simN, a_grid, k_grid,
        ap_grid, Nap, ap_min, ap_max,
        psi_terminal, kappa_terminal, omega, mu,
        Nt, t_grid, t_weight,
        Np, p_grid, p_transition, p_ar1, sigma_p,
        sol_c_work, sol_h_work, sol_v_work,
        sol_c_grad, sol_h_grad, sol_v_grad,
        sol_c_college, sol_h_college, sol_v_college,
        sol_tr_college, sol_tr_work, sol_tr_v_college, sol_tr_v_work,
        sim_c, sim_h, sim_a, sim_k, sim_p_idx,
        sim_a_init, sim_k_init, sim_p_init_idx, sim_bc_init, sim_income, sim_wage,
        draws_uniform_p, draws_uniform_t, college_cost,
        kappa_0, kappa_theta, kappa_ParEd,
        lnw0, beta_E, alpha_theta, alpha_thetaE, gamma1, gamma1E, gamma2, gamma2E,
        m_theta, tax_lambda,
        c_floor, delta_P
    )
end

# ===========================================================================
# Primitives: wage, tax, utility
# ===========================================================================

# Pre-tax hourly wage (no taxes here).
#
#   ln w = lnw0 + beta_E*E + (alpha_theta + alpha_thetaE*E)*(log theta - m_theta)
#               + (gamma1 + gamma1E*E)*age + (gamma2 + gamma2E*E)*age^2 + ln z
#
# alpha_theta is an ELASTICITY with respect to childhood human capital.
#
# `theta` is childhood human capital at 18, FIXED for life -- it no longer
# accumulates -- and `E` is the college indicator. Replaces w0*(1 + alpha*HC),
# which welded childhood skill, college and experience into one stock and priced
# all three with a single alpha. See docs/WAGE_PROCESS.md.
@inline function wage_func(model::ConSavLaborCollege_AR1, theta::Float64, t::Int,
                           E::Float64, p_shock::Float64)
    th  = log(max(theta, 1e-8)) - model.m_theta
    age = model_age(t)
    lw  = model.lnw0 +
          model.beta_E * E +
          (model.alpha_theta + model.alpha_thetaE * E) * th +
          (model.gamma1 + model.gamma1E * E) * age +
          (model.gamma2 + model.gamma2E * E) * age * age
    return exp(lw) * p_shock * WAGE_SCALING_FACTOR
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
    # kappa_X = kappa_0 + kappa_theta*log(theta), with kappa_theta < 0 so that
    # ability lowers the cost (Colas 2021, Daruich & Fernandez 2023 both use a log
    # form and both find a negative gradient). The parental-education term is
    # additive and constant over the college years, so it is applied once, as a
    # value offset at the college-vs-work comparison, rather than carried here.
    psychic_cost = model.kappa_0 + model.kappa_theta * log(max(k, 1e-8))
    return cons_utility - psychic_cost
end

"""
    pared_value_offset(model, bc)

Value offset from the `kappa_ParEd * BothCollege` term in the psychic cost.

The term is additive in utility and constant across the college years -- theta is
fixed, so nothing about it varies with t -- and it does not interact with
consumption. It therefore shifts the college value by a closed-form annuity and
leaves every college policy unchanged, so it can be applied once at the
college-vs-work comparison instead of being carried as an extra state. This is
exact, not an approximation.
"""
@inline function pared_value_offset(model::ConSavLaborCollege_AR1, bc::Float64)
    disc = sum(model.beta^s for s in 0:(model.t_college - 1))
    return -model.kappa_ParEd * bc * disc      # kappa_ParEd < 0 -> raises V_college
end

# ========== Terminal Value ==========
@inline function terminal_value(model::ConSavLaborCollege_AR1, k::Float64, a_terminal::Float64)
    @unpack psi_terminal, kappa_terminal = model
    return psi_terminal * log(k) + kappa_terminal * log(a_terminal)
end

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

"""
    college_premium_profile(model, E_age)   -- internal helper
    beta_E_from_rce(model, rce)

Convert a believed college/high-school earnings RATIO into the model's `beta_E`.

`rce` is what the belief survey elicits: a single number for how much more a college
graduate earns, e.g. 1.5 for "50% more". The model does not carry a single premium --
it carries a profile, `beta_E + gamma1E*age + gamma2E*age^2`, worth 0.25 at 22 and
0.51 at 50 -- so a scalar belief has to be anchored to a horizon. The anchor here is
the **unweighted career average over the graduate's working life**, ages
`t_college+1 .. T`, which is the natural reading of "relative college earnings" as one
number covering a career.

A believer's profile is the estimated one shifted in parallel:

    beta_E^m = beta_E* + [log(rce_m) - anchor*]

and since `anchor* = beta_E* + mean_a(gamma1E*a + gamma2E*a^2)`, the true intercept
cancels:

    beta_E^m = log(rce_m) - mean_a(gamma1E*a + gamma2E*a^2)

so the mapping depends only on the interaction terms, not on where `beta_E*` happens
to sit. Beliefs shift the LEVEL of the premium and leave its SHAPE at the estimated
one.

This replaces `college_boost = (rce - 1)/0.4`, which inverted the old
`w0*(1 + alpha*HC)` wage at `HC = 0` to express the same belief as a human-capital
increment. Under a log wage the belief is already in the model's units and no
inversion is needed -- which is also closer to Bleemer and Zafar (2018), who elicit
beliefs about the EARNINGS return.
"""
function college_premium_anchor(model::ConSavLaborCollege_AR1)
    ages = [model_age(t) for t in (model.t_college + 1):model.T]
    return sum(model.gamma1E * a + model.gamma2E * a^2 for a in ages) / length(ages)
end

beta_E_from_rce(model::ConSavLaborCollege_AR1, rce::Real) =
    log(rce) - college_premium_anchor(model)

beta_E_from_rce(model::ConSavLaborCollege_AR1, rce::AbstractVector) =
    [beta_E_from_rce(model, r) for r in rce]

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
# C7: `findfirst` returns `nothing` when the cumulative weights fall a floating-point
# hair short of the draw. Fall back to the last state rather than propagating `nothing`
# into an array index.
function discrete_draw(probs::AbstractVector{Float64}, draw::Float64)
    cdf = cumsum(probs)
    return something(findfirst(x -> x >= draw, cdf), length(cdf))
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

# ===========================================================================
# Constraints
# ===========================================================================

# ================================
# Constraints
# ================================
# C16: a' must also stay BELOW a_max, and k' below k_max. Constraining only a' >= a_min
# let the solver pick transitions off the top of the grid -- measured at 3.59% of stored
# asset transitions and 5.00% of HC transitions -- where the continuation value is a
# `Line()` extrapolation. Forward simulation reported 0%, so nothing caught it.
@inline function asset_constraint_work_upper(x::Vector, grad::Vector, model::ConSavLaborCollege_AR1,
    assets::Float64, capital::Float64, t::Int, E::Float64, p_shock::Float64)
    c, h = x[1], x[2]
    w_pre = wage_func(model, capital, t, E, p_shock)
    a_next = (1.0 + model.r) * assets + after_tax_income(model, w_pre, h) - c + model.y
    g = a_next - model.a_max
    if length(grad) > 0
        grad[1] = -1.0
        grad[2] = d_after_tax_dh(model, w_pre, h)
    end
    return g
end

# There is no k' constraint any more: human capital is fixed at theta, so k' = k
# can never leave the grid and k_max cannot bind through hours.

@inline function asset_constraint_work(x::Vector, grad::Vector, model::ConSavLaborCollege_AR1,
    assets::Float64, capital::Float64, t::Int, E::Float64, p_shock::Float64)
    c, h = x[1], x[2]
    w_pre = wage_func(model, capital, t, E, p_shock)
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

# ===========================================================================
# College feasibility
# ===========================================================================

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
    return findfirst(a -> a >= thr, model.ap_grid)   # N13: parental grid
end

# ===========================================================================
# Interpolation
# ===========================================================================

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

# ===========================================================================
# Objectives
# ===========================================================================

# ================================
# Objectives & constraints
# ================================

# --- Final period objective: work, consume everything (progressive tax) ---
@inline function obj_last_period(model::ConSavLaborCollege_AR1, h_vec::Vector, assets::Float64,
    capital::Float64, t::Int, E::Float64, p_shock::Float64, grad::Vector)
    h     = h_vec[1]
    w_pre = wage_func(model, capital, t, E, p_shock)
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
    t::Int, E::Float64, p_shock::Float64, i_p::Int, interp, grad::Vector)
    c, h = x[1], x[2]
    w_pre  = wage_func(model, capital, t, E, p_shock)
    y_lab  = after_tax_income(model, w_pre, h)               # λ (w h)^(1-τ)
    dy_dh  = d_after_tax_dh(model, w_pre, h)                 # λ (1-τ) w (w h)^(-τ)

    a_next = (1.0 + model.r) * assets + y_lab - c + model.y
    # Human capital no longer accumulates: k' = k = theta, fixed for life. The
    # dV/dk term that used to enter grad[2] is therefore gone -- it encoded
    # learning by doing, which this specification removes.
    k_next = capital

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

            V_next  += p_trans_prob * Vj
            gradV_c += p_trans_prob * (-dV_da)                      # ∂a_next/∂c = -1
            gradV_h += p_trans_prob * (dy_dh * dV_da)               # ∂a_next/∂h = dy_dh, ∂k_next/∂h = 0
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
    # College no longer adds to the stock; it buys beta_E in the wage instead.
    k_next = capital

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

# ===========================================================================
# Solver: backward induction
# ===========================================================================

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
"""
    solve_model_work!(model; E, sol_c, sol_h, sol_v, t_min, label)

The working-life problem for one education group. `E` is the college indicator,
which now enters only through the wage; it no longer has a counterpart in the law
of motion, because human capital does not accumulate.

Called twice. `E = 0` fills the work arrays over the whole horizon. `E = 1` fills
the college arrays from `t_college+1` onward, which is the graduate's working
life -- these used to be a straight copy of the high-school solution, so
graduates and non-graduates faced an identical wage at the same stock.
"""
function solve_model_work!(model::ConSavLaborCollege_AR1;
                           E::Float64 = 0.0,
                           sol_c::Array = model.sol_c_work,
                           sol_h::Array = model.sol_h_work,
                           sol_v::Array = model.sol_v_work,
                           t_min::Int = 1,
                           label::String = "working")
    @unpack T, Na, Nk, Np, a_grid, k_grid, p_grid = model
    @unpack y, c_floor = model

    # ---- Final period (t = T): work, consume everything, no bequest ----
    for i_p in 1:Np, i_k in 1:Nk, i_a in 1:Na
        assets, capital = a_grid[i_a], k_grid[i_k]
        p_shock = p_grid[i_p]

        function obj_wrapper(h_vec::Vector, grad::Vector)
            f = obj_last_period(model, h_vec, assets, capital, T, E, p_shock, grad)
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
        w_pre = wage_func(model, capital, T, E, p_shock)
        cons  = assets + after_tax_income(model, w_pre, h_opt) + y

        sol_h[T, i_a, i_k, i_p, :] .= h_opt
        sol_c[T, i_a, i_k, i_p, :] .= cons
        sol_v[T, i_a, i_k, i_p, :] .= -minf
    end

    # ---- Working ages (t = T-1 down to t_min) ----
    @showprogress 1 "Solving $label model..." for t in (T-1):-1:t_min
        interp = create_interpolator(model, sol_v, t + 1)
        for i_p in 1:Np, i_k in 1:Nk, i_a in 1:Na
            assets, capital = a_grid[i_a], k_grid[i_k]
            p_shock = p_grid[i_p]

            function obj_wrapper(x::Vector, grad::Vector)
                f = obj_work_period(model, x, assets, capital, t, E, p_shock, i_p, interp, grad)
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
            w_pre_t = wage_func(model, capital, t, E, p_shock)
            c_hi = max((1.0 + model.r) * assets + after_tax_income(model, w_pre_t, 1.0) + y, 0.02)

            # C16 is retired. It capped hours at k_max - capital so that k' = k + h
            # stayed on the grid, which meant a high-HC worker was forced to work
            # fewer hours because the grid ended, not for any economic reason. With
            # k' = k the ceiling cannot bind through hours at all.
            h_hi = 1.0

            opt = Opt(:LD_SLSQP, 2)
            lower_bounds!(opt, [c_floor, 1e-3])   # C17: the configured floor, not a literal
            upper_bounds!(opt, [c_hi, h_hi])
            ftol_rel!(opt, 1e-8)
            maxeval!(opt, 1000)
            inequality_constraint!(opt, (x, grad) -> asset_constraint_work(x, grad, model, assets, capital, t, E, p_shock), 1e-6)
            # C16: a' <= a_max. Always feasible -- at c = c_hi the whole budget is consumed
            # and a' = a_min -- so this one can stay a nonlinear constraint.
            inequality_constraint!(opt, (x, grad) -> asset_constraint_work_upper(x, grad, model, assets, capital, t, E, p_shock), 1e-6)
            min_objective!(opt, obj_wrapper)

            init = [clamp(sol_c[t + 1, i_a, i_k, i_p, 1], c_floor, c_hi),
                    clamp(0.4, 1e-3, h_hi)]
            (minf, x_opt, ret) = optimize(opt, init)
            check_nlopt!(ret, minf, x_opt, "$label t=$t ia=$i_a ik=$i_k ip=$i_p")
            sol_c[t, i_a, i_k, i_p, :] .= x_opt[1]
            sol_h[t, i_a, i_k, i_p, :] .= x_opt[2]
            sol_v[t, i_a, i_k, i_p, :] .= -minf
        end
    end
end

# ================================
# College-path solver (unchanged)
# ================================
"""
    solve_model_college!(model)

Fills the college arrays. Two stages:

  1. `t > t_college`: the GRADUATE's working life, solved as a work problem with
     `E = 1`. This used to be `sol_*_college .= sol_*_work`, which gave a
     graduate and a high-school worker with the same stock an identical wage --
     college paid only through the stock increment it added. College now pays
     through `beta_E` and the education-specific age profile instead, so the two
     working lives genuinely differ and both have to be solved.
  2. `t <= t_college`: the study years, consumption and saving only, with the
     continuation at `t_college+1` taken from stage 1.
"""
function solve_model_college!(model::ConSavLaborCollege_AR1)
    @unpack T, t_college, Na, Nk, Np, a_grid, k_grid, p_grid, c_floor = model
    @unpack sol_c_college, sol_h_college, sol_v_college, sol_v_work = model
    @unpack sol_c_grad, sol_h_grad, sol_v_grad = model

    # Stage 1: the graduate's working life, into its own eps-free arrays.
    solve_model_work!(model; E = 1.0,
                      sol_c = sol_c_grad, sol_h = sol_h_grad, sol_v = sol_v_grad,
                      t_min = t_college + 1, label = "graduate working")

    a_req = compute_min_assets(model)
    if a_req[1] > a_grid[end]
        error("College is infeasible at every asset grid point: a_req[1] = $(a_req[1]) > " *
              "a_max = $(a_grid[end]). Widen the grid or revisit college_cost / y / c_floor.")
    end

    # Stage 2: the study years. t > t_college is already solved above.
    @showprogress 1 "Solving college model..." for t in t_college:-1:1
        begin
            # Continuation: for t < t_college the next period is a college year and its
            # value is only defined on the feasible slice; at t == t_college the next
            # period is the work path, defined everywhere.
            # At t = t_college the next period is the GRADUATE's first working year,
            # which now lives in sol_v_grad.
            interp = t + 1 > t_college ?
                create_interpolator(model, model.sol_v_grad, t + 1) :
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
                # C16 (college branch): a' <= a_max, as a box LOWER bound on c. There is no
                # labor choice here, so a' is monotone in c alone and the ceiling is exactly
                # a floor on consumption. c_hi - c_lo = a_max - a_req[t+1] > 0, so the box is
                # never empty. At a = a_max the old bounds gave a' = 102.39 on a grid ending
                # at 100.
                c_lo = max(c_floor,
                           (1.0 + model.r) * assets - model.college_cost + model.y - model.a_max)
                c_lo = min(c_lo, c_hi)

                nodes = t == 1 ? (1:model.Nt) : (1:1)
                for i_t in nodes
                    eps_it = t == 1 ? model.t_grid[i_t] : 0.0

                    opt = Opt(:LD_SLSQP, 1)
                    lower_bounds!(opt, c_lo)
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

                    (minf, c_vec, ret) = optimize(opt, [clamp(0.13, c_lo, c_hi)])
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

# ===========================================================================
# Transfer stage
# ===========================================================================

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

# ========== Main Solvers ==========

function optimal_transfer_work!(model::ConSavLaborCollege_AR1)
    # N13: `assets` here is the PARENT's, so the outer loop runs over ap_grid. The child's
    # a_grid still indexes V1_work, which is evaluated at the transfer tr.
    @unpack Nap, Nk, a_grid, ap_grid, k_grid, p_transition, Np, delta_P = model
    π_p = stationary_dist(p_transition)

    V1_work = [extrapolate(interpolate((a_grid, k_grid), model.sol_v_work[1, :, :, ip, 1],
                                       Gridded(Linear())), Line()) for ip in 1:Np]

    for ia in 1:Nap, ik in 1:Nk
        assets = ap_grid[ia]
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
    # N13: `assets` is parental, so the loop runs over ap_grid; a_grid indexes V1_college.
    @unpack Nap, Nk, Nt, a_grid, ap_grid, k_grid, p_transition, Np, delta_P = model
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

    for ia in 1:Nap, ik in 1:Nk, it in 1:Nt
        assets = ap_grid[ia]
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

# ===========================================================================
# Terminal value handed to the parent
# ===========================================================================

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
    out = zeros(m.Nap, m.Nk)          # N13: parental asset dimension
    for it in 1:m.Nt
        w = m.t_weight[it]
        for ik in 1:m.Nk, ia in 1:m.Nap
            vw = m.sol_tr_v_work[ia, ik, ip, 1]
            vc = m.ap_grid[ia] >= a_col_min ? m.sol_tr_v_college[ia, ik, ip, it] : -Inf
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
    ok = [all(isfinite, view(V, ia, :)) for ia in 1:m.Nap]
    ia0 = findfirst(ok)
    ia0 === nothing && error("Terminal value is non-finite at every parental asset grid point")
    all(ok[ia0:end]) || error("Terminal value has interior non-finite rows: $(findall(.!ok))")
    return Spline2D(m.ap_grid[ia0:end], m.k_grid, V[ia0:end, :]; s = s)
end

"""
    valid_rows(m) -> UnitRange

Asset-grid rows over which the parent's terminal value is finite (assets above `delta_P`).
"""
function valid_rows(m::ConSavLaborCollege_AR1)
    V  = terminal_value_surface(m)
    ok = [all(isfinite, view(V, ia, :)) for ia in 1:m.Nap]
    ia0 = findfirst(ok)
    return ia0 === nothing ? (1:0) : (ia0:m.Nap)
end

# ===========================================================================
# Simulation
# ===========================================================================

# --------------------------
# Simulation (AR1 Shock Only + Shock-free Retirement)
# --------------------------
function simulate_model_child!(model::ConSavLaborCollege_AR1)
    @unpack simN, T, t_college, r, college_cost, a_min = model
    @unpack a_grid, k_grid, p_grid, p_transition = model
    @unpack sim_a, sim_k, sim_c, sim_h, sim_income, sim_wage = model
    @unpack sim_p_idx, sim_a_init, sim_k_init, sim_bc_init, sim_p_init_idx, draws_uniform_p, y = model
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
            for t in 1:t_college, i_p in 1:model.Np]
        for i_t in 1:Nt
    ]
    interp_h_college = [
        [LinearInterpolation((a_grid[csl(t)], k_grid), model.sol_h_college[t, csl(t), :, i_p, i_t]; extrapolation_bc=Flat())
            for t in 1:t_college, i_p in 1:model.Np]
        for i_t in 1:Nt
    ]
    interp_v_college = [
        [LinearInterpolation((a_grid[csl(1)], k_grid), model.sol_v_college[1, csl(1), :, i_p, i_t]; extrapolation_bc=Flat())
            for i_p in 1:model.Np]
        for i_t in 1:Nt
    ]

    # Work (per shock)
    # The graduate's working life: same shape as the work interpolators, read from
    # the grad arrays. Before the split these WERE the work arrays, so a graduate
    # and a high-school worker at the same stock got the same policy.
    interp_c_grad = [
        LinearInterpolation((a_grid, k_grid), model.sol_c_grad[t, :, :, i_p, 1]; extrapolation_bc=Flat())
        for t in 1:T, i_p in 1:model.Np
    ]
    interp_h_grad = [
        LinearInterpolation((a_grid, k_grid), model.sol_h_grad[t, :, :, i_p, 1]; extrapolation_bc=Flat())
        for t in 1:T, i_p in 1:model.Np
    ]

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
    # N15: cum_weights normalized to end at exactly 1, and drawn from the model's ONE
    # stored draw set rather than a locally seeded RNG.
    cum_weights = cumsum(t_weight); cum_weights ./= cum_weights[end]
    # C7: see discrete_draw -- cum_weights[end] can still be 1 - eps after rounding.
    eps_indices = [something(findfirst(w -> w ≥ model.draws_uniform_t[i], cum_weights), Nt)
                   for i in 1:simN]

    # -- 3. Initial path choice (stochastic via taste node) --
    path_choice = Vector{Symbol}(undef, simN)
    for i in 1:simN
        a0, k0, p0_idx = sim_a_init[i], sim_k_init[i], sim_p_init_idx[i]
        i_t = eps_indices[i]
        # -Inf enters ONLY here, at the discrete comparison.
        EV_college = a0 >= a_req_sim[1] ?
            interp_v_college[i_t][p0_idx](a0, k0) + pared_value_offset(model, sim_bc_init[i]) : -Inf
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
                # A graduate's working life is solved in the COLLEGE arrays with
                # E = 1, so it must be read from there, not from the work arrays.
                grad_i = path_choice[i] == :college
                if grad_i
                    c = interp_c_grad[t, p_idx](a, k)
                    h = interp_h_grad[t, p_idx](a, k)
                else
                    c = interp_c_work[t, p_idx](a, k)
                    h = interp_h_work[t, p_idx](a, k)
                end
                p_shock = p_grid[p_idx]
                E_i = grad_i ? 1.0 : 0.0
                w_pre = wage_func(model, k, t, E_i, p_shock)   # pre-tax hourly wage
                sim_wage[i, t] = w_pre / WAGE_SCALING_FACTOR
                sim_income[i, t] = after_tax_income(model, w_pre, h)
            end

            sim_c[i, t] = c
            sim_h[i, t] = h

            # ----- State transitions -----
            if t < T
                if path_choice[i] == :college && t <= t_college
                    a_next = (1 + r) * a - c - college_cost + y
                else
                    a_next = (1 + r) * a + sim_income[i, t] - c + y
                end
                # Human capital is fixed at theta for life on both paths.
                k_next = k

                # snap() fixes float-sized overshoot only; genuine excursions are left
                # visible for check_simulation rather than silently clipped.
                sim_a[i, t+1] = snap(a_next, a_min, model.a_max)
                sim_k[i, t+1] = snap(k_next, k_grid[1], model.k_max)

                p_draw = draws_uniform_p[i, t]
                p_trans_probs = p_transition[p_idx, :]
                sim_p_idx[i, t+1] = discrete_draw(p_trans_probs, p_draw)
            else
                # C12: the terminal period consumes everything and leaves no bequest, so
                # end-of-life assets are 0 by construction. This column was never written
                # and stayed NaN, which X3 then hid. sim_k has only T columns, so only
                # assets are recorded here.
                sim_a[i, t+1] = 0.0
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





# --------------------------
# Simulation (AR1 Shock Only + Family Optimization)
# --------------------------
function simulate_model_family!(model::ConSavLaborCollege_AR1)
    @unpack simN, T, t_college, r, college_cost, a_min = model
    @unpack a_grid, k_grid, p_grid, p_transition, Np = model
    @unpack ap_grid = model                       # N13: transfer arrays live on this grid
    @unpack sim_a, sim_k, sim_c, sim_h, sim_income, sim_wage = model
    @unpack sim_p_idx, sim_a_init, sim_k_init, sim_bc_init, sim_p_init_idx, draws_uniform_p, y = model
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
            for t in 1:t_college, i_p in 1:Np]
        for i_t in 1:Nt
    ]
    interp_h_college = [
        [LinearInterpolation((a_grid[csl(t)], k_grid), model.sol_h_college[t, csl(t), :, i_p, i_t]; extrapolation_bc=Flat())
            for t in 1:t_college, i_p in 1:Np]
        for i_t in 1:Nt
    ]
    # Work policy interpolators
    # The graduate's working life: same shape as the work interpolators, read from
    # the grad arrays. Before the split these WERE the work arrays, so a graduate
    # and a high-school worker at the same stock got the same policy.
    interp_c_grad = [
        LinearInterpolation((a_grid, k_grid), model.sol_c_grad[t, :, :, i_p, 1]; extrapolation_bc=Flat())
        for t in 1:T, i_p in 1:Np
    ]
    interp_h_grad = [
        LinearInterpolation((a_grid, k_grid), model.sol_h_grad[t, :, :, i_p, 1]; extrapolation_bc=Flat())
        for t in 1:T, i_p in 1:Np
    ]

    interp_c_work = [
        LinearInterpolation((a_grid, k_grid), model.sol_c_work[t, :, :, i_p, 1]; extrapolation_bc=Flat())
        for t in 1:T, i_p in 1:Np
    ]
    interp_h_work = [
        LinearInterpolation((a_grid, k_grid), model.sol_h_work[t, :, :, i_p, 1]; extrapolation_bc=Flat())
        for t in 1:T, i_p in 1:Np
    ]
    # C14: the college transfer arrays hold NaN wherever the parent cannot both fund
    # a_req[1] and retain delta_P, so these interpolants are built over the feasible
    # parent-asset slice. `col_min` below then reproduces discrete_college_choice's -Inf
    # at the point of comparison, so the slice is never evaluated outside its domain in
    # a way that can pick college.
    col_min = min_parent_assets_for_college(model)
    ip0 = first_feasible_parent_a(model)
    ip0 === nothing && error("College transfer infeasible at every asset grid point " *
                             "(needs a >= $col_min, a_grid ends at $(a_grid[end]))")
    ip0 > length(ap_grid) - 1 && error("College transfer feasible at only $(length(ap_grid) - ip0 + 1) " *
                                       "parental asset node(s); need at least 2 to interpolate")
    ag_p = ap_grid[ip0:end]

    # Transfer value interpolators
    sol_tr_v_college_interp = [
        [LinearInterpolation((ag_p, k_grid), model.sol_tr_v_college[ip0:end, :, ip, it]; extrapolation_bc=Line())
            for ip in 1:Np]
        for it in 1:Nt
    ]
    sol_tr_v_work_interp = [
        LinearInterpolation((ap_grid, k_grid), model.sol_tr_v_work[:, :, ip, 1]; extrapolation_bc=Line())
        for ip in 1:Np
    ]
    # Transfer amount interpolators
    sol_tr_college_interp = [
        [LinearInterpolation((ag_p, k_grid), model.sol_tr_college[ip0:end, :, ip, it]; extrapolation_bc=Flat())
            for ip in 1:Np]
        for it in 1:Nt
    ]
    sol_tr_work_interp = [
        LinearInterpolation((ap_grid, k_grid), model.sol_tr_work[:, :, ip, 1]; extrapolation_bc=Flat())
        for ip in 1:Np
    ]

    # -- 2. Assign a taste shock node to each agent for t == 1 --
    # N15: cum_weights normalized to end at exactly 1, and drawn from the model's ONE
    # stored draw set rather than a locally seeded RNG.
    cum_weights = cumsum(t_weight); cum_weights ./= cum_weights[end]
    # C7: see discrete_draw -- cum_weights[end] can still be 1 - eps after rounding.
    eps_indices = [something(findfirst(w -> w ≥ model.draws_uniform_t[i], cum_weights), Nt)
                   for i in 1:simN]

    # -- 3. Initial path choice based on parent's transfer decision --
    path_choice = Vector{Symbol}(undef, simN)
    tr_initial = Vector{Float64}(undef, simN)
    for i in 1:simN
        parent_assets = sim_a_init[i]  # Parent's assets
        HC = sim_k_init[i]             # Child's initial human capital
        ip = sim_p_init_idx[i]         # Persistent shock index
        it = eps_indices[i]            # Taste shock index

        # Compute parent's value for each path. C14: below col_min the college branch was
        # never solved, so it is -Inf here exactly as in discrete_college_choice -- not an
        # extrapolation of the feasible slice.
        f_college = parent_assets >= col_min ?
                    sol_tr_v_college_interp[it][ip](parent_assets, HC) +
                        pared_value_offset(model, sim_bc_init[i]) : -Inf
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
                # Graduates' working life lives in the COLLEGE arrays (E = 1).
                E_i = path_choice[i] == :college ? 1.0 : 0.0
                if E_i == 1.0
                    c = interp_c_grad[t, p_idx](a, k)
                    h = interp_h_grad[t, p_idx](a, k)
                else
                    c = interp_c_work[t, p_idx](a, k)
                    h = interp_h_work[t, p_idx](a, k)
                end
                p_shock = p_grid[p_idx]
                w_pre = wage_func(model, k, t, E_i, p_shock)  # Pre-tax hourly wage
                sim_wage[i, t] = w_pre / WAGE_SCALING_FACTOR
                sim_income[i, t] = after_tax_income(model, w_pre, h)
            end

            sim_c[i, t] = c
            sim_h[i, t] = h

            # ----- State transitions -----
            if t < T
                if path_choice[i] == :college && t <= t_college
                    a_next = (1 + r) * a - c - college_cost + y
                else
                    a_next = (1 + r) * a + sim_income[i, t] - c + y
                end
                # Human capital is fixed at theta for life on both paths.
                k_next = k
                # snap() fixes float-sized overshoot only; genuine excursions are left
                # visible for check_simulation rather than silently clipped.
                sim_a[i, t+1] = snap(a_next, a_min, model.a_max)
                sim_k[i, t+1] = snap(k_next, k_grid[1], model.k_max)

                # Transition for persistent shock
                p_draw = draws_uniform_p[i, t]
                p_trans_probs = p_transition[p_idx, :]
                sim_p_idx[i, t+1] = discrete_draw(p_trans_probs, p_draw)
            else
                # C12: the terminal period consumes everything and leaves no bequest, so
                # end-of-life assets are 0 by construction. This column was never written
                # and stayed NaN, which X3 then hid. sim_k has only T columns, so only
                # assets are recorded here.
                sim_a[i, t+1] = 0.0
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




