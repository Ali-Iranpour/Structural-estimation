# =============================================================================
# moments.jl -- SMM on four parent-block moments.
#
# Estimates four parent parameters against four data means: household
# consumption, parental leisure, and monetary investment in the child SPLIT BY
# CHILD AGE (1-9 and 10-17). Baseline only; nothing here touches the child
# lifecycle, the counterfactuals or the belief machinery.
#
# WHAT SMM IS DOING HERE, IN ONE PARAGRAPH
# ----------------------------------------
# The model has parameters we cannot observe (how much parents value leisure,
# how productive money is in producing child skill). For any guess at those
# parameters we can SOLVE the model and SIMULATE a cohort, which gives us
# simulated versions of things we CAN observe -- average consumption, average
# leisure, average investment. Simulated Method of Moments picks the parameters
# that make the simulated averages line up with the averages in the PSID/CDS
# data. "Method of moments" because we match moments (here, means) rather than
# a likelihood; "simulated" because the model has no closed form, so the moments
# come out of a simulation.
#
# FOUR MOMENTS, FOUR PARAMETERS: JUST-IDENTIFIED
# ----------------------------------------------
# One parameter per moment, so in principle all four can be matched exactly and
# the objective can reach zero. That is deliberate for a first estimation: if the
# fit is bad you know it is the MODEL failing, not a shortage of free parameters.
# It also makes the weighting matrix irrelevant at the optimum, which removes one
# thing to get wrong. Each parameter has a moment it moves most:
#
#   phi_1_0    weight on consumption      ->  mean c_p
#   phi_2_0    weight on leisure          ->  mean l_p
#   sigma_2_0  LEVEL of the e_p elasticity ->  mean e_p, ages 1-9
#   sigma_2_1  SLOPE of the e_p elasticity ->  mean e_p, ages 10-17
#
# They are not independent -- the budget ties them together (see BUDGET below) --
# but each has a clear first-order channel, which is what identification needs.
#
# WHY INVESTMENT IS SPLIT BY AGE
# ------------------------------
# sigma_2_t = exp(sigma_2_0 + sigma_2_1*(t-1)), so sigma_2_1 is an age SLOPE. A
# single pooled mean of e_p cannot separate a slope from a level: many
# (sigma_2_0, sigma_2_1) pairs reproduce the same average, and the search would
# slide along that ridge and return whichever point its Sobol seed sat nearest.
# Adding sigma_2_1 to a 3-moment design would have been under-identified -- an
# answer, but an arbitrary one. Splitting investment at child age 9 supplies the
# second investment moment that pins the slope down. Data: 0.3532 early against
# 0.4414 late, a 1.25x rise.
#
# The profile behind those two numbers is U-SHAPED, though -- 0.353 at age 1,
# down to 0.241 at 12, then up to 0.650 by 17 -- while exp(sigma_2_0 +
# sigma_2_1*(t-1)) is monotone. Two group means are therefore the most this
# functional form can honestly be asked to match, and a good fit on them is NOT
# the model reproducing the age profile of investment.
#
# BUDGET: THE MOMENTS ARE NOT FREE OF EACH OTHER
# ----------------------------------------------
# Every period,  c_p + e_p + saving = (1+r)a + after-tax income + y.
# So the three targets jointly imply a saving rate. At the current wage process
# (mean after-tax household income 5.2264, y = 0.6) the targets c = 3.158 and
# e = 0.394 leave  5.826 - 3.158 - 0.394 = 2.27  per period of saving, i.e. 39%
# of resources. That is high, and over 17 periods at r = 3% it accumulates to far
# more than the ~25 (i.e. $250k) terminal-asset figure discussed earlier. The
# report prints the implied saving rate and terminal assets on every run so this
# tension stays visible instead of hiding inside a converged objective.
#
# WHY THIS IS CHEAP
# -----------------
# The estimated parameters are ALL parent-block parameters. The child lifecycle,
# its transfer stage and the terminal value spline depend on NONE of them, so
# they are solved ONCE at startup and reused for every evaluation. This is exact,
# not an approximation. Each objective evaluation is then just: build the parent,
# backward-induct, simulate -- a few seconds rather than a full pipeline.
#
# PARALLELISM
# -----------
# Worker PROCESSES (Distributed.jl), never threads. NLopt.jl is not thread-safe
# in this project: with `parallel = true` and 8 threads the objective killed the
# process with exit 0 and no error. Each worker process has its own NLopt state,
# so the hazard cannot arise. See the header of tiktak.jl.
# =============================================================================

using TOML, Printf, Statistics

# -----------------------------------------------------------------------------
# Scale constants -- see Input/smm_targets_baseline.toml for the derivation
# -----------------------------------------------------------------------------
const DOLLARS_PER_MODEL_UNIT = 10_000.0
const HOURS_PER_WEEK         = 112.0
const SMM_AGE_LO, SMM_AGE_HI = 1, 17

# Child age at which investment splits into early/late. MUST match AGE_SPLIT in
# tools/make_smm_targets.py -- load_targets checks this against the generated file
# and refuses to run if they have drifted, because a mismatch would compare the
# model's ages 1..9 against the data's ages 1..8 and quietly report a bad fit as a
# model failure.
const SMM_AGE_SPLIT = 9

# The moments actually targeted, in report order. `mean_e_p` (the pooled
# investment mean) is still computed and printed, but it is NOT in this tuple: it
# is the sum of the two age groups and would add no information while making the
# system over-identified. To go back to the 3-moment design, put `mean_e_p` here
# in place of the two `_early`/`_late` entries and drop sigma_2_1 from SMM_PARAMS.
const SMM_MOMENTS = ("mean_c_p", "mean_l_p", "mean_e_p_early", "mean_e_p_late")

# A failed solve must return a large FINITE value, never Inf or an exception:
# a derivative-free local search needs to be able to form a descent direction
# away from a bad region, and Inf carries no direction.
const SMM_PENALTY = 1.0e6

# Penalised evaluations, by reason, on THIS process. A penalty is a real answer
# ("the model cannot live here"), but a search that penalises half its draws is
# telling you the box is wrong, not that the model is bad -- so the count is kept
# and reported instead of vanishing into a large finite number. run_smm.jl gathers
# these from every worker at the end of the run.
const SMM_PENALTY_LOG = Dict{Symbol,Int}()

function _penalize!(reason::Symbol)
    SMM_PENALTY_LOG[reason] = get(SMM_PENALTY_LOG, reason, 0) + 1
    return SMM_PENALTY
end

"""
    smm_feasible(kw) -> Bool

Is this parameter draw economically admissible, before any solving happens?

Only one restriction so far: the money share in the HC technology,
`sigma_2_t = exp(sigma_2_0 + sigma_2_1*(t-1))`, must stay below 1 for every
`t = 1..17`. At or above 1 the Cobb-Douglas technology is explosive in `e_p`, and
the parent's SLSQP solve diverges to a NaN iterate rather than failing cleanly.

The maximum is at one end or the other since the exponent is monotone in `t`, so
checking both endpoints is exact, not a sample.
"""
function smm_feasible(kw)
    s0 = hasproperty(kw, :sigma_2_0) ? kw.sigma_2_0 : PARENT_DEFAULTS.sigma_2_0
    s1 = hasproperty(kw, :sigma_2_1) ? kw.sigma_2_1 : PARENT_DEFAULTS.sigma_2_1
    lo, hi = SMM_AGE_LO - 1, SMM_AGE_HI - 1          # the (t-1) actually used
    return max(exp(s0 + s1 * lo), exp(s0 + s1 * hi)) < 1.0
end

# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------
"""
    load_targets(path) -> Dict{String,NamedTuple}

Read the generated target file. Each entry carries the data mean plus the source
variable and units, so a run can print exactly what it matched against.
"""
function load_targets(path::AbstractString)
    raw = TOML.parsefile(path)

    # The age split lives in two files and must agree in both. If the generator's
    # AGE_SPLIT is changed without changing SMM_AGE_SPLIT, the model's early group
    # and the data's early group cover different ages and every fit silently
    # compares the wrong things -- so fail here rather than produce a number.
    if haskey(raw, "age_split")
        got = Int(raw["age_split"])
        got == SMM_AGE_SPLIT || error("""
            age split mismatch: $path was generated with age_split = $got, but
            moments.jl has SMM_AGE_SPLIT = $SMM_AGE_SPLIT. Make them equal --
            they must describe the same child ages.""")
    end

    out = Dict{String,NamedTuple}()
    for k in SMM_MOMENTS
        haskey(raw, k) || error("""
            target file $path is missing [$k].
            Regenerate it:  uv run --with pandas --with numpy python tools/make_smm_targets.py""")
        e = raw[k]
        out[k] = (mean = Float64(e["mean"]), sd = Float64(e["sd"]),
                  n = Int(e["n"]), source = String(e["source"]),
                  units = String(e["units"]))
    end
    return out
end

# -----------------------------------------------------------------------------
# Model moments
# -----------------------------------------------------------------------------
"""
    model_moments(p) -> NamedTuple

The three simulated moments, on exactly the definitions the target file uses.

`sim_*` columns 1..17 are the family stage; column 18 is the terminal state and
is NOT a flow, so it is excluded. Means skip non-finite entries rather than
propagating them -- a single NaN would otherwise turn a moment into NaN and the
objective into a penalty, hiding a merely-partial simulation as a total failure.
"""
function model_moments(p::Parent_child_interaction_age_specific_AR1)
    cols  = SMM_AGE_LO:SMM_AGE_HI
    # Column t IS child age t, so the model's age groups are literally these
    # columns -- the same ages the generator selects on Child_Age in the data.
    early = SMM_AGE_LO:SMM_AGE_SPLIT
    late  = (SMM_AGE_SPLIT + 1):SMM_AGE_HI
    nanmean(v) = (w = filter(isfinite, v); isempty(w) ? NaN : mean(w))

    c = nanmean(vec(p.sim_c[:, cols]))
    e = nanmean(vec(p.sim_e[:, cols]))            # pooled: reported, not targeted
    # Leisure is a residual of the time budget, exactly as the data builds it:
    # 112 - work - active childcare, per parent.
    l = nanmean(vec(1.0 .- p.sim_h[:, cols] .- p.sim_t[:, cols]))

    return (mean_c_p = c, mean_l_p = l, mean_e_p = e,
            mean_e_p_early = nanmean(vec(p.sim_e[:, early])),
            mean_e_p_late  = nanmean(vec(p.sim_e[:, late])))
end

"""
    moment_diagnostics(p) -> NamedTuple

Things that are not targeted but decide whether a fit is believable: the implied
saving rate, terminal assets, and the two time uses leisure is the residual of.
"""
function moment_diagnostics(p::Parent_child_interaction_age_specific_AR1)
    cols = SMM_AGE_LO:SMM_AGE_HI
    nanmean(v) = (w = filter(isfinite, v); isempty(w) ? NaN : mean(w))
    inc = nanmean(vec(p.sim_income[:, cols]))
    c   = nanmean(vec(p.sim_c[:, cols]))
    e   = nanmean(vec(p.sim_e[:, cols]))
    res = inc + p.y
    return (income = inc, resources = res,
            saving_rate = (res - c - e) / res,
            terminal_assets = nanmean(p.sim_a[:, p.T + 1]),
            h_p = nanmean(vec(p.sim_h[:, cols])),
            t_p = nanmean(vec(p.sim_t[:, cols])))
end

# -----------------------------------------------------------------------------
# Estimated parameters
# -----------------------------------------------------------------------------
# Bounded parameters are searched on a LINKED scale so the optimizer cannot walk
# out of the economically meaningful region: strictly-positive weights are
# searched in logs, so a step can never produce a negative weight. sigma_2_0 is a
# log-elasticity already (sigma_2 = exp(sigma_2_0 + sigma_2_1*(t-1))) and is
# searched in levels inside a box.
#
# Bounds, and why:
#   phi_1_0   [0.2, 5.0]     weight on consumption; 1.0 is the incumbent and the
#                            de facto numeraire for the other weights.
#   phi_2_0   [0.05, 20.0]   weight on leisure; incumbent 0.5. Wide, because the
#                            leisure level is sensitive to it and eta = 2.
#   sigma_2_0 [-5.0, -0.5]   incumbent -1.80 (sigma_2 = 0.165). The box spans
#                            sigma_2 in [0.0067, 0.607]; staying below 1 keeps the
#                            Cobb-Douglas share sensible.
struct SMMParam
    name::Symbol
    lo::Float64
    hi::Float64
    link::Symbol       # :log or :level
end

#   sigma_2_1 [-0.05, 0.05]  the AGE SLOPE of the money elasticity, incumbent 0.02.
#                            sigma_2_t = exp(sigma_2_0 + sigma_2_1*(t-1)), so over
#                            t = 1..17 this box spans a 0.45x fall to a 2.2x rise in
#                            the elasticity. For scale, the data's own log-linear
#                            investment slope is +0.0158/yr, so the incumbent 0.02 is
#                            already close and this box is roughly +-3x around it.
#
#                            NOTE the interaction with sigma_2_0: the Cobb-Douglas
#                            share stays below 1 only while sigma_2_0 + 16*sigma_2_1
#                            < 0, so the top corner of the joint box (-0.5, 0.05)
#                            implies sigma_2_17 = 1.35 and an explosive HC technology.
#                            That corner is left reachable rather than boxed out
#                            because solve_model! throws there and smm_objective turns
#                            it into SMM_PENALTY -- infeasible is a legitimate answer,
#                            and the estimate sits at sigma_2_0 ~ -3.8, nowhere near it.
const SMM_PARAMS = [
    SMMParam(:phi_1_0,   0.2,  5.0,  :log),
    SMMParam(:phi_2_0,   0.05, 20.0, :log),
    SMMParam(:sigma_2_0, -5.0, -0.5, :level),
    SMMParam(:sigma_2_1, -0.05, 0.05, :level),
]

# Just-identified: one parameter per targeted moment. Keep it that way, or decide
# deliberately not to -- an over-identified system needs a weighting matrix, and
# Q can no longer reach 0, which changes how every number below is read.
length(SMM_PARAMS) == length(SMM_MOMENTS) || @warn """
    SMM is no longer just-identified: $(length(SMM_PARAMS)) parameters against \
    $(length(SMM_MOMENTS)) moments. Equal weights are then a real assumption, \
    not a harmless one."""

# INVARIANT -- the single assumption that makes this estimation affordable.
#
# Every estimated parameter must be a PARENT-block parameter. The child lifecycle
# is then invariant across evaluations, so run_smm.jl solves it ONCE per process
# and reuses the terminal value spline for every one of the ~800 evaluations.
# That is exact, not an approximation: nothing in child_lifecycle.jl reads any of
# these names.
#
# Add a CHILD parameter (rho, omega, psi_terminal, kappa_terminal, ...) and that
# silently stops being true -- the run would keep reusing a stale child solve and
# report a converged fit for a model it never actually solved. Wrong answers, no
# error. So it fails loudly here instead. Membership in PARENT_DEFAULTS is the
# test because that NamedTuple is the parent block's calibration, by construction.
#
# If you genuinely need to estimate a child parameter, this guard is not the thing
# to delete: build_child_value() has to move INSIDE smm_objective, and every
# evaluation then pays a full child solve.
let stray = [q.name for q in SMM_PARAMS if !hasproperty(PARENT_DEFAULTS, q.name)]
    isempty(stray) || error("""
        SMM_PARAMS contains non-parent parameter(s): $(join(stray, ", ")).

        run_smm.jl solves the child lifecycle once per process and reuses it for
        every evaluation, which is only valid while all estimated parameters are
        parent-block. See the INVARIANT note above moments.jl:SMM_PARAMS.""")
end

to_search(v, q::SMMParam)   = q.link === :log ? log(v) : v
from_search(z, q::SMMParam) = q.link === :log ? exp(z) : z

search_bounds() = ([to_search(q.lo, q) for q in SMM_PARAMS],
                   [to_search(q.hi, q) for q in SMM_PARAMS])

"""
    unpack(z) -> NamedTuple

Search vector -> model keyword arguments, clamped back into the box. The clamp
matters: NLopt's Nelder-Mead can propose a point marginally outside the bounds,
and `exp` of a slightly-too-large value is a silently absurd parameter.
"""
function unpack(z::AbstractVector{Float64})
    vals = map(enumerate(SMM_PARAMS)) do (i, q)
        clamp(from_search(z[i], q), q.lo, q.hi)
    end
    return NamedTuple{Tuple(q.name for q in SMM_PARAMS)}(Tuple(vals))
end

incumbent() = [to_search(getfield(PARENT_DEFAULTS, q.name), q) for q in SMM_PARAMS]

# -----------------------------------------------------------------------------
# Objective
# -----------------------------------------------------------------------------
"""
    smm_objective(z, targets, V_child; grids...) -> Float64

Weighted relative distance between simulated and data means:

    Q = sum_j w_j * ((m_j - mhat_j) / s_j)^2,     s_j = max(|mhat_j|, 0.05)

Scaling by the target puts every moment on a comparable footing regardless of
units -- without it, consumption (~3) would dominate leisure (~0.5) purely
because of its size. The 0.05 floor stops a near-zero target from exploding the
ratio.

Weights are EQUAL. With three moments and three parameters the system is
just-identified, so at the optimum the weights do not change the answer; equal
weights are then the choice that adds no unexamined assumption. (They would
matter for over-identified runs, e.g. once SDs are added.)

Common random numbers: every model is built with the same `seed`, so the initial
draws and shock paths are identical across evaluations. Without this the
objective is a step function of simulation noise and no derivative-free method
converges -- it would be chasing the RNG, not the parameters.
"""
function smm_objective(z::AbstractVector{Float64}, targets, V_child;
                       Na::Int = 30, Nk::Int = 2, Nhc::Int = 30,
                       simN::Int = 2000, seed::Int = 1234)
    kw = unpack(z)

    # ---- reject the infeasible region BEFORE paying for a solve --------------
    # sigma_2_t = exp(sigma_2_0 + sigma_2_1*(t-1)) is the Cobb-Douglas share on
    # money in the HC technology. Above 1 the technology is explosive and the
    # SLSQP solve does not merely fail, it wanders to a NaN iterate and trips the
    # @assert in HC_technology_full. Since sigma_2_1 became a free parameter the
    # joint box has such a corner (sigma_2_0 -> -0.5 with sigma_2_1 -> 0.05 gives
    # sigma_2_17 = 1.35), and a Sobol point landed in it on the 2026-08-27 run --
    # killing the whole estimation at evaluation 376 of 401.
    #
    # This is a genuine economic restriction, so it is stated here as one rather
    # than left for the solver to discover: infeasible draws cost nothing and are
    # scored, not crashed on.
    if !smm_feasible(kw)
        _penalize!(:infeasible_sigma_2)
        return SMM_PENALTY
    end

    try
        # No `w`. The struct HAS a `w` field, but nothing ever reads it: every wage in
        # the model comes from `wage_func`, the Mincer specification in beta0 /
        # beta_bothcollege / beta_age / beta_age2 / ... times WAGE_SCALING_FACTOR.
        # Passing w = 12.5 here looked like it pinned the wage and pinned nothing.
        p = Parent_child_interaction_age_specific_AR1(; Na = Na, Nk = Nk, Nhc = Nhc,
                                                        simN = simN, seed = seed, kw...)
        p.V_child_interp = V_child
        # solve_model! prints "Solving period t ..." UNGATED by `verbose` (lines 645,
        # 725, 807 of parent_family.jl), and simulate_model! prints its own summary.
        # Thousands of evaluations would bury the progress output, so silence them
        # here rather than editing the solver.
        redirect_stdout(devnull) do
            solve_model!(p; verbose = false)
            simulate_model!(p)
        end
        m = model_moments(p)

        q = 0.0
        for k in SMM_MOMENTS
            mhat = targets[k].mean
            mj   = getfield(m, Symbol(k))
            isfinite(mj) || return SMM_PENALTY
            s = max(abs(mhat), 0.05)
            q += ((mj - mhat) / s)^2
        end
        return q
    catch err
        # A parameter draw the model cannot solve is a legitimate answer -- "this
        # region is infeasible" -- not a bug, so it becomes a large finite penalty.
        # The list is EMPIRICAL; each entry is a failure actually observed:
        #
        #   ErrorException   solve_model! throws below a 95% converged share.
        #   DomainError      log/^ of a non-positive quantity in the technology.
        #   AssertionError   HC_technology_full's `@assert t_p > 0 && e_p > 0`, which
        #                    fires when SLSQP hands it a NaN iterate (NaN > 0 is
        #                    false). Killed the 2026-08-27 run at Sobol point 376/401,
        #                    ~2 min into a projected 137 min -- an AssertionError is
        #                    neither of the two types above, so it was re-thrown out
        #                    of pmap and took every worker down with it.
        #   InexactError     a non-finite intermediate narrowed to an Int.
        #
        # Anything NOT in this list is still re-thrown. A MethodError or an
        # UndefVarError is a coding error and must not be silently scored as a bad
        # parameter draw -- that would turn a broken objective into a converged run.
        if err isa ErrorException || err isa DomainError ||
           err isa AssertionError || err isa InexactError
            _penalize!(nameof(typeof(err)))
            return SMM_PENALTY
        end
        rethrow()
    end
end

# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
"""
    report_fit(z, targets, V_child; kwargs...)

Re-solve at `z` and print the moment table plus the untargeted diagnostics.
"""
function report_fit(z::AbstractVector{Float64}, targets, V_child;
                    Na::Int = 30, Nk::Int = 2, Nhc::Int = 30,
                    simN::Int = 2000, seed::Int = 1234,
                    out::IO = stdout)
    kw = unpack(z)
    p = Parent_child_interaction_age_specific_AR1(; Na = Na, Nk = Nk, Nhc = Nhc,
                                                    simN = simN, seed = seed, kw...)   # no `w` -- see smm_objective
    p.V_child_interp = V_child
    redirect_stdout(devnull) do            # see smm_objective
        solve_model!(p; verbose = false)
        simulate_model!(p)
    end
    m = model_moments(p)
    d = moment_diagnostics(p)

    println(out, "\nParameters")
    println(out, "-"^62)
    for (i, q) in enumerate(SMM_PARAMS)
        @printf(out, "  %-12s %10.4f   (was %.4f)\n", q.name, getfield(kw, q.name),
                getfield(PARENT_DEFAULTS, q.name))
    end

    println(out, "\nTargeted moments")
    println(out, "-"^76)
    @printf(out, "  %-10s %10s %10s %9s   %s\n", "moment", "model", "data", "gap %", "source")
    for k in SMM_MOMENTS
        mj, mhat = getfield(m, Symbol(k)), targets[k].mean
        @printf(out, "  %-10s %10.4f %10.4f %8.1f%%   %s\n",
                k, mj, mhat, 100*(mj - mhat)/abs(mhat), targets[k].source)
    end
    # Same numbers in the units the data was collected in, because "0.53" is
    # hard to sanity-check and "59 hours a week" is not.
    @printf(out, "\n  c_p        %.0f USD/yr  vs data %.0f\n",
            m.mean_c_p*DOLLARS_PER_MODEL_UNIT, targets["mean_c_p"].mean*DOLLARS_PER_MODEL_UNIT)
    @printf(out, "  l_p        %.1f hrs/wk  vs data %.1f\n",
            m.mean_l_p*HOURS_PER_WEEK, targets["mean_l_p"].mean*HOURS_PER_WEEK)
    @printf(out, "  e_p  1-%-2d  %.0f USD/yr  vs data %.0f\n", SMM_AGE_SPLIT,
            m.mean_e_p_early*DOLLARS_PER_MODEL_UNIT, targets["mean_e_p_early"].mean*DOLLARS_PER_MODEL_UNIT)
    @printf(out, "  e_p %2d-%-2d  %.0f USD/yr  vs data %.0f\n", SMM_AGE_SPLIT+1, SMM_AGE_HI,
            m.mean_e_p_late*DOLLARS_PER_MODEL_UNIT, targets["mean_e_p_late"].mean*DOLLARS_PER_MODEL_UNIT)
    # The age slope the two groups are really about, in one number each side.
    @printf(out, "  e_p late/early  model %.2fx  vs data %.2fx\n",
            m.mean_e_p_late/m.mean_e_p_early,
            targets["mean_e_p_late"].mean/targets["mean_e_p_early"].mean)

    println(out, "\nUntargeted -- does the fit stay believable?")
    println(out, "-"^62)
    @printf(out, "  after-tax income      %8.4f  (%.0f USD/yr)\n", d.income, d.income*DOLLARS_PER_MODEL_UNIT)
    @printf(out, "  implied saving rate   %8.1f%%\n", 100*d.saving_rate)
    @printf(out, "  terminal assets       %8.4f  (%.0f USD)\n", d.terminal_assets,
            d.terminal_assets*DOLLARS_PER_MODEL_UNIT)
    @printf(out, "  work  h_p             %8.4f  (%.1f hrs/wk)\n", d.h_p, d.h_p*HOURS_PER_WEEK)
    @printf(out, "  child time t_p        %8.4f  (%.1f hrs/wk)\n", d.t_p, d.t_p*HOURS_PER_WEEK)
    return (moments = m, diagnostics = d, params = kw)
end
