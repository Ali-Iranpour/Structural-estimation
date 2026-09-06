# =============================================================================
# moments.jl -- SMM on ten parent-block moments.
#
# Estimates NINE parent parameters against TEN data moments: household consumption,
# parental WORK hours, parent TIME with the child, monetary investment, the child's own
# study time, and the LEVEL OF CHILD SKILL -- the last four split by child age. Baseline
# only; nothing here touches the child lifecycle, the counterfactuals or the belief
# machinery.
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
# TEN MOMENTS, NINE PARAMETERS: OVER-IDENTIFIED BY ONE
# ----------------------------------------------------
# This is NOT the earlier just-identified design and must not be read as one. Q cannot
# reach zero, the weighting is no longer irrelevant at the optimum, and a residual gap is
# not by itself evidence of a bug. Counting nine against ten also establishes nothing
# about identification -- see the residual-Jacobian note below. Each parameter still has
# a moment it moves most:
#
#   phi_2      weight on leisure           ->  mean h_p   (work; l = 1 - h - t)
#   phi_3      parents' weight on skill    ->  mean t_p and mean e_p
#   lambda_2   child's weight on skill     ->  mean i_c   (study time)
#   R_0        HC technology TFP           ->  mean log HC
#   sigma_1_0  LEVEL of the t_p elasticity ->  mean t_p, ages 1-9
#   sigma_1_1  SLOPE of the t_p elasticity ->  mean t_p, ages 10-17
#   sigma_2_0  LEVEL of the e_p elasticity ->  mean e_p, ages 1-9
#   sigma_2_1  SLOPE of the e_p elasticity ->  mean e_p, ages 10-17
#   sigma_4_0  LEVEL of the i_c elasticity ->  mean i_c
#
# MEASURED at the incumbent (central differences, columns scaled to a full-box move):
# the residual Jacobian has full column rank with condition number 49 and smallest
# singular value 0.271. The weakest direction is lambda_2 against
# sigma_1_0 + sigma_4_0 + sigma_2_0 -- valuation against technology -- and the second
# weakest is sigma_2_1, whose whole-box effect on the investment moments is ~10x smaller
# than sigma_2_0's. Both are identified; neither is sharply identified.
#
# WHY h_p AND t_p RATHER THAN l_p
# -------------------------------
# l_p = 1 - h_p - t_p identically, so targeting leisure pins the SUM of work and
# child time and says nothing about the split. The 2026-08-27 estimate matched
# leisure exactly while working 29.6 hrs/wk against 34.4 in data and doing 23.2
# hrs of childcare against 18.2 -- two errors that cancel inside l_p and are
# invisible to it. Targeting h_p and t_p is strictly more information, and l_p
# comes along for free as the residual.
#
# CAVEAT ON t_p, by instruction 2026-08-28: it is matched on `par_time_tot`, the
# child-side union of active AND nearby parental presence. Nearby time overlaps
# leisure and work, so the h_p and t_p targets jointly imply about 33 hrs/wk of
# leisure against the 59.2 the same data measures. The identity forces the model
# to that number, and the ~26-hour difference is absorbed by phi_2_0. Read the
# estimated phi_2_0 as "whatever makes this time budget work", NOT as a taste for
# leisure. tools/make_smm_targets.py carries the full accounting and the one-line
# revert to per-parent active time.
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

# The HC moments start at child age 3, not 1. The Woodcock-Johnson composite is not
# administered before age 3, so `x_gach` has 0 observations at age 1 and 1 at age 2 --
# the data's "early" HC group is really ages 3-9. The model was averaging log(sim_hc)
# over 1..9 against it. MEASURED at the incumbent: ages 1-9 gives 6.5492 and ages 3-9
# gives 6.6588, so the coverage mismatch alone was worth 0.110 log points, i.e. 23% of
# the entire HC gap the estimation is trying to close. It has to match on both sides.
const SMM_AGE_HC_LO = 3

# The moments actually targeted, in report order. `mean_e_p` (the pooled
# investment mean) is still computed and printed, but it is NOT in this tuple: it
# is the sum of the two age groups and would add no information while making the
# system over-identified. To go back to the 3-moment design, put `mean_e_p` here
# in place of the two `_early`/`_late` entries and drop sigma_2_1 from SMM_PARAMS.
const SMM_MOMENTS = ("mean_c_p", "mean_h_p",
                     "mean_t_p_early", "mean_t_p_late",
                     "mean_e_p_early", "mean_e_p_late",
                     "mean_i_c_early", "mean_i_c_late",
                     "mean_hc_early",  "mean_hc_late")

# Ten moments against nine parameters -- over-identified by one, deliberately.
# The two HC moments are what make the set identified at all: phi_3 and lambda_2 (how
# much parent and child VALUE skill) and R_0, sigma_1, sigma_2, sigma_4 (how efficiently
# skill is PRODUCED) both raise investment, so investment moments alone cannot separate
# them. Only the resulting HC level can. Before HC was put in the data's units there was
# no such moment available.
#
# The child's own study time starts at t = T_CHILD_VOICE = 6; there is no child decision
# before that, so mean_i_c_early averages t = 6..9, not 1..9.

# Moments that are MEANS OF LOGS. Their residual is already a proportional error -- a
# log difference of 0.05 IS a 5% error in the level -- so it must NOT be divided by the
# target the way a level moment is.
#
# WHY THIS MATTERS. Dividing by the target puts level moments on a proportional footing,
# which is right for them. For a log moment it divides by the arbitrary level of the log:
# `x_gach` is a log W-score, so the target is ~6.1, and the residual gets shrunk 6.1x
# before squaring. MEASURED at the incumbent: the model's human capital was +60.2% in
# LEVELS and the objective scored it as a 7.7% miss. The HC moments carried 13.9% of
# R_0's identifying leverage -- and R_0 is in the estimated set precisely to fix the HC
# level. On the units-free scale below that becomes 86.1%, the residual Jacobian's
# condition number falls 162 -> 49, and its smallest singular value is 3.4x stronger.
#
# The scaling was also arbitrary in the literal sense: index HC to 1 instead of W-scores
# and log HC ~ 0, the 0.05 floor binds, and these two moments get ~150x MORE weight than
# they had. A moment's weight must not depend on the units its log happens to be in.
const SMM_LOG_MOMENTS = ("mean_hc_early", "mean_hc_late")

"""
    moment_scale(k, mhat) -> Float64

Denominator of moment `k`'s residual. Level moments are scaled by their own target so
every moment is measured in proportional error; log moments are already proportional and
are scaled by 1. The 0.05 floor stops a near-zero level target from exploding the ratio.
"""
moment_scale(k, mhat) = k in SMM_LOG_MOMENTS ? 1.0 : max(abs(mhat), 0.05)

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
    _root_cause(e) -> Exception

Unwrap the exception NLopt hands back so it can be classified by what actually
went wrong.

THIS IS LOAD-BEARING, and its absence cost two runs. `solve_model!` drives NLopt,
and anything thrown inside an NLopt *callback* crosses a C boundary: NLopt catches
it in `_catch_forced_stop`, stores `CapturedException(e, backtrace)`
(NLopt.jl:568), forces a stop, and re-throws THAT wrapper from `optimize!`
(NLopt.jl:807). So the objective never sees the `AssertionError` itself -- it sees
a `CapturedException` around one, matches none of the types below, and re-throws
out of `pmap`, killing every worker.

Errors thrown by `solve_model!` OUTSIDE a callback -- the 95%-convergence
`error()` -- arrive unwrapped, which is why `ErrorException` appeared to be
handled correctly while `AssertionError` was not.
"""
_root_cause(e) =
    e isa CapturedException   ? _root_cause(e.ex) :
    e isa TaskFailedException ? _root_cause(e.task.exception) :
    e

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
    lo, hi = SMM_AGE_LO - 1, SMM_AGE_HI - 1          # the (t-1) actually used
    _max_share(a, b) = max(exp(a + b * lo), exp(a + b * hi))
    for (n0, n1) in ((:sigma_1_0, :sigma_1_1), (:sigma_2_0, :sigma_2_1))
        a = hasproperty(kw, n0) ? getproperty(kw, n0) : getfield(PARENT_DEFAULTS, n0)
        b = hasproperty(kw, n1) ? getproperty(kw, n1) : getfield(PARENT_DEFAULTS, n1)
        _max_share(a, b) < 1.0 || return false
    end
    return true
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
    # The child only chooses study time from T_CHILD_VOICE; before that sim_i is not a
    # decision. Match the generator, which selects Child_Age >= 6 for the early group.
    early_i = T_CHILD_VOICE:SMM_AGE_SPLIT
    # HC is observed from age 3 only -- see SMM_AGE_HC_LO.
    early_hc = SMM_AGE_HC_LO:SMM_AGE_SPLIT

    # Non-finite entries are COUNTED, not silently dropped. Filtering them was the more
    # dangerous half of a NaN: a single bad cell used to vanish into a perfectly finite
    # mean, so a simulation that had partly failed reported an ordinary-looking fit.
    # VERIFIED: injecting one NaN into sim_c still returned mean_c_p = 3.703901. The
    # count travels with the moments and smm_objective refuses the draw if it is non-zero.
    n_bad = Ref(0)
    function nanmean(v)
        w = filter(isfinite, v)
        n_bad[] += length(v) - length(w)
        isempty(w) ? NaN : mean(w)
    end
    # log HC, and the mean of the agent-level LOGS -- the data's x_gach is a mean of logs,
    # and log(mean) differs from mean(log) by a Jensen term that moves with age.
    # Non-positive HC is a failure, not something to floor away, so it is counted here too.
    function loghc(rng)
        v = vec(p.sim_hc[:, rng])
        n_bad[] += count(x -> !(isfinite(x) && x > 0), v)
        w = filter(x -> isfinite(x) && x > 0, v)
        isempty(w) ? NaN : mean(log.(w))
    end

    c = nanmean(vec(p.sim_c[:, cols]))
    e = nanmean(vec(p.sim_e[:, cols]))            # pooled: reported, not targeted
    # Leisure is a residual of the time budget, exactly as the data builds it:
    # 112 - work - active childcare, per parent.
    l = nanmean(vec(1.0 .- p.sim_h[:, cols] .- p.sim_t[:, cols]))

    return (mean_c_p = c, mean_l_p = l, mean_e_p = e,
            mean_h_p = nanmean(vec(p.sim_h[:, cols])),
            mean_t_p_early = nanmean(vec(p.sim_t[:, early])),
            mean_t_p_late  = nanmean(vec(p.sim_t[:, late])),
            mean_e_p_early = nanmean(vec(p.sim_e[:, early])),
            mean_e_p_late  = nanmean(vec(p.sim_e[:, late])),
            mean_i_c_early = nanmean(vec(p.sim_i[:, early_i])),
            mean_i_c_late  = nanmean(vec(p.sim_i[:, late])),
            mean_hc_early  = loghc(early_hc),
            mean_hc_late   = loghc(late),
            n_nonfinite    = n_bad[])
end

# Tolerance for the domain checks below. The optimizer's own floors are 1e-4 (goods) and
# TIME_FLOOR = 1e-3 (time), and `snap_parent` repairs only float-sized violations
# (tol 1e-10) BY DESIGN -- a genuinely out-of-bounds value is meant to propagate rather
# than be silently rewritten. This tolerance is therefore loose enough to ignore
# interpolation noise at a bound and tight enough that a real violation is still a
# violation.
const SIM_FEAS_TOL = 1e-8

"""
    simulation_violations(p) -> NamedTuple

Cells of the simulation that leave the model's own domain, counted by KIND.

This exists because counting non-finite cells is not the same as checking validity, and
the difference was measured, not assumed. Injecting each pathology into a solved baseline:

    sim_c  = NaN                        caught (non-finite)
    sim_c  = -5.0  negative consumption NOT caught
    sim_h  = -0.3  negative hours       NOT caught
    sim_h  =  1.8  hours > time budget  NOT caught
    sim_a  = -50   below a_min = 0      NOT caught
    sim_hc = -1.0  negative skill       caught (the one series with a positivity test)

Everything finite was accepted, so a simulation could report an ordinary-looking fit on
economically impossible paths. A negative consumption is not a bad parameter draw with a
large objective -- it is a solve that failed, and it must be refused, not scored.

Assets are checked over ALL T+1 columns. Column T+1 is the terminal state that becomes
the child's initial assets at the handoff, so excluding it hides exactly the column that
propagates into the next block.
"""
function simulation_violations(p::Parent_child_interaction_age_specific_AR1)
    cols = SMM_AGE_LO:SMM_AGE_HI
    tol  = SIM_FEAS_TOL
    C, E, H, Tp = p.sim_c[:, cols], p.sim_e[:, cols], p.sim_h[:, cols], p.sim_t[:, cols]
    I, HC       = p.sim_i[:, cols], p.sim_hc[:, cols]
    A           = p.sim_a[:, 1:(p.T + 1)]

    nf = sum(M -> count(!isfinite, M), (C, E, H, Tp, I, HC)) + count(!isfinite, A)
    fin(f) = x -> isfinite(x) && f(x)
    unit   = x -> x < -tol || x > 1 + tol

    v = (nonfinite               = nf,
         c_nonpositive           = count(fin(x -> x <= 0.0), C),
         e_negative              = count(fin(x -> x < -tol), E),
         hc_nonpositive          = count(fin(x -> x <= 0.0), HC),
         h_outside_unit          = count(fin(unit), H),
         t_outside_unit          = count(fin(unit), Tp),
         i_outside_unit          = count(fin(unit), I),
         parent_leisure_negative = count(fin(x -> x < -tol), 1.0 .- H .- Tp),
         child_leisure_negative  = count(fin(x -> x < -tol), 1.0 .- Tp .- I),
         assets_below_min        = count(fin(x -> x < p.a_min - tol), A))
    return (total = sum(values(v)), v...)
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

    # GRID COVERAGE. Policies are interpolated with Flat() extrapolation, so a simulated
    # state above the solved grid silently reuses the policy at the top node. That is
    # defensible for a thin tail and indefensible if the mass lives out there, so it is
    # measured rather than assumed.
    #
    # MEASURED over ALL T+1 columns, and reported as HOUSEHOLDS, not as a share of cells.
    # An earlier version did neither, and the conclusion drawn from it was wrong: it
    # compared the fraction of HOUSEHOLDS above at t=1 (0.1%) against the fraction of
    # CELLS above over t=1..17 (0.1%), read the equality as "all of it is the initial
    # draw", and reported that. The two numbers match only because 2 households x 17
    # periods / 34,000 cells equals 2 / 2,000 -- different denominators, same digits.
    #
    # What is actually true at the incumbent: 2 households sit above the ceiling in EVERY
    # period 1..17, and 7 are above at the T+1 handoff, peaking at 259 against a ceiling
    # of 100. So five of them cross DURING the family stage; it is not only initial
    # wealth. The handoff column matters most of all -- it becomes the child's initial
    # assets -- and the old diagnostic excluded it from both the share and the maximum.
    a_hi  = maximum(p.a_grid)
    Aall  = p.sim_a[:, 1:(p.T + 1)]
    Aflow = p.sim_a[:, cols]
    n_sim = size(Aall, 1)
    return (income = inc, resources = res,
            saving_rate = (res - c - e) / res,
            terminal_assets = nanmean(p.sim_a[:, p.T + 1]),
            h_p = nanmean(vec(p.sim_h[:, cols])),
            t_p = nanmean(vec(p.sim_t[:, cols])),
            a_grid_max        = a_hi,
            a_max_sim         = maximum(Aall),          # over ALL columns, handoff included
            a_hh_ever_above   = count(i -> any(Aall[i, :] .> a_hi), 1:n_sim) / n_sim,
            a_hh_above_t1     = mean(p.sim_a[:, 1] .> a_hi),
            a_hh_above_handoff= mean(p.sim_a[:, p.T + 1] .> a_hi),
            a_cell_above_flow = mean(Aflow .> a_hi))
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
#   sigma_1_0 [-4.0, -0.2]   LEVEL of the HC elasticity to parent TIME, incumbent
#                            -0.90 (sigma_1 = 0.407). Spans sigma_1 in [0.018, 0.819].
#                            This is the parameter that moves t_p: parent_family.jl's
#                            own note records that tau_p is flat at 0.011-0.023 for
#                            every phi_2 from 0.05 to 3.0 because the FOC scales with
#                            phi_2 on both sides -- "tau_p is set by sigma_1 and the
#                            value of the child's HC". So phi_2_0 identifies h_p and
#                            sigma_1_0 identifies t_p, through separate channels.
#   sigma_1_1 [-0.20, 0.05]  AGE SLOPE of that elasticity, incumbent -0.08. The data
#                            supports this far better than its sigma_2 counterpart:
#                            t_p HALVES over the family stage (30.3 -> 9.7 hrs/wk,
#                            late/early 0.512x) and does so monotonically, which is
#                            exactly the shape exp(sigma_1_0 + sigma_1_1*(t-1)) can
#                            produce. Investment, by contrast, is U-shaped.
const SMM_PARAMS = [
    # phi_1 and lambda_1 are NOT here: utility is defined only up to relative weights, so
    # two of the five must be normalised. Both are pinned at 1.0 (instruction 2026-08-30).
    SMMParam(:phi_2,     0.01, 20.0, :log),    # mean hours of work
    SMMParam(:phi_3,     0.05, 20.0, :log),    # parental time + monetary investment
    SMMParam(:lambda_2,  0.05, 20.0, :log),    # the child's own study time
    # R_0 is the HC technology's TFP and therefore the natural parameter for the HC
    # LEVEL, exactly as sigma_1_0 is for parental time. It became estimable only once HC
    # was put in the data's units: before the rescaling there was no HC moment to
    # identify it against. Searched in logs -- it is a strictly positive scale.
    # Measured at the rescaled starting point R_0 = 81.55, the model overshoots the data
    # by ~50% from age 5 (815 against 423), which is the level error this closes.
    SMMParam(:R_0,       5.0, 300.0, :log),    # HC level, against the HC moments
    SMMParam(:sigma_1_0, -4.0, -0.2,  :level), # HC elasticity to parental TIME, level
    SMMParam(:sigma_1_1, -0.20, 0.05, :level), #   ... and its age slope
    SMMParam(:sigma_2_0, -5.0, -0.5,  :level), # HC elasticity to MONEY, level
    SMMParam(:sigma_2_1, -0.05, 0.05, :level), #   ... and its age slope
    SMMParam(:sigma_4_0, -6.0, -1.0,  :level), # HC elasticity to the CHILD'S own study
    # sigma_4_1 IS NOT ESTIMATED -- and note it is NOT ZERO either. It keeps its
    # PARENT_DEFAULTS value of 0.02, so sigma_4 RISES 24.6% over ages 6-17
    # (0.01133 -> 0.01412, measured). An earlier version of this comment claimed
    # "sigma_4 is held flat in t"; that was false, and it had reached the advisor memo
    # before it was caught. State what the code does.
    #
    # Why it is not estimated -- AND NOT THE REASON PREVIOUSLY GIVEN HERE. An earlier
    # version of this comment claimed sigma_4_1 and mu_1 form an identification ridge.
    # That is wrong. Taking logs of the child's study FOC ratio, with mu_0 = 1 and
    # lambda_1 = 1 so that (1 - mu_t) = -mu_1*(t-5):
    #
    #     log[ sigma_4_t / (1 - mu_t) ] = sigma_4_0 + sigma_4_1*(t-5) - log(-mu_1) - log(t-5)
    #
    # mu_1 enters ONLY the intercept, through -log(-mu_1). That makes it collinear with
    # sigma_4_0, NOT with sigma_4_1, which carries the age SLOPE. MEASURED on the scaled
    # residual Jacobian at the incumbent (|cos| between columns):
    #
    #     sigma_4_0 vs mu_1      0.991   <- the real near-collinearity, worst of all pairs
    #     sigma_4_0 vs sigma_4_1 0.813
    #     sigma_4_1 vs mu_1      0.805   <- correlated, but not a ridge
    #
    # So the pair that cannot be estimated together is (sigma_4_0, mu_1). mu_1 does have
    # other channels -- it also moves alpha_1 and alpha_2 in the family utility -- but
    # not enough to break that.
    #
    # The reason sigma_4_1 is nonetheless still out is CONDITIONING, not rank. Adding it
    # with mu_1 fixed leaves the system full rank but much harder to solve:
    #
    #     9 parameters (current)      condition number   49.2, smallest sv 0.278
    #     10 with sigma_4_1 added     condition number 1067.1, smallest sv 0.052
    #
    # because sigma_4_0 and sigma_4_1 are themselves 0.813 collinear: study time is
    # targeted only at ages 6-9 and 10-17, which are too close together to separate the
    # level of the elasticity from its slope. Adding a study-time moment further apart in
    # age would fix that; adding the parameter alone would not.
    #
    # mu_1 is not estimated either (it holds at -0.04), so the age profile of study time
    # is at present an assumption on BOTH sides. Which to free is an open question with
    # the advisor; do not resolve it here.
]

# Over-identification is now DELIBERATE (ten moments, nine parameters) and is explained
# in the header, so it is no longer warned about on every worker at every startup. What
# would be a real error is the other direction: fewer moments than parameters cannot be
# estimated at all, so that fails loudly.
length(SMM_MOMENTS) >= length(SMM_PARAMS) || error("""
    SMM is UNDER-identified: $(length(SMM_PARAMS)) parameters against \
    $(length(SMM_MOMENTS)) moments. Add moments or drop parameters -- the search would
    otherwise wander along a flat direction and return whichever point it started at.""")

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

    Q = sum_j ((m_j - mhat_j) / s_j)^2,     s_j = moment_scale(j, mhat_j)

`s_j` is the target for a LEVEL moment and 1 for a LOG moment, so every residual is a
proportional error in the underlying quantity -- see moment_scale. Without the level
scaling, consumption (~3) would dominate leisure (~0.5) purely because of its size;
without the log exception, the two HC moments were shrunk 6.1x by the arbitrary level
of a log W-score.

Weights are otherwise EQUAL, and that is now a real assumption rather than a harmless
one: the estimator is OVER-IDENTIFIED, ten moments against nine parameters, so Q cannot
reach zero and the weighting does change the answer at the optimum. Equal weights are
the choice that adds nothing unexamined, not a choice that is free. A covariance-based
weighting matrix is the principled successor and is not implemented.

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

        # A simulation that leaves the model's domain is not a bad parameter draw, it is
        # an invalid evaluation. Scoring it would let a partly-failed solve compete on
        # the strength of the cells that happened to survive. Checked by KIND so the
        # penalty log says WHICH economic law broke, not merely that something did.
        viol = simulation_violations(p)
        if viol.total > 0
            worst = argmax(Dict(k => v for (k, v) in pairs(viol) if k !== :total))
            _penalize!(Symbol("invalid_sim_", worst))
            return SMM_PENALTY
        end

        q = 0.0
        for k in SMM_MOMENTS
            mhat = targets[k].mean
            mj   = getfield(m, Symbol(k))
            isfinite(mj) || return SMM_PENALTY
            q += ((mj - mhat) / moment_scale(k, mhat))^2
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
        #                    false). Killed the 2026-08-27 run at Sobol 376/401 and
        #                    the 2026-08-28 run at Sobol 81/401.
        #   InexactError     a non-finite intermediate narrowed to an Int.
        #
        # CLASSIFY THE ROOT CAUSE, NOT `err`. Anything thrown inside an NLopt
        # callback comes back wrapped in a CapturedException, so testing `err isa
        # AssertionError` directly is always false and the run dies. That is
        # exactly what happened on 2026-08-28 -- the type was in this list already.
        # See _root_cause above.
        #
        # Anything NOT in this list is still re-thrown. A MethodError or an
        # UndefVarError is a coding error and must not be silently scored as a bad
        # parameter draw -- that would turn a broken objective into a converged run.
        cause = _root_cause(err)
        if cause isa ErrorException || cause isa DomainError ||
           cause isa AssertionError || cause isa InexactError
            _penalize!(nameof(typeof(cause)))
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
    q_tot = 0.0
    for k in SMM_MOMENTS
        mj, mhat = getfield(m, Symbol(k)), targets[k].mean
        q_tot += ((mj - mhat) / moment_scale(k, mhat))^2
        # For a log moment the "gap %" is the LEVEL gap, exp(dlog) - 1, not the gap in
        # the log -- reporting the latter is what made a 60% error in human capital look
        # like a 7.7% miss.
        gap = k in SMM_LOG_MOMENTS ? 100*(exp(mj - mhat) - 1) : 100*(mj - mhat)/abs(mhat)
        @printf(out, "  %-10s %10.4f %10.4f %8.1f%%   %s\n",
                k, mj, mhat, gap, targets[k].source)
    end
    @printf(out, "  %-10s %10s %10s %8.4f    (gap %% is in LEVELS for log moments)\n",
            "Q", "", "", q_tot)
    # Same numbers in the units the data was collected in, because "0.53" is
    # hard to sanity-check and "59 hours a week" is not.
    @printf(out, "\n  c_p        %8.0f USD/yr  vs data %.0f\n",
            m.mean_c_p*DOLLARS_PER_MODEL_UNIT, targets["mean_c_p"].mean*DOLLARS_PER_MODEL_UNIT)
    @printf(out, "  h_p        %8.1f hrs/wk  vs data %.1f\n",
            m.mean_h_p*HOURS_PER_WEEK, targets["mean_h_p"].mean*HOURS_PER_WEEK)
    @printf(out, "  t_p  1-%-2d  %8.1f hrs/wk  vs data %.1f\n", SMM_AGE_SPLIT,
            m.mean_t_p_early*HOURS_PER_WEEK, targets["mean_t_p_early"].mean*HOURS_PER_WEEK)
    @printf(out, "  t_p %2d-%-2d  %8.1f hrs/wk  vs data %.1f\n", SMM_AGE_SPLIT+1, SMM_AGE_HI,
            m.mean_t_p_late*HOURS_PER_WEEK, targets["mean_t_p_late"].mean*HOURS_PER_WEEK)
    @printf(out, "  e_p  1-%-2d  %8.0f USD/yr  vs data %.0f\n", SMM_AGE_SPLIT,
            m.mean_e_p_early*DOLLARS_PER_MODEL_UNIT, targets["mean_e_p_early"].mean*DOLLARS_PER_MODEL_UNIT)
    @printf(out, "  e_p %2d-%-2d  %8.0f USD/yr  vs data %.0f\n", SMM_AGE_SPLIT+1, SMM_AGE_HI,
            m.mean_e_p_late*DOLLARS_PER_MODEL_UNIT, targets["mean_e_p_late"].mean*DOLLARS_PER_MODEL_UNIT)
    # The two age slopes the split moments exist to identify.
    @printf(out, "\n  t_p late/early  model %.2fx  vs data %.2fx\n",
            m.mean_t_p_late/m.mean_t_p_early,
            targets["mean_t_p_late"].mean/targets["mean_t_p_early"].mean)
    @printf(out, "  e_p late/early  model %.2fx  vs data %.2fx\n",
            m.mean_e_p_late/m.mean_e_p_early,
            targets["mean_e_p_late"].mean/targets["mean_e_p_early"].mean)
    # l_p is not targeted, but l = 1 - h - t identically, so the h_p and t_p
    # targets IMPLY a leisure level. Compare against THAT, not against measured
    # leisure: t_p is matched on par_time_tot, which overlaps leisure and work, so
    # the implied figure sits ~26 hrs/wk below the 59.2 the data measures. That gap
    # is a property of the target choice, not a failure of the fit -- see the
    # header of tools/make_smm_targets.py.
    n_e, n_l = SMM_AGE_SPLIT - SMM_AGE_LO + 1, SMM_AGE_HI - SMM_AGE_SPLIT
    t_implied = (n_e*targets["mean_t_p_early"].mean + n_l*targets["mean_t_p_late"].mean) / (n_e + n_l)
    l_implied = 1 - targets["mean_h_p"].mean - t_implied
    @printf(out, "  l_p (residual)  model %.1f hrs/wk  vs %.1f implied by the h_p/t_p targets\n",
            m.mean_l_p*HOURS_PER_WEEK, l_implied*HOURS_PER_WEEK)
    @printf(out, "                  (measured leisure is %.1f hrs/wk -- par_time_tot overlaps it)\n",
            0.5286*HOURS_PER_WEEK)

    println(out, "\nUntargeted -- does the fit stay believable?")
    println(out, "-"^62)
    @printf(out, "  after-tax income      %8.4f  (%.0f USD/yr)\n", d.income, d.income*DOLLARS_PER_MODEL_UNIT)
    @printf(out, "  implied saving rate   %8.1f%%\n", 100*d.saving_rate)
    @printf(out, "  terminal assets       %8.4f  (%.0f USD)\n", d.terminal_assets,
            d.terminal_assets*DOLLARS_PER_MODEL_UNIT)
    @printf(out, "  leisure l_p           %8.4f  (%.1f hrs/wk)\n",
            1 - d.h_p - d.t_p, (1 - d.h_p - d.t_p)*HOURS_PER_WEEK)
    v = simulation_violations(p)
    @printf(out, "  invalid sim cells     %8d\n", v.total)
    if v.total > 0
        for (k, n) in pairs(v)
            k === :total || n == 0 || @printf(out, "     %-22s %8d\n", k, n)
        end
    end
    # Reported, NOT clamped. sim_a_init is LogNormal(0.296, 1.402) and its upper tail runs
    # past a_max, so clamping the draw would distort the initial wealth distribution to
    # flatter a grid. But it is NOT only the initial draw -- see moment_diagnostics.
    @printf(out, "  assets above a_max=%.0f  max %.1f  (households: %.2f%% ever, %.2f%% at t=1, %.2f%% at handoff)\n",
            d.a_grid_max, d.a_max_sim, 100*d.a_hh_ever_above,
            100*d.a_hh_above_t1, 100*d.a_hh_above_handoff)
    if d.a_hh_ever_above > d.a_hh_above_t1 + 1e-12
        @printf(out, "     %.2f%% of households CROSS the ceiling during t = 1..17 -- not just the initial draw\n",
                100*(d.a_hh_ever_above - d.a_hh_above_t1))
    end
    return (moments = m, diagnostics = d, params = kw)
end
