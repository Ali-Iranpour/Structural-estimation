# =============================================================================
# moments.jl -- SMM on ten parent-block moments.
#
# Estimates TEN parent parameters against TEN data moments: household consumption,
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
# TEN MOMENTS, TEN PARAMETERS: LOCAL IDENTIFICATION MUST BE RECHECKED
# ----------------------------------------------------
# Own study plus fixed school time, 2026-09-09. sigma_4_1 is now estimated;
# the late HC moment covers ages 12-17 (early HC remains 3-9). Equal counts do
# not establish identification or guarantee an exact fit. Earlier Jacobian
# diagnostics below describe the previous nine-parameter specification only.
#
#   phi_2      weight on leisure           ->  mean h_p   (work; l = 1 - h - t)
#   phi_3      parents' weight on skill    ->  mean t_p and mean e_p
#   lambda_2   child's weight on skill     ->  mean i_c   (child's time input)
#   R_0        HC technology TFP           ->  mean log HC
#   sigma_1_0  LEVEL of the t_p elasticity ->  mean t_p, ages 1-9
#   sigma_1_1  SLOPE of the t_p elasticity ->  mean t_p, ages 10-17
#   sigma_2_0  LEVEL of the e_p elasticity ->  mean e_p, ages 1-9
#   sigma_2_1  SLOPE of the e_p elasticity ->  mean e_p, ages 10-17
#   sigma_4_0  LEVEL of the i_c elasticity ->  mean i_c
#   sigma_4_1  SLOPE of the i_c elasticity ->  child study and HC age profiles
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
# Scale constants -- see the selected run folder's targets.toml for the derivation
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
const SMM_AGE_HC_LATE_LO = 12
const SMM_CHILD_TIME_SPEC = "own_study_fixed_school_v1"

# The moments actually targeted, in report order. `mean_e_p` (the pooled
# investment mean) is still computed and printed, but it is NOT in this tuple: it
# is the sum of the two age groups and would add no information while making the
# system over-identified. To go back to the 3-moment design, put `mean_e_p` here
# in place of the two `_early`/`_late` entries and drop sigma_2_1 from SMM_PARAMS.
const SMM_PARENT_MOMENTS = ("mean_c_p", "mean_h_p",
                            "mean_t_p_early", "mean_t_p_late",
                            "mean_e_p_early", "mean_e_p_late",
                            "mean_i_c_early", "mean_i_c_late",
                            "mean_hc_early",  "mean_hc_late")

# =============================================================================
# THE TAS BLOCK -- seven child-level moments, for the four kappa parameters
# =============================================================================
# These are NOT more parent moments. They are measured on a different file, a different
# unit of observation and a different frame: one row per TAS-linked child (4,248 children,
# 1,481 family clusters) against the parent block's one row per child-YEAR. They are
# unweighted, per Input/CODEBOOK.md, and they are produced by the SAME pipeline run -- the
# child block is resimulated from the simulated parents' terminal states before any of
# them is computed.
#
# WHAT EACH ONE IS FOR. These are joint identifying relationships, not a one-parameter-
# one-moment assignment; every kappa moves every one of them:
#
#   k0_complete          overall four-year completion   ->  kappa_0    (the level)
#   kth_ga17_t{1,2,3}_c  completion by ability tertile  ->  kappa_theta (the gradient)
#   kpe_g{0,1}_c         completion by parental ed      ->  kappa_ParEd
#   kterm_x_strict_w99   parental net worth retained    ->  kappa_terminal
#
# COMPLETION, NOT ENTRY. The model's college path is binary and has no dropout: enrol,
# study t_college = 4 years, then earn the graduate wage E = 1. Nobody enrols without
# finishing, so the path IS a completed four-year degree, and matching it to entry (0.616)
# would compare a mechanism that always pays the college premium against a population
# where 30 percentage points of it never does. Decision 2026-09-10; the codebook makes the
# same call independently ("COMPLETION IS THE PRIMARY OUTCOME").
const SMM_TAS_MOMENTS = ("k0_complete",
                         "kth_ga17_t1_c", "kth_ga17_t2_c", "kth_ga17_t3_c",
                         "kpe_g0_c", "kpe_g1_c",
                         "kterm_x_strict_w99")

# SEVENTEEN moments, FOURTEEN parameters. The order here must match TARGETED in
# tools/make_smm_targets.py -- it is the row/column order of the covariance matrix, and
# load_targets refuses to run if the two have drifted.
const SMM_MOMENTS = (SMM_PARENT_MOMENTS..., SMM_TAS_MOMENTS...)

# The model's ability tertiles are cut on the child's HC at THIS age, matching the data's
# `ach_age == 17` (the child's last CDS wave). Column t of the parent's sim_hc IS child
# age t, so this is literally column 17 -- the last family-stage column, one before the
# age-18 handoff that the enrolment decision is taken on. That ordering mirrors the data,
# where the assessment precedes the college decision.
const SMM_TAS_ACH_AGE = 17

# RANK-BASED tertiles (decision 2026-09-10), not the data's absolute W-score cut points.
# The model's HC level is already targeted by mean_hc_late; cutting the simulation at the
# data's cut points would make these three moments absorb any level or dispersion miss
# into kappa_theta and confound the gradient with the level. Cutting the simulation at its
# OWN terciles asks only what these moments exist to ask: how much does completion rise
# with rank in ability. Ties are broken by stable rank, so an exact tie cannot put the
# same HC in two different tertiles.
const SMM_TAS_TERTILE_RULE = "model-internal rank tertiles of sim_hc at child age 17"

# Ten parameters and ten moments. HC separates valuation from technology;
# freeing sigma_4_1 requires a fresh Jacobian check under this specification.
# Own study moments cover 6-9 and 10-17; HC covers 3-9 and 12-17.
# School time is exogenous and deducted from leisure, not included in sim_i.

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

LEGACY as of 2026-09-10. The objective no longer uses this: with seventeen moments against
fourteen parameters the system is over-identified and the weights decide the answer, so
`moment_weights` (diagonal inverse-variance on the joint clustered covariance) replaced it.

It is kept because `tools/test_smm_baseline.jl` pins a frozen Q computed with it, and that
regression is the only frozen reference to the parent solve that exists. Re-pinning the
number to the new objective would have thrown it away. Do not use this in new code.


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

# The ONE ErrorException message that is a model failure rather than a bug. `error()` is
# Julia's generic throw, and this project uses it for a genuine economic refusal -- the
# solver declining to return a solution built on failed optimizations -- but also, like any
# code, for programming mistakes. Matching the message is what separates them.
#
# A4 (2026-09-06): before this, EVERY ErrorException became SMM_PENALTY = 1e6. A typo that
# threw `error("...")` anywhere under solve_model! or simulate_model! was scored as "the
# model cannot live at this parameter draw", so a broken objective could return a converged
# run with a plausible-looking penalty rate.
const MODEL_FAILURE_PATTERNS = ("converged (floor", "Refusing to return a solution")

"""
    is_model_failure(e) -> Bool

Is this exception an EXPECTED model failure (score it) or a programming error (re-throw)?

Expected: a domain error in the technology, an assertion tripped by a NaN iterate from
SLSQP, a non-finite value narrowed to an Int, and the solver's own convergence refusal.
Everything else -- MethodError, UndefVarError, BoundsError, and any other `error()` -- is a
bug and must stay visible.
"""
function is_model_failure(e)
    e isa DomainError    && return true
    e isa AssertionError && return true
    e isa InexactError   && return true
    if e isa ErrorException
        return any(p -> occursin(p, e.msg), MODEL_FAILURE_PATTERNS)
    end
    return false
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

    get(raw, "child_time_spec", "") == SMM_CHILD_TIME_SPEC || error(
        "Target specification mismatch: regenerate targets for own study plus fixed school time; " * path)
    get(raw, "age_hc_early", []) == [SMM_AGE_HC_LO, SMM_AGE_SPLIT] ||
        error("Early HC age window mismatch: regenerate targets; " * path)
    get(raw, "age_hc_late", []) == [SMM_AGE_HC_LATE_LO, SMM_AGE_HI] ||
        error("Late HC age window mismatch: expected ages 12-17; " * path)
    school = Float64.(get(raw, "school_time", []))
    length(school) == SMM_AGE_HI || error("Missing school_time schedule: " * path)
    all(x -> isfinite(x) && 0 <= x < 1 - 2TIME_FLOOR, school) ||
        error("Invalid school_time schedule: " * path)
    all(iszero, school[1:T_CHILD_VOICE-1]) || error("School must be zero below age 6: " * path)

    # ---- the psychic-cost centring constant ---------------------------------
    # kappa_0 + kappa_theta*(log theta - m_psychic). FROZEN in the target file, not
    # recomputed from the simulation: a centring that moved with the parameter vector
    # would be a new nonlinearity rather than a reparameterisation, and the estimate would
    # then depend on the simulation draw. Required, not defaulted -- silently falling back
    # to 0.0 would leave kappa_0 on the old uncentred scale while its BOX is on the new
    # one, which is a factor-180 error and would look like a bad fit.
    haskey(raw, "m_psychic") || error("""
        target file $path has no m_psychic. It predates the centred psychic cost
        (2026-09-10). Regenerate it:
          uv run --with pandas --with numpy python tools/make_smm_targets.py""")
    m_psychic = Float64(raw["m_psychic"])
    isfinite(m_psychic) && 0 < m_psychic < 20 ||
        error("m_psychic = $m_psychic is not a plausible mean log W-score: " * path)

    # ---- the moment covariance ----------------------------------------------
    # The weighting matrix and every standard error come from here. Its row order MUST be
    # SMM_MOMENTS: a silent permutation would weight each residual by another moment's
    # precision and there would be no symptom except a wrong answer.
    haskey(raw, "moment_cov") || error("target file $path has no [moment_cov] block: " * path)
    mc = raw["moment_cov"]
    cov_names = String.(mc["names"])
    cov_names == collect(SMM_MOMENTS) || error("""
        [moment_cov] in $path is ordered
            $(join(cov_names, ", "))
        but moments.jl expects
            $(join(SMM_MOMENTS, ", "))
        These index the same vector, so they must be identical and in the same order.
        Regenerate the targets.""")
    se = Float64.(mc["se"])
    length(se) == length(SMM_MOMENTS) || error("[moment_cov].se has the wrong length: " * path)
    all(x -> isfinite(x) && x > 0, se) ||
        error("[moment_cov].se has a non-positive or non-finite entry: " * path)
    cov = [Float64.(row) for row in mc["cov"]]
    Sigma = reduce(vcat, (reshape(r, 1, :) for r in cov))
    size(Sigma) == (length(SMM_MOMENTS), length(SMM_MOMENTS)) ||
        error("[moment_cov].cov is not $(length(SMM_MOMENTS))x$(length(SMM_MOMENTS)): " * path)

    haskey(raw, "tas_wealth_winsor_cut") || error(
        "target file has no tas_wealth_winsor_cut; regenerate the targets: " * path)
    wcut = Float64(raw["tas_wealth_winsor_cut"])
    isfinite(wcut) && wcut > 0 || error("tas_wealth_winsor_cut is not positive: " * path)

    out = Dict{String,NamedTuple}("_spec" => (school_time = school,
                                              m_psychic  = m_psychic,
                                              wealth_cut = wcut,
                                              se         = se,
                                              Sigma      = Sigma,
                                              cov_names  = cov_names,
                                              n_clusters = Int(mc["n_clusters"])))
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

# Metadata travels with the frozen targets to every solve, including diagnostics.
# Fail on a missing schedule rather than silently reverting to the legacy model.
target_school_time(targets) = targets["_spec"].school_time
target_m_psychic(targets)   = targets["_spec"].m_psychic
target_wealth_cut(targets)  = targets["_spec"].wealth_cut
target_se(targets)          = targets["_spec"].se
target_Sigma(targets)       = targets["_spec"].Sigma

"""
    moment_weights(targets) -> Vector{Float64}

Diagonal inverse-variance weights, in `SMM_MOMENTS` order (decision 2026-09-10).

    Q = sum_j w_j * (m_j - mhat_j)^2 ,      w_j = 1 / se_j^2

so each residual is measured in STANDARD ERRORS of its own moment. This replaces the
proportional-error scaling that the square ten-moment system used: with seventeen moments
against fourteen parameters the system is over-identified, Q cannot reach zero, and the
relative weights therefore decide the answer rather than merely the path to it.

WHY THE DIAGONAL AND NOT THE FULL INVERSE. `target_Sigma` carries the full joint 17x17
covariance and it IS used -- for standard errors, sensitivity, and the identification
diagnostics. It is not used as the first-stage weight, because its cross-block terms rest
on the 488 family clusters the two samples share out of 2,629, and a noisily-estimated
optimal weight can move an estimate further than the efficiency it buys.

DO NOT READ THE CROSS-BLOCK CORRELATIONS AS A JUSTIFICATION. They are small -- -0.0350 to
+0.0453 across all 70 parent-by-TAS pairs -- but the correlations a diagonal weight actually
discards are the WITHIN-block ones, and the largest of those is +0.676. Whether the diagonal
costs much efficiency here is an open question for a second-stage comparison; it is not
settled by the cross-block figure, and an earlier version of this comment claimed it was.

READ THE WEIGHT CONCENTRATION IN THE REPORT BEFORE TRUSTING A FIT. Inverse-variance
weighting is efficient when the model can fit the moments to within sampling error. This
model cannot: `mean_hc_late` has an SE of 0.0014 against a current miss of ~0.19 log
points, i.e. ~137 standard errors, so it alone can claim most of Q and turn a seventeen-
moment estimation into a one-moment one. `report_fit` prints each moment's share of Q for
exactly this reason. If one moment dominates, the answer is a scale decision to be taken
deliberately -- not a wider box.
"""
moment_weights(targets) = 1.0 ./ (target_se(targets) .^ 2)

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
    # The child only chooses its time input from T_CHILD_VOICE; before that sim_i is not
    # a decision. Match the generator, which selects Child_Age >= 6 for the early group.
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
            mean_hc_late   = loghc(SMM_AGE_HC_LATE_LO:SMM_AGE_HI),
            n_nonfinite    = n_bad[])
end

"""
    rank_tertiles(x) -> Vector{Int}

Assign each element to a within-sample tertile 1..3 BY RANK.

Rank-based rather than at the data's absolute W-score cut points (decision 2026-09-10):
the model's HC level is already targeted by `mean_hc_late`, and cutting the simulation at
the data's cut points would fold any level or dispersion miss into these three moments and
so into `kappa_theta`. Cutting at the simulation's own terciles asks only what they exist
to ask -- how steeply completion rises with rank in ability.

`MergeSort` is passed explicitly because ties must break the same way on every evaluation.
With an unstable sort two draws that differ only in simulation noise could shuffle tied
households between tertiles and move the moment without any parameter having changed,
which turns the objective into a step function and defeats common random numbers.
"""
function rank_tertiles(x::AbstractVector{<:Real})
    n = length(x)
    ord = sortperm(x; alg = MergeSort)     # stable: equal values keep their index order
    t = Vector{Int}(undef, n)
    @inbounds for (rank, i) in enumerate(ord)
        t[i] = min(3, 1 + div(3 * (rank - 1), n))
    end
    return t
end

"""
    tas_moments(r, targets) -> NamedTuple

The seven TAS moments, computed from the RESIMULATED child block.

Each one is the model counterpart of a ratio of means over the TAS frame, so each is a
plain subgroup mean here -- the data's denominator indicator is the subgroup membership,
and the model has no missing outcomes to condition on.
"""
function tas_moments(r, targets)
    p, ch = r.parent, r.child
    col = ch.sim_college
    n_bad = 0

    function share(mask)
        v = col[mask]
        w = filter(isfinite, v)
        n_bad += length(v) - length(w)
        isempty(w) ? NaN : mean(w)
    end

    # --- kappa_0: the overall completion rate ---
    all_i = trues(length(col))
    k0 = share(all_i)

    # --- kappa_theta: completion by ability tertile ---
    # Column t of sim_hc IS child age t, so this is the child's HC at age 17 -- the last
    # family-stage column, matching the data's `ach_age == 17` assessment, and one period
    # before the age-18 handoff the enrolment decision is actually taken on.
    hc17 = p.sim_hc[:, SMM_TAS_ACH_AGE]
    n_bad += count(x -> !(isfinite(x) && x > 0), hc17)
    tert = rank_tertiles(hc17)
    kth = ntuple(k -> share(tert .== k), 3)

    # --- kappa_ParEd: completion by parental education ---
    # The model's BothCollege against the data's EITHER-parent group. Open mismatch, by
    # instruction; docs/ERRORS.md P7c has the measured size of it.
    bc = p.sim_k[:, 1]
    kpe0 = share(bc .< 0.5)
    kpe1 = share(bc .>= 0.5)

    # --- kappa_terminal: parental assets RETAINED after the transfer ---
    # `retained = sim_a_init - sim_tr_init`, i.e. the parent's terminal assets less what it
    # handed over. NOT the pre-transfer age-18 assets: those are a different object and are
    # ~30% larger at the incumbent.
    #
    # The SAME winsorisation as the data, so the two sides are the same functional. It is
    # expected to bind on nobody -- the cut is $4.45m and the child's asset grid stops at
    # a_max = 100 model units = $1m -- and `n_winsorised` reports whether that held.
    cut = target_wealth_cut(targets)
    ret = r.retained
    n_bad += count(!isfinite, ret)
    fin_ret = filter(isfinite, ret)
    n_wins = count(>(cut), fin_ret)
    kterm = isempty(fin_ret) ? NaN : mean(min.(fin_ret, cut))

    return (k0_complete = k0,
            kth_ga17_t1_c = kth[1], kth_ga17_t2_c = kth[2], kth_ga17_t3_c = kth[3],
            kpe_g0_c = kpe0, kpe_g1_c = kpe1,
            kterm_x_strict_w99 = kterm,
            # diagnostics, not targeted
            n_college = count(isequal(1.0), col),
            n_bothcollege = count(>=(0.5), bc),
            mean_transfer = (v = filter(isfinite, r.transfers); isempty(v) ? NaN : mean(v)),
            retained_negative = count(x -> isfinite(x) && x < 0, ret),
            n_winsorised = n_wins,
            tas_nonfinite = n_bad)
end

"""
    model_moments(r, targets) -> NamedTuple

All seventeen targeted moments plus the diagnostics, from one pipeline result.
"""
function model_moments(r::NamedTuple, targets)
    pm = model_moments(r.parent)
    tm = tas_moments(r, targets)
    return merge(pm, tm, (n_nonfinite = pm.n_nonfinite + tm.tas_nonfinite,))
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

Assets AND human capital are checked over ALL T+1 columns. Column T+1 is the terminal
state at the age-18 handoff: `sim_a[:, T+1]` becomes the child's initial assets and
`sim_hc[:, T+1]` becomes its initial `k`. Excluding it hides exactly the column that
propagates into the next block -- and HC was excluded until 2026-09-06 (A1).
"""
function simulation_violations(p::Parent_child_interaction_age_specific_AR1)
    cols = SMM_AGE_LO:SMM_AGE_HI
    tol  = SIM_FEAS_TOL
    C, E, H, Tp = p.sim_c[:, cols], p.sim_e[:, cols], p.sim_h[:, cols], p.sim_t[:, cols]
    I           = p.sim_i[:, cols]
    # A1: HUMAN CAPITAL IS CHECKED OVER ALL T+1 COLUMNS, ASSETS TOO.
    #
    # Columns 1..T are the family stage; column T+1 is the state at the age-18 handoff.
    # `sim_hc[:, T+1]` BECOMES the child's `sim_k_init` and `sim_a[:, T+1]` its initial
    # assets, so a non-finite or non-positive value there propagates straight into the
    # child block -- and it was the one column the HC check did not look at. The child's
    # wage and psychic cost both take `log(theta)`, so a non-positive handoff is a domain
    # error there, not merely a bad fit here.
    HC          = p.sim_hc[:, 1:(p.T + 1)]
    A           = p.sim_a[:, 1:(p.T + 1)]

    nf = sum(M -> count(!isfinite, M), (C, E, H, Tp, I)) +
         count(!isfinite, HC) + count(!isfinite, A)
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
         child_leisure_negative  = count(fin(x -> x < -tol), 1.0 .- Tp .- I .- reshape(p.school_time[1:p.T], 1, :)),
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
    # state outside the solved grid silently reuses the policy at the boundary node. That
    # is defensible for a thin tail and indefensible if the mass lives out there, so it is
    # measured rather than assumed -- for BOTH state variables the parent carries on a
    # grid, assets and human capital.
    #
    # MEASURED over ALL T+1 columns, and reported as HOUSEHOLDS, not as a share of cells.
    # An earlier version did neither, and the conclusion drawn from it was wrong: it
    # compared the fraction of HOUSEHOLDS above at t=1 (0.1%) against the fraction of
    # CELLS above over t=1..17 (0.1%), read the equality as "all of it is the initial
    # draw", and reported that. The two numbers match only because 2 households x 17
    # periods / 34,000 cells equals 2 / 2,000 -- different denominators, same digits.
    #
    # What is actually true at the incumbent: 2 households sit above the asset ceiling in
    # EVERY period 1..17, and 7 are above at the T+1 handoff, peaking at 259 against a
    # ceiling of 100. So five of them cross DURING the family stage; it is not only
    # initial wealth. The handoff column matters most of all -- it becomes the child's
    # initial assets -- and the old diagnostic excluded it from both the share and the
    # maximum.
    #
    # HC was not measured at all before. It has a FLOOR as well as a ceiling (hc_min = 50
    # in W-score units), and a state below the floor is extrapolated just as silently as
    # one above the ceiling, so both ends are counted.
    a_hi  = maximum(p.a_grid)
    a_lo  = minimum(p.a_grid)
    hc_hi = maximum(p.hc_grid)
    hc_lo = minimum(p.hc_grid)
    Aall  = p.sim_a[:, 1:(p.T + 1)]
    Hall  = p.sim_hc[:, 1:(p.T + 1)]
    n_sim = size(Aall, 1)

    # PER-PERIOD counts and maxima, so "a thin tail" can be checked against where and when
    # it happens rather than inferred from a single pooled share. Column T+1 is the
    # handoff and is included; it is the column that propagates into the child block.
    per_period_over(M, hi) = [count(x -> isfinite(x) && x > hi, view(M, :, t)) for t in 1:size(M, 2)]
    per_period_max(M)      = [(v = filter(isfinite, view(M, :, t)); isempty(v) ? NaN : maximum(v))
                              for t in 1:size(M, 2)]

    return (income = inc, resources = res,
            saving_rate = (res - c - e) / res,
            terminal_assets = nanmean(p.sim_a[:, p.T + 1]),
            h_p = nanmean(vec(p.sim_h[:, cols])),
            t_p = nanmean(vec(p.sim_t[:, cols])),
            n_sim = n_sim,
            # --- assets ---
            a_grid_min        = a_lo,
            a_grid_max        = a_hi,
            a_max_sim         = maximum(Aall),          # over ALL columns, handoff included
            a_min_sim         = minimum(Aall),
            a_hh_ever_above   = count(i -> any(view(Aall, i, :) .> a_hi), 1:n_sim) / n_sim,
            a_hh_above_t1     = mean(p.sim_a[:, 1] .> a_hi),
            a_hh_above_handoff= mean(p.sim_a[:, p.T + 1] .> a_hi),
            a_cell_above_flow = mean(p.sim_a[:, cols] .> a_hi),
            a_over_by_period  = per_period_over(Aall, a_hi),
            a_max_by_period   = per_period_max(Aall),
            # --- human capital ---
            hc_grid_min       = hc_lo,
            hc_grid_max       = hc_hi,
            hc_max_sim        = maximum(Hall),
            hc_min_sim        = minimum(Hall),
            hc_hh_ever_above  = count(i -> any(view(Hall, i, :) .> hc_hi), 1:n_sim) / n_sim,
            hc_hh_ever_below  = count(i -> any(view(Hall, i, :) .< hc_lo), 1:n_sim) / n_sim,
            hc_hh_above_handoff = mean(p.sim_hc[:, p.T + 1] .> hc_hi),
            hc_over_by_period = per_period_over(Hall, hc_hi),
            hc_under_by_period= [count(x -> isfinite(x) && x < hc_lo, view(Hall, :, t))
                                 for t in 1:size(Hall, 2)],
            hc_max_by_period  = per_period_max(Hall))
end

# -----------------------------------------------------------------------------
# Estimated parameters
# -----------------------------------------------------------------------------
# Bounded parameters are searched on a LINKED scale so the optimizer cannot walk
# out of the economically meaningful region: strictly-positive weights are
# searched in logs, so a step can never produce a negative weight. sigma_2_0 is a
# log-elasticity already (sigma_2 = exp(sigma_2_0 + sigma_2_1*(t-1))) and is
# searched in levels inside a box.

struct SMMParam
    name::Symbol
    lo::Float64
    hi::Float64
    link::Symbol       # :log or :level
    owner::Symbol      # :parent or :child -- which constructor the value is routed to
end

# Backward-compatible constructor: an SMMParam with no stated owner is a parent parameter,
# which is what every one of them was before 2026-09-10.
SMMParam(name, lo, hi, link) = SMMParam(name, lo, hi, link, :parent)

# =============================================================================
# THE CHILD BLOCK'S CALIBRATION
# =============================================================================
# The child-side counterpart of PARENT_DEFAULTS, and the single place the child's
# non-grid configuration is written down. Before 2026-09-10 these values were inline
# keyword arguments in run_smm.jl's build_child_value() and in run_all.jl, and the two
# had already drifted from the constructor's own defaults: `kappa_terminal` is 10.0 in
# child_lifecycle.jl and 5.0 in both callers. 5.0 is what every run since has actually
# used, so 5.0 is what is recorded here -- and now there is one definition instead of
# three.
#
# The four ESTIMATED entries are starting values; the eight others are fixed.
const CHILD_DEFAULTS = (
    # --- fixed ---
    rho          = 1.5,
    psi_terminal = 0.0,      # by instruction 2026-08-30
    omega        = 0.3,      # altruism
    a_max        = 100.0,    # must cover the parent's terminal assets + 51 periods
    w            = 20.0,
    # --- estimated: the psychic cost of college ---
    # kappa_0 IS ON THE CENTRED SCALE. The legacy uncentred value was 0.2728 with
    # kappa_theta = -0.0342; centring at m_psychic moves it to
    #     kappa_0_centred = 0.2728 + (-0.0342)*m_psychic = 0.0587  at m_psychic = 6.2611
    # which is the same psychic cost for the same child -- see check_psychic_centring.
    # Quoting the old 0.2728 against the new box would put the starting value 0.21 above
    # where it belongs, which at this scale is a large error.
    kappa_0      = 0.0587,
    kappa_theta  = -0.0342,
    kappa_ParEd  = -0.0070,
    # --- estimated: the parent's taste for retained assets ---
    kappa_terminal = 5.0,
)

# The legacy uncentred pair, kept so the centring can be CHECKED rather than trusted.
const LEGACY_KAPPA_0, LEGACY_KAPPA_THETA = 0.2728, -0.0342

"""
    check_psychic_centring(m_psychic)

Verify that `CHILD_DEFAULTS.kappa_0` really is the legacy psychic cost re-expressed at
this centring, rather than a number that was right for some earlier `m_psychic`.

The recentring is behaviourally neutral BY CONSTRUCTION, but only if the starting value
moves with it. If the target file's `m_psychic` changes -- a different achievement age, a
different frame -- and `CHILD_DEFAULTS.kappa_0` does not, then the incumbent silently
becomes a different model. This is the check that turns that into an error.
"""
function check_psychic_centring(m_psychic::Float64)
    implied = LEGACY_KAPPA_0 + LEGACY_KAPPA_THETA * m_psychic
    isapprox(CHILD_DEFAULTS.kappa_0, implied; atol = 5e-4) || error("""
        CHILD_DEFAULTS.kappa_0 = $(CHILD_DEFAULTS.kappa_0) does not match the legacy
        psychic cost re-centred at m_psychic = $m_psychic, which is $implied.

        The legacy uncentred pair is (kappa_0, kappa_theta) = ($LEGACY_KAPPA_0, $LEGACY_KAPPA_THETA)
        and centring maps kappa_0 -> kappa_0 + kappa_theta*m_psychic. Either the target
        file's m_psychic changed, or CHILD_DEFAULTS was edited without re-deriving it.""")
    return nothing
end

"""
    param_default(name) -> Float64

Starting value for an estimated parameter, from whichever block owns it. Both blocks are
searched and an ambiguous name is an error rather than a silent precedence rule -- a
parameter that existed in both would otherwise be routed by declaration order.
"""
function param_default(name::Symbol)
    inp = hasproperty(PARENT_DEFAULTS, name)
    inc = hasproperty(CHILD_DEFAULTS, name)
    inp && inc && error("`$name` is defined in BOTH PARENT_DEFAULTS and CHILD_DEFAULTS; " *
                        "routing it would be arbitrary. Rename one.")
    inp && return getfield(PARENT_DEFAULTS, name)
    inc && return getfield(CHILD_DEFAULTS, name)
    error("`$name` is in neither PARENT_DEFAULTS nor CHILD_DEFAULTS")
end

# The nine-parameter run 2026-09-06_183119 is frozen with its ORIGINAL bounds.
# Exploration bounds for the school-plus-study target pilot (2026-09-07).
# Keep the expanded parental-time/money limits. sigma_4_0 returns to [-6,-1]:
# retain the old incumbent while covering the higher child-time region near -3,
# without spending Sobol points on the old homework-only extension to -8.
# These are pilot choices, not confidence intervals; reassess after a joint fit.
# See docs/BASELINE_9PARAM.md and docs/REVIEW_TRIAGE.md.
const SMM_PARAMS = [
    SMMParam(:phi_2,     0.01, 20.0, :log),
    SMMParam(:phi_3,     0.05, 20.0, :log),
    # UPPER LIMIT RAISED 20 -> 100 (2026-09-08). lambda_2 has climbed across three runs
    # under the school-time targets -- 8.68, then 16.80, then EXACTLY 20.0 -- so the
    # ceiling is now what determines it, not the data.
    #
    # WHY IT CLIMBS, AND WHY THAT IS A SPECIFICATION QUESTION AND NOT ONLY A BOX ONE.
    # `mean_i_c` became `c_time_hrs` (school + own study) and jumped from ~0.039 to ~0.365,
    # i.e. from 4.4 to 41 hrs/wk. In this model the child CHOOSES i_c, trading it against
    # its own leisure, so the only way to make a child voluntarily spend 41 hrs/wk is to
    # make it value skill enormously relative to leisure -- which is what lambda_2 does.
    # But school attendance is COMPULSORY, not chosen. Reproducing a mandate through a
    # taste parameter fits the moment while attributing it to the wrong mechanism, and
    # every counterfactual that moves the return to skill inherits that.
    #
    # So the box is raised to let the estimate come to rest and reveal where it actually
    # wants to be -- but if it lands near 100, or the fit only holds at implausible values,
    # the answer is a modelling change (a compulsory-schooling floor on i_c below age 16,
    # say) rather than a wider box. Flag for Sahber either way: lambda_1 is normalised to
    # 1, so lambda_2 = 20 already means the child weights skill twenty times its leisure.
    SMMParam(:lambda_2,  0.05, 100.0, :log),
    SMMParam(:R_0,       0.5, 100.0, :log),
    SMMParam(:sigma_1_0, -4.0, -0.1,  :level), # upper limit was -0.2
    SMMParam(:sigma_1_1, -0.20, 0.05, :level),
    SMMParam(:sigma_2_0, -5.0, -0.5,  :level),
    # PROVISIONAL lower limit -0.15 (was -0.10, before that -0.05). sigma_2_1 has now
    # been pinned at BOTH previous lower bounds -- -0.049981 against -0.05, then
    # -0.099999 against -0.10 -- so widening has not yet freed it, it has only moved the
    # wall. Widened once more to find out which of two things is true, and the answer is
    # NOT decided by where this run lands:
    #
    #   (a) the box was genuinely binding, in which case Q keeps falling as sigma_2_1
    #       goes more negative and the estimate eventually comes to rest interior;
    #   (b) the objective is flat in this direction -- a ridge with sigma_2_0, whose
    #       scaled Jacobian cosine is 0.869 -- in which case the optimizer simply slides
    #       to whatever wall it is given and -0.15 will pin too.
    #
    # `code/smm/profile_param.jl` is what distinguishes them: it fixes sigma_2_1 at a
    # ladder of values and JOINTLY re-optimizes the other eight at each, so the question
    # is answered by the shape of Q, not by one more boundary hit.
    #
    # For scale: sigma_2_t = exp(sigma_2_0 + sigma_2_1*(t-1)), so -0.10 already means the
    # money elasticity falls 80% over ages 1-17 and -0.15 means 91%. Note also that the
    # e_p profile is already slightly OVER-steep (model 1.19x against data 1.14x), so the
    # pressure is not coming from the moment this slope exists to fit.
    # LOWER LIMIT -0.30 (was -0.15, before that -0.10, before that -0.05).
    #
    # FOURTH BOX, THIRD BOUNDARY HIT. The record:
    #     box [-0.05, 0.05]   ->  -0.049981   pinned
    #     box [-0.10, 0.05]   ->  -0.099999   pinned
    #     box [-0.15, 0.05]   ->  -0.120616   INTERIOR (run 2026-09-07_205033)
    #     box [-0.15, 0.05]   ->  -0.149997   pinned   (run 2026-09-08_100413)
    #
    # The one interior landing came before the baseline was promoted and the off-grid
    # initial assets were resampled; with the new starting point and draw it went back to
    # the wall. So a wider box has never yet produced a STABLE interior estimate, and each
    # widening moves the rest of the vector with it -- sigma_1_0 -0.685 -> -1.246,
    # sigma_2_0 -3.696 -> -4.274, lambda_2 20 -> 54 between the last two runs. Parameters
    # sliding together like that is the signature of a ridge, not of nine separately
    # determined numbers.
    #
    # The step is DELIBERATELY LARGE this time (0.05 -> 0.15 of extra room rather than
    # another 0.05). Three small steps have each cost a full estimation and returned the
    # same answer; a big step either finds an interior optimum or shows that none exists in
    # any reasonable range, and either outcome is worth more than a fifth wall.
    #
    # WHAT IT ALREADY MEANS AT -0.15. sigma_2_t = exp(sigma_2_0 + sigma_2_1*(t-1)), so at
    # the fitted sigma_2_0 = -4.274 the money elasticity runs
    #     t=1  0.0139   t=9  0.0042   t=17  0.0013
    # i.e. it falls 90.9% over the family stage and money is very nearly irrelevant to
    # human capital by adolescence. At -0.30 it would fall 99.2%. Before widening again,
    # ask whether that is a finding or a symptom: `code/smm/profile_param.jl` fixes
    # sigma_2_1 at a ladder of values and jointly re-optimizes the other eight, which is
    # what distinguishes a genuinely binding box from an optimizer sliding along a flat
    # direction. It costs ~3 h against ~13 h for another blind estimation.
    SMMParam(:sigma_2_1, -0.30, 0.05, :level),
    # LOWER LIMIT -10.0 (was -6.0, which was a leftover from the school-plus-study pilot).
    #
    # THE FLOOR BOUND UNDER THE OWN-STUDY SPECIFICATION. 3636d43 recorded this as open and
    # estimated the requirement at about -6.3. MEASURED here at grid 30, simN 2000, every
    # other parameter held at PARENT_DEFAULTS and sigma_4_1 at 0.02, that estimate is well
    # short -- at -6.3 the model still produces nearly three times the study target:
    #
    #     sigma_4_0   i_c early   gap     i_c late   gap        Q
    #        -6.000     0.1421   +262%      0.1157  +133%    6.04
    #        -6.300     0.1153   +193%      0.0907   +83%    3.06
    #        -6.500     0.0994   +153%      0.0766   +54%    1.81
    #        -7.000     0.0666    +70%      0.0492    -1%    0.38
    #        -7.500     0.0432    +10%      0.0310   -38%    0.23   <- univariate minimum
    #        -8.000     0.0274    -30%      0.0192   -61%    0.51
    #
    # So the univariate optimum is near -7.5, not -6.3, and Q is U-shaped around it. The
    # model solves cleanly with zero violations all the way to -12, so the floor is a
    # modelling choice rather than a numerical limit.
    #
    # -10.0 IS DELIBERATELY GENEROUS, and the reason is sigma_2_1: it was widened three
    # times in 0.05 steps, pinned each time, and cost a full estimation on each occasion
    # before a large step finally let it settle at -0.155. A floor 2.5 below the univariate
    # optimum should not need revisiting.
    #
    # The joint optimum will differ from the sweep above. The other nine parameters move,
    # and sigma_4_1 is now ESTIMATED, so the early/late tilt that the sweep cannot resolve
    # -- early wants about -7.5 while late wants about -7.0 -- is exactly what the slope is
    # there to absorb.
    SMMParam(:sigma_4_0, -10.0, -1.0,  :level),
    # Same candidate interval already used by jacobian.jl. This is a search box,
    # not an identification result; reassess it after the first own-study fit.
    SMMParam(:sigma_4_1, -0.05, 0.15, :level),
    # mu_1 remains fixed at PARENT_DEFAULTS.

    # =========================================================================
    # THE FOUR CHILD PARAMETERS -- added 2026-09-10
    # =========================================================================
    # PILOT BOXES, exactly as the sigma bounds above are: they are search regions chosen
    # to contain the answer, not confidence intervals, and they should be reassessed after
    # the first joint fit rather than defended.
    #
    # kappa_0 -- the LEVEL of the psychic cost of college, AT MEAN ABILITY.
    #
    # On the centred scale, so it is no longer the 0.2728 of the uncentred form. The
    # incumbent is 0.0587. The box has to let the college share fall a long way: the model
    # currently produces ~50.8% against a completion target of 32.3%, and the psychic cost
    # is the only parameter that moves that margin directly. Negative values are admitted
    # -- college can be intrinsically attractive -- because nothing rules them out a priori
    # and excluding them would impose the sign the moment is there to measure.
    #
    # MEASURED AFTER THE BOX WAS SET (2026-09-10), AND IT ARGUES FOR A NARROWER ONE.
    # The college margin is close to a STEP in kappa_0. At grid 20 / simN 500, holding the
    # rest at their defaults:
    #
    #     kappa_0    0.059   -0.50   -1.00   -1.50   -2.00
    #     completion 0.000    0.612   1.000   1.000   1.000
    #
    # So the whole transition sits in roughly [-1.0, 0.0] and everything above ~0 is a FLAT
    # ZERO. Two consequences, both confirmed:
    #
    #   * the Jacobian at the incumbent (kappa_0 = 0.0587, share 0) has rank 11/14 at a 2%
    #     step and 12/14 at 5% -- the model is LOCALLY UNIDENTIFIED there, because none of
    #     the six completion moments moves. At an interior-college point it is rank 14/14 at
    #     every step size tested. `tools/check_jacobian_rank.jl` reproduces both.
    #   * an 8-point Sobol smoke test walked kappa_0 UP to 1.87, further into the dead
    #     region, and improved Q on the other moments while all six completion moments
    #     stayed pinned at zero.
    #
    # About 71% of [-2, 5] is that dead region. RECOMMENDATION for the next box, after the
    # first joint fit: [-3, 1]. It covers the transition with margin and spends no Sobol
    # points where the derivative is identically zero. NOT applied here -- [-2, 5] is the
    # agreed pilot box, and a full Sobol stage over 14 dimensions will still find the live
    # region. Raise it with the advisor before the production run.
    SMMParam(:kappa_0, -2.0, 5.0, :level, :child),

    # kappa_theta -- the ABILITY GRADIENT. NEGATIVE: ability lowers the cost.
    #
    # THE BOX IS ~300x THE INCUMBENT, AND DELIBERATELY SO. The incumbent -0.0342 comes
    # from Colas Table 2's RATIO kappa_ParEd/kappa_theta = 0.205, with the levels set to
    # reproduce the old kappa/(HC+1)^4 cost -- it was never fitted to a gradient. What the
    # tertile moments demand is far larger, and the arithmetic is worth keeping:
    #
    #   the T1->T3 gap in completion is 0.629 - 0.113 = 0.516;
    #   the T1->T3 gap in log ability is about 0.076 (g_ACH ~508 against ~548);
    #   the psychic cost is paid for t_college = 4 years, an annuity of ~3.77 at beta=0.97;
    #   the taste shock has sigma_eps = 0.5, which sets the scale a value difference has
    #   to reach before it moves an enrolment decision.
    #
    # So |kappa_theta| * 0.076 * 3.77 has to be of order 1, i.e. |kappa_theta| ~ 3-4. At
    # the incumbent -0.0342 the psychic cost differs by 0.0026 across the whole ability
    # range and the model cannot produce an ability gradient in completion AT ALL. A box of
    # [-0.2, 0] would have looked generous and been hopeless.
    SMMParam(:kappa_theta, -10.0, 0.0, :level, :child),

    # kappa_ParEd -- the PARENTAL-EDUCATION shift. NEGATIVE, per Colas.
    #
    # Same scale argument: the targeted gap is 0.601 - 0.211 = 0.390 over the same 4-year
    # annuity, so |kappa_ParEd| of order 0.3-1.0 is what is being asked for, against an
    # incumbent of -0.0070. NOTE what this parameter is actually estimating here: the data
    # groups are EITHER-parent college and the model's state is BothCollege, so it absorbs
    # a definitional mismatch as well as an effect. Open by instruction -- docs/ERRORS.md
    # P7c has the measured size of the mismatch.
    #
    # MEASURED: THIS PARAMETER SATURATES, and the box is much wider than the region that
    # carries information. At grid 20 / simN 500 with kappa_0 = -0.5:
    #
    #     kappa_ParEd  -0.007   -0.30   -1.00   -2.00
    #     g1 - g0      +0.294   +0.480  +0.480  +0.480
    #
    # g1 reaches 1.000 at -0.30 and cannot rise further, so Q is FLAT in kappa_ParEd below
    # about -0.3 and the optimizer will slide to whatever wall it is given -- the same
    # signature already documented for sigma_2_1. It is also the weakest column of the
    # Jacobian at every step size tested (44.6-51.9 against 120+ for the next weakest), and
    # it is half of the least-identified direction, trading off against kappa_theta.
    # RECOMMENDATION for the next box: [-1, 0].
    SMMParam(:kappa_ParEd, -3.0, 0.0, :level, :child),

    # kappa_terminal -- the parent's taste for the assets it RETAINS after the transfer.
    #
    # Box and link from the previous fourteen-parameter specification
    # (archive/smm_14param_legacy.jl: `Par("kappa_terminal", 0.5, 40.0, 5.00, :log)`),
    # which is the one piece of this design that has been searched before. Log link: it is
    # a strictly positive weight, and a step that made it negative would inverte the sign
    # of log(a_terminal) in the transfer objective.
    SMMParam(:kappa_terminal, 0.5, 40.0, :log, :child),
]

# A moment-count check is necessary but does not establish local identification.
length(SMM_MOMENTS) >= length(SMM_PARAMS) || error("""
    SMM is UNDER-identified: $(length(SMM_PARAMS)) parameters against \
    $(length(SMM_MOMENTS)) moments. Add moments or drop parameters -- the search would
    otherwise wander along a flat direction and return whichever point it started at.""")

# =============================================================================
# PARAMETER ROUTING -- what replaced the parent-only invariant
# =============================================================================
# UNTIL 2026-09-10 this file asserted that every estimated parameter was a PARENT
# parameter, because that is what made the child lifecycle invariant across evaluations
# and let run_smm.jl solve it ONCE per process. That assumption is now false by design:
# four of the fourteen parameters are child parameters.
#
# THE GUARD WAS NOT SIMPLY DELETED. Deleting it would have left a run reusing a stale
# child solve and reporting a converged fit for a model it never solved -- wrong answers,
# no error, which is exactly what the old comment warned about. What replaces it is
# `build_child_solution`, which rebuilds the child from a COMPLETE dependency key, plus
# these two checks:
#
#   1. every estimated parameter is a field of exactly one of the two default sets, so it
#      is routed to exactly one constructor and never silently dropped;
#   2. no child parameter is ever passed to the parent constructor, or the reverse.
#
# The second is the one that would fail silently. `Parent_child_interaction_age_specific_AR1`
# takes keyword arguments; handing it `kappa_0 = 3.1` would either throw or, worse, be
# absorbed by a same-named field, and the child block would keep its default while the
# report printed the estimated value.
let unknown = [q.name for q in SMM_PARAMS
               if !hasproperty(PARENT_DEFAULTS, q.name) && !hasproperty(CHILD_DEFAULTS, q.name)]
    isempty(unknown) || error("""
        SMM_PARAMS names parameter(s) that belong to neither block: $(join(unknown, ", ")).
        Add them to PARENT_DEFAULTS or CHILD_DEFAULTS so they can be routed.""")
end
let mis = [q.name for q in SMM_PARAMS
           if (q.owner === :parent) != hasproperty(PARENT_DEFAULTS, q.name)]
    isempty(mis) || error("""
        SMM_PARAMS declares an owner that disagrees with the default sets for:
        $(join(mis, ", ")). A parameter's `owner` must be :parent exactly when it is a
        field of PARENT_DEFAULTS, and :child otherwise.""")
end

const SMM_CHILD_PARAMS  = Tuple(q.name for q in SMM_PARAMS if q.owner === :child)
const SMM_PARENT_PARAMS = Tuple(q.name for q in SMM_PARAMS if q.owner === :parent)

"""
    split_params(kw) -> (parent_kw, child_kw)

Route the unpacked parameter vector to the two constructors. Nothing is shared and nothing
is dropped: the two NamedTuples partition `kw`, which is asserted rather than assumed.
"""
function split_params(kw::NamedTuple)
    pn = Tuple(n for n in keys(kw) if hasproperty(PARENT_DEFAULTS, n))
    cn = Tuple(n for n in keys(kw) if !hasproperty(PARENT_DEFAULTS, n))
    bad = [n for n in cn if !hasproperty(CHILD_DEFAULTS, n)]
    isempty(bad) || error("""
        split_params cannot route: $(join(bad, ", ")) is in neither PARENT_DEFAULTS nor
        CHILD_DEFAULTS. Passing it to either constructor would either throw or be absorbed
        by a same-named field while the block kept its default -- a wrong answer with no
        error. Add it to the appropriate default set.""")
    pk = NamedTuple{pn}(map(n -> getfield(kw, n), pn))
    ck = NamedTuple{cn}(map(n -> getfield(kw, n), cn))
    length(pk) + length(ck) == length(kw) ||
        error("split_params lost a parameter: $(length(pk)) + $(length(ck)) != $(length(kw))")
    return pk, ck
end

"""
    named_point(vals::AbstractDict{Symbol,Float64}) -> NamedTuple

A `Dict`-shaped parameter point as a NamedTuple, for the diagnostic tools, which build
their points by name rather than as a search vector.
"""
named_point(vals::AbstractDict{Symbol,<:Real}) =
    NamedTuple{Tuple(keys(vals))}(Tuple(Float64.(values(vals))))

"""
    residuals_from(m, targets; weights) -> Vector{Float64}

The residual vector every tool must agree on: `sqrt(w_j) * (m_j - mhat_j)`, in
`SMM_MOMENTS` order, so that `sum(residuals_from(...).^2)` IS `smm_objective`'s Q.

BEFORE 2026-09-10 each of jacobian.jl, sensitivity.jl, profile_param.jl and
grid_sensitivity.jl computed its own residual with `moment_scale`, and each also built its
own child value function. Four copies of an objective is four chances for one of them to
drift from the one being optimised, and a Jacobian of the wrong objective is not a
diagnostic of anything. There is now exactly one definition, here.
"""
function residuals_from(m, targets; weights::Union{Nothing,Vector{Float64}} = nothing)
    w = weights === nothing ? moment_weights(targets) : weights
    return [sqrt(w[j]) * (getfield(m, Symbol(k)) - targets[k].mean)
            for (j, k) in enumerate(SMM_MOMENTS)]
end

"""
    evaluate_at(vals, targets; kwargs...) -> NamedTuple

Run the full pipeline at a NAMED parameter point and return the residuals, moments and
validity counts. This is the shared entry point for the Jacobian, sensitivity, profiling,
grid-check and standard-error tools; all of them used to hold their own copy of it.

Any parameter not named in `vals` stays at its own block's default.
"""
function evaluate_at(vals::AbstractDict{Symbol,<:Real}, targets;
                     Na::Int = 30, Nk::Int = 2, Nhc::Int = 30,
                     simN::Int = 2000, seed::Int = 1234,
                     child_grid = (Na = 30, Nk = 30, Nt = 5),
                     parent_extra::NamedTuple = (;),
                     weights::Union{Nothing,Vector{Float64}} = nothing,
                     demo_sim::Bool = false)
    r = run_pipeline(named_point(vals), targets; Na = Na, Nk = Nk, Nhc = Nhc,
                     simN = simN, seed = seed, child_grid = child_grid,
                     parent_extra = parent_extra, demo_sim = demo_sim)
    m = model_moments(r, targets)
    v = simulation_violations(r.parent)
    return (r = residuals_from(m, targets; weights = weights),
            moments = m, nviol = v.total, nbad = m.n_nonfinite, pipeline = r)
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

incumbent() = [to_search(param_default(q.name), q) for q in SMM_PARAMS]

# =============================================================================
# THE CHILD SOLUTION -- rebuilt per evaluation, from a complete dependency key
# =============================================================================
# WHAT CHANGED, AND WHY THE OLD DESIGN CANNOT SIMPLY BE PATCHED.
#
# run_smm.jl used to compute `V_CHILD = build_child_value()` ONCE per process and hand the
# same spline to every evaluation. That was exact while every estimated parameter was a
# parent parameter. It is now false: kappa_0, kappa_theta and kappa_ParEd change the
# child's college value and kappa_terminal changes the parent's transfer objective, so a
# reused spline would answer for a model the run never solved.
#
# WHAT IS ACTUALLY REUSABLE, MEASURED RATHER THAN ASSUMED. The child solve has four
# stages, and only some of them read the four parameters:
#
#   solve_model_work!            6.31 s   high-school path. Reads NONE of the four.
#   solve_model_college! stage 1 ~5.8 s   graduate working life, E = 1, ordinary `util`.
#                                         Reads NONE of the four.
#   solve_model_college! stage 2 ~0.5 s   the t_college study years. Reads kappa_0,
#                                         kappa_theta, m_psychic through util_college.
#   optimal_transfer_*!          0.50 s   reads kappa_terminal (and the college value).
#   terminal_value_spline        ~0.00 s  reads kappa_ParEd, at the enrolment max.
#
# So the two expensive stages -- 12.1 s of the 12.9 s -- are invariant across the whole
# search, and only ~1.2 s of it has to be redone. VERIFIED: refreshing from cached work
# and graduate blocks reproduces a full re-solve BIT-IDENTICALLY (max |diff| = 0.000e+00
# on sol_v_college, sol_c_college and sol_h_college, and an identical NaN feasibility
# pattern) at 16.9x the speed. This is not an approximation and there is no tolerance to
# tune -- the arrays are copied, not refitted.
#
# THE CACHE STORES ARRAYS, NOT A MODEL. Deliberately. A cached model would carry mutable
# `sim_*` state, and any simulator that touched it would contaminate every later
# evaluation with another parameter draw's simulation -- the exact hazard the brief warns
# about. Storing six immutable copies of the solution arrays makes that structurally
# impossible: nothing is ever simulated on the cached object because there is no cached
# object to simulate.

"""
    child_config(targets; Na, Nk, Nt, simN, seed) -> NamedTuple

The child's complete NON-ESTIMATED configuration, and therefore the cache key.

Everything the reusable stages depend on is in here and every estimated child parameter is
out of it. If a value belongs in the key and is missing, two different models share a cache
entry and the second silently inherits the first's solution; if an estimated parameter
leaks IN, the cache never hits and the run is merely slow. The first failure is silent, so
the key is written out in full rather than derived.
"""
child_config(targets; Na::Int, Nk::Int, Nt::Int, simN::Int, seed::Int) =
    (Na = Na, Nk = Nk, Nt = Nt, simN = simN, seed = seed,
     rho          = CHILD_DEFAULTS.rho,
     psi_terminal = CHILD_DEFAULTS.psi_terminal,
     omega        = CHILD_DEFAULTS.omega,
     a_max        = CHILD_DEFAULTS.a_max,
     w            = CHILD_DEFAULTS.w,
     m_psychic    = target_m_psychic(targets))

"""
    child_solve_key(cfg) -> NamedTuple

`cfg` less the settings that cannot change a SOLUTION: `simN` and `seed`.

Both are simulation settings. `solve_model_work!` and `solve_model_college!` read grids,
parameters and the shock discretisation; neither reads `draws_uniform_*`, `sim_*` or the
RNG. Leaving them in the key was not wrong -- it produced cache MISSES, not stale hits --
but it meant the report grid, the search grid and every diagnostic tool with a different
`simN` each re-solved the same 12.1 s of work-and-graduate block. Dropping them is safe
precisely because the cache stores solution arrays and nothing else.
"""
child_solve_key(cfg::NamedTuple) =
    Base.structdiff(cfg, NamedTuple{(:simN, :seed)})

# key -> the six kappa-independent solution arrays. One entry per process; the search grid
# and the report grid are different configurations and get different entries.
const CHILD_BASE_CACHE = Dict{Any,NamedTuple}()
const CHILD_BASE_FIELDS = (:sol_c_work, :sol_h_work, :sol_v_work,
                           :sol_c_grad, :sol_h_grad, :sol_v_grad)

# The FULL child solution -- every solution array plus the parent's terminal-value object --
# keyed on the solve configuration AND the four kappas.
#
# WHY A SECOND TIER. Tier A saves the 12.1 s that no estimated parameter can change. But a
# finite difference in a PARENT parameter changes no kappa either, so the ~1.2 s of study
# years and transfer stage is also being redone for nothing -- across the ten parent
# columns of a Jacobian, or a profile of a parent parameter, that is every evaluation.
#
# BOUNDED AT TWO ENTRIES, deliberately. Each entry is ~90 MB at grid 30 and this runs on
# 20 worker processes on a shared machine. Two is enough for the access pattern that
# motivates it -- consecutive evaluations at the same kappas -- and during the search
# proper, where every draw moves a kappa, the tier simply never hits and costs one dict
# lookup. It is a cache for the diagnostics, not for the optimizer.
const CHILD_FULL_CACHE = Dict{Any,NamedTuple}()
const CHILD_FULL_LIMIT = 2
const CHILD_FULL_FIELDS = (CHILD_BASE_FIELDS...,
                           :sol_c_college, :sol_h_college, :sol_v_college,
                           :sol_tr_college, :sol_tr_work,
                           :sol_tr_v_college, :sol_tr_v_work)

"""
    child_base(cfg) -> NamedTuple of solved, kappa-independent arrays

Solve (once per process per configuration) the two stages that no estimated parameter
touches. The kappa values used here are irrelevant to what is returned -- they affect only
the study years and the transfer stage, neither of which is stored -- but they must be
SOME valid values, so the block's own defaults are used.
"""
function child_base(cfg::NamedTuple)
    return get!(CHILD_BASE_CACHE, child_solve_key(cfg)) do
        # simN = 2 because the base model is NEVER simulated -- only its solution arrays
        # are copied out. Building it at the caller's simN allocated sim arrays that were
        # then thrown away, on every process.
        kap = (kappa_0 = CHILD_DEFAULTS.kappa_0, kappa_theta = CHILD_DEFAULTS.kappa_theta,
               kappa_ParEd = CHILD_DEFAULTS.kappa_ParEd,
               kappa_terminal = CHILD_DEFAULTS.kappa_terminal,
               simN = 2, seed = 1234)
        m = ConSavLaborCollege_AR1(; merge(cfg, kap)...)
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do      # ProgressMeter writes to STDERR
                solve_model_work!(m)
                solve_model_college!(m)      # stage 1 of this is the graduate block
            end
        end
        NamedTuple{CHILD_BASE_FIELDS}(map(f -> copy(getfield(m, f)), CHILD_BASE_FIELDS))
    end
end

"""
    build_child_solution(ckw, targets; Na, Nk, Nt, simN, seed) -> (child, V_child)

A child model solved at THIS draw's kappa values, plus the parent's terminal-value object.

Returns a FRESH model every call. Nothing is mutated in place across evaluations, so a
partly-failed solve cannot leave state behind for the next draw to inherit.

The terminal value is constructed from the SOLVED value functions -- `terminal_value_spline`
reads `sol_tr_v_college` and `sol_tr_v_work` -- not from any simulation.
"""
# The four estimated child parameters, and their SMM starting values. `CHILD_DEFAULTS`
# also holds the eight fixed settings, so this is the subset a partial parameter point
# must be completed with.
const CHILD_ESTIMATED = (:kappa_0, :kappa_theta, :kappa_ParEd, :kappa_terminal)
const CHILD_ESTIMATED_DEFAULTS =
    NamedTuple{CHILD_ESTIMATED}(map(n -> getfield(CHILD_DEFAULTS, n), CHILD_ESTIMATED))

function build_child_solution(ckw::NamedTuple, targets;
                              Na::Int, Nk::Int, Nt::Int, simN::Int, seed::Int)
    cfg = child_config(targets; Na = Na, Nk = Nk, Nt = Nt, simN = simN, seed = seed)

    # EVERY ESTIMATED CHILD PARAMETER IS SET EXPLICITLY, even when the caller omitted it.
    #
    # `merge(cfg, ckw)` alone left an omitted kappa at the CONSTRUCTOR's default, which is
    # not the SMM's: `kappa_0` would fall back to 0.2728 -- the LEGACY UNCENTRED value, on
    # a scale where the centred incumbent is 0.0587 -- and `kappa_terminal` to 10.0 against
    # an SMM default of 5.0. The runner always passes all fourteen and so never hit it, but
    # `evaluate_at` documents a partial-point interface, and every diagnostic tool uses it:
    # a Jacobian column for a PARENT parameter would have been computed on a child block
    # the estimation never uses. Silent, and wrong in the direction that looks plausible.
    kap = merge(CHILD_ESTIMATED_DEFAULTS, ckw)
    stray = [n for n in keys(ckw) if !(n in CHILD_ESTIMATED)]
    isempty(stray) || error("""
        build_child_solution was handed child parameter(s) it does not know how to
        complete: $(join(stray, ", ")). Add them to CHILD_ESTIMATED, or they will be
        merged into the constructor without being part of the cache key -- which would
        make the tier-B cache return a solution for different parameter values.""")

    full_key = (child_solve_key(cfg), kap)
    hit = get(CHILD_FULL_CACHE, full_key, nothing)
    ch  = ConSavLaborCollege_AR1(; merge(cfg, kap)...)

    if hit !== nothing
        # A FRESH model, filled from cached ARRAYS. The cached entry is never handed out
        # and never simulated, so no `sim_*` state can cross between evaluations.
        for f in CHILD_FULL_FIELDS
            copyto!(getfield(ch, f), getfield(hit, f))
        end
        return ch, hit.V_child
    end

    base = child_base(cfg)
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            for f in CHILD_BASE_FIELDS
                copyto!(getfield(ch, f), getfield(base, f))
            end
            # reuse_grad asserts the graduate block is already solved; see the seam in
            # child_lifecycle.jl for why skipping it is exact.
            solve_model_college!(ch; reuse_grad = true)
            optimal_transfer_work!(ch)
            optimal_transfer_college!(ch)
        end
    end
    V = terminal_value_spline(ch; s = 10.0)

    # Bounded, and cleared rather than evicted one at a time: the access pattern this
    # serves is "the same kappas repeatedly", so a stale pair is worth nothing and the
    # memory is worth a lot on 20 shared processes.
    length(CHILD_FULL_CACHE) >= CHILD_FULL_LIMIT && empty!(CHILD_FULL_CACHE)
    CHILD_FULL_CACHE[full_key] =
        merge(NamedTuple{CHILD_FULL_FIELDS}(map(f -> copy(getfield(ch, f)), CHILD_FULL_FIELDS)),
              (V_child = V,))
    return ch, V
end

# =============================================================================
# THE FULL EVALUATION PIPELINE
# =============================================================================
"""
    run_pipeline(kw, targets; Na, Nk, Nhc, simN, seed, child_grid, demo_sim) -> NamedTuple

One complete evaluation of the model, in the order the specification fixes:

    solve child lifecycle and transfer problems
      -> initial child simulation
      -> construct the child terminal-value object
      -> solve and simulate parents
      -> initialize children from simulated parent outcomes
      -> resimulate children
      -> calculate all moments

THE INITIAL CHILD SIMULATION IS A DEMONSTRATION AND SUPPLIES NOTHING. It runs on the
child's own `sim_a_init` -- a lognormal draw, not the parent's terminal assets -- so its
college shares and transfers describe a population that does not exist in this model's
equilibrium. Its outputs are therefore ERASED before the parent block runs, and
`model_moments` refuses to compute a TAS moment from a model whose handoff arrays are
still NaN. That is the mechanism by which an initial simulation cannot leak into a final
moment; a comment saying "don't use this" would not be one.

COMMON RANDOM NUMBERS AND MATCHING SIZES. The parent and the child are built with the same
`seed` and the same `simN`, and both are asserted. Without matching sizes the handoff
`child.sim_a_init .= parent.sim_a[:, T+1]` is a length error at best and a silent recycle
at worst; without a common seed the objective is a step function of simulation noise and no
derivative-free search converges on it.
"""
function run_pipeline(kw::NamedTuple, targets;
                      Na::Int = 30, Nk::Int = 2, Nhc::Int = 30,
                      simN::Int = 2000, seed::Int = 1234,
                      child_grid = (Na = 30, Nk = 30, Nt = 5),
                      parent_extra::NamedTuple = (;),
                      demo_sim::Bool = true)
    pk, ck = split_params(kw)

    # ---- 1. child lifecycle and transfer problems ---------------------------
    child, V_child = build_child_solution(ck, targets;
                                          Na = child_grid.Na, Nk = child_grid.Nk,
                                          Nt = child_grid.Nt, simN = simN, seed = seed)

    # ---- 2. initial (demonstration) child simulation ------------------------
    if demo_sim
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                simulate_model_child!(child)
            end
        end
        # ERASE IT. See the docstring: these are outcomes for children who never met the
        # simulated parents, and nothing downstream may read them.
        fill!(child.sim_college, NaN)
        fill!(child.sim_tr_init, NaN)
    end

    # ---- 3. the terminal-value object ---------------------------------------
    # Already built from the solved value functions inside build_child_solution; named
    # here so the ordering of the specification is visible in the code.

    # ---- 4. solve and simulate the parents ----------------------------------
    # `parent_extra` carries NON-ESTIMATED parent constructor settings that a diagnostic
    # needs to vary -- `a_max` for the asset-grid sweep is the only current use. It is
    # applied AFTER `pk`, so it can never silently override an estimated parameter without
    # saying so: that collision is an error, not a precedence rule.
    clash = [n for n in keys(parent_extra) if hasproperty(pk, n)]
    isempty(clash) || error("""
        parent_extra would override estimated parameter(s): $(join(clash, ", ")).
        An estimated parameter must come from the search vector, never from a caller's
        side channel.""")
    parent = Parent_child_interaction_age_specific_AR1(; Na = Na, Nk = Nk, Nhc = Nhc,
                                                         simN = simN, seed = seed,
                                                         school_time = target_school_time(targets),
                                                         pk..., parent_extra...)   # no `w` -- see smm_objective
    parent.V_child_interp = V_child
    redirect_stdout(devnull) do
        solve_model!(parent; verbose = false)
        simulate_model!(parent)
    end

    # ---- 5. initialize children from the simulated parent outcomes ----------
    size(child.sim_a_init, 1) == size(parent.sim_a, 1) || error(
        "handoff size mismatch: child simN = $(size(child.sim_a_init, 1)), " *
        "parent simN = $(size(parent.sim_a, 1)). They must match.")
    child.sim_a_init  .= parent.sim_a[:, parent.T + 1]
    child.sim_k_init  .= parent.sim_hc[:, parent.T + 1]
    # BothCollege is a fixed household type, so any column of parent.sim_k holds it. It
    # feeds the kappa_ParEd term in the child's psychic cost of college. CLAUDE.md records
    # that leaving this at its default of zeros silently switches that term off.
    child.sim_bc_init .= parent.sim_k[:, 1]

    # ---- 6. resimulate the children -----------------------------------------
    _, path_choice, _ = redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            simulate_model_family!(child)
        end
    end

    # The family simulator is the ONLY thing that fills these. If they are still NaN the
    # resimulation did not happen and every TAS moment below would be built on the erased
    # demonstration run.
    any(isnan, child.sim_college) && error(
        "resimulation did not populate sim_college -- the TAS moments would be built on " *
        "the erased demonstration simulation")

    return (parent = parent, child = child, V_child = V_child,
            path_choice = path_choice,
            transfers = copy(child.sim_tr_init),
            # Parental assets RETAINED after the transfer: the model's kappa_terminal
            # object, and what kterm_x_strict_w99 is matched against.
            retained = child.sim_a_init .- child.sim_tr_init,
            parent_kw = pk, child_kw = ck)
end

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

Weights are otherwise EQUAL. Ten moments and ten parameters do not guarantee
an exact fit; if residuals remain, their relative weights still affect the answer.
A covariance-based weighting matrix is not implemented.

Common random numbers: every model is built with the same `seed`, so the initial
draws and shock paths are identical across evaluations. Without this the
objective is a step function of simulation noise and no derivative-free method
converges -- it would be chasing the RNG, not the parameters.
"""
function smm_objective(z::AbstractVector{Float64}, targets;
                       Na::Int = 30, Nk::Int = 2, Nhc::Int = 30,
                       simN::Int = 2000, seed::Int = 1234,
                       child_grid = (Na = 30, Nk = 30, Nt = 5),
                       weights::Union{Nothing,Vector{Float64}} = nothing,
                       demo_sim::Bool = true)
    kw = unpack(z)
    w  = weights === nothing ? moment_weights(targets) : weights

    # ---- reject the infeasible region BEFORE paying for a solve --------------
    if !smm_feasible(kw)
        _penalize!(:infeasible_sigma_2)
        return SMM_PENALTY
    end

    try
        r = run_pipeline(kw, targets; Na = Na, Nk = Nk, Nhc = Nhc, simN = simN,
                         seed = seed, child_grid = child_grid, demo_sim = demo_sim)
        m = model_moments(r, targets)

        # A simulation that leaves the model's domain is not a bad parameter draw, it is
        # an invalid evaluation. Checked by KIND so the penalty log says WHICH economic
        # law broke, not merely that something did.
        viol = simulation_violations(r.parent)
        if viol.total > 0
            worst = argmax(Dict(k => v for (k, v) in pairs(viol) if k !== :total))
            _penalize!(Symbol("invalid_sim_", worst))
            return SMM_PENALTY
        end
        # The child block has its own way of failing: a non-finite handoff, or a
        # resimulation that produced no usable college decision. Neither shows up in the
        # parent's violation count, and both would otherwise be scored as a fit.
        if m.n_nonfinite > 0
            _penalize!(:invalid_sim_child_nonfinite)
            return SMM_PENALTY
        end

        q = 0.0
        for (j, k) in enumerate(SMM_MOMENTS)
            mhat = targets[k].mean
            mj   = getfield(m, Symbol(k))
            isfinite(mj) || return _penalize!(:nonfinite_moment)
            q += w[j] * (mj - mhat)^2
        end
        return q
    catch err
        cause = _root_cause(err)
        if is_model_failure(cause)
            _penalize!(nameof(typeof(cause)))
            return SMM_PENALTY
        end
        rethrow()
    end
end

"""
    smm_objective(z, targets, V_child; kwargs...)

REMOVED. The three-argument form took a child value function solved once per process and
reused for every evaluation, which was exact only while every estimated parameter was a
parent parameter. Four child parameters are now estimated, so a precomputed `V_child` is
stale the moment `kappa_0`, `kappa_theta`, `kappa_ParEd` or `kappa_terminal` moves.

This method exists to FAIL rather than to let an old call site silently score a model that
was never solved.
"""
smm_objective(::AbstractVector{Float64}, targets, V_child; kwargs...) = error("""
    smm_objective(z, targets, V_child) was removed on 2026-09-10.

    A precomputed child value function is no longer valid: kappa_0, kappa_theta,
    kappa_ParEd and kappa_terminal are estimated, and all four change the child solve.
    Call smm_objective(z, targets; ...) instead -- it rebuilds the child block from a
    complete dependency key and reuses only the two stages that provably do not depend
    on any estimated parameter. See build_child_solution.""")

# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
"""
    report_fit(z, targets, V_child; kwargs...)

Re-solve at `z` and print the moment table plus the untargeted diagnostics.
"""
function report_fit(z::AbstractVector{Float64}, targets;
                    Na::Int = 30, Nk::Int = 2, Nhc::Int = 30,
                    simN::Int = 2000, seed::Int = 1234,
                    child_grid = (Na = 30, Nk = 30, Nt = 5),
                    weights::Union{Nothing,Vector{Float64}} = nothing,
                    out::IO = stdout)
    kw = unpack(z)
    w  = weights === nothing ? moment_weights(targets) : weights
    r  = run_pipeline(kw, targets; Na = Na, Nk = Nk, Nhc = Nhc, simN = simN,
                      seed = seed, child_grid = child_grid)
    p  = r.parent
    m  = model_moments(r, targets)
    d  = moment_diagnostics(p)

    println(out, "\nParameters")
    println(out, "-"^62)
    for (i, q) in enumerate(SMM_PARAMS)
        @printf(out, "  %-12s %10.4f   (was %.4f)\n", q.name, getfield(kw, q.name),
                param_default(q.name))
    end

    se = target_se(targets)

    println(out, "\nTargeted moments -- 17 moments, 14 parameters (over-identified)")
    println(out, "-"^104)
    @printf(out, "  %-20s %11s %11s %9s %9s %8s   %s\n",
            "moment", "model", "data", "gap %", "t", "Q share", "source")
    q_tot = 0.0
    qk = Float64[]
    for (jj, k) in enumerate(SMM_MOMENTS)
        mj, mhat = getfield(m, Symbol(k)), targets[k].mean
        qj = w[jj] * (mj - mhat)^2
        push!(qk, qj); q_tot += qj
    end
    for (jj, k) in enumerate(SMM_MOMENTS)
        mj, mhat = getfield(m, Symbol(k)), targets[k].mean
        # For a log moment the "gap %" is the LEVEL gap, exp(dlog) - 1, not the gap in the
        # log -- reporting the latter is what made a 60% error in human capital look like
        # a 7.7% miss.
        gap = k in SMM_LOG_MOMENTS ? 100*(exp(mj - mhat) - 1) : 100*(mj - mhat)/abs(mhat)
        # t is the miss in STANDARD ERRORS of the data moment, which is the scale the
        # weighting actually uses. A |t| of 100 is not a near miss expressed in small
        # units; it is a moment the model cannot reach.
        tstat = (mj - mhat) / se[jj]
        if jj == length(SMM_PARENT_MOMENTS) + 1
            println(out, "  " * "-"^40 * " TAS block " * "-"^40)
        end
        @printf(out, "  %-20s %11.4f %11.4f %8.1f%% %9.1f %7.1f%%   %s\n",
                k, mj, mhat, gap, tstat, 100*qk[jj]/max(q_tot, eps()), targets[k].source)
    end
    @printf(out, "  %-20s %11s %11s %30.4f\n", "Q", "", "", q_tot)

    # WEIGHT CONCENTRATION. Inverse-variance weighting is efficient only if the model can
    # fit the moments to within sampling error. Where it cannot, Q is dominated by whichever
    # moment happens to be most precisely measured, and a seventeen-moment estimation
    # quietly becomes a one-moment one. This line is how that becomes visible instead of
    # being discovered from an implausible estimate.
    ord = sortperm(qk; rev = true)
    top = ord[1:min(3, length(ord))]
    @printf(out, "\n  Q concentration: top 3 moments carry %.1f%% of Q  (%s)\n",
            100*sum(qk[top])/max(q_tot, eps()),
            join((@sprintf("%s %.0f%%", SMM_MOMENTS[t], 100*qk[t]/max(q_tot, eps())) for t in top), ", "))
    if 100*qk[ord[1]]/max(q_tot, eps()) > 60
        @printf(out, "  WARNING: %s alone carries %.0f%% of Q. The other 16 moments are\n",
                SMM_MOMENTS[ord[1]], 100*qk[ord[1]]/max(q_tot, eps()))
        println(out, "           barely influencing the estimate. Treat this as a scaling decision")
        println(out, "           to be taken deliberately, NOT as a converged seventeen-moment fit.")
    end

    # ---- the TAS block in its own units ------------------------------------
    println(out, "\nTAS block -- college completion and terminal wealth")
    println(out, "-"^76)
    @printf(out, "  college completion    %8.4f  vs data %.4f   (%d of %d simulated children)\n",
            m.k0_complete, targets["k0_complete"].mean, m.n_college, size(r.child.sim_college, 1))
    @printf(out, "  ability gradient      T1 %.3f  T2 %.3f  T3 %.3f   (model)\n",
            m.kth_ga17_t1_c, m.kth_ga17_t2_c, m.kth_ga17_t3_c)
    @printf(out, "                        T1 %.3f  T2 %.3f  T3 %.3f   (data)\n",
            targets["kth_ga17_t1_c"].mean, targets["kth_ga17_t2_c"].mean, targets["kth_ga17_t3_c"].mean)
    @printf(out, "                        T3-T1 model %+.3f  vs data %+.3f\n",
            m.kth_ga17_t3_c - m.kth_ga17_t1_c,
            targets["kth_ga17_t3_c"].mean - targets["kth_ga17_t1_c"].mean)
    @printf(out, "  parental education    g0 %.3f  g1 %.3f   (model; %d of %d are BothCollege)\n",
            m.kpe_g0_c, m.kpe_g1_c, m.n_bothcollege, size(r.child.sim_college, 1))
    @printf(out, "                        g0 %.3f  g1 %.3f   (data -- EITHER-parent, see ERRORS.md P7c)\n",
            targets["kpe_g0_c"].mean, targets["kpe_g1_c"].mean)
    @printf(out, "  mean transfer         %8.4f  (%.0f USD)\n",
            m.mean_transfer, m.mean_transfer*DOLLARS_PER_MODEL_UNIT)
    @printf(out, "  retained assets       %8.4f  (%.0f USD)  vs data %.0f USD\n",
            m.kterm_x_strict_w99, m.kterm_x_strict_w99*DOLLARS_PER_MODEL_UNIT,
            targets["kterm_x_strict_w99"].mean*DOLLARS_PER_MODEL_UNIT)
    @printf(out, "  pre-transfer assets   %8.4f  (%.0f USD)  -- NOT the target; the target is post-transfer\n",
            d.terminal_assets, d.terminal_assets*DOLLARS_PER_MODEL_UNIT)
    if m.n_winsorised > 0
        @printf(out, "  NOTE %d simulated households exceed the data's p99 winsorisation cut\n", m.n_winsorised)
    end
    if m.retained_negative > 0
        @printf(out, "  NOTE %d simulated households retain NEGATIVE assets (delta_P should prevent this)\n",
                m.retained_negative)
    end
    println(out, "  LIMITATION: the data is measured at a median child age of ~29, a median 4 years")
    println(out, "  after independence; the model's object is assets at the transfer, child age 18.")

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
    viol = simulation_violations(p)
    v = viol
    @printf(out, "  invalid sim cells     %8d\n", v.total)
    if v.total > 0
        for (k, n) in pairs(v)
            k === :total || n == 0 || @printf(out, "     %-22s %8d\n", k, n)
        end
    end
    # Reported, NOT clamped. sim_a_init is LogNormal(0.296, 1.402) and its upper tail runs
    # past a_max, so clamping the draw would distort the initial wealth distribution to
    # flatter a grid. But it is NOT only the initial draw -- see moment_diagnostics.
    n_sim = d.n_sim
    @printf(out, "  assets above a_max=%.0f  max %.1f  (households: %d ever (%.2f%%), %d at t=1, %d at handoff)\n",
            d.a_grid_max, d.a_max_sim, round(Int, d.a_hh_ever_above*n_sim), 100*d.a_hh_ever_above,
            round(Int, d.a_hh_above_t1*n_sim), round(Int, d.a_hh_above_handoff*n_sim))
    if d.a_hh_ever_above > d.a_hh_above_t1 + 1e-12
        @printf(out, "     %d households CROSS the ceiling during t = 1..%d -- not just the initial draw\n",
                round(Int, (d.a_hh_ever_above - d.a_hh_above_t1)*n_sim), SMM_AGE_HI)
    end
    @printf(out, "  HC in [%.0f, %.0f]      range %.1f - %.1f  (households: %d ever above, %d ever below, %d above at handoff)\n",
            d.hc_grid_min, d.hc_grid_max, d.hc_min_sim, d.hc_max_sim,
            round(Int, d.hc_hh_ever_above*n_sim), round(Int, d.hc_hh_ever_below*n_sim),
            round(Int, d.hc_hh_above_handoff*n_sim))

    # PER-PERIOD, because a pooled share cannot distinguish "two households from the start"
    # from "everyone in the last two periods", and those call for different fixes. Column
    # T+1 is the handoff -- the one that becomes the child's initial state.
    if any(>(0), d.a_over_by_period) || any(>(0), d.hc_over_by_period) ||
       any(>(0), d.hc_under_by_period)
        println(out, "\n  off-grid states by period (n = ", n_sim, " households)")
        @printf(out, "    %-5s %8s %10s %8s %8s %10s\n",
                "t", "a>a_max", "max a", "hc>hi", "hc<lo", "max hc")
        for t in 1:length(d.a_over_by_period)
            lbl = t == length(d.a_over_by_period) ? "T+1" : string(t)
            (d.a_over_by_period[t] == 0 && d.hc_over_by_period[t] == 0 &&
             d.hc_under_by_period[t] == 0 && lbl != "T+1") && continue
            @printf(out, "    %-5s %8d %10.1f %8d %8d %10.1f\n", lbl,
                    d.a_over_by_period[t], d.a_max_by_period[t],
                    d.hc_over_by_period[t], d.hc_under_by_period[t], d.hc_max_by_period[t])
        end
        println(out, "    (rows with no off-grid state are omitted; T+1 is the age-18 handoff)")
    else
        println(out, "  every simulated state is inside both grids")
    end
    return (moments = m, diagnostics = d, params = kw, violations = viol,
            pipeline = r)
end

report_fit(::AbstractVector{Float64}, targets, V_child; kwargs...) = error(
    "report_fit(z, targets, V_child) was removed on 2026-09-10 -- see smm_objective.")
