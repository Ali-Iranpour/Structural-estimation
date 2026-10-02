# =============================================================================
# moments.jl -- SMM on the 67-moment vector of memo 19 (2026-10-01).
#
# WHAT SMM IS DOING HERE, IN ONE PARAGRAPH
# ----------------------------------------
# The model has parameters we cannot observe (how much parents value leisure, how
# productive parental time is in producing child skill). For any guess at those parameters
# we SOLVE the model and SIMULATE a cohort, which gives simulated versions of things we CAN
# observe. Simulated Method of Moments picks the parameters that make the simulated moments
# line up with the PSID/CDS/TAS moments. The weights are diagonal inverse variances.
#
# THE TARGET VECTOR (memo 19 section 3; targets.toml, written by tools/make_smm_targets.py
# from Child_Time_Study/Code/28_smm_moments.do). Order = [moment_cov].names:
#
#   P  (2)   mean_c_p, mean_h_p                 equal-age means over t = 1..17
#   S  (59)  the DFVW skill block (memo 18 section 3), raw Letter-Word through the analytic
#            binomial at the DATA's ages and composition:
#              S1 (15) mean LW by age 3..17        S5 (3)  5-year autocorrelation
#              S3 (4)  SD of LW, pooled bins        S6 (3)  corr(input, LW)
#              S4 (2)  mean 5-year change           S7 (9)  corr(input, 5-year change)
#              S8 (22) mean and SD of each input    S9 (1)  money / pre-tax labour income
#   T  (5)   k0_complete, kpe_bc0_c, kpe_bc1_c, kth_lw17_gap, m_eps
#   W  (1)   kterm_med22                        median retained assets a - tr at the half period
#
# THE TECHNOLOGY these moments identify is memo 18 / Del Boca, Flinn, Verriest & Wiswall (JPE
# 2026): ln k' = ln R_t + sum_j s_jt ln x_jt + s_3t ln k_t, s_jt = exp(a_j0 + a_j1 t), logistic
# TFP, no shock -- see PARENT_DEFAULTS in parent_family.jl.
#
# THE ANALYTIC BINOMIAL (memo 18 section 3.1, decision D4). A simulated child of latent skill
# k has raw score LW ~ Binomial(57, p(k)), p(k) = logistic(L0 + ln k). The model never draws
# LW: every moment uses pi = 57 p(k) and v = 57 p (1 - p), the conditional mean and variance,
# which is the "R draws per observation" estimator with R -> infinity. Test noise enters only
# where a variance does -- SDs, correlations' denominators, the m_eps regression.
#
# THE DATA'S COMPOSITION. A pooled moment in the data mixes ages (and, for pairs, base and end
# ages) in the data's proportions. The model counterpart uses the SAME proportions, read from
# the [composition] tables of the target file. Those tables are NOT YET EXPORTED by Stata
# (decision 2026-10-01: export them, and refuse to run without them); load_targets stops
# with the exact list. docs/SMM_COMPOSITION.md specifies them.
#
# OPEN, FLAGGED FOR REVIEW: kth_lw17_gap and m_eps use the analytic expectation of LW at 17
# rather than a literal binomial draw (decision 2026-10-01, "use analytical but flag this").
#
# PARALLELISM
# -----------
# Worker PROCESSES (Distributed.jl), never threads. NLopt.jl is not thread-safe in this
# project: with `parallel = true` and 8 threads the objective killed the process with exit 0
# and no error. Each worker process has its own NLopt state. See the header of tiktak.jl.
# =============================================================================

using TOML, Printf, Statistics, LinearAlgebra

# -----------------------------------------------------------------------------
# Scale constants -- see the selected run folder's targets.toml for the derivation
# -----------------------------------------------------------------------------
const DOLLARS_PER_MODEL_UNIT = 10_000.0
const HOURS_PER_WEEK         = 112.0
const SMM_AGE_LO, SMM_AGE_HI = 1, 17
const SMM_CHILD_TIME_SPEC = "own_study_fixed_school_v1"

# Letter-Word (WJ-R) has 57 items; DFVW app. C.1.5, footnote 11. L1 = 1 and L0 comes from the
# target file (-4.5951 = logit(0.01)), the same at every age (DFVW, Agostinelli-Wiswall).
const NQ_LW = 57

# The terminal skill is measured at AGE 17 (memo 19 decision 3). Column t of sim_hc IS child
# age t, so this is the last family-stage column, one before the age-18 handoff the college
# decision is taken on -- the assessment precedes the decision, as in the data.
const SMM_LW17_AGE = 17

# The moments actually targeted, in [moment_cov].names order. load_targets refuses a file
# whose order differs: these index the covariance matrix.
const SMM_P_MOMENTS = ("mean_c_p", "mean_h_p")
const SMM_S1_MOMENTS = Tuple("S1_mean_LW_age$a" for a in 3:17)

# Every pooled S moment: (kind, composition frame, input, lo, hi). `lo:hi` is the age bin --
# completed age at assessment for level sets, BASE age for pair sets. Kinds follow memo 18
# section 3: :sd_lw (S3), :mean_dlw (S4), :corr_lw (S5), :corr_x_lw (S6), :corr_x_dlw (S7),
# :mean_x / :sd_x (S8), :mean_ratio (S9).
const SMM_S_SPEC = (
    S3_sd_LW_3_17              = (:sd_lw,      :O_LW,   :none, 3, 17),
    S3_sd_LW_3_7               = (:sd_lw,      :O_LW,   :none, 3, 7),
    S3_sd_LW_8_11              = (:sd_lw,      :O_LW,   :none, 8, 11),
    S3_sd_LW_12_17             = (:sd_lw,      :O_LW,   :none, 12, 17),
    S4_mean_dLW_base3_7        = (:mean_dlw,   :P_LW,   :none, 3, 7),
    S4_mean_dLW_base8_12       = (:mean_dlw,   :P_LW,   :none, 8, 12),
    S5_corr_LW_LWt5_base3_12   = (:corr_lw,    :P_LW,   :none, 3, 12),
    S5_corr_LW_LWt5_base3_7    = (:corr_lw,    :P_LW,   :none, 3, 7),
    S5_corr_LW_LWt5_base8_12   = (:corr_lw,    :P_LW,   :none, 8, 12),
    S6_corr_taup_LW_3_17       = (:corr_x_lw,  :O_taup, :taup, 3, 17),
    S6_corr_tauc_LW_6_17       = (:corr_x_lw,  :O_tauc, :tauc, 6, 17),
    S6_corr_ep_LW_3_17         = (:corr_x_lw,  :O_ep,   :ep,   3, 17),
    S7_corr_taup_dLW_base3_12  = (:corr_x_dlw, :P_taup, :taup, 3, 12),
    S7_corr_taup_dLW_base3_7   = (:corr_x_dlw, :P_taup, :taup, 3, 7),
    S7_corr_taup_dLW_base8_12  = (:corr_x_dlw, :P_taup, :taup, 8, 12),
    S7_corr_tauc_dLW_base6_12  = (:corr_x_dlw, :P_tauc, :tauc, 6, 12),
    S7_corr_tauc_dLW_base6_7   = (:corr_x_dlw, :P_tauc, :tauc, 6, 7),
    S7_corr_tauc_dLW_base8_12  = (:corr_x_dlw, :P_tauc, :tauc, 8, 12),
    S7_corr_ep_dLW_base3_12    = (:corr_x_dlw, :P_ep,   :ep,   3, 12),
    S7_corr_ep_dLW_base3_7     = (:corr_x_dlw, :P_ep,   :ep,   3, 7),
    S7_corr_ep_dLW_base8_12    = (:corr_x_dlw, :P_ep,   :ep,   8, 12),
    S8_mean_taup_3_5   = (:mean_x, :D_taup, :taup, 3, 5),   S8_sd_taup_3_5   = (:sd_x, :D_taup, :taup, 3, 5),
    S8_mean_taup_6_8   = (:mean_x, :D_taup, :taup, 6, 8),   S8_sd_taup_6_8   = (:sd_x, :D_taup, :taup, 6, 8),
    S8_mean_taup_9_12  = (:mean_x, :D_taup, :taup, 9, 12),  S8_sd_taup_9_12  = (:sd_x, :D_taup, :taup, 9, 12),
    S8_mean_taup_13_17 = (:mean_x, :D_taup, :taup, 13, 17), S8_sd_taup_13_17 = (:sd_x, :D_taup, :taup, 13, 17),
    S8_mean_ep_3_5     = (:mean_x, :E_ep,   :ep,   3, 5),   S8_sd_ep_3_5     = (:sd_x, :E_ep,   :ep,   3, 5),
    S8_mean_ep_6_8     = (:mean_x, :E_ep,   :ep,   6, 8),   S8_sd_ep_6_8     = (:sd_x, :E_ep,   :ep,   6, 8),
    S8_mean_ep_9_12    = (:mean_x, :E_ep,   :ep,   9, 12),  S8_sd_ep_9_12    = (:sd_x, :E_ep,   :ep,   9, 12),
    S8_mean_ep_13_17   = (:mean_x, :E_ep,   :ep,   13, 17), S8_sd_ep_13_17   = (:sd_x, :E_ep,   :ep,   13, 17),
    S8_mean_tauc_6_8   = (:mean_x, :D_tauc, :tauc, 6, 8),   S8_sd_tauc_6_8   = (:sd_x, :D_tauc, :tauc, 6, 8),
    S8_mean_tauc_9_12  = (:mean_x, :D_tauc, :tauc, 9, 12),  S8_sd_tauc_9_12  = (:sd_x, :D_tauc, :tauc, 9, 12),
    S8_mean_tauc_13_17 = (:mean_x, :D_tauc, :tauc, 13, 17), S8_sd_tauc_13_17 = (:sd_x, :D_tauc, :tauc, 13, 17),
    S9_mean_ep_over_Y_3_17     = (:mean_ratio, :E_epY,  :ep,   3, 17),
)
# The order of the 59 S rows in the target file: S1, then the pooled rows in SMM_S_SPEC order.
const SMM_S_MOMENTS = (SMM_S1_MOMENTS..., String.(keys(SMM_S_SPEC))...)
const SMM_T_MOMENTS = ("k0_complete", "kpe_bc0_c", "kpe_bc1_c", "kth_lw17_gap", "m_eps")
const SMM_W_MOMENTS = ("kterm_med22",)
const SMM_MOMENTS = (SMM_P_MOMENTS..., SMM_S_MOMENTS..., SMM_T_MOMENTS..., SMM_W_MOMENTS...)
length(SMM_MOMENTS) == 67 || error("SMM_MOMENTS has $(length(SMM_MOMENTS)) entries; memo 19 targets 67")

# UNTARGETED diagnostics with a model counterpart (report only): the m_eps regression's
# coefficients and fit, the LW-at-17 means by college status, and corr(BothCollege, LW) by
# age bin -- the untargeted check of m_BC (memo 18 section 3.6).
const SMM_BC_LW_BINS = (bc_lw_corr_3_5 = (3, 5), bc_lw_corr_6_8 = (6, 8),
                        bc_lw_corr_9_12 = (9, 12), bc_lw_corr_13_17 = (13, 17))

# THE COMPOSITION FRAMES the pooled S moments are mixed over. `kind`: :level frames are
# counted by age, :pair frames by (base age a, end age a2); `odd` frames also carry n_odd,
# the rows observed at an ODD CDS wave (1997, 2007), where money is the mean of the two
# adjacent even PSID years (memo 18 D10) and the model mirrors that. Specified for the Stata
# export in docs/SMM_COMPOSITION.md.
const SMM_COMPOSITION_FRAMES = (
    O_LW   = (kind = :level, odd = false, uses = "S3; equals the S1 per-age N"),
    O_taup = (kind = :level, odd = false, uses = "S6 taup"),
    O_tauc = (kind = :level, odd = false, uses = "S6 tauc"),
    O_ep   = (kind = :level, odd = true,  uses = "S6 ep"),
    P_LW   = (kind = :pair,  odd = false, uses = "S4, S5"),
    P_taup = (kind = :pair,  odd = false, uses = "S7 taup"),
    P_tauc = (kind = :pair,  odd = false, uses = "S7 tauc"),
    P_ep   = (kind = :pair,  odd = true,  uses = "S7 ep"),
    D_taup = (kind = :level, odd = false, uses = "S8 taup"),
    D_tauc = (kind = :level, odd = false, uses = "S8 tauc"),
    E_ep   = (kind = :level, odd = false, uses = "S8 money"),
    E_epY  = (kind = :level, odd = false, uses = "S9"),
)

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

Every elasticity s_jt = exp(a_j0 + a_j1 t) must stay below one at every age it is used
(t = 1..17; own study from T_CHILD_VOICE). For persistence this is memo 18's restriction
(a_30 + a_31 t < 0): s_3 >= 1 makes ln k explosive and the parent solve diverges rather than
failing cleanly. For the inputs an elasticity of one or more is an explosive Cobb-Douglas
in that input. The exponent is linear in t, so the maximum is at an end and checking both
ends is exact, not a sample.
"""
function smm_feasible(kw)
    get_(n) = hasproperty(kw, n) ? getproperty(kw, n) : getfield(PARENT_DEFAULTS, n)
    for (n0, n1, lo) in ((:a_1_0, :a_1_1, SMM_AGE_LO), (:a_2_0, :a_2_1, SMM_AGE_LO),
                         (:a_3_0, :a_3_1, SMM_AGE_LO), (:a_4_0, :a_4_1, T_CHILD_VOICE))
        a0, a1 = get_(n0), get_(n1)
        max(a0 + a1 * lo, a0 + a1 * SMM_AGE_HI) < 0.0 || return false
    end
    return get_(:d_0) > 0.0 && get_(:d_1) > 0.0
end

# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------
_req(raw, k, path) = haskey(raw, k) ? raw[k] :
    error("target file $path has no `$k`; it predates memo 19. Regenerate it with tools/make_smm_targets.py")

"""
    parse_composition(c, path) -> NamedTuple of frames

Read and validate the [composition] tables (docs/SMM_COMPOSITION.md). Each frame becomes
`(a, a2, n, n_odd)` with `a2 = nothing` for a level frame and `n_odd = nothing` where the
frame has no odd-wave rows.
"""
function parse_composition(c::AbstractDict, path::AbstractString)
    frames = Pair{Symbol,Any}[]
    for (name, spec) in pairs(SMM_COMPOSITION_FRAMES)
        haskey(c, String(name)) || error("[composition.$name] is missing from $path " *
                                         "($(spec.uses)); see docs/SMM_COMPOSITION.md")
        f = c[String(name)]
        a  = Int.(f[spec.kind === :pair ? "a" : "age"])
        a2 = spec.kind === :pair ? Int.(f["a2"]) : nothing
        n  = Int.(f["n"])
        n_odd = spec.odd ? Int.(f["n_odd"]) : nothing
        m = length(n)
        (length(a) == m && (a2 === nothing || length(a2) == m) &&
         (n_odd === nothing || length(n_odd) == m)) ||
            error("[composition.$name] in $path: columns have different lengths")
        all(>=(0), n) && sum(n) > 0 || error("[composition.$name]: counts must be >= 0 and not all zero")
        all(x -> 3 <= x <= SMM_AGE_HI, a) || error("[composition.$name]: ages must lie in 3..$SMM_AGE_HI")
        if a2 !== nothing
            all(a2 .> a) && all(<=(SMM_AGE_HI), a2) ||
                error("[composition.$name]: pair end ages must exceed the base age and be <= $SMM_AGE_HI")
        end
        n_odd === nothing || (all(0 .<= n_odd .<= n) ||
                              error("[composition.$name]: n_odd must lie in 0..n"))
        push!(frames, name => (a = a, a2 = a2, n = n, n_odd = n_odd))
    end
    return (; frames...)
end

# Rows of a frame inside an age bin (base age for pairs).
_bin_count(fr, lo, hi) = sum(fr.n[i] for i in eachindex(fr.n) if lo <= fr.a[i] <= hi; init = 0)

"""
    check_composition(comp, out, path)

The composition must DESCRIBE the moments it weights: the S1 per-age N must equal O_LW age by
age, and every pooled S moment's N must equal its frame's count over the moment's bin. A
mismatch means the export and the moment file come from different samples.
"""
function check_composition(comp, out, path)
    O = comp.O_LW
    for a in 3:17
        na = sum(O.n[i] for i in eachindex(O.n) if O.a[i] == a; init = 0)
        na == out["S1_mean_LW_age$a"].n ||
            error("[composition.O_LW] has $na rows at age $a but S1_mean_LW_age$a has N = " *
                  "$(out["S1_mean_LW_age$a"].n): $path")
    end
    for (k, (kind, frame, _, lo, hi)) in pairs(SMM_S_SPEC)
        nb = _bin_count(getfield(comp, frame), lo, hi)
        nb == out[String(k)].n || error("[composition.$frame] counts $nb rows over $lo..$hi " *
                                        "but $k has N = $(out[String(k)].n): $path")
    end
    return nothing
end

"""
    load_targets(path; require_composition = true) -> Dict{String,NamedTuple}

Read the memo-19 target file. Each moment entry carries its mean, SE, N and description;
`"_spec"` carries the calibrated constants, the joint covariance and the composition.

`require_composition = false` loads a file without [composition] tables and leaves
`_spec.composition === nothing`; any S-block evaluation then refuses. It exists for the
parameter-recovery test, which attaches a synthetic composition of its own.
"""
function load_targets(path::AbstractString; require_composition::Bool = true)
    raw = TOML.parsefile(path)

    get(raw, "child_time_spec", "") == SMM_CHILD_TIME_SPEC || error(
        "Target specification mismatch: regenerate targets for own study plus fixed school time; " * path)
    Int(_req(raw, "n_targeted", path)) == length(SMM_MOMENTS) ||
        error("$path declares n_targeted = $(raw["n_targeted"]); moments.jl targets $(length(SMM_MOMENTS))")
    school = Float64.(get(raw, "school_time", []))
    length(school) == SMM_AGE_HI || error("Missing school_time schedule: " * path)
    all(x -> isfinite(x) && 0 <= x < 1 - 2TIME_FLOOR, school) ||
        error("Invalid school_time schedule: " * path)
    all(iszero, school[1:T_CHILD_VOICE-1]) || error("School must be zero below age 6: " * path)

    # ---- the psychic-cost centring constant, now in DFVW ln k units --------
    # kappa_0 + kappa_theta*(log theta - m_psychic). FROZEN in the target file, not recomputed
    # from the simulation: a centring that moved with the parameter vector would be a new
    # nonlinearity rather than a reparameterisation. Required, not defaulted.
    m_psychic = Float64(_req(raw, "m_psychic", path))
    isfinite(m_psychic) && 3 < m_psychic < 10 ||
        error("m_psychic = $m_psychic is not a plausible latent mean ln k at 17 (DFVW units): " * path)

    # ---- the measurement normalisation and the calibrated starting distribution ----
    L0 = Float64(_req(raw, "L0", path))
    abs(L0 - log(0.01 / 0.99)) < 1e-2 ||
        error("L0 = $L0 is not DFVW's normalisation logit(0.01) = -4.595: " * path)
    sd_lnk17 = Float64(_req(raw, "sd_lnk17", path))
    isfinite(sd_lnk17) && sd_lnk17 > 0 || error("sd_lnk17 must be positive: " * path)
    init = (m0 = Float64(_req(raw, "init_m0", path)), mBC = Float64(_req(raw, "init_mBC", path)),
            s0 = Float64(_req(raw, "init_s0", path)))
    all(isfinite, values(init)) && init.s0 >= 0 || error("invalid init_* constants: " * path)

    # ---- the child's bargaining weight ---------------------------------------
    Int.(_req(raw, "mu_ages", path)) == collect(T_CHILD_VOICE:SMM_AGE_HI) ||
        error("mu_ages must be $(T_CHILD_VOICE):$(SMM_AGE_HI): " * path)
    mu_by_age = Float64.(_req(raw, "mu_by_age", path))
    mu_half = Float64(_req(raw, "mu_half", path))
    all(x -> 0 <= x <= 1, mu_by_age) && 0 <= mu_half <= 1 ||
        error("child weights must lie in [0, 1]: " * path)

    # ---- the moment covariance ----------------------------------------------
    # Its row order MUST be SMM_MOMENTS: a silent permutation would weight each residual by
    # another moment's precision and there would be no symptom except a wrong answer.
    mc = _req(raw, "moment_cov", path)
    cov_names = String.(mc["names"])
    cov_names == collect(SMM_MOMENTS) || error("""
        [moment_cov] in $path is ordered
            $(join(cov_names, ", "))
        but moments.jl expects
            $(join(SMM_MOMENTS, ", "))
        These index the same vector, so they must be identical and in the same order.""")
    se = Float64.(mc["se"])
    length(se) == length(SMM_MOMENTS) || error("[moment_cov].se has the wrong length: " * path)
    all(x -> isfinite(x) && x > 0, se) ||
        error("[moment_cov].se has a non-positive or non-finite entry: " * path)
    Sigma = reduce(vcat, (reshape(Float64.(r), 1, :) for r in mc["cov"]))
    size(Sigma) == (length(SMM_MOMENTS), length(SMM_MOMENTS)) ||
        error("[moment_cov].cov is not $(length(SMM_MOMENTS))x$(length(SMM_MOMENTS)): " * path)

    out = Dict{String,NamedTuple}()
    for (j, k) in enumerate(SMM_MOMENTS)
        haskey(raw, k) || error("target file $path is missing [$k]")
        e = raw[k]
        get(e, "targeted", false) == true || error("[$k] is in SMM_MOMENTS but not flagged targeted in $path")
        isapprox(Float64(e["se"]), se[j]; rtol = 1e-8) ||
            error("[$k].se = $(e["se"]) differs from [moment_cov].se = $(se[j]): $path")
        out[k] = (mean = Float64(e["mean"]), se = Float64(e["se"]), n = Int(e["n"]),
                  block = String(e["block"]), source = String(get(e, "measure", "")),
                  units = String(get(e, "units", "")), targeted = true)
    end
    # UNTARGETED rows travel too: the fit report prints them beside their model counterparts.
    for (k, e) in raw
        (e isa Dict && haskey(e, "mean") && !haskey(out, k)) || continue
        get(e, "targeted", false) == false || error("$k is flagged targeted in $path but is not in SMM_MOMENTS")
        out[k] = (mean = Float64(e["mean"]), se = Float64(get(e, "se", NaN)), n = Int(get(e, "n", 0)),
                  block = String(get(e, "block", "")), source = String(get(e, "measure", "")),
                  units = String(get(e, "units", "")), targeted = false)
    end

    # ---- the data's composition (decision 2026-10-01: required) -------------
    comp = nothing
    if haskey(raw, "composition")
        comp = parse_composition(raw["composition"], path)
        check_composition(comp, out, path)
    elseif require_composition
        error("""
            target file $path has no [composition] tables. The pooled S moments are mixtures
            over the DATA's age composition (memo 18 section 3), and the model counterpart needs
            the same proportions. NOT PROVIDED by 28_smm_moments.do yet; the frames needed are
                $(join(keys(SMM_COMPOSITION_FRAMES), ", "))
            specified in docs/SMM_COMPOSITION.md. Refusing rather than approximating
            (decision 2026-10-01).""")
    end

    out["_spec"] = (school_time = school, m_psychic = m_psychic, L0 = L0, sd_lnk17 = sd_lnk17,
                    init = init, mu_by_age = mu_by_age, mu_half = mu_half,
                    se = se, Sigma = Sigma, cov_names = cov_names,
                    n_clusters = Int.(get(mc, "n_clusters_by_moment", zeros(Int, length(se)))),
                    composition = comp)
    return out
end

"""
    with_composition(targets, comp) -> Dict

A copy of `targets` carrying the composition `comp` (a NamedTuple of frames, as
`parse_composition` returns). For the parameter-recovery test, whose synthetic data need a
composition before Stata exports the real one. Checked exactly as a file's would be.
"""
function with_composition(targets, comp)
    out = copy(targets)
    check_composition(comp, out, "<with_composition>")
    out["_spec"] = merge(targets["_spec"], (composition = comp,))
    return out
end

# Metadata travels with the frozen targets to every solve, including diagnostics.
target_school_time(targets) = targets["_spec"].school_time
target_m_psychic(targets)   = targets["_spec"].m_psychic
target_L0(targets)          = targets["_spec"].L0
target_init(targets)        = targets["_spec"].init
target_mu_by_age(targets)   = targets["_spec"].mu_by_age
target_mu_half(targets)     = targets["_spec"].mu_half
target_se(targets)          = targets["_spec"].se
target_Sigma(targets)       = targets["_spec"].Sigma
target_composition(targets) = (c = targets["_spec"].composition;
    c === nothing ? error("no [composition] attached to these targets: the S block cannot be " *
                          "computed. See docs/SMM_COMPOSITION.md") : c)

"""
    moment_weights(targets) -> Vector{Float64}

Diagonal inverse-variance weights, in `SMM_MOMENTS` order:

    Q = sum_j w_j * (m_j - mhat_j)^2 ,      w_j = 1 / se_j^2

so each residual is measured in standard errors of its own moment. The full 67x67 bootstrap
covariance (`target_Sigma`) is used for standard errors and diagnostics, not as the
first-stage weight. READ THE Q SHARES IN THE REPORT: where the model cannot fit a precisely
measured moment, that moment can claim most of Q.
"""
moment_weights(targets) = 1.0 ./ (target_se(targets) .^ 2)

# -----------------------------------------------------------------------------
# Model moments
# -----------------------------------------------------------------------------
"""
    lw_pi_v(p, L0) -> (Pi, V, n_bad)

The analytic binomial building blocks of memo 18 section 3.1, for every simulated child
(rows) and age 1..17 (columns; column t of sim_hc IS age t):

    Pi = 57 p(k),   V = 57 p(k) (1 - p(k)),   p(k) = logistic(L0 + ln k)

the conditional mean and variance of the raw Letter-Word score. A non-positive or
non-finite k is COUNTED in `n_bad` and its cells set to NaN, never floored away.
"""
function lw_pi_v(p::Parent_child_interaction_age_specific_AR1, L0::Float64)
    K = p.sim_hc[:, 1:SMM_AGE_HI]
    n_bad = count(x -> !(isfinite(x) && x > 0), K)
    Pr = map(k -> (isfinite(k) && k > 0) ? 1 / (1 + exp(-(L0 + log(k)))) : NaN, K)
    return NQ_LW .* Pr, NQ_LW .* Pr .* (1 .- Pr), n_bad
end

# Population covariance across simulated children (memo 18 uses 1/N throughout).
function _popcov(y::AbstractVector, z::AbstractVector)
    my, mz = mean(y), mean(z)
    s = 0.0
    @inbounds for i in eachindex(y, z)
        s += (y[i] - my) * (z[i] - mz)
    end
    return s / length(y)
end

# MIXTURE MOMENTS over composition cells c with weights w (memo 18 section 3.1):
#     mu(y)    = sum_c w_c ybar_c
#     Cov(y,z) = sum_c w_c [cov_c(y,z) + (ybar_c - mu(y)) (zbar_c - mu(z))]
# which is the model analogue of pooling the data's observations across ages.
_mmean(w, Y) = sum(w[c] * mean(Y[c]) for c in eachindex(w)) / sum(w)
function _mcov(w, Y, Z)
    my, mz = _mmean(w, Y), _mmean(w, Z)
    return sum(w[c] * (_popcov(Y[c], Z[c]) + (mean(Y[c]) - my) * (mean(Z[c]) - mz))
               for c in eachindex(w)) / sum(w)
end

"""
    _cells(frame, lo, hi) -> Vector{(w, a, a2, odd)}

The composition cells of a frame inside the bin `lo:hi` (base age for pairs). A frame with
odd-wave rows splits each row into its even-wave and odd-wave parts, because the model's
money input differs between them (see `_input`).
"""
function _cells(fr, lo::Int, hi::Int)
    out = Tuple{Float64,Int,Int,Bool}[]
    for i in eachindex(fr.n)
        lo <= fr.a[i] <= hi || continue
        a2 = fr.a2 === nothing ? 0 : fr.a2[i]
        no = fr.n_odd === nothing ? 0 : fr.n_odd[i]
        fr.n[i] - no > 0 && push!(out, (Float64(fr.n[i] - no), fr.a[i], a2, false))
        no > 0 && push!(out, (Float64(no), fr.a[i], a2, true))
    end
    isempty(out) && error("composition bin $lo..$hi has no observations")
    return out
end

"""
    _input(p, x, a, odd) -> Vector

The simulated input `x` at age `a`, in the units the data moments use (time / 112, money in
10k USD/yr). At an ODD CDS wave the data's money is the mean of the two adjacent even PSID
years (memo 18 D10), so the model's is the mean of ages a-1 and a+1. At age 17 the model has
no age-18 money, so age 16 alone stands in -- the data's "single observed year" fallback.
`:bc` is the household's BothCollege type (constant in t), for the untargeted m_BC check.
"""
function _input(p, x::Symbol, a::Int, odd::Bool)
    x === :bc && return p.sim_k[:, 1]
    odd && x !== :ep && error("odd-wave averaging applies to money only, not $x")
    x === :taup && return p.sim_t[:, a]
    x === :tauc && return p.sim_i[:, a]
    if x === :ep
        odd || return p.sim_e[:, a]
        return a + 1 <= SMM_AGE_HI ? 0.5 .* (p.sim_e[:, a-1] .+ p.sim_e[:, a+1]) : p.sim_e[:, a-1]
    end
    error("unknown input $x")
end

# Parents' pre-tax LABOUR income, the denominator of S9 (memo 18 section 3.5): wage times
# hours, before the HSV tax. sim_wage stores wage / WAGE_SCALING_FACTOR, so this is exactly
# the `labor_pre` the budget uses.
_pretax_labour(p, a::Int) = p.sim_wage[:, a] .* WAGE_SCALING_FACTOR .* p.sim_h[:, a]

"""
    s_pooled(kind, cells, x, p, Pi, V) -> Float64

One pooled S moment, by the memo-18 section-3 formula for its kind. `cells` carry the
data's composition; test noise (V) enters only the variances it belongs in.
"""
function s_pooled(kind::Symbol, cells, x::Symbol, p, Pi, V)
    w = [c[1] for c in cells]
    col(M, j) = M[:, j]
    if kind === :sd_lw                                      # S3
        Y = [col(Pi, c[2]) for c in cells]
        return sqrt(_mcov(w, Y, Y) + _mmean(w, [col(V, c[2]) for c in cells]))
    elseif kind === :mean_dlw                               # S4
        return _mmean(w, [col(Pi, c[3]) .- col(Pi, c[2]) for c in cells])
    elseif kind === :corr_lw                                # S5: noise in the denominator only
        Y1 = [col(Pi, c[2]) for c in cells]; Y2 = [col(Pi, c[3]) for c in cells]
        v1 = _mmean(w, [col(V, c[2]) for c in cells]); v2 = _mmean(w, [col(V, c[3]) for c in cells])
        return _mcov(w, Y1, Y2) / sqrt((_mcov(w, Y1, Y1) + v1) * (_mcov(w, Y2, Y2) + v2))
    elseif kind === :corr_x_lw                              # S6 (and the BC check)
        X = [_input(p, x, c[2], c[4]) for c in cells]; Y = [col(Pi, c[2]) for c in cells]
        v = _mmean(w, [col(V, c[2]) for c in cells])
        return _mcov(w, X, Y) / sqrt(_mcov(w, X, X) * (_mcov(w, Y, Y) + v))
    elseif kind === :corr_x_dlw                             # S7: both waves' noise in Var(dLW)
        X = [_input(p, x, c[2], c[4]) for c in cells]
        D = [col(Pi, c[3]) .- col(Pi, c[2]) for c in cells]
        v = _mmean(w, [col(V, c[2]) .+ col(V, c[3]) for c in cells])
        return _mcov(w, X, D) / sqrt(_mcov(w, X, X) * (_mcov(w, D, D) + v))
    elseif kind === :mean_x                                 # S8
        return _mmean(w, [_input(p, x, c[2], c[4]) for c in cells])
    elseif kind === :sd_x
        X = [_input(p, x, c[2], c[4]) for c in cells]
        return sqrt(_mcov(w, X, X))
    elseif kind === :mean_ratio                             # S9
        return _mmean(w, [p.sim_e[:, c[2]] ./ _pretax_labour(p, c[2]) for c in cells])
    end
    error("unknown S-moment kind $kind")
end

"""
    s_block_moments(p, targets) -> (Dict, n_bad, Pi, V)

The 59 S moments (memo 18 section 3) plus the untargeted corr(BothCollege, LW) checks.
S1 is the plain per-age mean of Pi; every pooled row is mixed over the data's composition.
"""
function s_block_moments(p::Parent_child_interaction_age_specific_AR1, targets)
    comp = target_composition(targets)
    Pi, V, n_bad = lw_pi_v(p, target_L0(targets))
    out = Dict{String,Float64}()
    for a in 3:SMM_AGE_HI
        out["S1_mean_LW_age$a"] = mean(view(Pi, :, a))
    end
    for (k, (kind, frame, x, lo, hi)) in pairs(SMM_S_SPEC)
        out[String(k)] = s_pooled(kind, _cells(getfield(comp, frame), lo, hi), x, p, Pi, V)
    end
    for (k, (lo, hi)) in pairs(SMM_BC_LW_BINS)
        out[String(k)] = s_pooled(:corr_x_lw, _cells(comp.O_LW, lo, hi), :bc, p, Pi, V)
    end
    return out, n_bad, Pi, V
end

"""
    t_block_moments(r, Pi, V) -> NamedTuple

The five college moments, from the RESIMULATED child block (memo 19 section 3):

  k0_complete            share choosing college (the model's college path is a completed BA)
  kpe_bc0_c / kpe_bc1_c  the same by the household's BothCollege type
  kth_lw17_gap           mean LW at 17, college minus not
  m_eps                  residual variance of the OLS of college on [1, BothCollege, LW17]

LW AT 17 IS ANALYTIC (decision 2026-10-01, FLAGGED for review): the test draw is independent
of the college decision given k -- college depends on k at 18, assets, BothCollege and the
taste shock, never on the score -- so the gap needs only the means Pi. The regression uses
the population second moments of (college, BC, LW17), with the binomial variance V in
E[LW17^2]; that is the literal OLS with infinitely many draws per child. Its RSS/N is the
population residual variance, which the data's RSS/(N-3) estimates without bias.

An EMPTY group (nobody or everybody in college, or a single BothCollege type) leaves a
moment undefined: NaN, counted in `n_bad`, penalised by the objective -- never a zero that
would reward a draw at which the college margin has collapsed.
"""
function t_block_moments(r, Pi, V)
    p, ch = r.parent, r.child
    col = ch.sim_college
    bc  = p.sim_k[:, 1]
    n_bad = count(!isfinite, col)
    share(mask) = any(mask) ? mean(col[mask]) : (n_bad += 1; NaN)

    k0   = mean(col)
    kpe0 = share(bc .< 0.5)
    kpe1 = share(bc .>= 0.5)

    pi17, v17 = Pi[:, SMM_LW17_AGE], V[:, SMM_LW17_AGE]
    isc = col .>= 0.5
    m_c = any(isc)    ? mean(pi17[isc])    : (n_bad += 1; NaN)
    m_n = any(.!isc)  ? mean(pi17[.!isc])  : (n_bad += 1; NaN)

    Sxx = [1.0         mean(bc)          mean(pi17);
           mean(bc)    mean(bc .^ 2)     mean(bc .* pi17);
           mean(pi17)  mean(bc .* pi17)  mean(pi17 .^ 2 .+ v17)]
    Sxy = [mean(col), mean(bc .* col), mean(pi17 .* col)]
    if all(isfinite, Sxx) && cond(Sxx) < 1e12
        b  = Sxx \ Sxy
        rv = mean(col .^ 2) - dot(b, Sxy)
        r2 = 1 - rv / var(col; corrected = false)
    else
        b, rv, r2 = fill(NaN, 3), NaN, NaN
        n_bad += 1
    end
    return (k0_complete = k0, kpe_bc0_c = kpe0, kpe_bc1_c = kpe1,
            kth_lw17_gap = m_c - m_n, m_eps = rv,
            # diagnostics, not targeted
            kth_lw17_mean_c = m_c, kth_lw17_mean_n = m_n,
            lpm_b_bc = b[2], lpm_b_lw = b[3], lpm_r2 = r2,
            n_college = count(isequal(1.0), col), n_bothcollege = count(>=(0.5), bc),
            t_nonfinite = n_bad)
end

"""
    model_moments(r, targets) -> NamedTuple

All 67 targeted moments, in SMM_MOMENTS order, followed by the diagnostics, from one
pipeline result. `n_nonfinite` counts every cell that could not be used; the objective
refuses a draw with any.

  P  equal-age means of c_p and h_p over t = 1..17 (memo 19: "equal-age mean 1-17"); the
     simulated panel is balanced, so each age carries the same weight, as in the data.
  W  kterm_med22 = MEDIAN retained assets a - tr at the age-18 half period (decision
     2026-10-01: no adjustment for the data's first-child ages 21-22; documented gap).
"""
function model_moments(r::NamedTuple, targets)
    p = r.parent
    n_bad = Ref(0)
    function eqage(M)
        ms = [(v = filter(isfinite, view(M, :, t)); n_bad[] += size(M, 1) - length(v);
               isempty(v) ? NaN : mean(v)) for t in SMM_AGE_LO:SMM_AGE_HI]
        return mean(ms)
    end
    pm = (mean_c_p = eqage(p.sim_c), mean_h_p = eqage(p.sim_h))

    s, s_bad, Pi, V = s_block_moments(p, targets)
    n_bad[] += s_bad + count(x -> !isfinite(x), values(s))
    tm = t_block_moments(r, Pi, V)

    ret = r.retained
    n_bad[] += count(!isfinite, ret)
    fin = filter(isfinite, ret)
    kterm = isempty(fin) ? NaN : median(fin)

    vals = Dict{String,Float64}("mean_c_p" => pm.mean_c_p, "mean_h_p" => pm.mean_h_p)
    merge!(vals, s)
    for k in SMM_T_MOMENTS
        vals[k] = getfield(tm, Symbol(k))
    end
    vals["kterm_med22"] = kterm

    names_ = (Symbol.(SMM_MOMENTS)..., Symbol.(keys(SMM_BC_LW_BINS))...)
    targeted = NamedTuple{names_}(Tuple(vals[String(k)] for k in names_))
    diag = (kth_lw17_mean_c = tm.kth_lw17_mean_c, kth_lw17_mean_n = tm.kth_lw17_mean_n,
            lpm_b_bc = tm.lpm_b_bc, lpm_b_lw = tm.lpm_b_lw, lpm_r2 = tm.lpm_r2,
            n_college = tm.n_college, n_bothcollege = tm.n_bothcollege,
            mean_transfer = (v = filter(isfinite, r.transfers); isempty(v) ? NaN : mean(v)),
            retained_negative = count(x -> isfinite(x) && x < 0, ret),
            mean_lnk_by_age = Tuple(mean(log.(max.(p.sim_hc[:, a], 1e-300))) for a in 1:SMM_AGE_HI),
            sd_lnk_by_age   = Tuple(std(log.(max.(p.sim_hc[:, a], 1e-300))) for a in 1:SMM_AGE_HI),
            n_nonfinite = n_bad[] + tm.t_nonfinite)
    return merge(targeted, diag)
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
# `CHILD_DEFAULTS` is defined in code/src/child_lifecycle.jl since 2026-09-12 (it was here
# from 2026-09-10), beside the constructor whose defaults read from it, so that the
# notebook and run_all.jl build the fitted child without including the SMM. The five
# estimated entries are the exp16b fit; the five fixed ones are unchanged. Nothing about
# how this file uses it changed: `child_config` takes the fixed entries,
# `CHILD_ESTIMATED_DEFAULTS` the estimated ones, and `m_psychic` comes from the TARGET
# FILE, checked against the baseline's centring below.

# The legacy uncentred pair, kept so the centring can be CHECKED rather than trusted.
const LEGACY_KAPPA_0, LEGACY_KAPPA_THETA = 0.2728, -0.0342

"""
    check_psychic_centring(m_psychic)

Verify that the target file's `m_psychic` is the centring `CHILD_DEFAULTS.kappa_0` was
fitted at. The psychic cost is `kappa_0 + kappa_theta*(log theta - m_psychic)`, so
`kappa_0` is the cost AT `m_psychic`; a target file built on a different achievement age
or frame would carry a different `m_psychic`, and the same `kappa_0` would then be a
different model. This turns that into an error at load time.

Until 2026-09-12 the check derived `kappa_0` from the legacy uncentred pair instead
(`LEGACY_KAPPA_0 + LEGACY_KAPPA_THETA*m_psychic`); that pair is kept for the centring
regression test (tools/test_smm_tas.jl group 8) and no longer describes the baseline.
"""
function check_psychic_centring(m_psychic::Float64)
    isapprox(m_psychic, CHILD_DEFAULTS.m_psychic; atol = 1e-9) || error("""
        the target file's m_psychic = $m_psychic is not the centring the baseline
        kappa_0 was fitted at, CHILD_DEFAULTS.m_psychic = $(CHILD_DEFAULTS.m_psychic).

        Either the target file was generated on a different achievement frame, or
        CHILD_DEFAULTS was re-fitted without recording its m_psychic. Re-derive one or the
        other; do not run with a kappa_0 that belongs to another centring.""")
    return nothing
end

"""
    param_default(name) -> Float64

Starting value for an estimated parameter, from whichever block owns it. Both blocks are
searched and an ambiguous name is an error rather than a silent precedence rule -- a
parameter that existed in both would otherwise be routed by declaration order.
"""
# SEARCH STARTS THAT DIFFER FROM THE BLOCK DEFAULT. Empty since 2026-09-12: the block
# defaults ARE the fitted exp16b vector, including sigma_eta = 0.0315 (until then
# PARENT_DEFAULTS.sigma_eta was 0 and the search started the shock at 0.03 from here).
# The mechanism stays so a future parameter can be started away from its baseline value
# without editing the block that run_all.jl and the notebook read.
const SMM_START = (;)

function param_default(name::Symbol)
    inp = hasproperty(PARENT_DEFAULTS, name)
    inc = hasproperty(CHILD_DEFAULTS, name)
    inp && inc && error("`$name` is defined in BOTH PARENT_DEFAULTS and CHILD_DEFAULTS; " *
                        "routing it would be arbitrary. Rename one.")
    inp && return getfield(PARENT_DEFAULTS, name)
    inc && return getfield(CHILD_DEFAULTS, name)
    error("`$name` is in neither PARENT_DEFAULTS nor CHILD_DEFAULTS")
end

"""
    smm_start(name) -> Float64

The SEARCH starting value: `SMM_START` where it lists the parameter, the block default
otherwise. This is what the incumbent and the "(was ...)" column report; `param_default`
remains the value a partial point is completed with.
"""
smm_start(name::Symbol) = hasproperty(SMM_START, name) ? getfield(SMM_START, name) : param_default(name)

# THE ESTIMATED SET (2026-10-02): the memo-18 technology (12), three preference weights and
# the five child parameters -- 15 parent + 5 child = 20, against 67 moments.
#
# Dropped from the exp16b set: R_0 (TFP is now the logistic d_0..d_3), sigma_*_0/1 (now
# a_*_0/1 on AGE, not t-1), and sigma_eta (zero by memo 18). The memo-18 starts are DFVW Table
# 7 (see PARENT_DEFAULTS); the boxes below are SEARCH REGIONS around them, not confidence
# intervals. memo 18 fixes only the TFP box -- (d_0, d_1) in (0, 10), d_2 in (-4, 4), d_3 in
# (-20, 20). The a_j boxes are NOT PROVIDED by memo 18 and are set here: each contains the
# DFVW value with room on both sides, and smm_feasible keeps every elasticity below one.
const SMM_PARAMS = [
    SMMParam(:phi_2,     0.01, 20.0, :log),
    SMMParam(:phi_3,     0.05, 20.0, :log),
    SMMParam(:lambda_2,  0.05, 100.0, :log),
    # parental time: DFVW mother + father, exp(-0.631 - 0.115 t) at the start
    SMMParam(:a_1_0, -4.0,  1.0,  :level),
    SMMParam(:a_1_1, -0.40, 0.10, :level),
    # money: DFVW d4 = exp(-7.154 + 0.072 t)
    SMMParam(:a_2_0, -12.0, -1.0, :level),
    SMMParam(:a_2_1, -0.30,  0.30, :level),
    # persistence: DFVW d5 = exp(-0.254 + 0.005 t), 0.79 -> 0.84; must stay below one
    SMMParam(:a_3_0, -3.0,  0.0,  :level),
    SMMParam(:a_3_1, -0.10, 0.10, :level),
    # own study, from age 6: DFVW d3 = exp(-6.598 + 0.271 t)
    SMMParam(:a_4_0, -12.0, -1.0, :level),
    SMMParam(:a_4_1, -0.20,  0.60, :level),
    # TFP, memo 18's box. d_0, d_1 strictly positive so R_t is a convex combination of two
    # positive levels.
    SMMParam(:d_0,  0.01, 10.0, :level),
    SMMParam(:d_1,  0.01, 10.0, :level),
    SMMParam(:d_2, -4.0,   4.0, :level),
    SMMParam(:d_3, -20.0, 20.0, :level),

    # ---- the five child parameters ----
    # kappa_0 -- the level of the psychic cost at mean ability (ln k = m_psychic). The college
    # margin is close to a STEP in kappa_0 and everything above ~0 was a flat zero under
    # exp16b, so the box covers the transition with margin: [-3, 1] (unchanged).
    SMMParam(:kappa_0, -3.0, 1.0, :level, :child),
    # kappa_theta -- the ability gradient, per unit of ln k. NEW UNITS: ln k has SD 0.65 at 17
    # against 0.033 for the old log g_ACH, so the old box [-10, 0] is a 20x wider economic
    # range. [-3, 0] admits up to -2 per SD of skill; the start (-0.18) is the exp16b value
    # converted per SD.
    SMMParam(:kappa_theta, -3.0, 0.0, :level, :child),
    # kappa_ParEd -- the BothCollege shift. The data split is now BothCollege itself (memo 19
    # decision 2), so the either-parent mismatch it used to absorb is gone. Box unchanged.
    SMMParam(:kappa_ParEd, -1.0, 0.5, :level, :child),
    # kappa_terminal -- the parent's taste for retained assets; strictly positive weight.
    SMMParam(:kappa_terminal, 0.5, 40.0, :log, :child),
    # sigma_eps -- SD of the college taste shock, identified by m_eps; strictly positive.
    SMMParam(:sigma_eps, 0.1, 2.0, :log, :child),
]

# FIFTEEN parent + FIVE child. Asserted, because a stray entry in either default set would be
# routed silently.
let np = count(q -> q.owner === :parent, SMM_PARAMS), nc = count(q -> q.owner === :child, SMM_PARAMS)
    (np, nc) == (15, 5) || error("SMM_PARAMS has $np parent + $nc child parameters; the " *
                                 "memo-18 set is 15 + 5 = 20")
end
# The HC shock is zero by memo 18 and must not be estimated.
any(q -> q.name === :sigma_eta, SMM_PARAMS) && error("sigma_eta must not be estimated; memo 18 sets it to 0")
PARENT_DEFAULTS.sigma_eta == 0.0 || error("PARENT_DEFAULTS.sigma_eta = $(PARENT_DEFAULTS.sigma_eta); memo 18 sets it to 0")

# A moment-count check is necessary but does not establish local identification.
length(SMM_MOMENTS) >= length(SMM_PARAMS) || error("""
    SMM is UNDER-identified: $(length(SMM_PARAMS)) parameters against \
    $(length(SMM_MOMENTS)) moments.""")

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
                     demo_sim::Bool = false,
                     child_wage::NamedTuple = child_wage_config())
    r = run_pipeline(named_point(vals), targets; Na = Na, Nk = Nk, Nhc = Nhc,
                     simN = simN, seed = seed, child_grid = child_grid,
                     parent_extra = parent_extra, demo_sim = demo_sim,
                     child_wage = child_wage)
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

incumbent() = [to_search(smm_start(q.name), q) for q in SMM_PARAMS]

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
    child_wage_config(; sd_log_afqt = CHILD_DEFAULTS.sd_log_afqt) -> NamedTuple

The child's wage loading on skill, anchored per SD to Daruich & Fernandez Table B4
(`anchored_alpha`, decision 2026-10-01), with m_theta = m_psychic (both the latent mean ln k
at 17). REFUSES while sd(log AFQT) is NaN -- it is NOT PROVIDED yet -- so no estimation can
run on a wage loading nobody chose. Tests that need a running model pass an explicit
`child_wage` NamedTuple instead, labelled as a placeholder.

lnw0 is the constructor's normalisation for now. Once alpha_theta is fixed it is re-set to
keep the mean child wage at BASELINE_MEAN_CHILD_WAGE (`lnw0_for_mean_wage`), and the value
belongs here.
"""
function child_wage_config(; sd_log_afqt::Real = CHILD_DEFAULTS.sd_log_afqt)
    isfinite(sd_log_afqt) && sd_log_afqt > 0 || error("""
        child_wage_config: sd(log AFQT raw score) is NOT PROVIDED (CHILD_DEFAULTS.sd_log_afqt =
        $(sd_log_afqt)). alpha_theta = 0.654 * sd(log AFQT) / $(SD_LNK17) cannot be set without it.
        Supply it in CHILD_DEFAULTS, or pass `child_wage = (alpha_theta = ..., alpha_thetaE = ...,
        m_theta = ..., lnw0 = ...)` explicitly for a test.""")
    a, aE = anchored_alpha(sd_log_afqt)
    return (alpha_theta = a, alpha_thetaE = aE, m_theta = CHILD_DEFAULTS.m_psychic,
            lnw0 = log(CHILD_DEFAULTS.w) - 0.4144)
end

"""
    check_half_period_weight(mu_half)

The target file's mu_half (the CHILD's weight at 18) must be the one CHILD_DEFAULTS was set
from: the child module's `mu` is the PARENT's weight, 1 - mu_half. Same role as
check_psychic_centring -- a different file would silently make a different model.
"""
function check_half_period_weight(mu_half::Float64)
    isapprox(1 - mu_half, CHILD_DEFAULTS.mu; atol = 1e-9) || error("""
        the target file's mu_half = $mu_half implies a parent weight of $(1 - mu_half) at the
        half period, but CHILD_DEFAULTS.mu = $(CHILD_DEFAULTS.mu). Update one or the other.""")
    return nothing
end

"""
    child_config(targets; Na, Nk, Nt, simN, seed, child_wage) -> NamedTuple

The child's complete NON-ESTIMATED configuration, and therefore the cache key.

Everything the reusable stages depend on is in here and every estimated child parameter is
out of it. If a value belongs in the key and is missing, two different models share a cache
entry and the second silently inherits the first's solution; if an estimated parameter
leaks IN, the cache never hits and the run is merely slow. The first failure is silent, so
the key is written out in full rather than derived. The wage loading (`child_wage`) and the
half-period weight `mu` are in it since 2026-10-02: both change the solved child.
"""
child_config(targets; Na::Int, Nk::Int, Nt::Int, simN::Int, seed::Int,
             child_wage::NamedTuple) =
    (Na = Na, Nk = Nk, Nt = Nt, simN = simN, seed = seed,
     rho          = CHILD_DEFAULTS.rho,
     psi_terminal = CHILD_DEFAULTS.psi_terminal,
     omega        = CHILD_DEFAULTS.omega,
     a_max        = CHILD_DEFAULTS.a_max,
     w            = CHILD_DEFAULTS.w,
     m_psychic    = target_m_psychic(targets),
     mu           = 1 - target_mu_half(targets),
     alpha_theta  = Float64(child_wage.alpha_theta),
     alpha_thetaE = Float64(child_wage.alpha_thetaE),
     m_theta      = Float64(child_wage.m_theta),
     lnw0         = Float64(child_wage.lnw0))

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
# The five estimated child parameters, and their SMM starting values. `CHILD_DEFAULTS`
# also holds the fixed settings, so this is the subset a partial parameter point must be
# completed with. `sigma_eps` is here and NOT in `child_config`: it rebuilds `t_grid`,
# which only the study years (stage 2) and the transfer stage read -- the cached work and
# graduate blocks are eps-free (their arrays carry no Nt dimension), verified by
# tools/test_smm_tas.jl "cache parity" at a non-default sigma_eps.
const CHILD_ESTIMATED = (:kappa_0, :kappa_theta, :kappa_ParEd, :kappa_terminal, :sigma_eps)
const CHILD_ESTIMATED_DEFAULTS =
    NamedTuple{CHILD_ESTIMATED}(map(n -> getfield(CHILD_DEFAULTS, n), CHILD_ESTIMATED))

function build_child_solution(ckw::NamedTuple, targets;
                              Na::Int, Nk::Int, Nt::Int, simN::Int, seed::Int,
                              child_wage::NamedTuple = child_wage_config())
    cfg = child_config(targets; Na = Na, Nk = Nk, Nt = Nt, simN = simN, seed = seed,
                       child_wage = child_wage)

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
                      demo_sim::Bool = true,
                      child_wage::NamedTuple = child_wage_config())
    pk, ck = split_params(kw)
    check_half_period_weight(target_mu_half(targets))

    # ---- 1. child lifecycle and transfer problems ---------------------------
    child, V_child = build_child_solution(ck, targets;
                                          Na = child_grid.Na, Nk = child_grid.Nk,
                                          Nt = child_grid.Nt, simN = simN, seed = seed,
                                          child_wage = child_wage)

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
    # The calibrated constants come from the TARGET FILE, not from the compiled defaults:
    # the age-1 skill draw (memo 18 section 7) and the child's weight at ages 6..17.
    ini = target_init(targets)
    parent = Parent_child_interaction_age_specific_AR1(; Na = Na, Nk = Nk, Nhc = Nhc,
                                                         simN = simN, seed = seed,
                                                         school_time = target_school_time(targets),
                                                         init_m0 = ini.m0, init_mBC = ini.mBC,
                                                         init_s0 = ini.s0,
                                                         mu_child_by_age = target_mu_by_age(targets),
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
            # Parental assets RETAINED after the transfer, a_term = a - tr at the half
            # period: the model's kappa_terminal object, whose MEDIAN is kterm_med22.
            retained = child.sim_a_init .- child.sim_tr_init,
            parent_kw = pk, child_kw = ck)
end

# -----------------------------------------------------------------------------
# Objective
# -----------------------------------------------------------------------------
"""
    smm_objective(z, targets; grids..., child_wage) -> Float64

The SMM criterion, diagonal inverse-variance weighted (see `moment_weights`):

    Q = sum_j w_j (m_j - mhat_j)^2,     w_j = 1 / se_j^2,   j over the 67 SMM_MOMENTS

A draw outside the admissible region (`smm_feasible`), an invalid simulation, a non-finite
moment or an expected model failure scores SMM_PENALTY; a programming error is re-thrown.

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
                       demo_sim::Bool = true,
                       child_wage::NamedTuple = child_wage_config())
    kw = unpack(z)
    w  = weights === nothing ? moment_weights(targets) : weights

    # ---- reject the infeasible region BEFORE paying for a solve --------------
    if !smm_feasible(kw)
        _penalize!(:infeasible_elasticity)
        return SMM_PENALTY
    end

    try
        r = run_pipeline(kw, targets; Na = Na, Nk = Nk, Nhc = Nhc, simN = simN,
                         seed = seed, child_grid = child_grid, demo_sim = demo_sim,
                         child_wage = child_wage)
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
    report_fit(z, targets; grids..., child_wage, out = stdout)

Re-solve at `z` and print every targeted moment against the data, the Q shares, and the
untargeted checks that decide whether a fit is believable.
"""
function report_fit(z::AbstractVector{Float64}, targets;
                    Na::Int = 30, Nk::Int = 2, Nhc::Int = 30,
                    simN::Int = 2000, seed::Int = 1234,
                    child_grid = (Na = 30, Nk = 30, Nt = 5),
                    weights::Union{Nothing,Vector{Float64}} = nothing,
                    child_wage::NamedTuple = child_wage_config(),
                    out::IO = stdout)
    kw = unpack(z)
    w  = weights === nothing ? moment_weights(targets) : weights
    r  = run_pipeline(kw, targets; Na = Na, Nk = Nk, Nhc = Nhc, simN = simN,
                      seed = seed, child_grid = child_grid, child_wage = child_wage)
    p  = r.parent
    m  = model_moments(r, targets)
    d  = moment_diagnostics(p)
    se = target_se(targets)

    println(out, "\nParameters")
    println(out, "-"^62)
    for q in SMM_PARAMS
        @printf(out, "  %-14s %10.4f   (start %.4f)\n", q.name, getfield(kw, q.name), smm_start(q.name))
    end

    @printf(out, "\nTargeted moments -- %d moments, %d parameters\n", length(SMM_MOMENTS), length(SMM_PARAMS))
    println(out, "-"^100)
    @printf(out, "  %-28s %11s %11s %9s %8s   %s\n", "moment", "model", "data", "t", "Q share", "measure")
    qk = [w[j] * (getfield(m, Symbol(k)) - targets[k].mean)^2 for (j, k) in enumerate(SMM_MOMENTS)]
    q_tot = sum(qk)
    block = ""
    for (j, k) in enumerate(SMM_MOMENTS)
        if targets[k].block != block
            block = targets[k].block
            println(out, "  " * "-"^30 * " block " * block * " " * "-"^30)
        end
        mj, mhat = getfield(m, Symbol(k)), targets[k].mean
        # t is the miss in STANDARD ERRORS of the data moment -- the scale the weighting uses.
        @printf(out, "  %-28s %11.4f %11.4f %9.1f %7.1f%%   %s\n", k, mj, mhat, (mj - mhat) / se[j],
                100 * qk[j] / max(q_tot, eps()), first(targets[k].source, 46))
    end
    @printf(out, "  %-28s %11s %11s %18.4f\n", "Q", "", "", q_tot)
    for b in ("P", "S", "T", "W")
        @printf(out, "  Q share of block %s: %.1f%%\n", b,
                100 * sum(qk[j] for (j, k) in enumerate(SMM_MOMENTS) if targets[k].block == b) / max(q_tot, eps()))
    end
    # WEIGHT CONCENTRATION: where the model cannot fit a precisely measured moment, that
    # moment can claim most of Q, and a 67-moment estimation quietly becomes a few-moment one.
    ord = sortperm(qk; rev = true)
    top = ord[1:min(5, length(ord))]
    @printf(out, "\n  Q concentration: top 5 moments carry %.1f%% of Q  (%s)\n",
            100 * sum(qk[top]) / max(q_tot, eps()),
            join((@sprintf("%s %.0f%%", SMM_MOMENTS[t], 100 * qk[t] / max(q_tot, eps())) for t in top), ", "))

    println(out, "\nUntargeted -- the college regression, the skill gradient, the handoff")
    println(out, "-"^76)
    _t(k) = haskey(targets, k) ? targets[k].mean : NaN
    @printf(out, "  LW at 17: college %.2f / not %.2f   (data %.2f / %.2f)   [analytic LW17, FLAGGED]\n",
            m.kth_lw17_mean_c, m.kth_lw17_mean_n, _t("kth_lw17_mean_c"), _t("kth_lw17_mean_n"))
    @printf(out, "  m_eps regression: b_BC %.3f  b_LW %.4f  R2 %.3f   (data %.3f / %.4f / %.3f)\n",
            m.lpm_b_bc, m.lpm_b_lw, m.lpm_r2, _t("lpm_b_bc"), _t("lpm_b_lw"), _t("lpm_r2"))
    for k in keys(SMM_BC_LW_BINS)
        @printf(out, "  %-18s model %.3f  data %.3f\n", k, getfield(m, k), _t(String(k)))
    end
    @printf(out, "  college %d of %d simulated children; %d BothCollege\n",
            m.n_college, size(r.child.sim_college, 1), m.n_bothcollege)
    @printf(out, "  mean transfer %.4f (%.0f USD); median retained %.4f (%.0f USD) vs data %.0f USD\n",
            m.mean_transfer, m.mean_transfer * DOLLARS_PER_MODEL_UNIT,
            m.kterm_med22, m.kterm_med22 * DOLLARS_PER_MODEL_UNIT,
            targets["kterm_med22"].mean * DOLLARS_PER_MODEL_UNIT)
    println(out, "  LIMITATION: the data are at first-child ages 21-22; the model's a_term is at 18.")
    m.retained_negative > 0 && @printf(out, "  NOTE %d households retain NEGATIVE assets\n", m.retained_negative)
    print(out, "  mean ln k by age   ")
    for a in 1:SMM_AGE_HI; @printf(out, "%d:%.2f ", a, m.mean_lnk_by_age[a]); end
    print(out, "\n  sd ln k by age     ")
    for a in 1:SMM_AGE_HI; @printf(out, "%d:%.2f ", a, m.sd_lnk_by_age[a]); end
    @printf(out, "\n  (data: latent mean ln k at 17 = %.3f, SD %.3f)\n", target_m_psychic(targets),
            targets["_spec"].sd_lnk17)

    @printf(out, "\n  c_p  %8.0f USD/yr  vs data %.0f\n", m.mean_c_p * DOLLARS_PER_MODEL_UNIT,
            targets["mean_c_p"].mean * DOLLARS_PER_MODEL_UNIT)
    @printf(out, "  h_p  %8.1f hrs/wk  vs data %.1f\n", m.mean_h_p * HOURS_PER_WEEK,
            targets["mean_h_p"].mean * HOURS_PER_WEEK)
    @printf(out, "  after-tax income %.4f (%.0f USD/yr); implied saving rate %.1f%%\n",
            d.income, d.income * DOLLARS_PER_MODEL_UNIT, 100 * d.saving_rate)
    viol = simulation_violations(p)
    @printf(out, "  invalid sim cells %d\n", viol.total)
    if viol.total > 0
        for (k, n) in pairs(viol)
            k === :total || n == 0 || @printf(out, "     %-22s %8d\n", k, n)
        end
    end
    n_sim = d.n_sim
    @printf(out, "  assets above a_max=%.0f: %d households ever; skill outside [%.3g, %.3g]: %d ever above, %d ever below\n",
            d.a_grid_max, round(Int, d.a_hh_ever_above * n_sim), d.hc_grid_min, d.hc_grid_max,
            round(Int, d.hc_hh_ever_above * n_sim), round(Int, d.hc_hh_ever_below * n_sim))
    return (moments = m, diagnostics = d, params = kw, violations = viol, pipeline = r)
end

report_fit(::AbstractVector{Float64}, targets, V_child; kwargs...) = error(
    "report_fit(z, targets, V_child) was removed on 2026-09-10 -- see smm_objective.")
