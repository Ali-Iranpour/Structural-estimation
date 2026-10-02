# =============================================================================
# config.jl -- the optimizer configuration and its validation.
#
# Everything that decides the SEARCH PATH lives in `TikTakConfig` (the optimizer identity
# a checkpoint records); how the work is executed -- workers, pause limits, paths,
# callbacks -- is not part of it. Validation runs before the first evaluation, so an
# invalid configuration never costs a pre-testing stage (plan step 2.3).
# =============================================================================

"""
    SolverSettings

One NLopt local solve: algorithm, stopping rules and evaluation budget. `maxeval` must be
positive: NLopt reads 0 as NO LIMIT.

The INITIAL STEP (plan step 9; the baseline leaves it to NLopt): `initial_step = 0` keeps
NLopt's own heuristic -- a quarter of the box width per coordinate, less near a bound -- which
is what every run before 2026-09-27 used. A positive value is a fraction of each coordinate's
box width (Nelder-Mead's simplex size, BOBYQA's rhobeg). `step_schedule = :theta_shrink`
multiplies it for restart j by max(step_min, 1 - theta_j): early, exploratory restarts take
wide steps, late ones -- started close to the incumbent -- small ones. Settings, not tuning:
tools/bench_tiktak.jl measures them; the defaults stay the baseline until evidence says otherwise.
"""
struct SolverSettings
    alg::Symbol
    ftol_rel::Float64
    ftol_abs::Float64
    xtol_rel::Float64
    maxeval::Int
    initial_step::Float64
    step_schedule::Symbol
    step_min::Float64
end
SolverSettings(alg, ftol_rel, ftol_abs, xtol_rel, maxeval) =
    SolverSettings(alg, ftol_rel, ftol_abs, xtol_rel, maxeval, 0.0, :fixed, 0.0)

"""
    TikTakConfig

The settings that decide the search path. `nstar` is the REQUESTED number of local restarts.
Whether a smaller valid pool may reduce the restarts is `allow_fewer_restarts`; the effective
count is a property of the run (`TikTakResult.nstar_effective`), not of the configuration.

The pre-testing design (plan step 5) is one of two:
  fixed-attempt  `n_valid_target = 0`: the first `n_sobol` Sobol' draws, valid or not
  valid-target   `n_valid_target > 0`: the Sobol' sequence is continued until its shortest
                 prefix holding `n_valid_target` valid draws is complete, never beyond
                 `n_sobol` draws (the hard attempt cap). `require_valid_target` makes an
                 exhausted cap a failed pre-testing gate rather than a reported shortfall.
"""
struct TikTakConfig
    n_sobol::Int
    n_valid_target::Int
    require_valid_target::Bool
    skip_first::Bool
    invalid_value::Float64
    nstar::Int
    allow_fewer_restarts::Bool
    theta_p::Float64
    theta_lo::Float64
    theta_hi::Float64
    local_::SolverSettings
    polish::SolverSettings
    skip_polish::Bool
    stop_tol::Float64
    on_error::Symbol
    # VERIFICATION OF AN UNCHANGED POINT (plan step 3). A converged solve verifies the retained
    # point when it returns it within `verify_xtol` in box-normalized coordinates (max over
    # coordinates of |dx_i| / (hi_i - lo_i)) with a value within max(verify_ftol_abs,
    # verify_ftol_rel * |f|) of the retained one. Deliberately tight: the same point on the
    # same deterministic objective gives the same value, so a looser match would let a
    # neighbouring point -- or equal Q somewhere else -- pass as verification.
    verify_xtol::Float64
    verify_ftol_rel::Float64
    verify_ftol_abs::Float64
    # THE ASYNCHRONOUS START-UP (plan step 6; second option 2026-09-29, fix plan R2). Part of the
    # SEARCH PLAN in the optimizer identity: never changed on a resume. Only the asynchronous
    # local stage reads it -- serial and one-worker runs are sequential under either value.
    #   :first_alone     restart 1 runs by itself and is committed before any other restart
    #                    starts, so every later start mixes with a LOCAL MINIMUM. The cost: the
    #                    other workers wait for one whole restart (E1-P: ~11 h, 19 idle workers).
    #   :immediate_mixed the pool fills at once. Restart 1 is still the best seed, unmixed;
    #                    restart j >= 2 mixes seed j with the CURRENT incumbent under the same
    #                    theta schedule -- at first the best PRE-TESTED point (for a warm start,
    #                    the supplied point), later each newer committed result. Our variant, not
    #                    the reference's: its processes start from their own seeds UNMIXED until
    #                    a local result exists, which in E1-P would have meant 19 searches from
    #                    Q 10,090-74,684 while the incumbent was at 916 (tiktak_problems.md 20).
    # Wall time: with equal restarts of length T on P workers, first_alone takes
    # T + ceil((K-1)/P) T and immediate_mixed ceil(K/P) T -- one round less only when K fills
    # the rounds (K = 20 or 40 on 20 workers; K = 21 costs 2T either way).
    bootstrap::Symbol
    # SEARCH GEOMETRY (plan step 9): true runs every local solve in box-normalized coordinates
    # u = (x - lo) / (hi - lo) in [0, 1]^n. The mixing, the starts, the endpoints and every
    # checkpoint stay in search coordinates x; only the solver sees u (so xtol_rel is relative
    # to the box, not to |x|). Off by default: the baseline solves in x.
    normalize::Bool
end

"The asynchronous start-up policies (`TikTakConfig.bootstrap`); the first is the default."
const BOOTSTRAP_POLICIES = (:first_alone, :immediate_mixed)

"Raised when the pre-testing stage cannot supply the restarts the configuration requires."
struct PretestGateError <: Exception
    msg::String
end
Base.showerror(io::IO, e::PretestGateError) = print(io, "TikTak pre-testing gate FAILED: ", e.msg)

_check(ok::Bool, msg) = ok || throw(ArgumentError("TikTak configuration: " * msg))

function _check_solver(s::SolverSettings, what::String, n::Int)
    _check(s.maxeval >= 1, "$what maxeval = $(s.maxeval); NLopt reads 0 as NO LIMIT, so the budget must be positive" *
                           (what == "polish" ? " (use skip_polish = true to bypass the polish)" : ""))
    for (nm, v) in (("ftol_rel", s.ftol_rel), ("ftol_abs", s.ftol_abs), ("xtol_rel", s.xtol_rel))
        _check(isfinite(v) && v >= 0, "$what $nm = $v must be finite and >= 0")
    end
    _check(isfinite(s.initial_step) && 0 <= s.initial_step <= 1,
           "$what initial_step = $(s.initial_step) must be a fraction of the box width in [0, 1] (0 = NLopt's default)")
    _check(s.step_schedule in (:fixed, :theta_shrink), "$what step_schedule must be :fixed or :theta_shrink")
    _check(0 <= s.step_min <= 1, "$what step_min = $(s.step_min) must be in [0, 1]")
    (s.step_schedule === :theta_shrink && s.initial_step == 0) &&
        _check(false, "$what step_schedule = :theta_shrink needs an explicit initial_step > 0")
    ok = try
        NLopt.Opt(s.alg, n); true
    catch
        false
    end
    _check(ok, "$what algorithm :$(s.alg) is not an NLopt algorithm")
    return nothing
end

"""
    validate(cfg, lo, hi; resuming = false)

Refuse an invalid problem before anything is evaluated: bounds that are not finite, of
unequal length or not strictly ordered; non-positive budgets (a zero evaluation cap would
be NLopt's "no limit"); a mixing schedule outside [0, 1].
"""
function validate(cfg::TikTakConfig, lo::Vector{Float64}, hi::Vector{Float64}; n_supplied::Int = 0,
                  resuming::Bool = false)
    n = length(lo)
    _check(n >= 1, "the box has no coordinates")
    _check(length(hi) == n, "lo and hi must have the same length ($(n) vs $(length(hi)))")
    _check(all(isfinite, lo) && all(isfinite, hi), "every bound must be finite (Sobol' needs a finite box)")
    _check(all(lo .< hi), "every lo must be strictly below its hi")
    _check(cfg.n_sobol >= 0, "N = $(cfg.n_sobol) Sobol' draws must be >= 0")
    _check(cfg.nstar >= 1, "Nstar = $(cfg.nstar) restarts must be >= 1")
    _check(cfg.n_valid_target >= 0, "n_valid_target = $(cfg.n_valid_target) must be >= 0")
    cfg.n_valid_target > 0 && _check(cfg.n_sobol >= cfg.n_valid_target,
        "the attempt cap N = $(cfg.n_sobol) is below the valid-draw target $(cfg.n_valid_target)")
    pool = cfg.n_valid_target > 0 ? cfg.n_valid_target : cfg.n_sobol
    resuming || _check(cfg.nstar <= pool + n_supplied,
                       "need 1 <= Nstar <= (N, or n_valid_target) + #extra_seeds (Nstar = $(cfg.nstar), " *
                       "pool = $pool, #extra_seeds = $n_supplied)")
    _check(isfinite(cfg.theta_p) && cfg.theta_p > 0, "theta_p = $(cfg.theta_p) must be finite and > 0")
    _check(0.0 <= cfg.theta_lo <= cfg.theta_hi <= 1.0,
           "need 0 <= theta_lo <= theta_hi <= 1 (got $(cfg.theta_lo), $(cfg.theta_hi))")
    _check(!isnan(cfg.invalid_value), "invalid_value is NaN")
    _check(isfinite(cfg.stop_tol) && cfg.stop_tol >= 0, "stop_tol = $(cfg.stop_tol) must be finite and >= 0")
    _check(cfg.on_error in (:rethrow, :discard), "on_error must be :rethrow or :discard, got :$(cfg.on_error)")
    _check(cfg.bootstrap in BOOTSTRAP_POLICIES,
           "bootstrap must be one of $(join(repr.(BOOTSTRAP_POLICIES), ", ")), got :$(cfg.bootstrap)")
    for (nm, v) in (("verify_xtol", cfg.verify_xtol), ("verify_ftol_rel", cfg.verify_ftol_rel),
                    ("verify_ftol_abs", cfg.verify_ftol_abs))
        _check(isfinite(v) && v >= 0, "$nm = $v must be finite and >= 0")
    end
    _check_solver(cfg.local_, "local", n)
    cfg.skip_polish || _check_solver(cfg.polish, "polish", n)
    return nothing
end

"""
    checked_point(x, lo, hi; what, tol = 1e-9) -> Vector{Float64}

A caller-supplied point (an extra seed, a resumed incumbent) as a fresh vector: it must
have the box's dimension and finite coordinates, and lie inside the box. A coordinate
outside by at most `tol` of its box width is snapped onto the bound (TOML round trips are
not bit-exact); anything further is refused rather than silently clamped.
"""
function checked_point(x::AbstractVector{<:Real}, lo::Vector{Float64}, hi::Vector{Float64};
                       what::AbstractString = "point", tol::Float64 = 1e-9)
    length(x) == length(lo) || throw(ArgumentError(
        "TikTak: $what has dimension $(length(x)); the box has $(length(lo))"))
    y = Vector{Float64}(undef, length(lo))
    for i in eachindex(lo)
        v = Float64(x[i])
        isfinite(v) || throw(ArgumentError("TikTak: $what coordinate $i is not finite ($v)"))
        w = hi[i] - lo[i]
        (lo[i] - tol * w <= v <= hi[i] + tol * w) || throw(ArgumentError(
            "TikTak: $what coordinate $i = $v is outside its box [$(lo[i]), $(hi[i])]"))
        y[i] = clamp(v, lo[i], hi[i])
    end
    return y
end
