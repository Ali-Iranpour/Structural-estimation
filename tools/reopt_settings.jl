# =============================================================================
# reopt_settings.jl -- the solver settings and resume metadata tools/reopt.jl ACTUALLY uses
# (2026-09-28: tiktak_fix_plan.md follow-up 5, finding 15 of tiktak_problems.md)
#
# Until 2026-09-28 --ftol-rel and --init-step were parsed and written to results.toml in TikTak mode
# but never passed to the optimizer (the file said ftol_rel = 1e-4 while TikTak ran 1e-3), and every
# results.toml said resume_mode = "restart_from_best" -- the pure-local resume -- even when TikTak
# resumed from its own tiktak_state.toml. Pure functions, so tools/test_reopt_identity.jl tests them.
#
#   pure-local mode (--sobol 0)  --ftol-rel (default 1e-4, the 2026-09-20 pilots), --init-step (default
#                                0 = NLopt's heuristic), --local-alg neldermead|bobyqa
#   TikTak mode (--sobol N > 0)  --ftol-rel -> the restarts' local_tol and --init-step -> local_initial_step,
#                                FORWARDED when given; when not given TikTak's own defaults apply (local_tol
#                                1e-3), unchanged. --local-alg is REFUSED: TikTak's restarts are Nelder-Mead
#                                and its polish BOBYQA. Both step fractions are of the box width.
# =============================================================================

"The pure-local mode's defaults (unchanged since 2026-09-20)."
const REOPT_LOCAL_FTOL_DEFAULT = 1e-4
const REOPT_LOCAL_STEP_DEFAULT = 0.0

"""
    reopt_local_settings(; ftol_rel, init_step, local_alg) -> NamedTuple

The pure-local mode's solver settings from the flags (`nothing` = not given).
"""
function reopt_local_settings(; ftol_rel = nothing, init_step = nothing, local_alg = nothing)
    alg = local_alg === nothing ? :LN_NELDERMEAD :
          lowercase(String(local_alg)) == "neldermead" ? :LN_NELDERMEAD :
          lowercase(String(local_alg)) == "bobyqa" ? :LN_BOBYQA : error("--local-alg must be neldermead or bobyqa")
    ft = ftol_rel === nothing ? REOPT_LOCAL_FTOL_DEFAULT : Float64(ftol_rel)
    st = init_step === nothing ? REOPT_LOCAL_STEP_DEFAULT : Float64(init_step)
    (isfinite(ft) && ft > 0) || error("--ftol-rel = $ft must be finite and > 0")
    (isfinite(st) && 0 <= st <= 1) || error("--init-step = $st must be a fraction of the box width in [0, 1]")
    return (alg = alg, ftol_rel = ft, init_step = st)
end

"""
    reopt_tiktak_kw(; ftol_rel, init_step, local_alg) -> NamedTuple

The keyword arguments reopt.jl forwards to `tiktak` for the solver flags that were GIVEN --
`local_tol` for --ftol-rel, `local_initial_step` for --init-step -- and nothing for a flag that
was not, so TikTak's defaults stay the baseline. --local-alg in TikTak mode is an error.
"""
function reopt_tiktak_kw(; ftol_rel = nothing, init_step = nothing, local_alg = nothing)
    local_alg === nothing || error("--local-alg applies to the pure-local mode (--sobol 0) only: TikTak's restarts are " *
                                   "Nelder-Mead and its polish is BOBYQA. Drop the flag, or run the pure-local mode.")
    kw = (;)
    if ftol_rel !== nothing
        ft = Float64(ftol_rel); (isfinite(ft) && ft > 0) || error("--ftol-rel = $ft must be finite and > 0")
        kw = merge(kw, (local_tol = ft,))
    end
    if init_step !== nothing
        s = Float64(init_step); (isfinite(s) && 0 <= s <= 1) || error("--init-step = $s must be a fraction of the box width in [0, 1]")
        kw = merge(kw, (local_initial_step = s,))
    end
    return kw
end

"""
    effective_settings_lines(cfg) -> Vector{String}

The `[effective_settings]` block of results.toml, read from the configuration TikTak RAN with
(`result.config`), never from the command line.
"""
function effective_settings_lines(cfg)
    l, p = cfg.local_, cfg.polish
    return ["local_alg = \"$(l.alg)\"", "local_ftol_rel = $(l.ftol_rel)", "local_ftol_abs = $(l.ftol_abs)",
            "local_xtol_rel = $(l.xtol_rel)", "local_maxeval = $(l.maxeval)",
            "local_initial_step = $(l.initial_step)   # fraction of the box width; 0 = NLopt's heuristic",
            "local_step_schedule = \"$(l.step_schedule)\"",
            "polish_alg = \"$(p.alg)\"", "polish_ftol_rel = $(p.ftol_rel)", "polish_maxeval = $(p.maxeval)",
            "skip_polish = $(cfg.skip_polish)", "normalize = $(cfg.normalize)"]
end

"""
    reopt_resume_lines(; tiktak, local_resumed, evals_done, local_verification, tk_route, tk_semantics) -> Vector{String}

The resume fields of results.toml, from the route actually taken. Pure-local mode: the
RESTART-from-best resume of checkpoint_best.toml (not an exact optimizer-state resume). TikTak
mode: `tk_route` is "tiktak_state", "pretest_cache" or "none" (fresh), and `tk_semantics` the
module's resume_semantics; no pure-local field is written.
"""
function reopt_resume_lines(; tiktak::Bool, local_resumed::Bool = false, evals_done::Int = 0,
                            local_verification::AbstractString = "not a resume",
                            tk_route::AbstractString = "none", tk_semantics = :fresh)
    if !tiktak
        return ["resumed = $local_resumed", "evals_before_resume = $evals_done",
                "resume_mode = \"" * (local_resumed ? "restart_from_best (Nelder-Mead restarted at the checkpointed incumbent; " *
                                                      "not an exact optimizer-state resume)" : "fresh") * "\"",
                "resume_verification = \"$local_verification\""]
    end
    resumed = tk_route != "none"
    return ["resumed = $resumed",
            "resume_mode = \"" * (tk_route == "tiktak_state" ? "tiktak_state (the TikTak module's checkpoint: seeds, incumbent, " *
                                  "every committed restart; jobs in flight replayed)" :
                                  tk_route == "pretest_cache" ? "pretest_cache (completed pre-testing values reused)" : "fresh") * "\"",
            "resume_semantics = \"$(tk_semantics)\"   # fresh|serial_exact|async_continuation|changed_optimizer|already_complete",
            "resume_verification = \"" * (resumed ? "verified before the warm-up: objective identity field by field " *
                                          "(tools/reopt_identity.jl) and the TikTak preflight (objective, optimizer, state consistency)" :
                                          "not a resume") * "\""]
end
