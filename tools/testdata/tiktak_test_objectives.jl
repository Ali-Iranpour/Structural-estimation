# =============================================================================
# tiktak_test_objectives.jl -- synthetic objectives for tools/test_tiktak.jl, loaded on the
# master AND on every test worker (after code/src/tiktak.jl) and registered under keys.
# Several read TikTak.current_job() to make one chosen restart slow, failing or fatal, so the
# scheduler's behaviour under a known completion order can be tested without relying on
# wall-clock speed alone.
# =============================================================================

tt_rast(x) = 10length(x) + sum(xi^2 - 10cos(2π * xi) for xi in x)
TikTak.register_objective!(:rast, tt_rast)

"Restart `TT_SLOW_J[]` is slow (every evaluation sleeps), so later restarts overtake it."
const TT_SLOW_J = Ref(2)
function tt_slow(x)
    job = TikTak.current_job()
    job !== nothing && job.stage === :local && job.j == TT_SLOW_J[] && sleep(0.02)
    return tt_rast(x)
end
TikTak.register_objective!(:slow, tt_slow)

"Every evaluation of restart `TT_FAIL_J[]` throws a coding error (never retried)."
const TT_FAIL_J = Ref(3)
function tt_fail(x)
    job = TikTak.current_job()
    job !== nothing && job.stage === :local && job.j == TT_FAIL_J[] && error("tt_fail: deliberate coding error in restart $(job.j)")
    return tt_rast(x)
end
TikTak.register_objective!(:fail, tt_fail)

"The FIRST attempt of restart `TT_KILL_J[]` kills its worker process (a retry succeeds)."
const TT_KILL_J = Ref(3)
function tt_kill(x)
    job = TikTak.current_job()
    job !== nothing && job.stage === :local && job.j == TT_KILL_J[] && job.attempt == 1 && exit(17)
    return tt_rast(x)
end
TikTak.register_objective!(:kill, tt_kill)

"Every attempt of restart `TT_KILL_J[]` kills its worker (retries exhausted)."
function tt_kill_always(x)
    job = TikTak.current_job()
    job !== nothing && job.stage === :local && job.j == TT_KILL_J[] && exit(18)
    return tt_rast(x)
end
TikTak.register_objective!(:kill_always, tt_kill_always)

"A fast objective with a long evaluation budget: floods progress messages if they were not coalesced."
tt_flood(x) = sum(abs2, x .- 0.1) + 1e-3 * sum(sin, 50 .* x)
TikTak.register_objective!(:flood, tt_flood)

"Slow everywhere (each evaluation sleeps): jobs stay in flight long enough to pause or crash mid-way."
tt_slow_all(x) = (sleep(0.01); tt_rast(x))
TikTak.register_objective!(:slow_all, tt_slow_all)

# ---- 2026-09-28 (follow-up 3): APPLICATION I/O errors from a healthy worker (finding 12) ----
"Restart `TT_FAIL_J[]` reads a 'truncated file': EOFError from a live worker -- an objective error, not a lost worker."
function tt_eof(x)
    job = TikTak.current_job()
    job !== nothing && job.stage === :local && job.j == TT_FAIL_J[] && throw(EOFError())
    return tt_rast(x)
end
TikTak.register_objective!(:eof, tt_eof)

"Pre-testing (no job) of a point with x[1] > 4: an IOError from a live worker."
function tt_ioerr_pretest(x)
    TikTak.current_job() === nothing && x[1] > 4.0 && throw(Base.IOError("tt_ioerr_pretest: simulated broken input", -5))
    return tt_rast(x)
end
TikTak.register_objective!(:ioerr_pretest, tt_ioerr_pretest)

"The polish reads a truncated file: EOFError."
function tt_eof_polish(x)
    job = TikTak.current_job()
    job !== nothing && job.stage === :polish && throw(EOFError())
    return tt_rast(x)
end
TikTak.register_objective!(:eof_polish, tt_eof_polish)

"Every evaluation sleeps 0.2 s: a job that outlives a short drain deadline."
tt_sleepy(x) = (sleep(0.2); tt_rast(x))
TikTak.register_objective!(:sleepy, tt_sleepy)
