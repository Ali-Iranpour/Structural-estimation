# =============================================================================
# tiktak.jl -- adapter for the TikTak optimizer module (2026-09-27).
#
# The optimizer lives in code/src/TikTak/ as the module `TikTak`, with its own NLopt,
# Sobol and Printf imports (tiktak_fix_plan.md, J1). Every existing caller keeps writing
#
#     include(joinpath(SRC, "tiktak.jl"))
#
# and gets `tiktak`, `tiktak_selftest`, `ret_class`, `ret_tally` and `TikTakResult` in
# scope, exactly as before. Everything else is reached qualified (`TikTak.name`), so the
# module cannot shadow a name of the economic model (moments.jl defines `incumbent()`).
#
# Loading this file starts no worker and runs no estimation. Including it twice in one
# process is a no-op, so a script that includes it under @everywhere and again directly
# does not redefine the module.
# =============================================================================

isdefined(@__MODULE__, :TikTak) || include(joinpath(@__DIR__, "TikTak", "TikTak.jl"))
using .TikTak
