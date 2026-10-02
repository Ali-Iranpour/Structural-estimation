# =============================================================================
# presets.jl -- what a run is FOR, separate from how big it is (plan step 8).
#
# A preset names the purpose of a run and supplies DEFAULT budgets and gates for it. The
# caller's explicit settings always win; the label is recorded with the run, never inferred
# from the number of restarts. Five restarts are a development test; a final estimation may
# use 100 to 1,000 restarts with a matching pre-testing pool (the paper's K = 0.1 N).
#
# The budgets are starting points from tiktak_fix_plan.md's run ladder, not tuned values:
#   smoke        does it run? finite outputs, penalty reasons, workers, checkpoints. Polish
#                skipped; a budget-limited answer is expected and is not a failure.
#   integration  can one realistic local solve and a bounded polish converge and be reported
#                correctly? No global-fit claim.
#   pilot        search quality against time: cap adequacy, basin behaviour, worker counts.
#                A pool too small for the restarts is a failed gate.
#   production   a substantial global search, to be followed by numerical and economic
#                validation. The local evaluation cap should come from pilot traces.
# =============================================================================

"""
    Preset

Default budgets and gates of one purpose. `n_sobol` is the attempt cap when
`n_valid_target > 0`. `requires_convergence = false` means a run of this purpose reports
successful execution without claiming estimation convergence.
"""
struct Preset
    name::String
    n_sobol::Int
    n_valid_target::Int
    nstar::Int
    local_maxeval::Int
    polish_maxeval::Int
    skip_polish::Bool
    allow_fewer_restarts::Bool
    require_valid_target::Bool
    requires_convergence::Bool
    note::String
end

const PRESETS = Dict{String,Preset}(
    "smoke" => Preset("smoke", 64, 0, 5, 60, 120, true, true, false, false,
        "execution check -- finite outputs, penalties, workers, checkpoints; estimation convergence NOT required"),
    "integration" => Preset("integration", 1500, 150, 5, 500, 200, false, true, false, true,
        "one realistic local solve and a bounded polish, reported correctly; no global-fit claim"),
    "pilot" => Preset("pilot", 10_000, 1000, 50, 1500, 1000, false, false, true, true,
        "search-quality pilot: quality against time, cap adequacy, basins, worker counts"),
    "production" => Preset("production", 100_000, 10_000, 1000, 2000, 4000, false, false, true, true,
        "a substantial global search, to be followed by numerical and economic validation"),
)

"""
    resolve_preset(name) -> Preset

The preset of that name; "" or "custom" is no preset (every setting explicit or a caller
default). An unknown name is an error, never a silent fallback.
"""
function resolve_preset(name::AbstractString)
    (isempty(name) || name == "custom") && return Preset("custom", 0, 0, 0, 0, 0, false, true, false, true,
                                                          "no preset: settings as given")
    haskey(PRESETS, name) || throw(ArgumentError("TikTak: unknown preset \"$name\"; known: " *
                                                 join(sort(collect(keys(PRESETS))), ", ")))
    return PRESETS[name]
end

"""
    execution_verdict(purpose, acceptance) -> String

One line for the end of a run that keeps execution and estimation apart: a smoke run that
executed cleanly PASSES as a smoke run whatever its convergence; any other purpose reports
the acceptance verdict.
"""
function execution_verdict(purpose::AbstractString, acc::Acceptance)
    p = resolve_preset(purpose)
    if !p.requires_convergence
        return acc.execution_ok && acc.candidate_valid ?
            "$(p.name): execution PASSED; estimation convergence not required (and not claimed)" :
            "$(p.name): execution FAILED -- " * join(filter(r -> !occursin("convergence", r) && !occursin("bound", r), acc.reasons), "; ")
    end
    return acc.accepted ? "$(p.name): ACCEPTED (no claim of global optimality)" :
                          "$(p.name): NOT ACCEPTED -- " * join(acc.reasons, "; ")
end
