#!/usr/bin/env julia
# =============================================================================
# report_point.jl -- the fit report (code/smm/moments.jl report_fit) at a saved point
#
#     julia --project=. --threads=1 tools/report_point.jl <targets.toml> <estimates.toml> [--start <toml>] [--grid 30]
#
# (2026-10-02, Ali.) Prints, at the EXACT point of an estimates.toml ([search_vector]), the report a run
# prints at its end with the CURRENT report code: the parameters with their start, box and position in it,
# the targeted moments, and the untargeted checks (the college regression, the skill gradient, the transfer
# at 18 by path). For a run whose own log has an older report (e.g. one launched from frozen code).
# --start: a toml with [parameters] (an estimates.toml, theta0.toml) to show as the start column; without
# it, the [parameters] of the run's init-from file recorded in run_record.toml beside the estimates, when
# there is one, else the code's default start. Grids: parent Na = Nhc = --grid (30), child 30/30/5,
# simN 2000, seed 1234 -- the production report grid. Changes nothing on disk.
# =============================================================================
using Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames, Statistics, Dates
using ProgressMeter, Distributions, StatsBase, QuantEcon, FastGaussQuadrature, Parameters, Dierckx, TOML
const REPO = normpath(joinpath(@__DIR__, "..")); const SRC = joinpath(REPO, "code", "src")
include(joinpath(SRC, "paths.jl")); include(joinpath(SRC, "manifest.jl")); include(joinpath(SRC, "diagnostics.jl"))
include(joinpath(SRC, "child_lifecycle.jl")); include(joinpath(SRC, "parent_family.jl")); include(joinpath(SRC, "tiktak.jl"))
include(joinpath(REPO, "code", "smm", "moments.jl"))

argstr(f, d) = (i = findfirst(==(f), ARGS); i === nothing ? d : ARGS[i + 1])
length(ARGS) >= 2 || error("usage: report_point.jl <targets.toml> <estimates.toml> [--start <toml>] [--grid 30]")
const T = load_targets(abspath(ARGS[1]))
const EST = TOML.parsefile(abspath(ARGS[2]))
haskey(EST, "search_vector") || error("$(ARGS[2]) has no [search_vector]")
const Z = Float64.(EST["search_vector"]["z"])
length(Z) == length(SMM_PARAMS) || error("the search vector has $(length(Z)) entries; this specification has $(length(SMM_PARAMS))")
const G = parse(Int, argstr("--grid", "30"))
"The run's init-from file from the run_record.toml beside the estimates (in its [parameters] table), or \"\"."
function recorded_init_from(est_path)
    rr = joinpath(dirname(abspath(est_path)), "run_record.toml")
    isfile(rr) || return ""
    for tab in values(TOML.parsefile(rr))
        tab isa AbstractDict || continue
        f = get(tab, "init_from", "")
        f isa String && isfile(f) && return f
    end
    return ""
end
const start_file = let s = argstr("--start", ""); isempty(s) ? recorded_init_from(ARGS[2]) : s end
start = isempty(start_file) ? nothing :
    let pr = TOML.parsefile(start_file)["parameters"]
        Dict(q.name => Float64(pr[String(q.name)]) for q in SMM_PARAMS)
    end
println("report at the point of $(ARGS[2]); start column: ", isempty(start_file) ? "the code's default start" : start_file)
report_fit(Z, T; Na = G, Nk = 2, Nhc = G, simN = 2000, seed = 1234, child_grid = (Na = 30, Nk = 30, Nt = 5), start = start)
