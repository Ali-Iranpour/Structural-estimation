# =============================================================================
# paths.jl — single source of truth for every path in the project.
#
# Nothing else should build a path with a hard-coded folder name. If the
# repository is reorganized again, this file is the only one that changes.
#
# Layout (all paths derived from this file's own location):
#
#   <ROOT>/code/src/paths.jl   ← you are here
#   <ROOT>/output/figures/
#   <ROOT>/output/tables/
#   <ROOT>/output/data/
#   <ROOT>/output/reports/
#   <ROOT>/docs/
#
# Usage:
#     savefig(p, joinpath(figpath("Baseline"), "policy_functions.pdf"))
#     CSV.write(joinpath(tabpath(), "college_shares.csv"), df)
#
# The fig/tab/data/report helpers CREATE the directory and return its path,
# so they are safe to call inline. Use `figdir(...)` if you want the path
# without creating anything.
# =============================================================================

const ROOT     = normpath(joinpath(@__DIR__, "..", ".."))
const CODE_DIR = joinpath(ROOT, "code")
const SRC_DIR  = joinpath(CODE_DIR, "src")
const DOCS_DIR = joinpath(ROOT, "docs")

# STRUCT_EST_OUTPUT redirects every output helper somewhere else. The smoke test sets it,
# because it runs the notebook at shrunken grids and would otherwise drop 12x12x4 figures
# and 200-agent tables into output/, which is tracked. Provenance already records that a
# table came from a shrunken run; this stops it being committed in the first place.
const OUT_DIR = get(ENV, "STRUCT_EST_OUTPUT", joinpath(ROOT, "output"))

const FIG_DIR    = joinpath(OUT_DIR, "figures")
const TAB_DIR    = joinpath(OUT_DIR, "tables")
const DATA_DIR   = joinpath(OUT_DIR, "data")
const REPORT_DIR = joinpath(OUT_DIR, "reports")

# --- path builders (do NOT create anything) ---------------------------------
figdir(parts...)    = joinpath(FIG_DIR,    parts...)
tabdir(parts...)    = joinpath(TAB_DIR,    parts...)
datadir(parts...)   = joinpath(DATA_DIR,   parts...)
reportdir(parts...) = joinpath(REPORT_DIR, parts...)

# --- path builders that ensure the directory exists -------------------------
figpath(parts...)    = (d = figdir(parts...);    isdir(d) || mkpath(d); d)
tabpath(parts...)    = (d = tabdir(parts...);    isdir(d) || mkpath(d); d)
datapath(parts...)   = (d = datadir(parts...);   isdir(d) || mkpath(d); d)
reportpath(parts...) = (d = reportdir(parts...); isdir(d) || mkpath(d); d)

"""
    timestamp()

`yyyy-mm-dd_HHMMSS` string, for run-specific output folders.
"""
timestamp() = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")

"""
    unique_path(dir, basename, ext=".pdf")

Full path inside `dir` that does not already exist: returns
`basename.ext`, else `basename_2.ext`, `basename_3.ext`, …
Replaces the copy-pasted overwrite-avoidance loops that were scattered
through the plotting functions.
"""
function unique_path(dir::AbstractString, basename::AbstractString, ext::AbstractString=".pdf")
    isdir(dir) || mkpath(dir)
    p = joinpath(dir, basename * ext)
    isfile(p) || return p
    i = 2
    while isfile(joinpath(dir, "$(basename)_$(i)$(ext)"))
        i += 1
    end
    return joinpath(dir, "$(basename)_$(i)$(ext)")
end

"""
    sanitize(s)

Turn a plot title (possibly a LaTeXString) into a safe filename stem.
"""
function sanitize(x)
    s = string(x)
    s = replace(s, r"\$" => "", "\\" => " ", "{" => "", "}" => "",
                   "^" => "", ":" => "-", "/" => "-")
    s = replace(s, r"\s+" => "_")
    s = replace(s, r"[^\w\-\_]" => "")
    s = replace(s, r"_{2,}" => "_")
    s = replace(s, r"-{2,}" => "-")
    return isempty(s) ? "plot" : s
end

function __print_paths()
    println("ROOT       = ", ROOT)
    for (n, d) in (("figures", FIG_DIR), ("tables", TAB_DIR),
                   ("data", DATA_DIR), ("reports", REPORT_DIR))
        println(rpad("  " * n, 12), "= ", d, isdir(d) ? "" : "   (not created yet)")
    end
end
