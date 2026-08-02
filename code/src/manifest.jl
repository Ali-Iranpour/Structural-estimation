# =============================================================================
# manifest.jl — run provenance.
#
# Writes a small TOML next to a set of outputs recording WHEN they were made,
# from WHICH commit, on WHICH Julia/package versions, and with WHICH parameter
# values. Lets any figure in the paper be traced back to the code that made it.
#
# Usage:
#     dir = figpath("Parameters")
#     write_manifest(dir; experiment = "sigma counterfactuals",
#                         mu_1 = -0.04, rho = 1.5, Na = 30, simN = 5000)
#
# Writes <dir>/run_manifest.toml (or run_manifest_2.toml, … if one exists).
#
# Requires paths.jl (for unique_path, ROOT) and Dates.
# =============================================================================

import Pkg

"""
    git_sha(; short=true)

Current commit of the repository, or `"unknown"` if git is unavailable or
this is not a checkout. Suffixed `-dirty` when the tree has uncommitted
changes, so a manifest never claims a clean provenance it does not have.
"""
function git_sha(; short::Bool=true)
    try
        fmt = short ? "%h" : "%H"
        sha = strip(read(`git -C $ROOT log -1 --format=$fmt`, String))
        dirty = !isempty(strip(read(`git -C $ROOT status --porcelain`, String)))
        return dirty ? sha * "-dirty" : sha
    catch
        return "unknown"
    end
end

"""
    package_versions(names...)

`Dict` of package name => version string for the given direct dependencies.
Defaults to the packages this project's results actually depend on.
"""
function package_versions(names = ["NLopt", "Interpolations", "Dierckx", "QuantEcon",
                                   "FastGaussQuadrature", "Distributions"])
    out = Dict{String,String}()
    try
        for (_, p) in Pkg.dependencies()
            if p.name in names && p.version !== nothing
                out[p.name] = string(p.version)
            end
        end
    catch
    end
    return out
end

_toml_val(v::AbstractString) = "\"" * replace(v, "\"" => "\\\"") * "\""
_toml_val(v::Bool)           = string(v)
_toml_val(v::Real)           = string(v)
_toml_val(v::Symbol)         = _toml_val(string(v))
_toml_val(v::AbstractVector) = "[" * join(_toml_val.(v), ", ") * "]"
_toml_val(v)                 = _toml_val(string(v))

"""
    write_manifest(dir; kwargs...)

Write run provenance to `dir/run_manifest.toml`. Every keyword is recorded
under `[parameters]`; pass whatever identifies the run (the parameters you
varied, grid sizes, simN). Returns the path written.
"""
function write_manifest(dir::AbstractString; kwargs...)
    isdir(dir) || mkpath(dir)
    path = unique_path(dir, "run_manifest", ".toml")

    io = IOBuffer()
    println(io, "# Run provenance — written automatically. Do not edit by hand.")
    println(io)
    println(io, "[run]")
    println(io, "timestamp   = ", _toml_val(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")))
    println(io, "git_commit  = ", _toml_val(git_sha()))
    println(io, "julia       = ", _toml_val(string(VERSION)))
    println(io, "hostname    = ", _toml_val(gethostname()))
    # repo-relative when the output lives inside the project, absolute otherwise
    absdir = abspath(dir)
    shown  = startswith(absdir, abspath(ROOT)) ? relpath(absdir, ROOT) : absdir
    println(io, "output_dir  = ", _toml_val(shown))
    println(io)

    pv = package_versions()
    if !isempty(pv)
        println(io, "[packages]")
        for k in sort(collect(keys(pv)))
            println(io, rpad(k, 20), "= ", _toml_val(pv[k]))
        end
        println(io)
    end

    if !isempty(kwargs)
        println(io, "[parameters]")
        for (k, v) in pairs(kwargs)
            println(io, rpad(string(k), 20), "= ", _toml_val(v))
        end
    end

    write(path, String(take!(io)))
    @info "Wrote run manifest" path
    return path
end
