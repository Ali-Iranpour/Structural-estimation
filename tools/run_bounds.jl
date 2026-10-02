# run_bounds.jl -- RUN-SPECIFIC parameter boxes (Part B, the mu profile; 2026-09-21).
#
#     --bounds mu=0.2:0.8,kappa_terminal=0.5:200
#
# replaces the named entries of SMM_PARAMS IN THE CURRENT PROCESS ONLY. The vector is a
# mutable array of immutable SMMParam records, so the source constants (code/smm/moments.jl)
# and the production defaults are untouched; the run's own records say what was in force.
# Everything that reads the box then agrees: `unpack`'s clamp inside the objective, the
# optimiser's lower/upper bounds (reopt.jl), `--fix`'s admissibility check, the
# perturbation step sizes and feasibility clamp (followup_checks.jl perturb), the Jacobian
# steps (jacobian_v5e.jl) and the near-bound report.
#
# Why this is needed rather than widening the constants: `unpack` clamps EVERY evaluated
# point into SMM_PARAMS' box, so an optimiser handed a wider box but the old SMM_PARAMS would
# silently evaluate mu = 0.2 as mu = 0.5. And a point produced under one box and loaded by a
# tool running under another would be clamped silently too -- `check_point_in_box` turns that
# into an error (the "checkpoint compatibility" test of the Part B plan).
#
# Included AFTER code/smm/moments.jl. Pure functions plus one in-place mutation of SMM_PARAMS.

const RUN_BOUNDS_APPLIED = Dict{Symbol,Tuple{Float64,Float64,Float64,Float64}}()   # name => (lo, hi, was_lo, was_hi)

function parse_run_bounds(s::AbstractString)
    d = Dict{Symbol,Tuple{Float64,Float64}}()
    for kv in split(s, ',')
        isempty(strip(kv)) && continue
        parts = split(kv, '='); length(parts) == 2 || error("--bounds: expected name=lo:hi, got '$kv'")
        lohi = split(parts[2], ':'); length(lohi) == 2 || error("--bounds: expected name=lo:hi, got '$kv'")
        lo = parse(Float64, strip(lohi[1])); hi = parse(Float64, strip(lohi[2]))
        d[Symbol(strip(parts[1]))] = (lo, hi)
    end
    d
end

"""
    apply_run_bounds!(d) -> Vector{String}

Replace the box of every parameter named in `d` (name => (lo, hi)) for this process.
Validates: the name is an estimated parameter; lo < hi; a :log-linked parameter keeps lo > 0.
Returns one human-readable line per replaced box (old box included) for the run's log.
"""
function apply_run_bounds!(d::AbstractDict{Symbol,Tuple{Float64,Float64}})
    lines = String[]
    for (k, (lo, hi)) in d
        i = findfirst(q -> q.name === k, SMM_PARAMS)
        i === nothing && error("--bounds $k: not an estimated parameter (SMM_PARAMS)")
        q = SMM_PARAMS[i]
        lo < hi || error("--bounds $k=$lo:$hi: lo must be < hi")
        isfinite(lo) && isfinite(hi) || error("--bounds $k=$lo:$hi: bounds must be finite")
        q.link === :log && lo <= 0 && error("--bounds $k=$lo:$hi: a log-linked parameter needs lo > 0")
        SMM_PARAMS[i] = SMMParam(k, lo, hi, q.link, q.owner)
        RUN_BOUNDS_APPLIED[k] = (lo, hi, q.lo, q.hi)
        push!(lines, "$k = [$lo, $hi] (default [$(q.lo), $(q.hi)])")
    end
    lines
end

run_bounds_string(d::AbstractDict) = join(("$k=$(v[1]):$(v[2])" for (k, v) in sort(collect(d); by = first)), ",")

"""
    check_point_in_box(point; what = "point", tol = 1e-9)

Error if any parameter of `point` (name => value) lies outside the box currently in force.
A value outside would otherwise be clamped silently by `unpack` and the tool would answer for
a different point than the one it was given. Values within `tol` (relative to the box width)
of a bound are accepted (the optimiser writes points on the boundary).
"""
function check_point_in_box(point::AbstractDict{Symbol,<:Real}; what::AbstractString = "point", tol::Float64 = 1e-9)
    bad = String[]
    for q in SMM_PARAMS
        haskey(point, q.name) || continue
        v = Float64(point[q.name]); w = q.hi - q.lo
        (q.lo - tol * w <= v <= q.hi + tol * w) || push!(bad, "$(q.name) = $v ∉ [$(q.lo), $(q.hi)]")
    end
    isempty(bad) || error("$what: outside the parameter box in force (pass the run's --bounds, or the point belongs to another box): " * join(bad, "; "))
    nothing
end

# the TOML block every tool writes so that a downstream reader can verify the box in force
function run_bounds_record(io::IO)
    println(io, "\n[run_bounds]")
    for (k, (lo, hi, wlo, whi)) in sort(collect(RUN_BOUNDS_APPLIED); by = first)
        println(io, k, " = { lo = ", lo, ", hi = ", hi, ", default_lo = ", wlo, ", default_hi = ", whi, " }")
    end
end
