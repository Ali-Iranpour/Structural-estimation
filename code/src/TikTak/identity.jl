# =============================================================================
# identity.jl -- what a checkpoint must match before a run may continue from it.
#
# Two identities, verified SEPARATELY (plan step 4.5):
#
#   objective   what is minimised. Supplied by the caller: the runner's objective_identity()
#               (targets by content, model source, parameter box and links, moments,
#               weights, grids, simulation draws), reduced to `objective_id`. Pre-testing
#               values may be reused across optimizer experiments on the same objective.
#   optimizer   how the search runs: this module's version and source, NLopt and Sobol
#               versions, every setting in TikTakConfig, the box, the pre-testing design and
#               the supplied seeds.
#
# Execution settings -- local mode, worker count, a --stop-after-restarts limit -- are not
# part of either: changing them does not change what was committed.
#
# WHAT A RESUME MAY CHANGE (2026-09-28, follow-up 1.4-1.5). The optimizer fields fall in three
# classes (`identity_change`):
#
#   plan      the box (its dimension included), the pre-testing design, the supplied seeds, the
#             requested restarts (K), the mixing schedule, the invalid-value threshold, stop_tol,
#             the verification tolerances, the bootstrap -- WHAT search is being continued.
#             Never changed on a continuation, not even with allow_optimizer_change: the saved
#             seeds, incumbent and records belong to the saved plan. Such a change is a new run
#             (reuse the pre-testing cache, warm-start from the saved point).
#   solver    the local and polish solver settings, skip_polish, normalize, on_error. Allowed
#             only with allow_optimizer_change; it opens a new settings epoch, so completed
#             restarts and replayed jobs keep the settings they ran with.
#   software  this module's version and source hash, the NLopt and Sobol versions. Allowed
#             only with allow_optimizer_change (a bug fix must not strand a running search),
#             recorded, and the run is then labelled :changed_optimizer.
# =============================================================================

"sha256 of `s`, first `n` hex characters."
sha_hex(s::AbstractString, n::Int = 16) = bytes2hex(SHA.sha256(s))[1:n]
sha_hex(b::AbstractVector{UInt8}, n::Int = 16) = bytes2hex(SHA.sha256(collect(b)))[1:n]

"The TOML text of a Dict with sorted keys: the canonical form hashed into an identity."
function canonical_toml(d::AbstractDict)
    io = IOBuffer()
    TOML.print(io, d; sorted = true)
    return String(take!(io))
end

"An identity hash over a Dict of fields (order-independent)."
fields_id(d::AbstractDict) = sha_hex(canonical_toml(d))

"""
    tiktak_source_sha() -> String

Content hash of this module's source files. Part of the optimizer identity: a checkpoint
written by other optimizer code is not continued silently.
"""
function tiktak_source_sha()
    files = sort(filter(f -> endswith(f, ".jl"), readdir(MODULE_DIR)))
    return sha_hex(reduce(vcat, (read(joinpath(MODULE_DIR, f)) for f in files)))
end

_pkgver(m::Module) = try string(pkgversion(m)) catch; "unknown" end

"""
    design_fields(lo, hi, skip_first, supplied) -> Dict{String,Any}

The identity of the pre-testing CANDIDATE DESIGN: the box, the Sobol' generator (its
version and its first points), the skip rule and the supplied points. Not the number of
draws: a cache of the first 1000 draws is a valid prefix of a 10000-draw design.
"""
function design_fields(lo::Vector{Float64}, hi::Vector{Float64}, skip_first::Bool,
                       supplied::Vector{Vector{Float64}})
    first_pts = sobol_points(lo, hi, 4; skip_first = skip_first)
    return Dict{String,Any}("lo" => lo, "hi" => hi, "skip_first" => skip_first,
                            "sobol_version" => _pkgver(Sobol),
                            "first_points_sha" => sha_hex(reinterpret(UInt8, reduce(vcat, first_pts))),
                            "supplied" => supplied)
end

"""
    optimizer_fields(cfg, lo, hi, supplied) -> Dict{String,Any}

Every setting that decides the search path, with the software that runs it.
"""
function optimizer_fields(cfg::TikTakConfig, lo::Vector{Float64}, hi::Vector{Float64},
                          supplied::Vector{Vector{Float64}})
    solver = solver_fields                             # checkpoint.jl: the same fields an epoch records
    return Dict{String,Any}(
        "tiktak_version" => TIKTAK_VERSION,
        "tiktak_source_sha" => tiktak_source_sha(),
        "nlopt_version" => string(NLopt.version()),
        "nlopt_jl_version" => _pkgver(NLopt),
        "sobol_version" => _pkgver(Sobol),
        "lo" => lo, "hi" => hi,
        "n_sobol" => cfg.n_sobol, "n_valid_target" => cfg.n_valid_target,
        "require_valid_target" => cfg.require_valid_target, "skip_first" => cfg.skip_first,
        "invalid_value" => cfg.invalid_value,
        "supplied_seeds_sha" => sha_hex(isempty(supplied) ? UInt8[] :
                                        reinterpret(UInt8, reduce(vcat, supplied))),
        "n_supplied" => length(supplied),
        "nstar_requested" => cfg.nstar,
        "theta" => [cfg.theta_p, cfg.theta_lo, cfg.theta_hi],
        "local" => solver(cfg.local_), "polish" => solver(cfg.polish),
        "skip_polish" => cfg.skip_polish, "stop_tol" => cfg.stop_tol,
        "on_error" => String(cfg.on_error),
        "verify" => [cfg.verify_xtol, cfg.verify_ftol_rel, cfg.verify_ftol_abs],
        "bootstrap" => String(cfg.bootstrap), "normalize" => cfg.normalize)
end

"""
    identity_differences(saved, now) -> Vector{String}

Field-by-field differences between two identity Dicts, one readable line per field, so a
refusal names what changed rather than only that a hash differs.
"""
function identity_differences(saved::AbstractDict, now::AbstractDict)
    out = String[]
    for k in sort(unique(vcat(collect(String.(keys(saved))), collect(String.(keys(now))))))
        a = get(saved, k, nothing); b = get(now, k, nothing)
        a isa AbstractDict && b isa AbstractDict && (append!(out, ["$k." * d for d in identity_differences(a, b)]); continue)
        _same(a, b) || push!(out, "$k: saved $(repr(a)), now $(repr(b))")
    end
    return out
end
_same(a, b) = a == b
_same(a::AbstractVector, b::AbstractVector) = length(a) == length(b) && all(_same(x, y) for (x, y) in zip(a, b))
_same(a::Real, b::Real) = a == b || (isnan(a) && isnan(b))
_same(a::AbstractDict, b::AbstractDict) = Set(String.(keys(a))) == Set(String.(keys(b))) &&
                                          all(_same(a[k], b[k]) for k in keys(a))

"Optimizer-identity fields naming the software that runs the search (see the file header)."
const SOFTWARE_FIELDS = ("tiktak_version", "tiktak_source_sha", "nlopt_version", "nlopt_jl_version", "sobol_version")
"Optimizer-identity fields of the solver settings: changeable on a resume only explicitly, as a new epoch."
const SOLVER_FIELDS = ("local", "polish", "skip_polish", "normalize", "on_error")

"""
    identity_change(saved, now) -> (software, solver, plan)

The changed optimizer-identity fields, by class (the file header), each as the readable lines
of `identity_differences`. A field present on one side only is a PLAN change: an unknown
field cannot be judged harmless.
"""
function identity_change(saved::AbstractDict, now::AbstractDict)
    out = (software = String[], solver = String[], plan = String[])
    for k in sort(unique(vcat(collect(String.(keys(saved))), collect(String.(keys(now))))))
        a = get(saved, k, nothing); b = get(now, k, nothing)
        a !== nothing && b !== nothing && _same(a, b) && continue
        lines = a isa AbstractDict && b isa AbstractDict ? ["$k." * d for d in identity_differences(a, b)] :
                ["$k: saved $(repr(a)), now $(repr(b))"]
        cls = (a === nothing || b === nothing) ? :plan : k in SOFTWARE_FIELDS ? :software :
              k in SOLVER_FIELDS ? :solver : :plan
        append!(getfield(out, cls), lines)
    end
    return out
end

"Raised when a checkpoint does not describe the run it is asked to continue."
struct ResumeRefused <: Exception
    msg::String
end
Base.showerror(io::IO, e::ResumeRefused) = print(io, "TikTak refuses to resume: ", e.msg)
