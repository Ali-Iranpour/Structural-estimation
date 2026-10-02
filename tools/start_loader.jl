# start_loader.jl -- loading and VALIDATING a starting point or a checkpoint for tools/reopt.jl
# (2026-09-22; Part B correction 1). Included after code/smm/moments.jl and tools/run_bounds.jl.
#
# Before this file the estimator clamped an out-of-box start into the box with one log line
# ("CLAMPED into the box") and never checked finiteness; no run log holds that line (checked
# 2026-09-22 over batch 1, the audit and Part B), so no result started from a transformed point.
#
# Rules:
#   * a loaded value that is non-finite, or that lies outside the box in force by more than
#     START_BOUND_TOL of the box width (search coordinates), REFUSES the run (error before any
#     evaluation); a value within that tolerance is snapped onto the bound -- the optimiser
#     writes points exactly on the boundary and TOML round-trips are not bit-exact;
#   * both loading paths are validated: [search_vector] (the full z of a checkpoint / results
#     file) and [parameters] (a point by name; missing names default to smm_start and are named);
#   * a --fix'ed coordinate is taken from --fix, never from the file; when the file's value
#     differs the note says so (the mu profile starts every search from another mu);
#   * a WARM START is a start file whose recorded [run_bounds] differ from the box in force but
#     whose point is feasible in it: allowed, never transformed, and named in the log;
#   * a RESUME (--resume, checkpoint_best.toml present) additionally verifies the checkpoint's
#     [config] block -- run bounds, fixed parameters, targets and extra rows (the objective),
#     seed, simN, grid, grid overrides, budget, init step, ftol and the model-source hash --
#     against the run in force; any difference refuses the resume. A LEGACY checkpoint (written
#     before 2026-09-22, no [config]) can only be verified on the box and the fixed coordinates;
#     the resume proceeds with an explicit "UNVERIFIED" note (the driver relaunches the identical
#     command), and results.toml records resume_verification = "partial (legacy checkpoint)".
# Pure functions; no file writes.

const START_BOUND_TOL = 1e-9     # of the box width in search coordinates; narrowly defined, never widened

_fmt(v) = @sprintf("%.6g", v)

"""
    load_start_point(path; free_idx, fixed, params = SMM_PARAMS, tol = START_BOUND_TOL)
        -> (z::Vector{Float64}, how::String, notes::Vector{String})

`z` is the full search vector (one entry per `params`), finite and inside the box in force on
every FREE coordinate (snapped onto a bound when within `tol` of it); the fixed coordinates are
left as loaded (the caller embeds the --fix values). Errors name every offending coordinate.
"""
function load_start_point(path::AbstractString; free_idx::AbstractVector{Int}, fixed::AbstractDict{Symbol,Float64},
                          params = SMM_PARAMS, tol::Float64 = START_BOUND_TOL)
    raw = TOML.parsefile(path)
    notes = String[]; bad = String[]
    n = length(params)
    if haskey(raw, "search_vector") && haskey(raw["search_vector"], "z") && length(raw["search_vector"]["z"]) == n
        zraw = raw["search_vector"]["z"]
        all(x -> x isa Real, zraw) || error("$path: [search_vector] z holds a non-numeric entry")
        z = Float64.(zraw); how = "search_vector"
    else
        haskey(raw, "search_vector") && push!(notes, "[search_vector] ignored: length $(length(get(raw["search_vector"], "z", []))) ≠ $n")
        haskey(raw, "parameters") || error("$path has neither a usable [search_vector] nor [parameters]")
        P = raw["parameters"]; z = fill(NaN, n); missing_names = String[]
        for (i, q) in enumerate(params)
            k = String(q.name)
            if haskey(P, k)
                P[k] isa Real || error("$path: [parameters] $k is not a number")
                v = Float64(P[k])
                isfinite(v) || push!(bad, "$k = $v (non-finite)")
                q.link === :log && v <= 0 && isfinite(v) && push!(bad, "$k = $v (log-linked, needs > 0)")
                z[i] = (isfinite(v) && !(q.link === :log && v <= 0)) ? to_search(v, q) : NaN
            else
                z[i] = to_search(smm_start(q.name), q); push!(missing_names, k)
            end
        end
        isempty(missing_names) || push!(notes, "defaulted to smm_start (absent from [parameters]): " * join(missing_names, ", "))
        how = "parameters"
    end
    isempty(bad) || error("$path: refused, invalid loaded values: " * join(bad, "; "))
    # finiteness of the search vector (covers the search_vector path and log(0) = -Inf)
    for (i, q) in enumerate(params)
        isfinite(z[i]) || push!(bad, "$(q.name) (search coordinate $(z[i]))")
    end
    isempty(bad) || error("$path: refused, non-finite search coordinates: " * join(bad, "; "))
    # the box in force, FREE coordinates only: refuse a material violation, snap a boundary value
    snapped = String[]
    for i in free_idx
        q = params[i]; lo = to_search(q.lo, q); hi = to_search(q.hi, q); w = hi - lo
        if z[i] < lo - tol * w || z[i] > hi + tol * w
            push!(bad, @sprintf("%s = %s ∉ [%s, %s]", q.name, _fmt(from_search(z[i], q)), _fmt(q.lo), _fmt(q.hi)))
        elseif z[i] < lo || z[i] > hi
            z[i] = clamp(z[i], lo, hi); push!(snapped, String(q.name))
        end
    end
    isempty(bad) || error("$path: refused, outside the parameter box in force (a start is never clamped; pass the run's --bounds if this is the run's own point, or choose a feasible start): " * join(bad, "; "))
    isempty(snapped) || push!(notes, "snapped onto the bound (within $(tol) of the box width): " * join(snapped, ", "))
    # fixed coordinates: --fix governs; say when the file disagrees
    diff = String[]
    for (i, q) in enumerate(params)
        haskey(fixed, q.name) || continue
        v = from_search(z[i], q)
        isapprox(v, fixed[q.name]; rtol = 1e-9, atol = 1e-12) || push!(diff, @sprintf("%s %s → %s", q.name, _fmt(v), _fmt(fixed[q.name])))
    end
    isempty(diff) || push!(notes, "fixed coordinates taken from --fix, not from the file: " * join(diff, ", "))
    # warm start from another box: recorded [run_bounds] differ from the box in force (the point is feasible, see above)
    if haskey(raw, "run_bounds") && raw["run_bounds"] isa AbstractDict
        other = String[]
        for (i, q) in enumerate(params)
            rb = get(raw["run_bounds"], String(q.name), nothing); rb === nothing && continue
            (isapprox(Float64(rb["lo"]), q.lo) && isapprox(Float64(rb["hi"]), q.hi)) || push!(other, @sprintf("%s [%s, %s] vs [%s, %s] in force", q.name, _fmt(rb["lo"]), _fmt(rb["hi"]), _fmt(q.lo), _fmt(q.hi)))
        end
        for (i, q) in enumerate(params)      # a box in force that the file does not record at all
            haskey(RUN_BOUNDS_APPLIED, q.name) && !haskey(raw["run_bounds"], String(q.name)) && push!(other, @sprintf("%s [%s, %s] in force, file records the default", q.name, _fmt(q.lo), _fmt(q.hi)))
        end
        isempty(other) || push!(notes, "WARM START from another box (feasible here, not transformed): " * join(other, "; "))
    elseif !isempty(RUN_BOUNDS_APPLIED)
        push!(notes, "WARM START: the file records no [run_bounds]; box in force " * join(("$k [$(v[1]), $(v[2])]" for (k, v) in sort(collect(RUN_BOUNDS_APPLIED); by = first)), ", "))
    end
    return z, how, notes
end

"""
    load_start_omega(path, lo, hi; tol = START_BOUND_TOL) -> (omega::Float64, note::String)

The start value of an ESTIMATED omega (reopt.jl --free omega=LO:HI, the mu = 0.7 plan, 2026-09-24):
[parameters].omega of the start file. Same rules as the SMM coordinates: absent or non-finite refuses the
run, a value outside [lo, hi] by more than `tol` of the width refuses it, a value within `tol` is snapped.
A start carries omega explicitly -- tools/omega_starts.py --mu-fixed writes it -- never a silent default.
"""
function load_start_omega(path::AbstractString, lo::Float64, hi::Float64; tol::Float64 = START_BOUND_TOL)
    raw = TOML.parsefile(path)
    P = get(raw, "parameters", Dict{String,Any}())
    haskey(P, "omega") || error("$path: refused, omega is estimated (--free) but the start has no [parameters].omega " *
                                "(use tools/omega_starts.py --mu-fixed to transform a point)")
    P["omega"] isa Real || error("$path: [parameters] omega is not a number")
    w = Float64(P["omega"]); isfinite(w) || error("$path: refused, omega = $w (non-finite)")
    width = hi - lo
    (w < lo - tol * width || w > hi + tol * width) && error("$path: refused, omega = $(_fmt(w)) ∉ [$(_fmt(lo)), $(_fmt(hi))] (a start is never clamped)")
    (w < lo || w > hi) && return clamp(w, lo, hi), " (snapped onto the bound)"
    return w, ""
end

# ---- the configuration a checkpoint records, and its verification on --resume --------------------
const CHECKPOINT_CONFIG_KEYS = ("run_bounds", "fixed", "targets", "extra_moments", "seed", "simN", "grid", "parent_extra", "child_extra",
                                "local_maxeval", "init_step", "ftol_rel", "source_sha16", "tools_sha16", "free_extra", "mode")
# keys added 2026-09-24: a checkpoint written before then lacks them and is read with these values
const CHECKPOINT_CONFIG_DEFAULTS = Dict{String,Any}("free_extra" => "", "mode" => "local")

"""
    checkpoint_config(; kw...) -> Dict{String,Any}

The block `[config]` of a checkpoint: everything a resume must match. Values are TOML-printable.
"""
function checkpoint_config(; run_bounds::AbstractString, fixed::AbstractDict{Symbol,Float64}, targets::AbstractString, extra_moments,
                           seed::Int, simN::Int, grid::Int, parent_extra::AbstractString, child_extra::AbstractString,
                           local_maxeval::Int, init_step::Float64, ftol_rel::Float64, source_sha16::AbstractString, tools_sha16::AbstractString,
                           free_extra::AbstractString = "", mode::AbstractString = "local", objective_id::AbstractString = "")
    Dict{String,Any}("run_bounds" => String(run_bounds), "fixed" => Dict{String,Float64}(String(k) => v for (k, v) in fixed),
                     "targets" => String(targets), "extra_moments" => String[String(k) for k in extra_moments],
                     "seed" => seed, "simN" => simN, "grid" => grid, "parent_extra" => String(parent_extra), "child_extra" => String(child_extra),
                     "local_maxeval" => local_maxeval, "init_step" => init_step, "ftol_rel" => ftol_rel,
                     "source_sha16" => String(source_sha16), "tools_sha16" => String(tools_sha16),
                     "free_extra" => String(free_extra), "mode" => String(mode),
                     # the CONTENT identity of the objective (tools/reopt_identity.jl, 2026-09-28): the
                     # target files by content, the extra rows' weights, the SMM_* switches, model and
                     # adapter source. "" = not supplied (checked only when the run supplies it).
                     "objective_id" => String(objective_id))
end

"""
    verify_checkpoint(raw::AbstractDict, now::AbstractDict; free_idx, fixed, params = SMM_PARAMS)
        -> (status::String, notes::Vector{String})

`status` is "verified" (a [config] block matched item by item), "partial (legacy checkpoint)"
(no [config]: only the box and the fixed coordinates could be checked), or an error is thrown
naming every mismatch. The parameter point itself is validated by `load_start_point`.
"""
function verify_checkpoint(raw::AbstractDict, now::AbstractDict; free_idx, fixed::AbstractDict{Symbol,Float64}, params = SMM_PARAMS)
    notes = String[]
    # the fixed coordinates as written in the checkpoint's [parameters] must equal --fix
    if haskey(raw, "parameters")
        for (k, v) in fixed
            haskey(raw["parameters"], String(k)) || continue
            isapprox(Float64(raw["parameters"][String(k)]), v; rtol = 1e-9, atol = 1e-12) ||
                error("resume refused: the checkpoint holds $k = $(raw["parameters"][String(k)]) but --fix says $v")
        end
    end
    if !haskey(raw, "config") || !(raw["config"] isa AbstractDict)
        push!(notes, "UNVERIFIED on resume: legacy checkpoint without a [config] block -- verified the box and the fixed coordinates only; inputs, objective, seed and numerical settings are those of the relaunched command by construction, not by record")
        return "partial (legacy checkpoint)", notes
    end
    c = raw["config"]; mism = String[]
    # the objective's CONTENT identity (2026-09-28): verified whenever the run supplies it. A checkpoint
    # written before then has none -- its targets were checked by path, which says nothing about content.
    oid = String(get(now, "objective_id", ""))
    if !isempty(oid)
        saved_oid = String(get(c, "objective_id", ""))
        if isempty(saved_oid)
            push!(notes, "resume: the checkpoint predates the 2026-09-28 objective identity; its target files were verified " *
                         "by PATH only, not by content (a point is re-evaluated on a local-mode resume)")
        elseif saved_oid != oid
            push!(mism, "objective_id: checkpoint $(repr(saved_oid)) vs run $(repr(oid)) (target contents, extra-row " *
                        "weights, SMM_* switches, model or adapter source: tools/reopt_identity.jl)")
        end
    end
    for k in CHECKPOINT_CONFIG_KEYS
        haskey(c, k) || haskey(CHECKPOINT_CONFIG_DEFAULTS, k) || (push!(mism, "$k: absent from the checkpoint"); continue)
        a = get(c, k, get(CHECKPOINT_CONFIG_DEFAULTS, k, nothing)); b = now[k]
        same = if k == "fixed"
            Set(keys(a)) == Set(keys(b)) && all(isapprox(Float64(a[n]), Float64(b[n]); rtol = 1e-12, atol = 1e-15) for n in keys(a))
        elseif k == "extra_moments"
            String.(a) == String.(b)
        elseif a isa Real && b isa Real
            isapprox(Float64(a), Float64(b); rtol = 1e-12, atol = 0.0)
        else
            String(a) == String(b)
        end
        same || push!(mism, "$k: checkpoint $(repr(a)) vs run $(repr(b))")
    end
    # two keys are reported, not refused: the tools hash names the loader/driver version (a change there
    # does not change Q), and local_maxeval is the budget (not an input, objective, seed or numerical
    # setting; the driver relaunches the identical command, and an explicit budget extension is logged here)
    soft = filter(m -> startswith(m, "tools_sha16") || startswith(m, "local_maxeval"), mism)
    hard = filter(m -> !(startswith(m, "tools_sha16") || startswith(m, "local_maxeval")), mism)
    isempty(hard) || error("resume refused: the checkpoint was written under another configuration: " * join(mism, "; "))
    for m in soft
        push!(notes, "resume: " * (startswith(m, "tools") ? "tools changed since the checkpoint (" : "budget differs from the checkpoint (") * m * "); model source, inputs, objective, seed and numerics match")
    end
    return "verified", notes
end
