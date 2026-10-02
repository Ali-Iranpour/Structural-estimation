# =============================================================================
# reopt_identity.jl -- the OBJECTIVE IDENTITY of a tools/reopt.jl run
# (2026-09-28: tiktak_fix_plan.md follow-up 2, finding 17 of tiktak_problems.md)
#
# Until 2026-09-28 reopt.jl identified its objective by the target file's PATH and the
# extra moments' NAMES: new target means at the same path, or new standard errors for the same
# extra rows, changed Q without changing the identity, so a checkpoint or pre-testing cache of
# one objective could be continued under another -- old and new Q values are not comparable,
# and the monotone incumbent rule then means nothing.
#
# Pure functions: no model, no worker, no evaluation. reopt.jl builds the identity as soon
# as the run's constants are resolved and checks every checkpoint and cache against it BEFORE
# the child warm-up; tools/test_reopt_identity.jl tests them with temporary files and a counting
# objective. Everything that changes Q at a fixed search vector is in it:
#
#   targets_sha, extra_targets_sha   the target files BY CONTENT (paths are recorded elsewhere, not trusted)
#   moment_names                     the resolved order of the targeted moments
#   extra_names/_targets/_weights    every extra row as resolved: name, target mean, weight 1/se^2
#   free_names, free_links, lo, hi   the search coordinates: order, links, box (run bounds applied)
#   fixed, fixed_search, free_extra  held parameters (value and search coordinate); a freed omega's box
#   run_bounds                       the --bounds argument
#   seed, sim_n, grid, child_grid    common random numbers and the numerical problem
#   parent_extra, child_extra        the numerical and specification overrides, as resolved
#   spec_env                         the SMM_* switches moments.jl read from the environment at load
#   model_sha, adapter_sha           the model source and the objective-affecting adapter code
#                                    (tools/reopt_objective.jl, tools/run_bounds.jl), by content
#
# NOT in it: reopt.jl itself (orchestration and reporting -- it cannot change Q since the
# objective moved to tools/reopt_objective.jl), the optimizer settings (the TikTak module keeps its
# own optimizer identity), budgets. POLICY, conservative: any edit of a hashed file refuses a
# continuation, even a comment; an edit of reopt.jl alone never does.
# =============================================================================
using SHA, TOML

"Bumped when the identity's fields change; a checkpoint of another version is never continued as verified."
const REOPT_IDENTITY_VERSION = 2
const REOPT_MODEL_FILES = ("code/src/child_lifecycle.jl", "code/src/parent_family.jl", "code/smm/moments.jl")
const REOPT_ADAPTER_FILES = ("tools/reopt_objective.jl", "tools/run_bounds.jl")
"The environment switches moments.jl reads at load time; each changes the specification."
# This repository has no SMM_* specification switches (v2's are SMM_WORK_ROW, SMM_MU_FIXED, ...): the field stays
# in the identity and is always empty here. A switch added to moments.jl later must be listed in one of the two.
const REOPT_SPEC_ENV_KEYS = ()

"First 16 hex characters of the sha256 of a file's content."
file_sha16(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))[1:16]
"First 16 hex characters of the sha256 of several files' contents, concatenated in the given order."
files_sha16(repo::AbstractString, files) = bytes2hex(SHA.sha256(reduce(vcat, [read(joinpath(repo, f)) for f in files])))[1:16]

"The SMM_* switches as this process sees them (unset = \"\")."
# Switches recorded ONLY WHEN SET, so a run without them keeps its identity (v2: SMM_WEALTH_ROW). None here.
const REOPT_SPEC_ENV_KEYS_IF_SET = ()
spec_env(env = ENV) = merge(Dict{String,Any}(k => String(strip(get(env, k, ""))) for k in REOPT_SPEC_ENV_KEYS),
                            Dict{String,Any}(k => String(strip(env[k])) for k in REOPT_SPEC_ENV_KEYS_IF_SET
                                             if !isempty(strip(get(env, k, "")))))

"""
    extra_rows_from(names, cov_raw, mean_of) -> Vector{Tuple{Symbol,Float64,Float64}}

The extra targeted rows (experiment C) as reopt.jl uses them: for each name, the target
mean `mean_of(name)` and the weight 1/se^2 with se from the `[moment_cov]` block of the parsed
extra-targets file `cov_raw`. Pure: the caller supplies the parsed file and the means.
"""
function extra_rows_from(names, cov_raw::AbstractDict, mean_of)
    out = Tuple{Symbol,Float64,Float64}[]
    isempty(names) && return out
    mc = cov_raw["moment_cov"]
    fn = String.(mc["names"]); se = Float64.(mc["se"])
    for k in names
        i = findfirst(==(String(k)), fn)
        i === nothing && error("--extra-moments $k: no [moment_cov] row in the extra targets")
        push!(out, (Symbol(k), Float64(mean_of(String(k))), 1.0 / se[i]^2))
    end
    return out
end

"""
    reopt_objective_fields(; kw...) -> Dict{String,Any}

The objective identity of a reopt.jl run (see the file header for every field). Its hash,
`TikTak.fields_id(fields)`, is the objective id the TikTak module records with every point,
certificate, checkpoint and pre-testing cache.
"""
function reopt_objective_fields(; repo::AbstractString, targets_file::AbstractString, extra_targets_file::AbstractString,
                                extra_rows, moment_names, free_names, free_links, lo, hi,
                                fixed::AbstractDict, fixed_search::AbstractDict, free_extra::AbstractString,
                                run_bounds::AbstractString, seed::Integer, sim_n::Integer, grid::Integer,
                                child_grid::AbstractString, parent_extra::AbstractString, child_extra::AbstractString,
                                spec::AbstractDict = spec_env())
    fmt(x) = Float64(x)
    return Dict{String,Any}(
        "identity_version" => REOPT_IDENTITY_VERSION,
        "targets_sha"      => file_sha16(targets_file),
        "extra_targets_sha"=> isempty(extra_rows) ? "" : file_sha16(extra_targets_file),
        "moment_names"     => String[String(m) for m in moment_names],
        "extra_names"      => String[String(r[1]) for r in extra_rows],
        "extra_targets"    => Float64[fmt(r[2]) for r in extra_rows],
        "extra_weights"    => Float64[fmt(r[3]) for r in extra_rows],
        "free_names"       => String[String(n) for n in free_names],
        "free_links"       => String[String(l) for l in free_links],
        "lo"               => Float64[fmt(x) for x in lo],
        "hi"               => Float64[fmt(x) for x in hi],
        "fixed"            => Dict{String,Any}(String(k) => fmt(v) for (k, v) in fixed),
        "fixed_search"     => Dict{String,Any}(String(k) => fmt(v) for (k, v) in fixed_search),
        "free_extra"       => String(free_extra),
        "run_bounds"       => String(run_bounds),
        "seed"             => Int(seed), "sim_n" => Int(sim_n), "grid" => Int(grid),
        "child_grid"       => String(child_grid),
        "parent_extra"     => String(parent_extra), "child_extra" => String(child_extra),
        "spec_env"         => Dict{String,Any}(spec),
        "model_sha"        => files_sha16(repo, REOPT_MODEL_FILES),
        "adapter_sha"      => files_sha16(repo, REOPT_ADAPTER_FILES))
end

const _REOPT_FIELD_MEANING = Dict(
    "targets_sha" => "the targets file's CONTENT changed (at the same path or another)",
    "extra_targets_sha" => "the extra-targets file's content changed",
    "extra_weights" => "an extra row's weight 1/se^2 changed",
    "extra_targets" => "an extra row's target mean changed",
    "moment_names" => "the targeted moments or their order changed",
    "spec_env" => "an SMM_* specification switch in the environment changed",
    "model_sha" => "the model source changed ($(join(REOPT_MODEL_FILES, ", ")))",
    "adapter_sha" => "the objective adapter code changed ($(join(REOPT_ADAPTER_FILES, ", ")))",
    "identity_version" => "the checkpoint was written under another identity definition")

"One readable refusal line per changed field, with what the change means where that is not obvious."
function reopt_identity_differences(saved::AbstractDict, now::AbstractDict)
    lines = String[]
    for d in TikTak.identity_differences(saved, now)
        k = first(split(d, r"[.:]"; limit = 2))
        push!(lines, haskey(_REOPT_FIELD_MEANING, k) ? d * "   -- " * _REOPT_FIELD_MEANING[k] : d)
    end
    return lines
end

"""
    saved_objective(path, kind) -> (fields, id) or nothing

The objective identity recorded in a TikTak run state (`kind = :state`) or pre-testing cache
(`:cache`), read through the module's checksummed reader; `nothing` when there is no file.
"""
function saved_objective(path::AbstractString, kind::Symbol)
    TikTak.checkpoint_exists(path) || return nothing
    body, _ = TikTak.read_checksummed(path)
    d = TOML.parse(body)
    kind === :state && return (Dict{String,Any}(d["objective"]["fields"]), String(d["objective"]["id"]))
    return (Dict{String,Any}(get(d, "objective_fields", Dict{String,Any}())), String(d["objective_id"]))
end

"""
    check_reopt_resume(paths_kinds, now_fields) -> Vector{String}

Refuse -- with an error naming every changed field -- to continue a checkpoint or reuse a cache
whose objective identity is not `now_fields`. Runs before any model initialisation. A file
written under the incomplete pre-2026-09-28 identity (no `identity_version`) is refused as a
verified continuation: its values, certificates and pre-testing ranking cannot be shown to
belong to this objective. Returns notes for the log (one per verified file).
"""
function check_reopt_resume(paths_kinds, now_fields::AbstractDict)
    notes = String[]
    for (path, kind) in paths_kinds
        s = saved_objective(path, kind)
        s === nothing && continue
        saved, _ = s
        what = kind === :state ? "checkpoint" : "pre-testing cache"
        if get(saved, "identity_version", 0) != REOPT_IDENTITY_VERSION
            error("resume refused: the $what $path was written under the INCOMPLETE objective identity used before " *
                  "2026-09-28 (target file by path, extra-row weights absent), so its objective values, convergence " *
                  "evidence and pre-testing ranking cannot be shown to belong to the current objective. It is not " *
                  "continued. To build on it, start a NEW run in a new --outdir with --start <its best_estimates.toml> " *
                  "(parameters by name; the point is re-evaluated, nothing else carries over).")
        end
        diffs = reopt_identity_differences(saved, now_fields)
        isempty(diffs) || error("resume refused: the objective changed since the $what $path was written:\n  " *
                                join(diffs, "\n  ") * "\n  Start a new run (a new --outdir) for the new objective.")
        push!(notes, "$what $(basename(path)): objective identity verified field by field (identity version $REOPT_IDENTITY_VERSION)")
    end
    return notes
end
