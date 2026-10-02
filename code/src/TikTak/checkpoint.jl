# =============================================================================
# checkpoint.jl -- the versioned checkpoint of a run (plan step 4, J6).
#
# An EXPLICIT schema, written as TOML and converted field by field: never a Julia
# serialization of live tasks, channels, closures or NLopt objects, so a checkpoint stays
# readable, and the runtime types may change without breaking old files silently (a file of
# another schema version is refused by name).
#
# Crash safety. A checkpoint is written to <path>.tmp, flushed and fsync'ed, then the
# current file is renamed to <path>.prev and the new one renamed into place. The last line
# is a sha256 of everything above it. Reading takes the newest generation whose checksum
# verifies -- <path>, else <path>.tmp (a crash between the two renames), else <path>.prev
# -- and says which. A truncated or edited file therefore fails its checksum and is never
# half-read; with no valid generation the load fails with the reason.
#
# SCHEMA 2 (2026-09-28, follow-ups 1 and 4) adds the settings epochs (every record, in-flight
# job and the polish name theirs), the accounted attempts, `legacy_through` and the counters
# of unknown and duplicate work. A schema-1 file is MIGRATED on reading, explicitly: one
# epoch holding the saved optimizer settings, the committed attempts rebuilt from the
# records, and a `schema_migration` event. Nothing else is inferred.
# =============================================================================

const STATE_SCHEMA = 2
const STATE_SCHEMAS_READ = (1, 2)
const STATE_KIND = "tiktak_state"

_str(s::Symbol) = String(s)
_fvec(v) = Float64[Float64(x) for x in v]
_fvecs(v) = Vector{Float64}[_fvec(x) for x in v]

origin_dict(o::CandidateOrigin) = Dict{String,Any}("stage" => _str(o.stage), "restart" => o.restart,
    "attempt" => o.attempt, "candidate" => o.candidate, "ret" => _str(o.ret), "objective_id" => o.objective_id)
origin_from(d) = CandidateOrigin(Symbol(d["stage"]), Int(d["restart"]), Int(d["attempt"]), Int(d["candidate"]),
                                 Symbol(d["ret"]), String(d["objective_id"]))

verification_dict(v::Verification) = Dict{String,Any}("status" => _str(v.status), "stage" => _str(v.stage),
    "restart" => v.restart, "attempt" => v.attempt, "ret" => _str(v.ret), "x_tested" => v.x_tested,
    "f_tested" => v.f_tested, "distance" => v.distance, "coord_tol" => v.coord_tol,
    "objective_id" => v.objective_id, "solver" => v.solver)
verification_from(d) = Symbol(d["status"]) === :none ? NO_VERIFICATION :
    Verification(Symbol(d["status"]), Symbol(d["stage"]), Int(d["restart"]), Int(d["attempt"]), Symbol(d["ret"]),
                 _fvec(d["x_tested"]), Float64(d["f_tested"]), Float64(d["distance"]), Float64(d["coord_tol"]),
                 String(d["objective_id"]), String(d["solver"]))

record_dict(r::RestartRecord) = Dict{String,Any}("j" => r.j, "attempt" => r.attempt, "theta" => r.theta,
    "x0" => r.x0, "incumbent_version" => r.incumbent_version, "f_start" => r.f_start, "x" => r.x,
    "f_local" => r.f_local, "n_eval" => r.n_eval, "ret" => _str(r.ret), "action" => _str(r.action),
    "version_after" => r.version_after, "worker" => r.worker, "elapsed" => r.elapsed,
    "commit_seq" => r.commit_seq, "dispatch_seq" => r.dispatch_seq, "commits_at_dispatch" => r.commits_at_dispatch,
    "error" => r.error, "legacy" => r.legacy, "epoch" => r.epoch)
record_from(d) = RestartRecord(Int(d["j"]), Int(d["attempt"]), Float64(d["theta"]), _fvec(d["x0"]),
    Int(d["incumbent_version"]), Float64(d["f_start"]), _fvec(d["x"]), Float64(d["f_local"]), Int(d["n_eval"]),
    Symbol(d["ret"]), Symbol(d["action"]), Int(d["version_after"]), Int(d["worker"]), Float64(d["elapsed"]),
    Int(d["commit_seq"]), Int(d["dispatch_seq"]), Int(d["commits_at_dispatch"]), String(d["error"]), Bool(d["legacy"]),
    Int(get(d, "epoch", 1)))                        # schema 1: every record ran under the one epoch

"Solver settings as identity/checkpoint fields, and back."
solver_fields(s::SolverSettings) = Dict{String,Any}("alg" => String(s.alg), "ftol_rel" => s.ftol_rel,
    "ftol_abs" => s.ftol_abs, "xtol_rel" => s.xtol_rel, "maxeval" => s.maxeval, "initial_step" => s.initial_step,
    "step_schedule" => String(s.step_schedule), "step_min" => s.step_min)
solver_from_fields(d) = SolverSettings(Symbol(d["alg"]), Float64(d["ftol_rel"]), Float64(d["ftol_abs"]),
    Float64(d["xtol_rel"]), Int(d["maxeval"]), Float64(get(d, "initial_step", 0.0)),
    Symbol(get(d, "step_schedule", "fixed")), Float64(get(d, "step_min", 0.0)))

epoch_dict(e::SettingsEpoch) = Dict{String,Any}("index" => e.index, "local" => solver_fields(e.local_),
    "polish" => solver_fields(e.polish), "normalize" => e.normalize, "skip_polish" => e.skip_polish,
    "optimizer_id" => e.optimizer_id, "from_segment" => e.from_segment, "note" => e.note)
epoch_from(d) = SettingsEpoch(Int(d["index"]), solver_from_fields(d["local"]), solver_from_fields(d["polish"]),
    Bool(d["normalize"]), Bool(d["skip_polish"]), String(d["optimizer_id"]), Int(d["from_segment"]), String(d["note"]))
"Schema 1 had no epochs: the one epoch is the saved optimizer identity's settings."
epoch_from_optimizer(fields, opt_id) = SettingsEpoch(1, solver_from_fields(fields["local"]), solver_from_fields(fields["polish"]),
    Bool(get(fields, "normalize", false)), Bool(get(fields, "skip_polish", false)), String(opt_id), 1,
    "migrated from schema 1: the settings of the saved optimizer identity")

"An in-flight job: the dispatched job's own fields; its settings are those of its epoch."
inflight_dict(fl::InFlight) = Dict{String,Any}("j" => fl.job.j, "attempt" => fl.job.attempt, "stage" => _str(fl.job.stage),
    "theta" => fl.job.theta, "x0" => fl.job.x0, "incumbent_version" => fl.job.incumbent_version,
    "worker" => fl.worker, "dispatched" => fl.dispatched, "dispatch_seq" => fl.dispatch_seq,
    "commits_at_dispatch" => fl.commits_at_dispatch, "step_scale" => fl.job.step_scale, "epoch" => fl.job.epoch)
function inflight_from(d, st_run_id::String, cfg::TikTakConfig, epochs::Vector{SettingsEpoch}, lo, hi)
    ep = Int(get(d, "epoch", 1))
    1 <= ep <= length(epochs) || throw(ResumeRefused("an in-flight job names settings epoch $ep; the checkpoint has $(length(epochs))"))
    e = epochs[ep]
    job = RestartJob(st_run_id, Symbol(d["stage"]), Int(d["j"]), Int(d["attempt"]), Float64(d["theta"]), _fvec(d["x0"]),
                     Int(d["incumbent_version"]), true, NaN, e.local_, lo, hi, cfg.on_error, :default, 1, Inf,
                     Float64(get(d, "step_scale", 1.0)), e.normalize, ep)
    return InFlight(job, Int(d["worker"]), Float64(d["dispatched"]), Int(d["dispatch_seq"]), Int(d["commits_at_dispatch"]))
end

attempt_dict(k::AttemptKey, how::Symbol) = Dict{String,Any}("run_id" => k.run_id, "stage" => _str(k.stage), "j" => k.j,
                                                            "attempt" => k.attempt, "how" => _str(how))

event_dict(e::RunEvent) = Dict{String,Any}("seq" => e.seq, "time" => e.time, "kind" => _str(e.kind), "detail" => e.detail)
event_from(d) = RunEvent(Int(d["seq"]), String(d["time"]), Symbol(d["kind"]), String(d["detail"]))
segment_dict(s::Segment) = Dict{String,Any}("index" => s.index, "started" => s.started, "mode" => _str(s.mode),
    "workers" => s.workers, "stop_after" => s.stop_after, "next_j" => s.next_j, "note" => s.note)
segment_from(d) = Segment(Int(d["index"]), String(d["started"]), Symbol(d["mode"]), Int(d["workers"]),
                          Int(d["stop_after"]), Int(d["next_j"]), String(d["note"]))

pretest_dict(p::PretestSummary) = Dict{String,Any}(String(k) => (v isa Symbol ? _str(v) : v)
                                                   for (k, v) in zip(fieldnames(PretestSummary), ntuple(i -> getfield(p, i), fieldcount(PretestSummary))))
pretest_from(d) = PretestSummary((T === Symbol ? Symbol(d[String(k)]) : T(d[String(k)])
                                  for (k, T) in zip(fieldnames(PretestSummary), fieldtypes(PretestSummary)))...)

"The checkpoint of `st` as a Dict of TOML values (the explicit schema, version STATE_SCHEMA)."
function state_to_dict(st::RunState)
    return Dict{String,Any}(
        "schema_version" => STATE_SCHEMA, "kind" => STATE_KIND, "tiktak_version" => TIKTAK_VERSION,
        "run_id" => st.run_id, "checkpoint_seq" => st.checkpoint_seq, "written" => _now(),
        "stage" => _str(st.stage), "status" => _str(st.status), "resume_semantics" => _str(st.resume_semantics),
        "purpose" => st.purpose,
        "objective" => Dict{String,Any}("id" => st.objective_id, "fields" => st.objective_fields),
        "optimizer" => Dict{String,Any}("id" => st.optimizer_id, "fields" => st.optimizer_fields),
        "plan" => Dict{String,Any}("nstar_requested" => st.nstar_requested, "nstar_effective" => st.K,
                                   "schedule_denominator" => st.K, "lo" => st.lo, "hi" => st.hi),
        "pretest" => pretest_dict(st.pretest),
        "seeds" => Dict{String,Any}("x" => st.seeds, "f" => st.seed_f, "origin" => _str.(st.seed_origin),
                                    "candidate" => st.seed_candidate, "f_sobol_best" => st.f_sobol_best),
        "incumbent" => Dict{String,Any}("x" => st.inc.x, "f" => st.inc.f, "version" => st.inc.version,
                                        "origin" => origin_dict(st.inc.origin),
                                        "verification" => verification_dict(st.inc.verification)),
        "progress" => Dict{String,Any}("next_j" => st.next_j, "commit_seq" => st.commit_seq, "dispatch_seq" => st.dispatch_seq,
                                       "fZ_prev_distinct" => st.fZ_prev_distinct, "stopped_early" => st.stopped_early,
                                       "n_exception" => st.n_exception, "f_prepolish" => st.f_prepolish,
                                       "legacy_through" => st.legacy_through),
        "counters" => Dict{String,Any}("evals_pretest" => st.evals_pretest, "evals_local" => st.evals_local,
                                       "evals_polish" => st.evals_polish, "evals_abandoned" => st.evals_abandoned,
                                       "jobs_lost" => st.jobs_lost, "attempts_unknown" => st.attempts_unknown,
                                       "pretest_lost" => st.pretest_lost, "duplicates_ignored" => st.duplicates,
                                       "retries" => st.retries, "counts_complete" => st.counts_complete),
        "polish" => Dict{String,Any}("done" => st.polish.done, "ret" => _str(st.polish.ret),
                                     "improved" => st.polish.improved, "n_eval" => st.polish.n_eval,
                                     "epoch" => st.polish.epoch),
        "epochs" => Dict{String,Any}[epoch_dict(e) for e in st.epochs],
        "accounted" => Dict{String,Any}[attempt_dict(k, h) for (k, h) in
                                        sort(collect(st.accounted); by = p -> (p[1].stage, p[1].j, p[1].attempt))],
        "records" => Dict{String,Any}[record_dict(r) for r in st.records],
        "inflight" => Dict{String,Any}[inflight_dict(fl) for fl in sort(collect(values(st.inflight)); by = fl -> fl.job.j)],
        "segments" => Dict{String,Any}[segment_dict(s) for s in st.segments],
        "events" => Dict{String,Any}[event_dict(e) for e in st.events])
end

"""
    check_schema(d) -> Int

The schema version of a state Dict, refused by name unless this code reads it.
"""
function check_schema(d::AbstractDict)
    sv = get(d, "schema_version", nothing)
    (sv isa Integer && Int(sv) in STATE_SCHEMAS_READ && get(d, "kind", "") == STATE_KIND) ||
        throw(ResumeRefused("the checkpoint is not a TikTak state of a schema this code reads " *
                            "($(join(STATE_SCHEMAS_READ, ", "))): kind $(repr(get(d, "kind", nothing))), schema $(repr(sv))"))
    return Int(sv)
end

"The restarts a schema-1 legacy import covered: the `at restart J` of its import event, else its legacy records."
function _legacy_through_schema1(d)
    for e in get(d, "events", Any[])
        String(e["kind"]) == "legacy_import" || continue
        m = match(r"at restart (\d+) of", String(e["detail"]))
        m === nothing || return parse(Int, m.captures[1]) - 1
    end
    js = [Int(r["j"]) for r in d["records"] if Bool(r["legacy"])]
    return isempty(js) ? 0 : maximum(js)
end

"""
    state_from_dict(d, cfg) -> RunState

The explicit conversion back. `cfg` is the CURRENT configuration, which the caller has
already verified against the saved optimizer identity. A schema-1 Dict is migrated (see the
file header) and says so in its events.
"""
function state_from_dict(d::AbstractDict, cfg::TikTakConfig)
    schema = check_schema(d)
    p = d["plan"]; sd = d["seeds"]; ic = d["incumbent"]; pg = d["progress"]; c = d["counters"]; pl = d["polish"]
    inc = Incumbent(_fvec(ic["x"]), Float64(ic["f"]), Int(ic["version"]), origin_from(ic["origin"]),
                    verification_from(ic["verification"]))
    run_id = String(d["run_id"]); lo = _fvec(p["lo"]); hi = _fvec(p["hi"])
    epochs = schema >= 2 ? SettingsEpoch[epoch_from(e) for e in d["epochs"]] :
                           [epoch_from_optimizer(d["optimizer"]["fields"], d["optimizer"]["id"])]
    inflight = Dict{Int,InFlight}()
    for x in get(d, "inflight", Any[])
        fl = inflight_from(x, run_id, cfg, epochs, lo, hi); inflight[fl.job.j] = fl
    end
    records = RestartRecord[record_from(r) for r in d["records"]]
    accounted = Dict{AttemptKey,Symbol}()
    if schema >= 2
        for a in d["accounted"]
            accounted[AttemptKey(String(a["run_id"]), Symbol(a["stage"]), Int(a["j"]), Int(a["attempt"]))] = Symbol(a["how"])
        end
    else
        for r in records
            accounted[AttemptKey(run_id, :local, r.j, r.attempt)] = r.legacy ? :legacy : :committed
        end
        Bool(pl["done"]) && (accounted[AttemptKey(run_id, :polish, 0, 1)] = :committed)
    end
    polish = PolishRecord(Bool(pl["done"]), Symbol(pl["ret"]), Bool(pl["improved"]), Int(pl["n_eval"]),
                          Int(get(pl, "epoch", Bool(pl["done"]) ? 1 : 0)))
    legacy_through = schema >= 2 ? Int(pg["legacy_through"]) : _legacy_through_schema1(d)
    st = RunState(run_id, lo, hi, cfg,
                  String(d["objective"]["id"]), Dict{String,Any}(d["objective"]["fields"]),
                  String(d["optimizer"]["id"]), Dict{String,Any}(d["optimizer"]["fields"]),
                  Int(p["nstar_requested"]), Int(p["schedule_denominator"]),
                  _fvecs(sd["x"]), _fvec(sd["f"]), Symbol.(sd["origin"]), Int[Int(x) for x in sd["candidate"]],
                  Float64(sd["f_sobol_best"]), pretest_from(d["pretest"]),
                  inc, records, inflight,
                  Int(pg["next_j"]), Int(pg["commit_seq"]), Int(get(pg, "dispatch_seq", 0)), Int(d["checkpoint_seq"]),
                  Float64(pg["fZ_prev_distinct"]), Bool(pg["stopped_early"]), Int(pg["n_exception"]),
                  polish, Float64(pg["f_prepolish"]),
                  Int(c["evals_pretest"]), Int(c["evals_local"]), Int(c["evals_polish"]), Int(c["evals_abandoned"]),
                  Int(get(c, "jobs_lost", 0)), Int(get(c, "attempts_unknown", 0)), Int(get(c, "pretest_lost", 0)),
                  Int(get(c, "duplicates_ignored", 0)), Int(get(c, "retries", 0)), 0, Bool(c["counts_complete"]),
                  Symbol(d["stage"]), Symbol(d["status"]), Symbol(d["resume_semantics"]), String(get(d, "purpose", "custom")),
                  epochs, accounted, legacy_through,
                  Segment[segment_from(s) for s in d["segments"]], RunEvent[event_from(e) for e in d["events"]])
    schema == 1 && add_event!(st, :schema_migration,
        "checkpoint schema 1 read by schema $STATE_SCHEMA code: one settings epoch from the saved optimizer identity; " *
        "committed attempts rebuilt from the $(length(records)) records; legacy_through = $legacy_through")
    return st
end

# ---- atomic, checksummed files -------------------------------------------------
const _CHECKSUM_TAG = "# sha256 "

function _fsync(io::IOStream)
    try
        ccall(:fsync, Cint, (Cint,), Base.cconvert(Cint, fd(io)))
    catch
    end
    return nothing
end

"""
    write_checksummed(path, text; keep_previous = true)

Write `text` plus a sha256 trailer to `path` atomically (tmp file, fsync, rename), moving
the current file to `path.prev` first when `keep_previous`.
"""
function write_checksummed(path::AbstractString, text::AbstractString; keep_previous::Bool = true)
    body = String(text)
    endswith(body, "\n") || (body *= "\n")
    full = body * _CHECKSUM_TAG * bytes2hex(SHA.sha256(body)) * "\n"
    tmp = path * ".tmp"
    open(tmp, "w") do io
        write(io, full); flush(io); _fsync(io)
    end
    keep_previous && isfile(path) && mv(path, path * ".prev"; force = true)
    mv(tmp, path; force = true)
    return nothing
end

"The body of a checksummed file if its trailer verifies, else `nothing`."
function verified_body(file::AbstractString)
    isfile(file) || return nothing
    text = read(file, String)
    i = findlast(_CHECKSUM_TAG, text)
    i === nothing && return nothing
    body = text[1:first(i)-1]
    given = strip(text[last(i)+1:end])
    return bytes2hex(SHA.sha256(body)) == given ? body : nothing
end

"""
    read_checksummed(path) -> (body, which)

The newest valid generation: `path`, else `path.tmp`, else `path.prev` (`which` names it).
Errors when none verifies, listing what was found.
"""
function read_checksummed(path::AbstractString)
    for (file, which) in ((path, :current), (path * ".tmp", :tmp), (path * ".prev", :previous))
        body = verified_body(file)
        body === nothing || return body, which
    end
    found = [f for f in (path, path * ".tmp", path * ".prev") if isfile(f)]
    error("no valid checkpoint generation at $path" * (isempty(found) ? " (no file)" :
          " (found $(join(basename.(found), ", ")), none with a valid sha256 trailer: truncated or edited)"))
end

checkpoint_exists(path::AbstractString) = any(isfile, (path, path * ".tmp", path * ".prev"))

"Write the run state (increments the checkpoint sequence number)."
function write_state(path::AbstractString, st::RunState)
    st.checkpoint_seq += 1
    io = IOBuffer()
    println(io, "# TikTak run state, schema $STATE_SCHEMA (code/src/TikTak/checkpoint.jl). Machine-written;")
    println(io, "# the sha256 on the last line covers everything above it. Do not edit by hand.")
    TOML.print(io, state_to_dict(st); sorted = true)
    write_checksummed(path, String(take!(io)))
    return nothing
end

"Read a run state file: (Dict, which generation)."
function read_state_dict(path::AbstractString)
    body, which = read_checksummed(path)
    return TOML.parse(body), which
end
