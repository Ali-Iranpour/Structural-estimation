# =============================================================================
# pretest.jl -- the global (pre-testing) stage: evaluate a Sobol' design, keep every value
# in a recoverable cache, count what came back, and select the seeds.
#
# RECOVERABLE (plan step 5). With a cache path the values are written, checksummed and
# atomically, after every `chunk` completed evaluations: a crash costs at most the
# unfinished chunk, and a resumed stage evaluates only what is missing. A penalised value
# is an evaluated (invalid) point and is kept; a coding error is not a value and is never
# stored as one. The cache is keyed by the OBJECTIVE and the CANDIDATE DESIGN, not by the
# optimizer settings, so its values can serve another optimizer experiment on the same
# objective -- a declared new search, never a pretended continuation.
#
# DETERMINISTIC. The pool is defined by values in candidate-index order, never by which
# evaluation finished first: fixed-attempt = draws 1..N; valid-target = the shortest prefix
# 1..i* holding the target number of valid draws (draws evaluated past i* are "overshoot",
# kept in the cache and excluded from the pool); a time cap = the longest completed prefix.
# Ties in the ranking keep candidate order (stable sort).
# =============================================================================

"Is `v` a usable pre-testing value? Finite and strictly below the invalid-value threshold."
@inline valid_value(v::Float64, invalid_value::Float64) = isfinite(v) && v < invalid_value

"""
    select_seeds(fs, cfg, nstar_requested) -> (valid_order, nstar_effective)

`valid_order` holds the candidate indices of the VALID values, best first. Ties keep
candidate order (the sort is stable), so the selection is a function of the values alone.
The seeds, the first incumbent and its value are all read from this ONE array; before
2026-09-27 the incumbent's value came from the unfiltered sort, so a `-Inf` candidate
(excluded from the seeds) became an incumbent value no finite point could beat (finding 9).

With fewer valid values than requested restarts, `allow_fewer_restarts` reduces the
restarts (warned, and visible as `nstar_effective`); otherwise the pre-testing gate fails.
"""
function select_seeds(fs::Vector{Float64}, cfg::TikTakConfig, nstar_requested::Int; n_errors::Int = 0)
    order = sortperm(fs)                                  # stable: ties in candidate order
    valid_order = Int[k for k in order if valid_value(fs[k], cfg.invalid_value)]
    n_valid = length(valid_order)
    n_valid >= 1 || error(
        "every one of the $(length(fs)) pre-testing points failed to evaluate or was invalid" *
        (n_errors > 0 ? " ($n_errors threw and were discarded)" : "") *
        ". There is nothing to seed the local stage with.")
    nstar = nstar_requested
    if n_valid < nstar
        cfg.allow_fewer_restarts || throw(PretestGateError(
            "only $n_valid valid pre-testing values for $nstar requested restarts, and fewer restarts " *
            "were not allowed (allow_fewer_restarts = false). Enlarge the pre-testing design or " *
            "allow fewer restarts explicitly."))
        @warn "fewer VALID pre-testing values than restarts: reducing the restarts" n_valid Nstar = nstar
        nstar = n_valid
    end
    return valid_order, nstar
end

"""
    evaluate_candidates(f, cands, cfg; parallel, map_fn, on_each) -> (fs, errored)

Evaluate a batch of candidates: on threads (`parallel`), through a caller-supplied map
(`pmap` over worker processes), or serially. Under `on_error = :discard` a throw scores `Inf`
and is flagged in `errored` -- also when it happened on a worker process, which a counter in
this process's closure could not see. Under `:rethrow` (the default) `f` is called bare and a
coding error stops the run. `on_each(i, v)` runs after each value (serially: as it arrives).
"""
function evaluate_candidates(f::F, cands::Vector{Vector{Float64}}, cfg::TikTakConfig;
                             parallel::Bool = false, map_fn = map, on_each = (i, v) -> nothing) where {F}
    n = length(cands)
    fs = Vector{Float64}(undef, n)
    errored = falses(n)
    discard = cfg.on_error === :discard
    if parallel
        Threads.@threads for i in eachindex(cands)
            v, e = eval_one(f, cands[i], cfg); fs[i] = v; errored[i] = e
        end
        for i in 1:n; on_each(i, fs[i]); end
    elseif map_fn !== map
        # Distributed (or any user-supplied map): one batch, so on_each fires afterwards --
        # a worker process cannot write into this process's closure.
        xs = [copy(x) for x in cands]         # the objective never holds a stored candidate (J3)
        out = discard ? map_fn(x -> (try (Float64(f(x)), false) catch; (Inf, true) end), xs) :
                        map_fn(f, xs)
        for (i, o) in enumerate(out)
            fs[i], errored[i] = discard ? (Float64(o[1]), o[2]) : (Float64(o), false)
            on_each(i, fs[i])
        end
    else
        for (i, x) in pairs(cands)
            fs[i], errored[i] = eval_one(f, x, cfg)
            on_each(i, fs[i])
        end
    end
    return fs, errored
end

# -----------------------------------------------------------------------------
# The pre-testing stage and its cache
# -----------------------------------------------------------------------------
const CACHE_SCHEMA = 1
const CACHE_KIND = "tiktak_pretest_cache"

"""
    Pretest

The pre-testing stage in progress: the Sobol' generator and the points drawn so far, one
value / done / errored entry per candidate index, the same for the supplied points, and
how many values came from a cache instead of being evaluated here.
"""
mutable struct Pretest
    lo::Vector{Float64}
    hi::Vector{Float64}
    seq::Sobol.AbstractSobolSeq          # not hot: a draw per candidate; not parameterized on the dimension (J1)
    pts::Vector{Vector{Float64}}
    values::Vector{Float64}
    done::BitVector
    errored::BitVector
    supplied::Vector{Vector{Float64}}
    svalues::Vector{Float64}
    sdone::BitVector
    serrored::BitVector
    reused::Int                 # values loaded from a cache (not evaluated in this process)
    imported::Int               # of those, values from ANOTHER run's cache
    evals_segment::Int          # evaluated in this process
    lost::Int                   # evaluations dispatched whose value never arrived (a lost worker, an error,
                                # still running at a stop): their work is unknown (2026-09-28, follow-up 3)
    objective_id::String
    objective_fields::Dict{String,Any}
    design_id::String
    design::Dict{String,Any}
end

function Pretest(lo, hi, skip_first::Bool, supplied, objective_id::String,
                 objective_fields::Dict{String,Any} = Dict{String,Any}())
    seq = Sobol.SobolSeq(lo, hi)
    skip_first && Sobol.next!(seq)
    design = design_fields(lo, hi, skip_first, supplied)
    m = length(supplied)
    return Pretest(copy(lo), copy(hi), seq, Vector{Float64}[], Float64[], falses(0), falses(0),
                   supplied, fill(NaN, m), falses(m), falses(m), 0, 0, 0, 0, objective_id, objective_fields,
                   fields_id(design), design)
end

"The Sobol' candidate with index k (points are drawn in sequence and kept)."
function point!(pt::Pretest, k::Int)
    while length(pt.pts) < k
        push!(pt.pts, copy(Sobol.next!(pt.seq)))
    end
    return pt.pts[k]
end

function ensure_length!(pt::Pretest, k::Int)
    n = length(pt.values)
    if k > n
        append!(pt.values, fill(NaN, k - n)); append!(pt.done, falses(k - n)); append!(pt.errored, falses(k - n))
    end
    return pt
end

function write_cache(path::AbstractString, pt::Pretest)
    d = Dict{String,Any}("schema_version" => CACHE_SCHEMA, "kind" => CACHE_KIND,
        "objective_id" => pt.objective_id, "objective_fields" => pt.objective_fields,
        "design_id" => pt.design_id, "design" => pt.design,
        "values" => pt.values, "done" => collect(pt.done), "errored" => collect(pt.errored),
        "supplied_values" => pt.svalues, "supplied_done" => collect(pt.sdone),
        "supplied_errored" => collect(pt.serrored), "imported" => pt.imported, "lost" => pt.lost,
        "n_done" => count(pt.done), "updated" => _now())
    io = IOBuffer()
    println(io, "# TikTak pre-testing cache, schema $CACHE_SCHEMA (code/src/TikTak/pretest.jl). One value per")
    println(io, "# Sobol' candidate index (nan = not evaluated). Keyed by the objective and the design.")
    TOML.print(io, d; sorted = true)
    write_checksummed(path, String(take!(io)))
    return nothing
end

"""
    load_cache!(pt, path; imported = false) -> Int

Take the evaluated values of a cache file into `pt`, refusing a cache of another objective
or another candidate design. Returns the number of values taken. A value already present
in `pt` is kept (it cannot differ on the same objective and design).
"""
function load_cache!(pt::Pretest, path::AbstractString; imported::Bool = false)
    body, which = read_checksummed(path)
    which === :current || @warn "TikTak: the newest pre-testing cache generation did not verify; using the $which one" path
    d = TOML.parse(body)
    (Int(get(d, "schema_version", -1)) == CACHE_SCHEMA && get(d, "kind", "") == CACHE_KIND) ||
        throw(ResumeRefused("$path is not a TikTak pre-testing cache of schema $CACHE_SCHEMA"))
    String(d["objective_id"]) == pt.objective_id || throw(ResumeRefused(
        "the pre-testing cache $path belongs to another objective (saved $(d["objective_id"]), now $(pt.objective_id)):\n  " *
        join(identity_differences(get(d, "objective_fields", Dict{String,Any}()), pt.objective_fields), "\n  ")))
    String(d["design_id"]) == pt.design_id || throw(ResumeRefused(
        "the pre-testing cache $path belongs to another candidate design:\n  " *
        join(identity_differences(d["design"], pt.design), "\n  ")))
    vals = _fvec(d["values"]); dn = Bool.(d["done"]); er = Bool.(d["errored"])
    n = 0
    for k in eachindex(vals)
        dn[k] || continue
        ensure_length!(pt, k)
        pt.done[k] && continue
        pt.values[k] = vals[k]; pt.done[k] = true; pt.errored[k] = er[k]; n += 1
    end
    sv = _fvec(d["supplied_values"]); sd = Bool.(d["supplied_done"]); se = Bool.(d["supplied_errored"])
    for k in eachindex(sv)
        sd[k] && !pt.sdone[k] || continue
        pt.svalues[k] = sv[k]; pt.sdone[k] = true; pt.serrored[k] = se[k]; n += 1
    end
    pt.reused += n
    pt.imported += imported ? n : Int(get(d, "imported", 0))
    imported || (pt.lost += Int(get(d, "lost", 0)))       # this run's own unknown work carries over
    return n
end

"""
    PoolCursor

How far the pool is known to reach, scanned once, forward, in candidate-index order: draws
1..i are evaluated and `valid` of them are valid. Values never change once evaluated, so the
scan never goes back (O(N) over the stage).
"""
mutable struct PoolCursor
    i::Int
    valid::Int
end

"""
    advance!(c, pt, cfg) -> (i_end, status)

Move the cursor over evaluated draws:
  fixed-attempt:  (N, :complete) once 1..N are evaluated, else (i, :need_more)
  valid-target:   (i*, :reached) at the shortest evaluated prefix with the target number of
                  valid draws; (i, :need_more) when a missing draw comes first; (N, :cap) when
                  the attempt cap N is evaluated without reaching the target
"""
function advance!(c::PoolCursor, pt::Pretest, cfg::TikTakConfig)
    target = cfg.n_valid_target
    target > 0 && c.valid >= target && return (c.i, :reached)
    while c.i < cfg.n_sobol
        k = c.i + 1
        (k <= length(pt.done) && pt.done[k]) || return (c.i, :need_more)
        c.i = k
        if target > 0 && valid_value(pt.values[k], cfg.invalid_value)
            c.valid += 1
            c.valid >= target && return (c.i, :reached)
        end
    end
    return (c.i, target > 0 ? :cap : :complete)
end

"""
One guarded evaluation: under `on_error = :discard` a throw scores Inf and is flagged. The
objective receives a COPY of the candidate: a stored candidate (a future seed) is never the
objective's to modify (plan J3).
"""
eval_one(f::F, x, cfg::TikTakConfig) where {F} =
    cfg.on_error === :discard ? (try (Float64(f(copy(x))), false) catch; (Inf, true) end) : (Float64(f(copy(x))), false)

"""
    run_pretest!(pt, f, cfg; parallel, map_fn, chunk, cache_path, time_cap, on_sobol) -> (i_end, stop_reason)

Evaluate what the pool still needs: supplied points first, then Sobol' draws in index order.
Serially the draws go one at a time (so a valid-draw target is met with no overshoot);
through a batch map (`pmap`) or threads, `chunk` at a time. The cache is written after every
`chunk` completed values, at the end, and before an error propagates, so a crash loses at
most the unfinished chunk. A `time_cap` (seconds) stops the stage at the longest completed
prefix -- that pool depends on timing, and the stop reason says so.

`on_sobol(i, n, fx, best)` fires after each value with i < n, and once at the end with i == n
(the stage-end signal callers rely on).
"""
function run_pretest!(pt::Pretest, f::F, cfg::TikTakConfig; parallel::Bool = false, map_fn = map,
                      chunk::Int = 32, cache_path::String = "", time_cap::Float64 = Inf,
                      on_sobol = (i, n, fx, best) -> nothing) where {F}
    chunk >= 1 || throw(ArgumentError("TikTak: pretest_chunk must be >= 1"))
    t0 = time()
    planned = cfg.n_sobol + length(pt.supplied)
    serial = map_fn === map && !parallel
    best = Ref(Inf)
    for v in Iterators.flatten((pt.values[pt.done], pt.svalues[pt.sdone]))
        valid_value(v, cfg.invalid_value) && v < best[] && (best[] = v)
    end
    ndone = Ref(count(pt.done) + count(pt.sdone))
    since_save = Ref(0)
    save!() = (isempty(cache_path) || write_cache(cache_path, pt); since_save[] = 0)
    function store!(k::Int, supplied::Bool, v::Float64, err::Bool)
        if supplied
            pt.svalues[k] = v; pt.sdone[k] = true; pt.serrored[k] = err
        else
            ensure_length!(pt, k); pt.values[k] = v; pt.done[k] = true; pt.errored[k] = err
        end
        pt.evals_segment += 1; ndone[] += 1; since_save[] += 1
        valid_value(v, cfg.invalid_value) && v < best[] && (best[] = v)
        on_sobol(min(ndone[], planned - 1), planned, v, best[])
        since_save[] >= chunk && save!()
        return nothing
    end
    function batch!(ids::Vector{Int}, supplied::Bool)
        xs = supplied ? pt.supplied[ids] : [point!(pt, k) for k in ids]
        n0 = pt.evals_segment
        try
            if serial
                for (i, k) in enumerate(ids)
                    v, e = eval_one(f, xs[i], cfg)
                    store!(k, supplied, v, e)
                end
            else
                fs, er = evaluate_candidates(f, xs, cfg; parallel = parallel, map_fn = map_fn)
                for (i, k) in enumerate(ids); store!(k, supplied, fs[i], er[i]); end
            end
        catch
            # the evaluation that threw (serially) or the whole unfinished batch (a map over
            # workers, whose values never came back) ran without leaving a value: unknown work
            pt.lost += serial ? 1 : length(ids) - (pt.evals_segment - n0)
            save!()                                  # keep what was completed, then stop
            rethrow()
        end
        return nothing
    end

    todo = findall(!, pt.sdone)                      # supplied points: few, always in the pool
    isempty(todo) || batch!(todo, true)
    cur = PoolCursor(0, 0)
    stop_reason = :planned
    while true
        i_end, status = advance!(cur, pt, cfg)
        if status !== :need_more
            stop_reason = status === :reached ? :valid_target : status === :cap ? :attempt_cap : :planned
            break
        end
        if time() - t0 >= time_cap
            stop_reason = :time_cap
            break
        end
        width = serial && cfg.n_valid_target > 0 ? 1 : chunk
        ids = Int[]
        k = i_end + 1
        while length(ids) < width && k <= cfg.n_sobol
            (k <= length(pt.done) && pt.done[k]) || push!(ids, k)
            k += 1
        end
        batch!(ids, false)
    end
    save!()
    on_sobol(planned, planned, NaN, best[])          # the stage-end signal (i == n), exactly once
    return cur.i, stop_reason
end

"""
    pretest_summary(pt, i_end, stop_reason, cfg, valid_order, nstar) -> PretestSummary

Counts by kind over the POOL (Sobol' draws 1..i_end), the supplied points apart, and the
overshoot (draws evaluated beyond the pool).
"""
function pretest_summary(pt::Pretest, i_end::Int, stop_reason::Symbol, cfg::TikTakConfig,
                         valid_order::Vector{Int}, nstar::Int)
    iv = cfg.invalid_value
    pool = 1:i_end
    valid     = count(i -> !pt.errored[i] && valid_value(pt.values[i], iv), pool)
    invalid   = count(i -> !pt.errored[i] && isfinite(pt.values[i]) && pt.values[i] >= iv, pool)
    nonfinite = count(i -> !pt.errored[i] && !isfinite(pt.values[i]), pool)
    errors    = count(i -> pt.errored[i], pool)
    overshoot = count(i -> pt.done[i], i_end+1:length(pt.done))
    sel = view(valid_order, 1:nstar)
    target = cfg.n_valid_target
    return PretestSummary(i_end, valid, invalid, nonfinite, errors,
                          length(pt.supplied), count(v -> valid_value(v, iv), pt.svalues[pt.sdone]),
                          nstar, count(>(i_end), sel), pt.reused, overshoot, target,
                          target == 0 || valid >= target, stop_reason)
end
