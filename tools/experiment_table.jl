# =============================================================================
# experiment_table.jl -- one evaluation, one comparison row (ported on 2026-10-02 from
# apps/Structural-estimation-v2, where it carries the v5e audit's specification-specific columns)
#
# Included by tools/reopt.jl. `case_record(label, point, targets; ...)` evaluates a named parameter
# point under optional overrides (child_extra / parent_extra) and returns a Dict with Q, the
# contribution of every targeted row (t-statistic and share of Q), every targeted moment, the
# extra rows of an experiment kept apart (Q_extra), validity and runtime. It reads only
# SMM_MOMENTS and SMM_PARAMS, so it follows whatever moments and parameters moments.jl defines.
# `write_case` stores the record as TOML; `table_line` and `contributions_line` print it.
# =============================================================================
using Printf, TOML, Statistics

function case_record(label::String, point::Dict{Symbol,Float64}, targets;
                     child_extra::NamedTuple = (;), parent_extra::NamedTuple = (;),
                     child_grid = (Na = 30, Nk = 30, Nt = 5), Na::Int = 30, Nhc::Int = 30,
                     simN::Int = 2000, seed::Int = 1234,
                     extra::Vector{Tuple{Symbol,Float64,Float64}} = Tuple{Symbol,Float64,Float64}[])
    rec = Dict{String,Any}("label" => label,
                           "params" => Dict{String,Any}(String(k) => v for (k, v) in point),
                           "child_extra" => Dict{String,Any}(String(k) => string(v) for (k, v) in pairs(child_extra)),
                           "parent_extra" => Dict{String,Any}(String(k) => string(v) for (k, v) in pairs(parent_extra)),
                           "child_grid" => Dict("Na" => child_grid.Na, "Nk" => child_grid.Nk, "Nt" => child_grid.Nt),
                           "parent_grid" => Dict("Na" => Na, "Nhc" => Nhc), "simN" => simN, "seed" => seed)
    t0 = time()
    o = try
        evaluate_at(point, targets; Na = Na, Nk = 2, Nhc = Nhc, simN = simN, seed = seed, child_grid = child_grid,
                    child_extra = child_extra, parent_extra = parent_extra)
    catch err
        cause = _root_cause(err)
        rec["status"] = "exception"; rec["exception"] = first(sprint(showerror, cause), 300)
        rec["runtime_s"] = time() - t0
        return rec
    end
    rec["runtime_s"] = time() - t0
    m = o.moments
    rec["status"] = (o.nviol > 0 || o.nbad > 0) ? "invalid" : "valid"
    rec["nviol"] = o.nviol; rec["nbad"] = o.nbad
    rec["Q"] = sum(abs2, o.r)
    rec["t"] = Dict{String,Any}(String(k) => o.r[j] for (j, k) in enumerate(SMM_MOMENTS))
    rec["Qshare"] = Dict{String,Any}(String(k) => o.r[j]^2 / max(rec["Q"], 1e-300) for (j, k) in enumerate(SMM_MOMENTS))
    rec["moments"] = Dict{String,Any}(String(k) => Float64(getfield(m, Symbol(k))) for k in SMM_MOMENTS)
    # extra targeted rows of an experiment: their t-statistics and their own Q, kept apart from
    # Q over the targeted rows
    if !isempty(extra)
        rec["t_extra"] = Dict{String,Any}(String(k) => sqrt(w) * (getfield(m, k) - mhat) for (k, mhat, w) in extra)
        rec["Q_extra"] = sum(abs2, values(rec["t_extra"]))
    end
    return rec
end

function write_case(path::String, rec::Dict)
    open(path, "w") do io
        TOML.print(io, rec) do x; x isa Symbol ? String(x) : x; end
    end
end

const TABLE_HEADER = @sprintf("%-34s | %12s | %-7s | %s", "case", "Q", "status", "largest three rows of Q (t-statistic, share)")

function table_line(rec::Dict)
    haskey(rec, "Q") || return @sprintf("%-34s %s: %s", rec["label"], get(rec, "status", "?"), first(get(rec, "exception", ""), 80))
    top = sort(collect(keys(rec["Qshare"])); by = k -> -rec["Qshare"][k])[1:min(3, length(rec["Qshare"]))]
    @sprintf("%-34s | %12.4f | %-7s | %s", rec["label"], rec["Q"], rec["status"],
             join((@sprintf("%s %+.1f (%.0f%%)", k, rec["t"][k], 100 * rec["Qshare"][k]) for k in top), "  "))
end

function contributions_line(rec::Dict)
    haskey(rec, "t") || return ""
    parts = [@sprintf("%s %+.1f(%.0f%%)", k, rec["t"][k], 100 * rec["Qshare"][k]) for k in String.(SMM_MOMENTS)]
    return "    t-stats: " * join(parts, "  ")
end
