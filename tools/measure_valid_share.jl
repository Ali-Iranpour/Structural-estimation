#!/usr/bin/env julia
# =============================================================================
# measure_valid_share.jl -- how much of the search box gives a valid model evaluation?
#
#     julia --project=. --threads=1 tools/measure_valid_share.jl <targets.toml> <outdir> [--n 400] [--procs 10] [--grid 30]
#
# Needs a target file with the data's composition tables (from 2026-10-02 19:32 on); without them
# load_targets refuses, and SMM_TEST_FIXTURES=1 would measure the stand-ins instead of the model.
#
# (2026-10-02, Ali: "measure the valid share, then decide the sigma_j boxes".) Evaluates the SMM
# objective, exactly as the search does, at the first N Sobol' points of the search box (the points
# TikTak's pre-testing would draw), at the production grids (parent Na = Nhc = --grid, child 30/30/5,
# simN 2000, seed 1234), on --procs worker processes. Records, per point, Q and the penalty reason
# (the SMM_PENALTY_LOG entry it incremented, or "valid").
#
# Writes <outdir>/valid_share.csv (one row per point: natural parameter values, Q, reason, seconds)
# and <outdir>/valid_share_summary.txt: the valid share, the reasons tallied, and for every parameter
# the valid share in the low / middle / high third of its box (in search coordinates), which is what
# a box decision needs. A measurement, not a decision: nothing here changes a box.
# =============================================================================
using Distributed, Printf, TOML, Sobol, Statistics, Dates

argstr(f, d) = (i = findfirst(==(f), ARGS); i === nothing ? d : ARGS[i + 1])
length(ARGS) >= 2 || error("usage: measure_valid_share.jl <targets.toml> <outdir> [--n N] [--procs P] [--grid G]")
const TFILE, OUT = abspath(ARGS[1]), abspath(ARGS[2])
const N = parse(Int, argstr("--n", "400")); const P = parse(Int, argstr("--procs", "10")); const G = parse(Int, argstr("--grid", "30"))
mkpath(OUT)
const REPO = normpath(joinpath(@__DIR__, ".."))
addprocs(P; exeflags = `--project=$REPO --threads=1 --startup-file=no`)
@everywhere begin
    using Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames, Statistics, Dates
    using ProgressMeter, Distributions, StatsBase, QuantEcon, FastGaussQuadrature, Parameters, Dierckx, TOML
    LinearAlgebra.BLAS.set_num_threads(1)
    const REPO_ = $REPO; const SRC = joinpath(REPO_, "code", "src")
    include(joinpath(SRC, "paths.jl")); include(joinpath(SRC, "manifest.jl")); include(joinpath(SRC, "diagnostics.jl"))
    include(joinpath(SRC, "child_lifecycle.jl")); include(joinpath(SRC, "parent_family.jl")); include(joinpath(SRC, "tiktak.jl"))
    include(joinpath(REPO_, "code", "smm", "moments.jl"))
    const T = load_targets($TFILE)
    const GRID = $G
    function eval_point(z)
        before = copy(SMM_PENALTY_LOG); t0 = time()
        q = smm_objective(z, T; Na = GRID, Nk = 2, Nhc = GRID, simN = 2000, seed = 1234,
                          child_grid = (Na = 30, Nk = 30, Nt = 5), demo_sim = false)
        grew = [k for (k, v) in SMM_PENALTY_LOG if v > get(before, k, 0)]
        return (q = q, reason = q < SMM_PENALTY ? "valid" : (isempty(grew) ? "penalty (no reason logged)" : String(first(grew))),
                secs = time() - t0)
    end
end
lo, hi = search_bounds()
# the same sequence TikTak's pre-testing draws (code/src/TikTak/pretest.jl: SobolSeq, first point skipped)
seq = Sobol.SobolSeq(lo, hi)
Sobol.next!(seq)
pts = [copy(Sobol.next!(seq)) for _ in 1:N]
println("valid-share measurement: $N Sobol' points, grid $G, $(nworkers()) workers, targets $TFILE, ", now()); flush(stdout)
t_all = time()
res = pmap(eval_point, pts)
names_ = [String(q.name) for q in SMM_PARAMS]
open(joinpath(OUT, "valid_share.csv"), "w") do io
    println(io, join(vcat(["i"], names_, ["Q", "reason", "seconds"]), ","))
    for (i, (z, r)) in enumerate(zip(pts, res))
        nat = [from_search(z[j], SMM_PARAMS[j]) for j in eachindex(z)]
        println(io, join(vcat([string(i)], [@sprintf("%.10g", x) for x in nat], [@sprintf("%.10g", r.q), "\"$(r.reason)\"", @sprintf("%.2f", r.secs)]), ","))
    end
end
valid = [r.reason == "valid" for r in res]
open(joinpath(OUT, "valid_share_summary.txt"), "w") do io
    for out in (io, stdout)
        @printf(out, "%d points, grid %d: %d valid (%.1f%%); median seconds per evaluation %.1f; wall %.1f min\n",
                N, G, count(valid), 100 * mean(valid), median([r.secs for r in res]), (time() - t_all) / 60)
        println(out, "\npenalty reasons:")
        tally = Dict{String,Int}(); for r in res; tally[r.reason] = get(tally, r.reason, 0) + 1; end
        for (k, v) in sort(collect(tally); by = x -> -x[2]); @printf(out, "  %5d  %s\n", v, k); end
        println(out, "\nvalid share by third of each parameter's box (search coordinates): low / middle / high")
        for (j, q) in enumerate(SMM_PARAMS)
            u = [(p[j] - lo[j]) / (hi[j] - lo[j]) for p in pts]
            sh = [mean(valid[(u .>= a) .& (u .< b)]) for (a, b) in ((0, 1/3), (1/3, 2/3), (2/3, 1.0001))]
            @printf(out, "  %-15s [%9.4g, %9.4g]   %5.1f%%  %5.1f%%  %5.1f%%\n", q.name, q.lo, q.hi, 100 .* sh...)
        end
    end
end
println("wrote $(joinpath(OUT, "valid_share.csv")) and valid_share_summary.txt")
