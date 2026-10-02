# classify_nonfinite.jl (2026-10-02, diagnostic): for every Sobol' point of valid_share.csv that was penalised as
# invalid_sim_child_nonfinite, plus the recovery theta0 and the SMM start, solve once at the production grids and
# record the college share (overall and by BothCollege), mean ln k at 18 and Q. Read-only: changes nothing.
using Distributed, Printf, TOML, Statistics, Dates, DelimitedFiles
const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
const TFILE = joinpath(REPO, "output/smm_runs/2026-10-02_193217_281837_targets/targets.toml")
addprocs(32; exeflags = `--project=$REPO --threads=1 --startup-file=no`)
@everywhere begin
    using Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames, Statistics, Dates
    using ProgressMeter, Distributions, StatsBase, QuantEcon, FastGaussQuadrature, Parameters, Dierckx, TOML
    LinearAlgebra.BLAS.set_num_threads(1)
    const SRC = joinpath($REPO, "code", "src")
    include(joinpath(SRC, "paths.jl")); include(joinpath(SRC, "manifest.jl")); include(joinpath(SRC, "diagnostics.jl"))
    include(joinpath(SRC, "child_lifecycle.jl")); include(joinpath(SRC, "parent_family.jl")); include(joinpath(SRC, "tiktak.jl"))
    include(joinpath($REPO, "code", "smm", "moments.jl"))
    const T = load_targets($TFILE)
    function classify(nat::Dict{Symbol,Float64})
        z = [to_search(nat[q.name], q) for q in SMM_PARAMS]
        kw = unpack(z)
        try
            r = run_pipeline(kw, T; Na = 30, Nk = 2, Nhc = 30, simN = 2000, seed = 1234,
                             child_grid = (Na = 30, Nk = 30, Nt = 5), demo_sim = false)
            col = r.child.sim_college; bc = r.parent.sim_k[:, 1]
            lk = log.(r.parent.sim_hc[:, end])
            q = smm_objective(z, T; Na = 30, Nk = 2, Nhc = 30, simN = 2000, seed = 1234,
                              child_grid = (Na = 30, Nk = 30, Nt = 5), demo_sim = false)
            return (ok = true, college = mean(col), c_bc0 = mean(col[bc .< 0.5]), c_bc1 = mean(col[bc .>= 0.5]),
                    lnk18 = mean(lk), nonfinite_college = count(!isfinite, col), q = q, err = "")
        catch e
            return (ok = false, college = NaN, c_bc0 = NaN, c_bc1 = NaN, lnk18 = NaN, nonfinite_college = -1, q = NaN,
                    err = first(sprint(showerror, e), 120))
        end
    end
end
D = @__DIR__
raw, hdr = readdlm(joinpath(D, "valid_share.csv"), ','; header = true)
cols = Dict(String(h) => j for (j, h) in enumerate(vec(hdr)))
names_ = [q.name for q in SMM_PARAMS]
pts = Tuple{String,Dict{Symbol,Float64}}[]
for i in 1:size(raw, 1)
    strip(String(raw[i, cols["reason"]]), '"') == "invalid_sim_child_nonfinite" || continue
    push!(pts, ("sobol_$(raw[i, cols["i"]])", Dict(n => Float64(raw[i, cols[String(n)]]) for n in names_)))
end
theta0 = Dict{Symbol,Float64}(
    :phi_2 => 0.196278, :phi_3 => 0.100000, :lambda_2 => 1.577178,
    :sigma_1_0 => -0.630853, :sigma_1_1 => -0.115329, :sigma_2_0 => -7.154000, :sigma_2_1 => 0.072000,
    :sigma_3_0 => -0.244781, :sigma_3_1 => 0.005000, :sigma_4_0 => -6.598000, :sigma_4_1 => 0.271000,
    :d_0 => 4.109547, :d_1 => 4.532005, :d_2 => 2.000000, :d_3 => 2.878411,
    :kappa_0 => 0.413682, :kappa_theta => -0.182563, :kappa_ParEd => -0.108108,
    :kappa_terminal => 8.786782, :sigma_eps => 1.142235)   # tools/test_param_recovery.jl THETA0
pushfirst!(pts, ("smm_start", Dict(n => smm_start(n) for n in names_)))
pushfirst!(pts, ("recovery_theta0", theta0))
println("classifying $(length(pts)) points on $(nworkers()) workers, ", now()); flush(stdout)
res = pmap(p -> classify(p[2]), pts)
open(joinpath(D, "classify_nonfinite.csv"), "w") do io
    println(io, "point,ok,college,college_bc0,college_bc1,mean_lnk18,nonfinite_college,Q,error")
    for (p, r) in zip(pts, res)
        @printf(io, "%s,%s,%.4f,%.4f,%.4f,%.4f,%d,%.6g,\"%s\"\n", p[1], r.ok, r.college, r.c_bc0, r.c_bc1, r.lnk18,
                r.nonfinite_college, r.q, replace(r.err, '"' => '\''))
    end
end
sob = [r for (p, r) in zip(pts, res) if startswith(p[1], "sobol_") && r.ok]
c = [r.college for r in sob]
@printf("Sobol' points classified: %d (errors %d)\n", length(sob), count(r -> !r.ok, res))
@printf("  all college (share = 1): %d;  no college (share = 0): %d;  in between: %d\n",
        count(==(1.0), c), count(==(0.0), c), count(x -> 0 < x < 1, c))
for (p, r) in zip(pts[1:2], res[1:2])
    @printf("%-16s college %.3f (BC0 %.3f, BC1 %.3f)  mean ln k18 %.3f  Q %.6g\n", p[1], r.college, r.c_bc0, r.c_bc1, r.lnk18, r.q)
end
