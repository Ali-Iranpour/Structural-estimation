#!/usr/bin/env julia
# =============================================================================
# test_smm_baseline.jl -- the DETERMINISTIC baseline is reproduced by the current code.
#
#     julia --threads=1 --project=. tools/test_smm_baseline.jl
#
# Run 2026-09-10_183649 is the last estimate produced BEFORE the 2026-09-11 changes
# (the HC shock `sigma_eta` and the free taste scale `sigma_eps`). Its final fit --
# Q = 427.0098967 on seventeen moments at grid 30 / simN 2000, plus the full moment table
# in its run.log -- was computed by the pre-shock solver. The current code, evaluated at
# that run's parameters with sigma_eta = 0 and sigma_eps = 0.5, must reproduce it:
#
#   * Q to 1e-6 relative, rebuilt from that run's OWN targets file (its moment order and
#     its inverse-variance weights, parsed directly since load_targets now expects the
#     seven-TAS-moment order);
#   * the parent-block and TAS moments to the 4 decimals the run.log prints.
#
# This is the bit-identity claim of `eta_expected_interp` / `eval_child_value_eta` /
# `hc_apply_shock` at sigma_eta = 0, and of `sigma_eps` as a struct field, tested on the
# real grids rather than asserted from the code's structure.
#
# HISTORY. Until 2026-09-11 this file pinned the nine-parameter snapshot of
# 2026-09-06_183119. That snapshot predates the own-study respecification and the promoted
# PARENT_DEFAULTS, so no current code path could reproduce it and the test had been failing
# on its first assertion (`length(SMM_PARAMS) == 9`) since the fourteen-parameter design.
# The reference is moved forward to the last pre-shock run rather than dropped.
# =============================================================================
using Test, TOML, Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames
using Statistics, Dates, ProgressMeter, Distributions, StatsBase
using QuantEcon, FastGaussQuadrature, Parameters, Dierckx
BLAS.set_num_threads(1)
const REPO = normpath(joinpath(@__DIR__, ".."))
for f in ("paths.jl", "manifest.jl", "diagnostics.jl", "child_lifecycle.jl", "parent_family.jl")
    include(joinpath(REPO, "code/src", f))
end
include(joinpath(REPO, "code/smm/moments.jl"))

const BASE_RUN = joinpath(REPO, "output", "smm_runs", "2026-09-10_183649")
const EST  = TOML.parsefile(joinpath(BASE_RUN, "estimates.toml"))
const OLDT = TOML.parsefile(joinpath(BASE_RUN, "targets.toml"))     # its own moment order
const Q_REF = EST["Q_final"]

# The run.log's fit table, model column (4 decimals). Parsed rather than retyped.
function logged_model_moments()
    out = Dict{String,Float64}()
    lines = readlines(joinpath(BASE_RUN, "run.log"))
    i = findlast(l -> occursin("Targeted moments --", l), lines)
    for l in lines[i+3:i+25]
        f = split(strip(l))
        length(f) >= 3 && occursin(r"^[a-z]", f[1]) && (v = tryparse(Float64, f[2]); v === nothing || (out[f[1]] = v))
    end
    out
end
const LOGGED = logged_model_moments()

# The current target file is used only for the frozen scalars every evaluation needs
# (m_psychic, school_time, wealth cut) -- identical in both files, asserted below.
const T = load_targets(smm_targets_file())
@test target_m_psychic(T) == OLDT["m_psychic"]
@test T["_spec"].wealth_cut == OLDT["tas_wealth_winsor_cut"]
@test T["_spec"].school_time == Float64.(OLDT["school_time"])

@testset "specification" begin
    @test length(SMM_PARAMS) == 16
    # The block defaults are the fitted exp16b vector since 2026-09-12 (sigma_eta = 0.0315,
    # sigma_eps = 1.14); this regression sets both EXPLICITLY below, so it does not
    # depend on them.
    @test PARENT_DEFAULTS.R_1 == 0.0
    lo, hi = search_bounds(); z = incumbent()
    @test all(lo .<= z .<= hi)
    @test !smm_feasible((sigma_1_0 = -0.1, sigma_1_1 = 0.05))
end

println("Evaluating the 2026-09-10 fit at grid 30 / simN 2000 with sigma_eta = 0, sigma_eps = 0.5"); flush(stdout)
v = Dict{Symbol,Float64}(q.name => smm_start(q.name) for q in SMM_PARAMS)
for (k, x) in EST["parameters"]; v[Symbol(k)] = Float64(x); end
v[:sigma_eta] = 0.0; v[:sigma_eps] = 0.5
t0 = time()
res = evaluate_at(v, T; Na = 30, Nk = 2, Nhc = 30, simN = 2000, seed = 1234,
                  child_grid = (Na = 30, Nk = 30, Nt = 5), demo_sim = true)
@printf("  %.0f s\n", time() - t0)
m = res.moments

# Q on the OLD moment order with the OLD weights (1/se^2), from the old file directly.
old_names = String.(OLDT["moment_cov"]["names"])
old_se    = Float64.(OLDT["moment_cov"]["se"])
Q = sum(((getproperty(m, Symbol(k)) - OLDT[k]["mean"]) / old_se[j])^2 for (j, k) in enumerate(old_names))

@testset "Full-grid fit reproduces run 2026-09-10_183649" begin
    @test isapprox(Q, Q_REF; rtol = 1e-6)
    for (k, logged) in LOGGED
        hasproperty(m, Symbol(k)) || continue
        @test isapprox(getproperty(m, Symbol(k)), logged; atol = 6e-5)
    end
    @test res.nviol == 0 && res.nbad == 0
    @test all(isfinite, res.pipeline.parent.sim_hc[:, end]) && all(>(0), res.pipeline.parent.sim_hc[:, end])
    @test continuation_selftest(verbose = false)
end
@printf("Reproduced Q = %.10f  (reference %.10f);  %d logged moments compared\n", Q, Q_REF, length(LOGGED))
