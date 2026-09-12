#!/usr/bin/env julia
# =============================================================================
# test_hc_process_shock.jl -- the idiosyncratic HC shock in parent_family.jl (P13).
#
#     julia --project=. tools/test_hc_process_shock.jl [targets.toml]
#
#     log HC_{t+1} = log F_t(inputs, HC_t) + sigma_eta * z_{t+1},   z ~ N(0,1)
#
#   1  exact deterministic case   sigma_eta = 0 leaves every solver object UNTOUCHED
#                                 (same array identity, not merely equal values)
#   2  quadrature                 nodes/weights integrate N(0,1) exactly to degree 9;
#                                 E[exp(sigma z)] vs exp(sigma^2/2) at 3 and 5 nodes
#   3  analytical gradients       every parent objective's gradient matches central
#                                 finite differences at sigma_eta > 0, in the terminal
#                                 period (spline quadrature, chain rule) and the interior
#                                 periods (eta-smoothed PCHIP)
#   4  a positive shock           changes decisions, raises HC dispersion at 17 and at
#                                 the handoff, leaves the mean of log HC nearly unchanged
#   5  reproducibility            two independent builds at the same sigma_eta agree bit
#                                 for bit; the other four draw streams do not move
#   6  support                    simulated out-of-support share and the quadrature mass
#                                 that lands beyond hc_max, at the pilot sigma_eta and box top
#   7  three vs five nodes        moment differences at a fixed parameter vector, against
#                                 0.1 data SE as the screening criterion
#
# Small grids: a correctness harness. Section 6-7 numbers are REPORTED, not asserted
# beyond sanity -- see docs/SMM.md, appendix, for how to read them.
# =============================================================================

using Printf, Random, NLopt, LinearAlgebra, Interpolations, Statistics, Distributions,
      QuantEcon, FastGaussQuadrature, Parameters, Dierckx, ProgressMeter, TOML, Test

const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "code", "src", "paths.jl"))
include(joinpath(REPO, "code", "src", "child_lifecycle.jl"))
include(joinpath(REPO, "code", "src", "parent_family.jl"))
include(joinpath(REPO, "code", "smm", "moments.jl"))

const TFILE = length(ARGS) >= 1 ? ARGS[1] : smm_targets_file()
const T = load_targets(TFILE)
@printf("targets   %s\n", TFILE)

const G  = (Na = 12, Nk = 2, Nhc = 12)
const CG = (Na = 12, Nk = 12, Nt = 3)
const N  = 300

# One child solve, shared by every parent below (the shock lives in the parent block).
const CH, VCH = build_child_solution((;), T; Na = CG.Na, Nk = CG.Nk, Nt = CG.Nt, simN = N, seed = 1234)

function parent_at(sigma_eta; Neta = 5, seed = 1234, solve = true)
    p = Parent_child_interaction_age_specific_AR1(; Na = G.Na, Nk = G.Nk, Nhc = G.Nhc, simN = N,
            seed = seed, school_time = target_school_time(T), sigma_eta = sigma_eta, Neta = Neta)
    p.V_child_interp = VCH
    solve && redirect_stdout(devnull) do
        solve_model!(p; verbose = false); simulate_model!(p)
    end
    return p
end

@testset "HC process shock" begin

# ---- 1. exact deterministic case -------------------------------------------
@testset "1 sigma_eta = 0 is exact" begin
    p = parent_at(0.0; solve = false)
    @test p.sigma_eta == 0.0
    @test all(iszero, p.sigma_eta .* p.z_nodes)
    interp = expected_interp(p, create_interp(p, fill(1.0, p.T, p.Na, p.Nk, p.Nhc, p.Np), p.T))
    @test eta_expected_interp(p, interp) === interp          # returned untouched, not rebuilt
    a, hc = 3.0, 450.0
    @test eval_child_value_eta(p, VCH, a, hc, 0.0, true) == eval_child_value(VCH, a, hc, 0.0, true)
    @test hc_apply_shock(p, 400.0, 1.7) == 400.0
end

# ---- 2. quadrature ------------------------------------------------------------
@testset "2 quadrature" begin
    p = parent_at(0.03; solve = false)
    z, w = p.z_nodes, p.z_weights
    @test isapprox(sum(w), 1.0; atol = 1e-12)
    @test isapprox(sum(w .* z .^ 2), 1.0; atol = 1e-12)
    @test isapprox(sum(w .* z .^ 4), 3.0; atol = 1e-10)
    @test isapprox(sum(w .* z .^ 8), 105.0; atol = 1e-8)      # 5 nodes: exact to degree 9
    for (Neta, s) in ((3, 0.03), (5, 0.03), (3, 0.08), (5, 0.08))
        q = parent_at(s; Neta = Neta, solve = false)
        m = sum(q.z_weights .* exp.(s .* q.z_nodes))
        @printf("    E[exp(sigma z)] Neta=%d sigma=%.2f: %.10f  vs exp(sigma^2/2) %.10f  (rel err %.1e)\n",
                Neta, s, m, exp(s^2 / 2), abs(m / exp(s^2 / 2) - 1))
        @test abs(m / exp(s^2 / 2) - 1) < 1e-6
    end
end

# ---- 3. analytical gradients vs finite differences ----------------------------
# The two objects this change touched: the terminal continuation (spline quadrature with
# the exp(sigma z) chain-rule factor) and the eta-smoothed PCHIP continuation. Checked
# at sigma_eta = 0.05 -- larger than the pilot start, so a chain-rule slip is visible.
@testset "3 gradients" begin
    p = parent_at(0.05)
    t = p.T
    function fd_check(f, x, lo, hi; h = 1e-5)
        g = zeros(length(x)); f(x, g)
        ok = true
        for i in eachindex(x)
            xp = copy(x); xm = copy(x)
            xp[i] = min(hi[i], x[i] + h); xm[i] = max(lo[i], x[i] - h)
            fdv = (f(xp, Float64[]) - f(xm, Float64[])) / (xp[i] - xm[i])
            rel = abs(fdv - g[i]) / max(1.0, abs(g[i]))
            rel < 2e-4 || (ok = false; @printf("      component %d: analytic %.6g  fd %.6g\n", i, g[i], fdv))
        end
        ok
    end
    for (ia, ihc, ip) in ((3, 4, 2), (6, 8, 3), (9, 11, 4))
        assets, HC, ps, cap = p.a_grid[ia], p.hc_grid[ihc], p.p_grid[ip], 1.0
        # terminal period
        fT(x, g) = obj_last_period_full(p, x[1], x[2], x[3], x[4], x[5], assets, HC, cap, t, ps, p.V_child_interp, g)
        bmax = budget_ceiling(p, assets, cap, t, ps)
        x = [0.4bmax, 0.08, 0.05bmax, 0.35, 0.25]
        @test fd_check(fT, x, [1e-4, TIME_FLOOR, 1e-4, TIME_FLOOR, TIME_FLOOR], [bmax, 1, bmax, 1, 1])
        # an interior full period, on the eta-smoothed continuation
        tt = t - 2
        interp = eta_expected_interp(p, expected_interp(p, create_interp(p, p.sol_v, tt + 1)))
        fW(x, g) = obj_work_period_full(p, x[1], x[2], x[3], x[4], x[5], assets, HC, cap, tt, ps, ip, interp, g)
        @test fd_check(fW, x, [1e-4, TIME_FLOOR, 1e-4, TIME_FLOOR, TIME_FLOOR], [bmax, 1, bmax, 1, 1])
        # a parent-only period
        tp = T_CHILD_VOICE - 2
        interp2 = eta_expected_interp(p, expected_interp(p, create_interp(p, p.sol_v, tp + 1)))
        fP(x, g) = obj_work_period_parentonly(p, x[1], x[2], x[3], x[4], assets, HC, cap, tp, ps, ip, interp2, g)
        @test fd_check(fP, [0.4bmax, 0.05bmax, 0.35, 0.25], [1e-4, 1e-4, TIME_FLOOR, TIME_FLOOR], [bmax, bmax, 1, 1])
    end
end

# ---- 4. a positive shock changes the model ---------------------------------
@testset "4 positive shock" begin
    p0 = parent_at(0.0); p1 = parent_at(0.03)
    @test maximum(abs.(p1.sol_v .- p0.sol_v)) > 0          # decisions and values move
    @test maximum(abs.(p1.sol_t .- p0.sol_t)) > 0
    sd0, sd1 = std(log.(p0.sim_hc[:, 17])), std(log.(p1.sim_hc[:, 17]))
    sdh = std(log.(p1.sim_hc[:, p1.T + 1]))
    m0, m1 = mean(log.(p0.sim_hc[:, 17])), mean(log.(p1.sim_hc[:, 17]))
    @printf("    SD log HC17: %.4f -> %.4f (handoff %.4f);  mean log HC17: %.4f -> %.4f\n", sd0, sd1, sdh, m0, m1)
    @test sd1 > 2 * sd0
    @test abs(m1 - m0) < 0.02
    @test all(p1.sim_hc .> 0) && all(isfinite, p1.sim_hc)
end

# ---- 5. reproducibility and stream isolation --------------------------------
@testset "5 reproducible; other streams untouched" begin
    a = parent_at(0.03); b = parent_at(0.03)
    @test a.sol_v == b.sol_v && a.sim_hc == b.sim_hc && a.sim_a == b.sim_a
    c = parent_at(0.0; solve = false)
    @test a.sim_a_init == c.sim_a_init && a.sim_k_init == c.sim_k_init &&
          a.sim_hc_init == c.sim_hc_init && a.draws_uniform_p == c.draws_uniform_p
    d = parent_at(0.03; seed = 4321, solve = false)
    @test d.draws_eta != a.draws_eta
end

# ---- 6. support ---------------------------------------------------------------
@testset "6 support" begin
    for s in (0.03, 0.08)
        p = parent_at(s)
        out_sim = count(x -> x > p.hc_max || x < p.hc_min, p.sim_hc) / length(p.sim_hc)
        mult = exp.(s .* p.z_nodes)
        # quadrature mass that lands beyond hc_max from the top grid node, and beyond
        # hc_min from the bottom one -- where the interpolant extrapolates linearly
        above = sum(p.z_weights[j] for j in eachindex(mult) if p.hc_grid[end] * mult[j] > p.hc_max; init = 0.0)
        below = sum(p.z_weights[j] for j in eachindex(mult) if p.hc_grid[1] * mult[j] < p.hc_min; init = 0.0)
        # the mass that lands beyond hc_max from the simulated HC range at 17
        hi17 = maximum(p.sim_hc[:, 17])
        above_sim = sum(p.z_weights[j] for j in eachindex(mult) if hi17 * mult[j] > p.hc_max; init = 0.0)
        @printf("    sigma_eta=%.2f: simulated out-of-support share %.4f; quadrature mass beyond hc_max from the top node %.3f, from the max simulated HC17 %.3f; below hc_min from the bottom node %.3f; HC range %.1f-%.1f of [%.0f, %.0f]\n",
                s, out_sim, above, above_sim, below, extrema(p.sim_hc)..., p.hc_min, p.hc_max)
        @test out_sim == 0
        @test above_sim == 0
    end
end

# ---- 7. three vs five nodes ---------------------------------------------------
@testset "7 Neta 3 vs 5" begin
    se = target_se(T)
    for s in (0.03, 0.08)
        p3 = parent_at(s; Neta = 3); p5 = parent_at(s; Neta = 5)
        m3 = model_moments(p3); m5 = model_moments(p5)
        worst = 0.0; worst_k = ""
        for (j, k) in enumerate(SMM_PARENT_MOMENTS)
            d = abs(getfield(m3, Symbol(k)) - getfield(m5, Symbol(k))) / se[j]
            d > worst && (worst = d; worst_k = k)
        end
        d_sd = abs(std(log.(p3.sim_hc[:, 17])) - std(log.(p5.sim_hc[:, 17])))
        @printf("    sigma_eta=%.2f: worst parent-moment difference 3 vs 5 nodes = %.3f data SE (%s); SD log HC17 differs by %.5f\n",
                s, worst, worst_k, d_sd)
        @test isfinite(worst)
    end
end

end
