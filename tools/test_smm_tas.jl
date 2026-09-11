#!/usr/bin/env julia
# =============================================================================
# test_smm_tas.jl -- validation of the fourteen-parameter / seventeen-moment SMM.
#
#     julia --project=.. tools/test_smm_tas.jl [targets.toml]
#
# Covers exactly the things that can be wrong SILENTLY in this specification, which is a
# different list from the things that can be wrong loudly:
#
#   1  target reproduction      the TAS targets Julia reads are the published numbers
#   2  parameter routing        no child parameter reaches the parent constructor
#   3  child-solution refresh   every kappa actually changes the child solve
#   4  cache-key completeness   the cache cannot serve a stale solution
#   5  handoff arrays           the three handoff assignments are exact
#   6  demonstration isolation  the initial child simulation cannot supply a moment
#   7  determinism / CRN        the same z gives the same Q, twice
#   8  psychic centring         the reparameterisation is behaviourally neutral
#   9  wealth accounting        the target is POST-transfer, and it is not pre-transfer
#  10  cache parity             the cached refresh equals an INDEPENDENT full solve
#  11  mutation isolation       simulating a returned child cannot poison the cache
#  12  partial points           an omitted kappa uses the SMM default, not the constructor's
#  13  parent_extra             a non-estimated parent setting reaches the constructor
#
# Small grids throughout: this is a correctness harness, not an accuracy one.
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
@printf("spec      %d moments (%d parent + %d TAS), %d parameters (%d parent + %d child)\n",
        length(SMM_MOMENTS), length(SMM_PARENT_MOMENTS), length(SMM_TAS_MOMENTS),
        length(SMM_PARAMS), length(SMM_PARENT_PARAMS), length(SMM_CHILD_PARAMS))

const G  = (Na = 16, Nk = 2, Nhc = 16)
const CG = (Na = 16, Nk = 16, Nt = 3)
const N  = 400
base() = Dict{Symbol,Float64}(q.name => param_default(q.name) for q in SMM_PARAMS)
ev(v; kw...) = evaluate_at(v, T; Na = G.Na, Nk = G.Nk, Nhc = G.Nhc, simN = N,
                           seed = 1234, child_grid = CG, kw...)

@testset "SMM 14-parameter TAS specification" begin

# ---- 1. target reproduction ------------------------------------------------
# The published moment file is the reference. The generator rebuilds these from the
# microdata; if the two ever disagree, the target file is not what the codebook documents.
@testset "1 target reproduction" begin
    pub = Dict{String,Float64}()
    for ln in readlines(joinpath(REPO, "Input", "SMM_TAS_Moments.csv"))[2:end]
        f = split(ln, ",")
        length(f) >= 4 && (pub[f[1]] = parse(Float64, f[4]))
    end
    for k in ("k0_complete", "kth_ga17_t1_c", "kth_ga17_t2_c", "kth_ga17_t3_c",
              "kpe_g0_c", "kpe_g1_c")
        @test isapprox(T[k].mean, pub[k]; atol = 1e-9)
    end
    # Wealth is winsorised and converted, so it is NOT the published raw number. Check the
    # conversion instead: it must be below the raw mean and in model units.
    raw = pub["kterm_x_strict"] / 10_000
    @test T["kterm_x_strict_w99"].mean < raw
    @test 20 < T["kterm_x_strict_w99"].mean < 40
    # SEs are present, positive and finite for every targeted moment.
    @test all(x -> isfinite(x) && x > 0, target_se(T))
    @test length(target_se(T)) == length(SMM_MOMENTS)
    # The covariance is symmetric and its diagonal is the SE vector squared.
    S = target_Sigma(T)
    @test maximum(abs.(S .- S')) < 1e-12 * maximum(abs.(S))
    # 1e-12, not 0: the target file is text, so `se` and `cov` are round-trips through a
    # decimal literal. They are written at full double precision (%.17g), so this is a
    # check that the two blocks were built from the SAME influence functions, not a check
    # on floating-point formatting.
    @test isapprox(sqrt.(diag(S)), target_se(T); rtol = 1e-12)
end

# ---- 2. parameter routing ---------------------------------------------------
@testset "2 parameter routing" begin
    pk, ck = split_params(unpack(incumbent()))
    @test Set(keys(ck)) == Set(SMM_CHILD_PARAMS)
    @test Set(keys(pk)) == Set(SMM_PARENT_PARAMS)
    # No child name may be a parent field, or it would be silently absorbed.
    for n in SMM_CHILD_PARAMS
        @test !hasproperty(PARENT_DEFAULTS, n)
    end
    for n in SMM_PARENT_PARAMS
        @test !hasproperty(CHILD_DEFAULTS, n)
    end
    # An unroutable name must raise, not be dropped.
    @test_throws ErrorException split_params((; nonesuch = 1.0))
end

# ---- 3. child-solution refresh ----------------------------------------------
# THE CENTRAL CHECK. If any of these four failed to move the child solution, the run would
# report a converged fit for a parameter that does nothing -- the exact failure the old
# parent-only invariant existed to prevent.
@testset "3 every kappa moves the child solve" begin
    v0 = base(); r0 = ev(v0)
    for (nm, val) in ((:kappa_0, -0.60), (:kappa_theta, -3.0),
                      (:kappa_ParEd, -0.50), (:kappa_terminal, 15.0))
        v = base(); v[nm] = val
        r = ev(v)
        @test r.r != r0.r          # the residual vector must move
        moved = r.moments.k0_complete != r0.moments.k0_complete ||
                r.moments.kterm_x_strict_w99 != r0.moments.kterm_x_strict_w99
        @test moved
    end
end

# ---- 4. cache-key completeness ----------------------------------------------
# The cache may only hold objects that no estimated parameter can change. If a kappa
# leaked into the key the cache would never hit (slow but correct); if a kappa-dependent
# array were cached the answer would be wrong. Test the second, which is the silent one.
@testset "4 cache cannot serve a stale solution" begin
    cfg = child_config(T; Na = CG.Na, Nk = CG.Nk, Nt = CG.Nt, simN = N, seed = 1234)
    @test !any(n -> haskey(pairs(cfg), n), SMM_CHILD_PARAMS)
    # Two different psychic costs, same config: the college values must DIFFER ...
    c1, _ = build_child_solution((; kappa_0 = 0.0587, kappa_theta = -0.0342,
                                    kappa_ParEd = -0.007, kappa_terminal = 5.0), T;
                                 Na = CG.Na, Nk = CG.Nk, Nt = CG.Nt, simN = N, seed = 1234)
    c2, _ = build_child_solution((; kappa_0 = -0.60, kappa_theta = -0.0342,
                                    kappa_ParEd = -0.007, kappa_terminal = 5.0), T;
                                 Na = CG.Na, Nk = CG.Nk, Nt = CG.Nt, simN = N, seed = 1234)
    fin(a, b) = (m = isfinite.(a) .& isfinite.(b); (a[m], b[m]))
    v1, v2 = fin(c1.sol_v_college, c2.sol_v_college)
    @test maximum(abs.(v1 .- v2)) > 1e-6
    # ... while the two CACHED blocks stay bit-identical, since neither reads a kappa.
    w1, w2 = fin(c1.sol_v_work, c2.sol_v_work)
    g1, g2 = fin(c1.sol_v_grad, c2.sol_v_grad)
    @test maximum(abs.(w1 .- w2)) == 0.0
    @test maximum(abs.(g1 .- g2)) == 0.0
    # The cache must not be handing back a simulated object: nothing in it is a sim array.
    for (_, e) in CHILD_BASE_CACHE
        @test all(f -> startswith(String(f), "sol_"), keys(e))
    end
end

# ---- 5. handoff arrays -------------------------------------------------------
@testset "5 handoff is exact" begin
    r = run_pipeline(unpack(incumbent()), T; Na = G.Na, Nk = G.Nk, Nhc = G.Nhc,
                     simN = N, seed = 1234, child_grid = CG, demo_sim = true)
    p, c = r.parent, r.child
    @test c.sim_a_init  == p.sim_a[:,  p.T + 1]
    @test c.sim_k_init  == p.sim_hc[:, p.T + 1]
    @test c.sim_bc_init == p.sim_k[:, 1]
    @test length(c.sim_a_init) == size(p.sim_a, 1)      # matching simulation sizes
    @test c.simN == p.simN                               # matching simulation sizes
end

# ---- 6. the demonstration simulation cannot supply a moment ------------------
# Run the pipeline WITH and WITHOUT the initial child simulation. Every targeted moment
# must be identical: if the demonstration run leaked into any of them, they would not be.
@testset "6 demonstration simulation is isolated" begin
    a = run_pipeline(unpack(incumbent()), T; Na = G.Na, Nk = G.Nk, Nhc = G.Nhc,
                     simN = N, seed = 1234, child_grid = CG, demo_sim = true)
    b = run_pipeline(unpack(incumbent()), T; Na = G.Na, Nk = G.Nk, Nhc = G.Nhc,
                     simN = N, seed = 1234, child_grid = CG, demo_sim = false)
    ma, mb = model_moments(a, T), model_moments(b, T)
    for k in SMM_MOMENTS
        @test getfield(ma, Symbol(k)) === getfield(mb, Symbol(k))
    end
end

# ---- 7. determinism ----------------------------------------------------------
@testset "7 evaluations are deterministic" begin
    z = incumbent()
    q1 = smm_objective(z, T; Na = G.Na, Nk = G.Nk, Nhc = G.Nhc, simN = N,
                       seed = 1234, child_grid = CG, demo_sim = false)
    q2 = smm_objective(z, T; Na = G.Na, Nk = G.Nk, Nhc = G.Nhc, simN = N,
                       seed = 1234, child_grid = CG, demo_sim = false)
    @test q1 === q2
    @test isfinite(q1)
    # The removed three-argument forms must FAIL, not silently work.
    @test_throws ErrorException smm_objective(z, T, nothing)
    @test_throws ErrorException report_fit(z, T, nothing)
end

# ---- 8. the psychic recentring is behaviourally neutral ----------------------
# Same psychic cost, two parameterisations: uncentred (m_psychic = 0, legacy kappa_0) and
# centred (m_psychic = m, kappa_0 + kappa_theta*m). The college values must agree.
@testset "8 psychic centring is neutral" begin
    mp = target_m_psychic(T)
    check_psychic_centring(mp)
    common = (Na = CG.Na, Nk = CG.Nk, Nt = CG.Nt, rho = CHILD_DEFAULTS.rho,
              psi_terminal = CHILD_DEFAULTS.psi_terminal, omega = CHILD_DEFAULTS.omega,
              a_max = CHILD_DEFAULTS.a_max, w = CHILD_DEFAULTS.w,
              simN = N, seed = 1234, kappa_terminal = 5.0, kappa_ParEd = -0.007,
              kappa_theta = LEGACY_KAPPA_THETA)
    solved(m) = (redirect_stdout(devnull) do; redirect_stderr(devnull) do
                     solve_model_work!(m); solve_model_college!(m)
                 end; end; m)
    old = solved(ConSavLaborCollege_AR1(; common..., m_psychic = 0.0,
                                          kappa_0 = LEGACY_KAPPA_0))
    new = solved(ConSavLaborCollege_AR1(; common..., m_psychic = mp,
                                          kappa_0 = LEGACY_KAPPA_0 + LEGACY_KAPPA_THETA*mp))
    msk = isfinite.(old.sol_v_college) .& isfinite.(new.sol_v_college)
    @test maximum(abs.(old.sol_v_college[msk] .- new.sol_v_college[msk])) < 1e-8
end

# ---- 9. post-transfer wealth accounting --------------------------------------
@testset "9 the wealth moment is POST-transfer" begin
    r = run_pipeline(unpack(incumbent()), T; Na = G.Na, Nk = G.Nk, Nhc = G.Nhc,
                     simN = N, seed = 1234, child_grid = CG, demo_sim = false)
    m = model_moments(r, T)
    @test r.retained ≈ r.child.sim_a_init .- r.child.sim_tr_init
    @test all(isfinite, r.transfers)
    @test all(>=(0), r.transfers)                       # a transfer is never negative
    # Retained assets are strictly below pre-transfer assets wherever a transfer was made,
    # and the moment must NOT equal the pre-transfer mean -- substituting the latter is the
    # specific mistake the specification rules out.
    d = moment_diagnostics(r.parent)
    @test m.kterm_x_strict_w99 <= d.terminal_assets + 1e-12
    @test !isapprox(m.kterm_x_strict_w99, d.terminal_assets; rtol = 1e-6)
    @test m.retained_negative == 0                      # delta_P floor holds
end

# ---- 10. cached refresh vs an INDEPENDENT full solve ------------------------
# Group 4 compares two solutions that both came through the cache, so it can only show
# that the cache DISCRIMINATES on the kappas -- not that what it returns is right. This
# compares the cached path against a model built and solved from scratch, which is the
# claim actually being made ("bit-identical to a full re-solve").
@testset "10 cache parity with a full solve" begin
    kap = (kappa_0 = -0.45, kappa_theta = -2.0, kappa_ParEd = -0.25, kappa_terminal = 9.0)
    cached, Vc = build_child_solution(kap, T; Na = CG.Na, Nk = CG.Nk, Nt = CG.Nt,
                                      simN = N, seed = 1234)
    cfg = child_config(T; Na = CG.Na, Nk = CG.Nk, Nt = CG.Nt, simN = N, seed = 1234)
    fresh = ConSavLaborCollege_AR1(; merge(cfg, kap)...)
    redirect_stdout(devnull) do; redirect_stderr(devnull) do
        solve_model_work!(fresh)
        solve_model_college!(fresh)          # NOT reuse_grad: solves both stages itself
        optimal_transfer_work!(fresh)
        optimal_transfer_college!(fresh)
    end; end
    for f in (:sol_v_college, :sol_c_college, :sol_h_college,
              :sol_tr_work, :sol_tr_college, :sol_tr_v_work, :sol_tr_v_college)
        a, b = getfield(cached, f), getfield(fresh, f)
        @test isnan.(a) == isnan.(b)                       # identical feasibility pattern
        m = isfinite.(a) .& isfinite.(b)
        @test maximum(abs.(a[m] .- b[m])) == 0.0           # bit-identical, not "close"
    end
    # The terminal-value object the parent actually consumes must agree too.
    Vf = terminal_value_spline(fresh; s = 10.0)
    for a in (5.0, 25.0, 60.0), hc in (400.0, 520.0), bc in (0.0, 1.0)
        @test Vc(a, hc, bc) == Vf(a, hc, bc)
    end
end

# ---- 11. the cache cannot be contaminated by a simulation --------------------
# Tier B caches a fully solved child. If what it stored were a MODEL rather than copies of
# its arrays, simulating the returned object would write `sim_*` state that the next
# evaluation inherited. Simulate hard between two identical builds and check the second is
# clean and identical.
@testset "11 mutation isolation" begin
    kap = (kappa_0 = -0.45, kappa_theta = -2.0, kappa_ParEd = -0.25, kappa_terminal = 9.0)
    c1, _ = build_child_solution(kap, T; Na = CG.Na, Nk = CG.Nk, Nt = CG.Nt, simN = N, seed = 1234)
    before = copy(c1.sol_v_college)
    c1.sim_a_init  .= 20.0
    c1.sim_k_init  .= 500.0
    c1.sim_bc_init .= 1.0
    redirect_stdout(devnull) do; redirect_stderr(devnull) do
        simulate_model_family!(c1)
    end; end
    @test !any(isnan, c1.sim_college)              # it really did simulate
    c2, _ = build_child_solution(kap, T; Na = CG.Na, Nk = CG.Nk, Nt = CG.Nt, simN = N, seed = 1234)
    @test c2 !== c1                                 # a fresh object, not the cached one
    @test all(isnan, c2.sim_college)                # no simulation state carried over
    @test all(isnan, c2.sim_tr_init)
    a, b = before, c2.sol_v_college
    m = isfinite.(a) .& isfinite.(b)
    @test maximum(abs.(a[m] .- b[m])) == 0.0        # the solution survived unchanged
end

# ---- 12. a partial parameter point uses the SMM defaults --------------------
# `evaluate_at` documents a partial interface: anything not named stays at its own block's
# default. For CHILD parameters that used to mean the CONSTRUCTOR's default, not the SMM's
# -- kappa_0 = 0.2728 (the LEGACY UNCENTRED value) and kappa_terminal = 10.0 against SMM
# defaults of 0.0587 and 5.0. Every diagnostic tool uses this interface.
@testset "12 partial points complete from CHILD_DEFAULTS" begin
    c, _ = build_child_solution(NamedTuple(), T; Na = CG.Na, Nk = CG.Nk, Nt = CG.Nt,
                                simN = N, seed = 1234)
    @test c.kappa_0        == CHILD_DEFAULTS.kappa_0        # 0.0587, NOT 0.2728
    @test c.kappa_theta    == CHILD_DEFAULTS.kappa_theta
    @test c.kappa_ParEd    == CHILD_DEFAULTS.kappa_ParEd
    @test c.kappa_terminal == CHILD_DEFAULTS.kappa_terminal # 5.0, NOT 10.0
    @test c.m_psychic      == target_m_psychic(T)
    # Naming one leaves the other three at the SMM defaults, not the constructor's.
    c2, _ = build_child_solution((; kappa_theta = -2.0), T; Na = CG.Na, Nk = CG.Nk,
                                 Nt = CG.Nt, simN = N, seed = 1234)
    @test c2.kappa_theta    == -2.0
    @test c2.kappa_0        == CHILD_DEFAULTS.kappa_0
    @test c2.kappa_terminal == CHILD_DEFAULTS.kappa_terminal
    # An unknown child parameter must raise rather than bypass the cache key.
    @test_throws ErrorException build_child_solution((; nonesuch = 1.0), T;
        Na = CG.Na, Nk = CG.Nk, Nt = CG.Nt, simN = N, seed = 1234)
    # A partial point through the documented entry point evaluates without error.
    e = ev(Dict{Symbol,Float64}(:kappa_0 => -0.45))
    @test all(isfinite, e.r)
end

# ---- 13. non-estimated parent settings reach the constructor ----------------
# `grid_sensitivity.jl` varies the parent's asset ceiling. The refactor onto the shared
# pipeline silently dropped it, so every rung of the sweep solved at the same ceiling and
# the tool concluded the ceiling does not matter -- vacuously.
@testset "13 parent_extra reaches the parent" begin
    for am in (60.0, 140.0)
        r = run_pipeline(unpack(incumbent()), T; Na = G.Na, Nk = G.Nk, Nhc = G.Nhc,
                         simN = N, seed = 1234, child_grid = CG,
                         parent_extra = (a_max = am,), demo_sim = false)
        @test r.parent.a_max == am
    end
    # Two ceilings must not give identical simulated assets -- that equality is exactly
    # the symptom the dropped argument produced.
    r1 = run_pipeline(unpack(incumbent()), T; Na = G.Na, Nk = G.Nk, Nhc = G.Nhc, simN = N,
                      seed = 1234, child_grid = CG, parent_extra = (a_max = 60.0,), demo_sim = false)
    r2 = run_pipeline(unpack(incumbent()), T; Na = G.Na, Nk = G.Nk, Nhc = G.Nhc, simN = N,
                      seed = 1234, child_grid = CG, parent_extra = (a_max = 140.0,), demo_sim = false)
    @test r1.parent.sim_a != r2.parent.sim_a
    # An estimated parameter may never be injected through the side channel.
    @test_throws ErrorException run_pipeline(unpack(incumbent()), T; Na = G.Na, Nk = G.Nk,
        Nhc = G.Nhc, simN = N, seed = 1234, child_grid = CG,
        parent_extra = (phi_2 = 1.0,), demo_sim = false)
end

end
println("\nall validation groups completed")
