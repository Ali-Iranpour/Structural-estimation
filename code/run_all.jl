#!/usr/bin/env julia
# =============================================================================
# run_all.jl — one reproducible end-to-end run.
#
#     cd code && julia --project=.. run_all.jl [--quick]
#
# Solves the child lifecycle and the parent problem, simulates, runs every accuracy
# diagnostic, writes all LaTeX tables to output/tables/, and compiles them into
# output/reports/all_tables.pdf.
#
# Reproducible by construction:
#   * every RNG is seeded from SEED below;
#   * Project.toml/Manifest.toml pin the package set;
#   * each table carries a .meta.toml with the git commit, timestamp and parameters;
#   * the PDF wrapper is generated from whatever .tex files exist, so it can never
#     go stale relative to the tables.
#
# --quick runs a reduced grid for a smoke test (minutes, not hours).
# =============================================================================

using Printf, Random, NLopt, LinearAlgebra, Interpolations, DataFrames, Statistics, Dates
using ProgressMeter, Distributions, StatsBase, QuantEcon, FastGaussQuadrature
using Parameters, Dierckx

const SRC = joinpath(@__DIR__, "src")
include(joinpath(SRC, "paths.jl"))
include(joinpath(SRC, "manifest.jl"))
include(joinpath(SRC, "diagnostics.jl"))
include(joinpath(SRC, "tables.jl"))
include(joinpath(SRC, "child_lifecycle.jl"))
include(joinpath(SRC, "parent_family.jl"))

const QUICK = "--quick" in ARGS
const SEED  = 1234

# grid sizes
const C_NA, C_NK, C_NT = QUICK ? (20, 20,  6) : (50, 50, 10)
const P_NA, P_NK, P_NHC = QUICK ? (10,  2, 10) : (30,  2, 30)
const SIMN = QUICK ? 500 : 5000

banner(s) = (println("\n", "="^78); println(s); println("="^78))

banner("Structural Estimation — full run" * (QUICK ? "  [QUICK]" : ""))
@printf("commit %s   julia %s   %s\n", git_sha(), VERSION, Dates.format(now(), "yyyy-mm-dd HH:MM"))
@printf("child grid %d×%d×%d   parent grid %d×%d×%d   simN %d   seed %d\n",
        C_NA, C_NK, C_NT, P_NA, P_NK, P_NHC, SIMN, SEED)
t_start = time()

# -----------------------------------------------------------------------------
banner("1. Child lifecycle")
# -----------------------------------------------------------------------------
# a_max = 100 for the child, against the parent's 50: the child's asset grid has to cover
# the parent's TERMINAL assets (its own upper bound) plus the child's own accumulation over
# 51 periods. At a_max = 50 check_simulation reported 2.87% of simulated child assets above
# the grid, i.e. the model was being simulated where it had not been solved.
child = ConSavLaborCollege_AR1(Na = C_NA, Nk = C_NK, Nt = C_NT, rho = 1.5,
                               psi_terminal = 1.0, kappa_terminal = 5.0, omega = 0.3,
                               a_max = 100.0, w = 20.0, simN = SIMN, seed = SEED)
solve_model_work!(child)
solve_model_college!(child)
optimal_transfer_work!(child)
optimal_transfer_college!(child)
println("child solved in $(round(time()-t_start, digits=1))s")

banner("2. Child diagnostics")
check_solution(child)
# X4: check_solution allows NaN blanket-wide, so a solver failure INSIDE the feasible
# region would pass. This compares the NaN pattern against the two theoretical masks.
check_feasibility_mask(child)
# C16: the solution can leave the grid where the simulation does not. Reports both the
# off-grid share and how often the k_max ceiling binds.
check_solver_domain(child; throw_on_fail = false)
println()
bellman_residual(child; n_sample = QUICK ? 100 : 500)
# P5: the consistency residual above re-evaluates the STORED policy, so it detects an
# inconsistent value but never a suboptimal one. These two do.
opt_res = bellman_optimality_residual(child; n_sample = QUICK ? 60 : 200)
cont    = continuation_interpolation_test(child; n_sample = QUICK ? 60 : 150)
monotonicity_report(child)
shock_discretization_report(child.p_ar1, child.sigma_p, child.Np)

# -----------------------------------------------------------------------------
banner("3. Parent problem")
# -----------------------------------------------------------------------------
V_child = terminal_value_spline(child; s = 10.0)
parent  = Parent_child_interaction_age_specific_AR1(Na = P_NA, Nk = P_NK, Nhc = P_NHC,
                                                   w = 12.5, simN = SIMN, seed = SEED)
parent.V_child_interp = V_child
diag = solve_model!(parent; verbose = true)
@printf("parent solved; min converged share %.4f\n", minimum(d.converged_share for d in diag))
simulate_model!(parent)
a_p = filter(isfinite, vec(parent.sim_a))
@printf("parent sim: %.1f%% finite, %.2f%% outside the grid\n",
        100*length(a_p)/length(vec(parent.sim_a)),
        100*(count(<(parent.a_min), a_p) + count(>(parent.a_max), a_p))/length(a_p))

# -----------------------------------------------------------------------------
banner("4. Child adulthood, conditional on the parent's terminal states")
# -----------------------------------------------------------------------------
child.sim_a_init .= parent.sim_a[:, parent.T + 1]
child.sim_k_init .= parent.sim_hc[:, parent.T + 1]
_, path_choice, _ = simulate_model_family!(child)
check_simulation(child; throw_on_fail = false)

# -----------------------------------------------------------------------------
banner("5. Tables")
# -----------------------------------------------------------------------------
table_college_work(path_choice, "baseline_college_work_choice";
    caption = "College vs.\\ Work Decisions in the Baseline",
    label   = "college_work_choice",
    note    = "Number and share of individuals choosing each post-family-stage path in the baseline simulation.",
    seed = SEED, simN = SIMN, child_grid = "$(C_NA)x$(C_NK)x$(C_NT)")

table_outcomes([parent], ["Baseline"], "baseline_outcomes";
    caption = "End-of-Family Outcomes: Baseline",
    label   = "baseline_outcomes",
    seed = SEED, simN = SIMN)

br  = bellman_residual(child; n_sample = QUICK ? 100 : 500, verbose = false)
mono = monotonicity_report(child; verbose = false)
sim  = check_simulation(child; throw_on_fail = false, verbose = false)
dom  = check_solver_domain(child; throw_on_fail = false, verbose = false)
mcs  = mc_standard_errors(parent.sim_a[:, parent.T + 1])
table_diagnostics([
    "Minimum converged share (parent solve)" => fmt_num(minimum(d.converged_share for d in diag); digits = 4),
    "Bellman residual, max (child work path)" => @sprintf("%.2e", br.max),
    "Bellman residual, mean"                  => @sprintf("%.2e", br.mean),
    "Monotonicity violations, \$V\$ in assets" => fmt_num(100mono.V_violations; digits = 2) * "\\%",
    "Monotonicity violations, \$c\$ in assets" => fmt_num(100mono.c_violations; digits = 2) * "\\%",
    "Simulated assets below \$a_{\\min}\$"      => fmt_num(100sim.below_a; digits = 2) * "\\%",
    "Simulated assets above \$a_{\\max}\$"      => fmt_num(100sim.above_a; digits = 2) * "\\%",
    "Simulated states non-finite"             => fmt_num(100max(sim.nonfinite_a, sim.nonfinite_k); digits = 2) * "\\%",
    # The share alone reads as a domain failure; the magnitudes are what say it is not.
    "Stored transitions off-grid, assets"     => fmt_num(100dom.assets; digits = 2) * "\\% (max " * @sprintf("%.1e", dom.worst_a) * ")",
    "Stored transitions off-grid, HC"         => fmt_num(100dom.hc; digits = 2) * "\\% (max " * @sprintf("%.1e", dom.worst_k) * ")",
    "\$k_{\\max}\$ ceiling binds"               => fmt_num(100dom.hc_ceiling_binds; digits = 2) * "\\%",
    "Bellman optimality residual, max"        => @sprintf("%.2e", opt_res.max),
    "States improved by re-optimization"      => fmt_num(100opt_res.share_improved; digits = 2) * "\\%",
    "Linear vs.\\ cubic continuation, max \$|\\Delta c|\$" => @sprintf("%.2e", cont.max_dc),
    "Linear vs.\\ cubic continuation, max \$|\\Delta h|\$" => @sprintf("%.2e", cont.max_dh),
    "Mean terminal parental assets"           => fmt_num(mcs.mean; digits = 4),
    "Bootstrap standard error"                => fmt_num(mcs.se; digits = 4),
  ], "numerical_diagnostics";
    caption = "Numerical Diagnostics",
    label   = "numerical_diagnostics",
    note    = "Computed on the solved model and the simulated cohort. Bellman residuals are relative; standard errors are bootstrapped over 200 resamples. The optimality residual re-optimizes sampled states from four starts and compares the maximum against the stored value, so unlike the consistency residual it detects a suboptimal policy. The two off-grid rows report the share of stored transitions leaving the grid and the largest such excursion: these are bounded by the NLopt constraint tolerance and the labor lower bound respectively, so they are float-sized rather than genuine extrapolation. The last two rows re-solve the same states against an interpolating cubic continuation instead of the linear one the solver uses, and report how far the optimal policy moves.",
    seed = SEED, simN = SIMN)

write_manifest(tabpath(); experiment = "full_run", seed = SEED, simN = SIMN,
               child_Na = C_NA, child_Nk = C_NK, child_Nt = C_NT,
               parent_Na = P_NA, parent_Nk = P_NK, parent_Nhc = P_NHC,
               quick = QUICK)

# -----------------------------------------------------------------------------
banner("6. PDF")
# -----------------------------------------------------------------------------
pdf = build_tables_pdf()
pdf === nothing || @printf("PDF: %s\n", pdf)

banner(@sprintf("DONE in %.1f minutes", (time() - t_start)/60))
