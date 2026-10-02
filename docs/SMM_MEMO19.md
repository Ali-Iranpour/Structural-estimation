# The memo-19 moment vector and the memo-18 technology in the Julia model

2026-10-02. Implements the model side of Child_Time_Study memo 19, section 8, together with
the technology of memo 18. That technology is Del Boca, Flinn, Verriest & Wiswall (2026),
"Parenting with Patience", JPE 134(1), eq. (4) and online appendix C.1.2 / C.1.5. Nothing
here is estimated on data yet. Two inputs are missing (§3), and the code refuses to run an
estimation without them.

## 1. What the code now does

**Technology** (`code/src/parent_family.jl`, `PARENT_DEFAULTS`):

    ln k_{t+1} = ln R_t + s1_t ln tau_p + s2_t ln e_p + s3_t ln k_t + 1{t>=6} s4_t ln tau_c
    s_jt = exp(a_j0 + a_j1 t),  t = child age
    R_t  = d_0 + (d_1 - d_0) / (1 + exp(-d_2 (t - d_3)))

- **No shock.** `sigma_eta = 0`, exact; DFVW app. footnote 5.
- **No parental-schooling shifter.** Memo 18.
- **Inputs in model units.** Time is a share of 112 hours and money is in 10k USD/yr.
- **Starting skill draw.** ln k₁ = 0.629 + 0.434·BothCollege + 0.595·z, from targets.toml (K1).

**Measurement.** Raw LW ~ Binomial(57, logistic(−4.595 + ln k)) at every age. The model
never draws a score: every moment uses π = 57p and v = 57p(1−p) (memo 18 §3.1).

**Skill grid.** One log-spaced grid, ln k ∈ [−2, 10] with 30 nodes, shared node for node by
the parent's `hc_grid` and the child's `k_grid` (`hc_log_grid`). The child's terminal-value
spline is now fitted in ln k:
- A cubic spline in levels across five orders of magnitude made the marginal value of skill
  oscillate: dV/d ln k ranged 1.26–1.95 where the surface is smooth, 1.52 → 1.73.
- The parent's own PCHIP continuation stays in levels. On the same grid its dV/d ln k is flat
  to ±1%.

**Bargaining weights.**
- Parent's weight is 1 − μ_t at ages 6–17, from `mu_by_age`, and 1 before.
- At the half period the child module's parent weight is 1 − `mu_half` = 0.346. It drives
  both the transfer and the enrolment comparison.
- The μ₀ / μ₁ defaults are gone.

**Wage and psychic cost.**
- m_θ = m_psychic = 6.687, the latent mean ln k at 17.
- α_θ is anchored per SD to Daruich–Fernández Table B4: α_θ = 0.654·sd(log AFQT)/0.652, and
  α_θE likewise with 0.322. This refuses to run until sd(log AFQT) is supplied.
- κ_θ's starting value is converted per SD: −3.619 × 0.0329/0.652 = −0.183.

**Moments** (`code/smm/moments.jl`). All 67, in the target file's covariance order:
- **P.** `mean_c_p` and `mean_h_p` as equal-age means over t = 1..17.
- **S.** Memo 18 §3: S1 per age, and the 44 pooled rows as mixtures over the data's
  composition, with test noise only in the variances.
- **T.**
  - `k0_complete` is the college share.
  - `kpe_bc0_c` / `kpe_bc1_c` split it by BothCollege.
  - `kth_lw17_gap` and `m_eps` use the score at 17, analytically (§4).
  - `m_eps` is the population residual variance of college on [1, BC, LW17].
- **W.** `kterm_med22` is the median of a − tr at the half period.

**Estimated set: 20 parameters.**
- 12 technology: a_10 … a_41 and d_0 … d_3.
- φ2, φ3, λ2.
- Five child: κ0, κθ, κ_ParEd, κ_terminal, σ_ε.

## 2. Decisions behind it (user, 2026-10-01 / 02)

| item | decision |
|---|---|
| role of DFVW Table 7 | starting values and search boxes; all 12 technology parameters re-estimated |
| input units in the technology | model units (memo 18 as written) |
| parental-schooling shifter on s1 | none (memo 18) |
| data composition of pooled S moments | exported by Stata; Julia refuses without it |
| μ_half | 1 − 0.654 = 0.346 as the parent's weight, in both transfer and enrolment |
| κ_terminal timing | median a_term at 18; the 21–22 gap is documented, not adjusted |
| α_θ units | anchored per SD to D&F; needs sd(log AFQT) |
| LW at 17 for `kth_lw17_gap`, `m_eps` | analytic; **flagged for a second opinion** |

## 3. Not provided: inputs the estimation needs

1. **The data's age composition of the pooled S frames.** `docs/SMM_COMPOSITION.md` specifies
   the twelve frames, the export names and the generator pass-through. `load_targets`
   refuses a file without them.
2. **sd(log AFQT raw score)**, for the α_θ anchoring. Neither Daruich–Fernández nor Colas et
   al. report it. `child_wage_config` refuses until `CHILD_DEFAULTS.sd_log_afqt` is set.
   Once it is, `lnw0` has to be re-set to keep the mean child wage at its old level,
   14.464, measured on exp16b (`lnw0_for_mean_wage`).
3. **The p99 caps** on money ($47,004) and on the S9 ratio (0.443) are not in the target
   file. The model applies none.
4. **DFVW's mean parental schooling**, used to pool their mother/father time elasticities
   into one start for s1. I used 13 years. Going from 12 to 16 years moves a_10 by 0.10.
5. **Search boxes for a_j.** Memo 18 gives only the TFP box; the a_j boxes in `SMM_PARAMS`
   are mine.
6. **The BothCollege share.** The model draws Bernoulli(0.3). The data show 26% in the
   two-parent sample and 21% on the kpe frame. It is not in the target file and is
   unchanged.

## 4. Flagged for review

- **Analytic LW at 17.** The gap uses mean π among college-goers minus non-goers. The
  regression uses the population second moments of (college, BC, LW17), with the binomial
  variance in E[LW17²]. Against a literal OLS on drawn scores (400 draws per child):
  - `m_eps`: 0.2211 analytic against 0.2215 literal;
  - the gap: identical in expectation.

  A single literal draw adds noise with SD 0.15 points to the gap at 1,000 children (data SE
  0.55), and steps to the objective.
- **TFP in model units.** In model units DFVW's TFP is not monotone in age: 6.7, 4.8, 6.5,
  5.1, 6.6 at ages 1, 4, 7, 13, 17. The generalised logistic cannot follow it; the best fit
  in the memo-18 box is flat at 5.6. This is the cost of the model-units choice.
- **Skill dispersion collapses.** This is the memo-18 §4 risk, now measured. With no shock
  and persistence of 0.78–0.84, the age-1 dispersion decays and inputs add little.
  - At the pilot point, SD(ln k) runs from 0.64 at age 1 to 0.04 at 17; the data hold it
    near 0.65.
  - So the college choice barely depends on skill: the LW gap at 17 is 0.04 points against
    3.35.
  - The level-sample SD of LW at 12–17 is mostly test noise: 3.4 against 5.2.
  - The 5-year autocorrelation from base ages 8–12 is 0.40 against 0.72.

  Whether the estimator can close this by raising persistence toward one is an empirical
  question for the first estimation. If it cannot, it is a specification question for
  Sahber: the shock, or the initial dispersion.
- **Parents over-invest time at DFVW's elasticities.** The pilot pushed φ3 to its floor of
  0.05, and parental time at ages 3–5 still came out 0.48 of the week against 0.34. DFVW's
  persistence of 0.8 makes early investment far more valuable than under exp16b's 0.41.

## 5. Verification

- **Analytic binomial against brute force.** Each simulated child is given 200 drawn scores,
  pooled with the composition weights. All 21 pooled score moments (S3–S7) agree within
  Monte Carlo error: at most 0.006 LW points, and correlations within 0.002.
- **Determinism.** With common random numbers Q is exactly 0 at the generating point.
- **Local identification at the test point.** The 67 × 20 scaled Jacobian (central
  differences, per full-box move) has rank 20/20 and condition number 1.35×10⁴. The weakest
  direction is σ_ε with d_2, the TFP slope.
- **Parameter recovery.** `tools/test_param_recovery.jl`, results in
  `output/smm_runs/2026-10-02_100135_recovery/`. RESULTS: see §6.

## 6. Parameter recovery

(filled in below from the run logs)

## 7. Not updated, and why

The following pin the exp16b specification (17 moments, R_0, sigma_eta, the tertiles) and
fail or mislead until they are rewritten for this one:
- `code/smm/selftest.jl`, `jacobian.jl`, `profile_param.jl` (its default `--param
  sigma_2_1`), `standard_errors.jl`, `sensitivity.jl`, `grid_sensitivity.jl`;
- `tools/test_smm_*.jl`, `tools/test_hc_process_shock.jl`, `tools/check_jacobian_rank.jl`.

`code/run_all.jl` and the notebook build a child with the default wage loading, so they stop
at the α_θ refusal until sd(log AFQT) is supplied.
