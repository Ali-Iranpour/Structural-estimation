# Estimation: the model, the parameters, the moments, and how the SMM runs

The one document on the estimation (merged on 2026-10-02 from the former `SMM.md` and
`ESTIMATION_MEMO.md`, whose texts are in git history and in
`temp/docs_archive_2026-10-02/` on the server). It states the specification the code
implements now, what is estimated and what is calibrated, which moments are meant to
inform which parameters, how the search runs, what has been checked, and what has to
happen before a number can be quoted. Details live in their own records:

- [`SMM_MEMO19.md`](SMM_MEMO19.md): the memo-19 moment vector and the memo-18 technology,
  with the decisions behind them and the items flagged for review;
- [`SMM_COMPOSITION.md`](SMM_COMPOSITION.md): the data's age composition of the pooled
  moments (the export the estimation still needs);
- [`WAGE_RETURN_ANCHOR.md`](WAGE_RETURN_ANCHOR.md): the open question on the child's wage
  return to skill;
- [`../code/smm/README.md`](../code/smm/README.md): every runner flag, workers, resume,
  progress, acceptance.

```bash
uv run --with pandas --with numpy --with pyreadstat python tools/make_smm_targets.py      # freeze targets
julia --project=. code/smm/run_smm.jl --report-only --targets <targets.toml>              # fit at the start, no search
SMM_TEST_FIXTURES=1 julia --project=. code/smm/run_smm.jl --quick --targets <targets.toml> # smoke test on stand-in inputs
```

> **Status, 2 October 2026.** The specification is **memo 19: 20 parameters against 67
> moments** (2 parent P, 59 skill S, 5 college T, 1 wealth W). **It has not been estimated.**
> Two inputs are not yet provided and the code refuses to estimate without them: the data's
> age composition of the pooled S moments, and sd(log AFQT) for the child's wage return to
> skill. The parent's wage process and initial assets are calibrated in Child_Time_Study
> (steps 29 and 30) and read from the target file. The optimizer is the TikTak module of
> `apps/Structural-estimation-v2`, ported on 2026-10-02. Nothing here has been through the
> advisor yet; see [§7](#7-before-an-estimate-the-plan) for the plan and
> [§8](#8-caveats-that-travel-with-any-number) for the caveats.

---

## 1. The model as estimated

Period `t` is the child's age, `t = 1 … 17` (the family stage), followed at 18 by the
transfer and college decision and the child's own lifecycle (`T = 51`, ages 18–68).
Model units: money in 10,000 USD of 2015 per year; time as a share of the 112-hour
non-sleep week. Code: `code/src/parent_family.jl` (family stage), `code/src/child_lifecycle.jl`
(age 18 on), `code/smm/moments.jl` (the pipeline and the moments).

### 1.1 Preferences and the family objective

Parental leisure nets out work and time with the child, $l_{p,t} = 1 - h_{p,t} - \tau_{p,t}$;
the child's leisure nets out parental time, own study and a fixed school schedule,
$l_{c,t} = 1 - \tau_{p,t} - i_{c,t} - s_t$ ($s_t$ from the target file, zero below age 6).
The family maximises (`util_total`)

$$U_t = \phi_1\frac{c_t^{1-\rho}}{1-\rho} + \phi_2\,u_\eta(l_{p,t})
      + \tilde\mu_t\,\phi_3 \ln k_t
      + (1-\tilde\mu_t)\big(\lambda_1 \ln l_{c,t} + \lambda_2 \ln k_t\big)$$

with $u_\eta(l) = l^{1-\eta}/(1-\eta)$ (linearised below a floor), $\phi_1 = \lambda_1 = 1$
(normalisations), $\rho = 1.5$, $\eta = 2$, $\beta = 0.98$. The parents' weight is
$\tilde\mu_t = 1$ before age 6 and $\tilde\mu_t = 1 - \mu_t$ from 6 to 17, where $\mu_t$ is
the **child's** weight: a logistic in age fitted in Child_Time_Study to the
caregiver-reported autonomy index (0.334 at 6 rising to 0.589 at 17), calibrated, read from
the target file (`mu_by_age`).

### 1.2 Skill technology (memo 18; Del Boca, Flinn, Verriest & Wiswall 2026, eq. 4, app. C.1.2)

$$\ln k_{t+1} = \ln R_t + s_{1t}\ln\tau_{p,t} + s_{2t}\ln e_{p,t} + s_{3t}\ln k_t + s_{4t}\ln i_{c,t},
\qquad s_{jt} = \exp(\sigma_{j0} + \sigma_{j1}\,t)$$

$j = 1$ parental time, $2$ money, $3$ persistence (self-productivity), $4$ own study, with
$s_{4t} = 0$ before age 6. Every elasticity varies monotonically with age and all four,
persistence included, are estimated. TFP is DFVW's generalised logistic in age,

$$R_t = d_0 + \frac{d_1 - d_0}{1 + \exp(-d_2\,(t - d_3))}.$$

**No skill shock** ($\sigma_\eta = 0$): DFVW's technology is deterministic (app. fn. 5); the
shock code is kept and is exact at zero. **No parental-schooling shifter** on the time
elasticity (memo 18). Inputs enter in model units. The age-1 draw is
$\ln k_1 = m_0 + m_{BC}\,BC + s_0 z$ (0.629, 0.434, 0.595; Child_Time_Study block K1),
with $BC \sim$ Bernoulli(0.3), the household's BothCollege indicator, constant over $t$.

The elasticities are named `sigma_j_0`, `sigma_j_1` in the code (memo 19 wrote $a_{j0}$, $a_{j1}$;
renamed by decision on 2026-10-02). They are **not** the pre-memo-18 `sigma_*`, which used
$t-1$ and held persistence fixed.

### 1.3 Measurement

The data's skill measure is the raw Letter-Word score: 57 items, each answered correctly with
probability $p(k) = \text{logistic}(L_0 + \ln k)$, $L_0 = -4.595$ (DFVW's normalisation). The
model never draws a score: every moment uses the score's mean $\pi = 57p$ and test-noise
variance $v = 57p(1-p)$ analytically (memo 18 §3.1).

### 1.4 Budget, wages and initial assets

$$a_{t+1} = (1+r)\,a_t + \lambda\,(w_t h_{p,t})^{1-\tau} + y - c_t - e_{p,t}, \qquad a_{t+1} \ge 0,$$

with $r = 0.03$ and the HSV tax ($\lambda = 0.82$, $\tau = 0.18$); $y = 0.1632$ is the
government lump-sum transfer (Daruich & Fernández 2024, set 2026-09-27). The household's
wage is twice the mean parental hourly wage, $w_t = 2\exp(\ln\hat w_t + z_t)\cdot 0.584$,

$$\ln\hat w_t = \beta_0 + \beta_{bc}BC + \beta_{a}t + \beta_{a2}t^2 + \beta_{a2,bc}t^2 BC + \beta_{a,bc}t\,BC,$$

where $t$ = mean parental age − 25 (age 26 = period 1). **Calibrated in Child_Time_Study
`30_wage_process.do`** and read from the target file's `[constants]`:

| | value | | value |
|---|---|---|---|
| $\beta_0$ | 2.85994 | $\beta_{bc}$ | 0.33131 |
| $\beta_a$ | 0.016086 | $\beta_{a2}$ | −0.00026367 |
| $\beta_{a,bc}$ | 0.013519 | $\beta_{a2,bc}$ | −0.00031198 |
| AR(1) persistence $\rho_z$ (annual) | 0.97880 | innovation SD $\sigma_z$ | 0.07544 |
| initial variance of $z$ | 0.09366 | stationary SD (check) | 0.36835 |

The shock $z$ is discretised by Rouwenhorst on 5 nodes, which is exact on the stationary SD
(0.36835) and the autocorrelation (0.97880). Each household starts at the node nearest a draw
from $N(0, 0.0937)$ (the 5 nodes are 0.368 apart, so the discretised initial SD is 0.321
against 0.306). The data's measurement-error variance (0.063) is not part of the model.

**Initial assets** (`29_initial_assets.do`: net worth including home equity of fathers whose
first child was born at 25–27, at the child's age 0–1): with probability 0.2538 the household
starts at $a = 0$ (zero or negative net worth), otherwise $a \sim$ LogNormal(1.1444, 1.3673),
truncated to $[0, 100]$ by redrawing. Simulated: 25.5% at zero, median of the rest 3.13 (31k
USD), 0.56% of positive draws redrawn. Independent of BothCollege in the baseline (the
by-group values are in the target file and unused).

The parent constructor has **no defaults** for the wage process and initial assets: a call
without them fails, so the old values (AR(1) 0.9 / 0.1, $\beta_0$ = 2.799 …, LogNormal(0.296,
1.402), every household at the middle node) cannot come back silently.

### 1.5 Age 18: the transfer, college, and the parents' terminal value

At 18 the family chooses the transfer and the child's college decision with one objective,
$\bar c\,V_{\text{child}} + \mu_h\,V_{\text{parent}}$, where $\mu_h = 1 - \mu_{\text{half}} = 0.346$
is the parents' weight ($\mu_{\text{half}} = 0.654$, the child's autonomy at 18, from the
target file) and $\bar c = (1-\mu_h) + \mu_h\,\omega = 0.723$ with altruism $\omega = 0.2$. The
parents' terminal value is $\kappa_{\text{term}}\ln a$ on retained assets ($\psi = 0$: no
separate terminal weight on the child's skill).

College costs a net tuition of 0.6 per year and a psychic cost per college year
$\kappa_0 + \kappa_\theta(\ln\theta - m_{\text{psychic}}) + \kappa_{\text{ParEd}}BC$, centred at
$m_{\text{psychic}} = 6.687$, the latent mean $\ln k$ at 17 (centred because uncentred
$\kappa_0$ and $\kappa_\theta$ are near-collinear). A taste shock $\varepsilon \sim N(0,\sigma_\varepsilon^2)$
is integrated by Gauss–Hermite (5 nodes). The child's wage is
$\ln w = \ln w_0 + \beta_E E + (\alpha_\theta + \alpha_{\theta E}E)(\ln\theta - m_\theta) + \text{age profile}$,
with $\alpha_\theta = 0.654\cdot\text{sd}(\log\text{AFQT})/0.652$ anchored per SD to Daruich &
Fernández (Table B4) — **refused until sd(log AFQT) is supplied**. The child's government
transfer is 0.144.

---

## 2. The parameters

### 2.1 Estimated: 20

| parameter | box (start) | link | role | moments meant to inform it (Stata `parameter` column) |
|---|---|---|---|---|
| `phi_2` | [0.01, 20] (0.196) | log | weight on parental leisure | P: hours worked; the budget |
| `phi_3` | [0.05, 20] (1.53) | log | parents' weight on skill | S8 means and SDs of the inputs; S9 money over income ("money preference – budget share") |
| `lambda_2` | [0.05, 100] (13.9) | log | child's weight on skill | S8 (own study) |
| `sigma_1_0`, `sigma_1_1` | [−4, 1], [−0.4, 0.1] (−0.631, −0.115) | level | parental-time elasticity | S7 corr(time, 5-year LW change) |
| `sigma_2_0`, `sigma_2_1` | [−12, −1], [−0.3, 0.3] (−7.15, 0.072) | level | money elasticity | S7 corr(money, LW change) |
| `sigma_3_0`, `sigma_3_1` | [−3, 0], [−0.1, 0.1] (−0.254, 0.005) | level | persistence | S3 SD of LW; S5 5-year autocorrelations |
| `sigma_4_0`, `sigma_4_1` | [−12, −1], [−0.2, 0.6] (−6.60, 0.271) | level | own-study elasticity | S7 corr(study, LW change) |
| `d_0` … `d_3` | [0.01, 10] ×2, [−4, 4], [−20, 20] (5.6, 5.6, 1, 5.3) | level | TFP logistic | S1 mean LW by age; S4 mean 5-year change |
| `kappa_0` | [−3, 1] (−0.357) | level | psychic cost at mean ability | T: BA share |
| `kappa_theta` | [−3, 0] (−0.183) | level | ability gradient of the cost | T: LW gap at 17, BA minus no BA |
| `kappa_ParEd` | [−1, 0.5] (−0.108) | level | BothCollege shift of the cost | T: BA share by BothCollege |
| `kappa_terminal` | [0.5, 40] (8.79) | log | weight on retained assets | W: median net worth at first-child ages 21–22 |
| `sigma_eps` | [0.1, 2] (1.14) | log | taste-shock SD | T: residual variance of BA on BC and LW at 17 |

S6 (correlations of each input with the LW level) inform the elasticity levels jointly. The
technology starts are DFVW Table 7 (TFP started flat, see [§8](#8-caveats-that-travel-with-any-number));
the others are the code's starting values (`smm_start`), with $\kappa_\theta$ converted per SD to
the new skill units.
**These are joint identification arguments, not one-to-one assignments**: valuation and
technology both move inputs and therefore skill, and a moment's presence does not prove
separate identification ([§2.3](#23-identification-what-is-known)). Memo 18 gives only the TFP
box; the `sigma_j` boxes were set in the memo-19 implementation.

### 2.2 Calibrated or fixed

| quantity | value | source |
|---|---|---|
| child's weight $\mu_t$, ages 6–17; $\mu_{\text{half}}$ | 0.334 … 0.589; 0.654 | target file (autonomy index, Child_Time_Study 21) |
| age-1 skill draw $m_0, m_{BC}, s_0$ | 0.629, 0.434, 0.595 | target file (block K1) |
| $L_0$; $m_{\text{psychic}}$ | −4.595; 6.687 | target file |
| wage profile, AR(1), initial dispersion; initial assets | §1.4 | target file `[constants]` (blocks 29/30) |
| school schedule $s_t$ | median school time by age / 112 | target file |
| $\phi_1 = \lambda_1 = 1$; $\psi = 0$; $\sigma_\eta = 0$ | — | normalisations / memo 18 |
| $\beta, \rho, \eta, r, \lambda, \tau$ | 0.98, 1.5, 2, 0.03, 0.82, 0.18 | code |
| parent $y$; child $y$; net tuition; $\omega$ | 0.1632; 0.144; 0.6; 0.2 | code (2026-09-27, kept by Ali in the memo-19 merge) |
| BothCollege share | 0.3 | code — **open**: the data show 0.26 (two-parent) / 0.21 (kpe frame) |
| asset grid; skill grid; shock nodes | 30 nodes to 1M USD; 30 log-spaced nodes, $\ln k \in [-2, 10]$; 5 | code |

### 2.3 Identification: what is known

**Counts establish nothing**; the residual Jacobian (columns scaled to a full-box move, across
step sizes, simulation sizes and seeds) is the evidence, and the recovery test is the
practical check.

- **Jacobian** (memo 19, at the recovery test's pilot point, stand-in inputs): rank 20/20,
  condition number 1.35e4; the weakest direction is `sigma_eps` with `d_2`.
- **Recovery test** (`tools/test_param_recovery.jl`, 2026-10-02, merged code at `81884e3`,
  stand-in composition and wage loading, test grids; start displaced 5% of each box). **Neither
  run recovered the parameters**:
  - 12 technology parameters free: Q 21,373 → 2.50 after 1,522 evaluations (Nelder–Mead hit its
    cap); the eight elasticities came back within 2.6% of their boxes, the TFP did not
    (`d_2` 18.8%, `d_0` 16.6%);
  - all 20 free: Q 17,897 → 44.6 after 2,058 evaluations; `d_3` 22.6%, `d_2` 21.6%,
    `sigma_eps` 12.2%, `phi_3` 10.6%.
  Q did not reach zero, so the searches did not finish; that alone does not show
  non-identification, but the misses sit in the directions the Jacobian marks as weakest.
  The test ran before the new wage and asset calibration and must be rerun.
- **Skill dispersion** (memo 18 §4, `output/diagnostics/2026-10-02_dispersion/`): with no shock
  the SD of $\ln k$ falls from 0.64 at age 1 to 0.04 at 17 at the pilot point, against ≈0.65 in
  the data. Raising persistence toward one (TFP rescaled to keep the level) restores the SD at
  ages 12–17 (5.06 against 5.19 at persistence 0.951) but lowers the SD at 3–7 further and
  overshoots the 5-year correlations from base ages 3–7. Whether the estimator can close this
  is the first empirical question of an estimation; if it cannot, it is a specification
  question for the advisor (the shock, or the initial dispersion).

---

## 3. The moments

67 targeted rows and 32 untargeted diagnostics, all built in Child_Time_Study
`28_smm_moments.do` and passed through unchanged by `tools/make_smm_targets.py`
(`Input/SMM_Moments.csv`, `SMM_VCov.csv`, `SMM_Constants.csv` → `targets.toml`).

| block | rows | what |
|---|---|---|
| P | 2 | mean consumption (incl. housing, less money investment, p99 cap) and mean work hours per parent, equal-age means over 1–17 |
| S1 | 15 | mean raw LW at each age 3–17 |
| S3, S4, S5 | 4, 2, 3 | SD of LW by age group; mean 5-year LW change; 5-year autocorrelation (base ages 3–7, 8–12, 3–12) |
| S6, S7 | 3, 9 | correlation of each input (parental time, own study, money) with the LW level, and with the 5-year LW change |
| S8, S9 | 22, 1 | means and SDs of each input by age group; money over pre-tax labour income |
| T | 5 | BA share; BA share by BothCollege; LW gap at 17 (BA minus no BA); residual variance of BA on BC and LW at 17 |
| W | 1 | median family net worth (incl. home) at first-child ages 21–22, fathers 25–27 at first birth |

**Parental time is active time.** The time moments use `taup = parent_Act / 112` (active time
with any parent, from the CDS diary), matched to the model's $\tau_{p,t}$; the untargeted
by-age file uses the same definition since 2026-10-02. Note that `parent_Act` is time with
*any* parent seen from the child, so it does not close each parent's own time budget
(117.3 / 124.6 hours instead of 112); the overlap is about 5 hours a week, against 21 for the
former active-plus-nearby measure.

**Pooled moments are mixtures over the data's ages.** A pooled S moment mixes ages (and, for
pairs, base ages) in the data's proportions, and the model counterpart uses the same weights
from the target file's `[composition]` tables. **Not yet exported**; `load_targets` refuses a
target file without them ([`SMM_COMPOSITION.md`](SMM_COMPOSITION.md) specifies the frames).

### 3.1 How a correlation is used as a moment

Fifteen of the S rows are correlations: S5, the 5-year autocorrelation of Letter-Word (3 rows);
S6, each input (active parental time, own study, money) with the LW level (3); S7, each input
with the 5-year LW change (9). In the data, each is one correlation over all observations of its
frame: S6 parental time, for example, pools every child-wave aged 3–17. A correlation is
unit-free, so it does not depend on how skill or the inputs are scaled; its standard error comes
from Stata's joint bootstrap like every other row.

**The model rebuilds the same pooled statistic** (`s_pooled`, `moments.jl`). The data's
observations fall into composition cells $c$ (an age; for pairs, a base age and an end age; for
money, also an even or odd CDS wave) with counts $w_c$, read from the target file. The model
simulates every household at every age, so it forms the moments of each cell and mixes them with
the data's weights, including the between-age term that pooling creates:

$$\mu(y) = \sum_c w_c\,\bar y_c, \qquad
\mathrm{Cov}(y,z) = \sum_c w_c\big[\mathrm{cov}_c(y,z) + (\bar y_c - \mu(y))(\bar z_c - \mu(z))\big]$$

(weights normalised to one; population moments, $1/N$). The between-age term matters because both the inputs and LW change with age: a pooled correlation
is not the average of the per-age correlations, and the data's age mix decides how much each age
counts.

**Test noise enters only where it belongs.** The model uses each child's expected score
$\pi = 57p$ and the binomial variance $v = 57p(1-p)$, never a drawn score. Measurement noise is
independent of the inputs and across test waves, so it adds to variances and not to
covariances:

$$\mathrm{corr}(x, LW) = \frac{\mathrm{Cov}(x,\pi)}{\sqrt{\mathrm{Var}(x)\,(\mathrm{Var}(\pi) + \bar v)}},
\qquad
\mathrm{corr}(x, \Delta LW) = \frac{\mathrm{Cov}(x,\Delta\pi)}{\sqrt{\mathrm{Var}(x)\,(\mathrm{Var}(\Delta\pi) + \bar v_a + \bar v_{a_2})}},$$

with $\Delta\pi = \pi_{a_2} - \pi_a$ between the base age $a$ and the end age $a_2$ of each pair, and the S5 autocorrelation has the noise of each wave in its own variance only. This is what the
data's correlations contain: noise attenuates them, and the model reproduces the attenuation
rather than comparing noise-free model correlations with noisy data ones.

**Inputs**: parental time and own study are the model's choices at the cell's age. Money at an
odd CDS wave (1997, 2007) is, in the data, the mean of the two adjacent even PSID years, so the
model uses the mean of ages $a-1$ and $a+1$ (age 16 alone at 17).

**What they carry.** S7, the association between an input and the *growth* of skill that
follows, is the main information on that input's elasticity; S5, how strongly skill persists
over 5 years, informs persistence ($\sigma_{3\cdot}$) together with the SDs (S3); S6, the
association with the *level*, mixes technology with the fact that inputs respond to skill and
income, and informs the elasticity levels jointly. These are the intended mappings
([§2.1](#21-estimated-20)), not proofs of identification.

**The objective** is $Q(\theta) = \sum_j (m_j - \hat m_j)^2/\text{se}_j^2$, the diagonal
inverse-variance weight from the joint bootstrap covariance Stata exports (`SMM_VCov.csv`);
the residual vector every tool must agree on is $(m_j - \hat m_j)/\text{se}_j$ in
`SMM_MOMENTS` order. The diagonal weight is a first-stage choice: the full covariance is in the
target file for standard errors and diagnostics, and whether an efficient second stage moves
the estimate is untested.

---

## 4. How the search runs

### 4.1 TikTak

TikTak (Arnoud, Guvenen & Kleineberg 2022) is a structured multistart: evaluate $Q$ at $N$
Sobol' points over the box (pre-testing), keep the best $N^*$ as seeds, and run local searches
from mixtures of the next seed and the best point so far,
$S_j = (1-\theta_j)s_j + \theta_j p^*$ with $\theta_j = \min(\max(0.1, \sqrt{j/N^*}), 0.995)$, so
the search moves from exploration to exploitation on its own; a final BOBYQA polish follows.
$Q$ need not be differentiable, and the expensive part parallelises.

The implementation is the TikTak **module** `code/src/TikTak/` (v2's, ported 2026-10-02, plus the two
v1 rules below, version 2.3.0-dev; `code/src/tiktak.jl` is the adapter), driven by `code/smm/run_smm.jl`:

- **Pre-testing and restarts run on worker processes** (one Julia thread each: NLopt is not
  thread-safe). Restarts are asynchronous: by default (`--bootstrap immediate_mixed`) every
  local worker starts a restart at once, each mixed with the best point committed so far; the
  number of local workers is about $\sqrt{N^*}$. `--local-mode serial` keeps the published
  sequential algorithm.
- **Checkpoint**: `tiktak_state.toml`, versioned and checksummed, written after seed selection,
  every restart, a pause and the polish; `--resume` verifies the objective and the optimizer
  identity field by field and refuses any change of box, restart count or design. Runs written
  before the port cannot be resumed (`--init-from` warm-starts from them).
- **Presets** (`--preset smoke|integration|pilot|production`) record what a run is for; five
  restarts are a development test, not an estimate.
- `tools/reopt.jl` re-optimises from given points with parameters fixed (`--fix`), run-specific
  boxes (`--bounds`) or extra rows (`--extra-moments`), with its own checked objective identity.

**Departures from the paper.** Nelder–Mead locally (`ftol_rel` 1e-3; DFNLS has no maintained
Julia binding), plus absolute stopping rules so a search near $Q = 0$ can stop; the starting
point is forced into the seed pool (it competes like any Sobol' point); a short BOBYQA
refinement on the full grid when the search ran on a coarser one; and the asynchronous
parallel restarts above (the reference repository's way to scale; with one local worker it is
the sequential algorithm).

**Two v1 rules (2026-10-02, Ali; not yet in v2, to be ported in a separate step).** Both came from
the memo-19 pilot (`output/diagnostics/2026-10-02_pilot/`) and were checked against the authors'
Fortran (`apps/Structural-estimation-v2/archive/TikTak_serdarozkan_reference`):

- *A penalised mixed start falls back to the restart's own seed.* $S_j$ mixes two valid points,
  but the valid set is not convex (3.8% of the box is valid), so $S_j$ can be penalised. Nelder–Mead
  then sees a flat $10^{12}$ around its start and stops after $n+1$ evaluations with `FTOL_REACHED`:
  a lost restart (4 of 20 in the pilot's arm B). The seed $s_j$ is valid by construction and its
  value is known, so the restart starts there; `restarts.csv` marks it (`start_fallback`). The
  authors' code has no such case (its test objectives are defined everywhere).
- *A known start value is not recomputed.* The module evaluated each restart's start to record
  "start Q" and then Nelder–Mead evaluated the same point again as its first step (every restart
  showed evaluations 1 and 2 with the same Q); restart 1 re-evaluated its seed, the polish its
  incumbent. The solver's first call now returns the known value (the objective is deterministic,
  so it is the same number), and restart 1 uses its seed's pre-tested value. The authors' code
  likewise evaluates a start only inside its solver (`completeSearch`, `runAmoeba`). On the
  synthetic baseline the search path is bit-identical to the 2026-09-27 code and the count lower by
  exactly one per restart, one more for restart 1 and one for the polish.

Both change the optimizer identity: a run checkpointed before them resumes only with
`--allow-optimizer-change` (recorded); a job saved in flight before them replays exactly as it ran.

### 4.2 One evaluation

`run_pipeline` (`moments.jl`): solve the child's lifecycle and the age-18 problems (the stages
no estimated child parameter reads are cached as solution arrays under a complete key, the rest
rebuilt each evaluation; bit-identical to a full re-solve) → construct the terminal value →
solve and simulate the parents → hand the parents' assets, skill and BothCollege to the
children → resimulate the children → compute the 67 moments. Common random numbers throughout:
every stream is seeded from one `seed` (initial assets, BothCollege, skill, wage-shock path,
initial wage shock on its own stream `seed + 5`).

**What is not scored.** A draw whose elasticities reach one at some age (an explosive
technology) is rejected before solving; a simulation that leaves the model's domain, a
non-finite moment, or an expected solver failure returns the finite penalty $10^{12}$, counted
by reason. A `DomainError` or `InexactError` counts as a model failure only when it comes from
the model's own solver files; anything else is re-thrown, so a coding error cannot become a
converged run. **The penalty rate is high on memo 19**: at the production grids on the real inputs
15 of 400 Sobol' points (3.8%) are valid (`output/diagnostics/2026-10-02_valid_share/`): 53% break
the elasticity rule (rejected before a solve), 39% send no child to college (skill collapses), 3%
send every child. Ali decided on 2026-10-02 to keep the boxes and draw until enough points are
valid (`--sobol-valid N`); a solved draw costs about 11 s.

### 4.3 Test-only stand-ins

`SMM_TEST_FIXTURES=1` runs the objective on labelled stand-ins while the two inputs are
missing: a composition spread from the S1 counts and a placeholder wage loading
(`tools/smm_test_fixtures.jl`). It is fenced: the runner refuses it with `--preset pilot` or
`production`, the specification name gains `_TESTFIX` (so a test run never resumes into a real
one), and `run_record.toml` records it. Results on stand-ins are tests, never estimates.

---

## 5. Validation status (2 October 2026, branch `merge/port-memo19`)

| check | result |
|---|---|
| optimizer, synthetic (`tools/test_tiktak.jl`), module 2.3.0-dev (the two v1 rules, §4.1) | 479/479, incl. the new `start_rules` group 11/11 (evening) |
| runtime projection; reopt identity | 78/78; 38/38 |
| merged objective = memo-19 branch (memo 19's own values passed back in) | equal to the last digit at two points |
| runner on memo-19 code (stand-ins): resume, start, geometry, penalties, reopt | 39/39, 48/48, 14/14, 19/19, 20/20 |
| runner integration (stand-ins, module 2.2.0) | 30/30 (17:47) |
| runner tests on the REAL inputs (module 2.2.0; `temp/2026-10-02_merge_checks/real_inputs_suite/`) | resume, reopt integration, synthetic, projection, reopt identity pass; penalties 18/19, start 13/30, geometry 3/13, integration 25/27 fail from ONE cause: they start from the default point with 5-9 plain draws, all invalid on the real inputs -- to be given a valid start (§7) |
| module 2.3.0-dev on the real objective | resume 39/39; integration 30/30 (21:56, `output/diagnostics/2026-10-02_tiktak23_integration/`); a serial smoke shows the fallback and no repeated start evaluation |
| parent calibration on the built model | 25.5% at zero assets; Rouwenhorst SD 0.36835, autocorrelation 0.97880 |
| `selftest.jl`, `test_smm_tas.jl`, `test_smm_own_study.jl`, `test_smm_baseline.jl`, `test_hc_process_shock.jl` (check 7), `jacobian.jl`, `profile_param.jl`, `sensitivity.jl`, `grid_sensitivity.jl`, `check_jacobian_rank.jl` | **not yet rewritten for memo 19** (they pin the old specification or miss the wage loading) |

---

## 6. Robustness exercise to run after the first estimate

From the former estimation memo, adapted to memo 19 (proposed, not run). Following the
identification diagnostics in Sections 4.4 and 5.4 of *structural-robustness-Jan26.pdf*: move
one target at a time over a small range (in standard errors of the moment), hold the others,
the weights, the draws, the boxes and the solver settings fixed, **re-estimate all 20
parameters jointly** at each perturbation, and plot each estimate against the perturbed target,
recording $Q$, residuals, solver failures and binding bounds. With 67 targets this is long:
start with the T and W rows, the P rows, and one row per S block. It measures how the estimator
reallocates a change in the data across parameters; it is a sensitivity diagnostic, not a
confidence region. `code/smm/sensitivity.jl` implements it and needs the memo-19 update first.

---

## 7. Before an estimate: the plan

**A. Inputs (the code refuses without them)**
1. The composition tables: run block C of `28_smm_moments.do`, merge with Child_Time_Study
   commit `5aa297f`, copy the CSVs, regenerate the targets.
2. sd(log AFQT) for $\alpha_\theta$ ([`WAGE_RETURN_ANCHOR.md`](WAGE_RETURN_ANCHOR.md)); then set it
   and re-fit $\ln w_0$ to keep the mean child wage.
3. The remaining memo-19 decisions: the p99 caps, the BothCollege share, parents' mean
   schooling (13 years assumed for the time-elasticity start), the `sigma_j` boxes.
4. The advisor's sign-off on the memo-18/19 technology, $\sigma_\eta = 0$ and the calibration.

**B. Numerics**

5. The asset grid: 30 nodes with 12 (40%) below 150k USD and a 1M top, against the advisor's
   rule (at least 60% below 150k, top 1.5M; v2's G14). The new initial assets (a quarter at zero,
   median 31k) make the bottom matter more.

**C. Validation**

6. Rewrite the guards and tools listed in [§5](#5-validation-status-2-october-2026-branch-mergeport-memo19) for memo 19.
7. Measure the valid share of random draws at the production grids; tighten boxes if needed.
8. Rerun the recovery test with the real inputs and the new calibration, finish it with a
   BOBYQA polish, take the Jacobian across step sizes and seeds; if the TFP logistic stays weak,
   fixing its slope is a decision for the advisor.

**D. Estimation**

9. Smoke and pilot runs (time per evaluation, evaluations per restart).
10. A budget for approval, the production search through a driver, a converged polish, then
    acceptance, standard errors, identification and the [§6](#6-robustness-exercise-to-run-after-the-first-estimate) exercise.

---

## 8. Caveats that travel with any number

1. **Not through the advisor.** Memo 18/19, $\sigma_\eta = 0$ and the 2026-10-02 calibration are
   specification decisions; results built on them do not circulate until approved.
2. **Analytic LW at 17** for `kth_lw17_gap` and `m_eps` (population moments instead of drawn
   scores) is flagged in memo 19 for a second opinion.
3. **TFP in model units is not monotone.** DFVW's TFP converted to model units runs 6.7, 4.8,
   6.5, 5.1, 6.6 at ages 1, 4, 7, 13, 17; the logistic cannot follow it, so it starts flat at 5.6.
4. **Skill dispersion collapses without a shock** ([§2.3](#23-identification-what-is-known)).
5. **The parents over-invest time at DFVW's elasticities** in the pilot (0.48 of the week at
   ages 3–5 against 0.34), with $\phi_3$ at its floor: DFVW's persistence of ~0.8 makes early
   investment far more valuable than the former 0.41.
6. **Parental time is time with any parent** (§3): ~5 hours of overlap with the per-parent
   budget, which $\phi_2$ absorbs.
7. **The consumption profile is not targeted**, only its mean; its slope is the Euler equation
   at the calibrated $\beta = 0.98$, which recovers about half of the data's rise over the family
   stage (about 0.989 would match it).
8. **BothCollege share** 0.3 in the model against 0.21–0.26 in the data; **initial assets are
   independent of BothCollege** by decision.
9. **The initial wage shock's discretised SD** is 0.321 against 0.306 (5-node spacing).
10. **No standard errors and no sensitivity analysis** exist for any memo-19 estimate; the
    diagonal weight's cost is unmeasured.

---

## 9. How the specification got here

| date | parameters / moments | what changed |
|---|---|---|
| 2026-08-28 | 6 / 6 | just-identified parent block |
| 2026-09-06 | 9 / 10 | HC in the data's units; `R_0`, `phi_3`, `lambda_2`, `sigma_4_0`; time-invariant preferences; $\psi = 0$ |
| 2026-09-09 | 10 / 10 | own study only, fixed school schedule; `sigma_4_1` estimated |
| 2026-09-10 | 14 / 17 | four `kappa_*` and seven TAS moments; inverse-variance weights |
| 2026-09-11 | 16 / 17 | `sigma_eta` (skill shock) and `sigma_eps`; exp16b fit (Q 61.80) promoted to the baseline on 09-12 |
| 2026-09-27 | 15 / 16 | aligned with v2 (values and targets): `kse_w_gap` dropped, `mean_a_p_late` as the wealth target, `sigma_eps` fixed at 2.0, active parental time, $\mu$ 0.8, $\omega$ 0.2, parent $y$ 0.1632, child $y$ 0.144, net tuition 0.6 |
| 2026-10-02 | **20 / 67** | memo 18 technology (DFVW: all four elasticities on age, persistence estimated, logistic TFP, no shock, analytic LW measurement), memo 19 moments from Stata; $\mu_t$ from the autonomy data, $\mu_{\text{half}}$ 0.654; `sigma_eps` estimated again; kept from 09-27: $\omega$, both $y$, net tuition; elasticities named `sigma_j_k`; wage process and initial assets calibrated (29/30); TikTak module ported from v2 |

Estimates from different rows are not comparable: different targets, weights and objectives.
The former documents — the 2026-09-06 parent-block memo with the $\mu_1$/$\sigma_{41}$ question,
the TikTak write-up, the 16/17 moment definitions, the exp16b results and the 2026-09-11 work
order — are in git history and in `temp/docs_archive_2026-10-02/`.
