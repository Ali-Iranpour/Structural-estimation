# SMM Moments - Codebook

**Merged 2026-09-10.** This file consolidates what were two codebooks: the PSID/CDS
parent-block documentation (`19_smm_moments.do`) and the TAS-linked child-block
documentation (`23_smm_tas_moments.do`, which requires `21` and `22` first). Neither
do-file overwrites the other's outputs, and neither block's sample definition applies to
the other, so the two are kept as **separate parts below rather than interleaved**. Read
Part A for anything about parents and childhood; Part B for anything about college
outcomes, achievement tertiles, parental education or terminal wealth.

Do not edit by hand - rerun the do-file that owns the part you want to change, then
regenerate the targets (see *Generation workflow* below).

## The two sample frames are different objects

This is the single most important thing in the merged file, and it is the reason the parts
were not interleaved.

| | Part A - parent block | Part B - TAS child block |
|---|---|---|
| Source | `19_smm_moments.do` | `23_smm_tas_moments.do` |
| Micro file | `SMM_Moments_Micro.dta` | `SMM_TAS_Micro.dta` |
| Unit of observation | one child-**year** | one **child** |
| Rows | 17,791 | 4,248 |
| Frame | PSID one-child panel, CDS left-joined | TAS respondents linking to a CDS **or** PSID study child |
| Ages | child 0-17 | outcomes by age 25 |
| Weighting | unweighted (`CHILD_WT` available) | unweighted, **by decision** - see Part B |
| Cluster key | `Fam_id` | `famclust` |

**The cluster key is the same object in both**: the PSID 1968 family interview number,
ER30001. That is what makes a joint covariance possible. **488 families appear in both
frames** (1,794 in Part A restricted to child ages 1-17, 1,481 in Part B, 2,629 distinct
across the two). The blocks are therefore *not* independent and must not be treated as
such - though see *Measured overlap* below for how large the dependence actually turns out
to be.

## Files: what is supplied and what is not

`tools/make_smm_targets.py` reads only the files marked **supplied**. Everything else is
documented here because the source do-file produces it, not because this repository has it.

| File | Status | Contents |
|---|---|---|
| `SMM_Moments_ByAge.dta` / `.csv` | **supplied** | one row per child age 0-17 |
| `SMM_Moments_ByAge_Cohort.dta` | **supplied** | the same, restricted to `cohort_dad2530` |
| `SMM_Moments_Micro.dta` | **supplied** | one row per child-year; every Part A input |
| `SMM_Assets_ByChildAge.dta` / `.csv` | **supplied** | assets by child age |
| `SMM_TAS_Moments.dta` / `.csv` | **supplied** | moment, block, estimate, SE, N, clusters |
| `SMM_TAS_Micro.dta` | **supplied** | one row per child; every Part B input |
| `SMM_TAS_VCov.dta` / `.csv` | **NOT supplied** | joint cluster-robust covariance |
| `SMM_TAS_Gaps.dta` | **NOT supplied** | top-minus-bottom and parental-education differences |
| `SMM_TAS_TermWealth.dta` | **NOT supplied** | kappa_term means, medians, quartiles, negatives, timing |
| `SMM_TAS_Weighted.dta` | **NOT supplied** | full-frame vs weighted-subset rates |
| `SMM_TAS_Funnel.dta` | **NOT supplied** | usable sample at every step of every block |

### The missing covariance was reconstructed, not requested

`SMM_TAS_VCov.dta` is the only missing file the estimation actually needs, and it is
recoverable from `SMM_TAS_Micro.dta`, which is supplied. Part B specifies the estimator
exactly - every moment is `mean(numerator)/mean(denominator)` over the whole frame with
both zero off the subgroup, estimated by `ratio ..., cluster(famclust)` - and that is a
closed-form influence function:

```
    r     = mean(num) / mean(den)
    psi_i = (num_i - r*den_i) / mean(den) / N          N = the WHOLE-FRAME row count
    Var   = sum_c ( sum_{i in c} psi_i )^2             clustered on famclust
```

**VERIFIED**: this reproduces all seven targeted estimates *and* all seven published
standard errors in `SMM_TAS_Moments.csv` to six decimal places. `tools/make_smm_targets.py`
rebuilds it on every run and writes it into the frozen `targets.toml` as `[moment_cov]`,
stacked with the Part A influence functions on the shared `Fam_id` key.

The remaining four missing files are diagnostics and descriptive tables. Nothing in the
estimation reads them; the numbers this repository needs from `SMM_TAS_TermWealth.dta`
(negative share, timing gaps, quartiles) are recomputed from the microdata instead.

### Measured overlap between the blocks

Because the blocks share a cluster key, the cross-block covariance is computed rather than
assumed zero. Across all 70 parent-by-TAS moment pairs the correlation runs from **-0.0350
to +0.0453**, and no pair exceeds 0.05 in absolute value - so the two blocks are close to
independent in this target set.

**The within-block correlations are a different matter**, and they are what a diagonal
weighting matrix actually discards: the largest is **+0.676**. So the small cross-block
figure is not an argument that diagonal weighting is nearly efficient. Both figures are
reprinted every time the targets are regenerated, and the full joint covariance is what
standard errors and sensitivity are computed from.

(Corrected 2026-09-10: the printed cross-block figure previously divided by
`sqrt(se_i*se_j)` rather than `se_i*se_j` and was about an order of magnitude too small.
The exported covariance and correlation matrices were unaffected.)

## Generation workflow

```bash
# 1. Stata, in order. 21 and 22 must precede 23.
#    19_smm_moments.do      -> Part A files
#    21_tas_linkage.do, 22_*.do, 23_smm_tas_moments.do -> Part B files
#
# 2. Freeze the targets Julia will read (writes a timestamped, immutable snapshot
#    under output/smm_runs/<stamp>_targets/targets.toml):
uv run --with pandas --with numpy python tools/make_smm_targets.py
```

`make_smm_targets.py` reads BOTH micro files, rebuilds both blocks' moments and influence
functions, joins them on the cluster key, and writes one `targets.toml` carrying all
seventeen targeted moments, the joint covariance, the psychic-cost centring constant
`m_psychic`, the fixed school schedule and the wealth winsorisation cut. Julia never reads
a `.dta`.

**This documentation is part of that workflow.** If a do-file changes a sample definition,
change the corresponding part below in the same commit - the parts are the specification
that `make_smm_targets.py` implements, and `code/smm/moments.jl` refuses to run against a
target file whose spec fields disagree with its own constants.

## Decisions and open items recorded elsewhere

| Item | Where |
|---|---|
| `kappa_ParEd` targets use EITHER-parent college; the model means BOTH | `docs/ERRORS.md` P7c |
| `BothCollege` share hardcoded at `Bernoulli(0.3)` | `docs/ERRORS.md` P7b |
| Completion (not entry) is the targeted outcome | Part B, and `code/smm/moments.jl` |
| Terminal wealth is winsorised at p99 | Part B *Limitations*, and `tools/make_smm_targets.py` |
| Terminal-wealth timing gap is uncorrected | Part B *kappa_term* |

---

# Part A - PSID/CDS parent block

Generated by `Code/19_smm_moments.do`.

### Files

| File | Unit of observation | Rows | Variables |
|---|---|---|---|
| `SMM_Moments_ByAge.dta` | one child age (0-17) | 18 | 145 |
| `SMM_Moments_Micro.dta` | one child-year | 17791 | 62 |

### Sample

PSID one-child panel is the **base**; CDS is **left-joined** onto it, because PSID is the denser frame.
No parental-age restriction. One-child *situations* (`Num_Children_in_Family == 1`), 1996+.

| Rows | N |
|---|---|
| PSID child-years, ages 0-17 | 17791 |
| ... with a matched CDS time diary | 1075 |
| ... with a matched WJ test score | 802 |

Each moment uses every row on which it is defined and reports its own `n_` count.

### Naming convention (ByAge)

Every base moment below appears with six prefixes:

| Prefix | Statistic |
|---|---|
| `n_` | number of non-missing observations in the age cell |
| `mu_` | mean, unweighted |
| `sd_` | standard deviation, unweighted |
| `md_` | median, unweighted |
| `wmu_` | mean, weighted by `CHILD_WT` |
| `wmd_` | median, weighted by `CHILD_WT` |

So `sd_leis_hh` is the unweighted SD of household parental leisure in that age cell.

### Base moments

| Moment | Units | N | Description |
|---|---|---|---|
| `hc_lw` | W-score | 802 | Human capital LEVEL (not log): Woodcock-Johnson Letter-Word W-score, ~310-589 |
| `g_ACH` | W-score | 802 | Human capital LEVEL (not log): PCA achievement composite of LW/PC/AP W-scores, i |
| `x_lw` | - | 802 | Human capital in LOGS: ln(hc_lw) = the x of the production function, ~5.7-6.4 |
| `x_gach` | - | 802 | Human capital in LOGS: ln(g_ACH) = the x of the g_ACH specification, ~5.8-6.4 |
| `leis_mom` | hrs/week | 1053 | Mother leisure: 168 - work - 56 sleep - own active childcare (hrs/wk) |
| `leis_dad` | hrs/week | 1053 | Father leisure: 168 - work - 56 sleep - own active childcare (hrs/wk) |
| `leis_hh` | hrs/week | 1053 | Household parental leisure: mother + father (hrs/wk) |
| `leis_mom_wk` | hrs/week | 17266 | Mother leisure, work only: 112 - work (hrs/wk, annual coverage) |
| `leis_dad_wk` | hrs/week | 17266 | Father leisure, work only: 112 - work (hrs/wk, annual coverage) |
| `par_time_act` | hrs/week | 1075 | Parental time investment, ACTIVE only (hrs/wk, CDS diary) |
| `par_time_tot` | hrs/week | 1075 | Parental time investment, active + nearby (hrs/wk) = the p of the model |
| `study_hrs` | hrs/week | 1062 | Child own study time (hrs/wk, CDS diary) |
| `c_time_hrs` | hrs/week | 9725 | Child time input c (hrs/wk): median school by age-year + own study (07 c_final) |
| `study_own_share` | share | 755 | Own study as a share of the c input (diagnostic for the two measures) |
| `psid_mon_invest_real` | real 2015 USD/yr | 7577 | Parental monetary investment, OBSERVED (real 2015 USD/yr) |
| `m_method2_final` | real 2015 USD/yr | 16915 | Parental monetary investment, observed-or-imputed (real 2015 USD/yr) |
| `assets_real` | real 2015 USD/yr | 8145 | Family net worth EXCLUDING home equity (real 2015 USD; odd years only) |
| `assets_home_real` | real 2015 USD/yr | 8145 | Family net worth INCLUDING home equity (real 2015 USD; odd years only) |
| `cons_real` | real 2015 USD/yr | 7577 | Family consumption (real 2015 USD/yr) |
| `cons_exhous_real` | real 2015 USD/yr | 7577 | Family consumption excluding housing (real 2015 USD/yr) |
| `psid_mon_invest_real_w99` | real 2015 USD/yr | 7577 | Observed monetary investment, winsorised at p99 (real 2015 USD/yr) |
| `m_method2_final_w99` | real 2015 USD/yr | 16915 | Observed-or-imputed monetary investment, winsorised at p99 (real 2015 USD/yr) |
| `cons_real_w99` | real 2015 USD/yr | 7577 | Family consumption, winsorised at p99 (real 2015 USD/yr) |
| `cons_exhous_real_w99` | real 2015 USD/yr | 7577 | Family consumption ex housing, winsorised at p99 (real 2015 USD/yr) |

### Micro file variables

| Variable | Type | Description |
|---|---|---|
| `Sample_id` | float | PSID family-year sample identifier (key to Sample_Baseline.dta) |
| `Year` | int | PSID survey year |
| `Fam_id` | int | PSID 1968 family (lineage) identifier ER30001 |
| `Per_id` | int | PSID person number within the 1968 family ER30002 |
| `Age_Father` | byte | 1 Age |
| `Age_Mother` | byte | 2 Age |
| `Child_Age` | byte | Child age in years (PSID, every year) |
| `wh_mom` | double | Mother annual work hours (filled) |
| `wh_dad` | double | Father annual work hours (filled) |
| `study_total_miss` | float | Study/human-capital hrs/wk: homework, self-study, edu reading + academic clubs [ |
| `Dad_Total_Act` | float | Dad active time with child (hrs/wk; social context) |
| `Mom_Total_Act` | float | Mom active time with child (hrs/wk; social context) |
| `parent_Act` | float | Active time with any parent (hrs/wk; social context) |
| `CHILD_WT` | double | (mean) CH97PRWT |
| `LW_WS` | int | Woodcock-Johnson Letter-Word W-score (CDS assessment) |
| `PC_WS` | int | Woodcock-Johnson Passage Comprehension W-score (CDS assessment) |
| `AP_WS` | int | Woodcock-Johnson Applied Problems W-score (CDS assessment) |
| `psid_mon_invest_real` | double | Parental monetary investment, OBSERVED (real 2015 USD/yr) |
| `p_hours` | double | Observed parental time with child, hrs/wk (CDS diary, parent_Total) |
| `p_final_hrs` | double | Primary parental time back to hrs/wk = exp(p_final) (KDD 8) |
| `c_final_hrs` | double | Primary child-own time back to hrs/wk = exp(c_final) (KDD 7) |
| `parent_hours` | double | household work hours (mother + father), hrs/wk (KDD 11) |
| `m_method2_final` | double | Parental monetary investment, observed-or-imputed (real 2015 USD/yr) |
| `child_uid` | int | numeric child id (Fam_id x Per_id group) |
| `cds_matched` | byte | 1 = PSID child-year with a matched CDS time diary (the CDS wave rows) |
| `has_wj` | byte | 1 = PSID child-year with a matched Woodcock-Johnson score |
| `dad_age_birth` | byte | Father's age at this child's birth (Age_Father - Child_Age) |
| `mom_age_birth` | byte | Mother's age at this child's birth |
| `mean_age_birth` | double | Mean parental age at this child's birth |
| `cohort_dad2530` | byte | 1 = PRIMARY cohort: father aged 25-30 at the child's birth |
| `cohort_dad2527` | byte | 1 = tighter cohort: father aged 25-27 at the child's birth (thin, see KDD 10) |
| `cohort_mean2530` | byte | 1 = alternative cohort: MEAN parental age 25-30 at the child's birth |
| `leis_mom` | double | Mother leisure: 168 - work - 56 sleep - own active childcare (hrs/wk) |
| `leis_dad` | double | Father leisure: 168 - work - 56 sleep - own active childcare (hrs/wk) |
| `leis_mom_wk` | double | Mother leisure, work only: 112 - work (hrs/wk, annual coverage) |
| `leis_dad_wk` | double | Father leisure, work only: 112 - work (hrs/wk, annual coverage) |
| `leis_hh` | double | Household parental leisure: mother + father (hrs/wk) |
| `leis_mom_clip` | double | Mother leisure, clipped at 0 (diagnostic only - see KDD 2) |
| `leis_dad_clip` | double | Father leisure, clipped at 0 (diagnostic only - see KDD 2) |
| `leis_hh_clip` | double | Household leisure, clipped at 0 (diagnostic only) |
| `par_time_act` | double | Parental time investment, ACTIVE only (hrs/wk, CDS diary) |
| `par_time_tot` | double | Parental time investment, active + nearby (hrs/wk) = the p of the model |
| `Asset_Family` | double | 1 Asset |
| `Asset_Plus_Home_Family` | double | 1 Asset_With_Home |
| `assets_real` | double | Family net worth EXCLUDING home equity (real 2015 USD; odd years only) |
| `assets_home_real` | double | Family net worth INCLUDING home equity (real 2015 USD; odd years only) |
| `study_hrs` | double | Child own study time (hrs/wk, CDS diary) |
| `c_time_hrs` | double | Child time input c (hrs/wk): median school by age-year + own study (07 c_final) |
| `study_own_share` | double | Own study as a share of the c input (diagnostic for the two measures) |
| `g_ACH` | double | Human capital LEVEL (not log): PCA achievement composite of LW/PC/AP W-scores, i |
| `hc_lw` | int | Human capital LEVEL (not log): Woodcock-Johnson Letter-Word W-score, ~310-589 |
| `x_lw` | double | Human capital in LOGS: ln(hc_lw) = the x of the production function, ~5.7-6.4 |
| `x_gach` | double | Human capital in LOGS: ln(g_ACH) = the x of the g_ACH specification, ~5.8-6.4 |
| `Consumption` | float | (mean) Consumption |
| `Consumption_Exc_Housing` | float | (mean) Consumption_Exc_Housing |
| `CPI` | float | CPI |
| `cons_real` | double | Family consumption (real 2015 USD/yr) |
| `cons_exhous_real` | double | Family consumption excluding housing (real 2015 USD/yr) |
| `cons_real_w99` | double | Family consumption, winsorised at p99 (real 2015 USD/yr) |
| `cons_exhous_real_w99` | double | Family consumption ex housing, winsorised at p99 (real 2015 USD/yr) |
| `psid_mon_invest_real_w99` | double | Observed monetary investment, winsorised at p99 (real 2015 USD/yr) |
| `m_method2_final_w99` | double | Observed-or-imputed monetary investment, winsorised at p99 (real 2015 USD/yr) |

### Own study and fixed school time (checked 9 September 2026)

The current micro file also contains `med_school_ageyr`, `school_hrs`,
`study_hrs_measured` and `school_hrs_measured`. `school_hrs` is median school time
within `(Year, Child_Age)`, set to zero below age 6; `study_hrs` is own study,
also set to zero below age 6. The `_measured` variants retain the unzeroed values.
For future estimation, use `study_hrs / 112` as the chosen investment moment and
freeze the age means of `school_hrs / 112` as the school schedule deducted from
child leisure. Do not recover school by subtracting `study_hrs` from `c_time_hrs`:
the latter also has imputed observations and different coverage.

### Human capital: levels vs logs

Two of the four are **levels**, two are **logs**. Verified against the data:

| Moment | Scale | Mean | Range |
|---|---|---|---|
| `hc_lw` | **level**, W-score | 495.2 | 310-589 |
| `g_ACH` | **level**, W-score units | 495.5 | 324-587 |
| `x_lw` | **log** = ln(hc_lw) | 6.197 | 5.74-6.38 |
| `x_gach` | **log** = ln(g_ACH) | 6.200 | 5.78-6.38 |

`x_lw` and `x_gach` are exactly the natural logs of the two level series
(difference identically zero on all 802 rows), and they are what the production
function uses as x. Calibrate to the level or the log series, not a mix.

`g_ACH` is a PCA-weighted average of the LW / PC / AP W-scores, so it is in
W-score units, not standardised. Its mean sits slightly below the simple average
because 112 of the 802 rows fall back to a 2-test or 1-test composite.

### Two things to read before calibrating

1. **`c_time_hrs` is a conventional zero at ages 0-5**, not a measurement.
   `07` sets the child time input to zero below school age because the production
   function restricts beta^C to zero in G1. The same children record real study time
   in `study_hrs`. For current own-study estimation, use **`study_hrs` at ages 6–17**; use
   the `_measured` variant for descriptive study time below age 6.

2. **`c_time_hrs` dispersion is not comparable to `study_hrs`.** Its school component
   is a median by (Year, Age), identical for all children of an age, so it moves the
   level but carries no cross-child variation. Read its mean, not its SD.

### Parent leisure is constructed, not observed

The CDS diary records the **child's** 24 hours and who was present - never the parent's
own time use - so leisure is a residual: `112 - own work hours - own active child time`,
per parent (112 = 168 less a 56-hour sleep allowance). It therefore exists only at CDS
wave years. `leis_mom_wk` / `leis_dad_wk` are the work-only variant, defined in every
year, so the coverage cost of the definition is visible. Negative values are **not**
clipped - clipping would bias the mean upward.

---

# Part B - TAS-linked child block

Written by `Code/23_smm_tas_moments.do`. Run `21` and `22` first. Nothing in this part
overwrites the outputs of `19_smm_moments.do`.

**Sample frame warning.** Everything in this part is measured on TAS-linked children, one
row per child - not on the child-years of Part A. Denominators, weighting and cluster
counts here are not comparable to Part A's, and no moment from one part may be pooled with
a moment from the other.

### Sample

TAS respondents who link to a child in the **CDS diary sample OR the PSID
child-year extract** - not the full TAS, and not only the production-function
estimation sample. The union is deduplicated on the permanent (Fam_id, Per_id)
pair; `in_cds` and `in_psid_panel` are retained on every row, and
`SMM_TAS_Funnel.dta` reports the usable sample at every step of every block.

The link itself is the IND2023ER bridge built by `21_tas_linkage.do`:
(TAS year, family interview number, sequence number) -> (ER30001, ER30002).

### Horizon, denominators, missing values

| choice | value |
|---|---|
| Outcome horizon | achieved by age 25 (max over waves at age <= 25) |
| Follow-up requirement | at least one wave answered at age 23-25, SEPARATELY for each outcome |
| Horizon caveat | 'by age 25' is an APPROXIMATION: a child last observed at 23 or 24 contributes a zero not actually observed through 25. The log reports the last-usable-age distribution. |
| Denominator | every linked child in the frame, coded 0 |
| HS non-completers | in the denominator, coded 0 |
| 2-year / vocational only | in the denominator, coded 0 - never counted as four-year |
| Four-year dropouts | entry = 1, completion = 0 |
| Missing outcome | dropped from that moment only; each moment reports its own N |
| Repeated observations | collapsed to one row per child before any moment |
| Weighting | unweighted primary. CHILD_WT is a CDS CHILDHOOD weight, not a TAS adult follow-up weight, and is non-missing only in the CDS-PSID overlap (424 of 2,603). SMM_TAS_Weighted.dta reports the unweighted rate on the SAME records, because the full-frame-vs-weighted-subset difference is mostly selection, not weighting. |
| Clustering | Fam_id, the 1968 family interview number (siblings share it) |

### The four-year college measures

**COMPLETION IS THE PRIMARY OUTCOME** (user 2026-09-10). Entry is retained as a
secondary series but should not be used as the headline target: it rests on the
degree SOUGHT and runs near 0.62, against a national immediate-four-year-
enrollment figure of about 0.45. The same gap appears in the FULL TAS (0.59),
so it is the proxy, not our linkage. Completion (0.32) reconciles with the
national bachelor's figure of about 0.40 at ages 25-29. The achievement and
parental-education blocks condition on the completion frame, and the achievement
tertiles are cut on it.

The TAS does **not** record whether an institution is a two- or four-year
college. The only institution identifier is an IPEDS code, and no local
crosswalk exists. Two measures are therefore built:

- **Entry** (`y_entry`): the degree SOUGHT at college #1 or #2 (`G18N`) is a
  4-year or graduate degree, or a bachelor's-or-higher degree is held.
  A respondent at a 4-year school who reports seeking an associate degree is
  counted as two-year. This is the binding limitation on kappa_0.
- **Completion** (`y_complete`): `G18P` degree received in {2..6}
  (Bachelor's, Master's, Doctoral, MD, JD) or ENROLLMENT STATUS in {6, 7, 11}.

Code frames were audited wave by wave in the log rather than taken from the
published labels, which exist only for 2019 and 2023. The audit also confirms
no record completes a four-year degree while coded as a non-entrant.

### kappa_term: what 'terminal' means here

Two definitions, both reported:

| definition | conditions | waves available |
|---|---|---|
| loose | separated from the parental family AND not in a parental home | all 10 |
| strict | loose AND no tuition, housing or bills support | all 10 |
| broad | loose AND no support of any kind | all 10 |

**Separation** is the child's PSID family interview number differing from the
anchor parent's in the same year, both observed - which simultaneously verifies
that the parent is still in the panel after the child leaves.

**Residence** is the fall/winter primary-residence question (B1 in 2005-2015,
B15 in 2017-2023). Its code frame moves TWICE, and published value labels exist
only for 2019 and 2023, so the frames were read off each wave's own codebook and
reconciled against the code counts in the data:

| waves | parental-home codes | note |
|---|---|---|
| 2005-2009 | 1, 5 | 1 = parents' home, 5 = house R's parents own |
| 2011-2017 | 1, 5, 9 | 9 = spouse/partner's parents' home |
| 2019-2023 | 3, 4, 5 | frame renumbered; 1 becomes 'home owned by R' |

Applying the 2019 frame to the earlier waves would score **college dorms and
fraternities** as parental homes and miss the real parental-home code entirely.
The log prints the parental-home share by wave so a future renumbering appears
as a discontinuity at a frame boundary rather than as a finding.

**Support** is measured from the RECEIPT FLAGS, in all ten waves.

CORRECTION (2026-09-10, after audit). An earlier version of this file stated
that the support block did not exist before 2017 and restricted the strict
definition to 2017-2023. **That was wrong.** The same questions run through the
whole panel under the older F56 naming (F56D WTR TUITION COVERED, F56B WTR RENT
OR MORTGAGE COVERED, ...), which the 2017 redesign renamed to E42/E43-E49. The
original search matched only the 2017+ label wording and so missed six waves.

The flags, not the dollar amounts, are the receipt indicator. The old code summed
amounts only, so a respondent who answered the follow-up as a PERCENTAGE, or who
reported receipt without a usable amount, was scored as receiving nothing. For
2017+ the amounts and percentages are ORed in as corroboration.

**These are transfers from parents OR OTHER RELATIVES.** The E42 stem reads
'did your parents or other relatives...'; the short PSID label AMT PARENTS PAID
is what misleads. Parent-only support cannot be separated in these data.

Two aggregates are carried because the model's concept is college-period
support, not all lifetime help: **college** = tuition + rent/mortgage/dorm +
expenses and bills; **broad** = every category. Student-loan repayment help sits
outside the college aggregate because it usually follows the college spell.

**OPEN, needs the model:** the qualifying wave is the FIRST wave meeting the
condition. That is not a verified end-of-college-support date, and the wealth
is the LATEST observation at or after it, which can post-date qualification by
many years. The log reports the gap distribution and the child's age at the
wealth measurement so the reference point is explicit. Anchoring this to the
model's terminal date is a specification choice that has not been made.

**Age was never used as a proxy for independence.**

**Wealth** is the parental family's net worth from `Sample_Baseline.dta`, keyed
on the birth father (Sample_id = Fam_id*1000 + Per_id) with the birth mother as
fallback. The fallback triggers on WEALTH AVAILABILITY, not on whether the
father has identifiers: a father can be identified and still never appear in
the wealth extract, and those children used to be dropped even when the
mother's household was observable. pwsrc_* records which household was used.
It is taken at the LATEST odd year at or
after the qualifying wave in which the parental family is observed - PSID
measures wealth in odd years only (1999-2021) - and `wgap_*` carries the years
between qualification and measurement. Deflated to real 2015 USD.
**Negative net worth is retained**, and its share is reported per definition.

Wealth at ages 17-18 is NOT substituted anywhere in this block.

### kappa_theta: achievement

`LW_WS` and `g_ACH` only - no separate PC or AP blocks, per the specification.

`g_ACH` is the PCA achievement composite from `13_production_function_nls.do`
KDD 8: a weighted average of the Woodcock-Johnson **W-scores** whose weights are
the first principal component's loadings renormalised to sum to one, with a
3-test -> 2-test -> 1-test fallback cascade. It is in **W-score units, not
standardised** (mean around 495). The weights are re-estimated here on the CDS
diary universe rather than the master-panel rows and are printed in the log for
comparison with 13's.

Scores are the child's LAST CDS wave. They are NOT proven to precede college
entry (corrected 2026-09-10 after audit): a small number of children report
four-year enrollment in the SAME calendar year as the last CDS wave, and the
within-year order is unresolved because the pipeline does not retain assessment
and enrollment dates. Read the ability gradient with that caveat.
Tertiles are cut WITHIN age group, so T1/T2/T3 are not comparable in
level between the age-17 and age-18 panels - only the gradient within a panel
is. The age-18 group comes almost entirely from CDS-2002 and CDS-2007, since
CDS-2014 and CDS-2019 stop at 17.

### kappa_ParEd: parental education

TAS `COMPLETED EDUCATION OF MOTHER` / `OF FATHER`, all ten waves: 0-16 actual
years, 17 = at least some post-graduate work, 96 = parent unknown or never in
the study, 98 = DK, 99 = NA.

pared_col is THREE-VALUED (corrected 2026-09-10 after audit): 1 when either
observed parent has 16+ years, 0 only when BOTH parents are observed and both
are below 16, and missing otherwise. The old rule set 0 whenever any parent was
observed below 16, so 467 of the 1,633 zeros had one unobserved parent; that
estimates 'no OBSERVED college parent', a different quantity. The unknown group
is reported as its own moment rather than folded into zero.

**2013 is a bad release for this variable.** Mother's education is coded 0 for
1,539 of 1,804 respondents in 2013, against 2-45 zeros in every other wave.
Zero is inside the codebook's valid range, so no range filter catches it. The
2013 zeros are set missing; zeros elsewhere are kept and counted.

The child-level value is the FIRST valid report, not the maximum, so a later
correction or a post-horizon report cannot override what was true while the
college decision was being made. The log counts children where the two differ.

### Standard errors and the covariance matrix

Every moment is written as mean(numerator)/mean(denominator) over the whole
sample, with both zero off the subgroup, and all of them are estimated together
by `ratio ..., cluster(famclust)`. The off-diagonal terms of
`SMM_TAS_VCov.dta` are therefore real, not assumed zero - which matters
because the kappa_theta tertiles and the kappa_ParEd groups partition
overlapping sets of the same children.

Gaps in `SMM_TAS_Gaps.dta` are linear combinations of those ratios, so their
standard errors come straight off the joint V via `lincom`.

A moment whose denominator has fewer than 10 observations is dropped from the
system and named in the log, rather than being reported with a meaningless SE.

n_clusters is now PER MOMENT (corrected 2026-09-10 after audit); the old export
copied the whole system's count onto every row, overstating the independent
information behind thin cells. n_clusters_system carries the total.

The moment vector mixes probabilities with dollars, so the RAW covariance is
ill-conditioned by construction. Both condition numbers are printed in the log:
scale the moments consistently, or invert the correlation form, not the raw one.

The covariance conditions on the fitted PCA weights and the sample tertile cut
points, treating them as fixed. Propagating that uncertainty needs a
family-cluster bootstrap that re-runs those steps; it is not done here.

**Medians are not in the covariance matrix.** A median is not a ratio of means,
so it cannot join this ratio system; the kappa_term medians in
SMM_TAS_TermWealth.dta are descriptive and carry no standard error. This is a
choice of estimator, not an impossibility - a family-cluster bootstrap could
give medians both SEs and a joint covariance with the other moments.

### Files

| file | contents |
|---|---|
| `SMM_TAS_Moments.dta` / `.csv` | moment, block, estimate, SE, N, clusters |
| `SMM_TAS_VCov.dta` / `.csv` | joint cluster-robust covariance matrix |
| `SMM_TAS_Gaps.dta` | top-minus-bottom and parental-education differences |
| `SMM_TAS_TermWealth.dta` | kappa_term means, medians, quartiles, negatives, timing |
| SMM_TAS_Weighted.dta | full-frame vs weighted-subset rates, unweighted and weighted |
| `SMM_TAS_Funnel.dta` | usable sample at every step of every block |
| `SMM_TAS_Micro.dta` | one row per child: every input to every moment |

---

# Limitations carried into the estimation

Collected here so they travel with the targets rather than only with the block that
produced them. Each is documented in full in its own part above.

| # | Limitation | Affects | Status |
|---|---|---|---|
| 1 | 'By age 25' is an approximation: a child last observed at 23 or 24 contributes a zero not actually observed through 25. | every Part B outcome | inherent to the follow-up rule |
| 2 | Entry rests on the degree **sought**; the TAS records no two-/four-year institution flag. | `k0_entry` | why completion is the primary outcome |
| 3 | `pared_col` is EITHER-parent 16+, while the model's state is BothCollege. | `kpe_g0_c`, `kpe_g1_c` | **open by instruction**, `docs/ERRORS.md` P7c |
| 4 | The unknown-education group (N=526, completion 0.125) has no model counterpart. | `kappa_ParEd` | reported, untargeted |
| 5 | Terminal wealth is measured at a median child age of ~29, a median 4 years after qualifying independence; the model's object is assets at the transfer, child age 18. | `kterm_x_strict_w99` | **uncorrected**; the model has no post-separation parent to age forward |
| 6 | 13.0% of qualifying parental net worth is negative and is retained; the model floors retained assets at `delta_P` and cannot reproduce it. | `kterm_x_strict_w99` | recorded, not censored |
| 7 | The wealth mean is winsorised at p99 ($4.45m cut; raw mean $429,803, median $49,243, max $41.3m). | `kterm_x_strict_w99` | decision 2026-09-10, matching Part A's p99 treatment of consumption and investment |
| 8 | Support flags cover parents **or other relatives**; parent-only support cannot be separated. | the strict/broad terminal definitions | inherent to the PSID question wording |
| 9 | The last CDS assessment is not demonstrably before college entry for every child. | `kappa_theta` tertiles | corrected 2026-09-10 after audit; read the gradient with this caveat |
| 10 | Achievement tertiles are cut within age group, so T1/T2/T3 are not comparable in level between the age-17 and age-18 panels. | `kappa_theta` | only the within-panel gradient is meaningful |
| 11 | The covariance conditions on the fitted PCA weights and the sample tertile cut points, treating them as fixed. | all standard errors | propagating it needs a family-cluster bootstrap; not done |
| 12 | Medians are not ratios of means and carry no standard error in this system. | `kterm` medians | descriptive only |
| 13 | The raw moment vector mixes probabilities with dollars and is ill-conditioned by construction. | the weighting matrix | scale consistently, or invert the correlation form - `code/smm/standard_errors.jl` does the latter |
