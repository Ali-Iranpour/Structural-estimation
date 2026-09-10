# SMM moments from the TAS-linked children

Written by `Code/23_smm_tas_moments.do`. Run `21` and `22` first.
Nothing here overwrites the outputs of `19_smm_moments.do`.

## Sample

TAS respondents who link to a child in the **CDS diary sample OR the PSID
child-year extract** - not the full TAS, and not only the production-function
estimation sample. The union is deduplicated on the permanent (Fam_id, Per_id)
pair; `in_cds` and `in_psid_panel` are retained on every row, and
`SMM_TAS_Funnel.dta` reports the usable sample at every step of every block.

The link itself is the IND2023ER bridge built by `21_tas_linkage.do`:
(TAS year, family interview number, sequence number) -> (ER30001, ER30002).

## Horizon, denominators, missing values

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

## The four-year college measures

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

## kappa_term: what 'terminal' means here

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

## kappa_theta: achievement

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

## kappa_ParEd: parental education

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

## Standard errors and the covariance matrix

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

## Files

| file | contents |
|---|---|
| `SMM_TAS_Moments.dta` / `.csv` | moment, block, estimate, SE, N, clusters |
| `SMM_TAS_VCov.dta` / `.csv` | joint cluster-robust covariance matrix |
| `SMM_TAS_Gaps.dta` | top-minus-bottom and parental-education differences |
| `SMM_TAS_TermWealth.dta` | kappa_term means, medians, quartiles, negatives, timing |
| SMM_TAS_Weighted.dta | full-frame vs weighted-subset rates, unweighted and weighted |
| `SMM_TAS_Funnel.dta` | usable sample at every step of every block |
| `SMM_TAS_Micro.dta` | one row per child: every input to every moment |
