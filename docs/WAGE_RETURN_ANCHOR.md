# The return to childhood skill in the child's wage: two ways to set it

For Sahber. 2026-10-02. Nothing here is implemented. The model refuses to run until this
choice is made.

## 1. The problem

The child's adult wage (`WAGE_PROCESS.md` §1) loads on childhood skill through α_θ (high
school) and α_θ + α_θE (college):

```
ln w = ln w₀ + β_E·E + (α_θ + α_θE·E)·(ln θ − m_θ) + age terms + ln z
```

Under memo 18, θ is DFVW latent skill (ln k at 17, SD 0.652 in our data). The memo-19 code
sets the loading by carrying over Daruich & Fernández's return per SD:

```
α_θ = 0.654 · sd(log AFQT) / 0.652,   α_θ + α_θE = 0.976 · sd(log AFQT) / 0.652
```

Here 0.654 and 0.976 come from D&F (2023), Table B4. That table regresses the NLSY79 wage
residual on log AFQT raw score, separately by education.

The one input the formula needs, **sd(log AFQT)**, is not reported by D&F, and it is not in
our data (it is NLSY79). α_θ scales one-for-one with it.

## 2. Published values for sd(log AFQT)

| Source | What it says | Caveat |
|---|---|---|
| Daruich, *The Macroeconomic Consequences of Early Childhood Development Policies*, FRB St. Louis WP 2018-29 (version of 13 March 2019), fn. 28, p. 21 | "the standard deviation of log(AFQT) in the sample is approximately 0.05"; mean log(AFQT) 5.19 (high school) and 5.38 (college); "AFQT89 raw score" | Different returns in that version (0.533 / 0.904, Table 2). The group means alone imply a between-group SD near 0.09 (at a 70/30 split), above 0.05, so the 0.05 is probably within-group or loosely stated. |
| Altonji, Bharadwaj & Lange, *Constructing AFQT Scores that are Comparable Across the NLSY79 and the NLSY97* (August 2009), p. 3 | NLSY79 respondents tested at age 16: equated AFQT mean 155.93, SD 31.48 | A level, not a log; its coefficient of variation, 0.20, approximates sd(log) only roughly. Tested at age 16, not D&F's wage sample. |
| D&F, NBER WP 27351 (June 2020) and the 2023 revision | log(AFQT) defined as the log of the raw score; no SD reported | The returns differ across versions: 0.471 / 1.008 (2020), 0.654 / 0.976 (2023). |

The candidates differ by a factor of about four. They give α_θ = 0.050 (sd 0.05) or about
0.20 (sd 0.20). That is a return of 0.03 or 0.13 log points per SD of childhood skill for a
high-school worker. Writing to Daruich or Fernández for the sd in their Table B4 sample would
settle this route.

## 3. The alternative: estimate the return on our own children (`WAGE_PROCESS.md` §6.4)

The CDS children with a Letter-Word score at 17 are followed into adulthood by the
Transition into Adulthood Supplement. This is the design Lee & Seshadri use with CDS
Letter-Word.

**Data in `Child_Time_Study`.** TAS 2005–2023 records the respondent's own earnings last year
for jobs 1–5, plus total weeks and average weekly hours last year from 2007 on. 23 does not
extract these yet; the counts below come from a scratch diagnostic. An hourly wage is total
job earnings divided by (weeks × hours).

**Counts.** 372 TAS-linked children have an LW score at 17; 253 of them have BA status at
25/26. Of those 253:

| Children with at least one usable hourly wage | no BA (162) | BA (91) |
|---|---|---|
| at any age 18+ | 112 | 80 |
| at age 22+ | 80 | 67 |
| at age 24+ | 66 | 52 |
| at age 25+ | 65 | 49 |
| with positive earnings (no hours needed), age 22+ | 145 | 87 |

**Three problems with this sample.**
1. **Selection on independence.** In TAS 2007–2013, weeks and hours are coded zero for
   respondents who were already PSID heads or wives ("so few … were asked", codebook). In
   those waves an hourly wage exists only for dependents. The 2017+ waves copy the PSID
   values, so the problem is confined to the older cohorts.
2. **Young and few.** Wages are observed at 18–28 only, a few waves per child. After a
   $2–$400 hourly wage trim, 48 no-BA and 53 BA children remain at 22+.
3. **The score is noisy and near its ceiling at 17.** Raw LW at 17 has a binomial test
   error (memo 18). Regressing on it attenuates the return.

**What the precision looks like.** A crude scoping regression, not a result: per child, the
age- and year-purged log wage at 22+, averaged, on standardised LW at 17. The slope is 0.015
(SE 0.067) without a BA and 0.195 (SE 0.078) with one, before any noise correction. A
standard error near 0.07 per SD cannot separate the two published anchors (0.03 vs 0.13 per
SD) for high-school workers.

**How it would enter the SMM.** Correcting the binomial noise outside the model is fiddly. The
cleaner route is indirect inference: target the same regression as a moment, run on
simulated children whose LW at 17 carries the same binomial noise. The model already draws
that noise, analytically, for the S and T blocks. Age profiles would still come from the
PSID main panel (D&F's step 1), as §6.4 already proposes.

## 4. The two routes side by side

| | Daruich–Fernández per-SD anchor | Direct estimate on CDS/TAS |
|---|---|---|
| Skill measure | log AFQT (NLSY79), converted per SD | our ln k at 17 (Letter-Word) |
| Missing input | sd(log AFQT): published values differ by ×4 | none; the TAS earnings extract is to be added to 23 |
| Sample | NLSY79, ages 25–63, thousands of households | about 50–80 children per education group, ages 22–28 |
| Precision | high, given the sd | low: SE ≈ 0.07 per SD |
| Unit consistency | relies on a per-SD equivalence of AFQT and ln k | native |
| Main risk | the wrong sd scales the return by up to 4 | selection (2007–13 hours), early-career wages, noise |

## 5. Questions for Sahber

1. Which route, or both (D&F's value as the prior, the CDS regression as a moment)?
2. If D&F: which sd(log AFQT)? Should we ask the authors for the Table B4 sample value?
3. If CDS: hourly wage or annual earnings (earnings avoid the 2007–13 hours gap)? Which age
   window (22+ or 25+)? A pre-estimated parameter, or an SMM moment by indirect inference?

Sources: D&F 2023 (`docs/papers/Daruich and Fernandez - Universal Basic Income A Dynamic
Assessment.pdf`, Table B4); D&F NBER WP 27351 (2020); Daruich 2019 WP
(wpcarey.asu.edu/sites/g/files/litvpz246/files/documents/daruich_jmp.pdf); Altonji,
Bharadwaj & Lange 2009 (harris.uchicago.edu/files/aftq.pdf); TAS 2007 and 2023 codebooks
(`Child_Time_Study/Input/TAS/`).
