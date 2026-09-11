# SMM moments from the TAS-linked children - MERGED

**This file was merged into [`CODEBOOK.md`](CODEBOOK.md) on 2026-09-10.**

Its content is now **Part B - TAS-linked child block** of the consolidated codebook,
unchanged: the same sample definition, the same audit corrections (the support-block
correction, the three-valued `pared_col` correction, the per-moment `n_clusters`
correction, the assessment-timing correction), and the same open items.

The stub is kept rather than deleted because the filename is referenced from the Stata
pipeline and from earlier run records. **Do not add content here** - two codebooks
documenting one estimation is what the merge was for. Edit Part B of `CODEBOOK.md`.

What the merge added on top of the two originals:

- one table contrasting the two sample frames, and the warning that they are different
  units of observation which must not be pooled;
- a supplied-vs-missing file table (`SMM_TAS_VCov.dta`, `SMM_TAS_Gaps.dta`,
  `SMM_TAS_TermWealth.dta`, `SMM_TAS_Weighted.dta` and `SMM_TAS_Funnel.dta` are **not**
  in this repository);
- how the missing covariance is reconstructed from `SMM_TAS_Micro.dta`, and the check
  that it reproduces the published standard errors;
- the measured cross-block overlap (488 shared families; all 70 cross-block moment
  correlations under 0.005 in absolute value);
- a single consolidated table of the limitations that travel with the targets.
