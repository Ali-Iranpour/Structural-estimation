# TAS wage availability for the CDS children with a score at 17 (2026-10-02)

Scratch diagnostics behind `docs/WAGE_RETURN_ANCHOR.md` §3. Run with Stata 19 MP on the server from a scratch
directory (their paths point there; the inputs are Child_Time_Study's raw TAS waves, Temp/_tas_bridge.dta and
Output/Data/SMM/SMM_TAS_Micro.dta). Not part of any pipeline; nothing here enters the model.

- `tas_wage_counts.do/.log`: per TAS wave, job earnings last year (jobs 1-5), total weeks and average hours last
  year (2007+), head/wife status; usable hourly wage = all jobs known, total > 0, weeks 1-52, hours 1-112.
  Counts of children with a usable wage by BA status at 25/26 and by age.
- `tas_wage_precision.do/.log`: a crude scoping regression (per-child mean age/year-purged log wage at 22+ on
  the standardised raw LW score at 17, by BA; hourly wage trimmed to $2-$400). Not a result.
