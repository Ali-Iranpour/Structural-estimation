use "/tmp/claude-1105/-srv-project-speech/7f6dc90a-1729-4526-8754-9e9c68d446db/scratchpad/tmp.fxR2Z5jrVa/cds17_wages.dta", clear
keep if has_wage & agew >= 22 & !missing(ba)
* trim implausible hourly wages (below $2 or above $400, as Daruich's NLSY rule) for this scoping only
gen double w = exp(lnwage)
keep if inrange(w, 2, 400)
regress lnwage i.agew i.Year
predict double r, resid
collapse (mean) r (first) lw_raw ba (count) nw = r, by(Fam_id Per_id)
quietly summarize lw_raw
gen double z = (lw_raw - r(mean)) / r(sd)
di "=== per child: mean age/year-purged ln wage at 22+, on the LW score at 17 (standardised), by BA"
tab ba
regress r z if ba == 0, robust
regress r z if ba == 1, robust
summarize r if ba == 0
summarize r if ba == 1
