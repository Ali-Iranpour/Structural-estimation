clear all
set more off
set linesize 200
global TD "/srv/project/speech/apps/Child_Time_Study/Input/TAS"
global SD "/tmp/claude-1105/-srv-project-speech/7f6dc90a-1729-4526-8754-9e9c68d446db/scratchpad/tmp.fxR2Z5jrVa"
tempfile all
local first = 1
filefilter "$TD/ta2005/TA2005.do" "$SD/w.do", from("[path]\BS") to("$TD/ta2005/") replace
clear
quietly do "$SD/w.do"
keep TA050002 TA050003 TA050004 TA050220 TA050255 TA050290 TA050325 TA050360 TA050011
rename (TA050002 TA050003 TA050004) (tas_iw_id iw_num seq_num)
rename TA050220 earn1
rename TA050255 earn2
rename TA050290 earn3
rename TA050325 earn4
rename TA050360 earn5
gen weeks = .
gen hours = .
rename TA050011 hwstat
gen int Year = 2005
if `first' { 
 quietly save `all', replace 
 local first = 0 
 }
 else { 
 append using `all' 
 quietly save `all', replace 
 }
filefilter "$TD/ta2007/TA2007.do" "$SD/w.do", from("[path]\BS") to("$TD/ta2007/") replace
clear
quietly do "$SD/w.do"
keep TA070002 TA070003 TA070004 TA070248 TA070268 TA070288 TA070308 TA070328 TA070337 TA070338 TA070011
rename (TA070002 TA070003 TA070004) (tas_iw_id iw_num seq_num)
rename TA070248 earn1
rename TA070268 earn2
rename TA070288 earn3
rename TA070308 earn4
rename TA070328 earn5
rename TA070337 weeks
rename TA070338 hours
rename TA070011 hwstat
gen int Year = 2007
if `first' { 
 quietly save `all', replace 
 local first = 0 
 }
 else { 
 append using `all' 
 quietly save `all', replace 
 }
filefilter "$TD/ta2009/TA2009.do" "$SD/w.do", from("[path]\BS") to("$TD/ta2009/") replace
clear
quietly do "$SD/w.do"
keep TA090002 TA090003 TA090004 TA090265 TA090285 TA090305 TA090325 TA090345 TA090354 TA090355 TA090011
rename (TA090002 TA090003 TA090004) (tas_iw_id iw_num seq_num)
rename TA090265 earn1
rename TA090285 earn2
rename TA090305 earn3
rename TA090325 earn4
rename TA090345 earn5
rename TA090354 weeks
rename TA090355 hours
rename TA090011 hwstat
gen int Year = 2009
if `first' { 
 quietly save `all', replace 
 local first = 0 
 }
 else { 
 append using `all' 
 quietly save `all', replace 
 }
filefilter "$TD/ta2011/TA2011.do" "$SD/w.do", from("[path]\BS") to("$TD/ta2011/") replace
clear
quietly do "$SD/w.do"
keep TA110002 TA110003 TA110004 TA110255 TA110275 TA110295 TA110315 TA110335 TA110344 TA110345 TA110011
rename (TA110002 TA110003 TA110004) (tas_iw_id iw_num seq_num)
rename TA110255 earn1
rename TA110275 earn2
rename TA110295 earn3
rename TA110315 earn4
rename TA110335 earn5
rename TA110344 weeks
rename TA110345 hours
rename TA110011 hwstat
gen int Year = 2011
if `first' { 
 quietly save `all', replace 
 local first = 0 
 }
 else { 
 append using `all' 
 quietly save `all', replace 
 }
filefilter "$TD/ta2013/TA2013.do" "$SD/w.do", from("[path]\BS") to("$TD/ta2013/") replace
clear
quietly do "$SD/w.do"
keep TA130002 TA130003 TA130004 TA130254 TA130274 TA130294 TA130314 TA130334 TA130343 TA130344 TA130011
rename (TA130002 TA130003 TA130004) (tas_iw_id iw_num seq_num)
rename TA130254 earn1
rename TA130274 earn2
rename TA130294 earn3
rename TA130314 earn4
rename TA130334 earn5
rename TA130343 weeks
rename TA130344 hours
rename TA130011 hwstat
gen int Year = 2013
if `first' { 
 quietly save `all', replace 
 local first = 0 
 }
 else { 
 append using `all' 
 quietly save `all', replace 
 }
filefilter "$TD/ta2015/TA2015.do" "$SD/w.do", from("[path]\BS") to("$TD/ta2015/") replace
clear
quietly do "$SD/w.do"
keep TA150002 TA150003 TA150004 TA150247 TA150269 TA150291 TA150313 TA150335 TA150345 TA150346 TA150011
rename (TA150002 TA150003 TA150004) (tas_iw_id iw_num seq_num)
rename TA150247 earn1
rename TA150269 earn2
rename TA150291 earn3
rename TA150313 earn4
rename TA150335 earn5
rename TA150345 weeks
rename TA150346 hours
rename TA150011 hwstat
gen int Year = 2015
if `first' { 
 quietly save `all', replace 
 local first = 0 
 }
 else { 
 append using `all' 
 quietly save `all', replace 
 }
filefilter "$TD/TA2017/TA2017.do" "$SD/w.do", from("[path]\BS") to("$TD/TA2017/") replace
clear
quietly do "$SD/w.do"
keep TA170002 TA170003 TA170004 TA170238 TA170253 TA170268 TA170283 TA170298 TA170188 TA170190 TA170006
rename (TA170002 TA170003 TA170004) (tas_iw_id iw_num seq_num)
rename TA170238 earn1
rename TA170253 earn2
rename TA170268 earn3
rename TA170283 earn4
rename TA170298 earn5
rename TA170188 weeks
rename TA170190 hours
rename TA170006 hwstat
gen int Year = 2017
if `first' { 
 quietly save `all', replace 
 local first = 0 
 }
 else { 
 append using `all' 
 quietly save `all', replace 
 }
filefilter "$TD/TA2019/TA2019.do" "$SD/w.do", from("[path]\BS") to("$TD/TA2019/") replace
clear
quietly do "$SD/w.do"
keep TA190002 TA190003 TA190004 TA190292 TA190319 TA190346 TA190373 TA190400 TA190230 TA190232 TA190005
rename (TA190002 TA190003 TA190004) (tas_iw_id iw_num seq_num)
rename TA190292 earn1
rename TA190319 earn2
rename TA190346 earn3
rename TA190373 earn4
rename TA190400 earn5
rename TA190230 weeks
rename TA190232 hours
rename TA190005 hwstat
gen int Year = 2019
if `first' { 
 quietly save `all', replace 
 local first = 0 
 }
 else { 
 append using `all' 
 quietly save `all', replace 
 }
filefilter "$TD/TA2021/TA2021.do" "$SD/w.do", from("[path]\BS") to("$TD/TA2021/") replace
clear
quietly do "$SD/w.do"
keep TA210002 TA210003 TA210004 TA210292 TA210319 TA210346 TA210373 TA210400 TA210225 TA210227 TA210005
rename (TA210002 TA210003 TA210004) (tas_iw_id iw_num seq_num)
rename TA210292 earn1
rename TA210319 earn2
rename TA210346 earn3
rename TA210373 earn4
rename TA210400 earn5
rename TA210225 weeks
rename TA210227 hours
rename TA210005 hwstat
gen int Year = 2021
if `first' { 
 quietly save `all', replace 
 local first = 0 
 }
 else { 
 append using `all' 
 quietly save `all', replace 
 }
filefilter "$TD/TA2023/TA2023.do" "$SD/w.do", from("[path]\BS") to("$TD/TA2023/") replace
clear
quietly do "$SD/w.do"
keep TA230002 TA230003 TA230004 TA230315 TA230342 TA230369 TA230396 TA230423 TA230244 TA230246 TA230005
rename (TA230002 TA230003 TA230004) (tas_iw_id iw_num seq_num)
rename TA230315 earn1
rename TA230342 earn2
rename TA230369 earn3
rename TA230396 earn4
rename TA230423 earn5
rename TA230244 weeks
rename TA230246 hours
rename TA230005 hwstat
gen int Year = 2023
if `first' { 
 quietly save `all', replace 
 local first = 0 
 }
 else { 
 append using `all' 
 quietly save `all', replace 
 }
use `all', clear
merge m:1 Year iw_num seq_num using "/srv/project/speech/apps/Child_Time_Study/Temp/_tas_bridge.dta", keep(master match) keepusing(Fam_id Per_id ind_age)
tab _merge
drop _merge
* job earnings: 9,999,999 and -999,999 = DK/NA/RF; 0 = no such job; a DK on any job makes the total unknown
gen byte earn_dk = 0
forvalues k = 1/5 {
 replace earn_dk = 1 if inlist(earn`k', 9999999, -999999)
}
egen double earn_tot = rowtotal(earn1-earn5) if earn_dk == 0
gen byte wk_ok = inrange(weeks, 1, 52)
gen byte hr_ok = inrange(hours, 1, 112)
gen byte has_earn = earn_tot > 0 & !missing(earn_tot)
gen byte has_wage = has_earn & wk_ok & hr_ok
gen double lnwage = ln(earn_tot / (weeks * hours)) if has_wage
gen byte headwife = inlist(hwstat, 1, 2)
save "$SD/tas_wages.dta", replace
di "=== all TAS respondent-waves: usable hourly wage share by head/wife status and wave"
table Year headwife, statistic(mean has_wage) statistic(mean has_earn) nformat(%5.3f)
merge m:1 Fam_id Per_id using "/srv/project/speech/apps/Child_Time_Study/Output/Data/SMM/SMM_TAS_Micro.dta", keep(match) keepusing(ach_age lw_raw y_2526 has_2526 birth_year) nogenerate
keep if ach_age == 17 & !missing(lw_raw)
gen ba = cond(has_2526 == 1, y_2526, .)
label define bal 0 "no BA" 1 "BA" , replace
label values ba bal
gen int agew = ind_age
di "=== children with a score at 17: respondent-waves by age at interview, BA status at 25/26, usable hourly wage"
table agew ba if has_wage, statistic(frequency) missing
di "=== children (unique) with >= 1 usable hourly wage, by BA status and minimum age"
foreach a in 18 22 24 25 {
 bysort Fam_id Per_id: egen byte anyw`a' = max(has_wage & agew >= `a')
 bysort Fam_id Per_id: egen byte anye`a' = max(has_earn & agew >= `a')
}
bysort Fam_id Per_id: gen byte first = _n == 1
foreach a in 18 22 24 25 {
 di "--- age >= `a': children with a usable hourly wage / with positive earnings"
 tab ba anyw`a' if first, missing
 tab ba anye`a' if first, missing
}
di "=== waves per child with a usable wage at age >= 22"
bysort Fam_id Per_id: egen int nw22 = total(has_wage & agew >= 22)
tab nw22 ba if first, missing
di "=== head/wife share among their usable-wage waves (2007-13 hours coded 0 for heads/wives)"
table Year headwife if agew >= 22, statistic(frequency) statistic(mean has_wage) nformat(%5.3f)
save "$SD/cds17_wages.dta", replace
