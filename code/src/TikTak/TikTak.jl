# =============================================================================
# TikTak.jl -- the TikTak global optimizer as an importable module.
#
# Arnoud, Guvenen & Kleineberg (2022), "Benchmarking Global Optimizers": Section 2.1
# (the algorithm), Section 3.3 (polishing) and Appendix A.6, footnotes 26-27, in the
# PDF under docs/papers. Reference implementation: https://github.com/serdarozkan/TikTak
#
# 2026-09-27 (tiktak_fix_plan.md, J1): the optimizer used to be a file of top-level
# definitions that relied on NLopt and Printf being loaded in Main. It is now a module
# with its own imports, so it can be loaded and tested without starting an estimation
# or any worker. `code/src/tiktak.jl` is the adapter every existing caller includes.
# =============================================================================
module TikTak

import NLopt
import Sobol
import Distributed
import SHA
import TOML
import Dates
using Printf: @printf, @sprintf

export tiktak, tiktak_selftest, ret_class, ret_tally, TikTakResult

# `search_budget_complete(result)`, `work_known(result)`, `state_violations(state)`: reached as
# TikTak.name, like the rest of the module (see code/src/tiktak.jl).

const TIKTAK_VERSION = "2.3.0-dev"      # 2.3 (v1, 2026-10-02): penalised mixed start -> own seed; a known start value is reused; 2.2: bootstrap = :immediate_mixed (2026-09-29); 2.1: checkpoint schema 2
const MODULE_DIR = @__DIR__

include("config.jl")
include("types.jl")
include("status.jl")
include("localsearch.jl")
include("pretest.jl")
include("identity.jl")
include("state.jl")
include("checkpoint.jl")
include("workers.jl")
include("scheduler.jl")
include("presets.jl")
include("driver.jl")
include("selftest.jl")

end # module TikTak
