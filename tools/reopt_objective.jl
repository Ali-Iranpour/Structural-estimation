# =============================================================================
# reopt_objective.jl -- the OBJECTIVE-AFFECTING adapter code of tools/reopt.jl
# (moved out of it unchanged on 2026-09-28, tiktak_fix_plan.md follow-up 2)
#
# Everything in this file changes Q at a fixed search vector, so its source is hashed into the
# run's objective identity (tools/reopt_identity.jl, `adapter_sha`); reopt.jl keeps only
# orchestration and reporting, which cannot. Included on every process after moments.jl,
# run_bounds.jl and the run's constants: ZFIX, FREE_IDX, NF, CE, FREE_EXTRA_, TARGETS, PGRID,
# PSIMN, PSEED, PE_ARG_, EXTRA_.
# =============================================================================

"The full SMM search vector of a free vector: the fixed coordinates from ZFIX, the free ones from `zf`."
embed(zf::AbstractVector{Float64}) = (z = copy(ZFIX); z[FREE_IDX] .= view(zf, 1:NF); z)

"The child overrides of a free vector: under --free the last coordinate is omega (level link)."
ce_of(zf) = FREE_EXTRA_ === nothing ? CE : merge(CE, (omega = zf[NF + 1],))

"The objective: smm_objective -- the audited one, with its named penalties -- at the run's grids and draws."
obj(zf) = smm_objective(embed(zf), TARGETS; Na = PGRID, Nk = 2, Nhc = PGRID,
                        simN = PSIMN, seed = PSEED, child_grid = (Na = 30, Nk = 30, Nt = 5),
                        demo_sim = false, child_extra = ce_of(zf), parent_extra = PE_ARG_, extra_moments = EXTRA_)
