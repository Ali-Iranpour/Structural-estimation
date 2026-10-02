# =============================================================================
# smm_test_fixtures.jl -- SYNTHETIC inputs for testing the memo-19 SMM before the real
# ones exist. NOTHING HERE MAY ENTER AN ESTIMATION ON DATA.
#
#   fixture_composition(targets)   a [composition] built from the S1 per-age N
#   PLACEHOLDER_CHILD_WAGE         a wage loading on skill for a model that has to run
#
# WHY THEY EXIST. Two inputs of the memo-19 estimation are NOT PROVIDED yet:
#   * the data's age composition of every pooled S frame (docs/SMM_COMPOSITION.md), which
#     28_smm_moments.do has to export; load_targets refuses a file without it;
#   * sd(log AFQT) for the alpha_theta anchoring (child_wage_config refuses without it).
# A parameter-recovery test does not need either to be RIGHT -- it asks whether the
# estimator returns the parameters that generated a set of moments, at SOME fixed model --
# but it needs both to EXIST. These are those stand-ins, named so they cannot be mistaken
# for data.
# =============================================================================

"""
    fixture_composition(targets) -> NamedTuple of frames

O_LW is EXACT (the S1 per-age N). Every other frame spreads each moment's N over the ages of
its bin in proportion to the S1 counts at those ages (largest-remainder rounding, so the bin
totals match the moments' N exactly and `check_composition` passes). Pairs end five years
after the base age. Money frames at CDS waves put half their rows at odd waves. A FIXTURE:
the real counts differ (diary sets include 2019, money uses one-child families, pairs can be
4 or 6 years apart).
"""
function fixture_composition(targets)
    s1 = Dict(a => targets["S1_mean_LW_age$a"].n for a in 3:17)
    function alloc(ages, total)
        w = [s1[a] for a in ages]; q = total .* w ./ sum(w)
        n = floor.(Int, q); r = total - sum(n)
        for i in sortperm(q .- n; rev = true)[1:r]; n[i] += 1; end
        return n
    end
    nn(k) = targets[k].n
    function level(bins; odd = false)
        a = Int[]; n = Int[]
        for (lo, hi, tot) in bins
            append!(a, lo:hi); append!(n, alloc(lo:hi, tot))
        end
        return (a = a, a2 = nothing, n = n, n_odd = odd ? n .÷ 2 : nothing)
    end
    function pair(bins; odd = false)
        f = level(bins; odd = odd)
        return (a = f.a, a2 = f.a .+ 5, n = f.n, n_odd = f.n_odd)
    end
    O_LW = (a = collect(3:17), a2 = nothing, n = [s1[a] for a in 3:17], n_odd = nothing)
    return (
        O_LW   = O_LW,
        O_taup = level([(3, 17, nn("S6_corr_taup_LW_3_17"))]),
        O_tauc = level([(6, 17, nn("S6_corr_tauc_LW_6_17"))]),
        O_ep   = level([(3, 17, nn("S6_corr_ep_LW_3_17"))]; odd = true),
        P_LW   = pair([(3, 7, nn("S4_mean_dLW_base3_7")), (8, 12, nn("S4_mean_dLW_base8_12"))]),
        P_taup = pair([(3, 7, nn("S7_corr_taup_dLW_base3_7")), (8, 12, nn("S7_corr_taup_dLW_base8_12"))]),
        P_tauc = pair([(6, 7, nn("S7_corr_tauc_dLW_base6_7")), (8, 12, nn("S7_corr_tauc_dLW_base8_12"))]),
        P_ep   = pair([(3, 7, nn("S7_corr_ep_dLW_base3_7")), (8, 12, nn("S7_corr_ep_dLW_base8_12"))]; odd = true),
        D_taup = level([(3, 5, nn("S8_mean_taup_3_5")), (6, 8, nn("S8_mean_taup_6_8")),
                        (9, 12, nn("S8_mean_taup_9_12")), (13, 17, nn("S8_mean_taup_13_17"))]),
        D_tauc = level([(6, 8, nn("S8_mean_tauc_6_8")), (9, 12, nn("S8_mean_tauc_9_12")),
                        (13, 17, nn("S8_mean_tauc_13_17"))]),
        E_ep   = level([(3, 5, nn("S8_mean_ep_3_5")), (6, 8, nn("S8_mean_ep_6_8")),
                        (9, 12, nn("S8_mean_ep_9_12")), (13, 17, nn("S8_mean_ep_13_17"))]),
        E_epY  = level([(3, 17, nn("S9_mean_ep_over_Y_3_17"))]),
    )
end

# A PLACEHOLDER wage loading: 0.2 log points of wage per SD of childhood skill, i.e.
# alpha_theta = 0.2 / sd_lnk17 = 0.307, with Daruich & Fernandez's college/high-school ratio
# 0.976/0.654. Chosen only so the model runs; NOT the anchored value, which needs sd(log AFQT).
const PLACEHOLDER_CHILD_WAGE = let a = 0.2 / SD_LNK17
    (alpha_theta = a, alpha_thetaE = a * 0.322 / 0.654, m_theta = CHILD_DEFAULTS.m_psychic,
     lnw0 = log(CHILD_DEFAULTS.w) - 0.4144)
end
