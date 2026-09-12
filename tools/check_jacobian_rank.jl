#!/usr/bin/env julia
# =============================================================================
# check_jacobian_rank.jl -- local identification of the 16-parameter specification.
#
#     julia --project=.. tools/check_jacobian_rank.jl <targets.toml> [simNs] [seeds] [init_from]
#     e.g.  ... targets.toml 400,1000 1234,20260910 output/smm_runs/2026-09-10_183649/estimates.toml
#
# PARAMETER AND MOMENT COUNTS DO NOT ESTABLISH IDENTIFICATION. 17 >= 16 is necessary and
# says nothing about whether the seventeen moments can separate the sixteen parameters.
# R_1 is fixed at 0 and is not a column.
# This computes the residual Jacobian by central differences, at more than one step size
# and at more than one point, and reports its rank, conditioning and least-identified
# direction.
#
# WHY TWO POINTS. At the incumbent calibration the model's college share is EXACTLY ZERO,
# so all six completion moments sit on a corner: a symmetric difference across a kappa
# either stays at zero (no signal) or jumps (a difference quotient of a step function).
# A Jacobian there describes a corner, not the objective's local geometry. The second
# point puts the college margin in the interior, which is where the estimation will
# actually operate, and it is the one to read.
#
# WHY MORE THAN ONE SEED AND SIMULATION SIZE. Full numerical rank on a single draw of 400
# households is not evidence of robust identification. If rank, conditioning or the
# least-identified direction move with the seed, the finding belongs to the draw. Pass
# comma-separated lists as argument 2 (simN) and 3 (seeds) to widen or narrow the sweep.
#
# WHY THREE STEP SIZES. A derivative computed by finite differences on a simulated
# objective is a ratio of two noisy numbers. If rank or conditioning move a lot between
# 2%, 5% and 10% of the box, the "derivative" is measuring simulation noise rather than
# curvature, and no conclusion about identification survives.
#
# Columns are scaled to a FULL-BOX move, so they are comparable across log and level links
# and across parameters in different units. Pairwise geometry is invariant to that scaling;
# the condition number is not, which is why the scaling is stated rather than implied.
# =============================================================================

using Printf, Random, NLopt, LinearAlgebra, Interpolations, Statistics, Distributions,
      QuantEcon, FastGaussQuadrature, Parameters, Dierckx, ProgressMeter, TOML
const REPO = normpath(joinpath(@__DIR__, ".."))

include(joinpath(REPO, "code", "src", "paths.jl")); include(joinpath(REPO, "code", "src", "child_lifecycle.jl"))
include(joinpath(REPO, "code", "src", "parent_family.jl")); include(joinpath(REPO, "code", "smm", "moments.jl"))

const T = load_targets(ARGS[1])
check_psychic_centring(target_m_psychic(T))
# The EXPERIMENT'S shock discretisations: five taste nodes and five HC-shock nodes (the
# parent constructor default). Small state grids: this is a geometry check.
const G = (Na = 16, Nhc = 16); const CG = (Na = 16, Nk = 16, Nt = 5)

# SIMULATION SIZES AND SEEDS, not one of each.
#
# Full numerical rank on ONE draw of 400 households is not evidence of robust
# identification: the Jacobian is a ratio of differences between two simulated objectives,
# and at a small simN the smallest singular value can be simulation noise rather than
# curvature. If rank or the least-identified direction moves with the seed, the conclusion
# belongs to the draw and not to the model. Overridable from the command line for a
# cheaper or a more thorough pass.
const SIMS  = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : [400, 1000]
const SEEDS = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : [1234, 20260910]
const INIT_FROM = length(ARGS) >= 4 ? ARGS[4] :
    joinpath(REPO, "output", "smm_runs", "2026-09-10_183649", "estimates.toml")

# THE ACTUAL WARM START: the baseline estimates by name, the two new parameters at their
# SMM starts (sigma_eta 0.03, sigma_eps 0.5) -- exactly what `--init-from` hands the run.
function warm_start()
    v = Dict{Symbol,Float64}(q.name => smm_start(q.name) for q in SMM_PARAMS)
    if isfile(INIT_FROM)
        est = TOML.parsefile(INIT_FROM)["parameters"]
        for q in SMM_PARAMS
            haskey(est, String(q.name)) && (v[q.name] = Float64(est[String(q.name)]))
        end
    else
        @warn "init file not found; using the SMM starts" INIT_FROM
    end
    v
end
# At the warm start the college share is low (0.2 at the full grid, less here), so the
# TAS moments sit near a corner for kappa_ParEd (g0 = 0). The Jacobian is therefore also
# taken at a point where completion is well interior for BOTH parental-education groups.
const POINTS = Dict(
  "warm start (init-from)" => warm_start(),
  "interior completion"    => (v = warm_start(); v[:kappa_0] = -0.30; v[:kappa_ParEd] = -0.40;
                               v[:kappa_theta] = -6.0; v[:sigma_eps] = 0.8; v))

evalr(v, n, sd) = evaluate_at(v, T; Na = G.Na, Nk = 2, Nhc = G.Nhc, simN = n, seed = sd,
                              child_grid = CG).r

for (label, th0) in POINTS, N in SIMS, SD in SEEDS
    @printf("\n%s\nPOINT: %s   simN = %d   seed = %d\n", "="^96, label, N, SD)
    for q in SMM_PARAMS; @printf("  %-15s %9.4f\n", q.name, th0[q.name]); end
    for frac in (0.02, 0.05, 0.10)
        J = zeros(length(SMM_MOMENTS), length(SMM_PARAMS))
        for (j, q) in enumerate(SMM_PARAMS)
            zlo = q.link === :log ? log(q.lo) : q.lo
            zhi = q.link === :log ? log(q.hi) : q.hi
            z0  = q.link === :log ? log(th0[q.name]) : th0[q.name]
            h   = frac * (zhi - zlo)
            zp, zm = min(z0 + h/2, zhi), max(z0 - h/2, zlo)
            vp, vm = copy(th0), copy(th0)
            vp[q.name] = q.link === :log ? exp(zp) : zp
            vm[q.name] = q.link === :log ? exp(zm) : zm
            # scaled to a FULL-BOX move so columns are comparable across links and units
            J[:, j] = (evalr(vp, N, SD) .- evalr(vm, N, SD)) ./ (zp - zm) .* (zhi - zlo)
        end
        # A perturbed point at which a completion group is EMPTY returns NaN residuals -- a
        # controlled invalid evaluation, not a derivative. Those columns are reported and
        # left out of the SVD rather than zeroed (a zero column would fake a rank drop).
        bad = [j for j in 1:length(SMM_PARAMS) if any(!isfinite, J[:, j])]
        if !isempty(bad)
            @printf("\n  step %.0f%% of box:  INVALID columns (empty completion group at a perturbed point): %s\n",
                    100frac, join((String(SMM_PARAMS[j].name) for j in bad), ", "))
            @printf("    rank/conditioning below are over the %d finite columns only\n", length(SMM_PARAMS) - length(bad))
        end
        keep = setdiff(1:length(SMM_PARAMS), bad)
        length(keep) >= 2 || continue
        Jk = J[:, keep]; PK = SMM_PARAMS[keep]
        F = svd(Jk)
        tol = maximum(size(J)) * eps() * F.S[1]
        rk  = count(>(tol), F.S)
        @printf("\n  step %.0f%% of box:  rank %d/%d   cond %.3g   smallest sv %.4g\n",
                100frac, rk, length(PK), F.S[1]/F.S[end], F.S[end])
        @printf("    singular values: %s\n", join((@sprintf("%.3g", x) for x in F.S), " "))
        # column norms: which parameters the moments barely see at all
        cn = [norm(Jk[:, j]) for j in 1:length(PK)]
        ord = sortperm(cn)
        @printf("    weakest columns: %s\n",
                join((@sprintf("%s %.3g", PK[i].name, cn[i]) for i in ord[1:min(4, end)]), ", "))
        # the two directions the data is least able to resolve
        for c in (length(F.S), length(F.S) - 1)
            vmin = F.V[:, c]
            big = sortperm(abs.(vmin); rev = true)[1:min(3, end)]
            @printf("    weak direction (sv %.3g): %s\n", F.S[c],
                    join((@sprintf("%+.2f*%s", vmin[i], PK[i].name) for i in big), " "))
        end
        # THE TWO TRADE-OFFS THE EXPERIMENT INTRODUCES, as column cosines (1 = the moments
        # cannot tell the two parameters apart; 0 = orthogonal):
        #   sigma_eta vs kappa_theta -- both move the ability gap; sd_ga17 is what breaks it
        #   sigma_eps vs the psychic-cost levels -- the scale/level ambiguity of a probit;
        #                                           kse_w_gap is what breaks it
        idx(n) = findfirst(q -> q.name === n, SMM_PARAMS)
        okc(n) = !(idx(n) in bad)
        cosn(a, b) = !(okc(a) && okc(b)) ? NaN :
            (x = J[:, idx(a)]; y = J[:, idx(b)]; abs(dot(x, y)) / max(norm(x) * norm(y), eps()))
        @printf("    trade-offs |cos|: sigma_eta~kappa_theta %.3f;  sigma_eps~kappa_0 %.3f, ~kappa_theta %.3f, ~kappa_ParEd %.3f\n",
                cosn(:sigma_eta, :kappa_theta), cosn(:sigma_eps, :kappa_0),
                cosn(:sigma_eps, :kappa_theta), cosn(:sigma_eps, :kappa_ParEd))
        # rows: which moments each new parameter moves most (scaled sensitivities)
        for n in (:sigma_eta, :sigma_eps)
            okc(n) || continue
            col = J[:, idx(n)] ./ target_se(T)          # full-box move, in data SEs
            big = sortperm(abs.(col); rev = true)[1:3]
            @printf("    %s moves (full-box, in data SEs): %s\n", n,
                    join((@sprintf("%s %+.1f", SMM_MOMENTS[i], col[i]) for i in big), ", "))
        end
    end
end
println("\ndone")
