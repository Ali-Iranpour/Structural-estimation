#!/usr/bin/env julia
# =============================================================================
# check_jacobian_rank.jl -- local identification of the 14-parameter specification.
#
#     julia --project=.. tools/check_jacobian_rank.jl <targets.toml> [simNs] [seeds]
#     e.g.  ... targets.toml 400,1000 1234,20260910      (the defaults)
#
# PARAMETER AND MOMENT COUNTS DO NOT ESTABLISH IDENTIFICATION. 17 >= 14 is necessary and
# says nothing about whether the seventeen moments can separate the fourteen parameters.
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
const G = (Na = 16, Nhc = 16); const CG = (Na = 16, Nk = 16, Nt = 3)

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
base() = Dict{Symbol,Float64}(q.name => param_default(q.name) for q in SMM_PARAMS)

# The incumbent has a ZERO college share, so every TAS moment is a corner and its
# derivative is one-sided at best. The Jacobian is therefore also taken at a point where
# the college margin is interior -- which is the point that matters for identification.
const POINTS = Dict(
  "incumbent"         => base(),
  "interior college"  => (v = base(); v[:kappa_0] = -0.55; v[:kappa_theta] = -3.0;
                          v[:kappa_ParEd] = -0.30; v[:kappa_terminal] = 12.0; v))

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
        F = svd(J)
        tol = maximum(size(J)) * eps() * F.S[1]
        rk  = count(>(tol), F.S)
        @printf("\n  step %.0f%% of box:  rank %d/%d   cond %.3g   smallest sv %.4g\n",
                100frac, rk, length(SMM_PARAMS), F.S[1]/F.S[end], F.S[end])
        # column norms: which parameters the moments barely see at all
        cn = [norm(J[:, j]) for j in 1:length(SMM_PARAMS)]
        ord = sortperm(cn)
        @printf("    weakest columns: %s\n",
                join((@sprintf("%s %.3g", SMM_PARAMS[i].name, cn[i]) for i in ord[1:4]), ", "))
        # the direction the data is least able to resolve
        vmin = F.V[:, end]
        big = sortperm(abs.(vmin); rev = true)[1:3]
        @printf("    least-identified direction: %s\n",
                join((@sprintf("%+.2f*%s", vmin[i], SMM_PARAMS[i].name) for i in big), " "))
    end
end
println("\ndone")
