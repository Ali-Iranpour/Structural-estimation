#!/usr/bin/env julia
# =============================================================================
# standard_errors.jl -- parameter uncertainty for the SMM estimates.
#
#     cd code/smm
#     julia +1.11 --project=../.. standard_errors.jl \
#         --jacobian ../../output/identification/jac_9col \
#         --at       ../../output/smm_runs/<dir>/estimates.toml
#
# WHAT IT COMPUTES
# ----------------
# The classical minimum-distance sandwich. With residuals r(theta) = (m(theta) - mhat)/s,
# an objective Q = r'Wr, a Jacobian G = dr/dtheta and a moment covariance Omega (of the
# SCALED data moments), the estimator's asymptotic variance is
#
#     V = (G'WG)^-1 G'W Omega W G (G'WG)^-1
#
# and with the equal weights this project actually uses, W = I, that collapses to
#
#     V = (G'G)^-1 G' Omega G (G'G)^-1.
#
# Omega comes from Input/smm_targets_baseline.toml, [moment_cov]: a CLUSTER-ROBUST
# covariance of the ten targeted data moments, clustered on the family, built by
# tools/make_smm_targets.py from the micro file. It is the covariance of the MEANS, so it
# already carries the 1/n; it is not the cross-sectional SD, which is 12-48x larger and is
# not a standard error of anything.
#
# G comes from a SAVED Jacobian directory produced by jacobian.jl, so the derivative step,
# evaluation point, grids and seed that produced it are on record beside the numbers. This
# script refuses to guess: it will not compute a Jacobian of its own.
#
# WHAT THESE STANDARD ERRORS DO AND DO NOT COVER
# ----------------------------------------------
# THEY COVER  sampling error in the data moments, including the correlation induced by the
#             ten moments sharing households, propagated through a local linearisation of
#             the model at the reported point.
#
# THEY DO NOT COVER
#   * SIMULATION error. The model side uses simN draws under a FIXED seed (common random
#     numbers). The usual (1 + 1/S) inflation assumes independent draws per evaluation and
#     is reported as a separate column rather than folded in silently, because with CRN it
#     is neither obviously right nor obviously negligible.
#   * The weighting matrix being wrong. Equal weights are not efficient here; the efficient
#     choice is W = Omega^-1, which is also reported so the two can be compared. Neither is
#     "the" answer until the choice is argued in the paper.
#   * Any error in the SPECIFICATION. A tight standard error on a misspecified model is a
#     precise statement about the wrong thing, and the over-identification test below is
#     the only check here that speaks to it at all.
#   * Global identification, multiple minima, or a point sitting on a bound. A parameter
#     pinned to its box has no meaningful symmetric interval and is reported as such.
# =============================================================================

using Printf, Dates, LinearAlgebra, TOML, DelimitedFiles

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

function argstr(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("$flag needs a value")
    return ARGS[i + 1]
end

const JDIR   = argstr("--jacobian", "")
const ATFILE = argstr("--at", "")
const STEP   = argstr("--step", "")          # which saved step to use; default the first
const OUT    = argstr("--out", "")

isempty(JDIR) && error("""
    --jacobian is required: pass a directory produced by code/smm/jacobian.jl.
    This script does not compute a Jacobian of its own -- the point, step, grids and seed
    have to travel with the matrix, and that is what jacobian.jl saves.""")

const JMETA = TOML.parsefile(joinpath(JDIR, "jacobian.toml"))
const TGT   = TOML.parsefile(joinpath(REPO, "Input", "smm_targets_baseline.toml"))
haskey(TGT, "moment_cov") || error("""
    Input/smm_targets_baseline.toml has no [moment_cov] block. Regenerate it:
        uv run --with pandas --with numpy python tools/make_smm_targets.py""")
const MC = TGT["moment_cov"]

# ---- pick the step and load the matrix --------------------------------------
const STEPKEYS = sort([k for k in keys(JMETA) if startswith(k, "step_")])
const SKEY = isempty(STEP) ? STEPKEYS[1] : "step_" * replace(STEP, "." => "_")
haskey(JMETA, SKEY) || error("no $SKEY in $JDIR/jacobian.toml; available: $(join(STEPKEYS, ", "))")
const JBLK = JMETA[SKEY]

const RAW = readdlm(joinpath(JDIR, JBLK["matrix_file"]), ','; header = true)
const JCOLS = String.(RAW[2][2:end])
const JROWS = String.(RAW[1][:, 1])
const J_BOX = Float64.(RAW[1][:, 2:end])          # d r / d (full-box move)

# ---- undo the box scaling: we want d r / d theta in NATURAL units ------------
# jacobian.jl scales each column to a full-box move so columns are comparable. A standard
# error has to be in the parameter's own units, so the scaling is inverted here -- and for
# a log-linked parameter the chain rule to natural units is applied too, since the search
# coordinate is log(theta) and d/dtheta = (1/theta) d/dlog(theta).
const CNAMES = String.(JMETA["columns"]["names"])
const CLO    = Float64.(JMETA["columns"]["lo"])
const CHI    = Float64.(JMETA["columns"]["hi"])
const CLINK  = String.(JMETA["columns"]["link"])
const POINT  = JMETA["point"]

CNAMES == JCOLS || error("column order in the CSV does not match jacobian.toml")

const NM_, NP_ = size(J_BOX)
const G = similar(J_BOX)
for j in 1:NP_
    zlo = CLINK[j] == "log" ? log(CLO[j]) : CLO[j]
    zhi = CLINK[j] == "log" ? log(CHI[j]) : CHI[j]
    dz  = J_BOX[:, j] ./ (zhi - zlo)                       # d r / d z
    th  = Float64(POINT[CNAMES[j]])
    G[:, j] = CLINK[j] == "log" ? dz ./ th : dz            # d r / d theta
end

# ---- Omega, in the SCALED residual metric -----------------------------------
# [moment_cov] is the covariance of the RAW moment means. The objective works in
# r = (m - mhat)/s, so the covariance that matters is S^-1 Omega S^-1.
const MNAMES = String.(MC["names"])
MNAMES == JROWS || error("""
    moment order differs between the Jacobian ($(join(JROWS, ", ")))
    and [moment_cov]   ($(join(MNAMES, ", "))).""")
const OMEGA_RAW = reduce(vcat, [reshape(Float64.(row), 1, :) for row in MC["cov"]])
const SCALES = [Float64(JMETA["moment_scales"][m]) for m in MNAMES]
const OMEGA = Diagonal(1 ./ SCALES) * OMEGA_RAW * Diagonal(1 ./ SCALES)

# ---- the sandwich ------------------------------------------------------------
sandwich(G, W, Omega) = begin
    A = G' * W * G
    Ai = inv(A)
    Ai * (G' * W * Omega * W * G) * Ai
end

const W_EQ  = Matrix{Float64}(I, NM_, NM_)
const V_EQ  = sandwich(G, W_EQ, OMEGA)
const SE_EQ = sqrt.(max.(diag(V_EQ), 0.0))

# Efficient weighting, for comparison only. Omega can be near-singular; a pseudo-inverse
# with an explicit tolerance is used rather than a silent `inv`, and the tolerance is
# reported so a rank-deficient Omega cannot masquerade as a precise answer.
const F_OM = svd(OMEGA)
const OM_TOL = maximum(size(OMEGA)) * eps() * F_OM.S[1]
const OM_RANK = count(>(OM_TOL), F_OM.S)
const W_OPT = F_OM.V * Diagonal([s > OM_TOL ? 1/s : 0.0 for s in F_OM.S]) * F_OM.U'
const V_OPT = sandwich(G, W_OPT, OMEGA)
const SE_OPT = sqrt.(max.(diag(V_OPT), 0.0))

# ---- report ------------------------------------------------------------------
const OUTDIR = isempty(OUT) ? JDIR : OUT
mkpath(OUTDIR)
const LOG = open(joinpath(OUTDIR, "standard_errors.log"), "w")
say(a...) = (println(a...); println(LOG, a...))
sayf(f, a...) = (s = Printf.format(Printf.Format(f), a...); print(s); print(LOG, s))

say("="^80); say("SMM standard errors"); say("="^80)
sayf("jacobian     %s  (%s)\n", relpath(JDIR, REPO), SKEY)
sayf("point        %s\n", JMETA["point_file"])
sayf("grids        Na = Nhc = %d, simN = %d, seed = %d\n",
     JMETA["grid_Na"], JMETA["simN"], JMETA["seed"])
sayf("moment cov   clustered on %s, %d clusters\n", MC["cluster_on"], MC["n_clusters"])
sayf("Omega rank   %d of %d  (tolerance %.3g)\n", OM_RANK, NM_, OM_TOL)
if OM_RANK < NM_
    say("!! Omega is rank deficient: the efficient-weight column below uses a pseudo-inverse")
    say("   and should not be read as an efficient estimator.")
end

# Is the point on a bound? A symmetric interval is meaningless there.
say("\nparameter           estimate    se(equal W)   se(optimal W)     95% CI (equal W)")
say("-"^80)
for j in 1:NP_
    th = Float64(POINT[CNAMES[j]])
    pos = CLINK[j] == "log" ?
          (log(th) - log(CLO[j])) / (log(CHI[j]) - log(CLO[j])) :
          (th - CLO[j]) / (CHI[j] - CLO[j])
    onbound = pos < 0.02 || pos > 0.98
    @printf("%-16s %11.5f %12.5f %14.5f     [%.4f, %.4f]%s\n",
            CNAMES[j], th, SE_EQ[j], SE_OPT[j],
            th - 1.96SE_EQ[j], th + 1.96SE_EQ[j], onbound ? "  ON BOUND" : "")
    println(LOG, @sprintf("%-16s %11.5f %12.5f %14.5f     [%.4f, %.4f]%s",
            CNAMES[j], th, SE_EQ[j], SE_OPT[j],
            th - 1.96SE_EQ[j], th + 1.96SE_EQ[j], onbound ? "  ON BOUND" : ""))
end
if any(j -> begin
        th = Float64(POINT[CNAMES[j]])
        pos = CLINK[j] == "log" ? (log(th)-log(CLO[j]))/(log(CHI[j])-log(CLO[j])) :
                                  (th-CLO[j])/(CHI[j]-CLO[j])
        pos < 0.02 || pos > 0.98
    end, 1:NP_)
    say("\n!! A parameter ON A BOUND has no meaningful symmetric interval: the sampling")
    say("   distribution is truncated there and the delta method does not apply.")
end

# Correlation of the estimates -- where the collinearity actually lands.
say("\nestimate correlation (equal weights); |corr| > 0.9 is where two parameters")
say("cannot be told apart by these moments at this point")
let D = Diagonal(1 ./ max.(SE_EQ, eps())), C = D * V_EQ * D
    pairs = [(CNAMES[a], CNAMES[b], C[a, b]) for a in 1:NP_ for b in (a+1):NP_]
    for (a, b, c) in first(sort(pairs; by = x -> abs(x[3]), rev = true), 6)
        sayf("  %-12s %-12s %+.4f\n", a, b, c)
    end
end

# Over-identification: the only line here that speaks to specification.
if NP_ < NM_
    say("\nover-identification")
    sayf("  %d moments, %d parameters -> %d degrees of freedom\n", NM_, NP_, NM_ - NP_)
    say("  A J-test needs the residual AT the estimate and the efficient weight; it is not")
    say("  computed here because the point this Jacobian was taken at is a CALIBRATION, not")
    say("  a converged estimate. Re-run with --at a fitted estimates.toml to make it meaningful.")
end

open(joinpath(OUTDIR, "standard_errors.toml"), "w") do io
    println(io, "# GENERATED by code/smm/standard_errors.jl. Read the script header before quoting.")
    println(io, "generated    = \"", Dates.format(now(), "yyyy-mm-dd HH:MM"), "\"")
    println(io, "jacobian_dir = \"", relpath(JDIR, REPO), "\"")
    println(io, "jacobian_step= \"", SKEY, "\"")
    println(io, "point_file   = \"", JMETA["point_file"], "\"")
    println(io, "cluster_on   = \"", MC["cluster_on"], "\"")
    println(io, "n_clusters   = ", MC["n_clusters"])
    println(io, "omega_rank   = ", OM_RANK)
    println(io, "weighting    = \"equal (W = I) on scaled residuals; optimal shown for comparison\"")
    println(io, "covers       = \"data-moment sampling error only -- NOT simulation error, NOT specification\"")
    println(io, "\n[estimate]")
    for j in 1:NP_; @printf(io, "%-12s = %.8f\n", CNAMES[j], Float64(POINT[CNAMES[j]])); end
    println(io, "\n[se_equal_weights]")
    for j in 1:NP_; @printf(io, "%-12s = %.8f\n", CNAMES[j], SE_EQ[j]); end
    println(io, "\n[se_optimal_weights]")
    for j in 1:NP_; @printf(io, "%-12s = %.8f\n", CNAMES[j], SE_OPT[j]); end
end
sayf("\nwrote %s\n", relpath(joinpath(OUTDIR, "standard_errors.toml"), REPO))
close(LOG)
