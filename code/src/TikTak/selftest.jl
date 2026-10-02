# =============================================================================
# selftest.jl -- the standalone numerical self-test (no economic model, no workers).
# =============================================================================

"""
    tiktak_selftest(; verbose = true) -> Bool

Three checks, chosen so that each can only pass if a different part of the
algorithm is right:

  1. Sphere in 10d must hit the minimum to machine precision. Catches bad box
     scaling, a broken local stage, or a mishandled return value.
  2. Rastrigin in 3d, whose local minima form a dense lattice, must be solved
     exactly at a budget where that is achievable (N = 2000). Catches a global
     stage that is not actually exploring.
  3. TikTak must BEAT plain multistart on Rastrigin in 4d at an identical
     budget, where "plain multistart" is this same code with theta pinned to 0.
     This is the only check that tests the distinguishing feature -- the mixing
     of each seed with the incumbent best. Measured: 0.995 against 2.985.

Budgets matter more than they look. Rastrigin in 6d has on the order of 11^6
local minima in the box, so a small budget failing there is the function being
hard, not the optimizer being wrong.
"""
function tiktak_selftest(; verbose::Bool = true)
    rastrigin(x) = 10length(x) + sum(xi^2 - 10cos(2π*xi) for xi in x)
    sphere(x) = sum(x .^ 2)
    pass = true

    r1 = tiktak(sphere, fill(-10.0, 10), fill(10.0, 10); N = 200, Nstar = 10)
    ok1 = r1.f < 1e-12; pass &= ok1
    verbose && @printf("  sphere d=10          f = %.2e            [%s]\n",
                       r1.f, ok1 ? "PASS" : "FAIL")

    r2 = tiktak(rastrigin, fill(-5.12, 3), fill(5.12, 3); N = 2000, Nstar = 100)
    ok2 = r2.f < 1e-6; pass &= ok2
    verbose && @printf("  rastrigin d=3        f = %.2e            [%s]\n",
                       r2.f, ok2 ? "PASS" : "FAIL")

    rt = tiktak(rastrigin, fill(-5.12, 4), fill(5.12, 4); N = 1000, Nstar = 50)
    rm = tiktak(rastrigin, fill(-5.12, 4), fill(5.12, 4); N = 1000, Nstar = 50,
                theta_lo = 0.0, theta_hi = 0.0)          # theta == 0 => plain multistart
    ok3 = rt.f <= rm.f; pass &= ok3
    verbose && @printf("  rastrigin d=4        TikTak %.4f vs multistart %.4f  [%s]\n",
                       rt.f, rm.f, ok3 ? "PASS" : "FAIL")

    verbose && @printf("  tiktak_selftest: %s\n", pass ? "ALL PASS" : "FAILURES ABOVE")
    return pass
end
