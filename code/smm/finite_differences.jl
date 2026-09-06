"""
    bounded_stencil(z, lo, hi, step)

Derivative at `z` in search coordinates. Use a central difference in the interior,
otherwise a second-order forward/backward difference at the same evaluation point.
`step` is a fraction of box width. Never clip a point while keeping the old 2h divisor.
Returns the actual points, derivative weights and scheme for reproducible metadata.
"""
function bounded_stencil(z::Float64, lo::Float64, hi::Float64, step::Float64)
    all(isfinite, (z, lo, hi, step)) || throw(ArgumentError("non-finite difference input"))
    lo < hi && lo <= z <= hi || throw(ArgumentError("difference centre must be inside a nonempty box"))
    0 < step <= 0.5 || throw(ArgumentError("difference step must be in (0, 0.5]"))
    nominal_h = step * (hi - lo)
    if z - nominal_h >= lo && z + nominal_h <= hi
        points = [z - nominal_h, z + nominal_h]
        points[2] > points[1] || throw(ArgumentError("difference step is below floating-point resolution"))
        weights = [-1.0, 1.0] ./ (points[2] - points[1])
        return (points=points, weights=weights, scheme="central", nominal_h=nominal_h)
    end
    forward = z - nominal_h < lo
    h = min(nominal_h, (forward ? hi - z : z - lo) / 2)
    points = forward ? [z, z+h, min(hi,z+2h)] : [z, z-h, max(lo,z-2h)]
    d1, d2 = points[2]-z, points[3]-z
    d1 != 0 && d2 != 0 && d1 != d2 || throw(ArgumentError("difference step is below floating-point resolution"))
    # Derivative of the quadratic through the actual (possibly rounded) points.
    weights = [-(d1+d2)/(d1*d2), d2/(d1*(d2-d1)), -d1/(d2*(d2-d1))]
    return (points=points, weights=weights, scheme=forward ? "forward" : "backward", nominal_h=nominal_h)
end
