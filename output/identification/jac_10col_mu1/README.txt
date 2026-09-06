Residual Jacobian, produced by code/smm/jacobian.jl.

J_h<step>.csv   rows = the 10 targeted moments, in SMM_MOMENTS order.
                cols = phi_2, phi_3, lambda_2, R_0, sigma_1_0, sigma_1_1, sigma_2_0, sigma_2_1, sigma_4_0, mu_1.
                Entry (i,j) is the change in residual i -- (model - data)/scale --
                from moving parameter j across its ENTIRE box, estimated by a central
                difference of width 2*step*boxwidth in search coordinates.

jacobian.toml   singular values, condition numbers, weakest directions, pairwise
                cosines, and the full metadata: evaluation point, boxes, links,
                moment scales, grids, seed and steps.

Read with the qualifications in the script header: this is ONE local Jacobian at ONE
point under ONE scaling. It is not global identification and it is not inference.
