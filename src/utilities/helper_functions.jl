"""
    soft_heaviside(t::Real, nastiness::Real, step_time::Real) -> Real

Makes a soft analogue of the heaviside step function. Useful for inputs to
differential equations, as it's easier on the numerics.
"""
function soft_heaviside(t::Real, nastiness::Real, step_time::Real)
    return 1 / (1 + exp(nastiness * (step_time - t)))
end

"""
    soft_heaviside(nastiness::Real, step_time::Real) -> Function

Returns a closure that computes `soft_heaviside(t, nastiness, step_time)`.
"""
soft_heaviside(nastiness::Real, step_time::Real) = t -> soft_heaviside(t, nastiness, step_time)

get_ids_names(opArray) = repr.(opArray)

"""
    l2_hessian(nom_sol)

gets hessian according to L2 loss under the assumption that loss(θ₀) = 0. nom_sol is the solution of the nominal ODEProblem.
The Hessian then only requires first derivatives: it is sum_ij dyi/dθ * dyj/dtheta
"""
function l2_hessian(nom_sol)
    prob = nom_sol.prob
    function pToL2(p)
        pprob = remake(prob, p = p)
        psol = solve(pprob, nom_sol.alg, saveat = nom_sol.t) |> Array
        psol = reshape(psol, 1, :)
        return psol
    end

    gr = ForwardDiff.jacobian(pToL2, prob.p)
    u, d, v = svd(gr)
    return v * diagm(d .^ 2) * v'
    # = gr'*gr but a bit more accurate
end
