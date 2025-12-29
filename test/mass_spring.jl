using OrdinaryDiffEq, ForwardDiff, DiffEqCallbacks, LinearAlgebra, Test

"""
Core MDC functionality tests using simple cost functions.
Note: ModelingToolkit ODESystem integration tests are skipped for MTKv9+ compatibility.
"""

# Cost function with non-trivial Hessian for better MDC dynamics
p0 = [1.0, 2.0, 3.0]
function simple_cost(p)
    # Rosenbrock-like function modified to have minimum at p0
    a = 1.0
    b = 100.0
    sum((a .- p) .^ 2) + sum(b .* (p[2:end] .- p[1:(end - 1)] .^ 2) .^ 2)
end

function simple_cost_grad!(p, g)
    n = length(p)
    a = 1.0
    b = 100.0
    # Simple finite-difference for gradient to avoid allocation issues
    ε = 1e-8
    for i in 1:n
        p_plus = copy(p)
        p_plus[i] += ε
        g[i] = (simple_cost(p_plus) - simple_cost(p)) / ε
    end
    return simple_cost(p)
end

cost = DiffCost(simple_cost, simple_cost_grad!)

"""
Test DiffCost works correctly
"""
grad_holder = similar(p0)
# Test cost function returns a scalar
c0 = cost(p0)
@test c0 isa Number

# Test gradient
cost(p0, grad_holder)
@test length(grad_holder) == length(p0)

"""
Test transform_cost
"""
tr = logabs_transform(p0)
tr_cost, newp0 = transform_cost(cost, p0, tr)

# Test that transformed cost exists and is callable
@test tr_cost(newp0) isa Number

"""
Test sum_losses
"""
ll = sum_losses([cost, cost], p0)
g2 = similar(p0)
ll_val = ll(p0, grad_holder)
cost_val = cost(p0, g2)
@test ll_val ≈ 2cost_val

"""
Test MDCProblem creation and evolution
"""
H0 = ForwardDiff.hessian(cost, p0)
mom = 100.0  # Higher momentum for stability
span = (0.0, 1.0)  # Positive span for simpler testing
dp0 = (eigen(H0)).vectors[:, 1]
dp0 = dp0 / norm(dp0)  # Normalize

eprob = MDCProblem(cost, p0, dp0, mom, span)

cb = [
    Verbose([CurveDistance(0.1:0.1:1.0), HamiltonianResidual(0.3:0.4:1.0)]),
]

@time mdc = evolve(eprob, Tsit5; mdc_callback = cb)

"""
Test MDC solution properties
"""
# Use trajectory to get the parameter values at start and end
traj = trajectory(mdc)
@test size(traj, 1) == 3  # 3 parameters
@test size(traj, 2) > 0   # At least one point

# Check that the curve evolved (end differs from start)
if size(traj, 2) > 1
    @test traj[:, 1] != traj[:, end]
end

"""
Test with different callback configuration
"""
cb2 = [
    Verbose([CurveDistance(0.1:0.3:1.0)])
]
@time mdc2 = evolve(eprob, Tsit5; mdc_callback = cb2)

traj2 = trajectory(mdc2)
@test size(traj2, 1) == 3

"""
Test trajectory and distances utilities
"""
traj_final = trajectory(mdc)
@test size(traj_final, 1) == 3  # 3 parameters

dists = distances(mdc)
@test length(dists) > 0

println("All core MDC tests passed!")
