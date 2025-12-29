using MinimallyDisruptiveCurves
using LinearAlgebra
using Test

# AllocCheck is optional - only load if available
const HAVE_ALLOCCHECK = try
    @eval using AllocCheck
    true
catch
    false
end

"""
Allocation tests for performance-critical paths in MinimallyDisruptiveCurves.jl

These tests verify that the core dynamics function and related hot paths
do not allocate memory during execution, ensuring optimal performance.
"""

# Create a simple non-allocating cost function for testing
function test_cost_noalloc(p)
    s = 0.0
    @inbounds for i in eachindex(p)
        s += (p[i] - Float64(i))^2
    end
    return s
end

function test_cost_grad_noalloc!(p, g)
    s = 0.0
    @inbounds for i in eachindex(p)
        d = p[i] - Float64(i)
        g[i] = 2 * d
        s += d^2
    end
    return s
end

@testset "Allocation Tests" begin
    # Set up the test problem
    diff_cost = DiffCost(test_cost_noalloc, test_cost_grad_noalloc!)
    p0 = [1.0, 2.0, 3.0]
    dp0 = [1.0, 0.0, 0.0]
    momentum = 1.0
    tspan = (-2.0, 1.0)

    eprob = MDCProblem(diff_cost, p0, dp0, momentum, tspan)
    probs = eprob()
    prob = probs[1]
    f = prob.f
    u0 = prob.u0
    du = similar(u0)
    t = 0.0
    p = nothing

    # Warm up the function
    f(du, u0, p, t)
    f(du, u0, p, t)

    @testset "Dynamics function allocations" begin
        # Test that the dynamics function does not allocate
        # Using @allocated for runtime check
        allocs = @allocated f(du, u0, p, t)

        # We expect 0 allocations when using a non-allocating cost function
        @test allocs == 0
    end

    @testset "Dynamics function correctness" begin
        # Verify the dynamics produces valid (finite) output
        f(du, u0, p, t)
        @test all(isfinite, du)
        @test !all(iszero, du)  # Should produce non-zero derivatives
    end

    @testset "DiffCost allocations" begin
        # Test DiffCost evaluation does not allocate
        grad = similar(p0)

        # Warm up
        diff_cost(p0)
        diff_cost(p0, grad)

        # Test without gradient
        allocs_no_grad = @allocated diff_cost(p0)
        @test allocs_no_grad == 0

        # Test with gradient
        allocs_with_grad = @allocated diff_cost(p0, grad)
        @test allocs_with_grad == 0
    end

    @testset "Initial conditions" begin
        # Test that initial conditions computation is correct
        ic = MinimallyDisruptiveCurves.initial_conditions(eprob)
        @test length(ic) == 2 * length(p0)
        @test ic[1:length(p0)] == p0
        @test all(isfinite, ic)
    end
end
