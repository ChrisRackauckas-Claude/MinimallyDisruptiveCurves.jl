using MinimallyDisruptiveCurves
using Test

const GROUP = get(ENV, "GROUP", "All")

@testset "MinimallyDisruptiveCurves.jl" begin
    if GROUP == "All" || GROUP == "Core"
        @testset "mass_spring" begin
            include("mass_spring.jl")
        end
    end

    if GROUP == "All" || GROUP == "nopre" || GROUP == "Alloc"
        @testset "allocation_tests" begin
            include("alloc_tests.jl")
        end
    end
end
