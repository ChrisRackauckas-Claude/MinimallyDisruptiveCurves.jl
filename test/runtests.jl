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

    if GROUP == "All" || GROUP == "ExplicitImports"
        @testset "Explicit Imports" begin
            include("explicit_imports.jl")
        end
    end

    if GROUP == "All" || GROUP == "JET"
        # JET tests are optional - skip if JET can't be loaded (e.g., on pre-release Julia)
        jet_available = try
            @eval using JET
            true
        catch
            false
        end
        if jet_available
            @testset "JET Static Analysis" begin
                include("jet_tests.jl")
            end
        else
            @info "Skipping JET tests - JET.jl not available on this Julia version"
        end
    end
end
