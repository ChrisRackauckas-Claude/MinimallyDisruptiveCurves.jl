using ExplicitImports
using MinimallyDisruptiveCurves
using Test

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(MinimallyDisruptiveCurves) === nothing
    @test check_no_stale_explicit_imports(MinimallyDisruptiveCurves) === nothing
end
