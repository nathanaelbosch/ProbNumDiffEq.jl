using Test


@testset "ProbNumODE" begin
    @testset "Priors" begin include("priors.jl") end
    @testset "Correctness" begin include("correctness.jl") end
    @testset "Solution" begin include("solution.jl") end
    @testset "Sigmas" begin include("sigmas.jl") end
    @testset "Error Estimates" begin include("error_estimates.jl") end

end
