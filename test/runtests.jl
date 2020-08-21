using Test


@testset "Priors" begin include("priors.jl") end
@testset "Correctness" begin include("correctness.jl") end
@testset "Solution" begin include("solution.jl") end
@testset "Sigmas" begin include("sigmas.jl") end
@testset "State Initialization" begin include("state_init.jl") end
@testset "Preconditioning" begin include("preconditioning.jl") end
@testset "Step Control" begin include("step_controller.jl") end
