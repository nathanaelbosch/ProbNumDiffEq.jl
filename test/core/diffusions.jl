using ProbNumDiffEq
import ProbNumDiffEq as PNDE
using Test
using LinearAlgebra
using FillArrays

d, q = 2, 3
T = Float64

@testset "$diffusionmodel" for diffusionmodel in (
    DynamicDiffusion(),
    DynamicMVDiffusion(),
    FixedDiffusion(),
    FixedDiffusion(calibrate=false),
    FixedMVDiffusion(),
    FixedMVDiffusion(; initial_diffusion=rand(d)),
    FixedMVDiffusion(; initial_diffusion=Diagonal(rand(d))),
    FixedMVDiffusion(; initial_diffusion=Diagonal(rand(d)), calibrate=false),
)

    # Test the initial diffusion
    diffusion = PNDE.initial_diffusion(diffusionmodel, d, q, T)
    @assert size(diffusion) == (d, d)
    @assert diffusion isa Diagonal
    if !(diffusionmodel isa FixedMVDiffusion || diffusionmodel isa DynamicMVDiffusion)
        @assert diffusion isa Diagonal{T,<:Fill}
    end

    # Test applying the diffusion
    _, Q = PNDE.discretize(PNDE.IWP{T}(d, q), 0.1)
    Qmat = PSDMatrix(Matrix(Q.R))
    _diffusion = rand() * diffusion
    @testset "$FAC" for FAC in (
        PNDE.DenseCovariance{T}(d, q),
        PNDE.BlockDiagonalCovariance{T}(d, q),
        PNDE.IsometricKroneckerCovariance{T}(d, q),
    )
        if diffusion isa Diagonal{T,<:Vector} && FAC isa PNDE.IsometricKroneckerCovariance
            continue
        end

        _Q = PNDE.to_factorized_matrix(FAC, Q)
        Qdiff = @test_nowarn PNDE.apply_diffusion(_Q, _diffusion)
        Qmatdiff = @test_nowarn PNDE.apply_diffusion(Qmat, _diffusion)
        @test Qdiff == Qmatdiff

        Qdiff = @test_nowarn PNDE.apply_diffusion!(copy(_Q), _diffusion)
        @test Qdiff == Qmatdiff

        Qdiff = @test_nowarn PNDE.apply_diffusion!(copy(_Q), _Q, _diffusion)
        @test Qdiff == Qmatdiff
    end

    @testset "Calibration" begin
        # MLE
        # At their core, they all just compute z' * inv(S) * z
        # and then do something with the result
    end
end
