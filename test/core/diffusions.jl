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
    FixedMVDiffusion(; initial_diffusion=rand(2)),
    FixedMVDiffusion(; initial_diffusion=Diagonal(rand(2))),
    FixedMVDiffusion(; initial_diffusion=Diagonal(rand(2)), calibrate=false),
)

    # Test the initial diffusion
    diff = PNDE.initial_diffusion(diffusionmodel, d, q, T)
    @assert size(diff) == (d, d)
    @assert diff isa Diagonal
    if !(diffusionmodel isa FixedMVDiffusion || diffusionmodel isa DynamicMVDiffusion)
        @assert diff isa Diagonal{T,<:Fill}
    end

    # Test applying the diffusion
    _, Q = PNDE.discretize(PNDE.IWP{T}(d, q), 0.1)
    Qmat = PSDMatrix(Matrix(Q.R))
    _diff = rand() * diff
    @testset "$FAC" for FAC in (
        PNDE.DenseCovariance{T}(d, q),
        PNDE.BlockDiagonalCovariance{T}(d, q),
        PNDE.IsometricKroneckerCovariance{T}(d, q),
    )
        if diff isa Diagonal{T,<:Vector} && FAC isa PNDE.IsometricKroneckerCovariance
            continue
        end

        _Q = PNDE.to_factorized_matrix(FAC, Q)
        Qdiff = @test_nowarn PNDE.apply_diffusion(_Q, _diff)
        Qmatdiff = @test_nowarn PNDE.apply_diffusion(Qmat, _diff)
        @test Qdiff == Qmatdiff

        if !(diff isa Diagonal{T,<:Vector} && FAC isa PNDE.DenseCovariance)
            Qdiff = @test_nowarn PNDE.apply_diffusion!(copy(_Q), _diff)
            @test Qdiff == Qmatdiff

            Qdiff = @test_nowarn PNDE.apply_diffusion!(copy(_Q), _Q, _diff)
            @test Qdiff == Qmatdiff
        end
    end

    @testset "Calibration" begin
        # MLE
        # At their core, they all just compute z' * inv(S) * z
        # and then do something with the result
    end
end
