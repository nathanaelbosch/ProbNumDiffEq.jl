using ProbNumODE
using Test
using LinearAlgebra

using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_lotkavoltera


h = rand()
σ = rand()


@testset "Test IBM (d=2,q=2)" begin
    d, q = 2, 2

    A!, Q! = ProbNumODE.ibm(d, q; precond_dt=1.0)
    Ah = diagm(0 => ones(d*(q+1)))
    Qh = zeros(d*(q+1), d*(q+1))
    A!(Ah, h)
    Q!(Qh, h, σ^2)


    AH_22_IBM = [1 0 h 0 h^2/2 0;
                 0 1 0 h 0 h^2/2;
                 0 0 1 0 h 0;
                 0 0 0 1 0 h;
                 0 0 0 0 1 0;
                 0 0 0 0 0 1]
    @test AH_22_IBM ≈ Ah

    QH_22_IBM = σ^2 .* [h^5/20  0       h^4/8  0      h^3/6  0;
                        0       h^5/20  0      h^4/8  0      h^3/6;
                        h^4/8   0       h^3/3  0      h^2/2  0;
                        0       h^4/8   0      h^3/3  0      h^2/2;
                        h^3/6   0       h^2/2  0      h      0;
                        0       h^3/6   0      h^2/2  0      h]
    @test QH_22_IBM ≈ Qh
end



@testset "Test IBM with preconditioning (d=1,q=2)" begin
    d, q = 1, 2

    A!, Q! = ProbNumODE.ibm(d, q; precond_dt=h)
    Ah = diagm(0 => ones(d*(q+1)))
    Qh = zeros(d*(q+1), d*(q+1))
    A!(Ah, h)
    Q!(Qh, h, σ^2)

    AH_21_PRE = [1  1  0.5
                 0  1  1
                 0  0  1]

    QH_21_PRE = σ^2 * h * [1/20 1/8 1/6
                           1/8  1/3 1/2
                           1/6  1/2 1]

    @test AH_21_PRE ≈ Ah
    @test QH_21_PRE ≈ Qh
end



@testset "Verify correct prior dim" begin
    prob = prob_ode_lotkavoltera
    d = length(prob.u0)
    for q in 1:5
        integ = init(prob, EKF0(), q=q, initialize_derivatives=false, smooth=false)
        @test length(integ.cache.x.μ) == d*(q+1)
        sol = solve!(integ)
        @test length(integ.cache.x.μ) == d*(q+1)
        @test length(sol.x[end].μ) == d*(q+1)
    end
end
