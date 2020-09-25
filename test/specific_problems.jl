# Everytime I encounter something that raises some error and I fix it, I should add that
# specific problem to this list to make sure, that this specific run then works without
# bugs.
using ProbNumODE
using Test
using LinearAlgebra


@testset "EM-sigma with adaptive steps" begin
    prob = ProbNumODE.remake_prob_with_jac(fitzhugh_nagumo_iip())
    @test solve(prob, EKF0(), q=4, sigma=:EM, abstol=1e-8, reltol=1e-8) isa ProbNumODE.ProbODESolution
    @test solve(prob, EKF1(), q=4, sigma=:EM, abstol=1e-8, reltol=1e-8) isa ProbNumODE.ProbODESolution
end


@testset "Smoothing with small constant steps" begin
    prob = ProbNumODE.remake_prob_with_jac(fitzhugh_nagumo_iip())
    @test solve(prob, EKF0(), q=4, steprule=:constant, dt=1e-3, smooth=true) isa ProbNumODE.ProbODESolution
    @test solve(prob, EKF1(), q=4, steprule=:constant, dt=1e-3, smooth=true) isa ProbNumODE.ProbODESolution
end


@testset "Stiff Vanderpol" begin
    prob = ProbNumODE.remake_prob_with_jac(van_der_pol(p=[1e6]))
    prob = remake(prob, u0=big.(prob.u0))
    @test solve(prob, EKF1(), q=5) isa ProbNumODE.ProbODESolution
end
