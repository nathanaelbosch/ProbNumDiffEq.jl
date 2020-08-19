



@testset "Sigmas" begin
    prob = fitzhugh_nagumo()
    # Multiple different methods
    sol = solve(prob, EKF0(), steprule=:standard,
                sigmarule=:schober,
                abstol=1e-3, reltol=1e-3, q=2)
    @test_broken begin
        sol = solve(prob, EKF0(), steprule=:constant,
                    sigmarule=:fixedMLE,
                    abstol=1e-1, reltol=1e-1, q=2)
    end
    # sol = solve(prob, EKF0(), steprule=:constant, sigmarule=:fixedMAP,
    #             abstol=1e-1, reltol=1e-1, q=2)
    # sol = solve(prob, EKF0(), steprule=:constant, sigmarule=ProbNumODE.WeightedMLESigma(),
    #             abstol=1e-1, reltol=1e-1, q=2)
end
