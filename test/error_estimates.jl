


@testset "Error Estimations" begin
    prob = fitzhugh_nagumo()
    # Multiple different methods
    sol = solve(prob, EKF0(), steprule=:standard, local_errors=:schober,
                abstol=1e-3, reltol=1e-3, q=2)
    # sol = solve(prob, EKF0(), steprule=:standard, local_errors=:prediction,
    #             abstol=1e-3, reltol=1e-3, q=2)
    # sol = solve(prob, EKF0(), steprule=:standard, local_errors=:filtering,
    #             abstol=1e-3, reltol=1e-3, q=2)
end
