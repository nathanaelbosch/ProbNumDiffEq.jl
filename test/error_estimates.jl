


@testset "Correctness with Schober Errors" begin
    correctness_atol = 1e-8
    alg_abstol, alg_reltol = 1e-10, 1e-10

    prob = ProbNumODE.remake_prob_with_jac(prob_ode_lotkavoltera)
    setup = (
        steprule=:standard,
        abstol=alg_abstol,
        reltol=alg_reltol,
        local_errors=:schober,
        sigmarule=:schober,
        setprule=:standard,
    )
    for q in 1:2
        test_prob_solution_correctness(
            prob, correctness_atol, EKF0();
            q=q, setup...)
    end
end


@testset "Correctness with Prediction Errors" begin
    correctness_atol = 1e-5
    alg_abstol, alg_reltol = 1e-10, 1e-10

    prob = ProbNumODE.remake_prob_with_jac(prob_ode_lotkavoltera)
    setup = (
        steprule=:standard,
        abstol=alg_abstol,
        reltol=alg_reltol,
        local_errors=:prediction,
        sigmarule=:schober,
        setprule=:standard,
    )
    for q in 1:2
        test_prob_solution_correctness(
            prob, correctness_atol, EKF0();
            q=q, setup...)
    end
end



@testset "Correctness with Filtering Errors" begin
    correctness_atol = 1e-5
    alg_abstol, alg_reltol = 1e-10, 1e-10

    prob = ProbNumODE.remake_prob_with_jac(prob_ode_lotkavoltera)
    setup = (
        steprule=:standard,
        abstol=alg_abstol,
        reltol=alg_reltol,
        local_errors=:filtering,
        sigmarule=:schober,
        setprule=:standard,
    )
    for q in 1:2
        test_prob_solution_correctness(
            prob, correctness_atol, EKF0();
            q=q, setup...)
    end
end
