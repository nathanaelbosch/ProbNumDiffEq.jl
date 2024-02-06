# Parameter Inference with ProbNumDiffEq.jl



Let's assume we have an initial value problem (IVP)
```math
\begin{aligned}
\dot{y} &= f_\theta(y, t), \qquad y(t_0) = y_0,
\end{aligned}
```
which we observe through a set ``\mathcal{D} = \{u(t_n)\}_{n=1}^N`` of noisy data points
```math
\begin{aligned}
u(t_n) = H y(t_n) + v_n, \qquad v_n \sim \mathcal{N}(0, R).
\end{aligned}
```
The question of interest is: How can we compute the marginal likelihood ``p(\mathcal{D} \mid \theta)``?
Short answer: We can't. It's intractable, because computing the true IVP solution exactly ``y(t)`` is intractable.
What we can do however is compute an approximate marginal likelihood.
This is what `ProbNumDiffEq.DataLikelihoods` provides.

## The specific problem, in code
Let's assume that the true underlying dynamics are given by a FitzHugh-Nagumo model

```@example parameterinference
using ProbNumDiffEq, LinearAlgebra, OrdinaryDiffEq, Plots
function f(du, u, p, t)
    a, b, c = p
    du[1] = c*(u[1] - u[1]^3/3 + u[2])
    du[2] = -(1/c)*(u[1] -  a - b*u[2])
end
u0 = [-1.0, 1.0]
tspan = (0.0, 20.0)
p = (0.2, 0.2, 3.0)
true_prob = ODEProblem(f, u0, tspan, p)
```
from which we generate some artificial noisy data
```@example parameterinference
true_sol = solve(true_prob, Vern9(), abstol=1e-10, reltol=1e-10)

times = 1:0.5:20
σ = 1e-1
odedata = [true_sol(t) .+ σ * randn(length(u0)) for t in times]

plot(true_sol, color=:black, linestyle=:dash, label=["True Solution" ""])
scatter!(times, stack(odedata)', markersize=2, markerstrokewidth=0.1, color=1, label=["Noisy Data" ""])
```
Our goal is then to recover the true parameter `p` (and thus also the true trajectory plotted above) the noisy data.

## Computing the negative log-likelihood
To do parameter inference - be it maximum-likelihod, maximum a posteriori, or full Bayesian inference with MCMC - we need to evaluate the likelihood of given a parameter estimate ``\theta_\text{est}``, which corresponds to the probability of the data under the trajectory returned by the ODE solver
```@example parameterinference
θ_est = (0.1, 0.1, 2.0)
prob = remake(true_prob, p=θ_est)
plot(true_sol, color=:black, linestyle=:dash, label=["True Solution" ""])
scatter!(times, stack(odedata)', markersize=2, markerstrokewidth=0.1, color=1, label=["Noisy Data" ""])
plot!(solve(prob, EK1(), adaptive=false, dt=1e-1), color=2, label=["Numerical solution for θ_est"])
```
This quantity can be computed in multiple ways; see 
[Data Likelihoods](@ref).
Here we use 
[`ProbNumDiffEq.DataLikelihoods.fenrir_data_loglik`](@ref):
```@example parameterinference
using ProbNumDiffEq.DataLikelihoods

data = (t=times, u=odedata)
nll = -fenrir_data_loglik(prob, EK1(smooth=true); data, observation_noise_cov=σ^2, adaptive=false, dt=1e-1)
```
This is the negative marginal log-likelihood of the parameter `θ_est`.
You can use it as any other NLL: Optimize it to compute maximum-likelihood estimates or MAPs, or plug it into MCMC to sample from the posterior.
In our paper [tronarp22fenrir](@cite) we compute MLEs by pairing Fenrir with [Optimization.jl](http://optimization.sciml.ai/stable/) and [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/).
Let's quickly explore how to do this next.


## Maximum-likelihood parameter inference

To compute a maximum-likelihood estimate (MLE), we just need to maximize ``\theta \to p(\mathcal{D} \mid \theta)`` - that is, minimize the `nll` from above.
We use [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) for this.
First, define a loss function and create an `OptimizationProblem`
```@example parameterinference
using Optimization, OptimizationOptimJL

function loss(x, _)
    ode_params = x[begin:end-1]
    prob = remake(true_prob, p=ode_params)
    κ² = exp(x[end]) # we also optimize the diffusion parameter of the EK1
    return -fenrir_data_loglik(
        prob, EK1(smooth=true, diffusionmodel=FixedDiffusion(κ², false));
        data, observation_noise_cov=σ^2, adaptive=false, dt=1e-1
    )
end

fun = OptimizationFunction(loss, Optimization.AutoForwardDiff())
optprob = OptimizationProblem(
    fun, [θ_est..., 1e0];
    lb=[0.0, 0.0, 0.0, -10], ub=[1.0, 1.0, 5.0, 20] # lower and upper bounds
)
```

Then, just `solve` it! Here we use LBFGS:
```@example parameterinference
optsol = solve(optprob, LBFGS())
p_mle = optsol.u[1:3]
p_mle # hide
```

Success! The computed MLE is quite close to the true parameter which we used to generate the data.
As a final step, let's plot the true solution, the data, and the result of the MLE:

```@example parameterinference
plot(true_sol, color=:black, linestyle=:dash, label=["True Solution" ""])
scatter!(times, stack(odedata)', markersize=2, markerstrokewidth=0.1, color=1, label=["Noisy Data" ""])
mle_sol = solve(remake(true_prob, p=p_mle), EK1())
plot!(mle_sol, color=3, label=["MLE-parameter Solution" ""])
```

Looks good!


## API Documentation

For more details, see the API documentation of `ProbNumDiffEq.DataLikelihoods` at [Data Likelihoods](@ref).


### References

```@bibliography
Pages = []
Canonical = false

tronarp22fenrir
wu23dalton
```
