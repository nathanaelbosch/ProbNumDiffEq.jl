# Solvers

ProbNumDiffEq.jl provides two solvers: the [`EK1`](@ref) and the [`EK0`](@ref). Both based on extended Kalman filtering and smoothing, but the latter relies on evaluating the Jacobian of the vector field.

**Which solver should I use?**
- Use the [`EK1`](@ref) to get the best uncertainty quantification and to solve stiff problems.
- Use the [`EK0`](@ref) to get the fastest runtimes and to solve high-dimensional problems.

All solvers are compatible with DAEs in mass-matrix ODE form.
They also specialize on second-order ODEs: If the problem is of type [`SecondOrderODEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/dynamical_types/#SciMLBase.SecondOrderODEProblem), it solves the second-order problem directly; this is more efficient than solving the transformed first-order problem and provides more meaningful posteriors
[[1]](@ref solversrefs).

## [Smoothing and automatic differentiation](@id smoothing_ad)

Smoothing is enabled by default (`smooth=true`) and is required for dense output.
Two smoother implementations are available, selected with the solvers' `smoother`
keyword argument:

- `smoother=:mbf` (**default**): the square-root Modified Bryson--Frazier smoother
  (Gibbs, 2011). It never inverts the full predicted state covariance and is therefore
  robust to the numerical instability that the classic smoother can run into at very
  high solver orders.
- `smoother=:rts`: the classic Rauch--Tung--Striebel smoother, which marginalizes the
  backward transition kernels computed during the forward pass.

Both produce the same smoothed posterior up to numerical accuracy.

!!! warning "Forward-mode AD through smoothed covariances"
    The default smoother is `:mbf`; it used to be the RTS smoother. With `:mbf`,
    forward-mode automatic differentiation (e.g. with ForwardDiff.jl) of quantities that
    depend on the smoothed **covariances** can return `NaN` gradients. The smoothed
    *means* -- and therefore `sol.u` and everything computed from it -- are unaffected,
    and so are the values of the covariances themselves; only their partials are.
    This is an intrinsic conditioning property of the hyperbolic QR factorization that
    the MBF smoother uses to recover the smoothed covariances, not something that can be
    fixed in the implementation.

    So: if you differentiate a loss that involves smoothed covariances, such as anything
    built from `sol.pu[i].Σ` or `sol.x_smooth[i].Σ`, request the RTS smoother explicitly:
    ```julia
    solve(prob, EK1(smoother=:rts))
    ```
    If your loss only involves the solution means, which is the common case, both
    smoothers work and there is nothing to change.

## API
```@docs
EK1
EK0
```

### Probabilistic Exponential Integrators
```@docs
ExpEK
RosenbrockExpEK
```

## [References](@id solversrefs)


```@bibliography
Pages = []
Canonical = false

tronarp18probsol
krämer21highdim
bosch23expint
```
