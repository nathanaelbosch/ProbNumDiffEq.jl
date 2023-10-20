########################################################################################
# Algorithm
########################################################################################
abstract type AbstractEK <: OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm end

"""
    EK0(; order=3,
          smooth=true,
          prior=IWP(order),
          diffusionmodel=DynamicDiffusion(),
          initialization=TaylorModeInit())

**Gaussian ODE filter with zeroth-order vector field linearization.**

# Arguments
- `order::Integer`: Order of the integrated Wiener process (IWP) prior.
- `smooth::Bool`: Turn smoothing on/off; smoothing is required for dense output.
- `prior::AbstractODEFilterPrior`: Prior to be used by the ODE filter.
   By default, uses a 3-times integrated Wiener process prior `IWP(3)`.
   See also: [Priors](@ref).
- `diffusionmodel::ProbNumDiffEq.AbstractDiffusion`: See [Diffusion models and calibration](@ref).
- `initialization::ProbNumDiffEq.InitializationScheme`: See [Initialization](@ref).

# Examples
```julia-repl
julia> solve(prob, EK0())
```

# [References](@ref references)
"""
struct EK0{PT,DT,IT} <: AbstractEK
    prior::PT
    diffusionmodel::DT
    smooth::Bool
    initialization::IT
end
EK0(;
    order=3,
    prior=IWP(order),
    diffusionmodel=DynamicDiffusion(),
    smooth=true,
    initialization=TaylorModeInit(),
) = EK0(prior, diffusionmodel, smooth, initialization)

_unwrap_val(::Val{B}) where {B} = B
_unwrap_val(B) = B

"""
    EK1(; order=3,
          smooth=true,
          prior=IWP(order),
          diffusionmodel=DynamicDiffusion(),
          initialization=TaylorModeInit(),
          kwargs...)

**Gaussian ODE filter with first-order vector field linearization.**

# Arguments
- `order::Integer`: Order of the integrated Wiener process (IWP) prior.
- `smooth::Bool`: Turn smoothing on/off; smoothing is required for dense output.
- `prior::AbstractODEFilterPrior`: Prior to be used by the ODE filter.
   By default, uses a 3-times integrated Wiener process prior `IWP(3)`.
   See also: [Priors](@ref).
- `diffusionmodel::ProbNumDiffEq.AbstractDiffusion`: See [Diffusion models and calibration](@ref).
- `initialization::ProbNumDiffEq.InitializationScheme`: See [Initialization](@ref).

Some additional `kwargs` relating to implicit solvers are supported;
check out DifferentialEquations.jl's [Extra Options](https://diffeq.sciml.ai/stable/solvers/ode_solve/#Extra-Options) page.
Right now, we support `autodiff`, `chunk_size`, and `diff_type`.
In particular, `autodiff=false` can come in handy to use finite differences instead of
ForwardDiff.jl to compute Jacobians.

# Examples
```julia-repl
julia> solve(prob, EK1())
```

# [References](@ref references)
"""
struct EK1{CS,AD,DiffType,ST,CJ,PT,DT,IT} <: AbstractEK
    prior::PT
    diffusionmodel::DT
    smooth::Bool
    initialization::IT
end
EK1(;
    order=3,
    prior::PT=IWP(order),
    diffusionmodel::DT=DynamicDiffusion(),
    smooth=true,
    initialization::IT=TaylorModeInit(),
    chunk_size=Val{0}(),
    autodiff=Val{true}(),
    diff_type=Val{:forward},
    standardtag=Val{true}(),
    concrete_jac=nothing,
) where {PT,DT,IT} =
    EK1{
        _unwrap_val(chunk_size),
        _unwrap_val(autodiff),
        diff_type,
        _unwrap_val(standardtag),
        _unwrap_val(concrete_jac),
        PT,
        DT,
        IT,
    }(
        prior,
        diffusionmodel,
        smooth,
        initialization,
    )

"""
    ExpEK(; L, order=3, kwargs...)

Probabilistic exponential integrator

Probabilistic exponential integrators are a class of integrators for semi-linear stiff ODEs
that provide improved stability by essentially solving the linear part of the ODE exactly.
In probabilistic numerics, this amounts to including the linear part into the prior model
of the solver.

`ExpEK` is therefore just a short-hand for [`EK0`](@ref) with [`IOUP`](@ref) prior:
```julia
ExpEK(; order=3, L, kwargs...) = EK0(; prior=IOUP(order, L), kwargs...)
```

See also [`RosenbrockExpEK`](@ref), [`EK0`](@ref), [`EK1`](@ref).

# Arguments
See [`EK0`](@ref) for available keyword arguments.

# Examples
```julia-repl
julia> prob = ODEProblem((du, u, p, t) -> (@. du = - u + sin(u)), [1.0], (0.0, 10.0))
julia> solve(prob, ExpEK(L=-1))
```


# Reference
* [bosch23expint](@cite) Bosch et al, "Probabilistic Exponential Integrators", arXiv (2021)
"""
ExpEK(; L, order=3, kwargs...) = EK0(; prior=IOUP(order, L), kwargs...)

"""
    RosenbrockExpEK(; order=3, kwargs...)

**Probabilistic Rosenbrock-type exponential integrator**

A probabilistic exponential integrator similar to [`ExpEK`](@ref), but with automatic
linearization along the mean numerical solution. This brings the advantage that the
linearity does not need to be specified manually, and the more accurate local linearization
can sometimes also improve stability; but since the "prior" is adjusted at each step the
probabilistic interpretation becomes more complicated.

`RosenbrockExpEK` is just a short-hand for [`EK1`](@ref) with appropriete [`IOUP`](@ref)
prior:
```julia
RosenbrockExpEK(; order=3, kwargs...) = EK1(; prior=IOUP(order, update_rate_parameter=true), kwargs...)
```

See also [`ExpEK`](@ref), [`EK0`](@ref), [`EK1`](@ref).

# Arguments
See [`EK1`](@ref) for available keyword arguments.

# Examples
```julia-repl
julia> prob = ODEProblem((du, u, p, t) -> (@. du = - u + sin(u)), [1.0], (0.0, 10.0))
julia> solve(prob, RosenbrockExpEK())
```

# Reference
* [bosch23expint](@cite) Bosch et al, "Probabilistic Exponential Integrators", arXiv (2021)
"""
RosenbrockExpEK(; order=3, kwargs...) =
    EK1(; prior=IOUP(order, update_rate_parameter=true), kwargs...)

function DiffEqBase.remake(thing::EK1{CS,AD,DT,ST,CJ}; kwargs...) where {CS,AD,DT,ST,CJ}
    T = SciMLBase.remaker_of(thing)
    T(;
        SciMLBase.struct_as_namedtuple(thing)...,
        chunk_size=Val{CS}(),
        autodiff=Val{AD}(),
        standardtag=Val{ST}(),
        concrete_jac=CJ === nothing ? CJ : Val{CJ}(),
        diff_type=DT,
        kwargs...,
    )
end

function DiffEqBase.prepare_alg(alg::EK1{0}, u0::AbstractArray{T}, p, prob) where {T}
    # See OrdinaryDiffEq.jl: ./src/alg_utils.jl (where this is copied from).
    # In the future we might want to make EK1 an OrdinaryDiffEqAdaptiveImplicitAlgorithm and
    # use the prepare_alg from OrdinaryDiffEq; but right now, we do not use `linsolve` which
    # is a requirement.

    if (isbitstype(T) && sizeof(T) > 24) || (
        prob.f isa ODEFunction &&
        prob.f.f isa FunctionWrappersWrappers.FunctionWrappersWrapper
    )
        return remake(alg, chunk_size=Val{1}())
    end

    L = StaticArrayInterface.known_length(typeof(u0))
    @assert L === nothing "ProbNumDiffEq.jl does not support StaticArrays yet."

    x = if prob.f.colorvec === nothing
        length(u0)
    else
        maximum(prob.f.colorvec)
    end
    cs = ForwardDiff.pickchunksize(x)
    return remake(alg, chunk_size=Val{cs}())
end
