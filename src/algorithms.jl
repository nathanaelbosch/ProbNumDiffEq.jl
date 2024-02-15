########################################################################################
# Algorithm
########################################################################################
abstract type AbstractEK <: OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm end

function ekargcheck(
    alg;
    diffusionmodel,
    pn_observation_noise,
    covariance_factorization,
    kwargs...,
)
    if (isstatic(diffusionmodel) && diffusionmodel.calibrate) &&
       (!isnothing(pn_observation_noise) && !iszero(pn_observation_noise))
        throw(
            ArgumentError(
                "Automatic calibration of global diffusion models is not possible when using observation noise. If you want to calibrate a global diffusion parameter, do so setting `calibrate=false` and optimizing `sol.pnstats.log_likelihood` manually.",
            ),
        )
    end
    if alg == EK1
        if diffusionmodel isa FixedMVDiffusion && diffusionmodel.calibrate
            throw(
                ArgumentError(
                    "The `EK1` algorithm does not support automatic global calibration of multivariate diffusion models. Either use a scalar diffusion model, or set `calibrate=false` and calibrate manually by optimizing `sol.pnstats.log_likelihood`. Or use a different solve, like `EK0` or `DiagonalEK1`.",
                ),
            )
        elseif diffusionmodel isa DynamicMVDiffusion
            throw(
                ArgumentError(
                    "The `EK1` algorithm does not support automatic calibration of local multivariate diffusion models. Either use a scalar diffusion model, or use a different solve, like `EK0` or `DiagonalEK1`.",
                ),
            )
        end
    end
end

function covariance_structure(::Type{Alg}, prior, diffusionmodel) where {Alg<:AbstractEK}
    if Alg <: EK0
        if prior isa IWP
            if (diffusionmodel isa DynamicDiffusion || diffusionmodel isa FixedDiffusion)
                return IsometricKroneckerCovariance
            else
                return BlockDiagonalCovariance
            end
        else
            # This is not great as other priors can be Kronecker too; TODO
            return DenseCovariance
        end
    elseif Alg <: DiagonalEK1
        return BlockDiagonalCovariance
    elseif Alg <: EK1
        return DenseCovariance
    else
        throw(ArgumentError("Unknown algorithm type $Alg"))
    end
end
covariance_structure(alg) = covariance_structure(typeof(alg), alg.prior, alg.diffusionmodel)

"""
    EK0(; order=3,
          smooth=true,
          prior=IWP(order),
          diffusionmodel=DynamicDiffusion(),
          initialization=TaylorModeInit(num_derivatives(prior)))

**Gaussian ODE filter with zeroth-order vector field linearization.**

This is an _explicit_ ODE solver. It is fast and scales well to high-dimensional problems
[krämer21highdim](@cite), but it is not L-stable [tronarp18probsol](@cite). So for stiff
problems, use the [`EK1`](@ref).

Whenever possible this solver will use a Kronecker-factored implementation to achieve its
linear scaling and to get the best runtimes. This can currently be done only with an
`IWP` prior (default), with a scalar diffusion model (either `DynamicDiffusion` or
`FixedDiffusion`). _For other configurations the solver falls back to a dense implementation
which scales cubically with the problem size._

# Arguments
- `order::Integer`: Order of the integrated Wiener process (IWP) prior.
- `smooth::Bool`: Turn smoothing on/off; smoothing is required for dense output.
- `prior::AbstractGaussMarkovProcess`: Prior to be used by the ODE filter.
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
struct EK0{PT,DT,IT,RT,CF} <: AbstractEK
    prior::PT
    diffusionmodel::DT
    smooth::Bool
    initialization::IT
    pn_observation_noise::RT
    covariance_factorization::CF
    EK0(; order=3,
        prior::PT=IWP(order),
        diffusionmodel::DT=DynamicDiffusion(),
        smooth=true,
        initialization::IT=TaylorModeInit(num_derivatives(prior)),
        pn_observation_noise::RT=nothing,
        covariance_factorization::CF=covariance_structure(EK0, prior, diffusionmodel),
    ) where {PT,DT,IT,RT,CF} = begin
        ekargcheck(EK0; diffusionmodel, pn_observation_noise, covariance_factorization)
        new{PT,DT,IT,RT,CF}(
            prior, diffusionmodel, smooth, initialization, pn_observation_noise,
            covariance_factorization)
    end
end

_unwrap_val(::Val{B}) where {B} = B
_unwrap_val(B) = B

"""
    EK1(; order=3,
          smooth=true,
          prior=IWP(order),
          diffusionmodel=DynamicDiffusion(),
          initialization=TaylorModeInit(num_derivatives(prior)),
          kwargs...)

**Gaussian ODE filter with first-order vector field linearization.**

This is a _semi-implicit_, L-stable ODE solver so it can handle stiffness quite well [tronarp18probsol](@cite),
and it generally produces more expressive posterior covariances than the [`EK0`](@ref).
However, as typical implicit ODE solvers it scales cubically with the ODE dimension [krämer21highdim](@cite),
so if you're solving a high-dimensional non-stiff problem you might want to give the [`EK0`](@ref) a try.

# Arguments
- `order::Integer`: Order of the integrated Wiener process (IWP) prior.
- `smooth::Bool`: Turn smoothing on/off; smoothing is required for dense output.
- `prior::AbstractGaussMarkovProcess`: Prior to be used by the ODE filter.
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
struct EK1{CS,AD,DiffType,ST,CJ,PT,DT,IT,RT,CF} <: AbstractEK
    prior::PT
    diffusionmodel::DT
    smooth::Bool
    initialization::IT
    pn_observation_noise::RT
    covariance_factorization::CF
    EK1(;
        order=3,
        prior::PT=IWP(order),
        diffusionmodel::DT=DynamicDiffusion(),
        smooth=true,
        initialization::IT=TaylorModeInit(num_derivatives(prior)),
        chunk_size=Val{0}(),
        autodiff=Val{true}(),
        diff_type=Val{:forward},
        standardtag=Val{true}(),
        concrete_jac=nothing,
        pn_observation_noise::RT=nothing,
        covariance_factorization::CF=covariance_structure(EK1, prior, diffusionmodel),
    ) where {PT,DT,IT,RT,CF} = begin
        ekargcheck(EK1; diffusionmodel, pn_observation_noise, covariance_factorization)
        new{
            _unwrap_val(chunk_size),
            _unwrap_val(autodiff),
            diff_type,
            _unwrap_val(standardtag),
            _unwrap_val(concrete_jac),
            PT,
            DT,
            IT,
            RT,
            CF,
        }(
            prior,
            diffusionmodel,
            smooth,
            initialization,
            pn_observation_noise,
            covariance_factorization,
        )
    end
end

struct DiagonalEK1{CS,AD,DiffType,ST,CJ,PT,DT,IT,RT,CF} <: AbstractEK
    prior::PT
    diffusionmodel::DT
    smooth::Bool
    initialization::IT
    pn_observation_noise::RT
    covariance_factorization::CF
    DiagonalEK1(;
        order=3,
        prior::PT=IWP(order),
        diffusionmodel::DT=DynamicDiffusion(),
        smooth=true,
        initialization::IT=TaylorModeInit(num_derivatives(prior)),
        chunk_size=Val{0}(),
        autodiff=Val{true}(),
        diff_type=Val{:forward},
        standardtag=Val{true}(),
        concrete_jac=nothing,
        pn_observation_noise::RT=nothing,
        covariance_factorization::CF=covariance_structure(
            DiagonalEK1,
            prior,
            diffusionmodel,
        ),
    ) where {PT,DT,IT,RT,CF} = begin
        ekargcheck(DiagonalEK1; diffusionmodel, pn_observation_noise, covariance_factorization)
        new{
            _unwrap_val(chunk_size),
            _unwrap_val(autodiff),
            diff_type,
            _unwrap_val(standardtag),
            _unwrap_val(concrete_jac),
            PT,
            DT,
            IT,
            RT,
            CF,
        }(
            prior,
            diffusionmodel,
            smooth,
            initialization,
            pn_observation_noise,
            covariance_factorization,
        )
    end
end

"""
    ExpEK(; L, order=3, kwargs...)

**Probabilistic exponential integrator**

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

`RosenbrockExpEK` is just a short-hand for [`EK1`](@ref) with locally-updated [`IOUP`](@ref)
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
* [bosch23expint](@cite) Bosch et al, "Probabilistic Exponential Integrators", NeurIPS (2023)
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

function DiffEqBase.prepare_alg(
    alg::Union{EK1{0},DiagonalEK1{0}},
    u0::AbstractArray{T},
    p,
    prob,
) where {T}
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
