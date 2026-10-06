# `_process_AD_choice` normalizes the legacy `(autodiff, chunk_size, diff_type)`
# kwargs into an `ADTypes.AbstractADType`. v3's OrdinaryDiffEqCore exports this
# helper; v4 (OrdinaryDiffEq v7) removed it along with the legacy kwargs. We keep
# accepting the kwargs by delegating to OrdinaryDiffEqCore on v3 and inlining an
# equivalent shim on v4.
@static if isdefined(OrdinaryDiffEqCore, :_process_AD_choice)
    _process_AD_choice(autodiff, chunk_size, diff_type) =
        OrdinaryDiffEqCore._process_AD_choice(autodiff, chunk_size, diff_type)
else
    function _process_AD_choice(autodiff, chunk_size, diff_type)
        ad = autodiff
        if ad isa Bool
            # Mirror v3: `true` → AutoForwardDiff(chunksize=chunk_size), `false` →
            # AutoFiniteDiff(fdtype=diff_type, dir=1). Without this a Bool would
            # be stored as the `autodiff` field and break `AbstractADType` dispatch.
            if ad
                cs_int = _unwrap_val(chunk_size)
                cs = cs_int == 0 ? nothing : cs_int
                ad = ADTypes.AutoForwardDiff(; chunksize=cs)
            else
                fdtype = diff_type isa Val ? diff_type : Val(diff_type)
                ad = ADTypes.AutoFiniteDiff(; fdtype=fdtype, dir=1)
            end
        elseif ad isa ADTypes.AutoForwardDiff
            cs_int = _unwrap_val(chunk_size)
            # If the user set an explicit chunk_size kwarg, fold it into the ADType.
            # Otherwise defer to whatever chunksize is already on `ad`.
            if cs_int != 0
                ad = ADTypes.AutoForwardDiff(; chunksize=cs_int, tag=ad.tag)
            end
        elseif ad isa ADTypes.AutoFiniteDiff
            # Only override the ADType's fdtype if the user passed a non-default
            # legacy `diff_type` kwarg; otherwise preserve their pre-configured
            # AutoFiniteDiff (matching v3's behavior).
            fdtype = diff_type isa Val ? diff_type : Val(diff_type)
            if fdtype !== Val(:forward)
                ad = ADTypes.AutoFiniteDiff(; fdtype=fdtype)
            end
        end
        return (ad, chunk_size, diff_type)
    end
end

########################################################################################
# Linearizations
########################################################################################
"""
    AbstractLinearization

How an `ODEFilter` linearizes the ODE vector field, that is, which approximation of its
Jacobian enters the observation matrix `H`: `ZeroJacobian` in the [`EK0`](@ref),
`DiagonalJacobian` in the [`DiagonalEK1`](@ref) and `FullJacobian` in the [`EK1`](@ref).
"""
abstract type AbstractLinearization end
struct ZeroJacobian <: AbstractLinearization end
struct DiagonalJacobian <: AbstractLinearization end
struct FullJacobian <: AbstractLinearization end

"""
    needs_jacobian(component)

Whether a component of an `ODEFilter` needs the Jacobian of the ODE vector field.
"""
needs_jacobian(::ZeroJacobian) = false
needs_jacobian(::Union{DiagonalJacobian,FullJacobian}) = true
needs_jacobian(::AbstractGaussMarkovProcess) = false
needs_jacobian(prior::IOUP) = prior.update_rate_parameter

########################################################################################
# Algorithm
########################################################################################
"""
    ODEFilter(; linearization, prior, diffusionmodel, initialization, pn_observation_noise,
              smooth, covariance_factorization, autodiff, standardtag, concrete_jac)

**Gaussian ODE filter.** The solvers [`EK0`](@ref), [`EK1`](@ref) and [`DiagonalEK1`](@ref)
are `ODEFilter`s that differ only in their `linearization`; see them for the defaults.

- `covariance_factorization`: `IsometricKroneckerCovariance`, `BlockDiagonalCovariance`,
  `DenseCovariance`, or `nothing` for the most structured one that all other settings and
  the problem's mass matrix support.
- `autodiff`: an `ADTypes` choice for the Jacobian, or `nothing`. It is used if the
  `linearization` or the `prior` need a Jacobian, with `nothing` meaning
  `AutoForwardDiff()`, and is set to `nothing` otherwise.
"""
struct ODEFilter{LT,PT,DT,IT,RT,CF,AD,CJ} <:
       OrdinaryDiffEqCore.OrdinaryDiffEqAdaptiveAlgorithm
    linearization::LT
    prior::PT
    diffusionmodel::DT
    initialization::IT
    pn_observation_noise::RT
    smooth::Bool
    covariance_factorization::CF
    autodiff::AD
    standardtag::Bool
    concrete_jac::CJ
    function ODEFilter(;
        linearization,
        prior,
        diffusionmodel,
        initialization,
        pn_observation_noise,
        smooth,
        covariance_factorization,
        autodiff,
        standardtag,
        concrete_jac,
    )
        if (isstatic(diffusionmodel) && diffusionmodel.calibrate) &&
           (!isnothing(pn_observation_noise) && !iszero(pn_observation_noise))
            throw(
                ArgumentError(
                    "Automatic calibration of global diffusion models is not possible when using observation noise. If you want to calibrate a global diffusion parameter, do so setting `calibrate=false` and optimizing `sol.pnstats.log_likelihood` manually.",
                ),
            )
        end
        if !(needs_jacobian(linearization) || needs_jacobian(prior))
            autodiff = nothing
        elseif isnothing(autodiff)
            autodiff = AutoForwardDiff()
        end
        return new{
            typeof(linearization),
            typeof(prior),
            typeof(diffusionmodel),
            typeof(initialization),
            typeof(pn_observation_noise),
            _typeof(covariance_factorization),
            typeof(autodiff),
            typeof(concrete_jac),
        }(
            linearization,
            prior,
            diffusionmodel,
            initialization,
            pn_observation_noise,
            smooth,
            covariance_factorization,
            autodiff,
            standardtag,
            concrete_jac,
        )
    end
end

# `Type{T}` instead of `UnionAll`, so that `choose_covariance_structure` infers its result
_typeof(x) = typeof(x)
_typeof(T::Type) = Type{T}

_unwrap_val(::Val{B}) where {B} = B
_unwrap_val(B) = B

# The keyword arguments of `EK0`, `EK1` and `DiagonalEK1`, with their defaults. The
# `chunk_size` and `diff_type` of OrdinaryDiffEq 6 are folded into `autodiff`.
function _odefilter(
    linearization;
    order=3,
    prior=IWP(order),
    diffusionmodel=DynamicDiffusion(),
    smooth=true,
    initialization=TaylorModeInit(num_derivatives(prior)),
    pn_observation_noise=nothing,
    covariance_factorization=nothing,
    autodiff=AutoForwardDiff(),
    chunk_size=Val{0}(),
    diff_type=Val{:forward}(),
    standardtag=Val{true}(),
    concrete_jac=nothing,
)
    autodiff, _, _ = _process_AD_choice(autodiff, chunk_size, diff_type)
    return ODEFilter(;
        linearization,
        prior,
        diffusionmodel,
        initialization,
        pn_observation_noise,
        smooth,
        covariance_factorization,
        autodiff,
        standardtag=_unwrap_val(standardtag),
        concrete_jac=_unwrap_val(concrete_jac),
    )
end

"""
    EK0(; order=3,
          smooth=true,
          prior=IWP(order),
          diffusionmodel=DynamicDiffusion(),
          initialization=TaylorModeInit(num_derivatives(prior)))

**Gaussian ODE filter with zeroth-order vector field linearization.**

This is an _explicit_ ODE solver. It is fast and scales well to high-dimensional problems
[Krämer et al. (2022)](@cite krämer21highdim), but it is not L-stable [Tronarp et al. (2019)](@cite tronarp18probsol). So for stiff
problems, use the [`EK1`](@ref).

Whenever possible this solver will use a Kronecker-factored implementation to achieve its
linear scaling and to get the best runtimes. This can currently be done only with an
`IWP` prior (default), with a scalar diffusion model (either `DynamicDiffusion` or
`FixedDiffusion`). Otherwise it uses block-diagonal covariances, which also scale linearly,
for example with a multivariate diffusion model or a `Diagonal` mass matrix. _Settings that
only dense covariances can represent, such as an `IOUP` or `Matern` prior, throw an error
unless you pass `covariance_factorization=DenseCovariance`; dense covariances scale
cubically with the problem size._

# Arguments
- `order::Integer`: Order of the integrated Wiener process (IWP) prior.
- `smooth::Bool`: Turn smoothing on/off; smoothing is required for dense output.
- `prior::AbstractGaussMarkovProcess`: Prior to be used by the ODE filter.
   By default, uses a 3-times integrated Wiener process prior `IWP(3)`.
   See also: [Priors](@ref).
- `diffusionmodel::ProbNumDiffEq.AbstractDiffusion`: See [Diffusion models and calibration](@ref).
- `initialization::ProbNumDiffEq.InitializationScheme`: See [Initialization](@ref).

The keyword arguments of the [`EK1`](@ref) for computing Jacobians, such as `autodiff`,
are used for a prior that needs a Jacobian, such as an [`IOUP`](@ref) with
`update_rate_parameter=true`.

# Examples
```julia-repl
julia> solve(prob, EK0())
```

# [References](@ref references)
"""
EK0(; kwargs...) = _odefilter(ZeroJacobian(); kwargs...)

"""
    EK1(; order=3,
          smooth=true,
          prior=IWP(order),
          diffusionmodel=DynamicDiffusion(),
          initialization=TaylorModeInit(num_derivatives(prior)),
          kwargs...)

**Gaussian ODE filter with first-order vector field linearization.**

This is a _semi-implicit_, L-stable ODE solver so it can handle stiffness quite well [Tronarp et al. (2019)](@cite tronarp18probsol),
and it generally produces more expressive posterior covariances than the [`EK0`](@ref).
However, as typical implicit ODE solvers it scales cubically with the ODE dimension [Krämer et al. (2022)](@cite krämer21highdim),
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
EK1(; kwargs...) = _odefilter(FullJacobian(); kwargs...)

"""
    DiagonalEK1(; order=3,
                  smooth=true,
                  prior=IWP(order),
                  diffusionmodel=DynamicDiffusion(),
                  initialization=TaylorModeInit(num_derivatives(prior)),
                  kwargs...)

**Gaussian ODE filter with first-order vector field linearization and diagonal Jacobian approximation.**

A semi-implicit solver that approximates the Jacobian as diagonal, using a block-diagonal
covariance representation to achieve linear scaling with the ODE dimension
[Krämer et al. (2022)](@cite krämer21highdim). This makes it suitable for high-dimensional problems where
the full [`EK1`](@ref) would be too expensive. Settings that only dense covariances can
represent, such as an `IOUP` or `Matern` prior or a non-diagonal mass matrix, throw an error
unless you pass `covariance_factorization=DenseCovariance`.

!!! tip "Providing a Jacobian for linear scaling"
    For truly linear O(d) time and memory complexity, provide both a custom Jacobian
    function and a diagonal `jac_prototype` in your `ODEFunction`:
    ```julia
    ODEFunction(f; jac=jac, jac_prototype=Diagonal(ones(d)))
    ```
    Without these, the solver falls back to computing and storing the full d×d Jacobian
    via automatic differentiation and only extracting the diagonal afterwards, resulting
    in O(d²) time and memory for the Jacobian computation.

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

# Examples
```julia-repl
julia> solve(prob, DiagonalEK1())
```

# [References](@ref references)
"""
DiagonalEK1(; kwargs...) = _odefilter(DiagonalJacobian(); kwargs...)

"""
    ExpEK(; L, order=3, kwargs...)

**Probabilistic exponential integrator**

Probabilistic exponential integrators are a class of integrators for semi-linear stiff ODEs
that provide improved stability by essentially solving the linear part of the ODE exactly.
In probabilistic numerics, this amounts to including the linear part into the prior model
of the solver.

`ExpEK` is therefore just a short-hand for [`EK0`](@ref) with [`IOUP`](@ref) prior, which
needs dense covariances:
```julia
ExpEK(; order=3, L, kwargs...) =
    EK0(; prior=IOUP(order, L), covariance_factorization=DenseCovariance, kwargs...)
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
* [Bosch et al. (2023)](@cite bosch23expint) "Probabilistic Exponential Integrators", NeurIPS
"""
ExpEK(; L, order=3, kwargs...) =
    EK0(; prior=IOUP(order, L), covariance_factorization=DenseCovariance, kwargs...)

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
* [Bosch et al. (2023)](@cite bosch23expint) "Probabilistic Exponential Integrators", NeurIPS
"""
RosenbrockExpEK(; order=3, kwargs...) =
    EK1(; prior=IOUP(order, update_rate_parameter=true), kwargs...)

########################################################################################
# Covariance structure
########################################################################################
"""
    supports_covariance(input, S)

Whether the covariance structure `S` can represent `input`: a component of an `ODEFilter`,
its `ObservationNoise`, or the problem's `MassMatrix`.
"""
supports_covariance(::ZeroJacobian, S) = true
supports_covariance(::DiagonalJacobian, S) = S !== IsometricKroneckerCovariance
supports_covariance(::FullJacobian, S) = S === DenseCovariance
supports_covariance(::IWP, S) = true
supports_covariance(::AbstractGaussMarkovProcess, S) = S === DenseCovariance
supports_covariance(::Union{DynamicDiffusion,FixedDiffusion}, S) = true
supports_covariance(::DynamicMVDiffusion, S) = S === BlockDiagonalCovariance
supports_covariance(diffusion::FixedMVDiffusion, S) =
    S === BlockDiagonalCovariance || (S === DenseCovariance && !diffusion.calibrate)

struct ObservationNoise{T}
    value::T
end
supports_covariance(::ObservationNoise{<:Union{Nothing,Number,UniformScaling}}, S) = true
supports_covariance(::ObservationNoise{<:Diagonal{<:Number,<:FillArrays.Fill}}, S) = true
supports_covariance(::ObservationNoise{<:Diagonal}, S) = S !== IsometricKroneckerCovariance
supports_covariance(::ObservationNoise, S) = S === DenseCovariance

struct MassMatrix{T}
    value::T
end
supports_covariance(::MassMatrix{<:UniformScaling}, S) = true
supports_covariance(::MassMatrix{<:Diagonal}, S) = S !== IsometricKroneckerCovariance
supports_covariance(::MassMatrix, S) = S === DenseCovariance

"""
    choose_covariance_structure(alg::ODEFilter, mass_matrix)

The first of `IsometricKroneckerCovariance`, `BlockDiagonalCovariance` and
`DenseCovariance` that all settings of `alg` and the problem's `mass_matrix` support, or
`alg.covariance_factorization` if it is set and supported. Dense covariances are chosen
automatically only for a linearization that supports nothing else: a solver built for
structured covariances needs `covariance_factorization=DenseCovariance` for them.
"""
function choose_covariance_structure(alg::ODEFilter, mass_matrix)
    inputs = (;
        alg.linearization,
        alg.prior,
        alg.diffusionmodel,
        pn_observation_noise=ObservationNoise(alg.pn_observation_noise),
        mass_matrix=MassMatrix(mass_matrix),
    )
    supported(S) = all(map(input -> supports_covariance(input, S), Tuple(inputs)))
    C = alg.covariance_factorization
    if !isnothing(C)
        supported(C) && return _constant(C)
        msg =
            "`covariance_factorization = $C` is not supported by these inputs:\n" *
            _restrictions(inputs, (C,))
    elseif supported(IsometricKroneckerCovariance)
        return IsometricKroneckerCovariance
    elseif supported(BlockDiagonalCovariance)
        return BlockDiagonalCovariance
    elseif !supported(DenseCovariance)
        msg =
            "No covariance structure is supported by all of these inputs:\n" *
            _restrictions(inputs, COVARIANCE_STRUCTURES)
    elseif !supports_covariance(alg.linearization, BlockDiagonalCovariance)
        return DenseCovariance
    else
        msg =
            "These inputs rule out structured covariances:\n" *
            _restrictions(inputs, (BlockDiagonalCovariance,)) * "\n" *
            "Dense covariances scale cubically with the ODE dimension. For them, use " *
            "the `EK1`, or pass `covariance_factorization=DenseCovariance` to keep " *
            "this solver."
    end
    throw(ArgumentError(msg))
end

# A type read from a field is not a constant for inference, a static parameter is
_constant(::Type{C}) where {C} = C

const COVARIANCE_STRUCTURES =
    (IsometricKroneckerCovariance, BlockDiagonalCovariance, DenseCovariance)

# One line for each input that does not support all of `structures`
function _restrictions(inputs, structures)
    lines = String[]
    for (name, input) in pairs(inputs)
        supported = filter(S -> supports_covariance(input, S), COVARIANCE_STRUCTURES)
        if !all(in(supported), structures)
            push!(lines,
                "- `$name = $(_describe(input))` supports only $(join(supported, ", "))")
        end
    end
    return join(lines, "\n")
end
_describe(input) = repr(input)
_describe(input::Union{ObservationNoise,MassMatrix}) = summary(input.value)

# DAE-initialization hook.
#
# v3's `_default_dae_init!` is only extended for OrdinaryDiffEq's own implicit
# algorithms (in OrdinaryDiffEqNonlinearSolve), so without an override the EK
# solvers hit a MethodError on DAE / singular-mass-matrix problems. Route through
# `CheckInit` to match the OrdinaryDiffEq v7 default; users who want the previous
# silent-fix behavior pass `initializealg = BrownFullBasicInit()` explicitly.
# v4 dropped the hook and `CheckInit` is already its `DefaultInit` target.
@static if isdefined(OrdinaryDiffEqCore, :_default_dae_init!)
    function OrdinaryDiffEqCore._default_dae_init!(integrator, prob, x, alg::ODEFilter)
        return OrdinaryDiffEqCore._initialize_dae!(
            integrator,
            prob,
            DiffEqBase.CheckInit(),
            x,
        )
    end
end

function DiffEqBase.prepare_alg(alg::ODEFilter, u0::AbstractArray{T}, p, prob) where {T}
    isnothing(alg.autodiff) && return alg

    # See OrdinaryDiffEqCore.jl: ./src/alg_utils.jl (where this is copied from).
    # In the future we might want to make the ODEFilter an
    # OrdinaryDiffEqAdaptiveImplicitAlgorithm and use the prepare_alg from
    # OrdinaryDiffEqCore; but right now, we do not use `linsolve` which is a requirement.

    prepped_AD = OrdinaryDiffEqDifferentiation.prepare_ADType(
        OrdinaryDiffEqCore.alg_autodiff(alg),
        prob,
        u0,
        p,
        OrdinaryDiffEqCore.standardtag(alg),
    )

    sparse_prepped_AD =
        OrdinaryDiffEqDifferentiation.prepare_user_sparsity(prepped_AD, prob)

    L = StaticArrayInterface.known_length(typeof(u0))
    @assert L === nothing "ProbNumDiffEq.jl does not support StaticArrays yet."

    if (
        (
            (eltype(u0) <: Complex) ||
            (!(prob.f isa DAEFunction) && prob.f.mass_matrix isa MatrixOperator)
        ) && sparse_prepped_AD isa AutoSparse
    )
        @warn "Input type or problem definition is incompatible with sparse automatic differentiation. Switching to using dense automatic differentiation."
        autodiff = ADTypes.dense_ad(sparse_prepped_AD)
    else
        autodiff = sparse_prepped_AD
    end

    return remake(alg, autodiff=autodiff)
end
