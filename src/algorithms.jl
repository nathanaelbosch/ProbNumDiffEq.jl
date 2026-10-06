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

How an [`ODEFilter`](@ref) linearizes the ODE vector field, that is, which approximation of
its Jacobian enters the observation matrix `H`.
"""
abstract type AbstractLinearization end

"""
    ZeroJacobian()

Linearize the ODE vector field with a zero Jacobian, as the [`EK0`](@ref) does.
"""
struct ZeroJacobian <: AbstractLinearization end

"""
    DiagonalJacobian()

Linearize the ODE vector field with the diagonal of its Jacobian, as the
[`DiagonalEK1`](@ref) does.
"""
struct DiagonalJacobian <: AbstractLinearization end

"""
    FullJacobian()

Linearize the ODE vector field with its Jacobian, as the [`EK1`](@ref) does.
"""
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
    ODEFilter(; linearization, order=3, prior=IWP(order), diffusionmodel=DynamicDiffusion(),
              smooth=true, initialization=TaylorModeInit(num_derivatives(prior)),
              pn_observation_noise=nothing, covariance_factorization=nothing,
              autodiff=AutoForwardDiff(), standardtag=true, concrete_jac=nothing)

**Gaussian ODE filter**, the probabilistic ODE solver behind the [`EK0`](@ref), the
[`EK1`](@ref) and the [`DiagonalEK1`](@ref). These are `ODEFilter`s with a fixed
`linearization` and take all other keyword arguments below.

# Arguments
- `linearization::AbstractLinearization`: How the ODE vector field is linearized:
  [`ZeroJacobian()`](@ref ZeroJacobian), [`DiagonalJacobian()`](@ref DiagonalJacobian) or
  [`FullJacobian()`](@ref FullJacobian).
- `order::Integer`: Order of the default integrated Wiener process (IWP) prior.
- `prior::AbstractGaussMarkovProcess`: Prior to be used by the ODE filter.
   By default, an `order`-times integrated Wiener process prior `IWP(order)`.
   See also: [Priors](@ref).
- `diffusionmodel::ProbNumDiffEq.AbstractDiffusion`: See [Diffusion models and calibration](@ref).
- `smooth::Bool`: Turn smoothing on/off; smoothing is required for dense output.
- `initialization::ProbNumDiffEq.InitializationScheme`: See [Initialization](@ref).
- `pn_observation_noise`: Covariance of noise on the observation of the ODE, as a number, a
  `UniformScaling` or a matrix; `nothing` for none.
- `covariance_factorization`: `IsometricKroneckerCovariance`, `BlockDiagonalCovariance` or
  `DenseCovariance`. By default, the most structured one that all other settings and the
  problem's mass matrix support. The `EK0` and the `DiagonalEK1` use `DenseCovariance` only
  when it is set here, since dense covariances scale cubically with the ODE dimension.
- `autodiff`: How to compute the Jacobian if the `linearization` or the `prior` need one,
  e.g. `AutoForwardDiff()` or `AutoFiniteDiff()` from ADTypes.jl.
- `standardtag`, `concrete_jac`: As for the implicit solvers of OrdinaryDiffEq.jl.

For compatibility with OrdinaryDiffEq 6, the keywords `chunk_size`, `diff_type` and
`autodiff=true`/`false` are folded into `autodiff`.

# Examples
```julia-repl
julia> solve(prob, ODEFilter(linearization=FullJacobian(), order=5))
```
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
        standardtag=true,
        concrete_jac=nothing,
    )
        if (isstatic(diffusionmodel) && diffusionmodel.calibrate) &&
           (!isnothing(pn_observation_noise) && !iszero(pn_observation_noise))
            throw(
                ArgumentError(
                    "Automatic calibration of global diffusion models is not possible when using observation noise. If you want to calibrate a global diffusion parameter, do so setting `calibrate=false` and optimizing `sol.pnstats.log_likelihood` manually.",
                ),
            )
        end
        autodiff, _, _ = _process_AD_choice(autodiff, chunk_size, diff_type)
        if !(needs_jacobian(linearization) || needs_jacobian(prior))
            autodiff = nothing
        elseif isnothing(autodiff)
            autodiff = AutoForwardDiff()
        end
        concrete_jac = _unwrap_val(concrete_jac)
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
            _unwrap_val(standardtag),
            concrete_jac,
        )
    end
end

# `Type{T}` instead of `UnionAll`, so that `choose_covariance_structure` infers its result
_typeof(x) = typeof(x)
_typeof(T::Type) = Type{T}

_unwrap_val(::Val{B}) where {B} = B
_unwrap_val(B) = B

"""
    EK0(; order=3,
          smooth=true,
          prior=IWP(order),
          diffusionmodel=DynamicDiffusion(),
          initialization=TaylorModeInit(num_derivatives(prior)),
          kwargs...)

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

The `EK0` is the [`ODEFilter`](@ref) with `linearization=ZeroJacobian()`; see there for all
keyword arguments.

# Examples
```julia-repl
julia> solve(prob, EK0())
```

# [References](@ref references)
"""
EK0(; kwargs...) = ODEFilter(; linearization=ZeroJacobian(), kwargs...)

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

The `EK1` is the [`ODEFilter`](@ref) with `linearization=FullJacobian()`; see there for all
keyword arguments.

# Examples
```julia-repl
julia> solve(prob, EK1())
```

# [References](@ref references)
"""
EK1(; kwargs...) = ODEFilter(; linearization=FullJacobian(), kwargs...)

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

The `DiagonalEK1` is the [`ODEFilter`](@ref) with `linearization=DiagonalJacobian()`; see
there for all keyword arguments.

# Examples
```julia-repl
julia> solve(prob, DiagonalEK1())
```

# [References](@ref references)
"""
DiagonalEK1(; kwargs...) = ODEFilter(; linearization=DiagonalJacobian(), kwargs...)

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

_solver_name(::ZeroJacobian) = "EK0"
_solver_name(::FullJacobian) = "EK1"
_solver_name(::DiagonalJacobian) = "DiagonalEK1"
_solver_name(::AbstractLinearization) = nothing

# Shown as the call that constructs it, with the settings that differ from the defaults
function Base.show(io::IO, alg::ODEFilter)
    name = _solver_name(alg.linearization)
    settings = Pair{Symbol,Any}[]
    isnothing(name) && push!(settings, :linearization => alg.linearization)
    q = num_derivatives(alg.prior)
    if alg.prior != IWP(q)
        push!(settings, :prior => alg.prior)
    elseif q != 3
        push!(settings, :order => q)
    end
    default = ODEFilter(; alg.linearization, alg.prior)
    for field in fieldnames(ODEFilter)
        field in (:linearization, :prior) && continue
        value = getfield(alg, field)
        isequal(value, getfield(default, field)) || push!(settings, field => value)
    end
    print(io, something(name, "ODEFilter"), "(")
    join(io, (string(k, "=", repr(v; context=io)) for (k, v) in settings), ", ")
    print(io, ")")
end
Base.show(io::IO, ::MIME"text/plain", alg::ODEFilter) = show(io, alg)

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
                "- `$name = $(_describe(input))` supports only $(join(supported, ", "))" *
                _hint(input))
        end
    end
    return join(lines, "\n")
end
_describe(input) = repr(input)
_describe(input::Union{ObservationNoise,MassMatrix}) = summary(input.value)
_hint(input) = ""
_hint(input::Union{ObservationNoise,MassMatrix}) =
    !(input.value isa Diagonal) && input.value isa AbstractMatrix && isdiag(input.value) ?
    "; it is diagonal, so pass it as a `Diagonal`" : ""

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
