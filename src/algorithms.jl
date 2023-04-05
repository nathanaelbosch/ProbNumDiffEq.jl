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
struct EK0{CS,AD,DiffType,ST,CJ,PT,DT,IT} <: AbstractEK
    prior::PT
    diffusionmodel::DT
    smooth::Bool
    initialization::IT
end
EK0(;
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
) where {PT,DT,IT} = EK0{
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

function DiffEqBase.remake(
    thing::Union{EK1{CS,AD,DT,ST,CJ},EK0{CS,AD,DT,ST,CJ}};
    kwargs...,
) where {CS,AD,DT,ST,CJ}
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
    alg::Union{EK1{0},EK0{0}},
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
