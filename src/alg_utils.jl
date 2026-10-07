############################################################################################
# For the equivalent parts in OrdinaryDiffEqCore.jl, see:
# https://github.com/SciML/OrdinaryDiffEqCore.jl/blob/master/src/alg_utils.jl
############################################################################################

# v3 reaches autodiff settings via the private `_alg_autodiff`; v4 promoted
# `alg_autodiff` to the public entry point and deleted `_alg_autodiff`. Define both:
# the `_alg_autodiff` override is gated so v4 doesn't see it, and `alg_autodiff` is
# harmlessly redefined on v3 (an existing function gets a new method).
@static if isdefined(OrdinaryDiffEqDifferentiation, :_alg_autodiff)
    OrdinaryDiffEqDifferentiation._alg_autodiff(alg::ODEFilter) = alg.autodiff
end
OrdinaryDiffEqCore.alg_autodiff(alg::ODEFilter) = alg.autodiff
OrdinaryDiffEqCore.has_autodiff(alg::ODEFilter) = !isnothing(alg.autodiff)
OrdinaryDiffEqCore.standardtag(alg::ODEFilter) = alg.standardtag
OrdinaryDiffEqCore.concrete_jac(alg::ODEFilter) = alg.concrete_jac

# Unlike OrdinaryDiffEqCore's `_get_fwd_chunksize`, never `Val(nothing)` (early v3 versions)
_chunksize(::Type{<:AutoForwardDiff{CS}}) where {CS} = Val(something(CS, 0))
_chunksize(AD) = Val(0)
OrdinaryDiffEqCore.get_chunksize(alg::ODEFilter) = _chunksize(typeof(alg.autodiff))
@static if isdefined(SciMLBase, :forwarddiff_chunksize)
    SciMLBase.forwarddiff_chunksize(alg::ODEFilter) = OrdinaryDiffEqCore.get_chunksize(alg)
end

# Tell SciMLBase that the Jacobian is computed with ForwardDiff on the model function,
# so that FunctionWrappersWrapper registers Dual-compatible wrappers.
function SciMLBase.forwarddiffs_model(alg::ODEFilter)
    ad = alg.autodiff
    ad isa AutoSparse && return ADTypes.dense_ad(ad) isa AutoForwardDiff
    return ad isa AutoForwardDiff
end

@inline DiffEqBase.get_tmp_cache(integ, alg::ODEFilter, cache::AbstractODEFilterCache) =
    (cache.tmp, cache.atmp)
OrdinaryDiffEqCore.isfsal(::ODEFilter) = false

############################################
# Step size control
SciMLBase.isadaptive(::ODEFilter) = true
SciMLBase.alg_order(alg::ODEFilter) = num_derivatives(alg.prior)
# OrdinaryDiffEqCore.alg_adaptive_order(alg::ODEFilter) =

# PI control is the default. On OrdinaryDiffEqCore v3 we have to explicitly set the
# `isstandard`/`ispredictive` traits to false; on v4 those traits were removed and the
# default is PI already (via `default_controller`), so we only opt in on the old version.
@static if isdefined(OrdinaryDiffEqCore, :isstandard)
    OrdinaryDiffEqCore.isstandard(::ODEFilter) = false # proportional
    OrdinaryDiffEqCore.ispredictive(::ODEFilter) = false # not sure, maybe Gustafsson acceleration?
end

# OrdinaryDiffEqCore.qmin_default(alg::ODEFilter) =
# OrdinaryDiffEqCore.qmax_default(alg::ODEFilter) =
# OrdinaryDiffEqCore.beta2_default(alg::ODEFilter) = 2 // (5(OrdinaryDiffEqCore.alg_order(alg) + 1))
# OrdinaryDiffEqCore.beta1_default(alg::ODEFilter, beta2) = 7 // (10(OrdinaryDiffEqCore.alg_order(alg) + 1))
# OrdinaryDiffEqCore.gamma_default(alg::ODEFilter) =

# OrdinaryDiffEqCore.uses_uprev(alg::, adaptive::Bool) = adaptive
OrdinaryDiffEqCore.is_mass_matrix_alg(::ODEFilter) = true

SciMLBase.isautodifferentiable(::ODEFilter) = true
SciMLBase.allows_arbitrary_number_types(::ODEFilter) = true
SciMLBase.allowscomplex(::ODEFilter) = false
