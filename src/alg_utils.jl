############################################################################################
# For the equivalent parts in OrdinaryDiffEqCore.jl, see:
# https://github.com/SciML/OrdinaryDiffEqCore.jl/blob/master/src/alg_utils.jl
############################################################################################

# v3 reaches autodiff settings via the private `_alg_autodiff`; v4 promoted
# `alg_autodiff` to the public entry point and deleted `_alg_autodiff`. Define both:
# the `_alg_autodiff` override is gated so v4 doesn't see it, and `alg_autodiff` is
# harmlessly redefined on v3 (an existing function gets a new method).
@static if isdefined(OrdinaryDiffEqDifferentiation, :_alg_autodiff)
    OrdinaryDiffEqDifferentiation._alg_autodiff(::AbstractEK) = Val{true}()
end
OrdinaryDiffEqCore.alg_autodiff(::AbstractEK) = ADTypes.AutoForwardDiff()
OrdinaryDiffEqCore.standardtag(::AbstractEK) = false
OrdinaryDiffEqCore.concrete_jac(::AbstractEK) = nothing

@inline DiffEqBase.get_tmp_cache(integ, alg::AbstractEK, cache::AbstractODEFilterCache) =
    (cache.tmp, cache.atmp)
OrdinaryDiffEqCore.isfsal(::AbstractEK) = false

# Unlike OrdinaryDiffEqCore's `_get_fwd_chunksize`, never `Val(nothing)` (early v3 versions)
_chunksize(::Type{<:AutoForwardDiff{CS}}) where {CS} = Val(something(CS, 0))
_chunksize(AD) = Val(0)

for ALG in [:EK1, :DiagonalEK1]
    @static if isdefined(OrdinaryDiffEqDifferentiation, :_alg_autodiff)
        @eval OrdinaryDiffEqDifferentiation._alg_autodiff(alg::$ALG{CS,AD}) where {CS,AD} =
            alg.autodiff
    end
    @eval OrdinaryDiffEqCore.alg_autodiff(alg::$ALG) = alg.autodiff
    @eval OrdinaryDiffEqCore.alg_difftype(
        ::$ALG{CS,AD,DiffType},
    ) where {CS,AD,DiffType} =
        DiffType
    @eval OrdinaryDiffEqCore.standardtag(
        ::$ALG{CS,AD,DiffType,ST},
    ) where {CS,AD,DiffType,ST} =
        ST
    @eval OrdinaryDiffEqCore.concrete_jac(
        ::$ALG{CS,AD,DiffType,ST,CJ},
    ) where {CS,AD,DiffType,ST,CJ} = CJ
    @eval OrdinaryDiffEqCore.get_chunksize(::$ALG{CS,AD}) where {CS,AD} = _chunksize(AD)
    @static if isdefined(SciMLBase, :forwarddiff_chunksize)
        @eval SciMLBase.forwarddiff_chunksize(alg::$ALG) =
            OrdinaryDiffEqCore.get_chunksize(alg)
    end
    @eval OrdinaryDiffEqCore.has_autodiff(::$ALG) = true
    @eval OrdinaryDiffEqCore.isimplicit(::$ALG) = true
end

############################################
# Step size control
SciMLBase.isadaptive(::AbstractEK) = true
SciMLBase.alg_order(alg::AbstractEK) = num_derivatives(alg.prior)
# OrdinaryDiffEqCore.alg_adaptive_order(alg::AbstractEK) =

# PI control is the default. On OrdinaryDiffEqCore v3 we have to explicitly set the
# `isstandard`/`ispredictive` traits to false; on v4 those traits were removed and the
# default is PI already (via `default_controller`), so we only opt in on the old version.
@static if isdefined(OrdinaryDiffEqCore, :isstandard)
    OrdinaryDiffEqCore.isstandard(::AbstractEK) = false # proportional
    OrdinaryDiffEqCore.ispredictive(::AbstractEK) = false # not sure, maybe Gustafsson acceleration?
end

# OrdinaryDiffEqCore.qmin_default(alg::AbstractEK) =
# OrdinaryDiffEqCore.qmax_default(alg::AbstractEK) =
# OrdinaryDiffEqCore.beta2_default(alg::AbstractEK) = 2 // (5(OrdinaryDiffEqCore.alg_order(alg) + 1))
# OrdinaryDiffEqCore.beta1_default(alg::AbstractEK, beta2) = 7 // (10(OrdinaryDiffEqCore.alg_order(alg) + 1))
# OrdinaryDiffEqCore.gamma_default(alg::AbstractEK) =

# OrdinaryDiffEqCore.uses_uprev(alg::, adaptive::Bool) = adaptive
OrdinaryDiffEqCore.is_mass_matrix_alg(::AbstractEK) = true

SciMLBase.isautodifferentiable(::AbstractEK) = true
SciMLBase.allows_arbitrary_number_types(::AbstractEK) = true
SciMLBase.allowscomplex(::AbstractEK) = false
