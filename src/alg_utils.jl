############################################################################################
# For the equivalent parts in OrdinaryDiffEqCore.jl, see:
# https://github.com/SciML/OrdinaryDiffEqCore.jl/blob/master/src/alg_utils.jl
############################################################################################

# OrdinaryDiffEqCore v3 reached autodiff settings via the private `_alg_autodiff` hook;
# v4 renamed the public entry point to `alg_autodiff` and deleted `_alg_autodiff`. Set
# whichever one exists so prepare_alg / jacobian wrappers can find our AD choice on
# either version. Defining both on v3 is harmless (v3 still calls `_alg_autodiff`
# internally, and `alg_autodiff` simply gets a new method).
@static if isdefined(OrdinaryDiffEqDifferentiation, :_alg_autodiff)
    OrdinaryDiffEqDifferentiation._alg_autodiff(::AbstractEK) = Val{true}()
end
OrdinaryDiffEqDifferentiation.alg_autodiff(::AbstractEK) = ADTypes.AutoForwardDiff()
OrdinaryDiffEqDifferentiation.standardtag(::AbstractEK) = false
OrdinaryDiffEqDifferentiation.concrete_jac(::AbstractEK) = nothing

@inline DiffEqBase.get_tmp_cache(integ, alg::AbstractEK, cache::AbstractODEFilterCache) =
    (cache.tmp, cache.atmp)
OrdinaryDiffEqCore.isfsal(::AbstractEK) = false

for ALG in [:EK1, :DiagonalEK1]
    @static if isdefined(OrdinaryDiffEqDifferentiation, :_alg_autodiff)
        @eval OrdinaryDiffEqDifferentiation._alg_autodiff(alg::$ALG{CS,AD}) where {CS,AD} =
            alg.autodiff
    end
    @eval OrdinaryDiffEqDifferentiation.alg_autodiff(alg::$ALG) = alg.autodiff
    @eval OrdinaryDiffEqDifferentiation.alg_difftype(
        ::$ALG{CS,AD,DiffType},
    ) where {CS,AD,DiffType} =
        DiffType
    @eval OrdinaryDiffEqDifferentiation.standardtag(
        ::$ALG{CS,AD,DiffType,ST},
    ) where {CS,AD,DiffType,ST} =
        ST
    @eval OrdinaryDiffEqDifferentiation.concrete_jac(
        ::$ALG{CS,AD,DiffType,ST,CJ},
    ) where {CS,AD,DiffType,ST,CJ} = CJ
    @eval OrdinaryDiffEqDifferentiation.get_chunksize(::$ALG{CS}) where {CS} = Val(CS)
    @eval OrdinaryDiffEqCore.isimplicit(::$ALG) = true
end

############################################
# Step size control
OrdinaryDiffEqCore.isadaptive(::AbstractEK) = true
OrdinaryDiffEqCore.alg_order(alg::AbstractEK) = num_derivatives(alg.prior)
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
