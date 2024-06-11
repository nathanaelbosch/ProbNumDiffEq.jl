############################################################################################
# For the equivalent parts in OrdinaryDiffEq.jl, see:
# https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/src/alg_utils.jl
############################################################################################

OrdinaryDiffEq._alg_autodiff(::AbstractEK) = Val{true}()
OrdinaryDiffEq.standardtag(::AbstractEK) = false
OrdinaryDiffEq.concrete_jac(::AbstractEK) = nothing

@inline DiffEqBase.get_tmp_cache(integ, alg::AbstractEK, cache::AbstractODEFilterCache) =
    (cache.tmp, cache.atmp)
OrdinaryDiffEq.isfsal(::AbstractEK) = false

for ALG in [:EK1, :DiagonalEK1]
    @eval OrdinaryDiffEq._alg_autodiff(::$ALG{CS,AD}) where {CS,AD} = Val{AD}()
    @eval OrdinaryDiffEq.alg_difftype(::$ALG{CS,AD,DiffType}) where {CS,AD,DiffType} =
        DiffType
    @eval OrdinaryDiffEq.standardtag(::$ALG{CS,AD,DiffType,ST}) where {CS,AD,DiffType,ST} =
        ST
    @eval OrdinaryDiffEq.concrete_jac(
        ::$ALG{CS,AD,DiffType,ST,CJ},
    ) where {CS,AD,DiffType,ST,CJ} = CJ
    @eval OrdinaryDiffEq.get_chunksize(::$ALG{CS}) where {CS} = Val(CS)
    @eval OrdinaryDiffEq.isimplicit(::$ALG) = true
end

############################################
# Step size control
OrdinaryDiffEq.isadaptive(::AbstractEK) = true
OrdinaryDiffEq.alg_order(alg::AbstractEK) = num_derivatives(alg.prior)
# OrdinaryDiffEq.alg_adaptive_order(alg::AbstractEK) =

# PI control is the default!
OrdinaryDiffEq.isstandard(::AbstractEK) = true # proportional
OrdinaryDiffEq.ispredictive(::AbstractEK) = false # not sure, maybe Gustafsson acceleration?

# OrdinaryDiffEq.qmin_default(alg::AbstractEK) =
# OrdinaryDiffEq.qmax_default(alg::AbstractEK) =
# OrdinaryDiffEq.beta2_default(alg::AbstractEK) = 2 // (5(OrdinaryDiffEq.alg_order(alg) + 1))
# OrdinaryDiffEq.beta1_default(alg::AbstractEK, beta2) = 7 // (10(OrdinaryDiffEq.alg_order(alg) + 1))
# OrdinaryDiffEq.gamma_default(alg::AbstractEK) =

# OrdinaryDiffEq.uses_uprev(alg::, adaptive::Bool) = adaptive
OrdinaryDiffEq.is_mass_matrix_alg(::AbstractEK) = true

SciMLBase.isautodifferentiable(::AbstractEK) = true
SciMLBase.allows_arbitrary_number_types(::AbstractEK) = true
SciMLBase.allowscomplex(::AbstractEK) = false
