############################################################################################
# For the equivalent parts in OrdinaryDiffEq.jl, see:
# https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/src/alg_utils.jl
############################################################################################

OrdinaryDiffEq.alg_autodiff(::EK1{CS,AD}) where {CS,AD} = AD
OrdinaryDiffEq.alg_autodiff(::EK0{CS,AD}) where {CS,AD} = AD
OrdinaryDiffEq.alg_difftype(::EK1{CS,AD,DiffType}) where {CS,AD,DiffType} = DiffType
OrdinaryDiffEq.alg_difftype(::EK0{CS,AD,DiffType}) where {CS,AD,DiffType} = DiffType
OrdinaryDiffEq.standardtag(::EK1{CS,AD,DiffType,ST}) where {CS,AD,DiffType,ST} = ST
OrdinaryDiffEq.standardtag(::EK0{CS,AD,DiffType,ST}) where {CS,AD,DiffType,ST} = ST
OrdinaryDiffEq.concrete_jac(::EK1{CS,AD,DiffType,ST,CJ}) where {CS,AD,DiffType,ST,CJ} = CJ
OrdinaryDiffEq.concrete_jac(::EK0{CS,AD,DiffType,ST,CJ}) where {CS,AD,DiffType,ST,CJ} = CJ

@inline DiffEqBase.get_tmp_cache(integ, alg::AbstractEK, cache::AbstractODEFilterCache) =
    (cache.tmp, cache.atmp)
OrdinaryDiffEq.get_chunksize(::EK1{CS}) where {CS} = Val(CS)
OrdinaryDiffEq.get_chunksize(::EK0{CS}) where {CS} = Val(CS)
OrdinaryDiffEq.isfsal(::AbstractEK) = false

OrdinaryDiffEq.isimplicit(::EK1) = true

############################################
# Step size control
OrdinaryDiffEq.isadaptive(::AbstractEK) = true
OrdinaryDiffEq.alg_order(alg::AbstractEK) = alg.prior.num_derivatives
# OrdinaryDiffEq.alg_adaptive_order(alg::AbstractEK) =

# PI control is the default!
OrdinaryDiffEq.isstandard(::AbstractEK) = false # proportional
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
