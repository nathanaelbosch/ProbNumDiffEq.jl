########################################################################################
# Caches
########################################################################################
abstract type ODEFiltersCache <: OrdinaryDiffEq.OrdinaryDiffEqCache end
mutable struct GaussianODEFilterCache{
    RType, EType, F1, F2, uType, xType, matType, diffusionType, diffModelType,
} <: ODEFiltersCache
    # Constants
    d::Int                  # Dimension of the problem
    q::Int                  # Order of the prior
    A!
    Q!
    diffusionmodel::diffModelType
    R::RType
    E0::EType
    E1::EType
    Precond::F1
    InvPrecond::F2
    # Mutable stuff
    u::uType
    u_pred::uType
    u_filt::uType
    tmp::uType
    x::xType
    x_pred::xType
    x_filt::xType
    x_tmp::xType
    measurement
    Ah::matType
    Qh::matType
    H::matType
    du::uType
    ddu::matType
    K::matType
    diffmat::diffusionType
    err_tmp::uType
    log_likelihood
end

function OrdinaryDiffEq.alg_cache(
    alg::GaussianODEFilter, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, IIP)
    initialize_derivatives=true

    if length(u) == 1 && size(u) == ()
        error("Scalar-values problems are currently not supported. Please remake it with a
               1-dim Array instead")
    end

    if (alg isa EKF1 || alg isa IEKS) && isnothing(f.jac)
        error("""EKF1 requires the Jacobian. To automatically generate it with ModelingToolkit.jl use ODEFilters.remake_prob_with_jac(prob).""")
    end

    q = alg.order
    u0 = u
    t0 = t
    d = length(u)

    # Projections
    E0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    E1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))

    # Prior dynamics
    @assert alg.prior == :ibm "Only the ibm prior is implemented so far"
    Precond, InvPrecond = preconditioner(d, q)
    A!, Q! = ibm(d, q)

    # Measurement model
    R = PSDMatrix(LowerTriangular(zeros(d, d)))

    uType = typeof(u0)
    uElType = eltype(u0)
    matType = Matrix{uElType}

    # Initial states
    m0, P0 = initialize_derivatives ?
        initialize_with_derivatives(u0, f, p, t0, q) :
        initialize_without_derivatives(u0, f, p, t0, q)
    @assert iszero(P0)
    P0 = PSDMatrix(LowerTriangular(zero(P0)))
    x0 = Gaussian(m0, P0)

    # Pre-allocate a bunch of matrices
    Ah_empty = diagm(0=>ones(uElType, d*(q+1)))
    Qh_empty = zeros(uElType, d*(q+1), d*(q+1))
    h = E1 * x0.μ
    H = copy(E1)
    du = copy(u0)
    ddu = zeros(uElType, d, d)
    v, S = copy(h), copy(ddu)
    measurement = Gaussian(v, S)
    K = copy(H')

    diffusion_models = Dict(
        :dynamic => DynamicDiffusion(),
        :dynamicMV => MVDynamicDiffusion(),
        :fixed => FixedDiffusion(),
        :fixedMV => MVFixedDiffusion(),
        :fixedMAP => MAPFixedDiffusion(),
    )
    diffmodel = diffusion_models[alg.diffusionmodel]
    initdiff = initial_diffusion(diffmodel, d, q)

    return GaussianODEFilterCache{
        typeof(R), typeof(E0), typeof(Precond), typeof(InvPrecond),
        uType, typeof(x0), matType, typeof(initdiff),
        typeof(diffmodel),
    }(
        # Constants
        d, q, A!, Q!, diffmodel, R, E0, E1, Precond, InvPrecond,
        # Mutable stuff
        copy(u0), copy(u0), copy(u0), copy(u0),
        copy(x0), copy(x0), copy(x0), copy(x0),
        measurement,
        Ah_empty, Qh_empty, H, du, ddu, K, initdiff,
        copy(u0),
        0
    )
end
