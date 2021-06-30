########################################################################################
# Caches
########################################################################################
abstract type ODEFiltersCache <: OrdinaryDiffEq.OrdinaryDiffEqCache end
mutable struct GaussianODEFilterCache{
    RType, ProjType, SolProjType, FP, uType, xType, AType, QType, matType, diffusionType, diffModelType,
    measType, llType,
} <: ODEFiltersCache
    # Constants
    d::Int                  # Dimension of the problem
    q::Int                  # Order of the prior
    A::AType
    Q::QType
    diffusionmodel::diffModelType
    R::RType
    Proj::ProjType
    SolProj::SolProjType
    Precond::FP
    # Mutable stuff
    u::uType
    u_pred::uType
    u_filt::uType
    tmp::uType
    x::xType
    x_pred::xType
    x_filt::xType
    x_tmp::xType
    x_tmp2::xType
    measurement::measType
    H::matType
    du::uType
    ddu::matType
    K::matType
    G::matType
    covmatcache::matType
    local_diffusion::diffusionType
    global_diffusion::diffusionType
    err_tmp::uType
    log_likelihood::llType
end

function OrdinaryDiffEq.alg_cache(
    alg::GaussianODEFilter, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, IIP)
    initialize_derivatives=true

    if !(u isa AbstractVector)
        error("Problems which are not scalar- or vector-valued (e.g. u0 is a scalar ",
              "or a matrix) are currently not supported")
    end

    q = alg.order
    d = length(u)
    D = d*(q+1)

    u0 = u
    t0 = t

    uType = typeof(u0)
    uElType = eltype(u0)
    matType = Matrix{uElType}

    # Projections
    Proj(deriv) = deriv > q ? error("Projection called for non-modeled derivative") :
        kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    @assert f isa AbstractODEFunction
    SolProj = f isa DynamicalODEFunction ? [Proj(0); Proj(1)] : Proj(0)

    # Prior dynamics
    @assert alg.prior == :ibm "Only the ibm prior is implemented so far"
    Precond = preconditioner(d, q)
    A, Q = ibm(d, q, uElType)

    x0 = Gaussian(zeros(uElType, D), SRMatrix(Matrix(uElType(1.0)*I, D, D)))

    # Measurement model
    R = zeros(uElType, d, d)

    # Pre-allocate a bunch of matrices
    h = zeros(uElType, d)
    H = zeros(uElType, d, D)
    du = zeros(uElType, d)
    ddu = zeros(uElType, d, d)
    v, S = copy(h), copy(ddu)
    measurement = Gaussian(v, S)
    K = zeros(uElType, D, d)
    G = zeros(uElType, D, D)
    covmatcache = copy(G)

    diffusion_models = Dict(
        :dynamic => DynamicDiffusion(),
        :dynamicMV => MVDynamicDiffusion(),
        :fixed => FixedDiffusion(),
        :fixedMV => MVFixedDiffusion(),
        :fixedMAP => MAPFixedDiffusion(),
    )
    diffmodel = diffusion_models[alg.diffusionmodel]
    initdiff = initial_diffusion(diffmodel, d, q, uEltypeNoUnits)

    return GaussianODEFilterCache{
        typeof(R), typeof(Proj), typeof(SolProj), typeof(Precond),
        uType, typeof(x0), typeof(A), typeof(Q), matType, typeof(initdiff),
        typeof(diffmodel), typeof(measurement), uEltypeNoUnits,
    }(
        # Constants
        d, q, A, Q, diffmodel, R, Proj, SolProj, Precond,
        # Mutable stuff
        copy(u0), copy(u0), copy(u0), copy(u0),
        copy(x0), copy(x0), copy(x0), copy(x0), copy(x0),
        measurement,
        H, du, ddu, K, G, covmatcache, initdiff, initdiff,
        copy(u0),
        zero(uEltypeNoUnits),
    )
end
