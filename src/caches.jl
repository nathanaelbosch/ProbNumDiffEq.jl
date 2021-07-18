########################################################################################
# Caches
########################################################################################
abstract type ODEFiltersCache <: OrdinaryDiffEq.OrdinaryDiffEqCache end
mutable struct GaussianODEFilterCache{
    RType, ProjType, SolProjType, FP, uType, duType, xType, AType, QType, matType, diffusionType, diffModelType,
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
    du::duType
    ddu::matType
    K::matType
    G::matType
    covmatcache::matType
    local_diffusion::diffusionType
    global_diffusion::diffusionType
    err_tmp::duType
    log_likelihood::llType
end

function OrdinaryDiffEq.alg_cache(
    alg::GaussianODEFilter, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, IIP)
    initialize_derivatives=true

    if !(u isa AbstractVector)
        error("Problems which are not scalar- or vector-valued (e.g. u0 is a scalar ",
              "or a matrix) are currently not supported")
    end

    is_secondorder_ode = f isa DynamicalODEFunction
    if is_secondorder_ode
        @warn "Assuming that the given ODE is a SecondOrderODE. If this is not the case, e.g. because it is some other dynamical ODE, the solver will probably run into errors!"
    end

    q = alg.order
    d = is_secondorder_ode ? length(u[1, :]) : length(u)
    D = d*(q+1)

    u0 = u
    t0 = t

    uType = typeof(u0)
    uElType = eltype(u0)
    matType = Matrix{uElType}

    # Projections
    Proj(deriv) = deriv > q ? error("Projection called for non-modeled derivative") :
        kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(uElType, d)))
    @assert f isa AbstractODEFunction
    SolProj = f isa DynamicalODEFunction ? [Proj(1); Proj(0)] : Proj(0)

    # Prior dynamics
    @assert alg.prior == :ibm "Only the ibm prior is implemented so far"
    Precond = preconditioner(uElType, d, q)
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
        uType, typeof(du), typeof(x0), typeof(A), typeof(Q), matType, typeof(initdiff),
        typeof(diffmodel), typeof(measurement), uEltypeNoUnits,
    }(
        # Constants
        d, q, A, Q, diffmodel, R, Proj, SolProj, Precond,
        # Mutable stuff
        copy(u0), copy(u0), copy(u0), copy(u0),
        copy(x0), copy(x0), copy(x0), copy(x0), copy(x0),
        measurement,
        H, du, ddu, K, G, covmatcache, initdiff, initdiff,
        copy(du),
        zero(uEltypeNoUnits),
    )
end
