########################################################################################
# Caches
########################################################################################
abstract type ODEFiltersCache <: OrdinaryDiffEq.OrdinaryDiffEqCache end
mutable struct GaussianODEFilterCache{
    RType,
    ProjType,
    SolProjType,
    PType,
    PIType,
    EType,
    uType,
    duType,
    xType,
    AType,
    QType,
    matType,
    diffusionType,
    diffModelType,
    measType,
    puType,
    llType,
    CType,
} <: ODEFiltersCache
    # Constants
    d::Int                  # Dimension of the problem
    q::Int                  # Order of the prior
    A::AType
    Q::QType
    Ah::AType
    Qh::QType
    diffusionmodel::diffModelType
    R::RType
    Proj::ProjType
    SolProj::SolProjType
    # Also mutable
    P::PType
    PI::PIType
    E0::EType
    E1::EType
    E2::EType
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
    m_tmp::measType
    pu_tmp::puType
    H::matType
    du::duType
    ddu::matType
    K1::matType
    K2::matType
    G1::matType
    G2::matType
    covmatcache::matType
    local_diffusion::diffusionType
    global_diffusion::diffusionType
    err_tmp::duType
    log_likelihood::llType
    C1::CType
    C2::CType
end

function OrdinaryDiffEq.alg_cache(
    alg::GaussianODEFilter,
    u,
    rate_prototype,
    uEltypeNoUnits,
    uBottomEltypeNoUnits,
    tTypeNoUnits,
    uprev,
    uprev2,
    f,
    t,
    dt,
    reltol,
    p,
    calck,
    IIP,
)
    initialize_derivatives = true

    if u isa Number
        error("We currently don't support scalar-valued problems")
    end

    is_secondorder_ode = f isa DynamicalODEFunction
    if is_secondorder_ode
        @warn "Assuming that the given ODE is a SecondOrderODE. If this is not the case, e.g. because it is some other dynamical ODE, the solver will probably run into errors!"
    end

    q = alg.order
    d = is_secondorder_ode ? length(u[1, :]) : length(u)
    D = d * (q + 1)

    u_vec = u[:]
    t0 = t

    uType = typeof(u)
    uElType = eltype(u_vec)
    matType = Matrix{uElType}

    # Projections
    Proj = projection(d, q, uElType)
    E0, E1, E2 = Proj(0), Proj(1), Proj(2)
    @assert f isa AbstractODEFunction
    SolProj = f isa DynamicalODEFunction ? [Proj(1); Proj(0)] : Proj(0)

    # Prior dynamics
    @assert alg.prior == :ibm "Only the ibm prior is implemented so far"
    P, PI = init_preconditioner(d, q, uElType)

    A, Q = ibm(d, q, uElType)

    initial_variance = 1.0 * ones(uElType, D)
    x0 = Gaussian(zeros(uElType, D), SRMatrix(Matrix(Diagonal(sqrt.(initial_variance)))))

    # Measurement model
    R = zeros(uElType, d, d)

    # Pre-allocate a bunch of matrices
    h = zeros(uElType, d)
    H = f isa DynamicalODEFunction ? copy(E2) : copy(E1)
    du = f isa DynamicalODEFunction ? similar(u[2, :]) : similar(u)
    ddu = zeros(uElType, d, d)
    # v, S = similar(h), similar(ddu)
    v = similar(h)
    S =
        alg isa EK0 ? SRMatrix(zeros(uElType, d, D), Diagonal(zeros(uElType, d, d))) :
        SRMatrix(zeros(uElType, d, D), zeros(uElType, d, d))
    measurement = Gaussian(v, S)
    pu_tmp =
        f isa DynamicalODEFunction ?
        Gaussian(zeros(uElType, 2d), SRMatrix(zeros(uElType, 2d, D))) : similar(measurement)
    K = zeros(uElType, D, d)
    G = zeros(uElType, D, D)
    C1 = SRMatrix(zeros(uElType, D, 2D), zeros(uElType, D, D))
    C2 = SRMatrix(zeros(uElType, D, 3D), zeros(uElType, D, D))
    covmatcache = similar(G)

    if alg isa EK1FDB
        H = [E1; E2]
        v = [v; v]
        S = SRMatrix(zeros(uElType, 2d, D), zeros(uElType, 2d, 2d))
        measurement = Gaussian(v, S)
        K = zeros(uElType, D, 2d)
    end

    diffmodel =
        alg.diffusionmodel == :dynamic ? DynamicDiffusion() :
        alg.diffusionmodel == :fixed ? FixedDiffusion() :
        alg.diffusionmodel == :dynamicMV ? MVDynamicDiffusion() :
        alg.diffusionmodel == :fixedMV ? MVFixedDiffusion() :
        error("The specified diffusion could not be recognized! Use e.g. `:dynamic`.")

    initdiff = initial_diffusion(diffmodel, d, q, uEltypeNoUnits)

    Ah, Qh = copy(A), copy(Q)
    u_pred = similar(u)
    u_filt = similar(u)
    tmp = similar(u)
    x_pred = copy(x0)
    x_filt = similar(x0)
    x_tmp = similar(x0)
    x_tmp2 = similar(x0)
    m_tmp = similar(measurement)
    K2 = similar(K)
    G2 = similar(G)
    err_tmp = similar(du)
    return GaussianODEFilterCache{
        typeof(R),
        typeof(Proj),
        typeof(SolProj),
        typeof(P),
        typeof(PI),
        typeof(E0),
        uType,
        typeof(du),
        typeof(x0),
        typeof(A),
        typeof(Q),
        matType,
        typeof(initdiff),
        typeof(diffmodel),
        typeof(measurement),
        typeof(pu_tmp),
        uEltypeNoUnits,
        typeof(C1),
    }(
        # Constants
        d,
        q,
        A,
        Q,
        Ah,
        Qh,
        diffmodel,
        R,
        Proj,
        SolProj,
        P,
        PI,
        E0,
        E1,
        E2,
        # Mutable stuff
        u,
        u_pred,
        u_filt,
        tmp,
        x0,
        x_pred,
        x_filt,
        x_tmp,
        x_tmp2,
        measurement,
        m_tmp,
        pu_tmp,
        H,
        du,
        ddu,
        K,
        K2,
        G,
        G2,
        covmatcache,
        initdiff,
        initdiff,
        err_tmp,
        zero(uEltypeNoUnits),
        C1,
        C2,
    )
end
