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
    rateType,
    UF,
    JC,
    uNoUnitsType,
} <: ODEFiltersCache
    # Constants
    d_y::Int                  # State dimension
    d_z::Int                  # Measurement dimension
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
    atmp::uNoUnitsType
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
    default_diffusion::diffusionType
    local_diffusion::diffusionType
    global_diffusion::diffusionType
    err_tmp::duType
    log_likelihood::llType
    C1::CType
    C2::CType
    du1::rateType
    uf::UF
    jac_config::JC
end

function OrdinaryDiffEq.alg_cache(
    alg::GaussianODEFilter,
    u,
    rate_prototype,
    ::Type{uEltypeNoUnits},
    ::Type{uBottomEltypeNoUnits},
    ::Type{tTypeNoUnits},
    uprev,
    uprev2,
    m,
    t,
    dt,
    reltol,
    p,
    calck,
    ::Val{IIP},
) where {IIP,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits}
    initialize_derivatives = true

    if u isa Number
        error("We currently don't support scalar-valued problems")
    end

    @assert m isa SemiLinearModel
    f = m.b

    q = alg.order
    d_y = length(u)
    d_z = size(m.A, 1)
    D = d_y * (q + 1)
    @assert (d_z, D) == size(m.A)
    @assert (d_y, D) == size(m.C)

    u_vec = u[:]
    t0 = t

    uType = typeof(u)
    # uElType = eltype(u_vec)
    uElType = uBottomEltypeNoUnits
    matType = Matrix{uElType}

    # Projections
    Proj = projection(d_y, q, uElType)
    E0, E1, E2 = Proj(0), Proj(1), Proj(2)
    SolProj = Proj(0)

    # Prior dynamics
    P, PI = init_preconditioner(d_y, q, uElType)

    A, Q = ibm(d_y, q, uElType)

    initial_variance = ones(uElType, D)
    x0 = Gaussian(
        zeros(uElType, D),
        SRMatrix(diagm(sqrt.(initial_variance)), diagm(initial_variance)),
    )

    # Measurement model
    R = zeros(uElType, d_z, d_z)

    # Pre-allocate a bunch of matrices
    h = zeros(uElType, d_z)
    H = copy(m.A)
    du = similar(h)
    ddu = zeros(uElType, d_z, d_y)
    v = similar(h)
    S = SRMatrix(zeros(uElType, d_z, D), zeros(uElType, d_z, d_z))
    measurement = Gaussian(v, S)
    pu_tmp = Gaussian(zeros(uElType, d_y), SRMatrix(zeros(uElType, d_y, D), zeros(uElType, d_y, d_y)))
    K = zeros(uElType, D, d_z)
    G = zeros(uElType, D, D)
    C1 = SRMatrix(zeros(uElType, D, 2D), zeros(uElType, D, D))
    C2 = SRMatrix(zeros(uElType, D, 3D), zeros(uElType, D, D))
    covmatcache = similar(G)

    if alg isa EK1FDB
        error("Not supported right now")
        H = [E1; E2]
        v = [v; v]
        S = SRMatrix(zeros(uElType, 2d, D), zeros(uElType, 2d, 2d))
        measurement = Gaussian(v, S)
        K = zeros(uElType, D, 2d)
    end

    diffmodel = alg.diffusionmodel
    initdiff = initial_diffusion(diffmodel, d_y, q, uEltypeNoUnits)
    copy!(x0.Σ, apply_diffusion(x0.Σ, initdiff))

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
    err_tmp = similar(u)

    # Things for calc_J
    uf = get_uf(f, t, p, Val(IIP))
    du1 = similar(rate_prototype)
    dw1 = zero(u)
    atmp = similar(u, uEltypeNoUnits)
    if OrdinaryDiffEq.isimplicit(alg)
        jac_config = OrdinaryDiffEq.build_jac_config(alg, f, uf, du1, uprev, u, tmp, dw1)
    else
        jac_config = nothing
    end

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
        typeof(du1),
        typeof(uf),
        typeof(jac_config),
        typeof(atmp),
    }(
        # Constants
        d_y,
        d_z,
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
        atmp,
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
        initdiff * NaN,
        initdiff * NaN,
        err_tmp,
        zero(uEltypeNoUnits),
        C1,
        C2,
        du1,
        uf,
        jac_config,
    )
end

get_uf(f, t, p, ::Val{true}) = OrdinaryDiffEq.UJacobianWrapper(f, t, p)
get_uf(f, t, p, ::Val{false}) = OrdinaryDiffEq.UDerivativeWrapper(f, t, p)
