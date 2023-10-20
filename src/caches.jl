########################################################################################
# Caches
########################################################################################
mutable struct EKCache{
    RType,CFacType,ProjType,SolProjType,PType,PIType,EType,uType,duType,xType,PriorType,
    AType,QType,HType,matType,bkType,diffusionType,diffModelType,measModType,measType,
    puType,llType,dtType,rateType,UF,JC,uNoUnitsType,
} <: AbstractODEFilterCache
    # Constants
    d::Int                  # Dimension of the problem
    q::Int                  # Order of the prior
    covariance_factorization::CFacType
    prior::PriorType
    A::AType
    Q::QType
    Ah::AType
    Qh::QType
    diffusionmodel::diffModelType
    measurement_model::measModType
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
    xprev::xType
    x_pred::xType
    x_filt::xType
    x_tmp::xType
    x_tmp2::xType
    measurement::measType
    m_tmp::measType
    pu_tmp::puType
    H::HType
    du::duType
    ddu::matType
    K1::matType
    G1::matType
    Smat::HType
    C_dxd::matType
    C_dxD::matType
    C_Dxd::matType
    C_DxD::matType
    C_2DxD::matType
    C_3DxD::matType
    backward_kernel::bkType
    default_diffusion::diffusionType
    local_diffusion::diffusionType
    global_diffusion::diffusionType
    err_tmp::duType
    log_likelihood::llType
    dt_last::dtType
    du1::rateType
    uf::UF
    jac_config::JC
end

function OrdinaryDiffEq.alg_cache(
    alg::AbstractEK,
    u,
    rate_prototype,
    ::Type{uEltypeNoUnits},
    ::Type{uBottomEltypeNoUnits},
    ::Type{tTypeNoUnits},
    uprev,
    uprev2,
    f,
    t,
    dt,
    reltol,
    p,
    calck,
    ::Val{IIP},
) where {IIP,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits}
    if u isa Number
        error("We currently don't support scalar-valued problems")
    end

    is_secondorder_ode = f isa DynamicalODEFunction

    q = alg.prior.num_derivatives
    d = is_secondorder_ode ? length(u[1, :]) : length(u)
    D = d * (q + 1)

    FAC = get_covariance_factorization(alg)
    if FAC isa KroneckerCovariance && !(f.mass_matrix isa UniformScaling)
        error(
            "The selected algorithm uses an efficient Kronecker-factorized implementation which is incompatible with the provided mass matrix. Try using the `EK1` instead.",
        )
    end

    uType = typeof(u)
    # uElType = eltype(u_vec)
    uElType = uBottomEltypeNoUnits
    matType = Matrix{uElType}

    # Projections
    Proj = projection(FAC, d, q, uElType)
    E0, E1, E2 = Proj(0), Proj(1), Proj(2)
    @assert f isa SciMLBase.AbstractODEFunction
    SolProj = if is_secondorder_ode
        if E0 isa IKP
            IsoKroneckerProduct(d, [Proj(1).B; Proj(0).B])
        else
            SolProj = [Proj(1); Proj(0)]
        end
    else
        Proj(0)
    end

    # Prior dynamics
    prior = if alg.prior isa IWP
        IWP{uElType}(d, alg.prior.num_derivatives)
    elseif alg.prior isa IOUP && ismissing(alg.prior.rate_parameter)
        r = zeros(uElType, d, d)
        IOUP{uElType}(d, q, r, alg.prior.update_rate_parameter)
    elseif alg.prior isa IOUP
        IOUP{uElType}(d, q, alg.prior.rate_parameter, alg.prior.update_rate_parameter)
    elseif alg.prior isa Matern
        Matern{uElType}(d, q, alg.prior.lengthscale)
    else
        error("Invalid prior $(alg.prior)")
    end
    A, Q, Ah, Qh, P, PI = initialize_transition_matrices(FAC, prior, dt)

    # Measurement Model
    measurement_model = make_measurement_model(f)

    # Initial State
    initial_variance = ones(uElType, q + 1)
    μ0 = zeros(uElType, D)
    Σ0 = PSDMatrix(
        to_factorized_matrix(FAC, IsoKroneckerProduct(d, diagm(sqrt.(initial_variance)))),
    )
    x0 = Gaussian(μ0, Σ0)

    # Diffusion Model
    diffmodel = alg.diffusionmodel
    initdiff = initial_diffusion(diffmodel, d, q, uEltypeNoUnits)
    copy!(x0.Σ, apply_diffusion(x0.Σ, initdiff))

    # Measurement model related things
    R = zeros(uElType, d, d)
    H = factorized_zeros(FAC, uElType, d, D; d, q)
    v = zeros(uElType, d)
    S = PSDMatrix(factorized_zeros(FAC, uElType, D, d; d, q))
    measurement = Gaussian(v, S)

    # Caches
    du = is_secondorder_ode ? similar(u[2, :]) : similar(u)
    ddu = zeros(uElType, length(u), length(u))
    pu_tmp = if !is_secondorder_ode # same dimensions as `measurement`
        copy(measurement)
    else # then `u` has 2d dimensions
        Gaussian(zeros(uElType, 2d), PSDMatrix(factorized_zeros(FAC, uElType, D, 2d; d, q)))
    end
    K = zeros(uElType, D, d)
    G = zeros(uElType, D, D)
    Smat = factorized_zeros(FAC, uElType, d, d; d, q)

    C_dxd = zeros(uElType, d, d)
    C_dxD = zeros(uElType, d, D)
    C_Dxd = zeros(uElType, D, d)
    C_DxD = zeros(uElType, D, D)
    C_2DxD = zeros(uElType, 2D, D)
    C_3DxD = zeros(uElType, 3D, D)

    backward_kernel = AffineNormalKernel(
        factorized_zeros(FAC, uElType, D, D; d, q),
        zeros(uElType, D),
        PSDMatrix(factorized_zeros(FAC, uElType, 2D, D; d, q)),
    )

    u_pred = copy(u)
    u_filt = copy(u)
    tmp = copy(u)
    xprev = copy(x0)
    x_pred = copy(x0)
    x_filt = copy(x0)
    x_tmp = copy(x0)
    x_tmp2 = copy(x0)
    m_tmp = copy(measurement)
    err_tmp = copy(du)

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

    ll = zero(uEltypeNoUnits)
    return EKCache{
        typeof(R),typeof(FAC),typeof(Proj),typeof(SolProj),typeof(P),typeof(PI),typeof(E0),
        uType,typeof(du),typeof(x0),typeof(prior),typeof(A),typeof(Q),typeof(H),matType,
        typeof(backward_kernel),typeof(initdiff),
        typeof(diffmodel),typeof(measurement_model),typeof(measurement),typeof(pu_tmp),
        uEltypeNoUnits,typeof(dt),typeof(du1),typeof(uf),typeof(jac_config),typeof(atmp),
    }(
        d, q, FAC, prior, A, Q, Ah, Qh, diffmodel, measurement_model, R, Proj, SolProj,
        P, PI, E0, E1, E2,
        u, u_pred, u_filt, tmp, atmp,
        x0, xprev, x_pred, x_filt, x_tmp, x_tmp2,
        measurement, m_tmp, pu_tmp,
        H, du, ddu, K, G, Smat,
        C_dxd, C_dxD, C_Dxd, C_DxD, C_2DxD, C_3DxD,
        backward_kernel,
        initdiff, initdiff * NaN, initdiff * NaN,
        err_tmp, ll, dt, du1, uf, jac_config,
    )
end

get_uf(f, t, p, ::Val{true}) = OrdinaryDiffEq.UJacobianWrapper(f, t, p)
get_uf(f, t, p, ::Val{false}) = OrdinaryDiffEq.UDerivativeWrapper(f, t, p)
