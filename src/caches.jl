########################################################################################
# Caches
########################################################################################
mutable struct EKCache{
    RType,CFacType,ProjType,SolProjType,PType,PIType,EType,uType,duType,xType,PriorType,
    AType,QType,
    FType,LType,FHGMethodType,FHGCacheType,
    HType,vecType,dduType,matType,bkType,diffusionType,diffModelType,measModType,measType,
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
    F::FType # Prior SDE drift
    L::LType # Prior SDE dispersion
    FHG_method::FHGMethodType
    FHG_cache::FHGCacheType
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
    ddu::dduType
    K1::matType
    G1::matType
    Smat::HType
    C_d::vecType
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

OrdinaryDiffEqCore.get_fsalfirstlast(cache::AbstractODEFilterCache, rate_prototype) =
    (nothing, nothing)

function OrdinaryDiffEqCore.alg_cache(
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
    verbose,
) where {IIP,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits}
    if u isa Number
        error("We currently don't support scalar-valued problems")
    end

    is_secondorder_ode = f isa DynamicalODEFunction

    q = num_derivatives(alg.prior)
    d = is_secondorder_ode ? length(u.x[1]) : length(u)
    D = d * (q + 1)

    uType = typeof(u)
    # uElType = eltype(u_vec)
    uElType = uBottomEltypeNoUnits

    FAC = alg.covariance_factorization{uElType}(d, q)
    if FAC isa IsometricKroneckerCovariance && !(f.mass_matrix isa UniformScaling)
        throw(
            ArgumentError(
                "The selected algorithm uses an efficient Kronecker-factorized " *
                "implementation which is incompatible with the provided mass matrix. " *
                "Try using the `EK1` instead."),
        )
    end

    matType = typeof(factorized_similar(FAC, d, d))

    # Projections
    Proj = projection(FAC)
    E0, E1, E2 = Proj(0), Proj(1), Proj(2)
    @assert f isa SciMLBase.AbstractODEFunction
    SolProj = is_secondorder_ode ? [E1; E0] : copy(E0)

    # Prior dynamics
    if !(dim(alg.prior) == 1 || dim(alg.prior) == d)
        throw(
            DimensionMismatch(
                "The dimension of the prior is not compatible with the dimension " *
                "of the problem! The given ODE is $(d)-dimensional, but the prior is " *
                "$(dim(alg.prior))-dimensional. Please make sure that the dimension of " *
                "the prior is either 1 or $(d)."),
        )
    end
    prior = remake(alg.prior; elType=uElType, dim=d)
    if (prior isa IOUP) && prior.update_rate_parameter
        if !(prior.rate_parameter isa Missing)
            throw(
                ArgumentError(
                    "Do not manually set the `rate_parameter` of the IOUP prior when " *
                    "using the `update_rate_parameter=true` option." *
                    "Reset the prior and try again."),
            )
        end
        prior = remake(prior; rate_parameter=Array{uElType}(calloc, d, d))
    end

    A, Q, Ah, Qh, P, PI = initialize_transition_matrices(FAC, prior, dt)
    F, L = to_sde(prior)
    F, L = to_factorized_matrix(FAC, F), to_factorized_matrix(FAC, L)
    FHG_method, FHG_cache = if !(prior isa IWP)
        m = FiniteHorizonGramians.ExpAndGram{eltype(F),13}()
        c = FiniteHorizonGramians.alloc_mem(F, L, m)
        m, c
    else
        nothing, nothing
    end

    # Measurement Model
    measurement_model = make_measurement_model(f)

    # Initial State
    x0 = initial_distribution(prior)
    x0 = Gaussian(x0.μ, to_factorized_matrix(FAC, x0.Σ))

    # Diffusion Model
    diffmodel = alg.diffusionmodel
    initdiff = initial_diffusion(diffmodel, d, q, uEltypeNoUnits)
    apply_diffusion!(x0.Σ, initdiff)

    # Measurement model related things
    R =
        isnothing(alg.pn_observation_noise) ? nothing :
        to_factorized_matrix(FAC, cov2psdmatrix(alg.pn_observation_noise; d))
    H = factorized_similar(FAC, d, D)
    v = similar(Array{uElType}, d)
    S = factorized_zeros(FAC, d, d)
    measurement = Gaussian(v, S)

    # Caches
    du = is_secondorder_ode ? similar(u.x[2]) : similar(u)
    ddu =
        !isnothing(f.jac_prototype) ?
        f.jac_prototype : zeros(uElType, length(u), length(u))
    _d = is_secondorder_ode ? 2d : d
    pu_tmp = Gaussian(
        similar(Array{uElType}, _d),
        PSDMatrix(factorized_similar(FAC, D, _d)),
    )

    K = factorized_similar(FAC, D, d)
    G = factorized_similar(FAC, D, D)
    Smat = factorized_similar(FAC, d, d)

    C_d = similar(Array{uElType}, d)
    C_dxd = factorized_similar(FAC, d, d)
    C_dxD = factorized_similar(FAC, d, D)
    C_Dxd = factorized_similar(FAC, D, d)
    C_DxD = factorized_similar(FAC, D, D)
    C_2DxD = factorized_similar(FAC, 2D, D)
    C_3DxD = factorized_similar(FAC, 3D, D)

    backward_kernel = AffineNormalKernel(
        factorized_similar(FAC, D, D),
        similar(Vector{uElType}, D),
        PSDMatrix(factorized_similar(FAC, 2D, D)),
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
    if OrdinaryDiffEqCore.isimplicit(alg)
        jac_config = OrdinaryDiffEqDifferentiation.build_jac_config(
            alg, f, uf, du1, uprev, u, tmp, dw1,
        )
    else
        jac_config = nothing
    end

    ll = zero(uEltypeNoUnits)
    return EKCache{
        typeof(R),typeof(FAC),typeof(Proj),typeof(SolProj),typeof(P),typeof(PI),typeof(E0),
        uType,typeof(du),typeof(x0),typeof(prior),typeof(A),typeof(Q),
        typeof(F),typeof(L),typeof(FHG_method),typeof(FHG_cache),
        typeof(H),typeof(C_d),typeof(ddu),matType,
        typeof(backward_kernel),typeof(initdiff),
        typeof(diffmodel),typeof(measurement_model),typeof(measurement),typeof(pu_tmp),
        uEltypeNoUnits,typeof(dt),typeof(du1),typeof(uf),typeof(jac_config),typeof(atmp),
    }(
        d, q, FAC, prior, A, Q, Ah, Qh, F, L, FHG_method, FHG_cache, diffmodel,
        measurement_model, R, Proj, SolProj,
        P, PI, E0, E1, E2,
        u, u_pred, u_filt, tmp, atmp,
        x0, xprev, x_pred, x_filt, x_tmp, x_tmp2,
        measurement, m_tmp, pu_tmp,
        H, du, ddu, K, G, Smat,
        C_d, C_dxd, C_dxD, C_Dxd, C_DxD, C_2DxD, C_3DxD,
        backward_kernel,
        initdiff, initdiff * NaN, initdiff * NaN,
        err_tmp, ll, dt, du1, uf, jac_config,
    )
end

get_uf(f, t, p, ::Val{true}) = OrdinaryDiffEqDifferentiation.UJacobianWrapper(f, t, p)
get_uf(f, t, p, ::Val{false}) = OrdinaryDiffEqDifferentiation.UDerivativeWrapper(f, t, p)
