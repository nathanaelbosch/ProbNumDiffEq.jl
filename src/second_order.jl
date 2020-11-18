Base.@kwdef struct SecondOrderEKF0 <: AbstractEKF
    prior::Symbol = :ibm
    order::Int = 1
    diffusionmodel::Symbol = :dynamic
    smooth::Bool = true
end


function OrdinaryDiffEq.alg_cache(
    alg::SecondOrderEKF0, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, IIP)
    initialize_derivatives=true

    if length(u) == 1 && size(u) == ()
        error("Scalar-values problems are currently not supported. Please remake it with a
               1-dim Array instead")
    end

    if (alg isa EKF1 || alg isa IEKS) && isnothing(f.jac)
        error("""EKF1 requires the Jacobian. To automatically generate it with ModelingToolkit.jl use ODEFilters.remake_prob_with_jac(prob).""")
    end

    _du, _u = u[1,:], u[2,:]

    q = alg.order+1
    u0, du0 = _u, _du
    t0 = t
    d = length(_u)

    uType = typeof(u0)
    uElType = eltype(u0)
    matType = Matrix{uElType}

    # Projections
    Proj(deriv) = kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    SolProj = [Proj(1); Proj(0)]

    # Prior dynamics
    @assert alg.prior == :ibm "Only the ibm prior is implemented so far"
    Precond, InvPrecond = preconditioner(d, q)
    A, Q = ibm(d, q, uElType)

    # Measurement model
    R = PSDMatrix(LowerTriangular(zeros(d, d)))
    # Initial states
    m0, P0 = initialize_derivatives ?
        initialize_with_derivatives(u, f, p, t0, q-1) :
        initialize_without_derivatives(u, f, p, t0, q-1)
    @assert iszero(P0)
    m0 = [m0[d+1:2d];
          vcat([m0[(i*2d+1):(i*2d+d)] for i in 0:q-1]...)]
    P0 = zeros(uElType, d*(q+1), d*(q+1))
    P0 = PSDMatrix(LowerTriangular(zero(P0)))
    x0 = Gaussian(m0, P0)

    # Pre-allocate a bunch of matrices
    h = Proj(0) * x0.μ
    H = copy(Proj(1))
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
        typeof(R), typeof(Proj), typeof(SolProj), typeof(Precond), typeof(InvPrecond),
        uType, typeof(x0), matType, typeof(Q), typeof(initdiff),
        typeof(diffmodel),
    }(
        # Constants
        d, q, A, Q, diffmodel, R, Proj, SolProj, Precond, InvPrecond,
        # Mutable stuff
        copy(u), copy(u), copy(u), copy(u),
        copy(x0), copy(x0), copy(x0), copy(x0),
        measurement,
        H, du, ddu, K, initdiff,
        copy(u0),
        0
    )
end


function h!(integ::OrdinaryDiffEq.ODEIntegrator{SecondOrderEKF0}, x_pred, t)
    @unpack f, p, dt = integ
    @unpack d, q, du, Proj, SolProj, InvPrecond, measurement = integ.cache

    E0, E1, E2 = Proj(0), Proj(1), Proj(2)

    PI = InvPrecond(dt)
    z = measurement.μ

    u_pred = E0*PI*x_pred.μ
    du_pred = E1*PI*x_pred.μ
    IIP = isinplace(integ.f)
    if IIP
        ddu = copy(u_pred)
        f.f1(ddu, du_pred, u_pred, p, t)
    else
        error("TODO")
        du .= f(u_pred, p, t)
    end
    integ.destats.nf += 1

    z .= E2*PI*x_pred.μ .- ddu

    return z
end


function H!(integ::OrdinaryDiffEq.ODEIntegrator{SecondOrderEKF0}, x_pred, t)
    @unpack f, p, dt, alg = integ
    @unpack q, d, ddu, InvPrecond, H, Proj = integ.cache
    E2 = Proj(2)
    PI = InvPrecond(dt)
    H .= E2 * PI
    return H
end


function GaussianODEFilterPosterior(alg::SecondOrderEKF0, u0)
    uElType = eltype(u0)
    d = length(u0.x[1])
    q = alg.order+1
    Proj(deriv) = kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    SolProj = [Proj(1); Proj(0)]
    A, Q = ibm(d, q, uElType)
    Precond, InvPrecond = preconditioner(d, q)
    GaussianODEFilterPosterior(
        d, q, SolProj, A, Q, Precond, InvPrecond, alg.smooth)
end
