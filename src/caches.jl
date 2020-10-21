########################################################################################
# Caches
########################################################################################
abstract type ProbNumODECache <: DiffEqBase.DECache end
mutable struct GaussianODEFilterCache{RType, EType, F1, F2, uType, xType, matType, diffusionType} <: ProbNumODECache
    # Constants
    d::Int                  # Dimension of the problem
    q::Int                  # Order of the prior
    A!
    Q!
    R::RType
    E0::EType
    E1::EType
    method::Symbol
    Precond::F1
    InvPrecond::F2
    # Mutable stuff
    u::uType
    u_pred::uType
    u_filt::uType
    u_tmp::uType
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
end

function GaussianODEFilterCache(
    alg::ODEFilter, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, IIP,
    q, prior, method, initdiff, initialize_derivatives=true)

    u0 = u
    t0 = t
    d = length(u)

    # Projections
    E0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    E1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))

    # Prior dynamics
    @assert prior == :ibm
    Precond, InvPrecond = preconditioner(d, q)
    A!, Q! = ibm(d, q)

    # Measurement model
    @assert method in (:ekf0, :ekf1) ("Type of measurement model not in [:ekf0, :ekf1]")
    R = zeros(d, d)

    uType = typeof(u0)
    uElType = eltype(u0)
    matType = Matrix{uElType}

    # Initial states
    m0, P0 = initialize_derivatives ?
        initialize_with_derivatives(u0, f, p, t0, q) :
        initialize_without_derivatives(u0, f, p, t0, q)
    x0 = Gaussian(m0, P0)

    # Pre-allocate a bunch of matrices
    Ah_empty = diagm(0=>ones(uElType, d*(q+1)))
    Qh_empty = zeros(uElType, d*(q+1), d*(q+1))
    h = E1 * x0.μ
    H = copy(E1)
    du = copy(h)
    ddu = uElType.(zeros(d, d))
    v, S = copy(h), copy(ddu)
    measurement = Gaussian(v, S)
    K = copy(H')


    return GaussianODEFilterCache{
        typeof(R), typeof(E0), typeof(Precond), typeof(InvPrecond),
        uType, typeof(x0), matType, typeof(initdiff),
    }(
        # Constants
        d, q, A!, Q!, R, E0, E1, method, Precond, InvPrecond,
        # Mutable stuff
        copy(u0), copy(u0), copy(u0), copy(u0),
        copy(x0), copy(x0), copy(x0), copy(x0),
        measurement,
        Ah_empty, Qh_empty, H, du, ddu, K, initdiff,
        copy(u0),
    )

end
