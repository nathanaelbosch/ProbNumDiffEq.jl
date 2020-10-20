########################################################################################
# Caches
########################################################################################
abstract type ProbNumODECache <: DiffEqBase.DECache end
mutable struct GaussianODEFilterCache{RType, EType, F1, F2, uType, xType, matType, sigmaType} <: ProbNumODECache
    # Constants
    d::Int                  # Dimension of the problem
    q::Int                  # Order of the prior
    # dm                    # Dynamics Model
    # mm                    # Measurement Model
    A!
    Q!
    h!
    H!
    R::RType
    # Precond
    # Precond_inv
    E0::EType
    E1::EType
    jac
    Precond::F1
    InvPrecond::F2
    # Mutable stuff
    u::uType
    u_pred::uType
    u_filt::uType
    u_tmp::uType
    # tmp
    x::xType
    x_pred::xType
    x_filt::xType
    x_tmp::xType
    measurement
    Ah::matType
    Qh::matType
    h::uType
    H::matType
    du::uType
    ddu::matType
    K::matType
    σ_sq::sigmaType
    σ_sq_prev::sigmaType
    P_tmp::matType
    err_tmp::uType

end

function GaussianODEFilterCache(d, q, prob, prior, method, σ0, initialize_derivatives=true)

    d, f = length(prob.u0), prob.f
    @assert prior == :ibm
    Precond, InvPrecond = preconditioner(d, q)
    A!, Q! = ibm(d, q)

    E0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    # E0 = E0 * InvPrecond
    E1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    # E1 = E1 * InvPrecond

    @assert method in (:ekf0, :ekf1) ("Type of measurement model not in [:ekf0, :ekf1]")
    jac = method == :ekf1 ? f.jac : nothing
    h!(h, du, m) = h .= E1*m - du
    H!(H, ddu) = H .= E1 - ddu * E0
    # R = diagm(0 => eps(eltype(prob.u0)) .* ones(d))
    R = zeros(d, d)


    u0 = prob.u0

    uType = typeof(u0)
    uElType = eltype(u0)
    matType = Matrix{uElType}

    t0 = prob.tspan[1]
    # Initial states
    m0, P0 = initialize_derivatives ?
        initialize_with_derivatives(prob, q) :
        initialize_without_derivatives(prob, q)
    # P0 += eps(eltype(P0)) * I
    x0 = Gaussian(m0, P0)

    Ah_empty = diagm(0=>ones(uElType, d*(q+1)))
    Qh_empty = zeros(uElType, d*(q+1), d*(q+1))
    h = E1 * x0.μ
    H = uElType.(zeros(d, d*(q+1)))
    du = copy(h)
    ddu = uElType.(zeros(d, d))
    sigma = σ0
    v, S = copy(h), copy(ddu)
    measurement = Gaussian(v, S)
    K = copy(H')

    return GaussianODEFilterCache{
        typeof(R), typeof(E0), typeof(Precond), typeof(InvPrecond),
        uType, typeof(x0), matType, typeof(sigma),
    }(
        # Constants
        d, q, A!, Q!, h!, H!, R, E0, E1, jac, Precond, InvPrecond,
        # Mutable stuff
        copy(u0), copy(u0), copy(u0), copy(u0),
        copy(x0), copy(x0), copy(x0), copy(x0),
        measurement,
        Ah_empty, Qh_empty, h, H, du, ddu, K, sigma, sigma,
        copy(P0),
        copy(u0),
    )

end
