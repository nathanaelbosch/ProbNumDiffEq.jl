########################################################################################
# Integrated Brownian Motion
########################################################################################
abstract type AbstractODEFilterPrior end
"""
    IWP(wiener_process_dimension::Integer, num_derivatives::Integer)

Integrated Brownian motion
"""
Base.@kwdef struct IWP{elType} <: AbstractODEFilterPrior
    wiener_process_dimension::Int32
    num_derivatives::Int32
end
IWP(wiener_process_dimension, num_derivatives) =
    IWP{typeof(1.0)}(wiener_process_dimension, num_derivatives)

function preconditioned_discretize_1d(iwp::IWP{elType}) where {elType}
    q = iwp.num_derivatives

    A_breve = binomial.(q:-1:0, (q:-1:0)')
    Q_breve = Cauchy(collect(q:-1.0:0.0), collect((q+1):-1.0:1.0)) |> Matrix  # for Julia1.6

    QR_breve = cholesky(Q_breve).L'
    A_breve, QR_breve = elType.(A_breve), elType.(QR_breve)
    Q_breve = PSDMatrix(QR_breve)

    return A_breve, Q_breve
end

"""
    preconditioned_discretize(iwp::IWP)

Generate the discrete dynamics for a q-times integrated Wiener process (IWP).

The returned matrices `A::AbstractMatrix` and `Q::PSDMatrix` should be used in combination
with the preconditioners; see `./src/preconditioning.jl`.
"""
function preconditioned_discretize(iwp::IWP)
    A_breve, Q_breve = preconditioned_discretize_1d(iwp)
    QR_breve = Q_breve.R

    d = iwp.wiener_process_dimension
    A = kron(I(d), A_breve)
    QR = kron(I(d), QR_breve)
    Q = PSDMatrix(QR)

    return A, Q
end
preconditioned_discretize(iwp::IWP, dt) = begin
    # in the iWP case `dt` is not needed
    preconditioned_discretize(iwp)
end

function discretize_1d(iwp::IWP{elType}, dt::Real) where {elType}
    q = iwp.num_derivatives

    v = 0:q

    f = factorial.(v)
    A_breve = TriangularToeplitz(dt .^ v ./ f, :U) |> Matrix

    e = (2 * q + 1 .- v .- v')
    fr = reverse(f)
    Q_breve = @. dt^e / (e * fr * fr')

    QR_breve = cholesky(Q_breve).L'
    A_breve, QR_breve = elType.(A_breve), elType.(QR_breve)
    Q_breve = PSDMatrix(QR_breve)

    return A_breve, Q_breve
end

function discretize(p::IWP, dt::Real)
    A_breve, Q_breve = discretize_1d(p, dt)
    d = p.wiener_process_dimension
    A = kron(I(d), A_breve)
    QR = kron(I(d), Q_breve.R)
    Q = PSDMatrix(QR)
    return A, Q
end

Base.@kwdef struct IOUP{elType,R} <: AbstractODEFilterPrior
    wiener_process_dimension::Int32
    num_derivatives::Int32
    rate_parameter::R
end
IOUP{T}(wiener_process_dimension, num_derivatives, rate_parameter) where {T} =
    IOUP{T, typeof(rate_parameter)}(wiener_process_dimension, num_derivatives, rate_parameter)

function to_1d_sde(p::IOUP)
    q = p.num_derivatives
    r = p.rate_parameter

    F_breve = diagm(1 => ones(q))
    F_breve[end, end] = r

    L_breve = zeros(q + 1)
    L_breve[end] = 1.0

    return LTISDE(F_breve, L_breve)
end

struct LTISDE{AT<:AbstractMatrix,BT<:AbstractVecOrMat}
    F::AT
    L::BT
end
drift(sde::LTISDE) = sde.F
dispersion(sde::LTISDE) = sde.L

discretize(sde::LTISDE, dt::Real) =
    matrix_fraction_decomposition(drift(sde), dispersion(sde), dt)
function matrix_fraction_decomposition(
    drift::AbstractMatrix,
    dispersion::AbstractVecOrMat,
    dt::Real,
)
    d = size(drift, 1)
    M = [drift dispersion*dispersion'; zero(drift) -drift']
    Mexp = exponential!(dt * M)
    A = Mexp[1:d, 1:d]
    Q = Mexp[1:d, d+1:end] * A'
    # @info "matrix fraction decomposition" A Q
    return A, Q
end
function discretize(p::IOUP, dt::Real)
    A_breve, Q_breve = discretize_1d(p, dt)
    d = p.wiener_process_dimension

    # @info "discretize" p dt
    # QR_breve = cholesky!(Symmetric(Q_breve)).L'
    E = eigen(Q_breve)
    QR_breve = Diagonal(sqrt.(max.(E.values, 0))) * E.vectors'

    A = kron(I(d), A_breve)
    QR = kron(I(d), QR_breve)
    Q = PSDMatrix(QR)
    return A, Q
end

discretize_1d(p::IOUP, dt::Real) = discretize(to_1d_sde(p), dt)

function preconditioned_discretize(p::IOUP, dt::Real)
    A_breve, Q_breve = preconditioned_discretize_1d(p, dt)
    d = p.wiener_process_dimension

    # @info "prec_disc" p dt Q_breve
    # QR_breve = cholesky!(Symmetric(Q_breve)).L'
    E = eigen(Q_breve)
    QR_breve = Diagonal(sqrt.(E.values)) * E.vectors'

    A = kron(I(d), A_breve)
    QR = kron(I(d), QR_breve)
    Q = PSDMatrix(QR)
    return A, Q
end

preconditioned_discretize_1d(p::IOUP, dt::Real) = preconditioned_discretize(to_1d_sde(p), dt)
preconditioned_discretize(sde::LTISDE, dt::Real) = begin
    q = size(drift(sde), 1) - 1
    d = 1
    P, PI = init_preconditioner(1, q)
    make_preconditioner!(P, dt, d, q)
    make_preconditioner_inv!(PI, dt, d, q)
    return matrix_fraction_decomposition(drift(sde), dispersion(sde), dt)
end
