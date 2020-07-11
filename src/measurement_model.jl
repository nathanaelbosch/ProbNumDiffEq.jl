########################################################################################
# Measurement Models
########################################################################################
function measurement_model(kind::Symbol, d::Integer, q::Integer, f::Function, p)
    @assert kind in (:ekf0, :ekf1) ("Type of measurement model not in [:ekf0, :ekf1]")
    if kind == :ekf0
        return ekf0_measurement_model(d, q, f, p)
    elseif kind == :ekf1
        return ekf1_measurement_model(d, q, f, p)
    end
end
measurement_model(kind, d, q, ivp) = measurement_model(kind, d, q, ivp.f, ivp.p)

function ekf0_measurement_model(d::Integer, q::Integer, f::Function, p)
    H_0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    H_1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    R = zeros(d, d)

    h(m, t) = H_1*m - f(H_0*m, p, t)
    H(m, t) = H_1
    return (h=h, H=H, R=R)
end

function ekf1_measurement_model(d::Integer, q::Integer, f::Function, p)
    H_0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    H_1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    R = zeros(d, d)

    h(m, t) = H_1*m - f(H_0*m, p, t)
    Jf = (hasfield(typeof(f), :jac) && !isnothing(f.jac)) ? f.jac :
        (u, p, t) -> ForwardDiff.jacobian(_u -> f(_u, p, t), u)
    H(m, t) = H_1 - Jf(H_0*m, p, t) * H_0

    return (h=h, H=H, R=R)
end
