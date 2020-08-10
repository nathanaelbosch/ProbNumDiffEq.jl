########################################################################################
# Measurement Models
########################################################################################
function measurement_model(kind::Symbol, d::Integer, q::Integer, f::Function, p, IIP::Bool)
    @assert kind in (:ekf0, :ekf1) ("Type of measurement model not in [:ekf0, :ekf1]")
    if kind == :ekf0
        return ekf0_measurement_model(d, q, f, p, IIP::Bool)
    elseif kind == :ekf1
        return ekf1_measurement_model(d, q, f, p, IIP::Bool)
    end
end
measurement_model(kind, d, q, ivp) = measurement_model(kind, d, q, ivp.f, ivp.p, DiffEqBase.isinplace(ivp))

function ekf0_measurement_model(d::Integer, q::Integer, f::Function, p, IIP::Bool)
    E_0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    E_1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    R = zeros(d, d)

    h = IIP ? h_iip(f, p, d, q) : h_oop(f, p, d, q)
    H(m, t) = E_1
    return (h=h, H=H, R=R)
end
function ekf1_measurement_model(d::Integer, q::Integer, f::Function, p, IIP::Bool)
    E_0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    E_1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    R = zeros(d, d)

    h = IIP ? h_iip(f, p, d, q) : h_oop(f, p, d, q)
    Jf = (hasfield(typeof(f), :jac) && !isnothing(f.jac)) ? f.jac :
        (u, p, t) -> ForwardDiff.jacobian(_u -> f(_u, p, t), u)
    H(m, t) = E_1 - Jf(E_0*m, p, t) * E_0
    return (h=h, H=H, R=R)
end

function h_oop(f, p, d, q) 
    E_0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    E_1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    return h(m, t) = E_1*m - f(E_0*m, p, t)
end

function h_iip(f, p, d, q)
    E_0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    E_1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    function h(m, t)
        du = E_1*m
        f(du, E_0*m, p, t)
        return E_1*m - du
    end
    return h
end

