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

    if !IIP
        h_oop(m, t) = E_1*m - f(E_0*m, p, t)
        H_oop(m, t) = E_1
        return (h=h_oop, H=H_oop, R=R)
    else
        function h_iip(m, t) 
            du = E_1*m
            f(du, E_0*m, p, t)
            return E_1*m - du
        end
        H_iip(m, t) = E_1 
        return (h=h_iip, H=H_iip, R=R)
    end
end

function ekf1_measurement_model(d::Integer, q::Integer, f::Function, p, IIP::Bool)
    E_0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    E_1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    R = zeros(d, d)

    if !IIP
        h_oop(m, t) = E_1*m - f(E_0*m, p, t)
        Jf = (hasfield(typeof(f), :jac) && !isnothing(f.jac)) ? f.jac :
            (u, p, t) -> ForwardDiff.jacobian(_u -> f(_u, p, t), u)
        H_oop(m, t) = E_1 - Jf(E_0*m, p, t) * E_0
        return (h=h_oop, H=H_oop, R=R)
    else
        function h_iip(m, t) 
            du = E_1*m
            f(du, E_0*m, p, t)
            return E_1*m - du
        end
        Jf = (hasfield(typeof(f), :jac) && !isnothing(f.jac)) ? f.jac :
            (u, p, t) -> ForwardDiff.jacobian(_u -> f(_u, p, t), u)
        H_iip(m, t) = E_1 - Jf(E_0*m, p, t) * E_0
        return (h=h_iip, H=H_iip, R=R)
    end
end
