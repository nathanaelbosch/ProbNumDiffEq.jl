########################################################################################
# Basic Kalman Filtering
########################################################################################
function kf_predict(m::Vector, P::Matrix, A::Matrix, Q::Matrix)
    return (m=(A*m), P=(A*P*A' + Q))
end

function kf_update(m::Vector, P::Matrix, A::Matrix, Q::Matrix, H::Matrix, R::Matrix, y::Vector)
    v = y - H*m
    S = H * P * H' + R
    K = P * H' * inv(S)
    return (m=(m + K*v), P=(P - K*S*K'))
end

function kf_smooth(m_f_t::Vector, P_f_t::Matrix,
                   m_p_t1::Vector, P_p_t1::Matrix,
                   m_s_t1::Vector, P_s_t1::Matrix,
                   A::Matrix, Q::Matrix)
    G = P_f_t * A' * inv(P_p_t1)
    m = m_f_t + G * (m_s_t1 - m_p_t1)
    P = P_f_t + Symmetric(G * (P_s_t1 - P_p_t1) * G')

    # Sanity: Make sure that the diagonal of P is non-negative
    _min = minimum(diag(P))
    if _min < 0
        @assert abs(_min) < 1e-16
        P += - _min*I
    end
    @assert all(diag(P) .>= 0)

    return m, P
end


function ekf_predict(m::Vector, P::Matrix, f::Function, F::Function, Q::Matrix)
    return (m=f(m), P=(F(m)*P*F(m)' + Q))
end

function ekf_update(m::Vector, P::Matrix, h::Function, H::Function, R::Matrix, y::Vector)
    v = y - h(m)
    S = H(m) * P * H(m)' + R
    K = P * H(m)' * inv(S)
    @show K*S*K'
    return (m=(m + K*v), P=(P - K*S*K'))
end

function ekf_smooth(m_f_t::Vector, P_f_t::Matrix,
                    m_p_t1::Vector, P_p_t1::Matrix,
                    m_s_t1::Vector, P_s_t1::Matrix,
                    F::Function, Q::Matrix)
    G = P_f_t * F(m_f_t)' * inv(P_p_t1)
    m = m_f_t + G * (m_s_t1 - m_p_t1)
    P = P_f_t + G * (P_s_t1 - P_p_t1) * G'
    return m, P
end
