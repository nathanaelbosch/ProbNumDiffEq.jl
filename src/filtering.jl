
function kf_predict(m, P, A, Q)
    return (m=(A*m), P=(A*P*A' + Q))
end

function kf_update(m, P, A, Q, H, R, y)
    v = y - H*m
    S = H * P * H' + R
    K = P * H' * inv(S)
    return (m=(m + K*v), P=(P - K*S*K'))
end

function kf_smooth(m_f_t, P_f_t, m_p_t1, P_p_t1, m_s_t1, P_s_t1, A, Q)
    G = P_f_t * A' * inv(P_p_t1)
    m = m_f_t + G * (m_s_t1 - m_p_t1)
    P = P_f_t + G * (P_s_t1 - P_p_t1) * G'
    _min = minimum(diag(P))
    if _min < 0
        @assert abs(_min) < 1e-16
        P .+= -_min
    end
    @assert all(diag(P) .>= 0)
    return m, P
end


function ekf_predict(m, P, f, F, Q)
    return (m=f(m), P=(F(m)*P*F(m)' + Q))
end

function ekf_update(m, P, h, H, R, y)
    v = y - h(m)
    S = H(m) * P * H(m)' + R
    K = P * H(m)' * inv(S)
    @show K*S*K'
    return (m=(m + K*v), P=(P - K*S*K'))
end

function ekf_smooth(m_f_t, P_f_t, m_p_t1, P_p_t1, m_s_t1, P_s_t1, F, Q)
    G = P_f_t * F(m_f_t)' * inv(P_p_t1)
    m = m_f_t + G * (m_s_t1 - m_p_t1)
    P = P_f_t + G * (P_s_t1 - P_p_t1) * G'
    return m, P
end
