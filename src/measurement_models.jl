abstract type AbstractMeasurementModel end

struct StandardODEMeasurementModel{IIP,F} <: AbstractMeasurementModel
    f::F
end
StandardODEMeasurementModel(f::SciMLBase.AbstractODEFunction) =
    StandardODEMeasurementModel{isinplace(f),typeof(f)}(f)

function (m::StandardODEMeasurementModel{true})(z, x, p, t)
    d = length(z)
    D = length(x)
    q = D ÷ d - 1
    E0_x = @view x[1:(q+1):end]
    E1_x = @view x[2:(q+1):end]
    m.f(z, E0_x, p, t)
    mul!(z, m.f.mass_matrix, E1_x, 1, -1)
    return z
end
# function (m::StandardODEMeasurementModel{false})(x, p, t)
#     d = length(x) ÷ 2
#     E0x = @view x[1:d]
#     E1x = @view x[d+1:2d]
#     du = m.f(E0x, p, t)
#     return m.f.mass_matrix * E1x - du
# end

struct WrappedF{F,P,T}
    f::F
    p::P
    t::T
end
(wf::WrappedF)(du, u) = wf.f(du, u, wf.p, wf.t)
