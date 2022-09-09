abstract type AbstractMeasurementModel end

struct StandardODEMeasurementModel{IIP,F} <: AbstractMeasurementModel
    f::F
end
StandardODEMeasurementModel(f::SciMLBase.AbstractODEFunction) = begin
    f = SciMLBase.unwrapped_f(f)
    StandardODEMeasurementModel{isinplace(f),typeof(f)}(f)
end

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

struct SecondOrderODEMeasurementModel{IIP,F} <: AbstractMeasurementModel
    f1::F
end
SecondOrderODEMeasurementModel(f::SciMLBase.AbstractODEFunction) = begin
    # f = SciMLBase.unwrapped_f(f)
    SecondOrderODEMeasurementModel{isinplace(f.f1),typeof(f.f1)}(f.f1)
end

function (m::SecondOrderODEMeasurementModel{true})(z, x, p, t)
    d = length(z)
    D = length(x)
    q = D ÷ d - 1
    E0_x = @view x[1:(q+1):end]
    E1_x = @view x[2:(q+1):end]
    E2_x = @view x[3:(q+1):end]

    m.f1(z, E1_x, E0_x, p, t)

    z .= E2_x .- z
    return z
end

function make_measurement_model(f::SciMLBase.AbstractODEFunction)
    if f isa DynamicalODEFunction
        return SecondOrderODEMeasurementModel(f)
    else
        return StandardODEMeasurementModel(f)
    end
end
