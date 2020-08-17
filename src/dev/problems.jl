function exponential_decay(;λ=-50, u0=1., tspan=(0.0, 1.0))
    f(u, p, t) = λ .* u
    exp_decay = ODEProblem(f, u0, tspan)
    return exp_decay
end

function logistic_equation(; r=3, u0=1e-1, tspan=(0.0, 2.5))
    f(u, p, t) = r .* u .* (1 .- u)
    log_reg = ODEProblem(f, u0, tspan)
    return log_reg
end


function brusselator(; u0=[1.5; 3], tspan=(0., 20.))
    f(u, p, t) = [1 + u[1]^2*u[2] - 4 * u[1];
                  3 * u[1] - u[1]^2 * u[2]]
    brusselator = ODEProblem(f, u0, tspan)
    return brusselator
end


function fitzhugh_nagumo(; u0=[-1.0; 1.0], tspan=(0., 20.), p=[0.2,0.2,3.0])
    function fitz(u,p,t)
        V,R = u
        a,b,c = p
        return [
            c*(V - V^3/3 + R)
            -(1/c)*(V -  a - b*R)
        ]
    end
    return ODEProblem(fitz,u0,tspan,p)
end

function fitzhugh_nagumo_iip(; u0=[-1.0; 1.0], tspan=(0., 20.), p=[0.2,0.2,3.0])
    function fitz!(du,u,p,t)
        V,R = u
        a,b,c = p
        du[1] = c*(V - V^3/3 + R)
        du[2] = -(1/c)*(V -  a - b*R)
    end
    return ODEProblem(fitz!,u0,tspan,p)
end


function lotka_volterra(;
                        u0=[1.0;1.0],
                        tspan=(0.0,10.0),
                        p=[1.5,1.0,3.0,1.0])
    function f(u, p, t)
        a, b, c, d = p
        return [a*u[1] - b*u[1]*u[2];
                -c*u[2] + d*u[1]*u[2]]
    end
    return ODEProblem(f, u0, tspan, p)
end


function van_der_pol(; u0=[0;2.], tspan=(0.0,6.3), p=1e6)
    function f(u, p, t)
        μ = p
        return [
            μ*((1-u[2]^2)*u[1] - u[2])
            1*u[1]
        ]
    end
    return ODEProblem(f, u0, tspan, p)
end
