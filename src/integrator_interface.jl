# Perform a *successful* step
function DiffEqBase.step!(integrator::ODEFilterIntegrator)
    loopheader!(integrator)
    perform_step!(integrator)
    loopfooter!(integrator)
    while !integrator.accept_step && integrator.iter < integrator.opts.maxiters
        loopheader!(integrator)
        perform_step!(integrator)
        loopfooter!(integrator)
    end
end
