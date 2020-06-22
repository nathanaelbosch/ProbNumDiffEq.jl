

mutable struct SolverState
    x  # Current state estimate
    t  # Current time
end

struct fixed_solver_things
    dm  # Dynamics Model
    mm  # Dynamics Model
end

function initialize!(integrator, cache)
    # cache = constant cache, aka options
end
function initialize!(integrator,cache::RK_ALGCache)
end



function perform_step!(integrator, cache)
    # cache = constant cache, aka options
    @unpack t,dt,uprev,u,f,p = integrator
    @unpack a,b,c = cache
    # Compute stuff
    # Store stuff inside integrator
end


function prob_solve(ivp; dt, q=1,
                    steprule=:constant,
                    sigmarule=:mle,
                    method=:ekf0,
                    progressbar=false,
                    σ=1,
                    abstol=1e-6,
                    reltol=1e-3,
                    maxiters=1e5,
                    sigma_running=0,
                    )
    initialize
    h = dt
    d = length(ivp.u0)
    f(x, t) = ivp.f(x, ivp.p, t)  # parameters are currently unused


    # Initialize SSM
    dm = ibm(q, d; σ=σ)
    if method == :ekf0
        mm = ekf0_measurement_model(d, q, ivp)
    elseif method == :ekf1
        mm = ekf1_measurement_model(d, q, ivp)
    else
        throw(Error("method argument not in [:ekf0, :ekf1]"))
    end


    # Initialize problem
    t_0, T = ivp.tspan
    x_0 = ivp.u0
    m_0 = [x_0; f(x_0, t_0); zeros(d*(q-1))]  # Initializing with zeros is not great!
    P_0 = diagm(0 => [zeros(d); ones(d*q)] .+ 1e-16)
    initial_state = Gaussian(m_0, P_0)

    # Initialize objects to save into
    solution = StructArray([StateBelief(t_0, initial_state)])
    timesteps = [t_0]
    predictions = StructArray([initial_state])
    filter_estimates = StructArray([initial_state])
    initial_measurement = Gaussian(
        mm.h(initial_state.μ, t_0),
        (mm.H(initial_state.μ, t_0) * initial_state.Σ * mm.H(initial_state.μ, t_0)' + mm.R))
    measurements = StructArray([initial_measurement])
    sigmas = Real[]
    stepsizes = Real[]
    rejected = []
    retcode = :Default

    # Filtering
    t = t_last = t_0
    h_proposal = h
    filter_estimate = initial_state
    msmnt_errors = Real[]
    N = 0

    steprules = Dict(
        :constant => constant_steprule(),
        :pvalue => pvalue_steprule(0.05),
        :baseline => classic_steprule(abstol, reltol),
        :measurement_error => measurement_error_steprule(),
        :measurement_scaling => measurement_scaling_steprule(),
        :schober16 => schober16_steprule(),
    )
    steprule = steprules[steprule]

    sigmarules = Dict(
        :mle => sigma_mle,
        :mle_weighted => sigma_mle_weighted,
        :map => sigma_map,
        :running => sigma_running_average,
        :individual => (argv...; running, kwargv...) -> sigma_running_average(argv...; running=0, kwargv...),
    )
    sigmarule = sigmarules[sigmarule]


    proposals = []


    if progressbar
        pbar_update, pbar_close = make_progressbar(0.1)
    end
    iter = 0
    while t_last < T
        if progressbar
            pbar_update(fraction=(t_last-t_0)/(T-t_0))
        end

        # Here happens the main "work"
        t = t_last + h
        prediction, _filter_estimate, measurement, σ² = predict_update(
            filter_estimate, (A=dm.A(h), Q=dm.Q(h)), mm, t)

        # Error esitmation
        current_error = (measurement.μ' * measurement.Σ^(-1) * measurement.μ) / d
        sigma = sigmarule(msmnt_errors, current_error; stepsizes=stepsizes, h=h, d=d, running=sigma_running)

        previous_sigma = (N>0 ? sigmas[end] : sigma)
        accept, new_h = steprule(
            current_h=h,
            current_error=current_error,
            sigma=sigma,
            σ²=σ²,
            dm=dm,
            mm=mm,
            previous_sigma=previous_sigma,
            predictions=predictions,
            measurement=measurement,
            prediction=prediction,
            q=q,
            t=t,
            d=d)

        if accept
            # Save things and move on
            t_last = t
            push!(predictions, prediction)
            filter_estimate = _filter_estimate
            push!(filter_estimates, filter_estimate)
            push!(measurements, measurement)
            push!(timesteps, t)
            push!(stepsizes, h)
            push!(msmnt_errors, current_error)
            N += 1
            push!(sigmas, sigma)
        else
            push!(rejected, (t=t, h=h, current_error=current_error/previous_sigma))
        end

        h = min(new_h, T-t)
        iter += 1
        if iter >= maxiters
            break
            retcode = :MaxIters
        end
    end

    if progressbar
        pbar_close()
    end

    # Smoothing
    # THERE IS A PROBLEM WITH THE SMOOTHING ACTUALLY!
    # Ok the ploblem is obvious: I did not consider the fact that I need to chose the correct `h`
    # smoothed_estimates = StructArray{Gaussian}(undef, length(filter_estimates))
    # smoothed_estimates[end] = filter_estimates[end]
    # for i in length(smoothed_estimates)-1:-1:1
    #     smoothed_estimates[i] = smooth(filter_estimates[i],
    #                                    predictions[i+1],
    #                                    smoothed_estimates[i+1],
    #                                    (A=dm.A(h), Q=dm.Q(h)))
    # end

    # quadratic_errors = [measurements[i, :]' * measurement_covs[i, :, :]^(-1) * measurements[i, :] for i in 1:size(measurements)[1]]
    # sigma_squared_mll = cumsum(quadratic_errors, dims=1) ./ d ./ (1:length(quadratic_errors))

    return (
        ivp=ivp,
        filter=(
            dynamics_model=dm,
            measurement_model=mm,
        ),
        result=(
            predictions=predictions,
            filter_estimates=filter_estimates,
            # smoothed_estimates=smoothed_estimates,
            measurements=measurements,
            timesteps=timesteps,
        ),
        rejected=rejected,
        d=d,
        q=q,
        σ²=[sigmas[1]; sigmas],
        retcode=retcode,
    )
end
