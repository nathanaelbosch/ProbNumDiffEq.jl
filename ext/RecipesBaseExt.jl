module RecipesBaseExt

using RecipesBase
using ProbNumDiffEq
using Statistics

@recipe function f(
    p::AbstractArray{<:Gaussian};
    ribbon_width=1.96,
)
    means = mean.(p)
    stddevs = std.(p)
    ribbon --> ribbon_width*stddevs
    return means
end
@recipe function f(
    x, y::AbstractArray{<:Gaussian};
    ribbon_width=1.96,
)
    means = mean.(y)
    stddevs = std.(y)
    ribbon --> ribbon_width*stddevs
    return x, means
end
@recipe function f(
    x::AbstractArray{<:Gaussian}, y::AbstractArray{<:Gaussian}
)
    @warn "This plot does not visualize any uncertainties"
    xmeans = mean.(x)
    # xstddevs = std.(x)
    ymeans = mean.(y)
    # ystddevs = std.(y)
    # xerror := xstddevs
    # yerror := ystddevs
    return xmeans, ymeans
end
@recipe function f(
    x::AbstractArray{<:Gaussian},
    y::AbstractArray{<:Gaussian},
    z::AbstractArray{<:Gaussian},
)
    @warn "This plot does not visualize any uncertainties"
    xmeans = mean.(x)
    # xstddevs = std.(x)
    ymeans = mean.(y)
    # ystddevs = std.(y)
    zmeans = mean.(z)
    return xmeans, ymeans, zmeans
end

@recipe function f(
    process::ProbNumDiffEq.AbstractGaussMarkovProcess,
    plotrange;
    N_samples=10,
    plot_derivatives=false,
)
    marginals = ProbNumDiffEq.marginalize(process, plotrange)
    d = ProbNumDiffEq.dim(process)
    q = ProbNumDiffEq.num_derivatives(process)
    means = [mean(m) for m in marginals] |> stack |> permutedims
    stddevs = [std(m) for m in marginals] |> stack |> permutedims

    perm = permutedims(reshape(collect(1:d*(q+1)), q + 1, d))[:]
    reorder(X) = X[:, perm]

    E0 = ProbNumDiffEq.projection(d, q)(0)
    if !plot_derivatives
        stddevs = stddevs * E0'
        means = means * E0'
        q = 0
    end

    dui(i) = "u" * "'"^i
    if plot_derivatives
        title --> [i == 1 ? "$(dui(j))" : "" for i in 1:d for j in 0:q] |> permutedims
    end
    ylabel --> [q == 0 ? "u$i" : "" for i in 1:d for q in 0:q] |> permutedims
    xlabel --> if plot_derivatives
        [i == d ? "t" : "" for i in 1:d for q in 0:q] |> permutedims
    else
        "t"
    end

    @series begin
        ribbon --> 3stddevs
        label --> ""
        fillalpha --> 0.1
        layout --> if plot_derivatives
            (d, q + 1)
        else
            d
        end
        plotrange, means
    end

    if N_samples > 0
        samples = ProbNumDiffEq.sample(process, plotrange, N_samples) |> stack
        samples = permutedims(samples, (3, 1, 2))
        for i in 1:N_samples
            @series begin
                s = samples[:, :, i]
                if !plot_derivatives
                    s = s * E0'
                end
                primary --> false
                linealpha --> 0.3
                label := ""
                plotrange, s
            end
        end
    end
end

end
