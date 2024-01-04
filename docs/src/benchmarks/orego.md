# OREGO benchmark


!!! note "Summary"
    - [**The `EK1` is able to solve mass-matrix DAEs.** To achieve low error, use order 4 or higher.](@ref orego_results)
    - The order-to-error-tolerance heuristic holds: lower tolerance level ``\rightarrow`` higher order.


Adapted from
[SciMLBenchmarks.jl](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/DAE/OregoDAE/).

```@raw html
<details><summary>Code:</summary>
```
```julia
using LinearAlgebra, Statistics, Distributions
using DiffEqDevTools, ParameterizedFunctions, SciMLBase, OrdinaryDiffEq, Sundials, Plots
using ModelingToolkit
using ProbNumDiffEq

Plots.theme(
    :dao;
    markerstrokewidth=0.5,
    legend=:outertopright,
    margin=5Plots.mm,
    xticks=10.0 .^ (-16:1:16),
)

function plot_chisq_interval!(df, q=0.01)
    dist = Chisq(df)
    low, high, mid = quantile(dist, [q, 1-q])..., mean(dist)
    hline!([low, high], linestyle=:dash, color=:black, label="",
           fill_between=[high nothing], fillcolor=:green, fillalpha=0.15)
    hline!([mid], linestyle=:solid, color=:black, label="")
end
```
```@raw html
</details>
```


```@raw html
<details><summary>Code:</summary>
```
```julia
@variables t y1(t)=1.0 y2(t)=2.0 y3(t)=3.0
@parameters p1=77.27 p2=8.375e-6 p3=0.161
D = Differential(t)
eqs = [
  D(y1) ~ p1*(y2+y1*(1-p2*y1-y2))
  D(y2) ~ (y3-(1+y1)*y2)/p1
  D(y3) ~ p3*(y1-y3)
]
@named sys = ODESystem(eqs)
simpsys = structural_simplify(sys)
mmprob = ODEProblem(sys,[],(0.0,30.0))
daeprob = DAEProblem(sys,[D(y1)=>77.26935286375,
                          D(y2)=>-0.012941633234114146,
                          D(y3)=>-0.322],[],(0.0,30.0))
odaeprob = ODAEProblem(simpsys,[],(0.0,30.0))

ref_sol = solve(daeprob,IDA(),abstol=1/10^14,reltol=1/10^14)

plot(ref_sol, title="OREGO Solution", legend=false, xticks=:auto)
```
```@raw html
</details>
```

![](figures/orego_2_1.svg)



## [`EK1` across orders](@id orego_results)

```@raw html
<details><summary>Code:</summary>
```
```julia
DENSE = false;
SAVE_EVERYSTEP = false;

_setups = [
  "EK1($order)" => Dict(:alg => EK1(order=order, smooth=DENSE))
  for order in 2:6
]

labels = first.(_setups)
setups = last.(_setups)

abstols = 1.0 ./ 10.0 .^ (6:10)
reltols = 1.0 ./ 10.0 .^ (3:7)

wp = WorkPrecisionSet(
    mmprob, abstols, reltols, setups;
    names = labels,
    appxsol = ref_sol,
    dense = DENSE,
    save_everystep = SAVE_EVERYSTEP,
    numruns = 10,
    maxiters = Int(1e7),
)

plot(wp, palette=Plots.palette([:blue, :red], length(_setups)))


_ref_setups = [
    "Rosenbrock23" => Dict(:alg => Rosenbrock23())
    "Rodas4P" => Dict(:alg => Rodas4P())
    "RadauIIA" => Dict(:alg => RadauIIA5())
]
ref_labels = first.(_ref_setups)
ref_setups = last.(_ref_setups)
ref_wp = WorkPrecisionSet(
    mmprob, abstols, reltols, ref_setups;
    names = ref_labels,
    appxsol = ref_sol,
    dense = DENSE,
    save_everystep = SAVE_EVERYSTEP,
    maxiters = Int(1e7),
)
plot!(ref_wp, x=:final, color=:gray, alpha=0.7, linestyle=:dash)
```
```@raw html
</details>
```

![](figures/orego_3_1.svg)



### Calibration

```@raw html
<details><summary>Code:</summary>
```
```julia
plot(wp; x=:final, y=:chi2_final, yguide="Chi-squared (final)",
     palette=Plots.palette([:blue, :red], length(_setups)))
plot_chisq_interval!(3)
```
```@raw html
</details>
```

![](figures/orego_4_1.svg)




## Appendix

```@raw html
<details><summary>Computer information:</summary>
```

```julia
using InteractiveUtils
InteractiveUtils.versioninfo()
```

```
Julia Version 1.10.0
Commit 3120989f39b (2023-12-25 18:01 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 12 × Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, broadwell)
  Threads: 17 on 12 virtual cores
Environment:
  JULIA_NUM_THREADS = auto
```

```@raw html
</details>
```

```@raw html
<details><summary>Package information:</summary>
```

```julia
using Pkg
Pkg.status()
```

```
Status `~/.julia/dev/ProbNumDiffEq/benchmarks/Project.toml`
  [f3b72e0c] DiffEqDevTools v2.44.1
  [31c24e10] Distributions v0.25.104
  [7073ff75] IJulia v1.24.2
  [7f56f5a3] LSODA v0.7.5
  [e6f89c97] LoggingExtras v1.0.3
  [e2752cbe] MATLABDiffEq v1.2.0
⌃ [961ee093] ModelingToolkit v8.73.2
  [54ca160b] ODEInterface v0.5.0
  [09606e27] ODEInterfaceDiffEq v3.13.3
⌃ [1dea7af3] OrdinaryDiffEq v6.66.0
  [65888b18] ParameterizedFunctions v5.16.0
  [91a5bcdd] Plots v1.39.0
  [bf3e78b0] ProbNumDiffEq v0.13.1 `~/.julia/dev/ProbNumDiffEq`
⌃ [0bca4576] SciMLBase v2.10.0
  [505e40e9] SciPyDiffEq v0.2.1
  [ce78b400] SimpleUnPack v1.1.0
  [90137ffa] StaticArrays v1.9.0
  [c3572dad] Sundials v4.23.1
  [44d3d7a6] Weave v0.10.12
  [0518478a] deSolveDiffEq v0.1.1
Info Packages marked with ⌃ have new versions available and may be upgradable.
```

```@raw html
</details>
```

```@raw html
<details><summary>Full manifest:</summary>
```

```julia
Pkg.status(mode=Pkg.PKGMODE_MANIFEST)
```

```
Status `~/.julia/dev/ProbNumDiffEq/benchmarks/Manifest.toml`
  [47edcb42] ADTypes v0.2.6
⌅ [c3fe647b] AbstractAlgebra v0.34.7
  [621f4979] AbstractFFTs v1.5.0
  [1520ce14] AbstractTrees v0.4.4
  [7d9f7c33] Accessors v0.1.33
⌅ [79e6a3ab] Adapt v3.7.2
  [ec485272] ArnoldiMethod v0.2.0
  [c9d4266f] ArrayAllocators v0.3.0
  [4fba245c] ArrayInterface v7.7.0
  [4c555306] ArrayLayouts v1.4.5
  [6e4b80f9] BenchmarkTools v1.4.0
  [e2ed5e7c] Bijections v0.1.6
  [d1d4a3ce] BitFlags v0.1.8
  [62783981] BitTwiddlingConvenienceFunctions v0.1.5
  [fa961155] CEnum v0.5.0
  [2a0fbf3d] CPUSummary v0.2.4
  [00ebfdb7] CSTParser v3.3.6
  [49dc2e85] Calculus v0.5.1
  [324d7699] CategoricalArrays v0.10.8
  [d360d2e6] ChainRulesCore v1.19.0
  [fb6a15b2] CloseOpenIntervals v0.1.12
  [944b1d66] CodecZlib v0.7.3
  [35d6a980] ColorSchemes v3.24.0
  [3da002f7] ColorTypes v0.11.4
  [c3611d14] ColorVectorSpace v0.10.0
  [5ae59095] Colors v0.12.10
  [861a8166] Combinatorics v1.0.2
  [a80b9123] CommonMark v0.8.12
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v4.10.1
  [b152e2b5] CompositeTypes v0.1.3
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.3
  [f0e56b4a] ConcurrentUtilities v2.3.0
  [8f4d0f93] Conda v1.10.0
  [187b0558] ConstructionBase v1.5.4
  [d38c429a] Contour v0.6.2
  [587fd27a] CovarianceEstimation v0.2.11
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [717857b8] DSP v0.7.9
  [9a962f9c] DataAPI v1.15.0
  [a93c6f00] DataFrames v1.6.1
  [864edb3b] DataStructures v0.18.15
  [e2d170a0] DataValueInterfaces v1.0.0
  [8bb1440f] DelimitedFiles v1.9.1
⌃ [2b5f629d] DiffEqBase v6.145.2
  [459566f4] DiffEqCallbacks v2.36.1
  [f3b72e0c] DiffEqDevTools v2.44.1
  [77a26b50] DiffEqNoiseProcess v5.20.0
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
  [b4f34e82] Distances v0.10.11
  [31c24e10] Distributions v0.25.104
  [ffbed154] DocStringExtensions v0.9.3
⌅ [5b8099bc] DomainSets v0.6.7
  [fa6b7ba4] DualNumbers v0.6.8
  [7c1d4256] DynamicPolynomials v0.5.3
  [b305315f] Elliptic v1.0.1
  [4e289a0a] EnumX v1.0.4
  [f151be2c] EnzymeCore v0.6.4
  [6912e4f1] Espresso v0.6.1
  [460bff9d] ExceptionUnwrapping v0.1.10
  [d4d017d3] ExponentialUtilities v1.25.0
  [e2ba6199] ExprTools v0.1.10
  [c87230d0] FFMPEG v0.4.1
  [7a1cc6ca] FFTW v1.7.2
  [7034ab61] FastBroadcast v0.2.8
  [9aa1b823] FastClosures v0.3.2
  [442a2c76] FastGaussQuadrature v1.0.1
  [29a986be] FastLapackInterface v2.0.0
  [1a297f60] FillArrays v1.9.3
  [6a86dc24] FiniteDiff v2.22.0
⌃ [b59a298d] FiniteHorizonGramians v0.1.1
  [53c48c17] FixedPointNumbers v0.8.4
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.36
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
  [d9f16b24] Functors v0.4.5
⌃ [46192b85] GPUArraysCore v0.1.5
⌅ [28b8d3ca] GR v0.72.10
  [43dcc890] GaussianDistributions v0.5.2
  [c145ed77] GenericSchur v0.5.3
  [c27321d9] Glob v1.3.1
  [86223c79] Graphs v1.9.0
  [42e2da0e] Grisu v1.0.2
  [0b43b601] Groebner v0.5.1
⌅ [d5909c97] GroupsCore v0.4.2
  [cd3eb016] HTTP v1.10.1
  [eafb193a] Highlights v0.5.2
  [3e5b6fbb] HostCPUFeatures v0.1.16
  [34004b35] HypergeometricFunctions v0.3.23
  [7073ff75] IJulia v1.24.2
  [615f187c] IfElse v0.1.1
  [d25df0c9] Inflate v0.1.4
  [842dd82b] InlineStrings v1.4.0
  [18e54dd8] IntegerMathUtils v0.1.2
  [8197267c] IntervalSets v0.7.8
  [3587e190] InverseFunctions v0.1.12
  [41ab1584] InvertedIndices v1.3.0
  [92d709cd] IrrationalConstants v0.2.2
  [c8e1da08] IterTools v1.10.0
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.7
  [692b3bcd] JLLWrappers v1.5.0
  [682c06a0] JSON v0.21.4
  [98e50ef6] JuliaFormatter v1.0.45
  [ccbc3e58] JumpProcesses v9.10.1
  [ef3ab10e] KLU v0.4.1
  [2c470bb0] Kronecker v0.5.5
  [ba0b0d4f] Krylov v0.9.5
  [7f56f5a3] LSODA v0.7.5
  [b964fa9f] LaTeXStrings v1.3.1
  [2ee39098] LabelledArrays v1.15.0
  [984bce1d] LambertW v0.4.6
  [23fbe1c1] Latexify v0.16.1
  [73f95e8e] LatticeRules v0.0.1
  [10f19ff3] LayoutPointers v0.1.15
  [50d2b5c4] Lazy v0.15.1
  [5078a376] LazyArrays v1.8.3
  [1d6d02ad] LeftChildRightSiblingTrees v0.2.0
  [d3d80556] LineSearches v7.2.0
  [7ed4a6bd] LinearSolve v2.22.0
  [2ab3a3ac] LogExpFunctions v0.3.26
  [e6f89c97] LoggingExtras v1.0.3
  [bdcacae8] LoopVectorization v0.12.168
  [10e44e05] MATLAB v0.8.4
  [e2752cbe] MATLABDiffEq v1.2.0
  [d8e11817] MLStyle v0.4.17
  [1914dd2f] MacroTools v0.5.12
  [d125e4d3] ManualMemory v0.1.8
  [a3b82374] MatrixFactorizations v2.1.0
  [bb5d69b7] MaybeInplace v0.1.1
  [739be429] MbedTLS v1.1.9
  [442fdcdd] Measures v0.3.2
  [e1d29d7a] Missings v1.1.0
⌃ [961ee093] ModelingToolkit v8.73.2
  [46d2c3a1] MuladdMacro v0.2.4
  [102ac46a] MultivariatePolynomials v0.5.3
  [ffc61752] Mustache v1.0.19
  [d8a4904e] MutableArithmetics v1.4.0
  [d41bc354] NLSolversBase v7.8.3
  [2774e3e8] NLsolve v4.5.1
  [77ba4419] NaNMath v1.0.2
  [356022a1] NamedDims v1.2.1
⌃ [8913a72c] NonlinearSolve v3.1.0
  [54ca160b] ODEInterface v0.5.0
  [09606e27] ODEInterfaceDiffEq v3.13.3
  [6fd5a793] Octavian v0.3.27
  [6fe1bfb0] OffsetArrays v1.13.0
  [4d8831e6] OpenSSL v1.4.1
  [429524aa] Optim v1.7.8
  [bac558e1] OrderedCollections v1.6.3
⌃ [1dea7af3] OrdinaryDiffEq v6.66.0
  [90014a1f] PDMats v0.11.31
  [fe68d972] PSDMatrices v0.4.7
  [65ce6f38] PackageExtensionCompat v1.0.2
  [65888b18] ParameterizedFunctions v5.16.0
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.8.1
  [b98c9c47] Pipe v1.3.0
  [32113eaa] PkgBenchmark v0.2.12
  [ccf2f8ad] PlotThemes v3.1.0
  [995b91a9] PlotUtils v1.4.0
  [91a5bcdd] Plots v1.39.0
  [e409e4f3] PoissonRandom v0.4.4
  [f517fe37] Polyester v0.7.9
  [1d0040c9] PolyesterWeave v0.2.1
⌅ [f27b6e38] Polynomials v3.2.13
  [2dfb63ee] PooledArrays v1.4.3
  [85a6dd25] PositiveFactorizations v0.2.4
  [d236fae5] PreallocationTools v0.4.17
  [aea7be01] PrecompileTools v1.2.0
  [21216c6a] Preferences v1.4.1
  [08abe8d2] PrettyTables v2.3.1
  [27ebfcd6] Primes v0.5.5
  [bf3e78b0] ProbNumDiffEq v0.13.1 `~/.julia/dev/ProbNumDiffEq`
  [33c8b6b6] ProgressLogging v0.1.4
  [438e738f] PyCall v1.96.4
  [1fd47b50] QuadGK v2.9.1
  [8a4e6c94] QuasiMonteCarlo v0.3.3
  [6f49c342] RCall v0.13.18
  [74087812] Random123 v1.6.2
  [fb686558] RandomExtensions v0.4.4
  [e6cf234a] RandomNumbers v1.5.3
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
⌅ [731186ca] RecursiveArrayTools v2.38.10
  [f2c3362d] RecursiveFactorization v0.2.21
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.0
  [ae5879a3] ResettableStacks v1.1.1
  [79098fc4] Rmath v0.7.1
  [47965b36] RootedTrees v2.20.0
  [7e49a35a] RuntimeGeneratedFunctions v0.5.12
  [fdea26ae] SIMD v3.4.6
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.42
⌃ [0bca4576] SciMLBase v2.10.0
  [c0aeaf25] SciMLOperators v0.3.7
  [505e40e9] SciPyDiffEq v0.2.1
  [6c6a2e73] Scratch v1.2.1
  [91c51154] SentinelArrays v1.4.1
  [efcf1570] Setfield v1.1.1
  [1277b4bf] ShiftedArrays v2.0.0
  [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.1.0
  [727e6d20] SimpleNonlinearSolve v1.2.0
  [699a6c99] SimpleTraits v0.9.4
  [ce78b400] SimpleUnPack v1.1.0
  [ed01d8cd] Sobol v1.5.0
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.2.1
  [47a9eef4] SparseDiffTools v2.15.0
  [e56a9233] Sparspak v0.3.9
  [276daf66] SpecialFunctions v2.3.1
  [928aab9d] SpecialMatrices v3.0.0
  [aedffcd0] Static v0.8.8
  [0d7ed370] StaticArrayInterface v1.5.0
  [90137ffa] StaticArrays v1.9.0
  [1e83bf80] StaticArraysCore v1.4.2
  [82ae8749] StatsAPI v1.7.0
  [2913bbd2] StatsBase v0.34.2
  [4c63d2b9] StatsFuns v1.3.0
  [3eaba693] StatsModels v0.7.3
  [7792a7ef] StrideArraysCore v0.5.2
  [69024149] StringEncodings v0.3.7
  [892a3eda] StringManipulation v0.3.4
  [09ab397b] StructArrays v0.6.16
  [c3572dad] Sundials v4.23.1
⌅ [2efcf032] SymbolicIndexingInterface v0.2.2
⌃ [d1185830] SymbolicUtils v1.4.0
⌃ [0c5d862f] Symbolics v5.11.0
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.11.1
  [92b13dbe] TaylorIntegration v0.14.5
  [6aa5eb33] TaylorSeries v0.15.4
  [62fd8b95] TensorCore v0.1.1
  [5d786b92] TerminalLoggers v0.1.7
  [8290d209] ThreadingUtilities v0.5.2
  [a759f4b9] TimerOutputs v0.5.23
  [c751599d] ToeplitzMatrices v0.8.2
  [0796e94c] Tokenize v0.5.26
  [3bb67fe8] TranscodingStreams v0.10.2
  [d5829a12] TriangularSolve v0.1.20
  [410a4b4d] Tricks v0.1.8
  [781d530d] TruncatedStacktraces v1.4.0
  [5c2747f8] URIs v1.5.1
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [1986cc42] Unitful v1.19.0
  [45397f5d] UnitfulLatexify v1.6.3
  [a7c27f48] Unityper v0.1.6
  [41fe7b60] Unzip v0.2.0
  [3d5dd08c] VectorizationBase v0.21.65
  [81def892] VersionParsing v1.3.0
  [19fa3120] VertexSafeGraphs v0.2.0
  [44d3d7a6] Weave v0.10.12
  [1b915085] WinReg v1.0.0
  [ddb6d928] YAML v0.4.9
  [c2297ded] ZMQ v1.2.2
  [0518478a] deSolveDiffEq v0.1.1
  [6e34b625] Bzip2_jll v1.0.8+0
  [83423d85] Cairo_jll v1.16.1+1
  [2702e6a9] EpollShim_jll v0.0.20230411+0
  [2e619515] Expat_jll v2.5.0+0
  [b22a6f82] FFMPEG_jll v4.4.4+1
  [f5851436] FFTW_jll v3.3.10+0
  [a3f928ae] Fontconfig_jll v2.13.93+0
  [d7e528f0] FreeType2_jll v2.13.1+0
  [559328eb] FriBidi_jll v1.0.10+0
  [0656b61e] GLFW_jll v3.3.9+0
⌅ [d2c73de3] GR_jll v0.72.10+0
  [78b55507] Gettext_jll v0.21.0+0
  [7746bdde] Glib_jll v2.76.5+0
  [3b182d85] Graphite2_jll v1.3.14+0
  [2e76f6c2] HarfBuzz_jll v2.8.1+1
  [1d5cc7b8] IntelOpenMP_jll v2024.0.2+0
  [aacddb02] JpegTurbo_jll v3.0.1+0
  [c1c5ebd0] LAME_jll v3.100.1+0
  [88015f11] LERC_jll v3.0.0+1
  [1d63c593] LLVMOpenMP_jll v15.0.7+0
  [aae0fff6] LSODA_jll v0.1.2+0
  [dd4b983a] LZO_jll v2.10.1+0
⌅ [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.8.7+0
  [7e76a0d4] Libglvnd_jll v1.6.0+0
  [7add5ba3] Libgpg_error_jll v1.42.0+0
  [94ce4f54] Libiconv_jll v1.17.0+0
  [4b2f31a3] Libmount_jll v2.35.0+0
⌅ [89763e89] Libtiff_jll v4.5.1+1
  [38a345b3] Libuuid_jll v2.36.0+0
  [856f044c] MKL_jll v2024.0.0+0
  [c771fb93] ODEInterface_jll v0.0.1+0
  [e7412a2a] Ogg_jll v1.3.5+1
  [458c3c95] OpenSSL_jll v3.0.12+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [30392449] Pixman_jll v0.42.2+0
  [c0090381] Qt6Base_jll v6.5.3+1
  [f50d1b31] Rmath_jll v0.4.0+0
⌅ [fb77eaff] Sundials_jll v5.2.2+0
  [a44049a8] Vulkan_Loader_jll v1.3.243+0
  [a2964d1f] Wayland_jll v1.21.0+1
  [2381bf8a] Wayland_protocols_jll v1.31.0+0
  [02c8fc9c] XML2_jll v2.12.2+0
  [aed1982a] XSLT_jll v1.1.34+0
  [ffd25f8a] XZ_jll v5.4.5+0
  [f67eecfb] Xorg_libICE_jll v1.0.10+1
  [c834827a] Xorg_libSM_jll v1.2.3+0
  [4f6342f7] Xorg_libX11_jll v1.8.6+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.11+0
  [935fb764] Xorg_libXcursor_jll v1.2.0+4
  [a3789734] Xorg_libXdmcp_jll v1.1.4+0
  [1082639a] Xorg_libXext_jll v1.3.4+4
  [d091e8ba] Xorg_libXfixes_jll v5.0.3+4
  [a51aa0fd] Xorg_libXi_jll v1.7.10+4
  [d1454406] Xorg_libXinerama_jll v1.1.4+4
  [ec84b674] Xorg_libXrandr_jll v1.5.2+4
  [ea2f1a96] Xorg_libXrender_jll v0.9.10+4
  [14d82f49] Xorg_libpthread_stubs_jll v0.1.1+0
  [c7cfdc94] Xorg_libxcb_jll v1.15.0+0
  [cc61e674] Xorg_libxkbfile_jll v1.1.2+0
  [e920d4aa] Xorg_xcb_util_cursor_jll v0.1.4+0
  [12413925] Xorg_xcb_util_image_jll v0.4.0+1
  [2def613f] Xorg_xcb_util_jll v0.4.0+1
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.0+1
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.9+1
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.1+1
  [35661453] Xorg_xkbcomp_jll v1.4.6+0
  [33bec58e] Xorg_xkeyboard_config_jll v2.39.0+0
  [c5fb5394] Xorg_xtrans_jll v1.5.0+0
  [8f1865be] ZeroMQ_jll v4.3.4+0
  [3161d3a3] Zstd_jll v1.5.5+0
  [35ca27e7] eudev_jll v3.2.9+0
  [214eeab7] fzf_jll v0.43.0+0
  [1a1c6b14] gperf_jll v3.1.1+0
  [a4ae2306] libaom_jll v3.4.0+0
  [0ac62f75] libass_jll v0.15.1+0
  [2db6ffa8] libevdev_jll v1.11.0+0
  [f638f0a6] libfdk_aac_jll v2.0.2+0
  [36db933b] libinput_jll v1.18.0+0
  [b53b4c65] libpng_jll v1.6.40+0
  [a9144af2] libsodium_jll v1.0.20+0
  [f27f6e37] libvorbis_jll v1.3.7+1
  [009596ad] mtdev_jll v1.1.6+0
  [1270edf5] x264_jll v2021.5.5+0
  [dfaa095f] x265_jll v3.5.0+0
  [d8fb68d0] xkbcommon_jll v1.4.1+1
  [0dad84c5] ArgTools v1.1.1
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [ade2ca70] Dates
  [8ba89e20] Distributed
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching
  [9fa8497b] Future
  [b77e0a4c] InteractiveUtils
  [4af54fe1] LazyArtifacts
  [b27032c2] LibCURL v0.6.4
  [76f85450] LibGit2
  [8f399da3] Libdl
  [37e2e46d] LinearAlgebra
  [56ddb016] Logging
  [d6f4376e] Markdown
  [a63ad114] Mmap
  [ca575930] NetworkOptions v1.2.0
  [44cfe95a] Pkg v1.10.0
  [de0858da] Printf
  [9abbd945] Profile
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays v1.10.0
  [10745b16] Statistics v1.10.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v1.0.5+1
  [deac9b47] LibCURL_jll v8.4.0+0
  [e37daf67] LibGit2_jll v1.6.4+0
  [29816b5a] LibSSH2_jll v1.11.0+1
  [c8ffd9c3] MbedTLS_jll v2.28.2+1
  [14a3606d] MozillaCACerts_jll v2023.1.10
  [4536629a] OpenBLAS_jll v0.3.23+2
  [05823500] OpenLibm_jll v0.8.1+2
  [efcefdf7] PCRE2_jll v10.42.0+1
  [bea87d4a] SuiteSparse_jll v7.2.1+1
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.8.0+1
  [8e850ede] nghttp2_jll v1.52.0+1
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

```@raw html
</details>
```

