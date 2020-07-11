# ProbNumODE.jl: Probabilistic Numerics for Ordinary Differential Equations

[![Build Status](https://travis-ci.com/nathanaelbosch/ProbNumODE.jl.svg?branch=master)](https://travis-ci.com/nathanaelbosch/ProbNumODE.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/nathanaelbosch/ProbNumODE.jl?svg=true)](https://ci.appveyor.com/project/nathanaelbosch/ProbNumODE-jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://nathanaelbosch.github.io/ProbNumODE.jl/dev)
[![Coverage](https://codecov.io/gh/nathanaelbosch/ProbNumODE.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nathanaelbosch/ProbNumODE.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
<!-- [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://nathanaelbosch.github.io/ProbNumODE.jl/stable) -->
<!-- [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) -->


## Installation
The package can be installed directly from github:
```julia
] add https://github.com/nathanaelbosch/ProbNumODE.jl
```


## Example
```julia
using ProbNumODE
prob = fitzhugh_nagumo()
sol = solve(prob, EKF0())
using Plots
plot(sol)
```
![Fitzhugh-Nagumo Solution](./docs/src/figures/fitzhugh_nagumo.png?raw=true "Fitzhugh-Nagumo Solution")
