# ProbNumODE.jl

[![Build Status](https://travis-ci.com/nathanaelbosch/ProbNumODE.jl.svg?branch=master)](https://travis-ci.com/nathanaelbosch/ProbNumODE.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/nathanaelbosch/ProbNumODE.jl?svg=true)](https://ci.appveyor.com/project/nathanaelbosch/ProbNumODE-jl)
[![Coverage](https://codecov.io/gh/nathanaelbosch/ProbNumODE.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nathanaelbosch/ProbNumODE.jl)
[![Coverage](https://coveralls.io/repos/github/nathanaelbosch/ProbNumODE.jl/badge.svg?branch=master)](https://coveralls.io/github/nathanaelbosch/ProbNumODE.jl?branch=master)


## Installation
The package can be installed directly from github:
```julia
] add https://github.com/nathanaelbosch/ProbNumODE.jl
```


## Example
```julia
using ProbNumODE
prob = fitzhugh_nagumo()
sol = solve(prob, ODEFilter())
using Plots
plot(sol)
```
![Fitzhugh-Nagumo Solution](./docs/src/figures/fitzhugh_nagumo.png?raw=true "Fitzhugh-Nagumo Solution")
