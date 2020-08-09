# Solvers and Options
For now, just a list of available options, without much explanation.

## Solvers
- `EKF0()`
- `EKF1()`

## Adaptive Step Size Schemes
- `steprule=:schober16`
- `steprule=:baseline`
- `steprule=:constant`

- If choosing an adaptive scheme: `abstol` and `reltol` as in DifferentialEquations.jl

## Sigma Estimation
- `sigmarule=ProbNumODE.Schober16Sigma()`
- `sigmarule=ProbNumODE.MLESigma()`
- `sigmarule=ProbNumODE.MAPSigma()`
- `sigmarule=ProbNumODE.WeightedMLESigma()`

## Smoothing
- `smoothed=true`
- `smoothed=false`

## Preconditioning
- `precondition=true`
- `precondition=false`
