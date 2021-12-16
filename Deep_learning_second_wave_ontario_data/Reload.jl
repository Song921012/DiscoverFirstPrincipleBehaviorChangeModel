# It is for reload the training results
using DifferentialEquations
using LinearAlgebra, DiffEqSensitivity, Optim
using Flux: flatten, params
using DiffEqFlux, Flux
using Plots
using Flux: train!
using GalacticOptim
using Optim
using Turing, Distributions
using MCMCChains, StatsPlots
using CSV, DataFrames
using SymbolicRegression
using Random
Random.seed!(14);
source_data = DataFrame(CSV.File("./Source_Data/Provincial_Daily_Totals.csv"))
data_on = source_data[source_data.Province.=="ONTARIO", :]
n = 200
m = 200
data_acc = data_on.TotalCases[(n+1):n+m+1]
data_daily = data_on.TotalCases[(n+1):n+m+1] - data_on.TotalCases[n:n+m]
display(plot(data_daily, label = "Daily Confirmed Cases", lw = 2))
display(plot(data_acc, label = "Accumulated Confirmed Cases", lw = 2))
data_daily[1]
println(length(data_acc))

# Model generation
#using CUDA
#CUDA.allowscalar(false)
ann = FastChain(FastDense(1, 32, swish), FastDense(32, 1))
function SIR_nn(du, u, p, t)
    γ = 1 / 10
    N = 14570000.0
    S, I, R, H = u
    du[1] = -S * abs(ann(t, p)[1]) / N
    du[2] = S * abs(ann(t, p)[1]) / N - γ * I
    du[3] = γ * I
    du[4] = S * abs(ann(t, p)[1]) / N
end
u_0 = Float32[14570000, data_daily[1], 1, data_acc[1]]
tspan_data = (0.0f0, 200.0f0)
#prob_nn = ODEProblem(SIR_nn, u_0, tspan_data, p_0)
function train(θ)
    solvedata = Array(concrete_solve(prob_nn, Vern7(), u_0, θ, saveat = 1,
        abstol = 1e-6, reltol = 1e-6,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP())))
end


using BSON: @save, @load
@load "./Deep_learning_second_wave_ontario_data/ann_para_irbfgs100.bason" ann_param

p_min = ann_param
tspan_predict = (0.0, 200)
scatter(data_acc, label = "Real accumulated cases")
prob_prediction = ODEProblem(SIR_nn, u_0, tspan_data, p_min)
sol = solve(prob_prediction, Tsit5(), saveat = 1)
Sus(t) = sol(t, idx = 1)
data_prediction = Array(solve(prob_prediction, Tsit5(), saveat = 1))
plot!(data_prediction[4, :], label = "Fit accumulated cases")

mid = zeros(length(data_acc))
mid[2:end] = data_prediction[4, 1:end-1]
pred_daily = data_prediction[4, :] - mid
pred_daily[1] = data_daily[1]
scatter(data_daily, label = "Real daily confirmed")
plot!(pred_daily, label = "Fit daily confirmed by NN", lw = 2)


# Symbolic SymbolicRegression
power_abs(x, y) = abs(x)^y
xx = data_prediction[2, 10:end]
yy = data_prediction[3, 10:end]

X = [xx'; yy']
Y = ann_value[1, :]

options = SymbolicRegression.Options(
    binary_operators = (+, *, /, -, power_abs),
    unary_operators = (exp,),
    npopulations = 20
)

hallOfFame = EquationSearch(X, Y, niterations = 5, options = options, numprocs = 4)
dominating = calculateParetoFrontier(X, Y, hallOfFame, options)
eqn = node_to_symbolic(dominating[end].tree, options)
# 0.07236328x1 + 1.555432(x1 - 1.7383432)*((5.9498844 + 0.0044679674x2)^-1)

f(x1, x2) = power_abs(2.611907x1 * (power_abs(x2 - 0.9984887, 0.2599004)^-1) - 519.1816, 1.0043758)
zz = f.(xx, yy)

tspan = collect(10:1:200)'
ann_value = abs.(ann(tspan, ann_param))
plot(ann_value', label = "Neural Networks abs(NN(I,R))")
plot!(zz, label = "Symbolic Regression")

# Testing Symbolic Regression
f(x1, x2) = 0.07236328x1 + 1.555432abs(x1 - 1.7383432) * ((5.9498844 + 0.0044679674x2)^-1)
function SIR_sr(du, u, p, t)
    γ = 1 / 10
    N = 14570000.0
    S, I, R, H = u
    du[1] = -S * f(I, R) / N
    du[2] = S * f(I, R) / N - γ * I
    du[3] = γ * I
    du[4] = S * f(I, R) / N
end
u_0 = Float32[14570000, 1, 1, 1]
tspan_data = (0.0f0, 150.0f0)
prob_sr = ODEProblem(SIR_sr, u_0, tspan_data, p = nothing)
data_prediction = Array(solve(prob_sr, Vern7(), saveat = 1))
scatter(data_acc, label = "Real accumulated cases")
plot!(data_prediction[4, :], label = "Fit accumulated cases")

mid = zeros(length(data_acc))
mid[2:end] = data_prediction[4, 1:end-1]
pred_daily = data_prediction[4, :] - mid
scatter(data_daily, label = "Real daily confirmed")
plot!(pred_daily, label = "Fit daily confirmed by NN", lw = 2)