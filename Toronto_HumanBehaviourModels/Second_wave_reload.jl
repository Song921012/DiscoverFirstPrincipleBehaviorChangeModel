using DifferentialEquations
using LinearAlgebra, DiffEqSensitivity, Optim
using Flux: flatten, params
using DiffEqFlux, Flux
using Plots
using Flux:train!
using GalacticOptim
using Optim
using Turing, Distributions
using MCMCChains, StatsPlots
using CSV, DataFrames
using SymbolicRegression
using Random
using Dates
Random.seed!(14);

# Load the data
source_data = DataFrame(CSV.File("2021Toronto_HumanBehaviourModels/CityofToronto_COVID-19_RecoveryData.csv"))
data_on = source_data[:, "7 day moving average"]
data_on_acc = [sum(data_on[1:i]) for i = 1: length(data_on)]

province_name, n, m, popu = "Toronto", 275,79,2930000
start_data = Date(2020, 03, 01) + Dates.Day(n)
data_daily = data_on[(n + 1):(n + m +1)]
data_acc = data_on_acc[(n + 1):(n + m + 1)]
xlabel_str = "Days after $start_data"
display(plot(data_daily, label="Daily Confirmed Cases", lw=2))
display(plot(data_acc, label="Accumulated Confirmed Cases", lw=2))
data_daily[1]
println(length(data_acc))

savetitle = "Toronto_second_wave_ML_based_method_1128"
popu = 2930000
m = length(data_acc)-1



# Model generation
#using CUDA
#CUDA.allowscalar(false)
ann = FastChain(FastDense(1, 32, swish),FastDense(32, 1))
p_0 = initial_params(ann)
function SIR_nn(du, u, p, t)
    γ = 1 / 10
    N = popu
    S, I, R, H = u
    du[1] = - S * abs(ann(t, p)[1]) / N
    du[2] = S * abs(ann(t, p)[1]) / N - γ * I
    du[3] = γ * I
    du[4] = S * abs(ann(t, p)[1]) / N
end
u_0 = Float32[popu, data_daily[1], 1, data_acc[1]]
tspan_data = (0.0f0, m)
prob_nn = ODEProblem(SIR_nn, u_0, tspan_data, p_0)
function train(θ)
    solvedata = Array(concrete_solve(prob_nn, Vern7(), u_0, θ, saveat=1,
                        abstol=1e-6, reltol=1e-6,
                        sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

using BSON: @load
@load "./2021Toronto_HumanBehaviourModels/Saving_Data/second_wave_ann_nn_ir.bason" ann
@load "./2021Toronto_HumanBehaviourModels/Saving_Data/second_wave_ann_para_irbfgs100.bason" second_wave_ann_param
@load "./2021Toronto_HumanBehaviourModels/Saving_Data/second_wave_ann_para_irbfgs500.bason" second_wave_ann_param

p_min = second_wave_ann_param
tspan_predict = (0.0, length(data_acc)-1)
scatter(data_acc,label="Real accumulated cases")
prob_prediction = ODEProblem(SIR_nn, u_0, tspan_data, p_min)
data_prediction = Array(solve(prob_prediction, Tsit5(), saveat=1))
plot!(data_prediction[4,:],label="Fit accumulated cases")
savefig("./2021Toronto_HumanBehaviourModels/Saving_Data/Toronto/second_wave_toronto_nn_acc.png")
mid = zeros(length(data_acc))
mid[2:end] = data_prediction[4,1:end - 1]
pred_daily = data_prediction[4,:] - mid
pred_daily[1] = data_daily[1]
scatter(data_daily, label = "Real daily confirmed")
plot!(pred_daily,label="Fit daily confirmed by NN",lw =2)
savefig("./2021Toronto_HumanBehaviourModels/Saving_Data/Toronto/second_wave_toronto_nn_daily.png")

# Symbolic SymbolicRegression

xx= data_prediction[2,:]
yy = data_prediction[3,:]/100
X = [xx';yy']
tspan = collect(0:1:m)'
ann_value = abs.(ann(tspan,second_wave_ann_param))
Y = ann_value[1,:]
power_abs(x,y) = abs(x)^y
options = SymbolicRegression.Options(
    binary_operators=(+, *, /, -,power_abs),
    unary_operators=(exp,),
    npopulations=20
)
hallOfFame = EquationSearch(X, Y, niterations=5, options=options, numprocs=4)
dominating = calculateParetoFrontier(X, Y, hallOfFame, options)
eqn = node_to_symbolic(dominating[end].tree, options)


# Plot  Symbolic SymbolicRegression

f(x1,x2) = 166.31638 + power_abs(0.76200837x1*(0.60947824 + 0.18002248(x2^-1)), 0.82347095) - (0.9300104x2)
zz = f.(xx,yy)[6:end]
tspan = collect(6:1:m)'
ann_value = abs.(ann(tspan,second_wave_ann_param))
plot(ann_value', label = "NN(t)")
plot!(zz,label = "Symbolic Regression Function")
savefig("./2021Toronto_HumanBehaviourModels/Saving_Data/Toronto/second_wave_toronto_nn_symbolic_regression.png")