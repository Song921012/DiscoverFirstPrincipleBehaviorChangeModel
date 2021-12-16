# loading the Julia packages  needed.
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
                        abstol=1e-8, reltol=1e-8,
                        sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end
function loss(θ)
    pred = train(θ)
    pred_daily =  pred[4,:][2:end]- pred[4,:][1:end-1]
    sum(abs2, (log.(data_acc) .- log.(pred[4,:])))+2*sum(abs2, (log.(data_daily[1:end-1]) .- log.(pred_daily))), pred
end

const losses = []
callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses) % 50 == 0
        println(losses[end])
    end
    false
end

@time res1_node = DiffEqFlux.sciml_train(loss, p_0, ADAM(0.06), cb=callback, maxiters=500)
@time res2_node = DiffEqFlux.sciml_train(loss, res1_node.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters=300)
second_wave_ann_param = res2_node.minimizer


using BSON: @save
@save "./2021Toronto_HumanBehaviourModels/Saving_Data/second_wave_ann_nn_ir.bason" ann
@save "./2021Toronto_HumanBehaviourModels/Saving_Data/second_wave_ann_para_irbfgs100.bason" second_wave_ann_param
@time res3_node = DiffEqFlux.sciml_train(loss, res2_node.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters=200)
second_wave_ann_param = res3_node.minimizer
@save "./2021Toronto_HumanBehaviourModels/Saving_Data/second_wave_ann_para_irbfgs500.bason" second_wave_ann_param

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

f(x1,x2) =x1*power_abs(x2 - 8.204531, -0.22241311) - 37.3306 - (10.302171(x1^-1)*(x2 - 11509.189))
zz = f.(xx,yy)

tspan = collect(0:1:m)'
ann_value = abs.(ann(tspan,second_wave_ann_param))
plot(ann_value', label = "NN(t)")
plot!(zz,label = "Symbolic Regression")
# Testing Symbolic Regression
f(x1,x2) = 0.07236328x1 + 1.555432abs(x1 - 1.7383432)*((5.9498844 + 0.0044679674x2)^-1)
function SIR_sr(du, u, p, t)
    γ = 1 / 10
    N = 14570000.0
    S, I, R, H = u
    du[1] = - S * f(I,R) / N
    du[2] = S * f(I,R) / N - γ * I
    du[3] = γ * I
    du[4] = S * f(I,R) / N
end
u_0 = Float32[14570000, 1, 1, 1]
tspan_data = (0.0f0, 150.0f0)
prob_sr = ODEProblem(SIR_sr, u_0, tspan_data,p=nothing)
data_prediction = Array(solve(prob_sr, Vern7(), saveat=1))
scatter(data_acc,label="Real accumulated cases")
plot!(data_prediction[4,:],label="Fit accumulated cases")

mid = zeros(length(data_acc))
mid[2:end] = data_prediction[4,1:end - 1]
pred_daily = data_prediction[4,:] - mid
scatter(data_daily, label = "Real daily confirmed")
plot!(pred_daily,label="Fit daily confirmed by NN",lw =2)