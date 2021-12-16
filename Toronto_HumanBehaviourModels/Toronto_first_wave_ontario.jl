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
using BSON:@save, @load
using Random
using Dates
Random.seed!(14);
# Load the source data
source_data = DataFrame(CSV.File("2021Toronto_HumanBehaviourModels/CityofToronto_COVID-19_RecoveryData.csv"))
province_name, n, m, popu = "Toronto", 0,154,2930000
data_on = source_data[:, "7 day moving average"]
data_on_acc = [sum(data_on[1:i]) for i = 1: length(data_on)]
start_data = Date(2020, 03, 01) + Dates.Day(n)
data_daily = data_on[(n + 1):(n + m +1)]
data_acc = data_on_acc[(n + 1):(n + m + 1)]
xlabel_str = "Days after $start_data"
source_data_label_daily = "Daily Confirmed Cases"
source_data_title_daily = string("Daily Confirmed Cases of ", province_name)
source_data_savefig_daily = string("./2021Toronto_HumanBehaviourModels/Saving_Data/", province_name, "/", province_name, "_daily_cases.png")
source_data_label_acc = "Accumulated Confirmed Cases"
source_data_title_acc = string("Accumulated Cases of ", province_name)
source_data_savefig_acc = string("./2021Toronto_HumanBehaviourModels/Saving_Data/", province_name, "/", province_name, "_accumulated_cases.png")
# Save the Source data
display(plot(data_daily, label=source_data_label_daily, xlabel=xlabel_str, title=source_data_title_daily, lw=2))
savefig(source_data_savefig_daily)
display(plot(data_acc, label=source_data_label_acc, xlabel=xlabel_str, title=source_data_title_acc, lw=2))
savefig(source_data_savefig_acc)
data_daily[1]
println(length(data_acc))

# Bayesia Inference
# Model generation
function SIR_pred(du, u, p, t)
    a, b, c, d = p
    γ = 1 / 10
    N = popu
    S, I, R, H = u
    Foi = max(0,a * I + (b* I+d)/(c*R+ 1))
    du[1] = - S* Foi / N
    du[2] =  S * Foi / N - γ * I
    du[3] = γ * I
    du[4] = S * Foi / N
end
# u_0 = [14570000, 1, 1, 1]
u_0 = [popu, data_daily[1], 1, data_acc[1]]
@model function fitSIR(data, prob1) # data should be a Vector
    σ ~ InverseGamma(2, 3) # ~ is the tilde character
    a ~ truncated(Normal(0.4, 0.5), 0, 1)
    b ~ truncated(Normal(1, 0.5), 0, 10)
    c ~ truncated(Normal(2.799337206675763e-5, 0.00001), 0, 1)
    d ~ truncated(Normal(0, 1), -100, 100)
    p = [a,b,c,d]
    prob = remake(prob1, p=p)
    predicted = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat=0:1:length(data) - 1)

    for i = 1:length(predicted)
        data[i] ~ Normal(predicted[i][4], σ)
            # predicted[i][2] is the data for y - a scalar, so we use Normal instead of MvNormal
    end
end
# Online Learning
Time_learn  = 20:10:m
p_0 = [0.2,1,0.001,0]
parameter_saving = DataFrame(A=Float64[], B=Float64[], C=Float64[])
t_max = m
tspan_learn = (0.0, t_max)
prob_pred = ODEProblem(SIR_pred, u_0, tspan_learn, p_0)
data_to_learn = data_acc[1:t_max + 1]
data_daily_to_learn = data_daily[1:t_max + 1]
# chain data strings
chain_data_savefig = string("./2021Toronto_HumanBehaviourModels/Saving_Data/", province_name, "/", province_name, "_chain$t_max.png")
# plot datasaving


function train(θ)
    prob_pred_train = remake(prob_pred, p=θ)
    Array(solve(prob_pred_train, Vern7(), abstol=1e-12, reltol=1e-12, saveat=0:1:t_max))
end
function loss(θ, p)
    pred = train(θ)
    mid = zeros(length(data_to_learn))
    mid[2:end] = pred[4,1:end - 1]
    pred_daily = pred[4,:] - mid
    sum(abs2, (log.(data_to_learn) .- log.(pred[4,:]))) #+ sum(abs2, (log.(data_daily_to_learn .+ 1) .- log.(pred_daily .+ 1))) # + 1e-5*sum(sum.(abs, params(ann)))
end
println(loss([0.2,1,0.001,0], p_0))
lb = [0.0001, 0.000001,0.000001,-100]
ub = [1,10,1,100]
using GalacticOptim:OptimizationProblem
using Optim
loss1 = OptimizationFunction(loss,GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(loss1, p_0, lb=lb, ub=ub)
sol1 = GalacticOptim.solve(prob, IPNewton())
p_min = sol1.u
# p_0 = p_min
Turing.setadbackend(:forwarddiff)
model = fitSIR(data_to_learn, prob_pred)
chain = sample(model, NUTS(.45), MCMCThreads(), 2000, 3, progress=false)#, init_theta=sol1.u)
plot(chain)
savefig(chain_data_savefig)
p_min = [mean(chain[:a]),mean(chain[:b]),mean(chain[:c]),mean(chain[:d])]
push!(parameter_saving, p_min)
println("$t_max data parameter:", p_min)
p_0 = p_min
tspan_predict = (0.0, m)
scatter(data_to_learn, label="Training data")
plot!(data_acc, label="Real accumulated cases")
prob_prediction = ODEProblem(SIR_pred, u_0, tspan_predict, p_min)
data_prediction = Array(solve(prob_prediction, Tsit5(), saveat=1))

    
Fit_data_title_acc = string(province_name, "Accumulated Cases Train by $t_max days data")
Fit_data_savefig_acc = string("./2021Toronto_HumanBehaviourModels/Saving_Data/", province_name, "/", province_name, "_Fit_accumulated_cases_by$t_max.png")

Fit_data_title_daily = string(province_name, "_Daily Cases Train by $t_max days data")
Fit_data_savefig_daily = string("./2021Toronto_HumanBehaviourModels/Saving_Data/", province_name, "/", province_name, "_Fit_daily_cases_by$t_max.png")

display(plot!(data_prediction[4,:], label="Predicted accumulated cases", xlabel=xlabel_str, title=Fit_data_title_acc, lw=2))
savefig(Fit_data_savefig_acc)
mid = zeros(length(data_acc))
mid[2:end] = data_prediction[4,1:end - 1]
pred_daily = data_prediction[4,:] - mid
pred_daily[1] =data_daily[1]
scatter(data_daily, label="Real accumulated cases")
display(plot!(pred_daily, label="Predicted Daily cases", xlabel=xlabel_str, title=Fit_data_title_daily, lw=2))
savefig(Fit_data_savefig_daily)
para_save_path = string("./2021Toronto_HumanBehaviourModels/Saving_Data/", province_name, "/", province_name, "_parameters.csv")
CSV.write(para_save_path, parameter_saving)