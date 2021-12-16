# Step 0: loading the Julia packages needed.
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
Random.seed!(14);
# Step 1: Loading the data
data_on = DataFrame(CSV.File("2021Toronto_HumanBehaviourModels/first_wave_data.csv"))
data_acc = data_on[:,"Acc"]
data_daily = data_on[:,"Daily"]
savetitle, popu, m = "Toronto_first_wave_ML_based_method_II",2930000,length(data_acc)-1


# Step 2： Define the universal differential equation
ann = FastChain(FastDense(2, 16, tanh, bias = false), FastDense(16, 1, bias = false)) # define NN()
p_0 = initial_params(ann)
c_0 = 1.0
p_00 =[c_0;p_0]
function SIR_nn(du, u, p, t)
    γ = 1 / 10
    N = popu
    S, I, c, H = u
    I_prim  = S * c / N - γ
    du[1] = - S * c * I / N
    du[2] = S * c * I  / N - γ * I
    du[3] = -c * abs(ann([I/N, I_prim],p)[1])
    du[4] = S * c * I / N
end

# Step 3： Train the model
## step 3.1 Define the loss function
tspan_data = (0.0f0, m)
function train(θ)
    u_0 = [popu, 1.0, abs(θ[1]), 1.0]
    prob_nn = ODEProblem(SIR_nn, u_0, tspan_data, θ[2:end])
    solvedata = Array(concrete_solve(prob_nn, Vern7(), u_0, θ[2:end], saveat=0:1:length(data_acc)-1,
                        sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end
function loss(θ)
    pred = train(θ)
    sum(abs2, (log.(data_acc) .- log.(pred[4,:]))), pred # + 1e-5*sum(sum.(abs, params(ann)))
end
@time print(loss(p_00)[1])
const losses = []
callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses) % 50 == 0
        println(losses[end])
    end
    false
end

## Step 3.2. train
@time res1_node = DiffEqFlux.sciml_train(loss, p_00, ADAM(0.06), cb=callback, maxiters=100)
@time res2_node = DiffEqFlux.sciml_train(loss, res1_node.minimizer, BFGS(initial_stepnorm=0.01), cb=callback1, maxiters=100)
ann_param = res2_node.minimizer

## Step 3.3： Save the  final results!
using BSON: @save
save_ann_archi = string("./2021Toronto_HumanBehaviourModels/Saving_Data/",savetitle,"_ann_archi.bason")
save_ann_para_1 = string("./2021Toronto_HumanBehaviourModels/Saving_Data/",savetitle,"_ann_para_1.bason")
save_ann_para_2 = string("./2021Toronto_HumanBehaviourModels/Saving_Data/",savetitle,"_ann_para_2.bason")
@save save_ann_archi ann
@save save_ann_para_1 ann_param
@time res3_node = DiffEqFlux.sciml_train(loss, res2_node.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters=200)
ann_param = res3_node.minimizer
@save save_ann_para_2 ann_param

## Step 3.4 Data visualization of the results
p_min = ann_param
scatter(data_acc,label="Real accumulated cases")
data_prediction = train(p_min)
plot!(data_prediction[4,:],label="Fit accumulated cases")

mid = zeros(length(data_acc))
mid[2:end] = data_prediction[4,1:end - 1]
pred_daily = data_prediction[4,:] - mid
pred_daily[1] = data_daily[1]
scatter(data_daily, label = "Real daily confirmed")
plot!(pred_daily,label="Fit daily confirmed by NN",lw =2)


## Step 4: recover the equations by Symbolic Regression.

xx= data_prediction[2,:]/14570000.0
plot(xx)
plot(data_prediction[3,:])
yy = data_prediction[1,:].*data_prediction[3,:]./14570000.0 .- 0.1
plot(yy)
f(x1,x2) = 0.07236328x1 + 1.555432(x1 - 1.7383432)*((5.9498844 + 0.0044679674x2)^-1)
zz = f.(xx,yy)

tspan = collect(0:1:150)'
ann_value = abs.(ann(tspan,ann_param))
plot(ann_value', label = "NN(t)")
plot!(zz,label = "0.07236328I + 1.555432(I - 1.7383432)*((5.9498844 + 0.0044679674R)^-1)")

X = [xx';yy']
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
