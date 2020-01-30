using MAT, Distributed, SharedArrays, JLD2
# addprocs(12)
@everywhere include("code/acoustic_solver.jl");
@everywhere include("code/adj_l2.jl")

@load "temp_data/model_data.jld2";
println("Data loaded.")

eval_fn(x) = obj_fn_l2_parallel(data_true, x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
eval_grad(x) = grad_l2_parallel(data_true, x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);

min_value = minimum(c_true)
max_value = maximum(c_true)
alpha = 1e-7
iterNum = 30
rho = 0.5
cc = 1e-10
maxSearchTime = 3
# x0 = reshape(c, Nx*Ny, 1);

@load "temp_data/ex5_1-20/data_iter_20.jld2"
x0 = copy(xk)

println("Start nonlinear CG.")
xk, fn = nonlinear_cg(eval_fn, eval_grad, x0, alpha, iterNum, min_value, max_value; rho=rho, c=cc, maxSearchTime=maxSearchTime, threshold=1e-10);

# println("Start LBFGS.")
# xk, fn = LBFGS(eval_fn, eval_grad, x0, alpha, iterNum, min_value, max_value; m=5, alpha_search=1, rho=0.5, c=1e-10, maxSearchTime=5, threshold=1e-10)

@save "temp_data/result.jld2" xk fn
println("Done.")