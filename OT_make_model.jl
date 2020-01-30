using MAT, ImageFiltering
using Distributed, SharedArrays, JLD2
# addprocs(12)
@everywhere include("code/acoustic_solver.jl");

# vars = matread("marmousi_model/marmousi_dz10.mat")
vars = matread("marmousi_model/vp_5.mat")
println("Data loaded.")
println("===============================================")

c_true = get(vars,"vp_5",3)
c_true = c_true[1:6:end, 1:6:end]
c_true = c_true[:,150:450];

c = imfilter(c_true, Kernel.gaussian(15));
c0 = (c .- minimum(c)) ./ (maximum(c)-minimum(c)) * (maximum(c_true)-minimum(c_true)) .+ minimum(c_true);
c0[1:16,:] .= 1.5
c = copy(c0)
Nx, Ny = size(c_true)
println("Nx: ", Nx, ". Ny: ", Ny, ".")
rho = ones(Nx,Ny)
h = 30*1e-3

# time
Fs = 350;
dt = 1/Fs;
Nt = 1050;
t = range(0,length=Nt,step=dt);
println("CFL: ", maximum(c_true) * dt / h);

# source
source = source_ricker(5,0.2,t)
source_num = 11
source_position = zeros(Int,source_num,2)
for i = 1:source_num
        source_position[i,:] = [1 1+30(i-1)]
end
source = repeat(source, 1, 1);

# receiver
receiver_num = 101
receiver_position = zeros(Int,receiver_num,2)
for i = 1:receiver_num
    receiver_position[i,:] = [1, (i-1)*3+1]
end

# PML
pml_len = 20
pml_coef = 200;

println("Source number: ", source_num)
println("Receiver number: ", receiver_num)
println("===============================================")


println("Computing true data.")
@time data_true = multi_solver_parallel(c_true, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
println("True data computed.")

@save "temp_data/model_data.jld2" data_true c_true c rho Nx Ny Nt h dt source source_position receiver_position pml_len pml_coef
println("Data saved to temp_data/model_data.jld2")