using Distributed, SharedArrays, LinearAlgebra
include("acoustic_solver.jl")

function obj_fn_l2_parallel(received_data, c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100);
    c = reshape(c, Nx, Ny)
    source_num = size(source_position,1)
    obj_value = SharedArray{Float64}(source_num)

    @sync @distributed for ind = 1:source_num
        source_position_each = source_position[ind,:]
        data_forward, u = acoustic_eq_solver(c, rho, Nx, Ny, Nt, h, dt, source, source_position_each', receiver_position; pml_len=pml_len, pml_coef=pml_coef);
        # evaluate the objctive function
        obj_value[ind] = 0.5 * norm(data_forward-received_data[:,:,ind],2).^2
    end
    obj_value = sum(obj_value)
    return obj_value
end

function grad_l2_parallel(received_data, c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100);
    c = reshape(c, Nx, Ny)
    source_num = size(source_position,1)
    grad = SharedArray{Float64}(Nx, Ny, source_num);
    obj_value = SharedArray{Float64}(source_num)

    @sync @distributed for ind = 1:source_num
        source_position_each = source_position[ind,:]
        data_forward, u = acoustic_eq_solver(c, rho, Nx, Ny, Nt, h, dt, source, source_position_each', receiver_position; pml_len=pml_len, pml_coef=pml_coef);
        # adjoint source
        adj_source = data_forward - received_data[:,:,ind]
        adj_source = adj_source[end:-1:1,:]
        # adjoint wavefield
        data_backward, v = acoustic_eq_solver(c, rho, Nx, Ny, Nt, h, dt, adj_source, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
        utt = 0 .* u
        utt[:,:,2:end-1] = (u[:,:,3:end]-2u[:,:,2:end-1]+u[:,:,1:end-2]) ./ dt^2;
        grad0 = 1 ./ c.^3 .* utt .* v[:,:,end:-1:1];
        grad0 = sum(grad0,dims=3)
        grad[:,:,ind] = grad0

        # evaluate the objctive function
        obj_value[ind] = 0.5 * norm(data_forward-received_data[:,:,ind],2).^2

    end

    grad = Array(grad)
    grad = sum(grad, dims=3)
    grad = grad[:,:,1]

    obj_value = sum(obj_value)

    grad = reshape(grad, Nx*Ny, 1)

    return obj_value, grad
end

function grad_l2(received_data, c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100);
    c = reshape(c, Nx, Ny)
    source_num = size(source_position,1)
    grad = zeros(Nx, Ny, source_num);
    obj_value = zeros(source_num)

    for ind = 1:source_num
        source_position_each = source_position[ind,:]
        data_forward, u = acoustic_eq_solver(c, rho, Nx, Ny, Nt, h, dt, source, source_position_each', receiver_position; pml_len=pml_len, pml_coef=pml_coef);
        # adjoint source
        adj_source = data_forward - received_data[:,:,ind]
        adj_source = adj_source[end:-1:1,:]
        # adjoint wavefield
        data_backward, v = acoustic_eq_solver(c, rho, Nx, Ny, Nt, h, dt, adj_source, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
        utt = 0 .* u
        utt[:,:,2:end-1] = (u[:,:,3:end]-2u[:,:,2:end-1]+u[:,:,1:end-2]) ./ dt^2;
        grad0 = 1 ./ c.^3 .* utt .* v[:,:,end:-1:1];
        grad0 = sum(grad0,dims=3)
        grad[:,:,ind] = grad0

        # evaluate the objctive function
        obj_value[ind] = 0.5 * norm(data_forward-received_data[:,:,ind],2).^2
    end

    grad = sum(grad, dims=3)
    grad = grad[:,:,1]
    obj_value = sum(obj_value)

    return obj_value, grad
end

# function adj_source_sinkhorn(data1, data2, M; reg=1e-3, reg_m=1e2, iterMax=100, verbose=false);
#     adj = 0 .* data1;
#     dist = 0

#     if length(size(data1)) == 2
#         for i = 1:size(data1,2)
#             f = data1[:,i]
#             g = data2[:,i]
#             T, aaa, d1 = unbalanced_sinkhorn_1d_signal(f, g, M, reg, reg_m; iterMax=iterMax, verbose=verbose)
#             adj[:,i] = aaa
#             dist += d1
#         end
#     elseif length(size(data1)) == 3
#         for i = 1:size(data1,2)
#             for j = 1:size(data1,3)
#                 f = data1[:,i,j]
#                 g = data2[:,i,j]
#                 T, aaa, d1 = unbalanced_sinkhorn_1d_signal(f, g, M, reg, reg_m; iterMax=iterMax, verbose=verbose)
#                 adj[:,i,j] = aaa
#                 dist += d1
#             end
#         end
#     else
#         error("Please check the dimension of data1")
#     end

#     return adj
# end
        
# function adj_source_sinkhorn_parallel(data1, data2, M; reg_p=0, reg=1e-3, reg_m=1e2, iterMax=50, verbose=false);
# #     mi1 = minimum(data1)
# #     mi2 = minimum(data2)
# #     mi = -1.1*min(mi1,mi2)
    
#     Nt = size(data1,1)
#     adj = 0 .* data1;
#     adj = SharedArray{Float64}(adj);
#     dist = zeros(size(data1,2), size(data1,3))
#     dist = SharedArray{Float64}(dist);
    

#     if length(size(data1)) == 2
#         @sync @distributed for i = 1:size(data1,2)
#             f = data1[:,i]
#             g = data2[:,i]
#             T, aaa, d1 = unbalanced_sinkhorn_1d_signal_linear(f, g, M, reg, reg_m; reg_p=reg_p, iterMax=iterMax, verbose=verbose)
#             adj[:,i] = aaa
#             dist[i] = d1
#         end
#     elseif length(size(data1)) == 3
#         @sync @distributed for i = 1:size(data1,2)
#             for j = 1:size(data1,3)
#                 f = data1[:,i,j]
#                 g = data2[:,i,j]
#                 T, aaa, d1 = unbalanced_sinkhorn_1d_signal_linear(f, g, M, reg, reg_m; reg_p=reg_p, iterMax=iterMax, verbose=verbose)
#                 adj[:,i,j] = aaa
#                 dist[i,j] = d1
#             end
#         end
#     else
#         error("Please check the dimension of data1")
#     end
#     adj = Array(adj)
#     dist = sum(dist)

#     return adj, dist
# end

# function adj_source_sinkhorn_parallel_bal(data1, data2, M; reg_p=0, reg=1e-3, iterMax=50, verbose=false);
    
#     Nt = size(data1,1)
#     adj = 0 .* data1;
#     adj = SharedArray{Float64}(adj);
#     dist = zeros(size(data1,2), size(data1,3))
#     dist = SharedArray{Float64}(dist);
    

#     if length(size(data1)) == 2
#         @sync @distributed for i = 1:size(data1,2)
#             f = data1[:,i]
#             g = data2[:,i]
#             T, aaa, d1 = sinkhorn_1d_signal_linear(f, g, M, reg; reg_p=reg_p, iterMax=iterMax, verbose=verbose)
#             adj[:,i] = aaa
#             dist[i] = d1
#         end
#     elseif length(size(data1)) == 3
#         @sync @distributed for i = 1:size(data1,2)
#             for j = 1:size(data1,3)
#                 f = data1[:,i,j]
#                 g = data2[:,i,j]
#                 T, aaa, d1 = sinkhorn_1d_signal_linear(f, g, M, reg; reg_p=reg_p, iterMax=iterMax, verbose=verbose)
#                 adj[:,i,j] = aaa
#                 dist[i,j] = d1
#             end
#         end
#     else
#         error("Please check the dimension of data1")
#     end
#     adj = Array(adj)
#     dist = sum(dist)

#     return adj, dist
# end

# function grad_l2(data, u, data0, c, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; pml_len=10, pml_coef=100);
# #     input:
# #     data1: received data
# #     c1, rho1: 
    
#     adj_source = data - data0
    
# #     adjoint wavefield
#     vl = backward_solver(c, rho, Nx, Ny, Nt, h, dt, adj_source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
    
#     uu = 0 .* u;
#     uu[:,:,2:end-1,:] = (u[:,:,3:end,:] - 2*u[:,:,2:end-1,:] + u[:,:,1:end-2,:]) / (dt^2);
#     gradl = uu[:,:,end:-1:1,:].*vl
#     gradl = sum(gradl, dims=[3,4])
#     gradl = gradl[:,:,1,1]
# #     gradl = gradl ./ (maximum(abs.(gradl)))
    
#     return gradl
# end

# function grad_sinkhorn(data, u, data0, c, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; pml_len=10, pml_coef=100, lambda=1000, numItermax=10, stopThr = 1e-6)
    
#     adj_source = adj_source_sinkhorn(data, data0; lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=false);
    
# #     adjoint wavefield
#     v = backward_solver(c, rho, Nx, Ny, Nt, h, dt, adj_source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
    
#     uu = 0 .* u;
#     uu[:,:,2:end-1,:] = (u[:,:,3:end,:] - 2*u[:,:,2:end-1,:] + u[:,:,1:end-2,:]) / (dt^2);
#     grad = uu[:,:,end:-1:1,:].*v
#     grad = sum(grad, dims=[3,4])
#     grad = grad[:,:,1,1]
    
#     return -grad
# end

# function grad_l2_parallel(data, u, data0, c, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; save_ratio=1, pml_len=10, pml_coef=100);
# #     input:
# #     data1: received data
# #     c1, rho1: 
#     c = reshape(c, Nx, Ny)
#     adj_source = data - data0
    
# #     adjoint wavefield
#     vl = backward_solver_parallel(c, rho, Nx, Ny, Nt, h, dt, -adj_source, source_position, receiver_position; save_ratio=save_ratio, pml_len=pml_len, pml_coef=pml_coef);
    
#     uu = 0 .* u;
#     uu[:,:,2:end-1,:] = (u[:,:,3:end,:] - 2*u[:,:,2:end-1,:] + u[:,:,1:end-2,:]) / (dt^2);
#     gradl = uu[:,:,end:-1:1,:].*vl
#     gradl = sum(gradl, dims=[3,4])
#     gradl = gradl[:,:,1,1] * dt
    
#     gradl = -2 ./ (rho .* c.^3) .* gradl
    
# #     gradl .= gradl / maximum(abs.(gradl))
# #     gradl = gradl ./ norm(gradl,2)

#     return gradl
# end
        
# function grad_sinkhorn_parallel(data, u, data0, c, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; cut=100, reg_p=0, pml_len=10, pml_coef=100, reg=5e-3, reg_m=1e2, iterMax=50, verbose=false, save_ratio=1)
#     c = reshape(c, Nx, Ny)
# #     Nt = size(data,1)
#     t = range(0,step=dt,length=Nt)
#     M = cost_matrix_1d(t,t)
    
#     adj_source, fk = adj_source_sinkhorn_parallel(data, data0, M; reg_p=reg_p, reg=reg, reg_m=reg_m, iterMax=iterMax, verbose=verbose);
        
#     adj_source[1:cut,:,:] .= 0

# #     adjoint wavefield
#     v = backward_solver_parallel(c, rho, Nx, Ny, Nt, h, dt, -adj_source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef, save_ratio=save_ratio);
    
#     uu = 0 .* u;
#     uu[:,:,2:end-1,:] = (u[:,:,3:end,:] - 2*u[:,:,2:end-1,:] + u[:,:,1:end-2,:]) / (dt^2);
#     grad = uu[:,:,end:-1:1,:].*v
#     grad = sum(grad, dims=[3,4])
#     grad = grad[:,:,1,1]
    
# #     grad .= grad / maximum(abs.(grad))
#     grad = grad ./ norm(grad,2)
#     return grad, fk
# end

# function grad_sinkhorn_parallel_bal(data, u, data0, c, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; reg_p=0, pml_len=10, pml_coef=100, reg=5e-3, reg_m=1e2, iterMax=50, verbose=false)
#     c = reshape(c, Nx, Ny)
#     Nt = size(data,1)
#     t = range(0,step=dt,length=Nt)
#     M = cost_matrix_1d(t,t)
    
#     adj_source, fk = adj_source_sinkhorn_parallel_bal(data, data0, M; reg_p=reg_p, reg=reg, iterMax=iterMax, verbose=verbose);
        
# #     adjoint wavefield
#     v = backward_solver_parallel(c, rho, Nx, Ny, Nt, h, dt, adj_source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
    
#     uu = 0 .* u;
#     uu[:,:,2:end-1,:] = (u[:,:,3:end,:] - 2*u[:,:,2:end-1,:] + u[:,:,1:end-2,:]) / (dt^2);
#     grad = uu[:,:,end:-1:1,:].*v
#     grad = sum(grad, dims=[3,4])
#     grad = grad[:,:,1,1]
    
#     grad = grad ./ norm(grad,2)
#     return grad, fk
# end