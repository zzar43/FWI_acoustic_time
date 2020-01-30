# differential operator
function dx_central(A, Nx, Ny, dx)
    B = zeros(Nx, Ny);
    B[2:end-1, 2:end-1] = (A[3:end,2:end-1] - A[1:end-2,2:end-1])./(2*dx);
    return B
end
function dy_central(A, Nx, Ny, dy)
    B = zeros(Nx, Ny);
    B[2:end-1, 2:end-1] = (A[2:end-1,3:end] - A[2:end-1,1:end-2])./(2*dy);
    return B
end
function dx_forward(A, Nx, Ny, dx)
    B = zeros(Nx, Ny);
    B[2:end-1, 2:end-1] = (A[2:end-1,2:end-1] - A[1:end-2,2:end-1])./(dx);
    return B
end
function dy_forward(A, Nx, Ny, dy)
    B = zeros(Nx, Ny);
    B[2:end-1, 2:end-1] = (A[2:end-1,2:end-1] - A[2:end-1,1:end-2])./(dy);
    return B
end
function laplace_central(A, Nx, Ny, dx, dy)
    B = zeros(Nx, Ny);
    B[2:end-1, 2:end-1] = (A[3:end,2:end-1] - 2*A[2:end-1,2:end-1] + A[1:end-2,2:end-1])./dx^2 + (A[2:end-1,3:end] - 2*A[2:end-1,2:end-1] + A[2:end-1,1:end-2])./dy^2;
    return B
end
function dx_forward_4th(A, Nx, Ny, dx)
    B = zeros(Nx, Ny);
    B[5:end-4, 5:end-4] = (-25/12)*A[5:end-4, 5:end-4] + (4)*A[6:end-3, 5:end-4] + (-3)*A[7:end-2, 5:end-4] + (4/3)*A[8:end-1, 5:end-4] + (-1/4)*A[9:end, 5:end-4]
    B = B ./ dx
    return B
end
function dy_forward_4th(A, Nx, Ny, dy)
    B = zeros(Nx, Ny);
    B[5:end-4, 5:end-4] = (-25/12)*A[5:end-4, 5:end-4] + (4)*A[5:end-4, 6:end-3] + (-3)*A[5:end-4, 7:end-2] + (4/3)*A[5:end-4, 8:end-1] + (-1/4)*A[5:end-4, 9:end]
    B = B ./ dy
    return B
end
function dx_central_4th(A, Nx, Ny, dx)
    B = zeros(Nx, Ny);
    B[3:end-2,3:end-2] = (1/12)*A[1:end-4,3:end-2] + (-2/3)*A[2:end-3,3:end-2] + (2/3)*A[4:end-1,3:end-2] + (-1/12)*A[5:end,3:end-2]
    B = B ./ dx
    return B
end
function dy_central_4th(A, Nx, Ny, dy)
    B = zeros(Nx, Ny);
    B[3:end-2,3:end-2] = (1/12)*A[3:end-2,1:end-4] + (-2/3)*A[3:end-2,2:end-3] + (2/3)*A[3:end-2,4:end-1] + (-1/12)*A[3:end-2,5:end]
    B = B ./ dy
    return B
end
function laplace_central_4th(A, Nx, Ny, dx, dy)
    B1 = zeros(Nx, Ny);
    B2 = zeros(Nx, Ny);
    B1[3:end-2,3:end-2] = (-1/12)*A[1:end-4,3:end-2] + (4/3)*A[2:end-3,3:end-2]  + (-5/2)*A[3:end-2,3:end-2]  + (4/3)*A[4:end-1,3:end-2] + (-1/12)*A[5:end,3:end-2]
    B1 = B1 ./ dx^2
    B2[3:end-2,3:end-2] = (-1/12)*A[3:end-2,1:end-4] + (4/3)*A[3:end-2,2:end-3]  + (-5/2)*A[3:end-2,3:end-2]  + (4/3)*A[3:end-2,4:end-1] + (-1/12)*A[3:end-2,5:end]
    B2 = B2 ./ dy^2
    B = B1 + B2
    return B
end

# Ricker function for source
function source_ricker(center_fre, center_time, t)
    x = (1 .- 2*pi^2 .* center_fre^2 .* (t.-center_time).^2) .* exp.(-pi^2*center_fre^2 .* (t .- center_time).^2);
end

# extension of the domain
function extend_vel(vel, Nx_pml, Ny_pml, pml_len)
    vel_ex = zeros(Nx_pml, Ny_pml);
    vel_ex[pml_len+1:end-pml_len, pml_len+1:end-pml_len] .= vel;
    for i = 1:pml_len
        vel_ex[i,:] = vel_ex[pml_len+1,:];
        vel_ex[end-i+1,:] = vel_ex[end-pml_len,:];
        vel_ex[:,i] = vel_ex[:,pml_len+1];
        vel_ex[:,end-i+1] = vel_ex[:,end-pml_len];
    end
    return vel_ex
end

# core solver space 4 time 1
function acoustic_eq_solver(c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100)

    if size(receiver_position,2) != 2
        error("Please check the size of receiver_position.")
    end
    if size(source_position,2) != 2
        error("Please check the size of source_position.")
    end
    if ndims(source_position) == 1
        source_num = 1
    else
        source_num = size(source_position,1)
    end
    if ndims(receiver_position) == 1
        receiver_num = 1;
    else
        receiver_num = size(receiver_position,1)
    end

    # prepare PML coef
    pml_value = range(0, stop=pml_coef, length=pml_len);
    Nx_pml = Nx + 2*pml_len;
    Ny_pml = Ny + 2*pml_len;
    source_position_pml = source_position .+ pml_len;
    source_position_pml_vec = zeros(Int, source_num)
    if source_num == 1
        source_position_pml_vec[1] = source_position_pml[1] + (source_position_pml[2]-1)*Nx_pml
    else
        for i = 1:source_num
            source_position_pml_vec[i] = source_position_pml[i,1] + (source_position_pml[i,2]-1)*Nx_pml
        end
    end
    receiver_position_pml = receiver_position .+ pml_len;
    receiver_position_pml_vec = zeros(Int, receiver_num)
    if receiver_num == 1
        receiver_position_pml_vec[1] = receiver_position_pml[1] + (receiver_position_pml[2]-1)*Nx_pml
    else
        for i = 1:receiver_num
            receiver_position_pml_vec[i] = receiver_position_pml[i,1] + (receiver_position_pml[i,2]-1)*Nx_pml
        end
    end
    sigma_x = zeros(Nx_pml, Ny_pml);
    sigma_y = zeros(Nx_pml, Ny_pml);
    for i = 1:pml_len
        sigma_x[pml_len+1-i,:] .= pml_value[i];
        sigma_x[pml_len+Nx+i,:] .= pml_value[i];
        sigma_y[:,pml_len+1-i] .= pml_value[i];
        sigma_y[:,pml_len+Ny+i] .= pml_value[i];
    end
    alpha = 1 * (sigma_x + sigma_y);
    beta = 1 * sigma_x .* sigma_y;
    r = -0.25 * alpha.^2 + beta;

    # extend coef
    c_ex = extend_vel(c, Nx_pml, Ny_pml, pml_len);
    rho_ex = extend_vel(rho, Nx_pml, Ny_pml, pml_len);
    a = 1 ./ (rho_ex .* c_ex.^2);
    b = 1 ./ rho_ex;

    # Anonymous Functions
    dx(x) = dx_central(x, Nx_pml, Ny_pml, h)
    dy(x) = dy_central(x, Nx_pml, Ny_pml, h)
    lp(x) = laplace_central(x, Nx_pml, Ny_pml, h, h)
    ss(t) = exp.(-1/2*alpha*t);

    # initialize
    dxb = dx(b);
    dyb = dy(b);
    u0 = zeros(Nx_pml,Ny_pml);
    u1 = zeros(Nx_pml,Ny_pml);
    u2 = zeros(Nx_pml,Ny_pml);
    vx0 = zeros(Nx_pml,Ny_pml);
    vx1 = zeros(Nx_pml,Ny_pml);
    vx2 = zeros(Nx_pml,Ny_pml);
    vy0 = zeros(Nx_pml,Ny_pml);
    vy1 = zeros(Nx_pml,Ny_pml);
    vy2 = zeros(Nx_pml,Ny_pml);
    U = zeros(Nx,Ny,Nt);
    data = zeros(Nt, receiver_num);
    t = range(0, step=dt, length=Nt);

    # main algorithm
    # n = 0, initial value
    p0 = zeros(Nx_pml, Ny_pml);
    p1 = zeros(Nx_pml, Ny_pml);
    p2 = zeros(Nx_pml, Ny_pml);
    ux0 = zeros(Nx_pml, Ny_pml);
    ux1 = zeros(Nx_pml, Ny_pml);
    ux2 = zeros(Nx_pml, Ny_pml);
    uy0 = zeros(Nx_pml, Ny_pml);
    uy1 = zeros(Nx_pml, Ny_pml);
    uy2 = zeros(Nx_pml, Ny_pml);
    
    for iter = 1:Nt-1
        s2 = ss(t[iter+1]);
        s1 = ss(t[iter]);
        
        L1_1 = dxb.*vx1 + dyb.*vy1;
        L1_2 = b.*(dx(vx1) + dy(vy1));
        L1_3 = dxb .* dx(s1.*u1) + dyb .* dy(s1.*u1);
        L1_4 = b .* lp(s1.*u1);
        L1 = L1_1 + L1_2 + L1_3 + L1_4;
        
        u2 = (2 .- dt^2*r).*u1 .- u0 + dt^2*L1./(a.*s1);
        u2[source_position_pml_vec] .+= source[iter,:]
        
        vx2 = (1 .- 0.5*dt*sigma_x).*vx1 - 0.5*dt*(sigma_x-sigma_y).*(dx(s2.*u2)+dx(s1.*u1));
        vx2 = vx2 ./ (1 .+ 0.5*dt*sigma_x);
        vy2 = (1 .- 0.5*dt*sigma_y).*vy1 - 0.5*dt*(sigma_y-sigma_x).*(dy(s2.*u2)+dy(s1.*u1));
        vy2 = vy2 ./ (1 .+ 0.5*dt*sigma_y);
        
    #     update
        u0 = copy(u1); u1 = copy(u2);
        vx1 = copy(vx2);
        vy1 = copy(vy2);
        
    #     record
        U[:,:,iter] .= s1[pml_len+1:end-pml_len, pml_len+1:end-pml_len] .* u1[pml_len+1:end-pml_len, pml_len+1:end-pml_len];
        data[iter,:] = u1[receiver_position_pml_vec];
    end
    
    s2 = ss(t[Nt]);
    U[:,:,end] .= s2[pml_len+1:end-pml_len, pml_len+1:end-pml_len] .* u2[pml_len+1:end-pml_len, pml_len+1:end-pml_len];
    data[end,:] = u2[receiver_position_pml_vec];

    return data, U
end

# function acoustic_eq_solver_s4t2(c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100, save_ratio=1, source_type=1)

#     # prepare PML coef
#     if ndims(source_position) == 1
#         source_num = 1
#     else
#         source_num = size(source_position,1)
#     end
#     if ndims(receiver_position) == 1
#         receiver_num = 1;
#     else
#         receiver_num = size(receiver_position,1)
#     end
#     pml_value = range(0, stop=pml_coef, length=pml_len);
#     Nx_pml = Nx + 2*pml_len;
#     Ny_pml = Ny + 2*pml_len;
#     source_position_pml = source_position .+ pml_len;
#     source_position_pml_vec = zeros(Int, source_num)
#     if source_num == 1
#         source_position_pml_vec[1] = source_position_pml[1] + (source_position_pml[2]-1)*Nx_pml
#     else
#         for i = 1:source_num
#             source_position_pml_vec[i] = source_position_pml[i,1] + (source_position_pml[i,2]-1)*Nx_pml
#         end
#     end
#     receiver_position_pml = receiver_position .+ pml_len;
#     receiver_position_pml_vec = zeros(Int, receiver_num)
#     if receiver_num == 1
#         receiver_position_pml_vec[1] = receiver_position_pml[1] + (receiver_position_pml[2]-1)*Nx_pml
#     else
#         for i = 1:receiver_num
#             receiver_position_pml_vec[i] = receiver_position_pml[i,1] + (receiver_position_pml[i,2]-1)*Nx_pml
#         end
#     end
#     sigma_x = zeros(Nx_pml, Ny_pml);
#     sigma_y = zeros(Nx_pml, Ny_pml);
#     for i = 1:pml_len
#         sigma_x[pml_len+1-i,:] .= pml_value[i];
#         sigma_x[pml_len+Nx+i,:] .= pml_value[i];
#         sigma_y[:,pml_len+1-i] .= pml_value[i];
#         sigma_y[:,pml_len+Ny+i] .= pml_value[i];
#     end

#     # extend coef
#     c_ex = extend_vel(c, Nx_pml, Ny_pml, pml_len)
#     rho_ex = extend_vel(rho, Nx_pml, Ny_pml, pml_len)

#     # initialize
#     t = range(0, length=Nt, step=dt)
#     a = 1 ./ (rho_ex .* c_ex.^2);
#     b = 1 ./ rho_ex;
#     alpha = 1 * (sigma_x + sigma_y);
#     beta = 1 * sigma_x .* sigma_y;
#     r = -0.25 * alpha.^2 + beta;
#     NNt = div(Nt, save_ratio)
#     P = zeros(Nx, Ny, NNt);
#     data = zeros(Nt, receiver_num);

#     # Anonymous Functions
#     dx(A) = dx_central_4th(A, Nx_pml, Ny_pml, h);
#     dy(A) = dy_central_4th(A, Nx_pml, Ny_pml, h);
#     lp(A) = laplace_central_4th(A, Nx_pml, Ny_pml, h, h);
#     ss(t) = exp(-0.5 .* alpha .* t);
#     dxb = dx(b);
#     dyb = dy(b);

#     # main algorithm
#     # n = 0, initial value
#     p0 = zeros(Nx_pml, Ny_pml);
#     p1 = zeros(Nx_pml, Ny_pml);
#     p2 = zeros(Nx_pml, Ny_pml);
#     ux0 = zeros(Nx_pml, Ny_pml);
#     ux1 = zeros(Nx_pml, Ny_pml);
#     ux2 = zeros(Nx_pml, Ny_pml);
#     uy0 = zeros(Nx_pml, Ny_pml);
#     uy1 = zeros(Nx_pml, Ny_pml);
#     uy2 = zeros(Nx_pml, Ny_pml);
    
#     velocity_x1 = zeros(Nx_pml,Ny_pml);
#     velocity_y1 = zeros(Nx_pml,Ny_pml);

#     # main loop
#     for iter = 1:Nt-1

#         s2 = ss(t[iter+1]);
#         s1 = ss(t[iter]);
        
#         dx_p1 = dx(s1.*p1)
#         dy_p1 = dy(s1.*p1)
        
#         # space = dx_b.*ux1 + dy_b.*uy1
#         # space = space + b .* (dx(ux1) + dy(uy1))
#         # space = space + dx_b .* dx_p1 + dy_b .* dy_p1
#         # space = space + b .* laplace_central_4th(s1.*p1, Nx_pml, Ny_pml, h, h)

#         L1_1 = dxb.*ux1 + dyb.*uy1;
#         L1_2 = b.*(dx(ux1) + dy(uy1));
#         L1_3 = dxb .* dx(s1.*p1) + dyb .* dy(s1.*p1);
#         L1_4 = b .* lp(s1.*p1);
#         L1 = L1_1 + L1_2 + L1_3 + L1_4;

#         p2 = 2*p1 - p0 - dt^2 .* r.*p1 + (dt^2 ./ (a.*s1)) .* L1;
# #         p2[source_position_pml_vec] .= dt^2 ./ (a[source_position_pml_vec].*s1[source_position_pml_vec]) .* source[iter,:]
#         if source_type == 1
#             p2[source_position_pml_vec] .+= source[iter,:]
#         elseif source_type == 2
#             p2[source_position_pml_vec] .= source[iter,:]
#         end
        
#         ux2 = dt/2*(sigma_y-sigma_x) .* (dx_p1 + dx(s2.*p2)) + (1 .- dt/2*sigma_x).*ux1
#         ux2 = ux2 ./ (1 .+ dt/2*sigma_x)
#         uy2 = dt/2*(sigma_x-sigma_y) .* (dy_p1 + dy(s2.*p2)) + (1 .- dt/2*sigma_y).*uy1
#         uy2 = uy2 ./ (1 .+ dt/2*sigma_y);
        
#         p0[:] = p1[:]; p1[:] = p2[:]
#         ux1[:] = ux2[:]
#         uy1[:] = uy2[:]
        
#         if rem(iter,save_ratio) == 0
# #             ss = exp.(-0.5 * alpha * (t[iter+1]));
#             P[:,:,div(iter,save_ratio)] = s2[pml_len+1:end-pml_len, pml_len+1:end-pml_len] .* p2[pml_len+1:end-pml_len, pml_len+1:end-pml_len];
#         end
#         data[iter,:] = s2[receiver_position_pml_vec] .* p2[receiver_position_pml_vec];
#     end

#     return data, P
# end

# solver for multiple sources
# function multi_solver(c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100)
#     source_num = size(source_position,1)
#     receiver_num = size(receiver_position,1)
    
#     data = zeros(Nt, receiver_num, source_num)
#     P = zeros(Nx, Ny, Nt, source_num)

#     for ind = 1:source_num
#         source1 = source[:,ind]
#         source_position1 = zeros(Int, 1, 2)
#         source_position1 = source_position[ind,:]
#         d, p = acoustic_eq_solver(c, rho, Nx, Ny, Nt, h, dt, source1, source_position1, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
#         data[:,:,ind] = d;
#         P[:,:,:,ind] = p;
#     end
#     return data, P
# end

# # solver for backward wavefield in adjoint method
# function backward_solver(c, rho, Nx, Ny, Nt, h, dt, adj_source, source_position, receiver_position; pml_len=10, pml_coef=100)
#     adj_source = adj_source[end:-1:1,:,:]
#     source_num = size(adj_source, 3)
#     # receiver_num = size(receiver_position,1)

#     # data = zeros(Nt, receiver_num, source_num)
#     P = zeros(Nx, Ny, Nt, source_num)

#     for ind = 1:source_num
#         source1 = adj_source[:,:,ind]
#         d, p = acoustic_eq_solver_s4t2(c, rho, Nx, Ny, Nt, h, dt, source1, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
#         P[:,:,:,ind] = p;
#     end
#     return P
# end

# # Parallel solver for wave equation

using Distributed, SharedArrays

function multi_solver_parallel(c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100)
    c = reshape(c, Nx, Ny)
    source_num = size(source_position,1)
    receiver_num = size(receiver_position,1)
    data = SharedArray{Float64}(Nt, receiver_num, source_num)
#     P = SharedArray{Float64}(Nx, Ny, Nt, source_num)

    @sync @distributed for ind = 1:source_num
        source_position_each = source_position[ind,:]
        d, p = acoustic_eq_solver(c, rho, Nx, Ny, Nt, h, dt, source, source_position_each', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        data[:,:,ind] = d;
#         P[:,:,:,ind] = p;
        p = []
    end
    
    data = Array(data)
#     P = Array(P)

#     return data, P
    return data
end

# # solver for backward wavefield in adjoint method with parallel computing
# function backward_solver_parallel(c, rho, Nx, Ny, Nt, h, dt, adj_source, source_position, receiver_position; pml_len=10, pml_coef=100, save_ratio=1, source_type=2)
#     c = reshape(c, Nx, Ny)
#     adj_source = adj_source[end:-1:1,:,:]
#     source_num = size(adj_source, 3)
    
#     NNt = div(Nt, save_ratio)
#     P = SharedArray{Float64}(Nx, Ny, NNt, source_num)

#     @sync @distributed for ind = 1:source_num
#         source1 = adj_source[:,:,ind]
#         d, p = acoustic_eq_solver_s4t2(c, rho, Nx, Ny, Nt, h, dt, source1, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef, save_ratio=save_ratio, source_type=source_type)
#         P[:,:,:,ind] = p;
#     end
    
#     P = Array(P)
    
#     return P
# end
