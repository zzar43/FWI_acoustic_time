using Distributed, SharedArrays, LinearAlgebra, Printf, JLD2
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
        
        u = nothing
        utt = nothing
        v = nothing
    end

    grad = Array(grad)
    grad = sum(grad, dims=3)
    grad = grad[:,:,1]
    
#     cutoff
    grad[1:16,:] .= 0
    
    grad = reshape(grad, Nx*Ny, 1)
#     
#     grad = grad ./ maximum(abs.(grad))
    obj_value = sum(obj_value)
    return obj_value, grad
end

function nonlinear_cg(eval_fn, eval_grad, x0, alpha, iterNum, min_value, max_value; rho=0.9, c=1e-10, maxSearchTime=5, threshold=1e-10)
    xk = convert(Array{Float64,1}, x0[:])
    fn_value = zeros(iterNum+1)

    fk, gradk = eval_grad(xk)
    fn_value[1] = fk
    d0 = -gradk
    r0 = -gradk
    
    iter = 1
    println("Main iteration: ", iter)
    alpha0 = line_search_backtracking(eval_fn, xk, fk, -d0, alpha, min_value, max_value; rho=rho, c=c, maxSearchTime=maxSearchTime)

    if alpha0 == 0
        println("----------------------------------------------------------------")
        println("Line search Failed. Try decrease line search coef alpha. Interupt.")
        println("----------------------------------------------------------------")
    else

    #     update
        xk = update_fn(xk, alpha0, gradk, min_value, max_value)
        
        iter = 1
        file_name = "temp_data/data_iter_" * string(iter) * ".jld2"
        @save file_name xk gradk
        
    #     compute gradient for next iteration
        fk, gradk = eval_grad(xk)
        fn_value[2] = fk
        r1 = -gradk
        beta = (r1'*(r1-r0))/(r0'*r0)
        beta = max(beta[1], 0)
        d1 = r1 + beta*d0
        println("----------------------------------------------------------------")
        if beta == 0
            println("No CG direction.")
        else
            println("CG direction.")
        end
        for iter = 2:iterNum
            println("Main iteration: ", iter)
    #         line search
            alpha0 = line_search_backtracking(eval_fn, xk, fk, -d1, alpha, min_value, max_value; rho=rho, c=c, maxSearchTime=maxSearchTime)
    #         update
            if alpha0 == 0
                println("Line search Failed. Use fixed step.")
                xk = update_fn(xk, 0.1*alpha, -d1, min_value, max_value)
                file_name = "temp_data/data_iter_" * string(iter) * ".jld2"
                update_direction = -d1
                @save file_name xk update_direction
            else
                xk = update_fn(xk, alpha0, -d1, min_value, max_value)
                file_name = "temp_data/data_iter_" * string(iter) * ".jld2"
                update_direction = -d1
                @save file_name xk update_direction
            end
            if iter < iterNum
                r0[:] = r1[:]
                d0[:] = d1[:]
        #     compute gradient for next iteration
                fk, gradk = eval_grad(xk)
                fn_value[iter+1] = fk
                r1 = -gradk
                beta = (r1'*(r1-r0))/(r0'*r0)
                beta = max(beta[1], 0)
                d1 = r1 + beta*d0

                println("----------------------------------------------------------------")
                if beta == 0
                    println("No CG direction.")
                else
                    println("CG direction.")
                end
                if fk <= threshold
                    @printf "fk: %1.5e " fk
                    println("Iteration is done.")
                    println("----------------------------------------------------------------\n")
                    break
                end
            elseif iter == iterNum 
                @printf "fk: %1.5e " fk
                println("Iteration is done. \n")
                println("----------------------------------------------------------------\n")
            end
        end
    end

    return xk, fn_value
end

function LBFGS(eval_fn, eval_grad, x0, alpha, iterNum, min_value, max_value; m=5, alpha_search=1, rho=0.1, c=1e-10, maxSearchTime=5, threshold=1e-10)
    # lbfgs
    xk = convert(Array{Float64,1}, x0[:])
    fn_value = zeros(iterNum+1)
    n = length(xk)
    S = zeros(n, m)
    Y = zeros(n, m)
    alpha_lbfgs = zeros(m)
    rho_lbfgs = zeros(m)

    fk, gradk = eval_grad(xk)
    fn_value[1] = fk

    iter = 1
    println("Main iteration: ", iter)
    alpha0 = line_search_backtracking(eval_fn, xk, fk, gradk, alpha, min_value, max_value; rho=rho, c=c, maxSearchTime=maxSearchTime)
    if alpha0 == 0
        println("----------------------------------------------------------------")
        println("Line search Failed. Try decrease line search coef alpha. Interupt.")
        println("----------------------------------------------------------------")
        error("Change the initial step size.")
    end
    #     update
    xk1 = update_fn(xk, alpha0, gradk, min_value, max_value)
    iter = 1
    file_name = "temp_data/data_iter_" * string(iter) * ".jld2"
    @save file_name xk gradk
    fk1, gradk1 = eval_grad(xk1)
    fn_value[2] = fk1
    S[:,1] = xk1 - xk
    Y[:,1] = gradk1 - gradk
    rho_lbfgs[1] = 1 ./ (Y[:,1]' * S[:,1])
    
    println("----------------------------------------------------------------")
    println("Start LBFGS.")
    println("----------------------------------------------------------------")
    
    for iter = 2:iterNum
        println("Main iteration: ", iter)
        q = copy(gradk1[:])
        for i = 1:m
            alpha_lbfgs[i] = rho_lbfgs[i] * S[:,i]' * q
            q = q - alpha_lbfgs[i] * Y[:,i]
        end

        r = (S[:,1]'*Y[:,1])./(Y[:,1]'*Y[:,1]) * q
        for i = m:-1:1
            beta = rho_lbfgs[i] * Y[:,i]'* r
            r = r + S[:,i] * (alpha_lbfgs[i]-beta)
        end
        ggk = copy(r)
        alpha0 = line_search_backtracking(eval_fn, xk1, fk1, ggk, alpha_search, min_value, max_value; rho=rho, c=c, maxSearchTime=maxSearchTime)
        if alpha0 == 0
            println("Line search Failed. Use fixed step.")
            xk = update_fn(xk, 0.1*alpha, ggk, min_value, max_value)
            file_name = "temp_data/data_iter_" * string(iter) * ".jld2"
            @save file_name xk ggk
        else
            rho_lbfgs[2:end] = rho_lbfgs[1:end-1]
            S[:,2:end] = S[:,1:end-1]
            Y[:,2:end] = Y[:,1:end-1]
            xk = copy(xk1)
            gradk = copy(gradk1)
            xk1 = update_fn(xk, alpha0, ggk, min_value, max_value)
            file_name = "temp_data/data_iter_" * string(iter) * ".jld2"
            @save file_name xk ggk
            
            fk1, gradk1 = eval_grad(xk1)
            fn_value[iter+1] = fk1
            
            S[:,1] = xk1 - xk
            Y[:,1] = gradk1 - gradk
            rho_lbfgs[1] = 1 ./ (Y[:,1]' * S[:,1])
            println("----------------------------------------------------------------")
        end
        
        if fk1 <= threshold
            @printf "fk: %1.5e " fk1
            println("Iteration is done.")
            println("----------------------------------------------------------------\n")
            break
        end
        if iter == iterNum 
            @printf "fk: %1.5e " fk1
            println("Iteration is done. \n")
            println("----------------------------------------------------------------\n")
        end
    end

    return xk1, fn_value
end



# Tools function

function line_search_backtracking(eval_fn, xk, fk, gradk, alpha, min_value, max_value; rho=0.9, c=0.9, maxSearchTime=30)
    pk = -gradk
    xk = xk[:]
    @printf "Start line search. fk: %1.5e\n" fk
    xkk = update_fn(xk, alpha, gradk, min_value, max_value)
    fk1 = eval_fn(xkk)
    @printf "    alpha: %1.5e" alpha
    @printf "    fk1: %1.5e" fk1
    @printf "    fk-c*alpha*gradk^2: %1.5e\n" (fk + c*alpha*sum(gradk.*pk))
    
    searchTime = 0
    for iter = 1:maxSearchTime
        if fk1 <= (fk + c*alpha*sum(gradk.*pk))
            break
        end
        alpha = rho * alpha
        xkk = update_fn(xk, alpha, gradk, min_value, max_value)
        fk1 = eval_fn(xkk)   
        @printf "    alpha: %1.5e" alpha
        @printf "    fk1: %1.5e" fk1
        @printf "    fk-c*alpha*gradk^2: %1.5e\n" (fk + c*alpha*sum(gradk.*pk))
        searchTime += 1
    end
    
    if fk1 > fk + c*alpha*sum(gradk.*pk)
        println("Line search failed. Search time: ", searchTime, ". Try to decrease search coef c.")
        alpha = 0
    elseif fk1 == NaN
        println("Line search failed. Search time: ", searchTime, ". Function value is NaN.")
    else
        println("Line search succeed. Search time: ", searchTime, ".")
    end

    return alpha
end

function update_fn(xk, alphak, gradk, min_value, max_value)
    xk = xk[:]
    xk1 = xk - alphak * gradk
    if min_value != 0
        xk1[findall(ind->ind<min_value,xk1)] .= min_value
    end
    if max_value != 0
        xk1[findall(ind->ind>max_value,xk1)] .= max_value
    end
    return xk1
end