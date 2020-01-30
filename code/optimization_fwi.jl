using Printf

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

function gradient_descent(eval_fn, eval_grad, x0, alpha, iterNum, min_value, max_value; rho=0.9, c=1e-10, maxSearchTime=5, threshold=1e-10)

    xk = convert(Array{Float64,1}, x0[:])
    fn_value = zeros(iterNum+1)
    
    fk, gradk = eval_grad(xk)
    fn_value[1] = fk
    
    for iter = 1:iterNum
        println("Main iteration: ", iter)
        
        # Line search
        alpha0 = line_search_backtracking(eval_fn, xk, fk, gradk, alpha, min_value, max_value; rho=rho, c=c, maxSearchTime=maxSearchTime)
        # println(alpha0)
        if alpha0 == 0
            println("----------------------------------------------------------------")
            println("Line search Failed. Try decrease line search coef alpha. Interupt.")
            println("----------------------------------------------------------------")
            break
        else
            xk = update_fn(xk, alpha0, gradk, min_value, max_value)
        end
        
        # Compute gradient for next iteration
        fk, gradk = eval_grad(xk)
        fn_value[iter+1] = fk
        println("----------------------------------------------------------------")
        if fk <= threshold
            @printf "fk: %1.5e " fk
            println("Iteration is done.")
            println("----------------------------------------------------------------\n")
            break
        end
        if iter == iterNum 
            @printf "fk: %1.5e " fk
            println("Iteration is done. \n")
            println("----------------------------------------------------------------\n")
        end
    end

    return xk, fn_value
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
                println("----------------------------------------------------------------")
                println("Line search Failed. Try decrease line search coef alpha. Interupt.")
                println("----------------------------------------------------------------")
                break
            else
                xk = update_fn(xk, alpha0, gradk, min_value, max_value)
            end
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
            if iter == iterNum 
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
            println("----------------------------------------------------------------")
            println("Line search Failed. Try decrease line search coef alpha. Interupt.")
            println("----------------------------------------------------------------")
            break
        else
            rho_lbfgs[2:end] = rho_lbfgs[1:end-1]
            S[:,2:end] = S[:,1:end-1]
            Y[:,2:end] = Y[:,1:end-1]
            xk = copy(xk1)
            gradk = copy(gradk1)
            xk1 = update_fn(xk, alpha0, ggk, min_value, max_value)
            
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

# --------------------------------------------------------------------------------
# Optimization functions
# --------------------------------------------------------------------------------

function eval_grad_l2(data0, c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; save_ratio=1, pml_len=10, pml_coef=100, cutoff=0)
    x = reshape(c, Nx, Ny)
    if minimum(x) > 0
        data, u = multi_solver_parallel(x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; save_ratio=save_ratio, pml_len=pml_len, pml_coef=pml_coef)

        fk = 0.5 * norm(data - data0,2) ^ 2

        gradk = grad_l2_parallel(data, u, data0, x, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; save_ratio=save_ratio, pml_len=pml_len, pml_coef=pml_coef);
        
        if cutoff != 0
            gradk = reshape(gradk, Nx, Ny)
            gradk[1:cutoff,:] .= 0
        end
        gradk = reshape(gradk, Nx*Ny, 1)
    else
        fk = NaN
        gradk = zeros(Nx*Ny,1)
        print(" Wrong update, with entries of x <= 0. ")
    end
    
    return fk, gradk
end

function eval_fn_l2(data0, c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; save_ratio=1, pml_len=10, pml_coef=100)
    x = reshape(c, Nx, Ny)
    
    if minimum(x) > 0
        data, u = multi_solver_parallel(x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; save_ratio=save_ratio, pml_len=pml_len, pml_coef=pml_coef)
        fk = 0.5 * norm(data - data0,2) ^ 2
    else
        fk = NaN
        print(" Wrong update, with entries of x <= 0. ")
    end
    
    return fk
end

function eval_grad_OT_linear(data0, c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; save_ratio=1, pml_len=10, pml_coef=100, reg=5e-3, reg_m=1e2, reg_p=0, iterMax=1000, cutoff=0)
    x = reshape(c, Nx, Ny)
    if minimum(x) > 0
        data, u = multi_solver_parallel(x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; save_ratio=save_ratio, pml_len=pml_len, pml_coef=pml_coef)

        gradk, fk = grad_sinkhorn_parallel_linear(data, u, data0, x, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; reg_p=reg_p, pml_len=pml_len, pml_coef=pml_coef, reg=reg, reg_m=reg_m, iterMax=iterMax, verbose=false, save_ratio=save_ratio)
        
        if cutoff != 0
            gradk = reshape(gradk, Nx, Ny)
            gradk[1:cutoff,:] .= 0
        end
        gradk = reshape(gradk, Nx*Ny, 1)
    else
        fk = NaN
        gradk = zeros(Nx*Ny,1)
        print(" Wrong update, with entries of x <= 0. ")
    end
    
    return fk, gradk
end

function eval_fn_OT_linear(data0, c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; save_ratio=1, pml_len=10, pml_coef=100, reg=5e-3, reg_m=1e2, reg_p=0, iterMax=1000)
    x = reshape(c, Nx, Ny)
    if minimum(x) > 0
        t = range(0,step=dt,length=Nt)
        M = cost_matrix_1d(t,t)
        data, u = multi_solver_parallel(x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; save_ratio=save_ratio, pml_len=pml_len, pml_coef=pml_coef)
        adj_source, fk = adj_source_sinkhorn_parallel_linear(data, data0, M; reg_p=reg_p, reg=reg, reg_m=reg_m, iterMax=iterMax, verbose=false);
    else
        fk = NaN
        print(" Wrong update, with entries of x <= 0. ")
    end
    
    return fk
end

function eval_grad_OT_exp(data0, c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; save_ratio=1, pml_len=10, pml_coef=100, reg=5e-3, reg_m=1e2, reg_p=0, iterMax=1000, cutoff=0)
    x = reshape(c, Nx, Ny)
    if minimum(x) > 0
        data, u = multi_solver_parallel(x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; save_ratio=save_ratio, pml_len=pml_len, pml_coef=pml_coef)

        gradk, fk = grad_sinkhorn_parallel_exp(data, u, data0, x, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; reg_p=reg_p, pml_len=pml_len, pml_coef=pml_coef, reg=reg, reg_m=reg_m, iterMax=iterMax, verbose=false, save_ratio=save_ratio)
        
        if cutoff != 0
            gradk = reshape(gradk, Nx, Ny)
            gradk[1:cutoff,:] .= 0
        end
        gradk = reshape(gradk, Nx*Ny, 1)
    else
        fk = NaN
        gradk = zeros(Nx*Ny,1)
        print(" Wrong update, with entries of x <= 0. ")
    end
    
    return fk, gradk
end

function eval_fn_OT_exp(data0, c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; save_ratio=1, pml_len=10, pml_coef=100, reg=5e-3, reg_m=1e2, reg_p=0, iterMax=1000)
    x = reshape(c, Nx, Ny)
    if minimum(x) > 0
        t = range(0,step=dt,length=Nt)
        M = cost_matrix_1d(t,t)
        data, u = multi_solver_parallel(x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; save_ratio=save_ratio, pml_len=pml_len, pml_coef=pml_coef)
        adj_source, fk = adj_source_sinkhorn_parallel_exp(data, data0, M; reg_p=reg_p, reg=reg, reg_m=reg_m, iterMax=iterMax, verbose=false);
    else
        fk = NaN
        print(" Wrong update, with entries of x <= 0. ")
    end
    
    return fk
end


function LBFGS1(eval_fn, eval_grad, x0, alpha, iterNum, min_value, max_value; m=5, alpha_search=1, rho=0.1, c=1e-10, maxSearchTime=5, threshold=1e-10)
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
        error("Line search Failed.")
    end
    #     update
    xk1 = update_fn(xk, alpha0, gradk, min_value, max_value)
    fk1, gradk1 = eval_grad(xk1)
    fn_value[2] = fk1
    S[:,1] = xk1 - xk
    Y[:,1] = gradk1 - gradk
    rho_lbfgs[1] = 1 ./ (Y[:,1]' * S[:,1])
    
#     println("----------------------------------------------------------------")
#     println("Start LBFGS.")
#     println("----------------------------------------------------------------")
    
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
        if iter <= m
            ggk = gradk1[:]
            alpha_lbfgs_search = alpha0
        else
            ggk = copy(r)
            alpha_lbfgs_search = alpha_search
            println("LBFGS.")
        end
        alpha0 = line_search_backtracking(eval_fn, xk1, fk1, ggk, alpha_lbfgs_search, min_value, max_value; rho=rho, c=c, maxSearchTime=maxSearchTime)
        if alpha0 == 0
            println("----------------------------------------------------------------")
            println("Line search Failed. Try decrease line search coef alpha. Interupt.")
            println("----------------------------------------------------------------")
            break
        else
            rho_lbfgs[2:end] = rho_lbfgs[1:end-1]
            S[:,2:end] = S[:,1:end-1]
            Y[:,2:end] = Y[:,1:end-1]
            xk = copy(xk1)
            gradk = copy(gradk1)
            xk1 = update_fn(xk, alpha0, ggk, min_value, max_value)
            
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

    return xk, fn_value
end