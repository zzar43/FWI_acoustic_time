using LinearAlgebra

# Cost matrix

function cost_matrix_1d(x, y; p=2)
    Nx = length(x)
    Ny = length(y)

    M = zeros(Nx,Ny)
    for i = 1:Nx
        for j = 1:Ny
            M[i,j] = (x[i] - y[j])^p
        end
    end
    return M
end

function cost_matrix_2d(x, y; p=2)
    # x, y: row vectors, for the domain
    Nx = length(x)
    Ny = length(y)
    X = repeat(x, 1, Ny)
    Y = repeat(y', Nx, 1)
    X = reshape(X, Nx*Ny, 1)
    Y = reshape(Y, Nx*Ny, 1)
    M = zeros(Nx*Ny, Nx*Ny)
    for i = 1:Nx*Ny
        for j = 1:Nx*Ny
            M[i,j] = sqrt((X[i]-X[j])^2 + (Y[i]-Y[j])^2)^p
        end
    end
    
    return M
end

# test functions
function gauss_func(t, b, c)
    y = exp.(-(t.-b).^2 ./ (2*c^2));
    return y
end

function sin_func(t, omega, phi)
    return sin.(2*pi*omega*(t .- phi));
end

function ricker_func(t, t0, sigma)
    t = t.-t0;
    f = (1 .- t.^2 ./ sigma.^2) .* exp.(- t.^2 ./ (2 .* sigma.^2));
    return f
end

function gaussian_2d(X,Y,center,sigma)
    g = exp.(-(X.-center[1]).^2 ./ (sigma[1]^2) -(Y.-center[2]).^2 ./ (sigma[2]^2))
    g = g ./ maximum(g)
    return g
end

function ricker_2d(X,Y,center,sigma)
    g = (1 .- (X.-center[1]).^2 ./ (sigma[1]^2) .- (Y.-center[2]).^2 ./ (sigma[2]^2)) .* exp.(-(X.-center[1]).^2 ./ (sigma[1]^2) -(Y.-center[2]).^2 ./ (sigma[2]^2))
    g = g ./ maximum(g)
    return g
end


# basic shinkhorn functions
function sinkhorn_1d(a, b, M, reg; iterMax=100, errThr=1e-2, verbose=true)
    a = a ./ norm(a,1)
    b = b ./ norm(b,1)
    
    Na = length(a)
    Nb = length(b)
    size_M = size(M)
    if Na != size_M[1] || Nb != size_M[2]
        error("Please check the size of cost matrix.")
    end
    
    # threshold for the non-zero part
    ind_a = findall(x->x>0, a)
    aa = a[ind_a]
    Naa = length(aa)
    MM = M[ind_a,:]
    K = exp.(-MM./reg)
    
    u = ones(Naa) ./ Naa
    v = ones(Nb) ./ Nb
    K_tilde = (1 ./ aa) .* K
    
    iter = 0
    err = 1
    
    while iter < iterMax
        u0 = copy(u)
        v0 = copy(v)

        v = b ./ (K' * u)
        u = 1 ./ (K_tilde * v)

        if any(isnan.(u)) || any(isnan.(v)) || any(isinf.(u)) || any(isinf.(v)) 
            if verbose == true
                println("Numerical error. Try to increase reg.")
            end
            u = copy(u0)
            v = copy(v0)
            break
        end

        iter += 1
    end
    # v = b ./ (K' * u)

    # coupling
    T = 0 .* M
    T[ind_a, :] = (u .* K .* v')

    # test converge
    a_test = T * ones(Nb)
    err = norm(a_test-a,2)
    if verbose==true && err>errThr
        println("Not converge. Try to increase iterMax.")
    end

    # gradient w.r.t. a. Normalized with sum(grad) = 0
    grad = zeros(Na)
    grad[ind_a] = reg*log.(u) .- (reg/Naa) * sum(log.(u))

    # p-Wasserstein distance ^ p
    dist = sum(T .* M)
    
    return T, grad, dist
end

# Unbalanced Sinkhorn OT
function unbalanced_sinkhorn_1d(a, b, M, reg, reg_m; iterMax=100, verbose=true)
    Na = length(a)
    Nb = length(b)
    size_M = size(M)
    if Na != size_M[1] || Nb != size_M[2]
    #     println("size error")
        error("Please check the size of cost matrix.")
    end

    # threshold for the non-zero part
    ind_a = findall(x->x>0, a)
    aa = a[ind_a]
    Naa = length(aa)
    MM = M[ind_a,:]
    K = exp.(-MM./reg)
    fi = reg_m / (reg + reg_m)
    
    u = ones(Naa) ./ Naa
    v = ones(Nb) ./ Nb
    u0 = zeros(Naa)
    v0 = zeros(Nb)
    # K_tilde = (1 ./ aa) .* K;
    
    iter = 0
    err = 1
    
    while iter < iterMax
        u0 = copy(u)
        v0 = copy(v)

        v = b ./ (K' * u)
        v = v .^ fi
        # u = 1 ./ (K_tilde * v)
        u = aa ./ (K * v)
        u = u .^ fi

        if any(isnan.(u)) || any(isnan.(v)) || any(isinf.(u)) || any(isinf.(v)) 
            if verbose == true
                println("Numerical error. Try to increase reg.")
            end
            u = copy(u0)
            v = copy(v0)
            break
        end

        iter += 1
    end
    # v = b ./ (K' * u)
    # v = v .^ fi

    # coupling
    T = 0 .* M
    T[ind_a, :] = (u .* K .* v')

    # gradient w.r.t. a.
    ff = reg*log.(u)
    temp1 = exp.(-ff./reg_m) .- 1
    grad = zeros(Na)
    grad[ind_a] = - reg_m * temp1

    # p-Wasserstein distance ^ p
    a1 = T * ones(Na)
    b1 = T'* ones(Nb)

    loga = a1.*log.(a1./a)
    loga[isnan.(loga)] .= 0
    kla = sum(loga - a1 + a)

    logb = b1.*log.(b1./b)
    logb[isnan.(logb)] .= 0
    klb = sum(logb - b1 + b)

    lTK = log.(T[ind_a,:]./K)
    lTK = T[ind_a,:] .* lTK
    lTK[isnan.(lTK)] .= 0
    lTK = lTK - T[ind_a,:] + K

    dist = reg * sum(lTK) + reg_m * (kla+klb)
#     dist = reg * sum(T .* M) + reg_m * (kla + klb)
#     dist = reg * sum(lTK)
    
    return T, grad, dist
end

# Signal case

function proj_p(f)
    N = length(f)
    Pp = zeros(N)
    p_ind = findall(x->x>=0, f)
    Pp[p_ind] .= 1
    Pp = diagm(Pp)
    
    return Pp
end

function proj_n(f)
    N = length(f)
    Pn = zeros(N)
    p_ind = findall(x->x<0, f)
    Pn[p_ind] .= -1
    Pn = diagm(Pn)
    
    return Pn
end

function sinkhorn_1d_signal(a, b, M, reg; iterMax=100, errThr=1e-2, verbose=false)
    Pap = proj_p(a)
    Pan = proj_n(a)
    Pbp = proj_p(b)
    Pbn = proj_n(b)

    ap = Pap * a
    an = Pan * a
    bp = Pbp * b
    bn = Pbn * b
    
    if maximum(ap) == 0
        ap = ones(length(a)) ./ length(a)
    end
    if maximum(an) == 0
        an = ones(length(a)) ./ length(a)
    end
    if maximum(bp) == 0
        bp = ones(length(b)) ./ length(b)
    end
    if maximum(bn) == 0
        bn = ones(length(b)) ./ length(b)
    end
    
    Tp, gp, dp = sinkhorn_1d(ap, bp, M, reg; iterMax=iterMax, errThr=errThr, verbose=verbose)
    Tn, gn, dn = sinkhorn_1d(an, bn, M, reg; iterMax=iterMax, errThr=errThr, verbose=verbose)
    T = Tp - Tn
    grad = Pap*gp + Pan*gn
    dist = dp + dn
    
    return T, grad, dist
end

function unbalanced_sinkhorn_1d_signal(a, b, M, reg, reg_m; iterMax=100, verbose=false)
    Pap = proj_p(a)
    Pan = proj_n(a)
    Pbp = proj_p(b)
    Pbn = proj_n(b)

    ap = Pap * a
    an = Pan * a
    bp = Pbp * b
    bn = Pbn * b
    
    if maximum(ap) == 0
        ap = ones(length(a)) ./ length(a)
    end
    if maximum(an) == 0
        an = ones(length(a)) ./ length(a)
    end
    if maximum(bp) == 0
        bp = ones(length(b)) ./ length(b)
    end
    if maximum(bn) == 0
        bn = ones(length(b)) ./ length(b)
    end

    Tp, gp, dp = unbalanced_sinkhorn_1d(ap, bp, M, reg, reg_m; iterMax=iterMax, verbose=verbose)
    Tn, gn, dn = unbalanced_sinkhorn_1d(an, bn, M, reg, reg_m; iterMax=iterMax, verbose=verbose)
    T = Tp - Tn
    # grad = Pap*gp + Pan*gn
    grad = Pap*gp + Pan*gn
    dist = dp + dn

    return T, grad, dist
end

###
function unbalanced_sinkhorn_1d_signal_linear(a, b, M, reg, reg_m; reg_p=0, iterMax=100, verbose=false)

    if reg_p == 0
        mi_a = minimum(a)
        mi_b = minimum(b)
        mi = min(mi_a, mi_b)
        a = a .- 1.1*mi
        b = b .- 1.1*mi
    else
        a = a .+ reg_p
        b = b .+ reg_p
    end

    T, grad, dist = unbalanced_sinkhorn_1d(a, b, M, reg, reg_m; iterMax=iterMax, verbose=verbose)
    return T, grad, dist
end
###

###
function sinkhorn_1d_signal_linear(a, b, M, reg; reg_p=0, iterMax=100, verbose=false)

    if reg_p == 0
        a = a .- minimum(a)
        b = b .- minimum(b)
    else
        a = a .+ reg_p
        b = b .+ reg_p
    end
    
    a = a ./ norm(a,1)
    b = b ./ norm(b,1)
    T, grad, dist = sinkhorn_1d(a, b, M, reg; iterMax=iterMax, verbose=verbose)
    return T, grad, dist
end
###