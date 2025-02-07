using Statistics

# 数値微分
function grad_x(y, dx, gN=1)
    if ndims(y) == 1
        println("grad y ndim != 1", size(y))
    end

    grad_num = zeros(Float32, size(y))
    if gN == 1
        bibun = (y[3:end,:] .- y[1:end-2,:]) ./ (2*dx)
        grad_num[2:end-1,:] = bibun
    elseif gN == 2
        bibun = (y[3:end,:] .- 2*y[2:end-1,:] .+ y[1:end-2,:]) ./ (dx^2)
        grad_num[2:end-1,:] = bibun
    elseif gN == 3
        bibun = (y[5:end,:] - 2*y[4:end-1,:] + 2*y[2:end-3,:] - y[1:end-4,:])/((2*dx)^3)
        grad_num[2,:] = (-3*y[6,:] + 14*y[5,:] -24*y[4,:] +18*y[4,:]-5*y[2,:])/((2*dx)^3)
        grad_num[end-2,:] = (3*y[end-6,:] - 14*y[end-5,:] +24*y[end-4,:] -18*y[end-3,:]+5*y[end-2,:])/((2*dx)^3)
        grad_num[3:end-2,:] = bibun
    elseif gN == 4
        bibun = (y[5:end,:] - 4*y[4:end-1,:] + 6*y[3:end-2,:] - 4*y[2:end-3,:] + y[1:end-4,:])/(dx^4)
        grad_num[2,:] = (y[4,:] - 4*y[3,:] + 6*y[2,:] - 4*y[1,:])/(dx^4)
        grad_num[end-2,:] = (3*y[end-2,:] - 4*y[end-3,:] + y[end-4,:])/(dx^4)
        grad_num[3:end-2,:] = bibun
    else
        k = gN
        grad_num = y
        while k > 0
            if k >= 4
                grad_num = grad_x(grad_num, dx, 4)
                k -= 4
            elseif k > 0
                grad_num = grad_x(grad_num, dx, k)
                k = 0
            end
        end
    end

    return grad_num
end

# 交差微分
function grad_yx(y, dyx, gN)
    if ndims(y) == 1
        println("grad y ndim != 1",size(y))
    end

    grad_num = zeros(Float32, size(y))

    if all([i == 1 for i in gN])
        bibun = ((y[3:end,3:end] - y[1:end-2,3:end])-(y[3:end,1:end-2] - y[1:end-2,1:end-2]))/(4*dyx[1]*dyx[2])
        grad_num[2:end-1,2:end-1] = bibun

    else
        grad_num = y
        while !all([i <= 0 for i in gN])
            grad_num = grad_yx(grad_num, dyx, [1,1])
            for j in 1:size(gN)[1]
                gN[j] -= 1
            end
        end
    end

    return grad_num
end

# 相関係数の絶対値
function fit_compute(y,x)
    x1 = x .- mean(x)
    y1 = y .- mean(y)
    xy = sqrt(sum(x1.^2)) * sqrt(sum(y1.^2))
    if xy != 0
        s = sum(x1.*y1) / xy
    else
        # print("soukankkeisuu l2norm y1:",sum(y1.^2), ", x1:", sum(x1.^2))
        return NaN
    end

    return abs(s)
end