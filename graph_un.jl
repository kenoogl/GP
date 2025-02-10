using Statistics

include("compute_un.jl")

global Data = Dict()
global Diff_v = Dict()
global Diff_k = []
global Vc = 10.0
global Digit_num = 8
global V = []


function set_Data(v, d)
    global V = v
    # println(size(v))
    # println(size(d))

    for i in 1:size(v)[1]
        Data[v[i]] = d[i]
    end
end

function set_Diff(v, d)
    for i=1:size(v)[1]
        Diff_v[v[i]] = d[i]
        push!(Diff_k, v[i])
    end
end

function ffunc(func, x)
    if func == "log"
        return log(x)
    elseif func == "e^"
        return exp(x)
    elseif func == "sin"
        return sin(x)
    elseif func == "cos"
        return cos(x)
    elseif func[1] == "^"
        return x^func[1]
    else
        println("not ffunc")
        return ""
    end
end

function my_div(f1, f2)
    if f2 != 0.0
        return f1 ./ f2
    else
        # f2が0の時、最小の値で割りたい
        ff = floatmin(Float32) # 最小の正規化数
        if ndims(f2) < 1
            if f1 == 1.0
                return -Inf
            elseif f1 == 0.0
                return 0.0
            else
                return fill(Inf, size(f1))
            end
        else
            f = copy(f2)
            f[f==0] = 0.0
            f[f==0] += ff
            return f1 ./ f
        end
    end
end

function diff_load(y, diff)
    if isa(y, Float32) || y == "C"
        return 0.0
    else
        k = copy(diff)
        if y == "u"
            data_y = copy(Data[y])
            gn = maximum(diff) - minimum(diff)
            if gn != 0
                x_index = findfirst(Diff_k.=="x")
                t_index = findfirst(Diff_k.=="t")
                if diff[x_index] > diff[t_index]
                    z = grad_x(data_y, Diff_v["x"], gn)
                    k[x_index] -= gn
                elseif diff[x_index] == 0
                    return 0.0
                else
                    z = grad_x(data_y', Diff_v["t"], gn)'
                    k[t_index] -= gn
                end
            else
                z = data_y
            end

            if maximum(k) > 0
                return grad_yx(z, [Diff_v[i] for i in Diff_k], k)
            else
                return z
            end
        end

        diff_x = deleteat!(k, findfirst(Diff_k.==y))
        if diff_x == 1 && all([i == 0 for i in k])
            return 1.0
        else
            return 0.0
        end
    end
end

function diff_copy(j)
    new = []
    for i in j
        push!(new, copy(i))
    end

    return new
end

function make_diff_list(diff_num)

    diff_num_list = [[[[0 for _ in 1:length(diff_num)] for _ in 1:2]]]
    diff_c = diff_num[:]
    ii = 1
    n = 1

    while any([diff_c[i]>0 for i in 1:length(diff_num)])
        while diff_c[ii] == 0
            ii += 1
        end

        # [修正]juliaでは1始まりなので、ii=1からループに入るが、
        # lengthはpythonと同じく配列の長さが返るので、
        # 比較結果がずれる。[>=] -> [>]へ修正
        if ii > length(diff_num)
            break
        end
        
        new_diff = []
        i = 1
        for j in diff_num_list[n]
            j1 = diff_copy(j)
            j1[2][ii] += 1
            push!(new_diff, j1)
            j2 = diff_copy(j)
            j2[1][ii] += 1
            push!(new_diff, j2)
            i += 1
        end

        push!(diff_num_list, new_diff)
        n += 1
        diff_c[ii] -= 1
    end

    return diff_num_list[end]
end

function diff_mul(formula_list, ope1, ope2, diff_num, dd)
    D = make_diff_list(diff_num)
    S = 0
    S1 = Dict()
    S2 = Dict()

    for i in D
        if string(i[1]) in keys(S1)
            s1 = S1[string(i[1])]
        else
            if sum(i[1]) == 0
                s1 = calculate(formula_list, ope1, dd)
            else
                s1 = diff_cal(formula_list, ope1, i[1], dd)
            end
            S1[string(i[1])] = s1
        end

        if string(i[2]) in keys(S2)
            s2 = S2[string(i[2])]
        else
            if sum(i[2]) == 0
                s2 = calculate(formula_list, ope2, dd)
            else
                s2 = diff_cal(formula_list, ope2, i[2], dd)
            end
            S2[string(i[2])] = s2
        end
        S = S .+ (s1 .* s2)
    end

    return S
end

function diff_div(formula_list, ope1, ope2, diff_num, dd)
    D = make_diff_list(diff_num)
    S = 0
    S1 = Dict()
    S2 = Dict()

    for i in D
        if string(i[1]) in keys(S1)
            s1 = S1[string(i[1])]
        else
            if sum(i[1]) == 0
                s1 = calculate(formula_list, ope1, dd)
            else
                s1 = diff_cal(formula_list, ope1, i[1], dd)
            end
            S1[string(i[1])] = s1
        end

        if string(i[2]) in keys(S2)
            s2 = S2[string(i[2])]
        else
            if sum(i[2]) == 0
                s2 = calculate(formula_list, ope2, dd)
            else
                n = sum(i[2])+1
                k = prod(-Vector(1:1+n))
                d = diff_cal(formula_list, ope2, i[2], dd)
                s2 = k .* d .^ n
            end
            S2[string(i[2])] = s2
        end
        S = S .+ my_div(s1, s2)
    end

    return S
end

function diff_func(func, x)
    if func == "sin"
        return cos(x)
    elseif func == "cos"
        return -sin(x)
    else
        return nothing
    end
end

function diff_cal(formula_list, index, diff_num, dd)
    strIndex = string(index)
    operater = formula_list[strIndex]

    if operater[1] == "load"
        dd[strIndex] = diff_load(operater[2], diff_num)
    
    elseif operater[1] == "mul"
        dd[strIndex] = diff_mul(formula_list, operater[2], operater[3], diff_num, dd)

    elseif operater[1] == "div"
        dd[strIndex] = diff_div(formula_list, operater[2], operater[3], diff_num, dd)

    elseif findfirst("d_d", string(operater[1])) !== nothing
        kk = diff_num
        kk[findfirst(Diff_k.==string(operater[1][end]))] += 1
        dd[strIndex] = diff_cal(formula_list, operater[2], kk, dd)

    elseif operater[1] == "add"
        dc1 = diff_cal(formula_list, operater[2], diff_num, dd)
        dc2 = diff_cal(formula_list, operater[3], diff_num, dd)
        dd[strIndex] = dc1 + dc2

    elseif operater[1] == "sub"
        dc1 = diff_cal(formula_list, operater[2], diff_num, dd)
        dc2 = diff_cal(formula_list, operater[3], diff_num, dd)
        dd[strIndex] = dc1 - dc2
    else
        println("diff_cal function use.")
        df = diff_func(operater[1] , calculate(formula_list, operater[2], dd))
        dc = diff_cal(formula_list, operater[2], diff_num, dd)
        dd[strIndex] = df * dc
    end

    
    return dd[strIndex]
end

function calculate(formula_list, index, dd)
    strIndex = string(index)

    if strIndex in keys(dd)
        return dd[strIndex]
    end

    operater = formula_list[strIndex]

    if operater[1] == "load"
        dd[strIndex] = load(operater[2])

    elseif operater[1] == "mul"
        c2 = calculate(formula_list, operater[2], dd)
        c3 = calculate(formula_list, operater[3], dd)
        dd[strIndex] =  c2.*c3

    elseif operater[1] == "div"
        if operater[2] == operater[3]
            dd[strIndex] = 1.0

            return dd[strIndex]
        end
        
        f = calculate(formula_list, operater[3], dd)
        ff = calculate(formula_list, operater[2], dd)
        dd[strIndex] = my_div(ff, f)

    elseif !isnothing(findfirst("d_d", string(operater[1])))
        diff = [0 for _ in Diff_k]
        # juliaでは辞書は順番の情報は持たないため、kが変わってしまう。setDiffにDiff_k追加
        k = findfirst(Diff_k.==string(operater[1][end]))
        diff[k] += 1
        dd[strIndex] = diff_cal(formula_list, operater[2], diff, dd)

    elseif operater[1] == "add"
        dd[strIndex] = calculate(formula_list, operater[2], dd) + calculate(formula_list, operater[3], dd)
    
    elseif operater[1] == "sub"
        dd[strIndex] = calculate(formula_list, operater[2], dd) - calculate(formula_list, operater[3], dd)
    
    else
        dd[strIndex] = ffunc(operater[1], calculate(formula_list, operater[2], dd))
    end

    return dd[strIndex]
end

function load(x)
    if !isnothing(findfirst(x.==keys(Data)))
        return Data[x]
    elseif x == "C"
        x = digits_designate((rand() - 0.5) * 2 * Vc)
    end

    if isa(x, Float32)
        return float32(x)
    else
        println("load miss")
        return nothing
    end
end

function digits_designate(float_num, str=nothing)
    if isnothing(str)
        return Float32(float_num)
    else
        if floor(float_num) == float_num
            return string(Int(floor(float_num)))
        else
            return string(float_num) * ".8f"
        end
    end
end

function detect_num(x)
    if occursin("*", string(x)) || occursin("/", string(x))
        return false
    else
        l = [sort!([string(x[i]),"A"])[1] != "A" for i in 1:lastindex(x)]
        return sum(l) == length(x)
    end
end

function sentence_function(func, x)
    if func == "log"
        return func * "(" * x * ")"
    elseif func == "e^" || func == "sin" || func == "cos"
        if !detect_num(x)
            return func * "(" * x * ")"
        else
            f = ffunc(func, Float32(x))
            return digits_designate(f, "True")
        end
    elseif func[1] == "^"
        return "(" * x * "^" + string(func[1:ned]) * ")"
    else
        println("not sentence_function")
        return ""
    end
end


function sentence_calculate(formula_list, index, l=nothing)
    strIndex = string(index)
    # 計算済みの同じ項かどうか
    if strIndex in keys(l)
        return l[strIndex]
    end

    # 定数項の時
    if strIndex in keys(formula_list["subexp"])
        f1 = formula_list["subexp"][strIndex]
        if ndims(f1) < 1
            if sum([isnan(ff*0) for ff in f1]) > 0
                l[strIndex] = string(f1)
            else
                l[strIndex] = digits_designate(f1, "True")
            end

            return l[strIndex]
        end
    end

    operater = formula_list[strIndex]

    if operater[1] == "load"
        l[strIndex] = string(operater[2])
        
        return l[strIndex]
    end

    if operater[1] == "mul"
        f1 = sentence_calculate(formula_list, operater[2], l)
        f2 = sentence_calculate(formula_list, operater[3], l)

        if detect_num(f1) && detect_num(f2)
            f1 = typeof(f1) == Float32 ? f1 : parse(Float32,f1)
            f2 = typeof(f2) == Float32 ? f2 : parse(Float32,f2)
            l[strIndex] = digits_designate(f1*f2, "True")
        elseif f1 == "0" || f1 == 0.0 || f2 == "0" || f2 == 0.0
            l[strIndex] = "0"
        elseif f1 == "1"
            l[strIndex] =  f2
        elseif f2 == "1"
            l[strIndex] =  f1
        elseif f1 == "NaN" || f2 == "NaN"
            l[strIndex] = "NaN"
        elseif length(f1) > length(f2)
            l[strIndex] =  "(" * string(f2)  * "*" * string(f1) * ")"
        else
            l[strIndex] =  "(" * string(f1)  * "*" * string(f2) * ")"
        end

    elseif operater[1] == "div"
        f1 = sentence_calculate(formula_list, operater[2], l)
        f2 = sentence_calculate(formula_list, operater[3], l)

        if f1 == f2
            if f1 == "0" || f1 == "Inf"
                l[strIndex] =  "NaN"
            else
                l[strIndex] =  "1"
            end
        elseif f1 == "0" || f2 == "Inf"
            l[strIndex] =  "0"
        elseif f1 == "Inf" || f2 == "0"
            l[strIndex] =  "Inf"
        elseif f1 == "NaN" || f2 == "NaN"
            l[strIndex] =  "NaN"
        elseif detect_num(f1) && detect_num(f2)
            f1 = typeof(f1) == Float32 ? f1 : parse(Float32,f1)
            f2 = typeof(f2) == Float32 ? f2 : parse(Float32,f2)
            l[strIndex] =  digits_designate(my_div(f1, f2), "True")
        elseif f2 == "1"
            l[strIndex] = f1
        elseif length(f2) > 3 && "{1 /" == f2[1:4]
            l[strIndex] =  "(" * string(f1) * "*" * string(f2[5:end-1]) * ")"
        else
            l[strIndex] =  "{" * string(f1) * " /" * string(f2) * "}"
        end
    
    elseif !isnothing(findfirst("d_d", string(operater[1])))
        f1 = sentence_calculate(formula_list, operater[2], l)
        if detect_num(f1)
            l[strIndex] = "0"
        elseif operater[1][end] == f1
            l[strIndex] = "1"
        else
            l[strIndex] =  string(operater[1]) * "[" * string(f1) * "]"
        end

    elseif operater[1] == "add"
        print( "add exist")
    
    elseif operater[1] == "sub"
        print( "sub exist")
    
    else
        l[strIndex] =  sentence_function(operater[1], sentence_calculate(formula_list, operater[2], l))
    end

    return l[strIndex]
end
