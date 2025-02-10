using Dates
using Statistics
using LinearAlgebra
using Random
using Printf
using SharedArrays
using Distributed
using TimerOutputs

include("txt_read.jl")
include("graph_un.jl")
include("compute_un.jl")
include("regression.jl")

const to = TimerOutput()

Limit    = 10 #node number max
Ope_Node = 1
O   = ["load", "mul", "div", "d_dx", "d_dt"]
OPE = ["*", "/", "d_dx", "d_dt"]
Num_island = procs()[1]
gn = 32
Importance_moji = "Keisu"

function num_return()
    return (rand()-0.5)*2*Vc
end

function choice_O(x=nothing)
    if isnothing(x)
        return O[rand(1:length(O))]
    else
        n = findfirst(O.==x)
        l = vcat(O[1:n-1], O[n+1:end])

        return l[rand(1:length(l))]
    end
end

function choice_V(x=nothing)
    if isnothing(x)
        v = V[rand(1:length(V))]
    else
        n = findfirst(V.==x)
        l = vcat(V[1:n-1], V[n+1:end])
        v = l[rand(1:length(l))]
    end

    if v == "C"
        return num_return()
    end

    return v
end

function make_element(operater, k)
    if operater == "load"
        return ["load", choice_V()]
    else
        return [operater, rand(1:k), rand(1:k)] 
    end
end

function make_element_0(v)
    if v in V
        return ["load", choice_V(v)]
    else
        return ["load", num_return()] 
    end
end

function mutate(dict, k)
    str_k = string(k)
    operater = dict[str_k][1]
    next = choice_O(operater)

    # 突然変異させるkeyが1番目のとき、loadを作成
    if k == 1
        dict[str_k] = make_element_0(dict[str_k][2])

    # 突然変異させる演算がloadの場合
    elseif operater == "load"
        if dict[str_k][2] in V
            dict[str_k] = make_element(next, k-1)
        else
            dict[str_k][2] = num_return()
        end

    # 突然変異させる演算がload以外の場合
    else
        if next == "load"
            dict[str_k] = make_element(next, k-1)
        else
            dict[str_k][1] = next
        end
    end

    return dict
end

function make_ope(dict, mutation)
    # 突然変異させる場合
    if mutation
        # 演算リストが2つ以上の場合keyの数をランダムに決定
        k = 1
        if ("2" in keys(dict))
            k = rand(1:dict["len"])
        end
        dict = mutate(dict, k)

    # 突然変異させない場合
    else
        # 演算リストに新しい演算を追加
        k = findmax(parse.(Int, keys(dict)))[1]+1
        dict[string(k)] = make_element(choice_O(), k-1)
    end

    return dict
end

function make_list()
    # 1つ目のoperaterは必ずloadになる
    dict1 = Dict{String, Any}("1" => make_element("load", 1))

    # ランダムに決定したkeyの数分演算のリストを作成
    N = rand(1:Limit*Ope_Node)+1
    while length(dict1) < N
        dict1 = make_ope(dict1, false)
    end
    dict1["len"] = N

    return dict1
end

function make_data(dd)
    # gn(32)個の親のislandデータを作成
    ret = Vector(undef, gn)
    for i in 1:gn
        ret[i] = create_island_data(make_list(), dd[1])
    end

    return ret
end

function time_print(start)
    dtime = now() - start
    @printf(F, "time : %3.2f seconds\n", dtime.value/1000)
end

function cross_over(dict1, dict2)
    # 小さいほうのkey数を取得し、入れ替えるkeyの数をランダムに決定する
    k = minimum([dict1["len"], dict2["len"]])
    key_n = k>2 ? rand(2:k) : 2

    new_dict1 = Dict()
    new_dict2 = Dict()

    # key_nまでの辞書を作成
    for i in 1:key_n
        new_dict1[string(i)] = dict2[string(i)]
        new_dict2[string(i)] = dict1[string(i)]
    end
    
    # key_n以降の辞書は入れ替えずに追加
    for i in key_n:dict1["len"]
        new_dict1[string(i)] = dict1[string(i)]
    end
    new_dict1["len"] = dict1["len"]

    for i in key_n:dict2["len"]
        new_dict2[string(i)] = dict2[string(i)]
    end
    new_dict2["len"] = dict2["len"]
    
    return new_dict1, new_dict2
end

function dict_copy(dict1)
    new = Dict()
    for i in keys(dict1)
        if sum([string(k) >= "a" for k in i]) == 0
            new[i] = dict1[i]
        end
    end
    new["len"] = dict1["len"]

    return new
end

function model_save(list1, population)
    for popu in population
        l = popu["complexity"]
        if l <= gn
            if list1[l]["fit"] < popu["fit"]
                list1[l] = popu
            end
        end
    end
end

function evolution(data, dict_list, list1)
    index = shuffle(Vector(1:gn))
    p_cross = 0.75
    p_muta = 0.01

    # 子のislandデータを作成
    child = []
    for i in 1:2:gn
        # いらない？
        parent = []
        push!(parent, dict_list[index[i]])
        # 75%の確率で子同士を入れ替える
        if p_cross > rand()
            child0, child1 = cross_over(dict_list[index[i]]["dict"], dict_list[index[i+1]]["dict"])
        else
            child0 = dict_copy(dict_list[index[i]]["dict"])
            child1 = dict_copy(dict_list[index[i+1]]["dict"])
        end

        # 1%の確率で子を突然変異させる（演算リストのoperaterを変化させる）
        if p_muta > rand()
            child0 = make_ope(child0, true)
        end
        if p_muta > rand()
            child1 = make_ope(child1, true)
        end

        push!(child, create_island_data(child0, data))
        push!(child, create_island_data(child1, data))
    end

    # 親+子のislandデータを評価・選定
    model_save(list1, LARS_UN(vcat(dict_list, child), data))

    # 次の親となるislandデータを作成
    next_dict = []
    fit = []
    for i in vcat(dict_list, child)
        k = 1
        while k < length(next_dict)
            if next_dict[k]["importance"] > i["importance"]
                k += 1
            else
                break
            end
        end

        while k < length(next_dict)
            if next_dict[k]["cor"] > i["cor"]
                k += 1
            else
                break
            end
        end
        insert!(next_dict, k, i)
        push!(fit, i["cor"])
    end

    # mean:平均、std:標本標準偏差
    return next_dict[1:gn], [mean(fit), std(fit)]
end

function solu_evo(data, island, fit_array, list1)
    # 100世代分式の生成・評価を行う
    temp = island
    for _ in 1:100
        island, fit = evolution(data, island, list1)

        push!(fit_array, fit)

        # いらない？
        for i in 1:size(temp)[1]
            temp[i] = island[i]
        end
    end
end

# ";" : キーワード引数
function best_model_print(list_fit, island_num; pri=true, RE=false)
    maxi = Dict{String, Any}("fit" => -100.0)
    for i in 1:size(list_fit)[1]
        if maxi["fit"] < list_fit[i]["fit"]
            for j in keys(list_fit[i])
                maxi[j] = list_fit[i][j]
            end
        end
    end

    if pri
        @printf(F, "island:%2d -> fit=%.10f, f=%s\n", island_num, maxi["fit"], maxi["formula"])
    end

    if RE
        return maxi
    end
end

function A_island(i_d)
    data        = i_d[1]
    island      = i_d[2]
    predict_map = i_d[3]
    island_num  = i_d[4]
    fit_array   = i_d[5]
    model_fits  = i_d[6]

    solu_evo(data, island, fit_array, model_fits)

    best_model_print(model_fits, island_num, pri=false)
    
    return [island, predict_map, fit_array, 0, model_fits]
end

function model_marge(data)
    front = data[1]
    for i in 2:Num_island
        for j in size(front)[1]
            if front[j]["fit"] < data[i][j]["fit"]
                for k in keys(data[i][j])
                    front[j][k] = data[i][j][k]
                end
            end
        end
    end

    return front
end

function trans_popu(island)
    mat = zeros(Num_island, gn)
    idn = Vector(1:Num_island)
    for i in 1:gn
        idn = shuffle(idn)
        mat[:, i] = convert.(Int, idn)
    end

    for j in 1:Num_island
        for i in 1:gn
            temp = island[j][i]
            mat_index = convert(Int, mat[j, i])
            island[j][i] = island[mat_index][i]
            island[mat_index][i] = temp
        end
    end
end

function island_model(data, start, gp_n=0)
    Maxi = -Inf
    Maxi_popu = nothing
    predict_map = Dict()
    fit_array = [[[]] for _ in 1:Num_island]

    # 親データの作成 1:直列処理, 2:pmapを用いた並列処理
    if Num_island == 1
        island = [make_data([data])]
    else
        island = pmap(make_data, [[data] for _ in 1:Num_island])
    end

    kaisu = 0
    fit_value = []
    front = []
    fit = fitness(data)
    model_fits = [Dict("fit"      => fit["Least_mean"], 
                        "formula" => "", 
                        "fits0"   => fit) for _ in 1:gn+1+3]
    while kaisu < 3
        kaisu += 1
        time_print(start)
        println(F, "generation : ", kaisu)

        # Inputデータの作成
        Input = [[data, island[i], predict_map, i, fit_array[i], model_fits] for i in 1:Num_island]
        
        # 式の推定を実行 1:直列, 2:pmap
        if Num_island == 1
            Result = [A_island(Input[1])]
        else
            Result = pmap(A_island, Input)
        end

        println(F, "compute result finish")
        kari = []
        for i in 1:Num_island
            island[i]    = Result[i][1]
            predict_map  = Result[i][2]
            fit_array[i] = Result[i][3]
            push!(kari, Result[i][5])
        end
        model_fits = model_marge(kari)

        Maxi_model = best_model_print(model_fits, -1; pri=true, RE=true)
        
        if Maxi < Maxi_model["fit"]
            Maxi = Maxi_model["fit"]
        end

        push!(fit_value, Maxi)
        push!(front, model_fits)

        trans_popu(island)
    end

    return fit_value, Maxi_popu, fit_array, front
end

function dis_whole_f(y, x)
    d = fit_compute(y, x)

    return isnan(d) ? -Inf : abs(d)
end

function node_f(dict1)
    l = Dict()
    f = sentence_calculate(dict1, dict1["len"], l)

    return f, l
end

function f_object_compute(d)
    sub_dict = Dict()
    sdu = calculate(d, d["len"], sub_dict)

    d["subexp"] = sub_dict
    if ndims(sdu) < 1
        return ones(size(U).-2) * Inf
    else
        return sdu[2:end-1, 2:end-1]
    end
end

function create_island_data(d, data)
    obj = f_object_compute(d)
    mf, ll = node_f(d)
    
    # x微分せずにt微分している場合, objをInfで再生成
    if isnothing(findfirst("d_dx", mf)) && (!isnothing(findfirst("d_dt", mf)))
        obj = ones(size(U)[1]-2, size(U)[2]-2)*Inf
    end

    # 欠損データがある場合
    ave = mean(obj)
    if isnan(ave*0.0)
        return Dict(
            "dict" => d,
            "formula" => string(ave),
            "complexity" => Inf, 
            "simu" => obj,
            "cor" => 0,
            "ave" => ave, 
            "var" => ave, 
            "l2norm" => ave, 
            "importance" => 0,
            "age" => 0,
            "score" => 0.0
        )
    end

    # 複雑度を算出 -> juliaではindex関数が同じ動作をしないため、ロジックを変更
    global count = 0
    for v in vcat(OPE, V)
        i = 1
        # 文字列が含まれなくなる（i=nothing）までループ
        while !isnothing(i)
            # fのi番目以降の文字列にvが含まれるか検索し、最初のindexを返す
            i = findnext(v, mf, i)
            # 含まれれば、iにその次のindexを設定し、countする。
            if !isnothing(i)
                i = i[1]+1
                global count += 1
            end
        end
    end

    return Dict(
        "dict" => d, 
        "formula" => string(mf), 
        "complexity" => count, 
        "simu" => obj, 
        "cor" => dis_whole_f(data, obj), 
        "ave" => ave, 
        "var" => Statistics.var(obj, corrected=false), 
        "l2norm" => norm(obj.-ave, 2), 
        "importance" => 0,
        "age" => 0,
        "score" => 0.0
    )
end

function RegInit(data)
    reg_init = Vector(undef, length(V))
    for i in 1:size(V)[1]
        reg_init[i] = create_island_data(Dict("1" => ["load", V[i]], "len" => 1), data)
    end
    set_Init(reg_init)
end

# 機械的に変換しただけ（テスト未実施）
function add_noize(U_ori, noize)
    if noize != 0.0
        uvar = std(U_ori)^2
        noize_var = uvar * abs(noize)^2
        U_ori += rand(Normal(0, noize_var), size(U_ori))
        println(F, "U_var:$uvar, noize_var:$noize_var, U+noize-U.var:", var(U_ori)-uvar)
    end

    if noize > 0.0
        u_no = zeros(size(U_ori))
        u_no += vcat((U_ori[:,1], U_ori[:,1:end]))
        u_no += vcat((U_ori[:,1:end], U_ori[:,end]))
        u_no += hcat((U_ori[1,:].reshape(-1,1), U_ori[1:end,:]))
        u_no += hcat((U_ori[2:end,:], U_ori[end, :].reshape(-1,1)))
        u_no += 4*U_ori
        U_ori = u_no/8.0
        un_var = var(U_ori)
        println(F, "move average , U+noize:$un_var, U+noize-U.var:", un_var-uvar)
    end

    return U_ori
end

function make_target(U, noize, XT)
    U_ori = add_noize(U, noize)
    T_min = XT["T_min"]
    X_min = XT["X_min"]
    Nt = XT["Nt"]
    Nx = XT["Nx"]

    postU = U_ori[X_min:X_min+Nx-1, T_min+1:T_min+Nt]       # [1:200, 3:102]
    # U   = U_ori[X_min:X_min+Nx-1, T_min:T_min+Nt-1]       # [1:200, 2:101]
    preU  = U_ori[X_min:X_min+Nx-1, T_min-1:T_min+Nt-2]     # [1:200, 1:100]
    dU = (postU - preU)/(2*delta_t)
    ret = dU[2:end-1, 2:end-1]                              # [1:198, 1:98]

    return ret
end

function bar_print(N, i, ta)
    bar = string("[","#"^i, "."^(N-i), "]")
    percentage = @sprintf("%3.1f", 100*i/N)
    p = "%"
    h = @sprintf("%2d", ((N-i)*ta)/3600)
    m = @sprintf("%02d", (((N-i)*ta)%3600)/60)
    s = @sprintf("%02d", (((N-i)*ta)%3600)%60)
    println("$bar : $percentage $p done. Remaining time is $h:$m:$s [h/m/s]")
end


function main(U, name, noize, XT, path, Loop)
    ST = now() # ms

    Fit_value = Vector(undef, Loop)
    model     = Vector(undef, Loop)
    Array     = Vector(undef, Loop)
    Front     = Vector(undef, Loop)
    bar_print(Loop, 0, 0.0)
    stave = 0.0

    # Loop回 方程式を推定
    for i = 1:Loop
        st = now()
        data = make_target(U, noize, XT)

        RegInit(data)
        
        println(F,"============================================================")
        println(F, "\nGeneration $i start")
        Fit_value[i], model[i], Array[i], Front[i] = island_model(data, now(), i)

        println(F, "save data : $i")
        Dtime = now() - ST # 最初からの時間
        @printf(F, "Total time : %5.3f sec. / Gen. %d time : %5.3f sec.\n", Dtime.value/1000, i, (now()-st).value/1000)

        stave = (stave*(i-1) + (now()-st).value/1000) / i # i回目までの平均
        
        if Num_island == 1
            # \x1b：エスケープコード, [1F：カーソルを1つ上に移動, [2K：行全体を削除
            print("\x1b[1F\x1b[2K")
        end
        bar_print(Loop, i, stave)
    end

    println(F,"\n============================================================")

    for i in 1:size(Front)[1]
        best_model_print(Front[i][end], i, pri=true)
    end

    Dtime = now() - ST
    @printf(F, "Elapsed time : %5.3f sec.\n", Dtime.value/1000)
    println("Elapsed time : ", Dtime.value/1000, " sec.")
end


function GP_Init(v, x, t)
    xu = x
    for _ in 2:size(U)[2]
        xu = hcat(xu, x)
    end

    tu = t
    for _ in 2:size(U)[1]
        tu = hcat(tu, t)
    end

    set_Data(v, [U, xu, tu'])
    set_Diff(["x", "t"], [delta_x, delta_t])

    println(F, "load data:$v, graph make variable:$V")
    println(F, "U:", size(U))
end


function GP_DATA(tmin, nt, xmin, nx, name, noize, path, Loop)

    @timeit to "GP_DATA" begin

        global F = open("JuliaLog.txt", "w")

        # Uori:32bit浮動小数点数の配列, v:ヘッダーデータ
        U_ori, v = read_txt(name)

        global delta_t = v["delta_t"][1]
        global delta_x = v["delta_x"][1]
        println(F, "High fit_value is good.")
        println(F, "Limit:", Limit, ", Ope_Node:", Ope_Node)
        println(F, "delta_t:", delta_t, ", delta_x:", delta_x)
        println(F, "tmin:", tmin, ", nt:", nt)
        println(F, "xmin:", xmin, ", nx:", nx)
        println(F, "operation:", OPE)
        println(F, name, " Load.")
        println(F, "config island:", Num_island, ", gn:", gn)
        println(F, "Importance determinate method:", Importance_moji)
        println(F, now())
        println(F, keys(v))

        Nt = nt
        Nx = nx
        T_min = tmin+1  # Juliaでは1始まりのため+1
        X_min = xmin+1  # Juliaでは1始まりのため+1

        if nt == 0
            Nt = size(U_ori, 1) - 2
            print(F, "Nt:", Nt)
        end
        if nx == 0
            Nx = size(U_ori, 2)
            print(F, "Nx:", Nx)
        end

        global U = U_ori[X_min:X_min+Nx-1, T_min:T_min+Nt-1]

        if "x" in keys(v)
            x = v["x"][X_min:X_min+Nx-1]
        else
            x = [(X_min+x-1)*delta_x for x in 1:Nx]
        end

        if "t" in keys(v)
            t = v["t"][T_min:T_min+Nt-1]
        else
            t = [(T_min+t-1)*delta_t for t in 1:Nt]
        end

        XT = Dict(
            "T_min" => T_min,
            "Nt"    => Nt,
            "X_min" => X_min,
            "Nx"    => Nx)

        v = ["u","x","t"]

        @timeit to "GP_Init" begin
            GP_Init(v, x, t)
        end
    
        @timeit to "main" begin
            main(U_ori, path*name, noize, XT, path, Loop)
        end
            
        close(F)
    end # "GP_DATA"
end

GP_DATA(1, 100, 0, 200, "simulation_burgers.txt", 0, "", 1) #just compile
GP_DATA(1, 100, 0, 200, "simulation_burgers.txt", 0, "", 50)
show(to)

#using Profile
#Profile.clear()
#GP_DATA(1, 100, 0, 200, "simulation_burgers.txt", 0, "", 1) # コンパイルのためのダミー
#@profile GP_DATA(1, 100, 0, 200, "simulation_burgers.txt", 0, "", 50)

#io=open("profile.txt", "w")
#Profile.print(IOContext(io, :displaysize => (24, 500)), noisefloor=2)
#close(io)

#using Profile, PProf
#Profile.init(n = 10^7, delay = 0.01)
#@profile GP_DATA(1, 100, 0, 200, "simulation_burgers.txt", 0, "", 1) #just compile
#Profile.clear()
#@profile GP_DATA(1, 100, 0, 200, "simulation_burgers.txt", 0, "", 50)
#pprof()

# GP_DATA(tmin, nt, xmin, nx, name, noize, path, Loop)

# diffusion,sinの場合、データがないため、nx=100で実行する必要がある。
# GP_DATA(1, 100, 0, 100, "simulation_diffusion.txt", 0, "", 50)
# GP_DATA(1, 100, 0, 100, "simulation_sin.txt", 0, "", 50)
# GP_DATA(1, 100, 0, 100, "simulation_box.txt", 0, "", 50)