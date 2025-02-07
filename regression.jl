using Statistics
using LinearAlgebra
#using LARS  #   modify 20250205 by keno
include("MyLARS.jl")


function set_Init(Island)
    global Init_Island = Island
end

function LARS_UN(island, Y)
    x = []
    sen = []
    index = []
    reject = ["0", "1", "NaN", "Inf"]
    Island = vcat(Init_Island, island)
    
    for i in 1:size(Island)[1]
        # シミュレーションデータの中に欠損データ（Inf）がひとつでもあれば、false
        tf1 = !any([isnan(0.0*ii) for ii in Island[i]["simu"]])
        # 指定の演算（formula）を除外する。
        tf2 = !(Island[i]["formula"] in vcat(sen, reject))
        tf3 = Island[i]["var"]>0
        if tf1 && tf2 && tf3
            Island[i]["importance"] = 1
            d = vec(Island[i]["simu"])
            k = 1
            d1 = d .- Island[i]["ave"]
            ds = sum(d1.^2)
            s = 1

            while k < length(x)+1
                if Island[i]["cor"] > Island[index[k]]["cor"]
                    s += 1
                end

                x1 = x[k] .- Island[index[k]]["ave"]
                keisu = abs(dot(d1, x1) / sqrt(ds * sum(x1.^2)))

                if keisu > 0.95
                    break
                end
                k += 1
            end

            if k == length(x)+1
                insert!(sen, s, Island[i]["formula"])
                insert!(x, s, d)
                insert!(index, s, i)
            elseif Island[i]["complexity"] < Island[index[k]]["complexity"]
                push!(reject, sen[k])
                Island[index[k]]["importance"] -= 1
                sen[k] = Island[i]["formula"]
                x[k] = d
                index[k] = i
            else
                push!(reject, Island[i]["formula"])
                Island[i]["importance"] -= 1
            end
        else
            Island[i]["importance"] = 0
        end
    end

    # xを転置
    xx = []
    for xi in 1:size(x)[1]
        xx = xi==1 ? x[1] : hcat(xx,x[xi])
    end
    
    # Yを1次元ベクトルに変換
    y = vec(Y)
    xx = Float32.(xx)
    y = Float32.(y)
    c = MyLARS.lars(xx, y)

    coef_path = c.coefs[:, end]
    coef = c.coefs[:, end]
    F = ""
    s = 0
    X = zeros(size(Y))
    i=1
    vn = [abs(i) for i in coef_path]
    vn = vn / sum(vn)
    vn_index = sortperm(vn)[end:-1:1]   # vnを逆順にする
    j = 1
    Im = ""
    x_ave = 0
    y_ave = mean(Y)
    pre_fit = -100
    fit = -50
    Reg = []
    while sum(vn[vn_index[1:j-1]]) < 0.95 && pre_fit < fit
        pre_fit = fit
        i = vn_index[j]
        vn_sum = sum(vn[vn_index[1:j]])
        naiseki = []
        if coef[i] != 0.0
            ii = index[i]
            Island[ii]["importance"] = 1 + vn[i]
            s += 1
            X += coef[i] * Island[ii]["simu"]
            Im *= @sprintf("%.3f+", vn[i])

            x_ave += coef[i] * Island[ii]["ave"]
            m = mean(Y .* X)
            a = (m - y_ave * x_ave) / var(X)
            b = y_ave - a * x_ave

            j1 = 1
            F = ""
            term = []
            while j1 <= j
                i1 = vn_index[j1]
                ii1 = index[i1]
                F *= @sprintf("%.6f*%s + ", (a * coef[i1]), Island[ii1]["formula"])
                push!(term, [a*coef[i1], Island[ii1]["formula"], a*vn[i1]/(1.0+(a-1)*vn_sum)])
                j1 += 1
            end
            e = Y - a*X
            fit = fitness(e)
            push!(Reg, Dict(
                "formula"    => @sprintf("%s :%s", F[1:end-3], Im[1:end-1]),
                "complexity" => s,
                "fit"        => fit["Least_mean"],
                "term"       => term,
                "fits"       => fit
            ))

            F *= @sprintf("%.6f + ", b)
            push!(term, [b, "b=1"])
            fit = fitness(e .- b)
            push!(Reg, Dict(
                "formula"    => @sprintf("%s :%s", F[1:end-3], Im[1:end-1]),
                "complexity" => s+1,
                "fit"        => fit["Least_mean"], 
                "term"       => term,
                "fits"       => fit
            ))
            fit = fit["Least_mean"]
        end
        j += 1
    end

    return Reg
end

function fitness(e)
    S = sum(e.^2)
    LM = S/length(e)
    VAR = LM - mean(e).^2

    return Dict(
        "l2norm" => -sqrt(S),
        "var" => -VAR,
        "Least_mean" => -LM
    )
end