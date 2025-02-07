function read_txt(name)
    list = []
    dic = Dict()
    for line in eachline(name)
        lines = split(line)

        # 最初の要素が数値でなければ、その文字列をdic[]にいれ、行の残りを値として辞書に入れる
        # 数値の場合には、
        if tryparse(Float32, lines[1]) === nothing
            dic[lines[1]] = parse.(Float32,lines[2:end])
            #println(dic)
        else
            f_list = parse.(Float32, lines)
            list = size(list)[1] < 1 ? f_list : hcat(list, f_list)
        end
    end

    return list, dic
end

read_txt("./simulation_burgers.txt")