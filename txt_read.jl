function read_txt(name)
    list = []
    dic = Dict()
    for line in eachline(name)
        lines = split(line)

        if tryparse(Float32, lines[1]) === nothing
            dic[lines[1]] = parse.(Float32,lines[2:end])
        else
            f_list = parse.(Float32, lines)
            list = size(list)[1] < 1 ? f_list : hcat(list, f_list)
        end
    end

    return list, dic
end

# read_txt("simulation_burgers_ari.txt")