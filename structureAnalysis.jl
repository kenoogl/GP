# Source code abstraction

using CodeTracking, AbstractTrees

# 関数依存関係を保存する構造体
struct FNode
    name::Symbol
    children::Vector{FNode}
end

AbstractTrees.children(node::FNode) = node.children

# ソースコードの関数呼び出し関係を解析
function parse_dependencies(code::String)
    functions = Dict{Symbol, Vector{Symbol}}()
    current_function = nothing

	for line in eachline(code)
        if startswith(line, "function ")
            func_name = Symbol(split(line[10:end], "(")[1])
            current_function = func_name
            functions[func_name] = []
        elseif current_function !== nothing
            for (func, _) in functions
                if occursin(string(func), line)
                    push!(functions[current_function], func)
                end
            end
        end
    end

    return functions
end

# 依存関係ツリーを構築
function build_tree(func_name::Symbol, functions::Dict{Symbol, Vector{Symbol}})
    return FNode(func_name, [build_tree(child, functions) for child in get(functions, func_name, [])])
end


# 解析してツリーを表示
dependencies = parse_dependencies("gp_predictor_un.jl")
tree = build_tree(:main, dependencies)
print_tree(tree)
