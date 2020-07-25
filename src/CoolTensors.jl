module CoolTensors

# credits to @tkf on discourse: https://discourse.julialang.org/t/fun-one-liners/28352/57
@doc let path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    open(path) do io
        lines = eachline(io)
        # skip badges in lines 2-6
        s = first(lines) * '\n' * join(Iterators.drop(lines, 5), '\n')
        replace(s, "```julia" => "```jldoctest README")
    end
end
CoolTensors

export Tensor, @T_str, raise, lower, TIndex, TI,
    tbegin, tend

include("tensor.jl")

export TScalar, TVector, TCovector, TLinearMap, TBilinear

const TScalar = Tensor{T,0,T""} where {T}
const TVector = Tensor{T,1,T"'"} where {T}
const TCovector = Tensor{T,1,T","} where {T}
const TLinearMap = Tensor{T,2,T"',"} where {T}
const TBilinear = Tensor{T,2,T",,"} where {T}

include("contractions.jl")
include("tindex.jl")

include("tbeginend.jl")

export ⊗ # from TensorCore.jl

include("operations.jl")

include("tensoroperations.jl")

end
