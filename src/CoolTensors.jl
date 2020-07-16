module CoolTensors

export Tensor, @T_str, raise, lower, TCartesianIndex, TI

include("tensor.jl")

export TScalar, TVector, TCovector, TLinearMap, TBilinear

const TScalar = Tensor{T,0,T""} where {T}
const TVector = Tensor{T,1,T"'"} where {T}
const TCovector = Tensor{T,1,T","} where {T}
const TLinearMap = Tensor{T,2,T"',"} where {T}
const TBilinear = Tensor{T,2,T",,"} where {T}

include("contractions.jl")
include("tcartesianindex.jl")

export ⊗ # from TensorCore.jl

include("operations.jl")

end
