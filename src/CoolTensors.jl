module CoolTensors

export Tensor, @T_str, raise, lower, TCartesianIndex
export TScalar, TVector, TCovector, TLinearMap, TBilinear

struct Tensor{T,N,ipos,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
end

Base.size(t::Tensor) = size(t.parent)
function Base.getindex(t::Tensor, inds::Int...)
    @boundscheck checkbounds(t.parent, inds...)
    @inbounds t.parent[inds...]
end
Base.parent(t::Tensor) = t.parent

struct IndexPos{N}
    # TODO: Should this be UInt64 instead?
    x::UInt
end

index_pos(::Tensor{<:Any,<:Any,ipos}) where {ipos} = ipos
function (ipos::IndexPos{N})(a::AbstractArray{T,N}) where {T,N}
    Tensor{T,N,ipos,typeof(a)}(a)
end
(::IndexPos{0})(x::Number) = x


Base.getindex(ipos::IndexPos, i) = (ipos.x >> (i-1)) % Bool
Base.lastindex(::IndexPos{N}) where {N} = N
function deleteat(ipos::IndexPos{N}, i::Int) where {N}
    mask = (UInt(1) << i) - UInt(1)
    IndexPos{N-1}((ipos.x & (mask >> 1)) | ((ipos.x & ~mask) >> 1))
end
function Base.show(io::IO, ipos::IndexPos{N}) where {N}
    print(io, "T\"")
    for i in 1:N
        print(io, ipos[i] ? ''' : ',')
    end
    print(io, '\"')
    nothing
end

macro T_str(s)
    i = 0
    x = UInt(0)
    for c in s
        isspace(c) && continue
        x |= if c === ''' || c === '^'
            UInt(1) << i
        elseif c === ',' || c === '_' || c === '.'
            UInt(0)
        else
            throw(ArgumentError("Index positions can only be ''' and ','"))
        end
        i += 1
    end
    :(IndexPos{$i}($x))
end

for (typed_f, f) in [
    (:getindex, :vect), (:typed_hcat, :hcat), (:typed_vcat, :vcat), (:typed_hvcat, :hvcat),
]
    @eval Base.$typed_f(ipos::IndexPos, args...) = ipos(Base.$f(args...))
end
Base.getindex(::IndexPos{0}, x) = T""(fill(x))

const TScalar = Tensor{T,0,T""} where {T}
const TVector = Tensor{T,1,T"'"} where {T}
const TCovector = Tensor{T,1,T","} where {T}
const TLinearMap = Tensor{T,2,T"',"} where {T}
const TBilinear = Tensor{T,2,T",,"} where {T}

function (t::Tensor{<:Any,N})(v::Union{TVector,TCovector}; dims=N) where {N}
    N == 0 && throw(ArgumentError("Cannot contract Scalar with $(typeof(v))"))
    index_pos(t)[dims] !== index_pos(v)[1] ||
        throw(ArgumentError("Tensor indices don't match"))

    # TODO: use an external package for tensor reductions
    parent = mapreduce(
        (x, y) -> x .* y, (x, y) -> x .+ y, eachslice(t.parent, dims=dims), v;
        init=zeros(promote_type(eltype(t), eltype(v)), size(t)[setdiff(1:end, dims)]),
    )
    deleteat(index_pos(t), dims)(parent)
end
(t::Tensor)(::Colon; dims=0) = t
function (t::Tensor{<:Any,N})(args...) where {N}
    foldl((t, (i,v)) -> t(v, dims=N-i+1), enumerate(args), init=t)
end


function raise(t::Tensor{<:Any,N,ipos}, i::Int) where {N,ipos}
    IndexPos{N}(ipos.x | (UInt(1) << (i-1)))(t.parent)
end
raise(t::Tensor, inds::Vararg{Int}) = foldl((t, i) -> raise(t, i), inds, init=t)

function lower(t::Tensor{<:Any,N,ipos}, i::Int) where {N,ipos}
    IndexPos{N}(ipos.x & ~(UInt(1) << (i-1)))(t.parent)
end
lower(t::Tensor, inds::Vararg{Int}) = foldl((t, i) -> lower(t, i), inds, init=t)


struct TCartesianIndex{N,ipos}
    I::CartesianIndex{N}
end

function TCartesianIndex{N,ipos}(t::NTuple{N,Int}) where {N,ipos}
    TCartesianIndex{N,ipos}(CartesianIndex(t))
end
TCartesianIndex{N,ipos}(t::Vararg{Int,N}) where {N,ipos} = TCartesianIndex{N,ipos}(t)

function Base.getindex(t::Tensor{<:Any,N,ipos}, cI::TCartesianIndex{N,ipos}) where {N,ipos}
    t.parent[cI.I]
end

Base.typed_vcat(t::Tensor, i...) = Base.typed_hvcat(t, ntuple(one, length(i)), i...)
Base.typed_hcat(t::Tensor, i...) = Base.typed_hvcat(t, (length(i),), i...)

function index_pos_from_alternations(alt::Tuple, current_ipos=true)
    i = 0
    x = UInt(0)
    while !isempty(alt)
        if first(alt) > 0
            x |= UInt(current_ipos) << i
            i += 1
            alt = (first(alt) - 1, Base.tail(alt)...)
        else
            current_ipos = !current_ipos
            alt = Base.tail(alt)
        end
    end
    IndexPos{i}(x)
end

function Base.typed_hvcat(t::Tensor{<:Any,N,ipos}, alt::Tuple, i...) where {N,ipos}
    i, _ipos = if !isempty(i) && first(i) === (\)
        Base.tail(i), index_pos_from_alternations(Base.tail(alt), false)
    else
        i, index_pos_from_alternations(alt)
    end
    _ipos === ipos ||
        throw(ArgumentError("Index positions $_ipos don't match indices of Tensor $ipos"))
    I = TCartesianIndex{N,ipos}(reduce((x, y) -> (x..., y...), i))
    t[I]
end

using TensorCore
export ⊗

function TensorCore.tensor(t::Tensor{<:Any,N,ipos}, s::Tensor{<:Any,M,jpos}) where {N,ipos,M,jpos}
    ts = tensor(parent(t), parent(s))
    ijpos = IndexPos{N+M}((ipos.x | jpos.x << N))
    Tensor{eltype(ts), N+M, ijpos, typeof(ts)}(ts)
end

end
