###
### Tensor
###

struct Tensor{T,N,ipos,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
end

Base.size(t::Tensor) = size(t.parent)
function Base.getindex(t::Tensor, inds::Int...)
    @boundscheck checkbounds(t.parent, inds...)
    @inbounds t.parent[inds...]
end
function Base.setindex!(t::Tensor, x, inds::Int...)
    @boundscheck checkbounds(t.parent, inds...)
    @inbounds t.parent[inds...] = x
end
Base.parent(t::Tensor) = t.parent

###
### IndexPos
###

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

###
### T"..."
###

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

###
### T".."[a b; c d]
###

for (typed_f, f) in [
    (:getindex, :vect), (:typed_hcat, :hcat), (:typed_vcat, :vcat), (:typed_hvcat, :hvcat),
]
    @eval Base.$typed_f(ipos::IndexPos, args...) = ipos(Base.$f(args...))
end
Base.getindex(::IndexPos{0}, x) = T""(fill(x))
