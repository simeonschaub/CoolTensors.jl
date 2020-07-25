###
### Tensor
###

struct Tensor{T,N,ipos,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
end

Base.size(t::Tensor) = size(t.parent)
Base.firstindex(t::Tensor, i...) = firstindex(t.parent, i...)
Base.lastindex(t::Tensor, i...) = lastindex(t.parent, i...)
Base.parent(t::Tensor) = t.parent

tto_indices(t, inds) = expand_tbeginends(t, to_indices(t, (), inds))
Base.@propagate_inbounds function Base.getindex(t::Tensor, inds...)
    t.parent[tto_indices(t, inds)...]
end
Base.@propagate_inbounds function Base.setindex!(t::Tensor, x, inds...)
    t.parent[tto_indices(t, inds)...] = x
end
Base.checkbounds(t::Tensor, inds...) = checkbounds(t.parent, tto_indices(t, inds)...)

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


position(ipos::IndexPos, i) = (ipos.x >> (i-1)) % Bool
function deleteat(ipos::IndexPos{N}, i::Int) where {N}
    mask = (UInt(1) << i) - UInt(1)
    IndexPos{N-1}((ipos.x & (mask >> 1)) | ((ipos.x & ~mask) >> 1))
end
function Base.show(io::IO, ipos::IndexPos{N}) where {N}
    print(io, "T\"")
    for i in 1:N
        print(io, position(ipos, i) ? ''' : ',')
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

struct TAxis{T,ipos,R<:AbstractUnitRange{T}} <: AbstractUnitRange{T}
    parent::R
end
TAxis{T,ipos}(r::R) where {T,ipos,R} = TAxis{T,ipos,R}(r)
index_pos(::TAxis{<:Any,ipos}) where {ipos} = ipos
Base.parent(t::TAxis) = t.parent
Base.first(t::TAxis) = first(t.parent)
Base.last(t::TAxis) = last(t.parent)
function Base.show(io::IO, t::TAxis{T,ipos}) where {T,ipos}
    print(io, "TAxis{", T, ",", ipos, "}(")
    show(io, t.parent)
    print(io, ")")
end

function Base.axes(t::Tensor{<:Any,N,ipos}) where {N,ipos}
    ax = axes(t.parent)
    ntuple(i -> TAxis{Int,IndexPos{1}(UInt(position(ipos, i)))}(ax[i]), N)
end
#Base.UnitRange(t::TAxis) = UnitRange(t.parent)
#Base.UnitRange{Int}(t::Tensor) = UnitRange{Int}(t.parent)
#Base.LinearIndices(t::Tensor) = LinearIndices(t.parent)
function Base.similar(t::Tensor, eltype::Type, axes::NTuple{N,TAxis{Int}}) where {N}
    x = mapreduce(|, 1:N, init=UInt(0)) do i
        index_pos(axes[i]).x << (i-1)
    end
    ipos = IndexPos{N}(x)
    paren = similar(t.parent, eltype, map(parent, axes))
    ipos(paren)
end
Base.similar(t::Tensor, eltype::Type, axes::Tuple{}) = T""(similar(t.parent, eltype, ()))
