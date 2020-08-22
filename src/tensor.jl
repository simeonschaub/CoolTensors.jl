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

index_pos(::T) where {T} = index_pos(T)
@noinline index_pos(T::Type) = throw(MethodError(index_pos, (T,)))
index_pos(::Type{<:Tensor{<:Any,<:Any,ipos}}) where {ipos} = ipos
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

###
### TAxis
###

struct TAxis{T,ipos,R<:AbstractUnitRange{T}} <: AbstractUnitRange{T}
    parent::R
end
TAxis{T,ipos}(r::R) where {T,ipos,R} = TAxis{T,ipos,R}(r)
index_pos(::Type{<:TAxis{<:Any,ipos}}) where {ipos} = ipos
Base.parent(t::TAxis) = t.parent
Base.first(t::TAxis) = first(t.parent)
Base.last(t::TAxis) = last(t.parent)

function Base.show(io::IO, t::TAxis{T,ipos}) where {T,ipos}
    print(io, "TAxis{", T, ",", ipos, "}(")
    show(io, t.parent)
    print(io, ")")
end

# don't always explicitly show axes in default printing
Base.summary(io::IO, t::Tensor) = Base.array_summary(io, t, axes(t.parent))

function Base.axes(t::Tensor{<:Any,N,ipos}) where {N,ipos}
    ax = axes(t.parent)
    ntuple(i -> TAxis{Int,IndexPos{1}(UInt(position(ipos, i)))}(ax[i]), N)
end

function index_pos(axes::NTuple{N,TAxis{Int}}) where {N}
    x = mapreduce(|, 1:N, init=UInt(0)) do i
        index_pos(axes[i]).x << (i-1)
    end
    IndexPos{N}(x)
end

function Base.similar(
    t::AbstractArray,
    eltype::Type,
    axes::Tuple{TAxis{Int},Vararg{TAxis{Int}}}
)
    ipos = index_pos(axes)
    paren = similar(t, eltype, map(parent, axes))
    ipos(paren)
end
function Base.similar(
    t::Tensor,
    eltype::Type,
    axes::Tuple{TAxis{Int},Vararg{TAxis{Int}}}
)
    similar(t.parent, eltype, axes)
end
Base.similar(t::Tensor, eltype::Type, axes::Tuple{}) = T""(similar(t.parent, eltype, ()))
function Base.similar(
    ::Type{A},
    axes::Tuple{TAxis{Int},Vararg{TAxis{Int},Nm1}}
) where {A<:AbstractArray,Nm1}
    ipos = index_pos(axes)
    paren = similar(A, map(parent, axes))
    ipos(paren)
end

###
### unsafe and strided stuff
###

Base.unsafe_convert(::Type{Ptr{T}}, t::Tensor{T}) where {T} = Base.unsafe_convert(Ptr{T}, t.parent)
Base.strides(t::Tensor) = strides(t.parent)
