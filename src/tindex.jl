###
### TIndex
###

struct TIndex{N,ipos,T<:Tuple}
    I::T
    TIndex{N,ipos}(t::T) where {N,ipos,T<:Tuple} = new{N,ipos,T}(t)
end

TIndex{N,ipos}(t...) where {N,ipos} = TIndex{N,ipos}(t)

function Base.show(io::IO, ti::TIndex{N,ipos}) where {N,ipos}
    print(io, "TIndex{", N, ",", ipos, "}", ti.I)
end

###
### (::Tensor)[::TIndex]( = x)
###

@inline function Base.to_indices(
    t::Tensor{<:Any,N,ipos},
    inds,
    I::Tuple{TIndex{N,ipos}}
) where {N,ipos}
    #to_indices(t, inds, first(I).I)
    to_indices(t, inds, expand_tbeginends(t, first(I)).I)
end

@noinline function Base.to_indices(
    ::Tensor{<:Any,<:Any,ipos},
    _,
    ::Tuple{TIndex{<:Any,_ipos}},
) where {ipos,_ipos}
    throw(ArgumentError("Index positions $_ipos don't match indices of Tensor $ipos"))
end

# needed for views
Base.to_indices(t::Tensor, I::Tuple{TIndex}) = to_indices(t, axes(t), I)

###
### TI[1; 2 3; 4]
###

struct _TI end
const TI = _TI()

### `@generated` typed_hvcat

function static_index_pos_from_alternations(alt, ipos=true)
    i = 0
    x = UInt(0)
    while !isempty(alt)
        if first(alt) > 0
            x |= UInt(ipos) << i
            i += 1
            alt = (first(alt) - 1, Base.tail(alt)...)
        else
            ipos = !ipos
            alt = Base.tail(alt)
        end
    end
    :(IndexPos{$i}($x))
end

expanded_dims(::Type{<:Union{Integer,AbstractUnitRange{<:Integer},Colon}}) = 1
expanded_dims(::Type{<:Union{CartesianIndex{N},CartesianIndices{N}}}) where {N} = N
expanded_dims(::Type{BitArray{N}}) where {N} = N
expanded_dims(::Type{AbstractArray{T,N}}) where {T,N} = expanded_dims(T)
expanded_dims(::Type) = -1

function static_expand_indices(alt::Tuple, inds::Tuple{Vararg{Type}})
    N = 0
    new_alt = (0,)
    alt_ptr = 1
    for T_i in inds
        n = expanded_dims(T_i)::Int
        n == -1 && return nothing
        N += n
        new_alt = (Base.front(new_alt)..., last(new_alt) + n)
        if alt_ptr == alt[length(new_alt)]
            new_alt = (new_alt..., 0)
            alt_ptr = 1
        else
            alt_ptr += 1
        end
    end
    new_alt, N
end

function Base.typed_hvcat(
    ::_TI,
    ::Val{alt},
    inds...;
    start_ipos::Val{_start_ipos} = Val(true)
) where {alt,_start_ipos}
    if @generated
        e = static_expand_indices(alt, inds)
        e === nothing && return :(
            _typed_hvcat_fallback(TI, $alt, $(inds...), start_ipos=_start_ipos)
        )
        alt, N = e
        ipos = static_index_pos_from_alternations(alt, _start_ipos)
        :(Base.@_inline_meta; TIndex{$N,$ipos}(inds))
    else
        _typed_hvcat_fallback(TI, alt, inds..., start_ipos=_start_ipos)
    end
end
@inline function Base.typed_hvcat(::_TI, ::Val{alt}, ::typeof(\), inds...) where {alt}
    Base.typed_hvcat(TI, Val{Base.tail(alt)}(), inds...; start_ipos=Val(false))
end
@inline function Base.typed_hvcat(::_TI, alt::T, inds...) where {T<:Tuple}
    Base.typed_hvcat(TI, Val{alt}(), inds...)
end

### non-`@generated` typed_hvcat

function index_pos_from_alternations_fallback(alt::Tuple, ipos=true)
    i = 0
    x = UInt(0)
    while !isempty(alt)
        if first(alt) > 0
            x |= UInt(ipos) << i
            i += 1
            alt = (first(alt) - 1, Base.tail(alt)...)
        else
            ipos = !ipos
            alt = Base.tail(alt)
        end
    end
    IndexPos{i}(x)
end

function expand_indices_fallback(alt, inds)
    new_inds = ()
    new_alt = (0,)
    alt_ptr = 1
    for i in inds
        t = to_indices(AbstractArray, (), (i,))
        new_inds = (new_inds..., t...)
        new_alt = (Base.front(new_alt)..., last(new_alt) + length(t))
        if alt_ptr == alt[length(new_alt)]
            new_alt = (new_alt..., 0)
            alt_ptr = 1
        else
            alt_ptr += 1
        end
    end
    return new_alt, new_inds
end

function _typed_hvcat_fallback(::_TI, alt::Tuple, inds...; start_ipos=true) where {N}
    alt, _inds = expand_indices_fallback(alt, inds)
    ipos = index_pos_from_alternations_fallback(alt, start_ipos)
    TIndex{length(_inds),ipos}(inds)
end

for T in [:_TI, :Tensor]
    @eval begin
        @inline function Base.typed_vcat(x::$T, i::Vararg{Any,N}) where {N}
            Base.typed_hvcat(x, Val{ntuple(one, N)}(), i...)
        end
        @inline function Base.typed_hcat(x::$T, i::Vararg{Any,N}) where {N}
            Base.typed_hvcat(x, Val{(N,)}(), i...)
        end
    end
end
# for TI[] and TI[1]
@inline Base.getindex(::_TI) = Base.typed_hvcat(TI, Val{()}())
@inline Base.getindex(::_TI, i) = Base.typed_hvcat(TI, Val{(1,)}(), i)

###
### (::Tensor)[1; 2 3; 4]
###

# a[1; 2 3; 4] gets transformed into a[TI[1; 2 3; 4]]
@inline function Base.typed_hvcat(t::Tensor, alt::T, i...) where {T<:Tuple}
    Base.typed_hvcat(t, Val{alt}(), i...)
end
@inline function Base.typed_hvcat(
    t::Tensor{T,N,ipos},
    alt::Val{_alt},
    i...,
) where {T,N,ipos,_alt}
    t[Base.typed_hvcat(TI, alt, i...)]
end
@inline function Base.typed_hvcat(
    t::Tensor{T,N,ipos},
    alt::Val{_alt},
    ::typeof(\),
    i...,
) where {T,N,ipos,_alt}
    t[Base.typed_hvcat(TI, alt, \, i...)]
end
