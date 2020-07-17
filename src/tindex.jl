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

function Base.to_indices(
    t::Tensor{<:Any,N,ipos},
    inds,
    I::Tuple{TIndex{N,ipos}}
) where {N,ipos}
    #to_indices(t, inds, first(I).I)
    to_indices(t, inds, expand_tbeginends(t, first(I)).I)
end

function Base.to_indices(
    ::Tensor{<:Any,<:Any,ipos},
    _,
    ::Tuple{TIndex{<:Any,_ipos}},
) where {ipos,_ipos}
    throw(ArgumentError("Index positions $_ipos don't match indices of Tensor $ipos"))
end

###
### TI[1; 2 3; 4]
###

struct _TI end
const TI = _TI()

@generated function index_pos_from_alternations(::Val{_alt}, ::Val{ipos}=Val(true)) where {_alt,ipos}
    alt = _alt
    current_ipos = ipos
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
    :(IndexPos{$i}($x))
end

@generated function expand_indices(::Val{alt}, inds::NTuple{N}) where {alt,N}
    ex = quote
        Base.@_inline_meta
        new_inds = ()
        new_alt = (0,)
        alt_ptr = 1
    end
    for i in 1:N
        push!(ex.args, quote
            t = to_indices(AbstractArray, (), (inds[$i],))
            new_inds = (new_inds..., t...)
            new_alt = (Base.front(new_alt)..., last(new_alt) + length(t))
            if alt_ptr == $alt[length(new_alt)]
                new_alt = (new_alt..., 0)
                alt_ptr = 1
            else
                alt_ptr += 1
            end
        end)
    end
    push!(ex.args, :(Val{new_alt}(), new_inds))
    ex
end

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

function expand_indices(alt, inds)
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

@inline function Base.typed_hvcat(::_TI, alt::Val{_alt}, inds...) where {N,_alt}
    alt2, _inds = expand_indices(alt, inds)
    ipos = index_pos_from_alternations(alt2)
    TIndex{length(_inds),ipos}(inds)
end
@inline function Base.typed_hvcat(::_TI, alt::Val{_alt}, ::typeof(\), inds...) where {N,_alt}
    alt2, _inds = expand_indices(Val{Base.tail(_alt)}(), inds)
    ipos = index_pos_from_alternations(alt2, Val(false))
    TIndex{length(_inds),ipos}(inds)
end
function Base.typed_hvcat(::_TI, alt::Tuple, inds...) where {N}
    Base.typed_hvcat(TI, Val{alt}(), inds...)
end
#function Base.typed_hvcat(::_TI, alt::Tuple, ::typeof(\), inds...) where {N}
#    alt, _inds = expand_indices(Base.tail(alt), inds)
#    ipos = index_pos_from_alternations(alt, false)
#    TIndex{length(_inds),ipos}(inds)
#end

for T in [:_TI, :Tensor]
    @eval begin
        Base.typed_vcat(x::$T, i...) = Base.typed_hvcat(x, ntuple(one, length(i)), i...)
        Base.typed_hcat(x::$T, i...) = Base.typed_hvcat(x, (length(i),), i...)
    end
end
# for TI[] and TI[1]
Base.getindex(::_TI) = Base.typed_hvcat(TI, ())
Base.getindex(::_TI, i) = Base.typed_hvcat(TI, (1,), i)

###
### (::Tensor)[1; 2 3; 4]
###

# a[1; 2 3; 4] gets transformed into a[TI[1; 2 3; 4]]
@inline Base.typed_hvcat(t::Tensor, alt::Tuple, i...) = Base.typed_hvcat(t, Val{alt}(), i...)
@inline Base.typed_hvcat(t::Tensor{T,N,ipos}, alt::Val{_alt}, i...) where {T,N,ipos,_alt} = t[Base.typed_hvcat(TI, alt, i...)#=::TIndex{N,ipos,typeof(i)}=#]
@inline Base.typed_hvcat(t::Tensor{T,N,ipos}, alt::Val{_alt}, ::typeof(\), i...) where {T,N,ipos,_alt} = t[Base.typed_hvcat(TI, alt, \, i...)#=::TIndex{N,ipos,typeof(i)}=#]
