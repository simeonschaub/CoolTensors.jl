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

function Base.typed_hvcat(::_TI, alt::Tuple, inds...) where {N}
    alt, _inds = expand_indices(alt, inds)
    ipos = index_pos_from_alternations(alt)
    TIndex{length(_inds),ipos}(inds)
end
function Base.typed_hvcat(::_TI, alt::Tuple, ::typeof(\), inds...) where {N}
    alt, _inds = expand_indices(Base.tail(alt), inds)
    ipos = index_pos_from_alternations(alt, false)
    TIndex{length(_inds),ipos}(inds)
end

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
Base.typed_hvcat(t::Tensor, alt::Tuple, i...) = t[Base.typed_hvcat(TI, alt, i...)]
#Base.typed_hvcat(t::Tensor{T,N,ipos}, alt::Tuple, i...) where {T,N,ipos} = t[Base.typed_hvcat(TI, alt, i...)::TIndex{N,ipos,typeof(i)}]
#Base.typed_hvcat(t::Tensor{T,N,ipos}, alt::Tuple, ::typeof(\), i...) where {T,N,ipos} = t[Base.typed_hvcat(TI, alt, \, i...)::TIndex{N,ipos,typeof(i)}]
