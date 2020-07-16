###
### TCartesianIndex
###

struct TCartesianIndex{N,ipos}
    I::CartesianIndex{N}
end

function TCartesianIndex{N,ipos}(t::NTuple{N,Int}) where {N,ipos}
    TCartesianIndex{N,ipos}(CartesianIndex(t))
end
TCartesianIndex{N,ipos}(t::Vararg{Int,N}) where {N,ipos} = TCartesianIndex{N,ipos}(t)

function Base.show(io::IO, ti::TCartesianIndex)
    print(io, typeof(ti))
    print(io, Tuple(ti.I))
end

###
### (::Tensor)[::TCartesianIndex]( = x)
###

function Base.to_indices(
    t::Tensor{<:Any,N,ipos},
    inds,
    I::Tuple{TCartesianIndex{N,ipos},Vararg{Any}}
) where {N,ipos}
    to_indices(t, inds, (first(I).I, Base.tail(I)...))
end

function Base.to_indices(
    ::Tensor{<:Any,<:Any,ipos},
    _,
    ::Tuple{TCartesianIndex{<:Any,_ipos},Vararg{Any}},
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

function Base.typed_hvcat(::_TI, alt::Tuple, i::Vararg{Int,N}) where {N}
    ipos = index_pos_from_alternations(alt)
    TCartesianIndex{N,ipos}(i)
end
function Base.typed_hvcat(::_TI, alt::Tuple, ::typeof(\), i::Vararg{Int,N}) where {N}
    ipos = index_pos_from_alternations(Base.tail(alt), false)
    TCartesianIndex{N,ipos}(i)
end

for T in [:_TI, :Tensor]
    @eval begin
        Base.typed_vcat(x::$T, i...) = Base.typed_hvcat(x, ntuple(one, length(i)), i...)
        Base.typed_hcat(x::$T, i...) = Base.typed_hvcat(x, (length(i),), i...)
    end
end

###
### (::Tensor)[1; 2 3; 4]
###

# a[1; 2 3; 4] gets transformed into a[TI[1; 2 3; 4]]
Base.typed_hvcat(t::Tensor, alt::Tuple, i...) = t[Base.typed_hvcat(TI, alt, i...)]
