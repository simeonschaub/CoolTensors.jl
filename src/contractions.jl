###
### (::Tensor)(::TVector, :, TCovector, ...)
###

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

###
### raise/lower
###

function raise(t::Tensor{<:Any,N,ipos}, i::Int) where {N,ipos}
    IndexPos{N}(ipos.x | (UInt(1) << (i-1)))(t.parent)
end
raise(t::Tensor, inds::Vararg{Int}) = foldl((t, i) -> raise(t, i), inds, init=t)

function lower(t::Tensor{<:Any,N,ipos}, i::Int) where {N,ipos}
    IndexPos{N}(ipos.x & ~(UInt(1) << (i-1)))(t.parent)
end
lower(t::Tensor, inds::Vararg{Int}) = foldl((t, i) -> lower(t, i), inds, init=t)
