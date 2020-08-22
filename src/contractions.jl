using TensorOperations

###
### (::Tensor)(::TVector, :, TCovector, ...)
###

#function (t::Tensor{<:Any,N})(v::Union{TVector,TCovector}; dims=N) where {N}
#    N == 0 && throw(ArgumentError("Cannot contract Scalar with $(typeof(v))"))
#    position(index_pos(t), dims) !== position(index_pos(v), 1) ||
#        throw(ArgumentError("Tensor indices don't match"))
#
#    # TODO: use an external package for tensor reductions
#    parent = mapreduce(
#        (x, y) -> x .* y, (x, y) -> x .+ y, eachslice(t.parent, dims=dims), v;
#        init=zeros(promote_type(eltype(t), eltype(v)), size(t)[setdiff(1:end, dims)]),
#    )
#    deleteat(index_pos(t), dims)(parent)
#end
#(t::Tensor)(::Colon; dims=0) = t
@generated function (t::Tensor{<:Any,N})(args::Union{TVector,TCovector,Colon}...) where {N}
    length(args) <= N || throw(ArgumentError("Too many arguments"))
    args = (ntuple(_ -> Colon, N-length(args))..., reverse(args)...)
    tensors = Any[:t]
    cinds = [Symbol[gensym() for _ in 1:N]]
    oinds = Symbol[]

    for (i, v) in enumerate(args)
        if v == Colon
            push!(oinds, cinds[1][i])
            continue
        end

        position(index_pos(t), i) !== position(index_pos(v), 1) ||
            throw(ArgumentError("Tensor indices don't match"))

        v = :(args[$(N + 1 - i)])
        push!(tensors, v)
        push!(cinds, [cinds[1][i]])
    end
    if length(tensors) == 1
        :t
    else
        rhs = Expr(:call, :*, (Expr(:ref, t, i...) for (t,i) in zip(tensors, cinds))...)
        if isempty(oinds)
            :(@tensoropt x = $rhs)
        else
            :(@tensoropt x[$(oinds...)] := $rhs)
        end
    end
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
