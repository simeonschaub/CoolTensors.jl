###
### (::Tensor) ⊗ (::Tensor)
###

using TensorCore

function TensorCore.tensor(t::Tensor{<:Any,N,ipos}, s::Tensor{<:Any,M,jpos}) where {N,ipos,M,jpos}
    ts = tensor(parent(t), parent(s))
    ijpos = IndexPos{N+M}((ipos.x | jpos.x << N))
    Tensor{eltype(ts), N+M, ijpos, typeof(ts)}(ts)
end
