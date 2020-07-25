###
### (::Tensor) ⊗ (::Tensor)
###

using TensorCore

function TensorCore.tensor(t::Tensor{<:Any,N,ipos}, s::Tensor{<:Any,M,jpos}) where {N,ipos,M,jpos}
    ts = tensor(parent(t), parent(s))
    ijpos = IndexPos{N+M}((ipos.x | jpos.x << N))
    Tensor{eltype(ts), N+M, ijpos, typeof(ts)}(ts)
end

function Base.permutedims(t::Tensor{<:Any,N,ipos}, p) where {N,ipos}
    x = mapreduce(|, 1:N, init = UInt(0)) do i
        masked = ipos.x & (UInt(1) << (i-1))
        masked << (p[i] - i)
    end
    CoolTensors.IndexPos{N}(x)(permutedims(t.parent, p))
end
