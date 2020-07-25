using TensorOperations, Strided
using TensorOperations: IndexTuple

Strided.UnsafeStridedView(t::Tensor{<:Any,<:Any,ipos}) where {ipos} = ipos(Strided.UnsafeStridedView(t.parent))

TensorOperations.isblascontractable(t::Tensor, p1::Tuple{Vararg{Int}}, p2::Tuple{Vararg{Int}}, C::Symbol) = TensorOperations.isblascontractable(t.parent, p1, p2, C)

Strided.sreshape(t::Tensor{T,N,ipos}, d::Tuple{Vararg{Int}}) where {T,N,ipos} = #=ipos=#(sreshape(t.parent, d))

function TensorOperations.similarstructure_from_indices(T::Type, poA::IndexTuple, poB::IndexTuple,
    p1::IndexTuple, p2::IndexTuple, A::Tensor, B::Tensor, CA::Symbol, CB::Symbol)
    p = (p1..., p2...)
    ax_A, ax_B = axes(A), axes(B)
    ax_from_A = map(i -> ax_A[i], poA)
    ax_from_B = map(i -> ax_B[i], poB)
    ax_comb = (ax_A..., ax_B...)
    map(i -> ax_comb[i], p)
end
