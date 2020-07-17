using LegibleLambdas

###
### tbegin/tend
###

struct TBeginEnd{F}
    "maps `(firstindex(::Tensor, d), lastindex(::Tensor, d))` to an index"
    f::F
end

for (tend, _tend, l) in [(:tbegin, :_tbegin, :f), (:tend, :_tend, :l)]
    @eval begin
        $_tend(f, l) = $l
        const $tend = TBeginEnd($_tend)
        
        Base.show(io::IO, ::TBeginEnd{typeof($_tend)}) = print(io, $(string(tend)))
        function Base.show(io::IO, tend::TBeginEnd{<:LegibleLambda})
            show(io, TBeginEnd)
            show(io, tend.f)
        end
        
    end
end

for f in [:(:), :+, :-]
    @eval begin
        Base.$f(x::TBeginEnd, y::Int) = TBeginEnd(@λ (f, l) -> $f(x.f(f, l), y))
        Base.$f(x::Int, y::TBeginEnd) = TBeginEnd(@λ (f, l) -> $f(x, y.f(f, l)))
        Base.$f(x::TBeginEnd, y::TBeginEnd) = TBeginEnd(@λ (f, l) -> $f(x.f(f, l), y.f(f, l)))
    end
end

Base.:(:)(x::TBeginEnd, s::Int, y::Int) = TBeginEnd(@λ (f, l) -> x.f(f, l):s:y)
Base.:(:)(x::Int, s::Int, y::TBeginEnd) = TBeginEnd(@λ (f, l) -> x:s:y.f(f, l))
Base.:(:)(x::TBeginEnd, s::Int, y::TBeginEnd) = TBeginEnd(@λ (f, l) -> x.f(f, l):s:y.f(f, l))


Base.to_index(i::TBeginEnd) = i

expand_tbeginend(t, dim, x::TBeginEnd) = x.f(firstindex(t, dim), lastindex(t, dim))
expand_tbeginend(t, _, x) = x

function expand_tbeginends(t, inds::Tuple)
    ntuple(i -> expand_tbeginend(t, i, inds[i]), length(inds))
end
function expand_tbeginends(t, ti::TIndex{N,ipos}) where {N,ipos}
    TIndex{N,ipos}(expand_tbeginends(t, ti.I))
end
