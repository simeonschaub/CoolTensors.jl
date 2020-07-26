###
### tbegin/tend
###

struct TBeginEnd{F}
    "maps `(firstindex(::Tensor, d), lastindex(::Tensor, d))` to an index"
    f::F
    ast
end

for (tend, _tend, l) in [(:tbegin, :_tbegin, :f), (:tend, :_tend, :l)]
    @eval begin
        $_tend(f, l) = $l
        const $tend = TBeginEnd($_tend, $(Meta.quot(tend)))
    end
end

Base.show(io::IO, tend::TBeginEnd) = print(io, tend.ast)

macro _tbeginend(ex)
    @assert Meta.isexpr(ex, :call)
    showexpr = Expr(
        :call,
        ex.args[1],
        [
            if Meta.isexpr(i, :call) && Meta.isexpr(i.args[1], :.)
                Expr(:$, Expr(:., i.args[1].args[1], Meta.quot(:ast)))
            else
                Expr(:$, i)
            end
            for i in ex.args[2:end]
        ]...,
    )
    :(TBeginEnd((f, l) -> $ex, $(Meta.quot(showexpr))))
end

for f in [:(:), :+, :-, :*, :min, :max, :%, :mod, :mod1, :div, :fld]
    @eval begin
        Base.$f(x::TBeginEnd, y::Int) = @_tbeginend $f(x.f(f, l), y)
        Base.$f(x::Int, y::TBeginEnd) = @_tbeginend $f(x, y.f(f, l))
        Base.$f(x::TBeginEnd, y::TBeginEnd) = @_tbeginend $f(x.f(f, l), y.f(f, l))
    end
end

Base.:(:)(x::TBeginEnd, s::Int, y::Int) = @_tbeginend x.f(f, l):s:y
Base.:(:)(x::Int, s::Int, y::TBeginEnd) = @_tbeginend x:s:y.f(f, l)
Base.:(:)(x::TBeginEnd, s::Int, y::TBeginEnd) = @_tbeginend x.f(f, l):s:y.f(f, l)


Base.to_index(i::TBeginEnd) = i
expanded_dims(::Type{<:TBeginEnd}) = 1

expand_tbeginend(t, dim, x::TBeginEnd) = x.f(firstindex(t, dim), lastindex(t, dim))
expand_tbeginend(t, _, x) = x

function expand_tbeginends(t, inds::Tuple)
    ntuple(i -> expand_tbeginend(t, i, inds[i]), length(inds))
end
function expand_tbeginends(t, ti::TIndex{N,ipos}) where {N,ipos}
    TIndex{N,ipos}(expand_tbeginends(t, ti.I))
end
