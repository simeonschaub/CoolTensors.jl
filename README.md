# CoolTensors

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://simeonschaub.github.io/CoolTensors.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://simeonschaub.github.io/CoolTensors.jl/dev)
[![Build Status](https://github.com/simeonschaub/CoolTensors.jl/workflows/CI/badge.svg)](https://github.com/simeonschaub/CoolTensors.jl/actions)
[![Coverage](https://codecov.io/gh/simeonschaub/CoolTensors.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/simeonschaub/CoolTensors.jl)

The easiest way to create `Tensor` objects is using `IndexPos` objects. They can easily created with the custom `T"..."` string literal. `'` denotes contravariant indices and `,` covariant indices:

```julia
julia> x = T"'"[√2, 1]
2-element Tensor{Float64,1,T"'",Array{Float64,1}}:
 1.4142135623730951
 1.0

julia> Λ(ψ) = T"',"[cosh(ψ) -sinh(ψ); -sinh(ψ) cosh(ψ)]
Λ (generic function with 1 method)

julia> using LinearAlgebra

julia> g = T",,"(Diagonal([1, -1]))
2×2 Tensor{Int64,2,T",,",Diagonal{Int64,Array{Int64,1}}}:
 1   0
 0  -1
```

`lower` and `raise` lower and raise indices: (The metric is assumed to be euclidian.)

```julia
julia> lower(x, 1)
2-element Tensor{Float64,1,T",",Array{Float64,1}}:
 1.4142135623730951
 1.0

julia> 𝔤 = raise(g, 1, 2)
2×2 Tensor{Int64,2,T"''",Diagonal{Int64,Array{Int64,1}}}:
 1   0
 0  -1
```

Indexing can be done just like a regular array, using `TCartesianIndex`, which stores an additional `IndexPos` in the type parameter, or by using `hvcat` syntax. When using the latter, indices are separated by either whitespace or `;`, `;` switches the contra-/covariance of the following indices, whitespace means an index has the same contra-/covariance as the last index. To specify that the first index is covariant, write `\;` before the first index.

```julia
julia> x[1]
1.4142135623730951

julia> lower(x, 1)[\; 1]
1.4142135623730951

julia> Λ(.5)[1; 2]
-0.5210953054937474

julia> g[\; 2 2]
-1

julia> g[1; 2]
ERROR: ArgumentError: Index positions T"'," don't match indices of Tensor T",,"
Stacktrace:
[...]
```

Tensors can be called with vectors and covectors. The rightmost index is always contracted first. If called with multiple (co-)vectors, the first (co-)vector is contracted with the last index of the tensor, the second with the second-last index, and so on. `:` can be used to skip the contraction of an index.

```julia
julia> Λ(.5)(x)
2-element Tensor{Float64,1,T"'",Array{Float64,1}}:
 1.073608627785168
 0.3906859168881719

julia> Λ(.5)(lower(x, 1))
ERROR: ArgumentError: Tensor indices don't match
Stacktrace:
[...]

julia> Λ(.5)(:, lower(x, 1))
2-element Tensor{Float64,1,T",",Array{Float64,1}}:
 1.073608627785168
 0.3906859168881719

julia> lower(x, 1)(x)
3.0000000000000004
```

Let's prove that `g(x, x)` is Lorentz invariant!

```julia
julia> g(x, x)
1.0000000000000004

julia> g(Λ(.5)(x), Λ(.5)(x))
0.9999999999999998
```

QED.

`⊗` (`\otimes`) from [TensorCore.jl](https://github.com/JuliaMath/TensorCore.jl) is overloaded and exported to calculate the tensor product of two `Tensor`s:

```julia
julia> x ⊗ lower(x, 1)
2×2 Tensor{Float64,2,T"',",Array{Float64,2}}:
 2.0      1.41421
 1.41421  1.0
```

*Disclaimer:* This is currently only a prototype. It is still missing a lot of features. Performance should be pretty bad. There will be bugs. Don't use this for any productive work.
