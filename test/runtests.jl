using CoolTensors
using Test

@testset "CoolTensors.jl" begin
    # Write your tests here.
end

using Documenter
#docstring = replace(@doc(CoolTensors).content[], r"```.*?```jldoctest"s => "")
#@doc docstring CoolTensors
#println(@doc CoolTensors)
DocMeta.setdocmeta!(CoolTensors, :DocTestSetup, :(using CoolTensors); recursive=true)
doctest(CoolTensors; manual = false)
