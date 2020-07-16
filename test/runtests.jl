using CoolTensors
using Test

@testset "CoolTensors.jl" begin
    # Write your tests here.
end

using Documenter
DocMeta.setdocmeta!(CoolTensors, :DocTestSetup, :(using CoolTensors); recursive=true)
doctest(CoolTensors; manual = false)
