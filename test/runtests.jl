using CoolTensors
using Test

@testset "CoolTensors.jl" begin
    # Write your tests here.
end

if VERSION <= v"1.5"
    using Documenter
    DocMeta.setdocmeta!(CoolTensors, :DocTestSetup, :(using CoolTensors); recursive=true)
    doctest(CoolTensors; manual = false)
end
