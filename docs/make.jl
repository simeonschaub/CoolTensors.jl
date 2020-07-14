using CoolTensors
using Documenter

makedocs(;
    modules=[CoolTensors],
    authors="Simeon Schaub <simeondavidschaub99@gmail.com> and contributors",
    repo="https://github.com/simeonschaub/CoolTensors.jl/blob/{commit}{path}#L{line}",
    sitename="CoolTensors.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://simeonschaub.github.io/CoolTensors.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/simeonschaub/CoolTensors.jl",
)
