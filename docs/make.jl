# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using LegendHDF5IO

# Doctest setup
DocMeta.setdocmeta!(
    LegendHDF5IO,
    :DocTestSetup,
    :(using LegendHDF5IO);
    recursive=true,
)

makedocs(
    sitename = "LegendHDF5IO",
    modules = [LegendHDF5IO],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://legend-exp.github.io/LegendHDF5IO.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = !("nonstrict" in ARGS),
    strict = !("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/legend-exp/LegendHDF5IO.jl.git",
    forcepush = true,
    push_preview = true,
)
