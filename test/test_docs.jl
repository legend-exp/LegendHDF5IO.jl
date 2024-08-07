# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

using Test
using LegendHDF5IO
import Documenter

Documenter.DocMeta.setdocmeta!(
    LegendHDF5IO,
    :DocTestSetup,
    :(using LegendHDF5IO);
    recursive=true,
)
Documenter.doctest(LegendHDF5IO)
