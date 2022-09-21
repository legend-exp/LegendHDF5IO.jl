# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "Package LegendHDF5IO" begin
    include("ranges/range_to_namedtuple.jl")
    include("histograms/histogram_io.jl")
    include("test_wrappers.jl")
end # testset
