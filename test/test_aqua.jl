# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import LegendHDF5IO

#Test.@testset "Package ambiguities" begin
#    Test.@test isempty(Test.detect_ambiguities(LegendHDF5IO))
#end # testset

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        LegendHDF5IO,
        ambiguities = false,
        unbound_args = false
    )
end # testset
