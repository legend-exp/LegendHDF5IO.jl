# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import LegendHDF5IO
import LegendDataTypes

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        LegendHDF5IO,
        ambiguities = true,
        # readdata/writedata/getunits/setunits! form the interface that
        # LegendDataTypes defines for LEGEND I/O packages to implement:
        piracies = (treat_as_own = [
            LegendDataTypes.readdata, LegendDataTypes.writedata,
            LegendDataTypes.getunits, LegendDataTypes.setunits!,
        ],)
    )
end # testset
