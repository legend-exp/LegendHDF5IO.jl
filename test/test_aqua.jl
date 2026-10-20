# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import LegendHDF5IO
import LegendDataTypes

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        LegendHDF5IO,
        # The cat guards of lazy tables, which only throw, are ambiguous with
        # SparseArrays' cat methods on Julia pre-release builds. The call
        # errors either way:
        ambiguities = (exclude = [Base.cat],),
        # readdata/writedata/getunits/setunits! form the interface that
        # LegendDataTypes defines for LEGEND I/O packages to implement:
        piracies = (treat_as_own = [
            LegendDataTypes.readdata, LegendDataTypes.writedata,
            LegendDataTypes.getunits, LegendDataTypes.setunits!,
        ],)
    )
end # testset
