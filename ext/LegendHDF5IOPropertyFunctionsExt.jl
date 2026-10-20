# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

module LegendHDF5IOPropertyFunctionsExt

import PropertyFunctions
using LegendHDF5IO: AbstractLH5Table, LH5JoinedTable, _join_part

# The generic innermerge opens every column of every table to merge them:
PropertyFunctions.innermerge(tbl::AbstractLH5Table, tbls...) =
    LH5JoinedTable(map(_join_part, (tbl, tbls...))...)

end # module LegendHDF5IOPropertyFunctionsExt
