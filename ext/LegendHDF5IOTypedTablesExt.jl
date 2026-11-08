# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

module LegendHDF5IOTypedTablesExt

import TypedTables
using LegendHDF5IO: AbstractLH5Table

# A Table of the lazy columns would append through them without the table's
# checks, and its rows would hold views into the file:
TypedTables.Table(::AbstractLH5Table; kwargs...) = throw(ArgumentError(
    "Lazy tables do not convert to Tables, read the rows with tbl[:]"))

end # module LegendHDF5IOTypedTablesExt
