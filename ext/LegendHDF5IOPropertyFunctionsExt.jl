# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

module LegendHDF5IOPropertyFunctionsExt

import PropertyFunctions
using PropertyFunctions: PropSelFunction, PPath, PPaths, input_property_paths
using LegendHDF5IO: AbstractLH5Table, LH5JoinedTable, LH5TypedTable, _join_part

# The generic innermerge opens every column of every table to merge them:
PropertyFunctions.innermerge(tbl::AbstractLH5Table, tbls...) =
    LH5JoinedTable(map(_join_part, (tbl, tbls...))...)

_path(::PPath{path}) where {path} = path

# A selection fixes the row type from the first file and reads the selected
# columns only, where the generic selection would open them in every file of
# a chain:
Base.Broadcast.broadcasted(pf::PropSelFunction{<:PPaths, names}, tbl::AbstractLH5Table) where {names} =
    LH5TypedTable(tbl, names, map(_path, input_property_paths(pf)))
Base.Broadcast.broadcasted(::PropSelFunction{Tuple{}}, ::AbstractLH5Table) =
    throw(ArgumentError("A selection needs at least one column"))

end # module LegendHDF5IOPropertyFunctionsExt
