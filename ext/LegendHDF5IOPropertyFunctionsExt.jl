# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

module LegendHDF5IOPropertyFunctionsExt

import PropertyFunctions
using PropertyFunctions: PropertyFunction, PropSelFunction, PPath, PPaths, input_property_paths
using LegendHDF5IO: AbstractLH5Table, LH5JoinedTable, LH5TypedTable, LH5LazyBroadcast, _join_part, _select, _selection_tree

# The generic innermerge opens every column of every table to merge them, and
# with an in-memory table first it wraps the lazy columns in a StructArray. A
# lazy table in the first two positions is caught, not one behind two others.
# The third method resolves the ambiguity:
PropertyFunctions.innermerge(tbl::AbstractLH5Table, tbls...) = _lazy_join(tbl, tbls...)
PropertyFunctions.innermerge(tbl, tbl2::AbstractLH5Table, tbls...) = _lazy_join(tbl, tbl2, tbls...)
PropertyFunctions.innermerge(tbl::AbstractLH5Table, tbl2::AbstractLH5Table, tbls...) = _lazy_join(tbl, tbl2, tbls...)

_lazy_join(tbls...) = LH5JoinedTable(map(_join_part, tbls)...)

# Nested paths, `$y.b`, would otherwise read the whole nested table:
@inline PropertyFunctions.subcolumn(col::AbstractLH5Table, name::Symbol) = getproperty(col, name)
@inline PropertyFunctions.subcolumn(col::AbstractLH5Table, ::Val{name}) where {name} = getproperty(col, name)

_path(::PPath{path}) where {path} = path

# A selection fixes the row type from the first file and reads the selected
# columns only, where the generic selection would open them in every file of
# a chain:
Base.Broadcast.broadcasted(pf::PropSelFunction{<:PPaths, names}, tbl::AbstractLH5Table) where {names} =
    LH5TypedTable(tbl, names, map(_path, input_property_paths(pf)))
Base.Broadcast.broadcasted(::PropSelFunction{Tuple{}}, ::AbstractLH5Table) =
    throw(ArgumentError("A selection needs at least one column"))

# Over a typed table, a function broadcasts block by block with an inferable
# element type, a column accessor returns the column and a selection stays a
# selection. The repeated signatures resolve the ambiguities with
# PropertyFunctions' methods and the selection methods above:
Base.Broadcast.broadcasted(pf::PropertyFunction, tbl::LH5TypedTable) = _typed_broadcast(pf, tbl)
Base.Broadcast.broadcasted(pf::PropertyFunction{Tuple{}}, tbl::LH5TypedTable) = _typed_broadcast(pf, tbl)
Base.Broadcast.broadcasted(pf::PropSelFunction{<:PPaths, names}, tbl::LH5TypedTable) where {names} =
    LH5TypedTable(tbl, names, map(_path, input_property_paths(pf)))
Base.Broadcast.broadcasted(::PropSelFunction{Tuple{}}, ::LH5TypedTable) =
    throw(ArgumentError("A selection needs at least one column"))

_typed_broadcast(pf::PropertyFunction, tbl::LH5TypedTable) = LH5LazyBroadcast(pf, tbl, _used_columns(tbl, pf))

# The columns of `tbl` that `pf` reads, as a selection; a function without
# properties reads none, but needs the rows to count:
function _used_columns(tbl::LH5TypedTable, pf::PropertyFunction)
    paths = map(_path, input_property_paths(pf))
    isempty(paths) ? tbl : _select(tbl, _selection_tree(paths))
end
_typed_broadcast(pf::PropertyFunction{<:PPaths, <:PPath}, tbl::LH5TypedTable) =
    foldl(getproperty, _path(only(input_property_paths(pf))); init = tbl)

end # module LegendHDF5IOPropertyFunctionsExt
