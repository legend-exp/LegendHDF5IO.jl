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

# Any other function broadcasts block by block over a typed table of the
# columns it uses, and a column accessor returns the column. The second
# method resolves the ambiguity with PropertyFunctions' method for functions
# without properties:
Base.Broadcast.broadcasted(pf::PropertyFunction, tbl::AbstractLH5Table) = _lazy_broadcast(pf, tbl)
Base.Broadcast.broadcasted(pf::PropertyFunction{Tuple{}}, tbl::AbstractLH5Table) = _lazy_broadcast(pf, tbl)

_lazy_broadcast(pf::PropertyFunction, tbl::AbstractLH5Table) = _lazy_broadcast(pf, tbl, _used_columns(tbl, pf))
_lazy_broadcast(pf::PropertyFunction{<:PPaths, <:PPath}, tbl::AbstractLH5Table) =
    foldl(getproperty, _path(only(input_property_paths(pf))); init = tbl)

# The element type follows from the type of a typed table, and from the
# columns read otherwise:
_lazy_broadcast(pf, tbl::LH5TypedTable, used::LH5TypedTable) = LH5LazyBroadcast(pf, tbl, used)
_lazy_broadcast(pf, ::AbstractLH5Table, used::LH5TypedTable) = LH5LazyBroadcast(pf, used)

# The columns of `tbl` that `pf` reads, as a typed table. Nested tables hold
# the columns read only. A function without properties reads none, but needs
# the rows to count, from the whole of a typed table or from the first column:
function _used_columns(tbl::LH5TypedTable, pf::PropertyFunction)
    paths = map(_path, input_property_paths(pf))
    isempty(paths) ? tbl : _select(tbl, _selection_tree(paths))
end

function _used_columns(tbl::AbstractLH5Table, pf::PropertyFunction)
    paths = map(_path, input_property_paths(pf))
    isempty(paths) && (paths = ((first(propertynames(tbl)),),))
    heads = (unique(map(first, paths))...,)
    LH5TypedTable(_select(tbl, _selection_tree(paths)), heads, map(head -> (head,), heads))
end

end # module LegendHDF5IOPropertyFunctionsExt
