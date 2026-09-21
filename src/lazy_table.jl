# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

"""
    LH5LazyTable

A table stored in an LH5 file whose columns are opened on demand.

Reading a table returns one of these. Accessing a column, by
`tbl.colname`, `getproperties` or the Tables.jl interface, opens that
column and no other. Anything that treats the table as an array
(broadcasting, indexing, iteration) opens all of them, as it must.

# Extended help

Opening a column reads its "datatype" attribute, and for a vector of
vectors its `cumulative_length` as well. Those live in the dataset's
object header, which HDF5 places next to the column's data rather than
next to the other headers, so opening a wide table eagerly costs one
scattered read per column plus the `cumulative_length` arrays. On a
cluster filesystem that is the dominant cost of opening a file whose
columns are mostly unused.

Neither the column names nor their types are part of the type, so
opening a file compiles nothing that depends on the table's shape.
Column access is therefore not inferrable, which costs nothing in
practice: the datatype attributes are only read at run time, so code
that loops over rows has to pass the selected columns through a function
boundary either way.
"""
mutable struct LH5LazyTable
    data_store::HDF5.H5DataStore
    colnames::Vector{Symbol}
    columns::Dict{Symbol, Any}
    typed::Any
end

_lazy_table(ds::HDF5.H5DataStore, names) =
    LH5LazyTable(ds, collect(Symbol, names), Dict{Symbol, Any}(), nothing)

# The struct's own fields are shadowed by column access:
_table_store(tbl::LH5LazyTable) = getfield(tbl, :data_store)
_table_colnames(tbl::LH5LazyTable) = getfield(tbl, :colnames)
_table_columns(tbl::LH5LazyTable) = getfield(tbl, :columns)

Base.propertynames(tbl::LH5LazyTable) = (_table_colnames(tbl)...,)
Base.keys(tbl::LH5LazyTable) = propertynames(tbl)
Base.haskey(tbl::LH5LazyTable, name::Symbol) = name in _table_colnames(tbl)
Base.haskey(tbl::LH5LazyTable, name::AbstractString) = haskey(tbl, Symbol(name))

function Base.getproperty(tbl::LH5LazyTable, name::Symbol)
    get!(_table_columns(tbl), name) do
        haskey(tbl, name) || throw(ArgumentError("table has no column \"$name\""))
        LH5Array(_table_store(tbl)[String(name)])
    end
end

Base.getindex(tbl::LH5LazyTable, name::Symbol) = getproperty(tbl, name)
Base.getindex(tbl::LH5LazyTable, name::AbstractString) = getproperty(tbl, Symbol(name))

"""
    _typed_table(tbl::LH5LazyTable)

Open every column and return them as a `StructArray`, the representation
that array operations need. Memoized, since building it specializes on
the full column type tuple.
"""
function _typed_table(tbl::LH5LazyTable)
    cached = getfield(tbl, :typed)
    isnothing(cached) || return cached
    names = propertynames(tbl)
    typed = StructArray(NamedTuple{names}(map(name -> getproperty(tbl, name), names)))
    setfield!(tbl, :typed, typed)
    typed
end

_typed_table(tbl::StructArray{<:NamedTuple}) = tbl

StructArrays.StructArray(tbl::LH5LazyTable) = _typed_table(tbl)

function Base.length(tbl::LH5LazyTable)
    names = _table_colnames(tbl)
    isempty(names) ? 0 : length(getproperty(tbl, first(names)))
end

Base.size(tbl::LH5LazyTable) = (length(tbl),)
Base.isempty(tbl::LH5LazyTable) = length(tbl) == 0
Base.firstindex(tbl::LH5LazyTable) = 1
Base.lastindex(tbl::LH5LazyTable) = length(tbl)

# Left deliberately narrow: a bare `idxs...` is ambiguous with Indexing.jl's
# dictionary-indexed getindex/view.
const _TableIndex = Union{Integer, AbstractVector, Colon}
Base.getindex(tbl::LH5LazyTable, idxs::_TableIndex...) = _typed_table(tbl)[idxs...]
Base.view(tbl::LH5LazyTable, idxs::_TableIndex...) = view(_typed_table(tbl), idxs...)
Base.iterate(tbl::LH5LazyTable, state...) = iterate(_typed_table(tbl), state...)
Base.broadcastable(tbl::LH5LazyTable) = _typed_table(tbl)
Base.collect(tbl::LH5LazyTable) = collect(_typed_table(tbl))

Base.:(==)(a::LH5LazyTable, b::LH5LazyTable) = _typed_table(a) == _typed_table(b)
Base.:(==)(a::LH5LazyTable, b::AbstractArray) = _typed_table(a) == b
Base.:(==)(a::AbstractArray, b::LH5LazyTable) = a == _typed_table(b)

Tables.istable(::Type{LH5LazyTable}) = true
Tables.columnaccess(::Type{LH5LazyTable}) = true
Tables.rowaccess(::Type{LH5LazyTable}) = false
Tables.columns(tbl::LH5LazyTable) = tbl
Tables.columnnames(tbl::LH5LazyTable) = propertynames(tbl)
Tables.getcolumn(tbl::LH5LazyTable, name::Symbol) = getproperty(tbl, name)
Tables.getcolumn(tbl::LH5LazyTable, i::Int) = getproperty(tbl, _table_colnames(tbl)[i])
# The column types are only known once the columns are opened:
Tables.schema(::LH5LazyTable) = nothing

function Base.show(io::IO, tbl::LH5LazyTable)
    names = _table_colnames(tbl)
    opened = length(_table_columns(tbl))
    print(io, "LH5LazyTable(", length(names), " columns")
    opened > 0 && print(io, ", ", opened, " opened")
    print(io, "): ")
    join(io, names, ", ")
end

Base.show(io::IO, ::MIME"text/plain", tbl::LH5LazyTable) = show(io, tbl)
