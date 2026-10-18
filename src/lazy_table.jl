# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

"""
    LH5LazyTable

A table stored in an LH5 file whose columns are opened on demand.

Reading a table returns one of these. Accessing a column, with
`tbl.colname` or `Tables.getcolumn`, opens that column and no other.
Anything that treats the table as an array (broadcasting, indexing,
iteration) opens all of them, as it must.

# Extended help

Opening a column reads its "datatype" attribute, and for a vector of
vectors its `cumulative_length` as well. The attribute lives in the
dataset's object header, which HDF5 usually places next to the column's
data rather than next to the other headers, so opening a wide table
eagerly costs one scattered read per column plus the `cumulative_length`
arrays. On a network file system that is the dominant cost of opening a
file whose columns are mostly unused.

Neither the column names nor the column types are type parameters, so
opening a table compiles no code specialized on them. Column access is
therefore not inferable, which costs little in practice: the datatype
attributes are only read at run time, so code that loops over rows has to
pass the selected columns through a function boundary either way.
"""
mutable struct LH5LazyTable
    data_store::HDF5.H5DataStore
    colnames::Vector{Symbol}
    columns::Dict{Symbol, Any}
    typed::Any
    lock::ReentrantLock
end

_lazy_table(ds::HDF5.H5DataStore, names) =
    LH5LazyTable(ds, collect(Symbol, names), Dict{Symbol, Any}(), nothing, ReentrantLock())

# The struct's own fields are shadowed by column access:
_table_store(tbl::LH5LazyTable) = getfield(tbl, :data_store)
_table_colnames(tbl::LH5LazyTable) = getfield(tbl, :colnames)
_table_columns(tbl::LH5LazyTable) = getfield(tbl, :columns)
_table_lock(tbl::LH5LazyTable) = getfield(tbl, :lock)

Base.propertynames(tbl::LH5LazyTable) = (_table_colnames(tbl)...,)

# Several tasks may open columns at once, so the shared cache is locked:
function Base.getproperty(tbl::LH5LazyTable, name::Symbol)
    lock(_table_lock(tbl)) do
        get!(_table_columns(tbl), name) do
            hasproperty(tbl, name) || throw(ArgumentError("table has no column \"$name\""))
            LH5Array(_table_store(tbl)[String(name)])
        end
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
    lock(_table_lock(tbl)) do
        cached = getfield(tbl, :typed)
        isnothing(cached) || return cached
        names = propertynames(tbl)
        # Nested tables become typed tables as well, since StructArrays wants arrays as columns:
        cols = map(name -> _typed_column(getproperty(tbl, name)), names)
        typed = StructArray(NamedTuple{names}(cols))
        setfield!(tbl, :typed, typed)
        typed
    end
end

"""
    _refresh_cache!(tbl::LH5LazyTable)

Bring the cached columns up to date with the file. Columns whose dataset was
replaced are reopened, ragged arrays take their current element pointers and
waveforms their current time axes, in place. Appends through `tbl` call this
first, since other handles may have changed the table in the meantime.
Returns whether a column was reopened, which makes tables built from the old
columns stale.
"""
function _refresh_cache!(tbl::LH5LazyTable)
    lock(_table_lock(tbl)) do
        cols = _table_columns(tbl)
        store = _table_store(tbl)
        reopened = false
        for (name, col) in cols
            current = store[String(name)]
            if _disk_column(col) && _same_object(_column_object(col), current)
                close(current)
                reopened |= _refresh!(col)
            else
                cols[name] = LH5Array(current)
                reopened = true
            end
        end
        reopened && setfield!(tbl, :typed, nothing)
        reopened
    end
end

_refresh!(tbl::LH5LazyTable) = _refresh_cache!(tbl)

_typed_table(tbl::StructArray{<:NamedTuple}) = tbl

_typed_column(col) = col
_typed_column(col::LH5LazyTable) = _typed_table(col)

StructArrays.StructArray(tbl::LH5LazyTable) = _typed_table(tbl)

function Base.length(tbl::LH5LazyTable)
    names = _table_colnames(tbl)
    isempty(names) ? 0 : length(getproperty(tbl, first(names)))
end

Base.size(tbl::LH5LazyTable) = (length(tbl),)
Base.isempty(tbl::LH5LazyTable) = length(tbl) == 0
Base.firstindex(tbl::LH5LazyTable) = 1
Base.lastindex(tbl::LH5LazyTable) = length(tbl)
# Indexing and iteration, and so eachindex and pairs, run over rows:
Base.keys(tbl::LH5LazyTable) = Base.OneTo(length(tbl))

# Lazy tables are not arrays, so Base would silently collect them into a
# vector of handles, or append a handle to the rows of the other arguments.
# A table in the first two positions is caught, which covers `vcat(tbl, x)`,
# `vcat(x, tbl)` and reductions with an initial value:
_lazy_concat() = throw(ArgumentError("Lazy tables cannot be concatenated, read them with tbl[:] first"))
for f in (:vcat, :hcat)
    @eval begin
        Base.$f(::LH5LazyTable, ::Any...) = _lazy_concat()
        Base.$f(::AbstractArray, ::LH5LazyTable, ::Any...) = _lazy_concat()
    end
end
Base.cat(::LH5LazyTable, ::Any...; dims) = _lazy_concat()
Base.cat(::AbstractArray, ::LH5LazyTable, ::Any...; dims) = _lazy_concat()

# Left deliberately narrow: a bare `idxs...` is ambiguous with Indexing.jl's
# dictionary-indexed getindex/view.
const _TableIndex = Union{Integer, AbstractVector, Colon}
Base.getindex(tbl::LH5LazyTable, i::Integer) = _typed_table(tbl)[i]
# Selections rebuild the row type from the columns they read:
Base.getindex(tbl::LH5LazyTable, idxs::Union{AbstractVector, Colon}) =
    _getindex_columns(_typed_table(tbl), idxs)
Base.view(tbl::LH5LazyTable, idxs::_TableIndex...) = view(_typed_table(tbl), idxs...)
Base.iterate(tbl::LH5LazyTable, state...) = iterate(_typed_table(tbl), state...)
Base.IteratorEltype(::Type{LH5LazyTable}) = Base.EltypeUnknown()
Base.broadcastable(tbl::LH5LazyTable) = _typed_table(tbl)

# Row by row, these would read every column once per row:
Base.collect(tbl::LH5LazyTable) = collect(tbl[:])
Base.:(==)(a::LH5LazyTable, b::LH5LazyTable) = a[:] == b[:]
Base.:(==)(a::LH5LazyTable, b::AbstractArray) = a[:] == b
Base.:(==)(a::AbstractArray, b::LH5LazyTable) = a == b[:]
Base.hash(tbl::LH5LazyTable, h::UInt) = hash(tbl[:], h)

Tables.istable(::Type{LH5LazyTable}) = true
Tables.columnaccess(::Type{LH5LazyTable}) = true
Tables.rowaccess(::Type{LH5LazyTable}) = false
Tables.columns(tbl::LH5LazyTable) = tbl
Tables.columnnames(tbl::LH5LazyTable) = propertynames(tbl)
# Tables.jl consumers want arrays as columns, so nested tables come as StructArrays:
Tables.getcolumn(tbl::LH5LazyTable, name::Symbol) = _typed_column(getproperty(tbl, name))
Tables.getcolumn(tbl::LH5LazyTable, i::Int) = Tables.getcolumn(tbl, _table_colnames(tbl)[i])
Tables.columnindex(tbl::LH5LazyTable, name::Symbol) =
    something(findfirst(==(name), _table_colnames(tbl)), 0)
Tables.columntype(tbl::LH5LazyTable, name::Symbol) = eltype(Tables.getcolumn(tbl, name))
Tables.subset(tbl::LH5LazyTable, idxs; viewhint::Union{Bool, Nothing} = nothing) =
    viewhint === true ? view(tbl, idxs) : tbl[idxs]
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
