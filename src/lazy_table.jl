# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

"""
    AbstractLH5Table <: AbstractVector{NamedTuple}

Supertype of the lazy tables of LH5 data: [`LH5LazyTable`](@ref).

A lazy table is a vector of `NamedTuple` rows whose column names are known,
while their element types are only known once a column is opened. Accessing
a column, with `tbl.colname` or `Tables.getcolumn`, opens that column and no
other, and `Tables.columns(tbl)` is a dictionary of the columns (see
[`LH5LazyColumns`](@ref)). Column-wise broadcasts, such as PropertyFunctions,
open only the columns they use.

Treating the table as an array opens all columns. Indexing with a range,
index vector or mask reads the selected rows into a `StructArray`, `tbl[:]`
reads the whole table. Iteration reads the rows in blocks of a few thousand,
so loops, comprehensions and `map` read neither one row at a time nor
everything at once. `filter`, `findall` and reductions go through iteration,
`sort` and `sortperm` read the table first. Concatenation is refused, and so
is `StructArray(tbl)`: once a column was read eagerly, such a `StructArray`
would slice and append rows with a stale row type and bypass the append
checks.

Lazy tables stop working when the file they read from is closed, the rows
read from them do not.

# Implementation

Subtypes implement `_table_colnames(tbl)`, `_column(tbl, name)` and
`length(tbl)`, and may specialize `_typed_table(tbl)` and
`_getindex_rows(tbl, idxs)`.
"""
abstract type AbstractLH5Table <: AbstractVector{NamedTuple} end

"""
    LH5LazyTable <: AbstractLH5Table

A table stored in an LH5 file whose columns are opened on demand. Reading a
table returns one. It behaves as described for [`AbstractLH5Table`](@ref),
and `append!` appends to the stored columns.

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
mutable struct LH5LazyTable <: AbstractLH5Table
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

Base.propertynames(tbl::AbstractLH5Table) = (_table_colnames(tbl)...,)
Base.getproperty(tbl::AbstractLH5Table, name::Symbol) = _column(tbl, name)
Base.getindex(tbl::AbstractLH5Table, name::Symbol) = getproperty(tbl, name)
Base.getindex(tbl::AbstractLH5Table, name::AbstractString) = getproperty(tbl, Symbol(name))

_no_column(name) = throw(ArgumentError("table has no column \"$name\""))

# Several tasks may open columns at once, so the shared cache is locked:
function _column(tbl::LH5LazyTable, name::Symbol)
    lock(_table_lock(tbl)) do
        get!(_table_columns(tbl), name) do
            hasproperty(tbl, name) || _no_column(name)
            LH5Array(_table_store(tbl)[String(name)])
        end
    end
end

"""
    _typed_table(tbl::AbstractLH5Table)

Open every column and return them as a `StructArray`, which indexing and
appending work on.
"""
function _typed_table(tbl::AbstractLH5Table)
    names = propertynames(tbl)
    StructArray(NamedTuple{names}(map(name -> _typed_column(getproperty(tbl, name)), names)))
end

# Memoized, since building it specializes on the full column type tuple:
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
_typed_column(col::AbstractLH5Table) = _typed_table(col)

# Once a column was read eagerly, a StructArray over the columns would slice
# and append rows with a stale row type and bypass the append checks:
StructArrays.StructArray(::AbstractLH5Table; kwargs...) = throw(ArgumentError(
    "Lazy tables do not convert to StructArrays, read the rows with tbl[:]"))

# The shape of a column stored as a dataset gives the row count without
# opening the column. A ragged column would read its element pointers:
function Base.length(tbl::LH5LazyTable)
    names = _table_colnames(tbl)
    isempty(names) && return 0
    store = _table_store(tbl)
    i = findfirst(name -> _with_object(obj -> obj isa HDF5.Dataset, store, String(name)), names)
    isnothing(i) ? length(getproperty(tbl, first(names))) : _with_object(_nrows, store, String(names[i]))
end

_nrows(ds::HDF5.Dataset) = last(size(ds))

function _with_object(f, store, path::AbstractString)
    obj = store[path]
    try
        f(obj)
    finally
        close(obj)
    end
end

Base.size(tbl::AbstractLH5Table) = (length(tbl),)
Base.IndexStyle(::Type{<:AbstractLH5Table}) = IndexLinear()

# Selections rebuild the row type from the columns they read, and a single
# row holds its data, where a row of the typed table would hold views:
Base.getindex(tbl::AbstractLH5Table, i::Integer) = tbl[i:i][1]
Base.getindex(tbl::AbstractLH5Table, ::Colon) = _getindex_rows(tbl, :)
Base.getindex(tbl::AbstractLH5Table, idxs::AbstractVector) = _getindex_rows(tbl, idxs)
function Base.getindex(tbl::AbstractLH5Table, idxs::AbstractArray)
    checkbounds(tbl, idxs)
    reshape(tbl[vec(idxs)], axes(idxs))
end

"""
    _rows_per_block

Number of rows that iteration over a lazy table reads at a time.
"""
const _rows_per_block = 4096

_getindex_rows(tbl::AbstractLH5Table, idxs) = _getindex_columns(_typed_table(tbl), idxs)

# Reading rows in blocks keeps loops over a table from reading one row at a
# time, and keeps their memory bounded. The state holds the current block:
function Base.iterate(tbl::AbstractLH5Table, state = (1, nothing, 0))
    i, block, offset = state
    if block === nothing || i > offset + length(block)
        n = length(tbl)
        i > n && return nothing
        block = tbl[i:min(i + _rows_per_block - 1, n)]
        offset = i - 1
    end
    (block[i - offset], (i + 1, block, offset))
end

Base.broadcastable(tbl::AbstractLH5Table) = _typed_table(tbl)

# Base copies, filters and reduces arrays by indexing them, which would read
# one row at a time. These go through iteration or read the table at once:
function Base.copyto!(dest::Array, src::AbstractLH5Table)
    length(dest) < length(src) && throw(BoundsError(dest, LinearIndices(src)))
    for (i, row) in zip(eachindex(dest), src)
        @inbounds dest[i] = row
    end
    dest
end
Base.copyto!(dest::Array, doffs::Integer, src::AbstractLH5Table, soffs::Integer, n::Integer) =
    copyto!(dest, doffs, src[soffs:soffs + n - 1], 1, n)
Base.copy(tbl::AbstractLH5Table) = tbl[:]
Base.collect(tbl::AbstractLH5Table) = collect(tbl[:])
Base.hash(tbl::AbstractLH5Table, h::UInt) = hash(tbl[:], h)
Base.mapreduce(f, op, tbl::AbstractLH5Table; dims = :, kw...) =
    dims === Colon() ? mapfoldl(f, op, tbl; kw...) : mapreduce(f, op, tbl[:]; dims, kw...)
Base.findall(f::Function, tbl::AbstractLH5Table) = findall(map(f, tbl))
Base.findall(f::Base.Fix2{typeof(in)}, tbl::AbstractLH5Table) = findall(map(f, tbl))
Base.filter(f, tbl::AbstractLH5Table) = tbl[findall(map(f, tbl))]
Base.sort(tbl::AbstractLH5Table; kw...) = sort(tbl[:]; kw...)
Base.sortperm(tbl::AbstractLH5Table; kw...) = sortperm(tbl[:]; kw...)

# Concatenation would copy the rows one at a time. A table in the first two
# positions among arrays is caught, as is a reduction over a vector of tables;
# a table behind two arrays is not. vcat has methods for vectors alone, which
# need guards of their own to stay unambiguous:
_lazy_concat() = throw(ArgumentError("Lazy tables cannot be concatenated, read them with tbl[:] first"))
for f in (:vcat, :hcat)
    @eval begin
        Base.$f(::AbstractLH5Table, ::AbstractVecOrMat...) = _lazy_concat()
        Base.$f(::AbstractVecOrMat, ::AbstractLH5Table, ::AbstractVecOrMat...) = _lazy_concat()
        Base.$f(::AbstractLH5Table, ::AbstractLH5Table, ::AbstractVecOrMat...) = _lazy_concat()
        Base.reduce(::typeof($f), ::AbstractVector{<:AbstractLH5Table}) = _lazy_concat()
    end
end
Base.vcat(::AbstractVector, ::AbstractLH5Table, ::AbstractVector...) = _lazy_concat()
Base.vcat(::AbstractLH5Table, ::AbstractLH5Table, ::AbstractVector...) = _lazy_concat()
Base.cat(::AbstractLH5Table, ::AbstractArray...; dims) = _lazy_concat()
Base.cat(::AbstractArray, ::AbstractLH5Table, ::AbstractArray...; dims) = _lazy_concat()
Base.cat(::AbstractLH5Table, ::AbstractLH5Table, ::AbstractArray...; dims) = _lazy_concat()
# DiskArrays has cat methods of the same shape:
Base.cat(::AbstractLH5Table, ::DiskArrays.AbstractDiskArray, ::AbstractArray...; dims) = _lazy_concat()
Base.cat(::DiskArrays.AbstractDiskArray, ::AbstractLH5Table, ::AbstractArray...; dims) = _lazy_concat()

"""
    LH5LazyColumns <: AbstractDict{Symbol, AbstractVector}

The columns of a lazy table (see [`AbstractLH5Table`](@ref)), as
`Tables.columns` returns them: an ordered, read-only dictionary from column
name to column that opens a column when it is looked up. `cols.energy` and
`cols[:energy]` are the same lookup. Nested tables are lazy tables themselves.
"""
struct LH5LazyColumns <: AbstractDict{Symbol, AbstractVector}
    table::AbstractLH5Table
end

_columns_table(cols::LH5LazyColumns) = getfield(cols, :table)

Base.length(cols::LH5LazyColumns) = length(_table_colnames(_columns_table(cols)))
Base.isempty(cols::LH5LazyColumns) = length(cols) == 0
Base.keys(cols::LH5LazyColumns) = propertynames(_columns_table(cols))
Base.haskey(cols::LH5LazyColumns, name::Symbol) = hasproperty(_columns_table(cols), name)
Base.getindex(cols::LH5LazyColumns, name::Symbol) =
    haskey(cols, name) ? getproperty(_columns_table(cols), name) : throw(KeyError(name))
Base.get(cols::LH5LazyColumns, name::Symbol, default) = haskey(cols, name) ? cols[name] : default
Base.propertynames(cols::LH5LazyColumns) = keys(cols)
Base.getproperty(cols::LH5LazyColumns, name::Symbol) = cols[name]
function Base.iterate(cols::LH5LazyColumns, i = 1)
    names = _table_colnames(_columns_table(cols))
    i > length(names) ? nothing : (names[i] => cols[names[i]], i + 1)
end

Tables.istable(::Type{<:AbstractLH5Table}) = true
Tables.columnaccess(::Type{<:AbstractLH5Table}) = true
# Rows come from iteration, so Tables.rows(tbl) is tbl itself:
Tables.rowaccess(::Type{<:AbstractLH5Table}) = true
Tables.rowaccess(::Type{LH5LazyColumns}) = true
Tables.rows(cols::LH5LazyColumns) = _columns_table(cols)
Tables.columns(tbl::AbstractLH5Table) = LH5LazyColumns(tbl)
Tables.columnnames(tbl::AbstractLH5Table) = propertynames(tbl)
Tables.getcolumn(tbl::AbstractLH5Table, name::Symbol) = getproperty(tbl, name)
Tables.getcolumn(tbl::AbstractLH5Table, i::Int) = getproperty(tbl, _table_colnames(tbl)[i])
Tables.columnindex(tbl::AbstractLH5Table, name::Symbol) =
    something(findfirst(==(name), _table_colnames(tbl)), 0)
Tables.columntype(tbl::AbstractLH5Table, name::Symbol) = eltype(Tables.getcolumn(tbl, name))
# A view would read the rows one at a time, so the hint is ignored:
Tables.subset(tbl::AbstractLH5Table, idxs; viewhint::Union{Bool, Nothing} = nothing) = tbl[idxs]
# Tables.jl would read a view row by row and infer its columns from the rows,
# losing them for an empty selection. The selection is read at once instead:
const LH5LazyTableView = SubArray{<:NamedTuple, 1, <:AbstractLH5Table}
Tables.columnaccess(::Type{<:LH5LazyTableView}) = true
Tables.columns(v::LH5LazyTableView) = parent(v)[only(parentindices(v))]
# The column types are only known once the columns are opened:
Tables.schema(::Union{AbstractLH5Table, LH5LazyColumns}) = nothing
# A lazy table cannot be built from another source, so sinks get the rows in memory:
Tables.materializer(::Union{AbstractLH5Table, LH5LazyColumns}) = _in_memory_table
_in_memory_table(x) = _materialize(StructArray(Tables.columntable(x)))
DataAPI.nrow(tbl::AbstractLH5Table) = length(tbl)
DataAPI.ncol(tbl::AbstractLH5Table) = length(_table_colnames(tbl))

# Showing a table must not read it:
function Base.show(io::IO, tbl::AbstractLH5Table)
    names = _table_colnames(tbl)
    print(io, nameof(typeof(tbl)), "(", length(names), " columns")
    _show_details(io, tbl)
    print(io, "): ")
    join(io, names, ", ")
end

Base.show(io::IO, ::MIME"text/plain", tbl::AbstractLH5Table) = show(io, tbl)

_show_details(::IO, ::AbstractLH5Table) = nothing

function _show_details(io::IO, tbl::LH5LazyTable)
    opened = length(_table_columns(tbl))
    opened > 0 && print(io, ", ", opened, " opened")
end

function Base.show(io::IO, cols::LH5LazyColumns)
    print(io, "Tables.columns(")
    show(io, _columns_table(cols))
    print(io, ")")
end

Base.show(io::IO, ::MIME"text/plain", cols::LH5LazyColumns) = show(io, cols)
