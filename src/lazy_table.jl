# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

"""
    LH5LazyTable <: AbstractVector{NamedTuple}

A table stored in an LH5 file whose columns are opened on demand.

Reading a table returns one of these: a vector of `NamedTuple` rows whose
column names come from the file, while their element types are only known
once a column is opened. Accessing a column, with `tbl.colname` or
`Tables.getcolumn`, opens that column and no other, and `Tables.columns(tbl)`
is a dictionary of the columns (see [`LH5LazyColumns`](@ref)). Column-wise
broadcasts, such as PropertyFunctions, open only the columns they use.

Treating the table as an array opens all columns. Indexing with a range,
index vector or mask reads the selected rows into a `StructArray`, `tbl[:]`
reads the whole table. Iteration reads the rows in blocks of a few thousand,
so loops, comprehensions and `map` read neither one row at a time nor
everything at once. `filter`, `findall` and reductions go through iteration,
`sort` and `sortperm` read the table first. Concatenation is refused, and so
is `StructArray(tbl)`: once a column was read eagerly, such a `StructArray`
would slice and append rows with a stale row type and bypass the append
checks.

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
mutable struct LH5LazyTable <: AbstractVector{NamedTuple}
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

Open every column and return them as a `StructArray`, which indexing and
appending work on. Memoized, since building it specializes on the full
column type tuple.
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

# Once a column was read eagerly, a StructArray over the columns would slice
# and append rows with a stale row type and bypass the append checks:
StructArrays.StructArray(::LH5LazyTable; kwargs...) = throw(ArgumentError(
    "Lazy tables do not convert to StructArrays, read the rows with tbl[:]"))

# The shape of a column stored as a dataset tells the row count without
# opening the column, a ragged column would read its element pointers:
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

Base.size(tbl::LH5LazyTable) = (length(tbl),)
Base.IndexStyle(::Type{LH5LazyTable}) = IndexLinear()

# Selections rebuild the row type from the columns they read, and a single
# row holds its data, where a row of the typed table would hold views:
Base.getindex(tbl::LH5LazyTable, i::Integer) = tbl[i:i][1]
Base.getindex(tbl::LH5LazyTable, ::Colon) = _getindex_columns(_typed_table(tbl), :)
Base.getindex(tbl::LH5LazyTable, idxs::AbstractVector) = _getindex_columns(_typed_table(tbl), idxs)
function Base.getindex(tbl::LH5LazyTable, idxs::AbstractArray)
    checkbounds(tbl, idxs)
    reshape(tbl[vec(idxs)], axes(idxs))
end

"""
    _rows_per_block

Rows that iterating over an `LH5LazyTable` reads at a time.
"""
const _rows_per_block = 4096

# Reading rows in blocks keeps loops over a table from reading one row at a
# time, and keeps their memory bounded. The state holds the current block:
function Base.iterate(tbl::LH5LazyTable, state = (1, nothing, 0))
    i, block, offset = state
    if block === nothing || i > offset + length(block)
        n = length(tbl)
        i > n && return nothing
        block = tbl[i:min(i + _rows_per_block - 1, n)]
        offset = i - 1
    end
    (block[i - offset], (i + 1, block, offset))
end

Base.broadcastable(tbl::LH5LazyTable) = _typed_table(tbl)

# Base copies, filters and reduces arrays by indexing them, which would read
# one row at a time. These go through iteration or read the table at once:
function Base.copyto!(dest::Array, src::LH5LazyTable)
    length(dest) < length(src) && throw(BoundsError(dest, LinearIndices(src)))
    for (i, row) in zip(eachindex(dest), src)
        @inbounds dest[i] = row
    end
    dest
end
Base.copyto!(dest::Array, doffs::Integer, src::LH5LazyTable, soffs::Integer, n::Integer) =
    copyto!(dest, doffs, src[soffs:soffs + n - 1], 1, n)
Base.copy(tbl::LH5LazyTable) = tbl[:]
Base.collect(tbl::LH5LazyTable) = collect(tbl[:])
Base.hash(tbl::LH5LazyTable, h::UInt) = hash(tbl[:], h)
Base.mapreduce(f, op, tbl::LH5LazyTable; dims = :, kw...) =
    dims === Colon() ? mapfoldl(f, op, tbl; kw...) : mapreduce(f, op, tbl[:]; dims, kw...)
Base.findall(f::Function, tbl::LH5LazyTable) = findall(map(f, tbl))
Base.findall(f::Base.Fix2{typeof(in)}, tbl::LH5LazyTable) = findall(map(f, tbl))
Base.filter(f, tbl::LH5LazyTable) = tbl[findall(map(f, tbl))]
Base.sort(tbl::LH5LazyTable; kw...) = sort(tbl[:]; kw...)
Base.sortperm(tbl::LH5LazyTable; kw...) = sortperm(tbl[:]; kw...)

# Concatenation would copy the rows one at a time. A table in the first two
# positions among arrays is caught, as is a reduction over a vector of tables;
# a table behind two arrays is not. vcat has methods for vectors alone, which
# need guards of their own to stay unambiguous:
_lazy_concat() = throw(ArgumentError("Lazy tables cannot be concatenated, read them with tbl[:] first"))
for f in (:vcat, :hcat)
    @eval begin
        Base.$f(::LH5LazyTable, ::AbstractVecOrMat...) = _lazy_concat()
        Base.$f(::AbstractVecOrMat, ::LH5LazyTable, ::AbstractVecOrMat...) = _lazy_concat()
        Base.$f(::LH5LazyTable, ::LH5LazyTable, ::AbstractVecOrMat...) = _lazy_concat()
        Base.reduce(::typeof($f), ::AbstractVector{<:LH5LazyTable}) = _lazy_concat()
    end
end
Base.vcat(::AbstractVector, ::LH5LazyTable, ::AbstractVector...) = _lazy_concat()
Base.vcat(::LH5LazyTable, ::LH5LazyTable, ::AbstractVector...) = _lazy_concat()
Base.cat(::LH5LazyTable, ::AbstractArray...; dims) = _lazy_concat()
Base.cat(::AbstractArray, ::LH5LazyTable, ::AbstractArray...; dims) = _lazy_concat()
Base.cat(::LH5LazyTable, ::LH5LazyTable, ::AbstractArray...; dims) = _lazy_concat()
# DiskArrays has cat methods of the same shape:
Base.cat(::LH5LazyTable, ::DiskArrays.AbstractDiskArray, ::AbstractArray...; dims) = _lazy_concat()
Base.cat(::DiskArrays.AbstractDiskArray, ::LH5LazyTable, ::AbstractArray...; dims) = _lazy_concat()

"""
    LH5LazyColumns <: AbstractDict{Symbol, AbstractVector}

The columns of an [`LH5LazyTable`](@ref), as `Tables.columns` returns them: an
ordered, read-only dictionary from column name to column that opens a column
when it is looked up. `cols.energy` and `cols[:energy]` are the same lookup.
Nested tables are `LH5LazyTable`s themselves.
"""
struct LH5LazyColumns <: AbstractDict{Symbol, AbstractVector}
    table::LH5LazyTable
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

Tables.istable(::Type{LH5LazyTable}) = true
Tables.columnaccess(::Type{LH5LazyTable}) = true
# Rows come from iteration, so Tables.rows(tbl) is tbl itself:
Tables.rowaccess(::Type{LH5LazyTable}) = true
Tables.rowaccess(::Type{LH5LazyColumns}) = true
Tables.rows(cols::LH5LazyColumns) = _columns_table(cols)
Tables.columns(tbl::LH5LazyTable) = LH5LazyColumns(tbl)
Tables.columnnames(tbl::LH5LazyTable) = propertynames(tbl)
Tables.getcolumn(tbl::LH5LazyTable, name::Symbol) = getproperty(tbl, name)
Tables.getcolumn(tbl::LH5LazyTable, i::Int) = getproperty(tbl, _table_colnames(tbl)[i])
Tables.columnindex(tbl::LH5LazyTable, name::Symbol) =
    something(findfirst(==(name), _table_colnames(tbl)), 0)
Tables.columntype(tbl::LH5LazyTable, name::Symbol) = eltype(Tables.getcolumn(tbl, name))
# A view would read the rows one at a time, so the hint goes unheeded:
Tables.subset(tbl::LH5LazyTable, idxs; viewhint::Union{Bool, Nothing} = nothing) = tbl[idxs]
# Tables.jl would read a view row by row and infer its columns from the rows,
# losing them for an empty selection. The selection is read at once instead:
const LH5LazyTableView = SubArray{<:NamedTuple, 1, LH5LazyTable}
Tables.columnaccess(::Type{<:LH5LazyTableView}) = true
Tables.columns(v::LH5LazyTableView) = parent(v)[only(parentindices(v))]
# The column types are only known once the columns are opened:
Tables.schema(::Union{LH5LazyTable, LH5LazyColumns}) = nothing
# A lazy table cannot be built from another source, so sinks get the rows in memory:
Tables.materializer(::Union{LH5LazyTable, LH5LazyColumns}) = _in_memory_table
_in_memory_table(x) = _materialize(StructArray(Tables.columntable(x)))
DataAPI.nrow(tbl::LH5LazyTable) = length(tbl)
DataAPI.ncol(tbl::LH5LazyTable) = length(_table_colnames(tbl))

function Base.show(io::IO, tbl::LH5LazyTable)
    names = _table_colnames(tbl)
    opened = length(_table_columns(tbl))
    print(io, "LH5LazyTable(", length(names), " columns")
    opened > 0 && print(io, ", ", opened, " opened")
    print(io, "): ")
    join(io, names, ", ")
end

Base.show(io::IO, ::MIME"text/plain", tbl::LH5LazyTable) = show(io, tbl)

function Base.show(io::IO, cols::LH5LazyColumns)
    print(io, "Tables.columns(")
    show(io, _columns_table(cols))
    print(io, ")")
end

Base.show(io::IO, ::MIME"text/plain", cols::LH5LazyColumns) = show(io, cols)
