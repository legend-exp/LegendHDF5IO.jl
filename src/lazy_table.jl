# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

"""
    AbstractLH5Vector{T} <: AbstractVector{T}

Supertype of the lazy vectors of LH5 data, which read their elements in
blocks: the lazy tables, [`AbstractLH5Table`](@ref), and lazy broadcasts over
typed tables, [`LH5LazyBroadcast`](@ref).

Indexing with a range, index vector or mask reads the selected elements, and
`v[:]` reads them all. Iteration reads the elements in blocks of a few
thousand (see [`LegendHDF5IO.rows_per_block`](@ref)), so loops,
comprehensions and `map` read neither one element at a time nor everything at
once. `filter`, searches such as `findall`, `findfirst` and `findmax`, and
reductions go through iteration, while `sort` and `sortperm` read the vector
first.

# Implementation

Subtypes implement `length(v)` and `_getindex_rows(v, idxs)`, which reads the
elements at `idxs`, and may specialize the block iteration,
`_first_block_state(v)` and `_next_block(v, state)`.
"""
abstract type AbstractLH5Vector{T} <: AbstractVector{T} end

"""
    AbstractLH5Table{R<:NamedTuple} <: AbstractLH5Vector{R}

Supertype of the lazy tables of LH5 data: [`LH5LazyTable`](@ref),
[`LH5JoinedTable`](@ref), [`LH5ChainedTable`](@ref) and
[`LH5TypedTable`](@ref).

A lazy table is a vector of `NamedTuple` rows. Its column names are known up
front, the element type of a column only once the column is opened, so the
row type `R` is the abstract `NamedTuple`, except for typed tables. Accessing
a column, with `tbl.colname` or `Tables.getcolumn`, opens that column and no
other, and `Tables.columns(tbl)` is a dictionary of the columns (see
[`LH5LazyColumns`](@ref)). A PropertyFunction broadcast over a lazy table is
an [`LH5LazyBroadcast`](@ref) that reads only the columns the function uses,
a pure column accessor returns the column, and selections return typed
tables.

Treating the table as an array opens all columns. Indexing with a range,
index vector or mask reads the selected rows into a `StructArray`, and
iteration reads the rows in blocks, as described for
[`AbstractLH5Vector`](@ref). `vcat` of lazy tables chains them.
Concatenation with other arrays is refused if a table is among the first two
arguments. A table behind two other arrays is concatenated row by row.
`StructArray(tbl)`, `TypedTables.Table(tbl)` and `TypedTables.FlexTable(tbl)`
are refused as well: they would append through the columns without the
table's checks, and a
`StructArray` of a column read eagerly would slice and append rows with a
stale row type.

Lazy tables stop working when the file they read from is closed, and a
handle stops working when its table is appended to or its columns are
changed through another handle: its reads and appends then throw, and the
table must be reopened. Its own appends keep it current. The rows read from
a table do not depend on it.

# Implementation

Subtypes implement `_table_colnames(tbl)`, `_column(tbl, name)` and
`length(tbl)`, and may specialize `_typed_table(tbl)`,
`_getindex_rows(tbl, idxs)` and the block iteration, `_next_block`.
"""
abstract type AbstractLH5Table{R<:NamedTuple} <: AbstractLH5Vector{R} end

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
mutable struct LH5LazyTable <: AbstractLH5Table{NamedTuple}
    data_store::HDF5.H5DataStore
    colnames::Vector{Symbol}
    columns::Dict{Symbol, Any}
    typed::Any
    lock::ReentrantLock
    key::Tuple{UInt, HDF5.API.H5O_token_t}
    generation::Int
end

function _lazy_table(ds::HDF5.H5DataStore, names)
    key = _object_key(ds)
    LH5LazyTable(ds, collect(Symbol, names), Dict{Symbol, Any}(), nothing, ReentrantLock(), key, _generation(key))
end

# The table is used only while its group is as it was opened (see `_generations`):
_check_current(tbl::LH5LazyTable) =
    getfield(tbl, :generation) == _generation(getfield(tbl, :key)) || throw(_stale("table"))

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
    _check_current(tbl)
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
    _check_current(tbl)
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

_typed_table(tbl::StructArray{<:NamedTuple}) = tbl

_typed_column(col) = col
_typed_column(col::AbstractLH5Table) = _typed_table(col)

# Once a column was read eagerly, a StructArray over the columns would slice
# and append rows with a stale row type and bypass the append checks. The
# TypedTables extension refuses Table(tbl) and FlexTable(tbl) the same way:
StructArrays.StructArray(::AbstractLH5Table; kwargs...) = throw(ArgumentError(
    "Lazy tables do not convert to StructArrays, read the rows with tbl[:]"))

# The shape of a column stored as a dataset gives the row count without
# opening the column. A ragged column would read its element pointers:
function Base.length(tbl::LH5LazyTable)
    _check_current(tbl)
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

Base.size(v::AbstractLH5Vector) = (length(v),)
Base.IndexStyle(::Type{<:AbstractLH5Vector}) = IndexLinear()

# A single element is read as a block of one, so a row of a table holds its
# data, where a row of the typed table would hold views:
Base.getindex(v::AbstractLH5Vector, i::Integer) = v[i:i][1]
Base.getindex(v::AbstractLH5Vector, ::Colon) = _getindex_rows(v, :)
Base.getindex(v::AbstractLH5Vector, idxs::AbstractVector) = _getindex_rows(v, idxs)
function Base.getindex(v::AbstractLH5Vector, idxs::AbstractArray)
    checkbounds(v, idxs)
    reshape(v[vec(idxs)], axes(idxs))
end

_getindex_rows(tbl::AbstractLH5Table, idxs) = _getindex_columns(_typed_table(tbl), idxs)

# Iteration reads elements in blocks, which keeps loops over a table from
# reading one row at a time and keeps their memory bounded. `_next_block(v,
# state)` returns the next block with the state after it, or `nothing` at the
# end, and `_first_block_state(v)` the state before the first block. Tables
# whose length is costly read part by part instead, and typed tables and
# broadcasts read the blocks of the table they are made from:
_first_block_state(::AbstractLH5Vector) = 1

function _next_block(v::AbstractLH5Vector, i::Int)
    n = length(v)
    i > n && return nothing
    stop = min(i + _rows_per_block() - 1, n)
    (v[i:stop], stop + 1)
end

# The state holds the current block and the position in it:
function Base.iterate(v::AbstractLH5Vector, state = (nothing, 0, _first_block_state(v)))
    block, pos, next_state = state
    while isnothing(block) || pos == length(block)
        next = _next_block(v, next_state)
        isnothing(next) && return nothing
        block, next_state = next
        pos = 0
    end
    (block[pos + 1], (block, pos + 1, next_state))
end

Base.broadcastable(tbl::AbstractLH5Table) = _typed_table(tbl)

# Base copies, filters, searches and reduces arrays by indexing them, which
# would read one element at a time. These go through iteration or read the
# vector at once:
function Base.copyto!(dest::Array, src::AbstractLH5Vector)
    length(dest) < length(src) && throw(BoundsError(dest, LinearIndices(src)))
    for (i, x) in zip(eachindex(dest), src)
        @inbounds dest[i] = x
    end
    dest
end
Base.copyto!(dest::Array, doffs::Integer, src::AbstractLH5Vector, soffs::Integer, n::Integer) =
    copyto!(dest, doffs, src[soffs:soffs + n - 1], 1, n)
Base.copy(v::AbstractLH5Vector) = v[:]
Base.collect(v::AbstractLH5Vector) = collect(v[:])
Base.hash(v::AbstractLH5Vector, h::UInt) = hash(v[:], h)
Base.mapreduce(f, op, v::AbstractLH5Vector; dims = :, kw...) =
    dims === Colon() ? mapfoldl(f, op, v; kw...) : mapreduce(f, op, v[:]; dims, kw...)
Base.findall(mask::AbstractLH5Vector{Bool}) = findall(mask[:])
Base.findall(f::Function, v::AbstractLH5Vector) = findall(map(f, v))
Base.findall(f::Base.Fix2{typeof(in)}, v::AbstractLH5Vector) = findall(map(f, v))
Base.filter(f, v::AbstractLH5Vector) = v[findall(map(f, v))]
Base.findmax(f, v::AbstractLH5Vector; dims = :) = findmax(map(f, v); dims)
Base.findmin(f, v::AbstractLH5Vector; dims = :) = findmin(map(f, v); dims)
Base.findmax(v::AbstractLH5Vector; dims = :) = findmax(identity, v; dims)
Base.findmin(v::AbstractLH5Vector; dims = :) = findmin(identity, v; dims)
function Base.findfirst(f::Function, v::AbstractLH5Vector)
    for (i, x) in enumerate(v)
        f(x) && return i
    end
    nothing
end
Base.findlast(f::Function, v::AbstractLH5Vector) = findlast(map(f, v))
Base.findfirst(mask::AbstractLH5Vector{Bool}) = findfirst(identity, mask)
Base.findlast(mask::AbstractLH5Vector{Bool}) = findlast(identity, mask)
Base.sort(v::AbstractLH5Vector; kw...) = sort(v[:]; kw...)
Base.sortperm(v::AbstractLH5Vector; kw...) = sortperm(v[:]; kw...)

# Lazy tables concatenate lazily with each other (see LH5ChainedTable). The
# second method resolves the ambiguity with the guards below:
Base.vcat(tbl::AbstractLH5Table, tbls::AbstractLH5Table...) = _chain_tables(Any[tbl, tbls...])
Base.vcat(tbl1::AbstractLH5Table, tbl2::AbstractLH5Table, tbls::AbstractLH5Table...) =
    _chain_tables(Any[tbl1, tbl2, tbls...])
Base.reduce(::typeof(vcat), tbls::AbstractVector{<:AbstractLH5Table}) = _chain_tables(tbls)
ArraysOfArrays.vecflattened(tbls::AbstractVector{<:AbstractLH5Table}) = _chain_tables(tbls)

# Concatenated with other arrays, they would be copied a row at a time. A
# table in the first two positions among arrays is caught, but not one behind
# two arrays. vcat has methods for vectors alone, which need guards of their
# own to stay unambiguous:
_lazy_concat(::typeof(vcat)) = throw(ArgumentError(
    "Lazy tables concatenate with lazy tables only, read them with tbl[:] first"))
_lazy_concat(::typeof(cat)) = throw(ArgumentError("Concatenate lazy tables with vcat"))
_lazy_concat(::typeof(hcat)) = throw(ArgumentError(
    "Lazy tables only concatenate vertically, join their columns with innermerge"))
for f in (:vcat, :hcat)
    @eval begin
        Base.$f(::AbstractLH5Table, ::AbstractVecOrMat...) = _lazy_concat($f)
        Base.$f(::AbstractVecOrMat, ::AbstractLH5Table, ::AbstractVecOrMat...) = _lazy_concat($f)
        Base.$f(::AbstractLH5Table, ::AbstractLH5Table, ::AbstractVecOrMat...) = _lazy_concat($f)
    end
end
Base.reduce(::typeof(hcat), ::AbstractVector{<:AbstractLH5Table}) = _lazy_concat(hcat)
Base.vcat(::AbstractVector, ::AbstractLH5Table, ::AbstractVector...) = _lazy_concat(vcat)
Base.vcat(::AbstractLH5Table, ::AbstractLH5Table, ::AbstractVector...) = _lazy_concat(vcat)
Base.cat(::AbstractLH5Table, ::AbstractArray...; dims) = _lazy_concat(cat)
Base.cat(::AbstractArray, ::AbstractLH5Table, ::AbstractArray...; dims) = _lazy_concat(cat)
Base.cat(::AbstractLH5Table, ::AbstractLH5Table, ::AbstractArray...; dims) = _lazy_concat(cat)
# DiskArrays has cat methods of the same shape:
Base.cat(::AbstractLH5Table, ::DiskArrays.AbstractDiskArray, ::AbstractArray...; dims) = _lazy_concat(cat)
Base.cat(::DiskArrays.AbstractDiskArray, ::AbstractLH5Table, ::AbstractArray...; dims) = _lazy_concat(cat)

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
# Tables.jl would look these up in the schema, which is not known:
Tables.columnindex(cols::LH5LazyColumns, name::Symbol) = Tables.columnindex(_columns_table(cols), name)
Tables.columntype(cols::LH5LazyColumns, name::Symbol) = Tables.columntype(_columns_table(cols), name)
# A view would read the rows one at a time, so the hint is ignored:
Tables.subset(tbl::AbstractLH5Table, idxs; viewhint::Union{Bool, Nothing} = nothing) = tbl[idxs]
# Tables.jl would read a view row by row and infer its columns from the rows,
# losing them for an empty selection. The selection is read at once instead:
const LH5LazyTableView = SubArray{<:NamedTuple, 1, <:AbstractLH5Table}
Tables.columnaccess(::Type{<:LH5LazyTableView}) = true
Tables.columns(v::LH5LazyTableView) = parent(v)[only(parentindices(v))]
# The column types are only known once the columns are opened:
Tables.schema(::Union{AbstractLH5Table, LH5LazyColumns}) = nothing
# Like those of StructArrays and TypedTables, the materializer wraps the
# columns it is given. Columns that read from disk go into a lazy table, which
# slices them safely:
Tables.materializer(::Union{AbstractLH5Table, LH5LazyColumns}) = _wrap_columns
_wrap_columns(tbl::AbstractLH5Table) = tbl
function _wrap_columns(x)
    cols = Tables.columntable(x)
    any(_reads_from_disk, cols) ? LH5JoinedTable(cols) : StructArray(cols)
end
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
