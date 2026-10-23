# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

"""
    LH5TypedTable{R<:NamedTuple} <: AbstractLH5Table{R}

A lazy table with a fixed row type. PropertyFunctions selections over lazy
tables return one: `(@pf (e = \$energy, t = \$timestamp)).(tbl)` has the
columns `energy` and `timestamp` of `tbl` under the names `e` and `t`, and
nested paths such as `\$geds.energy` select from nested tables. The row type,
and the types of the columns of a block of rows, are fixed when the selection
is made, from the columns of the first file, which is the only file a
selection over a file chain opens. So rows read with a range or an index
vector, iteration and `filterby` have inferable types, and `Tables.schema`
is known.

Rows are read from the selected columns only. Blocks of rows take the column
types of the first part of a chain and convert those of the other parts, and
a value that does not fit throws. The columns, `sel.e`, are the lazy columns
of the table selected from. `append!` refuses typed tables.
"""
struct LH5TypedTable{R<:NamedTuple, B<:StructArray{R, 1}, T<:AbstractLH5Table} <: AbstractLH5Table{R}
    table::T
    paths::Vector{Tuple{Vararg{Symbol}}}
end

"""
    LH5TypedTable(tbl::AbstractLH5Table, names, paths)

The columns of `tbl` at `paths`, each a tuple of the names leading to a
column, as a typed table with the columns named `names`.
"""
function LH5TypedTable(tbl::AbstractLH5Table, names::NTuple{N, Symbol}, paths::NTuple{N, Tuple{Vararg{Symbol}}}) where {N}
    allunique(names) || throw(ArgumentError("Selected column names are not unique: $(join(names, ", "))"))
    projected = _select(tbl, unique(map(first, paths)))
    block = _block_columns(_pick_columns(projected[1:0], names, paths))
    LH5TypedTable{eltype(block), typeof(block), typeof(projected)}(projected, collect(paths))
end

# A selection of a typed table selects from the table it was made from:
LH5TypedTable(tbl::LH5TypedTable, names::NTuple{N, Symbol}, paths::NTuple{N, Tuple{Vararg{Symbol}}}) where {N} =
    LH5TypedTable(_typed_parent(tbl), names, map(path -> (_column_path(tbl, first(path))..., Base.tail(path)...), paths))

_typed_parent(tbl::LH5TypedTable) = getfield(tbl, :table)

function _column_path(tbl::LH5TypedTable{R}, name::Symbol) where {R}
    i = findfirst(==(name), fieldnames(R))
    isnothing(i) && _no_column(name)
    getfield(tbl, :paths)[i]
end

Base.propertynames(::LH5TypedTable{R}) where {R} = fieldnames(R)
_table_colnames(::LH5TypedTable{R}) where {R} = collect(Symbol, fieldnames(R))
_column(tbl::LH5TypedTable, name::Symbol) = foldl(getproperty, _column_path(tbl, name); init = _typed_parent(tbl))

Base.length(tbl::LH5TypedTable) = length(_typed_parent(tbl))
Base.isempty(tbl::LH5TypedTable) = isempty(_typed_parent(tbl))

function Base.first(tbl::LH5TypedTable)
    isempty(tbl) && throw(ArgumentError("collection must be non-empty"))
    tbl[1]
end
Base.first(tbl::LH5TypedTable, n::Integer) = _typed_block(tbl, first(_typed_parent(tbl), n))

# The selected columns of a block of rows, under their new names:
_pick_columns(rows::StructArray, names::Tuple, paths) =
    StructArray(NamedTuple{names}(map(path -> foldl(getproperty, path; init = rows), paths)))

# Blocks hold their data: lazily read columns are read, and ranges are
# collected, since reads with index vectors give vectors:
_block_columns(col) = _materialize(col)
_block_columns(r::AbstractRange) = collect(r)
_block_columns(rows::StructArray{<:NamedTuple}) = StructArray(map(_block_columns, StructArrays.components(rows)))

# Every block of rows has the type found at selection:
_typed_block(tbl::LH5TypedTable{R, B}, rows::StructArray) where {R, B} =
    _conform_block(_block_columns(_pick_columns(rows, fieldnames(R), Tuple(getfield(tbl, :paths)))), B)::B

_getindex_rows(tbl::LH5TypedTable, idxs) = _typed_block(tbl, _typed_parent(tbl)[idxs])

# The iteration state of a typed table wraps that of the table it selects from:
struct _TypedState{S}
    state::S
end

_first_block_state(tbl::LH5TypedTable) = _TypedState(_first_block_state(_typed_parent(tbl)))

function _next_block(tbl::LH5TypedTable, state::_TypedState)
    next = _next_block(_typed_parent(tbl), state.state)
    isnothing(next) ? nothing : (_typed_block(tbl, first(next)), _TypedState(last(next)))
end

# Blocks take the column types of the first part, so that reads over any
# rows return one type. Values that do not fit those types throw:
function _conform_block(rows::StructArray, ::Type{B}) where {R, C, B<:StructArray{R, 1, C}}
    rows isa B && return rows
    cols = StructArrays.components(rows)
    StructArray{R}(NamedTuple{fieldnames(C)}(map(name -> _convert_column(fieldtype(C, name), getproperty(cols, name)), fieldnames(C))))
end

_convert_column(::Type{T}, col) where {T} = col isa T ? col : throw(ArgumentError(
    "A column of type $(typeof(col)) cannot take the type $T of the first part"))
_convert_column(::Type{T}, col::AbstractArray) where {T<:Array} = convert(T, col)
_convert_column(::Type{T}, col::StructArray) where {T<:StructArray} = _conform_block(col, T)
# Waveforms are StructArrays of their time and signal columns, so they convert as nested tables:
_convert_column(::Type{T}, col::VectorOfVectors) where {T<:VectorOfVectors} = col isa T ? col :
    VectorOfArrays(_convert_column(fieldtype(T, :data), flatview(col)),
        convert(fieldtype(T, :elem_ptr), ArraysOfArrays.internal_element_ptr(col)), convert(fieldtype(T, :kernel_size), col.kernel_size))
_convert_column(::Type{T}, col::ArrayOfSimilarArrays) where {T<:ArrayOfSimilarArrays} = col isa T ? col :
    ArrayOfSimilarArrays{eltype(fieldtype(T, :data)), ndims(eltype(T))}(_convert_column(fieldtype(T, :data), flatview(col)))

# A join selects from the table a typed table was made from, with the block
# types of the columns it takes:
function _select(tbl::LH5TypedTable{R, B}, names) where {R, B}
    issetequal(names, fieldnames(R)) && return tbl
    selected = (names...,)
    paths = map(name -> _column_path(tbl, name), selected)
    projected = _select(_typed_parent(tbl), unique(map(first, paths)))
    block = _block_type(B, selected)
    LH5TypedTable{eltype(block), block, typeof(projected)}(projected, collect(paths))
end

# The type of the blocks of some of the columns of blocks of type `B`:
_block_type(::Type{StructArray{R, 1, C, I}}, names::Tuple) where {R, C, I} = StructArray{
    NamedTuple{names, Tuple{map(name -> fieldtype(R, name), names)...}}, 1,
    NamedTuple{names, Tuple{map(name -> fieldtype(C, name), names)...}}, I}

Tables.schema(::LH5TypedTable{R}) where {R} = Tables.Schema(fieldnames(R), fieldtypes(R))
Tables.columntype(::LH5TypedTable{R}, name::Symbol) where {R} = fieldtype(R, name)

Base.append!(tbl::LH5TypedTable, ::Any) = _read_only(tbl)
Base.append!(tbl::LH5TypedTable, ::EncodedArray) = _read_only(tbl)
