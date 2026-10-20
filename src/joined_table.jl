# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

"""
    LH5JoinedTable <: AbstractLH5Table

The columns of several tables side by side, without reading them.
`PropertyFunctions.innermerge` returns one for lazy tables, and so does
`Tables.materializer` of a lazy table when given columns that read from
disk.

The parts are lazy tables or `NamedTuple`s of columns, of equal length. A
column comes from the last part that has it, and the columns are ordered by
their first occurrence, as in `merge`. `append!` refuses joined tables. Their
columns are those of the parts, so writing to a column writes to its part.
"""
struct LH5JoinedTable{P<:Tuple} <: AbstractLH5Table
    parts::P
    colnames::Vector{Symbol}
    owners::Vector{Int}

    function LH5JoinedTable(parts::Union{AbstractLH5Table, NamedTuple}...)
        names, owners = Symbol[], Int[]
        for (k, part) in enumerate(parts), name in _part_colnames(part)
            i = findfirst(==(name), names)
            isnothing(i) ? (push!(names, name); push!(owners, k)) : (owners[i] = k)
        end
        lengths = mapreduce(_part_lengths, vcat, parts; init = Int[])
        allequal(lengths) || throw(DimensionMismatch(
            "Tables to join differ in length: $(join(unique(lengths), ", "))"))
        new{typeof(parts)}(parts, names, owners)
    end
end

_part_colnames(tbl::AbstractLH5Table) = _table_colnames(tbl)
_part_colnames(cols::NamedTuple) = collect(Symbol, keys(cols))

_part_column(tbl::AbstractLH5Table, name::Symbol) = getproperty(tbl, name)
_part_column(cols::NamedTuple, name::Symbol) = getfield(cols, name)

_part_lengths(tbl::AbstractLH5Table) = [length(tbl)]
_part_lengths(cols::NamedTuple) = Int[length(col) for col in values(cols)]

# Lazy tables and NamedTuples of columns join as they are, other tables with their columns:
_join_part(tbl::AbstractLH5Table) = tbl
_join_part(cols::NamedTuple) = cols
_join_part(tbl) = Tables.columntable(tbl)

_join_parts(tbl::LH5JoinedTable) = getfield(tbl, :parts)
_table_colnames(tbl::LH5JoinedTable) = getfield(tbl, :colnames)

function _column(tbl::LH5JoinedTable, name::Symbol)
    i = findfirst(==(name), _table_colnames(tbl))
    isnothing(i) && _no_column(name)
    _part_column(_join_parts(tbl)[getfield(tbl, :owners)[i]], name)
end

# The tables that provide columns must still agree in length:
function Base.length(tbl::LH5JoinedTable)
    parts, owners, names = _join_parts(tbl), getfield(tbl, :owners), _table_colnames(tbl)
    lengths = unique(mapreduce(k -> _owned_lengths(parts[k], names[owners .== k]), vcat, unique(owners); init = Int[]))
    length(lengths) <= 1 || throw(DimensionMismatch(
        "The tables of the join differ in length: $(join(lengths, ", "))"))
    isempty(lengths) ? 0 : only(lengths)
end

_owned_lengths(tbl::AbstractLH5Table, ::Any) = [length(tbl)]
_owned_lengths(cols::NamedTuple, names) = [length(getfield(cols, name)) for name in names]

# Each part reads the rows of the columns it provides, as they are now, which
# for a chained table touches only the files that hold them. The parts must
# still agree in length, or rows of different events would pair up:
function _getindex_rows(tbl::LH5JoinedTable, idxs)
    length(tbl)
    sel = idxs isa AbstractVector{Bool} ? _materialize(idxs) : idxs
    parts, owners, names = _join_parts(tbl), getfield(tbl, :owners), _table_colnames(tbl)
    rows = Dict(k => _join_rows(_select(parts[k], names[owners .== k]), sel) for k in unique(owners))
    StructArray(NamedTuple{(names...,)}(map((name, k) -> getproperty(rows[k], name), (names...,), Tuple(owners))))
end

# A part with only some of its columns, each of which must have the rows of the part:
function _select(tbl::AbstractLH5Table, names)
    issetequal(names, _table_colnames(tbl)) && return tbl
    cols = NamedTuple{(names...,)}(map(name -> getproperty(tbl, name), (names...,)))
    n = length(tbl)
    foreach(pairs(cols)) do (name, col)
        last(size(col)) == n || throw(DimensionMismatch("Column $name has $(last(size(col))) rows, its table has $n"))
    end
    LH5JoinedTable(cols)
end
_select(cols::NamedTuple, names) = NamedTuple{(names...,)}(cols)

# A join selects from its parts, so that their columns are not opened:
function _select(tbl::LH5JoinedTable, names)
    issetequal(names, _table_colnames(tbl)) && return tbl
    parts, owners, colnames = _join_parts(tbl), getfield(tbl, :owners), _table_colnames(tbl)
    selected = [k => colnames[(owners .== k) .& in.(colnames, Ref(names))] for k in unique(owners)]
    LH5JoinedTable((_select(parts[k], ks) for (k, ks) in selected if !isempty(ks))...)
end

_join_rows(tbl::AbstractLH5Table, idxs) = tbl[idxs]
_join_rows(cols::NamedTuple, idxs) = _getindex_columns(StructArray(cols), idxs)

_read_only(tbl) = throw(ArgumentError("$(nameof(typeof(tbl)))s are read-only, append to the stored tables instead"))
Base.append!(tbl::LH5JoinedTable, ::Any) = _read_only(tbl)
# EncodedArrays appends its arrays to any vector:
Base.append!(tbl::LH5JoinedTable, ::EncodedArray) = _read_only(tbl)
