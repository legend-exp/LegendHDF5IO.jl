# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

"""
    LH5ChainedTable <: AbstractLH5Table

Tables one after the other, without reading them. `vcat` and
`reduce(vcat, ...)` of lazy tables return one, as do
`ArraysOfArrays.vecflattened` of a vector of lazy tables and indexing a file
chain with the path of a table (see [`LH5FileChain`](@ref)).

Rows are read from the parts that hold them, so `first(tbl, n)` reads the
first parts only. Accessing a column concatenates the columns of all parts
lazily, which opens the column in each part. The parts must have the same
columns, and rows take the column order of the first part. Chained tables
and their columns are read-only.
"""
struct LH5ChainedTable <: AbstractLH5Table{NamedTuple}
    parts::Vector{Any}
    colnames::Vector{Symbol}
    lengths::Vector{Int}
    columns::Dict{Symbol, Any}
    rownames::Base.RefValue{Any}
    lock::ReentrantLock
end

_chained(parts::Vector{Any}, colnames, lengths = fill(-1, length(parts))) = LH5ChainedTable(
    parts, collect(Symbol, colnames), lengths, Dict{Symbol, Any}(), Ref{Any}(nothing), ReentrantLock())

_chain_parts(tbl::LH5ChainedTable) = getfield(tbl, :parts)
_table_colnames(tbl::LH5ChainedTable) = getfield(tbl, :colnames)
_table_columns(tbl::LH5ChainedTable) = getfield(tbl, :columns)
_table_lock(tbl::LH5ChainedTable) = getfield(tbl, :lock)

# A part is a lazy table, or a reference that opens one when it is used.
# Nested tables in memory, such as columns of joined tables, can be parts:
const _MemoryTable = StructArray{<:NamedTuple}

_part_rows(tbl::Union{AbstractLH5Table, _MemoryTable}, idxs) = tbl[idxs]
_part_length(tbl::Union{AbstractLH5Table, _MemoryTable}) = length(tbl)
_is_subtable(tbl::Union{AbstractLH5Table, _MemoryTable}, name::Symbol) =
    getproperty(tbl, name) isa Union{AbstractLH5Table, _MemoryTable}
_subpart(tbl::Union{AbstractLH5Table, _MemoryTable}, name::Symbol) = getproperty(tbl, name)
# A column of a chain would be concatenated over every part to look at it:
_is_subtable(tbl::LH5ChainedTable, name::Symbol) = _is_subtable(first(_chain_parts(tbl)), name)
_part_colnames(tbl::_MemoryTable) = collect(Symbol, propertynames(tbl))
_part_column(tbl::_MemoryTable, name::Symbol) = getproperty(tbl, name)
_select(tbl::_MemoryTable, tree) = StructArray(NamedTuple{_tree_names(tree)}(map(entry -> _select_column(tbl, entry), (tree...,))))
_check_colnames(part, names) = issetequal(_part_colnames(part), names) ||
    throw(ArgumentError("Tables to concatenate have different columns: " *
        "$(join(names, ", ")) and $(join(_part_colnames(part), ", "))"))

# Lazy tables may be appended to, so lengths and columns are kept only for
# parts that cannot change:
_fixed_part(::Any) = false

function _chain_tables(tbls::AbstractVector)
    isempty(tbls) && throw(ArgumentError("No tables to concatenate"))
    parts, lengths = Any[], Int[]
    for tbl in tbls
        if tbl isa LH5ChainedTable
            append!(parts, _chain_parts(tbl))
            append!(lengths, getfield(tbl, :lengths))
        else
            push!(parts, tbl)
            push!(lengths, -1)
        end
    end
    names = _part_colnames(first(parts))
    foreach(part -> _check_colnames(part, names), parts)
    _chained(parts, names, lengths)
end

# Lengths are found part by part, as far as they are needed. The parts are
# of any type, so the count is asserted for the iteration state's sake:
function _part_length(tbl::LH5ChainedTable, k::Integer)::Int
    part = _chain_parts(tbl)[k]
    _fixed_part(part) || return _part_length(part)
    lengths = getfield(tbl, :lengths)
    lock(_table_lock(tbl)) do
        lengths[k] < 0 && (lengths[k] = _part_length(part))
        lengths[k]
    end
end

Base.length(tbl::LH5ChainedTable) =
    sum(k -> _part_length(tbl, k), eachindex(_chain_parts(tbl)); init = 0)

# Row counts up to and including each part, as far as needed to reach row `upto`:
function _part_ends(tbl::LH5ChainedTable, upto::Integer)
    ends = Int[]
    for k in eachindex(_chain_parts(tbl))
        push!(ends, (isempty(ends) ? 0 : last(ends)) + _part_length(tbl, k))
        last(ends) >= upto && break
    end
    ends
end

Base.isempty(tbl::LH5ChainedTable) =
    all(k -> _part_length(tbl, k) == 0, eachindex(_chain_parts(tbl)))

function Base.first(tbl::LH5ChainedTable)
    isempty(tbl) && throw(ArgumentError("collection must be non-empty"))
    tbl[1]
end

function Base.first(tbl::LH5ChainedTable, n::Integer)
    n >= 0 || throw(ArgumentError("Number of elements must be non-negative"))
    tbl[1:min(n, last(_part_ends(tbl, n)))]
end

# The column names of the first part, with those of its nested tables:
function _row_names(tbl::LH5ChainedTable)
    lock(_table_lock(tbl)) do
        cached = getfield(tbl, :rownames)
        isnothing(cached[]) && (cached[] = _name_tree(first(_chain_parts(tbl)), _table_colnames(tbl)))
        cached[]
    end
end

_name_tree(part, tree) = _build_name_tree(part, tree)

# A name tree names the columns of a part, with the columns of its nested
# tables as subtrees. Built from a selection (see `_select`), it expands the
# names of whole nested tables and keeps the subtrees of partly selected ones:
_build_name_tree(part, tree) = Tuple(map((tree...,)) do entry
    name = _tree_name(entry)
    _is_subtable(part, name) || return name
    sub = _subpart(part, name)
    name => _name_tree(sub, entry isa Pair ? last(entry) : _part_colnames(sub))
end)

_tree_name(name::Symbol) = name
_tree_name(entry::Pair) = first(entry)

# Rows of every part must have the columns of the first, and take their order:
function _conform(rows::StructArray{<:NamedTuple}, tree)
    names = map(_tree_name, tree)
    have = propertynames(rows)
    issetequal(have, names) || throw(ArgumentError(
        "Tables to concatenate have different columns: $(join(names, ", ")) and $(join(have, ", "))"))
    StructArray(NamedTuple{names}(map(entry -> _conform_column(rows, entry), tree)))
end

_conform_column(rows, name::Symbol) = getproperty(rows, name)
_conform_column(rows, entry::Pair) = _conform(getproperty(rows, first(entry)), last(entry))

_read_rows(tbl::LH5ChainedTable, k::Integer, idxs) =
    _conform(_part_rows(_chain_parts(tbl)[k], idxs), _row_names(tbl))

_concat_rows(blocks::AbstractVector) = length(blocks) == 1 ? only(blocks) :
    _chain_like(first(blocks), blocks)

_getindex_rows(tbl::LH5ChainedTable, ::Colon) = _concat_rows(map(eachindex(_chain_parts(tbl))) do k
    rows = _read_rows(tbl, k, :)
    length(rows) == _part_length(tbl, k) || throw(DimensionMismatch(
        "The columns of part $k differ in length"))
    rows
end)

function _getindex_rows(tbl::LH5ChainedTable, mask::AbstractVector{Bool})
    m = _materialize(mask)
    checkbounds(tbl, m)
    _getindex_rows(tbl, findall(m))
end

# The indices are grouped by part, so that each part reads its rows at once,
# and the rows are put back into the order of `idxs` afterwards:
function _getindex_rows(tbl::LH5ChainedTable, idxs::AbstractVector{<:Integer})
    isempty(idxs) && return _read_rows(tbl, 1, Int[])
    lo, hi = extrema(idxs)
    ends = _part_ends(tbl, hi)
    lo >= 1 && hi <= last(ends) || throw(BoundsError(tbl, idxs))
    ks = [searchsortedfirst(ends, i) for i in idxs]
    order = issorted(ks) ? nothing : sortperm(ks; alg = MergeSort)
    ks_sorted, idxs_sorted = isnothing(order) ? (ks, idxs) : (ks[order], idxs[order])
    blocks = Any[]
    start = firstindex(ks_sorted)
    while start <= lastindex(ks_sorted)
        k = ks_sorted[start]
        stop = something(findnext(!=(k), ks_sorted, start), lastindex(ks_sorted) + 1) - 1
        push!(blocks, _read_rows(tbl, k, idxs_sorted[start:stop] .- (k == 1 ? 0 : ends[k - 1])))
        start = stop + 1
    end
    rows = _concat_rows(blocks)
    isnothing(order) ? rows : rows[invperm(order)]
end

# Blocks are read part by part, so iterating needs no total length. The
# state holds the part and the rows of it read so far:
_first_block_state(::LH5ChainedTable) = (1, 0)

function _next_block(tbl::LH5ChainedTable, state::Tuple{Int, Int})
    k, read = state
    k > length(_chain_parts(tbl)) && return nothing
    n = _part_length(tbl, k)
    read == n && return _next_block(tbl, (k + 1, 0))
    stop = min(read + _rows_per_block, n)
    (_read_rows(tbl, k, read + 1:stop), (k, stop))
end

function _column(tbl::LH5ChainedTable, name::Symbol)
    name in _table_colnames(tbl) || _no_column(name)
    all(_fixed_part, _chain_parts(tbl)) || return _chain_column(tbl, name)
    lock(() -> get!(() -> _chain_column(tbl, name), _table_columns(tbl), name), _table_lock(tbl))
end

# A chain of the parts with some of their columns, sharing what is known of
# their lengths:
_select(tbl::LH5ChainedTable, tree) = _selects_all(tree, _table_colnames(tbl)) ? tbl :
    LH5ChainedTable(Any[_select(part, tree) for part in _chain_parts(tbl)], collect(Symbol, _tree_names(tree)),
        getfield(tbl, :lengths), Dict{Symbol, Any}(), Ref{Any}(nothing), _table_lock(tbl))

# The column of every part must have the rows of the part, or the
# concatenated columns would pair up rows of different events:
function _chain_column(tbl::LH5ChainedTable, name::Symbol)
    parts = _chain_parts(tbl)
    if _is_subtable(first(parts), name)
        subparts = Any[_subpart(part, name) for part in parts]
        foreach(k -> _check_part_rows(tbl, k, name, _part_length(subparts[k])), eachindex(parts))
        _chain_tables(subparts)
    else
        cols = Any[_part_column(part, name) for part in parts]
        foreach(k -> _check_part_rows(tbl, k, name, last(size(cols[k]))), eachindex(parts))
        _chain_columns(cols)
    end
end

_check_part_rows(tbl::LH5ChainedTable, k::Integer, name::Symbol, n::Integer) =
    n == _part_length(tbl, k) || throw(DimensionMismatch(
        "Column $name of part $k has $n rows, the part has $(_part_length(tbl, k))"))

"""
    _chain_columns(cols::AbstractVector)

Concatenate the columns of the parts of a chained table, lazily and
read-only for columns that read from disk.
"""
function _chain_columns(cols::AbstractVector)
    kind = _column_kind(first(cols))
    all(col -> _column_kind(col) == kind, cols) || throw(ArgumentError(
        "Columns to concatenate differ in kind: " * join(unique(map(col -> nameof(typeof(col)), cols)), ", ")))
    _chain_like(first(cols), cols)
end

_column_kind(::AbstractArray) = :array
_column_kind(::VectorOfVectors) = :ragged
_column_kind(::ArrayOfSimilarArrays{<:Any, M}) where {M} = (:similar, M)
_column_kind(::ArrayOfRDWaveforms) = :waveforms
_column_kind(::StructArray{<:NamedTuple}) = :table
# Encoded data concatenates only for the same codec and decoded element type:
_column_kind(A::VectorOfEncodedArrays{T}) where {T} = (:encoded, A.codec, T)
_column_kind(A::VectorOfEncodedSimilarArrays{T}) where {T} = (:encoded_similar, A.codec, T)

# Plain arrays concatenate along their last dimension:
function _chain_like(::AbstractArray, cols)
    N = ndims(first(cols))
    all(col -> ndims(col) == N, cols) || throw(DimensionMismatch(
        "Columns to concatenate differ in their number of dimensions"))
    # Element types promote, to a union like Union{Missing, T} at most:
    T = mapreduce(eltype, promote_type, cols)
    isconcretetype(T) || T isa Union && all(isconcretetype, Base.uniontypes(T)) || throw(ArgumentError(
        "Columns to concatenate have incompatible element types: " * join(unique(map(eltype, cols)), ", ")))
    # Parts without rows have no say in the inner size, and DiskArrays cannot
    # concatenate them:
    arrays = [col for col in cols if size(col, N) > 0]
    isempty(arrays) && return _readonly(first(cols))
    inner = size(first(arrays))[1:N-1]
    all(col -> size(col)[1:N-1] == inner, arrays) || throw(DimensionMismatch(
        "Columns to concatenate differ in their inner size"))
    # Arrays of empty arrays hold no data, only their number:
    any(iszero, inner) && return Array{T}(undef, inner..., sum(col -> size(col, N), arrays))
    length(arrays) == 1 && return _readonly(only(arrays))
    if any(col -> col isa DiskArrays.AbstractDiskArray, arrays)
        LH5ChainedArray(DiskArrays.ConcatDiskArray(reshape(arrays, ntuple(i -> i == N ? length(arrays) : 1, N))))
    else
        N == 1 ? reduce(vcat, arrays) : cat(arrays...; dims = N)
    end
end

# The element pointers of each part shift by the data of the parts before it:
function _chain_like(::VectorOfVectors, cols)
    datas, ptrs = Any[], [1]
    for col in cols
        ptr = ArraysOfArrays.internal_element_ptr(col)
        push!(datas, flatview(col))
        append!(ptrs, view(ptr, firstindex(ptr) + 1:lastindex(ptr)) .+ (last(ptrs) - first(ptr)))
    end
    VectorOfVectors(_chain_columns(datas), ptrs)
end

function _chain_like(::ArrayOfSimilarArrays{<:Any, M}, cols) where {M}
    data = _chain_columns(Any[flatview(col) for col in cols])
    ArrayOfSimilarArrays{eltype(data), M}(data)
end

_chain_like(::ArrayOfRDWaveforms, cols) = ArrayOfRDWaveforms((
    _chain_columns(Any[col.time for col in cols]), _chain_columns(Any[col.signal for col in cols])))

function _chain_like(::StructArray{<:NamedTuple}, cols)
    names = propertynames(first(cols))
    all(col -> issetequal(propertynames(col), names), cols) || throw(ArgumentError(
        "Nested tables to concatenate have different columns"))
    StructArray(NamedTuple{names}(map(name -> _chain_columns(Any[getproperty(col, name) for col in cols]), names)))
end

_chain_like(A::VectorOfEncodedArrays{T}, cols) where {T} = VectorOfEncodedArrays{T}(A.codec,
    reduce(vcat, [col.innersizes for col in cols]), _chain_columns(Any[col.encoded for col in cols]))

function _chain_like(::VectorOfEncodedSimilarArrays{T}, cols) where {T}
    # The inner size of a part without rows says nothing:
    nonempty = filter(!isempty, cols)
    ref = isempty(nonempty) ? first(cols) : first(nonempty)
    all(col -> col.innersize == ref.innersize, nonempty) || throw(DimensionMismatch(
        "Columns to concatenate differ in their inner size"))
    VectorOfEncodedSimilarArrays{T}(ref.codec, ref.innersize, _chain_columns(Any[col.encoded for col in cols]))
end

"""
    LH5ChainedArray{T, N} <: DiskArrays.AbstractDiskArray{T, N}

A read-only column of an [`LH5ChainedTable`](@ref) that reads through the
disk arrays of its parts.
"""
struct LH5ChainedArray{T, N, A<:DiskArrays.AbstractDiskArray{T, N}} <: DiskArrays.AbstractDiskArray{T, N}
    parts::A
end

Base.size(A::LH5ChainedArray) = size(A.parts)
DiskArrays.haschunks(A::LH5ChainedArray) = DiskArrays.haschunks(A.parts)
DiskArrays.eachchunk(A::LH5ChainedArray) = DiskArrays.eachchunk(A.parts)
DiskArrays.readblock!(A::LH5ChainedArray{T, N}, aout, r::Vararg{AbstractUnitRange, N}) where {T, N} =
    DiskArrays.readblock!(A.parts, aout, r...)
# Writes would bypass the checks of the arrays of the parts:
DiskArrays.writeblock!(::LH5ChainedArray, ::Any, ::Vararg{AbstractUnitRange}) =
    throw(ArgumentError("The columns of chained tables are read-only"))

_readonly(A) = A
_readonly(A::DiskArrays.AbstractDiskArray) = LH5ChainedArray(A)
_readonly(A::LH5ChainedArray) = A

_show_details(io::IO, tbl::LH5ChainedTable) = print(io, ", ", length(_chain_parts(tbl)), " parts")

Base.append!(tbl::LH5ChainedTable, ::Any) = _read_only(tbl)
Base.append!(tbl::LH5ChainedTable, ::EncodedArray) = _read_only(tbl)
