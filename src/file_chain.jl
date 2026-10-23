# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

"""
    LH5FileChain

LH5 files of the same structure, read as one. [`lh5open`](@ref) of a vector
of file names returns one.

`chain[path]` concatenates the table or array at `path` over all files, as
an [`LH5ChainedTable`](@ref) for a table. Files are opened when they are read
from, and at most `max_open` of them stay open. Objects from the chain stop
working when the chain is closed.

The first file defines the structure: the column names of tables, and the
datatypes, units, codecs, element types and inner sizes of what they hold.
Other files are checked against it when a column is first read from them, so
they need to agree only on what is read. Chains are read-only.
"""
mutable struct LH5FileChain
    filenames::Vector{String}
    max_open::Int
    files::Dict{Int, Any}
    recent::Vector{Int}
    structures::Dict{String, Any}
    trees::Dict{Tuple{String, Tuple}, Any}
    kept::Dict{Tuple{Int, String}, Any}
    checked::Set{Tuple{Int, String}}
    lock::ReentrantLock
    isopen::Bool
end

mutable struct _OpenFile
    lhd::LHDataStore
    leases::Int
    tables::Dict{Tuple{String, Vector{Symbol}}, LH5LazyTable}
    datasets::Dict{String, HDF5.Dataset}
end

"""
    lh5open(filenames::AbstractVector{<:AbstractString}; max_open::Integer = 64)

Open the LEGEND HDF5 files `filenames`, which must have the same structure,
as one [`LH5FileChain`](@ref), for reading.
"""
function lh5open(filenames::AbstractVector{<:AbstractString}; max_open::Integer = 64)
    isempty(filenames) && throw(ArgumentError("No files to open"))
    max_open >= 1 || throw(ArgumentError("max_open must be at least 1"))
    # Files are reopened later, maybe from another working directory:
    LH5FileChain(abspath.(filenames), max_open, Dict{Int, Any}(), Int[],
        Dict{String, Any}(), Dict{Tuple{String, Tuple}, Any}(), Dict{Tuple{Int, String}, Any}(),
        Set{Tuple{Int, String}}(),
        ReentrantLock(), true)
end

"""
    lh5open(f, filenames::AbstractVector{<:AbstractString}; kwargs...)

Open the files `filenames` as one [`LH5FileChain`](@ref), apply `f` to it
and close it afterwards, returning the result of `f`.
"""
function lh5open(f::Function, filenames::AbstractVector{<:AbstractString}; kwargs...)
    chain = lh5open(filenames; kwargs...)
    try
        f(chain)
    finally
        close(chain)
    end
end

Base.isopen(chain::LH5FileChain) = chain.isopen

# Files in use are closed when they are released:
function Base.close(chain::LH5FileChain)
    lock(chain.lock) do
        chain.isopen = false
        _close_unused!(chain, 0)
    end
    nothing
end

# A file stays open while it is leased. Of the other files, the least
# recently used are closed to keep at most `max_open` open:
function _with_file(f, chain::LH5FileChain, k::Integer)
    file = lock(chain.lock) do
        chain.isopen || throw(ArgumentError("The file chain is closed"))
        file = get(chain.files, k, nothing)
        if isnothing(file)
            _close_unused!(chain, chain.max_open - 1)
            file = _OpenFile(lh5open(chain.filenames[k]), 0,
                Dict{Tuple{String, Vector{Symbol}}, LH5LazyTable}(), Dict{String, HDF5.Dataset}())
            chain.files[k] = file
        end
        filter!(!=(k), chain.recent)
        push!(chain.recent, k)
        file.leases += 1
        file
    end
    try
        f(file)
    finally
        lock(chain.lock) do
            file.leases -= 1
            _close_unused!(chain, chain.isopen ? chain.max_open : 0)
        end
    end
end

function _close_unused!(chain::LH5FileChain, n_max::Integer)
    i = firstindex(chain.recent)
    while length(chain.files) > n_max && i <= lastindex(chain.recent)
        k = chain.recent[i]
        file = chain.files[k]
        if file.leases == 0
            close(file.lhd)
            delete!(chain.files, k)
            deleteat!(chain.recent, i)
        else
            i += 1
        end
    end
end

_file_dataset(chain::LH5FileChain, file, path::AbstractString) =
    lock(() -> get!(() -> file.lhd.data_store[path], file.datasets, path), chain.lock)

_child_path(path::AbstractString, name::Symbol) = string(rstrip(path, '/'), "/", name)

_missing_object(chain::LH5FileChain, k::Integer, path::AbstractString) =
    throw(ArgumentError("\"$path\" is missing in $(chain.filenames[k])"))

# What must match between files: attributes, element types and inner sizes
# of datasets, and the same for the members of groups. Writers differ in the
# integer types of element counts, which every reader converts:
const _count_datasets = ("cumulative_length", "decoded_size")

# A scalar count is the inner size of all elements, as for encoded arrays of
# equal-size arrays, so its value must match as well:
function _structure(obj::HDF5.Dataset, name::AbstractString = "")
    counts = name in _count_datasets
    (datatype = _datatype_key(obj), units = getattribute(obj, :units, ""), codec = _codec(obj),
        eltype = counts ? Integer : _julia_eltype(obj), inner = size(obj)[1:end-1],
        value = counts && ndims(obj) == 0 ? read(obj) : nothing)
end

function _structure(obj::HDF5.Group, ::AbstractString = "")
    members = sort!(collect(keys(obj)))
    (datatype = _datatype_key(obj), units = getattribute(obj, :units, ""), codec = _codec(obj),
        members = [name => _with_object(member -> _structure(member, name), obj, name) for name in members])
end

# Encoded data decodes only with the codec it was encoded with, parameters included:
function _codec(obj)
    name = getattribute(obj, :codec, "")
    isempty(name) && return nothing
    read_from_properties(getattribute, obj, LegendDataTypes.array_codecs.by_name[Symbol(name)])
end

# Writers differ in the order of the fields of tables and structs:
function _datatype_key(obj)
    dt = getattribute(obj, :datatype, "")
    isnothing(match(datatype_regexp, dt)) ? dt : _sort_datatype_fields(dt)
end

function _structure(chain::LH5FileChain, file, k::Integer, path::AbstractString)
    store = file.lhd.data_store
    haskey(store, path) || _missing_object(chain, k, path)
    _with_object(_structure, store, path)
end

function _first_structure(chain::LH5FileChain, path::AbstractString)
    cached = lock(() -> get(chain.structures, path, nothing), chain.lock)
    isnothing(cached) || return cached
    structure = _with_file(file -> _structure(chain, file, 1, path), chain, 1)
    lock(() -> chain.structures[path] = structure, chain.lock)
end

function _check_structure(chain::LH5FileChain, file, k::Integer, path::AbstractString, expected)
    lock(() -> (k, path) in chain.checked, chain.lock) && return nothing
    _structure(chain, file, k, path) == expected || throw(ArgumentError(
        "\"$path\" in $(chain.filenames[k]) differs from $(chain.filenames[1]) in its datatype, units, codec, element type or size"))
    lock(() -> push!(chain.checked, (k, path)), chain.lock)
    nothing
end

# Lease file `k` with the objects at `paths` checked against the first file.
# Their structures in the first file are found beforehand, so that no lease
# of it is nested in another:
function _with_checked_file(f, chain::LH5FileChain, k::Integer, paths)
    expected = k == 1 ? () : [_first_structure(chain, path) for path in paths]
    _with_file(chain, k) do file
        foreach((path, structure) -> _check_structure(chain, file, k, path, structure), paths, expected)
        f(file)
    end
end

# Columns whose elements have identities are kept while the chain is open,
# since reading them again after their file was closed would make new ones:
_kept(chain::LH5FileChain, k::Integer, path::AbstractString, col) =
    _keeps_identity(col) ? lock(() -> get!(chain.kept, (k, path), col), chain.lock) : col

# Columns that outlive a lease reopen their file when they are read:
_detach(col, ::LH5FileChain, ::Integer) = col
_detach(col::LH5Array{T, N}, chain::LH5FileChain, k::Integer) where {T, N} =
    LH5PartArray{T, N}(chain, k, HDF5.name(col.file), size(col), DiskArrays.eachchunk(col))
_detach(col::DiskArrays.SubDiskArray{<:Any, <:Any, <:LH5Array}, chain::LH5FileChain, k::Integer) =
    view(_detach(parent(col), chain, k), parentindices(col)...)
# The flat view covers the data of the elements only:
function _detach(col::VectorOfVectors, chain::LH5FileChain, k::Integer)
    ptr = ArraysOfArrays.internal_element_ptr(col)
    VectorOfVectors(_detach(flatview(col), chain, k), ptr .- (first(ptr) - 1))
end
_detach(col::ArrayOfSimilarArrays{T, M}, chain::LH5FileChain, k::Integer) where {T, M} =
    ArrayOfSimilarArrays{T, M}(_detach(flatview(col), chain, k))
_detach(col::ArrayOfRDWaveforms, chain::LH5FileChain, k::Integer) =
    ArrayOfRDWaveforms((_detach(col.time, chain, k), _detach(col.signal, chain, k)))
_detach(col::VectorOfEncodedArrays{T}, chain::LH5FileChain, k::Integer) where {T} =
    VectorOfEncodedArrays{T}(col.codec, col.innersizes, _detach(col.encoded, chain, k))
_detach(col::VectorOfEncodedSimilarArrays{T}, chain::LH5FileChain, k::Integer) where {T} =
    VectorOfEncodedSimilarArrays{T}(col.codec, col.innersize, _detach(col.encoded, chain, k))

"""
    LH5PartArray{T, N} <: DiskArrays.AbstractDiskArray{T, N}

A dataset in one file of an [`LH5FileChain`](@ref), which reopens the file
when it is read.
"""
struct LH5PartArray{T, N, C} <: DiskArrays.AbstractDiskArray{T, N}
    chain::LH5FileChain
    k::Int
    path::String
    size::NTuple{N, Int}
    chunks::C
end

LH5PartArray{T, N}(chain::LH5FileChain, k::Integer, path::AbstractString, sz::NTuple{N, Int}, chunks) where {T, N} =
    LH5PartArray{T, N, typeof(chunks)}(chain, k, path, sz, chunks)

Base.size(A::LH5PartArray) = A.size
DiskArrays.haschunks(::LH5PartArray) = DiskArrays.Chunked()
DiskArrays.eachchunk(A::LH5PartArray) = A.chunks

function DiskArrays.readblock!(A::LH5PartArray{T, N}, aout, r::Vararg{AbstractUnitRange, N}) where {T, N}
    _with_file(A.chain, A.k) do file
        DiskArrays.readblock!(LH5Array{T, N}(_file_dataset(A.chain, file, A.path)), aout, r...)
    end
end

DiskArrays.writeblock!(::LH5PartArray, ::Any, ::Vararg{AbstractUnitRange}) =
    throw(ArgumentError("The arrays of a file chain are read-only"))

# The table at `path` in file `k`, with the columns of the first file, or a
# selection of them as a name tree (see `_select`). Rows are counted by all
# of them:
struct _FileTablePart
    chain::LH5FileChain
    k::Int
    path::String
    tree::Tuple
    allnames::Vector{Symbol}
end

_FileTablePart(chain::LH5FileChain, k::Integer, path::AbstractString, names::Vector{Symbol}) =
    _FileTablePart(chain, k, path, Tuple(names), names)

function _file_table(part::_FileTablePart, file)
    names = _part_colnames(part)
    lock(part.chain.lock) do
        get!(file.tables, (part.path, names)) do
            store = file.lhd.data_store
            haskey(store, part.path) || _missing_object(part.chain, part.k, part.path)
            _lazy_table(store[part.path], names)
        end
    end
end

_with_part_table(f, part::_FileTablePart, paths) =
    _with_checked_file(file -> f(_file_table(part, file)), part.chain, part.k, paths)

_part_colnames(part::_FileTablePart) = collect(Symbol, _tree_names(part.tree))
_fixed_part(::_FileTablePart) = true

# The selected columns of the first file, with those of its nested tables:
function _part_tree(part::_FileTablePart)
    chain, key = part.chain, (part.path, part.tree)
    cached = lock(() -> get(chain.trees, key, nothing), chain.lock)
    isnothing(cached) || return cached
    tree = _build_name_tree(part, part.tree)
    lock(() -> chain.trees[key] = tree, chain.lock)
end

function _name_tree(part::_FileTablePart, tree)
    t = (tree...,)
    t == part.tree || t == _tree_names(part.tree) ? _part_tree(part) : _build_name_tree(part, t)
end

_select(part::_FileTablePart, tree) = _FileTablePart(part.chain, part.k, part.path, (tree...,), part.allnames)

# The columns of a tree as the names that lead to them, and their paths:
_tree_leaves(tree) = mapreduce(vcat, tree; init = Tuple{Vararg{Symbol}}[]) do entry
    entry isa Pair ? [(first(entry), names...) for names in _tree_leaves(last(entry))] : [(entry,)]
end

_names_path(path::AbstractString, names) = foldl(_child_path, names; init = path)
_tree_paths(path::AbstractString, tree) = [_names_path(path, names) for names in _tree_leaves(tree)]

# Rows hold the columns of the first file, other columns are not even opened:
_tree_columns(part::_FileTablePart, tbl, tree, path::AbstractString = part.path) =
    StructArray(NamedTuple{map(_tree_name, tree)}(map(tree) do entry
        name = _tree_name(entry)
        entry isa Pair ? _tree_columns(part, getproperty(tbl, name), last(entry), _child_path(path, name)) :
            _kept(part.chain, part.k, _child_path(path, name), getproperty(tbl, name))
    end))

# Columns read together check each other's lengths. A selection of some
# columns must still have the rows of the whole table:
function _part_rows(part::_FileTablePart, idxs)
    tree = _part_tree(part)
    n = _selects_all(part.tree, part.allnames) ? nothing : _part_length(part)
    _with_part_table(part, _tree_paths(part.path, tree)) do tbl
        cols = _tree_columns(part, tbl, tree)
        isnothing(n) || length(cols) == n || throw(DimensionMismatch(
            "The columns of \"$(part.path)\" in $(part.chain.filenames[part.k]) differ in length"))
        _materialize(_getindex_columns(cols, idxs))
    end
end

# The row count comes from the shape of the first column that the file has
# and that the first file stores as a dataset, nested tables searched as well,
# so columns that are not read need not be present. The columns are looked at
# in table order until one is found, so counting rows reads neither the column
# nor the structures of the columns behind it:
function _part_length(part::_FileTablePart)
    chain, k, path, names = part.chain, part.k, part.path, part.allnames
    isempty(names) && return 0
    _with_file(chain, k) do file
        store = file.lhd.data_store
        _object_kind(store, path) == :group || _missing_object(chain, k, path)
        _with_object(store, path) do table
            witness = _row_witness(chain, path, table, names)
            isnothing(witness) && _missing_object(chain, k, _child_path(path, first(names)))
            leaf, kind = witness
            kind == :dataset ? _with_object(_nrows, table, join(map(String, leaf), '/')) :
                length(foldl(getproperty, leaf; init = _file_table(_FileTablePart(chain, k, path, names), file)))
        end
    end
end

# The first column, nested tables included, that the first file stores as a
# dataset and `table` has as one, with its kind. Failing that, the first
# column that `table` stores like the first file. Nothing if `table` has none
# of them:
function _row_witness(chain::LH5FileChain, path::AbstractString, table, names, prefix::Tuple = ())
    fallback = nothing
    for name in names
        leaf = (prefix..., Symbol(name))
        structure = _first_structure(chain, _names_path(path, leaf))
        if !hasproperty(structure, :members)
            _column_kind(table, leaf) == :dataset && return (leaf, :dataset)
        elseif _is_table_datatype(structure.datatype)
            found = _row_witness(chain, path, table, map(first, structure.members), leaf)
            isnothing(found) || last(found) == :group || return found
            isnothing(fallback) && (fallback = found)
        elseif isnothing(fallback) && _column_kind(table, leaf) == :group
            fallback = (leaf, :group)
        end
    end
    fallback
end

# :dataset, :group, or nothing if there is no object at the path, without
# failing on a path that leads through a dataset:
_object_kind(store::HDF5.H5DataStore, path::AbstractString) =
    _object_kind(store, split(path, '/'; keepempty = false))

function _object_kind(store::HDF5.H5DataStore, names::AbstractVector{<:AbstractString})
    isempty(names) && return :group
    name = String(first(names))
    haskey(store, name) || return nothing
    _with_object(store, name) do obj
        length(names) == 1 ? (obj isa HDF5.Dataset ? :dataset : :group) :
            obj isa HDF5.H5DataStore ? _object_kind(obj, names[2:end]) : nothing
    end
end

# The kind of a column of a table, or nothing if the table has no such column.
# Nested columns are looked up only inside nested tables, so a column that
# this file stores differently is not looked into:
function _column_kind(table::HDF5.H5DataStore, names::Tuple)
    name = String(first(names))
    haskey(table, name) || return nothing
    _with_object(table, name) do obj
        length(names) == 1 ? (obj isa HDF5.Dataset ? :dataset : :group) :
            obj isa HDF5.Group && _is_table_datatype(getattribute(obj, :datatype, "")) ?
                _column_kind(obj, names[2:end]) : nothing
    end
end

function _part_column(part::_FileTablePart, name::Symbol)
    path = _child_path(part.path, name)
    _with_part_table(tbl -> _detach(_kept(part.chain, part.k, path, getproperty(tbl, name)), part.chain, part.k),
        part, (path,))
end

# The first file decides whether a column is a nested table:
_first_column(f, part::_FileTablePart, name::Symbol) = _with_file(part.chain, 1) do file
    f(getproperty(_file_table(_FileTablePart(part.chain, 1, part.path, part.tree, part.allnames), file), name))
end

# Decided by the datatype, so that finding the structure opens no column:
_is_subtable(part::_FileTablePart, name::Symbol) =
    _is_table_datatype(_first_structure(part.chain, _child_path(part.path, name)).datatype)

_subpart(part::_FileTablePart, name::Symbol) = _FileTablePart(part.chain, part.k,
    _child_path(part.path, name), _first_column(col -> collect(Symbol, propertynames(col)), part, name))

function Base.getindex(chain::LH5FileChain, path::AbstractString)
    obj = _with_file(chain, 1) do file
        store = file.lhd.data_store
        haskey(store, path) || _missing_object(chain, 1, path)
        LH5Array(store[path])
    end
    if obj isa LH5LazyTable
        names = collect(Symbol, propertynames(obj))
        _chained(Any[_FileTablePart(chain, k, path, names) for k in eachindex(chain.filenames)], names)
    elseif obj isa AbstractArray
        _chain_columns(Any[_file_column(chain, k, path) for k in eachindex(chain.filenames)])
    else
        throw(ArgumentError("\"$path\" is neither a table nor an array, read it from a single file"))
    end
end

Base.getindex(chain::LH5FileChain, i::Any, j::Any...) = getindex(chain, join(string.((i, j...)), "/"))

_file_column(chain::LH5FileChain, k::Integer, path::AbstractString) = _with_checked_file(chain, k, (path,)) do file
    _detach(_kept(chain, k, path, LH5Array(file.lhd.data_store[path])), chain, k)
end

Base.keys(chain::LH5FileChain) = _with_file(file -> keys(file.lhd), chain, 1)
Base.haskey(chain::LH5FileChain, path::AbstractString) = _with_file(file -> haskey(file.lhd, path), chain, 1)

Base.show(io::IO, chain::LH5FileChain) =
    print(io, "LH5FileChain(", length(chain.filenames), " files, ", length(chain.files), " open)")
