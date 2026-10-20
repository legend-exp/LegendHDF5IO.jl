# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

export LHDataStore, LH5FileChain, LH5Array, AbstractLH5Table, LH5LazyTable, LH5JoinedTable, LH5ChainedTable, LH5LazyColumns

"""
    LH5Array{T, N} <: AbstractArray{T, N}

Array wrapper for an `HDF5.Dataset` following the LEGEND data format, as in
".lh5" files.

`getindex` reads only the requested part of the on-disk array. `append!`
extends the on-disk array along its last dimension, which requires a chunked
dataset.

`LH5Array` implements the [DiskArrays.jl](https://github.com/JuliaIO/DiskArrays.jl)
interface: views, iteration and reductions read the data chunk by chunk,
broadcasts are lazy, and `setindex!` writes back into the dataset within
its current size.

# Default constructors

```julia
LH5Array{T, N}(ds::HDF5.Dataset)
LH5Array{T}(ds::HDF5.Dataset)
LH5Array(ds::Union{HDF5.Dataset, HDF5.H5DataStore})
```

# Example

```julia
julia> using HDF5
julia> f = h5open("path/to/lh5/file", "r")
julia> lh = LH5Array(f["path/to/HDF5/Dataset"])
[...]
julia> x = lh[1:10]     # load the first 10 elements of the on-disk array
[...]
julia> append!(lh, x)   # append those 10 elements to the on-disk array
[...]
```
"""
mutable struct LH5Array{T, N} <: DiskArrays.AbstractDiskArray{T, N}
    file::HDF5.Dataset
end

const LH5AoSA{T, M, N, L} = ArrayOfSimilarArrays{T, M, N, LH5Array{T, L}}
const VectorOfRDWaveforms{T, U, VVT, VVU} = ArrayOfRDWaveforms{T, U, 1, VVT, VVU}
const LH5VoV{T} = VectorOfVectors{T, LH5Array{T, 1}}
const LH5ArrayOfRDWaveforms{T, U, N, VVT} = 
    ArrayOfRDWaveforms{T, U, N, VVT, <:Union{LH5VoV{U}, LH5AoSA{U}}}
const LH5VectorOfRDWaveforms{T, U} = LH5ArrayOfRDWaveforms{T, U, 1}
# StructArrays of disk-backed columns, lazy tables among them, with typed
# tables nested one level deep. StructArrays would slice and append them with
# the disk-backed row type:
const LH5TableColumn = Union{LH5Array{<:Any, 1}, LH5VoV, LH5AoSA{<:Any, <:Any, 1}, LH5VectorOfRDWaveforms, LH5LazyTable}
const LH5FlatTableColumns = NamedTuple{names, <:Tuple{Vararg{LH5TableColumn}}} where names
const LH5TableColumns = NamedTuple{names, <:Tuple{Vararg{Union{LH5TableColumn,
    StructArrays.StructVector{<:NamedTuple, <:LH5FlatTableColumns}}}}} where names
const LH5Table = StructArrays.StructVector{<:NamedTuple, <:LH5TableColumns}

LH5Array{T}(f::HDF5.Dataset) where {T} = LH5Array{T, _ndims(f)}(f)
LH5Array(f::Union{HDF5.Dataset, HDF5.H5DataStore}) = LH5Array(f, getdatatype(f))
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:RealQuantity})

return a value with type `RealQuantity`
"""
LH5Array(ds::HDF5.Dataset, ::Type{<:RealQuantity}) = begin
    u = getunits(ds)
    v = read(ds)
    (u == NoUnits) ? v : v * u
end
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:AbstractArray})

return a `LH5Array` with dimensions equal to that of `ds` and element type 
equal to `eltype(ds) * u`
"""
LH5Array(ds::HDF5.Dataset, DT::Type{<:AbstractArray}) = begin
    N_expected = _fixed_ndims(DT)
    isnothing(N_expected) || N_expected == _ndims(ds) || throw(ArgumentError(
        "Dataset has $(_ndims(ds)) dimensions but expected $N_expected from datatype"))
    u = getunits(ds)
    ET = (u == NoUnits) ? _julia_eltype(ds) : typeof(eltype(ds)(0) * u)
    LH5Array{ET}(ds)
end

# HDF5 reports string datasets by their storage type:
_julia_eltype(ds::HDF5.Dataset) = _julia_eltype(eltype(ds))
_julia_eltype(::Type{T}) where {T} = T
_julia_eltype(::Type{<:Union{Cstring, HDF5.FixedString}}) = String
"""
    LH5Array(ds::HDF5.H5DataStore, ::Type{<:Bool}) = begin

return a value with type Bool
"""
LH5Array(ds::HDF5.Dataset, ::Type{<:Bool}) = begin
    units = getunits(ds)
    units == NoUnits || throw(ArgumentError("Can't interpret dataset with units as Bool values"))
    data = getcontent(ds)
    data > 0
end
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:AbstractArray{<:Bool}}) = begin

return a `LH5Array` with dimensions equal to that of `ds` and element type 
`Bool`. Applying `getindex!` on `LH5Array{Bool}` will yield a BitArray.
"""
LH5Array(ds::HDF5.Dataset, ::Type{<:AbstractArray{<:Bool}}) = begin
    units = getunits(ds)
    units == NoUnits || throw(ArgumentError("Can't interpret dataset with units as Bool values"))
    LH5Array{Bool}(ds)
end
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:Enum})

return a value of the given `Enum` type.
"""
LH5Array(ds::HDF5.Dataset, ET::Type{<:Enum}) = ET(getcontent(ds))
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:AbstractArray{<:Enum}})

return an in-memory array of `Enum` values.
"""
LH5Array(ds::HDF5.Dataset, AT::Type{<:AbstractArray{<:Enum}}) =
    _enum_eltype(AT).(getcontent(ds))
"""
    LH5Array(ds::HDF5.H5DataStore, ::Type{<:AbstractArrayOfSimilarArrays{<:RealQuantity}})

return an `ArraysOfSimilarArrays` where the field `data` is a `LH5Array` 
(see `ArraysOfSimilarArrays`)
"""
LH5Array(ds::HDF5.Dataset, 
::Type{<:AbstractArrayOfSimilarArrays{<:RealQuantity}}) = begin
    VectorOfSimilarArrays(LH5Array(ds, AbstractArray{<:RealQuantity}))
end
LH5Array(ds::HDF5.Dataset, ::Type{<:AbstractArrayOfSimilarArrays{<:Bool}}) =
    VectorOfSimilarArrays(LH5Array(ds, AbstractArray{<:Bool}))
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:NamedTuple{T}}) where T

return a `NamedTuple` where each `field` is the output of `LH5Array` applied to it.
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:NamedTuple{T}}) where {T} =
    NamedTuple{T}(map(k -> LH5Array(ds[String(k)]), T))
"""
    LH5Array(ds::HDF5.H5DataStore, ::Type{<:StructArray{<:NamedTuple{(T)}}}) where T

return an `LH5LazyTable` that opens the columns named in `T` on demand.
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:StructArray{<:NamedTuple{(T)}}}) where T =
    _lazy_table(ds, T)
"""
    LH5Array(ds::HDF5.H5DataStore, ::Type{<:AbstractVector{<:RDWaveform}})

return an `ArrayOfRDWaveforms` where the field `signal` is either a 
`VectorOfSimilarVectors` with an `LH5Array` as `data` or `VectorOfVectors` 
with an `LH5Array` as `data` (see `ArrayOfRDWaveforms` and `ArraysOfArrays`) 
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:AbstractVector{<:RDWaveform}}) = begin
    tbl = StructArray(LH5Array(ds, NamedTuple{(:t0, :dt, :values)}))
    from_table(tbl, AbstractVector{<:RDWaveform})
end
"""
    LH5Array(ds::HDF5.H5DataStore, ::Type{<:AbstractVector{<:AbstractVector{<:RealQuantity}}})

return a `VectorOfVectors` object where `data` is an `LH5Array` 
(see `VectorOfArrays`)
"""
LH5Array(ds::HDF5.H5DataStore,
::Type{<:AbstractVector{<:AbstractVector}}) = begin
    data = LH5Array(ds["flattened_data"])
    clen_ds = ds["cumulative_length"]
    cumulen = try
        LH5Array(clen_ds)[:]
    finally
        close(clen_ds)
    end
    VectorOfVectors(data, _element_ptrs(cumulen))
end
"""
    LH5Array(ds::HDF5.H5DataStore, ::Type{<:Histogram{<:RealQuantity}})

return a `Histogram`.
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:Histogram{<:RealQuantity}}) = begin
    T = (:binning, :weights, :isdensity)
    nt = _materialize(LH5Array(ds, NamedTuple{T}))
    _nt_to_histogram(nt)
end
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:String})

return a `String`.
"""
LH5Array(ds::HDF5.Dataset, ::Type{<:String}) = read(ds)
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:Symbol})

return a `Symbol`.
"""
LH5Array(ds::HDF5.Dataset, ::Type{<:Symbol}) = Symbol(read(ds))
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:AbstractArray{<:StaticVector}})

return an in-memory array of `SVector`s.
"""
LH5Array(ds::HDF5.Dataset, AT::Type{<:AbstractArray{<:StaticVector}}) = begin
    data = getcontent(ds)
    u = getunits(ds)
    L = size(data, 1)
    L_expected = _inner_staticvector_length(AT)
    isnothing(L_expected) || L_expected == L || throw(ErrorException(
        "Trying to read array of static vectors of length $L_expected, but inner dimension of data has length $L"))
    qdata = u == NoUnits ? data : data * u
    reinterpret(reshape, SVector{L, eltype(qdata)}, qdata)
end
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:Tuple})

return an `Tuple`
"""
LH5Array(ds::HDF5.Dataset, ::Type{<:Tuple}) =
    Tuple(LH5Array(ds, AbstractArray{<:RealQuantity, 1}))
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:AbstractArray{<:Tuple}})

return an Array of NTuples
"""
LH5Array(ds::HDF5.Dataset, AT::Type{<:AbstractArray{<:NTuple}}) = begin
    raw = read(ds)
    u = getunits(ds)
    data = u == NoUnits ? raw : raw * u
    L = size(data, 1)
    L_expected = _inner_ntuple_length(AT)
    isnothing(L_expected) || L_expected == L || throw(ErrorException(
        "Trying to read array of NTuples of length $L_expected, but inner dimension of data has length $L"))
    _flatview_to_array_of_ntuple(data, NTuple{L, eltype(data)})
end
"""
    LH5Array(ds::HDF5.H5DataStore, ::Type{<:AbstractEncodedArray{T, 1} where {T}})

return an EncodedArray
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:AbstractEncodedArray{T, 1} where {T}}
    ) = begin

    data::Vector{UInt8} = read(ds["encoded_data"])
    size_vec_in::NTuple{1, Int} = LH5Array(ds["size"])
    U = eltype(ds["sample_data"])
    codec_name = Symbol(getattribute(ds, :codec, String))
    C = LegendDataTypes.array_codecs.by_name[codec_name]
    EncodedArray{U}(C(), size_vec_in, data)
end
"""
    LH5Array(ds::HDF5.H5DataStore, ::Type{<:VectorOfEncodedArrays{T, 1} where {T}})

return a VectorOfEncodedArrays
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:VectorOfEncodedArrays{T, 1} where {T}}
    ) = begin
    
    data_vec = LH5Array(
        ds["encoded_data"])
    size_vec::Vector{NTuple{1, Int}} = LH5Array(ds["decoded_size"])
    U = haskey(ds, "sample_data") ? eltype(ds["sample_data"]) : Int32
    codec_name = Symbol(getattribute(ds, :codec, String))
    C = LegendDataTypes.array_codecs.by_name[codec_name]
    codec = read_from_properties(getattribute, ds, C)
    VectorOfEncodedArrays{U}(codec, size_vec, data_vec)
end
"""
    LH5Array(ds::HDF5.H5DataStore, ::Type{<:VectorOfEncodedSimilarArrays{T, 1} where {T}})

return a VectorOfEncodedSimilarArrays
"""
LH5Array(ds::HDF5.H5DataStore, 
    ::Type{<:VectorOfEncodedSimilarArrays{T, 1} where {T}}) = begin

    data = LH5Array(
        ds["encoded_data"])
    innersize::NTuple{1, Int} = (LH5Array(ds["decoded_size"]),)
    U = haskey(ds, "sample_data") ? eltype(ds["sample_data"]) : Int32
    codec_name = Symbol(getattribute(ds, :codec, String))
    C = LegendDataTypes.array_codecs.by_name[codec_name]
    codec = read_from_properties(getattribute, ds, C)
    VectorOfEncodedSimilarArrays{U}(codec, innersize, data)
end

# HDF5.generic_read is not type-stable, but for numeric LH5 datasets the
# result type follows from the index types (scalar indices drop
# dimensions). Other element types (e.g. strings) may be normalized to a
# different Julia type by HDF5 and are left unasserted:

@inline _read_ndims() = 0
@inline _read_ndims(::Integer, idxs...) = _read_ndims(idxs...)
@inline _read_ndims(::Union{Colon, AbstractRange}, idxs...) = _read_ndims(idxs...) + 1

@inline function _read_result(val, ::Type{T}, idxs...) where {T}
    if T <: RealQuantity
        K = _read_ndims(idxs...)
        K == 0 ? val::T : val::Array{T, K}
    else
        val
    end
end

# All reads go through one Base method and dispatch internally on the
# element type and the kind of index:

Base.getindex(lh::LH5Array{T, N},
    idxs::Vararg{Union{HDF5.IndexType, AbstractVector{<:Integer}}, N}
) where {T, N} = begin
    inds = to_indices(lh, idxs)
    checkbounds(lh, inds...)
    # Empty selections would reach code in DiskArrays and HDF5 that cannot
    # handle them, and HDF5 hyperslabs cannot step backwards:
    any(isempty, inds) && return similar(Array{T}, Base.index_shape(inds...))
    any(_descending, idxs) && return _getindex_reversed(lh, idxs)
    _getindex_impl(lh, idxs...)
end

# Masks over the whole array select linearly, as in Base:
Base.getindex(lh::LH5Array{T, N}, mask::AbstractArray{Bool, N}) where {T, N} =
    lh[LinearIndices(lh)[_materialize(mask)]]

_descending(::Any) = false
_descending(r::AbstractRange) = step(r) < 0

function _getindex_reversed(lh::LH5Array, idxs)
    out = lh[map(i -> _descending(i) ? reverse(i) : i, idxs)...]
    kept = filter(k -> !(idxs[k] isa Integer), eachindex(idxs))
    reverse(out; dims = Tuple(findall(k -> _descending(idxs[k]), kept)))
end

_getindex_impl(lh::LH5Array{T, N}, idxs::Vararg{HDF5.IndexType, N}
) where {T, N} = begin
    dtype = HDF5.datatype(lh.file)
    val = try
        HDF5.generic_read(lh.file, dtype, T, idxs...)
    finally
        close(dtype)
    end
    _read_result(val, T, idxs...)
end

# Bool datasets are stored as UInt8 for h5py compatibility, and other
# writers may store bytes other than 0 and 1, so the bytes are compared
# rather than reinterpreted:
_getindex_impl(lh::LH5Array{Bool, N}, idxs::Vararg{HDF5.IndexType, N}
) where {N} = begin
    dtype = HDF5.datatype(lh.file)
    val = try
        HDF5.generic_read(lh.file, dtype, UInt8, idxs...)
    finally
        close(dtype)
    end
    map(!iszero, _read_result(val, UInt8, idxs...))
end

# HDF5 converts strings itself when asked for String:
_getindex_impl(lh::LH5Array{String, N}, idxs::Vararg{HDF5.IndexType, N}
) where {N} = read(lh.file, String, idxs...)

function _append_elemptr!(dest_ptr::AbstractVector{<:Integer}, src_ptr::AbstractVector{<:Integer})
    offset = last(dest_ptr) - first(src_ptr)
    append!(dest_ptr, view(src_ptr, firstindex(src_ptr) + 1:lastindex(src_ptr)) .+ offset)
end

# Scattered reads along the last (event) dimension. Depending on index
# density, dataset rank and layout they map to one read of the whole index
# span, one read with a scattered dataspace selection, or one hyperslab read
# per contiguous index run:

function _contiguous_runs(idxs::AbstractVector{<:Integer})
    runs = Vector{UnitRange{Int}}()
    isempty(idxs) && return runs
    start = prev = Int(first(idxs))
    for v in Iterators.drop(idxs, 1)
        if v == prev + 1
            prev = Int(v)
        else
            push!(runs, start:prev)
            start = prev = Int(v)
        end
    end
    push!(runs, start:prev)
    runs
end

"""
    LegendHDF5IO.scatter_bulk_max_bytes[]

Size up to which a scattered read reads the whole index span its events
cover, instead of reading the events themselves.

Reading the span costs one read operation. Reading the events costs one
scattered selection, or one read per run of adjacent events, depending on
element type, rank and layout. With the default `nothing` the limit depends
on the file system the file lives on: 1 MiB on local storage and 8 MiB on
network file systems such as GPFS, NFS or Lustre, where fewer and larger
reads pay off. Variable-length elements like strings are always read
selectively. Can also be set per environment as the preference
"scatter_bulk_max_bytes".

See also [`LegendHDF5IO.scatter_bulk_max_waste`](@ref).
"""
const scatter_bulk_max_bytes = Ref{Union{Nothing, Int}}(nothing)

"""
    LegendHDF5IO.scatter_bulk_max_waste[]

Factor by which a scattered read may exceed the requested amount of data
when it reads the whole index span its events cover.

Densely covered spans are read as a whole however large they are, so
raising this trades reading more data for fewer read operations. With the
default `nothing` the factor is 4 on local storage and 16 on network file
systems. Spans up to [`LegendHDF5IO.scatter_bulk_max_bytes`](@ref) are read
as a whole regardless. Can also be set per environment as the preference
"scatter_bulk_max_waste".
"""
const scatter_bulk_max_waste = Ref{Union{Nothing, Float64}}(nothing)

# Network file systems pay per operation rather than per byte:
const _scatter_defaults = (
    disk = (bytes = 2^20, waste = 4.0),
    network = (bytes = 2^23, waste = 16.0),
)

function _scatter_limits(network::Bool)
    defaults = network ? _scatter_defaults.network : _scatter_defaults.disk
    (something(scatter_bulk_max_bytes[], defaults.bytes),
        something(scatter_bulk_max_waste[], defaults.waste))
end

_scatter_limits(lh::LH5Array) = _scatter_limits(_is_network_file(HDF5.filename(lh.file)))

const _network_fstypes = Set(["nfs", "nfs4", "gpfs", "lustre", "cifs", "smb3", "ceph",
    "beegfs", "glusterfs", "panfs", "afs", "9p", "fuse.sshfs"])

# Looked up once per file:
const _network_files = Dict{String, Bool}()
const _network_files_lock = ReentrantLock()

_is_network_file(path::AbstractString) = lock(_network_files_lock) do
    get!(_network_files, path) do
        _fstype(path) in _network_fstypes
    end
end

# The type of the file system a path lives on, from the mount table of the
# process on Linux. Elsewhere, and for files that are not on disk, such as
# in-memory ones, nothing is known:
function _fstype(path::AbstractString)
    Sys.islinux() && isfile("/proc/self/mounts") && ispath(path) || return nothing
    _fstype(realpath(path), readlines("/proc/self/mounts"))
end

# Mount points escape spaces and other characters as octal sequences:
_unescape_mount(s::AbstractString) =
    replace(s, r"\\([0-7]{3})" => m -> string(Char(parse(Int, m[2:end]; base = 8))))

function _fstype(path::AbstractString, mounts::AbstractVector{<:AbstractString})
    best, bestlen = nothing, -1
    for line in mounts
        fields = split(line)
        length(fields) >= 3 || continue
        point = _unescape_mount(fields[2])
        covers = point == "/" || path == point || startswith(path, point * "/")
        if covers && length(point) > bestlen
            best, bestlen = String(fields[3]), length(point)
        end
    end
    best
end

_select_lastdim(A::AbstractArray{<:Any, K}, i) where {K} =
    A[ntuple(_ -> Colon(), Val(K - 1))..., i]

function _getindex_scattered_lastdim(
    lh::LH5Array{T, N}, front::NTuple{M, Any}, ilast::AbstractVector{<:Integer}
) where {T, N, M}
    isempty(ilast) && return lh[front..., 1:0]
    lo, hi = Int.(extrema(ilast))
    # A small or densely covered index span is cheaper to read as a whole
    # than by any scattered read, and the result is gathered from it in any
    # index order. Variable-length elements like strings are always read
    # selectively, since the size of their payload is unknown up front:
    if isbitstype(T) && T <: Number
        row_bytes = sizeof(T) * prod(Base.front(size(lh)))
        span_bytes = (hi - lo + 1) * row_bytes
        max_bytes, max_waste = _scatter_limits(lh)
        if span_bytes <= max_bytes || span_bytes <= max_waste * length(ilast) * row_bytes
            return _select_lastdim(lh[front..., lo:hi], ilast .- (lo - 1))
        end
    end
    # HDF5 reads selections in index order, so the scattered paths need
    # sorted indices:
    issorted(ilast) && return _getindex_scattered_sparse(lh, front, ilast)
    p = sortperm(ilast)
    _select_lastdim(_getindex_scattered_sparse(lh, front, ilast[p]), invperm(p))
end

function _getindex_scattered_sparse(
    lh::LH5Array{T, N}, front::NTuple{M, Any}, ilast::AbstractVector{<:Integer}
) where {T, N, M}
    # Point selections beat per-run reads for vectors of any layout, hyperslab
    # unions only for contiguous-layout datasets (libhdf5 maps large irregular
    # selections onto chunks slowly, while sorted per-run reads make good use
    # of the chunk cache). Both read raw bytes, so element types that HDF5
    # has to convert, like strings, always take the per-run path:
    if isbitstype(T) && T <: Number && all(i -> i isa Colon, front) && (N == 1 || _is_contiguous(lh.file))
        return _getindex_scattered_single_read(lh, ilast)
    end
    _getindex_scattered_runs(lh, front, ilast)
end

# One HDF5 hyperslab read per contiguous index run:
function _getindex_scattered_runs(
    lh::LH5Array{T, N}, front::NTuple{M, Any}, ilast::AbstractVector{<:Integer}
) where {T, N, M}
    runs = _contiguous_runs(ilast)
    parts = [lh[front..., r] for r in runs]
    _cat_lastdim(parts, length(ilast))
end

function _cat_lastdim(parts::AbstractVector{<:AbstractArray{T, K}}, n::Int) where {T, K}
    fp = first(parts)
    out = similar(fp, ntuple(i -> size(fp, i), Val(K - 1))..., n)
    colons = ntuple(_ -> Colon(), Val(K - 1))
    offset = 0
    for part in parts
        len = size(part, K)
        out[colons..., offset .+ (1:len)] = part
        offset += len
    end
    out
end

function _with_create_properties(f, ds::HDF5.Dataset)
    dcpl = HDF5.get_create_properties(ds)
    try
        f(dcpl)
    finally
        close(dcpl)
    end
end

_is_contiguous(ds::HDF5.Dataset) = _with_create_properties(p -> p.layout == :contiguous, ds)
_chunk_dims(ds::HDF5.Dataset) = _with_create_properties(p -> p.layout == :chunked ? p.chunk : nothing, ds)

# A single H5Dread with a scattered dataspace selection, for sorted indices:
function _getindex_scattered_single_read(
    lh::LH5Array{T, N}, ilast::AbstractVector{<:Integer}
) where {T, N}
    idxs, imap = _dedup_sorted(ilast)
    dims = size(lh)
    fspace = HDF5.dataspace(lh.file)
    out = try
        _select_scattered!(fspace, dims, idxs)
        _read_selection(lh.file, T, fspace, (Base.front(dims)..., length(idxs)))
    finally
        close(fspace)
    end
    isnothing(imap) ? out : out[ntuple(_ -> :, N - 1)..., imap]
end

# Unique values of a sorted vector and, if there were duplicates, the
# position of each input among them:
function _dedup_sorted(idxs::AbstractVector{<:Integer})
    u = Int[Int(first(idxs))]
    imap = Vector{Int}(undef, length(idxs))
    for (k, v) in enumerate(idxs)
        v > last(u) && push!(u, Int(v))
        imap[k] = length(u)
    end
    length(u) == length(idxs) ? (u, nothing) : (u, imap)
end

function _select_scattered!(fspace::HDF5.Dataspace, dims::Dims{1}, idxs::Vector{Int})
    coords = HDF5.API.hsize_t[i - 1 for i in idxs]
    HDF5.API.h5s_select_elements(fspace, HDF5.API.H5S_SELECT_SET, length(coords), coords)
    nothing
end

function _select_scattered!(fspace::HDF5.Dataspace, dims::Dims{N}, idxs::Vector{Int}) where {N}
    runs = _contiguous_runs(idxs)
    # HDF5 dimension order is the reverse of Julia's, index runs select
    # along HDF5 dimension 1:
    start = zeros(HDF5.API.hsize_t, N)
    count = HDF5.API.hsize_t[i == 1 ? 0 : dims[N - i + 1] for i in 1:N]
    _select_runs!(fspace, dims, start, count, runs, 1, length(runs), true)
    nothing
end

# OR-ing hyperslabs into a selection one by one is quadratic in libhdf5, so
# build sub-selections and merge them by divide and conquer (h5py PR 2603):
function _select_runs!(fspace::HDF5.Dataspace, dims::Dims,
    start::Vector{HDF5.API.hsize_t}, count::Vector{HDF5.API.hsize_t},
    runs::Vector{UnitRange{Int}}, lo::Int, hi::Int, fresh::Bool
)
    if hi - lo < 16
        for k in lo:hi
            start[1] = first(runs[k]) - 1
            count[1] = length(runs[k])
            op = fresh && k == lo ? HDF5.API.H5S_SELECT_SET : HDF5.API.H5S_SELECT_OR
            HDF5.API.h5s_select_hyperslab(fspace, op, start, C_NULL, count, C_NULL)
        end
    else
        mid = (lo + hi) >>> 1
        _select_runs!(fspace, dims, start, count, runs, lo, mid, fresh)
        fs2 = HDF5.dataspace(dims)
        try
            _select_runs!(fs2, dims, start, count, runs, mid + 1, hi, true)
            HDF5.API.h5s_modify_select(fspace, HDF5.API.H5S_SELECT_OR, fs2)
        finally
            close(fs2)
        end
    end
    nothing
end

function _read_selection(ds::HDF5.Dataset, ::Type{T}, fspace::HDF5.Dataspace,
    dims::Dims
) where {T}
    filetype = HDF5.datatype(ds)
    memtype = HDF5.Datatype(HDF5.API.h5t_get_native_type(filetype))
    memspace = HDF5.dataspace(dims)
    try
        sizeof(T) == sizeof(memtype) || throw(ArgumentError(
            "Can't read scattered $(sizeof(memtype))-byte elements into $T"))
        buf = Array{T}(undef, dims)
        HDF5.API.h5d_read(ds, memtype, memspace, fspace, ds.xfer, buf)
        buf
    finally
        close(memspace)
        close(memtype)
        close(filetype)
    end
end

# Bool datasets are stored as UInt8:
_read_selection(ds::HDF5.Dataset, ::Type{Bool}, fspace::HDF5.Dataspace, dims::Dims) =
    map(!iszero, _read_selection(ds, UInt8, fspace, dims))

_getindex_impl(lh::LH5Array{T, N},
    idxs::Vararg{Union{HDF5.IndexType, AbstractVector{<:Integer}}, N}
) where {T, N} = begin
    front, ilast = Base.front(idxs), idxs[end]
    if ilast isa AbstractVector{Bool}
        checkbounds(lh, front..., ilast)
        lh[front..., findall(_materialize(ilast))]
    elseif ilast isa Base.LogicalIndex
        # to_indices turns logical masks into an iterate-only LogicalIndex.
        # The mask is materialized first, since iterating a lazy mask would
        # read it element by element:
        checkbounds(lh, front..., ilast.mask)
        lh[front..., findall(_materialize(ilast.mask))]
    elseif ilast isa AbstractVector{<:Integer} && !(ilast isa AbstractRange) &&
        all(i -> i isa HDF5.IndexType, front)
        checkbounds(lh, idxs...)
        _getindex_scattered_lastdim(lh, front, ilast)
    else
        invoke(getindex, Tuple{DiskArrays.AbstractDiskArray{T, N}, Vararg{Any, N}}, lh, idxs...)
    end
end

# DiskArrays interface:

function DiskArrays.readblock!(lh::LH5Array{T, N}, aout, r::Vararg{AbstractUnitRange, N}) where {T, N}
    aout .= lh[map(UnitRange{Int}, r)...]
    nothing
end

# Writing back into the dataset, which DiskArrays turns into setindex!,
# broadcast assignment and copyto!. Growing a dataset is append!'s job, so
# writes stay within the current size:
function DiskArrays.writeblock!(lh::LH5Array{T, N}, v::AbstractArray,
    r::Vararg{AbstractUnitRange, N}
) where {T, N}
    # DiskArrays bounds-checks reads but not writes:
    checkbounds(lh, r...)
    lh.file[map(UnitRange{Int}, r)...] = _ustrip(convert(Array{T, N}, v))
    nothing
end

# Every write, direct, through a view or by broadcast assignment, passes
# through setindex_disk! of the array or of the view. DiskArrays checks bounds
# and the number of values block by block, after earlier blocks were written,
# and reads a block before writing part of it only when the indices are sparse
# over their span. Repeated indices inflate that density, so the elements
# between them would be written back uninitialized:
const LH5DiskArray = Union{LH5Array, DiskArrays.SubDiskArray{<:Any, <:Any, <:LH5Array}}

function DiskArrays.setindex_disk!(a::LH5DiskArray, v::AbstractArray, idxs...)
    inds = to_indices(a, idxs)
    checkbounds(a, inds...)
    n = prod(map(length, inds))
    length(v) == n || throw(DimensionMismatch("Cannot assign $(length(v)) values to $n elements"))
    any(_has_repeated_indices, inds) && throw(ArgumentError(
        "Repeated indices are not supported in writes"))
    # DiskArrays cannot plan a write without elements:
    n == 0 && return v
    # and indexes the values like the selection, so they take its shape:
    shape = mapreduce(size, (s, t) -> (s..., t...), inds; init = ())
    invoke(DiskArrays.setindex_disk!, Tuple{AbstractArray, AbstractArray, Vararg{Any}},
        a, reshape(v, shape), idxs...)
    v
end

_has_repeated_indices(i) = false
_has_repeated_indices(i::AbstractArray{<:Union{Integer, CartesianIndex}}) = !allunique(i)
_has_repeated_indices(::Union{AbstractRange, Base.LogicalIndex}) = false

# DiskArrays batches writes into a view chunk by chunk, so repeated parent
# indices must be rejected before the first chunk is written:
const LH5SubDiskArray{T, N} = DiskArrays.SubDiskArray{T, N, <:LH5Array}

function _check_view_indices(a::LH5SubDiskArray)
    any(_has_repeated_indices, parentindices(a)) && throw(ArgumentError(
        "Repeated indices are not supported in writes"))
end

function Base.copyto!(dest::LH5SubDiskArray, bc::Base.Broadcast.Broadcasted{Nothing})
    _check_view_indices(dest)
    invoke(copyto!, Tuple{DiskArrays.AbstractDiskArray, Base.Broadcast.Broadcasted{Nothing}}, dest, bc)
end
function Base.fill!(dest::LH5SubDiskArray, v)
    _check_view_indices(dest)
    invoke(fill!, Tuple{DiskArrays.AbstractDiskArray, Any}, dest, v)
end
# copyto! into a disk array funnels through _disk_copyto!, which has methods
# with and without matching dimensions:
function DiskArrays._disk_copyto!(dest::LH5SubDiskArray{<:Any, N}, src::AbstractArray{<:Any, N}) where {N}
    _check_view_indices(dest)
    invoke(DiskArrays._disk_copyto!, Tuple{AbstractArray{<:Any, N}, AbstractArray{<:Any, N}}, dest, src)
end
function DiskArrays._disk_copyto!(dest::LH5SubDiskArray, src::AbstractArray)
    _check_view_indices(dest)
    invoke(DiskArrays._disk_copyto!, Tuple{AbstractArray, AbstractArray}, dest, src)
end

# Resolves the ambiguity with EncodedArrays' setindex! for whole-array writes:
Base.setindex!(lh::LH5Array{T, 1}, v::EncodedArray, ::Colon) where {T} =
    invoke(setindex!, Tuple{DiskArrays.AbstractDiskArray, AbstractArray, Vararg{Any}}, lh, v, Colon())

# Contiguous datasets report a chunk grid as well, estimated below, so that
# DiskArrays never estimates one itself, which fails for empty views:
DiskArrays.haschunks(::LH5Array) = DiskArrays.Chunked()

# DiskArrays maps the parent's grid onto a view only through plain indices.
# Views through masks or CartesianIndex vectors get an estimated grid:
function DiskArrays.eachchunk(a::LH5SubDiskArray)
    # DiskArrays cannot map the grid onto an empty view, of a reversed range for example:
    isempty(a) && return DiskArrays.GridChunks(a, max.(size(a), 1))
    plain(i) = i isa Union{Int, AbstractRange, Colon} ||
        (i isa AbstractVector{<:Integer} && !(i isa Base.LogicalIndex))
    all(plain, parentindices(a)) &&
        return invoke(DiskArrays.eachchunk, Tuple{DiskArrays.AbstractSubDiskArray}, a)
    DiskArrays.estimate_chunksize(a)
end

# The return type is asserted since HDF5 property lists are not inferable,
# and DiskArrays iterates through the chunk grid:
function DiskArrays.eachchunk(lh::LH5Array{T, N}
)::DiskArrays.GridChunks{N, NTuple{N, DiskArrays.RegularChunks}} where {T, N}
    chunk = _chunk_dims(lh.file)
    isnothing(chunk) || return DiskArrays.GridChunks(lh, chunk::Dims{N})
    # DiskArrays cannot estimate a chunk size for an empty array:
    isempty(lh) ? DiskArrays.GridChunks(lh, max.(size(lh), 1)) : DiskArrays.estimate_chunksize(lh)
end

Base.size(lh::LH5Array{T, N}) where {T, N} = begin
    dspace = HDF5.dataspace(lh.file)
    try
        h5_dims = HDF5.API.h5s_get_simple_extent_dims(
            HDF5.checkvalid(dspace), nothing)
        return ntuple(i -> @inbounds(Int(h5_dims[N - i + 1])), N)
    finally
        close(dspace)
    end
end

# Reading straight into dest saves a copy, but HDF5 reinterprets the bytes
# instead of converting values, so only identical native number types
# qualify. Bool datasets hold bytes to compare and strings need conversion:
const _NativeNumber = Union{AbstractFloat, Signed, Unsigned}

function Base.copyto!(dest::Array{T, N}, src::LH5Array{T, N}) where {T<:_NativeNumber, N}
    length(dest) == length(src) || return invoke(copyto!, Tuple{Array, LH5Array}, dest, src)
    copyto!(dest, src.file, ntuple(_ -> Colon(), Val(N))...)
    dest
end
Base.copyto!(dest::Array, src::LH5Array{T, N}) where {T, N} =
    copyto!(dest, src[ntuple(_ -> Colon(), Val(N))...])

# Deep conversion of lazily read data into in-memory objects. Nested arrays
# are rebuilt around their materialized data:

_materialize(x) = x
_materialize(A::DiskArrays.AbstractDiskArray) = Array(A)
_materialize(A::VectorOfVectors) =
    VectorOfVectors(_materialize(A.data), copy(ArraysOfArrays.internal_element_ptr(A)))
_materialize(A::ArrayOfSimilarArrays{T, M}) where {T, M} =
    ArrayOfSimilarArrays{T, M}(_materialize(A.data))
_materialize(x::NamedTuple) = map(_materialize, x)
_materialize(A::StructArray{<:NamedTuple}) =
    StructArray(map(_materialize, StructArrays.components(A)))
_materialize(tbl::AbstractLH5Table) = _materialize(tbl[:])
_materialize(A::ArrayOfRDWaveforms) =
    ArrayOfRDWaveforms((_materialize(A.time), _materialize(A.signal)))
_materialize(A::VectorOfEncodedArrays{T}) where {T} =
    VectorOfEncodedArrays{T}(A.codec, A.innersizes, _materialize(A.encoded))
_materialize(A::VectorOfEncodedSimilarArrays{T}) where {T} =
    VectorOfEncodedSimilarArrays{T}(A.codec, A.innersize, _materialize(A.encoded))

@inline _ustrip(x::AbstractArray) = x
@inline _ustrip(x::AbstractArray{T}) where T<:Quantity = 
    reinterpret(Unitful.numtype(T), x) 

_append_impl!(dest::LH5Array{T, 1}, src::EncodedArray) where {T} =
    _append_impl!(dest, collect(src))

Base.append!(dest::LH5Array{T, N}, src::AbstractArray) where {T, N} =
    _append_impl!(dest, src)

_append_impl!(dest::LH5Array{T, N}, src::AbstractArray) where {T, N} = begin
    x = convert(Array{T, N}, src)
    old_size = size(dest)
    Base.front(size(x)) == Base.front(old_size) || throw(DimensionMismatch(
        "Cannot append an array of size $(size(x)) to an array of size $old_size along the last dimension"))
    new_size = (Base.front(old_size)..., old_size[N] + size(x, N))
    indices = (ntuple(_ -> Colon(), Val(N - 1))..., (old_size[N] + 1):new_size[N])
    HDF5.set_extent_dims(dest.file, new_size)
    try
        dest.file[indices...] = _ustrip(x)
    catch
        # The extension would otherwise stay behind, filled with fill values:
        HDF5.set_extent_dims(dest.file, old_size)
        rethrow()
    end
    dest
end

Base.append!(dest::LH5VoV, src::VectorOfVectors) = _append_impl!(dest, src)

# Table columns holding plain vectors of vectors:
_append_impl!(dest::LH5VoV, src::AbstractVector{<:AbstractVector}) =
    _append_impl!(dest, VectorOfVectors(src))

# Whether a column can be appended to. Ragged arrays and waveforms keep their
# element pointers and time axes from when they were opened. Another handle
# may have appended since, and appending from stale ones corrupts the array:
_check_appendable(::Any) = nothing

function _check_appendable(A::LH5VoV)
    ptr = ArraysOfArrays.internal_element_ptr(A)
    ptr isa Vector || throw(ArgumentError(
        "Cannot append through a view of a ragged array, its element pointers cannot grow"))
    clen = _column_object(A)["cumulative_length"]
    try
        length(clen) == length(A) && length(fused(A)) == last(ptr) - first(ptr) || throw(ArgumentError(
            "The array changed on disk since it was opened, reopen it to append"))
    finally
        close(clen)
    end
end

function _check_appendable(wf::LH5VectorOfRDWaveforms)
    _check_appendable(wf.signal)
    t0 = _column_object(wf)["t0"]
    try
        n = length(wf.time)
        length(wf.signal) == n && length(t0) == n || throw(ArgumentError(
            "The array changed on disk since it was opened, reopen it to append"))
    finally
        close(t0)
    end
end

_append_impl!(dest::LH5VoV, src::VectorOfVectors) = begin
    isempty(src) && return dest
    _check_appendable(dest)
    dest_ptr = ArraysOfArrays.internal_element_ptr(dest)
    dset = parent(fused(dest).file)["cumulative_length"]
    try
        clen = LH5Array(dset)
        old_len = last(dest_ptr) - first(dest_ptr)
        src_ptr = ArraysOfArrays.internal_element_ptr(src)
        src_flat = fused(src)[first(src_ptr):(last(src_ptr) - 1)]
        # Derived from src before dest changes, since src may be dest itself,
        # and converted first, so that an overflow leaves the data as it was:
        new_clen = convert(Vector{eltype(clen)}, cumsum(diff(src_ptr)) .+ old_len)
        append!(fused(dest), src_flat)
        _append_elemptr!(dest_ptr, src_ptr)
        append!(dest.kernel_size, src.kernel_size)
        append!(clen, new_clen)
    finally
        close(dset)
    end
    dest
end

# Non-scalar reads of disk-backed tables must recompute the row type, since
# the element type of lazy columns changes when they are read. Nested tables
# are columns of their own. A lazy mask or index vector is read once, not
# once per column:
function _getindex_columns(A::StructArray{<:NamedTuple}, idxs)
    inds = if idxs isa AbstractVector{Bool}
        mask = _materialize(idxs)
        checkbounds(A, mask)
        findall(mask)
    else
        _materialize(idxs)
    end
    StructArray(map(col -> _getindex_columns(col, inds), StructArrays.components(A)))
end
_getindex_columns(col, idxs) = col[idxs]

Base.getindex(A::LH5Table, idxs::Union{AbstractVector, Colon}) = _getindex_columns(A, idxs)

# The HDF5 object a lazily read column lives in. A nested table is the group
# of its columns, so the table of a column is the parent of its object:
_column_object(col::LH5Array) = col.file
_column_object(col::LH5AoSA) = fused(col).file
_column_object(col::LH5VoV) = parent(fused(col).file)
_column_object(wf::LH5VectorOfRDWaveforms) = parent(_column_object(wf.signal))
_column_object(A::StructArray{<:NamedTuple}) =
    parent(_column_object(first(StructArrays.components(A))))
_column_object(tbl::LH5LazyTable) = _table_store(tbl)

_object_token(obj) = HDF5.API.h5o_get_info(obj, HDF5.API.H5O_INFO_BASIC).token
_same_object(a, b) = _object_token(a) == _object_token(b)

# Columns that can be appended to on disk, nested tables when all of theirs can:
_disk_column(::Union{LH5Array, LH5VoV, LH5AoSA, LH5VectorOfRDWaveforms, LH5LazyTable}) = true
_disk_column(A::StructArray{<:NamedTuple}) = all(_disk_column, StructArrays.components(A))
_disk_column(::Any) = false

# Columns whose elements have identities that reading them again would not
# reproduce, such as measurements, also inside other arrays:
_keeps_identity(::Any) = false
_keeps_identity(A::Union{VectorOfArrays, ArrayOfSimilarArrays}) = _keeps_identity(flatview(A))
_keeps_identity(A::StructArray) = any(_keeps_identity, StructArrays.components(A))

# Columns that read from the file, which therefore must not outlive it:
_reads_from_disk(::Any) = false
_reads_from_disk(::Union{DiskArrays.AbstractDiskArray, AbstractLH5Table}) = true
_reads_from_disk(A::Union{VectorOfArrays, ArrayOfSimilarArrays}) = _reads_from_disk(flatview(A))
_reads_from_disk(A::StructArray) = any(_reads_from_disk, StructArrays.components(A))
_reads_from_disk(A::Union{VectorOfEncodedArrays, VectorOfEncodedSimilarArrays}) = _reads_from_disk(A.encoded)

# Ragged arrays and waveforms keep their element pointers and time axes in
# memory. Bringing them up to date keeps tables built from them coherent:
_refresh!(::Union{LH5Array, LH5AoSA}) = false

function _refresh!(A::LH5VoV)
    clen_ds = _column_object(A)["cumulative_length"]
    clen = try LH5Array(clen_ds)[:] finally close(clen_ds) end
    new_ptr = _element_ptrs(clen)
    ptr = ArraysOfArrays.internal_element_ptr(A)
    resize!(ptr, length(new_ptr))
    copyto!(ptr, new_ptr)
    resize!(A.kernel_size, length(clen))
    false
end

function _refresh!(wf::LH5VectorOfRDWaveforms)
    _refresh!(wf.signal)
    fresh = LH5Array(_column_object(wf), AbstractVector{<:RDWaveform})
    resize!(wf.time, length(fresh.time))
    copyto!(wf.time, fresh.time)
    false
end

function _check_table_columns(stored, dest)
    issetequal(stored, propertynames(dest)) || throw(ArgumentError(
        "The columns of the table changed on disk since it was opened, reopen it to append"))
end

Base.append!(dest::LH5LazyTable, src) = _append_table!(dest, src)
# EncodedArrays appends its arrays to any vector:
Base.append!(dest::LH5LazyTable, src::EncodedArray) = _append_table!(dest, src)

function _append_table!(dest::LH5LazyTable, src)
    _check_append_dest(dest)
    _append_impl!(_typed_table(dest), src)
    dest
end

Base.append!(dest::LH5Table, src) = _append_impl!(dest, src)

# Columns must be disk-backed and still be the objects stored under their
# names, and nested tables, like the table itself, must still have the columns
# they were opened with:
function _check_append_dest(dest::StructArray{<:NamedTuple})
    cols = StructArrays.components(dest)
    for (name, col) in pairs(cols)
        _disk_column(col) || throw(ArgumentError(
            "Column \"$name\" was read into memory and cannot be appended to on disk"))
    end
    group = _column_object(dest)
    _check_table_columns(propertynames(LH5Array(group)), dest)
    for (name, col) in pairs(cols)
        current = group[String(name)]
        try
            _same_object(_column_object(col), current) || throw(ArgumentError(
                "Column \"$name\" was replaced on disk since it was opened, reopen the table to append"))
        finally
            close(current)
        end
        _check_appendable(col)
    end
    foreach(col -> col isa Union{StructArray{<:NamedTuple}, LH5LazyTable} && _check_append_dest(col), cols)
end

# A lazy table may have gained or lost columns since it was opened, and its
# cached columns may have been appended to or replaced, so it is checked and
# brought up to date with the file first:
function _check_append_dest(tbl::LH5LazyTable)
    _check_table_columns(propertynames(LH5Array(_table_store(tbl))), tbl)
    _refresh_cache!(tbl)
    _check_append_dest(_typed_table(tbl))
end

# Column by column. Source columns are read before anything is written, since
# they may alias destination columns, and StructArrays would append row by row
# unless both element types match exactly:
function _append_impl!(dest::StructArray{<:NamedTuple}, src)
    _check_append_dest(dest)
    dcols = StructArrays.components(dest)
    scols = map(_materialize, Tables.columntable(src))
    issetequal(keys(dcols), keys(scols)) || throw(ArgumentError(
        "Cannot append table with columns $(keys(scols)) to table with columns $(keys(dcols))"))
    allequal(map(length, Tuple(scols))) || throw(ArgumentError(
        "The columns of the table to append differ in length"))
    for k in keys(dcols)
        _append_impl!(dcols[k], scols[k])
    end
    dest
end

_append_impl!(dest::LH5LazyTable, src) = _append_table!(dest, src)
_append_impl!(dest::StructArray{<:NamedTuple}, src::EncodedArray) =
    throw(ArgumentError("Cannot append an encoded array to a table"))

_append_impl!(dest::LH5AoSA, src::AbstractVector{<:AbstractArray}) = append!(dest, src)

# These add no behavior, they only resolve ambiguities with the append!
# methods of EncodedArrays and StructArrays:
Base.append!(dest::LH5Array{T, 1}, src::EncodedArray) where {T} = _append_impl!(dest, src)
Base.append!(dest::LH5Table, src::EncodedArray) = _append_impl!(dest, src)
Base.append!(dest::StructArrays.StructVector{T, <:LH5TableColumns},
    src::StructArrays.StructVector{T}) where {T<:NamedTuple} = _append_impl!(dest, src)

Base.append!(dest::LH5VectorOfRDWaveforms, src::VectorOfRDWaveforms) =
    _append_impl!(dest, src)
# StructArrays appends arrays of one element type component by component,
# which would leave the stored time axes behind:
Base.append!(dest::W, src::W) where {W<:LH5VectorOfRDWaveforms} = _append_impl!(dest, src)

_append_impl!(dest::LH5VectorOfRDWaveforms, src::VectorOfRDWaveforms) = begin
    length(src.time) == length(src.signal) || throw(ArgumentError(
        "The waveforms to append have more time axes than signals or vice versa"))
    _check_appendable(dest)
    grp = _column_object(dest)
    dset_t0 = grp["t0"]
    dset_dt = grp["dt"]
    try
        # Taken before anything changes, since src may be dest itself:
        src_t0 = first.(src.time)
        src_dt = step.(src.time)
        StructArrays.foreachfield(append!, dest, src)
        append!(LH5Array(dset_t0), src_t0)
        append!(LH5Array(dset_dt), src_dt)
    finally
        close(dset_t0)
        close(dset_dt)
    end
    dest
end

"""
    LHDataStore <: AbstractDict{String,Any}

Dictionary wrapper for an `HDF5.H5DataStore` (typically an `HDF5.File`, or
an `HDF5.Group`) following the LEGEND data format, as in ".lh5" files.

Constructor:

```julia
LHDataStore(h5ds::HDF5.H5DataStore, usechunks::Bool, compress::Symbol = :none)
```

To open ".lh5" files directly, without `HDF5.h5open`, use [`lh5open`](@ref).

`getindex(lh::LHDataStore, s)` returns the output of [`LH5Array`](@ref)
applied to `lh.data_store[s]`, so arrays other than those of `Enum`s,
`StaticVector`s, `NTuple`s and `Measurement`s are wrapped lazily. `setindex!`
creates and writes `HDF5.Group`s and `HDF5.Dataset`s.

With `usechunks = true`, array datasets are created chunked, so `append!` can
extend them, and the first array written sets the chunk size. With
`compress` set to `:zstd` or `:deflate`, array datasets are chunked and
compressed, with chunks of at most 1 MiB.

Supported value types include `RealQuantity` and arrays thereof, `Bool`,
`String`, `Symbol`, `Enum`s, (arrays of) `NTuple`s and `StaticVector`s,
`ArrayOfSimilarArrays`, `VectorOfVectors`, encoded arrays, `NamedTuple`s,
tables, `AbstractVector{<:RDWaveform}` and `Histogram`. **For arrays, the last
axis is assumed to correspond to the event number index.**

# Example

```julia
julia> lhf = lh5open("path/to/lh5/file", "cw")
julia> lhf["raw"]
[...]
julia> using Unitful
julia> x = rand(100) * u"ns"
julia> lhf["new"] = x
[...]
```
"""
mutable struct LHDataStore <: AbstractDict{String,Any}
    data_store::HDF5.H5DataStore
    usechunks::Bool
    compress::Symbol

    LHDataStore(data_store::HDF5.H5DataStore, usechunks::Bool, compress::Symbol) =
        new(data_store, usechunks, _compress_mode(compress))
end

LHDataStore(data_store::HDF5.H5DataStore, usechunks::Bool) =
    LHDataStore(data_store, usechunks, :none)

_compress_mode(compress::Bool) = compress ? :zstd : :none
function _compress_mode(compress::Symbol)
    compress === :gzip && return :deflate
    compress in (:none, :zstd, :deflate) || throw(ArgumentError(
        "Unsupported compression mode :$compress, expected :zstd, :deflate or :gzip"))
    compress
end

_dataset_filters(compress::Symbol) =
    compress === :none ? HDF5.Filters.Filter[] :
    compress === :zstd ? HDF5.Filters.Filter[H5Zzstd.ZstdFilter(3)] :
    HDF5.Filters.Filter[HDF5.Filters.Shuffle(), HDF5.Filters.Deflate(3)]

# A compressed chunk is decompressed as a whole, so compressed chunks stay
# small enough that a partial read decompresses little more than it needs:
const _compressed_chunk_bytes = 2^20

function _chunk_length(::Type{T}, sz_inner, sz_outer::Integer, compress::Symbol) where {T}
    compress === :none && return sz_outer
    # Variable-length elements like strings are stored as references in the chunk:
    elsize = isbitstype(T) ? sizeof(T) : sizeof(Ptr{Cvoid})
    min(sz_outer, max(1, fld(_compressed_chunk_bytes, elsize * prod(sz_inner))))
end

@deprecate LHDataStore(f::AbstractString, access::AbstractString = "r") lh5open(f, access)
@deprecate LHDataStore(f::Function, s::AbstractString, access::AbstractString = "r") lh5open(f, s, access)

Base.isopen(f::LHDataStore) = isopen(f.data_store)
Base.close(f::LHDataStore) = close(f.data_store)
Base.keys(lh::LHDataStore) = keys(lh.data_store)
Base.haskey(lh::LHDataStore, i::AbstractString) = haskey(lh.data_store, i)
Base.getindex(lh::LHDataStore, i::AbstractString) = LH5Array(lh.data_store[i])
Base.getindex(lh::LHDataStore, i::Any, j::Any...) =
    getindex(lh, join(string.((i, j...)), "/"))

Base.length(lh::LHDataStore) = length(keys(lh))

function Base.iterate(lh::LHDataStore)
    ks = keys(lh)
    r = iterate(ks)
    if isnothing(r)
        return nothing
    else
        k, i = r
        return (k => lh[k], (ks, i))
    end
end

function Base.iterate(lh::LHDataStore, state)
    ks, i_last = state
    r = iterate(ks, i_last)

    if isnothing(r)
        return nothing
    else
        k, i = r
        return (k => lh[k], (ks, i))
    end
end

Base.show(io::IO, m::MIME"text/plain", lh::LHDataStore) = HDF5.show_tree(io, lh.data_store)
Base.show(io::IO, lh::LHDataStore) = show(io, MIME"text/plain"(), lh)

function Base.setindex!(lh::LHDataStore, v, i::AbstractString)
    create_entry(lh, i, v, usechunks=lh.usechunks, compress=lh.compress)
    nothing
end

Base.setindex!(lh::LHDataStore, v, i::Any, j::Any...) =
    setindex!(lh, v, join(string.((i, j...)), "/"))

LegendDataTypes.readdata(input::LHDataStore, args...; kwargs...) = readdata(input.data_store, args...; kwargs...)
LegendDataTypes.writedata(output::LHDataStore, args...; kwargs...) = writedata(output.data_store, args...; kwargs...)    


# write <:Real
function create_entry(parent::LHDataStore, name::AbstractString, data::T; 
    kwargs...) where {T<:Real}

    parent.data_store[name] = data
    setdatatype!(parent.data_store[name], T)
    nothing
end

# write <:Quantity
function create_entry(parent::LHDataStore, name::AbstractString, data::T;
    kwargs...) where {T<:Quantity}

    create_entry(parent, name, ustrip(data); kwargs...)
    setunits!(parent.data_store[name], unit(T))
    nothing
end

# Store Bool values as UInt8 for h5py compatibility
function create_entry(parent::LHDataStore, name::AbstractString, data::Bool;
    kwargs...)

    parent.data_store[name] = UInt8(data)
    setdatatype!(parent.data_store[name], Bool)
    nothing
end

function create_entry(parent::LHDataStore, name::AbstractString,
    data::AbstractArray{Bool}; kwargs...)

    create_entry(parent, name, UInt8.(data); kwargs...)
    setdatatype!(parent.data_store[name], typeof(data))
    nothing
end

# write Enum values via their integer representation
function create_entry(parent::LHDataStore, name::AbstractString,
    data::Enum{T}; kwargs...) where {T}

    parent.data_store[name] = T(data)
    setdatatype!(parent.data_store[name], typeof(data))
    nothing
end

function create_entry(parent::LHDataStore, name::AbstractString,
    data::AbstractArray{<:Enum{T}}; kwargs...) where {T}

    create_entry(parent, name, T.(data); kwargs...)
    setdatatype!(parent.data_store[name], typeof(data))
    nothing
end

# write AbstractArray{<:Real} and AbstractArray{<:AbstractString}
function create_entry(parent::LHDataStore, name::AbstractString,
    data::AbstractArray{T}; usechunks::Bool=false, compress::Symbol=:none
) where {T<:Union{Real, AbstractString}}

    T <: AbstractString && T !== String && return create_entry(parent, name,
        convert(Array{String}, data); usechunks = usechunks, compress = compress)
    dtype = HDF5.datatype(T)
    ds = if !usechunks && compress === :none
        HDF5.create_dataset(parent.data_store, name, dtype, size(data))
    else
        # Compression requires a chunked dataset. The first written array
        # sets the chunk size:
        data_size = size(data)
        sz_inner, sz_outer = data_size[begin:end-1], data_size[end]
        sz_outer > 0 || throw(ArgumentError(
            "Cannot infer a chunk size for \"$name\" from an empty array"))
        dspace = (data_size, (sz_inner..., -1))
        chunk = (sz_inner..., _chunk_length(T, sz_inner, sz_outer, compress))
        HDF5.create_dataset(parent.data_store, name, dtype, dspace;
            chunk=chunk, filters=_dataset_filters(compress))
    end
    try
        HDF5.write_dataset(ds, dtype, Array(data))
        setdatatype!(ds, typeof(data))
    catch exc
        HDF5.delete_object(ds)
        rethrow(exc)
    finally
        close(ds)
        close(dtype)
    end
    nothing
end

# write AbstractArray{<:Quantity}
function create_entry(parent::LHDataStore, name::AbstractString,
    data::AbstractArray{T}; kwargs...) where {T<:Quantity}

    create_entry(parent, name, _ustrip(data); kwargs...)
    setdatatype!(parent.data_store[name], typeof(data))
    setunits!(parent.data_store[name], unit(T))
    nothing
end

# write AbstractArrayOfSimilarArrays{<:RealQuantity}
function create_entry(parent::LHDataStore, name::AbstractString,
    data::AbstractArrayOfSimilarArrays{T}; kwargs...) where {T<:RealQuantity}

    create_entry(parent, name, flatview(data); kwargs...)
    setdatatype!(parent.data_store[name], typeof(data))
    nothing
end

# write AbstractArray{<:StaticVector{L, <:RealQuantity}}
function create_entry(parent::LHDataStore, name::AbstractString,
    data::AbstractArray{<:StaticVector{L, T}}; kwargs...
    ) where {L, T<:RealQuantity}

    create_entry(parent, name, flatview(data); kwargs...)
    setdatatype!(parent.data_store[name], typeof(data))
    nothing
end

# write VectorOfVectors{<:RealQuantity}
function create_entry(parent::LHDataStore, name::AbstractString, 
    data::AbstractArray{<:AbstractArray{T, M}, N}; kwargs...) where {T, M, N}

    N == 1 || throw(ArgumentError("Output of multi-dimensional arrays of" 
    *" arrays to HDF5 is not supported"))
    # TODO: Support vectors of multi-dimensional arrays
    M == 1 || throw(ArgumentError("Output of vectors of multi-dimensional" 
    *" arrays to HDF5 is not supported"))
    create_entry(parent, "$name/flattened_data", flatview(data); kwargs...)
    create_entry(
        parent, "$name/cumulative_length", _cumulative_length(data); kwargs...)
    setdatatype!(parent.data_store[name], typeof(data))
    nothing
end

# write Vector{<:RDWaveforms}
function create_entry(parent::LHDataStore, name::AbstractString, 
    data::AbstractVector{<:RDWaveform{T, U}}; kwargs...
    ) where {T<:RealQuantity, U<:RealQuantity}

    create_entry(parent, name, to_table(data); kwargs...)
end

# write NamedTuple
function create_entry(parent::LHDataStore, name::AbstractString,
    data::NamedTuple; kwargs...)

    grp = HDF5.create_group(parent.data_store, name)
    try
        for k in keys(data)
            create_entry(parent, "$name/$(String(k))", data[k]; kwargs...)
        end
        setdatatype!(grp, typeof(data))
    catch
        # A half-written group would block writing the entry again:
        HDF5.delete_object(parent.data_store, name)
        rethrow()
    finally
        close(grp)
    end
    nothing
end

# write Histogram
function create_entry(parent::LHDataStore, name::AbstractString, 
    data::Histogram; kwargs...)

    create_entry(parent, name, _histogram_to_nt(data); kwargs...)
    setdatatype!(parent.data_store[name], typeof(data))
    nothing
end

# write String
function create_entry(parent::LHDataStore, name::AbstractString, 
    data::AbstractString; kwargs...)

    parent.data_store[name] = data
    setdatatype!(parent.data_store[name], typeof(data))
    nothing
end

# write Symbol
function create_entry(parent::LHDataStore, name::AbstractString, 
    data::Symbol; kwargs...)

    parent.data_store[name] = String(data)
    setdatatype!(parent.data_store[name], typeof(data))
    nothing
end

# write NTuple
function create_entry(parent::LHDataStore, name::AbstractString,
    data::NTuple{L,Any}; kwargs...) where {L}

    U = eltype(typeof(data))
    isconcretetype(U) || throw(ArgumentError("Only homogeneous tuples are supported"))
    create_entry(parent, name, collect(U, data); kwargs...)
    setdatatype!(parent.data_store[name], typeof(data))
    nothing
end

# write Array{<:NTuple}
function create_entry(parent::LHDataStore, name::AbstractString,
    data::AbstractArray{T}; kwargs...) where {T <: NTuple{L,Any} where L}

    isconcretetype(eltype(T)) || throw(ArgumentError("Only homogeneous tuples are supported"))
    create_entry(parent, name, _flatview_of_array_of_ntuple(data); kwargs...)
    setdatatype!(parent.data_store[name], typeof(data))
    nothing
end

# write EncodedArray
function create_entry(parent::LHDataStore, name::AbstractString, 
    data::T; kwargs...) where {C, U, T <: EncodedArray{U, 1, C}}

    create_entry(parent, name*"/encoded_data", data.encoded; kwargs...)
    create_entry(parent, name*"/size", data.size)

    # quick fix for avoiding hardcoding eltype while reading EncodedArray's
    parent[name*"/sample_data"] = U(1.0) 

    codec_name = LegendDataTypes.array_codecs.by_type[C]
    setattribute!(parent.data_store[name], :codec, String(codec_name))
    write_to_properties!(setattribute!, parent.data_store[name], data.codec)
    setdatatype!(parent.data_store[name], T)
    nothing
end

# write VectorOfEncodedArrays
function create_entry(parent::LHDataStore, name::AbstractString,
    data::T; kwargs...) where {C, U, T <: VectorOfEncodedArrays{U, 1, C}}

    create_entry(parent, name*"/encoded_data", data.encoded; kwargs...)
    create_entry(parent, name*"/decoded_size", data.innersizes)
        
        # quick fix for avoiding hardcoding eltype while reading EncodedArray's
    parent[name*"/sample_data"] = U(1.0)

    codec_name = LegendDataTypes.array_codecs.by_type[C]
    setattribute!(parent.data_store[name], :codec, String(codec_name))
    write_to_properties!(setattribute!, parent.data_store[name], data.codec)
    setdatatype!(parent.data_store[name], T)
    nothing
end

# write VectorOfEncodedSimilarArrays
function create_entry(parent::LHDataStore, name::AbstractString,
    data::T; kwargs...) where {C, U, T <: VectorOfEncodedSimilarArrays{U, 1, C}}

    create_entry(parent, name*"/encoded_data", data.encoded; kwargs...)
    create_entry(parent, name*"/decoded_size", only(data.innersize); kwargs...)
    parent[name*"/sample_data"] = U(1.0)
    codec_name = LegendDataTypes.array_codecs.by_type[C] |> String
    setattribute!(parent.data_store[name], :codec, codec_name)
    write_to_properties!(setattribute!, parent.data_store[name], data.codec)
    setdatatype!(parent.data_store[name], T)
    nothing
end

# A table is a vector of rows, so its columns must be vectors:
_check_vector_columns(cols) = all(col -> col isa AbstractVector, cols) ||
    throw(ArgumentError("The columns of a table must be vectors"))

# write tables (anything that satisfies the Tables.jl column interface)
function create_entry(parent::LHDataStore, name::AbstractString, data;
    kwargs...)

    Tables.istable(data) || throw(ArgumentError("Value to write, of type "
    *"$(typeof(data)), is not a table"))
    cols = Tables.columntable(data)
    _check_vector_columns(cols)
    create_entry(parent, name, cols; kwargs...)
    setdatatype!(parent.data_store[name], StructArray{NamedTuple{keys(cols)}})
    nothing
end

"""
    lh5open(filename::AbstractString, access::AbstractString = "r";
        usechunks::Bool = false, compress = false)

Open a LEGEND HDF5 file and return an [`LHDataStore`](@ref) object.

With `usechunks = true`, newly written array datasets are chunked and can be
extended with `append!`. `compress` selects their compression: `true` or
`:zstd` for Zstandard, `:deflate` (alias `:gzip`) for shuffle plus deflate.
Compressed datasets are always chunked. LEGEND HDF5 files typically use the
file extension ".lh5".
"""
function lh5open(filename::AbstractString, access::AbstractString = "r";
    usechunks::Bool = false, compress::Union{Bool,Symbol} = false)

    mode = _compress_mode(compress)
    LHDataStore(HDF5.h5open(filename, access), usechunks, mode)
end
export lh5open

"""
    lh5open(f, filename::AbstractString, access::AbstractString = "r"; kwargs...)

Open the LEGEND HDF5 file `filename`, apply `f` to the resulting
[`LHDataStore`](@ref) and close the file afterwards, returning the result
of `f`.
"""
function lh5open(f::Function, filename::AbstractString, 
    access::AbstractString = "r"; kwargs...)
    
    lhds = lh5open(filename, access; kwargs...)
    try
       f(lhds) 
    finally
        close(lhds)
    end
end

"""
    add_entries!(lhd::LHDataStore, i::AbstractString, src::StructArray)

Add the columns of `src` to the table at `lhd[i]`.
"""
function add_entries!(lhd::LHDataStore, i::AbstractString, src::StructArray{<:NamedTuple})
    current = LH5Array(lhd.data_store[i])
    current isa LH5LazyTable || throw(ArgumentError(
        "\"$i\" is not a table, add entries to a NamedTuple as a NamedTuple"))
    _check_vector_columns(StructArrays.components(src))
    length(current) == length(src) || throw(DimensionMismatch(
        "Cannot add columns of length $(length(src)) to table of length $(length(current))"))
    names = _add_entries!(lhd, i, StructArrays.components(src), propertynames(current))
    setdatatype!(lhd.data_store[i], StructArray{NamedTuple{names}})
    nothing
end

"""
    add_entries!(lhd::LHDataStore, i::AbstractString, src::NamedTuple)

Add the entries of `src` to the `NamedTuple` at `lhd[i]`.
"""
function add_entries!(lhd::LHDataStore, i::AbstractString, src::NamedTuple)
    current = LH5Array(lhd.data_store[i])
    current isa NamedTuple || throw(ArgumentError(
        "\"$i\" is not a NamedTuple, add columns to a table as a StructArray"))
    names = _add_entries!(lhd, i, src, keys(current))
    setdatatype!(lhd.data_store[i], NamedTuple{names})
    nothing
end
export add_entries!

# Writes the entries and returns the new list of names. The names come from
# the file, since a handle may predate earlier additions:
function _add_entries!(lhd::LHDataStore, i::AbstractString, src::NamedTuple, stored)
    clash = intersect(stored, keys(src))
    isempty(clash) || throw(ArgumentError("\"$i\" already has entries $clash"))
    for k in keys(src)
        lhd["$(i)/$(k)"] = src[k]
    end
    (stored..., keys(src)...)
end

"""
    delete_entry!(lhd::LHDataStore, i::AbstractString)

Remove the dataset `lhd[i]` and adjust the datatype of the parent if necessary.
Supported are entries of `NamedTuple`s, tables and `HDF5.Group`s.
"""
function delete_entry!(lhd::LHDataStore, i::AbstractString)
    parent, child = splitdir(i)
    if isempty(parent) || (parent == "/")
        HDF5.delete_object(lhd.data_store[i])
    else
        _delete_entry(lhd, lhd[parent], parent, child)
    end
end
export delete_entry!

function _remaining_entries(stored, parent::AbstractString, child::AbstractString)
    Symbol(child) in stored || throw(ArgumentError("\"$parent\" has no entry \"$child\""))
    remaining = filter(!=(Symbol(child)), stored)
    isempty(remaining) && throw(ArgumentError("Cannot delete last entry \"$child\" of \"$parent\""))
    remaining
end

function _delete_entry(lhd::LHDataStore, nt::NamedTuple,
    parent::AbstractString, child::AbstractString)

    remaining = _remaining_entries(keys(nt), parent, child)
    HDF5.delete_object(lhd.data_store["$(parent)/$(child)"])
    if hasattribute(lhd.data_store[parent], :datatype)
        setdatatype!(lhd.data_store[parent], NamedTuple{remaining})
    end
    nothing
end

function _delete_entry(lhd::LHDataStore, tbl::LH5LazyTable,
    parent::AbstractString, child::AbstractString)

    remaining = _remaining_entries(propertynames(tbl), parent, child)
    HDF5.delete_object(lhd.data_store["$(parent)/$(child)"])
    setdatatype!(lhd.data_store[parent], StructArray{NamedTuple{remaining}})
    nothing
end
