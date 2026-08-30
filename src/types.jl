# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

export LHDataStore, LH5Array

"""
    LH5Array{T, N} <: AbstractArray{T, N}

Array wrapper for an `HDF5.Dataset` following the LEGEND data format, as in
".lh5" files.

`getindex` reads the requested part of the on-disk array, so data can be
read partially without loading the whole array first. `append!` extends the
on-disk array (which requires it to be chunked); data is always appended
along the last dimension.

`LH5Array` implements the [DiskArrays.jl](https://github.com/JuliaIO/DiskArrays.jl)
interface: views, iteration and reductions read the data block-wise, and
broadcasts are lazy.

# Default constructors

```julia
LH5Array{T, N}(ds::HDF5.Dataset)
LH5Array{T}(ds::HDF5.Dataset)
LH5Array(ds::Union{HDF5.Dataset, HDF5.H5DataStore})
```

# Example:

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

# ArraysOfArrays v1 replaces the data dimensionality parameter of
# ArrayOfSimilarArrays by the element type:
@static if isdefined(ArraysOfArrays, :PartsView)
    const LH5AoSA{T, M, N, L} = ArrayOfSimilarArrays{T, M, N, LH5Array{T, L}}
else
    const LH5AoSA{T, M, N, L} = ArrayOfSimilarArrays{T, M, N, L, LH5Array{T, L}}
end
const LHIndexType = Union{Colon, AbstractRange{Int}, AbstractVector{Int}}
const VectorOfRDWaveforms{T, U, VVT, VVU} = ArrayOfRDWaveforms{T, U, 1, VVT, VVU}
const LH5VoV{T} = VectorOfVectors{T, LH5Array{T, 1}}
const LH5TableColumn = Union{LH5Array{<:Any, 1}, LH5VoV, LH5AoSA{<:Any, <:Any, 1}}
const LH5TableColumns = NamedTuple{names, <:Tuple{Vararg{LH5TableColumn}}} where names
const LH5Table = StructArrays.StructVector{<:NamedTuple, <:LH5TableColumns}
const LH5ArrayOfRDWaveforms{T, U, N, VVT} = 
    ArrayOfRDWaveforms{T, U, N, VVT, <:Union{LH5VoV{U}, LH5AoSA{U}}}
const LH5VectorOfRDWaveforms{T, U} = LH5ArrayOfRDWaveforms{T, U, 1}

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
    ET = (u == NoUnits) ? eltype(ds) : typeof(eltype(ds)(0) * u)
    LH5Array{ET}(ds)
end
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
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:NamedTuple{T}}) where T

return a `NamedTuple` where each `field` is the output of `LH5Array` applied to it.
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:NamedTuple{T}}) where {T} =
    NamedTuple{T}(map(k -> LH5Array(ds[String(k)]), T))
"""
    LH5Array(ds::HDF5.DataStore, ::Type{<:StructArray{<:NamedTuple{(T)}}}) where T

return a `StructArray` where each column is the output of `LH5Array` applied to it.
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:StructArray{<:NamedTuple{(T)}}}) where T =
    StructArray(LH5Array(ds, NamedTuple{T}))
"""
    LH5Array(ds::HDF5.DataStore, ::Type{<:AbstractVector{<:RDWaveform}})

return an `ArrayOfRDWaveforms` where the field `signal` is either a 
`VectorOfSimilarVectors` with an `LH5Array` as `data` or `VectorOfVectors` 
with an `LH5Array` as `data` (see `ArrayOfRDWaveforms` and `ArraysOfArrays`) 
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:AbstractVector{<:RDWaveform}}) = begin
    tbl = LH5Array(ds, StructArray{<:NamedTuple{(:t0, :dt, :values)}})
    from_table(tbl, AbstractVector{<:RDWaveform})
end
"""
    LH5Array(ds::HDF5.DataStore, ::Type{<:AbstractVector{<:AbstractVector{<:RealQuantity}}})

return a `VectorOfVectors` object where `data` is an `LH5Array` 
(see `VectorOfArrays`)
"""
LH5Array(ds::HDF5.H5DataStore, 
::Type{<:AbstractVector{<:AbstractVector}}) = begin
    data = LH5Array(ds["flattened_data"])
    cumulen = LH5Array(ds["cumulative_length"])[:]
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

Base.getindex(lh::LH5Array{T, N}, idxs::Vararg{HDF5.IndexType, N}
) where {T, N} = begin
    dtype = HDF5.datatype(lh.file)
    val = try
        HDF5.generic_read(lh.file, dtype, T, idxs...)
    finally
        close(dtype)
    end
    _read_result(val, T, idxs...)
end

Base.getindex(lh::LH5Array{Bool, N}, idxs::Vararg{HDF5.IndexType, N}
) where {N} = begin
    dtype = HDF5.datatype(lh.file)
    val = try
        HDF5.generic_read(lh.file, dtype, Bool, idxs...)
    finally
        close(dtype)
    end
    _read_result(val, Bool, idxs...) .> 0
end

Base.getindex(lh::LH5AoSA{T, M}, idxs::LHIndexType...) where {T, M} = begin
    indices = (ArraysOfArrays._ncolons(Val{M}())..., idxs...)
    ArrayOfSimilarArrays{T, M}(lh.data[indices...])
end

function _append_elemptr!(dest_ptr::AbstractVector{<:Integer}, src_ptr::AbstractVector{<:Integer})
    offset = last(dest_ptr) - first(src_ptr)
    append!(dest_ptr, view(src_ptr, firstindex(src_ptr) + 1:lastindex(src_ptr)) .+ offset)
end

# Scattered reads along the last (event) dimension. Depending on index
# density, dataset rank and layout they map to a single bounding-range read,
# a single read with a scattered dataspace selection, or one hyperslab read
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

const _scatter_bulk_max_bytes = 2^20

_select_lastdim(A::AbstractArray{<:Any, K}, i) where {K} =
    A[ntuple(_ -> Colon(), Val(K - 1))..., i]

function _getindex_scattered_lastdim(
    lh::LH5Array{T, N}, front::NTuple{M, Any}, ilast::AbstractVector{<:Integer}
) where {T, N, M}
    issorted(ilast) && return _getindex_scattered_sorted(lh, front, ilast)
    p = sortperm(ilast)
    _select_lastdim(_getindex_scattered_sorted(lh, front, ilast[p]), invperm(p))
end

function _getindex_scattered_sorted(
    lh::LH5Array{T, N}, front::NTuple{M, Any}, ilast::AbstractVector{<:Integer}
) where {T, N, M}
    isempty(ilast) && return lh[front..., 1:0]
    lo, hi = Int(first(ilast)), Int(last(ilast))
    span = hi - lo + 1
    row_bytes = sizeof(T) * prod(Base.front(size(lh)))
    # For small or densely covered index spans a single bounding read is
    # cheaper than any scattered read:
    if span * row_bytes <= _scatter_bulk_max_bytes || 4 * length(ilast) >= span
        return _select_lastdim(lh[front..., lo:hi], ilast .- (lo - 1))
    end
    # Point selections beat per-run reads for vectors of any layout, hyperslab
    # unions only for contiguous-layout datasets (libhdf5 maps large irregular
    # selections onto chunks slowly, while sorted per-run reads make good use
    # of the chunk cache):
    if all(i -> i isa Colon, front) && (N == 1 || _is_contiguous(lh.file))
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

# HDF5.jl gained bindings for these libhdf5 functions in v0.17:
@static if isdefined(HDF5.API, :h5s_select_elements)
    const _h5s_select_elements = HDF5.API.h5s_select_elements
    const _h5s_modify_select = HDF5.API.h5s_modify_select
else
    function _h5s_select_elements(space_id, op, num_elem, coord)
        ret = ccall((:H5Sselect_elements, HDF5.API.libhdf5), HDF5.API.herr_t,
            (HDF5.API.hid_t, HDF5.API.H5S_seloper_t, Csize_t, Ptr{HDF5.API.hsize_t}),
            space_id, op, num_elem, coord)
        ret < 0 && error("Error selecting dataspace elements")
        nothing
    end
    function _h5s_modify_select(space1_id, op, space2_id)
        ret = ccall((:H5Smodify_select, HDF5.API.libhdf5), HDF5.API.herr_t,
            (HDF5.API.hid_t, HDF5.API.H5S_seloper_t, HDF5.API.hid_t),
            space1_id, op, space2_id)
        ret < 0 && error("Error modifying dataspace selection")
        nothing
    end
end

# A single H5Dread with a scattered dataspace selection. Requires sorted
# indices, since HDF5 reads selections in index order:
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
# positions of the inputs therein:
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
    _h5s_select_elements(fspace, HDF5.API.H5S_SELECT_SET, length(coords), coords)
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
            _h5s_modify_select(fspace, HDF5.API.H5S_SELECT_OR, fs2)
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

Base.getindex(lh::LH5Array{T, N},
    idxs::Vararg{Union{HDF5.IndexType, AbstractVector{<:Integer}}, N}
) where {T, N} = begin
    front, ilast = Base.front(idxs), idxs[end]
    if ilast isa AbstractVector{Bool}
        lh[front..., findall(_materialize(ilast))]
    elseif ilast isa Base.LogicalIndex
        # to_indices turns logical masks into iterate-only LogicalIndex;
        # materialize the mask, iterating it would read lazy masks per element:
        lh[front..., findall(_materialize(ilast.mask))]
    elseif ilast isa AbstractVector{<:Integer} && !(ilast isa AbstractRange) &&
        all(i -> i isa HDF5.IndexType, front)
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

DiskArrays.haschunks(lh::LH5Array) =
    isnothing(_chunk_dims(lh.file)) ? DiskArrays.Unchunked() : DiskArrays.Chunked()

function DiskArrays.eachchunk(lh::LH5Array)
    chunk = _chunk_dims(lh.file)
    isnothing(chunk) ? DiskArrays.estimate_chunksize(lh) : DiskArrays.GridChunks(lh, chunk)
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

Base.copyto!(dest::Array, src::LH5Array) = begin
    indices = ArraysOfArrays._ncolons(Val{ndims(src)}())
    copyto!(dest, src.file, indices...)
end

# Deep conversion of lazily read data into in-memory objects:

_materialize(x) = x
_materialize(A::DiskArrays.AbstractDiskArray) = Array(A)
_materialize(A::LH5VoV) = VectorOfVectors(_materialize(A.data), copy(A.elem_ptr))
_materialize(A::LH5AoSA{T, M}) where {T, M} =
    ArrayOfSimilarArrays{T, M}(_materialize(A.data))
_materialize(x::NamedTuple) = map(_materialize, x)
_materialize(A::StructArray{<:NamedTuple}) =
    StructArray(map(_materialize, StructArrays.components(A)))
_materialize(A::ArrayOfRDWaveforms) =
    ArrayOfRDWaveforms((_materialize(A.time), _materialize(A.signal)))
_materialize(A::VectorOfEncodedArrays{T}) where {T} =
    VectorOfEncodedArrays{T}(A.codec, A.innersizes, _materialize(A.encoded))
_materialize(A::VectorOfEncodedSimilarArrays{T}) where {T} =
    VectorOfEncodedSimilarArrays{T}(A.codec, A.innersize, _materialize(A.encoded))

@inline _ustrip(x::AbstractArray{T}) where T<:Real = x
@inline _ustrip(x::AbstractArray{T}) where T<:Quantity = 
    reinterpret(Unitful.numtype(T), x) 

Base.append!(dest::LH5Array{T, 1}, src::EncodedArray) where {T} =
    append!(dest, collect(src))

Base.append!(dest::LH5Array{T, N}, src::AbstractArray) where {T, N} = begin
    x = convert(Array{T, N}, src)
    old_size = size(dest)
    new_size = (old_size[1:N-1]..., old_size[N] + size(src, N))
    from, to = old_size[N] + 1, new_size[N]
    indices = (ArraysOfArrays._ncolons(Val{N-1}())..., from:to)
    HDF5.set_extent_dims(dest.file, new_size)
    dest.file[indices...] = _ustrip(x)
    dest
end

Base.append!(dest::LH5VoV, src::VectorOfVectors) = begin
    if !isempty(src)
        src_flat = src.data[first(src.elem_ptr):(last(src.elem_ptr) - 1)]
        old_len = last(dest.elem_ptr) - first(dest.elem_ptr)
        append!(dest.data, src_flat)
        _append_elemptr!(dest.elem_ptr, src.elem_ptr)
        append!(dest.kernel_size, src.kernel_size)

        new_clen = cumsum(diff(src.elem_ptr)) .+ old_len
        dset = parent(dest.data.file)["cumulative_length"]
        try
            append!(LH5Array(dset), new_clen)
        finally
            close(dset)
        end
    end
    dest
end

# Non-scalar reads of disk-backed tables must recompute the row type, since
# the element type of lazy columns changes when they are read:
Base.getindex(A::LH5Table, idxs::Union{AbstractVector, Colon}) =
    StructArray(map(col -> col[idxs], StructArrays.components(A)))

# StructArrays appends column-wise only when the element types of both
# tables match exactly, which disk-backed and in-memory tables rarely do:
function _append_table!(dest, src)
    dcols = StructArrays.components(dest)
    scols = Tables.columntable(src)
    issetequal(keys(dcols), keys(scols)) || throw(ArgumentError(
        "Cannot append table with columns $(keys(scols)) to table with columns $(keys(dcols))"))
    for k in keys(dcols)
        append!(dcols[k], scols[k])
    end
    dest
end

Base.append!(dest::LH5Table, src) = _append_table!(dest, src)

# Disambiguation against the column-wise append of StructArrays and the
# element append of EncodedArrays:
Base.append!(dest::StructArrays.StructVector{T, <:LH5TableColumns},
    src::StructArrays.StructVector{T}) where {T<:NamedTuple} = _append_table!(dest, src)
Base.append!(dest::LH5Table, src::EncodedArray) =
    throw(ArgumentError("Cannot append an encoded array to a table"))

Base.append!(dest::LH5VectorOfRDWaveforms, src::VectorOfRDWaveforms) = begin
    # first append values to on-disk array
    StructArrays.foreachfield(append!, dest, src)

    # and then append time information to on disk array
    src_t0 = first.(src.time)
    src_dt = step.(src.time)
    grp = parent(dest.signal.data.file)
    dset_t0 = grp["t0"]
    dset_dt = grp["dt"]
    try
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

Dictionary wrapper for an `HDF5.H5DataStore` (typically an `HDF5.File`, but
may also be e.g. an `HDF5.Group`) following the LEGEND data format, as in
".lh5" files.

Constructor:

```julia
LHDataStore(h5ds::HDF5.H5DataStore, usechunks::Bool, compress::Symbol = :none)
```

To read from or write to ".lh5" files directly (without using `HDF5.h5open`
first), use [`lh5open`](@ref).

`getindex(lh::LHDataStore, s)` returns the output of [`LH5Array`](@ref)
applied to `lh.data_store[s]`, so arrays are wrapped lazily. `setindex!`
creates and writes `HDF5.Group`s and `HDF5.Dataset`s. With `usechunks = true`
datasets are created chunked and extensible, so they can later be extended
with `append!`; the chunk size is taken from the first array written. With
`compress` set to `:zstd` or `:deflate`, datasets are chunked and compressed. Supported
value types include `RealQuantity` and arrays thereof, `Bool`, `String`,
`Symbol`, `Enum`s, (arrays of) `NTuple`s and `StaticVector`s,
`ArraysOfSimilarArrays`, `VectorOfVectors`, encoded arrays, `NamedTuple`s,
tables, `AbstractVector{<:RDWaveform}` and `Histogram`. **For arrays, the
last axis is assumed to correspond to the event number index.**

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

# Store Bool values as UInt8 for h5py compatibility, like writedata does
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

# write AbstractArray{<:String}
function create_entry(parent::LHDataStore, name::AbstractString, 
    data::AbstractArray{String, N}; kwargs...) where {N}

    parent.data_store[name] = data
    setdatatype!(parent.data_store[name], Array{String, N})
    nothing
end

# write AbstractArray{<:Real}
function create_entry(parent::LHDataStore, name::AbstractString,
    data::AbstractArray{T}; usechunks::Bool=false, compress::Symbol=:none
) where {T<:Real}

    dtype = HDF5.datatype(T)
    ds = if !usechunks && compress === :none
        HDF5.create_dataset(parent.data_store, name, dtype, size(data))
    else
        # Compression requires a chunked dataset. The size of the first
        # written array informs the chunk size:
        data_size = size(data)
        sz_inner, sz_outer = data_size[begin:end-1], data_size[end]
        sz_outer > 0 || throw(ArgumentError(
            "Cannot infer a chunk size for \"$name\" from an empty array"))
        dspace = (data_size, (sz_inner..., -1))
        chunk = (sz_inner..., sz_outer)
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

# write tables (anything that satisfies the Tables.jl column interface)
function create_entry(parent::LHDataStore, name::AbstractString, data;
    kwargs...)

    Tables.istable(data) || throw(ArgumentError("Value to write, of type "
    *"$(typeof(data)), is not a table"))
    cols = Tables.columntable(data)
    create_entry(parent, name, cols; kwargs...)
    setdatatype!(parent.data_store[name], StructArray{NamedTuple{keys(cols)}})
    nothing
end

"""
    lh5open(filename::AbstractString, access::AbstractString = "r";
        usechunks::Bool = false, compress = false)

Open a LEGEND HDF5 file and return an [`LHDataStore`](@ref) object.

With `usechunks = true`, newly written datasets are chunked and can be
extended with `append!`. `compress` selects dataset compression: `true` or
`:zstd` for Zstandard, `:deflate` (alias `:gzip`) for shuffle plus deflate;
compressed datasets are always chunked. LEGEND HDF5 files typically use the
file extension ".lh5".
"""
function lh5open(filename::AbstractString, access::AbstractString = "r";
    usechunks::Bool = false, compress::Union{Bool,Symbol} = false)

    LHDataStore(HDF5.h5open(filename, access), usechunks, _compress_mode(compress))
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
    add_entries!(lhd::LHDataStore, i::AbstractString, src::StructArray,
        dest::StructArray=LH5Array(lhd.data_store[i]))

extend the table `dest` at `lhd[i]` with columns from `src`.
"""
function add_entries!(lhd::LHDataStore, i::AbstractString,
    src::StructArray{<:NamedTuple},
    dest::StructArray{<:NamedTuple} = LH5Array(lhd.data_store[i]))

    length(dest) == length(src) || throw(DimensionMismatch(
        "Cannot add columns of length $(length(src)) to table of length $(length(dest))"))
    new_cols = (; StructArrays.components(dest)..., StructArrays.components(src)...)
    add_entries!(lhd, i, StructArrays.components(src), StructArrays.components(dest))
    setdatatype!(lhd.data_store[i], StructArray{NamedTuple{keys(new_cols)}})
    nothing
end

"""
    add_entries!(lhd::LHDataStore, i::AbstractString, src::NamedTuple,
        dest::NamedTuple=LH5Array(lhd.data_store[i]))

extend the NamedTuple `dest` at `lhd[i]` with elements from `src`.
"""
function add_entries!(lhd::LHDataStore, i::AbstractString, src::NamedTuple,
    dest::NamedTuple=LH5Array(lhd.data_store[i]))

    new_nt = (;dest..., src...)
    for k in keys(src)
        lhd["$(i)/$(k)"] = src[k]
    end
    setdatatype!(lhd.data_store[i], typeof(new_nt))
    nothing
end
export add_entries!

"""
    delete_entry!(lhd::LHDataStore, i::AbstractString)

remove the dataset `lhd[i]` and adjust the datatype of the parent if necessary. 
Currently supported are elements of `NamedTuple`s, tables or
`HDF5.Group`s.
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

function _delete_entry(lhd::LHDataStore, nt::NamedTuple, 
    parent::AbstractString, child::AbstractString)

    if hasattribute(lhd.data_store[parent], :datatype)
        newkeys = setdiff(keys(nt), (Symbol(child),))
        isempty(newkeys) && throw(ArgumentError("Cannot delete last entry \"$child\" of \"$parent\""))
        new_nt = (;[k => nt[k] for k in newkeys]...)
        setdatatype!(lhd.data_store[parent], typeof(new_nt))
    end
    HDF5.delete_object(lhd.data_store["$(parent)/$(child)"])
    nothing
end

function _delete_entry(lhd::LHDataStore, tbl::StructArray{<:NamedTuple},
    parent::AbstractString, child::AbstractString)

    _delete_entry(lhd, StructArrays.components(tbl), parent, child)
    newkeys = filter(!=(Symbol(child)), propertynames(tbl))
    setdatatype!(lhd.data_store[parent], StructArray{NamedTuple{newkeys}})
    nothing
end