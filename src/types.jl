# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

export LHDataStore, LH5Array

"""
    LH5Array{T, N} <: AbstractArray{T, N}

Array wrapper for HDF5.Datasets following the LEGEND data format as in ".lh5"
files. 

An `LH5Array` contains a HDF5.Dataset `file` and Unitful.Unitlike `units` as 
returned by `getunits`(file)`. `getindex` and `append!` are supported.
`getindex` essentially falls back to `getindex` for `HDF5.Dataset`s, 
enabling the user to always read in the desired part of an ondisk array without 
having to load it in whole beforehand.
`append!` uses chunks to append the data provided to the ondisk array. **It is
important to note, that data is always appended along the last dimension of an 
array**

# Default constructors

```julia
LH5Array{T}(ds::HDF5.Dataset, u::Unitful.Unitlike)
LH5Array{T, N}(ds::HDF5.Dataset)
LH5Array(ds::Union{HDF5.Dataset, HDF5.H5DataStore})

```

# Examples:
```julia
julia> using HDF5
julia> f = h5open("path/to/lh5/file", "r")
julia> l5 = LH5Array(f["path/to/HDF5/Dataset"])
[...]
julia> x = lh[1:10]     # load the first 10 elements of the ondisk array
[...]
julia> append!(lh, x)   # append those 10 elements to the ondisk array 
[...]
```

"""
mutable struct LH5Array{T, N} <: AbstractArray{T, N}
    file::HDF5.Dataset
end

# ArraysOfArrays v1 replaces the data dimensionality parameter of
# ArrayOfSimilarArrays by the element type:
@static if isdefined(ArraysOfArrays, :PartsView)
    const LH5AoSA{T, M, N, L} = ArrayOfSimilarArrays{T, M, N, LH5Array{T, L}}
else
    const LH5AoSA{T, M, N, L} = ArrayOfSimilarArrays{T, M, N, L, LH5Array{T, L}}
end
const LHIndexType = Union{Colon, AbstractRange{Int}}
const VectorOfRDWaveforms{T, U, VVT, VVU} = ArrayOfRDWaveforms{T, U, 1, VVT, VVU}
const LH5VoV{T} = VectorOfVectors{T, LH5Array{T, 1}}
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
LH5Array(ds::HDF5.Dataset, ::Type{<:AbstractArray}) = begin
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
    NamedTuple{T}(LH5Array.([ds[k] for k in String.(T)]))
"""
    LH5Array(ds::HDF5.DataStore, ::Type{<:TypedTables.Table{<:NamedTuple{(T)}}}) where T

return a `Table` where each column is the output of `LH5Array` applied to it.
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:Table{<:NamedTuple{(T)}}}) where T = 
    Table(LH5Array(ds, NamedTuple{T}))
"""
    LH5Array(ds::HDF5.DataStore, ::Type{<:AbstractVector{<:RDWaveform}})

return an `ArrayOfRDWaveforms` where the field `signal` is either a 
`VectorOfSimilarVectors` with an `LH5Array` as `data` or `VectorOfVectors` 
with an `LH5Array` as `data` (see `ArrayOfRDWaveforms` and `ArraysOfArrays`) 
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:AbstractVector{<:RDWaveform}}) = begin
    tbl = LH5Array(ds, Table{<:NamedTuple{(:t0, :dt, :values)}})
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
""""
    LH5Array(ds::HDF5.H5DataStore, ::Type{<:Histogram{<:RealQuantity}})

return a `Histogram`. 
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:Histogram{<:RealQuantity}}) = begin
    T = (:binning, :weights, :isdensity)
    nt = LH5Array(ds, NamedTuple{T})
    nt = (
        binning=nt.binning,
        weights=Array(nt.weights),
        isdensity=nt.isdensity
    )
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
LH5Array(ds::HDF5.Dataset, ::Type{<:AbstractArray{<:StaticVector}}) = begin
    data = getcontent(ds)
    u = getunits(ds)
    L = size(data, 1)
    nestedview(u == NoUnits ? data : data * u, SVector{L})
end
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:Tuple})

return an `Tuple`
"""
LH5Array(ds::HDF5.Dataset, ::Type{<:Tuple}) = tuple(LH5Array(ds, Vector)...)
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:AbstractArray{<:Tuple}})

return an Array of NTuples
"""
LH5Array(ds::HDF5.Dataset, AT::Type{<:AbstractArray{<:NTuple}}) = begin
    data = read(ds)
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

Base.getindex(lh::LH5Array{T, N}, idxs::Vararg{HDF5.IndexType, N}
) where {T, N} = begin
    dtype = HDF5.datatype(lh.file)
    val = HDF5.generic_read(lh.file, dtype, T, idxs...)
    close(dtype)
    return val
end

Base.getindex(lh::LH5Array{Bool, N}, idxs::Vararg{HDF5.IndexType, N}
) where {N} = begin
    dtype = HDF5.datatype(lh.file)
    val = HDF5.generic_read(lh.file, dtype, Bool, idxs...)
    close(dtype)
    return val .> 0
end

Base.getindex(lh::LH5AoSA{T, M}, idxs::LHIndexType...) where {T, M} = begin
    indices = (ArraysOfArrays._ncolons(Val{M}())..., idxs...)
    ArrayOfSimilarArrays{T, M}(lh.data[indices...])
end

function _append_elemptr!(dest_ptr::AbstractVector{<:Integer}, src_ptr::AbstractVector{<:Integer})
    offset = last(dest_ptr) - first(src_ptr)
    append!(dest_ptr, view(src_ptr, firstindex(src_ptr) + 1:lastindex(src_ptr)) .+ offset)
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

Base.copyto!(dest::AbstractArray, src::LH5Array) = begin
    indices = ArraysOfArrays._ncolons(Val{ndims(src)}())
    copyto!(dest, src.file, indices...)
end

@inline _ustrip(x::AbstractArray{T}) where T<:Real = x
@inline _ustrip(x::AbstractArray{T}) where T<:Quantity = 
    reinterpret(Unitful.numtype(T), x) 

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

Dictionary wrapper for `HDF5.H5DataStore` objects, which were constructed 
according to the LEGEND data format in ".lh5" files. 

Constructor:

```julia
LHDataStore(h5ds::HDF5.DataStore)
```

This return an `LHDataStore` object that wraps an `h5ds` which will typically
be an `HDF5.File` be may also be an `HDF5.H5DataStore` (e.g. an `HDF5.Group`).
in general.

To read or write ".lh5" file directly (without using `HDF5.h5open` first),
we recommend using [`lh5open`](@ref).

Supports `getindex` and `setindex!` where `getindex(lh::LHDataStore, s)` returns 
the output of [`LH5Array`](@ref) applied to `data_store[s]` and `setindex!` 
creates and writes `HDF5.Group`s and `HDF5.Dataset`s using chunks of size 1000 
to the ondisk array. Currently supported are objects with types:
`AbstractArray{<:RealQuantity}`, `ArraysOfSimilarArrays{<:RealQuantity}`, 
`VectorOfVectors{<:RealQuantity}`, `NamedTuple`,`TypedTables.Table`, 
`Vector{<:RDWaveform}`. **For `AbstractArray{<:RealQuantity}` It is assumed 
that the last axis of the provided array corresponds to the event number 
index**.

# Example 

```julia
julia> using HDF5
julia> h5ds = h5open("path/to/lhf/file")
julia> lhf = LHDataStore(h5ds)
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
end

@deprecate LHDataStore(f::AbstractString, access::AbstractString = "r") lh5open(f, access)
@deprecate LHDataStore(f::Function, s::AbstractString, access::AbstractString = "r") lh5open(f, s, access)

Base.isopen(f::LHDataStore) = isopen(f.data_store)
Base.close(f::LHDataStore) = close(f.data_store)
Base.keys(lh::LHDataStore) = keys(lh.data_store)
Base.haskey(lh::LHDataStore, i::AbstractString) = haskey(lh.data_store, i)
Base.getindex(lh::LHDataStore, i::AbstractString) = LH5Array(lh.data_store[i])
Base.getindex(lh::LHDataStore, i::Any...) = getindex(lh, join(string.(i), "/"))

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
    create_entry(lh, i, v, usechunks=lh.usechunks)
    nothing
end

Base.setindex!(lh::LHDataStore, v, i::Any...) = 
    setindex!(lh, v, join(string.(i), "/"))

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
    data::AbstractArray{T}; usechunks::Bool=false) where {T<:Real}

    dtype = HDF5.datatype(T)
    ds = if !usechunks
        HDF5.create_dataset(parent.data_store, name, dtype, size(data))
    else
        # The size of the first written array informs the chunk size:
        data_size = size(data)
        sz_inner, sz_outer = data_size[begin:end-1], data_size[end]
        sz_outer > 0 || throw(ArgumentError(
            "Cannot infer a chunk size for \"$name\" from an empty array"))
        dspace = (data_size, (sz_inner..., -1))
        chunk = (sz_inner..., sz_outer)
        HDF5.create_dataset(parent.data_store, name, dtype, dspace; chunk=chunk)
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

# write ArrayOfSimilarArrays{<:RealQuantity}
function create_entry(parent::LHDataStore, name::AbstractString,
    data::ArrayOfSimilarArrays{T}; kwargs...) where {T<:RealQuantity}

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

# write Table
function create_entry(parent::LHDataStore, name::AbstractString, data; 
    kwargs...)

    Tables.istable(data) || throw(ArgumentError("Value to write, of type "
    *"$(typeof(data)), is not a table"))
    create_entry(parent, name, Tables.columns(Table(data)); kwargs...)
    setdatatype!(parent.data_store[name], typeof(data))
    nothing
end

"""
    lh5open(filename::AbstractString, access::AbstractString = "r")

Open a LEGEND HDF5 file and return an `LHDataStore` object.

LEGEND HDF5 files typically use the file extention ".lh5".
"""
function lh5open(filename::AbstractString, access::AbstractString = "r"; 
    usechunks::Bool = false)

    LHDataStore(HDF5.h5open(filename, access), usechunks)
end
export lh5open

"""
    lh5open(f, filename::AbstractString, access::AbstractString = "r")

Return f(lh5open(f, filename, access)).

Opens and closes the LEGEND HDF5 file `filename` automatically.
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
    add_entries!(lhd::LHDataStore, i::AbstractString, src::TypedTable.Table, 
        dest::TypedTable.Table=LH5Array(lhd.data_store[i]))

extend the Table `dest` at `lhd[i]` with columns from `src`.
"""
function add_entries!(lhd::LHDataStore, i::AbstractString,
    src::Table, dest::Table=LH5Array(lhd.data_store[i]))

    length(dest) == length(src) || throw(DimensionMismatch(
        "Cannot add columns of length $(length(src)) to table of length $(length(dest))"))
    tbl = Table(dest, src)
    add_entries!(lhd, i, columns(src), columns(dest))
    HDF5.rename_attribute(lhd.data_store[i], "datatype", "datatype_old")
    HDF5.delete_attribute(lhd.data_store[i], "datatype_old")
    setdatatype!(lhd.data_store[i], typeof(tbl))
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
    HDF5.rename_attribute(lhd.data_store[i], "datatype", "datatype_old")
    HDF5.delete_attribute(lhd.data_store[i], "datatype_old")
    setdatatype!(lhd.data_store[i], typeof(new_nt))
    nothing
end
export add_entries!

"""
    delete_entry!(lhd::LHDataStore, i::AbstractString)

remove the dataset `lhd[i]` and adjust the datatype of the parent if necessary. 
Currently supported are elements of `NamedTuple`, `TypedTable.Table` or 
`HDF5.Group`. 
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
        HDF5.rename_attribute(lhd.data_store[parent], "datatype", "datatype_old")
        HDF5.delete_attribute(lhd.data_store[parent], "datatype_old")
        setdatatype!(lhd.data_store[parent], typeof(new_nt))
    end
    HDF5.delete_object(lhd.data_store["$(parent)/$(child)"])
    nothing
end

function _delete_entry(lhd::LHDataStore, tbl::Table, parent::AbstractString, 
    child::AbstractString)

    _delete_entry(lhd, columns(tbl), parent, child)
    new_tbl = Table(lhd[parent])
    # adjust datatype of parent
    HDF5.rename_attribute(lhd.data_store[parent], "datatype", "datatype_old")
    HDF5.delete_attribute(lhd.data_store[parent], "datatype_old")
    setdatatype!(lhd.data_store[parent], typeof(new_tbl))
    nothing
end