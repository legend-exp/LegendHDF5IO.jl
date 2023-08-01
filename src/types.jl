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

const CHUNK_SIZE = 10_000
const LH5AoSA{T, M, N, L} = ArrayOfSimilarArrays{T, M, N, L, LH5Array{T, L}}
const LHIndexType = Union{Colon, AbstractRange{Int}}
const VectorOfRDWaveforms{T, U, VVT, VVU} = ArrayOfRDWaveforms{T, U, 1, VVT, VVU}
const LH5VoV{T} = VectorOfVectors{T, LH5Array{T, 1}}
const LH5ArrayOfRDWaveforms{T, U, N, VVT} = 
    ArrayOfRDWaveforms{T, U, N, VVT, <:Union{LH5VoV{U}, LH5AoSA{U}}}
const LH5VectorOfRDWaveforms{T, U} = LH5ArrayOfRDWaveforms{T, U, 1}

LH5Array{T}(f::HDF5.Dataset) where {T} = LH5Array{T, ndims(f)}(f)
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
    units == NoUnits || throw(ErrorExceptions("Can't interpret dataset with units as Bool values"))
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
    units == NoUnits || throw(ErrorExceptions("Can't interpret dataset with units as Bool values"))
    LH5Array{Bool}(ds)
end
"""
    LH5Array(ds::HDF5.H5DataStore, ::Type{<:AbstractArrayOfSimilarArrays{<:RealQuantity}})

return an `ArraysOfSimilarArrays` where the field `data` is a `LH5Array` 
(see `ArraysOfSimilarArrays`)
"""
LH5Array(ds::HDF5.Dataset, 
::Type{<:AbstractArrayOfSimilarArrays{<:RealQuantity}}) = begin
    A = LH5Array(ds, AbstractArray{<:RealQuantity})
    D = ndims(A)
    D == 2 ? nestedview(A) : nestedview(A, Val(D - 1))
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
LH5Array(ds::HDF5.H5DataStore, ::Type{<:TypedTables.Table{<:NamedTuple{(T)}}}
) where T =
    TypedTables.Table(LH5Array(ds, NamedTuple{T}))
"""
    LH5Array(ds::HDF5.DataStore, ::Type{<:AbstractVector{<:RDWaveform}})

return an `ArrayOfRDWaveforms` where the field `signal` is either a 
`VectorOfSimilarVectors` with an `LH5Array` as `data` or `VectorOfVectors` 
with an `LH5Array` as `data` (see `ArrayOfRDWaveforms` and `ArraysOfArrays`) 
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:AbstractVector{<:RDWaveform}}) = begin
    tbl = LH5Array(ds, TypedTables.Table{<:NamedTuple{(:t0, :dt, :values)}})
    from_table(tbl, AbstractVector{<:RDWaveform})
end
"""
    LH5Array(ds::HDF5.DataStore, ::Type{<:AbstractVector{<:AbstractVector{<:RealQuantity}}})

return a `VectorOfVectors` object where `data` is an `LH5Array` 
(see `VectorOfArrays`)
"""
LH5Array(ds::HDF5.H5DataStore, 
::Type{<:AbstractVector{<:AbstractVector{<:RealQuantity}}}) = begin
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

return a String object.
"""
LH5Array(ds::HDF5.Dataset, ::Type{<:String}) = read(ds)

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

Base.getindex(lh::LH5ArrayOfRDWaveforms, idxs::LHIndexType...) = 
    ArrayOfRDWaveforms((lh.time[idxs...], lh.signal[idxs...]))

_inv_element_ptrs(el_ptr::AbstractVector{<:Int}) = UInt32.(el_ptr .- 1)[2:end]

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
    x = convert(AbstractArray{T, N}, src)
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
        append!(dest.data, src.data)
        ArraysOfArrays.append_elemptr!(dest.elem_ptr, src.elem_ptr)
        append!(dest.kernel_size, src.kernel_size)

        # prepare elem_ptr to append to "cumulative_length"
        clen = _inv_element_ptrs(dest.elem_ptr)
        dset = parent(dest.data.file)["cumulative_length"]
        tmp = LH5Array(dset)
        from = length(tmp) + 1
        append!(tmp, clen[from:end])
    end
    dest
end

Base.append!(dest::LH5VectorOfRDWaveforms, src::VectorOfRDWaveforms) = begin
    # first append values to on-disk array
    StructArrays.foreachfield(append!, dest, src)

    # and then append time information to on disk array 
    src_t0 = first.(src.time)
    src_dt = step.(src.time)
    dset_t0 = parent(dest.signal.data.file)["t0"]
    dset_dt = parent(dest.signal.data.file)["dt"]
    append!(LH5Array(dset_t0), src_t0)
    append!(LH5Array(dset_dt), src_dt)
    dest
end

"""
    LHDataStore <: AbstractDict{String,Any}

Dictionary wrapper for `HDF5.H5DataStore` objects, which were constructed 
according to the LEGEND data format in ".lh5" files. 

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
julia> lhf = LHDataStore("path/to/lhf/file")
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
end

"""
    LHDataStore(f::HDF5.DataStore)
    LHDataStore(f::AbstractString, access::AbstractString = "r")

create a `LHDataStore` object, where `data_store` is either an 
`HDF5.file` created at path `f` with mode `access` (default is 
read-only), or a HDF5.Group. For more info on mode see `HDF5`.
"""
LHDataStore(f::AbstractString, access::AbstractString = "r") =
    LHDataStore(HDF5.h5open(f, access))

"""
    LHDataStore(f::Funtion, s::AbstractString, access::AbstractString = "r")

Apply the function `f` to the result of `LHDataStore(s, access)` and 
close the resulting `LHDataStore` object. Use with a `do` block:

#Example

    LHDataStore(s) do f
        f["key"] = [1, 2, 3, 4, 5]
    end
"""
LHDataStore(f::Function, s::AbstractString, access::AbstractString = "r"
) = begin
    lhds = LHDataStore(s, access)
    try
       f(lhds) 
    finally
        close(lhds)
    end
end

Base.close(f::LHDataStore) = close(f.data_store)
Base.keys(lh::LHDataStore) = keys(lh.data_store)
Base.haskey(lh::LHDataStore, i::AbstractString) = haskey(lh.data_store, i)
Base.getindex(lh::LHDataStore, i::AbstractString) = LH5Array(lh.data_store[i])

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


# write <:Real
Base.setindex!(output::LHDataStore, v::T, i::AbstractString, 
DT::DataType=typeof(v)) where {T<:Real} = begin
    output.data_store[i] = v
    DT != Nothing && setdatatype!(output.data_store[i], DT)
    nothing
end

# write <:Quantity
Base.setindex!(output::LHDataStore, v::T, i::AbstractString, 
DT::DataType=typeof(v)) where {T<:Quantity} = begin
    output[i, DT] = ustrip(v)
    setunits!(output.data_store[i], unit(T))
    nothing
end

# write AbstractArray{<:Real}
Base.setindex!(output::LHDataStore, v::AbstractArray{T}, i::AbstractString, 
DT::DataType=typeof(v)) where {T<:Real} = begin
    evntsize = size(v)[1:end-1]
    dspace = (size(v), (evntsize..., -1))
    chunk = (evntsize..., CHUNK_SIZE)
    dtype = HDF5.datatype(T)
    ds = HDF5.create_dataset(output.data_store, i, dtype, dspace; chunk=chunk)
    try
        HDF5.write_dataset(ds, dtype, Array(v))
        DT != Nothing && setdatatype!(ds, DT)
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
Base.setindex!(output::LHDataStore, v::AbstractArray{T}, i::AbstractString, 
DT::DataType=typeof(v)) where {T<:Quantity}  = begin
    output[i, DT] = _ustrip(v)
    setunits!(output.data_store[i], unit(T))
    nothing
end

# write ArrayOfSimilarArrays{<:RealQuantity}
Base.setindex!(output::LHDataStore, v::ArrayOfSimilarArrays{T}, 
i::AbstractString) where T<:RealQuantity = begin
    output[i, typeof(v)] = flatview(v)
    nothing
end

# write VectorOfVectors{<:RealQuantity}
Base.setindex!(output::LHDataStore, v::AbstractArray{<:AbstractArray{T, M}, N}, 
i::AbstractString) where {T<:RealQuantity, M, N} = begin
    N == 1 || throw(ArgumentError("Output of multi-dimensional arrays of" 
    *" arrays to HDF5 is not supported"))
    # TODO: Support vectors of multi-dimensional arrays
    M == 1 || throw(ArgumentError("Output of vectors of multi-dimensional" 
    *" arrays to HDF5 is not supported"))
    output["$i/flattened_data"] = flatview(v)
    output["$i/cumulative_length"] = _cumulative_length(v)
    setdatatype!(output.data_store["$i"], typeof(v))
    nothing
end

# write Vector{<:RDWaveforms}
Base.setindex!(output::LHDataStore, v::AbstractVector{<:RDWaveform}, 
i::AbstractString) = begin
    output[i] = to_table(v)
    nothing
end

# write NamedTuple 
Base.setindex!(output::LHDataStore, v::NamedTuple, i::AbstractString, 
DT::DataType=typeof(v)) = begin
    for k in keys(v)
        output[i*"/$(String(k))"] = v[k]
    end
    setdatatype!(output.data_store[i], DT)
    nothing
end

# write Bool Array
Base.setindex!(output::LHDataStore, v::Union{Bool, AbstractArray{Bool}}, 
i::AbstractString) = begin
   data = UInt8.(v)
   output[i, typeof(v)] = data
   nothing 
end

# write Histogram
Base.setindex!(output::LHDataStore, v::Histogram, i::AbstractString) = begin
    output[i, typeof(v)] = _histogram_to_nt(v)
    nothing
end

# write String
Base.setindex!(output::LHDataStore, v::AbstractString, i::AbstractString
) = begin 
    output.data_store[i] = v
    setdatatype!(output.data_store[i], typeof(v))
    nothing
end

# write Table
Base.setindex!(output::LHDataStore, v, i::AbstractString, 
DT::DataType=typeof(v)) = begin
    Tables.istable(v) || throw(ArgumentError("Value to write, of type "
    *"$(typeof(v)),is not a table"))
    cols = Tables.columns(v)
    output[i, typeof(v)] = Tables.columns(v)
    nothing
end
