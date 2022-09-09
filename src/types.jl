export LHDataStore, LH5Array

"""
    LH5Array{T, N} <: AbstractArray{T, N}

Array wrapper for HDF5.Datasets following the LEGEND data format as in ".lh5"
files. 

An `LH5Array` contains a HDF5.Dataset `file` and Unitful.Unitlike `units` as 
returned by [`getunits`](@ref)`(file)`. `getindex` and `append!` are supported.
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
    units::Unitful.Unitlike
end

const V{T} = AbstractVector{<:T}
const VV{T} = AbstractVector{<:AbstractVector{T}}
const LVV{T, M, N, L} = ArrayOfSimilarArrays{T, M, N, L, LH5Array{T, L}}
const MVV{T, M, N, L} = ArrayOfSimilarArrays{T, M, N, L, Array{T, L}}
const RDW{T, U, N} = ArrayOfRDWaveforms{T, U, N, <:VV{T}, MVV{U, 1, 1, 2}}
const LHRDW{T, U, N} = ArrayOfRDWaveforms{T, U, N, <:VV{T}, LVV{U, 1, 1, 2}}
const CHUNK_SIZE = 10_000
const LHIndexType = Union{Colon, AbstractRange{Int}}

LH5Array{T}(f::HDF5.Dataset, u::Unitful.Unitlike) where {T} = begin
    LH5Array{T, ndims(f)}(f, u)
end
LH5Array{T, N}(f::HDF5.Dataset) where {T, N} = LH5Array{T, N}(f, getunits(f))
LH5Array(f::Union{HDF5.Dataset, HDF5.H5DataStore}) = LH5Array(f, getdatatype(f))
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:AbstractArray{<:RealQuantity}})

return a `LH5Array` with dimensions equal to that of `ds` and element type 
equal to `eltype(ds) * u`
"""
LH5Array(ds::HDF5.Dataset, ::Type{<:AbstractArray{<:RealQuantity}}) = begin
    u = getunits(ds)
    ET = (u == NoUnits) ? eltype(ds) : typeof(eltype(ds)(0) * u)
    LH5Array{ET}(ds, u)
end
"""
    LH5Array(ds::HDF5.H5DataStore, ::Type{<:AbstractArrayOfSimilarArrays{<:RealQuantity}})

return an `ArraysOfSimilarArrays` where the field `data` is a `LH5Array` 
(see [`ArraysOfSimilarArrays`](@ref))
"""
LH5Array(ds::HDF5.Dataset, 
::Type{<:AbstractArrayOfSimilarArrays{<:RealQuantity}}) = begin
    nestedview(LH5Array(ds, AbstractArray{<:RealQuantity}))
end
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:NamedTuple{T}}) where T

return a `NamedTuple` where each `field` is the output of `LH5Array` applied to it.
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:NamedTuple{T}}) where T = begin
    NamedTuple{T}(LH5Array.([ds[k] for k in String.(T)]))
end
"""
    LH5Array(ds::HDF5.DataStore, ::Type{<:TypedTables.Table{<:NamedTuple{(T)}}}) where T

return a `Table` where each column is the output of `LH5Array` applied to it.
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:TypedTables.Table{<:NamedTuple{(T)}}}
) where T = begin
    TypedTables.Table(LH5Array(ds, NamedTuple{T}))
end
"""
    LH5Array(ds::HDF5.DataStore, ::Type{<:AbstractVector{<:RDWaveform}})

return an `ArrayOfRDWaveforms` where `value` is a `LH5Array` (see ) 
"""
LH5Array(ds::HDF5.H5DataStore, ::Type{<:AbstractVector{<:RDWaveform}}) = begin
    tbl = LH5Array(ds, TypedTables.Table{<:NamedTuple{(:t0, :dt, :values)}})
    from_table(tbl, AbstractVector{<:RDWaveform})
end
"""
    LH5Array(ds::HDF5.DataStore, ::Type{<:AbstractVector{<:AbstractVector{<:RealQuantity}}})

return a `VectorOfVectors` object where `data` is a `LH5Array` 
(see [`VectorOfArrays`](@ref))
"""
LH5Array(ds::HDF5.H5DataStore, 
::Type{<:AbstractVector{<:AbstractVector{<:RealQuantity}}}) = begin
    data = LH5Array(ds["flattened_data"])
    cumulen = LH5Array(ds["cumulative_length"])[:]
    VectorOfVectors(data, _element_ptrs(cumulen))
end

Base.getindex(lh::LH5Array{T, N}, idxs::Vararg{HDF5.IndexType, N}
) where {T, N} = begin
    dtype = HDF5.datatype(lh.file)
    val = HDF5.generic_read(lh.file, dtype, T, idxs...)
    close(dtype)
    return val
end

Base.getindex(lh::LVV{T, M}, idxs::LHIndexType...) where {T, M} = begin
    indices = (ArraysOfArrays._ncolons(Val{M}())..., idxs...)
    ArrayOfSimilarArrays{T, M}(lh.data[indices...])
end

Base.getindex(lh::LHRDW, idxs::LHIndexType...) = 
    ArrayOfRDWaveforms((lh.time[idxs...], lh.value[idxs...]))

_inv_element_ptrs(el_ptr::AbstractVector{<:Int}) = UInt32.(el_ptr .- 1)[2:end]
Base.isassigned(lh::LH5Array, i::Int) = 1 <= i <= length(lh)
Base.size(lh::LH5Array) = size(lh.file)
Base.elsize(::LH5Array{T}) where T = Base.elsize(Array{T})

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

Base.append!(dest::VectorOfVectors{T, LH5Array{T, 1}}, src::VectorOfVectors
) where T = begin
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

Base.append!(dest::LHRDW{T1, U}, src::RDW{T2, V}
) where {T1<:RealQuantity, T2<:RealQuantity, U<:RealQuantity, 
V<:RealQuantity} = begin
    # first append values to on-disk array
    StructArrays.foreachfield(append!, dest, src)

    # and then append time information to on disk array 
    src_t0 = first.(src.time)
    src_dt = step.(src.time)
    dset_t0 = parent(dest.value.data.file)["t0"]
    dset_dt = parent(dest.value.data.file)["dt"]
    append!(LH5Array(dset_t0), src_t0)
    append!(LH5Array(dset_dt), src_dt)
    dest
end

"""
    LHDataStore 

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
mutable struct LHDataStore
    data_store::HDF5.H5DataStore
end

"""
    LHDataStore(f::AbstractString)
    LHDataStore(f::HDF5.DataStore)

create a `LHDataStore` object, where `data_store` is an HDF5.file created 
at path `f` with mode `cw`. If a `HDF5.File` at `f` already exists, the data will 
be preserved. (see [`HDF5`](@ref))
"""
LHDataStore(f::AbstractString) = LHDataStore(HDF5.h5open(f, "cw"))

"""
    LHDataStore(f::Funtion, s::AbstractString)

Apply the function `f` to the result of `LHDataStore(s)` and close the 
resulting `LHDataStore` object. Use with a `do` block:

#Example

    LHDataStore(s) do f
        f["key"] = [1, 2, 3, 4, 5]
    end
"""

LHDataStore(f::Function, s::AbstractString) = begin
    lhds = LHDataStore(s)
    try
       f(lhds) 
    finally
        close(lhds)
    end
end

Base.close(f::LHDataStore) = close(f.data_store)
Base.keys(lh::LHDataStore) = keys(lh.data_store)
Base.getindex(lh::LHDataStore, i::AbstractString) = LH5Array(lh.data_store[i])

# write AbstractArray{<:Real} or <:Real
Base.setindex!(output::LHDataStore, v::Union{T, AbstractArray{T}}, 
i::AbstractString, DT::DataType=typeof(v)) where {T<:Real} = begin
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

# write AbstractArray{<:Quantity} or <:Quantity
Base.setindex!(output::LHDataStore, 
v::Union{T, AbstractArray{T}}, i::AbstractString, DT::DataType=typeof(v)
) where {T<:Quantity}  = begin
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
Base.setindex!(output::LHDataStore, v::NamedTuple, i::AbstractString) = begin
    for k in keys(v)
        output[i*"/$(String(k))"] = v[k]
    end
    nothing
end

# write Table
Base.setindex!(output::LHDataStore, v, i::AbstractString, 
DT::DataType=typeof(v)) = begin
    Tables.istable(v) || throw(ArgumentError("Value to write, of type "
    *"$(typeof(x)),is not a table"))
    cols = Tables.columns(v)
    output[i] = cols
    setdatatype!(output.data_store[i], DT)
    nothing
end

Base.show(io::IO, x::LHDataStore) = HDF5.show_tree(io, x.data_store)