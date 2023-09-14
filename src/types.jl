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

const LH5AoSA{T, M, N, L} = ArrayOfSimilarArrays{T, M, N, L, LH5Array{T, L}}
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
    D = _ndims(A)
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

return a `String`.
"""
LH5Array(ds::HDF5.Dataset, ::Type{<:String}) = read(ds)
"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:Symbol})

return a `Symbol`.
"""
LH5Array(ds::HDF5.Dataset, ::Type{<:Symbol}) = Symbol(read(ds))

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
Base.getindex(lh::LHDataStore, i::Any) = lh[string(i)]

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

Base.setindex!(output::LHDataStore, v, i, chunk_size=nothing) = begin
    output.usechunks = !isnothing(chunk_size)
    _setindex!(output, v, i, chunk_size)
end

# write <:Real
_setindex!(output::LHDataStore, v::T, i::AbstractString, args...
    ) where {T<:Real} = begin

    output.data_store[i] = v
    setdatatype!(output.data_store[i], T)
end

# write <:Quantity
_setindex!(output::LHDataStore, v::T, i::AbstractString, args...
    ) where {T<:Quantity} = begin

    _setindex!(output, ustrip(v), i)
    setunits!(output.data_store[i], unit(T))
end

# write AbstractArray{<:Real}
_setindex!(output::LHDataStore, v::AbstractArray{T}, i::AbstractString,
    chunk_size::Union{Nothing, Int}=nothing) where {T<:Real} = begin

    dtype = HDF5.datatype(T)
    ds = if isnothing(chunk_size)
        HDF5.create_dataset(output.data_store, i, dtype, size(v))
    else
        @assert chunk_size > 0 "chunk size has to be greater than zero"
        sizev = size(v)
        dspace = (sizev, (sizev[begin:end-1]..., -1))
        chunk = (sizev[begin:end-1]..., chunk_size)
        HDF5.create_dataset(output.data_store, i, dtype, dspace; chunk=chunk)
    end
    try
        HDF5.write_dataset(ds, dtype, Array(v))
        setdatatype!(ds, typeof(v))
    catch exc
        HDF5.delete_object(ds)
        rethrow(exc)
    finally
        close(ds)
        close(dtype)
    end
end

# write AbstractArray{<:Quantity}
_setindex!(output::LHDataStore, v::AbstractArray{T}, i::AbstractString,
    args...) where {T<:Quantity}  = begin

    _setindex!(output, ustrip(v), i, args...)
    setdatatype!(output.data_store[i], typeof(v))
    setunits!(output.data_store[i], unit(T))
end

# write ArrayOfSimilarArrays{<:RealQuantity}
_setindex!(output::LHDataStore, v::ArrayOfSimilarArrays{T}, i::AbstractString,
    args...) where {T<:RealQuantity} = begin

    _setindex!(output, flatview(v), i, args...)
    setdatatype!(output.data_store[i], typeof(v))
end

# write VectorOfVectors{<:RealQuantity}
_setindex!(output::LHDataStore, v::AbstractArray{<:AbstractArray{T, M}, N}, 
    i::AbstractString, args...) where {T<:RealQuantity, M, N} = begin

    N == 1 || throw(ArgumentError("Output of multi-dimensional arrays of" 
    *" arrays to HDF5 is not supported"))
    # TODO: Support vectors of multi-dimensional arrays
    M == 1 || throw(ArgumentError("Output of vectors of multi-dimensional" 
    *" arrays to HDF5 is not supported"))
    _setindex!(output, flatview(v), "$i/flattened_data", args...)
    _setindex!(output, _cumulative_length(v), "$i/cumulative_length", args...)
    setdatatype!(output.data_store[i], typeof(v))
end

# write Vector{<:RDWaveforms}
_setindex!(output::LHDataStore, v::AbstractVector{<:RDWaveform{T, U}}, 
    i::AbstractString, args...) where {T<:RealQuantity, U<:RealQuantity} = 
    _setindex!(output, to_table(v), i, args...)

# write NamedTuple 
_setindex!(output::LHDataStore, v::NamedTuple, i::AbstractString, args...
    ) = begin

    for k in keys(v)
        _setindex!(output, v[k], "$i/$(String(k))", args...)
    end
    setdatatype!(output.data_store[i], typeof(v))
end

# write Histogram
_setindex!(output::LHDataStore, v::Histogram, i::AbstractString, args...
    ) = begin

    _setindex!(output, _histogram_to_nt(v), i, args...)
    setdatatype!(output.data_store[i], typeof(v))
end

# write String
_setindex!(output::LHDataStore, v::AbstractString, i::AbstractString, args...
    ) = begin 

    output.data_store[i] = v
    setdatatype!(output.data_store[i], typeof(v))
end

# write Symbol
_setindex!(output::LHDataStore, v::Symbol, i::AbstractString, args...) = begin 
    output.data_store[i] = String(v)
    setdatatype!(output.data_store[i], typeof(v))
end

# write Table
_setindex!(output::LHDataStore, v, i::AbstractString, args...) = begin
    Tables.istable(v) || throw(ArgumentError("Value to write, of type "
    *"$(typeof(v)), is not a table"))
    _setindex!(output, Tables.columns(v), i, args...)
    setdatatype!(output.data_store[i], typeof(v))
end

"""
    lh5open(filename::AbstractString, access::AbstractString = "r")

Open a LEGEND HDF5 file and return an `LHDataStore` object.

LEGEND HDF5 files typically use the file extention ".lh5".
"""
function lh5open(filename::AbstractString, access::AbstractString = "r")
    LHDataStore(HDF5.h5open(filename, access), false)
end
export lh5open

"""
    lh5open(f, filename::AbstractString, access::AbstractString = "r")

Return f(lh5open(f, filename, access)).

Opens and closes the LEGEND HDF5 file `filename` automatically.
"""
function lh5open(f::Function, filename::AbstractString, access::AbstractString = "r")
    lhds = lh5open(filename, access)
    try
       f(lhds) 
    finally
        close(lhds)
    end
end

"""
    extend_datastore(lhd::LHDataStore, i::AbstractString, src::TypedTable.Table, 
        dest::TypedTable.Table=LH5Array(lhd.data_store[i]))

extend the Table `dest` at `lhd[i]` with columns from `src`.
"""
function extend_datastore(lhd::LHDataStore, i::AbstractString, 
    src::Table, dest::Table=LH5Array(lhd.data_store[i]))

    @assert length(dest) == length(src) "tables are not equal in length"
    tbl = Table(dest, src)
    extend_datastore(lhd, i, columns(src), columns(dest))
    HDF5.rename_attribute(lhd.data_store[i], "datatype", "datatype_old")
    HDF5.delete_attribute(lhd.data_store[i], "datatype_old")
    setdatatype!(lhd.data_store[i], typeof(tbl))
end

"""
    extend_datastore(lhd::LHDataStore, i::AbstractString, src::NamedTuple, 
        dest::NamedTuple=LH5Array(lhd.data_store[i]))

extend the NamedTuple `dest` at `lhd[i]` with elements from `src`.
"""
function extend_datastore(lhd::LHDataStore, i::AbstractString, src::NamedTuple,
    dest::NamedTuple=LH5Array(lhd.data_store[i]))

    new_nt = (;dest..., src...)
    for k in keys(src)
        lhd[joinpath(i, "$k")] = src[k]
    end
    HDF5.rename_attribute(lhd.data_store[i], "datatype", "datatype_old")
    HDF5.delete_attribute(lhd.data_store[i], "datatype_old")
    setdatatype!(lhd.data_store[i], typeof(new_nt))
end
export extend_datastore

"""
    reduce_datastore(lhd::LHDataStore, i::AbstractString)

remove the dataset `lhd[i]` and adjust the datatype of the parent if necessary. 
Currently supported are elements of `NamedTuple`, `TypedTable.Table` or 
`HDF5.Group`. 
"""
function reduce_datastore(lhd::LHDataStore, i::AbstractString)
    parent, child = splitdir(i)
    if isempty(parent)
        HDF5.delete_object(lhd.data_store[i])
    else
        _reduce_datastore(lhd, lhd[parent], parent, child)
    end
end
export reduce_datastore

function _reduce_datastore(lhd::LHDataStore, nt::NamedTuple, 
    parent::AbstractString, child::AbstractString)

    if hasattribute(lhd.data_store[parent], :datatype)
        newkeys = setdiff(keys(nt), (Symbol(child),))
        isempty(newkeys) && throw("Empty object at $parent not allowed")
        new_nt = (;[k => nt[k] for k in newkeys]...)
        HDF5.rename_attribute(lhd.data_store[parent], "datatype", "datatype_old")
        HDF5.delete_attribute(lhd.data_store[parent], "datatype_old")
        setdatatype!(lhd.data_store[parent], typeof(new_nt))
    end
    HDF5.delete_object(lhd.data_store[joinpath(parent, child)])
end

function _reduce_datastore(lhd::LHDataStore, tbl::Table, parent::AbstractString, 
    child::AbstractString)

    _reduce_datastore(lhd, columns(tbl), parent, child)
    new_tbl = Table(lhd[parent])
    # adjust datatype of parent
    HDF5.rename_attribute(lhd.data_store[parent], "datatype", "datatype_old")
    HDF5.delete_attribute(lhd.data_store[parent], "datatype_old")
    setdatatype!(lhd.data_store[parent], typeof(new_tbl))
end