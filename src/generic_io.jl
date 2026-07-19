# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT


# TODO: LegendHDF5File, LegendHDF5Input, LegendHDF5Output
# with Base.read/write, open/close, etc., and atomic file names.


const datatype_regexp = r"""^(([A-Za-z_]*)(<([0-9,]*)>)?)(\{(.*)\})?$"""

function _eldatatype_from_string(s::Union{Nothing,AbstractString})
    if isnothing(s) || isempty(s)
        RealQuantity
    else
        datatype_from_string(s)
    end
end

function _sort_datatype_fields(s::AbstractString)
    m = match(datatype_regexp, s)
    m isa Nothing && throw(ArgumentError("Invalid datatype string \"$s\""))
    kind = m[1]
    field_string = m[6]
    if kind != "struct" && kind != "table"
        return s
    else
        fields = split(field_string, ",")
        sorted_fields = sort(fields)
        return "$kind{"*join(sorted_fields, ",")*"}"
    end
end

_ndims(x) = ndims(x)
_ndims(::Type{<:AbstractArray{<:Any,N}}) where {N} = N

# Extract type parameters via dispatch, returning nothing where the given
# type does not determine them:

_fixed_ndims(::Type{<:AbstractArray{<:Any,N}}) where {N} = @isdefined(N) ? N : nothing
_fixed_ndims(::Type) = nothing

_inner_ntuple_length(::Type{<:AbstractArray{<:NTuple{L,Any}}}) where {L} = L
_inner_ntuple_length(::Type) = nothing

_inner_staticvector_length(::Type{<:AbstractArray{<:StaticVector{L}}}) where {L} = L
_inner_staticvector_length(::Type) = nothing

_enum_eltype(::Type{<:AbstractArray{E}}) where {E<:Enum} =
    @isdefined(E) ? E : throw(ArgumentError("Enum element type not determined by array type"))

_namedtuple_type(members::AbstractVector{<:AbstractString}) = NamedTuple{(Symbol.(members)...,)}


function datatype_from_string(s::AbstractString)
    s_sorted_fields = _sort_datatype_fields(s)
    if s == "real"
        RealQuantity
    elseif s == "bool"
        Bool
    elseif s == "string"
        String
    elseif s == "symbol"
        Symbol
    elseif haskey(_datatype_dict, s_sorted_fields)
        _datatype_dict[s_sorted_fields]
    else
        m = match(datatype_regexp, s)
        m isa Nothing && throw(ErrorException("Invalid datatype string \"$s\""))
        tp = m[2]
        content = m[6]
        if tp == "struct"
            _namedtuple_type(split(content, ","; keepempty = false))
        elseif tp == "table"
            TypedTables.Table{<:_namedtuple_type(split(content, ","; keepempty = false))}
        elseif tp == "ntuple"
            T = _eldatatype_from_string(content)
            (NTuple{N,<:T} where N)
        elseif tp == "enum"
            throw(ErrorException("Enum datatype \"$s\" is not registered," *
                " use LegendHDF5IO.register_datatype! to register enum types"))
        else
            dims = parse.(Int, split(m[4], ","))
            eltp = content
            T = _eldatatype_from_string(eltp)
            if tp == "array_of_equalsized_arrays"
                length(dims) == 2 || throw(ErrorException("Invalid dims $dims for datatype \"$tp\""))
                N = dims[1]; M = dims[2]
                # T isa Array || throw(ErrorException("Datatype \"$tp\" with \"array\" nested in \"array_of_equalsized_arrays\" currently not supported"))
                AbstractArrayOfSimilarArrays{<:T,M,N}
            elseif tp == "array_of_encoded_arrays"
                length(dims) == 2 || throw(ErrorException("Invalid dims $dims for datatype \"$tp\""))
                N = dims[1]; M = dims[2]
                N == 1 || throw(ErrorException("Only one-dimensional arrays of encoded arrays are supported"))
                VectorOfEncodedArrays{<:T,M}
            elseif tp == "array_of_encoded_equalsized_arrays"
                length(dims) == 2 || throw(ErrorException("Invalid dims $dims for datatype \"$tp\""))
                N = dims[1]; M = dims[2]
                N == 1 || throw(ErrorException("Only one-dimensional arrays of encoded arrays are supported"))
                VectorOfEncodedSimilarArrays{<:T,M}
            elseif tp == "fixedsize_array"
                length(dims) == 1 || throw(ErrorException("Invalid dims $dims for datatype \"$tp\""))
                N = dims[1]
                N == 1  || throw(ErrorException("Datatype fixedsize_array with $dims dims currently not supported\"$tp\""))
                # T <: RealQuantity || throw(ErrorException("Element type \"$eltp\" in datatype \"$tp\" currently not supported"))
                StaticArray{Tuple{L},<:T,1} where {L}
            elseif tp == "array"
                length(dims) == 1 || throw(ErrorException("Invalid dims $dims for datatype \"$tp\""))
                N = dims[1]
                _array_type(Array{T, N})
            elseif tp == "encoded_array"
                length(dims) == 1 || throw(ErrorException("Invalid dims $dims for datatype \"$tp\""))
                N = dims[1]
                EncodedArray{<:T,N}
            elseif tp == "histogram"
                length(dims) == 1 || throw(ErrorException("Invalid dims $dims for datatype \"$tp\""))
                N = dims[1]
                Histogram{<:T, N}
            else
                throw(ErrorException("Unknown datatype \"$tp\""))
            end
        end
    end
end

function _array_type(::Type{Array{T, N}}) where {T, N}
    isconcretetype(T) ? AbstractArray{T, N} : AbstractArray{<:T, N}
end

"""
    LegendHDF5IO.register_datatype!(name::AbstractString, ::Type{T})
    LegendHDF5IO.register_datatype!(::Type{T})

Register the LH5 datatype string `name` for type `T`, for reading and
writing. Without `name`, the canonical datatype string of `T` is used
(e.g. for `Enum` types).

To support reading and writing values of type `T`, also define

```julia
LegendHDF5IO.LH5Array(ds::HDF5.Dataset, ::Type{<:T})
LegendHDF5IO.create_entry(parent::LHDataStore, name::AbstractString, x::T; kwargs...)
```
"""
function register_datatype!(name::AbstractString, ::Type{T}) where {T}
    _datatype_dict[_sort_datatype_fields(name)] = T
    filter!(p -> p.first !== T, _datatype_names)
    push!(_datatype_names, T => name)
    nothing
end

register_datatype!(::Type{T}) where {T} = register_datatype!(datatype_to_string(T), T)

function _inner_datatype_to_string(::Type{T}) where T
    s = datatype_to_string(T)
    isempty(s) ? "" : "{$s}"
end


datatype_to_string(::Type{<:RealQuantity}) = "real"

datatype_to_string(::Type{Bool}) = "bool"

datatype_to_string(T::Type{<:Enum{U}}) where {U} = "enum{"*join(broadcast(x -> "$(string(x))=$(U(x))", instances(T)), ",")*"}"

datatype_to_string(::Type{<:AbstractString}) = "string"

datatype_to_string(::Type{<:Symbol}) = "symbol"

datatype_to_string(T::Type{<:NTuple{N,Any}}) where {N} = "ntuple$(_inner_datatype_to_string(eltype(T)))"

datatype_to_string(::Type{<:AbstractArray{T,N}}) where {T,N} =
    "array<$N>$(_inner_datatype_to_string(T))"

datatype_to_string(::Type{<:EncodedArray{T,N}}) where {T,N} =
    "encoded_array<$N>$(_inner_datatype_to_string(T))"

datatype_to_string(::Type{<:StaticArray{TPL,T,N}}) where {TPL,T<:RealQuantity,N} =
    "fixedsize_array<$N>$(_inner_datatype_to_string(T))"

datatype_to_string(::Type{<:ArrayOfSimilarArrays{T,M,N}}) where {T,M,N} =
    "array_of_equalsized_arrays<$N,$M>$(_inner_datatype_to_string(T))"

datatype_to_string(::Type{<:VectorOfEncodedArrays{T,N}}) where {T,N} =
    "array_of_encoded_arrays<1,$N>$(_inner_datatype_to_string(T))"

datatype_to_string(::Type{<:VectorOfEncodedSimilarArrays{T,M}}) where {T,M} =
    "array_of_encoded_equalsized_arrays<1,$M>$(_inner_datatype_to_string(T))"

datatype_to_string(::Type{<:NamedTuple{K}}) where K = "struct{$(join(K,","))}"

# ToDo: Make this more generic:
datatype_to_string(::Type{<:TypedTables.Table{<:NamedTuple{K}}}) where K = "table{$(join(K,","))}"
datatype_to_string(::Type{<:StructArrays.StructArray{<:NamedTuple{K}}}) where K = "table{$(join(K,","))}"

datatype_to_string(::Type{<:Histogram{T, N}}) where {T, N} =
    "histogram<$N>$(_inner_datatype_to_string(T))"

# Fallback for types registered via register_datatype!:
function datatype_to_string(::Type{T}) where {T}
    matches = filter(p -> T <: p.first, _datatype_names)
    isempty(matches) && throw(ArgumentError("No LH5 datatype registered for type $T"))
    reduce((a, b) -> b.first <: a.first ? b : a, matches).second
end

function _cumulative_length(A::VectorOfArrays)
    elem_ptr = ArraysOfArrays.internal_element_ptr(A)
    elem_ptr[(firstindex(elem_ptr) + 1):end] .- first(elem_ptr)
end

_cumulative_length(A::AbstractVector{<:AbstractArray}) = cumsum(length.(A))

function _element_ptrs(clen::Vector{<:Integer})
    vcat([1], Int.(clen) .+ 1)
end


function hasattribute(
    obj::Union{HDF5.Dataset, HDF5.H5DataStore}, key::Symbol
)
    key_str = String(key)
    attributes = HDF5.attributes(obj)
    haskey(attributes, key_str)
end


function getattribute(
    obj::Union{HDF5.Dataset, HDF5.H5DataStore}, key::Symbol, ::Type{T}
) where {T<:Union{AbstractString,Real}}
    # Close the attribute handle explicitly: a lingering open handle keeps
    # the attribute alive, so deleting and re-creating it under the same
    # name (setdatatype!) would leave readers seeing the old value.
    attr = HDF5.attributes(obj)[String(key)]
    x = try
        read(attr)
    finally
        close(attr)
    end
    x isa T ? x : convert(T, x)
end

function getattribute(
    obj::Union{HDF5.Dataset, HDF5.H5DataStore}, key::Symbol, default_value::T
) where {T<:Union{AbstractString,Real}}
    if hasattribute(obj, key)
        getattribute(obj, key, T)
    else
        default_value
    end
end


function setattribute!(
    obj::Union{HDF5.Dataset, HDF5.Group}, key::Symbol,
    value::Real
)
    HDF5.attributes(obj)[String(key)] = value
    nothing
end


function setattribute!(
    obj::Union{HDF5.Dataset, HDF5.Group}, key::Symbol,
    value::AbstractString
)
    # Write variable-length string for h5py compatibility (see https://github.com/h5py/h5py/issues/585).
    s_arr = Array{String,0}(undef)
    s_arr[] = convert(String, value)
    HDF5.attributes(obj)[String(key)] = s_arr
    nothing
end


LegendDataTypes.getunits(dset::HDF5.Dataset) = units_from_string(getattribute(dset, :units, ""))

function LegendDataTypes.setunits!(dset::HDF5.Dataset, units::Unitful.Unitlike)
    setattribute!(dset, :units, units_to_string(units))
end


default_datatype(dset::HDF5.Dataset) = AbstractArray{<:RealQuantity,length(size(dset))}
default_datatype(df::HDF5.H5DataStore) = NamedTuple{(Symbol.(keys(df))...,)}

function getdatatype(input::Union{HDF5.Dataset, HDF5.H5DataStore})
    dtstr = getattribute(input, :datatype, "")
    isempty(dtstr) ? default_datatype(input) : datatype_from_string(dtstr)
end

function setdatatype!(output::Union{HDF5.Dataset, HDF5.H5DataStore}, datatype::Type)
    dtstr = datatype_to_string(datatype)
    hasattribute(output, :datatype) && HDF5.delete_attribute(output, "datatype")
    setattribute!(output, :datatype, dtstr)
end


function _getcontent_impl(dset::HDF5.Dataset, idxs::NTuple{N,Colon}, axs::NTuple{N}) where {N}
    read(dset)
end

function _getcontent_impl(dset::HDF5.Dataset, idxs::Tuple{Integer}, axs::NTuple{0})
    idxs == (1,) || Base.throw_boundserror(dset, idxs)
    read(dset)
end

function _getcontent_impl(dset::HDF5.Dataset, idxs::NTuple{N,Any}, axs::NTuple{N}) where {N}
    # HDF5.generic_read doesn't like empty indices like `(Base.OneTo(0),)`:
    canonical_idxs = if idxs == axs && all(isempty, idxs)
        ()
    else
        Base.to_indices(dset, axs, idxs)
    end
    isinbounds = Base.checkbounds_indices(Bool, axs, idxs)
    isinbounds || Base.throw_boundserror(dset, idxs)
    dset[canonical_idxs...]
end

function getcontent(dset::HDF5.Dataset, idxs::Tuple = axes(dset))
    _getcontent_impl(dset, idxs, axes(dset))
end


"""
    readdata(input::Union{HDF5.Dataset, HDF5.H5DataStore}, name::AbstractString)
    readdata(input, name, datatype::Type)

Read the value stored under `name` from `input`, eagerly.

Reads via [`LH5Array`](@ref) and materializes the result in memory. The
value type is normally determined by the "datatype" attribute; the
three-argument form overrides it.
"""
function LegendDataTypes.readdata(
    input::Union{HDF5.Dataset, HDF5.H5DataStore}, name::AbstractString
)
    _materialize(LH5Array(input[name]))
end

function LegendDataTypes.readdata(
    input::Union{HDF5.Dataset, HDF5.H5DataStore}, name::AbstractString,
    DT::Type
)
    _materialize(LH5Array(input[name], DT))
end


"""
    writedata(output::HDF5.H5DataStore, name::AbstractString, x,
        fulldatatype::DataType = typeof(x))

Write the value `x` under `name` to `output`, via `create_entry`.

Passing `fulldatatype` overrides the "datatype" attribute written for `x`;
passing `Nothing` suppresses it.
"""
function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString, x,
    fulldatatype::DataType = typeof(x)
)
    create_entry(LHDataStore(output, false), name, x)
    if fulldatatype == Nothing
        hasattribute(output[name], :datatype) && HDF5.delete_attribute(output[name], "datatype")
    elseif fulldatatype != typeof(x)
        setdatatype!(output[name], fulldatatype)
    end
    nothing
end


function _flatview_of_array_of_ntuple(A::AbstractArray{<:NTuple{L,Any}, N}) where {L,N}
    T = eltype(eltype(A))
    reshape(reinterpret(T, A), L, size(A)...)
end

function _flatview_to_array_of_ntuple(A::AbstractArray{T,N}, TPL::Type{NTuple{L,T}}) where {T,N,L}
    size_A = size(A)
    sz_out = Base.tail(size_A)
    N_out = length(sz_out)
    size_A[1] == L || throw(DimensionMismatch("Length $L of NTuple type does not match first dimension of array of size $size_A"))
    tmp = reshape(reinterpret(TPL, A), sz_out...)
    convert(Array{NTuple{L,T},N_out}, tmp)
end
