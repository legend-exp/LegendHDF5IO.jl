# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT


# TODO: LegendHDF5File, LegendHDF5Input, LegendHDF5Output
# with Base.read/write, open/close, etc., and atomic file names.


const default_compression = ("shuffle", (), "deflate", 3)


const datatype_regexp = r"""^(([A-Za-z_]*)(<([0-9,]*)>)?)(\{(.*)\})?$"""
const arraydims_regexp = r"""^<([0-9,]*)>$"""

function _eldatatype_from_string(s::Union{Nothing,AbstractString})
    if isnothing(s) || isempty(s)
        RealQuantity
    else
        datatype_from_string(s)
    end
end

_ndims(x) = ndims(x)
_ndims(::Type{<:AbstractArray{<:Any,N}}) where {N} = N

Base.@propagate_inbounds _tuple_droplast(x::NTuple{N,Any}, ::Val{M}) where {N,M} =
    Base.ntuple(i -> x[i], Val{N-M}())

_namedtuple_type(members::AbstractVector{<:AbstractString}) = NamedTuple{(Symbol.(members)...,)}


function datatype_from_string(s::AbstractString)
    if s == "real"
        RealQuantity
    elseif s == "bool"
        Bool
    elseif s == "string"
        String
    elseif s == "symbol"
        Symbol
    elseif haskey(_datatype_dict, s)
        _datatype_dict[s]
    else
        m = match(datatype_regexp, s)
        m isa Nothing && throw(ErrorException("Invalid datatype string \"$s\""))
        tp = m[2]
        content = m[6]
        if tp == "struct"
            _namedtuple_type(split(content, ","))
        elseif tp == "table"
            TypedTables.Table{<:_namedtuple_type(split(content, ","))}
        elseif tp == "ntuple"
            T = _eldatatype_from_string(content)
            (NTuple{N,<:T} where N)
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
                N != 2 || throw(ErrorException("Only one-dimensional arrays of encoded arrays are supported"))
                VectorOfEncodedArrays{<:T,dims[1]}
            elseif tp == "array_of_encoded_equalsized_arrays"
                length(dims) == 2 || throw(ErrorException("Invalid dims $dims for datatype \"$tp\""))
                N = dims[1]; M = dims[2]
                N != 2 || throw(ErrorException("Only one-dimensional arrays of encoded arrays are supported"))
                VectorOfEncodedSimilarArrays{<:T,dims[1]}
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
    AbstractArray{<:T, N}
end

function _inner_datatype_to_string(::Type{T}) where T
    s = datatype_to_string(T)
    isempty(s) ? "" : "{$s}"
end


datatype_to_string(::Type{<:RealQuantity}) = "real"

datatype_to_string(::Type{Bool}) = "bool"

datatype_to_string(T::Type{<:Enum{U}}) where {U} = "enum{"*join(broadcast(x -> "$(string(x))=$(U(x))", instances.(T)), ",")*"}"

datatype_to_string(::Type{<:AbstractString}) = "string"

datatype_to_string(::Type{<:Symbol}) = "symbol"

datatype_to_string(::Type{NTuple{N,T}}) where {N,T} = "ntuple$(_inner_datatype_to_string(T))"

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

function _eltype(dset::HDF5.Dataset)
    dtype = HDF5.datatype(dset)
    try
        HDF5.hdf5_to_julia_eltype(dtype)
    finally
        close(dtype)
    end
end


function _cumulative_length(A::VectorOfArrays)
    elem_ptr = ArraysOfArrays.internal_element_ptr(A)
    elem_ptr[(firstindex(elem_ptr) + 1):end] .- first(elem_ptr)
end

function _cumulative_length(A::AbstractVector{<:AbstractArray})
    # ToDo: improve implementation
    cumsum(length.(eachindex.(A)))
end

function _element_ptrs(clen::Vector{<:Integer})
    vcat([1], Int.(clen) .+ 1)
end


# _without_units(A::Array, unit::Unitful.Unitlike) = ustrip.(uconvert.(unit, A))
# _with_units(A::Array, unit::Unitful.Unitlike) = A * unit


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
    key_str = String(key)
    attributes = HDF5.attributes(obj)
    x = read(attributes[key_str])
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
    ustr = units_to_string(units)
    # @debug "setunits!($(_infostring(dset)), \"$ustr\")"
    setattribute!(dset, :units, ustr)
end


default_datatype(dset::HDF5.Dataset) = AbstractArray{<:RealQuantity,length(size(dset))}
default_datatype(df::HDF5.H5DataStore) = NamedTuple{(Symbol.(keys(df))...,)}

_size(dset::HDF5.Dataset) = size(dset)
_size(df::HDF5.H5DataStore) = ()

# HDF5.H5DataStore
# HDF5.Dataset
_infostring(x::HDF5.Group) = "group \"$(HDF5.name(x))\""
_infostring(x::HDF5.Dataset) = "dataset \"$(HDF5.name(x))\" with size $(size(x)) of $(eltype(x))"

function getdatatype(input::Union{HDF5.Dataset, HDF5.H5DataStore})
    dtstr = getattribute(input, :datatype, "")
    dt = isempty(dtstr) ? default_datatype(input) : datatype_from_string(dtstr)
    # @debug "getdatatype($(_infostring(input))) = $dt"
    dt
end

function setdatatype!(output::Union{HDF5.Dataset, HDF5.H5DataStore}, datatype::Type)
    dtstr = datatype_to_string(datatype)
    hasattribute(output, :datatype) && HDF5.delete_attribute(output, "datatype")
    setattribute!(output, :datatype, dtstr)
end

function getinfo(dset::HDF5.Dataset)
    (
        datatype = getattribute(dset, :datatype, ""),
        units = getunits(dset),
        cumsum_length_ds = getattribute(dset, :cumsum_length_ds, ""),
    )
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


function LegendDataTypes.readdata(input::Union{HDF5.Dataset, HDF5.H5DataStore}, name::AbstractString)
    datatype = getdatatype(input[name])
    readdata(input, name, datatype)
end


function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString,
    x::Union{T,AbstractArray{T}},
    fulldatatype::DataType = typeof(x)
) where {T<:RealQuantity}
    units = unit(eltype(x))
    # @debug "name" name
    if units == NoUnits
        # @debug "without units"
        #output[name, "shuffle", (), "deflate", 3] = x
        output[name] = x
    else
        # @debug "with units"
        output[name] = ustrip.(x)
        setunits!(output[name], units)
    end
    if fulldatatype != Nothing
        setdatatype!(output[name], fulldatatype)
    end
    nothing
end


function LegendDataTypes.readdata(
    input::HDF5.H5DataStore, name::AbstractString,
    T::Type{<:Union{RealQuantity,AbstractArray}}
)
    dset = input[name]
    _ndims(dset) == _ndims(T) || throw(ArgumentError("Dataset has $(_ndims(dset)) dimensions but expected $(_ndims(T)) from datatype"))
    data = getcontent(dset)#::AT
    units = getunits(dset)
    if units == NoUnits
        data
    else
        data * units
    end
end


# Write Bool arrays as arrays of UInt8 for h5py compatibility (see
# https://github.com/h5py/h5py/issues/641). Issue may be fixed in recent
# versions of h5py (see https://github.com/h5py/h5py/pull/821).
function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString,
    x::Union{Bool,AbstractArray{Bool}},
    fulldatatype::DataType = typeof(x)
)
    data = UInt8.(x)
    output[name] = data
    if fulldatatype != Nothing
        setdatatype!(output[name], fulldatatype)
    end
    nothing
end

function LegendDataTypes.readdata(
    input::HDF5.H5DataStore, name::AbstractString,
    ::Type{<:Union{Bool,AbstractArray{<:Bool}}}
)
    dset = input[name]
    units = getunits(dset)
    units == NoUnits || throw(ErrorExceptions("Can't interpret dataset with units as Bool values"))
    data = getcontent(dset)
    # Broadcast will return BitArray, map would return Array{UInt8}:
    (x -> x > 0).(data)
end


function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString,
    x::AbstractArray{<:Enum{T}},
    fulldatatype::DataType = typeof(x)
) where {T}
    writedata(output, name, reinterpret(T, x), fulldatatype)
end

function LegendDataTypes.readdata(
    input::HDF5.H5DataStore, name::AbstractString,
    AT::Type{<:AbstractArray{<:Enum,N}}
) where N
    ET = AT.body.parameters[1].ub
    data = readdata(input, name, AbstractArray{RealQuantity,N})
    ET.(data)
end


function _flatview_of_array_of_ntuple(A::AbstractArray{NTuple{L, T}, N}) where {L,T,N}
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

_array_of_ntuple_innersize(A::AbstractArray{NTuple{L,T}}) where {L,T} = L

function _array_of_ntuple_innerconv(::Type{T}, A::AbstractArray{NTuple{L,U},N}) where {T,N,L,U}
    convert(Array{NTuple{L,T},N}, A)
end


function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString,
    x::AbstractArray{T,N},
    fulldatatype::DataType = typeof(x)
) where {L,T<:NTuple{L,RealQuantity},N}
    writedata(output, name, _flatview_of_array_of_ntuple(x), fulldatatype)
end

function LegendDataTypes.readdata(
    input::HDF5.H5DataStore, name::AbstractString,
    AT::Type{<:AbstractArray{<:NTuple}}
)
    N = length(size(input[name]))
    data = readdata(input, name, AbstractArray{RealQuantity,N})
    SV = AT.var.ub
    L = size(data, 1)
    if SV isa DataType
        L_expected = SV.parameters[1].parameters[1][1]
        L_expected == L || throw(ErrorException("Trying to read array of NTuples of length $L_expected, but inner dimension of data has length $L"))
    end
    _flatview_to_array_of_ntuple(data, NTuple{L,eltype(data)})
end


function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString,
    x::AbstractArray{T,N},
    fulldatatype::DataType = typeof(x)
) where {TPL,T<:StaticArray{TPL,<:RealQuantity},N}
    writedata(output, name, flatview(x), fulldatatype)
end

function LegendDataTypes.readdata(
    input::HDF5.H5DataStore, name::AbstractString,
    AT::Type{<:AbstractArray{<:StaticVector}}
)
    N = length(size(input[name]))
    data = readdata(input, name, AbstractArray{RealQuantity,N})
    SV = AT.var.ub
    L = size(data, 1)
    if SV isa DataType
        L_expected = SV.parameters[1].parameters[1][1]
        L_expected == L || throw(ErrorException("Trying to read array of static vectors of length $L_expected, but inner dimension of data has length $L"))
    end
    nestedview(data, SVector{L})
end


function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString,
    x::AbstractString,
    fulldatatype::DataType = typeof(x)
)
    output[name] = x
    if fulldatatype != Nothing
        setdatatype!(output[name], fulldatatype)
    end
    nothing
end

function LegendDataTypes.readdata(
    input::HDF5.H5DataStore, name::AbstractString,
    ::Type{<:String}
)
    dset = input[name]
    read(dset)#::AT
end


function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString,
    x::Symbol,
    fulldatatype::DataType = typeof(x)
)
    writedata(output, name, String(x), fulldatatype)
end

function LegendDataTypes.readdata(
    input::HDF5.H5DataStore, name::AbstractString,
    ::Type{<:Symbol}
)
    Symbol(readdata(input, name, String))
end


function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString,
    x::AbstractArray{<:AbstractArray{T,M},N},
    fulldatatype::DataType = typeof(x)
) where {T,M,N}
    N == 1 || throw(ArgumentError("Output of multi-dimensional arrays of arrays to HDF5 is not supported"))
    # ToDo: Support vectors of multi-dimensional arrays
    M ==1 || throw(ArgumentError("Output of vectors of multi-dimensional arrays to HDF5 is not supported"))
    writedata(output, "$(name)/flattened_data", flatview(x))
    writedata(output, "$(name)/cumulative_length", _cumulative_length(x))
    setdatatype!(output[name], fulldatatype)
    nothing
end

function LegendDataTypes.readdata(
    input::HDF5.H5DataStore, name::AbstractString,
    AT::Type{<:AbstractArray{<:AbstractArray}}
)
    data = readdata(input, "$name/flattened_data")
    clen = readdata(input, "$name/cumulative_length")
    data_vec = VectorOfVectors(data, _element_ptrs(clen))
end



function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString,
    x::VectorOfEncodedArrays{T,1,C},
    fulldatatype::DataType = typeof(x)
) where {T,C}
    codec = x.codec

    writedata(output, "$name/encoded_data", x.encoded)
    writedata(output, "$name/decoded_size", x.innersizes)
 
    dset = output[name]

    codec_name = LegendDataTypes.array_codecs[C]
    setattribute!(dset, :codec, String(codec_name))
    write_to_properties!(setattribute!, dset, codec)

    setdatatype!(dset, fulldatatype)
    nothing
end

function LegendDataTypes.readdata(
    input::HDF5.H5DataStore, name::AbstractString,
    AT::Type{<:VectorOfEncodedArrays}
)
    data_vec = readdata(input, "$name/encoded_data")
    size_vec_in = readdata(input, "$name/decoded_size")
    N = _array_of_ntuple_innersize(size_vec_in)
    size_vec = _array_of_ntuple_innerconv(Int, size_vec_in)

    SV = AT.var.ub
    if SV isa DataType
        N_expected = SV.parameters[1].parameters[1][1]
        N_expected == N || throw(ErrorException("Trying to read a vector of encoded arrays with $N_expected dimensions, but data indicates $N dimensions"))
    end

    dset = input[name]

    codec_name = Symbol(getattribute(dset, :codec, String))
    C = LegendDataTypes.array_codecs[codec_name]
    codec = read_from_properties(getattribute, dset, C)

    # ToDo: How to avoid hardcoding T?
    T = Int32

    return VectorOfEncodedArrays{T}(codec, size_vec, data_vec)
end


function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString,
    x::VectorOfEncodedSimilarArrays{T,1,C},
    fulldatatype::DataType = typeof(x)
) where {T,C}
    codec = x.codec

    # Limit to vector for encoded vectors for now:
    innerlen = only(x.innersize)

    writedata(output, "$name/encoded_data", x.encoded)
    writedata(output, "$name/decoded_size", innerlen)
 
    dset = output[name]

    codec_name = LegendDataTypes.array_codecs[C]
    setattribute!(dset, :codec, String(codec_name))
    write_to_properties!(setattribute!, dset, codec)

    setdatatype!(dset, fulldatatype)
    nothing
end

function LegendDataTypes.readdata(
    input::HDF5.H5DataStore, name::AbstractString,
    AT::Type{<:VectorOfEncodedSimilarArrays}
)
    data_vec = readdata(input, "$name/encoded_data")

    # Limit to vector for encoded vectors for now:
    innerlen = readdata(input, "$name/decoded_size")
    @assert innerlen isa Integer
    innersz = (innerlen,)
    N = 1

    dset = input[name]

    codec_name = Symbol(getattribute(dset, :codec, String))
    C = LegendDataTypes.array_codecs[codec_name]
    codec = read_from_properties(getattribute, dset, C)

    # ToDo: How to avoid hardcoding T?
    T = Int32

    VectorOfEncodedSimilarArrays{T}(codec, innersz, data_vec)
end


function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString,
    x::AbstractArrayOfSimilarArrays{T,M,N},
    fulldatatype::DataType = typeof(x)
) where {T,M,N}
    writedata(output, name, flatview(x), fulldatatype)
    nothing
end

# Hack:
function LegendDataTypes.readdata(
    input::HDF5.H5DataStore, name::AbstractString,
    AT::Type{<:AbstractVectorOfSimilarVectors}
)
    nestedview(readdata(input, name, AbstractArray{<:RealQuantity,2}))
end


function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString,
    x::NamedTuple,
    fulldatatype::DataType = typeof(x)
)
    for k in keys(x)
        y = x[k]
        writedata(output, "$name/$k", y, typeof(y))
    end
    setdatatype!(output[name], fulldatatype)
    nothing
end

function LegendDataTypes.readdata(
    input::HDF5.H5DataStore, name::AbstractString,
    AT::Type{<:NamedTuple}
)
    syms = AT.body.parameters[1]
    NamedTuple{syms}(map(k -> readdata(input, "$name/$k"), syms))
end


function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString,
    x::Any,
    fulldatatype::DataType = typeof(x)
)
    # @debug("writedata(\"$(HDF5.name(output))\", \"$name\", $fulldatatype")
    Tables.istable(x) || throw(ArgumentError("Value to write, of type $(typeof(x)),is not a table"))
    cols = Tables.columns(x)
    writedata(output, name, cols, fulldatatype)
end

function LegendDataTypes.readdata(
    input::HDF5.H5DataStore, name::AbstractString,
    AT::Type{<:TypedTables.Table}
)
    # Hack:
    TypedTables.Table(readdata(input, name, NamedTuple{AT.var.ub.body.parameters[1]}))
end
