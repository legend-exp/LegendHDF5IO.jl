# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

module LegendHDF5IOMeasurementsExt

using LegendHDF5IO
using LegendDataTypes
using LegendDataTypes: readdata, writedata
using Measurements
using HDF5
using Unitful

function __init__()
    LegendHDF5IO._datatype_dict["measurement"] = Measurement
end

LegendHDF5IO.datatype_to_string(::Type{<:Union{<:Measurement, <:Quantity{<:Measurement}}}) = "measurement"

function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString,
    x::Union{<:T, <:AbstractArray{<:T}},
    fulldatatype::DataType = typeof(ustrip(x))
) where {T <: Union{<:Measurement, Quantity{<:Measurement}}}
    nt::NamedTuple = (val = Measurements.value.(x), err = Measurements.uncertainty.(x))
    writedata(output, name, nt, fulldatatype)
end

function LegendDataTypes.readdata(
    input::HDF5.H5DataStore, name::AbstractString,
    ::Type{<:Union{<:T, <:AbstractArray{<:T}}}
) where {T <: Union{<:Measurement, <:Quantity{<:Measurement}}}
    nt = readdata(input, name, NamedTuple{(:val, :err)})
    measurement.(nt.val, nt.err)
end

"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:Measurement})

return a value with type `Measurement`
"""
function LegendHDF5IO.LH5Array(ds::HDF5.H5DataStore, ::Type{<:Union{
        <:Measurement, 
        <:Quantity{<:Measurement}, 
        <:AbstractArray{<:Union{<:Measurement, <:Quantity{<:Measurement}}}}}
    )
    nt::NamedTuple{(:val, :err)} = LegendHDF5IO.LH5Array(ds, NamedTuple{(:val, :err)})
    measurement.(nt.val, nt.err)
end


# write Measurement
function LegendHDF5IO.create_entry(parent::LegendHDF5IO.LHDataStore, name::AbstractString, 
    data::Union{<:T, <:AbstractArray{<:T}}; kwargs...) where {T <: Union{<:Measurement, <:Quantity{<:Measurement}}}
    LegendHDF5IO.create_entry(parent, name, (val = Measurements.value.(data), err = Measurements.uncertainty.(data)); kwargs...)
    LegendHDF5IO.setdatatype!(parent.data_store[name], Measurement)
    nothing
end


end # module