# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

module LegendHDF5IOMeasurementsExt

using LegendHDF5IO
using Measurements
using HDF5
using Unitful

function __init__()
    LegendHDF5IO.register_datatype!("measurement", Measurement)
end

const MeasurementLike = Union{Measurement, Quantity{<:Measurement}}

LegendHDF5IO.datatype_to_string(::Type{<:MeasurementLike}) = "measurement"

"""
    LH5Array(ds::HDF5.Dataset, ::Type{<:Measurement})

return a value with type `Measurement`
"""
function LegendHDF5IO.LH5Array(ds::HDF5.H5DataStore,
    ::Type{<:Union{MeasurementLike, AbstractArray{<:MeasurementLike}}}
)
    nt::NamedTuple{(:val, :err)} = LegendHDF5IO.LH5Array(ds, NamedTuple{(:val, :err)})
    measurement.(nt.val, nt.err)
end


# write Measurement
function LegendHDF5IO.create_entry(parent::LegendHDF5IO.LHDataStore, name::AbstractString,
    data::Union{MeasurementLike, AbstractArray{<:MeasurementLike}}; kwargs...)
    LegendHDF5IO.create_entry(parent, name, (val = Measurements.value.(data), err = Measurements.uncertainty.(data)); kwargs...)
    LegendHDF5IO.setdatatype!(parent.data_store[name], typeof(data))
    nothing
end


end # module