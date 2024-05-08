# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

module LegendHDF5IO

using ArraysOfArrays
using ElasticArrays
using EncodedArrays
using LegendDataTypes
using RadiationDetectorSignals
using StaticArrays
using StatsBase
using StructArrays
using Tables
using Unitful

import HDF5
import TypedTables
using Tables: columns
using TypedTables: Table
using LegendDataTypes: readdata, writedata, getunits, setunits!,
    units_from_string, units_to_string,
    read_from_properties, write_to_properties!
using RadiationDetectorSignals: RealQuantity, ArrayOfRDWaveforms


include("generic_io.jl")
include("radsig_io.jl")
include("geant4_hdf5.jl")
include("histogram_io.jl")
include("types.jl")

const _datatype_dict = Dict{String,Type}()

@static if !isdefined(Base, :get_extension)
    using Requires
end

function __init__()
    _datatype_dict[datatype_to_string(EventType)] = EventType
    _datatype_dict["table{t0,dt,values}"] = Vector{<:RDWaveform}
    _datatype_dict["struct{binning,weights,isdensity}"] = Histogram

    @static if !isdefined(Base, :get_extension)
        @require LegendDataManagement = "9feedd95-f0e0-423f-a8dc-de0970eae6b3" begin
            include("../ext/LegendHDF5IOLegendDataManagementExt.jl")
        end        
    end
end

end # module
