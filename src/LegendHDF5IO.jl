# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

module LegendHDF5IO

using ArraysOfArrays
using EncodedArrays
using LegendDataTypes
using RadiationDetectorSignals
using StaticArrays
using StatsBase
using StructArrays
using Tables
using Unitful

import DiskArrays
import HDF5
using LegendDataTypes: readdata, writedata, getunits, setunits!,
    units_from_string, units_to_string,
    read_from_properties, write_to_properties!
using RadiationDetectorSignals: RealQuantity, ArrayOfRDWaveforms

import H5Zzstd
using Preferences: @load_preference
using PrecompileTools: @setup_workload, @compile_workload


include("generic_io.jl")
include("radsig_io.jl")
include("geant4_hdf5.jl")
include("histogram_io.jl")
include("lazy_table.jl")
include("types.jl")

const _datatype_dict = Dict{String,Type}()
const _datatype_names = Vector{Pair{Type,String}}()

function _register_builtin_datatypes!()
    register_datatype!(EventType)
    register_datatype!(DAQType)
    _datatype_dict[_sort_datatype_fields("table{t0,dt,values}")] = Vector{<:RDWaveform}
    _datatype_dict[_sort_datatype_fields("struct{binning,weights,isdensity}")] = Histogram
    nothing
end

function __init__()
    scatter_bulk_max_bytes[] = @load_preference("scatter_bulk_max_bytes", 2^20)
    scatter_bulk_max_waste[] = @load_preference("scatter_bulk_max_waste", 4)
    _register_builtin_datatypes!()
end


# Reading and writing an LH5 file exercises most of the package, so
# precompiling it removes the bulk of the latency of the first read:
@setup_workload begin
    _register_builtin_datatypes!()
    tbl = StructArrays.StructArray((
        evtno = collect(Int32(1):Int32(4)),
        energy = rand(4),
        flag = rand(Bool, 4),
        wf = VectorOfSimilarVectors(rand(UInt16, 8, 4)),
        vov = VectorOfVectors([rand(Float32, 3) for _ in 1:4]),
    ))
    # Reading values with units evaluates the unit expression, which
    # precompilation does not allow, so the workload stays unitless:
    wfs = ArrayOfRDWaveforms((
        fill(range(0.0, 1.0, length = 8), 4),
        VectorOfSimilarVectors(rand(UInt16, 8, 4)),
    ))
    @compile_workload begin
        mktempdir() do dir
            path = joinpath(dir, "precompile.lh5")
            lh5open(path, "cw") do lhd
                lhd["tbl"] = tbl
                lhd["wfs"] = wfs
            end
            lh5open(path) do lhd
                t = lhd["tbl"]
                t[1:2]
                t[:]
                t.energy[[1, 3]]
                sum(t.energy)
                lhd["wfs"][1:2]
            end
        end
    end
end

end # module
