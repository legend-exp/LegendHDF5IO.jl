# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

using Test
using LegendHDF5IO

using ArraysOfArrays
using EncodedArrays
using HDF5
using Measurements
using RadiationDetectorSignals
using StatsBase
using TypedTables
using Unitful

using ArraysOfArrays: AbstractArrayOfSimilarArrays
using LegendDataTypes: readdata, writedata
using RadiationDetectorSignals: RealQuantity

@testset verbose=true "test generic IO" begin
    @testset verbose=true "data types" begin
        @test LegendHDF5IO._sort_datatype_fields("table{values,dt,t0}") == "table{dt,t0,values}"
        @test LegendHDF5IO._sort_datatype_fields("struct{weights,isdensity,binning}") == "struct{binning,isdensity,weights}"
        @test LegendHDF5IO._sort_datatype_fields("real") == "real"
        @test LegendHDF5IO._sort_datatype_fields("array<1>{encoded_array<1>{real}}") == "array<1>{encoded_array<1>{real}}"
        @test_throws ArgumentError LegendHDF5IO._sort_datatype_fields("no{valid}datatype")
    end
    @testset verbose=true "datatype string parsing" begin
        datatype_from_string = LegendHDF5IO.datatype_from_string
        @test datatype_from_string("array_of_encoded_arrays<1,1>{real}") <: VectorOfEncodedArrays{<:RealQuantity, 1}
        @test datatype_from_string("array_of_encoded_equalsized_arrays<1,1>{real}") <: VectorOfEncodedSimilarArrays{<:RealQuantity, 1}
        @test_throws ErrorException datatype_from_string("array_of_encoded_arrays<2,1>{real}")
        @test_throws ErrorException datatype_from_string("array_of_encoded_equalsized_arrays<2,1>{real}")
        @test_throws ErrorException datatype_from_string("no_such_datatype<1>{real}")
        @test datatype_from_string("array_of_equalsized_arrays<1,2>{real}") <: AbstractArrayOfSimilarArrays{<:RealQuantity, 2, 1}
    end
    @testset "Bool datasets with units are rejected" begin
        mktempdir(pwd()) do tmp
            path = joinpath(tmp, "tmp.lh5")
            HDF5.h5open(path, "w") do f
                f["bwu"] = UInt8[0, 1]
                LegendHDF5IO.setdatatype!(f["bwu"], Array{Bool,1})
                LegendHDF5IO.setunits!(f["bwu"], u"keV")
                f["swu"] = UInt8(1)
                LegendHDF5IO.setdatatype!(f["swu"], Bool)
                LegendHDF5IO.setunits!(f["swu"], u"keV")
            end
            HDF5.h5open(path, "r") do f
                @test_throws ArgumentError readdata(f, "bwu")
                @test_throws ArgumentError readdata(f, "swu")
            end
            lh5open(path) do lhd
                @test_throws ArgumentError lhd["bwu"]
                @test_throws ArgumentError lhd["swu"]
            end
        end
    end
end
