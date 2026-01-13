# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

using Test
using LegendHDF5IO

using ArraysOfArrays
using EncodedArrays
using Measurements
using RadiationDetectorSignals
using StatsBase
using TypedTables
using Unitful

@testset verbose=true "test generic IO" begin
    @testset verbose=true "data types" begin
        @test LegendHDF5IO._sort_datatype_fields("table{values,dt,t0}") == "table{dt,t0,values}"
        @test LegendHDF5IO._sort_datatype_fields("struct{weights,isdensity,binning}") == "struct{binning,isdensity,weights}"
        @test LegendHDF5IO._sort_datatype_fields("real") == "real"
        @test LegendHDF5IO._sort_datatype_fields("array<1>{encoded_array<1>{real}}") == "array<1>{encoded_array<1>{real}}"
    end
end
