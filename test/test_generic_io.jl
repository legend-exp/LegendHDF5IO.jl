# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

using Test
using LegendHDF5IO

using ArraysOfArrays
using EncodedArrays
using HDF5
using Measurements
using RadiationDetectorSignals
using StatsBase
using StructArrays
using Unitful

using ArraysOfArrays: AbstractArrayOfSimilarArrays
using LegendDataTypes: readdata, writedata
using RadiationDetectorSignals: RealQuantity

@enum TestEnum te_a=1 te_b=2

struct TestId
    no::Int
end

LegendHDF5IO.register_datatype!("testid", TestId)
LegendHDF5IO.LH5Array(ds::HDF5.Dataset, ::Type{<:TestId}) = TestId(read(ds))

function LegendHDF5IO.create_entry(parent::LegendHDF5IO.LHDataStore,
    name::AbstractString, x::TestId; kwargs...)
    parent.data_store[name] = x.no
    LegendHDF5IO.setdatatype!(parent.data_store[name], TestId)
    nothing
end

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
    @testset "register_datatype!" begin
        LegendHDF5IO.register_datatype!(TestEnum)
        @test LegendHDF5IO.datatype_to_string(TestId) == "testid"
        @test LegendHDF5IO.datatype_from_string("testid") == TestId
        @test_throws ArgumentError LegendHDF5IO.datatype_to_string(Base.RefValue{Int})
        mktempdir(pwd()) do tmp
            path = joinpath(tmp, "tmp.lh5")
            lh5open(path, "cw") do lhd
                lhd["id"] = TestId(42)
                @test lhd["id"] == TestId(42)
                lhd["te"] = [te_a, te_b, te_a]
                @test lhd["te"] == [te_a, te_b, te_a]
            end
            HDF5.h5open(path, "r+") do f
                @test readdata(f, "id") == TestId(42)
                writedata(f, "id2", TestId(7))
                @test readdata(f, "id2") == TestId(7)
                @test readdata(f, "te") == [te_a, te_b, te_a]
            end
        end
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
