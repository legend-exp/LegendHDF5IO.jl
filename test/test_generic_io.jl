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

@enum TestEnum te_a=1 te_b=2
@enum TestEnumNamed ten_a=1 ten_b=2
@enum TestStatusA ok_a=0 failed_a=1
@enum TestStatusB ok_b=0 failed_b=1

module EnumModA
    @enum Status st_ok=0 st_failed=1
end

module EnumModB
    @enum Status st_ok=0 st_failed=1
end

struct TestUnregistered end

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

struct TestChannelId <: Integer
    no::Int
end

struct TestPNum{T} <: Real
    x::T
end
LegendHDF5IO.register_datatype!("testpnum", TestPNum)

struct TestConcA end
struct TestConcB end
struct TestConcC end
struct TestConcD end

LegendHDF5IO.register_datatype!("testchannelid", TestChannelId)
LegendHDF5IO.LH5Array(ds::HDF5.Dataset, ::Type{<:TestChannelId}) = TestChannelId(read(ds))

function LegendHDF5IO.create_entry(parent::LegendHDF5IO.LHDataStore,
    name::AbstractString, x::TestChannelId; kwargs...)
    parent.data_store[name] = x.no
    LegendHDF5IO.setdatatype!(parent.data_store[name], TestChannelId)
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
        @test_throws ArgumentError datatype_from_string("no_such_datatype<1>{real}")
        @test datatype_from_string("array_of_equalsized_arrays<1,2>{real}") <: AbstractArrayOfSimilarArrays{<:RealQuantity, 2, 1}
    end
    @testset "register_datatype!" begin
        LegendHDF5IO.register_datatype!(TestEnum)
        @test LegendHDF5IO.datatype_to_string(TestEnum) == "enum{te_a=1,te_b=2}"
        # an enum registered under a name is written under that name:
        LegendHDF5IO.register_datatype!("testenumnamed", TestEnumNamed)
        @test LegendHDF5IO.datatype_to_string(TestEnumNamed) == "testenumnamed"
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
                lhd["ten"] = [ten_a, ten_b]
                @test LegendHDF5IO.getattribute(lhd.data_store["ten"], :datatype, "") == "array<1>{testenumnamed}"
                @test lhd["ten"] == [ten_a, ten_b]
            end
            HDF5.h5open(path, "r+") do f
                @test readdata(f, "id") == TestId(42)
                writedata(f, "id2", TestId(7))
                @test readdata(f, "id2") == TestId(7)
                @test readdata(f, "te") == [te_a, te_b, te_a]
            end
        end
    end
    @testset "registered names take precedence over built-in ones" begin
        @test LegendHDF5IO.datatype_to_string(TestChannelId) == "testchannelid"
        @test LegendHDF5IO.datatype_to_string(Vector{TestChannelId}) == "array<1>{testchannelid}"
        mktempdir(pwd()) do tmp
            path = joinpath(tmp, "tmp.lh5")
            lh5open(path, "cw") do lhd
                lhd["ch"] = TestChannelId(3)
                @test LegendHDF5IO.getattribute(lhd.data_store["ch"], :datatype, "") == "testchannelid"
                @test lhd["ch"] === TestChannelId(3)
            end
        end
    end
    @testset "registered supertypes take precedence over built-in naming" begin
        @test LegendHDF5IO.datatype_to_string(TestPNum{Float64}) == "testpnum"
        @test LegendHDF5IO.datatype_to_string(Vector{TestPNum{Int}}) == "array<1>{testpnum}"
    end
    @testset "unknown datatype names" begin
        @test_throws ArgumentError LegendHDF5IO.datatype_from_string("nosuchtype")
        @test_throws ArgumentError LegendHDF5IO.datatype_from_string("array<1>{nosuchtype}")
        err = try LegendHDF5IO.datatype_from_string("nosuchtype") catch e; e end
        @test occursin("register", sprint(showerror, err))
    end
    @testset "concurrent registrations" begin
        types = (TestConcA, TestConcB, TestConcC, TestConcD)
        results = Vector{Any}(undef, length(types))
        @sync for (i, T) in enumerate(types)
            Threads.@spawn results[i] = try LegendHDF5IO.register_datatype!("testconc", T); :ok catch e; e end
        end
        # exactly one registration wins, the registry names only that type:
        @test count(==(:ok), results) == 1
        @test all(r -> r === :ok || r isa ArgumentError, results)
        name_of(T) = try LegendHDF5IO.datatype_to_string(T) catch e; nothing end
        @test count(T -> name_of(T) == "testconc", types) == 1
        @test LegendHDF5IO.datatype_from_string("testconc") == types[findfirst(==(:ok), results)]
    end
    @testset "conflicting registrations" begin
        register_datatype! = LegendHDF5IO.register_datatype!
        datatype_to_string = LegendHDF5IO.datatype_to_string
        datatype_from_string = LegendHDF5IO.datatype_from_string

        register_datatype!("teststatus", TestStatusA)
        @test_throws ArgumentError register_datatype!("teststatus", TestStatusB)
        @test datatype_from_string("teststatus") == TestStatusA
        @test datatype_to_string(TestStatusB) == "enum{ok_b=0,failed_b=1}"
        @test register_datatype!("teststatus", TestStatusA) === nothing
        @test_throws ArgumentError register_datatype!("teststatus_alias", TestStatusA)
        @test datatype_to_string(TestStatusA) == "teststatus"

        register_datatype!(EnumModB.Status)
        @test_throws ArgumentError register_datatype!(EnumModA.Status)
        @test datatype_from_string(datatype_to_string(EnumModB.Status)) == EnumModB.Status
        @test register_datatype!(EnumModB.Status) === nothing

        for name in ("real", "bool", "string", "symbol", "struct{a,b}", "table{a,b}",
            "ntuple{real}", "array<1>{real}", "encoded_array<1>{real}",
            "fixedsize_array<1>{real}", "array_of_equalsized_arrays<1,1>{real}",
            "array_of_encoded_arrays<1,1>{real}",
            "array_of_encoded_equalsized_arrays<1,1>{real}", "histogram<1>{real}")
            @test_throws ArgumentError register_datatype!(name, TestUnregistered)
        end
        # names the datatype parser cannot read are rejected as well:
        @test_throws ArgumentError register_datatype!("status2", TestUnregistered)
        @test_throws ArgumentError register_datatype!("my type", TestUnregistered)
        @test_throws ArgumentError datatype_to_string(TestUnregistered)
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
