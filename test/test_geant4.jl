# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

using Test
using LegendHDF5IO

using HDF5
using StaticArrays
using StructArrays
using Unitful

function _write_gears_fixture(path::AbstractString)
    HDF5.h5open(path, "w") do f
        g = HDF5.create_group(f, "default_ntuples/t")
        g["de/pages"] = UInt8[2, 0, 1, 3]
        g["de_data/pages"] = Float64[1.0, 2.0, 0.0, 0.5, 0.25, 0.25]
        g["t_data/pages"] = Float64[10, 20, 30, 40, 50, 60]
        g["x_data/pages"] = Float64[1, 2, 3, 4, 5, 6]
    end
end

function _write_g4simple_fixture(path::AbstractString)
    HDF5.h5open(path, "w") do f
        g = HDF5.create_group(f, "default_ntuples/g4sntuple")
        g["event/pages"] = Int32[10, 10, 11]
        g["iRep/pages"] = Int32[1, 2, 1]
        g["Edep/pages"] = Float64[0.1, 0.2, 0.3]
        g["x/pages"] = Float64[1, 2, 3]
        g["step/pages"] = Int32[0, 1, 2]
    end
end

@testset verbose=true "Geant4 HDF5 input" begin
    mktempdir(pwd()) do tmp
        @testset "GEARS layout" begin
            path = joinpath(tmp, "gears.h5")
            _write_gears_fixture(path)
            hits = open(read, path, Geant4HDF5Input)
            # event 2 has no energy deposition and is dropped
            @test hits isa StructArray{<:NamedTuple}
            @test hits.evtno == Int32[1, 1, 2, 2, 2]
            @test hits.edep == Float32[1.0, 2.0, 0.5, 0.25, 0.25] .* u"keV"
            @test hits.thit == Float32[10, 20, 40, 50, 60] .* u"s"
            @test hits.pos == [SVector(x, 0.f0, 0.f0) * u"mm" for x in Float32[1, 2, 4, 5, 6]]
            @test hits.detno == ones(Int32, 5)
            @test hits.stp == zeros(Int32, 5)
        end
        @testset "g4simple layout" begin
            path = joinpath(tmp, "g4simple.h5")
            _write_g4simple_fixture(path)
            input = open(path, Geant4HDF5Input)
            try
                hits = input[:]
                @test hits isa StructArray{<:NamedTuple}
                @test hits.evtno == Int32[10, 10, 11]
                @test hits.detno == Int32[1, 2, 1]
                @test hits.volID == ones(Int32, 3)
                @test hits.edep == Float32[0.1, 0.2, 0.3] .* u"MeV"
                @test hits.pos == [SVector(x, 0.f0, 0.f0) * u"mm" for x in Float32[1, 2, 3]]
                @test hits.stp == Int32[0, 1, 2]
                @test hits.ekin == zeros(Float32, 3) .* u"MeV"
            finally
                close(input)
            end
        end
        @testset "unrecognized layout" begin
            path = joinpath(tmp, "other.h5")
            HDF5.h5open(path, "w") do f
                f["some_data"] = [1, 2, 3]
            end
            @test_throws ErrorException open(read, path, Geant4HDF5Input)
        end
    end
end
