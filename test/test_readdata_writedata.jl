# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

using Test
using LegendHDF5IO

using ArraysOfArrays
using EncodedArrays
using HDF5
using Measurements
using RadiationDetectorSignals
using StaticArrays
using StatsBase
using TypedTables
using Unitful

using LegendDataTypes
using LegendDataTypes: readdata, writedata

# Roundtrip through writedata and datatype-driven readdata
function _roundtrip(x)
    mktempdir(pwd()) do tmp
        HDF5.h5open(joinpath(tmp, "tmp.h5"), "w") do f
            writedata(f, "x", x)
            readdata(f, "x")
        end
    end
end

@testset verbose=true "readdata/writedata roundtrips" begin
    @testset "scalars" begin
        @test _roundtrip(1.5) === 1.5
        @test _roundtrip(42) === 42
        @test _roundtrip(1.5u"keV") === 1.5u"keV"
        @test _roundtrip(true) === true
        @test _roundtrip(false) === false
        @test _roundtrip("hello") == "hello"
        @test _roundtrip(:sym) === :sym
        @test _roundtrip(evt_pulser) === evt_pulser
        @test _roundtrip(daq_gerda) === daq_gerda
        @test _roundtrip((1.0, 2.0, 3.0)) === (1.0, 2.0, 3.0)
    end

    @testset "arrays" begin
        x = rand(10)
        @test _roundtrip(x) == x
        xq = rand(10)*u"mm"
        @test _roundtrip(xq) == xq
        xm = rand(3, 4, 5)
        @test _roundtrip(xm) == xm
        xb = rand(Bool, 10)
        @test _roundtrip(xb) == xb
        xs = ["a", "bb", "ccc"]
        @test _roundtrip(xs) == xs
        xe = [evt_real, evt_pulser, evt_baseline]
        @test _roundtrip(xe) == xe
        xt = [(1.0, 2.0), (3.0, 4.0)]
        @test _roundtrip(xt) == xt
        xsv = [SVector(1.0, 2.0, 3.0), SVector(4.0, 5.0, 6.0)]
        @test _roundtrip(xsv) == xsv
    end

    @testset "nested arrays" begin
        vov = VectorOfVectors([rand(rand(1:10)) for _ in 1:20])
        @test _roundtrip(vov) == vov
        vovq = VectorOfVectors([rand(rand(1:10))*u"m" for _ in 1:20])
        @test _roundtrip(vovq) == vovq
        vosv = VectorOfSimilarVectors(rand(5, 20))
        @test _roundtrip(vosv) == vosv
        vom = VectorOfSimilarArrays(rand(3, 4, 5))
        @test _roundtrip(vom) == vom
    end

    @testset "encoded arrays" begin
        codec = VarlenDiffArrayCodec()
        enc = rand(Int16(-100):Int16(100), 50) |> codec
        enc2 = _roundtrip(enc)
        @test enc2 == enc
        @test eltype(eltype(enc2)) == Int16

        vov = VectorOfVectors([rand(Int16(-5):Int16(5), rand(1:20)) for _ in 1:10])
        vov_enc = broadcast(|>, vov, codec)
        vov_enc2 = _roundtrip(vov_enc)
        @test vov_enc2 == vov_enc
        @test eltype(eltype(vov_enc2)) == Int16

        vosv = VectorOfSimilarVectors(rand(Int32(-5):Int32(5), 20, 10))
        vosv_enc = broadcast(|>, vosv, codec)
        vosv_enc2 = _roundtrip(vosv_enc)
        @test vosv_enc2 == vosv_enc
        @test eltype(eltype(vosv_enc2)) == Int32
    end

    @testset "structs and tables" begin
        nt = (a = 42, b = rand(5)*u"mm", c = "text")
        @test _roundtrip(nt) == nt
        tbl = Table(a = rand(10), b = rand(10)*u"keV")
        @test _roundtrip(tbl) == tbl
    end

    @testset "waveforms" begin
        trng = range(0.0u"μs", 10.0u"μs"; length = 50)
        wfs = ArrayOfRDWaveforms((fill(trng, 20), VectorOfSimilarVectors(rand(UInt16, 50, 20)*u"m")))
        r = _roundtrip(wfs)
        @test r.signal == wfs.signal
        @test r.time == wfs.time

        wfs_vov = ArrayOfRDWaveforms((
            fill(trng, 20),
            VectorOfVectors([rand(50)*u"m" for _ in 1:20])
        ))
        r_vov = _roundtrip(wfs_vov)
        @test r_vov.signal == wfs_vov.signal
        @test r_vov.time == wfs_vov.time

        codec = VarlenDiffArrayCodec()
        signal = VectorOfSimilarVectors(rand(Int32(-5):Int32(5), 50, 20))
        wfs_enc = ArrayOfRDWaveforms((fill(trng, 20), broadcast(|>, signal, codec)))
        r_enc = _roundtrip(wfs_enc)
        @test r_enc.signal == wfs_enc.signal
        @test r_enc.time == wfs_enc.time
    end

    @testset "histograms" begin
        h1 = fit(Histogram, rand(100), 0:0.1:1)
        @test _roundtrip(h1) == h1
        h2 = fit(Histogram, (rand(100), rand(100)), (0:0.2:1, Float64[0, 0.5, 1]))
        @test _roundtrip(h2) == h2
    end

    @testset "measurements" begin
        m = 2.0 ± 0.1
        @test _roundtrip(m) == m
        mv = measurement.(rand(10), rand(10))
        @test _roundtrip(mv) == mv
        mvu = mv .* u"s"
        @test _roundtrip(mvu) == mvu
    end

    @testset "error paths" begin
        mktempdir(pwd()) do tmp
            HDF5.h5open(joinpath(tmp, "tmp.h5"), "w") do f
                @test_throws ArgumentError writedata(f, "x", Dict(:a => 1))
                @test_throws ArgumentError writedata(f, "t", (1, 2.0))
                writedata(f, "sv", [SVector(1.0, 2.0)])
                @test_throws ErrorException readdata(f, "sv", AbstractArray{<:SVector{3}})
                writedata(f, "nt2", [(1.0, 2.0)])
                @test_throws ErrorException readdata(f, "nt2", AbstractArray{<:NTuple{3,Float64},1})
            end
        end
    end
end
