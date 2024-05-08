using LegendHDF5IO
using Unitful
using TypedTables
using ArraysOfArrays
using RadiationDetectorSignals
using StatsBase
using EncodedArrays
using LegendDataManagement

@testset verbose=true "test wrapper" begin
    @testset verbose=true "reading and writing" begin
        mktempdir(pwd()) do tmp
            path = joinpath(tmp, "tmp.lh5")
            lh5open(path, "cw") do lhd
                @testset "IO of real value" begin
                    w = rand()
                    @test setindex!(lhd, w, "w") |> isnothing
                    @test lhd["w"] == w
                end
                @testset "IO of real quantity" begin
                    v = rand()*u"keV"
                    @test setindex!(lhd, v, "v") |> isnothing
                    @test lhd["v"] == v
                end
                @testset "IO of Bool value" begin
                    z = rand(Bool)
                    @test setindex!(lhd, z, "z") |> isnothing
                    @test lhd["z"] == z 
                end
                @testset "IO of String" begin
                    s = "Test String"
                    @test setindex!(lhd, s, "s") |> isnothing
                    @test lhd["s"] == s
                end
                @testset "IO of Array{<:Real}" begin
                    x = rand(10, 10, 10)
                    @test setindex!(lhd, x, "x") |> isnothing
                    @test lhd["x"][:, :, :] == x        
                end
                @testset "IO of Array{<:Bool}" begin
                    boolarray = rand(Bool, 10)
                    @test setindex!(lhd, boolarray, "boolarray") |> isnothing
                    @test lhd["boolarray"][:] == boolarray
                end
                @testset "IO of Array{<:Quantity}" begin
                    y = rand(10)*u"mm"
                    @test setindex!(lhd, y, "y") |> isnothing
                    @test lhd["y"][:] == y
                end
                @testset "IO of VectorOfVectors" begin
                    vofvec = VectorOfVectors(
                        [rand(-5:5, rand(1:100)) for _ in 1:50])
                    @test setindex!(lhd, vofvec, "vofvec") |> isnothing
                    @test lhd["vofvec"][:] == vofvec
                end
                @testset "IO of VectorOfSimilarVectors" begin
                    vofsimvec = VectorOfSimilarVectors(
                        [rand(-5:5, 100) for _ in 1:50])
                    @test setindex!(lhd, vofsimvec, "vofsimvec") |> isnothing
                    @test lhd["vofsimvec"][:] == vofsimvec
                end
                @testset "IO of NamedTuple" begin
                    nt = (a=10, b=10.0u"mm")
                    @test setindex!(lhd, nt, "nt") |> isnothing
                    @test lhd["nt"] == nt
                end
                @testset "IO of Histogram" begin
                    h = fit(
                        Histogram, 
                        (rand(10), rand(10)), (0:0.2:1, Float64[0, 0.5, 1]))
                    @test setindex!(lhd, h, "h") |> isnothing
                    @test lhd["h"] == h
                end
                @testset "IO of EncodedArray" begin
                    codec = VarlenDiffArrayCodec()
                    x_enc = rand(-5:5, 50) |> codec
                    @test setindex!(lhd, x_enc, "x_enc") |> isnothing
                    @test lhd["x_enc"] == x_enc
                end
                @testset "IO of VectorOfEncodedArrays" begin
                    codec = VarlenDiffArrayCodec()
                    data = [rand(-5:5, rand(1:100)) for _ in 1:50]
                    vofvec = VectorOfVectors(data)
                    vofvec_enc = broadcast(|>, vofvec, codec)
                    @test setindex!(lhd, vofvec_enc, "vofvec_enc") |> isnothing
                    @test lhd["vofvec_enc"] == vofvec_enc
                end
                @testset "IO of VectorOfEncodedSimilarArrays" begin
                    codec = VarlenDiffArrayCodec()
                    data = [rand(-5:5, 100) for _ in 1:50]
                    vofsimvec = VectorOfSimilarVectors(data)
                    vofsimvec_enc = broadcast(|>, vofsimvec, codec)
                    @test setindex!(lhd, vofsimvec_enc, "vofsimvec_enc") |> isnothing
                    @test lhd["vofsimvec_enc"] == vofsimvec_enc
                end
                @testset "IO of ArrayOfRDWaveforms" begin
                    data = nestedview(rand(UInt16, 50, 50)*u"m")
                    trng = range(0.0u"μs", 10.0u"μs"; length=50)
                    waveform = ArrayOfRDWaveforms((fill(trng, 50), data))
                    @test setindex!(lhd, waveform, "waveform") |> isnothing
                    @test lhd["waveform"][:] == waveform
                end
                @testset "IO of Table" begin
                    tbl = Table(a=rand(10), b=rand(10))
                    @test setindex!(lhd, tbl, "tbl") |> isnothing
                    @test lhd["tbl"][:] == tbl
                end
                if @isdefined FileKey
                    @testset "IO of FileKey" begin
                        fk = FileKey("l200-p03-r006-cal-20221226T200846Z")
                        @test setindex!(lhd, fk, "fk") |> isnothing
                        @test lhd["fk"] == fk
                    end
                    @testset "IO of Array{FileKey}" begin
                        fk = FileKey("l200-p03-r006-cal-20221226T200846Z")
                        filekey = fill(fk, 50)
                        @test setindex!(lhd, filekey, "filekey") |> isnothing
                        @test lhd["filekey"] == filekey
                    end
                end
            end
        end
    end
    @testset verbose=true "test append functionality" begin
        mktempdir(pwd()) do tmp
            path = joinpath(tmp, "tmp.lh5")
            lh5open(path, "cw"; usechunks=true) do lhd
                @testset "append Tables" begin
                    tbl = Table(x=rand(10), y=rand(10))
                    newtbl = vcat(tbl, tbl)
                    lhd["tbl"] = tbl
                    @test append!(lhd["tbl"], tbl)[:] == newtbl
                end
                @testset "append VectorOfVectors" begin
                    data = collect(eachcol(rand(55, 50)*u"m"))
                    vofv = VectorOfVectors(data)
                    newvofv = vcat(vofv, vofv)
                    lhd["vofv"] = vofv
                    @test append!(lhd["vofv"], vofv)[:] == newvofv
                end
                @testset "append VectorOfSimilarVectors" begin
                    aofa = nestedview(rand(UInt16, 55, 50)*u"m")
                    newaofa = vcat(aofa, aofa)
                    lhd["aofa"] = aofa
                    @test append!(lhd["aofa"], aofa)[:] == newaofa
                end
                @testset "append VectorOfRDWaveforms" begin
                    aofa = nestedview(rand(UInt16, 55, 50)*u"m")
                    trng = range(0.0u"μs", 10.0u"μs"; length=55)
                    waveform = ArrayOfRDWaveforms((fill(trng, 50), aofa))
                    new_waveform = vcat(waveform, waveform)
                    lhd["waveform"] = waveform
                    @test append!(lhd["waveform"], waveform)[:] == new_waveform
                end
            end
        end
    end
    @testset verbose=true "test adding and removing entries" begin
        mktempdir(pwd()) do tmp
            path = joinpath(tmp, "tmp.lh5")
            lh5open(path, "cw") do lhd
                @testset verbose=true "Tables" begin
                    x, y = rand(10), rand(10)*u"mm"
                    tbl = Table(x=x, y=y)
                    lhd["tbl"] = tbl
                    @testset verbose=true "deleting entry" begin
                        @test delete_entry!(lhd, "tbl/y") |> isnothing
                        @test lhd["tbl"][:] == Table(tbl, y=nothing)
                    end
                    @testset "adding entry" begin
                        @test add_entries!(lhd, "tbl", Table(y=y)) |> isnothing
                        @test lhd["tbl"][:] == tbl
                    end
                end
                @testset verbose=true "NamedTuples" begin
                    x, y = 1.0, 1.0
                    nt = (x=x, y=y)
                    lhd["nt"] = nt
                    @testset "deleting entry" begin
                        @test delete_entry!(lhd, "nt/y") |> isnothing
                        @test lhd["nt"] == (x=x,)
                    end
                    @testset "add entry" begin
                        @test add_entries!(lhd, "nt", (y=y,)) |> isnothing
                        @test lhd["nt"] == nt
                    end    
                end
            end
        end
    end
end