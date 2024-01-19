using LegendHDF5IO
using Unitful
using TypedTables
using ArraysOfArrays
using RadiationDetectorSignals
using StatsBase

@testset "test wrapper" begin
    @testset "reading and writing" begin
        v = rand()
        w = v * u"W"
        z = rand(Bool)
        s = "Test String"
        h = fit(Histogram, (rand(10), rand(10)), (0:0.2:1, Float64[0, 0.5, 1]))
        data3 = (v=v, w=w, z=z, s=s, h=h)
        boolarray = rand(Bool, 50)
        x = rand(50)
        y = x*u"J"
        vofv1 = VectorOfVectors(fill(x, 50))
        vofv2 = VectorOfVectors(fill(y, 50))
        aofa1 = nestedview(rand(UInt16, 50, 50))
        aofa2 = nestedview(rand(UInt16, 50, 50)*u"m")
        trng = range(0.0u"μs", 10.0u"μs"; length=50)
        waveform = ArrayOfRDWaveforms((fill(trng, 50), aofa2))
        waveform2 = ArrayOfRDWaveforms((fill(trng, 50), vofv1))
        tbl = Table((a=x, b=y, c=vofv1, d=vofv2, e=aofa1, f=waveform, 
        g=waveform2, h=boolarray))
        nt = (data1=aofa2, data2=tbl, data3=data3)
        mktempdir(pwd()) do tmp
            path = joinpath(tmp, "tmp.lh5")
            lh5open(path, "cw") do f
                f["tmp"] = nt
            end
            # now check if datatypes and values are equal to the original 
            # data, that was written to tmp.lh5
            lh5open(path) do f
                NT = f["tmp"]
                @test keys(NT) == keys(nt)
                @test NT.data1 == nt.data1
                @testset "test first named tuple" begin
                    for k=keys(data3) @test nt.data3[k] == NT.data3[k] end
                end
                for (col1, col2) in zip(columns(NT.data2), columns(nt.data2))
                    if isa(col1, ArrayOfRDWaveforms)
                        @testset "check if RDWaveforms are equal" begin
                                @test col1[:].signal == col2.signal
                                @test col1[:].time == col2.time
                        end
                    else
                        @test col1 == col2
                    end
                end
            end
        end
    end
    @testset "test append functionality" begin
        x = rand(50)
        y = x*u"J"
        vofv1 = VectorOfVectors(collect(eachcol(rand(55, 50))))
        vofv2 = VectorOfVectors(collect(eachcol(rand(55, 50)*u"m")))
        aofa1 = nestedview(rand(UInt16, 55, 50))
        aofa2 = nestedview(rand(UInt16, 55, 50)*u"m")
        trng = range(0.0u"μs", 10.0u"μs"; length=55)
        waveform = ArrayOfRDWaveforms((fill(trng, 50), aofa2))
        tbl = Table((a=x, b=y, c=vofv1, d=vofv2, e=aofa1, f=waveform))
        nt = (data1=aofa2, data2=tbl)

        mktempdir(pwd()) do tmp
            path = joinpath(tmp, "tmp.lh5")
            lh5open(path, "cw") do f
                f["tmp", 50] = nt
                keys(f)
            end

            # first append data to in-memory-data
            nt2 = (data1=vcat(aofa2, aofa2), data2=vcat(tbl, tbl))

            # append data to on-disk-data
            lh5open(path, "cw") do f
                append!(f["tmp/data1"], aofa2)
                append!(f["tmp/data2"], tbl)
            end

            # now check if datatypes and values are equal to the data 
            # that was extended in memory
            lh5open(path) do f
                NT = f["tmp"]
                @test keys(NT) == keys(nt2)

                @test haskey(f, "tmp")
                @test haskey(f, "tmp/data1")
                @test haskey(f, "tmp/data2")
                @test !haskey(f, "tmp/data3")
                @test !haskey(f, "data1")
                @test !haskey(f, "data2")
                
                @test NT.data1[:] == nt2.data1
                for (col1, col2) in zip(columns(NT.data2), columns(nt2.data2))
                    if isa(col1, ArrayOfRDWaveforms)
                    @testset "check if RDWaveforms are equal" begin
                            @test col1.signal == col2.signal
                            @test col1.time == col2.time
                    end
                    else
                        @test col1 == col2
                    end
                end
            end
        end
    end
    @testset "test adding and removing columns from Tables" begin
        x, y = rand(10), rand(10)
        tbl = Table(x=x, y=y)
        mktempdir(pwd()) do tmp
            path = joinpath(tmp, "tmp.lh5")
            lh5open(path, "cw") do lhd
                lhd["tbl"] = tbl
                reduce_datastore(lhd, joinpath("tbl", "y"))
                @test lhd["tbl"][:] == Table(tbl, y=nothing)
                extend_datastore(lhd, "tbl", Table(y=y))
                @test lhd["tbl"][:] == tbl
            end
        end
    end
    @testset "test adding and removing columns from NamedTuples" begin
        x, y = rand(10), rand(10)
        nt = (x=x, y=y)
        mktempdir(pwd()) do tmp
            path = joinpath(tmp, "tmp.lh5")
            lh5open(path, "cw") do lhd
                lhd["nt"] = nt
                reduce_datastore(lhd, joinpath("nt", "y"))
                @test lhd["nt"] == (x=x,)
                extend_datastore(lhd, "nt", (y=y,))
                @test lhd["nt"] == nt
            end
        end
    end
end