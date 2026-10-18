# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

using Test
using LegendHDF5IO

using ArraysOfArrays
using DiskArrays
using EncodedArrays
using HDF5
using Measurements
using RadiationDetectorSignals
using StaticArrays
using StatsBase
using StructArrays
using Tables
using Unitful

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
                    # Bool data must be stored as integers for h5py compatibility:
                    dt = HDF5.datatype(lhd.data_store["boolarray"])
                    @test HDF5.API.h5t_get_class(dt.id) == HDF5.API.H5T_INTEGER
                    close(dt)
                    # other writers may store bytes other than 0 and 1, which every read path treats as true:
                    lhd.data_store["boolarray"][:] = UInt8[0, 1, 2, 128, 0, 1, 2, 128, 0, 1]
                    expected = Bool[0, 1, 1, 1, 0, 1, 1, 1, 0, 1]
                    @test lhd["boolarray"][:] == expected
                    @test lhd["boolarray"][3:4] == expected[3:4]
                    @test Array(lhd["boolarray"]) == expected
                    saved = LegendHDF5IO.scatter_bulk_max_bytes[], LegendHDF5IO.scatter_bulk_max_waste[]
                    LegendHDF5IO.scatter_bulk_max_bytes[] = LegendHDF5IO.scatter_bulk_max_waste[] = 0
                    try
                        @test lhd["boolarray"][[3, 4, 9]] == expected[[3, 4, 9]]
                    finally
                        LegendHDF5IO.scatter_bulk_max_bytes[], LegendHDF5IO.scatter_bulk_max_waste[] = saved
                    end
                end
                @testset "IO of Array{<:Quantity}" begin
                    y = rand(10)*u"mm"
                    @test setindex!(lhd, y, "y") |> isnothing
                    @test lhd["y"][:] == y
                end
                @testset "IO of VectorOfVectors" begin
                    vofvec = VectorOfArrays(
                        [rand(-5:5, rand(1:100)) for _ in 1:50])
                    @test setindex!(lhd, vofvec, "vofvec") |> isnothing
                    @test lhd["vofvec"][:] == vofvec
                    # opening a vector of vectors keeps only its data handle open:
                    n_open() = HDF5.API.h5f_get_obj_count(lhd.data_store.id, HDF5.API.H5F_OBJ_DATASET)
                    GC.enable(false)
                    n0 = n_open(); vv = lhd["vofvec"]; n1 = n_open()
                    GC.enable(true)
                    @test n1 - n0 == 1
                end
                @testset "IO of VectorOfSimilarVectors" begin
                    vofsimvec = convert(VectorOfSimilarVectors,
                        [rand(-5:5, 100) for _ in 1:50])
                    @test setindex!(lhd, vofsimvec, "vofsimvec") |> isnothing
                    @test lhd["vofsimvec"][:] == vofsimvec
                end
                @testset "IO of NamedTuple" begin
                    nt = (a=10, b=10.0u"mm")
                    @test setindex!(lhd, nt, "nt") |> isnothing
                    @test lhd["nt"] == nt
                    @test setindex!(lhd, NamedTuple(), "nt0") |> isnothing
                    @test lhd["nt0"] == NamedTuple()
                end
                @testset "IO of Array{<:SVector}" begin
                    sv = [SVector(1.0, 2.0, 3.0), SVector(4.0, 5.0, 6.0)]
                    @test setindex!(lhd, sv, "sv") |> isnothing
                    @test lhd["sv"] == sv
                    svu = [SVector(1.0, 2.0)u"mm", SVector(3.0, 4.0)u"mm"]
                    @test setindex!(lhd, svu, "svu") |> isnothing
                    @test lhd["svu"] == svu
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
                    vofvec = VectorOfArrays(data)
                    vofvec_enc = broadcast(|>, vofvec, codec)
                    @test setindex!(lhd, vofvec_enc, "vofvec_enc") |> isnothing
                    @test lhd["vofvec_enc"][:] == vofvec_enc
                end
                @testset "IO of VectorOfEncodedSimilarArrays" begin
                    codec = VarlenDiffArrayCodec()
                    data = [rand(-5:5, 100) for _ in 1:50]
                    vofsimvec = convert(VectorOfSimilarVectors, data)
                    vofsimvec_enc = broadcast(|>, vofsimvec, codec)
                    @test setindex!(lhd, vofsimvec_enc, "vofsimvec_enc") |> isnothing
                    @test lhd["vofsimvec_enc"][:] == vofsimvec_enc
                end
                @testset "IO of ArrayOfRDWaveforms" begin
                    data = VectorOfSimilarVectors(rand(UInt16, 50, 50)*u"m")
                    trng = range(0.0u"μs", 10.0u"μs"; length=50)
                    waveform = ArrayOfRDWaveforms((fill(trng, 50), data))
                    @test setindex!(lhd, waveform, "waveform") |> isnothing
                    @test lhd["waveform"][:] == waveform
                    # slices and scattered reads take the element type of what they read:
                    w = lhd["waveform"]
                    @test collect(w[1:2]) == waveform[1:2]
                    @test identity.(w[1:2]) == waveform[1:2]
                    @test collect(w[[1, 3]]) == waveform[[1, 3]]
                end
                @testset "failed group writes leave nothing behind" begin
                    @test_throws ArgumentError lhd["bad"] = (a = [1, 2], b = Dict(1 => 2))
                    @test !haskey(lhd.data_store, "bad")
                    @test_throws ArgumentError lhd["bad"] = StructArray((a = [1, 2], b = [Dict(1 => 2), Dict(3 => 4)]))
                    @test !haskey(lhd.data_store, "bad")
                    lhd["bad"] = (a = [1, 2],)
                    @test lhd["bad"].a[:] == [1, 2]
                end
                @testset "IO of tables" begin
                    tbl = StructArray((a = rand(10), b = rand(10)))
                    @test setindex!(lhd, tbl, "tbl") |> isnothing
                    @test lhd["tbl"][:] == tbl
                    # a table is a vector of rows, columns of other shapes are refused:
                    @test_throws ArgumentError lhd["tbl2d"] = StructArray((x = reshape(collect(1:6), 2, 3),))
                    @test !haskey(lhd.data_store, "tbl2d")
                end
                @testset "IO of Measurements" begin
                    m = 2.0 ± 0.1
                    @test setindex!(lhd, m, "m") |> isnothing
                    @test lhd["m"] == m
                    mu = (2.0 ± 0.1)u"s"
                    @test setindex!(lhd, mu, "m_unit") |> isnothing
                    @test lhd["m_unit"] == mu
                    mv = measurement.(rand(10), rand(10))
                    @test setindex!(lhd, mv, "mv") |> isnothing
                    @test lhd["mv"] == mv
                    mvu = mv .* u"s"
                    @test setindex!(lhd, mvu, "mv_unit") |> isnothing
                    @test lhd["mv_unit"] == mvu
                    getdt(k) = LegendHDF5IO.getattribute(lhd.data_store[k], :datatype, "")
                    @test getdt("m") == "measurement"
                    @test getdt("mv") == "array<1>{measurement}"
                    @test getdt("mv_unit") == "array<1>{measurement}"
                    # measurements are read eagerly, so an element stays correlated with itself:
                    x = lhd["mv"]
                    @test x isa Array
                    @test iszero(Measurements.uncertainty(x[1] - x[1]))
                    @test iszero(Measurements.uncertainty(sum(x .- x)))
                end
            end
        end
    end
    @testset verbose=true "scattered reads" begin
        mktempdir(pwd()) do tmp
            path = joinpath(tmp, "tmp.lh5")
            lh5open(path, "cw") do lhd
                x = rand(Float32, 1000)
                lhd["x"] = x
                A = lhd["x"]
                idxs = [1, 2, 3, 10, 11, 500, 999]
                @test A[idxs] == x[idxs]
                unsorted = [7, 3, 3, 950, 1]
                @test A[unsorted] == x[unsorted]
                @test A[Int[]] == Float32[]

                # rows large enough that scattered reads coalesce runs
                # instead of reading the whole index span:
                data = rand(UInt16, 700, 2000)
                lhd["aofa"] = VectorOfSimilarVectors(data)
                W = lhd["aofa"]
                sparse_idxs = [2, 3, 4, 700, 1999]
                @test W[sparse_idxs] == VectorOfSimilarVectors(data)[sparse_idxs]
                @test W.data[:, sparse_idxs] == data[:, sparse_idxs]
                dense_idxs = collect(4:2:1000)
                @test W[dense_idxs] == VectorOfSimilarVectors(data)[dense_idxs]

                # the bulk/scattered decision is tunable and must not
                # change the returned values:
                tune = [5, 5, 900, 17, 3]
                old_max, old_waste = LegendHDF5IO.scatter_bulk_max_bytes[],
                    LegendHDF5IO.scatter_bulk_max_waste[]
                try
                    LegendHDF5IO.scatter_bulk_max_bytes[] = 0
                    LegendHDF5IO.scatter_bulk_max_waste[] = 0
                    @test A[tune] == x[tune]
                    @test W[tune] == VectorOfSimilarVectors(data)[tune]
                    LegendHDF5IO.scatter_bulk_max_bytes[] = typemax(Int)
                    @test A[tune] == x[tune]
                    @test W[tune] == VectorOfSimilarVectors(data)[tune]
                finally
                    LegendHDF5IO.scatter_bulk_max_bytes[] = old_max
                    LegendHDF5IO.scatter_bulk_max_waste[] = old_waste
                end

                vv = VectorOfVectors([rand(rand(1:20)) for _ in 1:100])
                lhd["vv"] = vv
                @test lhd["vv"][[2, 3, 4, 70]] == vv[[2, 3, 4, 70]]

                # strings need HDF5's own normalization, whichever read path:
                strs = ["a", "bb", "ccc", "d"]
                lhd["strs"] = strs
                S = lhd["strs"]
                @test eltype(S) == String
                @test S[[4, 1, 1]] == strs[[4, 1, 1]]
                try
                    LegendHDF5IO.scatter_bulk_max_bytes[] = 0
                    LegendHDF5IO.scatter_bulk_max_waste[] = 0
                    @test S[[4, 1, 1]] == strs[[4, 1, 1]]
                finally
                    LegendHDF5IO.scatter_bulk_max_bytes[] = old_max
                    LegendHDF5IO.scatter_bulk_max_waste[] = old_waste
                end
            end

            # sparse index patterns on contiguous-layout data trigger
            # single-read scattered selections (point selection for vectors,
            # hyperslab-run union for higher dimensions):
            path2 = joinpath(tmp, "sparse.lh5")
            lh5open(path2, "cw") do lhd
                x = rand(Float32, 400_000)
                lhd["x"] = x * u"keV"
                X = lhd["x"]
                sp = [1, 5, 6, 350_000, 399_999]
                @test X[sp] == x[sp] * u"keV"
                unsorted_dups = [350_000, 1, 1, 399_999, 5]
                @test X[unsorted_dups] == x[unsorted_dups] * u"keV"

                b = rand(Bool, 2^20 + 5)
                lhd["b"] = b
                spb = [1, 3, 4, 2^20 + 3]
                @test lhd["b"][spb] == b[spb]
                @test lhd["b"][spb] isa Vector{Bool}

                # scattered reads of long strings fetch the selected elements only:
                long = repeat('x', 16384)
                lhd["ls"] = fill(long, 2048)
                LS = lhd["ls"]
                @test LS[[1, 2048]] == [long, long]
                @test (@allocated LS[[1, 2048]]) < 2^20

                data = rand(UInt16, 700, 2000)
                lhd["aofa"] = VectorOfSimilarVectors(data)
                W = lhd["aofa"]
                sp2 = [2, 3, 4, 700, 1999]
                @test W.data[:, sp2] == data[:, sp2]
                @test W.data[:, [1999, 2, 2, 700]] == data[:, [1999, 2, 2, 700]]
                # non-colon front indices use per-run reads:
                @test W.data[5, sp2] == data[5, sp2]
            end

            # on chunked datasets vectors still use point selections while
            # higher dimensions fall back to per-run reads:
            path3 = joinpath(tmp, "sparse_chunked.lh5")
            lh5open(path3, "cw"; usechunks=true) do lhd
                x = rand(Float32, 400_000)
                data = rand(UInt16, 700, 2000)
                lhd["x"] = x
                lhd["aofa"] = VectorOfSimilarVectors(data)
                sp = [1, 5, 6, 350_000, 399_999]
                sp2 = [2, 3, 4, 700, 1999]
                @test lhd["x"][sp] == x[sp]
                @test lhd["aofa"].data[:, sp2] == data[:, sp2]
            end
        end
    end
    @testset verbose=true "DiskArrays interface" begin
        mktempdir() do dir
            x = rand(Float32, 50_000)
            data = rand(UInt16, 100, 3000)
            flags = rand(Bool, 100)
            lh5open(joinpath(dir, "diskarrays.lh5"), "cw") do lhd
                lhd["x"] = x * u"keV"
                lhd["wf"] = VectorOfSimilarVectors(data)
                lhd["flags"] = flags
                lhd["strs"] = ["x$i" for i in 1:50]
                lhd["empty"] = Float32[]
                lhd["empty2d"] = zeros(Int, 3, 0)
                lhd["small"] = Float32[1, 2, 3]
                lhd["bvsv"] = VectorOfSimilarVectors([true false true; false true false])
                # other writers may store bytes other than 0 and 1:
                lhd.data_store["bvsv"][1, 2] = 0x02
                # a variable-length dataset without an LH5 datatype:
                lhd.data_store["vl"] = HDF5.VLen([Float64[i, i + 1] for i in 1:2000])
            end
            lh5open(joinpath(dir, "diskarrays_chunked.lh5"), "cw", usechunks = true) do lhd
                lhd["wf"] = VectorOfSimilarVectors(data)
            end
            lh5open(joinpath(dir, "diskarrays.lh5")) do lhd
                X = lhd["x"]
                @test X isa DiskArrays.AbstractDiskArray
                @test DiskArrays.haschunks(X) isa DiskArrays.Unchunked
                @test sum(X) ≈ sum(x) * u"keV"
                @test maximum(X) == maximum(x) * u"keV"
                @test collect(X) == x * u"keV"
                @test collect(Iterators.take(X, 3)) == x[1:3] * u"keV"
                v = view(X, 10:20)
                @test v isa DiskArrays.AbstractDiskArray
                @test collect(v) == x[10:20] * u"keV"
                bc = X .* 2
                @test bc isa DiskArrays.AbstractDiskArray
                @test collect(bc) == x * 2u"keV"
                # map is eager, as in Base, while broadcasting is lazy:
                @test map(v -> 2v, X) isa Vector
                @test map(v -> 2v, X) == x * 2u"keV"
                # explicit reads stay eager and type-stable:
                @test @inferred(X[1:10]) == x[1:10] * u"keV"
                @test @inferred(X[[1, 5, 40_000]]) == x[[1, 5, 40_000]] * u"keV"
                @test @inferred(X[3]) == x[3] * u"keV"
                mask = x .> 0.5f0
                @test @inferred(X[mask]) == x[mask] * u"keV"
                @test X[X .> 0.5f0 * u"keV"] == x[mask] * u"keV"
                # StructArrays and other to_indices users pass LogicalIndex:
                @test X[Base.to_indices(X, (mask,))...] == x[mask] * u"keV"
                @test X[Base.to_indices(X, (X .> 0.5f0 * u"keV",))...] == x[mask] * u"keV"

                W = lhd["wf"]
                @test W[3] isa DiskArrays.AbstractDiskArray
                @test sum(W[3]) == sum(data[:, 3])
                @test map(sum, view(W, 1:10)) == vec(sum(data[:, 1:10], dims = 1))
                @test W.data[[2, 5], 1:3] == data[[2, 5], 1:3]
                @test @inferred(W.data[:, [2, 5, 2999]]) == data[:, [2, 5, 2999]]
                @test W.data[:, mask[1:3000]] == data[:, mask[1:3000]]

                # empty selections give empty arrays of the right shape:
                M = W.data
                @test size(M[falses(100), :]) == (0, 3000)
                @test size(M[falses(100, 3000)]) == (0,)
                @test size(M[Int[], [1, 3000]]) == (0, 2)
                @test size(M[:, Int[]]) == (100, 0)
                @test X[Int[]] == Float32[] * u"keV"
                # reversed ranges and out-of-bounds indices behave as in Base:
                @test M[:, 10:-1:1] == data[:, 10:-1:1]
                @test X[10:-2:1] == x[10:-2:1] * u"keV"
                @test_throws BoundsError X[50_001]
                @test_throws BoundsError X[49_999:50_001]
                @test_throws BoundsError M[101, 1]
                # the chunk grid is inferable, so iteration stays typed:
                @test cumsum(X) isa Vector{<:Quantity{Float32}}
                # copyto! follows Base: the common length is copied and dest returned
                d = zeros(Float32, 5)
                @test copyto!(d, lhd["small"]) === d
                @test d == Float32[1, 2, 3, 0, 0]

                @test count(lhd["flags"]) == count(flags)
                # Bool datasets read as arrays of Bool, in every shape:
                B = lhd["flags"]
                @test B[:] isa Vector{Bool}
                @test collect(B) isa Vector{Bool}
                BV = lhd["bvsv"]
                @test eltype(flatview(BV)) == Bool
                @test BV[:] == [[true, false], [true, true], [true, false]]
                # variable-length numeric elements are converted on every read path:
                VL = lhd["vl"]
                saved = LegendHDF5IO.scatter_bulk_max_bytes[], LegendHDF5IO.scatter_bulk_max_waste[]
                LegendHDF5IO.scatter_bulk_max_bytes[] = LegendHDF5IO.scatter_bulk_max_waste[] = 0
                try
                    @test VL[[1, 3, 1500]] == [[1.0, 2.0], [3.0, 4.0], [1500.0, 1501.0]]
                finally
                    LegendHDF5IO.scatter_bulk_max_bytes[], LegendHDF5IO.scatter_bulk_max_waste[] = saved
                end

                # masks must match the axis they index:
                @test_throws BoundsError X[mask[1:10]]
                @test_throws BoundsError X[vcat(mask, false)]

                # string datasets take part in the lazy machinery like any other:
                S = lhd["strs"]
                @test eltype(S) == String
                @test first(S) == "x1"
                @test collect(view(S, 2:3)) == ["x2", "x3"]
                @test collect(S .* "!") == ["x$(i)!" for i in 1:50]

                # empty datasets have a valid chunk grid (DiskArrays itself
                # cannot view, broadcast or reduce empty arrays yet):
                E = lhd["empty"]
                @test isempty(DiskArrays.eachchunk(E))
                @test collect(E) == Float32[]
                @test isempty(DiskArrays.eachchunk(lhd["empty2d"]))
                @test collect(lhd["empty2d"]) == zeros(Int, 3, 0)
            end
            lh5open(joinpath(dir, "diskarrays_chunked.lh5")) do lhd
                W = lhd["wf"].data
                @test DiskArrays.haschunks(W) isa DiskArrays.Chunked
                @test DiskArrays.eachchunk(W) == DiskArrays.GridChunks(W, (100, 3000))
                @test sum(W) == sum(data)
            end
        end
    end

    @testset verbose=true "in-place writes" begin
        mktempdir(pwd()) do tmp
            path = joinpath(tmp, "write.lh5")
            lh5open(path, "cw") do lhd
                lhd["x"] = rand(Float32, 100) * u"keV"
                lhd["b"] = rand(Bool, 20)
                lhd["wf"] = VectorOfSimilarVectors(rand(UInt16, 10, 30))
                lhd["s"] = ["a", "bb", "ccc"]
            end
            lh5open(path, "r+") do lhd
                X, B, W = lhd["x"], lhd["b"], lhd["wf"]
                S = lhd["s"]
                S[1:2] = ["p", "q"]
                @test S[:] == ["p", "q", "ccc"]
                X[1:3] = Float32[1, 2, 3] * u"keV"
                @test X[1:3] == Float32[1, 2, 3] * u"keV"
                # values in a compatible unit are converted:
                X[4] = 5000f0 * u"eV"
                @test X[4] == 5f0 * u"keV"
                X[10:20] .= 0f0 * u"keV"
                @test all(iszero, X[10:20])
                copyto!(view(X, 30:39), fill(7f0 * u"keV", 10))
                @test all(==(7f0 * u"keV"), X[30:39])
                B[1:2] = [true, false]
                @test B[1:2] == [true, false]
                W.data[:, 2] = fill(0x0007, 10)
                @test W[2] == fill(0x0007, 10)
                @test_throws Unitful.DimensionError X[5] = 1f0
                @test_throws BoundsError X[101:102] = Float32[0, 0] * u"keV"
                # repeated indices in a write would corrupt the elements in between,
                # whichever path the write takes:
                before = X[1:3]
                v = Float32[7, 8, 9] * u"keV"
                @test_throws ArgumentError X[[1, 1, 3]] = v
                @test_throws ArgumentError X[[1, 1, 3]] .= v
                @test_throws ArgumentError view(X, [1, 1, 3]) .= v
                @test_throws ArgumentError view(X, 1:10)[[1, 1, 3]] = v
                @test_throws ArgumentError X[[CartesianIndex(1), CartesianIndex(1), CartesianIndex(3)]] = v
                @test X[1:3] == before
                M = W.data
                mbefore = M[:, :]
                @test_throws ArgumentError M[[1, 1, 3]] = fill(0x0001, 3)
                @test_throws ArgumentError M[[CartesianIndex(1, 1), CartesianIndex(1, 1), CartesianIndex(1, 2)]] = fill(0x0001, 3)
                @test_throws ArgumentError M[reshape([CartesianIndex(1, 1), CartesianIndex(1, 1), CartesianIndex(1, 2), CartesianIndex(2, 2)], 2, 2)] = fill(0x0001, 2, 2)
                @test_throws ArgumentError M[reshape([1, 1, 3, 4], 2, 2)] = fill(0x0001, 2, 2)
                @test M[:, :] == mbefore
                # an out-of-bounds write writes nothing:
                @test_throws BoundsError X[[2, 101]] = Float32[0, 0] * u"keV"
                @test X[1:3] == before
                X[[9, 8]] = Float32[90, 80] * u"keV"
                @test X[8:9] == Float32[80, 90] * u"keV"
                view(X, [5, 7]) .= 1f0 * u"keV"
                @test X[[5, 7]] == Float32[1, 1] * u"keV"
                M[[1, 3]] = UInt16[1, 3]
                @test M[1:3] == [0x0001, mbefore[2], 0x0003]
            end
            lh5open(path) do lhd
                @test lhd["x"][1:3] == Float32[1, 2, 3] * u"keV"
                @test lhd["wf"][2] == fill(0x0007, 10)
            end
            # writes into a view are checked as a whole, not chunk by chunk:
            lh5open(joinpath(tmp, "chunked.lh5"), "cw"; usechunks = true) do lhd
                lhd["c"] = Float32[1, 2, 3]
                append!(lhd["c"], Float32.(4:12))
                C = lhd["c"]
                @test_throws ArgumentError view(C, [1, 4, 4, 6]) .= 99f0
                @test_throws ArgumentError view(C, [1, 4, 7, 1]) .= 99f0
                @test_throws ArgumentError copyto!(view(C, [1, 4, 7, 1]), Float32[1, 2, 3, 4])
                @test_throws ArgumentError fill!(view(C, [2, 2]), 0f0)
                @test C[:] == Float32.(1:12)
                view(C, [1, 4, 7]) .= 99f0
                @test C[[1, 4, 7]] == fill(99f0, 3)
                # the values must fit the selection before anything is written:
                cbefore = C[:]
                @test_throws DimensionMismatch C[[2, 5, 8]] = Float32[1, 2]
                @test_throws DimensionMismatch C[[2, 5, 8]] = Float32[1, 2, 3, 4]
                @test C[:] == cbefore
                # an empty selection writes nothing:
                C[falses(12)] = Float32[]
                C[Int[]] = Float32[]
                @test C[:] == cbefore
                # values of another shape fit when their number does, as in Base:
                C[1:4] = reshape(cbefore[1:4], 2, 2)
                @test C[:] == cbefore
                # values are taken in the order of the selection, whatever its shape:
                C[reshape([8, 11, 2, 5], 2, 2)] = Float32[11, 12, 13, 14]
                @test C[[8, 11, 2, 5]] == Float32[11, 12, 13, 14]
            end
        end
    end

    @testset verbose=true "compressed writing" begin
        mktempdir(pwd()) do tmp
            for mode in (:zstd, :deflate)
                path = joinpath(tmp, "tmp_$(mode).lh5")
                lh5open(path, "cw"; compress=mode) do lhd
                    tbl = StructArray((
                        e = rand(Float32, 100),
                        wf = VectorOfSimilarVectors(rand(UInt16, 50, 100)),
                        s = string.(1:100),
                    ))
                    lhd["t"] = tbl
                    append!(lhd["t"], tbl)
                    @test lhd["t"][:] == vcat(tbl, tbl)
                    filters = HDF5.get_create_properties(lhd.data_store["t/wf"]).filters
                    @test length(filters) > 0
                end
            end
            @test_throws ArgumentError lh5open(joinpath(tmp, "x.lh5"), "cw", compress=:lzf)
            # compressed chunks are capped, so a partial read decompresses little more than it needs:
            x = rand(Float32, 2^20)
            m = rand(UInt16, 50, 2^16)
            lh5open(joinpath(tmp, "cap.lh5"), "cw"; compress = :zstd) do lhd
                lhd["x"] = x
                lhd["m"] = m
                @test HDF5.get_create_properties(lhd.data_store["x"]).chunk == (2^18,)
                @test HDF5.get_create_properties(lhd.data_store["m"]).chunk == (50, 10485)
                @test lhd["x"][2^18 - 1:2^18 + 1] == x[2^18 - 1:2^18 + 1]
                @test lhd["m"][:, 10485:10486] == m[:, 10485:10486]
            end
            # without compression the first write still defines the chunk:
            lh5open(joinpath(tmp, "nocap.lh5"), "cw"; usechunks = true) do lhd
                lhd["x"] = x
                @test HDF5.get_create_properties(lhd.data_store["x"]).chunk == (2^20,)
            end
        end
    end
    @testset verbose=true "test append functionality" begin
        mktempdir(pwd()) do tmp
            path = joinpath(tmp, "tmp.lh5")
            lh5open(path, "cw"; usechunks=true) do lhd
                @testset "append Tables" begin
                    tbl = StructArray((x = rand(10), y = rand(10)))
                    newtbl = vcat(tbl, tbl)
                    lhd["tbl"] = tbl
                    @test append!(lhd["tbl"], tbl)[:] == newtbl
                end
                @testset "append VectorOfVectors" begin
                    data = collect(eachcol(rand(55, 50)*u"m"))
                    vofv = VectorOfArrays(data)
                    newvofv = vcat(vofv, vofv)
                    lhd["vofv"] = vofv
                    @test append!(lhd["vofv"], vofv)[:] == newvofv
                end
                @testset "repeated append to VectorOfVectors" begin
                    vofv = VectorOfVectors([[1.0, 2, 3], [4.0, 5]])
                    lhd["vofv2"] = vofv
                    dest = lhd["vofv2"]
                    append!(dest, vofv)
                    append!(dest, VectorOfVectors(Vector{Float64}[]))
                    # backing array of this VectorOfVectors is longer than its content:
                    nonspanning = VectorOfVectors(collect(1.0:9.0), [4, 6, 10])
                    append!(dest, nonspanning)
                    expected = [collect.(vofv); collect.(vofv); collect.(nonspanning)]
                    @test lhd["vofv2"][:] == expected
                    clen = LegendHDF5IO.LH5Array(lhd.data_store["vofv2/cumulative_length"])[:]
                    @test clen == cumsum(length.(expected))
                end
                @testset "append VectorOfSimilarVectors" begin
                    aofa = VectorOfSimilarVectors(rand(UInt16, 55, 50)*u"m")
                    newaofa = vcat(aofa, aofa)
                    lhd["aofa"] = aofa
                    @test append!(lhd["aofa"], aofa)[:] == newaofa
                end
                @testset "append VectorOfRDWaveforms" begin
                    aofa = VectorOfSimilarVectors(rand(UInt16, 55, 50)*u"m")
                    trng = range(0.0u"μs", 10.0u"μs"; length=55)
                    waveform = ArrayOfRDWaveforms((fill(trng, 50), aofa))
                    new_waveform = vcat(waveform, waveform)
                    lhd["waveform"] = waveform
                    @test append!(lhd["waveform"], waveform)[:] == new_waveform
                end
                @testset "append converts values" begin
                    lhd["f"] = [1.0, 2.0]
                    lhd["i"] = [3, 4]
                    @test convert(Array{Float64, 1}, lhd["i"]) == [3.0, 4.0]
                    append!(lhd["f"], lhd["i"])
                    @test lhd["f"][:] == [1.0, 2.0, 3.0, 4.0]
                    lhd["e"] = [1.0, 2.0]u"keV"
                    lhd["e_ev"] = [3000.0]u"eV"
                    lhd["t"] = [1.0]u"s"
                    append!(lhd["e"], lhd["e_ev"])
                    @test lhd["e"][:] == [1.0, 2.0, 3.0]u"keV"
                    @test_throws Unitful.DimensionError append!(lhd["e"], lhd["t"])
                    @test lhd["e"][:] == [1.0, 2.0, 3.0]u"keV"
                end
                @testset "append through a stale ragged-array handle" begin
                    lhd["vofv_stale"] = VectorOfVectors([[1, 2], [3]])
                    h1, h2 = lhd["vofv_stale"], lhd["vofv_stale"]
                    append!(h1, VectorOfVectors([[4, 5]]))
                    # h2 still holds the element pointers from before that append:
                    @test_throws ArgumentError append!(h2, VectorOfVectors([[6]]))
                    @test lhd["vofv_stale"][:] == [[1, 2], [3], [4, 5]]
                end
                @testset "append through a lazy table after column changes" begin
                    lhd["tbl_cols"] = StructArray((a = [1, 2], b = [3, 4]))
                    t = lhd["tbl_cols"]
                    t[:]  # opens and caches the columns
                    delete_entry!(lhd, "tbl_cols/b")
                    add_entries!(lhd, "tbl_cols", StructArray((b = [30, 40],)))
                    append!(t, StructArray((a = [5], b = [50])))
                    @test lhd["tbl_cols"][:] == StructArray((a = [1, 2, 5], b = [30, 40, 50]))
                    @test t[:] == lhd["tbl_cols"][:]
                    lhd["tbl_vov"] = StructArray((v = VectorOfVectors([[1], [2, 3]]),))
                    orig = lhd["tbl_vov"]
                    orig[:]
                    append!(lhd["tbl_vov"], StructArray((v = VectorOfVectors([[4, 5]]),)))
                    append!(orig, StructArray((v = VectorOfVectors([[6, 7]]),)))
                    @test lhd["tbl_vov"][:] == StructArray((v = VectorOfVectors([[1], [2, 3], [4, 5], [6, 7]]),))
                end
                @testset "append tables with waveform and nested columns" begin
                    trng = range(0.0u"μs", 10.0u"μs"; length = 4)
                    wf = ArrayOfRDWaveforms((fill(trng, 2), VectorOfSimilarVectors(rand(UInt16, 4, 2))))
                    tbl = StructArray((e = [1.0, 2.0], wf = wf, sub = StructArray((v = VectorOfVectors([[1, 2], [3]]),))))
                    lhd["tbl_wf"] = tbl
                    append!(lhd["tbl_wf"], tbl)
                    @test lhd["tbl_wf"][:] == vcat(tbl, tbl)
                    # columns read eagerly cannot be appended to on disk:
                    lhd["tbl_svec"] = StructArray((s = [SVector(1, 2), SVector(3, 4)], a = [1, 2]))
                    @test_throws ArgumentError append!(lhd["tbl_svec"], StructArray((s = [SVector(5, 6)], a = [3])))
                    @test lhd["tbl_svec"][:].a == [1, 2]
                    # the StructArray conversion appends waveform columns with their time axes:
                    lhd["tbl_wf2"] = StructArray((e = [1.0, 2.0], wf = wf))
                    s = StructArray(lhd["tbl_wf2"])
                    append!(s, s)
                    @test lhd["tbl_wf2"][:] == vcat(StructArray((e = [1.0, 2.0], wf = wf)), StructArray((e = [1.0, 2.0], wf = wf)))
                    # nested tables are checked against their stored columns as well:
                    lhd["tbl_nest"] = StructArray((a = [1, 2], sub = StructArray((b = [3, 4],))))
                    t = lhd["tbl_nest"]
                    t[:]
                    add_entries!(lhd, "tbl_nest/sub", StructArray((c = [5, 6],)))
                    @test_throws ArgumentError append!(t, StructArray((a = [7], sub = StructArray((b = [8],)))))
                    @test length(lhd["tbl_nest/a"]) == 2 && length(lhd["tbl_nest/sub/b"]) == 2
                end
                @testset "append from aliased and inconsistent sources" begin
                    lhd["tbl_alias"] = StructArray((a = [1, 2], b = [10, 20]))
                    t = lhd["tbl_alias"]
                    append!(t, StructArray((a = t.b, b = t.a)))
                    @test lhd["tbl_alias"][:] == StructArray((a = [1, 2, 10, 20], b = [10, 20, 1, 2]))
                    @test_throws ArgumentError append!(t, (a = [5, 6, 7], b = [8]))
                    @test length(lhd["tbl_alias"]) == 4
                    lhd["tbl_src"] = StructArray((v = VectorOfVectors([[1, 2], [3]]), a = [1, 2]))
                    stale = lhd["tbl_src"]
                    stale[:]
                    append!(lhd["tbl_src"], StructArray((v = VectorOfVectors([[4]]), a = [3])))
                    # the ragged column of the stale handle is shorter than its other column:
                    @test_throws ArgumentError append!(lhd["tbl_src"], stale)
                    @test length(lhd["tbl_src"]) == 3
                end
                @testset "append through a table handle keeps its StructArray coherent" begin
                    lhd["tbl_coh"] = StructArray((a = [1.0, 2.0], v = VectorOfVectors([[1], [2, 3]])))
                    t = lhd["tbl_coh"]
                    typed = StructArray(t)
                    append!(t, StructArray((a = [3.0], v = VectorOfVectors([[4, 5]]))))
                    @test length(typed) == 3
                    @test typed[3] == (a = 3.0, v = [4, 5])
                    @test StructArray(t) === typed
                    # appends through a StructArray conversion check the stored columns as well:
                    add_entries!(lhd, "tbl_coh", StructArray((c = [7, 8, 9],)))
                    @test_throws ArgumentError append!(typed, StructArray((a = [4.0], v = VectorOfVectors([[6]]))))
                    @test lhd["tbl_coh"][:].c == [7, 8, 9]
                end
                @testset "rejected appends leave datasets unchanged" begin
                    lhd["m"] = rand(4, 3)
                    @test_throws DimensionMismatch append!(lhd["m"], ones(2, 2))
                    @test size(lhd["m"]) == (4, 3)
                    lhd["vv"] = VectorOfVectors([[10.0, 20.0], [30.0]])
                    # views cannot grow their element pointers:
                    @test_throws ArgumentError append!(view(lhd["vv"], 1:2), VectorOfVectors([[1.0, 2.0]]))
                    @test lhd["vv"][:] == [[10.0, 20.0], [30.0]]
                    # table columns holding plain vectors of vectors are accepted:
                    lhd["tbl_vv"] = StructArray((v = VectorOfVectors([[10.0, 20.0], [30.0]]),))
                    append!(lhd["tbl_vv"], StructArray((v = [[1.0], [2.0, 3.0]],)))
                    @test lhd["tbl_vv"][:].v == [[10.0, 20.0], [30.0], [1.0], [2.0, 3.0]]
                    # cumulative lengths that overflow their element type are rejected up front:
                    lhd["vv_narrow"] = VectorOfVectors([[1, 2], [3]])
                    grp = lhd.data_store["vv_narrow"]
                    HDF5.delete_object(grp, "cumulative_length")
                    grp["cumulative_length"] = UInt8[2, 3]
                    LegendHDF5IO.setdatatype!(grp["cumulative_length"], Vector{UInt8})
                    @test_throws InexactError append!(lhd["vv_narrow"], VectorOfVectors([collect(1:300)]))
                    @test lhd["vv_narrow"][:] == [[1, 2], [3]]
                end
                @testset "adding columns after appends" begin
                    lhd["tbl_grow"] = StructArray((v = VectorOfVectors([[1], [2, 3]]),))
                    append!(lhd["tbl_grow"], StructArray((v = VectorOfVectors([[4]]),)))
                    @test_throws DimensionMismatch add_entries!(lhd, "tbl_grow", StructArray((c = [7, 8],)))
                    add_entries!(lhd, "tbl_grow", StructArray((c = [7, 8, 9],)))
                    @test lhd["tbl_grow"][:].c == [7, 8, 9]
                end
                @testset "append waveforms to themselves and ragged waveforms" begin
                    trng = range(0.0u"μs", 10.0u"μs"; length = 4)
                    wf = ArrayOfRDWaveforms((fill(trng, 2), VectorOfSimilarVectors(rand(UInt16, 4, 2))))
                    lhd["wf_self"] = wf
                    w = lhd["wf_self"]
                    append!(w, w)
                    @test lhd["wf_self"][:] == vcat(wf, wf)
                    # a handle whose time axes predate an append elsewhere must be reopened:
                    lhd["wf_two"] = wf
                    h1, h2 = lhd["wf_two"], lhd["wf_two"]
                    append!(h1, wf)
                    @test_throws ArgumentError append!(h2, h2)
                    @test lhd["wf_two"][:] == vcat(wf, wf)
                    times = [range(0.0u"μs", 10.0u"μs"; length = n) for n in (3, 5)]
                    ragged = ArrayOfRDWaveforms((times, VectorOfVectors([rand(UInt16, 3), rand(UInt16, 5)])))
                    lhd["wf_ragged"] = ragged
                    append!(lhd["wf_ragged"], ragged)
                    @test lhd["wf_ragged"][:] == vcat(ragged, ragged)
                end
                @testset "append strings" begin
                    lhd["strs"] = ["a", "bb"]
                    append!(lhd["strs"], ["ccc"])
                    @test lhd["strs"][:] == ["a", "bb", "ccc"]
                end
                @testset "append through a stale handle" begin
                    tbl = StructArray((x = [1, 2], y = [3, 4]))
                    lhd["tbl_stale"] = tbl
                    stale = lhd["tbl_stale"]
                    add_entries!(lhd, "tbl_stale", StructArray((z = [5, 6],)))
                    # the handle does not know column z, appending through it would corrupt the table:
                    @test_throws ArgumentError append!(stale, tbl)
                    @test lhd["tbl_stale"][:] == StructArray((x = [1, 2], y = [3, 4], z = [5, 6]))
                end
                @testset "append through a StructArray of replaced columns" begin
                    lhd["tbl_sa"] = StructArray((a = [1, 2], b = [3, 4]))
                    t = lhd["tbl_sa"]
                    s = StructArray((a = t.a, b = t.b))
                    delete_entry!(lhd, "tbl_sa/b")
                    add_entries!(lhd, "tbl_sa", StructArray((b = [30, 40],)))
                    # column b of s is the deleted dataset, appending through s would
                    # leave the stored columns at different lengths:
                    @test_throws ArgumentError append!(s, StructArray((a = [5], b = [50])))
                    @test lhd["tbl_sa"][:] == StructArray((a = [1, 2], b = [30, 40]))
                end
                @testset "append through a StructArray of stale ragged columns" begin
                    lhd["tbl_rag"] = StructArray((a = [1, 2], v = VectorOfVectors([[1], [2, 3]])))
                    t = lhd["tbl_rag"]
                    s = StructArray((a = t.a, v = t.v))
                    append!(lhd["tbl_rag"], StructArray((a = [3], v = VectorOfVectors([[4]]))))
                    # the element pointers of s.v predate that append, and column a
                    # must not grow before this is noticed:
                    @test_throws ArgumentError append!(s, StructArray((a = [5], v = VectorOfVectors([[6]]))))
                    stored = lhd["tbl_rag"][:]
                    @test stored.a == [1, 2, 3] && stored.v == [[1], [2, 3], [4]]
                end
                @testset "append to itself" begin
                    vofv = VectorOfVectors([[1, 2], [3]])
                    lhd["vofv_self"] = vofv
                    dest = lhd["vofv_self"]
                    append!(dest, dest)
                    @test lhd["vofv_self"][:] == vcat(vofv, vofv)
                    tbl = StructArray((a = [1, 2], w = vofv))
                    lhd["tbl_self"] = tbl
                    t = lhd["tbl_self"]
                    append!(t, t)
                    @test lhd["tbl_self"][:] == vcat(tbl, tbl)
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
                    tbl = StructArray((x = x, y = y))
                    lhd["tbl"] = tbl
                    @testset verbose=true "deleting entry" begin
                        @test delete_entry!(lhd, "tbl/y") |> isnothing
                        @test lhd["tbl"][:] == StructArray((x = x,))
                    end
                    @testset "adding entry" begin
                        @test add_entries!(lhd, "tbl", StructArray((y = y,))) |> isnothing
                        @test lhd["tbl"][:] == tbl
                    end
                    @testset "adding entries repeatedly" begin
                        lhd["tbl_dest"] = StructArray((a = [1, 2], b = [3, 4]))
                        add_entries!(lhd, "tbl_dest", StructArray((c = [5, 6],)))
                        add_entries!(lhd, "tbl_dest", StructArray((d = [7, 8],)))
                        @test propertynames(lhd["tbl_dest"]) == (:a, :b, :c, :d)
                        # existing names are rejected before anything is written:
                        @test_throws ArgumentError add_entries!(lhd, "tbl_dest", StructArray((e = [1, 2], a = [0, 0])))
                        @test propertynames(lhd["tbl_dest"]) == (:a, :b, :c, :d)
                        @test !haskey(lhd.data_store, "tbl_dest/e")
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
                @testset verbose=true "error paths" begin
                    delete_entry!(lhd, "nt/y")
                    @test_throws ArgumentError delete_entry!(lhd, "nt/x")
                    @test_throws DimensionMismatch add_entries!(lhd, "tbl", StructArray((z = rand(3),)))
                    # a missing entry leaves the table and its datatype alone:
                    @test_throws ArgumentError delete_entry!(lhd, "tbl/nope")
                    @test lhd["tbl"] isa LegendHDF5IO.LH5LazyTable
                    @test propertynames(lhd["tbl"]) == (:x, :y)
                    # columns are added to a table as a table, not as a NamedTuple:
                    @test_throws ArgumentError add_entries!(lhd, "tbl", (c = rand(length(lhd["tbl"])),))
                    @test propertynames(lhd["tbl"]) == (:x, :y)
                    n = length(lhd["tbl"])
                    @test_throws ArgumentError add_entries!(lhd, "tbl", StructArray((m = reshape(rand(n), 1, n),)))
                    @test propertynames(lhd["tbl"]) == (:x, :y)
                    @test_throws ArgumentError add_entries!(lhd, "nt", StructArray((c = [30],)))
                    @test lhd["nt"] isa NamedTuple
                end
            end
        end
    end
    @testset "lazy tables" begin
        mktempdir() do dir
            fn = joinpath(dir, "lazy.lh5")
            n = 64
            a = rand(Float32, n)
            b = rand(Int32, n)
            vov = VectorOfVectors([rand(Float32, 3) for _ in 1:n])
            sv = [SVector(1.0, 2.0) * i for i in 1:n]
            lh5open(fn, "w") do lhd
                lhd["tbl"] = StructArray((a = a, b = b, vov = vov))
                # a table of tables, and one mixing eagerly read and disk-backed columns:
                lhd["nested"] = StructArray((x = StructArray((a = a,)), y = StructArray((b = b, vov = vov))))
                lhd["mixed"] = StructArray((sv = sv, vov = vov))
            end

            opened(tbl) = keys(LegendHDF5IO._table_columns(tbl))

            lh5open(fn, "r") do lhd
                tbl = lhd["tbl"]
                @test tbl isa LH5LazyTable
                # opening the table must not open any column:
                @test isempty(opened(tbl))
                @test propertynames(tbl) == (:a, :b, :vov)
                @test hasproperty(tbl, :a) && !hasproperty(tbl, :nope)

                @test tbl.a[:] == a
                @test Set(opened(tbl)) == Set([:a])
                @test tbl.b[:] == b
                @test tbl.vov[:] == vov
                @test Set(opened(tbl)) == Set([:a, :b, :vov])
                @test_throws ArgumentError tbl.nope
            end

            fn_wide = joinpath(dir, "wide.lh5")
            names = Tuple(Symbol.("c", 1:32))
            lh5open(fn_wide, "cw") do lhd
                lhd["wide"] = StructArray(NamedTuple{names}(ntuple(i -> fill(i, 8), 32)))
            end
            lh5open(fn_wide, "r") do lhd
                # columns opened concurrently all end up in the cache, once each:
                tbl = lhd["wide"]
                @sync for name in names
                    Threads.@spawn getproperty(tbl, name)
                end
                @test Set(opened(tbl)) == Set(names)
                @test all(getproperty(tbl, names[i])[:] == fill(i, 8) for i in 1:32)
            end

            lh5open(fn, "r") do lhd
                tbl = lhd["tbl"]
                @test length(tbl) == n
                # the row count needs one column, not all of them:
                @test length(opened(tbl)) == 1
                @test size(tbl) == (n,)
                @test !isempty(tbl)
                @test firstindex(tbl) == 1 && lastindex(tbl) == n
            end

            lh5open(fn, "r") do lhd
                tbl = lhd["tbl"]
                str = sprint(show, tbl)
                @test occursin("3 columns", str)
                @test all(name -> occursin(name, str), ("a", "b", "vov"))
                @test sprint(show, MIME"text/plain"(), tbl) == str
                # showing a table must not open any of its columns:
                @test isempty(opened(tbl))
            end

            lh5open(fn, "r") do lhd
                tbl = lhd["tbl"]
                @test Tables.istable(tbl)
                @test Tables.columnaccess(tbl)
                @test Tables.columnnames(tbl) == (:a, :b, :vov)
                @test Tables.getcolumn(tbl, :a)[:] == a
                @test Tables.getcolumn(tbl, 2)[:] == b
                # the column types are only known once columns are opened:
                @test isnothing(Tables.schema(tbl))
                @test Tables.columnindex(tbl, :b) == 2
                @test Tables.columnindex(tbl, :nope) == 0
                @test Tables.columntype(tbl, :b) == Int32
                @test Tables.subset(tbl, 2:3) == tbl[2:3]
                @test Tables.subset(tbl, 2:3; viewhint = true) == tbl[2:3]
                # Tables.jl consumers get nested tables as arrays, column access keeps them lazy:
                nested = lhd["nested"]
                @test Tables.getcolumn(nested, :y) isa StructArray
                @test nested.y isa LH5LazyTable
            end

            lh5open(fn, "r") do lhd
                # concatenating lazy tables is not supported yet, read them first:
                tbl = lhd["tbl"]
                @test_throws ArgumentError vcat(tbl, tbl)
                @test_throws ArgumentError reduce(vcat, [tbl, tbl])
                @test_throws ArgumentError vcat(tbl[:], tbl)
                @test_throws ArgumentError vcat(tbl, tbl[:])
                @test_throws ArgumentError reduce(vcat, [tbl, tbl]; init = tbl[1:0])
                @test_throws ArgumentError hcat(tbl, tbl)
                @test_throws ArgumentError cat(tbl, tbl; dims = 1)
                @test length(vcat(tbl[:], tbl[:])) == 2n
            end

            lh5open(fn, "r") do lhd
                tbl = lhd["tbl"]
                # selecting columns must not open the ones it skipped:
                sel = map(name -> Tables.getcolumn(tbl, name), (:a, :b))
                @test sel[1][:] == a
                @test Set(opened(tbl)) == Set([:a, :b])
                @test length(Tables.columntable(tbl)) == 3
                @test Set(opened(tbl)) == Set([:a, :b, :vov])
            end

            lh5open(fn, "r") do lhd
                tbl = lhd["tbl"]
                typed = StructArray(tbl)
                @test typed isa LegendHDF5IO.LH5Table
                # repeated conversion must reuse the same table:
                @test StructArray(tbl) === typed
                @test tbl[1:4].a == a[1:4]
                @test collect(tbl.a[1:4]) == a[1:4]
                @test (x -> 2x).(tbl.a) == 2 .* a
                @test first(tbl).a == a[1]
                @test tbl == typed
                @test tbl == tbl[:] && tbl[:] == tbl
                @test hash(tbl) == hash(tbl[:])
                @test collect(tbl) == collect(tbl[:])
                @test eltype(first(tbl, 2)) <: NamedTuple
                # a lazy mask selects rows like an in-memory one, and masks must fit the table:
                mask = tbl.a .> 0.5f0
                @test tbl[mask].a == a[a .> 0.5f0]
                @test_throws BoundsError tbl[[true]]
                @test_throws BoundsError tbl[trues(n + 1)]
                # keys, eachindex and pairs index rows, as iteration does:
                @test keys(tbl) == 1:n
                @test [tbl[i] for i in eachindex(tbl)] == collect(tbl)
                @test length(collect(pairs(tbl))) == n
            end

            lh5open(fn, "r") do lhd
                nested = lhd["nested"]
                @test nested.y isa LH5LazyTable
                @test StructArray(nested)[1:3].y.vov == vov[1:3]
                @test nested[2].y.b == b[2]
                @test readdata(lhd.data_store, "nested").x.a == a
                # slices rebuild the row type from the sliced columns:
                mixed = lhd["mixed"]
                rows = mixed[1:3]
                @test rows[2].vov == vov[2]
                @test rows.sv == sv[1:3]
                @test mixed[[1, n]].vov == vov[[1, n]]
            end

            # readdata and writing a lazy table back out:
            lh5open(fn, "r") do lhd
                eager = readdata(lhd.data_store, "tbl")
                @test eager isa StructVector
                @test eager.a == a && eager.vov == vov
            end
            lh5open(joinpath(dir, "copy.lh5"), "w") do dest
                lh5open(fn, "r") do src
                    dest["tbl"] = src["tbl"]
                    dest["nested"] = src["nested"]
                end
            end
            lh5open(joinpath(dir, "copy.lh5"), "r") do lhd
                @test lhd["tbl"].a[:] == a
                @test lhd["tbl"].vov[:] == vov
                @test lhd["nested"].y.vov[:] == vov
            end
        end
    end
end
