using StatsBase
using LegendHDF5IO
using LegendTestData
using Test
using LegendHDF5IO: _nt_to_histogram, _histogram_to_nt

@testset "Histogram <-> NamedTuple" begin
    h = fit(Histogram, rand(10))
    @test _nt_to_histogram(_histogram_to_nt(h)) == h
    h = fit(Histogram, (rand(10), rand(10)))
    @test _nt_to_histogram(_histogram_to_nt(h)) == h
    h = fit(Histogram, (rand(10), rand(10)), (0:0.2:1, Float64[0, 0.5, 1]))
    @test _nt_to_histogram(_histogram_to_nt(h)) == h
    # decimal steps whose multiples miss the last edge in the last bit:
    h = fit(Histogram, ([0.15, 0.55, 0.95], [0.05, 0.25, 0.45]), (0.0:0.2:1.0, 0.0:0.1:0.6))
    @test _nt_to_histogram(_histogram_to_nt(h)) == h
    h = fit(Histogram, [0.1, 0.5, 2.0], 0.0:0.7:2.1)
    @test _nt_to_histogram(_histogram_to_nt(h)) == h
    # integer edges stay integer:
    h = fit(Histogram, 1:10, 0:2:10)
    @test _nt_to_histogram(_histogram_to_nt(h)) == h && eltype(only(_nt_to_histogram(_histogram_to_nt(h)).edges)) == Int
end

@testset "Histogram IO" begin 
    fn = joinpath(legend_test_data_path(), "data", "lh5", "lgdo-histograms.lh5")
    lh5open(fn) do h5
        names = keys(h5)
        for name in names
            @testset "$name" begin
                h = @test_nowarn h5[name]
                @test h isa Histogram
            end
        end
    end
end