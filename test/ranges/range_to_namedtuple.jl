using LegendHDF5IO: _nt_to_range, _range_to_nt

@testset "Range <-> NamedTuple" begin
    # Readers know the length from the data the range belongs to:
    roundtrip(r) = _nt_to_range(_range_to_nt(r), length(r))
    @testset "UnitRange" begin
        @test roundtrip(1:10) == 1:10
    end
    @testset "StepRange" begin
        @test roundtrip(1:2:10) == 1:2:10
    end
    @testset "LinRange" begin
        @test roundtrip(LinRange(1, 10, 3)) == LinRange(1, 10, 3)
    end
    @testset "StepRangeLen" begin
        @test roundtrip(StepRangeLen(1.0:0.3:1.7)) == StepRangeLen(1.0:0.3:1.7)
    end
end