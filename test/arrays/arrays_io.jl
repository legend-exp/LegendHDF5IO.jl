using Unitful
using LegendHDF5IO: readdata, writedata, lh5open
using HDF5

@testset "Arrays I/O" begin
    
    a = rand(typeof(1.0u"keV"), 40)
    h5open("array_test.h5", "w") do h5f
        writedata(h5f, "array_in_keV", a)
        writedata(h5f, "array_no_units", ustrip.(a))
    end

    @testset "readdata" begin
        h5open("array_test.h5", "r") do h5f
            @test readdata(h5f, "array_in_keV") == a
            @test readdata(h5f, "array_no_units") == ustrip.(a)
        end
    end

    @testset "LHDataStore" begin
        lh5open("array_test.h5") do h5f
            @test h5f["array_in_keV"][:] == a
            @test h5f["array_no_units"][:] == ustrip.(a)
        end
    end
end
