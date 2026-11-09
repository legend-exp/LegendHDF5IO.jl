# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

using Test
using LegendHDF5IO

using HDF5
using ScopedSettings
using StructArrays

using LegendHDF5IO: max_chunk_bytes, max_compressed_chunk_bytes, usechunks, compress, max_open,
    rows_per_block, scatter_bulk_max_bytes, scatter_bulk_max_waste

@testset "settings" begin
    # the tests assume the built-in defaults, which environment variables override:
    withenv(["LEGENDHDF5IOJL_$k" => nothing for k in ("USECHUNKS", "COMPRESS", "MAX_CHUNK_BYTES",
            "MAX_COMPRESSED_CHUNK_BYTES", "SCATTER_BULK_MAX_BYTES", "SCATTER_BULK_MAX_WASTE", "MAX_OPEN",
            "ROWS_PER_BLOCK")]...) do
        @test max_chunk_bytes[] == 2^24 && max_compressed_chunk_bytes[] == 2^20
        @test usechunks[] === false && compress[] === :none
        @test max_open[] == 64 && rows_per_block[] == 4096
        @test scatter_bulk_max_bytes[] === nothing && scatter_bulk_max_waste[] === nothing

        # environment variables set the defaults, "auto" leaves a choice to the file system:
        withenv("LEGENDHDF5IOJL_USECHUNKS" => "true", "LEGENDHDF5IOJL_MAX_CHUNK_BYTES" => "8192",
            "LEGENDHDF5IOJL_COMPRESS" => "zstd", "LEGENDHDF5IOJL_SCATTER_BULK_MAX_BYTES" => "1234",
            "LEGENDHDF5IOJL_SCATTER_BULK_MAX_WASTE" => "auto") do
            @test usechunks[] === true && max_chunk_bytes[] == 8192 && compress[] === :zstd
            @test scatter_bulk_max_bytes[] == 1234 && scatter_bulk_max_waste[] === nothing
        end
        # as for lh5open, true and false stand for :zstd and :none:
        @test withenv(() -> compress[], "LEGENDHDF5IOJL_COMPRESS" => "true") === :zstd
        @test withenv(() -> compress[], "LEGENDHDF5IOJL_COMPRESS" => "false") === :none
        # a global override holds until default_value restores the default:
        scatter_bulk_max_bytes[] = 5
        @test scatter_bulk_max_bytes[] == 5
        scatter_bulk_max_bytes[] = default_value
        @test scatter_bulk_max_bytes[] === nothing

        mktempdir() do dir
            x = rand(1000)
            chunk(lhd, name) = HDF5.get_create_properties(lhd.data_store[name]).chunk
            # the first write is split evenly into chunks of at most the cap:
            @with max_chunk_bytes => 4096 begin
                lh5open(joinpath(dir, "cap.lh5"), "w"; usechunks = true) do lhd
                    lhd["x"] = x
                    lhd["small"] = x[1:100]
                    @test chunk(lhd, "x") == (500,) && chunk(lhd, "small") == (100,)
                    append!(lhd["x"], x)
                    @test lhd["x"][:] == vcat(x, x)
                end
            end
            @with max_compressed_chunk_bytes => 4096 begin
                lh5open(joinpath(dir, "ccap.lh5"), "w"; compress = :zstd) do lhd
                    lhd["x"] = x
                    # a chunk holds strings as 16-byte references to their contents:
                    lhd["s"] = string.(1:1000)
                    @test chunk(lhd, "x") == (500,) && chunk(lhd, "s") == (250,)
                end
            end
            # the defaults of lh5open's write options are settings:
            props(lhd) = HDF5.get_create_properties(lhd.data_store["x"])
            lh5open(joinpath(dir, "defaults.lh5"), "w") do lhd
                lhd["x"] = x
                @test props(lhd).layout == :contiguous
            end
            @with usechunks => true begin
                lh5open(joinpath(dir, "defaults.lh5"), "w") do lhd
                    lhd["x"] = x
                    @test props(lhd).layout == :chunked && isempty(props(lhd).filters)
                    append!(lhd["x"], x)
                    @test lhd["x"][:] == vcat(x, x)
                end
            end
            @with compress => :deflate begin
                lh5open(joinpath(dir, "defaults.lh5"), "w") do lhd
                    lhd["x"] = x
                    @test any(f -> f isa HDF5.Filters.Deflate, props(lhd).filters)
                end
            end
            # iteration reads blocks of rows_per_block rows, and chains keep max_open files open:
            fns = [joinpath(dir, "part$k.lh5") for k in 1:3]
            foreach(fn -> lh5open(lhd -> lhd["t"] = StructArray((x = x,)), fn, "w"), fns)
            @with rows_per_block => 300 max_open => 1 begin
                lh5open(fns) do fc
                    t = fc["t"]
                    @test length(first(LegendHDF5IO._next_block(t, LegendHDF5IO._first_block_state(t)))) == 300
                    @test sum(r -> r.x, t) ≈ 3 * sum(x)
                    @test length(fc.files) <= 1
                end
                lh5open(first(fns)) do lhd
                    t = lhd["t"]
                    @test length(first(LegendHDF5IO._next_block(t, LegendHDF5IO._first_block_state(t)))) == 300
                end
            end
        end
    end
end
