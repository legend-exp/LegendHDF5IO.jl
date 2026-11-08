# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

# A setting defaults to the preference of its name in the active
# LocalPreferences.toml, or to the environment variable LEGENDHDF5IOJL_<NAME>,
# which wins (see the Settings section of the guide):
_setting(::Type{T}, name::String, default; kwargs...) where {T} =
    ScopedSetting{T}(GetPreference{T}(@__MODULE__, name, default; kwargs...))

# The value "auto" leaves the choice to the file:
_auto_or(::Type{T}) where {T} = x -> x == "auto" ? nothing : x isa AbstractString ? parse(T, x) : convert(T, x)

# As for the compress argument of lh5open, true stands for :zstd:
_compress_or_bool(x) = x in (true, "true") ? :zstd : x in (false, "false") ? :none : Symbol(x)

"""
    LegendHDF5IO.usechunks[]

Default of the `usechunks` argument of [`lh5open`](@ref), `false`: whether
newly written array datasets are chunked, so that `append!` can extend them.
A setting, see the [Settings](@ref) section of the guide.
"""
const usechunks = _setting(Bool, "usechunks", false)

"""
    LegendHDF5IO.compress[]

Default of the `compress` argument of [`lh5open`](@ref), `:none`: the
compression of newly written array datasets, `:zstd` or `:deflate`. A
setting, see the [Settings](@ref) section of the guide, where "true" and
"false" stand for `:zstd` and `:none`.
"""
const compress = _setting(Symbol, "compress", :none; f_conv = _compress_or_bool)

"""
    LegendHDF5IO.max_chunk_bytes[]

Size in bytes up to which the first array written to a chunked, uncompressed
dataset sets the chunk size, 16 MiB by default. A larger first write is
split evenly into chunks of at most this size, so it wastes no space, and an
append then wastes at most one partially filled chunk. The default suits the
block sizes of cluster file systems and does not depend on the file system,
so that files written on one system read well on others. A setting, see the
[Settings](@ref) section of the guide.
"""
const max_chunk_bytes = _setting(Int, "max_chunk_bytes", 2^24)

"""
    LegendHDF5IO.max_compressed_chunk_bytes[]

Size in bytes up to which compressed datasets are chunked, 1 MiB by default.
A compressed chunk is decompressed as a whole, so compressed chunks stay
small enough that a partial read decompresses little more than it needs. A
setting, see the [Settings](@ref) section of the guide.
"""
const max_compressed_chunk_bytes = _setting(Int, "max_compressed_chunk_bytes", 2^20)

"""
    LegendHDF5IO.scatter_bulk_max_bytes[]

Size up to which a scattered read reads the whole index span its events
cover, instead of reading the events themselves.

Reading the span costs one read operation. Reading the events costs one
scattered selection, or one read per run of adjacent events, depending on
element type, rank and layout. With the default `nothing` the limit depends
on the file system the file lives on: 1 MiB on local storage and 8 MiB on
network file systems such as GPFS, NFS or Lustre, where fewer and larger
reads pay off. Variable-length elements like strings are always read
selectively. A setting, see the [Settings](@ref) section of the guide, where
"auto" stands for `nothing`.

See also [`LegendHDF5IO.scatter_bulk_max_waste`](@ref).
"""
const scatter_bulk_max_bytes = _setting(Union{Nothing, Int}, "scatter_bulk_max_bytes", nothing; f_conv = _auto_or(Int))

"""
    LegendHDF5IO.scatter_bulk_max_waste[]

Factor by which a scattered read may exceed the requested amount of data
when it reads the whole index span its events cover.

Densely covered spans are read as a whole however large they are, so
raising this trades reading more data for fewer read operations. With the
default `nothing` the factor is 4 on local storage and 16 on network file
systems. Spans up to [`LegendHDF5IO.scatter_bulk_max_bytes`](@ref) are read
as a whole regardless. A setting, see the [Settings](@ref) section of the
guide, where "auto" stands for `nothing`.
"""
const scatter_bulk_max_waste = _setting(Union{Nothing, Float64}, "scatter_bulk_max_waste", nothing; f_conv = _auto_or(Float64))

"""
    LegendHDF5IO.max_open[]

Default of the `max_open` argument of [`lh5open`](@ref) for file chains, 64:
the number of files a chain keeps open at a time. A setting, see the
[Settings](@ref) section of the guide.
"""
const max_open = _setting(Int, "max_open", 64)

"""
    LegendHDF5IO.rows_per_block[]

Number of rows that iteration over a lazy table reads at a time, 4096 by
default. A setting, see the [Settings](@ref) section of the guide.
"""
const rows_per_block = _setting(Int, "rows_per_block", 4096)
