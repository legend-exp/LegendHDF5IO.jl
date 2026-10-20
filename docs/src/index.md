# LegendHDF5IO.jl

LegendHDF5IO reads and writes HDF5 files that follow the
[LEGEND LH5 data format](https://legend-exp.github.io/legend-data-format-specs/dev/hdf5/).
Such files store arrays, tables, waveforms, histograms and nested structures
together with `datatype` and `units` attributes, and are interoperable with
the Python package [legend-pydataobj](https://github.com/legend-exp/legend-pydataobj).

## Getting started

Open a file with [`lh5open`](@ref), which returns an [`LHDataStore`](@ref).
Write with `setindex!`, read with `getindex`:

```julia
using LegendHDF5IO, Unitful

lh5open("data.lh5", "cw") do lhd
    lhd["energy"] = rand(1000) .* u"keV"
    lhd["metadata"] = (detector = "V01234A", threshold = 5.0u"keV")
end

lh5open("data.lh5") do lhd
    E = lhd["energy"]       # lazy, disk-backed LH5Array
    E_first = E[1:100]      # reads only the first 100 elements
    E_some = E[[1, 7, 42]]  # scattered reads are efficient, too
    E_all = E[:]            # reads the full array
    lhd["metadata"]         # NamedTuples are read back directly
end
```

Datasets are wrapped as lazy [`LH5Array`](@ref)s, so slicing reads only the
requested part of the data from disk. Arrays of `Enum`s, `StaticVector`s,
`NTuple`s and `Measurement`s are the exception and are read eagerly.
`LH5Array` implements the
[DiskArrays.jl](https://github.com/JuliaIO/DiskArrays.jl) interface: views,
iteration and reductions like `sum` read the data chunk by chunk, and
broadcasts over `LH5Array`s are lazy and evaluated the same way when collected
or written. Arithmetic, `vcat`, `reverse` and `diff` return such lazy arrays
as well, which read from the file on access and stop working once it is
closed. `map` is eager, as in Base. Broadcasts that expand an `LH5Array` along
a singleton dimension or add dimensions are not supported by DiskArrays yet;
read such data with `[:]` first.

Datasets opened for writing can be modified in place within their current
size (`E[1:10] = ...`); `append!` grows them. Nested structures (`NamedTuple`s,
tables, waveforms) are represented as HDF5 groups.

### Tables and waveforms

Tables and vectors of `RDWaveform`s round-trip through their LH5
representation. Tables are read as [`LH5LazyTable`](@ref)s, which open a
column only when it is accessed, so reading a few columns of a wide table costs
a few reads instead of one per column. They are vectors of `NamedTuple` rows
and satisfy the Tables.jl interface, with `Tables.columns(tbl)` an ordered
dictionary of the columns. Column-wise broadcasts, such as
[PropertyFunctions](https://github.com/oschulz/PropertyFunctions.jl) over the
table, open only the columns they use, while treating the table as an array
opens all of them. Anything that satisfies the Tables.jl column interface can
be written:

```julia
using StructArrays, ArraysOfArrays, RadiationDetectorSignals

tbl = StructArray((
    evtno = collect(1:100),
    energy = rand(100) .* u"keV",
    samples = VectorOfVectors([rand(-5:5, 50) for _ in 1:100]),
))

wfs = ArrayOfRDWaveforms((
    fill(range(0.0u"μs", 10.0u"μs", length = 1000), 100),
    VectorOfSimilarVectors(rand(UInt16, 1000, 100)),
))

lh5open("events.lh5", "cw") do lhd
    lhd["evt"] = tbl
    lhd["raw/waveform"] = wfs
end
```

Reading `lhd["evt"]` returns an `LH5LazyTable`; `lhd["evt"][:]`, or indexing
with a range, reads the rows into a `StructArray`. Iterating reads the rows in
blocks, so loops, comprehensions and `map` neither read one row at a time nor
the whole table at once; `filter` and `findall` scan the table this way,
`sort` reads it first. `StructArray(tbl)` is not supported: once a column was
read eagerly, such a `StructArray` would slice and append rows with a stale
row type and bypass the append checks. A table handle keeps the column names
it was opened with: columns added through other handles are not visible to it,
and `append!` refuses when the set of column names has changed. Columns it has
opened keep reading the data they were opened on, even after they were deleted
elsewhere; `append!` reopens such columns and refreshes the others.
Column-wise broadcasts, such as
[PropertyFunctions](https://github.com/oschulz/PropertyFunctions.jl) over the
table, open only the columns they use. Lazy tables cannot be concatenated with
`vcat`; read them first.

Scattered reads either read the whole index span the events cover, or the
events themselves. The thresholds for that choice,
[`LegendHDF5IO.scatter_bulk_max_bytes`](@ref) and
[`LegendHDF5IO.scatter_bulk_max_waste`](@ref), depend on the file system by
default: 1 MiB and a factor of 4 on local storage, 8 MiB and 16 on network
file systems such as GPFS, NFS or Lustre, which pay per read operation. The
file system is recognized from the mount table on Linux. Set the thresholds
to override the defaults.

### Appending and compression

Open the file with `usechunks = true` to create extensible array datasets, then
grow them with `append!` (always along the last dimension):

```julia
lh5open("events_chunked.lh5", "cw", usechunks = true) do lhd
    lhd["evt"] = tbl
    append!(lhd["evt"], tbl)
end
```

The first array written to a dataset sets its chunk size. Appends are not
transactional: one that fails midway, for example because a value cannot be
converted to the element type of a column, leaves the columns it has already
extended longer than the others.

With `compress = :zstd` (or `true`), array datasets are Zstandard-compressed,
and `compress = :deflate` uses shuffle plus deflate for maximum portability.
Compressed datasets are always chunked, so `append!` works on them as well,
and their chunks are capped at 1 MiB, since a partial read decompresses whole
chunks.

### Modifying structure

Columns and struct fields can be added and removed in place with
[`add_entries!`](@ref) and [`delete_entry!`](@ref):

```julia
lh5open("events.lh5", "r+") do lhd
    add_entries!(lhd, "evt", StructArray((quality = rand(Bool, 100),)))
    delete_entry!(lhd, "evt/quality")
end
```

### Geant4 output

Hit tables from Geant4 simulations in GEARS or g4simple HDF5 layout can be
read with [`Geant4HDF5Input`](@ref):

```julia
hits = open(read, "simulation.hdf5", Geant4HDF5Input)
```

## Supporting custom types

To support another type, register an LH5 datatype name for it and define a
reader and a writer. Register from the `__init__` function of your module or
package extension, since the registry is not part of the precompilation cache:

```julia
struct ChannelNo
    no::Int
end

function __init__()
    LegendHDF5IO.register_datatype!("channelno", ChannelNo)
end

LegendHDF5IO.LH5Array(ds::HDF5.Dataset, ::Type{<:ChannelNo}) =
    ChannelNo(read(ds))

function LegendHDF5IO.create_entry(parent::LHDataStore, name::AbstractString,
    x::ChannelNo; kwargs...)
    parent.data_store[name] = x.no
    LegendHDF5IO.setdatatype!(parent.data_store[name], ChannelNo)
    nothing
end
```

`Enum` types only need registration:
`LegendHDF5IO.register_datatype!(MyEnum)`.

A name can be registered for one type only, and a type under one name only.
Names consist of letters and underscores. Conflicting registrations, and
built-in names like `"real"`, are rejected with an `ArgumentError`. Subtypes
of a registered type, such as the instances of a parametric type, are written
under its name.

## The `readdata`/`writedata` API

The functions `LegendDataTypes.readdata` and `LegendDataTypes.writedata`
provide an eager alternative that operates directly on `HDF5.File` or
`HDF5.Group` objects:

```julia
using HDF5
import LegendDataTypes: readdata, writedata

HDF5.h5open("data.lh5", "w") do f
    writedata(f, "energy", rand(1000) .* u"keV")
end

E = HDF5.h5open(f -> readdata(f, "energy"), "data.lh5")
```

## Migrating from 0.1

Version 0.2 reads tables as [`LH5LazyTable`](@ref)s instead of TypedTables
`Table`s with lazily read columns. Column access (`tbl.energy`) works as before,
`tbl[:]` and indexing with a range return a `StructArray` rather than a `Table`,
and `TypedTables.Table(tbl)` gives a `Table` of the lazy columns again.
TypedTables is no longer a dependency. `StructArray(tbl)` is not supported, and
lazy tables cannot be concatenated with `vcat`; read them first.

`LH5Array` is a DiskArray now: broadcasts over it are lazy and read when
collected or written, `map` is eager. [`add_entries!`](@ref) takes the file,
the table name and the new columns; the destination argument is gone. Custom
types register their LH5 datatype with `LegendHDF5IO.register_datatype!`
instead of extending `datatype_to_string`.
