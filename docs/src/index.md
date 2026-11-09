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
a singleton dimension or add dimensions to it work if another `LH5Array` spans
that dimension. If only in-memory arrays do, DiskArrays does not support them
yet: they give wrong data or throw when collected, reduced or written. Nor
does it support lazy concatenations with an empty part: `vcat`, `hcat` and
`cat` refuse them if an `LH5Array` is among the first two arguments, while
concatenations of views and lazy broadcasts of `LH5Array`s otherwise have the
wrong length or throw. Read such data with `[:]` first. Empty `LH5Array`s also
need to be read first for reductions without an `init` value, such as `sum` or
`findall`, and for broadcasts that combine several of them.

Datasets opened for writing can be modified in place within their current
size (`E[1:10] = ...`), and `append!` grows them. An in-place write whose
source reads the dataset it writes to, like `E .= reverse(E)`, is refused,
since DiskArrays writes chunk by chunk; read the source first with `[:]`.
Nested structures (`NamedTuple`s, tables, waveforms) are represented as HDF5
groups.

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

Indexing `lhd["evt"]` with `[:]` or a range reads the rows into a
`StructArray`. Iteration reads the rows in blocks, so loops, comprehensions
and `map` read neither one row at a time nor the whole table at once. `filter`
and `findall` scan the table this way, while `sort` reads it first.
`StructArray(tbl)`, `TypedTables.Table(tbl)` and `TypedTables.FlexTable(tbl)`
are not supported: they would append through the columns without the table's
checks, and a `StructArray` of a column read eagerly would slice and append
rows with a stale row type. Read the rows with `tbl[:]` instead.

A handle, a table or an array, works while the stored object is as it was
when the handle was opened. An append or a structure change (`add_entries!`,
`delete_entry!`) through another handle makes its reads and appends throw,
and it must be reopened; its own appends keep it current. In-place writes
change no cached state, so every handle sees them.

Scattered reads either read the whole index span the events cover, or only
the events themselves. The thresholds for that choice,
[`LegendHDF5IO.scatter_bulk_max_bytes`](@ref) and
[`LegendHDF5IO.scatter_bulk_max_waste`](@ref), depend on the file system by
default: 1 MiB and a factor of 4 on local storage, 8 MiB and 16 on network
file systems such as GPFS, NFS or Lustre, which pay per read operation. The
file system is recognized from the mount table on Linux. Both are
[settings](@ref Settings).

### Selecting columns

A [PropertyFunctions](https://github.com/oschulz/PropertyFunctions.jl)
selection returns an [`LH5TypedTable`](@ref): a lazy table of the selected
columns under the names given to them, with the row type fixed when the
selection is made:

```julia
using PropertyFunctions

lh5open("events.lh5") do lhd
    evt = lhd["evt"]
    sel = (@pf (e = $energy, no = $evtno)).(evt)  # an LH5TypedTable
    eltype(sel)                                   # a concrete NamedTuple type
    for row in sel                                # rows are read in blocks
        row.e
    end
end
```

Nested paths such as `$geds.energy` select from nested tables, and only the
selected columns are read. The row type and the types of the columns of a
block of rows come from the first file, so rows read with a range or an index
vector, iteration and `filterby` have inferable types. The columns of a typed
table, `sel.e`, are the lazy columns of the table it selects from, while a
nested table, `sel.geds`, is a typed table of its selected columns, whose type
follows from the type of `sel`, so `sel.geds.<TAB>` completes in the REPL.

A PropertyFunction broadcast over any lazy table, `(@pf $energy^2).(evt)` or
`(@pf $e^2).(sel)`, returns an [`LH5LazyBroadcast`](@ref): a lazy vector that
reads the rows it needs of the columns the function uses and applies the
function to them block by block. Its element type is inferred from the row
type or, failing that, taken from the first block of rows. Over a typed table
the broadcast has an inferable type. Over other tables the row type is fixed
from the columns used, as for a selection, so a broadcast over a file chain
opens the first file only. A pure column accessor, `(@pf $energy).(evt)`,
returns the column.

### Joining and concatenating tables

`innermerge` from PropertyFunctions joins lazy tables column by column, and
`vcat` concatenates them row by row, both without reading anything:

```julia
lh5open("events.lh5") do lhd
    evt = lhd["evt"]
    twice = vcat(evt, evt)                                     # an LH5ChainedTable
    tagged = innermerge(evt, (tag = rand(Bool, length(evt)),)) # an LH5JoinedTable
    tagged[1:10]                                               # reads ten rows
end
```

A joined table takes each column from the last table that has it, as `merge`
does, and opens it when it is used. The tables must have the same length,
which is compared when the length of the join is needed. A chained table
reads rows from the parts that hold them and concatenates columns lazily.
`ArraysOfArrays.vecflattened` chains a vector of tables as well.
`Tables.materializer` of a lazy table wraps the given columns without
reading them.

`append!` refuses joined and chained tables. Writing to a column of a joined
table writes to the table it comes from, while the columns of a chained table
are read-only if they read from disk. Its columns held in memory, such as
those read eagerly, accept writes, which do not reach the files.
Concatenating lazy tables with other arrays, and `hcat`, are refused if a
table is among the first two arguments. A table behind two other arrays is
concatenated row by row into an `Array`.

### Reading many files

`lh5open` of a vector of file names opens files of the same structure as one
[`LH5FileChain`](@ref), similar to ROOT's `TChain`. Indexing it with a path
chains the table or array there over all files:

```julia
lh5open(["run1.lh5", "run2.lh5", "run3.lh5"]) do files
    evt = files["evt"]  # an LH5ChainedTable over the three files
    evt[1:100]          # reads from the first file only
    sum(evt.energy)     # reads the column from every file
end
```

Files are opened when they are read from, and at most `max_open` of them stay
open, by default the setting [`LegendHDF5IO.max_open`](@ref), 64. Indexing the
chain with the path of an array opens every file right away, as does accessing
a column of a chained table, since chaining them needs their sizes in all
files. The first file defines the structure: column names, datatypes, units,
codecs, element types and inner sizes. Another file is checked against it
when an array or column is first opened in it, so the files need to agree
only on what is used. A difference throws an `ArgumentError` that names the
file. Objects from the chain stop working when it is closed.

### Appending and compression

Open the file with `usechunks = true` to create extensible array datasets, then
grow them with `append!` (always along the last dimension):

```julia
lh5open("events_chunked.lh5", "cw", usechunks = true) do lhd
    lhd["evt"] = tbl
    append!(lhd["evt"], tbl)
end
```

The first array written to a dataset sets its chunk size, up to the setting
[`LegendHDF5IO.max_chunk_bytes`](@ref) (16 MiB): a larger first write is split
evenly into chunks of at most that size. Uncompressed chunks are stored in
full, so an append that does not fill its last chunk wastes at most one
chunk. Appends are not
transactional: one that fails midway, for example because a value cannot be
converted to the element type of a column, leaves the columns it has already
extended longer than the others.

With `compress = :zstd` (or `true`), array datasets are Zstandard-compressed,
and `compress = :deflate` uses shuffle plus deflate for maximum portability.
Compressed datasets are always chunked, so `append!` works on them as well,
and their chunks are capped at [`LegendHDF5IO.max_compressed_chunk_bytes`](@ref)
(1 MiB), since a partial read decompresses whole chunks. The defaults of
`usechunks` and `compress` are the settings [`LegendHDF5IO.usechunks`](@ref)
and [`LegendHDF5IO.compress`](@ref).

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

## Settings

The defaults of `lh5open`'s `usechunks` and `compress` arguments and of the
`max_open` of file chains, the chunk size caps, the thresholds of scattered
reads and the number of rows that iteration reads at a time are settings, as
[ScopedSettings](https://github.com/oschulz/ScopedSettings.jl) defines them:

| Setting | Default |
|:--|:--|
| [`LegendHDF5IO.usechunks`](@ref) | `false` |
| [`LegendHDF5IO.compress`](@ref) | `:none` |
| [`LegendHDF5IO.max_chunk_bytes`](@ref) | 16 MiB |
| [`LegendHDF5IO.max_compressed_chunk_bytes`](@ref) | 1 MiB |
| [`LegendHDF5IO.scatter_bulk_max_bytes`](@ref) | `nothing`, by file system |
| [`LegendHDF5IO.scatter_bulk_max_waste`](@ref) | `nothing`, by file system |
| [`LegendHDF5IO.max_open`](@ref) | 64 |
| [`LegendHDF5IO.rows_per_block`](@ref) | 4096 |

A setting is read with `setting[]`. It takes its default from the preference
of its name in the active `LocalPreferences.toml`, and the environment
variable `LEGENDHDF5IOJL_<NAME>` in upper case overrides that:

```toml
[LegendHDF5IO]
compress = "zstd"
scatter_bulk_max_bytes = "auto"
```

```sh
export LEGENDHDF5IOJL_MAX_CHUNK_BYTES=67108864
```

The settings that may be `nothing` take the value `"auto"` for it there. A
setting can also be set for the running session, with `setting[] = value`,
where `setting[] = default_value` restores the default, or for a scope:

```julia
using ScopedSettings

LegendHDF5IO.scatter_bulk_max_bytes[] = 2^23
LegendHDF5IO.scatter_bulk_max_bytes[] = default_value

@with LegendHDF5IO.usechunks => true LegendHDF5IO.compress => :zstd begin
    lh5open(lhd -> lhd["evt"] = tbl, "events_compressed.lh5", "w")
end
```

## Supporting custom types

To support another type, register an LH5 datatype name for it and define a
reader and a writer. Register from the `__init__` function of your module or
package extension, since the registry is not part of the precompilation cache:

```julia
using HDF5

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
and `tbl[:]` and indexing with a range return a `StructArray` rather than a
`Table`; `TypedTables.Table(tbl[:])` gives a `Table` of the rows. TypedTables
is no longer a dependency, and `StructArray(tbl)` and `TypedTables.Table(tbl)`
are not supported.
Joins with `innermerge` return lazy tables, as before. Named PropertyFunctions
selections return [`LH5TypedTable`](@ref)s, lazy tables with a fixed row type,
where 0.1 returned `Table`s of the lazy columns.

`LH5Array` is now a DiskArray: broadcasts over it are lazy and read when
collected or written, while `map` is eager. Reading Bool arrays gives an
`Array{Bool}` instead of a `BitArray`. [`add_entries!`](@ref) takes the
file, the table name and the new columns, without the destination argument.
Custom types register their LH5 datatype with
`LegendHDF5IO.register_datatype!` instead of extending `datatype_to_string`.
Encoded vectors of vectors are now written with the datatype
`array<1>{encoded_array<1>{...}}` of the LH5 specification, as legend-pydataobj
writes them, and files written by 0.1 still read, but not in one file chain
with newer files.
