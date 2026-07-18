# LegendHDF5IO.jl

LegendHDF5IO provides read and write functionality for HDF5 files that
follow the [LEGEND LH5 data format](https://legend-exp.github.io/legend-data-format-specs/dev/hdf5/).
Such files store arrays, tables, waveforms, histograms and nested structures
together with `datatype` and `units` attributes, and are interoperable with
the Python package [legend-pydataobj](https://github.com/legend-exp/legend-pydataobj).

## Getting started

Open a file with [`lh5open`](@ref), which returns an [`LHDataStore`](@ref).
Writing is done via `setindex!`, reading via `getindex`:

```julia
using LegendHDF5IO, Unitful

lh5open("data.lh5", "cw") do lhd
    lhd["energy"] = rand(1000) .* u"keV"
    lhd["metadata"] = (detector = "V01234A", threshold = 5.0u"keV")
end

lh5open("data.lh5") do lhd
    E = lhd["energy"]       # lazy, disk-backed LH5Array
    E_first = E[1:100]      # reads only the first 100 elements
    E_all = E[:]            # reads the full array
    lhd["metadata"]         # NamedTuples are read back directly
end
```

Datasets are wrapped as lazy [`LH5Array`](@ref)s, so slicing reads only the
requested part of the data from disk. Nested structures (`NamedTuple`s,
tables, waveforms) are represented as HDF5 groups.

### Tables and waveforms

Tables (e.g. `TypedTables.Table`) and vectors of `RDWaveform`s round-trip
through their LH5 representation:

```julia
using TypedTables, ArraysOfArrays, RadiationDetectorSignals

tbl = Table(
    evtno = collect(1:100),
    energy = rand(100) .* u"keV",
    samples = VectorOfVectors([rand(-5:5, 50) for _ in 1:100]),
)

wfs = ArrayOfRDWaveforms((
    fill(range(0.0u"μs", 10.0u"μs", length = 1000), 100),
    nestedview(rand(UInt16, 1000, 100)),
))

lh5open("events.lh5", "cw") do lhd
    lhd["evt"] = tbl
    lhd["raw/waveform"] = wfs
end
```

Reading `lhd["evt"]` returns a `Table` whose columns are disk-backed; use
`lhd["evt"][:]` or index with a range to materialize rows.

### Appending

Open the file with `usechunks = true` to create extensible datasets, then
grow them with `append!` (always along the last dimension):

```julia
lh5open("events.lh5", "cw", usechunks = true) do lhd
    lhd["evt"] = tbl
    append!(lhd["evt"], tbl)
end
```

### Modifying structure

Columns and struct fields can be added and removed in place with
[`add_entries!`](@ref) and [`delete_entry!`](@ref):

```julia
lh5open("events.lh5", "r+") do lhd
    add_entries!(lhd, "evt", Table(quality = rand(Bool, 100)))
    delete_entry!(lhd, "evt/quality")
end
```

### Geant4 output

Hit tables from Geant4 simulations in GEARS or g4simple HDF5 layout can be
read with [`Geant4HDF5Input`](@ref):

```julia
hits = open(read, "simulation.hdf5", Geant4HDF5Input)
```

## The `readdata`/`writedata` API

The functions `LegendDataTypes.readdata` and `LegendDataTypes.writedata`
provide an alternative, eager interface operating directly on `HDF5.File`
or `HDF5.Group` objects:

```julia
using HDF5
import LegendDataTypes: readdata, writedata

HDF5.h5open("data.lh5", "w") do f
    writedata(f, "energy", rand(1000) .* u"keV")
end

E = HDF5.h5open(f -> readdata(f, "energy"), "data.lh5")
```
