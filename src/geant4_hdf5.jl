# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

export Geant4HDF5Input

"""
    Geant4HDF5Input <: AbstractLegendInput

Input wrapper for Geant4 HDF5 files in GEARS or g4simple layout.

Use via `open(filename, Geant4HDF5Input)` (or the corresponding `do`-block
form) and `read(input)`, which returns a hits `Table`.
"""
struct Geant4HDF5Input <: AbstractLegendInput
    hdf5file::HDF5.File
end

struct GEARS_HDF5Input <: AbstractLegendInput
    hdf5file::HDF5.File
end

struct G4SIMPLE_HDF5Input <: AbstractLegendInput
    hdf5file::HDF5.File
end

Base.open(filename::AbstractString, ::Type{Geant4HDF5Input}) =
    Geant4HDF5Input(HDF5.h5open(filename, "r"))

function Base.open(f::Function, filename::AbstractString, ::Type{Geant4HDF5Input})
    input = open(filename, Geant4HDF5Input)
    try
        f(input)
    finally
        close(input)
    end
end

Base.close(input::Geant4HDF5Input) = close(input.hdf5file)

function Base.getindex(input::Geant4HDF5Input, ::Colon)
    read(input)
end

function Base.read(input::Geant4HDF5Input)
    if haskey(input.hdf5file, "/default_ntuples/t/")
        read(GEARS_HDF5Input(input.hdf5file))
    elseif haskey(input.hdf5file, "/default_ntuples/g4sntuple/")
        read(G4SIMPLE_HDF5Input(input.hdf5file))
    else
        throw(ErrorException("Unrecognized Geant4 HDF5 file layout, expected" *
            " group \"/default_ntuples/t\" (GEARS) or" *
            " \"/default_ntuples/g4sntuple\" (g4simple)"))
    end
end

_g4_column(::Type{T}, g, path::AbstractString, indices, default, n::Int) where {T} =
    haskey(g, path) ? T.(g[path][:][indices]) : fill(T(default), n)

# Consecutive per-event hit index ranges from the per-event hit counts
function _gears_hit_ranges(hits_per_event::AbstractVector{<:Integer})
    ranges = Vector{UnitRange{Int}}(undef, length(hits_per_event))
    stop = 0
    for (i, l) in enumerate(hits_per_event)
        start = stop + 1
        stop = start + Int(l) - 1
        ranges[i] = start:stop
    end
    ranges
end

function Base.read(input::GEARS_HDF5Input)
    g = input.hdf5file["default_ntuples/t/"]
    hit_ranges = _gears_hit_ranges(filter(!iszero, g["de/pages"][:]))
    h5_edep = g["de_data/pages"][:]
    # Drop events without any energy deposition:
    event_ranges = filter(r -> sum(view(h5_edep, r)) > 0, hit_ranges)
    indices = reduce(vcat, event_ranges; init = Int[])
    n_ind = length(indices)

    evtno = Vector{Int32}(undef, n_ind)
    offset = 0
    for (j, r) in enumerate(event_ranges)
        evtno[offset .+ (1:length(r))] .= Int32(j)
        offset += length(r)
    end

    col(::Type{T}, path, default) where {T} = _g4_column(T, g, path, indices, default, n_ind)

    x0 = col(Float32, "x_data/pages", 0)
    y0 = col(Float32, "y_data/pages", 0)
    z0 = col(Float32, "z_data/pages", 0)

    hits = TypedTables.Table(
        evtno = evtno,
        detno = col(Int32, "vlm_data/pages", 1),
        thit = col(Float32, "t_data/pages", 0) .* u"s",
        edep = Float32.(h5_edep[indices]) .* u"keV",
        pos = [SVector(x0[i], y0[i], z0[i]) * u"mm" for i in 1:n_ind],
        ekin = col(Float32, "k_data/pages", 0) .* u"keV",
        stp = col(Int32, "stp_data/pages", 0),
        l = col(Float32, "l_data/pages", 0) .* u"mm",
        mom = col(Int32, "pid_data/pages", 0),
        trk = col(Int32, "trk_data/pages", 0),
        pdg = col(Int32, "pdg_data/pages", 0),
        pro = col(Int32, "pro_data/pages", 0),
    )
    return hits
end

function Base.read(input::G4SIMPLE_HDF5Input)
    g = input.hdf5file["default_ntuples/g4sntuple/"]
    evtno = Int32.(g["event/pages"][:])
    n_ind = length(evtno)

    col(::Type{T}, path, default) where {T} = _g4_column(T, g, path, :, default, n_ind)

    x0 = col(Float32, "x/pages", 0)
    y0 = col(Float32, "y/pages", 0)
    z0 = col(Float32, "z/pages", 0)

    # The detector is identified by the volume copy number (iRep), while
    # volID identifies the logical volume:
    hits = TypedTables.Table(
        evtno = evtno,
        detno = col(Int32, "iRep/pages", 1),
        thit = col(Float32, "t/pages", 0) .* u"ns",
        edep = col(Float32, "Edep/pages", 0) .* u"MeV",
        pos = [SVector(x0[i], y0[i], z0[i]) * u"mm" for i in 1:n_ind],
        ekin = col(Float32, "KE/pages", 0) .* u"MeV",
        volID = col(Int32, "volID/pages", 1),
        stp = col(Int32, "step/pages", 0),
        mom = col(Int32, "parentID/pages", 0),
        trk = col(Int32, "trackID/pages", 0),
        pdg = col(Int32, "pid/pages", 0),
    )
    return hits
end
