# This file is a part of LegendDataTypes.jl, licensed under the MIT License (MIT).

export Geant4HDF5Input
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

Base.close(input::Geant4HDF5Input) = close(input.hdf5file)

function Base.getindex(input::Geant4HDF5Input, ::Colon)
    read(input)
end

function Base.read(input::Geant4HDF5Input)
    if haskey(input.hdf5file, "/default_ntuples/t/")
        read(GEARS_HDF5Input(input.hdf5file))
    elseif haskey(input.hdf5file, "/default_ntuples/g4sntuple/")
        read(G4SIMPLE_HDF5Input(input.hdf5file))
    end
end

function Base.read(input::GEARS_HDF5Input)

    function _get_indices_raw(g)
        len = filter(x->x!=0x00, g["e/pages"][:])
        indices_raw = []
        start = 0
        for i in len
            start+=1
            stop = start+i-1
            push!(indices_raw, collect(start:stop))
            start = stop
        end
        indices_raw
    end

    h5f = input.hdf5file
    g = h5f["default_ntuples/t/"]
    indices_raw = _get_indices_raw(g)
    h5_edep = g["e_data/pages"][:]
    indices_non_zero_energies  = filter(i-> sum(h5_edep[i]) > 0.0, indices_raw) # Filters events that don't deposit any energy
    indices = vcat(indices_non_zero_energies...)
    n_ind = length(indices)

    evtno = Int32.(vcat([ [j for i in 1:length(indices_non_zero_energies[j])] for j in 1:length(indices_non_zero_energies) ]...))
    detno = try Int32.(g["vlm_data/pages"][:][indices]) catch ; ones(Int32, n_ind) end
    thit = ( try Float32.(g["t0_data/pages"][:][indices]) catch ; zeros(Float32, n_ind) end ) .* u"s"
    edep = ( try Float32.(g["e_data/pages"][:][indices]) catch ; zeros(Float32, n_ind) end ) .* u"keV"

    x0 = try Float32.(g["x0_data/pages"][:][indices]) catch ; zeros(Float32, n_ind) end
    y0 = try Float32.(g["y0_data/pages"][:][indices]) catch ; zeros(Float32, n_ind) end
    z0 = try Float32.(g["z0_data/pages"][:][indices]) catch ; zeros(Float32, n_ind) end

    pos = [ SVector{3}(([ x0[i], y0[i], z0[i] ] .* u"mm")...) for i in 1:n_ind ]


    ekin = ( try Float32.(g["k_data/pages"][:][indices]) catch ; zeros(Float32, n_ind) end ) .* u"keV"# kinetic Energy of track [keV]
    stp = try Int32.(g["stp_data/pages"][:][indices]) catch ; zeros(Int32, n_ind) end # step number
    l = ( try Float32.(g["l_data/pages"][:][indices]) catch ; zeros(Float32, n_ind) end  ) .* u"mm"# step length
    mom = try Int32.(g["mom_data/pages"][:][indices]) catch ; zeros(Int32, n_ind) end # parent id
    trk = try Int32.(g["trk_data/pages"][:][indices]) catch ; zeros(Int32, n_ind) end # trk id
    pdg = try Int32.(g["pdg_data/pages"][:][indices]) catch ; zeros(Int32, n_ind) end # particle id
    pro = try Int32.(g["pro_data/pages"][:][indices]) catch ; zeros(Int32, n_ind) end # process id

    hits = TypedTables.Table(
        evtno = evtno,
        detno = detno,
        thit = thit,
        edep = edep,
        pos = pos,
        ekin = ekin,
        stp = stp,
        l = l,
        mom = mom,
        trk = trk,
        pdg = pdg,
        pro = pro,
    )
    return hits
end

function Base.read(input::G4SIMPLE_HDF5Input)
    function _get_indices_raw(g)
        len = filter(x->x!=0x00, g["e/pages"][:])
        indices_raw = []
        start = 0
        for i in len
            start+=1
            stop = start+i-1
            push!(indices_raw, collect(start:stop))
            start = stop
        end
        indices_raw
    end

    h5f = input.hdf5file
    g = h5f["default_ntuples/g4sntuple/"]
    # indices_raw = _get_indices_raw(g)
    h5_edep = g["event/pages"][:]
    # indices_non_zero_energies  = filter(i-> sum(h5_edep[i]) > 0.0, indices_raw) # Filters events that don't deposit any energy
    # indices = vcat(indices_non_zero_energies...)
    n_ind = length(h5_edep)
    indices = [i for i in 1:n_ind]

    evtno = try Int32.(g["event/pages"][:][indices]) catch ; ones(Int32, n_ind) end
    irep = try Int32.(g["iRep/pages"][:][indices]) catch ; ones(Int32, n_ind) end
    detno = try Int32.(g["volID/pages"][:][indices]) catch ; ones(Int32, n_ind) end
    thit = ( try Float32.(g["t/pages"][:][indices]) catch ; zeros(Float32, n_ind) end ) .* u"s"
    edep = ( try Float32.(g["Edep/pages"][:][indices]) catch ; zeros(Float32, n_ind) end ) .* u"keV"

    x0 = try Float32.(g["x/pages"][:][indices]) catch ; zeros(Float32, n_ind) end
    y0 = try Float32.(g["y/pages"][:][indices]) catch ; zeros(Float32, n_ind) end
    z0 = try Float32.(g["z/pages"][:][indices]) catch ; zeros(Float32, n_ind) end

    pos = [ SVector{3}(([ x0[i], y0[i], z0[i] ] .* u"mm")...) for i in 1:n_ind ]


    ekin = ( try Float32.(g["KE_data/pages"][:][indices]) catch ; zeros(Float32, n_ind) end ) .* u"keV"# kinetic Energy of track [keV]
    stp = try Int32.(g["step/pages"][:][indices]) catch ; zeros(Int32, n_ind) end # step number
    mom = try Int32.(g["parentID/pages"][:][indices]) catch ; zeros(Int32, n_ind) end # parent id
    trk = try Int32.(g["trackID/pages"][:][indices]) catch ; zeros(Int32, n_ind) end # trk id
    pdg = try Int32.(g["pid/pages"][:][indices]) catch ; zeros(Int32, n_ind) end # particle id

    hits = TypedTables.Table(
        evtno = evtno,
        detno = irep,
        thit = thit,
        edep = edep,
        pos = pos,
        ekin = ekin,
        volID = detno,
        stp = stp,
        mom = mom,
        trk = trk,
        pdg = pdg,
    )
    return hits
end
