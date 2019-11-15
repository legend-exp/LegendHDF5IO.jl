# This file is a part of LegendDataTypes.jl, licensed under the MIT License (MIT).

function read_geant4_hdf5 end
export read_geant4_hdf5


struct Geant4HDF5Input <: AbstractLegendInput
    hdf5file::HDF5.HDF5File
end

export Geant4HDF5Input

Base.open(filename::AbstractString, ::Type{Geant4HDF5Input}) =
    Geant4HDF5Input(HDF5.h5open(filename, "r"))

Base.close(input::Geant4HDF5Input) = close(input.hdf5file)


function Base.getindex(input::Geant4HDF5Input, ::Colon)
    read(input)
end


function Base.read(input::Geant4HDF5Input)

    function _d_open_if_exists(g, d_name::String)
        HDF5.exists(g, d_name) ? HDF5.d_open(g, d_name) : error("data set \"$d_name\" does not exist")
    end

    function _get_indices_raw(g)
        len = filter(x->x!=0x00, _d_open_if_exists(g, "e/pages")[:])
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
    g = HDF5.g_open(h5f, "default_ntuples/t/")
    indices_raw = _get_indices_raw(g)
    h5_edep = _d_open_if_exists(g, "e_data/pages")[:]
    indices_non_zero_energies  = filter(i-> sum(h5_edep[i]) > 0.0, indices_raw) # Filters events that don't deposit any energy
    indices = vcat(indices_non_zero_energies...)
    n_ind = length(indices)

    evtno = Int32.(vcat([ [j for i in 1:length(indices_non_zero_energies[j])] for j in 1:length(indices_non_zero_energies) ]...))
    detno = try Int32.(_d_open_if_exists(g, "vlm_data/pages")[:][indices]) catch ; ones(Int32, n_ind) end
    thit = ( try Float32.(_d_open_if_exists(g, "t0_data/pages")[:][indices]) catch ; zeros(Float32, n_ind) end ) .* u"s"
    edep = ( try Float32.(_d_open_if_exists(g, "e_data/pages")[:][indices]) catch ; zeros(Float32, n_ind) end ) .* u"keV"

    x0 = try Float32.(_d_open_if_exists(g, "x0_data/pages")[:][indices]) catch ; zeros(Float32, n_ind) end
    y0 = try Float32.(_d_open_if_exists(g, "y0_data/pages")[:][indices]) catch ; zeros(Float32, n_ind) end
    z0 = try Float32.(_d_open_if_exists(g, "z0_data/pages")[:][indices]) catch ; zeros(Float32, n_ind) end

    pos = [ SVector{3}(([ x0[i], y0[i], z0[i] ] .* u"mm")...) for i in 1:n_ind ]


    ekin = ( try Float32.(_d_open_if_exists(g, "k_data/pages")[:][indices]) catch ; zeros(Float32, n_ind) end ) .* u"keV"# kinetic Energy of track [keV]
    stp = try Int32.(_d_open_if_exists(g, "stp_data/pages")[:][indices]) catch ; zeros(Int32, n_ind) end # step number
    l = ( try Float32.(_d_open_if_exists(g, "l_data/pages")[:][indices]) catch ; zeros(Float32, n_ind) end  ) .* u"mm"# step length
    mom = try Int32.(_d_open_if_exists(g, "mom_data/pages")[:][indices]) catch ; zeros(Int32, n_ind) end # parent id
    trk = try Int32.(_d_open_if_exists(g, "trk_data/pages")[:][indices]) catch ; zeros(Int32, n_ind) end # trk id
    pdg = try Int32.(_d_open_if_exists(g, "pdg_data/pages")[:][indices]) catch ; zeros(Int32, n_ind) end # particle id
    pro = try Int32.(_d_open_if_exists(g, "pro_data/pages")[:][indices]) catch ; zeros(Int32, n_ind) end # process id

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
