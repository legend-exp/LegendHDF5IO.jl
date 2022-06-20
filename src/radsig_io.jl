# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT


function to_table(x::AbstractVector{<:RDWaveform})
    TypedTables.Table(
        t0 = first.(x.time),
        dt = step.(x.time),
        values = x.signal
    )
end


function _dtt02range(dt::RealQuantity, t0::RealQuantity, values::AbstractVector)
    # TODO: Handle different units for dt and t0
    t_idxs = Int32(0):Int32(size(values,1) - 1)
    t0 .+ t_idxs .* dt
end

function from_table(tbl, ::Type{<:AbstractVector{<:RDWaveform}})
    StructArray{RDWaveform}((
        _dtt02range.(tbl.dt, tbl.t0, tbl.values),
        tbl.values
    ))
end


function LegendDataTypes.writedata(
    output::HDF5.H5DataStore, name::AbstractString,
    x::AbstractVector{<:RDWaveform},
    fulldatatype::DataType = typeof(x)
) where {T}
    @assert fulldatatype == typeof(x)
    writedata(output, name, to_table(x))
end


function LegendDataTypes.readdata(
    input::HDF5.H5DataStore, name::AbstractString,
    AT::Type{<:AbstractVector{<:RDWaveform}}
)
    tbl = readdata(input, name, TypedTables.Table{<:NamedTuple{(:t0, :dt, :values)}})
    from_table(tbl, AbstractVector{<:RDWaveform})
end

