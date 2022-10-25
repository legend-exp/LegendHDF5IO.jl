# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT


function to_table(x::AbstractVector{<:RDWaveform})
    TypedTables.Table(
        t0 = first.(x.time),
        dt = step.(x.time),
        values = x.signal
    )
end

_dtt02range(dt::RealQuantity, t0::RealQuantity, len::Int) =
    t0 .+ (Int32(0):Int32(len - 1)) .* dt

_dtt02range(dt::AbstractArray, t0::AbstractArray, values) = 
    _dtt02range(dt[axes(dt)...], t0[axes(t0)...], values)

_dtt02range(dt::Array, t0::Array, values::ArrayOfSimilarArrays) =
    _dtt02range.(dt, t0, innersize(values)[1])

_dtt02range(dt::Array, t0::Array, values::VectorOfVectors) = 
    _dtt02range.(dt, t0, diff(values.elem_ptr))

# fallback to default implementation if values is just an array
_dtt02range(dt, t0, values) = _dtt02range.(dt, t0, size(values, 1))

function from_table(tbl, ::Type{<:AbstractVector{<:RDWaveform}})
    StructArray{RDWaveform}((
        _dtt02range(tbl.dt, tbl.t0, tbl.values),
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

