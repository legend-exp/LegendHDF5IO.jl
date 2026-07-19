# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT

function to_table(x::ArrayOfRDWaveforms)
    TypedTables.Table(
        t0 = first.(x.time),
        dt = step.(x.time),
        values = x.signal
    )
end

function to_table(x::AbstractVector{<:RDWaveform})
    to_table(ArrayOfRDWaveforms(x))
end

_dtt02range(dt::RealQuantity, t0::RealQuantity, len::Int) =
    t0 .+ (Int32(0):Int32(len - 1)) .* dt

_dtt02range(dt::AbstractArray, t0::AbstractArray, values) = 
    _dtt02range(dt[axes(dt)...], t0[axes(t0)...], values)

_dtt02range(dt::Array, t0::Array, values::AbstractArrayOfSimilarArrays) =
    _dtt02range.(dt, t0, innersize(values)[1])

_dtt02range(dt::Array, t0::Array, values::VectorOfVectors) = 
    _dtt02range.(dt, t0, diff(values.elem_ptr))

_dtt02range(dt::Array, t0::Array, values::VectorOfEncodedArrays) = 
    _dtt02range.(dt, t0, only.(values.innersizes))

# fallback to default implementation if values is just an array
_dtt02range(dt, t0, values) = _dtt02range.(dt, t0, size(values, 1))

function from_table(tbl, ::Type{<:AbstractVector{<:RDWaveform}})
    StructArray{RDWaveform}((
        _dtt02range(tbl.dt, tbl.t0, tbl.values),
        tbl.values
    ))
end
