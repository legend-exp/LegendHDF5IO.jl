module LegendHDF5IOLegendDataManagementExt

@static if isdefined(Base, :get_extension)
    using LegendDataManagement
else
    using ..LegendDataManagement
end

using LegendDataManagement: DataSelector
using LegendHDF5IO

const dataselector_bytypes = Dict{Type, String}()

LegendHDF5IO.datatype_to_string(::Type{<:T}) where {T <: DataSelector} = 
    dataselector_bytypes[T]

function LegendHDF5IO._array_type(::Type{Array{T, N}}
    ) where {T <: DataSelector, N}
    
    AbstractArray{T, N}
end

# write DataSelector
function LegendHDF5IO.create_entry(parent::LHDataStore, name::AbstractString, 
    data::T; kwargs...) where {T <:DataSelector}
    
    LegendHDF5IO.create_entry(parent, name, string(data); kwargs...)
    LegendHDF5IO.setdatatype!(parent.data_store[name], T)
    nothing
end

# write AbstractArray{<:DataSelector}
function LegendHDF5IO.create_entry(parent::LHDataStore, name::AbstractString, 
    data::T; kwargs...) where {T <:AbstractArray{<:DataSelector}}
    
    LegendHDF5IO.create_entry(parent, name, string.(data); kwargs...)
    LegendHDF5IO.setdatatype!(parent.data_store[name], T)
    nothing
end

LegendHDF5IO.LH5Array(ds::LegendHDF5IO.HDF5.Dataset, ::Type{<:T}
    ) where {T <: DataSelector} = begin
    
    s = read(ds)
    T(s)
end

function LegendHDF5IO.LH5Array(ds::LegendHDF5IO.HDF5.Dataset, ::Type{<:AbstractArray{<:T, N}}
    ) where {T <: DataSelector, N}
   
    s = read(ds)
    T.(s)
end

function __init__()
    LegendHDF5IO._datatype_dict["expsetup"] = ExpSetup
    LegendHDF5IO._datatype_dict["datatier"] = DataTier
    LegendHDF5IO._datatype_dict["datapartition"] = DataPartition
    LegendHDF5IO._datatype_dict["dataperiod"] = DataPeriod
    LegendHDF5IO._datatype_dict["datarun"] = DataRun
    LegendHDF5IO._datatype_dict["datacategory"] = DataCategory
    LegendHDF5IO._datatype_dict["timestamp"] = Timestamp
    LegendHDF5IO._datatype_dict["filekey"] = FileKey
    LegendHDF5IO._datatype_dict["channelid"] = ChannelId
    LegendHDF5IO._datatype_dict["detectorid"] = DetectorId

    dataselector_bytypes[ExpSetup] = "expsetup"
    dataselector_bytypes[DataTier] = "datatier"
    dataselector_bytypes[DataPartition] = "datapartition"
    dataselector_bytypes[DataPeriod] = "dataperiod"
    dataselector_bytypes[DataRun] = "datarun"
    dataselector_bytypes[DataCategory] = "datacategory"
    dataselector_bytypes[Timestamp] = "timestamp"
    dataselector_bytypes[FileKey] = "filekey"
    dataselector_bytypes[ChannelId] = "channelid"
    dataselector_bytypes[DetectorId] = "detectorid"
end

end