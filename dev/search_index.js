var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/#Modules","page":"API","title":"Modules","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Order = [:module]","category":"page"},{"location":"api/#Types-and-constants","page":"API","title":"Types and constants","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Order = [:type, :constant]","category":"page"},{"location":"api/#Functions-and-macros","page":"API","title":"Functions and macros","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Order = [:macro, :function]","category":"page"},{"location":"api/#Documentation","page":"API","title":"Documentation","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [LegendHDF5IO]\nOrder = [:module, :type, :constant, :macro, :function]","category":"page"},{"location":"api/#LegendHDF5IO.LH5Array","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array{T, N} <: AbstractArray{T, N}\n\nArray wrapper for HDF5.Datasets following the LEGEND data format as in \".lh5\" files. \n\nAn LH5Array contains a HDF5.Dataset file and Unitful.Unitlike units as  returned by getunits(file).getindexandappend!are supported.getindexessentially falls back togetindexforHDF5.Datasets,  enabling the user to always read in the desired part of an ondisk array without  having to load it in whole beforehand.append!` uses chunks to append the data provided to the ondisk array. It is important to note, that data is always appended along the last dimension of an  array\n\nDefault constructors\n\nLH5Array{T}(ds::HDF5.Dataset, u::Unitful.Unitlike)\nLH5Array{T, N}(ds::HDF5.Dataset)\nLH5Array(ds::Union{HDF5.Dataset, HDF5.H5DataStore})\n\n\nExamples:\n\njulia> using HDF5\njulia> f = h5open(\"path/to/lh5/file\", \"r\")\njulia> l5 = LH5Array(f[\"path/to/HDF5/Dataset\"])\n[...]\njulia> x = lh[1:10]     # load the first 10 elements of the ondisk array\n[...]\njulia> append!(lh, x)   # append those 10 elements to the ondisk array \n[...]\n\n\n\n\n\n","category":"type"},{"location":"api/#LegendHDF5IO.LH5Array-Tuple{HDF5.Dataset, Type{<:AbstractArray{<:Bool}}}","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.Dataset, ::Type{<:AbstractArray{<:Bool}}) = begin\n\nreturn a LH5Array with dimensions equal to that of ds and element type  Bool. Applying getindex! on LH5Array{Bool} will yield a BitArray.\n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Tuple{HDF5.Dataset, Type{<:AbstractArray{<:NTuple{N, T} where {N, T}}}}","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.Dataset, ::Type{<:AbstractArray{<:Tuple}})\n\nreturn an Array of NTuples\n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Tuple{HDF5.Dataset, Type{<:AbstractArray}}","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.Dataset, ::Type{<:AbstractArray})\n\nreturn a LH5Array with dimensions equal to that of ds and element type  equal to eltype(ds) * u\n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Tuple{HDF5.Dataset, Type{<:ArraysOfArrays.AbstractArrayOfSimilarArrays{<:Union{Real, Unitful.AbstractQuantity{<:Real}}}}}","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.H5DataStore, ::Type{<:AbstractArrayOfSimilarArrays{<:RealQuantity}})\n\nreturn an ArraysOfSimilarArrays where the field data is a LH5Array  (see ArraysOfSimilarArrays)\n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Tuple{HDF5.Dataset, Type{<:Bool}}","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.H5DataStore, ::Type{<:Bool}) = begin\n\nreturn a value with type Bool\n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Tuple{HDF5.Dataset, Type{<:String}}","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.Dataset, ::Type{<:String})\n\nreturn a String.\n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Tuple{HDF5.Dataset, Type{<:Symbol}}","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.Dataset, ::Type{<:Symbol})\n\nreturn a Symbol.\n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Tuple{HDF5.Dataset, Type{<:Tuple}}","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.Dataset, ::Type{<:Tuple})\n\nreturn an Tuple\n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Tuple{HDF5.Dataset, Type{<:Union{Real, Unitful.AbstractQuantity{<:Real}}}}","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.Dataset, ::Type{<:RealQuantity})\n\nreturn a value with type RealQuantity\n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Tuple{HDF5.H5DataStore, Type{<:AbstractVector{<:AbstractVector}}}","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.DataStore, ::Type{<:AbstractVector{<:AbstractVector{<:RealQuantity}}})\n\nreturn a VectorOfVectors object where data is an LH5Array  (see VectorOfArrays)\n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Tuple{HDF5.H5DataStore, Type{<:AbstractVector{<:RadiationDetectorSignals.RDWaveform}}}","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.DataStore, ::Type{<:AbstractVector{<:RDWaveform}})\n\nreturn an ArrayOfRDWaveforms where the field signal is either a  VectorOfSimilarVectors with an LH5Array as data or VectorOfVectors  with an LH5Array as data (see ArrayOfRDWaveforms and ArraysOfArrays) \n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Tuple{HDF5.H5DataStore, Type{<:EncodedArrays.AbstractEncodedArray{T, 1} where T}}","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.H5DataStore, ::Type{<:AbstractEncodedArray{T, 1} where {T}})\n\nreturn an EncodedArray\n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Tuple{HDF5.H5DataStore, Type{<:EncodedArrays.VectorOfEncodedArrays{T, 1, C, VS} where {T, C<:EncodedArrays.AbstractArrayCodec, VS<:(AbstractVector{<:Tuple{var\"#s12\"} where var\"#s12\"<:Integer})}}}","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.H5DataStore, ::Type{<:VectorOfEncodedArrays{T, 1} where {T}})\n\nreturn a VectorOfEncodedArrays\n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Tuple{HDF5.H5DataStore, Type{<:EncodedArrays.VectorOfEncodedSimilarArrays{T, 1} where T}}","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.H5DataStore, ::Type{<:VectorOfEncodedSimilarArrays{T, 1} where {T}})\n\nreturn a VectorOfEncodedSimilarArrays\n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Tuple{HDF5.H5DataStore, Type{<:StatsBase.Histogram{<:Union{Real, Unitful.AbstractQuantity{<:Real}}}}}","page":"API","title":"LegendHDF5IO.LH5Array","text":"\"     LH5Array(ds::HDF5.H5DataStore, ::Type{<:Histogram{<:RealQuantity}})\n\nreturn a Histogram. \n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Union{Tuple{T}, Tuple{HDF5.H5DataStore, Type{<:NamedTuple{T}}}} where T","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.Dataset, ::Type{<:NamedTuple{T}}) where T\n\nreturn a NamedTuple where each field is the output of LH5Array applied to it.\n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LH5Array-Union{Tuple{T}, Tuple{HDF5.H5DataStore, Type{<:TypedTables.Table{var\"#s18\", N} where {var\"#s18\"<:(NamedTuple{T}), N}}}} where T","page":"API","title":"LegendHDF5IO.LH5Array","text":"LH5Array(ds::HDF5.DataStore, ::Type{<:TypedTables.Table{<:NamedTuple{(T)}}}) where T\n\nreturn a Table where each column is the output of LH5Array applied to it.\n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.LHDataStore","page":"API","title":"LegendHDF5IO.LHDataStore","text":"LHDataStore <: AbstractDict{String,Any}\n\nDictionary wrapper for HDF5.H5DataStore objects, which were constructed  according to the LEGEND data format in \".lh5\" files. \n\nConstructor:\n\nLHDataStore(h5ds::HDF5.DataStore)\n\nThis return an LHDataStore object that wraps an h5ds which will typically be an HDF5.File be may also be an HDF5.H5DataStore (e.g. an HDF5.Group). in general.\n\nTo read or write \".lh5\" file directly (without using HDF5.h5open first), we recommend using lh5open.\n\nSupports getindex and setindex! where getindex(lh::LHDataStore, s) returns  the output of LH5Array applied to data_store[s] and setindex!  creates and writes HDF5.Groups and HDF5.Datasets using chunks of size 1000  to the ondisk array. Currently supported are objects with types: AbstractArray{<:RealQuantity}, ArraysOfSimilarArrays{<:RealQuantity},  VectorOfVectors{<:RealQuantity}, NamedTuple,TypedTables.Table,  Vector{<:RDWaveform}. For AbstractArray{<:RealQuantity} It is assumed  that the last axis of the provided array corresponds to the event number  index.\n\nExample\n\njulia> using HDF5\njulia> h5ds = h5open(\"path/to/lhf/file\")\njulia> lhf = LHDataStore(h5ds)\njulia> lhf[\"raw\"]\n[...]\njulia> using Unitful\njulia> x = rand(100) * u\"ns\"\njulia> lhf[\"new\"] = x\n[...]\n\n\n\n\n\n","category":"type"},{"location":"api/#LegendHDF5IO.add_entries!","page":"API","title":"LegendHDF5IO.add_entries!","text":"add_entries!(lhd::LHDataStore, i::AbstractString, src::NamedTuple, \n    dest::NamedTuple=LH5Array(lhd.data_store[i]))\n\nextend the NamedTuple dest at lhd[i] with elements from src.\n\n\n\n\n\n","category":"function"},{"location":"api/#LegendHDF5IO.add_entries!-2","page":"API","title":"LegendHDF5IO.add_entries!","text":"add_entries!(lhd::LHDataStore, i::AbstractString, src::TypedTable.Table, \n    dest::TypedTable.Table=LH5Array(lhd.data_store[i]))\n\nextend the Table dest at lhd[i] with columns from src.\n\n\n\n\n\n","category":"function"},{"location":"api/#LegendHDF5IO.delete_entry!-Tuple{LHDataStore, AbstractString}","page":"API","title":"LegendHDF5IO.delete_entry!","text":"delete_entry!(lhd::LHDataStore, i::AbstractString)\n\nremove the dataset lhd[i] and adjust the datatype of the parent if necessary.  Currently supported are elements of NamedTuple, TypedTable.Table or  HDF5.Group. \n\n\n\n\n\n","category":"method"},{"location":"api/#LegendHDF5IO.lh5open","page":"API","title":"LegendHDF5IO.lh5open","text":"lh5open(f, filename::AbstractString, access::AbstractString = \"r\")\n\nReturn f(lh5open(f, filename, access)).\n\nOpens and closes the LEGEND HDF5 file filename automatically.\n\n\n\n\n\n","category":"function"},{"location":"api/#LegendHDF5IO.lh5open-2","page":"API","title":"LegendHDF5IO.lh5open","text":"lh5open(filename::AbstractString, access::AbstractString = \"r\")\n\nOpen a LEGEND HDF5 file and return an LHDataStore object.\n\nLEGEND HDF5 files typically use the file extention \".lh5\".\n\n\n\n\n\n","category":"function"},{"location":"LICENSE/#LICENSE","page":"LICENSE","title":"LICENSE","text":"","category":"section"},{"location":"LICENSE/","page":"LICENSE","title":"LICENSE","text":"using Markdown\nMarkdown.parse_file(joinpath(@__DIR__, \"..\", \"..\", \"LICENSE.md\"))","category":"page"},{"location":"#LegendHDF5IO.jl","page":"Home","title":"LegendHDF5IO.jl","text":"","category":"section"}]
}
