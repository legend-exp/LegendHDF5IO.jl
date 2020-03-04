_range_to_nt(r::AbstractRange) = (
    first = first(r),
    last = last(r),
    step = step(r)
)
_nt_to_range(nt::NamedTuple) =
    range(nt.first, nt.last, step = nt.step)
    
_range_to_nt(r::UnitRange) = (
    first = first(r),
    last = last(r)
)
_nt_to_range(nt::NamedTuple{(:first, :last)}) =
    UnitRange(nt.first, nt.last)

_range_to_nt(r::LinRange) = (
    first = first(r),
    last = last(r),
    length = length(r)
)
_nt_to_range(nt::NamedTuple{(:first, :last, :length)}) =
    LinRange(nt.first, nt.last, nt.length)

_edge_to_nt(edge::AbstractRange) = _range_to_nt(edge)
_edge_to_nt(edge::AbstractVector) = collect(edge)
_edge_to_nt(edge::Vector{<:Real}) = edge

_nt_to_edge(nt::NamedTuple) = _nt_to_range(nt)
_nt_to_edge(nt::AbstractVector) = nt

function _nt_to_histogram(nt::NamedTuple)
    return StatsBase.Histogram(
        tuple(map(b ->_nt_to_edge(b.binedges), nt.binning)...), 
        nt.weights, 
        nt.binning[1].closedleft ? :left : :right, 
        nt.isdensity
    )  
end
    
function _histogram_to_nt(h::StatsBase.Histogram)
    n::Int = ndims(h.weights)
    axs_sym = Symbol.(["axis_$(i)" for i in Base.OneTo(n)])
    axs = [(
        binedges = _edge_to_nt(h.edges[i]),
        closedleft = h.closed == :left 
    ) for i in Base.OneTo(n)]
    return (
        ndims = n,
        binning = NamedTuple{tuple(axs_sym...)}(axs),
        weights = h.weights,
        isdensity = h.isdensity,
        closed = h.closed    
    )
end
        

function LegendDataTypes.writedata(
    output::HDF5.DataFile, name::AbstractString,
    x::Histogram,
    fulldatatype::DataType = typeof(x)
) where {T}
    @assert fulldatatype == typeof(x)
    writedata(output, name, _histogram_to_nt(x))
end


function LegendDataTypes.readdata(
    input::HDF5.DataFile, name::AbstractString,
    AT::Type{<:Histogram}
)
    _nt_to_histogram(readdata(input, name))
end

