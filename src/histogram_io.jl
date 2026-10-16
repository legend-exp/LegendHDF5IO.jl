# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

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
    closedleft = nt.binning[1].closedleft
    all(b -> b.closedleft == closedleft, values(nt.binning)) || throw(ArgumentError(
        "Histogram axes with mixed bin closedness are not supported"))
    return StatsBase.Histogram(
        tuple(map(b ->_nt_to_edge(b.binedges), nt.binning)...),
        nt.weights,
        closedleft ? :left : :right,
        nt.isdensity
    )
end
    
function _histogram_to_nt(h::StatsBase.Histogram)
    axs_idxs = eachindex(h.edges)
    axs_sym = Symbol.(["axis_$(i)" for i in axs_idxs])
    axs = [(
        binedges = _edge_to_nt(h.edges[i]),
        closedleft = h.closed == :left
    ) for i in axs_idxs]
    return (
        binning = NamedTuple{tuple(axs_sym...)}(axs),
        weights = h.weights,
        isdensity = h.isdensity
    )
end
