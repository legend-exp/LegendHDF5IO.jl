# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

_range_to_nt(r::AbstractRange) = (
    first = first(r),
    last = last(r),
    step = step(r)
)
# Rebuilt from first, last and the known length, which ranges compare by:
# first, last and step alone can give other ranges for decimal steps, and
# stepping from first can miss last in the last bit.
_nt_to_range(nt::NamedTuple, len::Integer) =
    range(nt.first, nt.last, length = len)
_nt_to_range(nt::NamedTuple{(:first, :last, :step), <:Tuple{Integer, Integer, Integer}}, ::Integer) =
    nt.first:nt.step:nt.last
    
_range_to_nt(r::UnitRange) = (
    first = first(r),
    last = last(r)
)
_nt_to_range(nt::NamedTuple{(:first, :last)}, ::Integer) =
    UnitRange(nt.first, nt.last)

_range_to_nt(r::LinRange) = (
    first = first(r),
    last = last(r),
    length = length(r)
)
_nt_to_range(nt::NamedTuple{(:first, :last, :length)}, ::Integer) =
    LinRange(nt.first, nt.last, nt.length)

_edge_to_nt(edge::AbstractRange) = _range_to_nt(edge)
_edge_to_nt(edge::AbstractVector) = collect(edge)
_edge_to_nt(edge::Vector{<:Real}) = edge

_nt_to_edge(nt::NamedTuple, len::Integer) = _nt_to_range(nt, len)
_nt_to_edge(nt::AbstractVector, ::Integer) = nt

function _nt_to_histogram(nt::NamedTuple)
    closedleft = nt.binning[1].closedleft
    all(b -> b.closedleft == closedleft, values(nt.binning)) || throw(ArgumentError(
        "Histogram axes with mixed bin closedness are not supported"))
    return StatsBase.Histogram(
        ntuple(i -> _nt_to_edge(nt.binning[i].binedges, size(nt.weights, i) + 1), length(nt.binning)),
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
