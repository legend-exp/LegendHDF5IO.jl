# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

"""
    LH5LazyBroadcast{T} <: AbstractLH5Vector{T}

A function broadcast lazily over a typed table: `f.(sel)` for a
PropertyFunction `f` and an [`LH5TypedTable`](@ref) `sel` returns one. It
reads nothing until it is indexed or iterated, then reads the rows of `sel`
it needs, in blocks when iterated, and applies `f` to them. The element type
`T` is inferred from `f` and the row type of `sel`, so `f.(sel)` and its
elements have inferable types. Pure column accessors, `(@pf \$energy).(sel)`,
return the column instead, and selections return typed tables.
"""
# The type carries what `f` returns for a block of rows, `RB`, so that reads
# infer, and not the type of the table, so that the table read from can be
# the columns `f` uses:
struct LH5LazyBroadcast{T, F, RB} <: AbstractLH5Vector{T}
    f::F
    table::LH5TypedTable
end

"""
    LH5LazyBroadcast(f, tbl::LH5TypedTable, reads::LH5TypedTable = tbl)

`f` broadcast over `tbl`, reading the rows from `reads`, a selection of the
columns of `tbl` that `f` uses.
"""
function LH5LazyBroadcast(f, tbl::LH5TypedTable{R, B}, reads::LH5TypedTable = tbl) where {R, B}
    RB = Base.promote_op(_broadcast, typeof(f), B)
    LH5LazyBroadcast{_array_eltype(RB), typeof(f), RB}(f, reads)
end

_broadcast(f, rows) = Broadcast.materialize(Broadcast.broadcasted(f, rows))

# The element type of an inferred array type, Any where inference gives none:
_array_eltype(::Type{A}) where {A} = A !== Union{} && A <: AbstractArray ? eltype(A) : Any

Base.length(bc::LH5LazyBroadcast) = length(bc.table)
_getindex_rows(bc::LH5LazyBroadcast{T, F, RB}, idxs) where {T, F, RB} = _broadcast(bc.f, bc.table[idxs])::RB
_first_block_state(bc::LH5LazyBroadcast) = _first_block_state(bc.table)

# The state is that of the typed table the broadcast reads from:
function _next_block(bc::LH5LazyBroadcast{T, F, RB}, state::_TypedState) where {T, F, RB}
    next = _next_block(bc.table, state)
    isnothing(next) ? nothing : (_broadcast(bc.f, first(next))::RB, last(next))
end

Base.broadcastable(bc::LH5LazyBroadcast) = bc[:]

# Showing a broadcast must not read it:
function Base.show(io::IO, bc::LH5LazyBroadcast)
    print(io, "LH5LazyBroadcast{", eltype(bc), "}(", bc.f, ", ")
    show(io, bc.table)
    print(io, ")")
end
Base.show(io::IO, ::MIME"text/plain", bc::LH5LazyBroadcast) = show(io, bc)
