# This file is a part of LegendHDF5IO.jl, licensed under the MIT License (MIT).

"""
    LH5LazyBroadcast{T} <: AbstractLH5Vector{T}

A function broadcast lazily over a lazy table: `f.(tbl)` for a
PropertyFunction `f` and an [`AbstractLH5Table`](@ref) `tbl` returns one. It
reads nothing until it is indexed or iterated, then reads the rows it needs
of the columns `f` uses, in blocks when iterated, and applies `f` to them.
Pure column accessors, `(@pf \$energy).(tbl)`, return the column instead,
and selections return typed tables.

The element type `T` is inferred from `f` and the row type. Over an
[`LH5TypedTable`](@ref) `sel`, that makes `f.(sel)` and its elements
inferable. Over other tables, the row type is fixed from the columns `f`
uses as for a selection, which opens the first file of a chain only. Where
inference gives no element type, `f` is applied to the first block of rows,
and the type of the result is that of all blocks: the other blocks are
converted to it, which throws for blocks that do not convert.
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
    isconcretetype(RB) && RB <: AbstractArray ? LH5LazyBroadcast{eltype(RB), typeof(f), RB}(f, reads) :
        _uninferred_broadcast(f, reads, RB)
end

_broadcast(f, rows) = Broadcast.materialize(Broadcast.broadcasted(f, rows))

# Inferred as a union of array types, the blocks give the union of their
# element types. Otherwise, all blocks take the type of the first, which an
# empty table does not have:
function _uninferred_broadcast(f, reads::LH5TypedTable, RB::Type)
    # TODO: Base.uniontypes is not public Julia API, other solution possible?
    types = Base.uniontypes(RB)
    if RB !== Union{} && all(A -> isconcretetype(A) && A <: AbstractArray, types)
        LH5LazyBroadcast{Union{map(eltype, types)...}, typeof(f), RB}(f, reads)
    else
        next = _next_block(reads, _first_block_state(reads))
        isnothing(next) && return LH5LazyBroadcast{Any, typeof(f), Any}(f, reads)
        FB = typeof(_broadcast(f, first(next)))
        LH5LazyBroadcast{eltype(FB), typeof(f), FB}(f, reads)
    end
end

# Only blocks typed by the first block can need converting:
function _block_result(::LH5LazyBroadcast{T, F, RB}, x) where {T, F, RB}
    x isa RB && return x
    try
        convert(RB, x)::RB
    catch err
        err isa Union{MethodError, InexactError} || rethrow()
        throw(ArgumentError("A broadcast that does not infer takes its type from the result for the first block of rows, $RB, but the result for other rows is a $(typeof(x))"))
    end
end

Base.length(bc::LH5LazyBroadcast) = length(bc.table)
_getindex_rows(bc::LH5LazyBroadcast, idxs) = _block_result(bc, _broadcast(bc.f, bc.table[idxs]))
_first_block_state(bc::LH5LazyBroadcast) = _first_block_state(bc.table)

# The state is that of the typed table the broadcast reads from:
function _next_block(bc::LH5LazyBroadcast, state::_TypedState)
    next = _next_block(bc.table, state)
    isnothing(next) ? nothing : (_block_result(bc, _broadcast(bc.f, first(next))), last(next))
end

Base.broadcastable(bc::LH5LazyBroadcast) = bc[:]

# Showing a broadcast must not read it:
function Base.show(io::IO, bc::LH5LazyBroadcast)
    print(io, "LH5LazyBroadcast{", eltype(bc), "}(", bc.f, ", ")
    show(io, bc.table)
    print(io, ")")
end
Base.show(io::IO, ::MIME"text/plain", bc::LH5LazyBroadcast) = show(io, bc)
