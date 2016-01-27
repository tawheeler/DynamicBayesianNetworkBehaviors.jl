abstract DirichletPrior
type UniformPrior <: DirichletPrior
    α::Float64
    UniformPrior(α::Float64=1.0) = new(α)
end
type BDeuPrior <: DirichletPrior
    #=
    Assigns equal scores to Markov equivalent structures

    α_ijk = x/{q_i * r_i} ∀ j, k and some given x

    see DMU section 2.4.3
    =#

    x::Float64

    BDeuPrior(x::Float64=1.0) = new(x)
end

Base.print(io::IO, p::UniformPrior) = Base.print(io, "UniformPrior(%.2f)", p.α)
Base.print(io::IO, p::BDeuPrior) = @printf(io, "BDeuPrior(%.2f)", p.x)

function Base.get{I<:Integer, J<:Integer}(p::UniformPrior,
    var_index::Integer,
    nintervals::AbstractVector{I}, # [nvars]
    parents::AbstractVector{J},    # [nvars]
    )

    r = nintervals[var_index]
    q = isempty(parents) ? 1 : prod(nintervals[parents])
    α = p.α

    fill(α, r, q)::Matrix{Float64}
end
function Base.get{I<:Integer, J<:Integer}(p::BDeuPrior,
    var_index::Integer,
    nintervals::AbstractVector{I}, # [nvars]
    parents::AbstractVector{J},    # [nvars]
    )

    x = p.x
    r = nintervals[var_index]
    q = isempty(parents) ? 1 : prod(nintervals[parents])

    α = x / (r*q)

    fill(α, r, q)::Matrix{Float64}
end