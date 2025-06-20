
# Linear combinations of Gaussian states

function bases_compatible(b1, b2)
    
    return b1.nmodes == b2.nmodes && typeof(b1) == typeof(b2)
end


"""
    GaussianLinearCombination{B<:SymplecticBasis,C,S}

Represents a linear combination of Gaussian states of the form Σᵢ cᵢ|ψᵢ⟩ where cᵢ are coefficients 
and |ψᵢ⟩ are Gaussian states, all sharing the same symplectic basis and ħ value.

## Fields
- `basis::B`: Symplectic basis shared by all states
- `coefficients::Vector{C}`: Complex coefficients for the linear combination  
- `states::Vector{S}`: Vector of Gaussian states
- `ħ::Number`: Reduced Planck's constant (must be same for all states)

## Examples

```julia
# Create from single state
basis = QuadPairBasis(1)
state1 = coherentstate(basis, 1.0)
lcgs1 = GaussianLinearCombination(state1)

# Create cat state as linear combination
state2 = coherentstate(basis, -1.0)
cat_state = 0.5 * GaussianLinearCombination(state1) + 0.5 * GaussianLinearCombination(state2)

# Create from coefficient-state pairs
lcgs2 = GaussianLinearCombination([0.6, 0.8], [state1, state2])
```
"""
mutable struct GaussianLinearCombination{B<:SymplecticBasis,C,S}
    basis::B
    coefficients::Vector{C}
    states::Vector{S}
    ħ::Number
    
    function GaussianLinearCombination(basis::B, coeffs::Vector{C}, states::Vector{S}) where {B<:SymplecticBasis,C,S}
        length(coeffs) == length(states) || throw(DimensionMismatch("Number of coefficients ($(length(coeffs))) must match number of states ($(length(states)))"))
        isempty(states) && throw(ArgumentError("Cannot create an empty linear combination"))
        
        ħ = first(states).ħ
        for (i, state) in enumerate(states)
            state isa GaussianState || throw(ArgumentError("Element $i is not a GaussianState: got $(typeof(state))"))
            if !bases_compatible(state.basis, basis)
                throw(ArgumentError("State $i has incompatible basis: expected $(typeof(basis))($(basis.nmodes)), got $(typeof(state.basis))($(state.basis.nmodes))"))
            end
            if state.ħ != ħ
                throw(ArgumentError("State $i has different ħ: expected $ħ, got $(state.ħ)"))
            end
        end
        
        return new{B,C,S}(basis, coeffs, states, ħ)
    end
end

function GaussianLinearCombination(state::GaussianState{B,M,V}) where {B,M,V}
    coeff_type = float(real(eltype(M)))
    return GaussianLinearCombination(state.basis, [one(coeff_type)], [state])
end

function GaussianLinearCombination(pairs::Vector{<:Tuple})
    isempty(pairs) && throw(ArgumentError("Cannot create an empty linear combination"))
    coeffs = [convert(Number, p[1]) for p in pairs]
    states = [p[2] for p in pairs]
    
    for (i, state) in enumerate(states)
        state isa GaussianState || throw(ArgumentError("Element $i: second element must be a GaussianState"))
    end
    
    basis = first(states).basis
    return GaussianLinearCombination(basis, coeffs, states)
end

function GaussianLinearCombination(coeffs::Vector{<:Number}, states::Vector{<:GaussianState})
    isempty(states) && throw(ArgumentError("Cannot create an empty linear combination"))
    basis = first(states).basis
    return GaussianLinearCombination(basis, coeffs, states)
end

function GaussianLinearCombination(pairs::Pair{<:Number,<:GaussianState}...)
    isempty(pairs) && throw(ArgumentError("Cannot create an empty linear combination"))
    coeffs = [p.first for p in pairs]
    states = [p.second for p in pairs]
    basis = first(states).basis
    return GaussianLinearCombination(basis, coeffs, states)
end

# basic arithmetic operations


"""
    +(lc1::GaussianLinearCombination, lc2::GaussianLinearCombination)

Add two linear combinations of Gaussian states. Both must have the same symplectic basis and ħ value.
"""
function Base.:+(lc1::GaussianLinearCombination{B}, lc2::GaussianLinearCombination{B}) where {B<:SymplecticBasis}
    bases_compatible(lc1.basis, lc2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    lc1.ħ == lc2.ħ || throw(ArgumentError(HBAR_ERROR))
    
    # Combine coefficients and states
    coeffs = vcat(lc1.coefficients, lc2.coefficients)
    states = vcat(lc1.states, lc2.states)
    
    return GaussianLinearCombination(lc1.basis, coeffs, states)
end



"""
    -(lc1::GaussianLinearCombination, lc2::GaussianLinearCombination)

Subtract two linear combinations of Gaussian states.
"""
function Base.:-(lc1::GaussianLinearCombination{B}, lc2::GaussianLinearCombination{B}) where {B<:SymplecticBasis}
    return lc1 + (-1) * lc2
end

# Handle operations between different basis types
function Base.:-(lc1::GaussianLinearCombination{B1}, lc2::GaussianLinearCombination{B2}) where {B1<:SymplecticBasis,B2<:SymplecticBasis}
    throw(ArgumentError(SYMPLECTIC_ERROR))
end

"""
    -(lc::GaussianLinearCombination)

Negate a linear combination of Gaussian states.
"""
function Base.:-(lc::GaussianLinearCombination)
    return (-1) * lc
end

"""
    *(α::Number, lc::GaussianLinearCombination)

Multiply a linear combination by a scalar from the left.
"""
function Base.:*(α::Number, lc::GaussianLinearCombination)
    new_coeffs = α .* lc.coefficients
    return GaussianLinearCombination(lc.basis, new_coeffs, copy(lc.states))
end

"""
    *(lc::GaussianLinearCombination, α::Number)

Multiply a linear combination by a scalar from the right.
"""
function Base.:*(lc::GaussianLinearCombination, α::Number)
    return α * lc
end

"""
    zero(lc::GaussianLinearCombination)

Create a zero linear combination with the same basis and ħ as the input.
Returns a linear combination with a single vacuum state and zero coefficient.
"""
function Base.zero(lc::GaussianLinearCombination)
    vac = vacuumstate(lc.basis, ħ = lc.ħ)
    coeff_type = eltype(lc.coefficients)
    return GaussianLinearCombination(lc.basis, [zero(coeff_type)], [vac])
end

"""
    Gabs.normalize!(lc::GaussianLinearCombination)

Normalize the coefficients of a linear combination in-place using L2 norm.
Note: Use Gabs.normalize! to avoid conflicts with LinearAlgebra.normalize!
"""
function normalize!(lc::GaussianLinearCombination)
    norm_val = sqrt(sum(abs2, lc.coefficients))
    if norm_val > 0
        lc.coefficients ./= norm_val
    end
    return lc
end

function Base.:+(lc1::GaussianLinearCombination{B1}, lc2::GaussianLinearCombination{B2}) where {B1<:SymplecticBasis,B2<:SymplecticBasis}
    throw(ArgumentError(SYMPLECTIC_ERROR))
end
function Base.:-(lc1::GaussianLinearCombination{B1}, lc2::GaussianLinearCombination{B2}) where {B1<:SymplecticBasis,B2<:SymplecticBasis}
    throw(ArgumentError(SYMPLECTIC_ERROR))
end

# Utility functions


"""
    length(lc::GaussianLinearCombination)

Return the number of terms in the linear combination.
"""
Base.length(lc::GaussianLinearCombination) = length(lc.coefficients)


"""
    getindex(lc::GaussianLinearCombination, i::Integer)

Access the i-th term as a (coefficient, state) tuple.
"""
Base.getindex(lc::GaussianLinearCombination, i::Integer) = (lc.coefficients[i], lc.states[i])


"""
    iterate(lc::GaussianLinearCombination, state=1)

Iterate over (coefficient, state) pairs in the linear combination.
"""
function Base.iterate(lc::GaussianLinearCombination, state::Int=1)
    if state > length(lc)
        return nothing
    else
        return ((lc.coefficients[state], lc.states[state]), state + 1)
    end
end

"""
    Gabs.simplify!(lc::GaussianLinearCombination; atol::Real=1e-14)

Simplify a linear combination by:
1. Removing coefficients smaller than `atol`
2. Combining terms with identical states
3. Ensuring at least one term remains (with vacuum state if all coefficients are zero)

Returns the modified linear combination.
"""
function simplify!(lc::GaussianLinearCombination; atol::Real=1e-14)
    if isempty(lc.coefficients)
        return lc
    end
    
    keep_mask = abs.(lc.coefficients) .> atol
    
    if !any(keep_mask)
        vac = vacuumstate(lc.basis, ħ = lc.ħ)
        coeff_type = eltype(lc.coefficients)
        lc.coefficients = [zero(coeff_type)]
        lc.states = [vac]
        return lc
    end
    
    coeffs = lc.coefficients[keep_mask]
    states = lc.states[keep_mask]
    
    unique_states = typeof(states[1])[]
    combined_coeffs = eltype(coeffs)[]
    sizehint!(unique_states, length(states))
    sizehint!(combined_coeffs, length(states))
    
    for (coeff, state) in zip(coeffs, states)
        existing_idx = findfirst(s -> isapprox(s, state, atol=1e-12), unique_states)
        
        if existing_idx === nothing
            push!(unique_states, state)
            push!(combined_coeffs, coeff)
        else
            combined_coeffs[existing_idx] += coeff
        end
    end
    
    final_mask = abs.(combined_coeffs) .> atol
    if !any(final_mask)
        vac = vacuumstate(lc.basis, ħ = lc.ħ)
        coeff_type = eltype(combined_coeffs)
        lc.coefficients = [zero(coeff_type)]
        lc.states = [vac]
    else
        lc.coefficients = combined_coeffs[final_mask]
        lc.states = unique_states[final_mask]
    end
    
    return lc
end



function Base.show(io::IO, mime::MIME"text/plain", lc::GaussianLinearCombination)
    basis_name = nameof(typeof(lc.basis))
    nmodes = lc.basis.nmodes
    
    print(io, "GaussianLinearCombination with $(length(lc)) terms")
    if nmodes == 1
        println(io, " for 1 mode.")
    else
        println(io, " for $(nmodes) modes.")
    end
    
    println(io, "  symplectic basis: ", basis_name)
    println(io, "  ħ = $(lc.ħ)")
    
    max_display = min(length(lc), 5)
    for i in 1:max_display
        coeff, state = lc[i]
        println(io, "  [$i] $(coeff) * GaussianState")
    end
    
    if length(lc) > max_display
        println(io, "  ⋮ ($(length(lc) - max_display) more terms)")
    end
end

function Base.show(io::IO, lc::GaussianLinearCombination)
    print(io, "GaussianLinearCombination($(length(lc)) terms)")
end

function Base.:(==)(lc1::GaussianLinearCombination, lc2::GaussianLinearCombination)
    bases_compatible(lc1.basis, lc2.basis) || return false

    lc1.ħ == lc2.ħ || return false
    length(lc1) == length(lc2) || return false
    
    return lc1.coefficients == lc2.coefficients && all(isequal(s1, s2) for (s1, s2) in zip(lc1.states, lc2.states))
end

function Base.isapprox(lc1::GaussianLinearCombination, lc2::GaussianLinearCombination; kwargs...)
    bases_compatible(lc1.basis, lc2.basis) || return false

    lc1.ħ == lc2.ħ || return false
    length(lc1) == length(lc2) || return false
    
    return isapprox(lc1.coefficients, lc2.coefficients; kwargs...) && 
           all(isapprox(s1, s2; kwargs...) for (s1, s2) in zip(lc1.states, lc2.states))
end


"""
    *(op::GaussianUnitary, lcgs::GaussianLinearCombination)

Apply a Gaussian unitary operator to a linear combination of Gaussian states.
Returns a new GaussianLinearCombination where the unitary is applied to each component state.
"""
function Base.:*(op::GaussianUnitary, lcgs::GaussianLinearCombination)
    op.basis == lcgs.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    op.ħ == lcgs.ħ || throw(ArgumentError(HBAR_ERROR))
    
    new_states = [op * state for state in lcgs.states]
    return GaussianLinearCombination(lcgs.basis, copy(lcgs.coefficients), new_states)
end

# Gaussian channel action on linear combinations
"""
    *(op::GaussianChannel, lcgs::GaussianLinearCombination)

Apply a Gaussian channel to a linear combination of Gaussian states.
Returns a new GaussianLinearCombination where the channel is applied to each component state.
"""
function Base.:*(op::GaussianChannel, lcgs::GaussianLinearCombination)
    op.basis == lcgs.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    op.ħ == lcgs.ħ || throw(ArgumentError(HBAR_ERROR))
    
    new_states = [op * state for state in lcgs.states]
    return GaussianLinearCombination(lcgs.basis, copy(lcgs.coefficients), new_states)
end

# Tensor products for linear combinations
"""
    tensor(lcgs1::GaussianLinearCombination, lcgs2::GaussianLinearCombination)

Compute the tensor product of two linear combinations of Gaussian states.
Returns a new GaussianLinearCombination with all pairwise tensor products.
"""
function tensor(lcgs1::GaussianLinearCombination{B}, lcgs2::GaussianLinearCombination{B}) where {B<:SymplecticBasis}
    typeof(lcgs1.basis) == typeof(lcgs2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    lcgs1.ħ == lcgs2.ħ || throw(ArgumentError(HBAR_ERROR))
    
    new_basis = lcgs1.basis ⊕ lcgs2.basis
    
    # Pre-allocate arrays for efficiency
    result_size = length(lcgs1) * length(lcgs2)
    CoeffType = promote_type(eltype(lcgs1.coefficients), eltype(lcgs2.coefficients))
    new_coeffs = Vector{CoeffType}(undef, result_size)
    new_states = Vector{GaussianState}(undef, result_size)
    
    idx = 1
    for (c1, s1) in lcgs1
        for (c2, s2) in lcgs2
            new_coeffs[idx] = c1 * c2
            new_states[idx] = s1 ⊗ s2
            idx += 1
        end
    end
    
    return GaussianLinearCombination(new_basis, new_coeffs, new_states)
end

function tensor(::Type{Tc}, ::Type{Ts}, lcgs1::GaussianLinearCombination, lcgs2::GaussianLinearCombination) where {Tc,Ts}
    result = tensor(lcgs1, lcgs2)
    new_coeffs = Tc <: Vector ? (Tc <: Vector{Float64} ? result.coefficients : Tc(result.coefficients)) : result.coefficients
    return GaussianLinearCombination(result.basis, new_coeffs, result.states)
end

function tensor(::Type{T}, lcgs1::GaussianLinearCombination, lcgs2::GaussianLinearCombination) where {T}
    return tensor(T, T, lcgs1, lcgs2)
end

# Add tensor products for different basis types with proper error handling
function tensor(lcgs1::GaussianLinearCombination{B1}, lcgs2::GaussianLinearCombination{B2}) where {B1<:SymplecticBasis,B2<:SymplecticBasis}
    throw(ArgumentError(SYMPLECTIC_ERROR))
end

# Tensor product alias
const ⊗ = tensor


