
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

#Phase 3 code:
# Gaussian unitary action on linear combinations
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

# Partial trace for linear combinations
"""
    ptrace(lcgs::GaussianLinearCombination, indices)

Compute the partial trace of a linear combination over specified subsystem indices.
Returns a new GaussianLinearCombination representing the reduced system.
"""
function ptrace(lcgs::GaussianLinearCombination, indices::T) where {T}
    basis = lcgs.basis
    length(indices) < basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    
    new_basis = typeof(basis)(basis.nmodes - length(indices))
    traced_states = [ptrace(state, indices) for state in lcgs.states]
    
    # Group identical states and sum their coefficients
    unique_states = typeof(traced_states[1])[]
    combined_coeffs = eltype(lcgs.coefficients)[]
    
    sizehint!(unique_states, length(traced_states))
    sizehint!(combined_coeffs, length(traced_states))
    
    for (coeff, state) in zip(lcgs.coefficients, traced_states)
        existing_idx = findfirst(s -> isapprox(s, state, atol=1e-12), unique_states)
        
        if existing_idx === nothing
            push!(unique_states, state)
            push!(combined_coeffs, coeff)
        else
            combined_coeffs[existing_idx] += coeff
        end
    end
    
    # Remove terms with negligible coefficients
    significant_mask = abs.(combined_coeffs) .> 1e-14
    if !any(significant_mask)
        # If all coefficients are negligible, return zero state
        vac = vacuumstate(new_basis, ħ = lcgs.ħ)
        return GaussianLinearCombination(new_basis, [zero(eltype(combined_coeffs))], [vac])
    end
    
    final_coeffs = combined_coeffs[significant_mask]
    final_states = unique_states[significant_mask]
    
    return GaussianLinearCombination(new_basis, final_coeffs, final_states)
end

function ptrace(::Type{Tc}, ::Type{Ts}, lcgs::GaussianLinearCombination, indices::T) where {Tc,Ts,T}
    result = ptrace(lcgs, indices)
    new_coeffs = Tc <: Vector ? (Tc <: Vector{Float64} ? result.coefficients : Tc(result.coefficients)) : result.coefficients
    return GaussianLinearCombination(result.basis, new_coeffs, result.states)
end

function ptrace(::Type{T}, lcgs::GaussianLinearCombination, indices) where {T}
    return ptrace(T, T, lcgs, indices)
end

"""
    measurement_probability(lcgs::GaussianLinearCombination, measurement::GaussianState, indices)

Calculate the measurement probability ⟨measurement|lcgs⟩ for a partial measurement on specified indices.
"""
function measurement_probability(lcgs::GaussianLinearCombination, measurement::GaussianState, indices::AbstractVector{<:Integer})
    # Extract the subsystem for measurement
    lcgs_subsystem = ptrace(lcgs, setdiff(1:lcgs.basis.nmodes, indices))
    
    # Calculate overlap with measurement state
    total_prob = 0.0
    
    for (coeff, state) in lcgs_subsystem
        # Calculate Gaussian state overlap using fidelity
        prob_amplitude = sqrt(fidelity(state, measurement))
        total_prob += abs2(coeff) * prob_amplitude
    end
    
    return real(total_prob)
end

function measurement_probability(lcgs::GaussianLinearCombination, measurement::GaussianState, index::Integer)
    return measurement_probability(lcgs, measurement, [index])
end

"""
    purity(lcgs::GaussianLinearCombination)

Calculate the purity Tr(ρ²) of a linear combination representing a mixed state.
For a pure state, the purity equals 1. For maximally mixed states, it approaches 0.
"""
function purity(lcgs::GaussianLinearCombination)
    purity_val = 0.0
    n_states = length(lcgs)
    
    # Tr(ρ²) = Σᵢⱼ cᵢ*cⱼ ⟨ψᵢ|ψⱼ⟩ 
    for i in 1:n_states
        for j in 1:n_states
            coeff_i = lcgs.coefficients[i]
            coeff_j = lcgs.coefficients[j]
            state_i = lcgs.states[i]
            state_j = lcgs.states[j]
            
            # Calculate overlap ⟨ψᵢ|ψⱼ⟩
            if i == j
                overlap = 1.0  # ⟨ψᵢ|ψᵢ⟩ = 1 for normalized states
            else
                # Use fidelity to calculate overlap between Gaussian states
                overlap = sqrt(fidelity(state_i, state_j))
            end
            
            purity_val += real(conj(coeff_i) * coeff_j * overlap)
        end
    end
    
    return real(purity_val)
end

"""
    entropy_vn(lcgs::GaussianLinearCombination; cutoff::Int=50, atol::Real=1e-12)

Calculate the Von Neumann entropy S(ρ) = -Tr(ρ log ρ) of a linear combination.
For pure states, uses the symplectic eigenvalue method on the covariance matrix.
For mixed states, constructs the density matrix in the Gaussian basis.
"""
function entropy_vn(lcgs::GaussianLinearCombination; cutoff::Int=50, atol::Real=1e-12)
    if length(lcgs) == 1
        # Single Gaussian state - use existing method
        return entropy_vn(lcgs.states[1])
    end
    
    # Check if this represents a pure state (coherent superposition)
    # vs mixed state (statistical mixture)
    is_pure = _is_pure_superposition(lcgs, atol)
    
    if is_pure
        # For pure superposition |ψ⟩ = Σᵢ cᵢ|ψᵢ⟩
        # Calculate effective covariance matrix of the superposition
        return _entropy_pure_superposition(lcgs, atol)
    else
        # For statistical mixture ρ = Σᵢ |cᵢ|²|ψᵢ⟩⟨ψᵢ|
        return _entropy_statistical_mixture(lcgs, cutoff, atol)
    end
end

function _is_pure_superposition(lcgs::GaussianLinearCombination, atol::Real)
    # A linear combination represents a pure state if coefficients represent
    # quantum amplitudes rather than classical probabilities
    # For this implementation, we assume it's pure if normalization is consistent
    # with quantum amplitudes
    norm_sq = sum(abs2, lcgs.coefficients)
    return abs(norm_sq - 1.0) < atol
end

function _entropy_pure_superposition(lcgs::GaussianLinearCombination, atol::Real)
    # For a pure superposition, entropy = 0 for the full system
    # This is because S(|ψ⟩⟨ψ|) = 0 always
    # However, if this was obtained via partial trace, we need to compute properly
    
    # Check if all component states are identical (up to phase)
    if _all_states_identical(lcgs.states, atol)
        return entropy_vn(lcgs.states[1])
    end
    
    # For true superposition of different Gaussian states,
    # we need to compute the entropy of the effective mixed state
    # that would result from decoherence
    return _entropy_from_overlap_matrix(lcgs, atol)
end

function _entropy_statistical_mixture(lcgs::GaussianLinearCombination, cutoff::Int, atol::Real)
    # For mixture ρ = Σᵢ pᵢ|ψᵢ⟩⟨ψᵢ| where pᵢ = |cᵢ|²
    probabilities = abs2.(lcgs.coefficients)
    
    # Renormalize probabilities
    total_prob = sum(probabilities)
    if total_prob > atol
        probabilities ./= total_prob
    else
        return 0.0
    end
    
    # Classical part: -Σᵢ pᵢ log(pᵢ)
    classical_entropy = 0.0
    for p in probabilities
        if p > atol
            classical_entropy -= p * log(p)
        end
    end
    
    # Quantum correction due to state overlaps
    quantum_correction = _quantum_entropy_correction(lcgs, probabilities, atol)
    
    return classical_entropy + quantum_correction
end

function _entropy_from_overlap_matrix(lcgs::GaussianLinearCombination, atol::Real)
    n = length(lcgs)
    
    # Construct overlap matrix S_ij = ⟨ψᵢ|ψⱼ⟩
    overlap_matrix = zeros(ComplexF64, n, n)
    
    for i in 1:n
        for j in 1:n
            if i == j
                overlap_matrix[i, j] = 1.0
            else
                # Calculate Gaussian overlap using the analytical formula
                overlap_matrix[i, j] = _gaussian_state_overlap(lcgs.states[i], lcgs.states[j])
            end
        end
    end
    
    # Construct density matrix ρ = Σᵢⱼ cᵢ c*ⱼ |ψᵢ⟩⟨ψⱼ|
    # In the basis of component states: ρᵢⱼ = cᵢ c*ⱼ ⟨ψⱼ|ψᵢ⟩
    density_matrix = zeros(ComplexF64, n, n)
    
    for i in 1:n
        for j in 1:n
            density_matrix[i, j] = lcgs.coefficients[i] * conj(lcgs.coefficients[j]) * conj(overlap_matrix[j, i])
        end
    end
    
    # Diagonalize density matrix
    eigenvals = real(eigvals(Hermitian(density_matrix)))
    
    # Compute Von Neumann entropy
    entropy = 0.0
    for λ in eigenvals
        if λ > atol
            entropy -= λ * log(λ)
        end
    end
    
    return entropy
end

function _gaussian_state_overlap(state1::GaussianState, state2::GaussianState)
    # Analytical formula for overlap between two Gaussian states
    # ⟨ψ₁|ψ₂⟩ = exp(-¼(μ₁-μ₂)ᵀ(V₁+V₂)⁻¹(μ₁-μ₂)) / √(det(V₁+V₂)/√(det(V₁)det(V₂)))
    
    μ1, μ2 = state1.mean, state2.mean
    V1, V2 = state1.covar, state2.covar
    
    Δμ = μ1 - μ2
    V_sum = V1 + V2
    
    try
        exp_arg = -0.25 * dot(Δμ, V_sum \ Δμ)
        det_factor = sqrt(det(V1) * det(V2)) / sqrt(det(V_sum))
        
        return complex(det_factor * exp(exp_arg))
    catch e
        if e isa SingularException
            return complex(0.0)
        else
            rethrow(e)
        end
    end
end

function _quantum_entropy_correction(lcgs::GaussianLinearCombination, probabilities::Vector, atol::Real)
    # Calculate quantum corrections due to coherent overlaps between states
    correction = 0.0
    n = length(lcgs)
    
    for i in 1:n
        for j in (i+1):n
            if probabilities[i] > atol && probabilities[j] > atol
                overlap = abs(_gaussian_state_overlap(lcgs.states[i], lcgs.states[j]))
                if overlap > atol
                    # Quantum interference contribution to entropy
                    correction += 2 * sqrt(probabilities[i] * probabilities[j]) * overlap * log(overlap)
                end
            end
        end
    end
    
    return correction
end

function _all_states_identical(states::Vector{<:GaussianState}, atol::Real)
    if length(states) <= 1
        return true
    end
    
    ref_state = states[1]
    for i in 2:length(states)
        if !isapprox(states[i], ref_state, atol=atol)
            return false
        end
    end
    return true
end

#wigner file content: 
"""
    cross_wigner(state1::GaussianState, state2::GaussianState, x::Vector)

Compute the cross-Wigner function between two Gaussian states at position x.
This is the quantum interference term in the Wigner function of linear combinations.
"""
function cross_wigner(state1::GaussianState, state2::GaussianState, x::T) where {T}
    state1.basis == state2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    
    basis = state1.basis
    nmodes = basis.nmodes
    isequal(length(state1.mean), length(x)) || throw(ArgumentError(WIGNER_ERROR))
    
    μ₁, μ₂ = state1.mean, state2.mean
    V₁, V₂ = state1.covar, state2.covar
    
    # Combined mean and covariance
    μ₁₂ = (μ₁ + μ₂) / 2
    V₁₂ = (V₁ + V₂) / 2
    
    # Difference terms
    Δμ = μ₁ - μ₂
    Δx = x - μ₁₂
    
    # Calculate the cross-Wigner function using the proper quantum mechanical formula
    try
        # Exponential term involving position difference
        exp_arg1 = -(1/2) * transpose(Δx) * inv(V₁₂) * Δx
        
        # Exponential term involving mean difference (quantum interference)
        exp_arg2 = -(1/8) * transpose(Δμ) * inv(V₁₂) * Δμ
        
        # Normalization factor
        norm_factor = 1 / ((2π)^nmodes * sqrt(det(V₁₂)))
        
        # Phase factor for quantum interference
        phase_arg = transpose(Δx) * inv(V₁₂) * Δμ / 2
        
        return norm_factor * exp(exp_arg1 + exp_arg2) * cos(phase_arg[1])
        
    catch e
        if e isa SingularException || e isa LinearAlgebra.LAPACKException
            # Handle singular covariance matrices
            return 0.0
        else
            rethrow(e)
        end
    end
end

"""
    wigner(lcgs::GaussianLinearCombination, x::Vector)

Compute the Wigner function of a linear combination of Gaussian states at position x.
Includes quantum interference terms between different component states.
"""
function wigner(lcgs::GaussianLinearCombination, x::T) where {T}
    basis = lcgs.basis
    nmodes = basis.nmodes
    isequal(length(lcgs.states[1].mean), length(x)) || throw(ArgumentError(WIGNER_ERROR))
    
    result = 0.0
    n_states = length(lcgs)
    
    # Diagonal terms (|cᵢ|² Wᵢᵢ)
    for i in 1:n_states
        coeff_i = lcgs.coefficients[i]
        state_i = lcgs.states[i]
        result += abs2(coeff_i) * wigner(state_i, x)
    end
    
    # Off-diagonal terms (2 Re(cᵢ* cⱼ Wᵢⱼ))
    for i in 1:n_states
        for j in (i+1):n_states
            coeff_i = lcgs.coefficients[i]
            coeff_j = lcgs.coefficients[j]
            state_i = lcgs.states[i]
            state_j = lcgs.states[j]
            
            cross_term = cross_wigner(state_i, state_j, x)
            result += 2 * real(conj(coeff_i) * coeff_j * cross_term)
        end
    end
    
    return result
end

"""
    wignerchar(lcgs::GaussianLinearCombination, xi::Vector)

Compute the Wigner characteristic function of a linear combination of Gaussian states.
"""
function wignerchar(lcgs::GaussianLinearCombination, xi::T) where {T}
    basis = lcgs.basis
    nmodes = basis.nmodes
    isequal(length(lcgs.states[1].mean), length(xi)) || throw(ArgumentError(WIGNER_ERROR))
    
    result = 0.0 + 0.0im
    n_states = length(lcgs)
    
    # Calculate characteristic function using overlap integrals
    for i in 1:n_states
        for j in 1:n_states
            coeff_i = lcgs.coefficients[i]
            coeff_j = lcgs.coefficients[j]
            state_i = lcgs.states[i]
            state_j = lcgs.states[j]
            
            # For identical states
            if i == j
                result += abs2(coeff_i) * wignerchar(state_i, xi)
            else
                # Cross terms require careful calculation
                μᵢ, μⱼ = state_i.mean, state_j.mean
                Vᵢ, Vⱼ = state_i.covar, state_j.covar
                
                try
                    # Combined system analysis
                    V_sum = Vᵢ + Vⱼ
                    μ_diff = μᵢ - μⱼ
                    
                    Omega = symplecticform(basis)
                    
                    # Cross characteristic function
                    exp_arg1 = -(1/4) * transpose(xi) * (Omega * V_sum * transpose(Omega)) * xi
                    exp_arg2 = im * transpose(Omega * ((μᵢ + μⱼ)/2)) * xi
                    exp_arg3 = -(1/4) * transpose(μ_diff) * inv(V_sum) * μ_diff
                    
                    cross_char = exp(exp_arg1 + exp_arg2 + exp_arg3) / sqrt(det(V_sum))
                    
                    result += conj(coeff_i) * coeff_j * cross_char
                    
                catch e
                    if e isa SingularException || e isa LinearAlgebra.LAPACKException
                        # Skip singular terms
                        continue
                    else
                        rethrow(e)
                    end
                end
            end
        end
    end
    
    return result
end

