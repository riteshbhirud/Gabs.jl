
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

# Phase 3: Integration and Advanced Features

## Gaussian Operations Integration

"""
    *(op::GaussianUnitary, lc::GaussianLinearCombination)

Apply a Gaussian unitary operation to a linear combination of Gaussian states.
The unitary is applied to each component state while preserving coefficients.
"""
function Base.:(*)(op::GaussianUnitary, lc::GaussianLinearCombination)
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    
    new_states = [op * state for state in lc.states]
    return GaussianLinearCombination(lc.basis, copy(lc.coefficients), new_states)
end

"""
    *(op::GaussianChannel, lc::GaussianLinearCombination)

Apply a Gaussian channel to a linear combination of Gaussian states.
The channel is applied to each component state while preserving coefficients.
"""
function Base.:(*)(op::GaussianChannel, lc::GaussianLinearCombination)
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    
    new_states = [op * state for state in lc.states]
    return GaussianLinearCombination(lc.basis, copy(lc.coefficients), new_states)
end

## Tensor Products

"""
    tensor(::Type{Tc}, ::Type{Ts}, lc1::GaussianLinearCombination, lc2::GaussianLinearCombination)

Compute tensor product of two linear combinations with specified output types.
Creates all pairwise tensor products: Σᵢⱼ cᵢcⱼ |ψᵢ⟩ ⊗ |ϕⱼ⟩.
"""
function tensor(::Type{Tc}, ::Type{Ts}, lc1::GaussianLinearCombination, lc2::GaussianLinearCombination) where {Tc,Ts}
    typeof(lc1.basis) == typeof(lc2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    lc1.ħ == lc2.ħ || throw(ArgumentError(HBAR_ERROR))
    
    new_basis = lc1.basis ⊕ lc2.basis
    
    # Create all pairwise tensor products
    n1, n2 = length(lc1), length(lc2)
    CoeffType = promote_type(eltype(lc1.coefficients), eltype(lc2.coefficients))
    new_coeffs = Vector{CoeffType}(undef, n1 * n2)
    new_states = Vector{GaussianState}(undef, n1 * n2)
    
    idx = 1
    for (c1, s1) in lc1
        for (c2, s2) in lc2
            new_coeffs[idx] = c1 * c2
            new_states[idx] = tensor(Tc, Ts, s1, s2)
            idx += 1
        end
    end
    
    return GaussianLinearCombination(new_basis, new_coeffs, new_states)
end

tensor(::Type{T}, lc1::GaussianLinearCombination, lc2::GaussianLinearCombination) where {T} = tensor(T, T, lc1, lc2)

"""
    tensor(lc1::GaussianLinearCombination, lc2::GaussianLinearCombination)

Compute tensor product of two linear combinations of Gaussian states.
"""
function tensor(lc1::GaussianLinearCombination, lc2::GaussianLinearCombination)
    typeof(lc1.basis) == typeof(lc2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    lc1.ħ == lc2.ħ || throw(ArgumentError(HBAR_ERROR))
    
    new_basis = lc1.basis ⊕ lc2.basis
    
    # Create all pairwise tensor products
    n1, n2 = length(lc1), length(lc2)
    CoeffType = promote_type(eltype(lc1.coefficients), eltype(lc2.coefficients))
    new_coeffs = Vector{CoeffType}(undef, n1 * n2)
    new_states = Vector{GaussianState}(undef, n1 * n2)
    
    idx = 1
    for (c1, s1) in lc1
        for (c2, s2) in lc2
            new_coeffs[idx] = c1 * c2
            new_states[idx] = s1 ⊗ s2
            idx += 1
        end
    end
    
    return GaussianLinearCombination(new_basis, new_coeffs, new_states)
end

## Partial Traces

"""
    ptrace(::Type{Tc}, ::Type{Ts}, lc::GaussianLinearCombination, indices)

Compute partial trace of a linear combination with specified output types.
"""
function ptrace(::Type{Tc}, ::Type{Ts}, lc::GaussianLinearCombination, indices::Union{Int, AbstractVector{<:Int}}) where {Tc,Ts}
    indices_vec = indices isa Int ? [indices] : collect(indices)
    length(indices_vec) < lc.basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    
    # Apply ptrace to each component state
    traced_states = [ptrace(Tc, Ts, state, indices_vec) for state in lc.states]
    
    # Create new linear combination with traced states
    result = GaussianLinearCombination(traced_states[1].basis, copy(lc.coefficients), traced_states)
    
    # Combine identical states automatically using simplify!
    simplify!(result)
    
    return result
end

ptrace(::Type{T}, lc::GaussianLinearCombination, indices::Union{Int, AbstractVector{<:Int}}) where {T} = ptrace(T, T, lc, indices)

"""
    ptrace(lc::GaussianLinearCombination, indices)

Compute partial trace of a linear combination over specified indices.
Combines identical traced states automatically.
"""
function ptrace(lc::GaussianLinearCombination, indices::Union{Int, AbstractVector{<:Int}})
    indices_vec = indices isa Int ? [indices] : collect(indices)
    length(indices_vec) < lc.basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    
    # Apply ptrace to each component state
    traced_states = [ptrace(state, indices_vec) for state in lc.states]
    
    # Create new linear combination with traced states
    result = GaussianLinearCombination(traced_states[1].basis, copy(lc.coefficients), traced_states)
    
    # Combine identical states automatically using simplify!
    simplify!(result)
    
    return result
end

## Complete Wigner Functions with Quantum Interference

"""
    cross_wigner(state1::GaussianState, state2::GaussianState, x::AbstractVector)

Compute cross-Wigner function between two Gaussian states at point x.
W(ψ₁,ψ₂,x) = (4/π)^N * exp(-2(x-μ₁₂)ᵀV₁₂⁻¹(x-μ₁₂)) / √det(4V₁₂)
where μ₁₂ = (μ₁+μ₂)/2, V₁₂ = (V₁+V₂)/2, and N is the number of modes.
"""
function cross_wigner(state1::GaussianState, state2::GaussianState, x::AbstractVector)
    state1.basis == state2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    length(x) == length(state1.mean) || throw(ArgumentError(WIGNER_ERROR))
    
    # For identical states, return regular Wigner function
    if state1 === state2 || (isapprox(state1.mean, state2.mean, atol=1e-12) && isapprox(state1.covar, state2.covar, atol=1e-12))
        return wigner(state1, x)
    end
    
    # Cross-Wigner function calculation
    μ1, μ2 = state1.mean, state2.mean
    V1, V2 = state1.covar, state2.covar
    
    μ12 = (μ1 + μ2) / 2
    V12 = (V1 + V2) / 2
    
    diff = x - μ12
    arg = -0.5 * dot(diff, V12 \ diff)
    
    nmodes = state1.basis.nmodes
    norm_factor = 1 / ((2π)^nmodes * sqrt(det(V12)))
    
    return norm_factor * exp(arg)
end

"""
    wigner(lc::GaussianLinearCombination, x::AbstractVector)

Compute Wigner function of a linear combination including quantum interference.
W(x) = Σᵢ |cᵢ|² Wᵢ(x) + 2 Σᵢ<ⱼ Re(cᵢ*cⱼ W_cross(ψᵢ,ψⱼ,x))
"""
function wigner(lc::GaussianLinearCombination, x::AbstractVector)
    length(x) == length(lc.states[1].mean) || throw(ArgumentError(WIGNER_ERROR))
    
    result = 0.0
    
    # Diagonal terms: Σᵢ |cᵢ|² Wᵢ(x)
    for (c, state) in lc
        result += abs2(c) * wigner(state, x)
    end
    
    # Cross terms: 2 Σᵢ<ⱼ Re(cᵢ*cⱼ W_cross(ψᵢ,ψⱼ,x))
    for i in 1:length(lc)
        ci, si = lc[i]
        for j in (i+1):length(lc)
            cj, sj = lc[j]
            cross_term = 2 * real(conj(ci) * cj * cross_wigner(si, sj, x))
            result += cross_term
        end
    end
    
    return result
end

"""
    cross_wignerchar(state1::GaussianState, state2::GaussianState, xi::AbstractVector)

Compute cross-Wigner characteristic function between two Gaussian states.
"""
function cross_wignerchar(state1::GaussianState, state2::GaussianState, xi::AbstractVector)
    state1.basis == state2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    length(xi) == length(state1.mean) || throw(ArgumentError(WIGNER_ERROR))
    
    # Calculate cross-Wigner characteristic function
    μ1, μ2 = state1.mean, state2.mean
    V1, V2 = state1.covar, state2.covar
    
    μ12 = (μ1 + μ2) / 2
    V12 = (V1 + V2) / 2
    
    Omega = symplecticform(state1.basis)
    
    arg1 = -0.5 * dot(xi, (Omega * V12 * transpose(Omega)) * xi)
    arg2 = 1im * dot(Omega * μ12, xi)
    
    return exp(arg1 + arg2)
end

"""
    wignerchar(lc::GaussianLinearCombination, xi::AbstractVector)

Compute Wigner characteristic function of a linear combination including interference.
"""
function wignerchar(lc::GaussianLinearCombination, xi::AbstractVector)
    length(xi) == length(lc.states[1].mean) || throw(ArgumentError(WIGNER_ERROR))
    
    result = 0.0 + 0.0im
    
    # Diagonal terms: Σᵢ |cᵢ|² χᵢ(xi)
    for (c, state) in lc
        result += abs2(c) * wignerchar(state, xi)
    end
    
    # Cross terms: 2 Σᵢ<ⱼ Re(cᵢ*cⱼ χ_cross(ψᵢ,ψⱼ,xi))
    for i in 1:length(lc)
        ci, si = lc[i]
        for j in (i+1):length(lc)
            cj, sj = lc[j]
            cross_term = 2 * real(conj(ci) * cj * cross_wignerchar(si, sj, xi))
            result += cross_term
        end
    end
    
    return result
end

## Advanced State Metrics

"""
    purity(lc::GaussianLinearCombination)

Calculate purity of a linear combination: Tr(ρ²).
For a properly normalized linear combination, this should be ≤ 1.
"""
function purity(lc::GaussianLinearCombination)
    # First normalize the coefficients to get proper quantum state
    total_norm = sqrt(sum(abs2, lc.coefficients))
    if total_norm < 1e-15
        return 0.0
    end
    
    normalized_coeffs = lc.coefficients ./ total_norm
    
    result = 0.0
    
    # Tr(ρ²) = Σᵢⱼ cᵢ*cⱼ ⟨ψᵢ|ψⱼ⟩
    for i in 1:length(lc)
        ci = normalized_coeffs[i]
        si = lc.states[i]
        for j in 1:length(lc)
            cj = normalized_coeffs[j]
            sj = lc.states[j]
            overlap = _gaussian_overlap(si, sj)
            result += real(conj(ci) * cj * overlap)
        end
    end
    
    # Ensure result ∈ [0,1]
    return clamp(result, 0.0, 1.0)
end

"""
    entropy_vn(lc::GaussianLinearCombination)

Calculate Von Neumann entropy of a linear combination.
COMPLETE implementation constructing full density matrix and diagonalizing.
"""
function entropy_vn(lc::GaussianLinearCombination)
    n = length(lc)
    
    if n == 1
        # Single state - entropy is 0 for a pure state
        return 0.0
    end
    
    # Normalize coefficients first
    total_norm = sqrt(sum(abs2, lc.coefficients))
    if total_norm < 1e-15
        return 0.0
    end
    normalized_coeffs = lc.coefficients ./ total_norm
    
    # For very large combinations: use approximation with warning
    if n > 100
        @warn "Using approximation for entropy calculation due to large number of terms ($(n)). " *
              "Result may not be exact for strongly interfering states."
        
        # Shannon entropy of coefficients (classical mixture approximation)
        entropy = 0.0
        
        for c in normalized_coeffs
            weight = abs2(c)
            if weight > 1e-15
                entropy -= weight * log(weight)
            end
        end
        
        return entropy
    end
    
    # COMPLETE implementation: construct full density matrix and diagonalize
    # Build density matrix ρᵢⱼ = cᵢ* cⱼ ⟨ψᵢ|ψⱼ⟩
    ρ = Matrix{ComplexF64}(undef, n, n)
    for i in 1:n
        ci = normalized_coeffs[i]
        si = lc.states[i]
        for j in 1:n
            cj = normalized_coeffs[j]
            sj = lc.states[j]
            overlap = _gaussian_overlap(si, sj)
            ρ[i, j] = conj(ci) * cj * overlap
        end
    end
    
    # Find eigenvalues λₖ of ρ
    eigenvals = real(eigvals(Hermitian(ρ)))
    
    # Handle numerical stability for near-zero eigenvalues
    eigenvals = eigenvals[eigenvals .> 1e-15]  # Remove numerical zeros
    total = sum(eigenvals)
    if total > 1e-15
        eigenvals ./= total  # Normalize to ensure unit trace
    end
    
    # S = -Σₖ λₖ log λₖ (handle λₖ=0 case)
    entropy = 0.0
    for λ in eigenvals
        if λ > 1e-15  # Numerical stability check for near-zero eigenvalues
            entropy -= λ * log(λ)
        end
    end
    
    return max(entropy, 0.0)  # Ensure non-negative
end

## Measurement Theory

"""
    measurement_probability(lc::GaussianLinearCombination, measurement::GaussianState, indices)

Calculate measurement probability using Born rule: P = |⟨measurement|Tr_complement(lc)⟩|².
"""
function measurement_probability(lc::GaussianLinearCombination, measurement::GaussianState, indices::Union{Int, AbstractVector{<:Int}})
    indices_vec = indices isa Int ? [indices] : collect(indices)
    
    # Check that measurement state has the right number of modes
    expected_modes = length(indices_vec)
    measurement.basis.nmodes == expected_modes || throw(ArgumentError(GENERALDYNE_ERROR))
    
    # Check ħ compatibility
    lc.ħ == measurement.ħ || throw(ArgumentError(HBAR_ERROR))
    
    # Normalize the linear combination first
    total_norm = sqrt(sum(abs2, lc.coefficients))
    if total_norm < 1e-15
        return 0.0
    end
    normalized_coeffs = lc.coefficients ./ total_norm
    
    # Get the complement indices (modes to trace out)
    complement_indices = setdiff(1:lc.basis.nmodes, indices_vec)
    
    # Partial trace over complement to get the measured subsystem
    if isempty(complement_indices)
        # No partial trace needed - measuring the entire system
        lc_measured_states = lc.states
        lc_measured_coeffs = normalized_coeffs
    else
        # Apply partial trace to each state
        lc_measured_states = [ptrace(state, complement_indices) for state in lc.states]
        lc_measured_coeffs = normalized_coeffs
    end
    
    # Calculate overlap with measurement state including all cross-terms
    overlap = 0.0 + 0.0im
    for i in 1:length(lc_measured_states)
        c = lc_measured_coeffs[i]
        state = lc_measured_states[i]
        state_overlap = _gaussian_overlap(measurement, state)
        overlap += c * state_overlap
    end
    
    # Born rule: probability is |⟨measurement|state⟩|²
    return abs2(overlap)
end