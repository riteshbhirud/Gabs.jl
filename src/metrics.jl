"""
    purity(state::GaussianState)

Calculate the purity of a Gaussian state, defined by `1/sqrt((2/ħ) det(V))`.
"""
purity(x::GaussianState) = (b = x.basis; (x.ħ/2)^(b.nmodes)/sqrt(det(x.covar)))

"""
    entropy_vn(state::GaussianState; tol::Real = 128 * eps(1/2))

Calculate the Von Neumann entropy of a Gaussian state, defined as

```math
S(\\rho) = -Tr(\\rho \\log(\\rho)) = \\sum_i f(v_i)
```

such that ``\\log`` denotes the natural logarithm, ``v_i`` is the symplectic
spectrum of ``\\mathbf{V}/\\hbar``, and the ``f`` is taken to be

```math
f(x) = (x + 1/2) \\log(x + 1/2) - (x - 1/2) \\log(x - 1/2)
```
wherein it is understood that ``0 \\log(0) \\equiv 0``.

# Arguments
* `state`: Gaussian state whose Von Neumann entropy is to be calculated.
* `tol`: Tolerance (exclusive) above the cut-off at ``1/2`` for computing ``f(x)``.
"""
function entropy_vn(state::GaussianState{B, M, V}; tol::Real = real(eltype(V)) <: AbstractFloat ? 128 * eps(real(eltype(V))(1) / real(eltype(V))(2)) : 128 * eps(1/2)) where {B, M, V}
    T = real(eltype(V))
    T = T <: AbstractFloat ? T : Float64
    S = _sympspectrum(state.covar, x -> (x - (T(1) / T(2))) > tol; pre = symplecticform(state.basis), invscale = state.ħ)
    return reduce(+, _entropy_vn.(S))
end

# this is the same as f(x)
_entropy_vn(x) = x < 19 ?
    (x + (1/2)) * log(x + (1/2)) - (x - (1/2)) * log(x - (1/2)) :
    log(x) + 1 - (1/(24 * x^2)) - (1/(320 * x^4)) - (1/(2688 * x^6))

"""
    fidelity(state1::GaussianState, state2::GaussianState; tol::Real = 128 * eps(1))

Calculate the joint fidelity of two Gaussian states, defined as

```math
F(\\rho, \\sigma) = Tr(\\sqrt{\\sqrt{\\rho} \\sigma \\sqrt{\\rho}}).
```

See: Banchi, Braunstein, and Pirandola, Phys. Rev. Lett. 115, 260501 (2015)

# Arguments
* `state1`, `state2`: Gaussian states whose joint fidelity is to be calculated.
* `tol`: Tolerance (inclusive) above the cut-off at ``1`` for computing ``x + \\sqrt{x^2 - 1}``.
"""
function fidelity(state1::GaussianState{B1, M1, V1}, state2::GaussianState{B2, M2, V2}; tol::Real = real(promote_type(eltype(V1), eltype(V2))) <: AbstractFloat ? 128 * eps(real(promote_type(eltype(V1), eltype(V2)))) : 128 * eps(1/1)) where {B1, M1, V1, B2, M2, V2}
    state1.basis == state2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    A = state2.mean - state1.mean
    B = state1.covar + state2.covar
    # many nasty factors of ħ ahead, tread carefully
    output = state1.ħ^(state1.basis.nmodes/2) * exp(- (transpose(A) * (B \ A)) / 4) / (det(B))^(1/4)
    A = symplecticform(state1.basis)
    # slightly different from Banachi, Braunstein, and Pirandola
    B = (B \ ((A .* ((state1.ħ^2)/4)) + (state2.covar * A * state1.covar)))
    T = real(promote_type(eltype(V1), eltype(V2)))
    T = T <: AbstractFloat ? T : Float64
    B = _sympspectrum(B, x -> (x - T(1)) >= tol; invscale = (state1.ħ / T(2)))
    return output * sqrt(reduce(*, _fidelity.(B)))
end

# this is the same as x + sqrt(x^2 - 1) when x > 0, but overflows gradually
_fidelity(x) = x^2 < floatmax(typeof(x)) ? x + sqrt(x^2 - 1) : 2 * x

"""
    logarithmic_negativity(state::GaussianState, indices::Union{Integer, AbstractVector{<:Integer}}; tola::Real = 0, tolb::Real = 128 * eps(1))

Calculate the logarithmic negativity of a Gaussian state partition, defined as

```math
N(\\rho) = \\log\\|\\rho^{T_B}\\|_1 = - \\sum_i \\log(2 \\tilde{v}_i^<)
```

such that ``\\log`` denotes the natural logarithm, ``\\tilde{v}_i^<`` is the
symplectic spectrum of ``\\mathbf{\\tilde{V}}/\\hbar`` which is ``< 1/2``.

Therein, ``\\mathbf{\\tilde{V}} = \\mathbf{T} \\mathbf{V} \\mathbf{T}`` where
```math
\\forall k : \\mathbf{T} q_k = q_k
\\forall k \\in \\mathrm{B} : \\mathbf{T} p_k = -p_k
\\forall k \\notin \\mathrm{B} : \\mathbf{T} p_k = p_k
```

# Arguments
* `state`: Gaussian state whose logarithmic negativity is to be calculated.
* `indices`: Integer or collection thereof, specifying the binary partition.
* `tola`: Tolerance (inclusive) above the cut-off at ``0`` for computing ``\\log(x)``.
* `tolb`: Tolerance (inclusive) below the cut-off at ``1`` for computing ``\\log(x)``.
"""
function logarithmic_negativity(state::GaussianState{B, M, V}, indices::Union{Integer, AbstractVector{<:Integer}}; tola::Real = 0, tolb::Real = real(eltype(V)) <: AbstractFloat ? 128 * eps(real(eltype(V))) : 128 * eps(1/1)) where {B, M, V}
    S = _tilde(state, indices)
    T = real(eltype(V))
    T = T <: AbstractFloat ? T : Float64
    S = _sympspectrum(S, x -> x >= tola && (T(1) - x) >= tolb; pre = symplecticform(state.basis), invscale = (state.ħ / T(2)))
    S = reduce(+, log.(S))
    # in case the reduction happened over an empty set
    return S < 0 ? -S : S
end

function _tilde(state::GaussianState{B,M,V}, indices::Union{Integer, AbstractVector{<:Integer}}) where {B<:QuadPairBasis,M,V}
    nmodes = state.basis.nmodes
    indices = collect(indices)
    all(x -> x >= 1 && x <= nmodes, indices) || throw(ArgumentError(INDEX_ERROR))
    T = copy(state.covar)
    @inbounds for i in indices
        # first loop is cache friendly, second one thrashes
        @inbounds for j in Base.OneTo(2*nmodes)
            T[j, 2*i] *= -1
        end
        @inbounds for j in Base.OneTo(2*nmodes)
            T[2*i, j] *= -1
        end
    end
    return T
end
function _tilde(state::GaussianState{B,M,V}, indices::Union{Integer, AbstractVector{<:Integer}}) where {B<:QuadBlockBasis,M,V}
    nmodes = state.basis.nmodes
    indices = collect(indices)
    all(x -> x >= 1 && x <= nmodes, indices) || throw(ArgumentError(INDEX_ERROR))
    T = copy(state.covar)
    @inbounds for i in indices
        # first loop is cache friendly, second one thrashes
        @inbounds for j in Base.OneTo(2*nmodes)
            T[j, nmodes + i] *= -1
        end
        @inbounds for j in Base.OneTo(2*nmodes)
            T[nmodes + i, j] *= -1
        end
    end
    return T
end

#phase 3 code from here:

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

