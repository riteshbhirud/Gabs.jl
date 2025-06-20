"""
    wigner(state::GaussianState, x)

Compute the Wigner function of an N-mode Gaussian state at `x`, a vector of size 2N.
"""
function wigner(state::GaussianState, x::T) where {T}
    basis = state.basis
    nmodes = basis.nmodes
    mean = state.mean
    isequal(length(mean), length(x)) || throw(ArgumentError(WIGNER_ERROR))

    V = state.covar
    diff = x .- mean
    arg = -(1/2) * transpose(diff) * inv(V) * diff

    return exp(arg)/((2pi)^nmodes * sqrt(det(V)))
end

"""
    wignerchar(state::GaussianState, xi)

Compute the Wigner characteristic function of an N-mode Gaussian state at `xi`, a vector of size 2N.
"""
function wignerchar(state::GaussianState, xi::T) where {T}
    basis = state.basis
    nmodes = basis.nmodes
    mean = state.mean
    isequal(length(mean), length(xi)) || throw(ArgumentError(WIGNER_ERROR))

    V = state.covar
    Omega = symplecticform(basis)

    arg1 = -(1/2) * transpose(xi) * (Omega*V*transpose(Omega))*xi
    arg2 = im * transpose(Omega*mean) * xi

    return exp(arg1 .- arg2)
end

#phase 3 code from here:

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

