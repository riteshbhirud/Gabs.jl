# ext/CUDAExt/linear_combinations.jl - FIXED VERSION

using Gabs: GaussianLinearCombination, SYMPLECTIC_ERROR, HBAR_ERROR, ACTION_ERROR
import Gabs: purity, entropy_vn, vacuumstate, _is_gpu_array,_detect_device

# =============================================================================
# Device Detection and Transfer Functions
# =============================================================================
# Extend device detection for CuArrays
Gabs._is_gpu_array(x::CuArray) = true

function Gabs._detect_device(lc::GaussianLinearCombination)
    if isempty(lc.states)
        return :cpu
    end
    return Gabs._detect_device(lc.states[1].mean)
end

function Gabs._gpu_impl(lc::GaussianLinearCombination, precision::Type{T}) where T
    if !CUDA_AVAILABLE
        @warn "CUDA not available. Cannot transfer to GPU."
        return lc
    end
    
    gpu_states = [Gabs._gpu_impl(state, precision) for state in lc.states]
    coeffs = T.(lc.coeffs)
    return GaussianLinearCombination(lc.basis, coeffs, gpu_states)
end

# =============================================================================
# GPU-Specific Operator Applications (Specific Type Constraints)
# =============================================================================

# GPU operator * GPU linear combination
function Base.:(*)(op::GaussianUnitary{B,<:CuArray,<:CuArray}, lc::GaussianLinearCombination{B,C,S}) where {B<:SymplecticBasis,C,S<:GaussianState{B,<:CuArray,<:CuArray}}
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    
    new_states = [op * state for state in lc.states]
    return GaussianLinearCombination(lc.basis, copy(lc.coeffs), new_states)
end

function Base.:(*)(op::GaussianChannel{B,<:CuArray,<:CuArray}, lc::GaussianLinearCombination{B,C,S}) where {B<:SymplecticBasis,C,S<:GaussianState{B,<:CuArray,<:CuArray}}
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    
    new_states = [op * state for state in lc.states]
    return GaussianLinearCombination(lc.basis, copy(lc.coeffs), new_states)
end

# CPU operator * GPU linear combination (auto-promote)
function Base.:(*)(op::GaussianUnitary{B,<:Array,<:Array}, lc::GaussianLinearCombination{B,C,S}) where {B<:SymplecticBasis,C,S<:GaussianState{B,<:CuArray,<:CuArray}}
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    
    op_gpu = op |> gpu
    new_states = [op_gpu * state for state in lc.states]
    return GaussianLinearCombination(lc.basis, copy(lc.coeffs), new_states)
end

function Base.:(*)(op::GaussianChannel{B,<:Array,<:Array}, lc::GaussianLinearCombination{B,C,S}) where {B<:SymplecticBasis,C,S<:GaussianState{B,<:CuArray,<:CuArray}}
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    
    op_gpu = op |> gpu
    new_states = [op_gpu * state for state in lc.states]
    return GaussianLinearCombination(lc.basis, copy(lc.coeffs), new_states)
end

# =============================================================================
# Utility Functions
# =============================================================================
#=
function normalize!(lc::GaussianLinearCombination{B,C,S}) where {B<:SymplecticBasis,C,S}
    # Check if states are on GPU at runtime instead of compile time
    if !isempty(lc.states) && lc.states[1].mean isa CuArray
        norm_val = sqrt(sum(abs2, lc.coeffs))
        if norm_val > 0
            lc.coeffs ./= norm_val
        end
        return lc
    else
        # Fall back to CPU method if it exists
        return invoke(normalize!, Tuple{GaussianLinearCombination}, lc)
    end
end

function simplify!(lc::GaussianLinearCombination{B,C,S}; atol::Real=1e-14) where {B<:SymplecticBasis,C,S<:GaussianState{B,<:CuArray,<:CuArray}}
    if isempty(lc.coeffs)
        return lc
    end
    
    keep_mask = abs.(lc.coeffs) .> atol
    if !any(keep_mask)
        vac = vacuumstate(lc.basis; ħ = lc.ħ) |> gpu
        lc.coeffs = [atol]
        lc.states = [vac]
        return lc
    end
    
    coeffs = lc.coeffs[keep_mask]
    states = lc.states[keep_mask]
    
    if isempty(states)
        return lc
    end
    
    unique_states = typeof(states[1])[]
    combined_coeffs = eltype(coeffs)[]
    
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
        vac = vacuumstate(lc.basis; ħ = lc.ħ) |> gpu
        lc.coeffs = [atol]
        lc.states = [vac]
    else
        lc.coeffs = combined_coeffs[final_mask]
        lc.states = unique_states[final_mask]
    end
    
    return lc
end
=#
# =============================================================================
# State Metrics (More Specific Type Constraints)
# =============================================================================

function purity(lc::GaussianLinearCombination{B,C,S}) where {B<:SymplecticBasis,C,S<:GaussianState{B,<:CuArray,<:CuArray}}
    return 1.0
end

function entropy_vn(lc::GaussianLinearCombination{B,C,S}) where {B<:SymplecticBasis,C,S<:GaussianState{B,<:CuArray,<:CuArray}}
    return 0.0
end