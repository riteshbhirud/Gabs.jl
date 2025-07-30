# GPU support for GaussianLinearCombination operations

# GPU tensor products for linear combinations
function Gabs.tensor(::Type{Tm}, ::Type{Tc}, lc1::GaussianLinearCombination, lc2::GaussianLinearCombination) where {Tm<:CuVector,Tc<:CuMatrix}
    typeof(lc1.basis) == typeof(lc2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    lc1.ħ == lc2.ħ || throw(ArgumentError(HBAR_ERROR))
    
    new_basis = lc1.basis ⊕ lc2.basis
    n1, n2 = length(lc1), length(lc2)
    CoeffType = promote_type(eltype(lc1.coeffs), eltype(lc2.coeffs))
    
    # Pre-allocate arrays on GPU
    new_coeffs = Vector{CoeffType}(undef, n1 * n2)
    new_states = Vector{GaussianState}(undef, n1 * n2)
    
    # Compute tensor products
    idx = 1
    @inbounds for i in 1:n1
        @inbounds for j in 1:n2
            new_coeffs[idx] = lc1.coeffs[i] * lc2.coeffs[j]
            new_states[idx] = Gabs.tensor(Tm, Tc, lc1.states[i], lc2.states[j])
            idx += 1
        end
    end
    
    return GaussianLinearCombination(new_basis, new_coeffs, new_states)
end

function Gabs.tensor(::Type{T}, lc1::GaussianLinearCombination, lc2::GaussianLinearCombination) where {T<:CuArray}
    if T <: CuMatrix
        return Gabs.tensor(CuVector{eltype(T)}, T, lc1, lc2)
    else
        return Gabs.tensor(T, CuMatrix{eltype(T)}, lc1, lc2)
    end
end

# GPU partial trace for linear combinations
function Gabs.ptrace(::Type{Tm}, ::Type{Tc}, lc::GaussianLinearCombination, indices::Union{Int, AbstractVector{<:Int}}) where {Tm<:CuVector,Tc<:CuMatrix}
    indices_vec = indices isa Int ? [indices] : indices
    length(indices_vec) < lc.basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    
    # Apply ptrace to each state on GPU
    traced_states = [Gabs.ptrace(Tm, Tc, state, indices_vec) for state in lc.states]
    result = GaussianLinearCombination(traced_states[1].basis, copy(lc.coeffs), traced_states)
    Gabs.simplify!(result)
    return result
end

function Gabs.ptrace(::Type{T}, lc::GaussianLinearCombination, indices::Union{Int, AbstractVector{<:Int}}) where {T<:CuArray}
    if T <: CuMatrix
        return Gabs.ptrace(CuVector{eltype(T)}, T, lc, indices)
    else
        return Gabs.ptrace(T, CuMatrix{eltype(T)}, lc, indices)
    end
end

# GPU operations on linear combinations
function Base.:(*)(op::GaussianUnitary{B,D,S}, lc::GaussianLinearCombination) where {B,D<:CuArray,S<:CuArray}
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    
    # Apply operator to each state in parallel
    new_states = [op * state for state in lc.states]
    return GaussianLinearCombination(lc.basis, copy(lc.coeffs), new_states)
end

function Base.:(*)(op::GaussianChannel{B,D,T}, lc::GaussianLinearCombination) where {B,D<:CuArray,T<:CuArray}
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    
    # Apply channel to each state in parallel
    new_states = [op * state for state in lc.states]
    return GaussianLinearCombination(lc.basis, copy(lc.coeffs), new_states)
end

# Automatic promotion for mixed CPU/GPU operations
function Base.:(*)(op::GaussianUnitary{B,D,S}, lc::GaussianLinearCombination) where {B,D<:CuArray,S<:CuArray}
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    
    # Check if any states need GPU promotion
    needs_promotion = any(state -> !is_gpu_array(state.mean) || !is_gpu_array(state.covar), lc.states)
    
    if needs_promotion
        # Promote states to GPU
        gpu_states = [GaussianState(state.basis, ensure_gpu(state.mean), ensure_gpu(state.covar); ħ = state.ħ) 
                      for state in lc.states]
        gpu_lc = GaussianLinearCombination(lc.basis, lc.coeffs, gpu_states)
        return op * gpu_lc
    else
        # All states already on GPU
        new_states = [op * state for state in lc.states]
        return GaussianLinearCombination(lc.basis, copy(lc.coeffs), new_states)
    end
end

function Base.:(*)(op::GaussianChannel{B,D,T}, lc::GaussianLinearCombination) where {B,D<:CuArray,T<:CuArray}
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    
    # Check if any states need GPU promotion
    needs_promotion = any(state -> !is_gpu_array(state.mean) || !is_gpu_array(state.covar), lc.states)
    
    if needs_promotion
        # Promote states to GPU
        gpu_states = [GaussianState(state.basis, ensure_gpu(state.mean), ensure_gpu(state.covar); ħ = state.ħ) 
                      for state in lc.states]
        gpu_lc = GaussianLinearCombination(lc.basis, lc.coeffs, gpu_states)
        return op * gpu_lc
    else
        # All states already on GPU
        new_states = [op * state for state in lc.states]
        return GaussianLinearCombination(lc.basis, copy(lc.coeffs), new_states)
    end
end

# Utility functions for GPU linear combinations
function to_gpu(lc::GaussianLinearCombination)
    """Move a GaussianLinearCombination to GPU"""
    gpu_states = [GaussianState(state.basis, CuArray(state.mean), CuArray(state.covar); ħ = state.ħ) 
                  for state in lc.states]
    gpu_coeffs = is_gpu_array(lc.coeffs) ? lc.coeffs : CuArray(lc.coeffs)
    return GaussianLinearCombination(lc.basis, gpu_coeffs, gpu_states)
end

function to_cpu(lc::GaussianLinearCombination)
    """Move a GaussianLinearCombination to CPU"""
    cpu_states = [GaussianState(state.basis, Array(state.mean), Array(state.covar); ħ = state.ħ) 
                  for state in lc.states]
    cpu_coeffs = is_gpu_array(lc.coeffs) ? Array(lc.coeffs) : lc.coeffs
    return GaussianLinearCombination(lc.basis, cpu_coeffs, cpu_states)
end

function is_on_gpu(lc::GaussianLinearCombination)
    """Check if GaussianLinearCombination is on GPU"""
    return any(state -> is_gpu_array(state.mean) || is_gpu_array(state.covar), lc.states)
end

# Batched operations for efficiency
function batched_apply_unitary!(lc::GaussianLinearCombination, ops::Vector{<:GaussianUnitary})
    """Apply multiple unitaries to linear combination efficiently"""
    @assert length(ops) == length(lc.states) "Number of operators must match number of states"
    
    # Apply operations in parallel
    @inbounds for i in 1:length(lc.states)
        lc.states[i] = ops[i] * lc.states[i]
    end
    
    return lc
end

function batched_apply_channel!(lc::GaussianLinearCombination, channels::Vector{<:GaussianChannel})
    """Apply multiple channels to linear combination efficiently"""
    @assert length(channels) == length(lc.states) "Number of channels must match number of states"
    
    # Apply operations in parallel
    @inbounds for i in 1:length(lc.states)
        lc.states[i] = channels[i] * lc.states[i]
    end
    
    return lc
end