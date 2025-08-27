# GPU State Creation Operations

# The base state creation functions already work with CuArrays through dispatch
# We just need to ensure proper GPU array handling and add optimized GPU-specific versions where beneficial

# Vacuum state - optimized GPU version
function Gabs.vacuumstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}; ħ = 2) where {Tm<:CuVector, Tc<:CuMatrix, N<:Int}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.vacuumstate(Vector{eltype(Tm)}, Matrix{eltype(Tc)}, basis; ħ = ħ)
    end
    
    nmodes = basis.nmodes
    T = eltype(Tm)
    
    # Create directly on GPU for efficiency
    mean = CUDA.zeros(T, 2*nmodes)
    covar = CuMatrix{T}((ħ/2) * I(2*nmodes))
    
    return GaussianState(basis, mean, covar; ħ = ħ)
end

# Thermal state - optimized GPU version  
function Gabs.thermalstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, photons::P; ħ = 2) where {Tm<:CuVector, Tc<:CuMatrix, N<:Int, P}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.thermalstate(Vector{eltype(Tm)}, Matrix{eltype(Tc)}, basis, photons; ħ = ħ)
    end
    
    # Use the existing internal function but ensure GPU arrays
    mean, covar = _thermalstate(basis, photons; ħ = ħ)
    
    # Convert to GPU arrays
    gpu_mean = CuArray(mean)
    gpu_covar = CuArray(covar)
    
    return GaussianState(basis, gpu_mean, gpu_covar; ħ = ħ)
end

# Coherent state - optimized GPU version
function Gabs.coherentstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {Tm<:CuVector, Tc<:CuMatrix, N<:Int, A}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.coherentstate(Vector{eltype(Tm)}, Matrix{eltype(Tc)}, basis, alpha; ħ = ħ)
    end
    
    # Use the existing internal function but ensure GPU arrays
    mean, covar = _coherentstate(basis, alpha; ħ = ħ)
    
    # Convert to GPU arrays
    gpu_mean = CuArray(mean)
    gpu_covar = CuArray(covar)
    
    return GaussianState(basis, gpu_mean, gpu_covar; ħ = ħ)
end

# Squeezed state - optimized GPU version
function Gabs.squeezedstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {Tm<:CuVector, Tc<:CuMatrix, N<:Int, R}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.squeezedstate(Vector{eltype(Tm)}, Matrix{eltype(Tc)}, basis, r, theta; ħ = ħ)
    end
    
    # Use the existing internal function but ensure GPU arrays
    mean, covar = _squeezedstate(basis, r, theta; ħ = ħ)
    
    # Convert to GPU arrays
    gpu_mean = CuArray(mean)
    gpu_covar = CuArray(covar)
    
    return GaussianState(basis, gpu_mean, gpu_covar; ħ = ħ)
end

# EPR state - optimized GPU version
function Gabs.eprstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {Tm<:CuVector, Tc<:CuMatrix, N<:Int, R}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.eprstate(Vector{eltype(Tm)}, Matrix{eltype(Tc)}, basis, r, theta; ħ = ħ)
    end
    
    # Use the existing internal function but ensure GPU arrays
    mean, covar = _eprstate(basis, r, theta; ħ = ħ)
    
    # Convert to GPU arrays  
    gpu_mean = CuArray(mean)
    gpu_covar = CuArray(covar)
    
    return GaussianState(basis, gpu_mean, gpu_covar; ħ = ħ)
end

# Single-argument versions for convenience
Gabs.vacuumstate(::Type{T}, basis::SymplecticBasis{N}; ħ = 2) where {T<:CuVector, N<:Int} = 
    Gabs.vacuumstate(T, CuMatrix{eltype(T)}, basis; ħ = ħ)

Gabs.thermalstate(::Type{T}, basis::SymplecticBasis{N}, photons::P; ħ = 2) where {T<:CuVector, N<:Int, P} = 
    Gabs.thermalstate(T, CuMatrix{eltype(T)}, basis, photons; ħ = ħ)

Gabs.coherentstate(::Type{T}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {T<:CuVector, N<:Int, A} = 
    Gabs.coherentstate(T, CuMatrix{eltype(T)}, basis, alpha; ħ = ħ)

Gabs.squeezedstate(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {T<:CuVector, N<:Int, R} = 
    Gabs.squeezedstate(T, CuMatrix{eltype(T)}, basis, r, theta; ħ = ħ)

Gabs.eprstate(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {T<:CuVector, N<:Int, R} = 
    Gabs.eprstate(T, CuMatrix{eltype(T)}, basis, r, theta; ħ = ħ)

# Ensure tensor products work correctly with GPU arrays
function Gabs.tensor(::Type{Tm}, ::Type{Tc}, state1::GaussianState{B,M1,V1}, state2::GaussianState{B,M2,V2}) where {
    Tm<:CuVector, Tc<:CuMatrix, B<:SymplecticBasis, M1<:CuArray, V1<:CuArray, M2<:CuArray, V2<:CuArray}
    
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.tensor(Vector{eltype(Tm)}, Matrix{eltype(Tc)}, state1, state2)
    end
    
    typeof(state1.basis) == typeof(state2.basis) || throw(ArgumentError(Gabs.SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(Gabs.HBAR_ERROR))
    
    # Use existing tensor implementation which should work with CuArrays
    mean, covar = Gabs._tensor(state1, state2)
    
    # Ensure result is on GPU
    gpu_mean = ensure_gpu_array(mean)
    gpu_covar = ensure_gpu_array(covar)
    
    return GaussianState(state1.basis ⊕ state2.basis, gpu_mean, gpu_covar; ħ = state1.ħ)
end

# Partial trace for GPU arrays
function Gabs.ptrace(::Type{Tm}, ::Type{Tc}, state::GaussianState{B,M,V}, indices::T) where {
    Tm<:CuVector, Tc<:CuMatrix, B<:SymplecticBasis, M<:CuArray, V<:CuArray, T}
    
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.ptrace(Vector{eltype(Tm)}, Matrix{eltype(Tc)}, state, indices)
    end
    
    basis = state.basis
    
    # Use existing _ptrace implementation which should work with CuArrays
    mean, covar = Gabs._ptrace(state, indices)
    
    # Ensure result is on GPU
    gpu_mean = ensure_gpu_array(mean)
    gpu_covar = ensure_gpu_array(covar)
    
    return GaussianState(typeof(basis)(basis.nmodes - length(indices)), gpu_mean, gpu_covar; ħ = state.ħ)
end