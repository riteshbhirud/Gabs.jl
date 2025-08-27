# GPU Unitary and Channel Operations

# Displacement operator - GPU version
function Gabs.displace(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {Td<:CuVector, Ts<:CuMatrix, N<:Int, A}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.displace(Vector{eltype(Td)}, Matrix{eltype(Ts)}, basis, alpha; ħ = ħ)
    end
    
    # Use existing internal function
    disp, symplectic = _displace(basis, alpha; ħ = ħ)
    
    # Convert to GPU arrays
    gpu_disp = CuArray(disp)
    gpu_symplectic = CuArray(symplectic)
    
    return GaussianUnitary(basis, gpu_disp, gpu_symplectic; ħ = ħ)
end

# Squeeze operator - GPU version
function Gabs.squeeze(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {Td<:CuVector, Ts<:CuMatrix, N<:Int, R}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.squeeze(Vector{eltype(Td)}, Matrix{eltype(Ts)}, basis, r, theta; ħ = ħ)
    end
    
    # Use existing internal function
    disp, symplectic = _squeeze(basis, r, theta)
    
    # Convert to GPU arrays
    gpu_disp = CuArray(disp)
    gpu_symplectic = CuArray(symplectic)
    
    return GaussianUnitary(basis, gpu_disp, gpu_symplectic; ħ = ħ)
end

# Two-mode squeeze operator - GPU version
function Gabs.twosqueeze(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {Td<:CuVector, Ts<:CuMatrix, N<:Int, R}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.twosqueeze(Vector{eltype(Td)}, Matrix{eltype(Ts)}, basis, r, theta; ħ = ħ)
    end
    
    # Use existing internal function
    disp, symplectic = _twosqueeze(basis, r, theta)
    
    # Convert to GPU arrays
    gpu_disp = CuArray(disp)
    gpu_symplectic = CuArray(symplectic)
    
    return GaussianUnitary(basis, gpu_disp, gpu_symplectic; ħ = ħ)
end

# Phase shift operator - GPU version
function Gabs.phaseshift(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, theta::R; ħ = 2) where {Td<:CuVector, Ts<:CuMatrix, N<:Int, R}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.phaseshift(Vector{eltype(Td)}, Matrix{eltype(Ts)}, basis, theta; ħ = ħ)
    end
    
    # Use existing internal function
    disp, symplectic = _phaseshift(basis, theta)
    
    # Convert to GPU arrays
    gpu_disp = CuArray(disp)
    gpu_symplectic = CuArray(symplectic)
    
    return GaussianUnitary(basis, gpu_disp, gpu_symplectic; ħ = ħ)
end

# Beam splitter operator - GPU version
function Gabs.beamsplitter(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, transmit::R; ħ = 2) where {Td<:CuVector, Ts<:CuMatrix, N<:Int, R}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.beamsplitter(Vector{eltype(Td)}, Matrix{eltype(Ts)}, basis, transmit; ħ = ħ)
    end
    
    # Use existing internal function
    disp, symplectic = _beamsplitter(basis, transmit)
    
    # Convert to GPU arrays
    gpu_disp = CuArray(disp)
    gpu_symplectic = CuArray(symplectic)
    
    return GaussianUnitary(basis, gpu_disp, gpu_symplectic; ħ = ħ)
end

# Attenuator channel - GPU version
function Gabs.attenuator(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}, theta::R, n::M; ħ = 2) where {Td<:CuVector, Tt<:CuMatrix, N<:Int, R, M}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.attenuator(Vector{eltype(Td)}, Matrix{eltype(Tt)}, basis, theta, n; ħ = ħ)
    end
    
    # Use existing internal function
    disp, transform, noise = _attenuator(basis, theta, n)
    
    # Convert to GPU arrays
    gpu_disp = CuArray(disp)
    gpu_transform = CuArray(transform)
    gpu_noise = CuArray(noise)
    
    return GaussianChannel(basis, gpu_disp, gpu_transform, gpu_noise; ħ = ħ)
end

# Amplifier channel - GPU version
function Gabs.amplifier(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}, r::R, n::M; ħ = 2) where {Td<:CuVector, Tt<:CuMatrix, N<:Int, R, M}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.amplifier(Vector{eltype(Td)}, Matrix{eltype(Tt)}, basis, r, n; ħ = ħ)
    end
    
    # Use existing internal function
    disp, transform, noise = _amplifier(basis, r, n)
    
    # Convert to GPU arrays
    gpu_disp = CuArray(disp)
    gpu_transform = CuArray(transform)
    gpu_noise = CuArray(noise)
    
    return GaussianChannel(basis, gpu_disp, gpu_transform, gpu_noise; ħ = ħ)
end

# Single-argument versions for convenience
Gabs.displace(::Type{T}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {T<:CuVector, N<:Int, A} = 
    Gabs.displace(T, CuMatrix{eltype(T)}, basis, alpha; ħ = ħ)

Gabs.squeeze(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {T<:CuVector, N<:Int, R} = 
    Gabs.squeeze(T, CuMatrix{eltype(T)}, basis, r, theta; ħ = ħ)

Gabs.twosqueeze(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {T<:CuVector, N<:Int, R} = 
    Gabs.twosqueeze(T, CuMatrix{eltype(T)}, basis, r, theta; ħ = ħ)

Gabs.phaseshift(::Type{T}, basis::SymplecticBasis{N}, theta::R; ħ = 2) where {T<:CuVector, N<:Int, R} = 
    Gabs.phaseshift(T, CuMatrix{eltype(T)}, basis, theta; ħ = ħ)

Gabs.beamsplitter(::Type{T}, basis::SymplecticBasis{N}, transmit::R; ħ = 2) where {T<:CuVector, N<:Int, R} = 
    Gabs.beamsplitter(T, CuMatrix{eltype(T)}, basis, transmit; ħ = ħ)

Gabs.attenuator(::Type{T}, basis::SymplecticBasis{N}, theta::R, n::M; ħ = 2) where {T<:CuVector, N<:Int, R, M} = 
    Gabs.attenuator(T, CuMatrix{eltype(T)}, basis, theta, n; ħ = ħ)

Gabs.amplifier(::Type{T}, basis::SymplecticBasis{N}, r::R, n::M; ħ = 2) where {T<:CuVector, N<:Int, R, M} = 
    Gabs.amplifier(T, CuMatrix{eltype(T)}, basis, r, n; ħ = ħ)

# Tensor products for GPU unitaries and channels
function Gabs.tensor(::Type{Td}, ::Type{Ts}, op1::GaussianUnitary{B,D1,S1}, op2::GaussianUnitary{B,D2,S2}) where {
    Td<:CuVector, Ts<:CuMatrix, B<:SymplecticBasis, D1<:CuArray, S1<:CuArray, D2<:CuArray, S2<:CuArray}
    
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.tensor(Vector{eltype(Td)}, Matrix{eltype(Ts)}, op1, op2)
    end
    
    typeof(op1.basis) == typeof(op2.basis) || throw(ArgumentError(Gabs.SYMPLECTIC_ERROR))
    op1.ħ == op2.ħ || throw(ArgumentError(Gabs.HBAR_ERROR))
    
    # Use existing tensor implementation
    disp, symplectic = Gabs._tensor(op1, op2)
    
    # Ensure result is on GPU
    gpu_disp = ensure_gpu_array(disp)
    gpu_symplectic = ensure_gpu_array(symplectic)
    
    return GaussianUnitary(op1.basis ⊕ op2.basis, gpu_disp, gpu_symplectic; ħ = op1.ħ)
end

function Gabs.tensor(::Type{Td}, ::Type{Tt}, op1::GaussianChannel{B,D1,T1}, op2::GaussianChannel{B,D2,T2}) where {
    Td<:CuVector, Tt<:CuMatrix, B<:SymplecticBasis, D1<:CuArray, T1<:CuArray, D2<:CuArray, T2<:CuArray}
    
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return Gabs.tensor(Vector{eltype(Td)}, Matrix{eltype(Tt)}, op1, op2)
    end
    
    typeof(op1.basis) == typeof(op2.basis) || throw(ArgumentError(Gabs.SYMPLECTIC_ERROR))
    op1.ħ == op2.ħ || throw(ArgumentError(Gabs.HBAR_ERROR))
    
    # Use existing tensor implementation
    disp, transform, noise = Gabs._tensor(op1, op2)
    
    # Ensure result is on GPU
    gpu_disp = ensure_gpu_array(disp)
    gpu_transform = ensure_gpu_array(transform)
    gpu_noise = ensure_gpu_array(noise)
    
    return GaussianChannel(op1.basis ⊕ op2.basis, gpu_disp, gpu_transform, gpu_noise; ħ = op1.ħ)
end

# GPU-optimized operation application - the existing * operators should work
# but we can add specialized versions if needed for performance

# For very large operations, we could add specialized GPU kernels here
# but for now, the existing linear algebra operations should be efficient on GPU