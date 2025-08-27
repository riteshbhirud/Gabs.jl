# GPU Unitary and Channel Operations

# Import the functions we want to extend
import Gabs: displace, squeeze, twosqueeze, phaseshift, beamsplitter, attenuator, amplifier

# Displacement operator - GPU version
function displace(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {Td<:CuVector, Ts<:CuMatrix, N<:Int, A}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return displace(Vector{eltype(Td)}, Matrix{eltype(Ts)}, basis, alpha; ħ = ħ)
    end
    
    T = eltype(Td)
    
    # Use existing internal function
    disp, symplectic = _displace(basis, alpha; ħ = ħ)
    
    # Convert to GPU arrays with correct element type
    gpu_disp = CuArray{T}(disp)
    gpu_symplectic = CuArray{T}(symplectic)
    
    return GaussianUnitary(basis, gpu_disp, gpu_symplectic; ħ = ħ)
end

# Squeeze operator - GPU version
function squeeze(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {Td<:CuVector, Ts<:CuMatrix, N<:Int, R}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return squeeze(Vector{eltype(Td)}, Matrix{eltype(Ts)}, basis, r, theta; ħ = ħ)
    end
    
    T = eltype(Td)
    
    # Use existing internal function
    disp, symplectic = _squeeze(basis, r, theta)
    
    # Convert to GPU arrays with correct element type
    gpu_disp = CuArray{T}(disp)
    gpu_symplectic = CuArray{T}(symplectic)
    
    return GaussianUnitary(basis, gpu_disp, gpu_symplectic; ħ = ħ)
end

# Two-mode squeeze operator - GPU version
function twosqueeze(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {Td<:CuVector, Ts<:CuMatrix, N<:Int, R}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return twosqueeze(Vector{eltype(Td)}, Matrix{eltype(Ts)}, basis, r, theta; ħ = ħ)
    end
    
    T = eltype(Td)
    
    # Use existing internal function
    disp, symplectic = _twosqueeze(basis, r, theta)
    
    # Convert to GPU arrays with correct element type
    gpu_disp = CuArray{T}(disp)
    gpu_symplectic = CuArray{T}(symplectic)
    
    return GaussianUnitary(basis, gpu_disp, gpu_symplectic; ħ = ħ)
end

# Phase shift operator - GPU version
function phaseshift(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, theta::R; ħ = 2) where {Td<:CuVector, Ts<:CuMatrix, N<:Int, R}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return phaseshift(Vector{eltype(Td)}, Matrix{eltype(Ts)}, basis, theta; ħ = ħ)
    end
    
    T = eltype(Td)
    
    # Use existing internal function
    disp, symplectic = _phaseshift(basis, theta)
    
    # Convert to GPU arrays with correct element type
    gpu_disp = CuArray{T}(disp)
    gpu_symplectic = CuArray{T}(symplectic)
    
    return GaussianUnitary(basis, gpu_disp, gpu_symplectic; ħ = ħ)
end

# Beam splitter operator - GPU version
function beamsplitter(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, transmit::R; ħ = 2) where {Td<:CuVector, Ts<:CuMatrix, N<:Int, R}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return beamsplitter(Vector{eltype(Td)}, Matrix{eltype(Ts)}, basis, transmit; ħ = ħ)
    end
    
    T = eltype(Td)
    
    # Use existing internal function
    disp, symplectic = _beamsplitter(basis, transmit)
    
    # Convert to GPU arrays with correct element type
    gpu_disp = CuArray{T}(disp)
    gpu_symplectic = CuArray{T}(symplectic)
    
    return GaussianUnitary(basis, gpu_disp, gpu_symplectic; ħ = ħ)
end

# Attenuator channel - GPU version
function attenuator(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}, theta::R, n::M; ħ = 2) where {Td<:CuVector, Tt<:CuMatrix, N<:Int, R, M}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return attenuator(Vector{eltype(Td)}, Matrix{eltype(Tt)}, basis, theta, n; ħ = ħ)
    end
    
    T = eltype(Td)
    
    # Use existing internal function
    disp, transform, noise = _attenuator(basis, theta, n)
    
    # Convert to GPU arrays with correct element type
    gpu_disp = CuArray{T}(disp)
    gpu_transform = CuArray{T}(transform)
    gpu_noise = CuArray{T}(noise)
    
    return GaussianChannel(basis, gpu_disp, gpu_transform, gpu_noise; ħ = ħ)
end

# Amplifier channel - GPU version
function amplifier(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}, r::R, n::M; ħ = 2) where {Td<:CuVector, Tt<:CuMatrix, N<:Int, R, M}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return amplifier(Vector{eltype(Td)}, Matrix{eltype(Tt)}, basis, r, n; ħ = ħ)
    end
    
    T = eltype(Td)
    
    # Use existing internal function
    disp, transform, noise = _amplifier(basis, r, n)
    
    # Convert to GPU arrays with correct element type
    gpu_disp = CuArray{T}(disp)
    gpu_transform = CuArray{T}(transform)
    gpu_noise = CuArray{T}(noise)
    
    return GaussianChannel(basis, gpu_disp, gpu_transform, gpu_noise; ħ = ħ)
end

# Convenience methods
function displace(::Type{T}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {T<:CuVector, N<:Int, A}
    return displace(T, CuMatrix{eltype(T)}, basis, alpha; ħ = ħ)
end

function squeeze(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {T<:CuVector, N<:Int, R}
    return squeeze(T, CuMatrix{eltype(T)}, basis, r, theta; ħ = ħ)
end

function twosqueeze(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {T<:CuVector, N<:Int, R}
    return twosqueeze(T, CuMatrix{eltype(T)}, basis, r, theta; ħ = ħ)
end

function phaseshift(::Type{T}, basis::SymplecticBasis{N}, theta::R; ħ = 2) where {T<:CuVector, N<:Int, R}
    return phaseshift(T, CuMatrix{eltype(T)}, basis, theta; ħ = ħ)
end

function beamsplitter(::Type{T}, basis::SymplecticBasis{N}, transmit::R; ħ = 2) where {T<:CuVector, N<:Int, R}
    return beamsplitter(T, CuMatrix{eltype(T)}, basis, transmit; ħ = ħ)
end

function attenuator(::Type{T}, basis::SymplecticBasis{N}, theta::R, n::M; ħ = 2) where {T<:CuVector, N<:Int, R, M}
    return attenuator(T, CuMatrix{eltype(T)}, basis, theta, n; ħ = ħ)
end

function amplifier(::Type{T}, basis::SymplecticBasis{N}, r::R, n::M; ħ = 2) where {T<:CuVector, N<:Int, R, M}
    return amplifier(T, CuMatrix{eltype(T)}, basis, r, n; ħ = ħ)
end