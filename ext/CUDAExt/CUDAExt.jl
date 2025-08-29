module CUDAExt

using CUDA
using CUDA: CuArray, CuVector, CuMatrix, @cuda, threadIdx, blockIdx, blockDim, gridDim
using Gabs
using Gabs: SymplecticBasis, QuadPairBasis, QuadBlockBasis
using Gabs: GaussianState, GaussianUnitary, GaussianChannel, GaussianLinearCombination
using Gabs: _promote_output_vector, _promote_output_matrix
using Gabs: _vacuumstate, _coherentstate, _squeezedstate, _thermalstate, _eprstate
using Gabs: _displace, _squeeze, _twosqueeze, _phaseshift, _beamsplitter
using Gabs: _attenuator, _amplifier
using Gabs: symplecticform, WIGNER_ERROR
using LinearAlgebra: I, det, mul!, eigvals, Diagonal, logdet, dot, inv
using Random: randn!

const CUDA_AVAILABLE = CUDA.functional()

function __init__()
    if CUDA_AVAILABLE
        @info "CUDA.jl extension loaded successfully. GPU acceleration enabled."
    else
        @warn "CUDA not available. GPU operations will fall back to CPU."
    end
end

# Core GPU implementations for convenience API
function Gabs._gpu_impl(state::GaussianState, precision)
    gpu_mean = CuArray{precision}(state.mean)
    gpu_covar = CuArray{precision}(state.covar)
    return GaussianState(state.basis, gpu_mean, gpu_covar; ħ = state.ħ)
end

function Gabs._gpu_impl(op::GaussianUnitary, precision)
    gpu_disp = CuArray{precision}(op.disp)
    gpu_symplectic = CuArray{precision}(op.symplectic)
    return GaussianUnitary(op.basis, gpu_disp, gpu_symplectic; ħ = op.ħ)
end

function Gabs._gpu_impl(op::GaussianChannel, precision)
    gpu_disp = CuArray{precision}(op.disp)
    gpu_transform = CuArray{precision}(op.transform)
    gpu_noise = CuArray{precision}(op.noise)
    return GaussianChannel(op.basis, gpu_disp, gpu_transform, gpu_noise; ħ = op.ħ)
end

function Gabs._gpu_impl(lc::GaussianLinearCombination, precision)
    gpu_coeffs = CuArray{precision}(lc.coeffs)
    gpu_states = [Gabs._gpu_impl(state, precision) for state in lc.states]
    return GaussianLinearCombination(lc.basis, gpu_coeffs, gpu_states)
end

function Gabs._gpu_impl(x::AbstractArray, precision)
    return CuArray{precision}(x)
end

function Gabs._detect_device(x::CuArray)
    return :gpu
end

function _extract_gpu_scalar(x::CuArray{T}) where T
    if ndims(x) == 0
        # 0-dimensional array - convert to CPU scalar safely
        return T(Array(x)[])
    elseif length(x) == 1
        # Single element - convert to CPU scalar safely
        return T(Array(x)[1])
    else
        # Multi-element - convert to CPU vector for internal functions
        # This is still efficient - only transfers the small input vector, not large state matrices
        return Vector{T}(Array(x))
    end
end

# EFFICIENT automatic dispatch functions - NO scalar indexing
function Gabs.coherentstate(basis::SymplecticBasis{N}, alpha::CuArray; ħ=2) where {N<:Int}
    T = real(eltype(alpha))
    # Extract scalar efficiently - no GPU scalar indexing
    alpha_val = _extract_gpu_scalar(alpha)
    return coherentstate(CuVector{T}, CuMatrix{T}, basis, alpha_val; ħ=ħ)
end

function Gabs.squeezedstate(basis::SymplecticBasis{N}, r::CuArray, theta; ħ=2) where {N<:Int}
    T = eltype(r)
    # Extract scalar efficiently - no GPU scalar indexing
    r_val = _extract_gpu_scalar(r)
    return squeezedstate(CuVector{T}, CuMatrix{T}, basis, r_val, theta; ħ=ħ)
end

function Gabs.thermalstate(basis::SymplecticBasis{N}, photons::CuArray; ħ=2) where {N<:Int}
    T = eltype(photons)
    # Extract scalar efficiently - no GPU scalar indexing
    photons_val = _extract_gpu_scalar(photons)
    return thermalstate(CuVector{T}, CuMatrix{T}, basis, photons_val; ħ=ħ)
end

function Gabs.eprstate(basis::SymplecticBasis{N}, r::CuArray, theta; ħ=2) where {N<:Int}
    T = eltype(r)
    # Extract scalar efficiently - no GPU scalar indexing
    r_val = _extract_gpu_scalar(r)
    return eprstate(CuVector{T}, CuMatrix{T}, basis, r_val, theta; ħ=ħ)
end

function Gabs.displace(basis::SymplecticBasis{N}, alpha::CuArray; ħ=2) where {N<:Int}
    T = real(eltype(alpha))
    # Extract scalar efficiently - no GPU scalar indexing
    alpha_val = _extract_gpu_scalar(alpha)
    return displace(CuVector{T}, CuMatrix{T}, basis, alpha_val; ħ=ħ)
end

function Gabs.squeeze(basis::SymplecticBasis{N}, r::CuArray, theta; ħ=2) where {N<:Int}
    T = eltype(r)
    # Extract scalar efficiently - no GPU scalar indexing
    r_val = _extract_gpu_scalar(r)
    return squeeze(CuVector{T}, CuMatrix{T}, basis, r_val, theta; ħ=ħ)
end

function Gabs.twosqueeze(basis::SymplecticBasis{N}, r::CuArray, theta; ħ=2) where {N<:Int}
    T = eltype(r)
    # Extract scalar efficiently - no GPU scalar indexing
    r_val = _extract_gpu_scalar(r)
    return twosqueeze(CuVector{T}, CuMatrix{T}, basis, r_val, theta; ħ=ħ)
end

function Gabs.phaseshift(basis::SymplecticBasis{N}, theta::CuArray; ħ=2) where {N<:Int}
    T = eltype(theta)
    # Extract scalar efficiently - no GPU scalar indexing
    theta_val = _extract_gpu_scalar(theta)
    return phaseshift(CuVector{T}, CuMatrix{T}, basis, theta_val; ħ=ħ)
end

function Gabs.beamsplitter(basis::SymplecticBasis{N}, transmit::CuArray; ħ=2) where {N<:Int}
    T = eltype(transmit)
    # Extract scalar efficiently - no GPU scalar indexing
    transmit_val = _extract_gpu_scalar(transmit)
    return beamsplitter(CuVector{T}, CuMatrix{T}, basis, transmit_val; ħ=ħ)
end

function Gabs.attenuator(basis::SymplecticBasis{N}, theta::CuArray, n; ħ=2) where {N<:Int}
    T = eltype(theta)
    # Extract scalar efficiently - no GPU scalar indexing
    theta_val = _extract_gpu_scalar(theta)
    return attenuator(CuVector{T}, CuMatrix{T}, basis, theta_val, n; ħ=ħ)
end

function Gabs.amplifier(basis::SymplecticBasis{N}, r::CuArray, n; ħ=2) where {N<:Int}
    T = eltype(r)
    # Extract scalar efficiently - no GPU scalar indexing  
    r_val = _extract_gpu_scalar(r)
    return amplifier(CuVector{T}, CuMatrix{T}, basis, r_val, n; ħ=ħ)
end

# Efficient tensor products for GPU arrays
function Gabs.tensor(state1::GaussianState{B,M1,V1}, state2::GaussianState{B,M2,V2}) where {B<:SymplecticBasis, M1<:CuArray, V1<:CuArray, M2<:CuArray, V2<:CuArray}
    T = promote_type(eltype(M1), eltype(M2))
    return tensor(CuVector{T}, CuMatrix{T}, state1, state2)
end

function Gabs.tensor(::Type{T}, op1::GaussianUnitary{B,D1,S1}, op2::GaussianUnitary{B,D2,S2}) where {
    T<:CuVector, B<:SymplecticBasis, D1<:CuArray, S1<:CuArray, D2<:CuArray, S2<:CuArray}
    return tensor(T, CuMatrix{eltype(T)}, op1, op2)
end

function Gabs.tensor(op1::GaussianUnitary{B,D1,S1}, op2::GaussianUnitary{B,D2,S2}) where {
    B<:SymplecticBasis, D1<:CuArray, S1<:CuArray, D2<:CuArray, S2<:CuArray}
    T = promote_type(eltype(D1), eltype(D2))
    return tensor(CuVector{T}, CuMatrix{T}, op1, op2)
end

function Gabs.tensor(::Type{T}, op1::GaussianChannel{B,D1,T1}, op2::GaussianChannel{B,D2,T2}) where {
    T<:CuVector, B<:SymplecticBasis, D1<:CuArray, T1<:CuArray, D2<:CuArray, T2<:CuArray}
    return tensor(T, CuMatrix{eltype(T)}, op1, op2)
end

function Gabs.tensor(op1::GaussianChannel{B,D1,T1}, op2::GaussianChannel{B,D2,T2}) where {
    B<:SymplecticBasis, D1<:CuArray, T1<:CuArray, D2<:CuArray, T2<:CuArray}
    T = promote_type(eltype(D1), eltype(D2))
    return tensor(CuVector{T}, CuMatrix{T}, op1, op2)
end

# Efficient GPU adaptation helper 
function Gabs._adapt_device_gpu(target_constructor, source_obj, args...)
    T = eltype(source_obj isa GaussianState ? source_obj.mean : 
              source_obj isa GaussianUnitary ? source_obj.disp : source_obj.disp)
    precision = real(T)
    return target_constructor(CuVector{precision}, CuMatrix{precision}, args...)
end

include("utils.jl")
include("state_operations.jl") 
include("unitary_operations.jl")
include("wigner_kernels.jl")

end