module CUDAExt

using CUDA
using CUDA: CuArray, CuVector, CuMatrix, @cuda, threadIdx, blockIdx, blockDim, gridDim
using Gabs
using Gabs: SymplecticBasis, QuadPairBasis, QuadBlockBasis
using Gabs: GaussianState, GaussianUnitary, GaussianChannel
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

function Gabs.coherentstate(basis::SymplecticBasis{N}, alpha::CuArray; ħ=2) where {N<:Int64}
    T = real(eltype(alpha))
    alpha_cpu = Array(alpha)
    alpha_val = ndims(alpha_cpu) == 0 ? alpha_cpu[] : (length(alpha_cpu) == 1 ? alpha_cpu[1] : alpha_cpu)
    return coherentstate(CuVector{T}, CuMatrix{T}, basis, alpha_val; ħ=ħ)
end

function Gabs.squeezedstate(basis::SymplecticBasis{N}, r::CuArray, theta; ħ=2) where {N<:Int64}
    T = eltype(r)
    r_cpu = Array(r)
    r_val = ndims(r_cpu) == 0 ? r_cpu[] : (length(r_cpu) == 1 ? r_cpu[1] : r_cpu)
    return squeezedstate(CuVector{T}, CuMatrix{T}, basis, r_val, theta; ħ=ħ)
end

function Gabs.thermalstate(basis::SymplecticBasis{N}, photons::CuArray; ħ=2) where {N<:Int64}
    T = eltype(photons)
    photons_cpu = Array(photons)
    photons_val = ndims(photons_cpu) == 0 ? photons_cpu[] : (length(photons_cpu) == 1 ? photons_cpu[1] : photons_cpu)
    return thermalstate(CuVector{T}, CuMatrix{T}, basis, photons_val; ħ=ħ)
end

function Gabs.eprstate(basis::SymplecticBasis{N}, r::CuArray, theta; ħ=2) where {N<:Int64}
    T = eltype(r)
    r_cpu = Array(r)
    r_val = ndims(r_cpu) == 0 ? r_cpu[] : (length(r_cpu) == 1 ? r_cpu[1] : r_cpu)
    return eprstate(CuVector{T}, CuMatrix{T}, basis, r_val, theta; ħ=ħ)
end

function Gabs.displace(basis::SymplecticBasis{N}, alpha::CuArray; ħ=2) where {N<:Int64}
    T = real(eltype(alpha))
    alpha_cpu = Array(alpha)
    alpha_val = ndims(alpha_cpu) == 0 ? alpha_cpu[] : (length(alpha_cpu) == 1 ? alpha_cpu[1] : alpha_cpu)
    return displace(CuVector{T}, CuMatrix{T}, basis, alpha_val; ħ=ħ)
end

function Gabs.squeeze(basis::SymplecticBasis{N}, r::CuArray, theta; ħ=2) where {N<:Int64}
    T = eltype(r)
    r_cpu = Array(r)
    r_val = ndims(r_cpu) == 0 ? r_cpu[] : (length(r_cpu) == 1 ? r_cpu[1] : r_cpu)
    return squeeze(CuVector{T}, CuMatrix{T}, basis, r_val, theta; ħ=ħ)
end

function Gabs.twosqueeze(basis::SymplecticBasis{N}, r::CuArray, theta; ħ=2) where {N<:Int64}
    T = eltype(r)
    r_cpu = Array(r)
    r_val = ndims(r_cpu) == 0 ? r_cpu[] : (length(r_cpu) == 1 ? r_cpu[1] : r_cpu)
    return twosqueeze(CuVector{T}, CuMatrix{T}, basis, r_val, theta; ħ=ħ)
end

function Gabs.phaseshift(basis::SymplecticBasis{N}, theta::CuArray; ħ=2) where {N<:Int64}
    T = eltype(theta)
    theta_cpu = Array(theta)
    theta_val = ndims(theta_cpu) == 0 ? theta_cpu[] : (length(theta_cpu) == 1 ? theta_cpu[1] : theta_cpu)
    return phaseshift(CuVector{T}, CuMatrix{T}, basis, theta_val; ħ=ħ)
end

function Gabs.beamsplitter(basis::SymplecticBasis{N}, transmit::CuArray; ħ=2) where {N<:Int64}
    T = eltype(transmit)
    transmit_cpu = Array(transmit)
    transmit_val = ndims(transmit_cpu) == 0 ? transmit_cpu[] : (length(transmit_cpu) == 1 ? transmit_cpu[1] : transmit_cpu)
    return beamsplitter(CuVector{T}, CuMatrix{T}, basis, transmit_val; ħ=ħ)
end

function Gabs.attenuator(basis::SymplecticBasis{N}, theta::CuArray, n; ħ=2) where {N<:Int64}
    T = eltype(theta)
    theta_cpu = Array(theta)
    theta_val = ndims(theta_cpu) == 0 ? theta_cpu[] : (length(theta_cpu) == 1 ? theta_cpu[1] : theta_cpu)
    return attenuator(CuVector{T}, CuMatrix{T}, basis, theta_val, n; ħ=ħ)
end

function Gabs.amplifier(basis::SymplecticBasis{N}, r::CuArray, n; ħ=2) where {N<:Int64}
    T = eltype(r)
    r_cpu = Array(r)
    r_val = ndims(r_cpu) == 0 ? r_cpu[] : (length(r_cpu) == 1 ? r_cpu[1] : r_cpu)
    return amplifier(CuVector{T}, CuMatrix{T}, basis, r_val, n; ħ=ħ)
end

function Gabs._adapt_device_gpu(target_constructor, source_obj, args...)
    T = eltype(source_obj isa GaussianState ? source_obj.mean : 
              source_obj isa GaussianUnitary ? source_obj.disp : source_obj.disp)
    precision = real(T)
    return target_constructor(CuVector{precision}, CuMatrix{precision}, args...)
end
function Gabs.tensor(state1::GaussianState{B,M1,V1}, state2::GaussianState{B,M2,V2}) where {B<:SymplecticBasis, M1<:CuArray, V1<:CuArray, M2<:CuArray, V2<:CuArray}
    T = promote_type(eltype(M1), eltype(M2))
    return tensor(CuVector{T}, CuMatrix{T}, state1, state2)
end
include("utils.jl")
include("state_operations.jl") 
include("unitary_operations.jl")
include("wigner_kernels.jl")

end