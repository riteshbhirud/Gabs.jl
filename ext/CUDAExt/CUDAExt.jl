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

# Check CUDA availability and provide graceful fallbacks
const CUDA_AVAILABLE = CUDA.functional()

function __init__()
    if CUDA_AVAILABLE
        @info "CUDA.jl extension loaded successfully. GPU acceleration enabled."
    else
        @warn "CUDA not available. GPU operations will fall back to CPU."
    end
end

include("utils.jl")
include("state_operations.jl") 
include("unitary_operations.jl")
include("wigner_kernels.jl")

end # module CUDAExt