module CUDAExt

using CUDA
using Gabs
using LinearAlgebra: I, det, mul!, diag, eigvals, Diagonal, dot, logdet

# Import internal functions and types that we need to extend for GPU
import Gabs: _promote_output_vector, _promote_output_matrix,
             _vacuumstate, _thermalstate, _coherentstate, _squeezedstate, _eprstate,
             _displace, _squeeze, _twosqueeze, _phaseshift, _beamsplitter,
             _attenuator, _amplifier,
             GaussianState, GaussianUnitary, GaussianChannel,
             QuadPairBasis, QuadBlockBasis, SymplecticBasis,
             symplecticform, wigner, wignerchar

# Check CUDA availability at module load time
const CUDA_AVAILABLE = CUDA.functional()

if CUDA_AVAILABLE
    @info "CUDA GPU support enabled for Gabs.jl"
else
    @warn "CUDA not available - GPU operations will fall back to CPU"
end

# Include all GPU implementation files
include("utils.jl")
include("state_operations.jl") 
include("unitary_operations.jl")
include("wigner_kernels.jl")

end # module CUDAExt