module CUDAExt

using CUDA
using LinearAlgebra
using Random

import Gabs
using Gabs: GaussianState, GaussianUnitary, GaussianChannel, GaussianLinearCombination,
           SymplecticBasis, QuadPairBasis, QuadBlockBasis,
           _promote_output_vector, _promote_output_matrix,
           symplecticform, wigner, wignerchar, cross_wigner, cross_wignerchar,
           vacuumstate, coherentstate, squeezedstate, thermalstate, eprstate,
           displace, squeeze, twosqueeze, phaseshift, beamsplitter,
           attenuator, amplifier, randstate, randunitary, randchannel,
           tensor, ptrace, changebasis, apply!,
           STATE_ERROR, UNITARY_ERROR, CHANNEL_ERROR, ACTION_ERROR, 
           SYMPLECTIC_ERROR, HBAR_ERROR, WIGNER_ERROR, INDEX_ERROR

include("arrays.jl")
include("states.jl") 
include("operations.jl")
include("linearcombinations.jl")
include("wigner.jl")
include("kernels.jl")
include("randoms.jl")
include("utils.jl")

end