# GPU kernels for Wigner function evaluation

"""
    wigner_kernel!(output, mean, covar_inv, covar_det, x_points, nmodes)

GPU kernel for computing Wigner function values at multiple phase space points.
"""
function wigner_kernel!(output, mean, covar_inv, covar_det, x_points, nmodes)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    npoints = size(x_points, 2)
    
    if idx <= npoints
        # Compute (x - mean)
        diff_sum = 0.0
        @inbounds for i in 1:(2*nmodes)
            x_i = x_points[i, idx]
            diff_i = x_i - mean[i]
            
            # Compute quadratic form: diff' * covar_inv * diff
            quad_term = 0.0
            for j in 1:(2*nmodes)
                x_j = x_points[j, idx]
                diff_j = x_j - mean[j]
                quad_term += diff_i * covar_inv[i, j] * diff_j
            end
            diff_sum += quad_term
        end
        
        # Compute Wigner function value
        normalization = 1.0 / ((2π)^nmodes * sqrt(covar_det))
        output[idx] = normalization * exp(-0.5 * diff_sum)
    end
    
    return nothing
end

"""
    wigner_kernel_optimized!(output, mean, covar_inv, covar_det, x_points, nmodes)

Optimized GPU kernel for Wigner function with reduced memory access.
"""
function wigner_kernel_optimized!(output, mean, covar_inv, covar_det, x_points, nmodes)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    npoints = size(x_points, 2)
    
    if idx <= npoints
        # Load x point into local memory
        diff_sum = 0.0
        
        @inbounds for i in 1:(2*nmodes)
            diff_i = x_points[i, idx] - mean[i]
            
            # Compute row of quadratic form
            for j in 1:(2*nmodes)
                diff_j = x_points[j, idx] - mean[j]
                diff_sum += diff_i * covar_inv[i, j] * diff_j
            end
        end
        
        # Compute final result
        normalization = 1.0 / ((2π)^nmodes * sqrt(covar_det))
        output[idx] = normalization * exp(-0.5 * diff_sum)
    end
    
    return nothing
end

"""
    wignerchar_kernel!(output, mean, covar, omega, xi_points, nmodes, hbar)

GPU kernel for computing Wigner characteristic function values.
"""
function wignerchar_kernel!(output, mean, covar, omega, xi_points, nmodes, hbar)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    npoints = size(xi_points, 2)
    
    if idx <= npoints
        # Compute arg1: -(1/2) * xi^T * (Omega*V*Omega^T) * xi
        arg1 = 0.0
        @inbounds for i in 1:(2*nmodes)
            xi_i = xi_points[i, idx]
            for j in 1:(2*nmodes)
                xi_j = xi_points[j, idx]
                
                # Compute (Omega*V*Omega^T)[i,j]
                omega_v_omega = 0.0
                for k in 1:(2*nmodes), l in 1:(2*nmodes)
                    omega_v_omega += omega[i,k] * covar[k,l] * omega[j,l]
                end
                
                arg1 += xi_i * omega_v_omega * xi_j
            end
        end
        arg1 *= -0.5
        
        # Compute arg2: im * (Omega*mean)^T * xi
        arg2_real = 0.0
        @inbounds for i in 1:(2*nmodes)
            omega_mean_i = 0.0
            for j in 1:(2*nmodes)
                omega_mean_i += omega[i,j] * mean[j]
            end
            arg2_real += omega_mean_i * xi_points[i, idx]
        end
        
        # Compute complex exponential
        exp_real = exp(arg1) * cos(arg2_real)
        exp_imag = exp(arg1) * (-sin(arg2_real))  # Negative because of -im
        
        # Store complex result (assuming output is complex)
        output[idx] = complex(exp_real, exp_imag)
    end
    
    return nothing
end

# GPU-accelerated Wigner function for single states
function Gabs.wigner(state::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, x::AbstractVector)
    if !cuda_available()
        @warn "CUDA not available, falling back to CPU" maxlog=1
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar), ħ=state.ħ)
        return Gabs.wigner(cpu_state, Array(x))
    end
    
    basis = state.basis
    nmodes = basis.nmodes
    mean = state.mean
    covar = state.covar
    
    # Validate input
    length(x) == length(mean) || throw(ArgumentError("Length of x must match mean vector length"))
    
    # Compute matrix inverse and determinant on GPU
    covar_inv = inv(covar)
    covar_det = det(covar)
    
    # Convert x to GPU if needed
    x_gpu = x isa CuArray ? x : CuArray(x)
    
    # Compute difference vector
    diff = x_gpu - mean
    
    # Compute quadratic form
    quad_form = dot(diff, covar_inv * diff)
    
    # Compute normalization
    normalization = 1.0 / ((2π)^nmodes * sqrt(covar_det))
    
    return normalization * exp(-0.5 * quad_form)
end

# GPU-accelerated Wigner function for batched evaluation  
function Gabs.wigner(state::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, x_grid::CuMatrix)
    if !cuda_available()
        @warn "CUDA not available, falling back to CPU" maxlog=1
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar), ħ=state.ħ)
        return Gabs.wigner(cpu_state, Array(x_grid))
    end
    
    basis = state.basis
    nmodes = basis.nmodes
    mean = state.mean
    covar = state.covar
    
    # Validate input dimensions
    size(x_grid, 1) == length(mean) || throw(ArgumentError("First dimension of x_grid must match mean vector length"))
    
    npoints = size(x_grid, 2)
    
    # Pre-compute matrix inverse and determinant
    covar_inv = inv(covar)
    covar_det = det(covar)
    
    # Allocate output
    output = CuArray{eltype(covar)}(undef, npoints)
    
    # Launch kernel
    threads = min(npoints, 256)
    blocks = cld(npoints, threads)
    
    @cuda threads=threads blocks=blocks wigner_kernel_optimized!(
        output, mean, covar_inv, covar_det, x_grid, nmodes
    )
    CUDA.synchronize()
    
    return output
end

# GPU-accelerated Wigner characteristic function for single states
function Gabs.wignerchar(state::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, xi::AbstractVector)
    if !cuda_available()
        @warn "CUDA not available, falling back to CPU" maxlog=1
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar), ħ=state.ħ)
        return Gabs.wignerchar(cpu_state, Array(xi))
    end
    
    basis = state.basis
    nmodes = basis.nmodes
    mean = state.mean
    covar = state.covar
    hbar = state.ħ
    
    # Validate input
    length(xi) == length(mean) || throw(ArgumentError("Length of xi must match mean vector length"))
    
    # Get symplectic form on GPU
    omega = CuArray(symplecticform(basis))
    
    # Convert xi to GPU if needed
    xi_gpu = xi isa CuArray ? xi : CuArray(xi)
    
    # Compute arg1: -(1/2) * xi^T * (Omega*V*Omega^T) * xi
    temp_mat = omega * covar * omega'
    arg1 = -0.5 * dot(xi_gpu, temp_mat * xi_gpu)
    
    # Compute arg2: im * (Omega*mean)^T * xi  
    omega_mean = omega * mean
    arg2 = 1im * dot(omega_mean, xi_gpu)
    
    return exp(arg1 - arg2)
end

# GPU-accelerated Wigner characteristic function for batched evaluation
function Gabs.wignerchar(state::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, xi_grid::CuMatrix)
    if !cuda_available()
        @warn "CUDA not available, falling back to CPU" maxlog=1
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar), ħ=state.ħ)
        return Gabs.wignerchar(cpu_state, Array(xi_grid))
    end
    
    basis = state.basis
    nmodes = basis.nmodes
    mean = state.mean
    covar = state.covar
    hbar = state.ħ
    
    # Validate input dimensions  
    size(xi_grid, 1) == length(mean) || throw(ArgumentError("First dimension of xi_grid must match mean vector length"))
    
    npoints = size(xi_grid, 2)
    
    # Get symplectic form on GPU
    omega = CuArray(symplecticform(basis))
    
    # Allocate output for complex results
    output = CuArray{ComplexF64}(undef, npoints)
    
    # Launch kernel
    threads = min(npoints, 256)
    blocks = cld(npoints, threads)
    
    @cuda threads=threads blocks=blocks wignerchar_kernel!(
        output, mean, covar, omega, xi_grid, nmodes, hbar
    )
    CUDA.synchronize()
    
    return output
end

# Fallback methods for CPU arrays with GPU states (promote to GPU)
function Gabs.wigner(state::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, x::Vector)
    return Gabs.wigner(state, CuArray(x))
end

function Gabs.wigner(state::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, x_grid::Matrix)
    return Array(Gabs.wigner(state, CuArray(x_grid)))
end

function Gabs.wignerchar(state::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, xi::Vector)
    return Gabs.wignerchar(state, CuArray(xi))
end

function Gabs.wignerchar(state::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, xi_grid::Matrix)
    return Array(Gabs.wignerchar(state, CuArray(xi_grid)))
end