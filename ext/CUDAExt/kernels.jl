# ext/CUDAExt/kernels.jl
# Complete optimized CUDA kernels for high-performance operations

using CUDA
using LinearAlgebra

# Optimized Wigner function kernel for linear combinations
function wigner_kernel!(output, means, covars, x_points, coeffs, n_states, n_points)
    """
    Parallel Wigner function evaluation for linear combinations of Gaussian states
    Includes both diagonal terms and cross-interference terms
    """
    # Thread and block indices
    point_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if point_idx <= n_points
        result = 0.0
        
        # Extract current point coordinates
        x1 = x_points[1, point_idx]
        x2 = x_points[2, point_idx]
        
        # Diagonal terms: |cᵢ|² W_ψᵢ(x)
        @inbounds for i in 1:n_states
            coeff_i = coeffs[i]
            
            # Extract mean and covariance for state i
            μ1 = means[1, i]
            μ2 = means[2, i]
            
            V11 = covars[1, 1, i]
            V12 = covars[1, 2, i]
            V21 = covars[2, 1, i]
            V22 = covars[2, 2, i]
            
            # Displacement from mean
            diff1 = x1 - μ1
            diff2 = x2 - μ2
            
            # Inverse of covariance matrix
            det_V = V11 * V22 - V12 * V21
            
            # Check for numerical stability
            if abs(det_V) < 1e-16
                continue  # Skip degenerate states
            end
            
            V_inv11 = V22 / det_V
            V_inv12 = -V12 / det_V
            V_inv21 = -V21 / det_V
            V_inv22 = V11 / det_V
            
            # Quadratic form: (x-μ)ᵀ V⁻¹ (x-μ)
            quad_form = diff1 * (V_inv11 * diff1 + V_inv12 * diff2) + 
                       diff2 * (V_inv21 * diff1 + V_inv22 * diff2)
            
            # Wigner function value with proper normalization
            normalization = 1.0 / (2π * sqrt(abs(det_V)))
            w_i = normalization * exp(-0.5 * quad_form)
            
            # Add diagonal contribution
            result += abs2(coeff_i) * w_i
        end
        
        # Cross terms (quantum interference): 2Re[cᵢ* cⱼ W_ψᵢψⱼ(x)]
        @inbounds for i in 1:n_states
            coeff_i = coeffs[i]
            
            # Extract state i parameters
            μ1_i = means[1, i]
            μ2_i = means[2, i]
            V11_i = covars[1, 1, i]
            V12_i = covars[1, 2, i]
            V21_i = covars[2, 1, i]
            V22_i = covars[2, 2, i]
            
            @inbounds for j in (i+1):n_states
                coeff_j = coeffs[j]
                
                # Extract state j parameters
                μ1_j = means[1, j]
                μ2_j = means[2, j]
                V11_j = covars[1, 1, j]
                V12_j = covars[1, 2, j]
                V21_j = covars[2, 1, j]
                V22_j = covars[2, 2, j]
                
                # Cross-Wigner function calculation
                # Average covariance: V_avg = (Vᵢ + Vⱼ)/2
                V_avg11 = 0.5 * (V11_i + V11_j)
                V_avg12 = 0.5 * (V12_i + V12_j)
                V_avg21 = 0.5 * (V21_i + V21_j)
                V_avg22 = 0.5 * (V22_i + V22_j)
                
                # Average mean: μ_avg = (μᵢ + μⱼ)/2
                μ_avg1 = 0.5 * (μ1_i + μ1_j)
                μ_avg2 = 0.5 * (μ2_i + μ2_j)
                
                # Displacement from average
                dx1 = x1 - μ_avg1
                dx2 = x2 - μ_avg2
                
                # Inverse of average covariance
                det_V_avg = V_avg11 * V_avg22 - V_avg12 * V_avg21
                
                if abs(det_V_avg) < 1e-16
                    continue  # Skip if average covariance is degenerate
                end
                
                V_avg_inv11 = V_avg22 / det_V_avg
                V_avg_inv12 = -V_avg12 / det_V_avg
                V_avg_inv21 = -V_avg21 / det_V_avg
                V_avg_inv22 = V_avg11 / det_V_avg
                
                # Quadratic form for cross term
                quad_avg = dx1 * (V_avg_inv11 * dx1 + V_avg_inv12 * dx2) +
                          dx2 * (V_avg_inv21 * dx1 + V_avg_inv22 * dx2)
                
                # Mean difference factor for phase
                Δμ1 = μ1_i - μ1_j
                Δμ2 = μ2_i - μ2_j
                
                # Phase factor from mean difference
                phase_factor = -(dx1 * Δμ1 + dx2 * Δμ2) / sqrt(abs(det_V_avg))
                
                # Cross-Wigner normalization
                det_Vi = V11_i * V22_i - V12_i * V21_i
                det_Vj = V11_j * V22_j - V12_j * V21_j
                
                if abs(det_Vi) < 1e-16 || abs(det_Vj) < 1e-16
                    continue
                end
                
                cross_norm = sqrt(sqrt(abs(det_Vi * det_Vj)) / abs(det_V_avg)) / (2π)
                
                # Complete cross-Wigner function
                cross_wigner = cross_norm * exp(-0.5 * quad_avg) * cos(phase_factor)
                
                # Add cross term contribution (factor of 2 for symmetry)
                cross_coeff = 2.0 * real(conj(coeff_i) * coeff_j)
                result += cross_coeff * cross_wigner
            end
        end
        
        output[point_idx] = result
    end
    
    return nothing
end

# Optimized batch Gaussian state transformation kernel
function gaussian_transform_kernel!(means_out, covars_out, means_in, covars_in, 
                                   transform_matrices, displacements, n_states)
    """
    Parallel transformation of multiple Gaussian states
    Applies: μ_out = T * μ_in + d, V_out = T * V_in * Tᵀ
    """
    state_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if state_idx <= n_states
        # Extract transformation matrix (assuming 2x2 for single mode)
        T11 = transform_matrices[1, 1, state_idx]
        T12 = transform_matrices[1, 2, state_idx]
        T21 = transform_matrices[2, 1, state_idx]
        T22 = transform_matrices[2, 2, state_idx]
        
        # Extract displacement
        d1 = displacements[1, state_idx]
        d2 = displacements[2, state_idx]
        
        # Extract input mean
        μ1_in = means_in[1, state_idx]
        μ2_in = means_in[2, state_idx]
        
        # Transform mean: μ_out = T * μ_in + d
        means_out[1, state_idx] = T11 * μ1_in + T12 * μ2_in + d1
        means_out[2, state_idx] = T21 * μ1_in + T22 * μ2_in + d2
        
        # Extract input covariance
        V11_in = covars_in[1, 1, state_idx]
        V12_in = covars_in[1, 2, state_idx]
        V21_in = covars_in[2, 1, state_idx]
        V22_in = covars_in[2, 2, state_idx]
        
        # First step: T * V_in
        TV11 = T11 * V11_in + T12 * V21_in
        TV12 = T11 * V12_in + T12 * V22_in
        TV21 = T21 * V11_in + T22 * V21_in
        TV22 = T21 * V12_in + T22 * V22_in
        
        # Second step: (T * V_in) * Tᵀ
        covars_out[1, 1, state_idx] = TV11 * T11 + TV12 * T21
        covars_out[1, 2, state_idx] = TV11 * T12 + TV12 * T22
        covars_out[2, 1, state_idx] = TV21 * T11 + TV22 * T21
        covars_out[2, 2, state_idx] = TV21 * T12 + TV22 * T22
    end
    
    return nothing
end

# Kernel for parallel overlap calculations between Gaussian states
function overlap_kernel!(overlaps, means1, covars1, means2, covars2, n_pairs)
    """
    Compute overlaps ⟨ψ₁|ψ₂⟩ between pairs of Gaussian states
    Formula: exp(-¼(μ₁-μ₂)ᵀ(V₁+V₂)⁻¹(μ₁-μ₂)) * √(det(V₁)det(V₂)/det(V₁+V₂))
    """
    pair_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if pair_idx <= n_pairs
        # Extract means
        μ1_1 = means1[1, pair_idx]
        μ2_1 = means1[2, pair_idx]
        μ1_2 = means2[1, pair_idx]
        μ2_2 = means2[2, pair_idx]
        
        # Extract covariances for state 1
        V11_1 = covars1[1, 1, pair_idx]
        V12_1 = covars1[1, 2, pair_idx]
        V21_1 = covars1[2, 1, pair_idx]
        V22_1 = covars1[2, 2, pair_idx]
        
        # Extract covariances for state 2
        V11_2 = covars2[1, 1, pair_idx]
        V12_2 = covars2[1, 2, pair_idx]
        V21_2 = covars2[2, 1, pair_idx]
        V22_2 = covars2[2, 2, pair_idx]
        
        # Sum of covariances
        V_sum11 = V11_1 + V11_2
        V_sum12 = V12_1 + V12_2
        V_sum21 = V21_1 + V21_2
        V_sum22 = V22_1 + V22_2
        
        # Mean difference
        Δμ1 = μ1_1 - μ1_2
        Δμ2 = μ2_1 - μ2_2
        
        # Determinant and inverse of sum covariance
        det_V_sum = V_sum11 * V_sum22 - V_sum12 * V_sum21
        
        # Check for numerical stability
        if abs(det_V_sum) < 1e-16
            overlaps[pair_idx] = 0.0
            return nothing
        end
        
        V_sum_inv11 = V_sum22 / det_V_sum
        V_sum_inv12 = -V_sum12 / det_V_sum
        V_sum_inv21 = -V_sum21 / det_V_sum
        V_sum_inv22 = V_sum11 / det_V_sum
        
        # Quadratic form for exponential argument
        quad_form = Δμ1 * (V_sum_inv11 * Δμ1 + V_sum_inv12 * Δμ2) +
                   Δμ2 * (V_sum_inv21 * Δμ1 + V_sum_inv22 * Δμ2)
        
        # Individual determinants
        det_V1 = V11_1 * V22_1 - V12_1 * V21_1
        det_V2 = V11_2 * V22_2 - V12_2 * V21_2
        
        # Check individual determinants
        if abs(det_V1) < 1e-16 || abs(det_V2) < 1e-16
            overlaps[pair_idx] = 0.0
            return nothing
        end
        
        # Overlap calculation with proper normalization
        exp_factor = exp(-0.25 * quad_form)
        det_factor = sqrt(abs(det_V1 * det_V2 / det_V_sum))
        
        overlaps[pair_idx] = exp_factor * det_factor
    end
    
    return nothing
end

# Optimized multi-mode Wigner kernel with shared memory
function multimode_wigner_kernel!(output, means, covars, x_points, coeffs, 
                                 n_states, n_points, n_modes)
    """
    Optimized Wigner function for multi-mode states using shared memory
    """
    # Shared memory for frequently accessed data
    shmem_size = 32  # Adjust based on available shared memory
    shared_means = @cuStaticSharedMem(Float64, (2, shmem_size))
    shared_covars = @cuStaticSharedMem(Float64, (4, shmem_size))  # Flattened 2x2 matrices
    
    point_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    thread_id = threadIdx().x
    
    if point_idx <= n_points
        result = 0.0
        
        # Extract current point (generalized for n_modes)
        x_coords = ntuple(i -> x_points[i, point_idx], 2*n_modes)
        
        # Process states in chunks that fit in shared memory
        for chunk_start in 1:shmem_size:n_states
            chunk_end = min(chunk_start + shmem_size - 1, n_states)
            chunk_size = chunk_end - chunk_start + 1
            
            # Load chunk into shared memory
            if thread_id <= chunk_size
                state_idx = chunk_start + thread_id - 1
                if state_idx <= n_states
                    shared_means[1, thread_id] = means[1, state_idx]
                    shared_means[2, thread_id] = means[2, state_idx]
                    shared_covars[1, thread_id] = covars[1, 1, state_idx]
                    shared_covars[2, thread_id] = covars[1, 2, state_idx]
                    shared_covars[3, thread_id] = covars[2, 1, state_idx]
                    shared_covars[4, thread_id] = covars[2, 2, state_idx]
                end
            end
            
            sync_threads()
            
            # Process chunk
            for local_idx in 1:chunk_size
                global_idx = chunk_start + local_idx - 1
                if global_idx <= n_states
                    coeff = coeffs[global_idx]
                    
                    # Extract from shared memory
                    μ1 = shared_means[1, local_idx]
                    μ2 = shared_means[2, local_idx]
                    V11 = shared_covars[1, local_idx]
                    V12 = shared_covars[2, local_idx]
                    V21 = shared_covars[3, local_idx]
                    V22 = shared_covars[4, local_idx]
                    
                    # Compute Wigner function (same as before but using shared memory)
                    diff1 = x_coords[1] - μ1
                    diff2 = x_coords[2] - μ2
                    
                    det_V = V11 * V22 - V12 * V21
                    if abs(det_V) > 1e-16
                        V_inv11 = V22 / det_V
                        V_inv12 = -V12 / det_V
                        V_inv21 = -V21 / det_V
                        V_inv22 = V11 / det_V
                        
                        quad_form = diff1 * (V_inv11 * diff1 + V_inv12 * diff2) + 
                                   diff2 * (V_inv21 * diff1 + V_inv22 * diff2)
                        
                        normalization = 1.0 / (2π * sqrt(abs(det_V)))
                        w = normalization * exp(-0.5 * quad_form)
                        
                        result += abs2(coeff) * w
                    end
                end
            end
            
            sync_threads()
        end
        
        output[point_idx] = result
    end
    
    return nothing
end

# Kernel for batched linear algebra operations
function batch_matrix_multiply_kernel!(C, A, B, n_matrices, matrix_size)
    """
    Batched matrix multiplication: C[i] = A[i] * B[i]
    For matrix_size x matrix_size matrices
    """
    matrix_idx = blockIdx().x
    thread_idx = threadIdx().x
    
    if matrix_idx <= n_matrices && thread_idx <= matrix_size * matrix_size
        # Convert linear thread index to matrix indices
        row = (thread_idx - 1) ÷ matrix_size + 1
        col = (thread_idx - 1) % matrix_size + 1
        
        if row <= matrix_size && col <= matrix_size
            result = 0.0
            
            @inbounds for k in 1:matrix_size
                result += A[row, k, matrix_idx] * B[k, col, matrix_idx]
            end
            
            C[row, col, matrix_idx] = result
        end
    end
    
    return nothing
end

# High-level interface functions using the kernels

function gpu_batch_wigner_evaluation(lc::GaussianLinearCombination, x_points::CuMatrix{Float64})
    """
    High-level interface for batch Wigner function evaluation
    """
    n_states = length(lc)
    n_points = size(x_points, 2)
    
    # Prepare data arrays on GPU
    means = CuArray(zeros(Float64, 2, n_states))
    covars = CuArray(zeros(Float64, 2, 2, n_states))
    coeffs = CuArray(Float64[c for c in lc.coeffs])
    
    # Fill data arrays
    for i in 1:n_states
        state = lc.states[i]
        means[:, i] = state.mean
        covars[:, :, i] = state.covar
    end
    
    # Output array
    output = CuArray(zeros(Float64, n_points))
    
    # Launch kernel with optimal configuration
    threads_per_block = min(256, n_points)
    blocks = cld(n_points, threads_per_block)
    
    @cuda threads=threads_per_block blocks=blocks wigner_kernel!(
        output, means, covars, x_points, coeffs, n_states, n_points
    )
    
    return output
end

function gpu_batch_state_transform(states::Vector{GaussianState}, transforms, displacements)
    """
    Apply transformations to multiple states in parallel
    """
    n_states = length(states)
    
    # Prepare input arrays
    means_in = CuArray(hcat([s.mean for s in states]...))
    covars_in = CuArray(cat([s.covar for s in states]..., dims=3))
    
    # Output arrays
    means_out = similar(means_in)
    covars_out = similar(covars_in)
    
    # Launch transformation kernel
    threads_per_block = min(256, n_states)
    blocks = cld(n_states, threads_per_block)
    
    @cuda threads=threads_per_block blocks=blocks gaussian_transform_kernel!(
        means_out, covars_out, means_in, covars_in, transforms, displacements, n_states
    )
    
    return means_out, covars_out
end

function gpu_batch_overlaps(states1::Vector{GaussianState}, states2::Vector{GaussianState})
    """
    Compute overlaps between pairs of states in parallel
    """
    n_pairs = min(length(states1), length(states2))
    
    # Prepare data arrays
    means1 = CuArray(hcat([s.mean for s in states1[1:n_pairs]]...))
    covars1 = CuArray(cat([s.covar for s in states1[1:n_pairs]]..., dims=3))
    means2 = CuArray(hcat([s.mean for s in states2[1:n_pairs]]...))
    covars2 = CuArray(cat([s.covar for s in states2[1:n_pairs]]..., dims=3))
    
    # Output array
    overlaps = CuArray(zeros(Float64, n_pairs))
    
    # Launch kernel
    threads_per_block = min(256, n_pairs)
    blocks = cld(n_pairs, threads_per_block)
    
    @cuda threads=threads_per_block blocks=blocks overlap_kernel!(
        overlaps, means1, covars1, means2, covars2, n_pairs
    )
    
    return overlaps
end

# Utility function for optimal kernel configuration
function optimal_kernel_config(n_elements::Int, max_threads_per_block::Int=1024)
    """
    Determine optimal CUDA kernel configuration
    """
    if n_elements <= 32
        threads = 32
    elseif n_elements <= 128
        threads = 128
    elseif n_elements <= 512
        threads = 512
    else
        threads = min(max_threads_per_block, 1024)
    end
    
    blocks = cld(n_elements, threads)
    
    return (threads=threads, blocks=blocks)
end

# Memory-optimized kernel launcher
function launch_kernel_with_memory_check(kernel_func, args...; memory_threshold_mb=512)
    """
    Launch kernel with automatic memory management
    """
    # Check available memory
    free_mem, total_mem = CUDA.memory_info()
    free_mb = free_mem ÷ (1024^2)
    
    if free_mb < memory_threshold_mb
        @warn "Low GPU memory: $(free_mb) MB free. Consider reducing batch size."
        CUDA.reclaim()  # Force garbage collection
    end
    
    # Launch kernel
    return kernel_func(args...)
end