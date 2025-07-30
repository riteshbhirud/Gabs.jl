# Custom CUDA kernels for high-performance operations

# Kernel for parallel Wigner function evaluation
function wigner_kernel!(output, means, covars, x_points, coeffs, n_states, n_points)
    # Thread and block indices
    point_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if point_idx <= n_points
        result = 0.0
        
        # Extract current point
        x1 = x_points[1, point_idx]
        x2 = x_points[2, point_idx]
        
        # Diagonal terms
        @inbounds for i in 1:n_states
            coeff_i = coeffs[i]
            
            # Extract mean and covariance for state i
            μ1 = means[1, i]
            μ2 = means[2, i]
            
            V11 = covars[1, 1, i]
            V12 = covars[1, 2, i]
            V21 = covars[2, 1, i]
            V22 = covars[2, 2, i]
            
            # Compute Wigner function for state i at point x
            diff1 = x1 - μ1
            diff2 = x2 - μ2
            
            # Inverse of 2x2 covariance matrix
            det_V = V11 * V22 - V12 * V21
            V_inv11 = V22 / det_V
            V_inv12 = -V12 / det_V
            V_inv21 = -V21 / det_V
            V_inv22 = V11 / det_V
            
            # Quadratic form
            quad_form = diff1 * (V_inv11 * diff1 + V_inv12 * diff2) + 
                       diff2 * (V_inv21 * diff1 + V_inv22 * diff2)
            
            # Wigner function value
            normalization = 1.0 / (2π * sqrt(det_V))
            w_i = normalization * exp(-0.5 * quad_form)
            
            result += abs2(coeff_i) * w_i
        end
        
        # Cross terms (interference)
        @inbounds for i in 1:n_states
            @inbounds for j in (i+1):n_states
                coeff_i = coeffs[i]
                coeff_j = coeffs[j]
                
                # Extract means for states i and j
                μ1_i = means[1, i]
                μ2_i = means[2, i]
                μ1_j = means[1, j]
                μ2_j = means[2, j]
                
                # Extract covariances
                V11_i = covars[1, 1, i]
                V12_i = covars[1, 2, i]
                V21_i = covars[2, 1, i]
                V22_i = covars[2, 2, i]
                
                V11_j = covars[1, 1, j]
                V12_j = covars[1, 2, j]
                V21_j = covars[2, 1, j]
                V22_j = covars[2, 2, j]
                
                # Average quantities
                μ1_avg = 0.5 * (μ1_i + μ1_j)
                μ2_avg = 0.5 * (μ2_i + μ2_j)
                
                V11_avg = 0.5 * (V11_i + V11_j)
                V12_avg = 0.5 * (V12_i + V12_j)
                V21_avg = 0.5 * (V21_i + V21_j)
                V22_avg = 0.5 * (V22_i + V22_j)
                
                # Difference from average
                dx1 = x1 - μ1_avg
                dx2 = x2 - μ2_avg
                
                # Phase calculation (symplectic form for QuadPair: [[0,1],[-1,0]])
                Δμ1 = μ1_i - μ1_j
                Δμ2 = μ2_i - μ2_j
                phase_arg = Δμ1 * dx2 - Δμ2 * dx1  # Simplified for 2D case
                
                # Gaussian part
                det_V_avg = V11_avg * V22_avg - V12_avg * V21_avg
                V_avg_inv11 = V22_avg / det_V_avg
                V_avg_inv12 = -V12_avg / det_V_avg
                V_avg_inv21 = -V21_avg / det_V_avg
                V_avg_inv22 = V11_avg / det_V_avg
                
                quad_form_avg = dx1 * (V_avg_inv11 * dx1 + V_avg_inv12 * dx2) + 
                               dx2 * (V_avg_inv21 * dx1 + V_avg_inv22 * dx2)
                
                normalization_cross = 1.0 / (2π * sqrt(det_V_avg))
                gauss_part = normalization_cross * exp(-0.5 * quad_form_avg)
                
                # Cross-Wigner value (real part of complex cross term)
                cross_w_real = gauss_part * cos(phase_arg)
                
                # Add interference term
                interference = 2.0 * real(conj(coeff_i) * coeff_j) * cross_w_real
                result += interference
            end
        end
        
        output[point_idx] = result
    end
    
    return nothing
end

# Optimized kernel for batched Gaussian transformations
function batch_gaussian_transform_kernel!(means_out, covars_out, means_in, covars_in, 
                                        transform_matrices, displacements, n_states)
    # Thread index
    state_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if state_idx <= n_states
        # For single-mode case (2x2 matrices)
        # Mean transformation: μ_out = T * μ_in + d
        T11 = transform_matrices[1, 1, state_idx]
        T12 = transform_matrices[1, 2, state_idx]
        T21 = transform_matrices[2, 1, state_idx]
        T22 = transform_matrices[2, 2, state_idx]
        
        d1 = displacements[1, state_idx]
        d2 = displacements[2, state_idx]
        
        μ1_in = means_in[1, state_idx]
        μ2_in = means_in[2, state_idx]
        
        means_out[1, state_idx] = T11 * μ1_in + T12 * μ2_in + d1
        means_out[2, state_idx] = T21 * μ1_in + T22 * μ2_in + d2
        
        # Covariance transformation: V_out = T * V_in * T^T
        V11_in = covars_in[1, 1, state_idx]
        V12_in = covars_in[1, 2, state_idx]
        V21_in = covars_in[2, 1, state_idx]
        V22_in = covars_in[2, 2, state_idx]
        
        # T * V_in
        TV11 = T11 * V11_in + T12 * V21_in
        TV12 = T11 * V12_in + T12 * V22_in
        TV21 = T21 * V11_in + T22 * V21_in
        TV22 = T21 * V12_in + T22 * V22_in
        
        # (T * V_in) * T^T
        covars_out[1, 1, state_idx] = TV11 * T11 + TV12 * T21
        covars_out[1, 2, state_idx] = TV11 * T12 + TV12 * T22
        covars_out[2, 1, state_idx] = TV21 * T11 + TV22 * T21
        covars_out[2, 2, state_idx] = TV21 * T12 + TV22 * T22
    end
    
    return nothing
end

# Kernel for parallel overlap calculations
function overlap_kernel!(overlaps, means1, covars1, means2, covars2, n_pairs)
    pair_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if pair_idx <= n_pairs
        # Extract means
        μ1_1 = means1[1, pair_idx]
        μ2_1 = means1[2, pair_idx]
        μ1_2 = means2[1, pair_idx]
        μ2_2 = means2[2, pair_idx]
        
        # Extract covariances
        V11_1 = covars1[1, 1, pair_idx]
        V12_1 = covars1[1, 2, pair_idx]
        V21_1 = covars1[2, 1, pair_idx]
        V22_1 = covars1[2, 2, pair_idx]
        
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
        
        # Inverse of sum covariance
        det_V_sum = V_sum11 * V_sum22 - V_sum12 * V_sum21
        V_sum_inv11 = V_sum22 / det_V_sum
        V_sum_inv12 = -V_sum12 / det_V_sum
        V_sum_inv21 = -V_sum21 / det_V_sum
        V_sum_inv22 = V_sum11 / det_V_sum
        
        # Quadratic form for exponential
        quad_form = Δμ1 * (V_sum_inv11 * Δμ1 + V_sum_inv12 * Δμ2) +
                   Δμ2 * (V_sum_inv21 * Δμ1 + V_sum_inv22 * Δμ2)
        
        # Determinants for normalization
        det_V1 = V11_1 * V22_1 - V12_1 * V21_1
        det_V2 = V11_2 * V22_2 - V12_2 * V21_2
        
        # Overlap calculation
        exp_factor = exp(-0.25 * quad_form)
        det_factor = sqrt(det_V1 * det_V2 / det_V_sum)
        
        overlaps[pair_idx] = exp_factor * det_factor
    end
    
    return nothing
end

# High-level interface functions for kernels
function gpu_batch_wigner_evaluation(lc::GaussianLinearCombination, x_points::CuMatrix{Float64})
    n_states = length(lc)
    n_points = size(x_points, 2)
    
    # Prepare data arrays on GPU
    means = CuArray(zeros(Float64, 2, n_states))
    covars = CuArray(zeros(Float64, 2, 2, n_states))
    coeffs = CuArray(Float64[abs(c) for c in lc.coeffs])  # Take absolute values for weights
    
    # Fill data arrays
    for i in 1:n_states
        state = lc.states[i]
        means[:, i] = state.mean
        covars[:, :, i] = state.covar
    end
    
    # Output array
    output = CuArray(zeros(Float64, n_points))
    
    # Launch kernel
    threads_per_block = 256
    blocks = cld(n_points, threads_per_block)
    
    @cuda threads=threads_per_block blocks=blocks wigner_kernel!(
        output, means, covars, x_points, coeffs, n_states, n_points
    )
    
    return Array(output)
end

function gpu_batch_gaussian_transform(states::Vector{GaussianState}, transforms::Vector{<:AbstractMatrix}, displacements::Vector{<:AbstractVector})
    n_states = length(states)
    @assert length(transforms) == n_states
    @assert length(displacements) == n_states
    
    # Prepare input arrays
    means_in = CuArray(zeros(Float64, 2, n_states))
    covars_in = CuArray(zeros(Float64, 2, 2, n_states))
    transform_mats = CuArray(zeros(Float64, 2, 2, n_states))
    disps = CuArray(zeros(Float64, 2, n_states))
    
    for i in 1:n_states
        means_in[:, i] = states[i].mean
        covars_in[:, :, i] = states[i].covar
        transform_mats[:, :, i] = transforms[i]
        disps[:, i] = displacements[i]
    end
    
    # Prepare output arrays
    means_out = CuArray(zeros(Float64, 2, n_states))
    covars_out = CuArray(zeros(Float64, 2, 2, n_states))
    
    # Launch kernel
    threads_per_block = 256
    blocks = cld(n_states, threads_per_block)
    
    @cuda threads=threads_per_block blocks=blocks batch_gaussian_transform_kernel!(
        means_out, covars_out, means_in, covars_in, transform_mats, disps, n_states
    )
    
    # Convert back to GaussianState objects
    result_states = Vector{GaussianState}(undef, n_states)
    for i in 1:n_states
        state = states[i]
        new_mean = Array(means_out[:, i])
        new_covar = Array(covars_out[:, :, i])
        result_states[i] = GaussianState(state.basis, new_mean, new_covar; ħ = state.ħ)
    end
    
    return result_states
end

function gpu_batch_overlap_calculation(states1::Vector{GaussianState}, states2::Vector{GaussianState})
    n_pairs = length(states1)
    @assert length(states2) == n_pairs
    
    # Prepare data arrays
    means1 = CuArray(zeros(Float64, 2, n_pairs))
    covars1 = CuArray(zeros(Float64, 2, 2, n_pairs))
    means2 = CuArray(zeros(Float64, 2, n_pairs))
    covars2 = CuArray(zeros(Float64, 2, 2, n_pairs))
    
    for i in 1:n_pairs
        means1[:, i] = states1[i].mean
        covars1[:, :, i] = states1[i].covar
        means2[:, i] = states2[i].mean
        covars2[:, :, i] = states2[i].covar
    end
    
    # Output array
    overlaps = CuArray(zeros(Float64, n_pairs))
    
    # Launch kernel
    threads_per_block = 256
    blocks = cld(n_pairs, threads_per_block)
    
    @cuda threads=threads_per_block blocks=blocks overlap_kernel!(
        overlaps, means1, covars1, means2, covars2, n_pairs
    )
    
    return Array(overlaps)
end