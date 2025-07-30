# GPU-accelerated Wigner function calculations

# GPU Wigner function for single Gaussian states
function Gabs.wigner(state::GaussianState{B,M,V}, x::AbstractVector) where {B,M<:CuArray,V<:CuArray}
    length(x) == length(state.mean) || throw(ArgumentError(WIGNER_ERROR))
    
    # Ensure x is on GPU
    x_gpu = ensure_gpu(x)
    
    # Compute on GPU
    V_inv = inv(state.covar)
    diff = x_gpu - state.mean
    arg = -0.5 * dot(diff, V_inv * diff)
    normalization = 1.0 / ((2π)^(state.basis.nmodes) * sqrt(det(state.covar)))
    
    return Array(normalization * exp(arg))[1]  # Return scalar to CPU
end

# GPU Wigner characteristic function for single Gaussian states
function Gabs.wignerchar(state::GaussianState{B,M,V}, xi::AbstractVector) where {B,M<:CuArray,V<:CuArray}
    length(xi) == length(state.mean) || throw(ArgumentError(WIGNER_ERROR))
    
    # Ensure xi is on GPU
    xi_gpu = ensure_gpu(xi)
    
    # Compute symplectic form on GPU
    Omega_cpu = Gabs.symplecticform(state.basis)
    Omega = CuArray(Omega_cpu)
    
    # Compute characteristic function on GPU
    arg1 = -0.5 * dot(xi_gpu, (Omega * state.covar * Omega') * xi_gpu)
    arg2 = 1im * dot((Omega * state.mean), xi_gpu)
    
    result = exp(arg1 - arg2)
    return Array(result)[1]  # Return scalar to CPU
end

# GPU cross-Wigner function between two Gaussian states
function Gabs.cross_wigner(state1::GaussianState{B1,M1,V1}, state2::GaussianState{B2,M2,V2}, x::AbstractVector) where {B1,B2,M1<:CuArray,V1<:CuArray,M2<:CuArray,V2<:CuArray}
    state1.basis == state2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    length(x) == length(state1.mean) || throw(ArgumentError(WIGNER_ERROR))
    
    # Ensure x is on GPU
    x_gpu = ensure_gpu(x)
    
    μ1, μ2 = state1.mean, state2.mean
    V1, V2 = state1.covar, state2.covar
    n = length(μ1) ÷ 2
    ħ = state1.ħ
    
    # Compute average quantities on GPU
    μ_avg = 0.5 * (μ1 + μ2)
    V_avg = 0.5 * (V1 + V2)
    dx = x_gpu - μ_avg
    
    # Compute symplectic form on GPU
    Omega_cpu = Gabs.symplecticform(state1.basis)
    Omega = CuArray(Omega_cpu)
    
    # Phase calculation
    Δμ = μ1 - μ2
    phase_arg = dot(Δμ, Omega * dx) / ħ
    phase = exp(1im * phase_arg)
    
    # Gaussian and normalization
    V_avg_inv = inv(V_avg)
    gauss_arg = -0.5 * dot(dx, V_avg_inv * dx)
    gauss = exp(gauss_arg)
    
    log_norm = -n * log(2π) - 0.5 * logdet(V_avg)
    norm = exp(log_norm)
    
    result = norm * gauss * phase
    return Array(result)[1]  # Return scalar to CPU
end

# Handle mixed CPU/GPU cases for cross_wigner
function Gabs.cross_wigner(state1::GaussianState{B1,M1,V1}, state2::GaussianState{B2,M2,V2}, x::AbstractVector) where {B1,B2,M1<:CuArray,V1<:CuArray,M2<:AbstractArray,V2<:AbstractArray}
    # Promote state2 to GPU
    gpu_state2 = GaussianState(state2.basis, CuArray(state2.mean), CuArray(state2.covar); ħ = state2.ħ)
    return Gabs.cross_wigner(state1, gpu_state2, x)
end

function Gabs.cross_wigner(state1::GaussianState{B1,M1,V1}, state2::GaussianState{B2,M2,V2}, x::AbstractVector) where {B1,B2,M1<:AbstractArray,V1<:AbstractArray,M2<:CuArray,V2<:CuArray}
    # Promote state1 to GPU
    gpu_state1 = GaussianState(state1.basis, CuArray(state1.mean), CuArray(state1.covar); ħ = state1.ħ)
    return Gabs.cross_wigner(gpu_state1, state2, x)
end

# GPU cross-Wigner characteristic function
function Gabs.cross_wignerchar(state1::GaussianState{B1,M1,V1}, state2::GaussianState{B2,M2,V2}, xi::AbstractVector) where {B1,B2,M1<:CuArray,V1<:CuArray,M2<:CuArray,V2<:CuArray}
    state1.basis == state2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    length(xi) == length(state1.mean) || throw(ArgumentError(WIGNER_ERROR))
    
    if state1 === state2
        return Gabs.wignerchar(state1, xi)
    end
    
    # Ensure xi is on GPU
    xi_gpu = ensure_gpu(xi)
    
    μ1, μ2 = state1.mean, state2.mean
    V1, V2 = state1.covar, state2.covar
    
    # Compute average quantities on GPU
    μ12 = 0.5 * (μ1 + μ2)
    V12 = 0.5 * (V1 + V2)
    
    # Compute symplectic form on GPU
    Omega_cpu = Gabs.symplecticform(state1.basis)
    Omega = CuArray(Omega_cpu)
    
    # Compute characteristic function
    temp_mat = Omega * V12 * Omega'
    temp_mat .*= -1
    temp_vec = temp_mat * xi_gpu
    arg1 = -0.5 * dot(xi_gpu, temp_vec)
    
    temp_vec2 = Omega * μ12
    arg2 = 1im * dot(temp_vec2, xi_gpu)
    
    result = exp(arg1 - arg2)
    return Array(result)[1]  # Return scalar to CPU
end

# GPU Wigner function for linear combinations with full interference
function Gabs.wigner(lc::GaussianLinearCombination, x::AbstractVector)
    length(x) == length(lc.states[1].mean) || throw(ArgumentError(WIGNER_ERROR))
    
    # Check if we should compute on GPU
    use_gpu = is_on_gpu(lc) || length(lc) > 10  # Use GPU for large combinations
    
    if use_gpu && CUDA.functional()
        return gpu_wigner_with_interference(lc, x)
    else
        # Fallback to CPU implementation
        return cpu_wigner_with_interference(lc, x)
    end
end

function gpu_wigner_with_interference(lc::GaussianLinearCombination, x::AbstractVector)
    # Ensure everything is on GPU
    gpu_lc = is_on_gpu(lc) ? lc : to_gpu(lc)
    x_gpu = ensure_gpu(x)
    
    result = 0.0
    n = length(gpu_lc)
    
    # Diagonal terms
    @inbounds for i in 1:n
        ci, si = gpu_lc[i]
        wi = Gabs.wigner(si, Array(x_gpu))  # Convert x back to CPU for individual wigner calls
        result += abs2(ci) * wi
    end
    
    # Off-diagonal interference terms
    @inbounds for i in 1:n
        ci, si = gpu_lc[i]
        @inbounds for j in (i+1):n
            cj, sj = gpu_lc[j]
            cross_w = Gabs.cross_wigner(si, sj, Array(x_gpu))
            cross_term = 2 * real(conj(ci) * cj * cross_w)
            result += cross_term
        end
    end
    
    return result
end

function cpu_wigner_with_interference(lc::GaussianLinearCombination, x::AbstractVector)
    # CPU fallback implementation
    result = 0.0
    n = length(lc)
    
    # Diagonal terms
    @inbounds for i in 1:n
        ci, si = lc[i]
        wi = Gabs.wigner(si, x)
        result += abs2(ci) * wi
    end
    
    # Off-diagonal interference terms
    @inbounds for i in 1:n
        ci, si = lc[i]
        @inbounds for j in (i+1):n
            cj, sj = lc[j]
            cross_w = Gabs.cross_wigner(si, sj, x)
            cross_term = 2 * real(conj(ci) * cj * cross_w)
            result += cross_term
        end
    end
    
    return result
end

# GPU Wigner characteristic function for linear combinations
function Gabs.wignerchar(lc::GaussianLinearCombination, xi::AbstractVector)
    length(xi) == length(lc.states[1].mean) || throw(ArgumentError(WIGNER_ERROR))
    
    # Check if we should compute on GPU
    use_gpu = is_on_gpu(lc) || length(lc) > 10
    
    if use_gpu && CUDA.functional()
        return gpu_wignerchar_with_interference(lc, xi)
    else
        return cpu_wignerchar_with_interference(lc, xi)
    end
end

function gpu_wignerchar_with_interference(lc::GaussianLinearCombination, xi::AbstractVector)
    # Ensure everything is on GPU
    gpu_lc = is_on_gpu(lc) ? lc : to_gpu(lc)
    xi_gpu = ensure_gpu(xi)
    
    result = 0.0 + 0.0im
    n = length(gpu_lc)
    
    # Diagonal terms
    @inbounds for i in 1:n
        ci, si = gpu_lc[i]
        chi = Gabs.wignerchar(si, Array(xi_gpu))
        result += abs2(ci) * chi
    end
    
    # Off-diagonal interference terms
    @inbounds for i in 1:n
        ci, si = gpu_lc[i]
        @inbounds for j in (i+1):n
            cj, sj = gpu_lc[j]
            cross_chi = Gabs.cross_wignerchar(si, sj, Array(xi_gpu))
            cross_term = 2 * real(conj(ci) * cj * cross_chi)
            result += cross_term
        end
    end
    
    return result
end

function cpu_wignerchar_with_interference(lc::GaussianLinearCombination, xi::AbstractVector)
    result = 0.0 + 0.0im
    n = length(lc)
    
    # Diagonal terms
    @inbounds for i in 1:n
        ci, si = lc[i]
        chi = Gabs.wignerchar(si, xi)
        result += abs2(ci) * chi
    end
    
    # Off-diagonal interference terms
    @inbounds for i in 1:n
        ci, si = lc[i]
        @inbounds for j in (i+1):n
            cj, sj = lc[j]
            cross_chi = Gabs.cross_wignerchar(si, sj, xi)
            cross_term = 2 * real(conj(ci) * cj * cross_chi)
            result += cross_term
        end
    end
    
    return result
end

# Batched Wigner function evaluation for multiple points
function batch_wigner_evaluation(lc::GaussianLinearCombination, x_points::AbstractMatrix)
    """
    Evaluate Wigner function at multiple points efficiently
    x_points: matrix where each column is a phase space point
    """
    use_gpu = is_on_gpu(lc) || size(x_points, 2) > 100
    
    if use_gpu && CUDA.functional()
        return gpu_batch_wigner(lc, x_points)
    else
        return cpu_batch_wigner(lc, x_points)
    end
end

function gpu_batch_wigner(lc::GaussianLinearCombination, x_points::AbstractMatrix)
    gpu_lc = is_on_gpu(lc) ? lc : to_gpu(lc)
    x_gpu = ensure_gpu(x_points)
    
    n_points = size(x_gpu, 2)
    results = Vector{Float64}(undef, n_points)
    
    # Parallel evaluation over points
    @inbounds for k in 1:n_points
        x_k = view(x_gpu, :, k)
        results[k] = gpu_wigner_with_interference(gpu_lc, Array(x_k))
    end
    
    return results
end

function cpu_batch_wigner(lc::GaussianLinearCombination, x_points::AbstractMatrix)
    n_points = size(x_points, 2)
    results = Vector{Float64}(undef, n_points)
    
    @inbounds for k in 1:n_points
        x_k = view(x_points, :, k)
        results[k] = cpu_wigner_with_interference(lc, x_k)
    end
    
    return results
end

# Grid-based Wigner function evaluation
function wigner_on_grid(lc::GaussianLinearCombination, q_range, p_range; resolution=100)
    """
    Evaluate Wigner function on a 2D grid for single-mode states
    """
    lc.basis.nmodes == 1 || throw(ArgumentError("Grid evaluation only supported for single-mode states"))
    
    q_vals = range(q_range[1], q_range[2], length=resolution)
    p_vals = range(p_range[1], p_range[2], length=resolution)
    
    # Create grid points
    x_points = zeros(2, resolution^2)
    idx = 1
    for q in q_vals
        for p in p_vals
            x_points[1, idx] = q
            x_points[2, idx] = p
            idx += 1
        end
    end
    
    # Evaluate Wigner function
    w_vals = batch_wigner_evaluation(lc, x_points)
    
    # Reshape to grid
    W = reshape(w_vals, resolution, resolution)
    
    return collect(q_vals), collect(p_vals), W
end