"""
    wigner_kernel!(output, x_points, mean, covar_inv, det_covar, nmodes)

CUDA kernel for computing Wigner function values at multiple phase space points.
"""
function wigner_kernel!(output::CuDeviceVector{T}, x_points::CuDeviceMatrix{T}, 
                       mean::CuDeviceVector{T}, covar_inv::CuDeviceMatrix{T}, 
                       det_covar::T, nmodes::Int) where T
    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= size(x_points, 2)
        diff_sum = zero(T)
        
        for i in 1:length(mean)
            diff_i = x_points[i, idx] - mean[i]
            temp_sum = zero(T)
            
            for j in 1:length(mean)
                diff_j = x_points[j, idx] - mean[j]
                temp_sum += covar_inv[i, j] * diff_j
            end
            
            diff_sum += diff_i * temp_sum
        end
        

        normalization = (2 * T(π))^nmodes * sqrt(det_covar)
        exponent = -0.5 * diff_sum
        
        output[idx] = exp(exponent) / normalization
    end
    
    return nothing
end

"""
    wignerchar_kernel!(output, xi_points, mean, covar, omega, nmodes, hbar)

CUDA kernel for computing Wigner characteristic function values at multiple points.
"""
function wignerchar_kernel!(output::CuDeviceVector{Complex{T}}, xi_points::CuDeviceMatrix{T},
                                 precomputed_quadratic::CuDeviceMatrix{T}, precomputed_linear::CuDeviceVector{T}) where T
    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= size(xi_points, 2)
        quadratic_term = zero(T)
        for i in 1:size(xi_points, 1)
            for j in 1:size(xi_points, 1)
                quadratic_term += xi_points[i, idx] * precomputed_quadratic[i, j] * xi_points[j, idx]
            end
        end
        
        linear_term = zero(T)
        for i in 1:size(xi_points, 1)
            linear_term += precomputed_linear[i] * xi_points[i, idx]
        end
        

        real_part = -0.5 * quadratic_term
        imag_part = linear_term
        
        output[idx] = Complex{T}(cos(imag_part), sin(imag_part)) * exp(real_part)
    end
    
    return nothing
end

function Gabs.wigner(state::GaussianState{B,M,V}, x::AbstractVector) where {B<:SymplecticBasis, M<:CuArray, V<:CuArray}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar); ħ = state.ħ)
        return Gabs.wigner(cpu_state, x)
    end
    
    basis = state.basis
    nmodes = basis.nmodes
    mean = state.mean
    covar = state.covar
    
    length(mean) == length(x) || throw(ArgumentError(WIGNER_ERROR))
    
    T = eltype(mean)
    x_gpu = CuArray(T.(x))
    
    diff = x_gpu .- mean
    covar_inv = inv(covar)
    
    quad_form = dot(diff, covar_inv * diff)
    
    det_covar = det(covar)
    normalization = (2 * T(π))^nmodes * sqrt(det_covar)
    
    result = exp(-0.5 * quad_form) / normalization
    
    return result  
end

function Gabs.wigner(state::GaussianState{B,M,V}, x_points::CuMatrix{T}) where {B<:SymplecticBasis, M<:CuArray, V<:CuArray, T}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar); ħ = state.ħ)
        return [Gabs.wigner(cpu_state, Array(x_points[:, i])) for i in 1:size(x_points, 2)]
    end
    
    basis = state.basis
    nmodes = basis.nmodes
    mean = state.mean
    covar = state.covar
    
    size(x_points, 1) == length(mean) || throw(ArgumentError(WIGNER_ERROR))
    
    num_points = size(x_points, 2)
    
    output = CUDA.zeros(T, num_points)
    covar_inv = inv(covar)
    det_covar = det(covar)
    
    threads_per_block = min(256, num_points)
    num_blocks = cld(num_points, threads_per_block)
    
    @cuda threads=threads_per_block blocks=num_blocks wigner_kernel!(
        output, x_points, mean, covar_inv, det_covar, nmodes
    )
    
    return output
end

function Gabs.wignerchar(state::GaussianState{B,M,V}, xi::AbstractVector) where {B<:SymplecticBasis, M<:CuArray, V<:CuArray}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar); ħ = state.ħ)
        return Gabs.wignerchar(cpu_state, xi)
    end
    
    basis = state.basis
    nmodes = basis.nmodes
    mean = state.mean
    covar = state.covar
    
    length(mean) == length(xi) || throw(ArgumentError(WIGNER_ERROR))
    
    T = eltype(mean)
    xi_gpu = CuArray(T.(xi))
    
    omega = CuArray(symplecticform(basis))
    
    temp1 = omega * covar
    temp2 = temp1 * transpose(omega)
    quadratic_term = dot(xi_gpu, temp2 * xi_gpu)
    
    omega_mean = omega * mean
    linear_term = dot(omega_mean, xi_gpu)
    
    real_part = -0.5 * quadratic_term  
    imag_part = linear_term
    
    result = Complex{T}(cos(imag_part), sin(imag_part)) * exp(real_part)
    
    return result  
end

function Gabs.wignerchar(state::GaussianState{B,M,V}, xi_points::CuMatrix{T}) where {B<:SymplecticBasis, M<:CuArray, V<:CuArray, T}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar); ħ = state.ħ)
        return [Gabs.wignerchar(cpu_state, Array(xi_points[:, i])) for i in 1:size(xi_points, 2)]
    end
    
    basis = state.basis
    nmodes = basis.nmodes
    mean = state.mean
    covar = state.covar
    
    size(xi_points, 1) == length(mean) || throw(ArgumentError(WIGNER_ERROR))
    
    num_points = size(xi_points, 2)
    
    output = CUDA.zeros(Complex{T}, num_points)
    
    omega = CuArray{T}(symplecticform(basis))
    
    temp_matrix = omega * covar * transpose(omega)
    precomputed_quadratic = CuArray{T}(temp_matrix)
    
    precomputed_linear = CuArray{T}(omega * mean)
    
    threads_per_block = min(256, num_points)
    num_blocks = cld(num_points, threads_per_block)
    
    @cuda threads=threads_per_block blocks=num_blocks wignerchar_kernel!(
        output, xi_points, precomputed_quadratic, precomputed_linear
    )
    
    return output
end

"""
    wigner_grid(state::GaussianState, x_range, p_range, nx::Int, np::Int)

Create a phase space grid for Wigner function evaluation.
Returns points as a 2×(nx*np) CuMatrix for efficient GPU evaluation.
"""
function wigner_grid(state::GaussianState{B,M,V}, x_range, p_range, nx::Int, np::Int) where {B<:SymplecticBasis, M<:CuArray, V<:CuArray}
    T = eltype(M)
    
    x_vals = range(x_range[1], x_range[2], length=nx)
    p_vals = range(p_range[1], p_range[2], length=np)
    
    total_points = nx * np
    points = CUDA.zeros(T, 2, total_points)
    
    for i in 1:nx
        for j in 1:np
            idx = (i-1) * np + j
            points[1, idx] = T(x_vals[i])
            points[2, idx] = T(p_vals[j])
        end
    end
    
    return points
end