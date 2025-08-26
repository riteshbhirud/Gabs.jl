# GPU utility functions and array promotion rules

"""
    cuda_available()

Check if CUDA is available and functional.
"""
cuda_available() = CUDA_AVAILABLE

"""
    ensure_gpu_array(arr, T=eltype(arr))

Convert array to GPU array with specified element type, or return CPU array if CUDA not available.
"""
function ensure_gpu_array(arr::AbstractArray, T::Type=eltype(arr))
    if cuda_available()
        return CuArray{T}(arr)
    else
        @warn "CUDA not available, keeping array on CPU" maxlog=1
        return Array{T}(arr)
    end
end

"""
    ensure_gpu_array(arr::CuArray, T=eltype(arr))

Keep CuArray as-is or convert element type if needed.
"""
function ensure_gpu_array(arr::CuArray, T::Type=eltype(arr))
    if T == eltype(arr)
        return arr
    else
        return CuArray{T}(arr)
    end
end

# Override array promotion utilities for CuArrays
function Gabs._promote_output_vector(::Type{<:CuVector{T}}, vec_out, vec_length::Int) where {T}
    if cuda_available()
        return CuArray{T}(vec_out)
    else
        @warn "CUDA not available, falling back to CPU array" maxlog=1
        return Vector{T}(vec_out)
    end
end

function Gabs._promote_output_vector(::Type{<:CuVector{T}}, ::Type{S}, vec_out) where {T,S}
    if cuda_available()
        return CuArray{T}(vec_out)
    else
        @warn "CUDA not available, falling back to CPU array" maxlog=1
        return Vector{T}(vec_out)
    end
end

function Gabs._promote_output_matrix(::Type{<:CuMatrix{T}}, mat_out, out_dim) where {T}
    if cuda_available()
        return CuArray{T}(mat_out)
    else
        @warn "CUDA not available, falling back to CPU array" maxlog=1
        return Matrix{T}(mat_out)
    end
end

function Gabs._promote_output_matrix(::Type{<:CuMatrix{T}}, ::Type{S}, mat_out) where {T,S}
    if cuda_available()
        return CuArray{T}(mat_out)
    else
        @warn "CUDA not available, falling back to CPU array" maxlog=1
        return Matrix{T}(mat_out)
    end
end

"""
    gpu_zeros(T, dims...)

Create zeros array on GPU if available, otherwise on CPU.
"""
function gpu_zeros(T::Type, dims...)
    if cuda_available()
        return CUDA.zeros(T, dims...)
    else
        return zeros(T, dims...)
    end
end

"""
    gpu_fill(val, dims...)

Create filled array on GPU if available, otherwise on CPU.
"""
function gpu_fill(val, dims...)
    if cuda_available()
        return CUDA.fill(val, dims...)
    else
        return fill(val, dims...)
    end
end

"""
    gpu_identity(T, n)

Create identity matrix on GPU if available, otherwise on CPU.
"""
function gpu_identity(T::Type, n::Int)
    if cuda_available()
        return CuArray{T}(I, n, n)
    else
        return Matrix{T}(I, n, n)
    end
end

"""
    safe_gpu_operation(f, args...; fallback_warn=true)

Safely execute GPU operation with fallback to CPU if needed.
"""
function safe_gpu_operation(f::Function, args...; fallback_warn=true)
    if !cuda_available()
        fallback_warn && @warn "CUDA not available, falling back to CPU operation" maxlog=1
        # Convert GPU arrays to CPU arrays for fallback
        cpu_args = map(arg -> arg isa CuArray ? Array(arg) : arg, args)
        return f(cpu_args...)
    end
    
    try
        return f(args...)
    catch e
        if e isa CUDA.OutOfGPUMemoryError
            @warn "GPU out of memory, falling back to CPU operation"
            # Convert GPU arrays to CPU arrays for fallback
            cpu_args = map(arg -> arg isa CuArray ? Array(arg) : arg, args)
            return f(cpu_args...)
        else
            rethrow(e)
        end
    end
end