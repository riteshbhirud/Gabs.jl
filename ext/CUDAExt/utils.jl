# GPU Array promotion utilities

Base.@propagate_inbounds function _promote_output_vector(::Type{T1}, ::Type{T2}, vec_out) where {T1<:CuVector, T2<:CuVector}
    return CuArray(vec_out)
end

Base.@propagate_inbounds function _promote_output_vector(::Type{T1}, ::Type{T2}, vec_out) where {T1<:CuVector, T2<:Vector}
    return CuArray(vec_out)
end

Base.@propagate_inbounds function _promote_output_vector(::Type{T1}, ::Type{T2}, vec_out) where {T1<:Vector, T2<:CuVector}
    return CuArray(vec_out)
end

Base.@propagate_inbounds function _promote_output_vector(::Type{<:CuVector}, vec_out, vec_length::Int)
    return CuArray(vec_out)
end

Base.@propagate_inbounds function _promote_output_matrix(::Type{T1}, ::Type{T2}, mat_out) where {T1<:CuMatrix, T2<:CuMatrix}
    return CuArray(mat_out)
end

Base.@propagate_inbounds function _promote_output_matrix(::Type{T1}, ::Type{T2}, mat_out) where {T1<:CuMatrix, T2<:Matrix}
    return CuArray(mat_out)
end

Base.@propagate_inbounds function _promote_output_matrix(::Type{T1}, ::Type{T2}, mat_out) where {T1<:Matrix, T2<:CuMatrix}
    return CuArray(mat_out)
end

Base.@propagate_inbounds function _promote_output_matrix(::Type{<:CuMatrix}, mat_out, out_dim::Int)
    return CuArray(mat_out)
end

Base.@propagate_inbounds function _promote_output_matrix(::Type{<:CuMatrix}, mat_out, out_dim::Tuple)
    return CuArray(mat_out)
end

"""
    ensure_gpu_array(x::AbstractArray)

Convert array to GPU array if CUDA is available, otherwise return unchanged.
"""
function ensure_gpu_array(x::AbstractArray)
    if CUDA_AVAILABLE && !(x isa CuArray)
        return CuArray(x)
    end
    return x
end

"""
    ensure_gpu_compatibility(x::AbstractArray, target_type::Type{<:CuArray})

Ensure array is GPU-compatible with target type.
"""
function ensure_gpu_compatibility(x::AbstractArray, target_type::Type{<:CuArray})
    if CUDA_AVAILABLE
        return CuArray(x)
    else
        # Fallback to CPU
        return Array(x)
    end
end

"""
    gpu_fallback_warning()

Issue warning when falling back to CPU due to CUDA unavailability.
"""
function gpu_fallback_warning()
    if !CUDA_AVAILABLE
        @warn "CUDA not available. Falling back to CPU computation. Install CUDA.jl and ensure GPU drivers are properly configured for GPU acceleration."
    end
end