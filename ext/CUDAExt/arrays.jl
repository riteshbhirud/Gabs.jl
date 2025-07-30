# GPU Array promotion rules for Gabs.jl types

# Vector promotion rules
function Gabs._promote_output_vector(::Type{T1}, ::Type{T2}, vec_out) where {T1<:CuVector, T2<:CuVector}
    T = promote_type(T1, T2)
    return T <: CuVector ? CuArray(vec_out) : T(vec_out)
end

function Gabs._promote_output_vector(::Type{T1}, ::Type{T2}, vec_out) where {T1<:CuVector, T2<:AbstractVector}
    return CuArray(vec_out)
end

function Gabs._promote_output_vector(::Type{T1}, ::Type{T2}, vec_out) where {T1<:AbstractVector, T2<:CuVector}
    return CuArray(vec_out)
end

function Gabs._promote_output_vector(::Type{T}, vec_out, vec_length::Int) where {T<:CuVector}
    return CuArray(vec_out)
end

# Matrix promotion rules
function Gabs._promote_output_matrix(::Type{T1}, ::Type{T2}, mat_out) where {T1<:CuMatrix, T2<:CuMatrix}
    T = promote_type(T1, T2)
    return T <: CuMatrix ? CuArray(mat_out) : T(mat_out)
end

function Gabs._promote_output_matrix(::Type{T1}, ::Type{T2}, mat_out) where {T1<:CuMatrix, T2<:AbstractMatrix}
    return CuArray(mat_out)
end

function Gabs._promote_output_matrix(::Type{T1}, ::Type{T2}, mat_out) where {T1<:AbstractMatrix, T2<:CuMatrix}
    return CuArray(mat_out)
end

function Gabs._promote_output_matrix(::Type{T}, mat_out, out_dim::Int) where {T<:CuMatrix}
    return CuArray(mat_out)
end

function Gabs._promote_output_matrix(::Type{T}, mat_out, out_dim::Tuple) where {T<:CuMatrix}
    return CuArray(mat_out)
end

# Type checking utilities
is_gpu_array(::CuArray) = true
is_gpu_array(::AbstractArray) = false

function ensure_gpu(x::AbstractArray)
    return is_gpu_array(x) ? x : CuArray(x)
end

function ensure_cpu(x::CuArray)
    return Array(x)
end
ensure_cpu(x::AbstractArray) = x

# Memory management helpers
function gpu_similar(x::AbstractArray{T}, dims...) where T
    return CUDA.zeros(T, dims...)
end

function gpu_zeros(T::Type, dims...)
    return CUDA.zeros(T, dims...)
end

function gpu_ones(T::Type, dims...)
    return CUDA.ones(T, dims...)
end

# Check if arrays are compatible for operations
function check_gpu_compatibility(arrays...)
    gpu_count = sum(is_gpu_array, arrays)
    if gpu_count > 0 && gpu_count < length(arrays)
        @warn "Mixing CPU and GPU arrays may impact performance. Consider moving all arrays to the same device."
    end
    return gpu_count > 0
end

# Automatic device selection
function select_compute_device(arrays...)
    return any(is_gpu_array, arrays)
end