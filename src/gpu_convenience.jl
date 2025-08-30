# src/gpu_convenience.jl - Fixed Conditional GPU API

"""
    gpu(x)

Transfer data to GPU. Works with states, operators, and arrays.
Requires CUDA.jl to be loaded.
"""
function gpu(state::GaussianState; precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _gpu_impl(state, precision)
end

function gpu(op::GaussianUnitary; precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _gpu_impl(op, precision)
end

function gpu(op::GaussianChannel; precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _gpu_impl(op, precision)
end

function gpu(lc::GaussianLinearCombination; precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _gpu_impl(lc, precision)
end


# FIX: Remove precision parameter from basis version
function gpu(basis::SymplecticBasis)
    return basis  # Basis itself doesn't need GPU arrays
end

function gpu(x::AbstractArray; precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _gpu_impl(x, precision)
end

"""
    cpu(x)

Transfer data to CPU. Works with states, operators, and arrays.
"""
function cpu(state::GaussianState)
    cpu_mean = isa(state.mean, AbstractArray) ? Array(state.mean) : state.mean
    cpu_covar = isa(state.covar, AbstractArray) ? Array(state.covar) : state.covar
    return GaussianState(state.basis, cpu_mean, cpu_covar; ħ = state.ħ)
end

function cpu(op::GaussianUnitary)
    cpu_disp = isa(op.disp, AbstractArray) ? Array(op.disp) : op.disp
    cpu_symplectic = isa(op.symplectic, AbstractArray) ? Array(op.symplectic) : op.symplectic
    return GaussianUnitary(op.basis, cpu_disp, cpu_symplectic; ħ = op.ħ)
end

function cpu(op::GaussianChannel)
    cpu_disp = isa(op.disp, AbstractArray) ? Array(op.disp) : op.disp
    cpu_transform = isa(op.transform, AbstractArray) ? Array(op.transform) : op.transform
    cpu_noise = isa(op.noise, AbstractArray) ? Array(op.noise) : op.noise
    return GaussianChannel(op.basis, cpu_disp, cpu_transform, cpu_noise; ħ = op.ħ)
end

function cpu(lc::GaussianLinearCombination)
    cpu_coeffs = isa(lc.coeffs, AbstractArray) ? Array(lc.coeffs) : lc.coeffs
    cpu_states = [cpu(state) for state in lc.states]
    return GaussianLinearCombination(lc.basis, cpu_coeffs, cpu_states)
end

function cpu(x::AbstractArray)
    return Array(x)
end

"""
    device(x)

Detect device of arrays/objects.
"""
device(x::Array) = :cpu
device(state::GaussianState) = _detect_device(state.mean)
device(op::GaussianUnitary) = _detect_device(op.disp)
device(op::GaussianChannel) = _detect_device(op.disp)
device(lc::GaussianLinearCombination) = _detect_device(lc)
function _detect_device(x)
    return :cpu  # Will be extended by CUDAExt for CuArrays
end

"""
    adapt_device(target_constructor, source_obj, args...)

Create target object on same device as source.
"""
function adapt_device(target_constructor, source_obj, args...)
    if _detect_device(source_obj) == :gpu
        return _adapt_device_gpu(target_constructor, source_obj, args...)
    else
        return target_constructor(args...)
    end
end

function _gpu_impl end
function _adapt_device_gpu end