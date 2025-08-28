# Professional GPU API - replace src/gpu_convenience.jl with this

# Device transfer functions (Flux.jl style)
"""
    gpu(x)

Transfer data to GPU. Works with states, operators, and arrays.

# Examples
```julia
state = vacuumstate(basis) |> gpu
operator = displace(basis, 1.0) |> gpu
basis = QuadPairBasis(1) |> gpu
```
"""
function gpu(state::GaussianState; precision=Float32)
    gpu_mean = CuArray{precision}(state.mean)
    gpu_covar = CuArray{precision}(state.covar)
    return GaussianState(state.basis, gpu_mean, gpu_covar; ħ = state.ħ)
end

function gpu(op::GaussianUnitary; precision=Float32)
    gpu_disp = CuArray{precision}(op.disp)
    gpu_symplectic = CuArray{precision}(op.symplectic)
    return GaussianUnitary(op.basis, gpu_disp, gpu_symplectic; ħ = op.ħ)
end

function gpu(op::GaussianChannel; precision=Float32)
    gpu_disp = CuArray{precision}(op.disp)
    gpu_transform = CuArray{precision}(op.transform)
    gpu_noise = CuArray{precision}(op.noise)
    return GaussianChannel(op.basis, gpu_disp, gpu_transform, gpu_noise; ħ = op.ħ)
end

function gpu(lc::GaussianLinearCombination; precision=Float32)
    gpu_coeffs = CuArray{precision}(lc.coeffs)
    gpu_states = [gpu(state; precision=precision) for state in lc.states]
    return GaussianLinearCombination(lc.basis, gpu_coeffs, gpu_states)
end

function gpu(basis::SymplecticBasis)
    # Mark basis as GPU-preferred (could store in basis metadata)
    return basis  # Basis itself doesn't need GPU arrays
end

function gpu(x::AbstractArray; precision=Float32)
    return CuArray{precision}(x)
end

"""
    cpu(x)

Transfer data to CPU. Works with states, operators, and arrays.

# Examples  
```julia
state_cpu = gpu_state |> cpu
operator_cpu = gpu_operator |> cpu
```
"""
function cpu(state::GaussianState)
    cpu_mean = Array(state.mean)
    cpu_covar = Array(state.covar)
    return GaussianState(state.basis, cpu_mean, cpu_covar; ħ = state.ħ)
end

function cpu(op::GaussianUnitary)
    cpu_disp = Array(op.disp)
    cpu_symplectic = Array(op.symplectic)
    return GaussianUnitary(op.basis, cpu_disp, cpu_symplectic; ħ = op.ħ)
end

function cpu(op::GaussianChannel)
    cpu_disp = Array(op.disp)
    cpu_transform = Array(op.transform)
    cpu_noise = Array(op.noise)
    return GaussianChannel(op.basis, cpu_disp, cpu_transform, cpu_noise; ħ = op.ħ)
end

function cpu(lc::GaussianLinearCombination)
    cpu_coeffs = Array(lc.coeffs)
    cpu_states = [cpu(state) for state in lc.states]
    return GaussianLinearCombination(lc.basis, cpu_coeffs, cpu_states)
end

function cpu(x::AbstractArray)
    return Array(x)
end

# Automatic dispatch based on input types
"""
Smart dispatch: automatically use GPU when inputs are GPU arrays.
"""

# States - automatic GPU detection
function coherentstate(basis, α::CuArray; ħ=2)
    T = real(eltype(α))
    return coherentstate(CuVector{T}, CuMatrix{T}, basis, Array(α); ħ=ħ)
end

function squeezedstate(basis, r::CuArray, θ; ħ=2) 
    T = eltype(r)
    return squeezedstate(CuVector{T}, CuMatrix{T}, basis, Array(r), θ; ħ=ħ)
end

function thermalstate(basis, n::CuArray; ħ=2)
    T = eltype(n)
    return thermalstate(CuVector{T}, CuMatrix{T}, basis, Array(n); ħ=ħ)
end

# Operations - automatic GPU detection  
function displace(basis, α::CuArray; ħ=2)
    T = real(eltype(α))
    return displace(CuVector{T}, CuMatrix{T}, basis, Array(α); ħ=ħ)
end

function squeeze(basis, r::CuArray, θ; ħ=2)
    T = eltype(r) 
    return squeeze(CuVector{T}, CuMatrix{T}, basis, Array(r), θ; ħ=ħ)
end

# Device detection helpers
"""
    device(x)

Detect device of arrays/objects.
"""
device(x::CuArray) = :gpu
device(x::Array) = :cpu
device(state::GaussianState) = device(state.mean)
device(op::GaussianUnitary) = device(op.disp)
device(op::GaussianChannel) = device(op.disp)

"""
    adapt_device(target, source)

Create target object on same device as source.
"""
function adapt_device(target_constructor, source_obj, args...)
    if device(source_obj) == :gpu
        T = eltype(source_obj isa GaussianState ? source_obj.mean : 
                  source_obj isa GaussianUnitary ? source_obj.disp : source_obj.disp)
        precision = real(T)
        return target_constructor(CuVector{precision}, CuMatrix{precision}, args...)
    else
        return target_constructor(args...)
    end
end

# Mixed device operations - automatic promotion
function Base.:*(op::GaussianUnitary, state::GaussianState)
    # If devices don't match, promote to GPU
    if device(op) != device(state)
        if device(op) == :gpu
            state = gpu(state)
        elseif device(state) == :gpu  
            op = gpu(op)
        end
    end
    
    # Use existing multiplication
    return invoke(*, Tuple{GaussianUnitary, GaussianState}, op, state)
end