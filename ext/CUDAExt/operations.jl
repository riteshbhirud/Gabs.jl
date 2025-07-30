# GPU-enabled Gaussian operations (unitaries and channels)

# GPU Gaussian Unitary constructors
function Gabs.displace(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {Td<:CuVector,Ts<:CuMatrix,N<:Int,A}
    nmodes = basis.nmodes
    
    if alpha isa Number
        if basis isa QuadPairBasis
            disp_cpu = repeat([sqrt(2*ħ) * real(alpha), sqrt(2*ħ) * imag(alpha)], nmodes)
        else # QuadBlockBasis
            disp_cpu = repeat([sqrt(2*ħ) * real(alpha), sqrt(2*ħ) * imag(alpha)], inner = nmodes)
        end
        disp = CuArray(disp_cpu)
        symplectic_cpu = Matrix{real(A)}(I, 2*nmodes, 2*nmodes)
        symplectic = CuArray(symplectic_cpu)
    else # Vector of alphas
        Rt = real(eltype(A))
        if basis isa QuadPairBasis
            disp_cpu = sqrt(2*ħ) * reinterpret(Rt, alpha)
        else # QuadBlockBasis
            re = reinterpret(Rt, alpha)
            disp_cpu = vcat(view(re, 1:2:length(re)), view(re, 2:2:length(re)))
            disp_cpu .*= sqrt(2*ħ)
        end
        disp = CuArray(disp_cpu)
        symplectic_cpu = Matrix{Rt}(I, 2*nmodes, 2*nmodes)
        symplectic = CuArray(symplectic_cpu)
    end
    
    return GaussianUnitary(basis, disp, symplectic; ħ = ħ)
end

function Gabs.displace(::Type{T}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {T<:CuArray,N<:Int,A}
    return displace(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis, alpha; ħ = ħ)
end

function Gabs.squeeze(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {Td<:CuVector,Ts<:CuMatrix,N<:Int,R}
    nmodes = basis.nmodes
    disp = CUDA.zeros(real(R), 2*nmodes)
    symplectic_cpu = zeros(real(R), 2*nmodes, 2*nmodes)
    
    if r isa Real && theta isa Real
        cr, sr = cosh(r), sinh(r)
        ct, st = cos(theta), sin(theta)
        
        if basis isa QuadPairBasis
            for i in Base.OneTo(nmodes)
                symplectic_cpu[2*i-1, 2*i-1] = cr - sr*ct
                symplectic_cpu[2*i-1, 2*i] = -sr * st
                symplectic_cpu[2*i, 2*i-1] = -sr * st
                symplectic_cpu[2*i, 2*i] = cr + sr*ct
            end
        else # QuadBlockBasis
            for i in Base.OneTo(nmodes)
                symplectic_cpu[i, i] = cr - sr*ct
                symplectic_cpu[i, i+nmodes] = -sr * st
                symplectic_cpu[i+nmodes, i] = -sr * st
                symplectic_cpu[i+nmodes, i+nmodes] = cr + sr*ct
            end
        end
    else # Vector parameters
        Rt = eltype(R)
        if basis isa QuadPairBasis
            for i in Base.OneTo(nmodes)
                cr, sr = cosh(r[i]), sinh(r[i])
                ct, st = cos(theta[i]), sin(theta[i])
                symplectic_cpu[2*i-1, 2*i-1] = cr - sr*ct
                symplectic_cpu[2*i-1, 2*i] = -sr * st
                symplectic_cpu[2*i, 2*i-1] = -sr * st
                symplectic_cpu[2*i, 2*i] = cr + sr*ct
            end
        else # QuadBlockBasis
            for i in Base.OneTo(nmodes)
                cr, sr = cosh(r[i]), sinh(r[i])
                ct, st = cos(theta[i]), sin(theta[i])
                symplectic_cpu[i, i] = cr - sr*ct
                symplectic_cpu[i, i+nmodes] = -sr * st
                symplectic_cpu[i+nmodes, i] = -sr * st
                symplectic_cpu[i+nmodes, i+nmodes] = cr + sr*ct
            end
        end
    end
    
    symplectic = CuArray(symplectic_cpu)
    return GaussianUnitary(basis, disp, symplectic; ħ = ħ)
end

function Gabs.squeeze(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {T<:CuArray,N<:Int,R}
    return squeeze(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis, r, theta; ħ = ħ)
end

function Gabs.phaseshift(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, theta::R; ħ = 2) where {Td<:CuVector,Ts<:CuMatrix,N<:Int,R}
    nmodes = basis.nmodes
    disp = CUDA.zeros(real(R), 2*nmodes)
    symplectic_cpu = zeros(real(R), 2*nmodes, 2*nmodes)
    
    if theta isa Real
        ct, st = cos(theta), sin(theta)
        if basis isa QuadPairBasis
            for i in Base.OneTo(nmodes)
                symplectic_cpu[2*i-1, 2*i-1] = ct
                symplectic_cpu[2*i-1, 2*i] = st
                symplectic_cpu[2*i, 2*i-1] = -st
                symplectic_cpu[2*i, 2*i] = ct
            end
        else # QuadBlockBasis
            for i in Base.OneTo(nmodes)
                symplectic_cpu[i, i] = ct
                symplectic_cpu[i, i+nmodes] = st
                symplectic_cpu[i+nmodes, i] = -st
                symplectic_cpu[i+nmodes, i+nmodes] = ct
            end
        end
    else # Vector parameters
        Rt = eltype(R)
        if basis isa QuadPairBasis
            for i in Base.OneTo(nmodes)
                ct, st = cos(theta[i]), sin(theta[i])
                symplectic_cpu[2*i-1, 2*i-1] = ct
                symplectic_cpu[2*i-1, 2*i] = st
                symplectic_cpu[2*i, 2*i-1] = -st
                symplectic_cpu[2*i, 2*i] = ct
            end
        else # QuadBlockBasis
            for i in Base.OneTo(nmodes)
                ct, st = cos(theta[i]), sin(theta[i])
                symplectic_cpu[i, i] = ct
                symplectic_cpu[i, i+nmodes] = st
                symplectic_cpu[i+nmodes, i] = -st
                symplectic_cpu[i+nmodes, i+nmodes] = ct
            end
        end
    end
    
    symplectic = CuArray(symplectic_cpu)
    return GaussianUnitary(basis, disp, symplectic; ħ = ħ)
end

function Gabs.phaseshift(::Type{T}, basis::SymplecticBasis{N}, theta::R; ħ = 2) where {T<:CuArray,N<:Int,R}
    return phaseshift(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis, theta; ħ = ħ)
end

# GPU Gaussian Channel constructors
function Gabs.attenuator(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}, theta::R, n::M; ħ = 2) where {Td<:CuVector,Tt<:CuMatrix,N<:Int,R,M}
    nmodes = basis.nmodes
    
    if theta isa Real && n isa Real
        disp = CUDA.zeros(real(R), 2*nmodes)
        transform_cpu = Matrix{real(R)}(cos(theta) * I, 2*nmodes, 2*nmodes)
        noise_cpu = Matrix{real(R)}((sin(theta))^2 * n * I, 2*nmodes, 2*nmodes)
        transform = CuArray(transform_cpu)
        noise = CuArray(noise_cpu)
    else # Vector parameters
        Rt = real(eltype(R))
        disp = CUDA.zeros(Rt, 2*nmodes)
        transform_cpu = zeros(Rt, 2*nmodes, 2*nmodes)
        noise_cpu = zeros(Rt, 2*nmodes, 2*nmodes)
        
        if basis isa QuadPairBasis
            for i in Base.OneTo(nmodes)
                ct, st = cos(theta[i]), sin(theta[i])
                ni = n[i]
                
                transform_cpu[2*i-1, 2*i-1] = ct
                transform_cpu[2*i, 2*i] = ct
                
                noise_cpu[2*i-1, 2*i-1] = st^2 * ni
                noise_cpu[2*i, 2*i] = st^2 * ni
            end
        else # QuadBlockBasis
            for i in Base.OneTo(nmodes)
                ct, st = cos(theta[i]), sin(theta[i])
                ni = n[i]
                
                transform_cpu[i, i] = ct
                transform_cpu[i+nmodes, i+nmodes] = ct
                
                noise_cpu[i, i] = st^2 * ni
                noise_cpu[i+nmodes, i+nmodes] = st^2 * ni
            end
        end
        transform = CuArray(transform_cpu)
        noise = CuArray(noise_cpu)
    end
    
    return GaussianChannel(basis, disp, transform, noise; ħ = ħ)
end

function Gabs.attenuator(::Type{T}, basis::SymplecticBasis{N}, theta::R, n::M; ħ = 2) where {T<:CuArray,N<:Int,R,M}
    return attenuator(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis, theta, n; ħ = ħ)
end

function Gabs.amplifier(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}, r::R, n::M; ħ = 2) where {Td<:CuVector,Tt<:CuMatrix,N<:Int,R,M}
    nmodes = basis.nmodes
    
    if r isa Real && n isa Real
        disp = CUDA.zeros(real(R), 2*nmodes)
        transform_cpu = Matrix{real(R)}(cosh(r) * I, 2*nmodes, 2*nmodes)
        noise_cpu = Matrix{real(R)}((sinh(r))^2 * n * I, 2*nmodes, 2*nmodes)
        transform = CuArray(transform_cpu)
        noise = CuArray(noise_cpu)
    else # Vector parameters
        Rt = real(eltype(R))
        disp = CUDA.zeros(Rt, 2*nmodes)
        transform_cpu = zeros(Rt, 2*nmodes, 2*nmodes)
        noise_cpu = zeros(Rt, 2*nmodes, 2*nmodes)
        
        if basis isa QuadPairBasis
            for i in Base.OneTo(nmodes)
                cr, sr = cosh(r[i]), sinh(r[i])
                ni = n[i]
                
                transform_cpu[2*i-1, 2*i-1] = cr
                transform_cpu[2*i, 2*i] = cr
                
                noise_cpu[2*i-1, 2*i-1] = sr^2 * ni
                noise_cpu[2*i, 2*i] = sr^2 * ni
            end
        else # QuadBlockBasis
            for i in Base.OneTo(nmodes)
                cr, sr = cosh(r[i]), sinh(r[i])
                ni = n[i]
                
                transform_cpu[i, i] = cr
                transform_cpu[i+nmodes, i+nmodes] = cr
                
                noise_cpu[i, i] = sr^2 * ni
                noise_cpu[i+nmodes, i+nmodes] = sr^2 * ni
            end
        end
        transform = CuArray(transform_cpu)
        noise = CuArray(noise_cpu)
    end
    
    return GaussianChannel(basis, disp, transform, noise; ħ = ħ)
end

function Gabs.amplifier(::Type{T}, basis::SymplecticBasis{N}, r::R, n::M; ħ = 2) where {T<:CuArray,N<:Int,R,M}
    return amplifier(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis, r, n; ħ = ħ)
end

# GPU Gaussian state/unitary/channel applications
function Base.:(*)(op::GaussianUnitary{B,D,S}, state::GaussianState{B,M,V}) where {B,D<:CuArray,S<:CuArray,M<:CuArray,V<:CuArray}
    op.basis == state.basis || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    
    d, S_op = op.disp, op.symplectic
    mean′ = S_op * state.mean + d
    covar′ = S_op * state.covar * S_op'
    
    return GaussianState(state.basis, mean′, covar′; ħ = state.ħ)
end

function Base.:(*)(op::GaussianChannel{B,D,T}, state::GaussianState{B,M,V}) where {B,D<:CuArray,T<:CuArray,M<:CuArray,V<:CuArray}
    op.basis == state.basis || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    
    d, T_op, N = op.disp, op.transform, op.noise
    mean′ = T_op * state.mean + d
    covar′ = T_op * state.covar * T_op' + N
    
    return GaussianState(state.basis, mean′, covar′; ħ = state.ħ)
end

# Automatic GPU promotion for mixed CPU/GPU operations
function Base.:(*)(op::GaussianUnitary{B,D,S}, state::GaussianState{B,M,V}) where {B,D<:CuArray,S<:CuArray,M<:AbstractArray,V<:AbstractArray}
    # Promote state to GPU
    gpu_state = GaussianState(state.basis, CuArray(state.mean), CuArray(state.covar); ħ = state.ħ)
    return op * gpu_state
end

function Base.:(*)(op::GaussianUnitary{B,D,S}, state::GaussianState{B,M,V}) where {B,D<:AbstractArray,S<:AbstractArray,M<:CuArray,V<:CuArray}
    # Promote operator to GPU
    gpu_op = GaussianUnitary(op.basis, CuArray(op.disp), CuArray(op.symplectic); ħ = op.ħ)
    return gpu_op * state
end

function Base.:(*)(op::GaussianChannel{B,D,T}, state::GaussianState{B,M,V}) where {B,D<:CuArray,T<:CuArray,M<:AbstractArray,V<:AbstractArray}
    # Promote state to GPU
    gpu_state = GaussianState(state.basis, CuArray(state.mean), CuArray(state.covar); ħ = state.ħ)
    return op * gpu_state
end

function Base.:(*)(op::GaussianChannel{B,D,T}, state::GaussianState{B,M,V}) where {B,D<:AbstractArray,T<:AbstractArray,M<:CuArray,V<:CuArray}
    # Promote operator to GPU
    gpu_op = GaussianChannel(op.basis, CuArray(op.disp), CuArray(op.transform), CuArray(op.noise); ħ = op.ħ)
    return gpu_op * state
end

# In-place GPU operations for efficiency
function Gabs.apply!(state::GaussianState{B,M,V}, op::GaussianUnitary{B,D,S}) where {B,D<:CuArray,S<:CuArray,M<:CuArray,V<:CuArray}
    op.basis == state.basis || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    
    d, S_op = op.disp, op.symplectic
    # Use in-place operations when possible
    temp_mean = S_op * state.mean
    state.mean .= temp_mean .+ d
    
    temp_covar = S_op * state.covar
    state.covar .= temp_covar * S_op'
    
    return state
end

function Gabs.apply!(state::GaussianState{B,M,V}, op::GaussianChannel{B,D,T}) where {B,D<:CuArray,T<:CuArray,M<:CuArray,V<:CuArray}
    op.basis == state.basis || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    
    d, T_op, N = op.disp, op.transform, op.noise
    # Use in-place operations when possible
    temp_mean = T_op * state.mean
    state.mean .= temp_mean .+ d
    
    temp_covar = T_op * state.covar
    state.covar .= temp_covar * T_op' .+ N
    
    return state
end