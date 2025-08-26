# GPU implementations for Gaussian unitary and channel operations

# Most unitary operations work naturally with CuArrays through linear algebra
# We provide GPU-optimized implementations where beneficial

# Displacement - GPU optimized implementations
function Gabs._displace(basis::QuadPairBasis{N}, alpha::A; ħ = 2) where {N<:Int,A<:Number}
    nmodes = basis.nmodes
    Rt = real(eltype(A))
    
    if cuda_available()
        disp_vals = CuArray([sqrt(2*ħ) * real(alpha), sqrt(2*ħ) * imag(alpha)])
        disp = repeat(disp_vals, nmodes)
        symplectic = CuArray{Rt}(I, 2*nmodes, 2*nmodes)
    else
        disp = repeat([sqrt(2*ħ) * real(alpha), sqrt(2*ħ) * imag(alpha)], nmodes)
        symplectic = Matrix{Rt}(I, 2*nmodes, 2*nmodes)
    end
    
    return disp, symplectic
end

function Gabs._displace(basis::QuadPairBasis{N}, alpha::A; ħ = 2) where {N<:Int,A<:Vector}
    nmodes = basis.nmodes
    Rt = real(eltype(A))
    
    if cuda_available()
        alpha_gpu = ensure_gpu_array(alpha)
        disp = sqrt(2*ħ) * CuArray(reinterpret(Rt, alpha_gpu))
        symplectic = CuArray{Rt}(I, 2*nmodes, 2*nmodes)
    else
        disp = collect(Iterators.flatten((real(i), imag(i)) for i in alpha))
        disp .*= sqrt(2*ħ)
        symplectic = Matrix{Rt}(I, 2*nmodes, 2*nmodes)
    end
    
    return disp, symplectic
end

# Squeezing - GPU optimized for vector case
function Gabs._squeeze(basis::QuadPairBasis{N}, r::R, theta::R) where {N<:Int,R<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = gpu_zeros(Rt, 2*nmodes)
    symplectic = gpu_zeros(Rt, 2*nmodes, 2*nmodes)
    
    if cuda_available()
        function squeeze_kernel!(symplectic, r, theta, nmodes)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if i <= nmodes
                cr, sr = cosh(r[i]), sinh(r[i])
                ct, st = cos(theta[i]), sin(theta[i])
                symplectic[2*i-1, 2*i-1] = cr - sr*ct
                symplectic[2*i-1, 2*i] = -sr * st
                symplectic[2*i, 2*i-1] = -sr * st
                symplectic[2*i, 2*i] = cr + sr*ct
            end
            return nothing
        end
        
        r_gpu = ensure_gpu_array(r, Rt)
        theta_gpu = ensure_gpu_array(theta, Rt)
        threads = min(nmodes, 256)
        blocks = cld(nmodes, threads)
        @cuda threads=threads blocks=blocks squeeze_kernel!(symplectic, r_gpu, theta_gpu, nmodes)
        CUDA.synchronize()
    else
        # CPU fallback
        @inbounds for i in Base.OneTo(nmodes)
            cr, sr = cosh(r[i]), sinh(r[i])
            ct, st = cos(theta[i]), sin(theta[i])
            symplectic[2*i-1, 2*i-1] = cr - sr*ct
            symplectic[2*i-1, 2*i] = -sr * st
            symplectic[2*i, 2*i-1] = -sr * st
            symplectic[2*i, 2*i] = cr + sr*ct
        end
    end
    
    return disp, symplectic
end

# Phase shift - GPU optimized for vector case
function Gabs._phaseshift(basis::QuadPairBasis{N}, theta::R) where {N<:Int,R<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = gpu_zeros(Rt, 2*nmodes)
    symplectic = gpu_zeros(Rt, 2*nmodes, 2*nmodes)
    
    if cuda_available()
        function phaseshift_kernel!(symplectic, theta, nmodes)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if i <= nmodes
                ct, st = cos(theta[i]), sin(theta[i])
                symplectic[2*i-1, 2*i-1] = ct
                symplectic[2*i-1, 2*i] = st
                symplectic[2*i, 2*i-1] = -st
                symplectic[2*i, 2*i] = ct
            end
            return nothing
        end
        
        theta_gpu = ensure_gpu_array(theta, Rt)
        threads = min(nmodes, 256)
        blocks = cld(nmodes, threads)
        @cuda threads=threads blocks=blocks phaseshift_kernel!(symplectic, theta_gpu, nmodes)
        CUDA.synchronize()
    else
        # CPU fallback
        @inbounds for i in Base.OneTo(nmodes)
            ct, st = cos(theta[i]), sin(theta[i])
            symplectic[2*i-1, 2*i-1] = ct
            symplectic[2*i-1, 2*i] = st
            symplectic[2*i, 2*i-1] = -st
            symplectic[2*i, 2*i] = ct
        end
    end
    
    return disp, symplectic
end

# Attenuator - GPU optimized for vector parameters
function Gabs._attenuator(basis::QuadPairBasis{N}, theta::R, n::M) where {N<:Int,R<:Vector,M<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = gpu_zeros(Rt, 2*nmodes)
    transform = gpu_zeros(Rt, 2*nmodes, 2*nmodes)
    noise = gpu_zeros(Rt, 2*nmodes, 2*nmodes)
    
    if cuda_available()
        function attenuator_kernel!(transform, noise, theta, n, nmodes)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if i <= nmodes
                ct, st = cos(theta[i]), sin(theta[i])
                ni = n[i]
                
                transform[2*i-1, 2*i-1] = ct
                transform[2*i, 2*i] = ct
                
                noise[2*i-1, 2*i-1] = st^2 * ni
                noise[2*i, 2*i] = st^2 * ni
            end
            return nothing
        end
        
        theta_gpu = ensure_gpu_array(theta, Rt)
        n_gpu = ensure_gpu_array(n, Rt)
        threads = min(nmodes, 256)
        blocks = cld(nmodes, threads)
        @cuda threads=threads blocks=blocks attenuator_kernel!(transform, noise, theta_gpu, n_gpu, nmodes)
        CUDA.synchronize()
    else
        # CPU fallback
        @inbounds for i in Base.OneTo(nmodes)
            ct, st = cos(theta[i]), sin(theta[i])
            ni = n[i]

            transform[2*i-1, 2*i-1] = ct
            transform[2*i, 2*i] = ct

            noise[2*i-1, 2*i-1] = st^2 * ni
            noise[2*i, 2*i] = st^2 * ni
        end
    end
    
    return disp, transform, noise
end

# Attenuator - GPU optimized for QuadBlockBasis vector parameters
function Gabs._attenuator(basis::QuadBlockBasis{N}, theta::R, n::M) where {N<:Int,R<:Vector,M<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = gpu_zeros(Rt, 2*nmodes)
    transform = gpu_zeros(Rt, 2*nmodes, 2*nmodes)
    noise = gpu_zeros(Rt, 2*nmodes, 2*nmodes)
    
    if cuda_available()
        function attenuator_block_kernel!(transform, noise, theta, n, nmodes)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if i <= nmodes
                ct, st = cos(theta[i]), sin(theta[i])
                ni = n[i]

                transform[i, i] = ct
                transform[i+nmodes, i+nmodes] = ct

                noise[i, i] = st^2 * ni
                noise[i+nmodes, i+nmodes] = st^2 * ni
            end
            return nothing
        end
        
        theta_gpu = ensure_gpu_array(theta, Rt)
        n_gpu = ensure_gpu_array(n, Rt)
        threads = min(nmodes, 256)
        blocks = cld(nmodes, threads)
        @cuda threads=threads blocks=blocks attenuator_block_kernel!(transform, noise, theta_gpu, n_gpu, nmodes)
        CUDA.synchronize()
    else
        # CPU fallback
        @inbounds for i in Base.OneTo(nmodes)
            ct, st = cos(theta[i]), sin(theta[i])
            ni = n[i]

            transform[i, i] = ct
            transform[i+nmodes, i+nmodes] = ct

            noise[i, i] = st^2 * ni
            noise[i+nmodes, i+nmodes] = st^2 * ni
        end
    end
    
    return disp, transform, noise
end

# Amplifier - GPU optimized for vector parameters
function Gabs._amplifier(basis::QuadPairBasis{N}, r::R, n::M) where {N<:Int,R<:Vector,M<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = gpu_zeros(Rt, 2*nmodes)
    transform = gpu_zeros(Rt, 2*nmodes, 2*nmodes)
    noise = gpu_zeros(Rt, 2*nmodes, 2*nmodes)
    
    if cuda_available()
        function amplifier_kernel!(transform, noise, r, n, nmodes)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if i <= nmodes
                cr, sr = cosh(r[i]), sinh(r[i])
                ni = n[i]

                transform[2*i-1, 2*i-1] = cr
                transform[2*i, 2*i] = cr

                noise[2*i-1, 2*i-1] = sr^2 * ni
                noise[2*i, 2*i] = sr^2 * ni
            end
            return nothing
        end
        
        r_gpu = ensure_gpu_array(r, Rt)
        n_gpu = ensure_gpu_array(n, Rt)
        threads = min(nmodes, 256)
        blocks = cld(nmodes, threads)
        @cuda threads=threads blocks=blocks amplifier_kernel!(transform, noise, r_gpu, n_gpu, nmodes)
        CUDA.synchronize()
    else
        # CPU fallback
        @inbounds for i in Base.OneTo(nmodes)
            cr, sr = cosh(r[i]), sinh(r[i])
            ni = n[i]

            transform[2*i-1, 2*i-1] = cr
            transform[2*i, 2*i] = cr

            noise[2*i-1, 2*i-1] = sr^2 * ni
            noise[2*i, 2*i] = sr^2 * ni
        end
    end
    
    return disp, transform, noise
end