# GPU implementations for Gaussian state creation

# Vacuum state - GPU implementation
function Gabs._vacuumstate(basis::SymplecticBasis{N}; ħ = 2) where {N<:Int}
    nmodes = basis.nmodes
    mean = gpu_zeros(Float64, 2*nmodes)
    covar = gpu_identity(Float64, 2*nmodes) .* (ħ/2)
    return mean, covar
end

# Thermal state - GPU implementation for uniform case
function Gabs._thermalstate(basis::Union{QuadPairBasis{N},QuadBlockBasis{N}}, photons::P; ħ = 2) where {N<:Int,P<:Number}
    nmodes = basis.nmodes
    Rt = float(typeof(photons))
    mean = gpu_zeros(Rt, 2*nmodes)
    covar = gpu_identity(Rt, 2*nmodes) .* ((2 * photons + 1) * (ħ/2))
    return mean, covar
end

# Thermal state - GPU implementation for vector case (QuadPairBasis)
function Gabs._thermalstate(basis::QuadPairBasis{N}, photons::P; ħ = 2) where {N<:Int,P<:Vector}
    nmodes = basis.nmodes
    Rt = float(eltype(P))
    mean = gpu_zeros(Rt, 2*nmodes)
    covar = gpu_zeros(Rt, 2*nmodes, 2*nmodes)
    
    if cuda_available()
        # GPU kernel for setting diagonal elements
        function thermal_kernel!(covar, photons, ħ, nmodes)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if i <= nmodes
                val = (2 * photons[i] + 1) * (ħ/2)
                covar[2*i-1, 2*i-1] = val
                covar[2*i, 2*i] = val
            end
            return nothing
        end
        
        photons_gpu = ensure_gpu_array(photons, Rt)
        threads = min(nmodes, 256)
        blocks = cld(nmodes, threads)
        @cuda threads=threads blocks=blocks thermal_kernel!(covar, photons_gpu, ħ, nmodes)
        CUDA.synchronize()
    else
        # CPU fallback
        @inbounds for i in Base.OneTo(nmodes)
            val = (2 * photons[i] + 1) * (ħ/2)
            covar[2*i-1, 2*i-1] = val
            covar[2*i, 2*i] = val
        end
    end
    
    return mean, covar
end

# Thermal state - GPU implementation for vector case (QuadBlockBasis)
function Gabs._thermalstate(basis::QuadBlockBasis{N}, photons::P; ħ = 2) where {N<:Int,P<:Vector}
    nmodes = basis.nmodes
    Rt = float(eltype(P))
    mean = gpu_zeros(Rt, 2*nmodes)
    covar = gpu_zeros(Rt, 2*nmodes, 2*nmodes)
    
    if cuda_available()
        # GPU kernel for QuadBlockBasis
        function thermal_block_kernel!(covar, photons, ħ, nmodes)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if i <= nmodes
                val = (2 * photons[i] + 1) * (ħ/2)
                covar[i, i] = val
                covar[i+nmodes, i+nmodes] = val
            end
            return nothing
        end
        
        photons_gpu = ensure_gpu_array(photons, Rt)
        threads = min(nmodes, 256)
        blocks = cld(nmodes, threads)
        @cuda threads=threads blocks=blocks thermal_block_kernel!(covar, photons_gpu, ħ, nmodes)
        CUDA.synchronize()
    else
        # CPU fallback
        @inbounds for i in Base.OneTo(nmodes)
            val = (2 * photons[i] + 1) * (ħ/2)
            covar[i, i] = val
            covar[i+nmodes, i+nmodes] = val
        end
    end
    
    return mean, covar
end

# Coherent state - GPU implementation for single amplitude (QuadPairBasis)
function Gabs._coherentstate(basis::QuadPairBasis{N}, alpha::A; ħ = 2) where {N<:Int,A<:Number}
    nmodes = basis.nmodes
    mean_val = sqrt(2*ħ) * [real(alpha), imag(alpha)]
    
    if cuda_available()
        mean = CuArray(repeat(mean_val, nmodes))
    else
        mean = repeat(mean_val, nmodes)
    end
    
    covar = gpu_identity(real(A), 2*nmodes) .* (ħ/2)
    return mean, covar
end

# Coherent state - GPU implementation for vector amplitude (QuadPairBasis)
function Gabs._coherentstate(basis::QuadPairBasis{N}, alpha::A; ħ = 2) where {N<:Int,A<:Vector}
    nmodes = basis.nmodes
    Rt = real(eltype(A))
    
    if cuda_available()
        alpha_gpu = ensure_gpu_array(alpha)
        # Use reinterpret to interleave real and imaginary parts
        mean = sqrt(2*ħ) * CuArray(reinterpret(Rt, alpha_gpu))
    else
        mean = sqrt(2*ħ) * reinterpret(Rt, alpha)
    end
    
    covar = gpu_identity(Rt, 2*nmodes) .* (ħ/2)
    return mean, covar
end

# Coherent state - GPU implementation for single amplitude (QuadBlockBasis)
function Gabs._coherentstate(basis::QuadBlockBasis{N}, alpha::A; ħ = 2) where {N<:Int,A<:Number}
    nmodes = basis.nmodes
    
    if cuda_available()
        mean = CuArray{real(A)}(undef, 2*nmodes)
        mean[1:nmodes] .= sqrt(2*ħ) * real(alpha)
        mean[nmodes+1:2*nmodes] .= sqrt(2*ħ) * imag(alpha)
    else
        mean = repeat([sqrt(2*ħ) * real(alpha), sqrt(2*ħ) * imag(alpha)], inner = nmodes)
    end
    
    covar = gpu_identity(real(A), 2*nmodes) .* (ħ/2)
    return mean, covar
end

# Coherent state - GPU implementation for vector amplitude (QuadBlockBasis)
function Gabs._coherentstate(basis::QuadBlockBasis{N}, alpha::A; ħ = 2) where {N<:Int,A<:Vector}
    nmodes = basis.nmodes
    Rt = real(eltype(A))
    
    if cuda_available()
        alpha_gpu = ensure_gpu_array(alpha)
        mean = CuArray{Rt}(undef, 2*nmodes)
        mean[1:nmodes] = sqrt(2*ħ) * real.(alpha_gpu)
        mean[nmodes+1:2*nmodes] = sqrt(2*ħ) * imag.(alpha_gpu)
    else
        re = reinterpret(Rt, alpha)
        mean = vcat(@view(re[1:2:end]), @view(re[2:2:end]))
        mean .*= sqrt(2*ħ)
    end
    
    covar = gpu_identity(Rt, 2*nmodes) .* (ħ/2)
    return mean, covar
end

# Squeezed state - GPU implementation for single parameters (QuadPairBasis)
function Gabs._squeezedstate(basis::QuadPairBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int,R<:Real}
    nmodes = basis.nmodes
    mean = gpu_zeros(R, 2*nmodes)
    covar = gpu_zeros(R, 2*nmodes, 2*nmodes)
    
    cr, sr = cosh(2*r), sinh(2*r)
    ct, st = cos(theta), sin(theta)
    
    if cuda_available()
        # GPU kernel for squeezed state covariance
        function squeezed_kernel!(covar, cr, sr, ct, st, ħ, nmodes)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if i <= nmodes
                covar[2*i-1, 2*i-1] = (ħ/2) * (cr - sr*ct)
                covar[2*i-1, 2*i] = -(ħ/2) * sr * st
                covar[2*i, 2*i-1] = -(ħ/2) * sr * st
                covar[2*i, 2*i] = (ħ/2) * (cr + sr*ct)
            end
            return nothing
        end
        
        threads = min(nmodes, 256)
        blocks = cld(nmodes, threads)
        @cuda threads=threads blocks=blocks squeezed_kernel!(covar, cr, sr, ct, st, ħ, nmodes)
        CUDA.synchronize()
    else
        # CPU fallback
        @inbounds for i in Base.OneTo(nmodes)
            covar[2*i-1, 2*i-1] = (ħ/2) * (cr - sr*ct)
            covar[2*i-1, 2*i] = -(ħ/2) * sr * st
            covar[2*i, 2*i-1] = -(ħ/2) * sr * st
            covar[2*i, 2*i] = (ħ/2) * (cr + sr*ct)
        end
    end
    
    return mean, covar
end

# Squeezed state - GPU implementation for vector parameters (QuadPairBasis)
function Gabs._squeezedstate(basis::QuadPairBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int,R<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    mean = gpu_zeros(Rt, 2*nmodes)
    covar = gpu_zeros(Rt, 2*nmodes, 2*nmodes)
    
    if cuda_available()
        function squeezed_vector_kernel!(covar, r, theta, ħ, nmodes)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if i <= nmodes
                cr, sr = cosh(2*r[i]), sinh(2*r[i])
                ct, st = cos(theta[i]), sin(theta[i])
                covar[2*i-1, 2*i-1] = (ħ/2) * (cr - sr*ct)
                covar[2*i-1, 2*i] = -(ħ/2) * sr * st
                covar[2*i, 2*i-1] = -(ħ/2) * sr * st
                covar[2*i, 2*i] = (ħ/2) * (cr + sr*ct)
            end
            return nothing
        end
        
        r_gpu = ensure_gpu_array(r, Rt)
        theta_gpu = ensure_gpu_array(theta, Rt)
        threads = min(nmodes, 256)
        blocks = cld(nmodes, threads)
        @cuda threads=threads blocks=blocks squeezed_vector_kernel!(covar, r_gpu, theta_gpu, ħ, nmodes)
        CUDA.synchronize()
    else
        # CPU fallback
        @inbounds for i in Base.OneTo(nmodes)
            cr, sr = cosh(2*r[i]), sinh(2*r[i])
            ct, st = cos(theta[i]), sin(theta[i])
            covar[2*i-1, 2*i-1] = (ħ/2) * (cr - sr*ct)
            covar[2*i-1, 2*i] = -(ħ/2) * sr * st
            covar[2*i, 2*i-1] = -(ħ/2) * sr * st
            covar[2*i, 2*i] = (ħ/2) * (cr + sr*ct)
        end
    end
    
    return mean, covar
end

# Squeezed state - GPU implementation for QuadBlockBasis
function Gabs._squeezedstate(basis::QuadBlockBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int,R<:Real}
    nmodes = basis.nmodes
    mean = gpu_zeros(R, 2*nmodes)
    covar = gpu_zeros(R, 2*nmodes, 2*nmodes)
    
    cr, sr = cosh(2*r), sinh(2*r)
    ct, st = cos(theta), sin(theta)
    
    if cuda_available()
        function squeezed_block_kernel!(covar, cr, sr, ct, st, ħ, nmodes)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if i <= nmodes
                covar[i, i] = (ħ/2) * (cr - sr*ct)
                covar[i, i+nmodes] = -(ħ/2) * sr * st
                covar[i+nmodes, i] = -(ħ/2) * sr * st
                covar[i+nmodes, i+nmodes] = (ħ/2) * (cr + sr*ct)
            end
            return nothing
        end
        
        threads = min(nmodes, 256)
        blocks = cld(nmodes, threads)
        @cuda threads=threads blocks=blocks squeezed_block_kernel!(covar, cr, sr, ct, st, ħ, nmodes)
        CUDA.synchronize()
    else
        # CPU fallback
        @inbounds for i in Base.OneTo(nmodes)
            covar[i, i] = (ħ/2) * (cr - sr*ct)
            covar[i, i+nmodes] = -(ħ/2) * sr * st
            covar[i+nmodes, i] = -(ħ/2) * sr * st
            covar[i+nmodes, i+nmodes] = (ħ/2) * (cr + sr*ct)
        end
    end
    
    return mean, covar
end

# EPR state - GPU implementation (QuadPairBasis)
function Gabs._eprstate(basis::QuadPairBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int,R<:Real}
    nmodes = basis.nmodes
    mean = gpu_zeros(R, 2*nmodes)
    cr, sr = (ħ/2)*cosh(2*r), (ħ/2)*sinh(2*r)
    ct, st = cos(theta), sin(theta)
    covar = gpu_zeros(R, 2*nmodes, 2*nmodes)
    
    if cuda_available()
        function epr_kernel!(covar, cr, sr, ct, st, nmodes)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if i <= div(nmodes, 2)
                # Mode pair indices
                idx1, idx2 = 4*i-3, 4*i-1
                idy1, idy2 = 4*i-2, 4*i
                
                # Fill covariance matrix elements
                covar[idx1, idx1] = cr
                covar[idx1, idx2] = -sr * ct
                covar[idx1, idy2] = -sr * st
                
                covar[idy1, idy1] = cr
                covar[idy1, idx2] = -sr * st
                covar[idy1, idy2] = sr * ct
                
                covar[idx2, idx1] = -sr * ct
                covar[idx2, idy1] = -sr * st
                covar[idx2, idx2] = cr
                
                covar[idy2, idx1] = -sr * st
                covar[idy2, idy1] = sr * ct
                covar[idy2, idy2] = cr
            end
            return nothing
        end
        
        threads = min(div(nmodes, 2), 256)
        blocks = cld(div(nmodes, 2), threads)
        @cuda threads=threads blocks=blocks epr_kernel!(covar, cr, sr, ct, st, nmodes)
        CUDA.synchronize()
    else
        # CPU fallback
        @inbounds for i in Base.OneTo(Int(nmodes/2))
            covar[4*i-3, 4*i-3] = cr
            covar[4*i-3, 4*i-1] = -sr * ct
            covar[4*i-3, 4*i] = -sr * st

            covar[4*i-2, 4*i-2] = cr
            covar[4*i-2, 4*i-1] = -sr * st
            covar[4*i-2, 4*i] = sr * ct

            covar[4*i-1, 4*i-3] = -sr * ct
            covar[4*i-1, 4*i-2] = -sr * st
            covar[4*i-1, 4*i-1] = cr

            covar[4*i, 4*i-3] = -sr * st
            covar[4*i, 4*i-2] = sr * ct
            covar[4*i, 4*i] = cr
        end
    end
    
    return mean, covar
end