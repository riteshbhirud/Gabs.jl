# GPU-enabled Gaussian state constructors

# Vacuum state on GPU
function Gabs.vacuumstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}; ħ = 2) where {Tm<:CuVector,Tc<:CuMatrix,N<:Int}
    nmodes = basis.nmodes
    mean = CUDA.zeros(eltype(Tm), 2*nmodes)
    covar = CuArray((ħ/2) * Matrix{Float64}(I, 2*nmodes, 2*nmodes))
    return GaussianState(basis, mean, covar; ħ = ħ)
end

function Gabs.vacuumstate(::Type{T}, basis::SymplecticBasis{N}; ħ = 2) where {T<:CuArray,N<:Int}
    return vacuumstate(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis; ħ = ħ)
end

# Thermal state on GPU  
function Gabs.thermalstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, photons::P; ħ = 2) where {Tm<:CuVector,Tc<:CuMatrix,N<:Int,P}
    nmodes = basis.nmodes
    Rt = float(eltype(P))
    mean = CUDA.zeros(Rt, 2*nmodes)
    
    if photons isa Number
        covar_cpu = Matrix{Rt}((2 * photons + 1) * (ħ/2) * I, 2*nmodes, 2*nmodes)
        covar = CuArray(covar_cpu)
    else
        # Handle vector of photons for different modes
        covar_cpu = zeros(Rt, 2*nmodes, 2*nmodes)
        if basis isa QuadPairBasis
            for i in Base.OneTo(nmodes)
                val = (2 * photons[i] + 1) * (ħ/2)
                covar_cpu[2*i-1, 2*i-1] = val
                covar_cpu[2*i, 2*i] = val
            end
        else # QuadBlockBasis
            for i in Base.OneTo(nmodes)
                val = (2 * photons[i] + 1) * (ħ/2)
                covar_cpu[i, i] = val
                covar_cpu[i+nmodes, i+nmodes] = val
            end
        end
        covar = CuArray(covar_cpu)
    end
    
    return GaussianState(basis, mean, covar; ħ = ħ)
end

function Gabs.thermalstate(::Type{T}, basis::SymplecticBasis{N}, photons::P; ħ = 2) where {T<:CuArray,N<:Int,P}
    return thermalstate(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis, photons; ħ = ħ)
end

# Coherent state on GPU
function Gabs.coherentstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {Tm<:CuVector,Tc<:CuMatrix,N<:Int,A}
    nmodes = basis.nmodes
    
    if alpha isa Number
        if basis isa QuadPairBasis
            mean_cpu = repeat([sqrt(2*ħ) * real(alpha), sqrt(2*ħ) * imag(alpha)], nmodes)
        else # QuadBlockBasis
            mean_cpu = repeat([sqrt(2*ħ) * real(alpha), sqrt(2*ħ) * imag(alpha)], inner = nmodes)
        end
        mean = CuArray(mean_cpu)
        covar_cpu = Matrix{real(A)}((ħ/2) * I, 2*nmodes, 2*nmodes)
        covar = CuArray(covar_cpu)
    else # Vector of alphas
        Rt = real(eltype(A))
        if basis isa QuadPairBasis
            mean_cpu = sqrt(2*ħ) * reinterpret(Rt, alpha)
        else # QuadBlockBasis
            re = reinterpret(Rt, alpha)
            mean_cpu = vcat(view(re, 1:2:length(re)), view(re, 2:2:length(re)))
            mean_cpu .*= sqrt(2*ħ)
        end
        mean = CuArray(mean_cpu)
        covar_cpu = Matrix{Rt}((ħ/2) * I, 2*nmodes, 2*nmodes)
        covar = CuArray(covar_cpu)
    end
    
    return GaussianState(basis, mean, covar; ħ = ħ)
end

function Gabs.coherentstate(::Type{T}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {T<:CuArray,N<:Int,A}
    return coherentstate(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis, alpha; ħ = ħ)
end

# Squeezed state on GPU
function Gabs.squeezedstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {Tm<:CuVector,Tc<:CuMatrix,N<:Int,R}
    nmodes = basis.nmodes
    
    if r isa Real && theta isa Real
        # Single mode parameters
        mean = CUDA.zeros(real(R), 2*nmodes)
        covar_cpu = zeros(real(R), 2*nmodes, 2*nmodes)
        cr, sr = cosh(2*r), sinh(2*r)
        ct, st = cos(theta), sin(theta)
        
        if basis isa QuadPairBasis
            for i in Base.OneTo(nmodes)
                covar_cpu[2*i-1, 2*i-1] = (ħ/2) * (cr - sr*ct)
                covar_cpu[2*i-1, 2*i] = -(ħ/2) * sr * st
                covar_cpu[2*i, 2*i-1] = -(ħ/2) * sr * st
                covar_cpu[2*i, 2*i] = (ħ/2) * (cr + sr*ct)
            end
        else # QuadBlockBasis
            for i in Base.OneTo(nmodes)
                covar_cpu[i, i] = (ħ/2) * (cr - sr*ct)
                covar_cpu[i, i+nmodes] = -(ħ/2) * sr * st
                covar_cpu[i+nmodes, i] = -(ħ/2) * sr * st
                covar_cpu[i+nmodes, i+nmodes] = (ħ/2) * (cr + sr*ct)
            end
        end
        covar = CuArray(covar_cpu)
    else
        # Vector parameters - different for each mode
        Rt = eltype(R)
        mean = CUDA.zeros(Rt, 2*nmodes)
        covar_cpu = zeros(Rt, 2*nmodes, 2*nmodes)
        
        if basis isa QuadPairBasis
            for i in Base.OneTo(nmodes)
                cr, sr = cosh(2*r[i]), sinh(2*r[i])
                ct, st = cos(theta[i]), sin(theta[i])
                covar_cpu[2*i-1, 2*i-1] = (ħ/2) * (cr - sr*ct)
                covar_cpu[2*i-1, 2*i] = -(ħ/2) * sr * st
                covar_cpu[2*i, 2*i-1] = -(ħ/2) * sr * st
                covar_cpu[2*i, 2*i] = (ħ/2) * (cr + sr*ct)
            end
        else # QuadBlockBasis
            for i in Base.OneTo(nmodes)
                cr, sr = cosh(2*r[i]), sinh(2*r[i])
                ct, st = cos(theta[i]), sin(theta[i])
                covar_cpu[i, i] = (ħ/2) * (cr - sr*ct)
                covar_cpu[i, i+nmodes] = -(ħ/2) * sr * st
                covar_cpu[i+nmodes, i] = -(ħ/2) * sr * st
                covar_cpu[i+nmodes, i+nmodes] = (ħ/2) * (cr + sr*ct)
            end
        end
        covar = CuArray(covar_cpu)
    end
    
    return GaussianState(basis, mean, covar; ħ = ħ)
end

function Gabs.squeezedstate(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {T<:CuArray,N<:Int,R}
    return squeezedstate(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis, r, theta; ħ = ħ)
end

# EPR state on GPU
function Gabs.eprstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {Tm<:CuVector,Tc<:CuMatrix,N<:Int,R}
    nmodes = basis.nmodes
    mean = CUDA.zeros(real(R), 2*nmodes)
    covar_cpu = zeros(real(R), 2*nmodes, 2*nmodes)
    
    if r isa Real && theta isa Real
        cr, sr = (ħ/2)*cosh(2*r), (ħ/2)*sinh(2*r)
        ct, st = cos(theta), sin(theta)
        
        if basis isa QuadPairBasis
            for i in Base.OneTo(Int(nmodes/2))
                covar_cpu[4*i-3, 4*i-3] = cr
                covar_cpu[4*i-3, 4*i-1] = -sr * ct
                covar_cpu[4*i-3, 4*i] = -sr * st
                
                covar_cpu[4*i-2, 4*i-2] = cr
                covar_cpu[4*i-2, 4*i-1] = -sr * st
                covar_cpu[4*i-2, 4*i] = sr * ct
                
                covar_cpu[4*i-1, 4*i-3] = -sr * ct
                covar_cpu[4*i-1, 4*i-2] = -sr * st
                covar_cpu[4*i-1, 4*i-1] = cr
                
                covar_cpu[4*i, 4*i-3] = -sr * st
                covar_cpu[4*i, 4*i-2] = sr * ct
                covar_cpu[4*i, 4*i] = cr
            end
        else # QuadBlockBasis - similar pattern adapted for block structure
            for i in Base.OneTo(Int(nmodes/2))
                covar_cpu[2*i-1, 2*i-1] = cr
                covar_cpu[2*i-1, 2*i] = -sr * ct
                covar_cpu[2*i, 2*i-1] = -sr * ct
                covar_cpu[2*i, 2*i] = cr
                
                covar_cpu[2*i-1, 2*i+nmodes] = -sr * st
                covar_cpu[2*i, 2*i+nmodes-1] = -sr * st
                
                covar_cpu[2*i+nmodes-1, 2*i+nmodes-1] = cr
                covar_cpu[2*i+nmodes-1, 2*i+nmodes] = sr * ct
                covar_cpu[2*i+nmodes, 2*i+nmodes-1] = sr * ct
                covar_cpu[2*i+nmodes, 2*i+nmodes] = cr
                
                covar_cpu[2*i+nmodes-1, 2*i] = -sr * st
                covar_cpu[2*i+nmodes, 2*i-1] = -sr * st
            end
        end
    else
        # Vector parameters - handle multiple pairs
        Rt = eltype(R)
        if basis isa QuadPairBasis
            for i in Base.OneTo(Int(nmodes/2))
                cr, sr = (ħ/2)*cosh(2*r[i]), (ħ/2)*sinh(2*r[i])
                ct, st = cos(theta[i]), sin(theta[i])
                
                covar_cpu[4*i-3, 4*i-3] = cr
                covar_cpu[4*i-3, 4*i-1] = -sr * ct
                covar_cpu[4*i-3, 4*i] = -sr * st
                
                covar_cpu[4*i-2, 4*i-2] = cr
                covar_cpu[4*i-2, 4*i-1] = -sr * st
                covar_cpu[4*i-2, 4*i] = sr * ct
                
                covar_cpu[4*i-1, 4*i-3] = -sr * ct
                covar_cpu[4*i-1, 4*i-2] = -sr * st
                covar_cpu[4*i-1, 4*i-1] = cr
                
                covar_cpu[4*i, 4*i-3] = -sr * st
                covar_cpu[4*i, 4*i-2] = sr * ct
                covar_cpu[4*i, 4*i] = cr
            end
        else # QuadBlockBasis
            for i in Base.OneTo(Int(nmodes/2))
                cr, sr = (ħ/2)*cosh(2*r[i]), (ħ/2)*sinh(2*r[i])
                ct, st = cos(theta[i]), sin(theta[i])
                
                covar_cpu[2*i-1, 2*i-1] = cr
                covar_cpu[2*i-1, 2*i] = -sr * ct
                covar_cpu[2*i, 2*i-1] = -sr * ct
                covar_cpu[2*i, 2*i] = cr
                
                covar_cpu[2*i-1, 2*i+nmodes] = -sr * st
                covar_cpu[2*i, 2*i+nmodes-1] = -sr * st
                
                covar_cpu[2*i+nmodes-1, 2*i+nmodes-1] = cr
                covar_cpu[2*i+nmodes-1, 2*i+nmodes] = sr * ct
                covar_cpu[2*i+nmodes, 2*i+nmodes-1] = sr * ct
                covar_cpu[2*i+nmodes, 2*i+nmodes] = cr
                
                covar_cpu[2*i+nmodes-1, 2*i] = -sr * st
                covar_cpu[2*i+nmodes, 2*i-1] = -sr * st
            end
        end
    end
    
    covar = CuArray(covar_cpu)
    return GaussianState(basis, mean, covar; ħ = ħ)
end

function Gabs.eprstate(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {T<:CuArray,N<:Int,R}
    return eprstate(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis, r, theta; ħ = ħ)
end