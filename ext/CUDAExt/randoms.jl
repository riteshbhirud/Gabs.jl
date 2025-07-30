# GPU-enabled random state generation

# GPU random state generation
function Gabs.randstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}; pure = false, ħ = 2) where {Tm<:CuVector,Tc<:CuMatrix,N<:Int}
    nmodes = basis.nmodes
    
    # Generate random mean on GPU
    mean = CuArray(randn(Float64, 2*nmodes))
    
    if pure
        # Pure state: V = S * S^T * (ħ/2)
        # Generate random symplectic matrix on CPU first, then move to GPU
        S_cpu = Gabs._randsymplectic(basis)
        S = CuArray(S_cpu)
        covar = (ħ/2) * S * S'
    else
        # Mixed state using Williamson decomposition
        # Generate symplectic eigenvalues
        sympeigs_cpu = (ħ/2) * (abs.(randn(Float64, nmodes)) .+ 1.0)
        
        if basis isa QuadPairBasis
            will_diag = repeat(sympeigs_cpu, inner = 2)
        else # QuadBlockBasis
            will_diag = repeat(sympeigs_cpu, outer = 2)
        end
        
        # Generate random symplectic matrix
        S_cpu = Gabs._randsymplectic(basis)
        S = CuArray(S_cpu)
        will = CuArray(diagm(will_diag))
        
        covar = S * will * S'
    end
    
    return GaussianState(basis, mean, covar; ħ = ħ)
end

function Gabs.randstate(::Type{T}, basis::SymplecticBasis{N}; pure = false, ħ = 2) where {T<:CuArray,N<:Int}
    return Gabs.randstate(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis; pure = pure, ħ = ħ)
end

# GPU random unitary generation
function Gabs.randunitary(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}; passive = false, ħ = 2) where {Td<:CuVector,Ts<:CuMatrix,N<:Int}
    nmodes = basis.nmodes
    
    # Generate random displacement on GPU
    disp = CuArray(randn(Float64, 2*nmodes))
    
    # Generate random symplectic matrix on CPU, then move to GPU
    symp_cpu = Gabs._randsymplectic(basis; passive = passive)
    symp = CuArray(symp_cpu)
    
    return GaussianUnitary(basis, disp, symp; ħ = ħ)
end

function Gabs.randunitary(::Type{T}, basis::SymplecticBasis{N}; passive = false, ħ = 2) where {T<:CuArray,N<:Int}
    return Gabs.randunitary(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis; passive = passive, ħ = ħ)
end

# GPU random channel generation
function Gabs.randchannel(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}; ħ = 2) where {Td<:CuVector,Tt<:CuMatrix,N<:Int}
    nmodes = basis.nmodes
    
    # Generate random displacement on GPU
    disp = CuArray(randn(Float64, 2*nmodes))
    
    # Generate symplectic matrix for system + environment on CPU
    symp_cpu = Gabs._randsymplectic(typeof(basis)(3*nmodes))
    
    # Extract system transformation and environment coupling
    transform_cpu = symp_cpu[1:2*nmodes, 1:2*nmodes]
    B_cpu = symp_cpu[1:2*nmodes, 2*nmodes+1:6*nmodes]
    
    # Move to GPU
    transform = CuArray(transform_cpu)
    B = CuArray(B_cpu)
    
    # Compute noise matrix on GPU
    noise = B * B'
    
    return GaussianChannel(basis, disp, transform, noise; ħ = ħ)
end

function Gabs.randchannel(::Type{T}, basis::SymplecticBasis{N}; ħ = 2) where {T<:CuArray,N<:Int}
    return Gabs.randchannel(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis; ħ = ħ)
end

# Batch random state generation for efficiency
function batch_randstates(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, n_states::Int; pure = false, ħ = 2) where {Tm<:CuVector,Tc<:CuMatrix,N<:Int}
    """Generate multiple random states efficiently on GPU"""
    nmodes = basis.nmodes
    
    # Pre-allocate arrays for all states
    all_means = CuArray(randn(Float64, 2*nmodes, n_states))
    all_covars = CuArray(zeros(Float64, 2*nmodes, 2*nmodes, n_states))
    
    if pure
        # Generate random symplectic matrices for all states
        for i in 1:n_states
            S_cpu = Gabs._randsymplectic(basis)
            S = CuArray(S_cpu)
            all_covars[:, :, i] = (ħ/2) * S * S'
        end
    else
        # Mixed states
        for i in 1:n_states
            # Generate symplectic eigenvalues
            sympeigs_cpu = (ħ/2) * (abs.(randn(Float64, nmodes)) .+ 1.0)
            
            if basis isa QuadPairBasis
                will_diag = repeat(sympeigs_cpu, inner = 2)
            else # QuadBlockBasis
                will_diag = repeat(sympeigs_cpu, outer = 2)
            end
            
            # Generate random symplectic matrix
            S_cpu = Gabs._randsymplectic(basis)
            S = CuArray(S_cpu)
            will = CuArray(diagm(will_diag))
            
            all_covars[:, :, i] = S * will * S'
        end
    end
    
    # Convert to vector of GaussianState objects
    states = Vector{GaussianState}(undef, n_states)
    for i in 1:n_states
        mean_i = all_means[:, i]
        covar_i = all_covars[:, :, i]
        states[i] = GaussianState(basis, mean_i, covar_i; ħ = ħ)
    end
    
    return states
end

function batch_randstates(::Type{T}, basis::SymplecticBasis{N}, n_states::Int; pure = false, ħ = 2) where {T<:CuArray,N<:Int}
    return batch_randstates(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis, n_states; pure = pure, ħ = ħ)
end

function batch_randstates(basis::SymplecticBasis{N}, n_states::Int; pure = false, ħ = 2, use_gpu = true) where {N<:Int}
    if use_gpu && CUDA.functional()
        return batch_randstates(CuVector{Float64}, CuMatrix{Float64}, basis, n_states; pure = pure, ħ = ħ)
    else
        # CPU fallback
        return [Gabs.randstate(basis; pure = pure, ħ = ħ) for _ in 1:n_states]
    end
end

# Batch random unitary generation
function batch_randunitaries(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, n_ops::Int; passive = false, ħ = 2) where {Td<:CuVector,Ts<:CuMatrix,N<:Int}
    """Generate multiple random unitaries efficiently on GPU"""
    nmodes = basis.nmodes
    
    # Pre-allocate arrays
    all_disps = CuArray(randn(Float64, 2*nmodes, n_ops))
    
    # Generate unitaries
    unitaries = Vector{GaussianUnitary}(undef, n_ops)
    for i in 1:n_ops
        # Generate random symplectic matrix on CPU, then move to GPU
        symp_cpu = Gabs._randsymplectic(basis; passive = passive)
        symp = CuArray(symp_cpu)
        disp_i = all_disps[:, i]
        
        unitaries[i] = GaussianUnitary(basis, disp_i, symp; ħ = ħ)
    end
    
    return unitaries
end

function batch_randunitaries(::Type{T}, basis::SymplecticBasis{N}, n_ops::Int; passive = false, ħ = 2) where {T<:CuArray,N<:Int}
    return batch_randunitaries(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis, n_ops; passive = passive, ħ = ħ)
end

function batch_randunitaries(basis::SymplecticBasis{N}, n_ops::Int; passive = false, ħ = 2, use_gpu = true) where {N<:Int}
    if use_gpu && CUDA.functional()
        return batch_randunitaries(CuVector{Float64}, CuMatrix{Float64}, basis, n_ops; passive = passive, ħ = ħ)
    else
        # CPU fallback
        return [Gabs.randunitary(basis; passive = passive, ħ = ħ) for _ in 1:n_ops]
    end
end

# Batch random channel generation
function batch_randchannels(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}, n_channels::Int; ħ = 2) where {Td<:CuVector,Tt<:CuMatrix,N<:Int}
    """Generate multiple random channels efficiently on GPU"""
    nmodes = basis.nmodes
    
    # Pre-allocate displacement arrays
    all_disps = CuArray(randn(Float64, 2*nmodes, n_channels))
    
    # Generate channels
    channels = Vector{GaussianChannel}(undef, n_channels)
    for i in 1:n_channels
        # Generate symplectic matrix for system + environment on CPU
        symp_cpu = Gabs._randsymplectic(typeof(basis)(3*nmodes))
        
        # Extract components
        transform_cpu = symp_cpu[1:2*nmodes, 1:2*nmodes]
        B_cpu = symp_cpu[1:2*nmodes, 2*nmodes+1:6*nmodes]
        
        # Move to GPU
        transform = CuArray(transform_cpu)
        B = CuArray(B_cpu)
        noise = B * B'
        disp_i = all_disps[:, i]
        
        channels[i] = GaussianChannel(basis, disp_i, transform, noise; ħ = ħ)
    end
    
    return channels
end

function batch_randchannels(::Type{T}, basis::SymplecticBasis{N}, n_channels::Int; ħ = 2) where {T<:CuArray,N<:Int}
    return batch_randchannels(CuVector{eltype(T)}, CuMatrix{eltype(T)}, basis, n_channels; ħ = ħ)
end

function batch_randchannels(basis::SymplecticBasis{N}, n_channels::Int; ħ = 2, use_gpu = true) where {N<:Int}
    if use_gpu && CUDA.functional()
        return batch_randchannels(CuVector{Float64}, CuMatrix{Float64}, basis, n_channels; ħ = ħ)
    else
        # CPU fallback
        return [Gabs.randchannel(basis; ħ = ħ) for _ in 1:n_channels]
    end
end

# Optimized random linear combination generation
function rand_linearcombination(basis::SymplecticBasis{N}, n_terms::Int; 
                               pure_states = false, complex_coeffs = false, 
                               normalize = true, ħ = 2, use_gpu = true) where {N<:Int}
    """Generate random GaussianLinearCombination efficiently"""
    
    # Generate random states
    if use_gpu && CUDA.functional()
        states = batch_randstates(CuVector{Float64}, CuMatrix{Float64}, basis, n_terms; pure = pure_states, ħ = ħ)
    else
        states = [Gabs.randstate(basis; pure = pure_states, ħ = ħ) for _ in 1:n_terms]
    end
    
    # Generate random coefficients
    if complex_coeffs
        coeffs = randn(ComplexF64, n_terms)
    else
        coeffs = randn(Float64, n_terms)
    end
    
    # Create linear combination
    lc = GaussianLinearCombination(basis, coeffs, states)
    
    # Normalize if requested
    if normalize
        Gabs.normalize!(lc)
    end
    
    return lc
end

# Random cat state generation
function rand_catstate(basis::SymplecticBasis{1}; max_alpha = 2.0, ħ = 2, use_gpu = true)
    """Generate random cat state with random amplitude"""
    alpha = (rand() * 2 - 1) * max_alpha + 1im * (rand() * 2 - 1) * max_alpha
    return Gabs.catstate_even(basis, alpha; ħ = ħ)
end

# Random GKP state generation
function rand_gkpstate(basis::SymplecticBasis{1}; 
                      lattice = rand(["square", "hexagonal"]),
                      delta_range = (0.05, 0.3),
                      nmax_range = (2, 5),
                      ħ = 2)
    """Generate random GKP state with random parameters"""
    delta = rand() * (delta_range[2] - delta_range[1]) + delta_range[1]
    nmax = rand(nmax_range[1]:nmax_range[2])
    
    return Gabs.gkpstate(basis; lattice = lattice, delta = delta, nmax = nmax, ħ = ħ)
end