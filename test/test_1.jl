@testitem "gpu-1" begin
using Gabs
using LinearAlgebra
using CUDA

# Skip tests if CUDA not available
const CUDA_TESTS_ENABLED = CUDA.functional()

if !CUDA_TESTS_ENABLED
    @warn "CUDA not available - skipping GPU tests"
end

@testset "GPU Foundation Tests" begin
    if !CUDA_TESTS_ENABLED
        @test_skip "CUDA not available"
        return
    end

    @testset "GPU Utilities and Infrastructure" begin
        @testset "Array Promotion" begin
            # Test vector promotion
            vec = [1.0, 2.0, 3.0, 4.0]
            gpu_vec = Gabs._promote_output_vector(CuVector{Float32}, vec, 4)
            @test gpu_vec isa CuVector{Float32}
            @test Array(gpu_vec) ≈ Float32.(vec)
            
            # Test matrix promotion
            mat = [1.0 2.0; 3.0 4.0]
            gpu_mat = Gabs._promote_output_matrix(CuMatrix{Float32}, mat, (2, 2))
            @test gpu_mat isa CuMatrix{Float32}
            @test Array(gpu_mat) ≈ Float32.(mat)
        end
        
        @testset "GPU Detection" begin
            @test CUDAExt.cuda_available() == true
            @test CUDAExt.CUDA_AVAILABLE == true
        end
    end

    @testset "GPU State Creation" begin
        for basis in [QuadPairBasis(1), QuadPairBasis(3), QuadBlockBasis(1), QuadBlockBasis(2)]
            nmodes = basis.nmodes
            
            @testset "$(typeof(basis)) - $nmodes modes" begin
                @testset "Vacuum State" begin
                    # Test Float32 and Float64
                    for T in [Float32, Float64]
                        cpu_vac = vacuumstate(basis, ħ=2)
                        gpu_vac = vacuumstate(CuVector{T}, CuMatrix{T}, basis, ħ=2)
                        
                        @test gpu_vac.mean isa CuVector{T}
                        @test gpu_vac.covar isa CuMatrix{T}
                        @test gpu_vac.basis == basis
                        @test gpu_vac.ħ == 2
                        
                        # Check values match CPU version
                        @test Array(gpu_vac.mean) ≈ T.(cpu_vac.mean) atol=1e-6
                        @test Array(gpu_vac.covar) ≈ T.(cpu_vac.covar) atol=1e-6
                    end
                end
                
                @testset "Coherent State" begin
                    for T in [Float32, Float64]
                        α = T(1.0) + T(0.5)im
                        cpu_coh = coherentstate(basis, α, ħ=2)
                        gpu_coh = coherentstate(CuVector{T}, CuMatrix{T}, basis, α, ħ=2)
                        
                        @test gpu_coh.mean isa CuVector{T}
                        @test gpu_coh.covar isa CuMatrix{T}
                        @test Array(gpu_coh.mean) ≈ T.(cpu_coh.mean) atol=1e-6
                        @test Array(gpu_coh.covar) ≈ T.(cpu_coh.covar) atol=1e-6
                        
                        # Test vector of alphas
                        if nmodes > 1
                            αs = [T(i) + T(0.1*i)im for i in 1:nmodes]
                            cpu_coh_vec = coherentstate(basis, αs, ħ=2)
                            gpu_coh_vec = coherentstate(CuVector{T}, CuMatrix{T}, basis, αs, ħ=2)
                            @test Array(gpu_coh_vec.mean) ≈ T.(cpu_coh_vec.mean) atol=1e-6
                            @test Array(gpu_coh_vec.covar) ≈ T.(cpu_coh_vec.covar) atol=1e-6
                        end
                    end
                end
                
                @testset "Squeezed State" begin
                    for T in [Float32, Float64]
                        r, θ = T(0.3), T(π/4)
                        cpu_sq = squeezedstate(basis, r, θ, ħ=2)
                        gpu_sq = squeezedstate(CuVector{T}, CuMatrix{T}, basis, r, θ, ħ=2)
                        
                        @test Array(gpu_sq.mean) ≈ T.(cpu_sq.mean) atol=1e-6
                        @test Array(gpu_sq.covar) ≈ T.(cpu_sq.covar) atol=1e-6
                        
                        # Test vector parameters
                        if nmodes > 1
                            rs = [T(0.1*i) for i in 1:nmodes]
                            θs = [T(π*i/4) for i in 1:nmodes]
                            cpu_sq_vec = squeezedstate(basis, rs, θs, ħ=2)
                            gpu_sq_vec = squeezedstate(CuVector{T}, CuMatrix{T}, basis, rs, θs, ħ=2)
                            @test Array(gpu_sq_vec.mean) ≈ T.(cpu_sq_vec.mean) atol=1e-6
                            @test Array(gpu_sq_vec.covar) ≈ T.(cpu_sq_vec.covar) atol=1e-6
                        end
                    end
                end
                
                @testset "Thermal State" begin
                    for T in [Float32, Float64]
                        n = T(2.0)
                        cpu_th = thermalstate(basis, n, ħ=2)
                        gpu_th = thermalstate(CuVector{T}, CuMatrix{T}, basis, n, ħ=2)
                        
                        @test Array(gpu_th.mean) ≈ T.(cpu_th.mean) atol=1e-6
                        @test Array(gpu_th.covar) ≈ T.(cpu_th.covar) atol=1e-6
                        
                        # Test vector of photon numbers
                        if nmodes > 1
                            ns = [T(i) for i in 1:nmodes]
                            cpu_th_vec = thermalstate(basis, ns, ħ=2)
                            gpu_th_vec = thermalstate(CuVector{T}, CuMatrix{T}, basis, ns, ħ=2)
                            @test Array(gpu_th_vec.mean) ≈ T.(cpu_th_vec.mean) atol=1e-6
                            @test Array(gpu_th_vec.covar) ≈ T.(cpu_th_vec.covar) atol=1e-6
                        end
                    end
                end
                
                if nmodes % 2 == 0  # EPR state requires even number of modes
                    @testset "EPR State" begin
                        for T in [Float32, Float64]
                            r, θ = T(0.5), T(π/6)
                            cpu_epr = eprstate(basis, r, θ, ħ=2)
                            gpu_epr = eprstate(CuVector{T}, CuMatrix{T}, basis, r, θ, ħ=2)
                            
                            @test Array(gpu_epr.mean) ≈ T.(cpu_epr.mean) atol=1e-6
                            @test Array(gpu_epr.covar) ≈ T.(cpu_epr.covar) atol=1e-6
                        end
                    end
                end
            end
        end
    end

    @testset "GPU Unitary Operations" begin
        for basis in [QuadPairBasis(1), QuadPairBasis(2), QuadBlockBasis(1)]
            nmodes = basis.nmodes
            
            @testset "$(typeof(basis)) - $nmodes modes" begin
                # Create test state
                α = 1.0f0 + 0.5f0im
                gpu_state = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, α)
                cpu_state = coherentstate(basis, α)
                
                @testset "Displacement" begin
                    β = 0.5f0 - 0.3f0im
                    cpu_disp = displace(basis, β)
                    gpu_disp = displace(CuVector{Float32}, CuMatrix{Float32}, basis, β)
                    
                    cpu_result = cpu_disp * cpu_state
                    gpu_result = gpu_disp * gpu_state
                    
                    @test Array(gpu_result.mean) ≈ cpu_result.mean atol=1e-6
                    @test Array(gpu_result.covar) ≈ cpu_result.covar atol=1e-6
                end
                
                @testset "Squeezing" begin
                    r, θ = 0.2f0, π/3
                    cpu_squeeze = squeeze(basis, r, θ)
                    gpu_squeeze = squeeze(CuVector{Float32}, CuMatrix{Float32}, basis, r, θ)
                    
                    cpu_result = cpu_squeeze * cpu_state
                    gpu_result = gpu_squeeze * gpu_state
                    
                    @test Array(gpu_result.mean) ≈ cpu_result.mean atol=1e-6
                    @test Array(gpu_result.covar) ≈ cpu_result.covar atol=1e-6
                end
                
                @testset "Phase Shift" begin
                    φ = π/4
                    cpu_phase = phaseshift(basis, φ)
                    gpu_phase = phaseshift(CuVector{Float32}, CuMatrix{Float32}, basis, φ)
                    
                    cpu_result = cpu_phase * cpu_state
                    gpu_result = gpu_phase * gpu_state
                    
                    @test Array(gpu_result.mean) ≈ cpu_result.mean atol=1e-6
                    @test Array(gpu_result.covar) ≈ cpu_result.covar atol=1e-6
                end
            end
        end
    end

    @testset "GPU Channel Operations" begin
        for basis in [QuadPairBasis(1), QuadBlockBasis(1)]
            @testset "$(typeof(basis))" begin
                # Create test state
                α = 1.0f0 + 0.5f0im
                gpu_state = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, α)
                cpu_state = coherentstate(basis, α)
                
                @testset "Attenuator" begin
                    θ, n = π/6, 2.0f0
                    cpu_att = attenuator(basis, θ, n)
                    gpu_att = attenuator(CuVector{Float32}, CuMatrix{Float32}, basis, θ, n)
                    
                    cpu_result = cpu_att * cpu_state
                    gpu_result = gpu_att * gpu_state
                    
                    @test Array(gpu_result.mean) ≈ cpu_result.mean atol=1e-6
                    @test Array(gpu_result.covar) ≈ cpu_result.covar atol=1e-6
                end
                
                @testset "Amplifier" begin
                    r, n = 0.3f0, 1.5f0
                    cpu_amp = amplifier(basis, r, n)
                    gpu_amp = amplifier(CuVector{Float32}, CuMatrix{Float32}, basis, r, n)
                    
                    cpu_result = cpu_amp * cpu_state
                    gpu_result = gpu_amp * gpu_state
                    
                    @test Array(gpu_result.mean) ≈ cpu_result.mean atol=1e-6
                    @test Array(gpu_result.covar) ≈ cpu_result.covar atol=1e-6
                end
            end
        end
    end

    @testset "GPU Wigner Functions" begin
        for basis in [QuadPairBasis(1), QuadPairBasis(2)]
            nmodes = basis.nmodes
            
            @testset "$(typeof(basis)) - $nmodes modes" begin
                # Create test states
                α = 1.0f0 + 0.5f0im
                gpu_coh = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, α)
                cpu_coh = coherentstate(basis, α)
                
                gpu_sq = squeezedstate(CuVector{Float32}, CuMatrix{Float32}, basis, 0.3f0, π/4)
                cpu_sq = squeezedstate(basis, 0.3f0, π/4)
                
                @testset "Single Point Evaluation" begin
                    x = zeros(Float32, 2*nmodes)
                    x[1] = 0.5f0
                    x[2] = -0.3f0
                    
                    # Test coherent state
                    gpu_w = wigner(gpu_coh, x)
                    cpu_w = wigner(cpu_coh, x)
                    @test gpu_w ≈ cpu_w atol=1e-5
                    
                    # Test squeezed state
                    gpu_w_sq = wigner(gpu_sq, x)
                    cpu_w_sq = wigner(cpu_sq, x)
                    @test gpu_w_sq ≈ cpu_w_sq atol=1e-5
                    
                    # Test characteristic function
                    xi = zeros(Float32, 2*nmodes)
                    xi[1] = 0.2f0
                    xi[2] = 0.1f0
                    
                    gpu_chi = wignerchar(gpu_coh, xi)
                    cpu_chi = wignerchar(cpu_coh, xi)
                    @test real(gpu_chi) ≈ real(cpu_chi) atol=1e-5
                    @test imag(gpu_chi) ≈ imag(cpu_chi) atol=1e-5
                end
                
                @testset "Batch Evaluation" begin
                    # Create grid of test points
                    npoints = 100
                    x_grid = CuArray(randn(Float32, 2*nmodes, npoints))
                    x_grid_cpu = Array(x_grid)
                    
                    # Test batched Wigner evaluation
                    gpu_w_batch = wigner(gpu_coh, x_grid)
                    @test gpu_w_batch isa CuVector{Float32}
                    @test length(gpu_w_batch) == npoints
                    
                    # Compare with individual evaluations
                    cpu_w_batch = [wigner(cpu_coh, x_grid_cpu[:, i]) for i in 1:npoints]
                    @test Array(gpu_w_batch) ≈ cpu_w_batch atol=1e-5
                    
                    # Test batched characteristic function
                    xi_grid = CuArray(randn(Float32, 2*nmodes, npoints))
                    gpu_chi_batch = wignerchar(gpu_coh, xi_grid)
                    @test gpu_chi_batch isa CuVector{ComplexF64}
                    @test length(gpu_chi_batch) == npoints
                end
                
                @testset "CPU Array Fallback" begin
                    # Test GPU state with CPU arrays (should promote to GPU)
                    x_cpu = zeros(Float32, 2*nmodes)
                    x_cpu[1] = 0.5f0
                    
                    gpu_w = wigner(gpu_coh, x_cpu)
                    cpu_w = wigner(cpu_coh, x_cpu)
                    @test gpu_w ≈ cpu_w atol=1e-5
                    
                    # Test batch with CPU arrays
                    x_grid_cpu = randn(Float32, 2*nmodes, 10)
                    gpu_w_batch = wigner(gpu_coh, x_grid_cpu)
                    @test gpu_w_batch isa Vector{Float32}
                end
            end
        end
    end

    @testset "Performance Benchmarks" begin
        @testset "State Creation Performance" begin
            basis = QuadPairBasis(5)
            α = 1.0f0 + 0.5f0im
            
            # Benchmark CPU
            cpu_time = @elapsed for _ in 1:100
                coherentstate(basis, α)
            end
            
            # Benchmark GPU
            CUDA.synchronize()
            gpu_time = @elapsed begin
                for _ in 1:100
                    coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, α)
                end
                CUDA.synchronize()
            end
            
            @test gpu_time > 0  # Just check it ran
            @test cpu_time > 0
            
            println("State creation - CPU: $(cpu_time*1000) ms, GPU: $(gpu_time*1000) ms")
        end
        
        @testset "Wigner Function Performance" begin
            basis = QuadPairBasis(1)
            α = 1.0f0 + 0.5f0im
            
            gpu_coh = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, α)
            cpu_coh = coherentstate(basis, α)
            
            # Large batch for performance testing
            npoints = 10000
            x_grid = randn(Float32, 2, npoints)
            x_grid_gpu = CuArray(x_grid)
            
            # CPU benchmark
            cpu_time = @elapsed begin
                cpu_results = [wigner(cpu_coh, x_grid[:, i]) for i in 1:npoints]
            end
            
            # GPU benchmark  
            CUDA.synchronize()
            gpu_time = @elapsed begin
                gpu_results = wigner(gpu_coh, x_grid_gpu)
                CUDA.synchronize()
            end
            
            speedup = cpu_time / gpu_time
            println("Wigner evaluation ($npoints points) - CPU: $(cpu_time*1000) ms, GPU: $(gpu_time*1000) ms, Speedup: $(speedup)x")
            
            @test speedup > 1.0  # GPU should be faster for large batches
        end
    end

    @testset "Error Handling" begin
        @testset "Invalid Dimensions" begin
            basis = QuadPairBasis(1)
            gpu_state = vacuumstate(CuVector{Float32}, CuMatrix{Float32}, basis)
            
            # Wrong size x vector for Wigner
            x_wrong = CuArray(Float32[1.0, 2.0, 3.0])  # Should be size 2
            @test_throws ArgumentError wigner(gpu_state, x_wrong)
            
            # Wrong size xi vector for characteristic function
            @test_throws ArgumentError wignerchar(gpu_state, x_wrong)
        end
        
        @testset "Mixed CPU/GPU Operations" begin
            basis = QuadPairBasis(1)
            cpu_state = coherentstate(basis, 1.0f0)
            gpu_unitary = displace(CuVector{Float32}, CuMatrix{Float32}, basis, 0.5f0)
            
            # This should work (promote CPU state)
            result = gpu_unitary * cpu_state
            @test result.mean isa Vector{Float32}  # Falls back to CPU
        end
    end

    @testset "Memory Management" begin
        @testset "Large State Creation" begin
            basis = QuadPairBasis(10)  # 20-dimensional phase space
            
            # This should not crash due to memory issues
            gpu_state = vacuumstate(CuVector{Float32}, CuMatrix{Float32}, basis)
            @test gpu_state.mean isa CuVector{Float32}
            @test size(gpu_state.covar) == (20, 20)
            
            # Free memory
            gpu_state = nothing
            CUDA.reclaim()
        end
        
        @testset "Batch Size Limits" begin
            basis = QuadPairBasis(1)
            gpu_state = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, 1.0f0)
            
            # Test increasingly large batches
            for batch_size in [100, 1000, 10000]
                x_grid = CuArray(randn(Float32, 2, batch_size))
                result = wigner(gpu_state, x_grid)
                @test length(result) == batch_size
                @test all(isfinite, Array(result))
            end
        end
    end

    @testset "Numerical Accuracy" begin
        @testset "Different Precisions" begin
            basis = QuadPairBasis(1)
            α = 1.0 + 0.5im
            
            # Create states with different precisions
            gpu_f32 = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, Float32(α))
            gpu_f64 = coherentstate(CuVector{Float64}, CuMatrix{Float64}, basis, α)
            cpu_f64 = coherentstate(basis, α)
            
            x = [0.5, -0.3]
            
            w_f32 = wigner(gpu_f32, CuArray(Float32.(x)))
            w_f64 = wigner(gpu_f64, CuArray(x))
            w_cpu = wigner(cpu_f64, x)
            
            # Float64 should be more accurate
            @test abs(w_f64 - w_cpu) < abs(w_f32 - w_cpu)
            @test abs(w_f64 - w_cpu) < 1e-10
        end
        
        @testset "Extreme Values" begin
            basis = QuadPairBasis(1)
            
            # Very large amplitude
            α_large = 10.0f0 + 5.0f0im
            gpu_large = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, α_large)
            cpu_large = coherentstate(basis, α_large)
            
            x = [0.0f0, 0.0f0]
            w_gpu = wigner(gpu_large, CuArray(x))
            w_cpu = wigner(cpu_large, x)
            
            @test w_gpu ≈ w_cpu rtol=1e-5
            @test isfinite(w_gpu)
        end
    end
end

end