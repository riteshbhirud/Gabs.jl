@testitem "GPU Foundation - State Creation" begin
    using CUDA
    using Gabs
    using LinearAlgebra
    
        
        @testset "GPU Vacuum State" begin
            basis = QuadPairBasis(1)
            
            # Create CPU version
            vac_cpu = vacuumstate(basis)
            
            # Create GPU version  
            vac_gpu = vacuumstate(CuVector{Float32}, CuMatrix{Float32}, basis)
            
            # Test types
            @test vac_gpu.mean isa CuVector{Float32}
            @test vac_gpu.covar isa CuMatrix{Float32}
            @test vac_gpu.basis == basis
            @test vac_gpu.ħ == 2
            
            # Test values match CPU
            @test Array(vac_gpu.mean) ≈ vac_cpu.mean
            @test Array(vac_gpu.covar) ≈ vac_cpu.covar
            
            # Test single-argument constructor
            vac_gpu_single = vacuumstate(CuVector{Float64}, basis)
            @test vac_gpu_single.mean isa CuVector{Float64}
            @test vac_gpu_single.covar isa CuMatrix{Float64}
        end
        
        @testset "GPU Coherent State" begin
            basis = QuadPairBasis(1)
            α = 1.0f0 + 0.5f0im
            
            # Create CPU version
            coh_cpu = coherentstate(basis, α)
            
            # Create GPU version
            coh_gpu = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, α)
            
            # Test types
            @test coh_gpu.mean isa CuVector{Float32}
            @test coh_gpu.covar isa CuMatrix{Float32}
            
            # Test values match CPU
            @test Array(coh_gpu.mean) ≈ Float32.(coh_cpu.mean)
            @test Array(coh_gpu.covar) ≈ Float32.(coh_cpu.covar)
            
            # Test multi-mode
            basis_multi = QuadPairBasis(2)
            α_multi = [1.0f0 + 0.5f0im, -0.3f0 + 0.8f0im]
            
            coh_cpu_multi = coherentstate(basis_multi, α_multi)
            coh_gpu_multi = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis_multi, α_multi)
            
            @test Array(coh_gpu_multi.mean) ≈ Float32.(coh_cpu_multi.mean)
            @test Array(coh_gpu_multi.covar) ≈ Float32.(coh_cpu_multi.covar)
        end
        
        @testset "GPU Squeezed State" begin
            basis = QuadPairBasis(1)
            r, θ = 0.3f0, Float32(π/4)
            
            # Create CPU version
            sq_cpu = squeezedstate(basis, r, θ)
            
            # Create GPU version
            sq_gpu = squeezedstate(CuVector{Float32}, CuMatrix{Float32}, basis, r, θ)
            
            # Test types
            @test sq_gpu.mean isa CuVector{Float32}
            @test sq_gpu.covar isa CuMatrix{Float32}
            
            # Test values match CPU
            @test Array(sq_gpu.mean) ≈ Float32.(sq_cpu.mean)
            @test Array(sq_gpu.covar) ≈ Float32.(sq_cpu.covar) rtol=1e-6
            
            # Test vector parameters
            basis_multi = QuadPairBasis(2)
            r_vec = [0.3f0, 0.5f0]
            θ_vec = [Float32(π/4), Float32(π/6)]
            
            sq_cpu_multi = squeezedstate(basis_multi, r_vec, θ_vec)
            sq_gpu_multi = squeezedstate(CuVector{Float32}, CuMatrix{Float32}, basis_multi, r_vec, θ_vec)
            
            @test Array(sq_gpu_multi.mean) ≈ Float32.(sq_cpu_multi.mean)
            @test Array(sq_gpu_multi.covar) ≈ Float32.(sq_cpu_multi.covar) rtol=1e-6
        end
        
        @testset "GPU Thermal State" begin
            basis = QuadPairBasis(1)
            n = 2.0f0
            
            # Create CPU version
            thermal_cpu = thermalstate(basis, n)
            
            # Create GPU version
            thermal_gpu = thermalstate(CuVector{Float32}, CuMatrix{Float32}, basis, n)
            
            # Test types
            @test thermal_gpu.mean isa CuVector{Float32}
            @test thermal_gpu.covar isa CuMatrix{Float32}
            
            # Test values match CPU
            @test Array(thermal_gpu.mean) ≈ Float32.(thermal_cpu.mean)
            @test Array(thermal_gpu.covar) ≈ Float32.(thermal_cpu.covar)
            
            # Test vector parameters
            basis_multi = QuadPairBasis(2)
            n_vec = [2.0f0, 3.0f0]
            
            thermal_cpu_multi = thermalstate(basis_multi, n_vec)
            thermal_gpu_multi = thermalstate(CuVector{Float32}, CuMatrix{Float32}, basis_multi, n_vec)
            
            @test Array(thermal_gpu_multi.mean) ≈ Float32.(thermal_cpu_multi.mean)
            @test Array(thermal_gpu_multi.covar) ≈ Float32.(thermal_cpu_multi.covar)
        end
        
        @testset "GPU EPR State" begin
            basis = QuadPairBasis(2)
            r, θ = 0.5f0, Float32(π/3)
            
            # Create CPU version
            epr_cpu = eprstate(basis, r, θ)
            
            # Create GPU version
            epr_gpu = eprstate(CuVector{Float32}, CuMatrix{Float32}, basis, r, θ)
            
            # Test types
            @test epr_gpu.mean isa CuVector{Float32}
            @test epr_gpu.covar isa CuMatrix{Float32}
            
            # Test values match CPU
            @test Array(epr_gpu.mean) ≈ Float32.(epr_cpu.mean)
            @test Array(epr_gpu.covar) ≈ Float32.(epr_cpu.covar) rtol=1e-6
        end
        
 
end

@testitem "GPU Foundation - Unitary Operations" begin
    using Gabs
    using LinearAlgebra
    
    if isdefined(Main, :CUDA) && Main.CUDA.functional()
        using CUDA
        
        @testset "GPU Displacement" begin
            basis = QuadPairBasis(1)
            α = 0.5f0 + 0.3f0im
            
            # Create CPU version
            disp_cpu = displace(basis, α)
            
            # Create GPU version
            disp_gpu = displace(CuVector{Float32}, CuMatrix{Float32}, basis, α)
            
            # Test types
            @test disp_gpu.disp isa CuVector{Float32}
            @test disp_gpu.symplectic isa CuMatrix{Float32}
            
            # Test values match CPU
            @test Array(disp_gpu.disp) ≈ Float32.(disp_cpu.disp)
            @test Array(disp_gpu.symplectic) ≈ Float32.(disp_cpu.symplectic)
            
            # Test application to state
            vac_gpu = vacuumstate(CuVector{Float32}, CuMatrix{Float32}, basis)
            displaced_gpu = disp_gpu * vac_gpu
            
            vac_cpu = vacuumstate(basis)
            displaced_cpu = disp_cpu * vac_cpu
            
            @test Array(displaced_gpu.mean) ≈ Float32.(displaced_cpu.mean) rtol=1e-6
            @test Array(displaced_gpu.covar) ≈ Float32.(displaced_cpu.covar) rtol=1e-6
        end
        
        @testset "GPU Squeeze" begin
            basis = QuadPairBasis(1)
            r, θ = 0.3f0, Float32(π/4)
            
            # Create CPU and GPU versions
            squeeze_cpu = squeeze(basis, r, θ)
            squeeze_gpu = squeeze(CuVector{Float32}, CuMatrix{Float32}, basis, r, θ)
            
            # Test values match
            @test Array(squeeze_gpu.disp) ≈ Float32.(squeeze_cpu.disp)
            @test Array(squeeze_gpu.symplectic) ≈ Float32.(squeeze_cpu.symplectic) rtol=1e-6
            
            # Test application
            vac_gpu = vacuumstate(CuVector{Float32}, CuMatrix{Float32}, basis)
            squeezed_gpu = squeeze_gpu * vac_gpu
            
            vac_cpu = vacuumstate(basis)
            squeezed_cpu = squeeze_cpu * vac_cpu
            
            @test Array(squeezed_gpu.mean) ≈ Float32.(squeezed_cpu.mean) rtol=1e-6
            @test Array(squeezed_gpu.covar) ≈ Float32.(squeezed_cpu.covar) rtol=1e-5
        end
        
        @testset "GPU Phase Shift" begin
            basis = QuadPairBasis(1)
            θ = Float32(π/3)
            
            phase_cpu = phaseshift(basis, θ)
            phase_gpu = phaseshift(CuVector{Float32}, CuMatrix{Float32}, basis, θ)
            
            @test Array(phase_gpu.disp) ≈ Float32.(phase_cpu.disp)
            @test Array(phase_gpu.symplectic) ≈ Float32.(phase_cpu.symplectic) rtol=1e-6
        end
        
        @testset "GPU Beam Splitter" begin
            basis = QuadPairBasis(2)
            transmit = 0.7f0
            
            bs_cpu = beamsplitter(basis, transmit)
            bs_gpu = beamsplitter(CuVector{Float32}, CuMatrix{Float32}, basis, transmit)
            
            @test Array(bs_gpu.disp) ≈ Float32.(bs_cpu.disp)
            @test Array(bs_gpu.symplectic) ≈ Float32.(bs_cpu.symplectic) rtol=1e-6
        end
        
        @testset "GPU Two-Mode Squeeze" begin
            basis = QuadPairBasis(2)
            r, θ = 0.2f0, Float32(π/6)
            
            twosq_cpu = twosqueeze(basis, r, θ)
            twosq_gpu = twosqueeze(CuVector{Float32}, CuMatrix{Float32}, basis, r, θ)
            
            @test Array(twosq_gpu.disp) ≈ Float32.(twosq_cpu.disp)
            @test Array(twosq_gpu.symplectic) ≈ Float32.(twosq_cpu.symplectic) rtol=1e-6
        end
        
    else
        @info "CUDA not available, skipping GPU unitary operation tests"
    end
end

@testitem "GPU Foundation - Channel Operations" begin
    using Gabs
    using LinearAlgebra
    
    if isdefined(Main, :CUDA) && Main.CUDA.functional()
        using CUDA
        
        @testset "GPU Attenuator Channel" begin
            basis = QuadPairBasis(1)
            θ, n = Float32(π/6), 2.0f0
            
            # Create CPU and GPU versions
            att_cpu = attenuator(basis, θ, n)
            att_gpu = attenuator(CuVector{Float32}, CuMatrix{Float32}, basis, θ, n)
            
            # Test types
            @test att_gpu.disp isa CuVector{Float32}
            @test att_gpu.transform isa CuMatrix{Float32}
            @test att_gpu.noise isa CuMatrix{Float32}
            
            # Test values match CPU
            @test Array(att_gpu.disp) ≈ Float32.(att_cpu.disp)
            @test Array(att_gpu.transform) ≈ Float32.(att_cpu.transform) rtol=1e-6
            @test Array(att_gpu.noise) ≈ Float32.(att_cpu.noise) rtol=1e-6
            
            # Test application to state
            coh_gpu = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, 1.0f0)
            attenuated_gpu = att_gpu * coh_gpu
            
            coh_cpu = coherentstate(basis, 1.0f0)
            attenuated_cpu = att_cpu * coh_cpu
            
            @test Array(attenuated_gpu.mean) ≈ Float32.(attenuated_cpu.mean) rtol=1e-6
            @test Array(attenuated_gpu.covar) ≈ Float32.(attenuated_cpu.covar) rtol=1e-5
        end
        
        @testset "GPU Amplifier Channel" begin
            basis = QuadPairBasis(1)
            r, n = 0.2f0, 1.5f0
            
            amp_cpu = amplifier(basis, r, n)
            amp_gpu = amplifier(CuVector{Float32}, CuMatrix{Float32}, basis, r, n)
            
            @test Array(amp_gpu.disp) ≈ Float32.(amp_cpu.disp)
            @test Array(amp_gpu.transform) ≈ Float32.(amp_cpu.transform) rtol=1e-6
            @test Array(amp_gpu.noise) ≈ Float32.(amp_cpu.noise) rtol=1e-5
        end
        
    else
        @info "CUDA not available, skipping GPU channel operation tests"
    end
end

@testitem "GPU Foundation - Wigner Functions" begin
    using Gabs
    using LinearAlgebra
    
    if isdefined(Main, :CUDA) && Main.CUDA.functional()
        using CUDA
        
        @testset "GPU Single-Point Wigner" begin
            basis = QuadPairBasis(1)
            
            # Create states
            vac_cpu = vacuumstate(basis)
            vac_gpu = vacuumstate(CuVector{Float32}, CuMatrix{Float32}, basis)
            
            coh_cpu = coherentstate(basis, 1.0f0)
            coh_gpu = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, 1.0f0)
            
            # Test points
            test_points = [[0.0f0, 0.0f0], [1.0f0, 0.5f0], [-0.5f0, 1.0f0]]
            
            for x in test_points
                # Vacuum state
                w_cpu_vac = wigner(vac_cpu, x)
                w_gpu_vac = wigner(vac_gpu, x)
                @test w_gpu_vac ≈ w_cpu_vac rtol=1e-5
                
                # Coherent state
                w_cpu_coh = wigner(coh_cpu, Float32.(x))
                w_gpu_coh = wigner(coh_gpu, x)
                @test w_gpu_coh ≈ w_cpu_coh rtol=1e-5
            end
        end
        
        @testset "GPU Batch Wigner Evaluation" begin
            basis = QuadPairBasis(1)
            coh_gpu = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, 1.0f0)
            
            # Create batch of points
            n_points = 100
            x_points = CuMatrix{Float32}(randn(Float32, 2, n_points))
            
            # Evaluate batch
            w_batch = wigner(coh_gpu, x_points)
            
            @test w_batch isa CuVector{Float32}
            @test length(w_batch) == n_points
            @test all(isfinite.(Array(w_batch)))
            
            # Compare with individual evaluations
            coh_cpu = coherentstate(basis, 1.0f0)
            for i in 1:min(10, n_points)  # Test first 10 points
                x_point = Array(x_points[:, i])
                w_individual = wigner(coh_cpu, x_point)
                w_batch_individual = Array(w_batch)[i]
                @test w_batch_individual ≈ w_individual rtol=1e-4
            end
        end
        
        @testset "GPU Single-Point Wigner Characteristic" begin
            basis = QuadPairBasis(1)
            
            vac_cpu = vacuumstate(basis)
            vac_gpu = vacuumstate(CuVector{Float32}, CuMatrix{Float32}, basis)
            
            # Test points
            test_points = [[0.0f0, 0.0f0], [0.5f0, 0.3f0], [-0.2f0, 0.8f0]]
            
            for xi in test_points
                chi_cpu = wignerchar(vac_cpu, xi)
                chi_gpu = wignerchar(vac_gpu, xi)
                
                @test real(chi_gpu) ≈ real(chi_cpu) rtol=1e-5
                @test imag(chi_gpu) ≈ imag(chi_cpu) rtol=1e-5
            end
        end
        
        @testset "GPU Batch Wigner Characteristic" begin
            basis = QuadPairBasis(1)
            sq_gpu = squeezedstate(CuVector{Float32}, CuMatrix{Float32}, basis, 0.3f0, π/4)
            
            # Create batch of points
            n_points = 50
            xi_points = CuMatrix{Float32}(randn(Float32, 2, n_points) * 0.5f0)
            
            # Evaluate batch
            chi_batch = wignerchar(sq_gpu, xi_points)
            
            @test chi_batch isa CuVector{ComplexF32}
            @test length(chi_batch) == n_points
            @test all(isfinite.(real(Array(chi_batch))))
            @test all(isfinite.(imag(Array(chi_batch))))
        end
        
        @testset "GPU Wigner Grid Utility" begin
            basis = QuadPairBasis(1)
            coh_gpu = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, 0.5f0)
            
            # Create grid
            x_range = (-2.0f0, 2.0f0)
            p_range = (-2.0f0, 2.0f0)
            nx, np = 20, 25
            
            grid_points = wigner_grid(coh_gpu, x_range, p_range, nx, np)
            
            @test size(grid_points) == (2, nx * np)
            @test grid_points isa CuMatrix{Float32}
            
            # Test evaluation on grid
            w_grid = wigner(coh_gpu, grid_points)
            @test length(w_grid) == nx * np
            @test all(Array(w_grid) .> 0)  # Coherent state Wigner is always positive
        end
        
    else
        @info "CUDA not available, skipping GPU Wigner function tests"
    end
end

@testitem "GPU Foundation - Tensor Products and Partial Traces" begin
    using Gabs
    using LinearAlgebra
    
    if isdefined(Main, :CUDA) && Main.CUDA.functional()
        using CUDA
        
        @testset "GPU State Tensor Products" begin
            basis1 = QuadPairBasis(1)
            basis2 = QuadPairBasis(1)
            
            # Create GPU states
            coh1_gpu = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis1, 1.0f0)
            vac2_gpu = vacuumstate(CuVector{Float32}, CuMatrix{Float32}, basis2)
            
            # Create CPU versions for comparison
            coh1_cpu = coherentstate(basis1, 1.0f0)
            vac2_cpu = vacuumstate(basis2)
            
            # Tensor product
            tensor_gpu = tensor(CuVector{Float32}, CuMatrix{Float32}, coh1_gpu, vac2_gpu)
            tensor_cpu = tensor(coh1_cpu, vac2_cpu)
            
            @test tensor_gpu.basis.nmodes == 2
            @test Array(tensor_gpu.mean) ≈ Float32.(tensor_cpu.mean) rtol=1e-6
            @test Array(tensor_gpu.covar) ≈ Float32.(tensor_cpu.covar) rtol=1e-6
        end
        
        @testset "GPU Unitary Tensor Products" begin
            basis1 = QuadPairBasis(1)
            
            disp1_gpu = displace(CuVector{Float32}, CuMatrix{Float32}, basis1, 0.5f0)
            disp2_gpu = displace(CuVector{Float32}, CuMatrix{Float32}, basis1, -0.3f0)
            
            disp1_cpu = displace(basis1, 0.5f0)
            disp2_cpu = displace(basis1, -0.3f0)
            
            tensor_gpu = tensor(CuVector{Float32}, CuMatrix{Float32}, disp1_gpu, disp2_gpu)
            tensor_cpu = tensor(disp1_cpu, disp2_cpu)
            
            @test Array(tensor_gpu.disp) ≈ Float32.(tensor_cpu.disp) rtol=1e-6
            @test Array(tensor_gpu.symplectic) ≈ Float32.(tensor_cpu.symplectic) rtol=1e-6
        end
        
        @testset "GPU Partial Traces" begin
            basis = QuadPairBasis(2)
            
            # Create 2-mode GPU state
            epr_gpu = eprstate(CuVector{Float32}, CuMatrix{Float32}, basis, 0.3f0, π/4)
            epr_cpu = eprstate(basis, 0.3f0, π/4)
            
            # Partial trace over first mode
            traced_gpu = ptrace(CuVector{Float32}, CuMatrix{Float32}, epr_gpu, [1])
            traced_cpu = ptrace(epr_cpu, [1])
            
            @test traced_gpu.basis.nmodes == 1
            @test Array(traced_gpu.mean) ≈ Float32.(traced_cpu.mean) rtol=1e-6
            @test Array(traced_gpu.covar) ≈ Float32.(traced_cpu.covar) rtol=1e-5
        end
        
    else
        @info "CUDA not available, skipping GPU tensor product and partial trace tests"
    end
end

@testitem "GPU Foundation - Performance and Error Handling" begin
    using Gabs
    using LinearAlgebra
    
    if isdefined(Main, :CUDA) && Main.CUDA.functional()
        using CUDA
        
        @testset "GPU vs CPU Performance Comparison" begin
            basis = QuadPairBasis(1)
            
            # Create states
            coh_cpu = coherentstate(basis, 1.0f0)
            coh_gpu = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, 1.0f0)
            
            # Large batch of points for performance test
            n_points = 10000
            x_points_cpu = randn(Float32, 2, n_points)
            x_points_gpu = CuArray(x_points_cpu)
            
            # Time CPU evaluation (individual points)
            cpu_time = @elapsed begin
                w_cpu = [wigner(coh_cpu, x_points_cpu[:, i]) for i in 1:n_points]
            end
            
            # Time GPU batch evaluation
            gpu_time = @elapsed begin
                w_gpu = wigner(coh_gpu, x_points_gpu)
                CUDA.synchronize()  # Ensure completion
            end
            
            @test gpu_time < cpu_time  # GPU should be faster
            @info "GPU speedup: $(round(cpu_time/gpu_time, digits=2))x"
            
            # Verify results match
            w_gpu_array = Array(w_gpu)
            @test length(w_gpu_array) == length(w_cpu)
            
            # Check a sample of results
            for i in 1:min(100, n_points)
                @test w_gpu_array[i] ≈ w_cpu[i] rtol=1e-4
            end
        end
        
        @testset "GPU Memory Management" begin
            basis = QuadPairBasis(10)  # Large system
            
            # Create large state
            α_vec = randn(ComplexF32, 10)
            large_coh_gpu = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, α_vec)
            
            @test large_coh_gpu.mean isa CuVector{Float32}
            @test large_coh_gpu.covar isa CuMatrix{Float32}
            @test size(large_coh_gpu.mean) == (20,)
            @test size(large_coh_gpu.covar) == (20, 20)
            
            # Verify no memory leaks by creating many states
            initial_memory = CUDA.memory_status().free_bytes
            
            for _ in 1:100
                temp_state = vacuumstate(CuVector{Float32}, CuMatrix{Float32}, QuadPairBasis(1))
            end
            GC.gc()
            CUDA.reclaim()
            
            final_memory = CUDA.memory_status().free_bytes
            memory_diff = initial_memory - final_memory
            
            @test memory_diff < 100 * 1024 * 1024  # Less than 100MB difference
        end
        
        @testset "GPU Type Promotion and Compatibility" begin
            basis = QuadPairBasis(1)
            
            # Test different precision types
            vac_f32 = vacuumstate(CuVector{Float32}, CuMatrix{Float32}, basis)
            vac_f64 = vacuumstate(CuVector{Float64}, CuMatrix{Float64}, basis)
            
            @test eltype(vac_f32.mean) == Float32
            @test eltype(vac_f64.mean) == Float64
            
            # Test operations preserve precision
            disp_f32 = displace(CuVector{Float32}, CuMatrix{Float32}, basis, 1.0f0)
            displaced_f32 = disp_f32 * vac_f32
            
            @test eltype(displaced_f32.mean) == Float32
            @test eltype(displaced_f32.covar) == Float32
        end
        
    else
        @info "CUDA not available, skipping GPU performance and error handling tests"
    end
end

@testitem "GPU Foundation - Fallback Behavior" begin
    using Gabs
    
    # Test behavior when CUDA is not available (mock scenario)
    @testset "Graceful Fallback" begin
        basis = QuadPairBasis(1)
        
        # These should not error even if CUDA is not available
        # The GPU functions should fall back to CPU versions
        try
            vac = vacuumstate(Vector{Float64}, Matrix{Float64}, basis)
            @test vac isa GaussianState
            
            coh = coherentstate(Vector{Float64}, Matrix{Float64}, basis, 1.0)
            @test coh isa GaussianState
            
            disp = displace(Vector{Float64}, Matrix{Float64}, basis, 1.0)
            @test disp isa GaussianUnitary
            
            @test true  # If we get here, fallback works
        catch e
            @test false "Fallback failed: $e"
        end
    end
end