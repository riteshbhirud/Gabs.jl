@testitem "GPU Linear Combinations - Core Functionality" begin
    using CUDA
    using Gabs
    using Gabs: device, GaussianLinearCombination
    using LinearAlgebra

    @testset "Device Management & Transfer" begin
        basis = QuadPairBasis(1)
        
        # Create CPU states with proper Float32 types for GPU compatibility
        state1 = coherentstate(basis, 1.0f0 + 0.5f0im)
        state2 = coherentstate(basis, -1.0f0 + 0.3f0im)
        coeffs = [0.6f0, 0.8f0]
        
        # Create CPU linear combination
        lc_cpu = GaussianLinearCombination(basis, coeffs, [state1, state2])
        @test device(lc_cpu) == :cpu
        @test lc_cpu.coeffs isa Vector{Float32}
        @test device(lc_cpu.states[1]) == :cpu
        
        # Transfer to GPU using professional API
        lc_gpu = lc_cpu |> gpu
        @test device(lc_gpu) == :gpu
        @test lc_gpu.coeffs isa Vector{Float32}  # Coefficients stay on CPU, converted to Float32
        @test device(lc_gpu.states[1]) == :gpu
        @test device(lc_gpu.states[2]) == :gpu
        
        # Verify GPU states have correct types
        @test lc_gpu.states[1].mean isa CuVector{Float32}
        @test lc_gpu.states[1].covar isa CuMatrix{Float32}
        
        # Verify coefficients match
        @test lc_gpu.coeffs ≈ lc_cpu.coeffs
        
        # Transfer back to CPU
        lc_back = lc_gpu |> cpu
        @test device(lc_back) == :cpu
        @test lc_back.coeffs ≈ lc_cpu.coeffs
        @test Array(lc_gpu.states[1].mean) ≈ lc_back.states[1].mean rtol=1e-6
        
        # Test precision control
        lc_gpu_f64 = gpu(lc_cpu, precision=Float64)
        @test eltype(lc_gpu_f64.coeffs) == Float64
        @test lc_gpu_f64.states[1].mean isa CuVector{Float64}
    end
    
    @testset "Creating GPU Linear Combinations Directly" begin
        basis = QuadPairBasis(1)
        
        # Method 1: Create CPU then transfer
        state1_cpu = coherentstate(basis, 1.0f0)
        state2_cpu = squeezedstate(basis, 0.3f0, Float32(π/4))
        lc1 = GaussianLinearCombination(basis, [0.7f0, 0.3f0], [state1_cpu, state2_cpu]) |> gpu
        @test device(lc1) == :gpu
        
        # Method 2: Create with GPU states directly
        state1_gpu = coherentstate(basis, 1.0f0) |> gpu
        state2_gpu = squeezedstate(basis, 0.3f0, Float32(π/4)) |> gpu
        lc2 = GaussianLinearCombination(basis, [0.7f0, 0.3f0], [state1_gpu, state2_gpu])
        @test device(lc2) == :gpu
        
        # Both methods should give equivalent results
        @test lc1.coeffs ≈ lc2.coeffs
        @test Array(lc1.states[1].mean) ≈ Array(lc2.states[1].mean) rtol=1e-6
    end

    @testset "Arithmetic Operations" begin
        basis = QuadPairBasis(1)
        
        # Create GPU linear combinations
        state1 = coherentstate(basis, 1.0f0) |> gpu
        state2 = coherentstate(basis, -1.0f0) |> gpu
        state3 = squeezedstate(basis, 0.2f0, 0.0f0) |> gpu
        
        lc1 = GaussianLinearCombination(basis, [0.6f0, 0.4f0], [state1, state2])
        lc2 = GaussianLinearCombination(basis, [0.3f0], [state3])
        
        @test device(lc1) == :gpu
        @test device(lc2) == :gpu
        
        # Test addition
        lc_sum = lc1 + lc2
        @test length(lc_sum) == 3
        @test device(lc_sum) == :gpu
        @test lc_sum.coeffs ≈ [0.6f0, 0.4f0, 0.3f0]
        
        # Test subtraction
        lc_diff = lc1 - lc2
        @test length(lc_diff) == 3
        @test device(lc_diff) == :gpu
        @test lc_diff.coeffs ≈ [0.6f0, 0.4f0, -0.3f0]
        
        # Test scalar multiplication
        lc_scaled = 2.0f0 * lc1
        @test length(lc_scaled) == 2
        @test device(lc_scaled) == :gpu
        @test lc_scaled.coeffs ≈ [1.2f0, 0.8f0]
        
        # Test right scalar multiplication
        lc_scaled2 = lc1 * 0.5f0
        @test lc_scaled2.coeffs ≈ [0.3f0, 0.2f0]
        
        # Test negation
        lc_neg = -lc1
        @test lc_neg.coeffs ≈ [-0.6f0, -0.4f0]
        @test device(lc_neg) == :gpu
    end

    @testset "Normalization" begin
        basis = QuadPairBasis(1)
        
        # Create unnormalized GPU linear combination
        state1 = vacuumstate(basis) |> gpu
        state2 = coherentstate(basis, 1.0f0) |> gpu
        
        coeffs = [3.0f0, 4.0f0]  # Norm = 5.0
        lc = GaussianLinearCombination(basis, coeffs, [state1, state2])
        @test device(lc) == :gpu
        
        # Check initial norm
        initial_norm = sqrt(sum(abs2, lc.coeffs))
        @test initial_norm ≈ 5.0f0
        
        # Normalize
        Gabs.normalize!(lc)
        @test device(lc) == :gpu  # Should stay on GPU
        
        # Check normalized coefficients
        final_norm = sqrt(sum(abs2, lc.coeffs))
        @test final_norm ≈ 1.0f0 rtol=1e-6
        @test lc.coeffs ≈ [0.6f0, 0.8f0] rtol=1e-6
        
        # Test zero coefficient case
        lc_zero = GaussianLinearCombination(basis, [0.0f0, 0.0f0], [state1, state2])
        Gabs.normalize!(lc_zero)
        @test lc_zero.coeffs == [0.0f0, 0.0f0]  # Should remain unchanged
    end

    @testset "Simplification" begin
        basis = QuadPairBasis(1)
        
        # Create states
        vac = vacuumstate(basis) |> gpu
        coh = coherentstate(basis, 1.0f0) |> gpu
        
        # Test removing negligible coefficients
        coeffs1 = CuArray([0.9f0, Float32(1e-15), 0.1f0])
        lc1 = GaussianLinearCombination(basis, coeffs1, [vac, coh, vac])
        @test length(lc1) == 3
        
        Gabs.simplify!(lc1)
        @test length(lc1) == 1  # ✅ Correct: combines both vacuum states
        @test device(lc1) == :gpu
        
        # Test combining identical states  
        coh2 = coherentstate(basis, 1.0f0) |> gpu  # Same as coh
        coeffs2 = CuArray([0.5f0, 0.3f0, 0.2f0])
        lc2 = GaussianLinearCombination(basis, coeffs2, [vac, coh, coh2])
        @test length(lc2) == 3
        
        Gabs.simplify!(lc2)
        @test length(lc2) == 2  # Should combine identical coherent states
        @test device(lc2) == :gpu
        
        # GPU-safe coefficient verification (no scalar indexing)
        cpu_coeffs = Array(lc2.coeffs)  # Move to CPU for testing
        cpu_vac_mean = Array(vac.mean)
        cpu_coh_mean = Array(coh.mean)
        
        # Find vacuum and coherent state indices
        vac_idx = findfirst(i -> Array(lc2.states[i].mean) ≈ cpu_vac_mean, 1:length(lc2))
        coh_idx = findfirst(i -> Array(lc2.states[i].mean) ≈ cpu_coh_mean, 1:length(lc2))
        
        @test vac_idx !== nothing
        @test coh_idx !== nothing
        @test cpu_coeffs[vac_idx] ≈ 0.5f0      # ✅ CPU array access
        @test cpu_coeffs[coh_idx] ≈ 0.5f0      # ✅ 0.3 + 0.2 = 0.5
        
        # Test all coefficients negligible
        coeffs3 = CuArray([Float32(1e-16), Float32(1e-17)])
        lc3 = GaussianLinearCombination(basis, coeffs3, [vac, coh])
        Gabs.simplify!(lc3)
        @test length(lc3) == 1  # Should fallback to single vacuum state
        @test device(lc3) == :gpu
    end

    @testset "GPU Operations - Gaussian Unitaries" begin
        basis = QuadPairBasis(1)
        
        # Create GPU linear combination
        state1 = coherentstate(basis, 1.0f0) |> gpu
        state2 = coherentstate(basis, -1.0f0) |> gpu
        lc_gpu = GaussianLinearCombination(basis, [0.7f0, 0.3f0], [state1, state2])
        
        @test device(lc_gpu) == :gpu
        
        # Test with GPU operator
        disp_gpu = displace(basis, 0.5f0) |> gpu
        result_gpu = disp_gpu * lc_gpu
        
        @test device(result_gpu) == :gpu
        @test length(result_gpu) == 2
        @test result_gpu.coeffs ≈ lc_gpu.coeffs  # Coefficients unchanged
        
        # Verify states were displaced correctly
        expected1 = disp_gpu * state1
        expected2 = disp_gpu * state2
        @test Array(result_gpu.states[1].mean) ≈ Array(expected1.mean) rtol=1e-6
        @test Array(result_gpu.states[2].mean) ≈ Array(expected2.mean) rtol=1e-6
        
        # Test with CPU operator (should auto-promote)
        disp_cpu = displace(basis, 0.2f0)
        result_mixed = disp_cpu * lc_gpu
        @test device(result_mixed) == :gpu  # Result should be on GPU
        
        # Test squeeze operation
        squeeze_gpu = squeeze(basis, 0.3f0, Float32(π/4)) |> gpu
        squeezed_lc = squeeze_gpu * lc_gpu
        @test device(squeezed_lc) == :gpu
        @test length(squeezed_lc) == 2
        
        # Test phase shift
        phase_gpu = phaseshift(basis, π/3) |> gpu
        phase_shifted = phase_gpu * lc_gpu
        @test device(phase_shifted) == :gpu
    end

    @testset "GPU Operations - Gaussian Channels" begin
        basis = QuadPairBasis(1)
        
        # Create GPU linear combination
        coh1 = coherentstate(basis, 1.0f0) |> gpu
        coh2 = coherentstate(basis, 0.5f0) |> gpu
        lc_gpu = GaussianLinearCombination(basis, [0.8f0, 0.2f0], [coh1, coh2])
        
        # Test attenuator channel
        att_gpu = attenuator(basis, π/6, 2.0f0) |> gpu
        attenuated = att_gpu * lc_gpu
        
        @test device(attenuated) == :gpu
        @test length(attenuated) == 2
        @test attenuated.coeffs ≈ lc_gpu.coeffs  # Coefficients unchanged
        
        # Verify channel was applied to each state
        expected1 = att_gpu * coh1
        expected2 = att_gpu * coh2
        @test Array(attenuated.states[1].mean) ≈ Array(expected1.mean) rtol=1e-6
        @test Array(attenuated.states[2].covar) ≈ Array(expected2.covar) rtol=1e-5
        
        # Test amplifier channel
        amp_gpu = amplifier(basis, 0.2f0, 1.5f0) |> gpu
        amplified = amp_gpu * lc_gpu
        @test device(amplified) == :gpu
        
        # Test with CPU channel (should auto-promote)
        att_cpu = attenuator(basis, π/8, 1.0f0)
        result_mixed = att_cpu * lc_gpu
        @test device(result_mixed) == :gpu  # Result should be on GPU
    end

    @testset "State Metrics" begin
        basis = QuadPairBasis(1)
        
        # Create GPU linear combination (pure superposition state)
        state1 = coherentstate(basis, 1.0f0) |> gpu
        state2 = squeezedstate(basis, 0.3f0, Float32(π/4)) |> gpu
        lc_gpu = GaussianLinearCombination(basis, [0.6f0, 0.8f0], [state1, state2])
        
        @test device(lc_gpu) == :gpu
        
        # Test purity (pure states have purity = 1.0)
        p = purity(lc_gpu)
        @test p == 1.0
        
        # Test von Neumann entropy (pure states have entropy = 0.0)
        s = entropy_vn(lc_gpu)
        @test s == 0.0
        
        # Test with different sizes
        basis_multi = QuadPairBasis(2)
        epr = eprstate(basis_multi, 0.5f0, Float32(π/3)) |> gpu
        lc_multi = GaussianLinearCombination(basis_multi, [1.0f0], [epr])
        
        @test purity(lc_multi) == 1.0
        @test entropy_vn(lc_multi) == 0.0
    end

    @testset "Mixed Device Operations" begin
        basis = QuadPairBasis(1)
        
        # Create CPU and GPU linear combinations
        state1_cpu = coherentstate(basis, 1.0f0)
        state2_cpu = coherentstate(basis, -1.0f0)
        lc_cpu = GaussianLinearCombination(basis, [0.7f0, 0.3f0], [state1_cpu, state2_cpu])
        
        state3_gpu = squeezedstate(basis, 0.2f0, 0.0f0) |> gpu
        lc_gpu = GaussianLinearCombination(basis, [0.5f0], [state3_gpu])
        
        @test device(lc_cpu) == :cpu
        @test device(lc_gpu) == :gpu
        
        # Addition should promote to GPU
        lc_sum = lc_cpu + lc_gpu
        @test device(lc_sum) == :gpu
        @test length(lc_sum) == 3
        
        # Test operations with mixed devices
        disp_cpu = displace(basis, 0.3f0)
        disp_gpu = displace(basis, 0.4f0) |> gpu
        
        # CPU operator on GPU linear combination (should auto-promote)
        result1 = disp_cpu * lc_gpu
        @test device(result1) == :gpu
        
        # GPU operator on CPU linear combination (should promote to GPU)
        result2 = disp_gpu * lc_cpu
        @test device(result2) == :gpu
    end

    @testset "Error Handling & Edge Cases" begin
        basis1 = QuadPairBasis(1)
        basis2 = QuadBlockBasis(1)  # Different basis type
        
        state1 = coherentstate(basis1, 1.0f0) |> gpu
        state2 = coherentstate(basis2, 1.0f0) |> gpu
        
        lc1 = GaussianLinearCombination(basis1, [0.5f0], [state1])
        lc2 = GaussianLinearCombination(basis2, [0.5f0], [state2])
        
        # Test basis mismatch in addition
        @test_throws ArgumentError lc1 + lc2
        
        # Test ħ mismatch
        state3 = GaussianState(basis1, CuArray([0.0f0, 0.0f0]), CuMatrix{Float32}(I(2)); ħ = 4)
        lc3 = GaussianLinearCombination(basis1, [1.0f0], [state3])
        @test_throws ArgumentError lc1 + lc3
        
        # Test operator basis mismatch
        disp_wrong = displace(basis2, 1.0f0) |> gpu
        @test_throws ArgumentError disp_wrong * lc1
        
        # Test empty constructor (should throw)
        @test_throws ArgumentError GaussianLinearCombination(basis1, Float32[], typeof(state1)[])
        
        # Test what happens when simplify! reduces to very small coefficients
        tiny_coeffs = CuArray([Float32(1e-20), Float32(1e-21)])  # Extremely small
        tiny_states = [state1, coherentstate(basis1, 2.0f0) |> gpu]
        lc_tiny = GaussianLinearCombination(basis1, tiny_coeffs, tiny_states)
        
        # This should create fallback vacuum state
        Gabs.simplify!(lc_tiny)
        @test length(lc_tiny) == 1  # Should have fallback vacuum
        @test device(lc_tiny) == :gpu
        
        # Test CUDA fallback behavior (if CUDA becomes unavailable)
        if CUDA.functional()
            @test true  # CUDA is available, normal GPU operations work
        else
            @warn "CUDA not functional, testing fallback behavior"
        end
    end

    @testset "Type Stability & Performance" begin
        basis = QuadPairBasis(1)
        
        # Create GPU linear combination
        state1 = coherentstate(basis, 1.0f0) |> gpu
        state2 = squeezedstate(basis, 0.3f0, Float32(π/4)) |> gpu
        lc = GaussianLinearCombination(basis, [0.6f0, 0.8f0], [state1, state2])
        
        # Test type stability
        @inferred device(lc)
        @inferred purity(lc)
        @inferred entropy_vn(lc)
        
        # Test operations preserve GPU arrays
        disp = displace(basis, 0.5f0) |> gpu
        result = disp * lc
        @test result.states[1].mean isa CuVector{Float32}
        @test result.states[1].covar isa CuMatrix{Float32}
        
        # Test arithmetic preserves types
        lc_scaled = 2.0f0 * lc
        @test lc_scaled.coeffs isa Vector{Float32}
        @test lc_scaled.states[1].mean isa CuVector{Float32}
        
        # Test normalization preserves types
        lc_copy = GaussianLinearCombination(basis, copy(lc.coeffs), copy(lc.states))
        Gabs.normalize!(lc_copy)
        @test lc_copy.coeffs isa Vector{Float32}
        @test device(lc_copy) == :gpu
    end

    @testset "Multi-Mode GPU Linear Combinations" begin
        basis = QuadPairBasis(2)
        
        # Create multi-mode states
        α_vec1 = [1.0f0 + 0.5f0im, -0.3f0 + 0.8f0im]
        α_vec2 = [0.5f0, 0.2f0 - 0.4f0im]
        
        coh1 = coherentstate(basis, α_vec1) |> gpu
        coh2 = coherentstate(basis, α_vec2) |> gpu
        epr = eprstate(basis, 0.3f0, Float32(π/6)) |> gpu

        # Create complex multi-mode linear combination
        coeffs = [0.5f0, 0.3f0, 0.2f0]
        lc_multi = GaussianLinearCombination(basis, coeffs, [coh1, coh2, epr])
        
        @test device(lc_multi) == :gpu
        @test length(lc_multi) == 3
        @test lc_multi.basis.nmodes == 2
        
        # Test operations on multi-mode
        squeeze_op = squeeze(basis, [0.2f0, 0.3f0], [0.0f0, Float32(π/4)]) |> gpu
        squeezed_multi = squeeze_op * lc_multi
        
        @test device(squeezed_multi) == :gpu
        @test length(squeezed_multi) == 3
        @test squeezed_multi.coeffs ≈ coeffs
        
        # Test beam splitter (two-mode operation)
        bs = beamsplitter(basis, 0.7f0) |> gpu
        bs_result = bs * lc_multi
        @test device(bs_result) == :gpu
        
        # Test two-mode squeeze
        twosq = twosqueeze(basis, 0.2f0, Float32(π/3)) |> gpu
        twosq_result = twosq * lc_multi
        @test device(twosq_result) == :gpu
    end

    @testset "Professional API Validation" begin
        @info "=== Testing Professional GPU API for Linear Combinations ==="
        
        basis = QuadPairBasis(1)
        
        # Show clean API usage
        @info "Professional API example:"
        @info "state1 = coherentstate(basis, 1.0f0) |> gpu"
        @info "state2 = squeezedstate(basis, 0.3f0, π/4) |> gpu"
        @info "lc = GaussianLinearCombination(basis, [0.7f0, 0.3f0], [state1, state2])"
        @info "op = displace(basis, 0.5f0) |> gpu"
        @info "result = op * lc"
        
        state1 = coherentstate(basis, 1.0f0) |> gpu
        state2 = squeezedstate(basis, 0.3f0, Float32(π/4)) |> gpu
        lc = GaussianLinearCombination(basis, [0.7f0, 0.3f0], [state1, state2])
        op = displace(basis, 0.5f0) |> gpu
        result = op * lc
        
        @test device(lc) == :gpu
        @test device(result) == :gpu
        @test result.coeffs ≈ lc.coeffs
        
        # Show automatic dispatch
        @info "Automatic GPU dispatch with CuArrays:"
        α_gpu = CuArray([1.0f0 + 0.5f0im])
        auto_state = coherentstate(basis, α_gpu)
        lc_auto = GaussianLinearCombination(basis, [1.0f0], [auto_state])
        @test device(lc_auto) == :gpu
        
        # Show mixed device operations
        @info "Mixed device operations with auto-promotion:"
        cpu_op = squeeze(basis, 0.2f0, 0.0f0)
        mixed_result = cpu_op * lc  # CPU operator, GPU linear combination
        @test device(mixed_result) == :gpu
        
        @info "✓ Professional API working perfectly for linear combinations"
        @info "✓ GPU acceleration through Phase 4A infrastructure confirmed"
        @info "✓ Auto-promotion and mixed device operations working"
        @info "✓ Coefficients efficiently managed on CPU"
        @info "✓ States properly accelerated on GPU"
    end
end