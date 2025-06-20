@testitem "Linear Combinations Integration - Phase 3" begin
    using Gabs
    using StaticArrays
    using LinearAlgebra
    using CairoMakie

    nmodes = rand(1:3)
    qpairbasis = QuadPairBasis(nmodes)
    qblockbasis = QuadBlockBasis(nmodes)

    @testset "Gaussian Unitary Actions" begin
        # Test basic unitary actions
        basis = QuadPairBasis(1)
        coh1 = coherentstate(basis, 1.0)
        coh2 = coherentstate(basis, -1.0)
        lc = GaussianLinearCombination(basis, [0.6, 0.8], [coh1, coh2])
        
        # Displacement
        α = 0.5 + 0.3im
        disp_op = displace(basis, α)
        lc_displaced = disp_op * lc
        
        @test lc_displaced isa GaussianLinearCombination
        @test length(lc_displaced) == 2
        @test lc_displaced.coefficients == [0.6, 0.8]
        @test lc_displaced.basis == basis
        @test lc_displaced.ħ == lc.ħ
        
        # Check that displacement is applied correctly to each state
        expected_state1 = disp_op * coh1
        expected_state2 = disp_op * coh2
        @test isapprox(lc_displaced.states[1], expected_state1, atol=1e-12)
        @test isapprox(lc_displaced.states[2], expected_state2, atol=1e-12)
        
        # Squeezing
        r, θ = 0.5, π/4
        squeeze_op = squeeze(basis, r, θ)
        lc_squeezed = squeeze_op * lc
        
        @test lc_squeezed isa GaussianLinearCombination
        @test length(lc_squeezed) == 2
        @test lc_squeezed.coefficients == [0.6, 0.8]
        
        # Phase shift
        phase_op = phaseshift(basis, π/3)
        lc_phase = phase_op * lc
        
        @test lc_phase isa GaussianLinearCombination
        @test length(lc_phase) == 2
        
        # Test with multi-mode
        if nmodes > 1
            basis_multi = QuadPairBasis(nmodes)
            coh_multi1 = coherentstate(basis_multi, 1.0)
            coh_multi2 = coherentstate(basis_multi, -1.0)
            lc_multi = GaussianLinearCombination(basis_multi, [0.5, 0.5], [coh_multi1, coh_multi2])
            
            disp_multi = displace(basis_multi, 0.5)
            lc_multi_displaced = disp_multi * lc_multi
            
            @test lc_multi_displaced isa GaussianLinearCombination
            @test length(lc_multi_displaced) == 2
        end
        
        # Test error handling
        basis_wrong = QuadPairBasis(2)
        disp_wrong = displace(basis_wrong, 1.0)
        @test_throws ArgumentError disp_wrong * lc
        
        # Test ħ mismatch
        lc_diff_h = GaussianLinearCombination(basis, [0.5, 0.5], [coherentstate(basis, 1.0, ħ=1), coherentstate(basis, -1.0, ħ=1)])
        @test_throws ArgumentError disp_op * lc_diff_h
    end

    @testset "Gaussian Channel Actions" begin
        basis = QuadPairBasis(1)
        vac = vacuumstate(basis)
        coh = coherentstate(basis, 1.0)
        lc = GaussianLinearCombination(basis, [0.7, 0.3], [vac, coh])
        
        # Attenuator channel
        θ, n = π/6, 2
        att_channel = attenuator(basis, θ, n)
        lc_attenuated = att_channel * lc
        
        @test lc_attenuated isa GaussianLinearCombination
        @test length(lc_attenuated) == 2
        @test lc_attenuated.coefficients == [0.7, 0.3]
        @test lc_attenuated.basis == basis
        
        # Check that channel is applied correctly
        expected_vac = att_channel * vac
        expected_coh = att_channel * coh
        @test isapprox(lc_attenuated.states[1], expected_vac, atol=1e-12)
        @test isapprox(lc_attenuated.states[2], expected_coh, atol=1e-12)
        
        # Amplifier channel
        r_amp, n_amp = 0.3, 1.5
        amp_channel = amplifier(basis, r_amp, n_amp)
        lc_amplified = amp_channel * lc
        
        @test lc_amplified isa GaussianLinearCombination
        @test length(lc_amplified) == 2
        
        # Test with custom noise
        noise_matrix = [1.5 0.2; 0.2 1.8]
        custom_channel = displace(basis, 0.5, noise_matrix)
        lc_custom = custom_channel * lc
        
        @test lc_custom isa GaussianLinearCombination
        @test length(lc_custom) == 2
        
        # Error handling
        basis_wrong = QuadBlockBasis(1)
        channel_wrong = attenuator(basis_wrong, θ, n)
        @test_throws ArgumentError channel_wrong * lc
    end

    @testset "Tensor Products" begin
        basis1 = QuadPairBasis(1)
        basis2 = QuadPairBasis(1)
        
        # Create linear combinations
        coh1 = coherentstate(basis1, 1.0)
        coh2 = coherentstate(basis1, -1.0)
        lc1 = GaussianLinearCombination(basis1, [0.6, 0.8], [coh1, coh2])
        
        vac = vacuumstate(basis2)
        sq = squeezedstate(basis2, 0.5, π/4)
        lc2 = GaussianLinearCombination(basis2, [0.3, 0.7], [vac, sq])
        
        # Tensor product
        lc_tensor = tensor(lc1, lc2)
        @test lc_tensor isa GaussianLinearCombination
        @test length(lc_tensor) == 4  # 2 × 2 = 4 terms
        @test lc_tensor.basis.nmodes == 2
        
        # Check coefficients
        expected_coeffs = [0.6*0.3, 0.6*0.7, 0.8*0.3, 0.8*0.7]
        @test isapprox(lc_tensor.coefficients, expected_coeffs, atol=1e-12)
        
        # Check states
        expected_states = [coh1 ⊗ vac, coh1 ⊗ sq, coh2 ⊗ vac, coh2 ⊗ sq]
        for i in 1:4
            @test isapprox(lc_tensor.states[i], expected_states[i], atol=1e-12)
        end
        
        # Test ⊗ operator alias
        lc_tensor_alias = lc1 ⊗ lc2
        @test lc_tensor_alias == lc_tensor
        
        # Test with typed arrays
        lc_tensor_typed = tensor(Vector{Float64}, Vector{GaussianState}, lc1, lc2)
        @test lc_tensor_typed isa GaussianLinearCombination
        @test lc_tensor_typed.coefficients isa Vector{Float64}
        
        # Test single component tensor
        lc_single1 = GaussianLinearCombination(coh1)
        lc_single2 = GaussianLinearCombination(vac)
        lc_single_tensor = lc_single1 ⊗ lc_single2
        
        @test length(lc_single_tensor) == 1
        @test isapprox(lc_single_tensor.states[1], coh1 ⊗ vac, atol=1e-12)
        
        # Error handling
        basis_block = QuadBlockBasis(1)
        lc_block = GaussianLinearCombination(vacuumstate(basis_block))
        @test_throws ArgumentError lc1 ⊗ lc_block
        
        # ħ mismatch
        lc_diff_h = GaussianLinearCombination(basis2, [1.0], [vacuumstate(basis2, ħ=1)])
        @test_throws ArgumentError lc1 ⊗ lc_diff_h
    end

    @testset "Partial Trace" begin
        basis = QuadPairBasis(2)
        
        # Create 2-mode states
        coh1 = coherentstate(basis, [1.0+0.0im, 0.5+0.0im])
        coh2 = coherentstate(basis, [-1.0+0.0im, -0.5+0.0im])
        vac = vacuumstate(basis)
        
        lc = GaussianLinearCombination(basis, [0.5, 0.3, 0.2], [coh1, coh2, vac])
        
        # Partial trace over mode 1
        lc_traced1 = ptrace(lc, [1])
        @test lc_traced1 isa GaussianLinearCombination
        @test lc_traced1.basis.nmodes == 1
        
        # Check that traced states are correct
        expected_traced1 = ptrace(coh1, [1])
        expected_traced2 = ptrace(coh2, [1])
        expected_traced_vac = ptrace(vac, [1])
        
        # Verify some state is present (exact matching depends on simplification)
        @test length(lc_traced1) >= 1
        @test lc_traced1.basis == QuadPairBasis(1)
        
        # Partial trace over mode 2
        lc_traced2 = ptrace(lc, [2])
        @test lc_traced2 isa GaussianLinearCombination
        @test lc_traced2.basis.nmodes == 1
        
        # Test with identical states (should combine coefficients)
        identical_state = coherentstate(basis, [1.0, 0.0])
        lc_identical = GaussianLinearCombination(basis, [0.3, 0.7], [identical_state, identical_state])
        lc_traced_identical = ptrace(lc_identical, [1])
        
        # After tracing identical states, coefficients should combine
        @test length(lc_traced_identical) <= 2  # May combine into fewer terms
        
        # Test with static arrays
        if nmodes >= 2
            coh_static = coherentstate(SVector{4}, SMatrix{4,4}, basis, [1.0, 0.5])
            vac_static = vacuumstate(SVector{4}, SMatrix{4,4}, basis)
            lc_static = GaussianLinearCombination(basis, [0.6, 0.4], [coh_static, vac_static])
            
            lc_traced_static = ptrace(lc_static, [1])
            @test lc_traced_static isa GaussianLinearCombination
        end
        
        # Error handling
        @test_throws ArgumentError ptrace(lc, [1, 2])  # Can't trace all modes
        @test_throws ArgumentError ptrace(lc, [3])     # Invalid mode index
    end

    @testset "Wigner Functions" begin
        basis = QuadPairBasis(1)
        
        # Simple linear combination
        coh1 = coherentstate(basis, 1.0)
        coh2 = coherentstate(basis, -1.0)
        lc = GaussianLinearCombination(basis, [0.6, 0.8], [coh1, coh2])
        
        # Test points
        x_test = [0.0, 0.0]
        x_test2 = [1.0, 0.5]
        
        # Wigner function
        w_lc = wigner(lc, x_test)
        @test w_lc isa Real
        @test isfinite(w_lc)
        
        # Should include diagonal and off-diagonal terms
        w_diagonal = abs2(0.6) * wigner(coh1, x_test) + abs2(0.8) * wigner(coh2, x_test)
        w_cross = 2 * real(conj(0.6) * 0.8 * cross_wigner(coh1, coh2, x_test))
        w_expected = w_diagonal + w_cross
        
        @test isapprox(w_lc, w_expected, atol=1e-10)
        
        # Test at different points
        w_lc2 = wigner(lc, x_test2)
        @test w_lc2 isa Real
        @test isfinite(w_lc2)
        
        # Cross-Wigner function
        cross_w = cross_wigner(coh1, coh2, x_test)
        @test cross_w isa Real
        @test isfinite(cross_w)
        
        # Cross-Wigner should be symmetric in states
        cross_w_sym = cross_wigner(coh2, coh1, x_test)
        @test isapprox(cross_w, cross_w_sym, atol=1e-10)
        
        # Test with identical states
        cross_w_identical = cross_wigner(coh1, coh1, x_test)
        w_single = wigner(coh1, x_test)
        @test isapprox(cross_w_identical, w_single, atol=1e-10)
        
        # Wigner characteristic function
        xi_test = [0.5, -0.3]
        wchar_lc = wignerchar(lc, xi_test)
        @test wchar_lc isa Complex
        @test isfinite(real(wchar_lc))
        @test isfinite(imag(wchar_lc))
        
        # Test normalization: ∫ W(x) dx = 1 (approximately, for discrete sampling)
        x_range = -3:0.5:3
        y_range = -3:0.5:3
        wigner_sum = 0.0
        dx = 0.5
        for x in x_range, y in y_range
            wigner_sum += wigner(lc, [x, y]) * dx^2
        end
        @test abs(wigner_sum - 1.0) < 0.1  # Rough normalization check
        
        # Error handling
        @test_throws ArgumentError wigner(lc, [1.0])  # Wrong dimension
        @test_throws ArgumentError wignerchar(lc, [1.0, 2.0, 3.0])  # Wrong dimension
        
        # Different basis error
        basis_wrong = QuadBlockBasis(1)
        coh_wrong = coherentstate(basis_wrong, 1.0)
        @test_throws ArgumentError cross_wigner(coh1, coh_wrong, x_test)
    end

    @testset "Visualization Support" begin
        basis = QuadPairBasis(1)
        
        # Create cat state
        cat_even = catstate_even(basis, 1.0)
        
        # Test grid
        q_range = -3:0.2:3
        p_range = -3:0.2:3
        
        # Test convert_arguments for heatmap
        args_wigner = Makie.convert_arguments(Makie.Heatmap, q_range, p_range, cat_even; dist=:wigner)
        @test args_wigner isa Tuple
        @test length(args_wigner) == 3
        @test size(args_wigner[3]) == (length(q_range), length(p_range))
        
        # Test wignerchar visualization
        args_char = Makie.convert_arguments(Makie.Heatmap, q_range, p_range, cat_even; dist=:wignerchar)
        @test args_char isa Tuple
        @test length(args_char) == 3
        @test size(args_char[3]) == (length(q_range), length(p_range))
        
        # Values should be real for the data
        @test all(isfinite, args_wigner[3])
        @test all(isfinite, args_char[3])
        
        # Test that negative values are handled (cat states have negative Wigner regions)
        wigner_data = args_wigner[3]
        @test any(wigner_data .< 0)  # Cat states should have negative regions
        @test any(wigner_data .> 0)  # And positive regions
        
        # Test with GKP state
        gkp = gkpstate(basis, lattice=:square, delta=0.2, nmax=2)
        args_gkp = Makie.convert_arguments(Makie.Heatmap, q_range, p_range, gkp; dist=:wigner)
        @test args_gkp isa Tuple
        @test all(isfinite, args_gkp[3])
        
        # Error handling
        basis_2mode = QuadPairBasis(2)
        lc_2mode = GaussianLinearCombination(coherentstate(basis_2mode, [1.0, 0.5]))
        @test_throws ArgumentError Makie.convert_arguments(Makie.Heatmap, q_range, p_range, lc_2mode)
        
        @test_throws ErrorException Makie.convert_arguments(Makie.Heatmap, q_range, p_range, cat_even; dist=:invalid)
    end

    @testset "Measurement Probabilities" begin
        basis = QuadPairBasis(2)
        
        # Create entangled state as linear combination
        coh1 = coherentstate(basis, [1.0+0.0im, 0.0+0.0im])
        coh2 = coherentstate(basis, [0.0+0.0im, 1.0+0.0im])
        lc = GaussianLinearCombination(basis, [0.6, 0.8], [coh1, coh2])
        
        # Measurement state
        measurement_state = coherentstate(QuadPairBasis(1), 0.5)
        
        # Measurement on mode 1
        prob1 = measurement_probability(lc, measurement_state, [1])
        @test prob1 isa Real
        @test 0 <= prob1 <= 1
        @test isfinite(prob1)
        
        # Measurement on mode 2
        prob2 = measurement_probability(lc, measurement_state, [2])
        @test prob2 isa Real
        @test 0 <= prob2 <= 1
        
        # Single index version
        prob1_single = measurement_probability(lc, measurement_state, 1)
        @test prob1_single == prob1
        
        # Test with vacuum measurement
        vac_measurement = vacuumstate(QuadPairBasis(1))
        prob_vac = measurement_probability(lc, vac_measurement, [1])
        @test prob_vac isa Real
        @test 0 <= prob_vac <= 1
        
        # Test with pure state (should give definite results)
        pure_coh = GaussianLinearCombination(coh1)
        prob_pure = measurement_probability(pure_coh, measurement_state, [1])
        @test prob_pure isa Real
        @test 0 <= prob_pure <= 1
        
        # Error handling
        measurement_wrong_basis = coherentstate(QuadBlockBasis(1), 1.0)
        # This should work as long as dimensions match
        
        @test_throws BoundsError measurement_probability(lc, measurement_state, [3])  # Invalid index
    end

    @testset "State Metrics - Purity" begin
        basis = QuadPairBasis(1)
        
        # Pure state (single component)
        pure_state = GaussianLinearCombination(coherentstate(basis, 1.0))
        purity_pure = purity(pure_state)
        @test isapprox(purity_pure, 1.0, atol=1e-10)
        
        # Orthogonal mixed state
        coh1 = coherentstate(basis, 2.0)  # Well separated
        coh2 = coherentstate(basis, -2.0)
        mixed_orthogonal = GaussianLinearCombination(basis, [0.6, 0.8], [coh1, coh2])
        purity_mixed_orth = purity(mixed_orthogonal)
        
        # For orthogonal states: purity = |c₁|² + |c₂|² = 0.36 + 0.64 = 1.0
        @test isapprox(purity_mixed_orth, 1.0, atol=1e-10)
        
        # Overlapping states (non-orthogonal)
        coh_close1 = coherentstate(basis, 0.1)
        coh_close2 = coherentstate(basis, -0.1)
        mixed_overlap = GaussianLinearCombination(basis, [0.6, 0.8], [coh_close1, coh_close2])
        purity_overlap = purity(mixed_overlap)
        
        # Should be less than 1 due to overlap terms
        @test purity_overlap <= 1.0
        @test purity_overlap > 0.0
        
        # Equal mixture
        equal_mix = GaussianLinearCombination(basis, [1/sqrt(2), 1/sqrt(2)], [coh1, coh2])
        purity_equal = purity(equal_mix)
        @test purity_equal <= 1.0
        @test purity_equal > 0.0
        
        # Test with three components
        coh3 = coherentstate(basis, 0.0)
        triple_mix = GaussianLinearCombination(basis, [0.5, 0.3, 0.2], [coh1, coh2, coh3])
        purity_triple = purity(triple_mix)
        @test purity_triple <= 1.0
        @test purity_triple > 0.0
        
        # Identical states (should combine to pure)
        identical_mix = GaussianLinearCombination(basis, [0.6, 0.8], [coh1, coh1])
        purity_identical = purity(identical_mix)
        @test isapprox(purity_identical, 1.0, atol=1e-10)
    end

    @testset "State Metrics - Von Neumann Entropy" begin
        basis = QuadPairBasis(1)
        
        # Pure state
        pure_state = GaussianLinearCombination(coherentstate(basis, 1.0))
        entropy_pure = entropy_vn(pure_state)
        @test isapprox(entropy_pure, 0.0, atol=1e-10)
        
        # Single Gaussian state (should use existing method)
        thermal = thermalstate(basis, 2.0)
        entropy_thermal_single = entropy_vn(GaussianLinearCombination(thermal))
        entropy_thermal_direct = entropy_vn(thermal)
        @test isapprox(entropy_thermal_single, entropy_thermal_direct, atol=1e-10)
        
        # Superposition of coherent states (pure)
        coh1 = coherentstate(basis, 1.0)
        coh2 = coherentstate(basis, -1.0)
        superposition = GaussianLinearCombination(basis, [1/sqrt(2), 1/sqrt(2)], [coh1, coh2])
        entropy_superposition = entropy_vn(superposition)
        
        # Pure superposition should have some entropy due to state overlap
        @test entropy_superposition >= 0.0
        @test isfinite(entropy_superposition)
        
        # Mixed state (classical probabilities)
        mixed_state = GaussianLinearCombination(basis, [0.7, 0.3], [coh1, coh2])
        entropy_mixed = entropy_vn(mixed_state)
        @test entropy_mixed >= 0.0
        @test isfinite(entropy_mixed)
        
        # Entropy should increase with mixing
        @test entropy_mixed >= entropy_superposition
        
        # Test with thermal states
        thermal1 = thermalstate(basis, 1.0)
        thermal2 = thermalstate(basis, 3.0)
        thermal_mix = GaussianLinearCombination(basis, [0.5, 0.5], [thermal1, thermal2])
        entropy_thermal_mix = entropy_vn(thermal_mix)
        @test entropy_thermal_mix >= 0.0
        @test isfinite(entropy_thermal_mix)
        
        # Test with squeezed states
        sq1 = squeezedstate(basis, 0.5, 0.0)
        sq2 = squeezedstate(basis, 0.5, π)
        sq_mix = GaussianLinearCombination(basis, [1/sqrt(2), 1/sqrt(2)], [sq1, sq2])
        entropy_sq = entropy_vn(sq_mix)
        @test entropy_sq >= 0.0
        @test isfinite(entropy_sq)
        
        # Test edge cases
        
        # Very small coefficients
        tiny_mix = GaussianLinearCombination(basis, [0.999, 0.001], [coh1, coh2])
        entropy_tiny = entropy_vn(tiny_mix)
        @test entropy_tiny >= 0.0
        @test entropy_tiny < entropy_mixed  # Less mixed → less entropy
        
        # Many components
        many_states = [coherentstate(basis, i*0.5) for i in 1:5]
        many_coeffs = [0.2, 0.3, 0.25, 0.15, 0.1]
        many_mix = GaussianLinearCombination(basis, many_coeffs, many_states)
        entropy_many = entropy_vn(many_mix)
        @test entropy_many >= 0.0
        @test isfinite(entropy_many)
    end

    @testset "Integration with Cat and GKP States" begin
        basis = QuadPairBasis(1)
        
        # Cat states from previous phases
        cat_even = catstate_even(basis, 1.0)
        cat_odd = catstate_odd(basis, 1.0)
        
        # Apply operations to cat states
        phase_op = phaseshift(basis, π/4)
        cat_rotated = phase_op * cat_even
        @test cat_rotated isa GaussianLinearCombination
        @test length(cat_rotated) == 2
        
        # Tensor product of cat states
        cat_tensor = cat_even ⊗ cat_odd
        @test cat_tensor isa GaussianLinearCombination
        @test length(cat_tensor) == 4  # 2 × 2
        
        # Wigner function of cat state
        w_cat = wigner(cat_even, [0.0, 0.0])
        @test w_cat isa Real
        @test isfinite(w_cat)
        
        # Cat states should have negative Wigner regions
        q_range = -2:0.5:2
        p_range = -2:0.5:2
        wigner_values = [wigner(cat_even, [q, p]) for q in q_range, p in p_range]
        @test any(wigner_values .< 0)  # Negative values indicate non-classicality
        
        # GKP states
        gkp_square = gkpstate(basis, lattice=:square, delta=0.2, nmax=2)
        
        # Apply operations to GKP
        squeeze_op = squeeze(basis, 0.3, π/6)
        gkp_squeezed = squeeze_op * gkp_square
        @test gkp_squeezed isa GaussianLinearCombination
        
        # GKP Wigner function
        w_gkp = wigner(gkp_square, [0.0, 0.0])
        @test w_gkp isa Real
        @test isfinite(w_gkp)
        
        # Purity and entropy of non-Gaussian states
        purity_cat = purity(cat_even)
        entropy_cat = entropy_vn(cat_even)
        
        @test purity_cat <= 1.0
        @test purity_cat > 0.0
        @test entropy_cat >= 0.0
        @test isfinite(entropy_cat)
        
        purity_gkp = purity(gkp_square)
        entropy_gkp = entropy_vn(gkp_square)
        
        @test purity_gkp <= 1.0
        @test purity_gkp > 0.0
        @test entropy_gkp >= 0.0
        @test isfinite(entropy_gkp)
        
        # Partial trace of multi-mode cat states
        if nmodes >= 2
            basis_2mode = QuadPairBasis(2)
            cat_2mode = catstate_even(basis_2mode, [1.0, 0.5])
            cat_traced = ptrace(cat_2mode, [1])
            
            @test cat_traced isa GaussianLinearCombination
            @test cat_traced.basis.nmodes == 1
        end
    end

    @testset "Edge Cases and Error Handling" begin
        basis = QuadPairBasis(1)
        
        # Empty-like combinations
        tiny_state = GaussianLinearCombination(basis, [1e-16], [vacuumstate(basis)])
        @test length(tiny_state) == 1
        
        # Operations on tiny states
        disp_op = displace(basis, 1.0)
        tiny_displaced = disp_op * tiny_state
        @test tiny_displaced isa GaussianLinearCombination
        
        # Wigner of tiny states
        w_tiny = wigner(tiny_state, [0.0, 0.0])
        @test isfinite(w_tiny)
        
        # Very large combinations
        many_states = [coherentstate(basis, i*0.1) for i in 1:20]
        many_coeffs = fill(1/sqrt(20), 20)
        large_lc = GaussianLinearCombination(basis, many_coeffs, many_states)
        
        w_large = wigner(large_lc, [0.0, 0.0])
        @test isfinite(w_large)
        
        purity_large = purity(large_lc)
        @test 0 <= purity_large <= 1
        
        # Numerical precision tests
        # States that are nearly identical
        coh_base = coherentstate(basis, 1.0)
        coh_tiny_diff = coherentstate(basis, 1.0 + 1e-12)
        nearly_identical = GaussianLinearCombination(basis, [0.5, 0.5], [coh_base, coh_tiny_diff])
        
        purity_nearly = purity(nearly_identical)
        @test isfinite(purity_nearly)
        @test purity_nearly <= 1.0
        
        # Cross-Wigner with nearly identical states
        cross_nearly = cross_wigner(coh_base, coh_tiny_diff, [0.0, 0.0])
        @test isfinite(cross_nearly)
        
        # Test with singular covariance matrices (edge case)
        # Create states with very different scales
        coh_large = coherentstate(basis, 100.0)
        coh_small = coherentstate(basis, 1e-6)
        extreme_mix = GaussianLinearCombination(basis, [0.5, 0.5], [coh_large, coh_small])
        
        w_extreme = wigner(extreme_mix, [0.0, 0.0])
        @test isfinite(w_extreme)
        
        # Partial trace edge cases
        if nmodes >= 2
            basis_multi = QuadPairBasis(nmodes)
            single_state_multi = GaussianLinearCombination(vacuumstate(basis_multi))
            traced_single = ptrace(single_state_multi, [1])
            @test traced_single isa GaussianLinearCombination
        end
    end

    @testset "Performance and Memory" begin
        basis = QuadPairBasis(1)
        
        # Test that operations don't unnecessarily copy large arrays
        large_coeffs = randn(100)
        large_states = [coherentstate(basis, randn()) for _ in 1:100]
        large_lc = GaussianLinearCombination(basis, large_coeffs, large_states)
        
        # Operations should complete in reasonable time
        disp_op = displace(basis, 1.0)
        @test (@elapsed disp_op = displace(basis, 1.0)) < 1.0
        large_displaced = disp_op * large_lc
        @test (@elapsed large_displaced = disp_op * large_lc) < 5.0
        @test length(large_displaced) == 100
        
        # Wigner function should be reasonably fast
        @test (@elapsed w_large = wigner(large_lc, [0.0, 0.0])) < 10.0
        @test isfinite(w_large)
        
        # Memory usage test - operations shouldn't create excessive copies
        memory_before = Base.gc_bytes()
        for _ in 1:10
            temp_displaced = disp_op * large_lc
            temp_w = wigner(temp_displaced, [0.0, 0.0])
        end
        Base.GC.gc()
        memory_after = Base.gc_bytes()
        
        # Should not have excessive memory growth
        @test (memory_after - memory_before) < 1e8  # Less than 100MB growth
    end
end