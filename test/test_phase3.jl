@testitem "Phase 3: Integration and Advanced Features" begin
    using Gabs
    using LinearAlgebra
    using StaticArrays
    using Test

    # Test setup
    nmodes = 2
    qpairbasis = QuadPairBasis(nmodes)
    qblockbasis = QuadBlockBasis(nmodes)
    single_basis = QuadPairBasis(1)

    @testset "Gaussian Operations Integration" begin
        # Create test states
        coh1 = coherentstate(single_basis, 1.0)
        coh2 = coherentstate(single_basis, -1.0)
        lc = GaussianLinearCombination(single_basis, [0.6, 0.8], [coh1, coh2])

        @testset "Gaussian Unitary Operations" begin
            # Test displacement operation
            disp_op = displace(single_basis, 0.5 + 0.3im)
            lc_displaced = disp_op * lc
            
            @test lc_displaced isa GaussianLinearCombination
            @test length(lc_displaced) == 2
            @test lc_displaced.coefficients == lc.coefficients
            @test lc_displaced.basis == lc.basis
            @test lc_displaced.ħ == lc.ħ

            # Verify displacement was applied to each state
            expected_state1 = disp_op * coh1
            expected_state2 = disp_op * coh2
            @test isapprox(lc_displaced.states[1], expected_state1)
            @test isapprox(lc_displaced.states[2], expected_state2)

            # Test squeeze operation
            squeeze_op = squeeze(single_basis, 0.5, π/4)
            lc_squeezed = squeeze_op * lc
            
            @test lc_squeezed isa GaussianLinearCombination
            @test length(lc_squeezed) == 2
            @test lc_squeezed.coefficients == lc.coefficients

            # Test phase shift operation
            phase_op = phaseshift(single_basis, π/3)
            lc_phased = phase_op * lc
            
            @test lc_phased isa GaussianLinearCombination
            @test length(lc_phased) == 2

            # Test with different basis types
            coh_block = coherentstate(QuadBlockBasis(1), 1.0)
            lc_block = GaussianLinearCombination(coh_block)
            disp_block = displace(QuadBlockBasis(1), 0.5)
            lc_displaced_block = disp_block * lc_block
            
            @test lc_displaced_block isa GaussianLinearCombination
            @test lc_displaced_block.basis isa QuadBlockBasis
        end

        @testset "Gaussian Channel Operations" begin
            # Test attenuator channel
            att_channel = attenuator(single_basis, π/6, 3)
            lc_attenuated = att_channel * lc
            
            @test lc_attenuated isa GaussianLinearCombination
            @test length(lc_attenuated) == 2
            @test lc_attenuated.coefficients == lc.coefficients
            @test lc_attenuated.basis == lc.basis

            # Test amplifier channel
            amp_channel = amplifier(single_basis, 0.5, 2)
            lc_amplified = amp_channel * lc
            
            @test lc_amplified isa GaussianLinearCombination
            @test length(lc_amplified) == 2

            # Test that channel increases covariance (noise)
            original_covar_trace = sum(tr(state.covar) for (_, state) in lc)
            amplified_covar_trace = sum(tr(state.covar) for (_, state) in lc_amplified)
            @test amplified_covar_trace > original_covar_trace
        end

        @testset "Operations Error Handling" begin
            # Create displacement operator in this scope
            disp_single = displace(single_basis, 1.0)
            
            # Test basis mismatch
            lc_wrong_basis = GaussianLinearCombination(coherentstate(qpairbasis, 1.0))
            @test_throws ArgumentError disp_single * lc_wrong_basis

            # Test ħ mismatch
            coh_diff_hbar = coherentstate(single_basis, 1.0, ħ=1)
            lc_diff_hbar = GaussianLinearCombination(coh_diff_hbar)
            @test_throws ArgumentError disp_single * lc_diff_hbar
        end
    end

    @testset "Tensor Products" begin
        @testset "Basic Tensor Products" begin
            # Single mode linear combinations
            coh1 = coherentstate(single_basis, 1.0)
            coh2 = coherentstate(single_basis, -1.0)
            lc1 = GaussianLinearCombination(single_basis, [0.6, 0.4], [coh1, coh2])
            
            vac = vacuumstate(single_basis)
            squeezed = squeezedstate(single_basis, 0.5, π/4)
            lc2 = GaussianLinearCombination(single_basis, [0.8, 0.6], [vac, squeezed])

            # Test basic tensor product
            lc_tensor = tensor(lc1, lc2)
            
            @test lc_tensor isa GaussianLinearCombination
            @test length(lc_tensor) == 4  # 2 × 2 = 4
            @test lc_tensor.basis == single_basis ⊕ single_basis
            @test lc_tensor.ħ == lc1.ħ

            # Verify coefficient multiplication
            expected_coeffs = [0.6*0.8, 0.6*0.6, 0.4*0.8, 0.4*0.6]
            @test isapprox(lc_tensor.coefficients, expected_coeffs)

            # Verify state tensor products
            @test isapprox(lc_tensor.states[1], coh1 ⊗ vac)
            @test isapprox(lc_tensor.states[2], coh1 ⊗ squeezed)
            @test isapprox(lc_tensor.states[3], coh2 ⊗ vac)
            @test isapprox(lc_tensor.states[4], coh2 ⊗ squeezed)

            # Test ⊗ operator alias
            lc_tensor2 = lc1 ⊗ lc2
            @test lc_tensor == lc_tensor2
        end

        @testset "Typed Tensor Products" begin
            coh = coherentstate(single_basis, 1.0)
            vac = vacuumstate(single_basis)
            lc1 = GaussianLinearCombination(single_basis, [0.7, 0.3], [coh, vac])
            lc2 = GaussianLinearCombination(single_basis, [0.9, 0.1], [vac, coh])

            # Test with correct type specifications: Vector for mean, Matrix for covar
            lc_tensor_typed = tensor(Vector{Float64}, Matrix{Float64}, lc1, lc2)
            @test lc_tensor_typed isa GaussianLinearCombination
            @test eltype(lc_tensor_typed.states[1].mean) == Float64
            @test eltype(lc_tensor_typed.states[1].covar) == Float64

            # Test with single type parameter (applies to both mean and covar types)
            lc_tensor_single_type = tensor(Matrix{Float64}, lc1, lc2)
            @test lc_tensor_single_type isa GaussianLinearCombination
        end

        @testset "Tensor Products with Different Bases" begin
            # Test QuadBlockBasis
            coh_block = coherentstate(QuadBlockBasis(1), 1.0)
            vac_block = vacuumstate(QuadBlockBasis(1))
            lc_block1 = GaussianLinearCombination(QuadBlockBasis(1), [0.6, 0.4], [coh_block, vac_block])
            lc_block2 = GaussianLinearCombination(QuadBlockBasis(1), [0.8, 0.2], [vac_block, coh_block])

            lc_tensor_block = tensor(lc_block1, lc_block2)
            @test lc_tensor_block.basis isa QuadBlockBasis
            @test lc_tensor_block.basis.nmodes == 2
        end

        @testset "Tensor Product Error Handling" begin
            coh_pair = coherentstate(single_basis, 1.0)
            coh_block = coherentstate(QuadBlockBasis(1), 1.0)
            lc_pair = GaussianLinearCombination(coh_pair)
            lc_block = GaussianLinearCombination(coh_block)

            # Test basis type mismatch
            @test_throws ArgumentError tensor(lc_pair, lc_block)

            # Test ħ mismatch
            coh_diff_hbar = coherentstate(single_basis, 1.0, ħ=1)
            lc_diff_hbar = GaussianLinearCombination(coh_diff_hbar)
            @test_throws ArgumentError tensor(lc_pair, lc_diff_hbar)
        end

        @testset "Large Tensor Products" begin
            # Test with more terms
            states = [coherentstate(single_basis, Float64(i)) for i in 1:5]
            coeffs = [0.2, 0.3, 0.1, 0.25, 0.15]
            lc_large1 = GaussianLinearCombination(single_basis, coeffs, states)
            
            vac = vacuumstate(single_basis)
            lc_large2 = GaussianLinearCombination(vac)
            
            lc_tensor_large = tensor(lc_large1, lc_large2)
            @test length(lc_tensor_large) == 5  # 5 × 1 = 5
            @test lc_tensor_large.coefficients == coeffs
        end
    end

    @testset "Partial Traces" begin
        @testset "Basic Partial Trace" begin
            # Create two-mode system
            coh1 = coherentstate(single_basis, 1.0)
            coh2 = coherentstate(single_basis, -1.0)
            vac = vacuumstate(single_basis)
            
            # Create tensor product linear combination
            lc1 = GaussianLinearCombination(single_basis, [0.6, 0.4], [coh1, vac])
            lc2 = GaussianLinearCombination(single_basis, [0.8, 0.2], [coh2, vac])
            lc_2mode = tensor(lc1, lc2)

            # Test partial trace over first mode
            lc_traced = ptrace(lc_2mode, 1)
            @test lc_traced isa GaussianLinearCombination
            @test lc_traced.basis.nmodes == 1
            @test lc_traced.ħ == lc_2mode.ħ

            # Test partial trace over second mode
            lc_traced2 = ptrace(lc_2mode, 2)
            @test lc_traced2 isa GaussianLinearCombination
            @test lc_traced2.basis.nmodes == 1

            # Test partial trace with vector indices
            basis_3mode = QuadPairBasis(3)
            coh3 = coherentstate(single_basis, 2.0)
            lc3 = GaussianLinearCombination(coh3)
            lc_3mode = tensor(tensor(lc1, lc2), lc3)
            
            lc_traced_multi = ptrace(lc_3mode, [1, 3])
            @test lc_traced_multi.basis.nmodes == 1
            @test length(lc_traced_multi) <= length(lc_3mode)  # May be simplified
        end

        @testset "Partial Trace with Simplification" begin
            # Create system where partial trace leads to identical states
            coh = coherentstate(single_basis, 1.0)
            vac = vacuumstate(single_basis)
            
            # Both terms have same first mode state
            state1 = coh ⊗ vac
            state2 = coh ⊗ coh
            lc_2mode = GaussianLinearCombination(QuadPairBasis(2), [0.5, 0.5], [state1, state2])
            
            # Trace out second mode - should combine identical first modes
            lc_traced = ptrace(lc_2mode, 2)
            @test length(lc_traced) == 1  # Should be simplified to single state
            @test abs(lc_traced.coefficients[1] - 1.0) < 1e-12
        end

        @testset "Typed Partial Trace" begin
            coh = coherentstate(single_basis, 1.0)
            vac = vacuumstate(single_basis)
            lc_2mode = tensor(GaussianLinearCombination(coh), GaussianLinearCombination(vac))

            # Test with correct type specifications: Vector for mean, Matrix for covar
            lc_traced_typed = ptrace(Vector{Float64}, Matrix{Float64}, lc_2mode, 1)
            @test lc_traced_typed isa GaussianLinearCombination
            @test eltype(lc_traced_typed.states[1].mean) == Float64

            lc_traced_single_type = ptrace(Matrix{Float64}, lc_2mode, 1)
            @test lc_traced_single_type isa GaussianLinearCombination
        end

        @testset "Partial Trace Error Handling" begin
            coh = coherentstate(single_basis, 1.0)
            lc_single = GaussianLinearCombination(coh)

            # Test tracing all modes - expect ArgumentError
            @test_throws ArgumentError ptrace(lc_single, 1)

            # Test invalid indices - expect ArgumentError
            lc_2mode = tensor(lc_single, lc_single)
            @test_throws ArgumentError ptrace(lc_2mode, [1, 2])  # All modes
            @test_throws ArgumentError ptrace(lc_2mode, 3)  # Index too large
        end

        @testset "QuadBlockBasis Partial Trace" begin
            coh_block = coherentstate(QuadBlockBasis(1), 1.0)
            vac_block = vacuumstate(QuadBlockBasis(1))
            lc_block_2mode = tensor(GaussianLinearCombination(coh_block), GaussianLinearCombination(vac_block))

            lc_traced_block = ptrace(lc_block_2mode, 1)
            @test lc_traced_block.basis isa QuadBlockBasis
            @test lc_traced_block.basis.nmodes == 1
        end
    end

    @testset "Wigner Functions with Quantum Interference" begin
        @testset "Cross-Wigner Function" begin
            coh1 = coherentstate(single_basis, 1.0)
            coh2 = coherentstate(single_basis, -1.0)
            
            # Test cross-Wigner function
x = [0.5, 0.3]

cross_w = cross_wigner(coh1, coh2, x)
@test cross_w isa Complex  # ✅ CORRECT - expects complex value
println("Cross-Wigner at $x: $cross_w")
@test isfinite(real(cross_w))
@test isfinite(imag(cross_w))

# Additional test: for identical states, should equal regular Wigner
cross_w_identical = cross_wigner(coh1, coh1, x)
regular_w = wigner(coh1, x)
@test isapprox(cross_w_identical, ComplexF64(regular_w), atol=1e-12)


            

            # Test at mean positions
            x_mean1 = coh1.mean
            x_mean2 = coh2.mean
            cross_w1 = cross_wigner(coh1, coh2, x_mean1)
            cross_w2 = cross_wigner(coh1, coh2, x_mean2)
            @test isfinite(cross_w1)
            @test isfinite(cross_w2)

            # For identical states, cross-Wigner equals regular Wigner
            cross_w_identical = cross_wigner(coh1, coh1, x)
            regular_w = wigner(coh1, x)
            @test isapprox(cross_w_identical, regular_w, atol=1e-12)
        end

        @testset "Linear Combination Wigner Function" begin
            coh1 = coherentstate(single_basis, 1.0)
            coh2 = coherentstate(single_basis, -1.0)
            
            # Create cat state
            lc = GaussianLinearCombination(single_basis, [0.5, 0.5], [coh1, coh2])
            
            x_points = [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]
            
            for x in x_points
                w_lc = wigner(lc, x)
                @test w_lc isa Real
                @test isfinite(w_lc)
                
                # Compare with manual calculation including interference
                w1 = wigner(coh1, x)
                w2 = wigner(coh2, x)
                w_cross = cross_wigner(coh1, coh2, x)
                
                w_manual = 0.25 * w1 + 0.25 * w2 + 2 * real(0.5 * 0.5 * w_cross)
                @test isapprox(w_lc, w_manual, atol=1e-12)
            end

            # Test that interference can create negative values (non-classical)
            x_interference = [0.0, 0.0]  # Midpoint between coherent states
            w_interference = wigner(lc, x_interference)
            
            # For a cat state at the origin, we expect significant interference
            w1_origin = wigner(coh1, x_interference)
            w2_origin = wigner(coh2, x_interference)
            # The cross term should be significant here
            @test abs(w_interference - 0.25 * (w1_origin + w2_origin)) > 1e-6
        end

        @testset "Cross-Wigner Characteristic Function" begin
            coh1 = coherentstate(single_basis, 1.0 + 0.5im)
            coh2 = coherentstate(single_basis, -0.5 + 1.0im)
            
            xi_points = [[0.0, 0.0], [0.5, 0.3], [1.0, -0.8]]
            
            for xi in xi_points
                cross_char = cross_wignerchar(coh1, coh2, xi)
                @test cross_char isa Complex
                @test isfinite(real(cross_char))
                @test isfinite(imag(cross_char))
                
                # For identical states, should equal regular characteristic function
                cross_char_identical = cross_wignerchar(coh1, coh1, xi)
                regular_char = wignerchar(coh1, xi)
                @test isapprox(cross_char_identical, regular_char, atol=1e-12)
            end
        end

        @testset "Linear Combination Wigner Characteristic Function" begin
            coh1 = coherentstate(single_basis, 0.8)
            coh2 = coherentstate(single_basis, -0.8)
            lc = GaussianLinearCombination(single_basis, [0.6, 0.8], [coh1, coh2])
            
            xi_points = [[0.0, 0.0], [0.2, 0.5], [-0.3, 0.7]]
            
            for xi in xi_points
                char_lc = wignerchar(lc, xi)
                @test char_lc isa Complex
                @test isfinite(real(char_lc))
                @test isfinite(imag(char_lc))
                
                # Manual calculation with cross-terms
                char1 = wignerchar(coh1, xi)
                char2 = wignerchar(coh2, xi)
                char_cross = cross_wignerchar(coh1, coh2, xi)
                
                char_manual = 0.36 * char1 + 0.64 * char2 + 2 * real(0.6 * 0.8 * char_cross)
                @test isapprox(real(char_lc), real(char_manual), atol=1e-12)
                @test isapprox(imag(char_lc), imag(char_manual), atol=1e-12)
            end
        end

        @testset "Wigner Function Error Handling" begin
            coh = coherentstate(single_basis, 1.0)
            lc = GaussianLinearCombination(coh)
            
            # Test wrong vector length
            @test_throws ArgumentError wigner(lc, [1.0])  # Should be length 2
            @test_throws ArgumentError wigner(lc, [1.0, 2.0, 3.0])  # Too long
            @test_throws ArgumentError wignerchar(lc, [1.0])
            
            # Test cross-Wigner with incompatible states
            coh_diff_basis = coherentstate(qpairbasis, 1.0)
            @test_throws ArgumentError cross_wigner(coh, coh_diff_basis, [1.0, 2.0])
            
            coh_diff_hbar = coherentstate(single_basis, 1.0, ħ=1)
            @test_throws ArgumentError cross_wigner(coh, coh_diff_hbar, [1.0, 2.0])
        end
    end

    @testset "Advanced State Metrics" begin
        @testset "Purity Calculation" begin
            # Pure state should have purity = 1
            coh = coherentstate(single_basis, 1.0)
            lc_pure = GaussianLinearCombination(coh)
            @test isapprox(purity(lc_pure), 1.0, atol=1e-12)
        
            # Quantum superposition is STILL pure (purity = 1)
            coh1 = coherentstate(single_basis, 5.0)
            coh2 = coherentstate(single_basis, -5.0)
            lc_superposition = GaussianLinearCombination(single_basis, [0.5, 0.5], [coh1, coh2])
            
            @test isapprox(purity(lc_superposition), 1.0, atol=1e-12)  # ✅ CORRECT
            
            # Test with complex coefficients (still pure)
            lc_complex = GaussianLinearCombination(single_basis, [0.6 + 0.8im, 0.0], [coh1, coh2])
            Gabs.normalize!(lc_complex)
            @test isapprox(purity(lc_complex), 1.0, atol=1e-12)
            
            # Test coherence measure for orthogonality assessment
            coherence_well_separated = coherence_measure(lc_superposition)
            
            coh3 = coherentstate(single_basis, 0.1)
            coh4 = coherentstate(single_basis, -0.1)
            lc_overlapping = GaussianLinearCombination(single_basis, [0.5, 0.5], [coh3, coh4])
            coherence_overlapping = coherence_measure(lc_overlapping)
            
            @test coherence_well_separated > coherence_overlapping  # Well-separated states have higher coherence
        end

        @testset "Von Neumann Entropy Calculation" begin
            # Pure state should have entropy = 0
            coh = coherentstate(single_basis, 1.0)
            lc_pure = GaussianLinearCombination(coh)
            @test isapprox(entropy_vn(lc_pure), 0.0, atol=1e-12)

            # Two-state superposition
            coh1 = coherentstate(single_basis, 1.5)
            coh2 = coherentstate(single_basis, -1.5)
            lc_two = GaussianLinearCombination(single_basis, [0.6, 0.8], [coh1, coh2])
            
            entropy_two = entropy_vn(lc_two)
            @test entropy_two >= 0.0
            @test isfinite(entropy_two)
            
            # Equal superposition should have higher entropy
            lc_equal = GaussianLinearCombination(single_basis, [0.5, 0.5], [coh1, coh2])
            entropy_equal = entropy_vn(lc_equal)
            @test entropy_equal >= entropy_two  # Equal weights should have higher entropy

            # Test with three states
            vac = vacuumstate(single_basis)
            lc_three = GaussianLinearCombination(single_basis, [0.5, 0.3, 0.2], [coh1, coh2, vac])
            entropy_three = entropy_vn(lc_three)
            @test entropy_three >= 0.0
            @test isfinite(entropy_three)

            # Test complex coefficients
            lc_complex = GaussianLinearCombination(single_basis, 
                [0.5 + 0.3im, 0.4 - 0.2im], [coh1, coh2])
            entropy_complex = entropy_vn(lc_complex)
            @test entropy_complex >= 0.0

            # Test large system (should use approximation with warning)
            large_states = [coherentstate(single_basis, Float64(i)/10) for i in 1:120]
            large_coeffs = ones(120) / sqrt(120)
            lc_large = GaussianLinearCombination(single_basis, large_coeffs, large_states)
            
            entropy_large = entropy_vn(lc_large)
            @test entropy_large >= 0.0
            @test isfinite(entropy_large)
        end

        @testset "Numerical Stability of Metrics" begin
            # Test with very small coefficients
            coh1 = coherentstate(single_basis, 1.0)
            coh2 = coherentstate(single_basis, -1.0)
            lc_small = GaussianLinearCombination(single_basis, [1e-8, 1.0], [coh1, coh2])
            
            @test 0.0 <= purity(lc_small) <= 1.0
            @test entropy_vn(lc_small) >= 0.0

            # Test with nearly identical states
            coh_close = coherentstate(single_basis, 1.001)
            lc_close = GaussianLinearCombination(single_basis, [0.5, 0.5], [coh1, coh_close])
            
            @test 0.0 <= purity(lc_close) <= 1.0
            @test entropy_vn(lc_close) >= 0.0

            # Test normalized combinations
            lc_normalized = GaussianLinearCombination(single_basis, [0.6, 0.8], [coh1, coh2])
            Gabs.normalize!(lc_normalized)
            
            @test 0.0 <= purity(lc_normalized) <= 1.0
            @test entropy_vn(lc_normalized) >= 0.0
        end
    end

    @testset "Measurement Theory" begin
        @testset "Basic Measurement Probability" begin
            # Create two-mode system
            coh1 = coherentstate(single_basis, 1.0)
            coh2 = coherentstate(single_basis, -1.0)
            vac = vacuumstate(single_basis)
            
            lc_2mode = tensor(GaussianLinearCombination(single_basis, [0.6, 0.4], [coh1, vac]),
                            GaussianLinearCombination(single_basis, [0.8, 0.2], [coh2, vac]))

            # Measure first mode in vacuum state
            prob_vac = measurement_probability(lc_2mode, vac, 1)
            @test 0.0 <= prob_vac <= 1.0
            @test prob_vac isa Real

            # Measure first mode in coherent state
            prob_coh = measurement_probability(lc_2mode, coh1, 1)
            @test 0.0 <= prob_coh <= 1.0

            # Measure second mode
            prob_second = measurement_probability(lc_2mode, coh2, 2)
            @test 0.0 <= prob_second <= 1.0

            # Perfect overlap should give probability close to coefficient squared
            single_coherent = GaussianLinearCombination(tensor(coh1, vac))
            prob_perfect = measurement_probability(single_coherent, coh1, 1)
            @test isapprox(prob_perfect, 1.0, atol=1e-12)
        end

        @testset "Partial Measurements" begin
            # Three-mode system
            coh = coherentstate(single_basis, 1.0)
            vac = vacuumstate(single_basis)
            squeezed = squeezedstate(single_basis, 0.5, π/4)
            
            # Create 3-mode state
            lc_3mode = tensor(tensor(GaussianLinearCombination(coh), 
                                   GaussianLinearCombination(vac)),
                            GaussianLinearCombination(squeezed))

            # Measure first mode only
            prob_1 = measurement_probability(lc_3mode, vac, 1)
            @test 0.0 <= prob_1 <= 1.0

            # Measure multiple modes
            measurement_2mode = tensor(coh, vac)
            prob_12 = measurement_probability(lc_3mode, measurement_2mode, [1, 2])
            @test 0.0 <= prob_12 <= 1.0

            # Measure all modes (no partial trace)
            measurement_3mode = tensor(tensor(coh, vac), squeezed)
            prob_all = measurement_probability(lc_3mode, measurement_3mode, [1, 2, 3])
            @test 0.0 <= prob_all <= 1.0
        end

        @testset "Born Rule Verification" begin
            # Create superposition state
            coh1 = coherentstate(single_basis, 1.0)
            coh2 = coherentstate(single_basis, -1.0)
            lc = GaussianLinearCombination(single_basis, [0.6, 0.8], [coh1, coh2])

            # Measure in first component state
            prob1 = measurement_probability(lc, coh1, 1)
            
            # Measure in second component state  
            prob2 = measurement_probability(lc, coh2, 1)
            
            # Both should be non-zero and reasonable
            @test prob1 > 0.0
            @test prob2 > 0.0
            
            # Test with orthogonal measurement (should be less probable)
            orthogonal_state = coherentstate(single_basis, 5.0)  # Far from both states
            prob_ortho = measurement_probability(lc, orthogonal_state, 1)
            @test prob_ortho < max(prob1, prob2)
        end

        @testset "Measurement Error Handling" begin
            coh = coherentstate(single_basis, 1.0)
            lc_single = GaussianLinearCombination(coh)
            
            # Wrong number of modes in measurement
            coh_2mode = coherentstate(qpairbasis, 1.0)
            @test_throws ArgumentError measurement_probability(lc_single, coh_2mode, 1)
            
            # ħ mismatch
            coh_diff_hbar = coherentstate(single_basis, 1.0, ħ=1)
            @test_throws ArgumentError measurement_probability(lc_single, coh_diff_hbar, 1)
            
            # Invalid indices
            lc_2mode = tensor(lc_single, lc_single)
            @test_throws ArgumentError measurement_probability(lc_2mode, coh, 3)
        end

        @testset "Complex Coefficients in Measurements" begin
            coh1 = coherentstate(single_basis, 1.0)
            coh2 = coherentstate(single_basis, -1.0)
            
            # Create state with complex coefficients
            lc_complex = GaussianLinearCombination(single_basis, 
                [0.6 + 0.8im, 0.0 + 1.0im], [coh1, coh2])
            
            # Measurement probabilities should still be real and in [0,1]
            prob_complex = measurement_probability(lc_complex, coh1, 1)
            @test prob_complex isa Real
            @test 0.0 <= prob_complex <= 1.0
            
            # Test with normalized complex state
            Gabs.normalize!(lc_complex)
            prob_normalized = measurement_probability(lc_complex, coh1, 1)
            @test 0.0 <= prob_normalized <= 1.0
        end
    end

    @testset "Static Arrays Compatibility" begin
        # Test all Phase 3 functions with StaticArrays
        coh_static = coherentstate(SVector{2}, SMatrix{2,2}, single_basis, 1.0)
        vac_static = vacuumstate(SVector{2}, SMatrix{2,2}, single_basis)
        lc_static = GaussianLinearCombination(single_basis, [0.6, 0.8], [coh_static, vac_static])

        @testset "Static Operations" begin
            # Gaussian operations
            disp_static = displace(SVector{2}, SMatrix{2,2}, single_basis, 0.5)
            lc_disp_static = disp_static * lc_static
            @test lc_disp_static.states[1].mean isa SVector
            @test lc_disp_static.states[1].covar isa SMatrix

            # Tensor products
            lc_tensor_static = tensor(SVector{4}, SMatrix{4,4}, lc_static, lc_static)
            @test lc_tensor_static.states[1].mean isa SVector
            @test lc_tensor_static.states[1].covar isa SMatrix

            # Partial trace
            lc_traced_static = ptrace(SVector{2}, SMatrix{2,2}, lc_tensor_static, 1)
            @test lc_traced_static.states[1].mean isa SVector
            @test lc_traced_static.states[1].covar isa SMatrix
        end

        @testset "Static Wigner Functions" begin
            x = @SVector [0.5, 0.3]
            
            # Cross-Wigner (FIXED)
            cross_w_static = cross_wigner(coh_static, vac_static, x)
            @test cross_w_static isa Complex  # ✅ CORRECT - expects Complex value
            @test isfinite(real(cross_w_static))
            @test isfinite(imag(cross_w_static))
            
            # Test for identical states (should equal regular Wigner)
            cross_w_identical = cross_wigner(coh_static, coh_static, x)
            regular_w = wigner(coh_static, x)
            @test isapprox(cross_w_identical, ComplexF64(regular_w), atol=1e-12)
            
            # Wigner with interference (already correct)
            w_static = wigner(lc_static, x)
            @test w_static isa Real
            @test isfinite(w_static)
            
            # Characteristic functions (already correct)
            xi = @SVector [0.2, 0.4]
            char_static = wignerchar(lc_static, xi)
            @test char_static isa Complex
            @test isfinite(real(char_static))
            @test isfinite(imag(char_static))
            
            # Cross-characteristic function (add this for completeness)
            cross_char_static = cross_wignerchar(coh_static, vac_static, xi)
            @test cross_char_static isa Complex
            @test isfinite(real(cross_char_static))
            @test isfinite(imag(cross_char_static))
        end

        @testset "Static Metrics" begin
            purity_static = purity(lc_static)
            @test purity_static isa Real
            @test 0.0 <= purity_static <= 1.0
            
            entropy_static = entropy_vn(lc_static)
            @test entropy_static isa Real
            @test entropy_static >= 0.0
        end

        @testset "Static Measurements" begin
            # Use well-separated states to ensure proper normalization
            coh_static_far = coherentstate(SVector{2}, SMatrix{2,2}, single_basis, 10.0)
            lc_static_norm = GaussianLinearCombination(coh_static_far)
            
            prob_static = measurement_probability(lc_static_norm, coh_static_far, 1)
            @test prob_static isa Real
            @test 0.0 <= prob_static <= 1.0
        end
    end
end