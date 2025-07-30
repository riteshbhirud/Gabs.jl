# GPU utility functions and memory management

# Memory management and optimization utilities
struct GPUMemoryPool
    max_pool_size::Int
    current_usage::Ref{Int}
    allocated_arrays::Vector{CuArray}
    
    function GPUMemoryPool(max_size_mb::Int = 1024)
        max_bytes = max_size_mb * 1024 * 1024
        new(max_bytes, Ref(0), CuArray[])
    end
end

const DEFAULT_GPU_POOL = GPUMemoryPool()

function allocate_gpu_workspace(::Type{T}, dims...) where T
    """Allocate GPU workspace with memory pooling"""
    size_bytes = sizeof(T) * prod(dims)
    
    if DEFAULT_GPU_POOL.current_usage[] + size_bytes > DEFAULT_GPU_POOL.max_pool_size
        # Clean up unused arrays
        cleanup_gpu_pool!()
    end
    
    arr = CUDA.zeros(T, dims...)
    push!(DEFAULT_GPU_POOL.allocated_arrays, arr)
    DEFAULT_GPU_POOL.current_usage[] += size_bytes
    
    return arr
end

function cleanup_gpu_pool!()
    """Clean up GPU memory pool"""
    # Remove arrays that are no longer referenced
    filter!(arr -> Base.summarysize(arr) > 0, DEFAULT_GPU_POOL.allocated_arrays)
    
    # Recalculate current usage
    DEFAULT_GPU_POOL.current_usage[] = sum(sizeof, DEFAULT_GPU_POOL.allocated_arrays)
    
    # Force garbage collection
    CUDA.reclaim()
end

function get_gpu_memory_info()
    """Get current GPU memory usage information"""
    free_bytes, total_bytes = CUDA.memory_info()
    used_bytes = total_bytes - free_bytes
    
    return (
        total_mb = total_bytes ÷ (1024^2),
        used_mb = used_bytes ÷ (1024^2),
        free_mb = free_bytes ÷ (1024^2),
        pool_usage_mb = DEFAULT_GPU_POOL.current_usage[] ÷ (1024^2)
    )
end

# Performance optimization utilities
function optimal_block_size(n_elements::Int, max_threads_per_block::Int = 1024)
    """Calculate optimal CUDA block size for given number of elements"""
    if n_elements <= 32
        return 32
    elseif n_elements <= 128
        return 128
    elseif n_elements <= 512
        return 512
    else
        return min(max_threads_per_block, 1024)
    end
end

function optimal_batch_size(element_size_bytes::Int, available_memory_bytes::Int)
    """Calculate optimal batch size based on available GPU memory"""
    # Use 80% of available memory to leave room for other operations
    usable_memory = Int(0.8 * available_memory_bytes)
    batch_size = usable_memory ÷ element_size_bytes
    
    # Ensure batch size is at least 1 and at most 10000
    return max(1, min(batch_size, 10000))
end

# Automatic device selection and optimization
function auto_select_compute_device(operation_size::Int, data_transfer_cost::Float64 = 0.0)
    """
    Automatically select CPU or GPU based on operation characteristics
    Returns: (use_gpu::Bool, reason::String)
    """
    if !CUDA.functional()
        return (false, "CUDA not available")
    end
    
    # Get GPU info
    device = CUDA.device()
    props = CUDA.properties(device)
    memory_info = get_gpu_memory_info()
    
    # Size-based heuristics
    if operation_size < 100
        return (false, "Operation too small for GPU overhead")
    elseif operation_size > 10000 && memory_info.free_mb > 100
        return (true, "Large operation benefits from GPU parallelism")
    elseif data_transfer_cost > 0.1  # High transfer cost
        return (false, "Data transfer cost too high")
    elseif memory_info.free_mb < 50  # Low memory
        return (false, "Insufficient GPU memory")
    else
        # Medium-sized operations - consider GPU
        return (true, "Moderate-sized operation suitable for GPU")
    end
end

# Benchmarking and profiling utilities
struct OperationBenchmark
    cpu_time::Float64
    gpu_time::Float64
    speedup::Float64
    memory_transfer_time::Float64
    
    function OperationBenchmark(cpu_time, gpu_time, transfer_time = 0.0)
        speedup = cpu_time / (gpu_time + transfer_time)
        new(cpu_time, gpu_time, speedup, transfer_time)
    end
end

function benchmark_operation(cpu_func, gpu_func, args...; n_trials::Int = 5)
    """Benchmark CPU vs GPU operation performance"""
    
    # Warm up
    try
        cpu_func(args...)
        if CUDA.functional()
            gpu_func(args...)
            CUDA.synchronize()
        end
    catch
        # Ignore warmup errors
    end
    
    # Benchmark CPU
    cpu_times = Float64[]
    for _ in 1:n_trials
        t_start = time()
        cpu_func(args...)
        t_end = time()
        push!(cpu_times, t_end - t_start)
    end
    cpu_time = minimum(cpu_times)
    
    # Benchmark GPU if available
    if CUDA.functional()
        gpu_times = Float64[]
        transfer_times = Float64[]
        
        for _ in 1:n_trials
            # Measure data transfer time
            t_transfer_start = time()
            gpu_args = [is_gpu_array(arg) ? arg : CuArray(arg) for arg in args]
            t_transfer_end = time()
            transfer_time = t_transfer_end - t_transfer_start
            
            # Measure GPU computation time
            CUDA.synchronize()
            t_gpu_start = time()
            result = gpu_func(gpu_args...)
            CUDA.synchronize()
            t_gpu_end = time()
            
            gpu_time = t_gpu_end - t_gpu_start
            push!(gpu_times, gpu_time)
            push!(transfer_times, transfer_time)
        end
        
        min_gpu_time = minimum(gpu_times)
        min_transfer_time = minimum(transfer_times)
        
        return OperationBenchmark(cpu_time, min_gpu_time, min_transfer_time)
    else
        return OperationBenchmark(cpu_time, Inf, 0.0)
    end
end

# Error handling and diagnostics
function check_gpu_compatibility(arrays...)
    """Check if arrays are compatible for GPU operations"""
    errors = String[]
    
    for (i, arr) in enumerate(arrays)
        if isa(arr, CuArray)
            # Check for NaN or Inf values
            if any(isnan, arr) || any(isinf, arr)
                push!(errors, "Array $i contains NaN or Inf values")
            end
            
            # Check for very large values that might cause overflow
            if any(abs.(arr) .> 1e10)
                push!(errors, "Array $i contains very large values (>1e10)")
            end
        end
    end
    
    return isempty(errors) ? nothing : errors
end

function gpu_health_check()
    """Perform comprehensive GPU health check"""
    report = Dict{String, Any}()
    
    # CUDA availability
    report["cuda_functional"] = CUDA.functional()
    if !CUDA.functional()
        report["cuda_error"] = "CUDA not functional"
        return report
    end
    
    # Device info
    device = CUDA.device()
    props = CUDA.properties(device)
    report["device_name"] = props.name
    report["compute_capability"] = props.major * 10 + props.minor
    report["total_memory_mb"] = props.totalGlobalMem ÷ (1024^2)
    
    # Memory info
    memory_info = get_gpu_memory_info()
    report["memory_info"] = memory_info
    
    # Performance test
    try
        test_size = 1000
        test_array = CUDA.rand(Float32, test_size, test_size)
        CUDA.synchronize()
        
        t_start = time()
        result = test_array * test_array
        CUDA.synchronize()
        t_end = time()
        
        report["matrix_multiply_time"] = t_end - t_start
        report["performance_test"] = "PASS"
    catch e
        report["performance_test"] = "FAIL"
        report["performance_error"] = string(e)
    end
    
    return report
end

# Adaptive optimization based on system characteristics
mutable struct GPUOptimizationSettings
    prefer_gpu::Bool
    min_size_for_gpu::Int
    max_batch_size::Int
    memory_threshold_mb::Int
    
    function GPUOptimizationSettings()
        # Initialize with conservative defaults
        new(true, 100, 1000, 100)
    end
end

const GPU_SETTINGS = GPUOptimizationSettings()

function optimize_gpu_settings!()
    """Automatically optimize GPU settings based on system characteristics"""
    if !CUDA.functional()
        GPU_SETTINGS.prefer_gpu = false
        return
    end
    
    health = gpu_health_check()
    
    # Adjust settings based on available memory
    if health["memory_info"].free_mb < 100
        GPU_SETTINGS.prefer_gpu = false
        GPU_SETTINGS.memory_threshold_mb = 50
    elseif health["memory_info"].free_mb > 1000
        GPU_SETTINGS.max_batch_size = 5000
        GPU_SETTINGS.min_size_for_gpu = 50
    end
    
    # Adjust based on compute capability
    if haskey(health, "compute_capability") && health["compute_capability"] >= 70
        # Modern GPU - can handle smaller operations efficiently
        GPU_SETTINGS.min_size_for_gpu = 50
    end
    
    return GPU_SETTINGS
end

# Initialize optimization settings
optimize_gpu_settings!()

# Convenience functions for common patterns
function with_gpu_fallback(gpu_func, cpu_func, args...; force_cpu::Bool = false)
    """Execute GPU function with automatic CPU fallback"""
    if force_cpu || !CUDA.functional() || !GPU_SETTINGS.prefer_gpu
        return cpu_func(args...)
    end
    
    try
        return gpu_func(args...)
    catch e
        @warn "GPU operation failed, falling back to CPU" exception=e
        return cpu_func(args...)
    end
end

function ensure_sufficient_gpu_memory(required_mb::Int)
    """Ensure sufficient GPU memory is available"""
    if !CUDA.functional()
        return false
    end
    
    memory_info = get_gpu_memory_info()
    if memory_info.free_mb < required_mb
        cleanup_gpu_pool!()
        memory_info = get_gpu_memory_info()
        
        if memory_info.free_mb < required_mb
            @warn "Insufficient GPU memory: need $(required_mb)MB, have $(memory_info.free_mb)MB"
            return false
        end
    end
    
    return true
end