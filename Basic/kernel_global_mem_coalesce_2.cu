// #pragma once

// #include <cassert>
// #include <cstdio>
// #include <cstdlib>
// #include <cublas_v2.h>
// #include <cuda_runtime.h>

// template <const uint BLOCKSIZE>
// __global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
//                                           const float *A, const float *B,
//                                           float beta, float *C) {
//   const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
//   const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

//   // if statement is necessary to make things work under tile quantization
//   if (cRow < M && cCol < N) {
//     float tmp = 0.0;
//     for (int i = 0; i < K; ++i) {
//       tmp += A[cRow * K + i] * B[i * N + cCol];
//     }
//     C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
//   }
// }
// collab https://colab.research.google.com/drive/1ngCZksFRhglndPlEHp4HZhJlpobThZPa?usp=sharing
// done with opus 16k thinking




---------------------------------------------------------------------------------------------------------------------
# Cell 1: Setup and Kernel Definition
%%writefile sgemm_kernel.cu
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    
    if (cRow < M && cCol < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
    }
}

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "CUBLAS error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

float* create_and_init_matrix(int rows, int cols) {
    float* mat = new float[rows * cols];
    init_matrix(mat, rows, cols);
    return mat;
}

bool verify_result(float* C1, float* C2, int M, int N, float tolerance = 1e-3) {
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(C1[i] - C2[i]);
        if (diff > tolerance) {
            printf("Mismatch at index %d: %f vs %f (diff: %f)\n", 
                   i, C1[i], C2[i], diff);
            return false;
        }
    }
    return true;
}

extern "C" {
    void benchmark_kernels(int M, int N, int K, int iterations) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const int BLOCKSIZE = 32;
        
        // Allocate host memory
        float* h_A = create_and_init_matrix(M, K);
        float* h_B = create_and_init_matrix(K, N);
        float* h_C_custom = new float[M * N]();
        float* h_C_cublas = new float[M * N]();
        
        // Allocate device memory
        float *d_A, *d_B, *d_C_custom, *d_C_cublas;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C_custom, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C_cublas, M * N * sizeof(float)));
        
        // Copy data to device
        CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C_custom, 0, M * N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_C_cublas, 0, M * N * sizeof(float)));
        
        // Setup for custom kernel
        dim3 gridDim((M + BLOCKSIZE - 1) / BLOCKSIZE, 
                     (N + BLOCKSIZE - 1) / BLOCKSIZE);
        dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
        
        // Warmup
        for (int i = 0; i < 5; i++) {
            sgemm_global_mem_coalesce<BLOCKSIZE><<<gridDim, blockDim>>>(
                M, N, K, alpha, d_A, d_B, beta, d_C_custom);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Benchmark custom kernel
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            sgemm_global_mem_coalesce<BLOCKSIZE><<<gridDim, blockDim>>>(
                M, N, K, alpha, d_A, d_B, beta, d_C_custom);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float custom_time_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&custom_time_ms, start, stop));
        custom_time_ms /= iterations;
        
        // Setup cuBLAS
        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));
        
        // Warmup cuBLAS
        for (int i = 0; i < 5; i++) {
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K, &alpha, d_B, N, d_A, K,
                                     &beta, d_C_cublas, N));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Benchmark cuBLAS
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K, &alpha, d_B, N, d_A, K,
                                     &beta, d_C_cublas, N));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float cublas_time_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&cublas_time_ms, start, stop));
        cublas_time_ms /= iterations;
        
        // Calculate GFLOPS
        double flops = 2.0 * M * N * K;
        double custom_gflops = (flops / custom_time_ms) / 1e6;
        double cublas_gflops = (flops / cublas_time_ms) / 1e6;
        
        // Copy results back for verification
        CHECK_CUDA(cudaMemcpy(h_C_custom, d_C_custom, M * N * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_C_cublas, d_C_cublas, M * N * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        // Verify results
        bool results_match = verify_result(h_C_custom, h_C_cublas, M, N);
        
        // Print results
        printf("Matrix Size: %dx%dx%d\n", M, N, K);
        printf("Custom Kernel: %.3f ms (%.2f GFLOPS)\n", custom_time_ms, custom_gflops);
        printf("cuBLAS:        %.3f ms (%.2f GFLOPS)\n", cublas_time_ms, cublas_gflops);
        printf("Speedup:       %.2fx\n", custom_time_ms / cublas_time_ms);
        printf("Results Match: %s\n", results_match ? "YES" : "NO");
        printf("----------------------------------------\n");
        
        // Write results to file for Python to read
        FILE* fp = fopen("/tmp/benchmark_results.txt", "a");
        fprintf(fp, "%d,%d,%d,%.3f,%.3f,%.2f,%.2f\n", 
                M, N, K, custom_time_ms, cublas_time_ms, custom_gflops, cublas_gflops);
        fclose(fp);
        
        // Cleanup
        CHECK_CUBLAS(cublasDestroy(handle));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C_custom));
        CHECK_CUDA(cudaFree(d_C_cublas));
        delete[] h_A;
        delete[] h_B;
        delete[] h_C_custom;
        delete[] h_C_cublas;
    }
}

int main() {
    // Check CUDA device
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("========================================\n\n");
    
    // Clear results file
    FILE* fp = fopen("/tmp/benchmark_results.txt", "w");
    fprintf(fp, "M,N,K,custom_ms,cublas_ms,custom_gflops,cublas_gflops\n");
    fclose(fp);
    
    // Test different matrix sizes
    int sizes[][3] = {
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {512, 1024, 256},
       
    };
    
    int iterations = 100;
    
    for (auto& size : sizes) {
        benchmark_kernels(size[0], size[1], size[2], iterations);
    }
    
    return 0;
}

=============================
# Cell 2: Compile the CUDA code
!nvcc -o sgemm_benchmark sgemm_kernel.cu -lcublas -O3 -arch=sm_75 -std=c++11
# Cell 3: Run the benchmark
!./sgemm_benchmark
===============================
# Cell 4: Load and visualize results
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load results
df = pd.read_csv('/tmp/benchmark_results.txt')

# Create matrix size label
df['Size'] = df.apply(lambda row: f"{row['M']}x{row['N']}x{row['K']}", axis=1)
df['Speedup'] = df['custom_ms'] / df['cublas_ms']

# Display results table
print("Benchmark Results:")
print("=" * 80)
display(df[['Size', 'custom_ms', 'cublas_ms', 'custom_gflops', 'cublas_gflops', 'Speedup']])
===================================
# Cell 5: Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Time comparison
ax1 = axes[0, 0]
x = np.arange(len(df))
width = 0.35
ax1.bar(x - width/2, df['custom_ms'], width, label='Custom Kernel', color='steelblue')
ax1.bar(x + width/2, df['cublas_ms'], width, label='cuBLAS', color='coral')
ax1.set_xlabel('Matrix Size')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Execution Time Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Size'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# GFLOPS comparison
ax2 = axes[0, 1]
ax2.bar(x - width/2, df['custom_gflops'], width, label='Custom Kernel', color='steelblue')
ax2.bar(x + width/2, df['cublas_gflops'], width, label='cuBLAS', color='coral')
ax2.set_xlabel('Matrix Size')
ax2.set_ylabel('GFLOPS')
ax2.set_title('Performance (GFLOPS) Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(df['Size'], rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Speedup plot
ax3 = axes[1, 0]
colors = ['red' if s < 1 else 'green' for s in df['Speedup']]
ax3.bar(x, df['Speedup'], color=colors)
ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
ax3.set_xlabel('Matrix Size')
ax3.set_ylabel('Speedup (custom/cuBLAS)')
ax3.set_title('Speedup Factor (lower is better)')
ax3.set_xticks(x)
ax3.set_xticklabels(df['Size'], rotation=45, ha='right')
ax3.grid(True, alpha=0.3)

# Efficiency plot
ax4 = axes[1, 1]
df['Efficiency'] = (df['cublas_gflops'] / df['custom_gflops']) * 100
ax4.plot(x, df['Efficiency'], marker='o', linewidth=2, markersize=8, color='purple')
ax4.set_xlabel('Matrix Size')
ax4.set_ylabel('Efficiency (%)')
ax4.set_title('cuBLAS Efficiency vs Custom Kernel')
ax4.set_xticks(x)
ax4.set_xticklabels(df['Size'], rotation=45, ha='right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
======================================================
# Cell 6: Performance Summary
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)

avg_speedup = df['Speedup'].mean()
best_speedup = df['Speedup'].min()
worst_speedup = df['Speedup'].max()

print(f"\nAverage Speedup Factor: {avg_speedup:.2f}x")
print(f"Best Speedup Factor: {best_speedup:.2f}x")
print(f"Worst Speedup Factor: {worst_speedup:.2f}x")

print(f"\nAverage Custom Kernel Performance: {df['custom_gflops'].mean():.2f} GFLOPS")
print(f"Average cuBLAS Performance: {df['cublas_gflops'].mean():.2f} GFLOPS")

efficiency = (df['custom_gflops'].mean() / df['cublas_gflops'].mean()) * 100
print(f"\nOverall Efficiency vs cuBLAS: {efficiency:.1f}%")

# Find best and worst cases
best_case = df.loc[df['Speedup'].idxmin()]
worst_case = df.loc[df['Speedup'].idxmax()]

print(f"\nBest Performance Case: {best_case['Size']}")
print(f"  Custom: {best_case['custom_ms']:.2f} ms ({best_case['custom_gflops']:.2f} GFLOPS)")
print(f"  cuBLAS: {best_case['cublas_ms']:.2f} ms ({best_case['cublas_gflops']:.2f} GFLOPS)")

print(f"\nWorst Performance Case: {worst_case['Size']}")
print(f"  Custom: {worst_case['custom_ms']:.2f} ms ({worst_case['custom_gflops']:.2f} GFLOPS)")
print(f"  cuBLAS: {worst_case['cublas_ms']:.2f} ms ({worst_case['cublas_gflops']:.2f} GFLOPS)")
