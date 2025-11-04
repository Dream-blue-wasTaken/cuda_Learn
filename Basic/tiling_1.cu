// collab : https://colab.research.google.com/drive/10V9fMOfKx6NwcRnlXNculzdl8tctIyYt
// done with opus 16k thinking
# Google Colab Notebook: SGEMM Benchmark - Custom Kernel vs cuBLAS

# Cell 1: Setup and Write CUDA Code
%%writefile sgemm_benchmark.cu
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}

// Helper function to initialize matrices with random values
void initMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function to verify results (check if two matrices are close enough)
bool verifyResults(const float* A, const float* B, int size, float tolerance = 1e-3) {
    for (int i = 0; i < size; i++) {
        if (std::abs(A[i] - B[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << A[i] << " vs " << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Benchmark function for custom kernel
template <const int BLOCKSIZE>
float benchmarkCustomKernel(int M, int N, int K, int iterations) {
    float *d_A, *d_B, *d_C;
    float alpha = 1.0f, beta = 0.0f;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Initialize matrices
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));

    // Setup execution parameters
    dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);

    // Warmup
    for (int i = 0; i < 5; i++) {
        sgemm_shared_mem_block<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        sgemm_shared_mem_block<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    float milliseconds = std::chrono::duration<float, std::milli>(end - start).count();

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return milliseconds / iterations;
}

// Benchmark function for cuBLAS
float benchmarkCublas(int M, int N, int K, int iterations) {
    float *d_A, *d_B, *d_C;
    float alpha = 1.0f, beta = 0.0f;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Initialize matrices
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Warmup
    for (int i = 0; i < 5; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    float milliseconds = std::chrono::duration<float, std::milli>(end - start).count();

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    delete[] h_A;
    delete[] h_B;
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return milliseconds / iterations;
}

int main() {
    // Test different matrix sizes
    std::vector<int> sizes = {128, 256, 512, 768, 1024, 1536, 2048};
    const int iterations = 100;
    const int BLOCKSIZE = 32;

    std::cout << "SGEMM Benchmark: Custom Kernel vs cuBLAS\n";
    std::cout << "=========================================\n";
    std::cout << std::setw(10) << "Size"
              << std::setw(15) << "Custom (ms)"
              << std::setw(15) << "cuBLAS (ms)"
              << std::setw(15) << "Custom GFLOPS"
              << std::setw(15) << "cuBLAS GFLOPS"
              << std::setw(12) << "Speedup\n";
    std::cout << "=========================================\n";

    for (int size : sizes) {
        int M = size, N = size, K = size;

        // Run benchmarks
        float customTime = benchmarkCustomKernel<BLOCKSIZE>(M, N, K, iterations);
        float cublasTime = benchmarkCublas(M, N, K, iterations);

        // Calculate GFLOPS
        double flops = 2.0 * M * N * K;
        double customGflops = (flops / customTime) / 1e6;
        double cublasGflops = (flops / cublasTime) / 1e6;

        // Calculate speedup
        float speedup = customTime / cublasTime;

        std::cout << std::setw(10) << size
                  << std::setw(15) << std::fixed << std::setprecision(3) << customTime
                  << std::setw(15) << std::fixed << std::setprecision(3) << cublasTime
                  << std::setw(15) << std::fixed << std::setprecision(1) << customGflops
                  << std::setw(15) << std::fixed << std::setprecision(1) << cublasGflops
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x\n";
    }

    return 0;
}

===========================================================
# Cell 2: Compile the CUDA code
!nvcc -o sgemm_benchmark sgemm_benchmark.cu -lcublas -O3 -arch=sm_75 -std=c++11

# Note: Change -arch=sm_75 based on your GPU architecture
# For T4: sm_75, For V100: sm_70, For A100: sm_80

# Cell 3: Run the benchmark
!./sgemm_benchmark

=============================================================
04/11/25