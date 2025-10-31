%%writefile benchmark.cu
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdlib>
#include <ctime>



// ----------------------------
// Kernel 1: Basic Matmul
// ----------------------------
__global__ void matmul_gpu(float* A, float* B, float* C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; ++l) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
}

// ----------------------------
// Kernel 2: SGEMM Naive
// ----------------------------
__global__ void sgemm_naive(int M, int N, int K,
                            float alpha, const float* A,
                            const float* B, float beta, float* C) {
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

// ----------------------------
// Helper: Measure kernel time
// ----------------------------
float measure_time_ms(void (*kernel)(float*, float*, float*, int, int, int),
                      float* A, float* B, float* C,
                      int M, int K, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    // Warm-up
    kernel<<<blocks, threads>>>(A, B, C, M, K, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel<<<blocks, threads>>>(A, B, C, M, K, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA(cudaGetLastError());

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

float measure_sgemm_naive(int M, int N, int K, float* A, float* B, float* C) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    // Warm-up
    sgemm_naive<<<blocks, threads>>>(M, N, K, 1.0f, A, B, 0.0f, C);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sgemm_naive<<<blocks, threads>>>(M, N, K, 1.0f, A, B, 0.0f, C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA(cudaGetLastError());

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

float measure_cublas(int M, int N, int K, float* A, float* B, float* C) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;

    // Warm-up
    CHECK_CUBLAS(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    CHECK_CUBLAS(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUBLAS(cublasDestroy(handle));
    return ms;
}

// ----------------------------
// Main Benchmark
// ----------------------------
int main() {
    srand(time(NULL));

    int M = 512, K = 512, N = 512;
    std::cout << "Matrix sizes: " << M << "x" << K << " * " << K << "x" << N << " = " << M << "x" << N << std::endl;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *A, *B, *C;
    CHECK_CUDA(cudaMallocManaged(&A, sizeA));
    CHECK_CUDA(cudaMallocManaged(&B, sizeB));
    CHECK_CUDA(cudaMallocManaged(&C, sizeC));

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; ++i) B[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;

    // Prefetch to GPU
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaMemPrefetchAsync(A, sizeA, device));
    CHECK_CUDA(cudaMemPrefetchAsync(B, sizeB, device));
    CHECK_CUDA(cudaMemPrefetchAsync(C, sizeC, device));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "\nRunning benchmarks..." << std::endl;

    // Benchmark
    float t1 = measure_time_ms(matmul_gpu, A, B, C, M, K, N);
    float t2 = measure_sgemm_naive(M, N, K, A, B, C);
    float t3 = measure_cublas(M, N, K, A, B, C);

    // GFLOPS = (2 * M * N * K) / (time * 1e6)
    float gflops1 = (2.0f * M * N * K) / (t1 * 1e6f);
    float gflops2 = (2.0f * M * N * K) / (t2 * 1e6f);
    float gflops3 = (2.0f * M * N * K) / (t3 * 1e6f);

    std::cout << "\n===== RESULTS =====" << std::endl;
    std::cout << "matmul_gpu:    " << t1 << " ms (" << gflops1 << " GFLOPS)" << std::endl;
    std::cout << "sgemm_naive:   " << t2 << " ms (" << gflops2 << " GFLOPS)" << std::endl;
    std::cout << "cuBLAS SGEMM:  " << t3 << " ms (" << gflops3 << " GFLOPS)" << std::endl;

    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
