// Placeholder kernel so CI proves nvcc builds cuda/. Replaced by the SGM kernels in Sprint 3.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_one(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}

int main() {
    const int n = 256;
    float* d = nullptr;
    if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) {
        std::printf("no CUDA device\n");
        return 1;
    }
    cudaMemset(d, 0, n * sizeof(float));
    add_one<<<1, n>>>(d, n);
    float h[n];
    cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
    cudaFree(d);
    std::printf("x[0] = %.1f\n", h[0]);
    return h[0] == 1.0f ? 0 : 1;
}
