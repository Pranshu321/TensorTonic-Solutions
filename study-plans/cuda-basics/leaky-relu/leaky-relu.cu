#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float alpha, int N) {
    // Write the float32 results into output; this kernel returns no value.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx<N){
        output[idx] = fmaxf(input[idx] , 0) + alpha*(fminf(input[idx],0));
    }
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}