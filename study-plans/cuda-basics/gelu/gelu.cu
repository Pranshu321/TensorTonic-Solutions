#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int N) {
    // Write the float32 results into output; this kernel returns no value.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = input[idx];

        const float kAlpha = 0.7978845608f; // sqrt(2/pi)

        float x3 = x * x * x;

        output[idx] =
            0.5f * x *
            (1.0f + tanhf(kAlpha * (x + 0.044715f * x3)));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    dim3 blocks((N + 255) / 256);
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
