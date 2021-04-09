#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__
void my_first_kernel(float *x) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    x[tid] = (float) threadIdx.x;
}

int main(int argc, char **argv) {
    float *h_x, *d_x;
    int nblocks = 3, nthreads = 4, nsize = 3*4;

    h_x = (float *)malloc(nsize*sizeof(float));
    cudaMalloc((void **)&d_x, nsize*sizeof(float));

    my_first_kernel<<<nblocks, nthreads>>>(d_x);
    cudaMemcpy(h_x, d_x, nsize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int n=0; n < nsize; n++) {
        printf(" n, x = %d %f \n", n , h_x[n]);
    }
    cudaFree(d_x); free(h_x);

    return 0;
}