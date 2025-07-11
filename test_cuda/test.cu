#include <iostream>

__global__ void hello() {
    // 不使用 printf
}

int main() {
    hello<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "CUDA Kernel executed successfully!" << std::endl;
    return 0;
}

