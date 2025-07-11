#include <torch/extension.h>

__global__ void add_kernel(float* x, float* y, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] + y[idx];
    }
}

void add_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor out) {
    int size = x.size(0);
    add_kernel<<<(size + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), size);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_cuda, "Add two tensors (CUDA)");
}

