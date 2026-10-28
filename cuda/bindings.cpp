// PyTorch bindings for the SGM kernels, built as the `stereo_depth_cuda` extension by setup.py.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "sgm/sgm.h"

namespace {

void check_image(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kUInt8, name, " must be uint8");
    TORCH_CHECK(t.dim() == 2, name, " must be a 2D grey image (H, W)");
}

void check(cudaError_t err) { TORCH_CHECK(err == cudaSuccess, "SGM failed: ", cudaGetErrorString(err)); }

struct Workspace {
    torch::Tensor census_left, census_right, cost, cost_sum;
    sgm::Buffers buffers() {
        return {reinterpret_cast<uint64_t*>(census_left.data_ptr<int64_t>()),
                reinterpret_cast<uint64_t*>(census_right.data_ptr<int64_t>()),
                cost.data_ptr<uint8_t>(), cost_sum.data_ptr<int16_t>()};
    }
};

Workspace allocate(const torch::Tensor& like, int64_t max_disp) {
    const auto h = like.size(0), w = like.size(1);
    const auto opts = like.options();
    return {torch::empty({h, w}, opts.dtype(torch::kInt64)),
            torch::empty({h, w}, opts.dtype(torch::kInt64)),
            torch::empty({h, w, max_disp}, opts.dtype(torch::kUInt8)),
            torch::empty({h, w, max_disp}, opts.dtype(torch::kInt16))};
}

sgm::Params params(int64_t max_disp, int64_t p1, int64_t p2) {
    TORCH_CHECK(max_disp > 0 && max_disp % 32 == 0 && max_disp <= sgm::kMaxDisp,
                "max_disp must be a multiple of 32 in 32..", sgm::kMaxDisp);
    TORCH_CHECK(0 < p1 && p1 < p2 && p2 <= 255 - sgm::kInvalidCost, "need 0 < p1 < p2 <= ",
                255 - sgm::kInvalidCost);
    return {static_cast<int>(max_disp), static_cast<int>(p1), static_cast<int>(p2)};
}

void check_pair(const torch::Tensor& left, const torch::Tensor& right) {
    check_image(left, "left");
    check_image(right, "right");
    TORCH_CHECK(left.sizes() == right.sizes(), "left and right must be the same size");
    TORCH_CHECK(left.device() == right.device(), "left and right must be on the same device");
}

torch::Tensor census(torch::Tensor image) {
    check_image(image, "image");
    const c10::cuda::CUDAGuard guard(image.device());
    image = image.contiguous();
    auto out = torch::empty(image.sizes(), image.options().dtype(torch::kInt64));
    check(sgm::census(image.data_ptr<uint8_t>(), image.size(0), image.size(1),
                      reinterpret_cast<uint64_t*>(out.data_ptr<int64_t>()),
                      at::cuda::getCurrentCUDAStream()));
    return out;
}

torch::Tensor cost_sum(torch::Tensor left, torch::Tensor right, int64_t max_disp, int64_t p1,
                       int64_t p2) {
    check_pair(left, right);
    const auto p = params(max_disp, p1, p2);
    const c10::cuda::CUDAGuard guard(left.device());
    left = left.contiguous();
    right = right.contiguous();
    auto ws = allocate(left, max_disp);
    check(sgm::aggregate(left.data_ptr<uint8_t>(), right.data_ptr<uint8_t>(), left.size(0),
                         left.size(1), p, ws.buffers(), at::cuda::getCurrentCUDAStream()));
    return ws.cost_sum;
}

torch::Tensor disparity(torch::Tensor left, torch::Tensor right, int64_t max_disp, int64_t p1,
                        int64_t p2) {
    check_pair(left, right);
    const auto p = params(max_disp, p1, p2);
    const c10::cuda::CUDAGuard guard(left.device());
    left = left.contiguous();
    right = right.contiguous();
    auto ws = allocate(left, max_disp);
    auto out = torch::empty(left.sizes(), left.options().dtype(torch::kFloat32));
    check(sgm::disparity(left.data_ptr<uint8_t>(), right.data_ptr<uint8_t>(), left.size(0),
                         left.size(1), p, ws.buffers(), out.data_ptr<float>(),
                         at::cuda::getCurrentCUDAStream()));
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Semi-global matching stereo on CUDA";
    m.def("census", &census, "9x7 census transform as int64 bit patterns", py::arg("image"));
    m.def("cost_sum", &cost_sum, "4-path aggregated cost volume (H, W, max_disp) int16",
          py::arg("left"), py::arg("right"), py::arg("max_disp") = 128, py::arg("p1") = 10,
          py::arg("p2") = 120);
    m.def("disparity", &disparity, "Disparity map (H, W) float32 in pixels", py::arg("left"),
          py::arg("right"), py::arg("max_disp") = 128, py::arg("p1") = 10, py::arg("p2") = 120);
}
