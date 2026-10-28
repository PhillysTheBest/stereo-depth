// Standalone GPU check that needs no PyTorch: `make -C cuda test` on a machine with a GPU.
// Builds a random-texture pair with a known constant shift and checks SGM recovers it.
#include <cstdio>
#include <random>
#include <vector>

#include "../sgm/sgm.h"

#define CHECK(call)                                                                     \
    do {                                                                                \
        cudaError_t err_ = (call);                                                      \
        if (err_ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            return 1;                                                                   \
        }                                                                               \
    } while (0)

int main() {
    const int h = 375, w = 450, shift = 23;
    const sgm::Params params{64, 10, 120};

    // left(x) = base(x), right(x) = base(x + shift), so left x matches right x - shift.
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> pixel(0, 255);
    std::vector<uint8_t> base(static_cast<size_t>(h) * (w + shift));
    for (auto& v : base) v = static_cast<uint8_t>(pixel(rng));
    std::vector<uint8_t> left(static_cast<size_t>(h) * w), right(left.size());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            left[y * w + x] = base[y * (w + shift) + x];
            right[y * w + x] = base[y * (w + shift) + x + shift];
        }

    const size_t pixels = left.size(), volume = pixels * params.max_disp;
    uint8_t *d_left, *d_right;
    float* d_out;
    sgm::Buffers buf{};
    CHECK(cudaMalloc(&d_left, pixels));
    CHECK(cudaMalloc(&d_right, pixels));
    CHECK(cudaMalloc(&d_out, pixels * sizeof(float)));
    CHECK(cudaMalloc(&buf.census_left, pixels * sizeof(uint64_t)));
    CHECK(cudaMalloc(&buf.census_right, pixels * sizeof(uint64_t)));
    CHECK(cudaMalloc(&buf.cost, volume));
    CHECK(cudaMalloc(&buf.cost_sum, volume * sizeof(int16_t)));
    CHECK(cudaMemcpy(d_left, left.data(), pixels, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_right, right.data(), pixels, cudaMemcpyHostToDevice));

    CHECK(sgm::disparity(d_left, d_right, h, w, params, buf, d_out, nullptr));  // warm-up
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    CHECK(sgm::disparity(d_left, d_right, h, w, params, buf, d_out, nullptr));
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::vector<float> out(pixels);
    CHECK(cudaMemcpy(out.data(), d_out, pixels * sizeof(float), cudaMemcpyDeviceToHost));
    int good = 0, counted = 0;
    for (int y = 0; y < h; ++y)
        for (int x = shift + 4; x < w; ++x, ++counted) good += out[y * w + x] == shift;
    const double ratio = static_cast<double>(good) / counted;
    std::printf("%dx%d, %d disparities: %.2f ms, %.2f%% exact\n", w, h, params.max_disp, ms,
                100 * ratio);
    return ratio > 0.98 ? 0 : 1;
}
