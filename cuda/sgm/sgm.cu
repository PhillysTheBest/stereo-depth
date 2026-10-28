#include "sgm.h"

#include <algorithm>
#include <climits>

namespace sgm {
namespace {

constexpr int kWarp = 32;

__device__ __forceinline__ int clampi(int v, int lo, int hi) { return min(max(v, lo), hi); }

// One thread per pixel. Bit order matches the reference: row-major over the window, centre skipped.
__global__ void census_kernel(const uint8_t* __restrict__ image, int height, int width,
                              uint64_t* __restrict__ out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const uint8_t centre = image[y * width + x];
    uint64_t bits = 0;
    for (int dy = -kCensusH / 2; dy <= kCensusH / 2; ++dy) {
        const int row = clampi(y + dy, 0, height - 1) * width;
        for (int dx = -kCensusW / 2; dx <= kCensusW / 2; ++dx) {
            if (dy == 0 && dx == 0) continue;
            const uint8_t neighbour = image[row + clampi(x + dx, 0, width - 1)];
            bits = (bits << 1) | static_cast<uint64_t>(neighbour < centre);
        }
    }
    out[y * width + x] = bits;
}

// One thread per (pixel, disparity), grid-stride so any image size fits.
__global__ void cost_kernel(const uint64_t* __restrict__ census_left,
                            const uint64_t* __restrict__ census_right, int width, int max_disp,
                            size_t total, uint8_t* __restrict__ cost) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int d = static_cast<int>(i % max_disp);
        const size_t pixel = i / max_disp;
        const int x = static_cast<int>(pixel % width);
        cost[i] = x >= d ? static_cast<uint8_t>(__popcll(census_left[pixel] ^ census_right[pixel - d]))
                         : static_cast<uint8_t>(kInvalidCost);
    }
}

// One block per scanline, one thread per disparity. (DX, DY) is the step between consecutive
// pixels: (1, 0) left to right, (-1, 0) right to left, (0, 1) top down, (0, -1) bottom up.
// Lr(p, d) = C(p, d) + min(Lr(q, d), Lr(q, d +- 1) + P1, min_k Lr(q, k) + P2) - min_k Lr(q, k)
template <int DX, int DY>
__global__ void path_kernel(const uint8_t* __restrict__ cost, int height, int width, int max_disp,
                            int p1, int p2, int16_t* __restrict__ cost_sum) {
    extern __shared__ int shared[];
    int* prev = shared;                 // Lr at the previous pixel, one per disparity
    int* warp_min = shared + max_disp;  // per-warp minimum of the current Lr

    const int d = threadIdx.x;
    const int lane = d % kWarp;
    const int warp = d / kWarp;
    const int n_warps = max_disp / kWarp;
    const int length = DX != 0 ? width : height;

    int lr_prev = 0;
    int min_prev = 0;
    for (int i = 0; i < length; ++i) {
        int x, y;
        if (DX != 0) {
            y = blockIdx.x;
            x = DX > 0 ? i : width - 1 - i;
        } else {
            x = blockIdx.x;
            y = DY > 0 ? i : height - 1 - i;
        }
        const size_t at = (static_cast<size_t>(y) * width + x) * max_disp + d;
        const int c = cost[at];

        int lr = c;
        if (i > 0) {
            int best = min(lr_prev, min_prev + p2);
            if (d > 0) best = min(best, prev[d - 1] + p1);
            if (d < max_disp - 1) best = min(best, prev[d + 1] + p1);
            lr = c + best - min_prev;
        }
        // Path kernels run one after another on the stream and each thread owns its (pixel, d),
        // so the sum needs no atomics.
        cost_sum[at] = static_cast<int16_t>(cost_sum[at] + lr);

        __syncthreads();  // every thread has read its neighbours in prev[]
        prev[d] = lr;
        int m = lr;
        for (int offset = kWarp / 2; offset > 0; offset /= 2)
            m = min(m, __shfl_xor_sync(0xffffffffu, m, offset));
        if (lane == 0) warp_min[warp] = m;
        __syncthreads();  // prev[] and warp_min[] are complete
        m = warp_min[0];
        for (int k = 1; k < n_warps; ++k) m = min(m, warp_min[k]);
        lr_prev = lr;
        min_prev = m;
    }
}

// One thread per pixel. Strict less-than keeps the smallest disparity on ties, like np.argmin.
__global__ void wta_kernel(const int16_t* __restrict__ cost_sum, int height, int width,
                           int max_disp, float* __restrict__ out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int16_t* s = cost_sum + (static_cast<size_t>(y) * width + x) * max_disp;
    int best = INT_MAX;
    int best_d = 0;
    for (int d = 0; d < max_disp; ++d) {
        if (s[d] < best) {
            best = s[d];
            best_d = d;
        }
    }
    out[y * width + x] = static_cast<float>(best_d);
}

dim3 pixel_grid(int height, int width, dim3 block) {
    return dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
}

bool valid_params(const Params& p) {
    return p.max_disp > 0 && p.max_disp % kWarp == 0 && p.max_disp <= kMaxDisp && p.p1 > 0 &&
           p.p1 < p.p2 && p.p2 <= 255 - kInvalidCost;
}

template <int DX, int DY>
void launch_path(const uint8_t* cost, int height, int width, const Params& p, int16_t* cost_sum,
                 cudaStream_t stream) {
    const int lines = DX != 0 ? height : width;
    const size_t shared = (p.max_disp + p.max_disp / kWarp) * sizeof(int);
    path_kernel<DX, DY><<<lines, p.max_disp, shared, stream>>>(cost, height, width, p.max_disp,
                                                               p.p1, p.p2, cost_sum);
}

}  // namespace

cudaError_t census(const uint8_t* image, int height, int width, uint64_t* out, cudaStream_t stream) {
    if (height <= 0 || width <= 0) return cudaErrorInvalidValue;
    const dim3 block(32, 8);
    census_kernel<<<pixel_grid(height, width, block), block, 0, stream>>>(image, height, width, out);
    return cudaGetLastError();
}

cudaError_t aggregate(const uint8_t* left, const uint8_t* right, int height, int width,
                      const Params& params, const Buffers& buf, cudaStream_t stream) {
    if (!valid_params(params)) return cudaErrorInvalidValue;
    cudaError_t err = census(left, height, width, buf.census_left, stream);
    if (err != cudaSuccess) return err;
    err = census(right, height, width, buf.census_right, stream);
    if (err != cudaSuccess) return err;

    const size_t total = static_cast<size_t>(height) * width * params.max_disp;
    const int threads = 256;
    const int blocks = static_cast<int>(std::min<size_t>((total + threads - 1) / threads, 65535));
    cost_kernel<<<blocks, threads, 0, stream>>>(buf.census_left, buf.census_right, width,
                                                params.max_disp, total, buf.cost);
    if ((err = cudaGetLastError()) != cudaSuccess) return err;

    err = cudaMemsetAsync(buf.cost_sum, 0, total * sizeof(int16_t), stream);
    if (err != cudaSuccess) return err;
    launch_path<1, 0>(buf.cost, height, width, params, buf.cost_sum, stream);
    launch_path<-1, 0>(buf.cost, height, width, params, buf.cost_sum, stream);
    launch_path<0, 1>(buf.cost, height, width, params, buf.cost_sum, stream);
    launch_path<0, -1>(buf.cost, height, width, params, buf.cost_sum, stream);
    return cudaGetLastError();
}

cudaError_t disparity(const uint8_t* left, const uint8_t* right, int height, int width,
                      const Params& params, const Buffers& buf, float* out, cudaStream_t stream) {
    cudaError_t err = aggregate(left, right, height, width, params, buf, stream);
    if (err != cudaSuccess) return err;
    const dim3 block(32, 8);
    wta_kernel<<<pixel_grid(height, width, block), block, 0, stream>>>(buf.cost_sum, height, width,
                                                                       params.max_disp, out);
    return cudaGetLastError();
}

}  // namespace sgm
