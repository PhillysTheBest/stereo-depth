// Semi-global matching on the GPU. Mirrors backend/src/stereo_depth/sgm/reference.py exactly:
// same census window, border clamping, integer path costs and smallest-disparity tie-break.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace sgm {

constexpr int kCensusW = 9;
constexpr int kCensusH = 7;
// One more than the largest Hamming distance a 9x7 census can produce (62 bits).
constexpr int kInvalidCost = 64;
constexpr int kMaxDisp = 1024;

struct Params {
    int max_disp;  // multiple of 32, at most kMaxDisp
    int p1;
    int p2;
};

// All pointers are device memory. Volumes are laid out (H, W, max_disp), disparity fastest.
struct Buffers {
    uint64_t* census_left;   // H*W
    uint64_t* census_right;  // H*W
    uint8_t* cost;           // H*W*max_disp
    int16_t* cost_sum;       // H*W*max_disp
};

// Each call returns cudaErrorInvalidValue for bad sizes or params, otherwise the first launch error.
cudaError_t census(const uint8_t* image, int height, int width, uint64_t* out, cudaStream_t stream);

// Fills buf.census_*, buf.cost and buf.cost_sum (the 4-path aggregated cost).
cudaError_t aggregate(const uint8_t* left, const uint8_t* right, int height, int width,
                      const Params& params, const Buffers& buf, cudaStream_t stream);

// aggregate() followed by winner-take-all into `out` (H*W floats).
cudaError_t disparity(const uint8_t* left, const uint8_t* right, int height, int width,
                      const Params& params, const Buffers& buf, float* out, cudaStream_t stream);

}  // namespace sgm
