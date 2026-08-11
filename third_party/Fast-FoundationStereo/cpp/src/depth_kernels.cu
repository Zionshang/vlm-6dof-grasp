/**
 * CUDA kernels for the Fast-FoundationStereo TensorRT depth inference pipeline.
 *
 * Active kernels:
 *   1.  GWC volume (mixed precision) -- FP32 features -> FP16 correlation volume
 *   2.  HWC uint8 -> CHW float      -- exact-match path (no resize, with zero-padding)
 *   3. Uniform resize + pad        -- aspect-ratio-preserving bilinear resize with
 *                                      border-replicate padding to model dimensions
 *   4.  Disparity crop              -- removes padding from model-resolution disparity
 *   5.  Disparity upsample          -- nearest-neighbor upsample with scale correction
 *   6.  Disparity clamp             -- clamps to minimum value
 *   7.  Disparity to depth          -- depth_m = fx * baseline_m / disparity (float32, meters)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cstdint>

namespace ffs_depth {
namespace cuda {

// =========================================================================
// 1. GWC Volume
// =========================================================================
template <typename T>
__device__ __forceinline__ float toFloat(T v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float toFloat<__half>(__half v) {
    return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float v) {
    return static_cast<T>(v);
}

template <>
__device__ __forceinline__ __half fromFloat<__half>(float v) {
    return __float2half(v);
}

template <typename InputT, typename OutputT>
__global__ void buildGWCVolumeKernel(
    const InputT* __restrict__ feat_left,
    const InputT* __restrict__ feat_right,
    OutputT* __restrict__ gwc_volume,
    int B, int C, int H, int W,
    int max_disp, int num_groups, bool normalize)
{
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int dgb = blockIdx.z;

    if (w >= W || h >= H) return;

    const int d = dgb % max_disp;
    const int g = (dgb / max_disp) % num_groups;
    const int b = dgb / (max_disp * num_groups);
    if (b >= B) return;

    const int K = C / num_groups;
    const int w_right = w - d;

    const int out_idx = ((b * num_groups + g) * max_disp + d) * H * W + h * W + w;

    if (w_right < 0) { gwc_volume[out_idx] = fromFloat<OutputT>(0.0f); return; }

    float dot = 0.f, nl = 0.f, nr = 0.f;
    const int left_base  = (b * C + g * K) * H * W + h * W + w;
    const int right_base = (b * C + g * K) * H * W + h * W + w_right;
    const int stride = H * W;

    for (int k = 0; k < K; ++k) {
        float l = toFloat<InputT>(feat_left [left_base  + k * stride]);
        float r = toFloat<InputT>(feat_right[right_base + k * stride]);
        dot += l * r;  nl += l * l;  nr += r * r;
    }
    if (normalize)
        gwc_volume[out_idx] = fromFloat<OutputT>(dot / (sqrtf(nl) * sqrtf(nr) + 1e-5f));
    else
        gwc_volume[out_idx] = fromFloat<OutputT>(dot);
}

// =========================================================================
// 2. Preprocess RGB HWC uint8 -> CHW float  (exact-match path, no resize)
//    Used when input resolution matches model resolution exactly.
// =========================================================================
__global__ void preprocessRGBToCHWKernel(
    const uint8_t* __restrict__ d_rgb_hwc,
    float* __restrict__ d_chw,
    int src_h, int src_w,
    int dst_h, int dst_w)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const int hw  = dst_h * dst_w;
    const int idx = y * dst_w + x;

    if (x < src_w && y < src_h) {
        const int si = (y * src_w + x) * 3;
        d_chw[idx]          = static_cast<float>(d_rgb_hwc[si + 2]);  // R
        d_chw[hw + idx]     = static_cast<float>(d_rgb_hwc[si + 1]);  // G
        d_chw[2 * hw + idx] = static_cast<float>(d_rgb_hwc[si]);      // B
    } else {
        d_chw[idx] = 0.f;  d_chw[hw + idx] = 0.f;  d_chw[2 * hw + idx] = 0.f;
    }
}

// =========================================================================
// 3. Uniform resize + border-replicate padding  (aspect-ratio preserving)
//    Pixels in [0, scaled_w) x [0, scaled_h) are bilinear-sampled from src.
//    Pixels in the padding region replicate the nearest edge pixel.
// =========================================================================
__global__ void resizeUniformAndPadKernel(
    const uint8_t* __restrict__ d_rgb_hwc,
    float* __restrict__ d_chw,
    int src_h, int src_w,
    int scaled_h, int scaled_w,
    int dst_h, int dst_w)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const int hw  = dst_h * dst_w;
    const int idx = y * dst_w + x;

    const int cx = min(x, scaled_w - 1);
    const int cy = min(y, scaled_h - 1);

    const float sx = static_cast<float>(src_w) / scaled_w;
    const float sy = static_cast<float>(src_h) / scaled_h;

    const float fx = (cx + 0.5f) * sx - 0.5f;
    const float fy = (cy + 0.5f) * sy - 0.5f;

    const int x0 = max(0, min(__float2int_rd(fx), src_w - 1));
    const int y0 = max(0, min(__float2int_rd(fy), src_h - 1));
    const int x1 = min(x0 + 1, src_w - 1);
    const int y1 = min(y0 + 1, src_h - 1);

    const float wx = fx - floorf(fx);
    const float wy = fy - floorf(fy);

    for (int c = 0; c < 3; ++c) {
        const int sc = 2 - c;  // RGB channel reorder
        float v00 = static_cast<float>(d_rgb_hwc[(y0 * src_w + x0) * 3 + sc]);
        float v01 = static_cast<float>(d_rgb_hwc[(y0 * src_w + x1) * 3 + sc]);
        float v10 = static_cast<float>(d_rgb_hwc[(y1 * src_w + x0) * 3 + sc]);
        float v11 = static_cast<float>(d_rgb_hwc[(y1 * src_w + x1) * 3 + sc]);
        float val = (1.f - wy) * ((1.f - wx) * v00 + wx * v01)
                  +        wy  * ((1.f - wx) * v10 + wx * v11);
        d_chw[c * hw + idx] = val;
    }
}

// =========================================================================
// 4. Crop disparity  (remove border-replicate padding)
//    Extracts the valid (scaled_w x scaled_h) region from the
//    model-resolution (model_w x model_h) disparity output.
// =========================================================================
__global__ void cropDisparityKernel(
    const float* __restrict__ src, float* __restrict__ dst,
    int src_w, int dst_h, int dst_w)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;
    dst[y * dst_w + x] = src[y * src_w + x];
}

// =========================================================================
// 5. Upsample float disparity (nearest neighbor) with scale correction
//    Upsamples from cropped model resolution to input resolution.
//    disp_scale converts disparity from scaled-pixel units to input-pixel units.
// =========================================================================
__global__ void upsampleDisparityKernel(
    const float* __restrict__ src, float* __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float disp_scale)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const float sx = static_cast<float>(src_w) / dst_w;
    const float sy = static_cast<float>(src_h) / dst_h;

    const float fx = (x + 0.5f) * sx - 0.5f;
    const float fy = (y + 0.5f) * sy - 0.5f;

    const int x0 = max(0, min(__float2int_rn(fx), src_w - 1));
    const int y0 = max(0, min(__float2int_rn(fy), src_h - 1));
    float val = src[y0 * src_w + x0];

    dst[y * dst_w + x] = val * disp_scale;
}

// =========================================================================
// 6. Clamp disparity to minimum value (removes negative/zero artifacts)
// =========================================================================
__global__ void clampDisparityKernel(
    float* __restrict__ disp, int count, float min_val)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    disp[idx] = fmaxf(disp[idx], min_val);
}

// =========================================================================
// 7. Disparity to depth: depth_m = fx * baseline_m / disparity (float32 meters)
//
//    Matches scripts/run_demo.py behaviour:
//      - disparity is assumed already clamped to >= 0 (see clampDisparityKernel)
//      - pixels where the right-image correspondence would fall off-image
//        (x - disp < 0) are marked invalid by setting disparity to +inf,
//        which yields depth = 0 after division.
//      - disp == 0 yields depth = +inf (matches numpy's float divide-by-zero).
//      - no clip on the upper end; output is float32 meters.
// =========================================================================
__global__ void dispToDepthKernel(
    const float* __restrict__ disp,
    float* __restrict__ depth_m,
    int height, int width, float fx, float baseline_m)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int i = y * width + x;

    float d = disp[i];
    if (static_cast<float>(x) - d < 0.0f) {
        d = CUDART_INF_F;  // remove_invisible: drop pixel via inf -> depth 0
    }
    depth_m[i] = fx * baseline_m / d;
}

// =========================================================================
// Host wrappers  (extern "C" for linkage)
// =========================================================================
extern "C" {

void ffsCudaBuildGWCVolumeFloat(
    const float* d_fl, const float* d_fr, float* d_gwc,
    int B, int C, int H, int W, int max_disp, int ngroups, bool normalize, cudaStream_t s)
{
    dim3 blk(16, 16);
    dim3 grd((W + 15) / 16, (H + 15) / 16, B * ngroups * max_disp);
    buildGWCVolumeKernel<float, float><<<grd, blk, 0, s>>>(d_fl, d_fr, d_gwc, B, C, H, W, max_disp, ngroups, normalize);
}

void ffsCudaBuildGWCVolumeHalf(
    const __half* d_fl, const __half* d_fr, __half* d_gwc,
    int B, int C, int H, int W, int max_disp, int ngroups, bool normalize, cudaStream_t s)
{
    dim3 blk(16, 16);
    dim3 grd((W + 15) / 16, (H + 15) / 16, B * ngroups * max_disp);
    buildGWCVolumeKernel<__half, __half><<<grd, blk, 0, s>>>(d_fl, d_fr, d_gwc, B, C, H, W, max_disp, ngroups, normalize);
}

void ffsCudaBuildGWCVolumeHalfToFloat(
    const __half* d_fl, const __half* d_fr, float* d_gwc,
    int B, int C, int H, int W, int max_disp, int ngroups, bool normalize, cudaStream_t s)
{
    dim3 blk(16, 16);
    dim3 grd((W + 15) / 16, (H + 15) / 16, B * ngroups * max_disp);
    buildGWCVolumeKernel<__half, float><<<grd, blk, 0, s>>>(d_fl, d_fr, d_gwc, B, C, H, W, max_disp, ngroups, normalize);
}

void ffsCudaBuildGWCVolumeMixed(
    const float* d_fl, const float* d_fr, __half* d_gwc,
    int B, int C, int H, int W, int max_disp, int ngroups, bool normalize, cudaStream_t s)
{
    dim3 blk(16, 16);
    dim3 grd((W + 15) / 16, (H + 15) / 16, B * ngroups * max_disp);
    buildGWCVolumeKernel<float, __half><<<grd, blk, 0, s>>>(d_fl, d_fr, d_gwc, B, C, H, W, max_disp, ngroups, normalize);
}

void ffsCudaPreprocessRGBToCHW(
    const uint8_t* d_rgb, float* d_chw,
    int src_h, int src_w, int dst_h, int dst_w, cudaStream_t s)
{
    dim3 blk(32, 16);
    dim3 grd((dst_w + 31) / 32, (dst_h + 15) / 16);
    preprocessRGBToCHWKernel<<<grd, blk, 0, s>>>(d_rgb, d_chw, src_h, src_w, dst_h, dst_w);
}

void ffsCudaResizeUniformAndPad(
    const uint8_t* d_rgb, float* d_chw,
    int src_h, int src_w, int scaled_h, int scaled_w,
    int dst_h, int dst_w, cudaStream_t s)
{
    dim3 blk(32, 16);
    dim3 grd((dst_w + 31) / 32, (dst_h + 15) / 16);
    resizeUniformAndPadKernel<<<grd, blk, 0, s>>>(d_rgb, d_chw, src_h, src_w, scaled_h, scaled_w, dst_h, dst_w);
}

void ffsCudaCropDisparity(
    const float* d_src, float* d_dst,
    int src_h, int src_w, int dst_h, int dst_w, cudaStream_t s)
{
    dim3 blk(32, 16);
    dim3 grd((dst_w + 31) / 32, (dst_h + 15) / 16);
    cropDisparityKernel<<<grd, blk, 0, s>>>(d_src, d_dst, src_w, dst_h, dst_w);
}

void ffsCudaUpsampleDisparity(
    const float* d_src, float* d_dst,
    int src_w, int src_h, int dst_w, int dst_h,
    float disp_scale, cudaStream_t s)
{
    dim3 blk(32, 16);
    dim3 grd((dst_w + 31) / 32, (dst_h + 15) / 16);
    upsampleDisparityKernel<<<grd, blk, 0, s>>>(d_src, d_dst, src_w, src_h, dst_w, dst_h, disp_scale);
}

void ffsCudaClampDisparity(
    float* d_disp, int count, float min_val, cudaStream_t s)
{
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    clampDisparityKernel<<<blocks, threads, 0, s>>>(d_disp, count, min_val);
}

void ffsCudaDispToDepth(
    const float* d_disp, float* d_depth_m,
    int height, int width, float fx, float baseline_m, cudaStream_t s)
{
    dim3 blk(32, 16);
    dim3 grd((width + 31) / 32, (height + 15) / 16);
    dispToDepthKernel<<<grd, blk, 0, s>>>(d_disp, d_depth_m, height, width, fx, baseline_m);
}

}  // extern "C"

}  // namespace cuda
}  // namespace ffs_depth
