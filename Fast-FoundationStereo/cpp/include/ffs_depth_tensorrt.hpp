#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ffs_depth {

/**
 * FoundationStereo TensorRT depth inference.
 *
 * Uses the two-engine architecture (feature_runner + post_runner).
 * The GWC (Group-wise Correlation) volume is computed on GPU between the two engines.
 *
 * Expected engine directory layout:
 *   - feature_runner.engine
 *   - post_runner.engine
 *   - onnx.yaml  (contains image_size, max_disp, cv_group, valid_iters)
 */
class FFSDepthInference {
public:
    struct Config {
        int image_height = 480;   // Model input height (must be divisible by 32)
        int image_width  = 864;   // Model input width (must be divisible by 32)
        int max_disp     = 192;   // Maximum disparity search range in pixels
        int cv_group     = 8;     // Number of groups for group-wise correlation (GWC) volume
        int valid_iters  = 8;     // Number of GRU refinement iterations during inference
        bool normalize   = true;  // Whether to L2-normalize features in GWC correlation
    };

    /**
     * Load engines and allocate all GPU buffers.
     * @param engine_dir  Directory containing the two .engine files and onnx.yaml.
     */
    explicit FFSDepthInference(const std::string& engine_dir);
    ~FFSDepthInference();

    FFSDepthInference(const FFSDepthInference&) = delete;
    FFSDepthInference& operator=(const FFSDepthInference&) = delete;

    /**
     * Run stereo disparity inference entirely on the GPU.
     *
     * This method is asynchronous: all work (preprocessing, TensorRT
     * inference, postprocessing) is enqueued on an internal CUDA stream
     * and the call returns immediately. Call sync() to block until the
     * output buffer is safe to read.
     *
     * Input images are expected in HWC uint8 BGR (OpenCV) format on device memory.
     * If (input_h, input_w) differs from the model size, the images are
     * uniformly (aspect-preserving) bilinearly resized down to fit the model
     * input and the remaining right/bottom strip is filled with replicate
     * padding. After inference the disparity is cropped back to the scaled
     * region and nearest-neighbour upsampled to (input_h, input_w), with
     * a horizontal scale correction so the values are in input-pixel units.
     *
     * @param d_left_rgb    Left  image on GPU  (input_h x input_w x 3, uint8 BGR)
     * @param d_right_rgb   Right image on GPU  (input_h x input_w x 3, uint8 BGR)
     * @param input_h       Height of input images (must be > 0)
     * @param input_w       Width  of input images (must be > 0)
     * @param d_disp_out    Output disparity on GPU (input_h x input_w, float32,
     *                      in input-pixel units, clamped to >= 0).
     */
    void infer(const uint8_t* d_left_rgb, const uint8_t* d_right_rgb,
               int input_h, int input_w,
               float* d_disp_out);

    /**
     * Convert an input-resolution disparity map to float32 depth in meters.
     *
     * This method is asynchronous: work is enqueued on the internal CUDA
     * stream and the call returns immediately. Call sync() before reading.
     *
     * Conversion semantics match scripts/run_demo.py (see inferDepth docstring
     * for the +inf / off-image-correspondence handling).
     *
     * @param d_disp       Input disparity on GPU (height x width, float32).
     *                     Must be expressed in INPUT-image pixel units (which
     *                     is what infer() produces).
     * @param height       Disparity / depth height
     * @param width        Disparity / depth width
     * @param fx           Focal length in pixels at INPUT resolution
     * @param baseline_m   Stereo baseline in meters
     * @param d_depth_out  Output depth on GPU (height x width, float32, meters)
     */
    void dispToDepth(const float* d_disp,
                     int height, int width,
                     float fx, float baseline_m,
                     float* d_depth_out);

    /**
     * Run stereo inference and convert disparity to float32 depth (meters) in one call.
     *
     * This method is asynchronous: all work is enqueued on an internal
     * CUDA stream and the call returns immediately. Call sync() to block
     * until d_depth_out is safe to read.
     *
     * The conversion matches scripts/run_demo.py:
     *   depth_m = fx * baseline_m / disparity
     * where:
     *   - disparity is in INPUT-image pixel units (infer() already rescales it),
     *     so `fx` must come from the INPUT-resolution intrinsics (i.e. K[0,0]
     *     of the unscaled camera matrix, not the model-resolution intrinsics).
     *   - pixels with x - disparity < 0 (right-image correspondence off-image)
     *     are marked invalid: disparity is replaced with +inf, yielding depth 0.
     *   - disparity == 0 yields depth = +inf (consumers should mask non-finite
     *     values before using the depth map for downstream geometry).
     *
     * @param d_left_rgb   Left  image on GPU (input_h x input_w x 3, uint8 BGR/RGB)
     * @param d_right_rgb  Right image on GPU (input_h x input_w x 3, uint8 BGR/RGB)
     * @param input_h      Height of input images
     * @param input_w      Width of input images
     * @param fx           Focal length in pixels at INPUT resolution
     * @param baseline_m   Stereo baseline in meters
     * @param d_depth_out  Output depth on GPU (input_h x input_w, float32, meters)
     */
    void inferDepth(const uint8_t* d_left_rgb, const uint8_t* d_right_rgb,
                    int input_h, int input_w,
                    float fx, float baseline_m,
                    float* d_depth_out);

    /** Block until all async work on the internal CUDA stream has completed. */
    void sync();

    int modelHeight()  const { return config_.image_height; }
    int modelWidth()   const { return config_.image_width;  }
    int maxDisp()      const { return config_.max_disp;     }
    int cvGroup()      const { return config_.cv_group;     }
    const Config& config() const { return config_; }
    cudaStream_t stream() const { return stream_; }

private:
    void loadConfig(const std::string& config_path);
    void loadEngine(const std::string& path,
                    std::unique_ptr<nvinfer1::ICudaEngine>& engine,
                    std::unique_ptr<nvinfer1::IExecutionContext>& context);
    void allocateBuffers();
    void allocateFeatureBuffers();
    void allocatePostBuffers();
    void freeDeviceBuffers();  // safe to call repeatedly; nulls every pointer
    void preprocessRGBGPU(const uint8_t* d_rgb, int src_h, int src_w, float* d_output);

    void runFeatureRunner();
    void buildGWCVolume();
    void runPostRunner();

    Config config_;

    std::unique_ptr<nvinfer1::IRuntime>          runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>       feature_engine_;
    std::unique_ptr<nvinfer1::IExecutionContext>  feature_context_;
    std::unique_ptr<nvinfer1::ICudaEngine>       post_engine_;
    std::unique_ptr<nvinfer1::IExecutionContext>  post_context_;

    cudaStream_t stream_ = nullptr;

    float* d_left_  = nullptr;
    float* d_right_ = nullptr;

    float* d_feat_left_04_  = nullptr;
    float* d_feat_left_08_  = nullptr;
    float* d_feat_left_16_  = nullptr;
    float* d_feat_left_32_  = nullptr;
    float* d_feat_right_04_ = nullptr;
    float* d_stem_2x_       = nullptr;

    std::vector<int> feat_04_dims_;
    std::vector<int> feat_08_dims_;
    std::vector<int> feat_16_dims_;
    std::vector<int> feat_32_dims_;
    std::vector<int> stem_2x_dims_;

    float* d_gwc_volume_   = nullptr;
    int    gwc_disp_levels_ = 0;

    float* d_disp_         = nullptr;
    float* d_disp_cropped_ = nullptr;
    float* d_disp_for_depth_ = nullptr;
    int64_t depth_alloc_pixels_ = 0;

    int    scaled_w_ = 0;  // Uniform-scaled width (before padding)
    int    scaled_h_ = 0;  // Uniform-scaled height (before padding)

    bool gwc_fp16_  = false;  // GWC volume tensor is FP16 (vs. FP32)
};

}  // namespace ffs_depth
