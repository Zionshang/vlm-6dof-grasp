#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <string>

#include "ffs_depth_tensorrt.hpp"

namespace ffs_depth {

/**
 * Single TensorRT engine inference path.
 *
 * Expected engine directory layout:
 *   - fast_foundationstereo.engine
 *   - onnx.yaml
 *
 * The engine is built from the plugin ONNX export path where the GWC cost
 * volume is represented by an FFSGWCVolume TensorRT plugin node.
 */
class FFSSingleEngineInference {
public:
    explicit FFSSingleEngineInference(const std::string& engine_dir);
    ~FFSSingleEngineInference();

    FFSSingleEngineInference(const FFSSingleEngineInference&) = delete;
    FFSSingleEngineInference& operator=(const FFSSingleEngineInference&) = delete;

    void infer(const uint8_t* d_left_rgb, const uint8_t* d_right_rgb,
               int input_h, int input_w,
               float* d_disp_out);

    void dispToDepth(const float* d_disp,
                     int height, int width,
                     float fx, float baseline_m,
                     float* d_depth_out);

    void inferDepth(const uint8_t* d_left_rgb, const uint8_t* d_right_rgb,
                    int input_h, int input_w,
                    float fx, float baseline_m,
                    float* d_depth_out);

    void sync();

    int modelHeight() const { return config_.image_height; }
    int modelWidth() const { return config_.image_width; }
    int maxDisp() const { return config_.max_disp; }
    int cvGroup() const { return config_.cv_group; }
    const FFSDepthInference::Config& config() const { return config_; }
    cudaStream_t stream() const { return stream_; }

private:
    void loadConfig(const std::string& config_path);
    void loadEngine(const std::string& path);
    void allocateBuffers();
    void freeDeviceBuffers();
    void preprocessRGBGPU(const uint8_t* d_rgb, int src_h, int src_w, float* d_output);

    FFSDepthInference::Config config_;

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    cudaStream_t stream_ = nullptr;

    float* d_left_ = nullptr;
    float* d_right_ = nullptr;
    float* d_disp_ = nullptr;
    float* d_disp_cropped_ = nullptr;
    float* d_disp_for_depth_ = nullptr;

    int64_t depth_alloc_pixels_ = 0;
    int scaled_w_ = 0;
    int scaled_h_ = 0;
};

}  // namespace ffs_depth
