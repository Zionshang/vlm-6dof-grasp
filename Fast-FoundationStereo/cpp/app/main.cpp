/**
 * Standalone Fast-FoundationStereo TensorRT inference demo.
 *
 * Usage:
 *   ./ffs_depth_main <engine_dir> <left_image> <right_image> <intrinsic_file> [output_dir]
 *
 * Outputs:
 *   <output_dir>/disparity.bin   int32 H, int32 W, then H*W float32 disparity
 *   <output_dir>/depth_meter.bin int32 H, int32 W, then H*W float32 depth in meters
 *   <output_dir>/depth_meter.npy NumPy float32 depth in meters, shape (H, W)
 *   <output_dir>/disp_vis.png    left/right/colorized-disparity visualization
 *   <output_dir>/depth_vis.png   colorized depth visualization
 */

#include "ffs_depth_tensorrt.hpp"
#include "ffs_depth_single_tensorrt.hpp"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void checkCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void printUsage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <engine_dir> <left_image> <right_image> <intrinsic_file> [output_dir]\n"
        << "\n"
        << "  engine_dir : directory containing feature_runner.engine, post_runner.engine, onnx.yaml\n"
        << "  left_image : left stereo image readable by OpenCV\n"
        << "  right_image: right stereo image with the same size as left_image\n"
        << "  intrinsic_file: text file with 3x3 K on line 1 and baseline in meters on line 2\n"
        << "  output_dir : output directory (default: ffs_output)\n";
}

struct CudaBuffer {
    void* ptr = nullptr;

    ~CudaBuffer() {
        if (ptr) cudaFree(ptr);
    }

    CudaBuffer() = default;
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    void allocate(size_t bytes) {
        checkCuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
    }

    template <typename T>
    T* as() {
        return static_cast<T*>(ptr);
    }
};

struct Intrinsics {
    float k[9] = {};
    float baseline = 0.0f;
};

Intrinsics loadIntrinsics(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open intrinsic file: " + path);
    }

    std::string k_line;
    std::string baseline_line;
    if (!std::getline(in, k_line) || !std::getline(in, baseline_line)) {
        throw std::runtime_error("intrinsic file must contain K on line 1 and baseline on line 2");
    }

    Intrinsics intr;
    std::istringstream k_stream(k_line);
    for (float& value : intr.k) {
        if (!(k_stream >> value)) {
            throw std::runtime_error("intrinsic file K line must contain 9 float values");
        }
    }

    std::istringstream baseline_stream(baseline_line);
    if (!(baseline_stream >> intr.baseline)) {
        throw std::runtime_error("intrinsic file baseline line must contain one float value");
    }
    if (!(intr.k[0] > 0.0f) || !(intr.baseline > 0.0f)) {
        throw std::runtime_error("invalid focal length or baseline in intrinsic file");
    }

    return intr;
}

void saveFloatMatrix(const std::filesystem::path& path,
                     const std::vector<float>& values,
                     int height,
                     int width) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write matrix: " + path.string());
    }
    const int32_t dims[2] = {height, width};
    out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    out.write(reinterpret_cast<const char*>(values.data()),
              values.size() * sizeof(float));
}

void saveNpyFloat32(const std::filesystem::path& path,
                    const std::vector<float>& values,
                    int height,
                    int width) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write npy matrix: " + path.string());
    }

    std::ostringstream header_stream;
    header_stream << "{'descr': '<f4', 'fortran_order': False, 'shape': ("
                  << height << ", " << width << "), }";
    std::string header = header_stream.str();

    const size_t prefix_len = 10;  // magic(6) + version(2) + header_len(2)
    size_t padded_len = header.size() + 1;
    const size_t rem = (prefix_len + padded_len) % 16;
    if (rem != 0) {
        padded_len += 16 - rem;
    }
    header.append(padded_len - header.size() - 1, ' ');
    header.push_back('\n');

    if (header.size() > 65535) {
        throw std::runtime_error("npy header too large: " + path.string());
    }
    const uint16_t header_len = static_cast<uint16_t>(header.size());

    out.write("\x93NUMPY", 6);
    const char version[2] = {1, 0};
    out.write(version, 2);
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
}

cv::Mat colorizeDisparity(const std::vector<float>& disparity, int height, int width) {
    std::vector<float> safe_disp(disparity.size(), 0.0f);
    std::vector<uint8_t> valid_mask(disparity.size(), 0);
    for (size_t i = 0; i < disparity.size(); ++i) {
        if (std::isfinite(disparity[i])) {
            safe_disp[i] = std::max(disparity[i], 0.0f);
            valid_mask[i] = 255;
        }
    }

    cv::Mat disp_mat(height, width, CV_32FC1, safe_disp.data());
    cv::Mat valid(height, width, CV_8UC1, valid_mask.data());

    if (cv::countNonZero(valid) == 0) {
        return cv::Mat::zeros(height, width, CV_8UC3);
    }

    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(disp_mat, &min_val, &max_val, nullptr, nullptr, valid);
    if (max_val <= min_val) {
        max_val = min_val + 1.0;
    }

    cv::Mat disp_u8;
    disp_mat.convertTo(disp_u8, CV_8UC1, 255.0 / (max_val - min_val),
                       -255.0 * min_val / (max_val - min_val));

    cv::Mat colored;
    cv::applyColorMap(disp_u8, colored, cv::COLORMAP_TURBO);
    colored.setTo(cv::Scalar(0, 0, 0), ~valid);
    return colored;
}

cv::Mat colorizeDepth(const std::vector<float>& depth, int height, int width) {
    std::vector<float> safe_depth(depth.size(), 0.0f);
    std::vector<uint8_t> valid_mask(depth.size(), 0);
    for (size_t i = 0; i < depth.size(); ++i) {
        if (std::isfinite(depth[i]) && depth[i] > 0.0f) {
            safe_depth[i] = depth[i];
            valid_mask[i] = 255;
        }
    }

    cv::Mat depth_mat(height, width, CV_32FC1, safe_depth.data());
    cv::Mat valid(height, width, CV_8UC1, valid_mask.data());

    if (cv::countNonZero(valid) == 0) {
        return cv::Mat::zeros(height, width, CV_8UC3);
    }

    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(depth_mat, &min_val, &max_val, nullptr, nullptr, valid);
    if (max_val <= min_val) {
        max_val = min_val + 1.0;
    }

    cv::Mat depth_u8;
    depth_mat.convertTo(depth_u8, CV_8UC1, 255.0 / (max_val - min_val),
                        -255.0 * min_val / (max_val - min_val));

    cv::Mat colored;
    cv::applyColorMap(depth_u8, colored, cv::COLORMAP_TURBO);
    colored.setTo(cv::Scalar(0, 0, 0), ~valid);
    return colored;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        const std::string engine_dir = argv[1];
        const std::string left_path = argv[2];
        const std::string right_path = argv[3];
        const std::string intrinsic_path = argv[4];
        const std::filesystem::path output_dir = (argc == 6) ? argv[5] : "ffs_output";
        const Intrinsics intr = loadIntrinsics(intrinsic_path);

        cv::Mat left_bgr = cv::imread(left_path, cv::IMREAD_COLOR);
        cv::Mat right_bgr = cv::imread(right_path, cv::IMREAD_COLOR);

        if (left_bgr.empty()) {
            throw std::runtime_error("cannot read left image: " + left_path);
        }
        if (right_bgr.empty()) {
            throw std::runtime_error("cannot read right image: " + right_path);
        }
        if (left_bgr.size() != right_bgr.size()) {
            throw std::runtime_error("left and right images must have identical dimensions");
        }
        if (!left_bgr.isContinuous()) left_bgr = left_bgr.clone();
        if (!right_bgr.isContinuous()) right_bgr = right_bgr.clone();

        std::filesystem::create_directories(output_dir);

        const int height = left_bgr.rows;
        const int width = left_bgr.cols;
        const size_t image_bytes = static_cast<size_t>(height) * width * 3 * sizeof(uint8_t);
        const size_t map_bytes = static_cast<size_t>(height) * width * sizeof(float);

        CudaBuffer d_left;
        CudaBuffer d_right;
        CudaBuffer d_disparity;
        CudaBuffer d_depth;
        d_left.allocate(image_bytes);
        d_right.allocate(image_bytes);
        d_disparity.allocate(map_bytes);
        d_depth.allocate(map_bytes);

        checkCuda(cudaMemcpy(d_left.ptr, left_bgr.data, image_bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy left image");
        checkCuda(cudaMemcpy(d_right.ptr, right_bgr.data, image_bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy right image");

        std::cout << "Input images: " << width << "x" << height << "\n";
        std::cout << "Depth: fx=" << intr.k[0] << " baseline=" << intr.baseline << " m\n";
        std::cout << "Loading engines from: " << engine_dir << "\n";
        const bool use_single_engine =
            std::filesystem::exists(std::filesystem::path(engine_dir) / "fast_foundationstereo.engine");

        if (use_single_engine) {
            std::cout << "Runtime: single TensorRT engine with FFSGWCVolume plugin\n";
            ffs_depth::FFSSingleEngineInference ffs(engine_dir);
            ffs.infer(d_left.as<uint8_t>(), d_right.as<uint8_t>(), height, width,
                      d_disparity.as<float>());
            ffs.dispToDepth(d_disparity.as<float>(), height, width,
                            intr.k[0], intr.baseline, d_depth.as<float>());
            ffs.sync();
        } else {
            std::cout << "Runtime: two TensorRT engines with external CUDA GWC\n";
            ffs_depth::FFSDepthInference ffs(engine_dir);
            ffs.infer(d_left.as<uint8_t>(), d_right.as<uint8_t>(), height, width,
                      d_disparity.as<float>());
            ffs.dispToDepth(d_disparity.as<float>(), height, width,
                            intr.k[0], intr.baseline, d_depth.as<float>());
            ffs.sync();
        }

        std::vector<float> disparity(static_cast<size_t>(height) * width);
        std::vector<float> depth(static_cast<size_t>(height) * width);
        checkCuda(cudaMemcpy(disparity.data(), d_disparity.ptr, map_bytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy disparity");
        checkCuda(cudaMemcpy(depth.data(), d_depth.ptr, map_bytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy depth");

        const auto disparity_path = output_dir / "disparity.bin";
        saveFloatMatrix(disparity_path, disparity, height, width);

        const auto depth_path = output_dir / "depth_meter.bin";
        saveFloatMatrix(depth_path, depth, height, width);

        const auto depth_npy_path = output_dir / "depth_meter.npy";
        saveNpyFloat32(depth_npy_path, depth, height, width);

        const cv::Mat disp_color = colorizeDisparity(disparity, height, width);
        cv::Mat disp_vis;
        cv::hconcat(std::vector<cv::Mat>{left_bgr, right_bgr, disp_color}, disp_vis);
        const auto disp_vis_path = output_dir / "disp_vis.png";
        if (!cv::imwrite(disp_vis_path.string(), disp_vis)) {
            throw std::runtime_error("failed to write disparity visualization: " + disp_vis_path.string());
        }

        const cv::Mat depth_vis = colorizeDepth(depth, height, width);
        const auto depth_vis_path = output_dir / "depth_vis.png";
        if (!cv::imwrite(depth_vis_path.string(), depth_vis)) {
            throw std::runtime_error("failed to write depth visualization: " + depth_vis_path.string());
        }

        std::cout << "Saved: " << disparity_path << "\n";
        std::cout << "Saved: " << depth_path << "\n";
        std::cout << "Saved: " << depth_npy_path << "\n";
        std::cout << "Saved: " << disp_vis_path << "\n";
        std::cout << "Saved: " << depth_vis_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
