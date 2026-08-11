#include "ffs_depth_tensorrt.hpp"
#include "ffs_depth_single_tensorrt.hpp"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
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

struct CudaBuffer {
    void* ptr = nullptr;
    ~CudaBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t bytes) { checkCuda(cudaMalloc(&ptr, bytes), "cudaMalloc"); }
    template <typename T> T* as() { return static_cast<T*>(ptr); }
};

struct Intrinsics {
    float k[9] = {};
    float baseline = 0.0f;
};

struct Args {
    std::string engine_dir;
    std::string left_path;
    std::string right_path;
    std::string intrinsic_path;
    std::string mode = "auto";
    int warmup = 10;
    int runs = 30;
    bool include_depth = false;
};

struct Stats {
    double mean = 0.0;
    double min = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

void printUsage(const char* prog) {
    std::cerr
        << "Usage: " << prog
        << " <engine_dir> <left_image> <right_image> <intrinsic_file>"
        << " [--mode auto|two|single] [--warmup N] [--runs N] [--include-depth]\n";
}

Args parseArgs(int argc, char** argv) {
    if (argc < 5) {
        printUsage(argv[0]);
        throw std::runtime_error("missing required arguments");
    }

    Args args;
    args.engine_dir = argv[1];
    args.left_path = argv[2];
    args.right_path = argv[3];
    args.intrinsic_path = argv[4];

    for (int i = 5; i < argc; ++i) {
        const std::string key = argv[i];
        auto requireValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (key == "--mode") {
            args.mode = requireValue("--mode");
        } else if (key == "--warmup") {
            args.warmup = std::stoi(requireValue("--warmup"));
        } else if (key == "--runs") {
            args.runs = std::stoi(requireValue("--runs"));
        } else if (key == "--include-depth") {
            args.include_depth = true;
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }

    if (args.mode != "auto" && args.mode != "two" && args.mode != "single") {
        throw std::runtime_error("--mode must be auto, two, or single");
    }
    if (args.warmup < 0 || args.runs <= 0) {
        throw std::runtime_error("--warmup must be >= 0 and --runs must be > 0");
    }
    return args;
}

Intrinsics loadIntrinsics(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open intrinsic file: " + path);
    std::string k_line;
    std::string baseline_line;
    if (!std::getline(in, k_line) || !std::getline(in, baseline_line)) {
        throw std::runtime_error("intrinsic file must contain K on line 1 and baseline on line 2");
    }

    Intrinsics intr;
    std::istringstream ks(k_line);
    for (float& v : intr.k) {
        if (!(ks >> v)) throw std::runtime_error("K line must contain 9 floats");
    }
    std::istringstream bs(baseline_line);
    if (!(bs >> intr.baseline)) throw std::runtime_error("baseline line must contain one float");
    return intr;
}

Stats summarize(std::vector<float> values) {
    if (values.empty()) throw std::runtime_error("cannot summarize empty timing vector");
    std::sort(values.begin(), values.end());
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const double mean = sum / static_cast<double>(values.size());
    double var = 0.0;
    for (float v : values) {
        const double d = static_cast<double>(v) - mean;
        var += d * d;
    }
    var /= static_cast<double>(values.size());

    auto percentile = [&](double p) {
        const size_t idx = static_cast<size_t>(
            std::llround((values.size() - 1) * p / 100.0));
        return static_cast<double>(values[std::min(idx, values.size() - 1)]);
    };

    Stats stats;
    stats.mean = mean;
    stats.min = values.front();
    stats.p50 = percentile(50.0);
    stats.p90 = percentile(90.0);
    stats.max = values.back();
    stats.stddev = std::sqrt(var);
    return stats;
}

void printStats(const char* label, const Stats& s) {
    std::cout
        << label
        << " mean_ms=" << s.mean
        << " p50_ms=" << s.p50
        << " p90_ms=" << s.p90
        << " min_ms=" << s.min
        << " max_ms=" << s.max
        << " std_ms=" << s.stddev
        << "\n";
}

template <typename Runner>
std::vector<float> profileRunner(
    Runner& runner,
    uint8_t* d_left,
    uint8_t* d_right,
    int height,
    int width,
    float* d_disp,
    float* d_depth,
    float fx,
    float baseline,
    int warmup,
    int runs,
    bool include_depth,
    std::vector<float>& host_ms)
{
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    std::vector<float> gpu_ms;
    gpu_ms.reserve(static_cast<size_t>(runs));
    host_ms.clear();
    host_ms.reserve(static_cast<size_t>(runs));

    for (int i = 0; i < warmup + runs; ++i) {
        const auto host_start = std::chrono::steady_clock::now();
        checkCuda(cudaEventRecord(start, runner.stream()), "cudaEventRecord start");
        runner.infer(d_left, d_right, height, width, d_disp);
        if (include_depth) {
            runner.dispToDepth(d_disp, height, width, fx, baseline, d_depth);
        }
        checkCuda(cudaEventRecord(stop, runner.stream()), "cudaEventRecord stop");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
        const auto host_stop = std::chrono::steady_clock::now();

        if (i >= warmup) {
            float elapsed = 0.0f;
            checkCuda(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
            gpu_ms.push_back(elapsed);
            host_ms.push_back(static_cast<float>(
                std::chrono::duration<double, std::milli>(host_stop - host_start).count()));
        }
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return gpu_ms;
}

template <typename Runner>
void runProfile(const Args& args,
                Runner& runner,
                uint8_t* d_left,
                uint8_t* d_right,
                int height,
                int width,
                float* d_disp,
                float* d_depth,
                const Intrinsics& intr,
                const char* mode_name) {
    std::vector<float> host_ms;
    const std::vector<float> gpu_ms = profileRunner(
        runner,
        d_left,
        d_right,
        height,
        width,
        d_disp,
        d_depth,
        intr.k[0],
        intr.baseline,
        args.warmup,
        args.runs,
        args.include_depth,
        host_ms);

    std::cout << "mode=" << mode_name << "\n";
    std::cout << "image=" << width << "x" << height << "\n";
    std::cout << "model=" << runner.modelWidth() << "x" << runner.modelHeight() << "\n";
    std::cout << "warmup=" << args.warmup << " runs=" << args.runs << "\n";
    std::cout << "timed_region=" << (args.include_depth ? "infer+dispToDepth" : "infer") << "\n";
    printStats("gpu", summarize(gpu_ms));
    printStats("host", summarize(host_ms));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parseArgs(argc, argv);
        const Intrinsics intr = loadIntrinsics(args.intrinsic_path);

        cv::Mat left = cv::imread(args.left_path, cv::IMREAD_COLOR);
        cv::Mat right = cv::imread(args.right_path, cv::IMREAD_COLOR);
        if (left.empty()) throw std::runtime_error("cannot read left image: " + args.left_path);
        if (right.empty()) throw std::runtime_error("cannot read right image: " + args.right_path);
        if (left.size() != right.size()) throw std::runtime_error("left/right size mismatch");
        if (!left.isContinuous()) left = left.clone();
        if (!right.isContinuous()) right = right.clone();

        const int height = left.rows;
        const int width = left.cols;
        const size_t image_bytes = static_cast<size_t>(height) * width * 3;
        const size_t map_bytes = static_cast<size_t>(height) * width * sizeof(float);

        CudaBuffer d_left;
        CudaBuffer d_right;
        CudaBuffer d_disp;
        CudaBuffer d_depth;
        d_left.allocate(image_bytes);
        d_right.allocate(image_bytes);
        d_disp.allocate(map_bytes);
        d_depth.allocate(map_bytes);
        checkCuda(cudaMemcpy(d_left.ptr, left.data, image_bytes, cudaMemcpyHostToDevice), "copy left");
        checkCuda(cudaMemcpy(d_right.ptr, right.data, image_bytes, cudaMemcpyHostToDevice), "copy right");

        std::string mode = args.mode;
        if (mode == "auto") {
            mode = std::filesystem::exists(
                std::filesystem::path(args.engine_dir) / "fast_foundationstereo.engine")
                ? "single"
                : "two";
        }

        if (mode == "single") {
            ffs_depth::FFSSingleEngineInference runner(args.engine_dir);
            runProfile(args, runner, d_left.as<uint8_t>(), d_right.as<uint8_t>(),
                       height, width, d_disp.as<float>(), d_depth.as<float>(),
                       intr, "single");
        } else {
            ffs_depth::FFSDepthInference runner(args.engine_dir);
            runProfile(args, runner, d_left.as<uint8_t>(), d_right.as<uint8_t>(),
                       height, width, d_disp.as<float>(), d_depth.as<float>(),
                       intr, "two");
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
