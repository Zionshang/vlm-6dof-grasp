#include "ffs_gwc_plugin.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cerr << "[TRT] " << msg << "\n";
        }
    }
};

struct TrtDestroy {
    template <typename T>
    void operator()(T* p) const {
        delete p;
    }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDestroy>;

void printUsage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <plugin_onnx> <output_engine> [--fp32] [--workspace-mb N]\n"
        << "\n"
        << "Builds a single TensorRT engine from an ONNX graph containing the\n"
        << "FFSGWCVolume custom plugin node.\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        const std::filesystem::path onnx_path = argv[1];
        const std::filesystem::path engine_path = argv[2];
        bool fp16 = true;
        size_t workspace_mb = 4096;

        for (int i = 3; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--fp32") {
                fp16 = false;
            } else if (arg == "--workspace-mb" && i + 1 < argc) {
                workspace_mb = static_cast<size_t>(std::stoull(argv[++i]));
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        if (!std::filesystem::exists(onnx_path)) {
            throw std::runtime_error("ONNX file does not exist: " + onnx_path.string());
        }
        if (!engine_path.parent_path().empty()) {
            std::filesystem::create_directories(engine_path.parent_path());
        }

        if (!ffs_depth::registerFFSGWCPlugin()) {
            throw std::runtime_error("failed to register FFSGWCVolume plugin");
        }

        Logger logger;
        TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        if (!builder) throw std::runtime_error("createInferBuilder failed");

        const auto explicit_batch =
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicit_batch));
        if (!network) throw std::runtime_error("createNetworkV2 failed");

        TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
        if (!parser) throw std::runtime_error("createParser failed");

        if (!parser->parseFromFile(onnx_path.string().c_str(),
                                   static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING))) {
            for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
                std::cerr << parser->getError(i)->desc() << "\n";
            }
            throw std::runtime_error("failed to parse ONNX: " + onnx_path.string());
        }

        TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        if (!config) throw std::runtime_error("createBuilderConfig failed");
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                   workspace_mb * 1024ULL * 1024ULL);
        if (fp16 && builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        TrtPtr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
        if (!serialized) throw std::runtime_error("buildSerializedNetwork failed");

        std::ofstream out(engine_path, std::ios::binary);
        if (!out) throw std::runtime_error("cannot write engine: " + engine_path.string());
        out.write(static_cast<char const*>(serialized->data()), serialized->size());

        std::cout << "Built engine: " << engine_path << "\n";
        std::cout << "Precision: " << (fp16 ? "FP16 allowed" : "FP32") << "\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
