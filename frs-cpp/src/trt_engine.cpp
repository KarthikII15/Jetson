// src/trt_engine.cpp — TensorRT 10.x compatible
#include "trt_engine.hpp"
#include <fstream>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <NvInferRuntime.h>

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            spdlog::warn("[TRT] {}", msg);
    }
};
static TRTLogger gLogger;

TRTEngine::TRTEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.good()) throw std::runtime_error("Cannot open engine: " + engine_path);
    size_t size = file.tellg(); file.seekg(0);
    std::vector<char> data(size); file.read(data.data(), size);

    runtime_.reset(nvinfer1::createInferRuntime(gLogger));
    engine_.reset(runtime_->deserializeCudaEngine(data.data(), size));
    if (!engine_) throw std::runtime_error("Failed to deserialize: " + engine_path);
    context_.reset(engine_->createExecutionContext());
    cudaStreamCreate(&stream_);
    spdlog::info("[TRT] Loaded: {} ({} tensors)", engine_path, engine_->getNbIOTensors());
}

TRTEngine::~TRTEngine() { cudaStreamDestroy(stream_); }

int TRTEngine::numBindings() const { return engine_->getNbIOTensors(); }

bool TRTEngine::isInput(int idx) const {
    return engine_->getTensorIOMode(engine_->getIOTensorName(idx)) == nvinfer1::TensorIOMode::kINPUT;
}

std::vector<int> TRTEngine::inputDims(int idx) const {
    auto dims = engine_->getTensorShape(engine_->getIOTensorName(idx));
    std::vector<int> out;
    for (int i = 0; i < dims.nbDims; ++i) out.push_back(dims.d[i]);
    return out;
}

size_t TRTEngine::bindingSize(int idx) const {
    auto dims = engine_->getTensorShape(engine_->getIOTensorName(idx));
    size_t n = 1;
    for (int i = 0; i < dims.nbDims; ++i) n *= (dims.d[i] > 0 ? dims.d[i] : 1);
    return n * sizeof(float);
}

int TRTEngine::maxBatchSize() const { return 1; }

void TRTEngine::infer(const std::vector<const float*>& inputs,
                      const std::vector<float*>& outputs) {
    int nb = engine_->getNbIOTensors();
    if (d_buffers_.empty()) {
        d_buffers_.resize(nb, nullptr);
        h_sizes_.resize(nb, 0);
        for (int i = 0; i < nb; ++i) {
            h_sizes_[i] = bindingSize(i);
            cudaMalloc(&d_buffers_[i], h_sizes_[i]);
            context_->setTensorAddress(engine_->getIOTensorName(i), d_buffers_[i]);
        }
    }
    int ii = 0;
    for (int i = 0; i < nb; ++i)
        if (isInput(i)) cudaMemcpyAsync(d_buffers_[i], inputs[ii++], h_sizes_[i], cudaMemcpyHostToDevice, stream_);
    context_->enqueueV3(stream_);
    int oi = 0;
    for (int i = 0; i < nb; ++i)
        if (!isInput(i)) cudaMemcpyAsync(outputs[oi++], d_buffers_[i], h_sizes_[i], cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
}
