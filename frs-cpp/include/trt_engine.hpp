#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

class TRTEngine {
public:
    explicit TRTEngine(const std::string& engine_path);
    ~TRTEngine();
    void infer(const std::vector<const float*>& inputs, const std::vector<float*>& outputs);
    int              numBindings()    const;
    bool             isInput(int idx) const;
    std::vector<int> inputDims(int idx) const;
    size_t           bindingSize(int idx) const;
    int              maxBatchSize()   const;
private:
    std::unique_ptr<nvinfer1::IRuntime>         runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_{};
    mutable std::vector<void*>  d_buffers_;
    mutable std::vector<size_t> h_sizes_;
};
