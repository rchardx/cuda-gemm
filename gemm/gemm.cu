#include <algorithm>
#include <iostream>
#include <random>
#include <ranges>
#include <span>
#include <vector>

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include "gemm/impl.h"

// For power limited testing, please run 2000+ iterations
constexpr int warmup_iterations = 10;
constexpr int iterations = 2000;

constexpr size_t l2_size = 50 * 1024 * 1024;
void *to_flush = nullptr;

std::vector<uint64_t> kernel_durations;

void CUPTIAPI
getTimestampCallback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
    const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *)cbdata;
    uint64_t start_timestamp, end_timestamp;

    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API && cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
        if (cbInfo->callbackSite == CUPTI_API_ENTER) {
            cuptiGetTimestamp(&start_timestamp);
            *((uint64_t *)userdata) = start_timestamp;
        } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
            cuptiGetTimestamp(&end_timestamp);
            start_timestamp = *((uint64_t *)userdata);
        }
    }
}

void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t stream_id, uint8_t *buffer, size_t size, size_t valid_size) {
    CUptiResult status;
    CUpti_Activity *record = NULL;

    if (valid_size > 0) {
        do {
            status = cuptiActivityGetNextRecord(buffer, valid_size, &record);
            if (status == CUPTI_SUCCESS) {
                switch (record->kind) {
                case CUPTI_ACTIVITY_KIND_KERNEL:
                    {
                        CUpti_ActivityKernel5 *kernel = (CUpti_ActivityKernel5 *)record;
                        kernel_durations.push_back(kernel->end - kernel->start);
                        break;
                    }
                default:
                    break;
                }
            } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            }
        } while (1);
    }

    free(buffer);
}

void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size, size_t *max_num_records) {
    *size = 1024 * 1024;
    *buffer = (uint8_t *)malloc(*size);
    *max_num_records = 0;
}

void flush(void *ptr) {
    cudaMemsetAsync(ptr, 0, l2_size);
}

void fill_randn(std::vector<__nv_bfloat16> &data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0f, 0.5f);

    for (auto &value : data) {
        const float rand_value = std::clamp(distribution(gen), -1.0f, 1.0f);
        value = __nv_bfloat16(rand_value);
    }
}

void cublas(__nv_bfloat16 *A_device, __nv_bfloat16 *B_device, __nv_bfloat16 *C_device, int M, int N, int K) {
    cublasLtHandle_t handle;
    cublasLtCreate(&handle);

    cublasLtMatrixLayout_t A_desc, B_desc, C_desc;

    cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_16BF, N, K, K);
    cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_16BF, K, M, K);
    cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_16BF, N, M, N);

    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;

    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

    cublasDataType_t scale_type = CUDA_R_32F;
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type));

    float alpha = 1.0f;
    float beta = 0.0f;

    size_t workspace_size = 32 * 1024 * 1024;  // 32 MiB workspace
    void *workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);

    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size));

    cublasLtMatmulHeuristicResult_t heuristic_result;
    int returned_results = 0;

    cublasLtMatmulAlgoGetHeuristic(
        handle, matmul_desc, A_desc, B_desc, C_desc, C_desc, preference, 1, &heuristic_result, &returned_results);

    if (returned_results == 0) {
        std::cerr << "No algorithm found!" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < warmup_iterations; i++) {
        cublasLtMatmul(
            handle, matmul_desc, &alpha, B_device, A_desc, A_device, B_desc, &beta, C_device, C_desc, C_device, C_desc,
            &heuristic_result.algo, workspace, workspace_size, 0);
    }

    kernel_durations.clear();

    CUpti_SubscriberHandle subscriber;
    uint64_t userdata = 0;
    cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getTimestampCallback, &userdata);
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);

    cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);

    for (int i = 0; i < iterations; i++) {
        flush(to_flush);
        cublasLtMatmul(
            handle, matmul_desc, &alpha, B_device, A_desc, A_device, B_desc, &beta, C_device, C_desc, C_device, C_desc,
            &heuristic_result.algo, workspace, workspace_size, 0);
    }

    cudaDeviceSynchronize();

    cuptiActivityFlushAll(0);

    cuptiUnsubscribe(subscriber);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);

    double time_ns = 0.0;
    if (!kernel_durations.empty()) {
        time_ns = std::accumulate(kernel_durations.begin(), kernel_durations.end(), 0ull) / kernel_durations.size();
    }

    double flops = 2.0 * double(M) * double(N) * double(K);
    double tflops = flops / time_ns * 1e-3;

    std::cout << "cuBLAS Average time (ns): " << time_ns << std::endl;
    std::cout << "cuBLAS Performance: " << tflops << " TFLOPS" << std::endl;

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(matmul_desc);
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(C_desc);
    cublasLtDestroy(handle);

    cudaFree(workspace);
}

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t NUM_STAGES>
uint32_t get_smem_size() {
    uint32_t smem_size = 0;
    smem_size += sizeof(__nv_bfloat16) * (BLOCK_M * BLOCK_N);
    smem_size += sizeof(__nv_bfloat16) * (BLOCK_M * BLOCK_K) * NUM_STAGES;
    smem_size += sizeof(__nv_bfloat16) * (BLOCK_N * BLOCK_K) * NUM_STAGES;
    smem_size += sizeof(uint64_t) * 2 * NUM_STAGES;
    return smem_size;
}

void impl(
    __nv_bfloat16 *A_device, __nv_bfloat16 *B_device, __nv_bfloat16 *C_device, uint32_t M, uint32_t N, uint32_t K) {
    constexpr uint32_t BLOCK_M = 128;
    constexpr uint32_t BLOCK_N = 256;
    constexpr uint32_t BLOCK_K = 64;
    constexpr uint32_t NUM_STAGES = 3;
    constexpr uint32_t NUM_TMA_MULTICAST = 2;

    using GemmType = Gemm<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, NUM_TMA_MULTICAST>;

    auto tma_a_desc = GemmType::make_2d_tma_a_desc(A_device, M, K);
    auto tma_b_desc = GemmType::make_2d_tma_b_desc(B_device, K, N);
    auto tma_c_desc = GemmType::make_2d_tma_c_desc(C_device, M, N);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    uint32_t num_sms = deviceProp.multiProcessorCount;

    const uint32_t num_blocks = ceil_div(M, BLOCK_M) * ceil_div(N, BLOCK_N);
    const uint32_t num_waves = ceil_div(num_blocks, num_sms);
    num_sms = std::min(num_sms, ceil_div(num_blocks, num_waves));
    num_sms = ceil_div(num_sms, NUM_TMA_MULTICAST) * NUM_TMA_MULTICAST;

    const uint32_t smem_size = get_smem_size<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES>();

    for (int i = 0; i < warmup_iterations; i++) {
        GemmType::run(C_device, M, N, K, tma_a_desc, tma_b_desc, tma_c_desc, 0, num_sms, smem_size);
    }

    kernel_durations.clear();

    CUpti_SubscriberHandle subscriber;
    uint64_t userdata = 0;
    cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getTimestampCallback, &userdata);
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);

    cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);

    for (int i = 0; i < iterations; i++) {
        flush(to_flush);
        GemmType::run(C_device, M, N, K, tma_a_desc, tma_b_desc, tma_c_desc, 0, num_sms, smem_size);
    }

    cudaDeviceSynchronize();

    cuptiActivityFlushAll(0);

    cuptiUnsubscribe(subscriber);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);

    double time_ns = 0.0;
    if (!kernel_durations.empty()) {
        time_ns = std::accumulate(kernel_durations.begin(), kernel_durations.end(), 0ull) / kernel_durations.size();
    }

    double flops = 2.0 * double(M) * double(N) * double(K);
    double tflops = flops / time_ns * 1e-3;

    std::cout << "Impl Average time (ns): " << time_ns << std::endl;
    std::cout << "Impl Performance: " << tflops << " TFLOPS" << std::endl;
}

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    __nv_bfloat16 *A_device, *B_device, *C_device, *C_cublas_device;

    size_t A_bytes = M * K * sizeof(__nv_bfloat16);
    size_t B_bytes = N * K * sizeof(__nv_bfloat16);
    size_t C_bytes = M * N * sizeof(__nv_bfloat16);

    cudaMalloc(&A_device, A_bytes);
    cudaMalloc(&B_device, B_bytes);
    cudaMalloc(&C_device, C_bytes);
    cudaMalloc(&C_cublas_device, C_bytes);

    cudaMalloc(&to_flush, l2_size);

    std::vector<__nv_bfloat16> A_host(M * K);
    std::vector<__nv_bfloat16> B_host(N * K);

    fill_randn(A_host);
    fill_randn(B_host);

    cudaMemcpy(A_device, A_host.data(), A_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host.data(), B_bytes, cudaMemcpyHostToDevice);

    impl(A_device, B_device, C_device, M, N, K);

    cublas(A_device, B_device, C_cublas_device, M, N, K);

    std::vector<__nv_bfloat16> C_host(M * N);
    std::vector<__nv_bfloat16> C_cublas_host(M * N);
    cudaMemcpy(C_host.data(), C_device, C_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_cublas_host.data(), C_cublas_device, C_bytes, cudaMemcpyDeviceToHost);

    bool is_correct = std::ranges::equal(C_host, C_cublas_host);
    std::cout << (is_correct ? "Correct!" : "Incorrect!") << std::endl;

    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
    cudaFree(C_cublas_device);

    cudaFree(to_flush);

    return 0;
}
