#pragma once

#include <cassert>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

template <typename T>
__device__ __host__ __forceinline__ constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ void prefetch_tma_descriptor(void const *desc_ptr) {
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    asm volatile("prefetch.tensormap [%0];" : : "l"(gmem_int_desc) : "memory");
}

union GmmaDescriptor {
    __host__ __device__ constexpr GmmaDescriptor() noexcept : desc_(0) {
    }

    __host__ __device__ constexpr GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {
    }

    __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor const &t) noexcept : desc_(t.desc_) {
    }

    __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor &&t) noexcept : desc_(t.desc_) {
    }

    __host__ __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor const &t) noexcept {
        desc_ = t.desc_;
        return *this;
    }

    __host__ __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor &&t) noexcept {
        desc_ = t.desc_;
        return *this;
    }

    uint64_t desc_;
    uint32_t reg32_[2];
    uint16_t reg16_[4];

    struct {
        uint16_t start_address_ : 14, : 2;
        uint16_t leading_byte_offset_ : 14, : 2;
        uint16_t stride_byte_offset_ : 14, : 2;
        uint8_t : 1, base_offset_ : 3, : 4;
        uint8_t : 6, layout_type_ : 2;
    } bitfield;

    // Decay to an `uint64_t`
    __host__ __device__ constexpr operator uint64_t() const noexcept {
        return desc_;
    }
};

template <class PointerType>
__device__ GmmaDescriptor make_k_major_smem_desc(
    PointerType smem_ptr, int layout_type, int leading_byte_offset = 0, int stride_byte_offset = 1024) {
    GmmaDescriptor desc;
    auto uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = layout_type;
    desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
    desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.bitfield.base_offset_ = 0;
    return desc;
}

__device__ __forceinline__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_fence_operand(float &reg) {
    asm volatile("" : "+f"(reg)::"memory");
}

template <int N>
__device__ __forceinline__ void warpgroup_wait() {
    static_assert(N >= 0 and N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" : : "n"(N) : "memory");
}

template <int Count>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(Count) : "memory");
}

__device__ __forceinline__ void tma_store_fence() {
    asm volatile("fence.proxy.async.shared::cta;");
}

__device__ __forceinline__ void tma_store_arrive() {
    asm volatile("cp.async.bulk.commit_group;");
}

struct WGMMA {
    __device__ static void wgmma(
        uint64_t const &desc_a, uint64_t const &desc_b, float &d000, float &d001, float &d002, float &d003, float &d004,
        float &d005, float &d006, float &d007, float &d008, float &d009, float &d010, float &d011, float &d012,
        float &d013, float &d014, float &d015, float &d016, float &d017, float &d018, float &d019, float &d020,
        float &d021, float &d022, float &d023, float &d024, float &d025, float &d026, float &d027, float &d028,
        float &d029, float &d030, float &d031, float &d032, float &d033, float &d034, float &d035, float &d036,
        float &d037, float &d038, float &d039, float &d040, float &d041, float &d042, float &d043, float &d044,
        float &d045, float &d046, float &d047, float &d048, float &d049, float &d050, float &d051, float &d052,
        float &d053, float &d054, float &d055, float &d056, float &d057, float &d058, float &d059, float &d060,
        float &d061, float &d062, float &d063, float &d064, float &d065, float &d066, float &d067, float &d068,
        float &d069, float &d070, float &d071, float &d072, float &d073, float &d074, float &d075, float &d076,
        float &d077, float &d078, float &d079, float &d080, float &d081, float &d082, float &d083, float &d084,
        float &d085, float &d086, float &d087, float &d088, float &d089, float &d090, float &d091, float &d092,
        float &d093, float &d094, float &d095, float &d096, float &d097, float &d098, float &d099, float &d100,
        float &d101, float &d102, float &d103, float &d104, float &d105, float &d106, float &d107, float &d108,
        float &d109, float &d110, float &d111, float &d112, float &d113, float &d114, float &d115, float &d116,
        float &d117, float &d118, float &d119, float &d120, float &d121, float &d122, float &d123, float &d124,
        float &d125, float &d126, float &d127, bool scale_d) {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %130, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
            "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
            " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
            " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
            " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
            " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
            " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
            " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
            " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
            " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
            " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
            " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
            " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
            " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
            " %104, %105, %106, %107, %108, %109, %110, %111, "
            " %112, %113, %114, %115, %116, %117, %118, %119, "
            " %120, %121, %122, %123, %124, %125, %126, %127},"
            " %128,"
            " %129,"
            " p,   1,   1,   0,   0;\n"
            "}\n"
            : "+f"(d000), "+f"(d001), "+f"(d002), "+f"(d003), "+f"(d004), "+f"(d005), "+f"(d006), "+f"(d007),
              "+f"(d008), "+f"(d009), "+f"(d010), "+f"(d011), "+f"(d012), "+f"(d013), "+f"(d014), "+f"(d015),
              "+f"(d016), "+f"(d017), "+f"(d018), "+f"(d019), "+f"(d020), "+f"(d021), "+f"(d022), "+f"(d023),
              "+f"(d024), "+f"(d025), "+f"(d026), "+f"(d027), "+f"(d028), "+f"(d029), "+f"(d030), "+f"(d031),
              "+f"(d032), "+f"(d033), "+f"(d034), "+f"(d035), "+f"(d036), "+f"(d037), "+f"(d038), "+f"(d039),
              "+f"(d040), "+f"(d041), "+f"(d042), "+f"(d043), "+f"(d044), "+f"(d045), "+f"(d046), "+f"(d047),
              "+f"(d048), "+f"(d049), "+f"(d050), "+f"(d051), "+f"(d052), "+f"(d053), "+f"(d054), "+f"(d055),
              "+f"(d056), "+f"(d057), "+f"(d058), "+f"(d059), "+f"(d060), "+f"(d061), "+f"(d062), "+f"(d063),
              "+f"(d064), "+f"(d065), "+f"(d066), "+f"(d067), "+f"(d068), "+f"(d069), "+f"(d070), "+f"(d071),
              "+f"(d072), "+f"(d073), "+f"(d074), "+f"(d075), "+f"(d076), "+f"(d077), "+f"(d078), "+f"(d079),
              "+f"(d080), "+f"(d081), "+f"(d082), "+f"(d083), "+f"(d084), "+f"(d085), "+f"(d086), "+f"(d087),
              "+f"(d088), "+f"(d089), "+f"(d090), "+f"(d091), "+f"(d092), "+f"(d093), "+f"(d094), "+f"(d095),
              "+f"(d096), "+f"(d097), "+f"(d098), "+f"(d099), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103),
              "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111),
              "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119),
              "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123), "+f"(d124), "+f"(d125), "+f"(d126), "+f"(d127)
            : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const &desc_a, uint64_t const &desc_b, float *d, bool scale_d) {
        wgmma(
            desc_a, desc_b, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13],
            d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28],
            d[29], d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39], d[40], d[41], d[42], d[43],
            d[44], d[45], d[46], d[47], d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55], d[56], d[57], d[58],
            d[59], d[60], d[61], d[62], d[63], d[64], d[65], d[66], d[67], d[68], d[69], d[70], d[71], d[72], d[73],
            d[74], d[75], d[76], d[77], d[78], d[79], d[80], d[81], d[82], d[83], d[84], d[85], d[86], d[87], d[88],
            d[89], d[90], d[91], d[92], d[93], d[94], d[95], d[96], d[97], d[98], d[99], d[100], d[101], d[102], d[103],
            d[104], d[105], d[106], d[107], d[108], d[109], d[110], d[111], d[112], d[113], d[114], d[115], d[116],
            d[117], d[118], d[119], d[120], d[121], d[122], d[123], d[124], d[125], d[126], d[127], scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 256;
    static constexpr int K = 16;
    static constexpr int NUM_ACCUMS = M * N / 128;
};

template <typename T>
struct SM90_U32x4_STSM_N {
    __device__ __forceinline__ static void copy(T src0, T src1, T src_2, T src_3, void *smem_dst) {
        const uint32_t src[4] = {
            *reinterpret_cast<uint32_t *>(&src0), *reinterpret_cast<uint32_t *>(&src1),
            *reinterpret_cast<uint32_t *>(&src_2), *reinterpret_cast<uint32_t *>(&src_3)};
        asm volatile(
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"l"(smem_dst), "r"(src[0]),
            "r"(src[1]), "r"(src[2]), "r"(src[3]));
    }
};

__device__ __forceinline__ uint32_t get_lane_id() {
    uint32_t lane_id;
    asm("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

__device__ __forceinline__ uint32_t block_rank_in_cluster() {
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank) :);
    return rank;
}

template <uint32_t RegCount>
__device__ __forceinline__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ __forceinline__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

__device__ __forceinline__ void init_barrier(uint64_t *barrier, int arrive_count) {
    uint32_t barrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile(
        "{\n\t"
        "mbarrier.init.shared::cta.b64 [%1], %0; \n"
        "}"
        :
        : "r"(arrive_count), "r"(barrier_ptr));
}

__device__ static __forceinline__ void wait(uint64_t *barrier, int phase) {
    uint32_t barrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    constexpr uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
        :
        : "r"(barrier_ptr), "r"(phase), "r"(ticks));
}

__device__ __forceinline__ void arrive_and_expect_tx(uint64_t *barrier, uint32_t transaction_bytes) {
    uint32_t barrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(barrier_ptr));
}

__device__ __forceinline__ void arrive(uint64_t const *barrier) {
    uint32_t barrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"
        "}"
        :
        : "r"(barrier_ptr));
}

__device__ __forceinline__ void arrive_cluster(uint64_t *barrier, uint32_t cta_id) {
    uint32_t barrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));

    asm volatile(
        "{\n"
        ".reg .b32 remAddr32;\n"
        "mapa.shared::cluster.u32  remAddr32, %0, %1;\n"
        "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n"
        "}"
        :
        : "r"(barrier_ptr), "r"(cta_id));
}

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t NUM_TMA_MULTICAST, uint32_t NUM_BLOCKS_PER_GROUP = 16>
struct Scheduler {
    int current_iter = -1;
    uint32_t num_aligned_m_blocks;
    uint32_t num_aligned_n_blocks;
    uint32_t num_blocks;

    __device__ explicit Scheduler(uint32_t shape_m, uint32_t shape_n) {
        num_aligned_m_blocks = ceil_div(shape_m, BLOCK_M);
        num_aligned_n_blocks = ceil_div(shape_n, BLOCK_N);
        num_blocks = num_aligned_m_blocks * num_aligned_n_blocks;
    }

    __device__ void get_swizzled_block_idx(int block_idx, uint32_t &m_block_idx, uint32_t &n_block_idx) {
        static_assert(NUM_BLOCKS_PER_GROUP % NUM_TMA_MULTICAST == 0, "Invalid group size");

        const auto num_blocks_per_group = num_aligned_n_blocks * NUM_BLOCKS_PER_GROUP;
        const auto group_idx = block_idx / num_blocks_per_group;
        const auto in_group_idx = block_idx % num_blocks_per_group;

        const auto first_m_block_idx = group_idx * NUM_BLOCKS_PER_GROUP;
        const auto num_m_blocks_in_group = min(NUM_BLOCKS_PER_GROUP, num_aligned_m_blocks - first_m_block_idx);
        m_block_idx = first_m_block_idx + in_group_idx % num_m_blocks_in_group;
        n_block_idx = in_group_idx / num_m_blocks_in_group;
    }

    __device__ bool get_next_block(uint32_t &m_block_idx, uint32_t &n_block_idx) {
        const auto next_block_idx = (++current_iter) * gridDim.x + blockIdx.x;
        if (next_block_idx >= num_blocks)
            return false;
        get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
        return true;
    }
};

template <uint32_t NUM_TMA_MULTICAST>
__device__ __forceinline__ void load_async_multicast(
    __nv_bfloat16 *smem, void const *src_tma_map, uint64_t *barrier, int crd0, int crd1) {
    constexpr uint64_t CACHE_HINT = 0x1000000000000000ull;
    constexpr uint16_t MASK = (1 << NUM_TMA_MULTICAST) - 1;

    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t barrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
        " [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :
        : "r"(smem_int_ptr), "l"(tma_ptr), "r"(barrier_ptr), "h"(MASK), "r"(crd0), "r"(crd1), "l"(CACHE_HINT)
        : "memory");
}

__device__ __forceinline__ void load_async(
    __nv_bfloat16 *smem, void const *src_tma_map, uint64_t *barrier, int crd0, int crd1) {
    constexpr uint64_t CACHE_HINT = 0x1000000000000000ull;

    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t barrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;"
        :
        : "r"(smem_int_ptr), "l"(tma_ptr), "r"(barrier_ptr), "r"(crd0), "r"(crd1), "l"(CACHE_HINT)
        : "memory");
}

__device__ __forceinline__ void store_async(void const *dst_tma_map, __nv_bfloat16 *smem, int crd0, int crd1) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst_tma_map);
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
                 :
                 : "l"(tma_ptr), "r"(smem_int_ptr), "r"(crd0), "r"(crd1)
                 : "memory");
}

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t NUM_STAGES, uint32_t NUM_TMA_MULTICAST>
__global__ void __launch_bounds__(384, 1) bf16_gemm_kernel_nt(
    __nv_bfloat16 *gmem_c, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
    const __grid_constant__ CUtensorMap tensor_map_a, const __grid_constant__ CUtensorMap tensor_map_b,
    const __grid_constant__ CUtensorMap tensor_map_c) {
    constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);

    constexpr uint32_t MATH_NUM_THREADS = 256;

    constexpr uint32_t FULL_K_ALL_STAGES = NUM_STAGES * BLOCK_K;
    const uint32_t num_k_full_iterations = ceil_div(shape_k, FULL_K_ALL_STAGES);

    const uint32_t warp_group_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const uint32_t in_group_idx = threadIdx.x % 128;
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_id();

    if (threadIdx.x == MATH_NUM_THREADS) {
        prefetch_tma_descriptor(&tensor_map_a);
        prefetch_tma_descriptor(&tensor_map_b);
        prefetch_tma_descriptor(&tensor_map_c);
    }
    __syncwarp();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    static_assert(SMEM_D_SIZE % 1024 == 0, "Shared memory of D must be aligned to 1024 bytes");
    static_assert(SMEM_A_SIZE_PER_STAGE % 1024 == 0, "Shared memory of A must be aligned to 1024 bytes");
    static_assert(SMEM_B_SIZE_PER_STAGE % 1024 == 0, "Shared memory of B must be aligned to 1024 bytes");

    auto smem_d = reinterpret_cast<__nv_bfloat16 *>(smem_buffer);
    __nv_bfloat16 *smem_a[NUM_STAGES];
    __nv_bfloat16 *smem_b[NUM_STAGES];

    uint64_t *full_barriers[NUM_STAGES];
    uint64_t *empty_barriers[NUM_STAGES];

#pragma unroll
    for (uint32_t i = 0; i < NUM_STAGES; ++i) {
        smem_a[i] = reinterpret_cast<__nv_bfloat16 *>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<__nv_bfloat16 *>(
            smem_buffer + SMEM_D_SIZE + NUM_STAGES * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    }

    auto barrier_start_ptr = reinterpret_cast<uint64_t *>(
        smem_buffer + SMEM_D_SIZE + NUM_STAGES * SMEM_A_SIZE_PER_STAGE + NUM_STAGES * SMEM_B_SIZE_PER_STAGE);
#pragma unroll
    for (uint32_t i = 0; i < NUM_STAGES; ++i) {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + NUM_STAGES + i;
    }

    if (threadIdx.x == MATH_NUM_THREADS) {
#pragma unroll
        for (uint32_t i = 0; i < NUM_STAGES; ++i) {
            init_barrier(full_barriers[i], 1);
            init_barrier(empty_barriers[i], NUM_TMA_MULTICAST * MATH_NUM_THREADS / 128);
        }

        asm volatile("fence.proxy.async.shared::cta;\n");
        if constexpr (NUM_TMA_MULTICAST > 1) {
            asm volatile("fence.mbarrier_init.release.cluster;\n");
        }
    }

    if constexpr (NUM_TMA_MULTICAST > 1) {
        asm volatile("barrier.cluster.arrive.aligned;\n");
        asm volatile("barrier.cluster.wait.aligned;\n");
    } else {
        __syncthreads();
    }

    const uint32_t block_rank = block_rank_in_cluster();

    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<BLOCK_M, BLOCK_N, NUM_TMA_MULTICAST>(shape_m, shape_n);

    if (threadIdx.x >= MATH_NUM_THREADS) {
        warpgroup_reg_dealloc<40>();

        if (threadIdx.x == MATH_NUM_THREADS) {
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                for (uint32_t k_iter = 0; k_iter < num_k_full_iterations; ++k_iter) {
#pragma unroll
                    for (uint32_t s = 0; s < NUM_STAGES; ++s) {
                        wait(empty_barriers[s], (scheduler.current_iter * num_k_full_iterations + k_iter + 1) & 1);

                        const uint32_t k_idx = k_iter * FULL_K_ALL_STAGES + s * BLOCK_K;
                        if (k_idx >= shape_k) {
                            arrive(full_barriers[s]);
                            continue;
                        }

                        load_async(smem_a[s], &tensor_map_a, full_barriers[s], k_idx, m_block_idx * BLOCK_M);

                        if constexpr (NUM_TMA_MULTICAST > 1) {
                            if (block_rank == 0) {
                                load_async_multicast<NUM_TMA_MULTICAST>(
                                    smem_b[s], &tensor_map_b, full_barriers[s], k_idx, n_block_idx * BLOCK_N);
                            }
                        } else {
                            load_async(smem_b[s], &tensor_map_b, full_barriers[s], k_idx, n_block_idx * BLOCK_N);
                        }

                        arrive_and_expect_tx(full_barriers[s], SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
                    }
                }
            }

            if constexpr (NUM_TMA_MULTICAST > 1) {
#pragma unroll
                for (uint32_t s = 0; s < NUM_STAGES; ++s)
                    wait(empty_barriers[s], (scheduler.current_iter * num_k_full_iterations + 1) & 1);
            }
        }
    } else {
        warpgroup_reg_alloc<232>();

        auto empty_barrier_arrive = [&](uint32_t s) {
            if constexpr (NUM_TMA_MULTICAST == 1) {
                in_group_idx == 0 ? arrive(empty_barriers[s]) : void();
            } else {
                in_group_idx < NUM_TMA_MULTICAST ? arrive_cluster(empty_barriers[s], in_group_idx) : void();
            }
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            float accum[WGMMA::NUM_ACCUMS];

            auto compute_wgmma_stage = [&](uint32_t s, bool scale_d = true) {
                const auto smem_a_warp_group_offset = warp_group_idx * WGMMA::M * BLOCK_K;

#pragma unroll
                for (int i = 0; i < WGMMA::NUM_ACCUMS; ++i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_arrive();

                auto desc_a = make_k_major_smem_desc(smem_a[s] + smem_a_warp_group_offset, 1);
                auto desc_b = make_k_major_smem_desc(smem_b[s], 1);
                WGMMA::wgmma(desc_a, desc_b, accum, scale_d);
#pragma unroll
                for (int k = 1; k < BLOCK_K / WGMMA::K; ++k) {
                    auto desc_a = make_k_major_smem_desc(smem_a[s] + k * WGMMA::K + smem_a_warp_group_offset, 1);
                    auto desc_b = make_k_major_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                    WGMMA::wgmma(desc_a, desc_b, accum, true);
                }

                warpgroup_commit_batch();
#pragma unroll
                for (int i = 0; i < WGMMA::NUM_ACCUMS; ++i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_wait<0>();
            };

            wait(full_barriers[0], (scheduler.current_iter * num_k_full_iterations) & 1);
            compute_wgmma_stage(0, false);
            empty_barrier_arrive(0);

#pragma unroll
            for (uint32_t s = 1; s < NUM_STAGES; ++s) {
                wait(full_barriers[s], (scheduler.current_iter * num_k_full_iterations) & 1);

                const uint32_t k_idx = s * BLOCK_K;
                if (k_idx >= shape_k) {
                    empty_barrier_arrive(s);
                    continue;
                }

                compute_wgmma_stage(s);

                empty_barrier_arrive(s);
            }
            for (uint32_t k_iter = 1; k_iter < num_k_full_iterations; ++k_iter) {
#pragma unroll
                for (uint32_t s = 0; s < NUM_STAGES; ++s) {
                    wait(full_barriers[s], (scheduler.current_iter * num_k_full_iterations + k_iter) & 1);

                    const uint32_t k_idx = k_iter * FULL_K_ALL_STAGES + s * BLOCK_K;
                    if (k_idx >= shape_k) {
                        empty_barrier_arrive(s);
                        continue;
                    }

                    compute_wgmma_stage(s);

                    empty_barrier_arrive(s);
                }
            }

            tma_store_wait<0>();
            asm volatile("bar.sync %0, 128;\n" ::"r"(warp_group_idx + 8) : "memory");

            uint32_t smem_store_offset = (warp_idx * 16 + lane_idx % 16) * 32 + 8 * (lane_idx / 16);

            uint32_t tma_store_smem_offset = warp_group_idx * WGMMA::M * 32;
            uint32_t tma_store_gmem_n = n_block_idx * BLOCK_N;
            uint32_t tma_store_gmem_m = m_block_idx * BLOCK_M + warp_group_idx * WGMMA::M;

#pragma unroll
            for (auto j = 0; j < WGMMA::NUM_ACCUMS / 16; ++j) {
                const auto i0 = j * 2 + 0;
                SM90_U32x4_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({accum[i0 * 8 + 0], accum[i0 * 8 + 1]}),
                    __float22bfloat162_rn({accum[i0 * 8 + 2], accum[i0 * 8 + 3]}),
                    __float22bfloat162_rn({accum[i0 * 8 + 4], accum[i0 * 8 + 5]}),
                    __float22bfloat162_rn({accum[i0 * 8 + 6], accum[i0 * 8 + 7]}), smem_d + smem_store_offset);

                const auto i1 = j * 2 + 1;
                SM90_U32x4_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({accum[i1 * 8 + 0], accum[i1 * 8 + 1]}),
                    __float22bfloat162_rn({accum[i1 * 8 + 2], accum[i1 * 8 + 3]}),
                    __float22bfloat162_rn({accum[i1 * 8 + 4], accum[i1 * 8 + 5]}),
                    __float22bfloat162_rn({accum[i1 * 8 + 6], accum[i1 * 8 + 7]}), smem_d + smem_store_offset + 16);

                smem_store_offset += BLOCK_M * 32;

                tma_store_fence();
                asm volatile("bar.sync %0, 128;\n" ::"r"(warp_group_idx + 8) : "memory");

                if (in_group_idx == 0) {
                    store_async(&tensor_map_c, smem_d + tma_store_smem_offset, tma_store_gmem_n, tma_store_gmem_m);

                    tma_store_arrive();
                }
                __syncwarp();

                tma_store_smem_offset += BLOCK_M * 32;
                tma_store_gmem_n += 32;
            }
        }
    }
}

enum class Layout { RowMajor, ColMajor };

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t NUM_STAGES, uint32_t NUM_TMA_MULTICAST>
class Gemm {
   public:
    Gemm() = default;

    static void run(
        __nv_bfloat16 *gmem_c, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, const CUtensorMap &tma_a_desc,
        const CUtensorMap &tma_b_desc, const CUtensorMap &tma_c_desc, cudaStream_t stream, int num_sms,
        uint32_t smem_size) {
        auto kernel = bf16_gemm_kernel_nt<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, NUM_TMA_MULTICAST>;
        assert(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        cudaLaunchConfig_t config;
        config.gridDim = num_sms;
        config.blockDim = 384;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {NUM_TMA_MULTICAST, 1, 1};
        config.attrs = &attr;
        config.numAttrs = 1;

        auto status =
            cudaLaunchKernelEx(&config, kernel, gmem_c, shape_m, shape_n, shape_k, tma_a_desc, tma_b_desc, tma_c_desc);
        assert(status == cudaSuccess);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_a_desc(T *global_address, uint32_t shape_m, uint32_t shape_k) {
        return make_2d_tma_desc(global_address, Layout::RowMajor, shape_m, shape_k, BLOCK_M, BLOCK_K);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_b_desc(T *global_address, uint32_t shape_k, uint32_t shape_n) {
        return make_2d_tma_desc(global_address, Layout::ColMajor, shape_k, shape_n, BLOCK_K, BLOCK_N);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_c_desc(T *global_address, uint32_t shape_m, uint32_t shape_n) {
        return make_2d_tma_desc(
            global_address, Layout::RowMajor, shape_m, shape_n, BLOCK_M / 2, 32,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE, false);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_desc(
        T *global_address, Layout layout, uint32_t gmem_rows, uint32_t gmem_cols, uint32_t smem_rows,
        uint32_t smem_cols, CUtensorMapSwizzle swizzle_type = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        bool enable_l2_promotion = true) {
        if (layout == Layout::RowMajor) {
            uint64_t gmem_dim[2] = {gmem_cols, gmem_rows};
            uint32_t smem_dim[2] = {smem_cols, smem_rows};
            return make_2d_tma_copy_desc(
                global_address, gmem_dim, gmem_cols * sizeof(T), smem_dim, swizzle_type, enable_l2_promotion);
        } else {
            uint64_t gmem_dim[2] = {gmem_rows, gmem_cols};
            uint32_t smem_dim[2] = {smem_rows, smem_cols};
            return make_2d_tma_copy_desc(
                global_address, gmem_dim, gmem_rows * sizeof(T), smem_dim, swizzle_type, enable_l2_promotion);
        }
    }

    static PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
        cudaDriverEntryPointQueryResult driver_status;
        void *cuTensorMapEncodeTiled_ptr = nullptr;

#if CUDA_VERSION >= 12050
        cudaGetDriverEntryPointByVersion(
            "cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status);
#else
        cudaGetDriverEntryPoint(
            "cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, cudaEnableDefault, &driver_status);
#endif

        if (driver_status != cudaDriverEntryPointSuccess)
            throw std::runtime_error("driver_status != cudaDriverEntryPointSuccess");
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(cuTensorMapEncodeTiled_ptr);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_copy_desc(
        T *global_address, uint64_t gmem_dim[2], uint64_t stride_in_bytes, uint32_t smem_dim[2],
        CUtensorMapSwizzle swizzle_type, bool enable_l2_promotion = true) {
        CUtensorMap tensor_map{};
        constexpr uint32_t rank = 2;
        uint64_t global_stride[rank - 1] = {stride_in_bytes};
        uint32_t elem_strides[rank] = {1, 1};

        PFN_cuTensorMapEncodeTiled encode_func = get_cuTensorMapEncodeTiled();

        auto result = encode_func(
            &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, global_address, gmem_dim, global_stride, smem_dim,
            elem_strides, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_type,
            enable_l2_promotion ? CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B :
                                  CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        assert(result == CUDA_SUCCESS);
        return tensor_map;
    }
};
