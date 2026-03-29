/*
 * TurboQuant CUDA kernels: complete kernel set for KV cache quantization.
 *
 * Four kernels ported from open_gororoba + steinmarder patterns:
 *   1. turboquant_quantize_boundary: boundary-search scalar quantization
 *   2. turboquant_dequant_dot: fused dequant + dot product (attention hot path)
 *   3. turboquant_sign_dot: QJL sign-sketch inner product with bit-packed signs
 *   4. turboquant_fast_jl_rotate: in-place Walsh-Hadamard Transform
 *
 * Design rules (from steinmarder SASS measurements on Ada sm_89):
 *   - Maximize FFMA (4.54 cyc, 44.6 ops/clk/SM) -- the workhorse
 *   - AVOID MUFU.RCP (41.53 cyc) and MUFU.EX2 (17.55 cyc) -- too slow
 *   - Minimize LDG L2 miss (92-123 cyc), target L1 residency (128KB Ada)
 *   - POPC = 7-8 cyc, fast enough for inline sign operations
 *   - FTZ = zero overhead on Ada
 *
 * Layout: SoA (Structure-of-Arrays)
 *   indices[coord_idx * n_vectors + vec_idx] -- coalesced per-coordinate access
 *   signs packed as u32 words (32 signs per word)
 *
 * Storage-compute split (from steinmarder INT8 SoA LBM, 5956 MLUPS):
 *   Store as u8 indices, promote to f32 centroid values at point of use.
 *   Codebook fits in L1 cache (max 16 entries @ 4-bit = 64 bytes).
 *
 * References:
 *   steinmarder/src/cuda_lbm/kernels_int8_soa_lloydmax.cu
 *   open_gororoba/crates/cd_kernel/src/turboquant/cuda/kernels/turboquant.cu
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* ====================================================================
 * Kernel 1: Boundary-search scalar quantization
 *
 * Quantize f32 values using sorted Lloyd-Max boundaries.
 * Index = count(boundaries[j] < value). For 3-bit (7 boundaries),
 * this is 7 comparisons -- the compiler unrolls for small counts.
 *
 * Pattern: steinmarder lm_encode() with boundary count instead of
 * linear mapping. More accurate for non-uniform Lloyd-Max codebooks.
 * ==================================================================== */

__global__ void turboquant_quantize_boundary_kernel(
    const float* __restrict__ boundaries,
    const float* __restrict__ values,
    unsigned char* __restrict__ indices,
    int n_boundaries,
    int n_values
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_values) return;

    float v = values[idx];
    unsigned char count = 0;

    /* Boundary count: unrolled by compiler for small n_boundaries */
    for (int b = 0; b < n_boundaries; b++) {
        if (v > boundaries[b]) count++;
    }
    indices[idx] = count;
}

/* ====================================================================
 * Kernel 2: Fused dequant + dot product (attention hot path)
 *
 * For each (query, key) pair, compute the attention score directly
 * from compressed key indices without full decompression.
 *
 * SoA layout: key_indices[coord * n_keys + key_idx] for coalesced reads.
 * Storage-compute split: read u8 index, look up f32 centroid, FFMA.
 *
 * Pattern: steinmarder storage-compute split from kernels_int8_soa.cu
 * ==================================================================== */

__global__ void turboquant_dequant_dot_kernel(
    const float* __restrict__ queries,       /* (n_queries, d) row-major */
    const unsigned char* __restrict__ key_indices, /* (d, n_keys) SoA */
    const float* __restrict__ centroids,     /* (n_levels,) codebook */
    const float* __restrict__ key_norms,     /* (n_keys,) per-key scale */
    float* __restrict__ scores,              /* (n_queries, n_keys) output */
    int d,
    int n_queries,
    int n_keys,
    int n_levels
) {
    int qi = blockIdx.y;
    int ki = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= n_queries || ki >= n_keys) return;

    const float* q = queries + qi * d;
    float dot = 0.0f;

    /* Storage-compute split: u8 -> f32 codebook lookup, then FFMA */
    for (int c = 0; c < d; c++) {
        unsigned char idx = key_indices[(long long)c * n_keys + ki]; /* SoA coalesced */
        float k_val = centroids[idx];                                /* L1 cached */
        dot = fmaf(q[c], k_val, dot);                               /* FFMA */
    }

    scores[qi * n_keys + ki] = dot * key_norms[ki];
}

/* ====================================================================
 * Kernel 3: QJL sign-sketch inner product with bit-packed signs
 *
 * Compute the QJL correction: ||r|| * sqrt(pi/2)/m * <S@q, signs>
 * Signs are packed 32 per u32 word; uses POPC (7-8 cyc on Ada).
 *
 * Pattern: open_gororoba sign_pack.rs popcount inner product
 * ==================================================================== */

__global__ void turboquant_sign_dot_kernel(
    const float* __restrict__ s_matrix,        /* (m, d) projection */
    const float* __restrict__ query,           /* (d,) single query */
    const unsigned int* __restrict__ packed_signs, /* (n_keys, n_words) */
    const float* __restrict__ residual_norms,  /* (n_keys,) */
    float* __restrict__ correction,            /* (n_keys,) output */
    int d,
    int m,
    int n_keys,
    int n_words,
    float scale /* sqrt(pi/2) / m */
) {
    int ki = blockIdx.x * blockDim.x + threadIdx.x;
    if (ki >= n_keys) return;

    const unsigned int* signs = packed_signs + ki * n_words;
    float result = 0.0f;

    for (int j = 0; j < m; j++) {
        /* Project query through S for dimension j */
        float proj = 0.0f;
        const float* s_row = s_matrix + j * d;
        for (int i = 0; i < d; i++) {
            proj = fmaf(s_row[i], query[i], proj);
        }

        /* Extract sign bit: +1 if set, -1 if clear */
        int word_idx = j >> 5;   /* j / 32 */
        int bit_idx = j & 31;    /* j % 32 */
        float sign = (signs[word_idx] >> bit_idx) & 1 ? 1.0f : -1.0f;
        result = fmaf(proj, sign, result);
    }

    correction[ki] = residual_norms[ki] * scale * result;
}

/* ====================================================================
 * Kernel 4: Fast Walsh-Hadamard Transform (in-place)
 *
 * Each thread transforms one d-dimensional vector.
 * For d=128: 7 butterfly levels, 896 FFMA per vector ~ 100 cycles.
 *
 * Pattern: open_gororoba turboquant_fast_jl_rotate
 * ==================================================================== */

__global__ void turboquant_fast_jl_rotate_kernel(
    float* __restrict__ data,    /* (n_vectors, d) in-place */
    const float* __restrict__ d1, /* (d,) Rademacher diagonal 1 */
    const float* __restrict__ d2, /* (d,) Rademacher diagonal 2 */
    int d,
    int n_vectors,
    float inv_sqrt_d
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= n_vectors) return;

    float* vec = data + vid * d;

    /* Step 1: multiply by D2 */
    for (int i = 0; i < d; i++) {
        vec[i] *= d2[i];
    }

    /* Step 2: in-place WHT butterfly */
    for (int h = 1; h < d; h *= 2) {
        for (int i = 0; i < d; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = vec[j];
                float b = vec[j + h];
                vec[j] = a + b;
                vec[j + h] = a - b;
            }
        }
    }

    /* Step 3: normalize and multiply by D1 */
    for (int i = 0; i < d; i++) {
        vec[i] *= inv_sqrt_d * d1[i];
    }
}

/* ====================================================================
 * PyTorch C++ extension wrappers
 * ==================================================================== */

torch::Tensor turboquant_quantize_boundary(
    torch::Tensor boundaries,
    torch::Tensor values
) {
    TORCH_CHECK(values.is_cuda(), "values must be on CUDA");
    int n_boundaries = boundaries.size(0);
    int n_values = values.numel();

    auto indices = torch::empty({n_values}, torch::dtype(torch::kUInt8).device(values.device()));

    int threads = 256;
    int blocks = (n_values + threads - 1) / threads;

    turboquant_quantize_boundary_kernel<<<blocks, threads>>>(
        boundaries.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<unsigned char>(),
        n_boundaries, n_values
    );
    return indices.reshape_as(values);
}

torch::Tensor turboquant_dequant_dot(
    torch::Tensor queries,
    torch::Tensor key_indices,
    torch::Tensor centroids,
    torch::Tensor key_norms
) {
    TORCH_CHECK(queries.is_cuda(), "queries must be on CUDA");
    int n_queries = queries.size(0);
    int d = queries.size(1);
    int n_keys = key_indices.size(1);
    int n_levels = centroids.size(0);

    auto scores = torch::empty({n_queries, n_keys}, queries.options());

    int threads = 256;
    dim3 blocks_dim((n_keys + threads - 1) / threads, n_queries);

    turboquant_dequant_dot_kernel<<<blocks_dim, threads>>>(
        queries.data_ptr<float>(),
        key_indices.data_ptr<unsigned char>(),
        centroids.data_ptr<float>(),
        key_norms.data_ptr<float>(),
        scores.data_ptr<float>(),
        d, n_queries, n_keys, n_levels
    );
    return scores;
}

torch::Tensor turboquant_sign_dot(
    torch::Tensor s_matrix,
    torch::Tensor query,
    torch::Tensor packed_signs,
    torch::Tensor residual_norms,
    float scale
) {
    TORCH_CHECK(query.is_cuda(), "query must be on CUDA");
    int d = s_matrix.size(1);
    int m = s_matrix.size(0);
    int n_keys = packed_signs.size(0);
    int n_words = packed_signs.size(1);

    auto correction = torch::empty({n_keys}, query.options());

    int threads = 256;
    int blocks = (n_keys + threads - 1) / threads;

    turboquant_sign_dot_kernel<<<blocks, threads>>>(
        s_matrix.data_ptr<float>(),
        query.data_ptr<float>(),
        (const unsigned int*)packed_signs.data_ptr<int>(),
        residual_norms.data_ptr<float>(),
        correction.data_ptr<float>(),
        d, m, n_keys, n_words, scale
    );
    return correction;
}

void turboquant_fast_jl_rotate(
    torch::Tensor data,
    torch::Tensor d1,
    torch::Tensor d2
) {
    TORCH_CHECK(data.is_cuda(), "data must be on CUDA");
    int n_vectors = data.size(0);
    int d = data.size(1);
    float inv_sqrt_d = 1.0f / sqrtf((float)d);

    int threads = 128;
    int blocks = (n_vectors + threads - 1) / threads;

    turboquant_fast_jl_rotate_kernel<<<blocks, threads>>>(
        data.data_ptr<float>(),
        d1.data_ptr<float>(),
        d2.data_ptr<float>(),
        d, n_vectors, inv_sqrt_d
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("turboquant_quantize_boundary", &turboquant_quantize_boundary,
          "Boundary-search Lloyd-Max quantization");
    m.def("turboquant_dequant_dot", &turboquant_dequant_dot,
          "Fused dequant + dot product (attention hot path)");
    m.def("turboquant_sign_dot", &turboquant_sign_dot,
          "QJL sign-sketch inner product with bit-packed signs");
    m.def("turboquant_fast_jl_rotate", &turboquant_fast_jl_rotate,
          "In-place Fast Walsh-Hadamard Transform with Rademacher diagonals");
}
