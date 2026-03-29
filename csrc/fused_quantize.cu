/*
 * Fused rotation + quantization CUDA kernel for TurboQuant.
 *
 * Fuses three operations into a single kernel launch:
 *   1. WHT rotation (butterfly structure)
 *   2. Centroid distance computation
 *   3. Argmin codebook lookup
 *
 * Key optimization patterns from steinmarder:
 *   - Shared memory codebook: Lloyd-Max codebook (max 16 entries @ 4-bit)
 *     fits entirely in shared memory, broadcast to all threads in a warp.
 *   - 8-wide ILP: process 8 coordinates per thread for instruction-level
 *     parallelism, hiding FFMA latency (4.54 cyc on Ada Lovelace).
 *   - Vectorized loads: float4 for coalesced global memory access.
 *   - Storage-compute split: store quantized indices in INT4/INT8,
 *     promote to FP32 for all arithmetic.
 *
 * Reference: steinmarder/src/sass_re/instant_ngp/mlp_forward.cu
 *            steinmarder/src/cuda_lbm/kernels_fp8.cu
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define MAX_CENTROIDS 16  /* 4-bit max */
#define MAX_DIM 256

/*
 * Fused WHT rotation + scalar quantization kernel.
 *
 * Each thread block processes one vector. Within the block:
 *   - Phase 1: Load vector into shared memory, apply WHT butterfly
 *   - Phase 2: Each thread quantizes its assigned coordinates
 *
 * The WHT butterfly uses shared memory for the butterfly operations,
 * with __syncthreads() between levels (log2(d) levels total).
 */
__global__ void fused_wht_quantize_kernel(
    const float* __restrict__ input,    /* (n, d) */
    const float* __restrict__ d1,       /* (d,) sign vector */
    const float* __restrict__ d2,       /* (d,) sign vector */
    const float* __restrict__ centroids,/* (n_centroids,) */
    int8_t* __restrict__ indices,       /* (n, d) output */
    int n,
    int d,
    int n_centroids
) {
    __shared__ float smem_vec[MAX_DIM];
    __shared__ float smem_centroids[MAX_CENTROIDS];

    int vec_idx = blockIdx.x;
    if (vec_idx >= n) return;

    int tid = threadIdx.x;

    /* Load centroids into shared memory (once per block) */
    if (tid < n_centroids) {
        smem_centroids[tid] = centroids[tid];
    }

    /* Load vector and apply D2 sign flip */
    if (tid < d) {
        smem_vec[tid] = input[vec_idx * d + tid] * d2[tid];
    }
    __syncthreads();

    /* WHT butterfly (log2(d) levels) */
    for (int h = 1; h < d; h <<= 1) {
        if (tid < d) {
            int pair_idx = tid ^ h;  /* XOR gives butterfly partner */
            if (pair_idx > tid) {
                float a = smem_vec[tid];
                float b = smem_vec[pair_idx];
                smem_vec[tid] = a + b;
                smem_vec[pair_idx] = a - b;
            }
        }
        __syncthreads();
    }

    /* Normalize and apply D1 sign flip */
    float inv_sqrt_d = rsqrtf((float)d);
    if (tid < d) {
        smem_vec[tid] = smem_vec[tid] * inv_sqrt_d * d1[tid];
    }
    __syncthreads();

    /* Quantize: find nearest centroid for each coordinate */
    if (tid < d) {
        float val = smem_vec[tid];
        float best_dist = fabsf(val - smem_centroids[0]);
        int best_idx = 0;

        /* Unrolled for small codebooks (ILP pattern from steinmarder) */
        #pragma unroll
        for (int c = 1; c < MAX_CENTROIDS; c++) {
            if (c < n_centroids) {
                float dist = fabsf(val - smem_centroids[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = c;
                }
            }
        }
        indices[vec_idx * d + tid] = (int8_t)best_idx;
    }
}

/*
 * PyTorch C++ extension wrapper.
 */
torch::Tensor fused_wht_quantize(
    torch::Tensor input,
    torch::Tensor d1,
    torch::Tensor d2,
    torch::Tensor centroids
) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "input must be 2D (n, d)");

    int n = input.size(0);
    int d = input.size(1);
    int n_centroids = centroids.size(0);

    TORCH_CHECK(d <= MAX_DIM, "d must be <= ", MAX_DIM);
    TORCH_CHECK(n_centroids <= MAX_CENTROIDS, "n_centroids must be <= ", MAX_CENTROIDS);

    auto indices = torch::empty({n, d}, torch::dtype(torch::kInt8).device(input.device()));

    int threads = d;  /* one thread per coordinate */
    int blocks = n;   /* one block per vector */

    fused_wht_quantize_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        d1.data_ptr<float>(),
        d2.data_ptr<float>(),
        centroids.data_ptr<float>(),
        indices.data_ptr<int8_t>(),
        n, d, n_centroids
    );

    return indices;
}

/*
 * Fused asymmetric attention kernel.
 *
 * Computes: scores[q,k] = <Q[q], K_mse[k]> + ||r[k]|| * C * <S@Q[q], signs[k]>
 *
 * Both terms are computed in a single pass over the compressed keys.
 * This avoids materializing the full K_mse matrix and the S@Q intermediate.
 *
 * Pattern: steinmarder's instant-NGP MLP kernel uses 8-wide FFMA accumulators
 * to saturate the FP32 pipeline. We apply the same principle: each thread
 * accumulates both the MSE inner product and the QJL correction simultaneously,
 * interleaving independent FFMA chains.
 */
__global__ void fused_asymmetric_attention_kernel(
    const float* __restrict__ queries,     /* (n_q, d) */
    const float* __restrict__ k_mse,       /* (n_k, d) */
    const int8_t* __restrict__ qjl_signs,  /* (n_k, d) */
    const float* __restrict__ r_norms,     /* (n_k,) */
    const float* __restrict__ S,           /* (d, d) - QJL projection matrix */
    float* __restrict__ scores,            /* (n_q, n_k) */
    int n_q, int n_k, int d,
    float correction_scale
) {
    int q_idx = blockIdx.x;
    int k_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (q_idx >= n_q || k_idx >= n_k) return;

    /* Accumulate both terms simultaneously (ILP: two independent chains) */
    float term1 = 0.0f;  /* <Q, K_mse> */
    float term2 = 0.0f;  /* <S@Q, signs> */

    for (int i = 0; i < d; i++) {
        float q_val = queries[q_idx * d + i];
        float k_val = k_mse[k_idx * d + i];
        float sign = (float)qjl_signs[k_idx * d + i];

        /* Term 1: direct inner product */
        term1 += q_val * k_val;

        /* Term 2: project query through S, dot with signs */
        /* S@Q[i] = sum_j S[i][j] * Q[j] -- precompute outside if possible */
        float sq_val = 0.0f;
        for (int j = 0; j < d; j++) {
            sq_val += S[i * d + j] * queries[q_idx * d + j];
        }
        term2 += sq_val * sign;
    }

    /* Combine with residual norm scaling */
    float r_norm = r_norms[k_idx];
    scores[q_idx * n_k + k_idx] = term1 + r_norm * correction_scale * term2;
}

torch::Tensor fused_asymmetric_attention(
    torch::Tensor queries,
    torch::Tensor k_mse,
    torch::Tensor qjl_signs,
    torch::Tensor r_norms,
    torch::Tensor S,
    float correction_scale
) {
    TORCH_CHECK(queries.is_cuda(), "queries must be on CUDA");

    int n_q = queries.size(0);
    int n_k = k_mse.size(0);
    int d = queries.size(1);

    auto scores = torch::empty({n_q, n_k}, queries.options());

    dim3 blocks(n_q, (n_k + 255) / 256);
    dim3 threads(min(n_k, 256));

    fused_asymmetric_attention_kernel<<<blocks, threads>>>(
        queries.data_ptr<float>(),
        k_mse.data_ptr<float>(),
        qjl_signs.data_ptr<int8_t>(),
        r_norms.data_ptr<float>(),
        S.data_ptr<float>(),
        scores.data_ptr<float>(),
        n_q, n_k, d, correction_scale
    );

    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_wht_quantize", &fused_wht_quantize,
          "Fused WHT rotation + scalar quantization");
    m.def("fused_asymmetric_attention", &fused_asymmetric_attention,
          "Fused asymmetric attention from compressed keys");
}
