"""
Tensor decomposition + quantization for KV cache compression.

Combines low-rank approximation with quantization for joint compression.
If the KV cache has low-rank structure, we can decompose it first
(reducing dimensionality) and then quantize the factors.

For a key matrix K of shape (seq_len, d):
    K ~ U_r @ (S_r @ V_r^T)
where r << min(seq_len, d).

Quantizing the factors costs:
    (seq_len * r + r * d) * bits instead of seq_len * d * bits

This module also provides joint rank-bitwidth optimization: given a fixed
memory budget, find the (rank, bits) pair that minimizes inner product RMSE.

Inspired by open_gororoba's tensor-train cross approximation, adapted for
the simpler case of matrix (2D) decomposition in the KV cache setting.
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple


def svd_compress(
    K: Tensor,
    rank: int,
) -> Tuple[Tensor, Tensor]:
    """
    Truncated SVD decomposition: K ~ U_r @ S_V_r.

    Args:
        K: key/value matrix, shape (seq_len, d)
        rank: truncation rank

    Returns:
        (U_r, S_V_r) where U_r: (seq_len, rank), S_V_r: (rank, d).
        Reconstruct as U_r @ S_V_r.
    """
    U, S, Vh = torch.linalg.svd(K, full_matrices=False)
    U_r = U[:, :rank]           # (seq_len, rank)
    S_r = S[:rank]              # (rank,)
    Vh_r = Vh[:rank, :]         # (rank, d)
    S_V_r = S_r.unsqueeze(-1) * Vh_r  # (rank, d)
    return U_r, S_V_r


def svd_reconstruct(U_r: Tensor, S_V_r: Tensor) -> Tensor:
    """
    Reconstruct from SVD factors.

    Args:
        U_r: shape (seq_len, rank)
        S_V_r: shape (rank, d)

    Returns:
        Reconstructed matrix, shape (seq_len, d).
    """
    return U_r @ S_V_r


def svd_inner_product(
    query: Tensor,
    U_r: Tensor,
    S_V_r: Tensor,
) -> Tensor:
    """
    Compute inner products <query, K_i> using SVD factored form.

    <q, K_i> = <q, U_r[i] @ S_V_r> = U_r[i] @ (S_V_r @ q)

    This is O(rank * d + seq_len * rank) instead of O(seq_len * d).

    Args:
        query: shape (d,) or (batch, d)
        U_r: shape (seq_len, rank)
        S_V_r: shape (rank, d)

    Returns:
        Inner products, shape (seq_len,) or (batch, seq_len).
    """
    # S_V_r @ q: (rank, d) @ (d,) -> (rank,) or (rank, d) @ (batch, d).T -> (rank, batch)
    if query.dim() == 1:
        projected_q = S_V_r @ query  # (rank,)
    else:
        projected_q = S_V_r @ query.mT  # (rank, batch)
    if projected_q.dim() == 1:
        return U_r @ projected_q  # (seq_len,)
    else:
        return (U_r @ projected_q).T  # (batch, seq_len)


def compression_ratio(
    seq_len: int,
    d: int,
    rank: int,
    bits: int,
    original_bits: int = 16,
) -> float:
    """
    Compute compression ratio for SVD + quantization.

    Args:
        seq_len: sequence length
        d: head dimension
        rank: SVD truncation rank
        bits: quantization bits for the factors
        original_bits: original precision (default fp16 = 16)

    Returns:
        Compression ratio (higher = more compression).
    """
    original = seq_len * d * original_bits
    compressed = (seq_len * rank + rank * d) * bits
    return original / compressed if compressed > 0 else float("inf")


def joint_rank_bitwidth(
    K: Tensor,
    memory_budget_bits: int,
    rank_range: Optional[List[int]] = None,
    bits_range: Optional[List[int]] = None,
    metric: str = "mse",
) -> Dict:
    """
    Grid search over (rank, bits) pairs to minimize reconstruction error
    within a memory budget.

    Args:
        K: key matrix, shape (seq_len, d)
        memory_budget_bits: maximum total bits for compressed representation
        rank_range: list of ranks to try (default: [4, 8, 16, 32, 64])
        bits_range: list of bit-widths to try (default: [1, 2, 3, 4])
        metric: "mse" or "cosine" for evaluation

    Returns:
        Dict with:
            best_rank: optimal rank
            best_bits: optimal bits
            best_error: error at optimal point
            compression_ratio: achieved compression ratio
            all_results: list of (rank, bits, error, budget_ok) tuples
    """
    seq_len, d = K.shape
    if rank_range is None:
        rank_range = [r for r in [4, 8, 16, 32, 64] if r < min(seq_len, d)]
    if bits_range is None:
        bits_range = [1, 2, 3, 4]

    all_results = []
    best_error = float("inf")
    best_rank = rank_range[0]
    best_bits = bits_range[0]

    for rank in rank_range:
        U_r, S_V_r = svd_compress(K, rank)

        for bits in bits_range:
            # Check budget
            total_bits = (seq_len * rank + rank * d) * bits
            budget_ok = total_bits <= memory_budget_bits

            # Measure reconstruction error (using SVD only, before quantization)
            K_hat = svd_reconstruct(U_r, S_V_r)

            if metric == "mse":
                error = ((K - K_hat) ** 2).mean().item()
            elif metric == "cosine":
                cos_sim = torch.nn.functional.cosine_similarity(
                    K.reshape(1, -1), K_hat.reshape(1, -1)
                )
                error = 1.0 - cos_sim.item()
            else:
                raise ValueError(f"Unknown metric: {metric}")

            all_results.append((rank, bits, error, budget_ok))

            if budget_ok and error < best_error:
                best_error = error
                best_rank = rank
                best_bits = bits

    return {
        "best_rank": best_rank,
        "best_bits": best_bits,
        "best_error": best_error,
        "compression_ratio": compression_ratio(seq_len, d, best_rank, best_bits),
        "all_results": all_results,
    }


def explained_variance_ratio(K: Tensor, max_rank: Optional[int] = None) -> Tensor:
    """
    Compute the fraction of variance explained by each singular component.

    Useful for choosing the SVD rank: if the first r components explain
    >99% of variance, rank=r is a good choice.

    Args:
        K: key matrix, shape (seq_len, d)
        max_rank: max number of components (default: min(seq_len, d))

    Returns:
        Cumulative explained variance ratio, shape (max_rank,).
    """
    _, S, _ = torch.linalg.svd(K, full_matrices=False)
    if max_rank is not None:
        S = S[:max_rank]
    total_var = (S ** 2).sum()
    cumulative = (S ** 2).cumsum(dim=0) / total_var
    return cumulative
