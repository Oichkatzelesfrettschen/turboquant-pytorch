# Full Conversation Audit: Origin to Current State

## Atomic Decomposition of Claims, Gaps, and Next Steps

### PHASE 1: Claims Made (Falsifiable Theses)

| ID | Claim | Evidence | Falsifiable Test | Status |
|----|-------|----------|------------------|--------|
| T1 | CD-structured rotation for quantization is novel (March 2026) | Literature search: no prior work found | Search for "Cayley-Dickson quantization rotation" post-March 2026 | VALIDATED (web search + 95 PDFs) |
| T2 | WHT butterfly IS the CD doubling at dim=2 | Mathematical identity: (a,b)->(a+b,a-b) | Z3 Proof 22: materialized=butterfly | Z3-PROVEN |
| T3 | E8 lattice gives 14% MSE improvement over scalar | G(E8)=0.07168 vs G(Z)=0.08333 | Measure MSE ratio on random Gaussian data | MEASURED: E8 with grid_search MSE=0.000326 vs LM-3b MSE=0.000357 = 8.7% (not 14% -- the 14% is theoretical at same rate) |
| T4 | NSN is the dominant contributor (+136bp) | Ablation on Qwen2.5-3B | Remove NSN, measure cosine drop | ABLATION-VALIDATED: L0=0.9849, L2=0.9985 |
| T5 | Sign packing gives 8x memory reduction | 128 signs * 1 byte = 128B vs 2 * int64 = 16B | Measure actual memory | MEASURED: 9.0 MB -> 2.8 MB = 3.2x (not 8x for full pipeline; 8x for signs alone) |
| T6 | Post-VQ scaling hurts (-2bp) | RCA: increases residual norm 5.3% | Measure with/without on real model | RCA-VALIDATED |
| T7 | WHT only +0.6bp despite 30% synthetic gain | RCA: compressor normalization removes magnitude structure | Measure cross-correlation post-normalization | RCA-VALIDATED: xcorr 0.207 (Haar) vs 0.397 (WHT) |
| T8 | Quaternion norm multiplicativity | ||ab||^2 = ||a||^2 * ||b||^2 | Z3 Proof 7 | Z3-PROVEN |
| T9 | Octonion alternativity | [a,a,b] = 0 for all a,b | Z3 Proofs 9, 19 | Z3-PROVEN |
| T10 | Sedenion non-alternativity | exists a,b with [a,a,b] != 0 | Z3 Proof 26: witness (e3+e10, e6) | Z3-PROVEN |
| T11 | Cariow-30 for regular octonions | N(N-1)/2+2 = 30 at N=8 | Verify against paywalled paper | LITERATURE-VALIDATED (arXiv variants confirm structure) |
| T12 | CD fidelity ratio 0.9998 on real model | Phase-geometry preserved despite 3.1% cosine distortion | Measure on Qwen2.5-3B | MEASURED: 1.3814 (higher than expected -- needs investigation) |
| T13 | TF32 gives 3.02x GPU speedup | cuBLAS math mode measurement | Benchmark Haar FP32 vs TF32 | MEASURED: 186M/s -> 561M/s = 3.02x |
| T14 | RotorQuant uses Cl(3,0) not CD | Pope March 2026, scrya.com | Read the actual code/paper | LITERATURE-VALIDATED |
| T15 | No Rust FWHT crate exists | open_gororoba wht_crate_scope.rs | Search crates.io | UNVERIFIED (need to check crates.io) |

### PHASE 2: Gaps Identified

| ID | Gap | Severity | Resolution |
|----|-----|----------|------------|
| G1 | T3 discrepancy: 14% theoretical vs 8.7% measured | MEDIUM | The 14% is at same bit rate; our E8 operates at ~1 bit/dim vs LM at 3 bits/dim -- different rates. Clarify in paper. |
| G2 | T5 discrepancy: 8x claimed vs 3.2x measured | MEDIUM | 8x is for QJL signs alone (128B -> 16B); 3.2x is full pipeline including non-sign data. Clarify in paper. |
| G3 | T12 fidelity ratio 1.38 != 0.9998 | HIGH | The 0.9998 came from open_gororoba on plasma data, not our ablation. Our measurement on Qwen2.5-3B gave 1.3814. These are different data/dimensions. Reconcile. |
| G4 | True perplexity never measured | HIGH | Only attention cosine similarity measured. Need monkey-patched attention for real PPL. |
| G5 | Cariow-30 never actually implemented | MEDIUM | Analysis only. The actual 30-mult algorithm needs paywalled paper coefficients. |
| G6 | MICRO 2026 format unknown | CRITICAL | Need to verify page limits, format, deadlines. |
| G7 | Paper is 6 pages but MICRO typically requires 11 | CRITICAL | Need to expand significantly. |
| G8 | Current paper has no double-blind anonymization | HIGH | Author name visible. MICRO is double-blind. |
| G9 | Clifford rotor 3D blocks shown to be worse than WHT | LOW | Documented honestly. Not a gap, just a negative result. |
| G10 | Hierarchical tower worse than uniform on Gaussian data | LOW | Documented. Only helps with structured outliers that NSN already handles. |

### PHASE 3: Debt Inventory

| Type | Count | Items |
|------|-------|-------|
| **Tech debt** | 3 | E8Block needs CUDA cd_multiply; Triton kernel slower than PyTorch; KMeans training slow |
| **Test debt** | 9 | 12/27 files still have zero dedicated tests (hierarchical, sign_pack, e8_rotation, etc.) |
| **Documentation debt** | 2 | README.md not updated; CHANGELOG.md missing |
| **Build debt** | 1 | pytest can't discover tests due to __init__.py in same directory |
| **Verification debt** | 6 | 6/36 properties need Lean/Coq/SymPy (not Z3-feasible) |
| **Paper debt** | 5 | Need 11 pages (have 6), double-blind, MICRO format, expanded experiments, related work |
| **Reproducibility debt** | 3 | No CI, no Docker, no pinned deps |

### PHASE 4: Novel Connections Identified But Not Exploited

1. **Cariow-Hadamard duality**: WHT rotation and Cariow multiplication share H_2 butterfly. A fused hardware kernel could do both in one pass. NOT BUILT.

2. **MERA-CD duality**: The CD tower doubling (4D->128D) is the algebraic dual of MERA coarse-graining (128D->4D). NOT EXPLOITED beyond documentation.

3. **Barnes-Wall lattice = CD tower at each level**: BW_8 = E8 = Octonion root system. The quantization codebook IS the algebra's root system. NOT EXPLOITED for adaptive codebook selection.

4. **G2 = Aut(O)**: The automorphism group of octonions constrains which rotations preserve the algebraic structure. Could inform rotation learning. NOT EXPLOITED.

5. **F4 = Aut(J3(O))**: The Albert exceptional Jordan algebra is 27D. Could inform 27D block rotations for models with head_dim near 27. SPECULATIVE.

6. **Zero-divisor geometry for adaptive allocation**: ZD-affinity scoring identifies structurally vulnerable directions. Could drive per-coordinate bit allocation instead of per-head. NOT EXPLOITED beyond diagnostics.
