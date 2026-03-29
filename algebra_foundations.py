"""
Unified algebraic foundations for quantization rotations.

Decomposes and re-derives the connections between:
    - Cayley-Dickson algebras (R -> C -> H -> O -> S -> ...)
    - Clifford algebras Cl(p,q) (geometric product, rotors)
    - Exceptional Lie algebras (G2, F4, E6, E7, E8)
    - Albert exceptional Jordan algebra (27D, octonionic Hermitian 3x3)
    - Lattice quantizers (E8, D8, Barnes-Wall hierarchy)
    - Scalar quantizers (Lloyd-Max, K-Means)

The key unification: all these structures are manifestations of the
SAME algebraic hierarchy, and their connections yield insights that
no individual approach provides alone.

=== The Fundamental Chain ===

    R -> C -> H -> O -> S  (Cayley-Dickson tower: doubling)
    |    |    |    |
    Cl(0,0) Cl(0,1) Cl(0,2) Cl(0,3)  (Clifford: geometric product)
    |    |    |    |
    Z    Z^2  D4   E8  (Lattice: sphere packing)
    |    |    |    |
    1b   2b   4b   8b  (Quantizer: bits per block)

At each dimension, the CD algebra, Clifford algebra, and lattice quantizer
are DIFFERENT VIEWS of the same structure:
    - CD provides the ROTATION (left-multiplication by unit element)
    - Clifford provides the SANDWICH PRODUCT (grade-preserving rotation)
    - The lattice provides the CODEBOOK (optimal quantization dictionary)

The exceptional algebras (G2, F4, E6, E7, E8) appear as the SYMMETRY GROUPS
of these structures:
    - G2 = Aut(O): automorphisms of octonions -> rotation symmetries at dim=8
    - F4 = Aut(J3(O)): Albert algebra automorphisms -> 27D Jordan symmetries
    - E8 = largest exceptional: the lattice itself IS the root system

=== Quantization Connection ===

For neural network quantization, these translate to:

1. ROTATION STEP: CD or Clifford structured rotation
   - CD left-multiply: O(d) params, O(d log d) cost, isometric for dim<=8
   - Clifford sandwich: O(d) params, always norm-preserving, 3D blocks
   - Haar/WHT: special cases (WHT = CD at dim=2 recursion)

2. CODEBOOK STEP: Lattice or learned quantizer
   - E8 lattice: optimal 8D sphere packing, G(E8)=0.07168
   - Z8 prefix-cut: CD tower filtration hierarchy
   - K-Means: data-driven, calibration-free on N(0,1)
   - Lloyd-Max: 1D optimal for scalar quantization

3. CORRECTION STEP: QJL sign sketching
   - Johnson-Lindenstrauss projection: random Gaussian
   - Sign packing: bit-packed for 8x memory reduction
   - Unbiased inner product estimation

=== Novel Insights from Unification ===

1. The WHT butterfly IS the CD doubling at dim=2 (proven in rotations.py)
2. The E8 lattice roots ARE the octonion unit elements (both have 240 vectors)
3. The Barnes-Wall hierarchy BW_2k matches the CD tower: BW_2=Z^2, BW_4=D4, BW_8=E8
4. The Cariow factorization exploits the SAME Hadamard structure as WHT rotation
5. The CD fidelity metric (associator norm) measures what Clifford sandwich preserves
6. Zero-divisors (dim>=16) mark where CD rotation loses isometry but Clifford doesn't
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor


# ===================================================================
# The Normed Division Algebras (Hurwitz theorem: only R, C, H, O)
# ===================================================================

@dataclass
class NormedDivisionAlgebra:
    """Properties of the four normed division algebras."""
    name: str
    dim: int
    commutative: bool
    associative: bool
    alternative: bool
    composition: bool  # ||ab|| = ||a||*||b||
    n_zero_divisors: int  # 0 for dim <= 8
    automorphism_group: str
    lattice: str  # optimal lattice at this dimension
    lattice_nsm: float  # normalized second moment G


DIVISION_ALGEBRAS = [
    NormedDivisionAlgebra("Reals", 1, True, True, True, True, 0, "Z/2Z", "Z", 1/12),
    NormedDivisionAlgebra("Complex", 2, True, True, True, True, 0, "Z/2Z", "Z^2 (hexagonal)", 0.08018),
    NormedDivisionAlgebra("Quaternions", 4, False, True, True, True, 0, "SO(3)", "D4", 0.07655),
    NormedDivisionAlgebra("Octonions", 8, False, False, True, True, 0, "G2", "E8", 0.07168),
]


# ===================================================================
# The CD Tower Beyond Composition Algebras
# ===================================================================

@dataclass
class CDTowerLevel:
    """Properties of each level in the Cayley-Dickson tower."""
    name: str
    dim: int
    associative: bool
    alternative: bool
    composition: bool
    power_associative: bool
    n_zero_divisor_pairs: int  # approximate
    cariow_mults: Optional[int]  # Cariow reduced multiplication count
    standard_mults: int  # standard n^2


CD_TOWER = [
    CDTowerLevel("Reals", 1, True, True, True, True, 0, None, 1),
    CDTowerLevel("Complex", 2, True, True, True, True, 0, None, 4),
    CDTowerLevel("Quaternions", 4, True, True, True, True, 0, 8, 16),
    CDTowerLevel("Octonions", 8, False, True, True, True, 0, 30, 64),
    CDTowerLevel("Sedenions", 16, False, False, False, True, 84, 84, 256),
    CDTowerLevel("Pathions", 32, False, False, False, True, None, 498, 1024),
    CDTowerLevel("Chingons", 64, False, False, False, True, None, 1494, 4096),
    CDTowerLevel("Routons", 128, False, False, False, True, None, None, 16384),
]


# ===================================================================
# Exceptional Lie Algebra Dimensions and Connections
# ===================================================================

@dataclass
class ExceptionalAlgebra:
    """An exceptional Lie algebra and its connection to quantization."""
    name: str
    rank: int
    dimension: int  # dimension of the Lie algebra
    root_count: int  # number of roots in the root system
    connection_to_cd: str
    connection_to_quantization: str


EXCEPTIONAL_ALGEBRAS = [
    ExceptionalAlgebra(
        "G2", 2, 14, 12,
        "Automorphism group of octonions: Aut(O) = G2",
        "Preserves the 7D imaginary octonion subspace. CD octonion "
        "rotations are elements of G2 x SO(1) (rotation + scalar).",
    ),
    ExceptionalAlgebra(
        "F4", 4, 52, 48,
        "Automorphism group of the Albert algebra J3(O): Aut(J3(O)) = F4",
        "The Albert algebra is 27D (3x3 octonionic Hermitian matrices). "
        "F4 symmetries could inform 27D or 54D block rotations.",
    ),
    ExceptionalAlgebra(
        "E6", 6, 78, 72,
        "Structure group of the Albert algebra's determinant form",
        "E6 lattice exists in 6D. Not directly used in current quantizers "
        "but the 72-root system could serve as a 6D codebook.",
    ),
    ExceptionalAlgebra(
        "E7", 7, 133, 126,
        "Related to the Freudenthal magic square at level C x O",
        "E7 lattice in 7D (126 roots). The Barnes-Wall lattice BW_7 "
        "is related. Could provide 7D codebook.",
    ),
    ExceptionalAlgebra(
        "E8", 8, 248, 240,
        "The E8 lattice IS the root system. E8 = densest 8D packing "
        "(Viazovska 2016). Direct connection: 240 E8 roots provide the "
        "optimal 8D quantization codebook AND rotation elements.",
        "E8 lattice quantizer: G(E8)=0.07168 (14% MSE improvement over "
        "scalar). Used by QuIP# for weight codebooks. We use E8 roots "
        "as rotation elements (E8BlockRotation).",
    ),
]


# ===================================================================
# The Barnes-Wall Lattice Hierarchy (matches CD tower)
# ===================================================================

def barnes_wall_connection() -> Dict[int, Dict]:
    """
    The Barnes-Wall lattice hierarchy BW_{2^k} matches the CD tower:

        BW_1  = Z     (integers)          -> Reals
        BW_2  = Z^2   (hexagonal)         -> Complex numbers
        BW_4  = D4    (checkerboard)       -> Quaternions
        BW_8  = E8    (exceptional)        -> Octonions
        BW_16 = Lambda_16 (Barnes-Wall 16) -> Sedenions

    At each level, the lattice is constructed via "Construction A" from
    the corresponding Reed-Muller code, and the CD doubling provides
    the algebraic structure for the doubling step.

    This is the MERA-CD duality: the lattice hierarchy coarsens
    (BW_128 -> BW_64 -> ... -> BW_4) while the CD tower refines
    (R^4 -> R^8 -> ... -> R^128).
    """
    return {
        1: {"lattice": "Z", "algebra": "R", "nsm": 1/12, "roots": 2},
        2: {"lattice": "A2 (hexagonal)", "algebra": "C", "nsm": 0.08018, "roots": 6},
        4: {"lattice": "D4", "algebra": "H", "nsm": 0.07655, "roots": 24},
        8: {"lattice": "E8", "algebra": "O", "nsm": 0.07168, "roots": 240},
        16: {"lattice": "Lambda_16 (BW)", "algebra": "S", "nsm": 0.06833, "roots": 4320},
    }


# ===================================================================
# Unified Rotation-Codebook-Correction Framework
# ===================================================================

@dataclass
class QuantizationMethod:
    """A complete quantization method specified by its algebraic components."""
    rotation_type: str     # "haar", "wht", "cd4"..."cd128", "e8block", "clifford"
    codebook_type: str     # "lloyd_max", "e8", "z8_*", "kmeans", "group_affine"
    correction_type: str   # "none", "qjl", "qjl_packed"
    preprocess_type: str   # "none", "nsn"
    bits: int
    # Algebraic properties
    rotation_algebra: str  # which algebra provides the rotation
    codebook_algebra: str  # which algebra/lattice provides the codebook
    params_per_block: int  # storage for rotation parameters
    mults_per_block: int   # multiplications for rotation application
    isometric: bool        # does the rotation preserve norms?


def enumerate_methods(d: int = 128) -> List[QuantizationMethod]:
    """Enumerate all valid quantization methods at dimension d."""
    methods = []

    # Scalar methods (any d)
    for rot in ["haar", "wht"]:
        methods.append(QuantizationMethod(
            rot, "lloyd_max", "qjl_packed", "nsn", 3,
            "trivial" if rot == "haar" else "CD tower (dim=2 recursion)",
            "Gaussian 1D",
            d*d if rot == "haar" else 2*d,
            d*d if rot == "haar" else d * int(math.log2(d)),
            True,
        ))

    # CD block methods
    for block_dim in [4, 8, 16]:
        if d % block_dim == 0:
            n_blocks = d // block_dim
            algebra = {4: "H (quaternions)", 8: "O (octonions)", 16: "S (sedenions)"}[block_dim]
            methods.append(QuantizationMethod(
                f"cd{block_dim}", "lloyd_max", "qjl_packed", "nsn", 3,
                algebra, "Gaussian 1D",
                n_blocks * block_dim,
                n_blocks * (CD_TOWER[int(math.log2(block_dim))].cariow_mults or block_dim**2),
                block_dim <= 8,
            ))

    # E8Block rotation + E8 codebook (full algebraic harmony)
    if d % 16 == 0:
        n_blocks = d // 16
        methods.append(QuantizationMethod(
            "e8block", "e8", "qjl_packed", "nsn", 3,
            "E8 roots embedded in S (sedenions)",
            "E8 lattice (Viazovska-optimal 8D packing)",
            n_blocks * 8,
            n_blocks * 30,  # Cariow-30 for octonion sub-blocks
            True,
        ))

    # Clifford rotor
    n_blocks_cl = d // 3
    methods.append(QuantizationMethod(
        "clifford", "lloyd_max", "qjl_packed", "nsn", 3,
        "Cl(3,0) (Euclidean 3D Clifford)",
        "Gaussian 1D",
        n_blocks_cl * 4,
        n_blocks_cl * 16,  # geometric product cost
        True,  # sandwich product always preserves norms
    ))

    # K-Means VQ
    methods.append(QuantizationMethod(
        "wht", "kmeans", "qjl_packed", "nsn", 3,
        "CD tower (dim=2 recursion)",
        "K-Means on N(0,1) (data-driven 8D codebook)",
        2 * d,
        d * int(math.log2(d)),
        True,
    ))

    return methods


def print_method_comparison(d: int = 128):
    """Print a formatted comparison of all quantization methods."""
    methods = enumerate_methods(d)
    print(f"\n=== Quantization Methods at d={d} ===\n")
    print(f"{'Rotation':<12} {'Codebook':<12} {'Algebra':<30} "
          f"{'Params':>8} {'Mults':>8} {'Iso':>4}")
    print("-" * 84)
    for m in methods:
        print(f"{m.rotation_type:<12} {m.codebook_type:<12} "
              f"{m.rotation_algebra[:30]:<30} "
              f"{m.params_per_block:>8} {m.mults_per_block:>8} "
              f"{'yes' if m.isometric else 'no':>4}")


# ===================================================================
# The Cariow-Hadamard Connection
# ===================================================================

def cariow_hadamard_duality():
    """
    The Cariow factorization and the WHT rotation use the SAME structure.

    WHT butterfly: at each level, (a,b) -> (a+b, a-b)
    Cariow factorization: at each level, decompose B = B_toeplitz + 2*M_sparse,
        then factor B_toeplitz via H_2 kron I_k into sum/difference blocks.

    The Hadamard matrix H_2 = [[1,1],[1,-1]] appears in BOTH:
        - WHT uses H_2 to decorrelate coordinates (rotation)
        - Cariow uses H_2 to factorize the bilinear multiplication matrix (algebra)

    This means: the WHT rotation and the Cariow-optimized CD multiply are
    DUAL operations -- one decorrelates data, the other factorizes algebra.
    Both exploit the same recursive halving structure of the CD tower.

    Novel insight: combining WHT rotation WITH Cariow-factorized CD block
    rotation should be optimally efficient because both share the same
    butterfly infrastructure. A fused implementation could reuse the
    butterfly hardware/instructions for both rotation and algebraic multiplication.
    """
    return {
        "wht_butterfly": "Decorrelates coordinates: O(d log d)",
        "cariow_butterfly": "Factorizes bilinear map: reduces mults by 50-67%",
        "shared_structure": "H_2 = [[1,1],[1,-1]] at every level",
        "fusion_opportunity": "Combined WHT + CD multiply shares butterfly HW",
        "novel_insight": "WHT and Cariow are DUAL operations in the CD tower",
    }
