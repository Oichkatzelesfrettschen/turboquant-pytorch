"""
Cayley-Dickson algebra engine for PyTorch.

Implements the recursive doubling construction that builds hypercomplex number
systems: reals (1D) -> complex (2D) -> quaternions (4D) -> octonions (8D) ->
sedenions (16D) -> ... for any power-of-two dimension.

The doubling formula:
    Conjugation: (a, b)* = (a*, -b)
    Multiplication: (a,b)(c,d) = (ac - d*b, da + bc*)

All functions operate on tensors with shape (..., 2^k) where the last dimension
is the algebra dimension. Batch dimensions are fully supported.

Key properties by dimension:
    dim=1:  Reals. Commutative, associative, ordered.
    dim=2:  Complex. Commutative, associative, no order.
    dim=4:  Quaternions. Non-commutative, associative.
    dim=8:  Octonions. Non-commutative, non-associative, alternative.
    dim=16: Sedenions. Non-alternative, has zero divisors.
    dim=32+: Higher CD algebras. Increasingly loose algebraic structure.

Composition algebra property (||ab|| = ||a|| ||b||) holds ONLY for dim <= 8.
At dim >= 16, zero divisors exist: nonzero a,b with ab = 0.

Ported from open_gororoba/crates/cd_kernel/src/cayley_dickson/arith.rs.
"""

import torch
from torch import Tensor


def _check_cd_dim(x: Tensor) -> int:
    """Validate that the last dimension is a power of 2 and return it."""
    d = x.shape[-1]
    if d == 0 or (d & (d - 1)) != 0:
        raise ValueError(f"Last dimension must be a power of 2, got {d}")
    return d


def cd_conjugate(x: Tensor) -> Tensor:
    """
    Cayley-Dickson conjugation: negate all imaginary components.

    x* = (x0, -x1, -x2, ..., -x_{n-1})

    For quaternions (a + bi + cj + dk), this gives (a - bi - cj - dk).
    Conjugation is an involution: (x*)* = x.
    For any CD algebra: x * x* = x* * x = ||x||^2 (real scalar).

    Args:
        x: tensor of shape (..., 2^k)

    Returns:
        Conjugate, same shape as x.
    """
    _check_cd_dim(x)
    result = x.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def cd_multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Cayley-Dickson multiplication via the recursive doubling formula.

    (a,b)(c,d) = (ac - d*b, da + bc*)

    where * denotes conjugation. Recursion bottoms out at dim=1
    (scalar multiplication).

    Specialized fast paths for common dimensions avoid intermediate tensors:
        dim=1: scalar multiply (0 intermediates)
        dim=2: complex multiply (0 intermediates, 2 fused ops)
        dim=4: quaternion Hamilton product (0 intermediates, direct formula)

    For dim >= 8: recursive with the general CD formula.

    Complexity: O(d * log(d)) multiplications -- same asymptotic cost as FFT.

    Args:
        a: tensor of shape (..., 2^k)
        b: tensor of shape (..., 2^k), same shape as a

    Returns:
        Product a*b, same shape as inputs.
    """
    d = _check_cd_dim(a)
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            f"Dimension mismatch: a has {a.shape[-1]}, b has {b.shape[-1]}"
        )

    if d == 1:
        return a * b

    if d == 2:
        return _complex_multiply(a, b)

    if d == 4:
        return _quaternion_multiply(a, b)

    # General recursive case for dim >= 8
    half = d // 2
    a_l, a_r = a[..., :half], a[..., half:]
    c_l, c_r = b[..., :half], b[..., half:]

    conj_c_r = cd_conjugate(c_r)
    conj_c_l = cd_conjugate(c_l)

    term1 = cd_multiply(a_l, c_l)
    term2 = cd_multiply(conj_c_r, a_r)
    term3 = cd_multiply(c_r, a_l)
    term4 = cd_multiply(a_r, conj_c_l)

    left = term1 - term2
    right = term3 + term4

    return torch.cat([left, right], dim=-1)


def _complex_multiply(a: Tensor, b: Tensor) -> Tensor:
    """Complex multiplication: (a0+a1*i)(b0+b1*i) = (a0*b0-a1*b1) + (a0*b1+a1*b0)*i"""
    a0, a1 = a[..., 0], a[..., 1]
    b0, b1 = b[..., 0], b[..., 1]
    return torch.stack([a0 * b0 - a1 * b1, a0 * b1 + a1 * b0], dim=-1)


def _quaternion_multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Quaternion Hamilton product -- direct formula, zero intermediate tensors.

    (a0 + a1*i + a2*j + a3*k)(b0 + b1*i + b2*j + b3*k) =
        (a0*b0 - a1*b1 - a2*b2 - a3*b3) +
        (a0*b1 + a1*b0 + a2*b3 - a3*b2)*i +
        (a0*b2 - a1*b3 + a2*b0 + a3*b1)*j +
        (a0*b3 + a1*b2 - a2*b1 + a3*b0)*k
    """
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0,
    ], dim=-1)


def cd_norm_sq(x: Tensor) -> Tensor:
    """
    Squared Cayley-Dickson norm: ||x||^2 = sum of squares of all components.

    Equivalently, ||x||^2 = Re(x * x*) where Re extracts the real (index 0) part.

    For composition algebras (dim <= 8): ||ab||^2 = ||a||^2 * ||b||^2.
    For dim >= 16 this multiplicativity FAILS due to zero divisors.

    Args:
        x: tensor of shape (..., 2^k)

    Returns:
        Squared norm, shape (...).
    """
    return (x * x).sum(dim=-1)


def cd_norm(x: Tensor) -> Tensor:
    """
    Cayley-Dickson norm: ||x|| = sqrt(sum of squares).

    Args:
        x: tensor of shape (..., 2^k)

    Returns:
        Norm, shape (...).
    """
    return cd_norm_sq(x).sqrt()


def cd_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Normalize to unit CD element: x / ||x||.

    Args:
        x: tensor of shape (..., 2^k)
        eps: small constant to avoid division by zero

    Returns:
        Unit element, same shape as x.
    """
    n = cd_norm(x).unsqueeze(-1)
    return x / (n + eps)


def cd_inverse(x: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Cayley-Dickson inverse: x^{-1} = x* / ||x||^2.

    For composition algebras (dim <= 8), x * x^{-1} = x^{-1} * x = 1.
    For dim >= 16, zero divisors exist where ||x|| can be zero for nonzero x,
    making the inverse undefined. The eps parameter prevents division by zero
    but the result is algebraically meaningless at zero divisors.

    Args:
        x: tensor of shape (..., 2^k)
        eps: regularization for near-zero norms

    Returns:
        Inverse, same shape as x.
    """
    conj = cd_conjugate(x)
    nsq = cd_norm_sq(x).unsqueeze(-1)
    return conj / (nsq + eps)


def cd_real_part(x: Tensor) -> Tensor:
    """
    Extract the real (scalar) component.

    Args:
        x: tensor of shape (..., 2^k)

    Returns:
        Real part, shape (...).
    """
    return x[..., 0]


def cd_imag_part(x: Tensor) -> Tensor:
    """
    Extract the imaginary components.

    Args:
        x: tensor of shape (..., 2^k) where k >= 1

    Returns:
        Imaginary part, shape (..., 2^k - 1).
    """
    return x[..., 1:]


def cd_from_real(r: Tensor, dim: int) -> Tensor:
    """
    Embed a real scalar into a CD algebra of given dimension.

    Args:
        r: tensor of shape (...)
        dim: target algebra dimension (power of 2)

    Returns:
        CD element with r as real part and zeros elsewhere, shape (..., dim).
    """
    zeros = torch.zeros(*r.shape, dim - 1, device=r.device, dtype=r.dtype)
    return torch.cat([r.unsqueeze(-1), zeros], dim=-1)


def cd_commutator(a: Tensor, b: Tensor) -> Tensor:
    """
    Commutator: [a,b] = ab - ba.

    Zero for reals and complex (commutative algebras).
    Nonzero for quaternions and above.

    Args:
        a, b: tensors of shape (..., 2^k)

    Returns:
        Commutator, same shape as inputs.
    """
    return cd_multiply(a, b) - cd_multiply(b, a)


def cd_associator(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """
    Associator: [a,b,c] = (ab)c - a(bc).

    Zero for reals, complex, and quaternions (associative algebras).
    Nonzero for octonions (but alternative: [a,a,b] = [a,b,b] = 0).
    Fully nonzero for sedenions and above.

    The associator measures how far the algebra is from being associative.

    Args:
        a, b, c: tensors of shape (..., 2^k)

    Returns:
        Associator, same shape as inputs.
    """
    ab = cd_multiply(a, b)
    bc = cd_multiply(b, c)
    return cd_multiply(ab, c) - cd_multiply(a, bc)


def cd_associator_norm(a: Tensor, b: Tensor, c: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Normalized associator norm: ||[a,b,c]|| / (||a|| * ||b|| * ||c||).

    This is a scale-invariant measure of non-associativity. Values:
        dim <= 4 (quaternions): exactly 0 (associative)
        dim = 8 (octonions): nonzero but bounded (alternative)
        dim >= 16 (sedenions+): can be large

    Useful for characterizing algebraic distortion when using CD rotations
    for quantization -- higher associator norms mean the rotation deviates
    more from an isometry.

    Args:
        a, b, c: tensors of shape (..., 2^k)
        eps: regularization for near-zero norms

    Returns:
        Normalized associator norm, shape (...).
    """
    assoc = cd_associator(a, b, c)
    return cd_norm(assoc) / (cd_norm(a) * cd_norm(b) * cd_norm(c) + eps)


def cd_is_zero_divisor_pair(a: Tensor, b: Tensor, atol: float = 1e-6) -> Tensor:
    """
    Check if (a, b) is a zero divisor pair: both nonzero but ab = 0.

    Zero divisors first appear at dim=16 (sedenions). They do not exist
    in composition algebras (dim <= 8).

    Args:
        a, b: tensors of shape (..., 2^k)
        atol: absolute tolerance for "zero" and "nonzero"

    Returns:
        Boolean tensor, shape (...).
    """
    product = cd_multiply(a, b)
    product_is_zero = cd_norm(product) < atol
    a_nonzero = cd_norm(a) > atol
    b_nonzero = cd_norm(b) > atol
    return product_is_zero & a_nonzero & b_nonzero


def cd_left_mult_matrix(a: Tensor) -> Tensor:
    """
    Construct the left-multiplication matrix L_a where L_a @ e_j = a * e_j.

    This makes the linear map x -> a*x explicit as a matrix. For unit
    quaternions/octonions, this matrix is orthogonal (rotation/reflection).
    For sedenions+, it is NOT orthogonal due to zero divisors.

    Args:
        a: single CD element of shape (2^k,) -- no batch dimension.

    Returns:
        Matrix of shape (2^k, 2^k).
    """
    d = _check_cd_dim(a)
    if a.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got {a.dim()}D")

    matrix = torch.zeros(d, d, device=a.device, dtype=a.dtype)
    basis = torch.zeros(d, device=a.device, dtype=a.dtype)
    for j in range(d):
        basis.zero_()
        basis[j] = 1.0
        col = cd_multiply(a.unsqueeze(0), basis.unsqueeze(0)).squeeze(0)
        matrix[:, j] = col
    return matrix


def cd_right_mult_matrix(a: Tensor) -> Tensor:
    """
    Construct the right-multiplication matrix R_a where R_a @ e_j = e_j * a.

    Args:
        a: single CD element of shape (2^k,) -- no batch dimension.

    Returns:
        Matrix of shape (2^k, 2^k).
    """
    d = _check_cd_dim(a)
    if a.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got {a.dim()}D")

    matrix = torch.zeros(d, d, device=a.device, dtype=a.dtype)
    basis = torch.zeros(d, device=a.device, dtype=a.dtype)
    for j in range(d):
        basis.zero_()
        basis[j] = 1.0
        col = cd_multiply(basis.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
        matrix[:, j] = col
    return matrix


def cd_random_unit(
    *batch_shape: int,
    dim: int,
    seed: int = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Generate random unit CD elements (uniform on the unit hypersphere).

    Args:
        *batch_shape: batch dimensions
        dim: algebra dimension (power of 2)
        seed: optional random seed
        device: torch device
        dtype: tensor dtype

    Returns:
        Unit CD elements of shape (*batch_shape, dim).
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    x = torch.randn(*batch_shape, dim, generator=gen, dtype=dtype)
    x = x.to(device)
    return cd_normalize(x)
