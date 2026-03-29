"""
Pytest wrapper for formal_verify.py Z3 SMT proofs.

Each proof group runs as a separate test with the 'slow' marker.
Use: pytest test_formal_verify.py -v
Skip: pytest -m "not slow"
"""
import pytest

z3 = pytest.importorskip("z3", reason="z3-solver not installed")

from turboquant.formal_verify import (
    prove_cl3_multiplication_table,
    prove_wht_self_inverse,
    prove_quaternion_identities,
    prove_searchsorted_argmin_equivalence,
    prove_sign_packing_roundtrip,
    prove_cd_conjugation_involution,
    prove_quaternion_norm_multiplicativity,
    prove_octonion_norm_multiplicativity,
    prove_octonion_alternativity,
    prove_quaternion_inverse,
    prove_quaternion_sandwich_norm,
    prove_clifford_sandwich_norm,
    prove_normalize_unit,
    prove_nsn_invertibility,
    prove_e8_root_properties,
    prove_complex_commutativity,
    prove_quaternion_associativity,
    prove_quaternion_non_commutativity,
    prove_octonion_right_alternativity,
    prove_octonion_non_associativity,
    prove_lagrange_optimality,
    prove_cd_norm_is_real,
    prove_wht_materialized_equals_butterfly,
    prove_hadamard_orthogonality,
    prove_e8_decoder_on_roots,
    prove_sign_pack_inner_product,
    prove_sedenion_non_alternativity,
    prove_sedenion_zero_divisor_exists,
    prove_octonion_inverse,
    prove_wht_rotation_orthogonal,
    prove_lloyd_max_boundaries_sorted,
)


@pytest.mark.slow
class TestZ3Proofs:
    """Z3 SMT proofs wrapped as pytest tests."""

    def test_cl3_multiplication_table(self):
        prove_cl3_multiplication_table()

    def test_wht_self_inverse(self):
        prove_wht_self_inverse()

    def test_quaternion_identities(self):
        prove_quaternion_identities()

    def test_searchsorted_argmin_equivalence(self):
        prove_searchsorted_argmin_equivalence()

    def test_sign_packing_roundtrip(self):
        prove_sign_packing_roundtrip()

    def test_cd_conjugation_involution(self):
        prove_cd_conjugation_involution()

    def test_quaternion_norm_multiplicativity(self):
        prove_quaternion_norm_multiplicativity()

    def test_octonion_norm_multiplicativity(self):
        prove_octonion_norm_multiplicativity()

    def test_octonion_alternativity(self):
        prove_octonion_alternativity()

    def test_quaternion_inverse(self):
        prove_quaternion_inverse()

    def test_quaternion_sandwich_norm(self):
        prove_quaternion_sandwich_norm()

    def test_clifford_sandwich_norm(self):
        prove_clifford_sandwich_norm()

    def test_normalize_unit(self):
        prove_normalize_unit()

    def test_nsn_invertibility(self):
        prove_nsn_invertibility()

    def test_e8_root_properties(self):
        prove_e8_root_properties()

    def test_complex_commutativity(self):
        prove_complex_commutativity()

    def test_quaternion_associativity(self):
        prove_quaternion_associativity()

    def test_quaternion_non_commutativity(self):
        prove_quaternion_non_commutativity()

    def test_octonion_right_alternativity(self):
        prove_octonion_right_alternativity()

    def test_octonion_non_associativity(self):
        prove_octonion_non_associativity()

    def test_cd_norm_is_real(self):
        prove_cd_norm_is_real()

    def test_wht_materialized_equals_butterfly(self):
        prove_wht_materialized_equals_butterfly()

    def test_hadamard_orthogonality(self):
        prove_hadamard_orthogonality()

    def test_e8_decoder_on_roots(self):
        prove_e8_decoder_on_roots()

    def test_sign_pack_inner_product(self):
        prove_sign_pack_inner_product()

    def test_sedenion_non_alternativity(self):
        prove_sedenion_non_alternativity()

    def test_sedenion_zero_divisor_exists(self):
        prove_sedenion_zero_divisor_exists()

    def test_octonion_inverse(self):
        prove_octonion_inverse()

    def test_wht_rotation_orthogonal(self):
        prove_wht_rotation_orthogonal()

    def test_lloyd_max_boundaries_sorted(self):
        prove_lloyd_max_boundaries_sorted()

    def test_lagrange_optimality(self):
        prove_lagrange_optimality()
