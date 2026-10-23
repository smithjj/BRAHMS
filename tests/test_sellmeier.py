import numpy as np
import pytest

from gui.core_py.sellmeier import SellmeierFormula

C = 299.792458  # um/ps

# Two-term-free Sellmeier: n^2 = 1 + A L^2 / (L^2 - B)
FORMULA = "sqrt(1 + A*L**2/(L**2 - B))"
COEFFS = {"A": 2.0, "B": 0.04}


@pytest.fixture
def sf():
    return SellmeierFormula(FORMULA, COEFFS)


def analytic_n(L):
    return np.sqrt(1 + 2.0 * L**2 / (L**2 - 0.04))


def test_formula_is_ready(sf):
    assert sf.is_ready and sf.error is None


def test_n_matches_closed_form(sf):
    L = np.array([0.5, 1.064, 3.0])
    assert sf.n(L) == pytest.approx(analytic_n(L), rel=1e-12)


def test_n_accepts_scalar(sf):
    assert float(sf.n(1.064)) == pytest.approx(analytic_n(1.064), rel=1e-12)


def test_k_is_two_pi_n_over_lambda(sf):
    L = 1.064
    assert float(sf.k(L)) == pytest.approx(2 * np.pi * analytic_n(L) / L)


@pytest.mark.parametrize("order,name", [(1, "dn_dL"), (2, "d2n_dL2"), (3, "d3n_dL3")])
def test_derivatives_match_finite_difference(sf, order, name):
    L0, h = 1.064, 1e-3
    lower = {1: sf.n, 2: sf.dn_dL, 3: sf.d2n_dL2}[order]
    fd = (lower(L0 + h) - lower(L0 - h)) / (2 * h)
    assert float(getattr(sf, name)(L0)) == pytest.approx(float(fd), rel=1e-4)


def test_group_velocity_beta1_consistency(sf):
    L = 1.064
    assert float(sf.beta1(L)) * float(sf.GV(L)) == pytest.approx(1.0)
    n, dn = float(sf.n(L)), float(sf.dn_dL(L))
    assert float(sf.GV(L)) == pytest.approx(C / (n - L * dn))


def test_beta2_formula_and_unit_conversion(sf):
    L = 1.064
    expected = L**3 * float(sf.d2n_dL2(L)) / (2 * np.pi * C**2)
    assert float(sf.beta2(L)) == pytest.approx(expected)
    assert float(sf.GVD_fs2_mm(L)) == pytest.approx(expected * 1e9)


def test_beta3_formula(sf):
    L = 1.064
    d2, d3 = float(sf.d2n_dL2(L)), float(sf.d3n_dL3(L))
    expected = -L**4 / (4 * np.pi**2 * C**3) * (3 * d2 + L * d3)
    assert float(sf.beta3(L)) == pytest.approx(expected)


def test_dispersionless_medium_has_no_gvd():
    flat = SellmeierFormula("n0", {"n0": 2.2})
    assert float(flat.n(1.0)) == 2.2
    assert float(flat.beta2(1.0)) == 0.0
    assert float(flat.beta1(1.0)) == pytest.approx(2.2 / C)


def test_temperature_dependent_formula():
    sf = SellmeierFormula("n0 + dndT*(T - 20)", {"n0": 2.0, "dndT": 1e-5})
    assert float(sf.n(1.0, 20.0)) == pytest.approx(2.0)
    assert float(sf.n(1.0, 120.0)) == pytest.approx(2.001)


def test_undefined_coefficient_is_reported_not_raised_later():
    bad = SellmeierFormula(FORMULA, {"A": 2.0})
    assert not bad.is_ready
    assert "B" in bad.error


def test_unparseable_formula_is_reported():
    bad = SellmeierFormula("sqrt(1 +", {})
    assert not bad.is_ready and bad.error


def test_evaluating_a_broken_formula_raises():
    bad = SellmeierFormula("sqrt(1 +", {})
    with pytest.raises(Exception):
        bad.n(1.0)
