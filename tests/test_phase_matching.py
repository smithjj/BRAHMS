import numpy as np
import pytest

from gui.core_py.phase_matching import (
    PhaseMatchingCalc, UniaxialPMCalc, _three_wavelengths, crystal_sign,
    wavelengths_for_scan)
from gui.core_py.sellmeier import SellmeierFormula


@pytest.fixture
def ppln(db):
    return PhaseMatchingCalc(db.sellmeier("MgO:PPLN", "e"))


@pytest.fixture
def flat():
    """Dispersionless medium: k = 2*pi*n/lambda exactly, so dk is analytic."""
    return PhaseMatchingCalc(SellmeierFormula("n0", {"n0": 2.0}))


class TestWavelengthHelpers:
    def test_three_wavelengths_shg(self):
        assert _three_wavelengths(1.064, "SHG") == (0.532, 1.064, 1.064)

    @pytest.mark.parametrize("proc", ["OPG", "SFG", "DFG"])
    def test_three_wavelengths_pump_driven(self, proc):
        assert _three_wavelengths(0.8, proc) == (0.8, 1.6, 1.6)

    def test_scan_shg(self):
        l1, l2, l3 = wavelengths_for_scan(np.array([1.0, 2.0]), 0.0, "SHG")
        assert np.allclose(l1, l2) and np.allclose(l3, l1 / 2)

    @pytest.mark.parametrize("proc", ["SFG", "DFG", "OPG"])
    def test_scan_energy_conservation(self, proc):
        l1, l2, l3 = wavelengths_for_scan(np.array([1.2, 1.5, 3.0]), 0.8, proc)
        assert np.allclose(1 / l3, 1 / l1 + 1 / l2)
        assert np.all(l3 < l1) and np.all(l3 < l2)

    def test_scan_nan_where_energy_conservation_impossible(self):
        _, l2, _ = wavelengths_for_scan(np.array([0.5, 1.5]), 0.8, "OPG")
        assert np.isnan(l2[0]) and not np.isnan(l2[1])


class TestDeltaK:
    def test_flat_medium_is_automatically_phase_matched(self, flat):
        assert flat.dk(0.5, 1.0, 1.0, 25.0) == pytest.approx(0.0, abs=1e-12)

    def test_qpm_term_is_minus_two_pi_over_period(self, flat):
        lam = 6.9
        assert flat.dk(0.5, 1.0, 1.0, 27.0, Lambda=lam) == pytest.approx(
            -2 * np.pi / lam)

    def test_grating_expands_with_temperature(self, flat):
        lam0 = 6.9
        expected = lam0 * (1 + flat.alpha_th * (60.0 - flat.T0))
        assert flat._Lambda_T(lam0, 60.0) == pytest.approx(expected)
        assert flat._Lambda_T(lam0, flat.T0) == pytest.approx(lam0)

    def test_dk_array_matches_scalar(self, ppln):
        lp = np.array([0.5, 0.532]); ls = np.array([1.0, 1.064])
        arr = ppln.dk_array(lp, ls, ls, 27.0, Lambda=6.9)
        for i in range(2):
            assert arr[i] == pytest.approx(ppln.dk(lp[i], ls[i], ls[i], 27.0, 6.9))


class TestQpmTemperature:
    def test_pm_temperature_recovers_design_temperature(self, ppln):
        """Pick the grating period that phase-matches SHG at 40 C, then check
        pm_temperature finds 40 C again (self-consistency of dk + root finder)."""
        lp, lf, T_design = 0.532, 1.064, 40.0
        k_sh = float(ppln.sf_e.k(lp, T_design))
        k_f = float(ppln.sf_e.k(lf, T_design))
        Lam_at_T = 2 * np.pi / (k_sh - 2 * k_f)
        Lam0 = Lam_at_T / (1 + ppln.alpha_th * (T_design - ppln.T0))
        T_pm = ppln.pm_temperature(lp, lf, lf, Lam0, np.linspace(10, 120, 56))
        assert T_pm == pytest.approx(T_design, abs=0.05)

    def test_pm_temperature_nan_when_no_root(self, ppln):
        assert np.isnan(ppln.pm_temperature(0.532, 1.064, 1.064, 50.0,
                                            np.linspace(10, 100, 10)))


class TestUniaxialAngle:
    @pytest.fixture
    def bbo(self, db):
        return UniaxialPMCalc(db.sellmeier("BBO", "o"), db.sellmeier("BBO", "e"),
                              lam_ref=1.064)

    def test_bbo_is_negative(self, bbo):
        assert bbo.sign == "negative"
        assert bbo.valid_pm_types() == ["ooe", "oee", "eoe"]

    def test_bbo_type_I_shg_1064_angle(self, bbo):
        # Handbook of Nonlinear Optical Crystals: BBO ooe SHG of 1064 nm ~ 22.8 deg
        theta = bbo.theta_pm(1.064, 1.064, 0.532, 25.0, "ooe")
        assert theta == pytest.approx(22.8, abs=0.5)

    def test_wrong_sign_pm_type_rejected(self, bbo):
        with pytest.raises(ValueError):
            bbo.theta_pm(1.064, 1.064, 0.532, 25.0, "eeo")

    def test_vectorised_over_wavelength(self, bbo):
        lam = np.array([1.0, 1.064, 1.2])
        th = bbo.theta_pm(lam, lam, lam / 2, 25.0, "ooe")
        assert th.shape == (3,) and np.all(np.diff(th) < 0)  # longer lambda -> smaller angle

    def test_no_real_solution_gives_nan(self, bbo):
        # BBO type-I SHG has a short-wavelength PM cutoff near 0.41 um fundamental
        assert np.isnan(bbo.theta_pm(0.35, 0.35, 0.175, 25.0, "ooe"))
        assert not np.isnan(bbo.theta_pm(0.5, 0.5, 0.25, 25.0, "ooe"))

    def test_crystal_sign_swaps_with_axes(self, db):
        so, se = db.sellmeier("BBO", "o"), db.sellmeier("BBO", "e")
        assert crystal_sign(so, se, 1.064) == "negative"
        assert crystal_sign(se, so, 1.064) == "positive"
