import json

import numpy as np
import pytest

from gui.core_py.config_builder import (
    _dk_qpm, _field_block, _idler_wavelength, build_config, config_summary,
    energy_from_peak_power, peak_power_from_energy)


def make_params(process="OPG", crystal="MgO:PPLN", **over):
    p = {
        "process": process,
        "crystal": crystal,
        "temperature": 27.0,
        "grating_um": 6.99,
        "length_mm": 10.0,
        "lx_mm": 0.5, "ly_mm": 0.5,
        "pump": {"lambda_um": 1.064 if process == "SHG" else 0.8,
                 "power_W": 1.0, "waist_um": 30.0, "fwhm_ps": 0.2,
                 "mode": "focused-cw"},
        "signal": {"lambda_um": 1.5, "excitation": "noise"},
        "idler": {"lambda_um": 1.5, "degenerate": False, "excitation": "noise"},
    }
    p.update(over)
    return p


class TestHelpers:
    def test_idler_energy_conservation(self):
        li = _idler_wavelength(0.8, 1.5)
        assert 1 / li == pytest.approx(1 / 0.8 - 1 / 1.5)

    def test_idler_degenerate_point(self):
        assert _idler_wavelength(0.532, 1.064) == pytest.approx(1.064)

    @pytest.mark.parametrize("lp,ls", [(1.5, 0.8), (1.0, 1.0)])
    def test_unphysical_idler_raises(self, lp, ls):
        with pytest.raises(ValueError):
            _idler_wavelength(lp, ls)

    def test_dk_qpm(self):
        assert _dk_qpm(10.0, 4.0, 5.0, 2 * np.pi) == pytest.approx(0.0)
        assert _dk_qpm(10.0, 4.0, 5.0, 0.0) == pytest.approx(1.0)

    def test_field_block_noise(self):
        fb = _field_block({"excitation": "noise"}, 30.0, 0.2, 10.0)
        assert fb["noise"] is True
        assert fb["waist_um"] == 30.0 and fb["fwhm_ps"] == 0.2
        assert fb["focal_point_um"] == 5000.0

    def test_field_block_inject(self):
        fb = _field_block({"excitation": "inject", "power_W": 0.01, "waist_um": 20.0,
                           "fwhm_ps": 0.5, "mode": "focused-pulsed"}, 30.0, 0.2, 10.0)
        assert fb["noise"] is False
        assert (fb["power_W"], fb["waist_um"], fb["fwhm_ps"]) == (0.01, 20.0, 0.5)
        assert fb["mode"] == "focused-pulsed"


class TestPulseEnergy:
    def test_gaussian_energy_peak_power_relation(self):
        # E = P_peak · FWHM · sqrt(pi / (4 ln 2)) ≈ 1.0645 · P_peak · FWHM
        assert energy_from_peak_power(1.0, 1.0) == pytest.approx(1.0645e-12, rel=1e-4)

    @pytest.mark.parametrize("E,F", [(1e-9, 0.2), (3e-6, 5.0), (1e-12, 0.001)])
    def test_roundtrip(self, E, F):
        assert energy_from_peak_power(peak_power_from_energy(E, F), F) == pytest.approx(E)

    def test_matches_numerical_integral_of_intensity_fwhm_pulse(self):
        F, P = 0.35, 12.0                                     # ps, W
        t = np.linspace(-20 * F, 20 * F, 400001)
        I = P * np.exp(-4 * np.log(2) * (t / F) ** 2)         # intensity FWHM = F
        E_num = np.trapezoid(I, t) * 1e-12
        assert energy_from_peak_power(P, F) == pytest.approx(E_num, rel=1e-6)

    def test_nonpositive_fwhm_rejected(self):
        with pytest.raises(ValueError):
            peak_power_from_energy(1e-9, 0.0)

    def test_pump_energy_becomes_peak_power(self, db):
        p = make_params()
        p["pump"].update(mode="focused-pulsed", fwhm_ps=0.5)
        p["pump"].pop("power_W"); p["pump"]["energy_J"] = 2e-9
        got = build_config(p, db)["fields"]["pump"]["power_W"]
        assert got == pytest.approx(peak_power_from_energy(2e-9, 0.5))

    def test_energy_wins_over_stale_power(self, db):
        p = make_params()
        p["pump"].update(mode="focused-pulsed", fwhm_ps=0.5, power_W=999.0, energy_J=2e-9)
        assert build_config(p, db)["fields"]["pump"]["power_W"] == \
            pytest.approx(peak_power_from_energy(2e-9, 0.5))

    def test_cw_pump_rejects_energy(self, db):
        p = make_params()
        p["pump"]["energy_J"] = 1e-9
        with pytest.raises(ValueError, match="pulsed"):
            build_config(p, db)

    def test_injected_pulsed_signal_energy_uses_its_own_fwhm(self, db):
        p = make_params()
        p["signal"].update(excitation="inject", mode="focused-pulsed",
                           fwhm_ps=2.0, energy_J=1e-12)
        got = build_config(p, db)["fields"]["signal"]["power_W"]
        assert got == pytest.approx(peak_power_from_energy(1e-12, 2.0))

    def test_noise_signal_ignores_energy(self, db):
        p = make_params()
        p["signal"].update(energy_J=1e-12, mode="focused-cw")   # would be invalid if used
        assert build_config(p, db)["fields"]["signal"]["noise"] is True


class TestBuildConfig:
    def test_is_json_serialisable(self, db):
        json.dumps(build_config(make_params(), db))

    def test_opg_wavelengths_obey_energy_conservation(self, db):
        wl = build_config(make_params(), db)["crystal"]["wavelengths"]
        assert 1 / wl["pump_um"] == pytest.approx(1 / wl["signal_um"] + 1 / wl["idler_um"])

    def test_degenerate_uses_signal_wavelength_for_idler(self, db):
        p = make_params()
        p["signal"]["lambda_um"] = 1.6
        p["idler"]["degenerate"] = True
        wl = build_config(p, db)["crystal"]["wavelengths"]
        assert wl["idler_um"] == 1.6

    def test_shg_roles(self, db):
        cfg = build_config(make_params("SHG"), db)
        wl = cfg["crystal"]["wavelengths"]
        assert wl["pump_um"] == 1.064
        assert wl["signal_um"] == wl["idler_um"] == pytest.approx(0.532)
        assert cfg["crystal"]["degenerate"] is True

    def test_shg_qpm_phase_matched_period_gives_zero_dk(self, db):
        # Choose Lambda that exactly phase-matches SHG, then dk must vanish.
        sf = db.sellmeier("MgO:PPLN", "e")
        T = 27.0
        Lam = 2 * np.pi / (float(sf.k(0.532, T)) - 2 * float(sf.k(1.064, T)))
        cfg = build_config(make_params("SHG", grating_um=Lam, temperature=T), db)
        assert cfg["crystal"]["wavelengths"]["dk_um-1"] == pytest.approx(0.0, abs=1e-9)

    def test_dk_override(self, db):
        cfg = build_config(make_params(dk=0.123), db)
        assert cfg["crystal"]["wavelengths"]["dk_um-1"] == 0.123

    def test_optics_match_sellmeier(self, db):
        cfg = build_config(make_params(), db)
        sf = db.sellmeier("MgO:PPLN", "e")
        assert cfg["crystal"]["optics"]["np"] == pytest.approx(float(sf.n(0.8, 27.0)))

    def test_signal_seed_reaches_config(self, db):
        p = make_params()
        p["signal"].update(excitation="inject", power_W=0.02, mode="focused-pulsed")
        f = build_config(p, db)["fields"]
        assert f["signal"]["noise"] is False and f["signal"]["power_W"] == 0.02
        assert f["idler"]["noise"] is True

    def test_pump_mode_from_regime_and_profile(self, db):
        p = make_params()
        p["pump"].pop("mode")
        p["pump"].update(regime="Pulsed", profile="Plane wave")
        cfg = build_config(p, db)
        assert cfg["fields"]["pump"]["mode"] == "waveplane-pulsed"
        assert cfg["dispersion"] is True

    def test_cw_disables_dispersion_and_uses_unit_time_window(self, db):
        cfg = build_config(make_params(), db)
        assert cfg["dispersion"] is False
        assert cfg["fields"]["t_window_ps"] == 1.0

    def test_pulsed_time_window_is_ten_fwhm_or_override(self, db):
        p = make_params()
        p["pump"].update(mode="focused-pulsed", fwhm_ps=0.5)
        assert build_config(p, db)["fields"]["t_window_ps"] == 5.0
        p["t_window_ps"] = 3.0
        assert build_config(p, db)["fields"]["t_window_ps"] == 3.0

    def test_absorption_flags(self, db):
        on = build_config(make_params(), db)["crystal"]["nonlinear"]
        off = build_config(make_params(alpha_p_active=False), db)["crystal"]["nonlinear"]
        assert on["alpha_crp_cm-1"] > 0 and off["alpha_crp_cm-1"] == 0.0
        db.update("MgO:PPLN", {"beta_s": 5e-5})
        off_tpa = build_config(make_params(), db)["crystal"]["nonlinear"]
        assert off_tpa["beta_crs_um_W"] == 0.0      # TPA off unless requested
        tpa = build_config(make_params(beta_s_active=True), db)["crystal"]["nonlinear"]
        assert tpa["beta_crs_um_W"] > 0

    def test_qpm_section_present_for_qpm_only(self, db):
        assert "qpm" in build_config(make_params(), db)["crystal"]
        bbo = make_params("SHG", crystal="BBO", pm_type="ooe", theta_deg=22.8)
        assert "qpm" not in build_config(bbo, db)["crystal"]

    def test_thermal_section(self, db):
        assert build_config(make_params(), db)["thermal"] == {"enabled": False}
        p = make_params(thermal={"enabled": True, "T_oven_C": 50.0, "tol": 1e-3})
        th = build_config(p, db)["thermal"]
        assert th["enabled"] and th["T_oven_C"] == 50.0 and th["tol"] == 1e-3

    def test_unknown_crystal_raises(self, db):
        with pytest.raises(ValueError, match="not found"):
            build_config(make_params(crystal="Unobtainium"), db)

    def test_birefringent_requires_pm_type_and_angle(self, db):
        with pytest.raises(ValueError, match="pm_type"):
            build_config(make_params("SHG", crystal="BBO"), db)

    def test_birefringent_type_II_shg_rejected(self, db):
        with pytest.raises(ValueError, match="Type II"):
            build_config(make_params("SHG", crystal="BBO", pm_type="oee", theta_deg=30), db)

    def test_invalid_pm_type_string(self, db):
        with pytest.raises(ValueError, match="Invalid PM type"):
            build_config(make_params("SHG", crystal="BBO", pm_type="xyz", theta_deg=30), db)

    def test_summary_mentions_crystal_and_process(self, db):
        s = config_summary(build_config(make_params(), db))
        assert "MgO:PPLN" in s and "OPG" in s
