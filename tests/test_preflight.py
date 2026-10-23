"""Preflight estimates: peak irradiance per wave and characteristic irradiance S0."""

import math

import pytest

from gui.core_py.config_builder import build_config, config_summary
from gui.core_py.preflight import (
    characteristic_irradiance_W_cm2, format_irradiance, peak_irradiance_W_cm2, preflight)

EPS0, C = 8.8541878128e-12, 299792458.0


def make_params(process="OPG", **over):
    p = {
        "process": process, "crystal": "MgO:PPLN",
        "temperature": 27.0, "grating_um": 6.99, "length_mm": 10.0,
        "lx_mm": 0.5, "ly_mm": 0.5,
        "pump": {"lambda_um": 1.064 if process == "SHG" else 0.8, "power_W": 1000.0,
                 "waist_um": 30.0, "fwhm_ps": 0.2, "mode": "focused-pulsed"},
        "signal": {"lambda_um": 1.5, "excitation": "noise"},
        "idler": {"lambda_um": 1.5, "degenerate": False, "excitation": "noise"},
    }
    p.update(over)
    return p


def _s0_omega_form(cfg, n1, n2, n3, lam1_um, lam2_um):
    """S0 = eps0 c^3 n1 n2 n3 / (2 d^2 w1 w2 L^2), independent of the lambda form."""
    d = cfg["crystal"]["nonlinear"]["deff_pm_V"] * 1e-12 * 2 / math.pi     # QPM
    L = cfg["crystal"]["geometry"]["Lcr_mm"] * 1e-3
    w1 = 2 * math.pi * C / (lam1_um * 1e-6)
    w2 = 2 * math.pi * C / (lam2_um * 1e-6)
    return EPS0 * C ** 3 * n1 * n2 * n3 / (2 * d ** 2 * w1 * w2 * L ** 2) * 1e-4   # W/cm²


class TestPeakIrradiance:
    def test_gaussian_peak(self):
        # 1 W through w = 100 um: 2/(pi 1e-4 cm²)·… = 2P/(pi w²)
        assert peak_irradiance_W_cm2(1.0, 100.0) == pytest.approx(2 / (math.pi * 1e4) * 1e8)

    def test_scales_with_power_and_inverse_waist_squared(self):
        a = peak_irradiance_W_cm2(1.0, 50.0)
        assert peak_irradiance_W_cm2(3.0, 50.0) == pytest.approx(3 * a)
        assert peak_irradiance_W_cm2(1.0, 100.0) == pytest.approx(a / 4)


class TestCharacteristicIrradiance:
    def test_opg_matches_independent_omega_form(self, db):
        cfg = build_config(make_params("OPG"), db)
        o, wl = cfg["crystal"]["optics"], cfg["crystal"]["wavelengths"]
        want = _s0_omega_form(cfg, o["ns"], o["ni"], o["np"], wl["signal_um"], wl["idler_um"])
        assert characteristic_irradiance_W_cm2(cfg) == pytest.approx(want, rel=1e-9)

    def test_shg_uses_fundamental_for_both_redder_waves(self, db):
        cfg = build_config(make_params("SHG"), db)
        o, wl = cfg["crystal"]["optics"], cfg["crystal"]["wavelengths"]
        want = _s0_omega_form(cfg, o["np"], o["np"], o["ns"], wl["pump_um"], wl["pump_um"])
        assert characteristic_irradiance_W_cm2(cfg) == pytest.approx(want, rel=1e-9)

    def test_magnitude_for_ppln_shg_is_megawatts_per_cm2(self, db):
        s0 = characteristic_irradiance_W_cm2(build_config(make_params("SHG"), db))
        assert 1e5 < s0 < 1e8

    def test_scales_as_inverse_length_squared(self, db):
        a = characteristic_irradiance_W_cm2(build_config(make_params(length_mm=10.0), db))
        b = characteristic_irradiance_W_cm2(build_config(make_params(length_mm=20.0), db))
        assert b == pytest.approx(a / 4, rel=1e-6)

    def test_qpm_applies_two_over_pi(self, db):
        cfg = build_config(make_params("SHG"), db)
        with_factor = characteristic_irradiance_W_cm2(cfg)
        no_qpm = {**cfg, "crystal": {k: v for k, v in cfg["crystal"].items() if k != "qpm"}}
        assert characteristic_irradiance_W_cm2(no_qpm) == pytest.approx(with_factor * (2 / math.pi) ** 2)

    def test_zero_deff_gives_none(self, db):
        cfg = build_config(make_params(), db)
        cfg["crystal"]["nonlinear"]["deff_pm_V"] = 0.0
        assert characteristic_irradiance_W_cm2(cfg) is None


class TestPreflight:
    def test_pump_peak_and_ratio(self, db):
        cfg = build_config(make_params(), db)
        pf = preflight(cfg)
        pump = pf["waves"]["pump"]
        assert pump["peak_W_cm2"] == pytest.approx(peak_irradiance_W_cm2(1000.0, 30.0))
        assert pump["ratio"] == pytest.approx(pump["peak_W_cm2"] / pf["S0_W_cm2"])
        assert pump["lambda_um"] == 0.8

    def test_noise_fields_have_no_peak_irradiance(self, db):
        w = preflight(build_config(make_params(), db))["waves"]
        assert w["signal"]["peak_W_cm2"] is None and w["signal"]["ratio"] is None
        assert w["idler"]["peak_W_cm2"] is None

    def test_injected_signal_uses_its_own_power_and_waist(self, db):
        p = make_params()
        p["signal"].update(excitation="inject", mode="focused-pulsed", power_W=10.0, waist_um=20.0)
        w = preflight(build_config(p, db))["waves"]
        assert w["signal"]["peak_W_cm2"] == pytest.approx(peak_irradiance_W_cm2(10.0, 20.0))

    def test_pump_energy_input_is_converted_before_the_estimate(self, db):
        p = make_params()
        p["pump"].pop("power_W"); p["pump"]["energy_J"] = 1e-6
        peak_W = build_config(p, db)["fields"]["pump"]["power_W"]
        got = preflight(build_config(p, db))["waves"]["pump"]["peak_W_cm2"]
        assert got == pytest.approx(peak_irradiance_W_cm2(peak_W, 30.0))

    def test_summary_lines_in_config_summary(self, db):
        s = config_summary(build_config(make_params(), db))
        assert "S₀" in s and "I_pump" in s and "I/S₀" in s
        assert "I_signal" not in s            # noise-seeded: nothing to report


@pytest.mark.parametrize("v,text", [(None, "—"), (0.0, "0 W/cm²"),
                                    (2.5e9, "2.5 GW/cm²"), (4.9e6, "4.9 MW/cm²")])
def test_format_irradiance(v, text):
    assert format_irradiance(v) == text


class TestPanel:
    @pytest.fixture
    def sim_tab(self, qapp):
        from gui.tabs.simulation import SimulationTab
        return SimulationTab()

    @pytest.fixture
    def full_tab(self, qapp):
        from gui.tabs.full_simulation import FullSimulationTab
        return FullSimulationTab()

    def test_shows_values_on_open(self, sim_tab):
        assert sim_tab.preflight.lbl_s0.text().startswith("S₀ = ") \
            and sim_tab.preflight.lbl_s0.text() != "S₀ = —"
        assert sim_tab.preflight._cells["pump"][2].text() != "—"

    def test_updates_when_an_input_changes(self, sim_tab):
        pf = sim_tab.preflight
        before = pf._cells["pump"][2].text()
        sim_tab.sb_power.setValue(sim_tab.sb_power.value() * 100)
        pf.refresh()                                   # what the debounce timer fires
        assert pf._cells["pump"][2].text() != before

    def test_debounce_timer_is_started_by_edits(self, sim_tab):
        pf = sim_tab.preflight
        pf._timer.stop()
        sim_tab.sb_waist.setValue(sim_tab.sb_waist.value() + 5)
        assert pf._timer.isActive()

    def test_unbuildable_config_shows_the_reason_and_clears_values(self, sim_tab):
        pf = sim_tab.preflight
        sim_tab.cb_crystal.setCurrentText("BBO")       # no PM angle loaded yet
        pf.refresh()
        assert pf.lbl_msg.isVisibleTo(pf) and "pm_type" in pf.lbl_msg.text()
        assert pf.lbl_s0.text() == "S₀ = —" and pf._cells["pump"][2].text() == "—"

    def test_message_clears_once_inputs_are_valid_again(self, sim_tab):
        pf = sim_tab.preflight
        sim_tab.cb_crystal.setCurrentText("BBO"); pf.refresh()
        sim_tab.cb_crystal.setCurrentText("MgO:PPLN"); pf.refresh()
        assert not pf.lbl_msg.isVisibleTo(pf) and pf.lbl_s0.text() != "S₀ = —"

    def test_idler_row_hidden_when_degenerate(self, sim_tab):
        sim_tab.cb_process.setCurrentText("SHG")
        sim_tab.preflight.refresh()
        assert not sim_tab.preflight._cells["idler"][0].isVisibleTo(sim_tab.preflight)

    def test_full_tab_has_panel_and_pump_ratio(self, full_tab):
        pf = full_tab.preflight
        assert pf._cells["pump"][3].text() != "—"
        full_tab.sb_power.setValue(full_tab.sb_power.value() * 10)
        pf.refresh()
        assert pf._cells["pump"][3].text() != "—"
