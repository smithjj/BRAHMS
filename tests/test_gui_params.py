"""Headless checks that the tabs turn widget state into the params dict that
build_config() consumes, and that the result produces a valid engine config."""

import pytest

from gui.core_py.config_builder import (
    build_config, energy_from_peak_power, peak_power_from_energy)


@pytest.fixture
def sim_tab(qapp):
    from gui.tabs.simulation import SimulationTab
    return SimulationTab()


@pytest.fixture
def full_tab(qapp):
    from gui.tabs.full_simulation import FullSimulationTab
    return FullSimulationTab()


def test_simulation_tab_defaults_build_a_config(sim_tab):
    cfg = build_config(sim_tab._collect_params(backend="cpu"))
    assert cfg["crystal"]["process"] == "SHG"


def test_backend_is_forwarded(sim_tab):
    assert sim_tab._collect_params(backend="cpu")["backend"] == "cpu"
    assert sim_tab._collect_params(backend="cuda")["backend"] == "cuda"


def test_simulation_signal_seed_roundtrip(sim_tab):
    sim_tab.cb_process.setCurrentText("OPG")
    sim_tab.signal_gb.rb_inject.setChecked(True)
    sim_tab.signal_gb.sb_power.setValue(0.05)
    p = sim_tab._collect_params()
    assert p["signal"]["excitation"] == "inject"
    assert p["signal"]["power_W"] == 0.05
    assert p["idler"]["excitation"] == "noise"
    f = build_config(p)["fields"]
    assert f["signal"]["noise"] is False and f["signal"]["power_W"] == 0.05


def test_full_tab_defaults_to_noise_seeds(full_tab):
    p = full_tab._collect_params()
    assert p["signal"]["excitation"] == "noise"
    assert p["idler"]["excitation"] == "noise"


def test_full_tab_exposes_seeding(full_tab):
    full_tab.cb_process.setCurrentText("DFG")
    full_tab.signal_gb.rb_inject.setChecked(True)
    full_tab.signal_gb.sb_power.setValue(0.01)
    p = full_tab._collect_params()
    assert p["signal"]["excitation"] == "inject" and p["signal"]["power_W"] == 0.01
    f = build_config(p)["fields"]
    assert f["signal"]["noise"] is False
    assert f["signal"]["mode"] == "focused-pulsed"     # full tab seeds default to pulsed
    assert f["idler"]["noise"] is True


def test_full_tab_shg_disables_seed_boxes(full_tab):
    full_tab.cb_process.setCurrentText("SHG")
    assert not full_tab.signal_gb.isEnabled()
    assert not full_tab.idler_gb.isEnabled()
    full_tab.cb_process.setCurrentText("OPG")
    assert full_tab.signal_gb.isEnabled() and full_tab.idler_gb.isEnabled()


def test_full_tab_degenerate_disables_idler_box_only(full_tab):
    full_tab.cb_process.setCurrentText("OPG")
    full_tab.chk_degen.setChecked(True)
    assert not full_tab.idler_gb.isEnabled()
    assert full_tab.signal_gb.isEnabled()
    full_tab.chk_degen.setChecked(False)
    assert full_tab.idler_gb.isEnabled()


def _select_energy(ctl):
    ctl.cb_kind.setCurrentIndex(ctl.cb_kind.findData("energy"))


def test_energy_selector_disabled_for_cw_pump(sim_tab):
    sim_tab.cb_field_mode.setCurrentIndex(sim_tab.cb_field_mode.findData("focused-cw"))
    assert not sim_tab.power_ctl.cb_kind.isEnabled()
    assert "energy_J" not in sim_tab._collect_params()["pump"]


def test_pump_energy_is_sent_and_converted(sim_tab):
    sim_tab.cb_field_mode.setCurrentIndex(sim_tab.cb_field_mode.findData("focused-pulsed"))
    sim_tab.sb_fwhm.setValue(0.5)
    _select_energy(sim_tab.power_ctl)
    sim_tab.sb_power.setValue(2e-9)
    pump = sim_tab._collect_params()["pump"]
    assert pump["energy_J"] == 2e-9
    cfg = build_config(sim_tab._collect_params())
    assert cfg["fields"]["pump"]["power_W"] == pytest.approx(peak_power_from_energy(2e-9, 0.5))


def test_switching_units_keeps_the_same_pulse(sim_tab):
    sim_tab.cb_field_mode.setCurrentIndex(sim_tab.cb_field_mode.findData("focused-pulsed"))
    sim_tab.sb_fwhm.setValue(0.4)
    sim_tab.sb_power.setValue(100.0)                     # W
    _select_energy(sim_tab.power_ctl)
    assert sim_tab.sb_power.value() == pytest.approx(energy_from_peak_power(100.0, 0.4))
    sim_tab.power_ctl.cb_kind.setCurrentIndex(0)
    assert sim_tab.sb_power.value() == pytest.approx(100.0)


def test_going_cw_reverts_energy_to_power(sim_tab):
    sim_tab.cb_field_mode.setCurrentIndex(sim_tab.cb_field_mode.findData("focused-pulsed"))
    sim_tab.sb_power.setValue(100.0)
    _select_energy(sim_tab.power_ctl)
    sim_tab.cb_field_mode.setCurrentIndex(sim_tab.cb_field_mode.findData("focused-cw"))
    p = sim_tab._collect_params()["pump"]
    assert "energy_J" not in p and p["power_W"] == pytest.approx(100.0)
    build_config(sim_tab._collect_params())              # must not raise


def test_energy_change_follows_fwhm_edit(sim_tab):
    sim_tab.cb_field_mode.setCurrentIndex(sim_tab.cb_field_mode.findData("focused-pulsed"))
    _select_energy(sim_tab.power_ctl)
    sim_tab.sb_power.setValue(1e-9)
    sim_tab.sb_fwhm.setValue(1.0)
    p1 = sim_tab.power_ctl.peak_power_W()
    sim_tab.sb_fwhm.setValue(2.0)
    assert sim_tab.power_ctl.peak_power_W() == pytest.approx(p1 / 2)


def test_injected_signal_energy(sim_tab):
    sim_tab.cb_process.setCurrentText("OPG")
    gb = sim_tab.signal_gb
    gb.rb_inject.setChecked(True)
    gb.cb_field_mode.setCurrentIndex(gb.cb_field_mode.findData("focused-pulsed"))
    gb.sb_fwhm.setValue(2.0)
    _select_energy(gb.power_ctl)
    gb.sb_power.setValue(1e-12)
    f = build_config(sim_tab._collect_params())["fields"]
    assert f["signal"]["power_W"] == pytest.approx(peak_power_from_energy(1e-12, 2.0))


def test_full_tab_pump_energy(full_tab):
    assert full_tab.power_ctl.cb_kind.isEnabled()        # always pulsed
    _select_energy(full_tab.power_ctl)
    full_tab.sb_power.setValue(5e-6)
    full_tab.sb_fwhm.setValue(0.2)
    cfg = build_config(full_tab._collect_params())
    assert cfg["fields"]["pump"]["power_W"] == pytest.approx(peak_power_from_energy(5e-6, 0.2))


def test_forced_dk_is_only_sent_when_checked(sim_tab):
    sim_tab.sb_dk_forced.setValue(2.5e-4)
    assert "dk" not in sim_tab._collect_params()
    sim_tab.chk_force_dk.setChecked(True)
    assert sim_tab._collect_params()["dk"] == pytest.approx(2.5e-4)
    assert build_config(sim_tab._collect_params())["crystal"]["wavelengths"]["dk_um-1"] \
        == pytest.approx(2.5e-4)
