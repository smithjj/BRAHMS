"""Headless checks that the tabs turn widget state into the params dict that
build_config() consumes, and that the result produces a valid engine config."""

import pytest

from gui.core_py.config_builder import build_config


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


def test_forced_dk_is_only_sent_when_checked(sim_tab):
    sim_tab.sb_dk_forced.setValue(2.5e-4)
    assert "dk" not in sim_tab._collect_params()
    sim_tab.chk_force_dk.setChecked(True)
    assert sim_tab._collect_params()["dk"] == pytest.approx(2.5e-4)
    assert build_config(sim_tab._collect_params())["crystal"]["wavelengths"]["dk_um-1"] \
        == pytest.approx(2.5e-4)
