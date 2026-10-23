"""Beam Analysis tab: table, options, time slider/movie, CSV export."""

import csv
import json

import numpy as np
import pytest

from gui.tabs.beam_analysis import BeamAnalysisTab, _COLUMNS
from test_beam_analysis import K0, N, D, gauss, run_dir, xy, _write_field  # noqa: F401  (fixtures)

COL = {c[2]: j for j, c in enumerate(_COLUMNS, start=1)}


@pytest.fixture
def tab(qapp):
    t = BeamAnalysisTab()
    yield t
    t._timer.stop()


def cell(tab, row, field):
    return tab.table.item(row, COL[field]).text()


def row_of(tab, name):
    return next(i for i in range(tab.table.rowCount()) if tab.table.item(i, 0).text() == name)


def test_starts_empty(tab):
    assert tab.table.rowCount() == 0 and "No data" in tab.lbl_note.text()
    assert not tab.btn_csv.isEnabled()


def test_loading_a_run_fills_one_row_per_wave(tab, run_dir):
    tab.load_output_dir(str(run_dir))
    names = [tab.table.item(i, 0).text() for i in range(tab.table.rowCount())]
    assert names == ["Input pump", "Pump", "Signal", "Idler"]
    assert tab.btn_csv.isEnabled()
    # whole-pulse view is the default; the signal was built with w = 30 μm, M² = 1
    r = row_of(tab, "Signal")
    assert float(cell(tab, r, "wx")) == pytest.approx(30, rel=5e-3)
    assert float(cell(tab, r, "M2x")) == pytest.approx(1.0, abs=0.02)


def test_analysis_is_deferred_until_the_tab_is_needed(tab, run_dir):
    tab.set_output_dir(str(run_dir))
    assert tab._run is None and tab.table.rowCount() == 0           # not shown yet: nothing computed
    tab.refresh_if_needed()                                         # what showEvent does
    assert tab._run is not None and tab.table.rowCount() == 4
    tab.show()                                                      # already analysed: no repeat
    assert tab._stale is False


def test_curvature_sign_and_units_in_the_table(tab, run_dir):
    tab.load_output_dir(str(run_dir))
    n = tab._run.waves["Pump"].n
    # the field was built with R = +8000 μm at n = 2.2; Pump uses n = 2.17
    assert float(cell(tab, row_of(tab, "Pump"), "Rx")) == pytest.approx(8.0 * 2.17 / 2.2, rel=0.02)   # mm
    tab.cb_frame.setCurrentIndex(1)                                  # after a flat exit face
    assert float(cell(tab, row_of(tab, "Pump"), "Rx")) == pytest.approx(8.0 * 2.17 / 2.2 / n, rel=0.02)
    assert "air" in tab.lbl_note.text() and "R < 0: converging" in tab.lbl_note.text()


def test_flat_phase_front_shows_infinity(tab, tmp_path, xy):
    d = tmp_path / "flat"; d.mkdir()
    (d / "config.json").write_text(json.dumps({"crystal": {
        "geometry": {"LX_mm": N * D * 1e-3, "LY_mm": N * D * 1e-3}, "degenerate": True,
        "wavelengths": {"pump_um": 0.8, "signal_um": 0.4, "idler_um": 0.4},
        "optics": {"np": 2.2, "ns": 2.3, "ni": 2.3}}}))
    _write_field(d / "pump_output.h5", gauss(*xy, 40, 40)[None])
    tab.load_output_dir(str(d))
    assert cell(tab, 0, "Rx") in ("∞",) or abs(float(cell(tab, 0, "Rx"))) > 1e3


def test_single_slice_view_follows_the_slider(tab, run_dir):
    tab.load_output_dir(str(run_dir))
    tab.cb_mode.setCurrentIndex(1)
    assert tab.sl_t.isEnabled() and tab.btn_play.isEnabled()
    tab.sb_power.setValue(0.0); tab._recompute()                     # analyse every slice
    tab.sl_t.setValue(1)
    assert tab.lbl_t.text() == "-0.500 ps"
    # the pulse envelope does not change the beam: same numbers at every time
    a = cell(tab, row_of(tab, "Signal"), "wx")
    tab.sl_t.setValue(3)
    assert cell(tab, row_of(tab, "Signal"), "wx") == a and tab.lbl_t.text() == "0.500 ps"


def test_slices_below_the_power_threshold_are_marked_not_analysed(tab, run_dir):
    tab.load_output_dir(str(run_dir))
    tab.cb_mode.setCurrentIndex(1)
    tab.sb_power.setValue(50.0); tab._recompute()
    tab.sl_t.setValue(0)                                             # the weak edge of the pulse
    assert cell(tab, row_of(tab, "Signal"), "M2x") == "—"
    assert "not analysed" in tab.lbl_note.text()
    tab.sl_t.setValue(2)                                             # the peak
    assert cell(tab, row_of(tab, "Signal"), "M2x") != "—"


def test_selected_time_survives_an_option_change(tab, run_dir):
    tab.load_output_dir(str(run_dir))
    tab.cb_mode.setCurrentIndex(1)
    tab.sl_t.setValue(3)
    tab.sb_noise.setValue(1.0); tab._recompute()
    assert tab.sl_t.value() == 3


def test_option_edits_are_debounced_then_reanalysed(tab, run_dir):
    tab.load_output_dir(str(run_dir))
    first = tab._run
    tab.sb_noise.setValue(5.0)
    assert tab._opt_timer.isActive() and tab._run is first
    tab._opt_timer.timeout.emit()                                    # the timer firing
    assert tab._run is not first


def test_noise_floor_is_off_by_default(tab):
    assert tab.sb_noise.value() == 0.0 and tab.sb_noise.specialValueText() == "off"


def test_edge_warning_highlights_the_cell(tab, tmp_path, xy):
    d = tmp_path / "wide"; d.mkdir()
    (d / "config.json").write_text(json.dumps({"crystal": {
        "geometry": {"LX_mm": N * D * 1e-3, "LY_mm": N * D * 1e-3}, "degenerate": True,
        "wavelengths": {"pump_um": 0.8, "signal_um": 0.4, "idler_um": 0.4},
        "optics": {"np": 2.2, "ns": 2.3, "ni": 2.3}}}))
    _write_field(d / "pump_output.h5", gauss(*xy, 140, 140)[None])
    tab.load_output_dir(str(d))
    item = tab.table.item(0, COL["edge_fraction"])
    assert item.background().color().name() == "#a65e00" and "edge" in tab.lbl_note.text()


def test_play_runs_only_in_single_slice_view(tab, run_dir):
    tab.load_output_dir(str(run_dir))
    assert not tab.btn_play.isEnabled()                              # whole pulse: nothing to play
    tab.cb_mode.setCurrentIndex(1)
    tab.btn_play.setChecked(True)
    assert tab._timer.isActive() and tab._timer.interval() == 100
    tab.sb_fps.setValue(20)
    assert tab._timer.interval() == 50
    tab.cb_mode.setCurrentIndex(0)                                   # leaving the view stops the movie
    assert not tab._timer.isActive() and not tab.btn_play.isChecked()


def test_playback_loops_through_the_slices(tab, run_dir):
    tab.load_output_dir(str(run_dir))
    tab.cb_mode.setCurrentIndex(1)
    tab.sl_t.setValue(3)
    seen = []
    for _ in range(4):
        tab._advance_frame(); seen.append(tab.sl_t.value())
    assert seen == [4, 0, 1, 2]


def test_every_plot_quantity_draws_all_waves(tab, run_dir):
    tab.load_output_dir(str(run_dir))
    tab.cb_mode.setCurrentIndex(1)
    for i in range(tab.cb_plot.count()):
        tab.cb_plot.setCurrentIndex(i)
        ax = tab.canvas.get_ax(0)
        assert ax.get_title() == tab.cb_plot.itemText(i)
        assert len(ax.lines) >= 2 * 4 + 1                            # x and y per wave + time marker
    tab.cb_plot.setCurrentIndex(2)
    assert "1/R" in tab.canvas.get_ax(0).get_ylabel()


def test_time_marker_only_in_single_slice_view(tab, run_dir):
    tab.load_output_dir(str(run_dir))
    n_fluence = len(tab.canvas.get_ax(0).lines)
    tab.cb_mode.setCurrentIndex(1)
    assert len(tab.canvas.get_ax(0).lines) == n_fluence + 1


def test_bad_directory_reports_the_problem(tab, tmp_path):
    tab.load_output_dir(str(tmp_path))                               # no config.json
    assert tab.table.rowCount() == 0 and "Could not analyse" in tab.lbl_note.text()
    assert not tab.btn_csv.isEnabled()


def test_export_csv(tab, run_dir, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    tab.load_output_dir(str(run_dir))
    out = tmp_path / "beams.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    tab._export_csv()
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 4 and rows[0]["slice"] == "all"                 # whole pulse
    tab.cb_mode.setCurrentIndex(1)
    tab._export_csv()
    assert len(list(csv.DictReader(open(out)))) == 4 * 5


def test_export_cancelled_writes_nothing(tab, run_dir, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    tab.load_output_dir(str(run_dir))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    tab._export_csv()
    assert not list(tmp_path.glob("*.csv"))


def test_main_window_has_a_beam_analysis_tab_in_each_results_group(qapp):
    from gui.mainwindow import MainWindow
    w = MainWindow()
    for group, beam in ((w.single_sim_tabs, w.beam_tab), (w.full_3d_tabs, w.full_beam_tab)):
        titles = [group.tabText(i).strip() for i in range(group.count())]
        assert titles[-1] == "Beam Analysis" and group.indexOf(beam) == group.count() - 1
    assert w.beam_tab is not w.full_beam_tab
