"""Full-results tab: the spatial panel shows irradiance (W/cm²), not |A|²."""

import numpy as np
import pytest

from gui.tabs.full_results import (
    FullResultsTab, _C, _EPS0, irradiance_W_cm2, irradiance_scale)


def test_irradiance_of_engine_gaussian_matches_2P_over_pi_w2():
    # The engine seeds A0 = sqrt(4 P / (eps0 c pi n w^2)), i.e. peak
    # irradiance 2P / (pi w^2). Units: P [W], w [um] -> W/um^2 -> W/cm^2.
    P, w, n = 1.0, 30.0, 2.2
    A0 = np.sqrt(4 * P / (_EPS0 * _C * np.pi * n * w ** 2))
    expected = 2 * P / (np.pi * w ** 2) * 1e8
    assert irradiance_W_cm2(np.array(A0), n) == pytest.approx(expected, rel=1e-9)


def test_irradiance_scales_with_index_and_ignores_phase():
    a = np.array([1.0 + 0j, 1j, -1.0])
    i = irradiance_W_cm2(a, 2.0)
    assert i == pytest.approx(np.full(3, i[0]))
    assert irradiance_W_cm2(a, 4.0)[0] == pytest.approx(2 * i[0])


@pytest.mark.parametrize("i,unit", [(5e12, "TW/cm²"), (2.5e9, "GW/cm²"), (3e6, "MW/cm²"),
                                    (4e3, "kW/cm²"), (7.0, "W/cm²"), (2e-2, "mW/cm²"),
                                    (1e-5, "μW/cm²")])
def test_irradiance_scale_units(i, unit):
    scale, u = irradiance_scale(i)
    assert u == unit and 1.0 <= i * scale < 1000.0


@pytest.fixture
def tab(qapp):
    return FullResultsTab()


def _load_synthetic(tab, amp=1.0, nt=9, ny=16, nx=16, n=2.0):
    t = np.arange(nt)
    env = np.exp(-((t - nt // 2) / 2.0) ** 2)                  # peaks at the middle slice
    A = (amp * env[:, None, None] * np.ones((1, ny, nx))).astype(np.complex64)
    tab._fields["Signal"] = A
    tab._t_ps = np.linspace(-1, 1, nt).astype(np.float32)
    tab._cfg = {"crystal": {"geometry": {"LX_mm": 0.5, "LY_mm": 0.5},
                            "optics": {"np": n, "ns": n, "ni": n}},
                "fields": {"t_window_ps": 2.0}}
    tab.sl_t.setRange(0, nt - 1)
    tab.cb_wave.setCurrentText("Signal")
    return A


def test_title_reports_max_irradiance_at_selected_time(tab):
    A = _load_synthetic(tab, amp=3.0e-2)
    tab.sl_t.setValue(4)
    tab._draw_xy()
    ax = tab.canvas_xy.get_ax(0)
    title = ax.get_title()
    peak = float(irradiance_W_cm2(A[4], 2.0).max())
    scale, unit = irradiance_scale(peak)
    assert "irradiance" in title and "t = " in title
    assert f"max = {peak * scale:.3g} {unit}" in title
    # the image shows the same quantity, in the same unit
    assert ax.images[0].get_array().max() == pytest.approx(peak * scale, rel=1e-5)


def test_title_maximum_follows_the_time_slider(tab):
    _load_synthetic(tab, amp=1.0)
    tab.sl_t.setValue(4); tab._draw_xy()
    mid = tab.canvas_xy.get_ax(0).get_title()
    tab.sl_t.setValue(0); tab._draw_xy()
    edge = tab.canvas_xy.get_ax(0).get_title()
    assert mid != edge


def test_fixed_color_scale_uses_the_maximum_over_all_times(tab):
    A = _load_synthetic(tab, amp=3.0e-2)
    peak = float(irradiance_W_cm2(A, 2.0).max())
    scale, _ = irradiance_scale(peak)
    tab.sl_t.setValue(0)                               # a dim frame
    tab.chk_fixed_scale.setChecked(True); tab._draw_xy()
    assert tab.canvas_xy.get_ax(0).images[0].get_clim() == pytest.approx((0.0, peak * scale))
    tab.chk_fixed_scale.setChecked(False)              # redraws via the toggle
    frame_max = float(irradiance_W_cm2(A[0], 2.0).max())
    assert tab.canvas_xy.get_ax(0).images[0].get_clim()[1] == \
        pytest.approx(frame_max * irradiance_scale(frame_max)[0])


def test_play_button_starts_and_stops_the_timer(tab):
    _load_synthetic(tab)
    tab.btn_play.setChecked(True)
    assert tab._timer.isActive() and tab._timer.interval() == 100     # 10 fps
    tab.sb_fps.setValue(20)
    assert tab._timer.interval() == 50
    tab.btn_play.setChecked(False)
    assert not tab._timer.isActive()


def test_play_does_nothing_without_data(tab):
    tab.btn_play.setChecked(True)
    assert not tab.btn_play.isChecked() and not tab._timer.isActive()


def test_playback_steps_through_all_slices_and_loops(tab):
    _load_synthetic(tab, nt=5)
    tab.sl_t.setValue(0)
    seen = []
    for _ in range(7):
        tab._advance_frame()
        seen.append(tab.sl_t.value())
    assert seen == [1, 2, 3, 4, 0, 1, 2]
    # each step redrew the plot at the new time
    assert f"t = {tab._t_label(2)}" in tab.canvas_xy.get_ax(0).get_title()


def test_colorbar_is_labelled_with_the_unit(tab):
    A = _load_synthetic(tab, amp=3.0e-2)
    tab.sl_t.setValue(4); tab._draw_xy()
    unit = irradiance_scale(float(irradiance_W_cm2(A[4], 2.0).max()))[1]
    assert tab._cbar_xy.ax.get_ylabel() == f"Irradiance ({unit})"
