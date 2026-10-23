"""Beam analysis: second-moment width, M², curvature, centroid and tilt, checked
against analytic beams (units: μm, rad/μm)."""

import csv
import json
import math
import shutil

import numpy as np
import pytest

from gui.core_py import beam_analysis as ba

N, D = 128, 2.0                  # grid points, spacing [μm]  → 256 μm window
LAMBDA, IDX = 0.8, 2.2           # vacuum wavelength [μm], index
K0 = 2 * math.pi * IDX / LAMBDA


@pytest.fixture(scope="module")
def xy():
    return ba.centered_axis(N, D), ba.centered_axis(N, D)


def gauss(x, y, wx=40.0, wy=40.0, Rx=np.inf, Ry=np.inf, x0=0.0, y0=0.0, kx=0.0, ky=0.0):
    """exp(−(x/w)²) amplitude with quadratic phase k0·x²/2R (R > 0 diverging) and tilt."""
    X, Y = x[None, :], y[:, None]
    ph = K0 * ((X - x0) ** 2 / (2 * Rx) + (Y - y0) ** 2 / (2 * Ry)) if np.isfinite(Rx) else 0.0
    return np.exp(-((X - x0) / wx) ** 2 - ((Y - y0) / wy) ** 2 + 1j * ph + 1j * (kx * X + ky * Y))


def waist(w, R):
    """Waist radius of a Gaussian with width w and radius of curvature R."""
    return w / math.sqrt(1 + (K0 * w * w / (2 * R)) ** 2)


class TestGaussian:
    def test_plain_gaussian(self, xy):
        r = ba.analyze_slices(gauss(*xy, 40, 25), *xy, K0)
        assert r.M2x[0] == pytest.approx(1.0, abs=5e-3) and r.M2y[0] == pytest.approx(1.0, abs=5e-3)
        assert r.wx[0] == pytest.approx(40, rel=1e-3) and r.wy[0] == pytest.approx(25, rel=1e-3)
        assert r.w0x[0] == pytest.approx(40, rel=5e-3)
        assert math.isinf(r.Rx[0]) or abs(r.Rx[0]) > 1e7        # flat phase front
        assert r.xbar[0] == pytest.approx(0, abs=1e-6) and r.thx[0] == pytest.approx(0, abs=1e-9)

    @pytest.mark.parametrize("Rx,Ry", [(6000.0, -9000.0), (-2500.0, 2500.0), (1500.0, -1800.0)])
    def test_astigmatic_curved_gaussian(self, xy, Rx, Ry):
        wx, wy = 40.0, 28.0
        r = ba.analyze_slices(gauss(*xy, wx, wy, Rx, Ry), *xy, K0)
        assert r.Rx[0] == pytest.approx(Rx, rel=0.01) and r.Ry[0] == pytest.approx(Ry, rel=0.01)
        assert r.M2x[0] == pytest.approx(1.0, abs=0.01) and r.M2y[0] == pytest.approx(1.0, abs=0.01)
        assert r.wx[0] == pytest.approx(wx, rel=1e-3) and r.wy[0] == pytest.approx(wy, rel=1e-3)
        assert r.w0x[0] == pytest.approx(waist(wx, Rx), rel=0.01)
        assert r.w0y[0] == pytest.approx(waist(wy, Ry), rel=0.01)

    def test_sign_convention_and_distance_to_waist(self, xy):
        # Positive R = diverging: the plane lies z0 = w² k0 /(2·...) past the waist.
        # For a Gaussian z = R / (1 + (R/z_R)^2)... easier: z0 from w and w0.
        w, R = 40.0, 3000.0
        r = ba.analyze_slices(gauss(*xy, w, w, R, -R), *xy, K0)
        w0 = waist(w, R)
        zR = K0 * w0 ** 2 / 2
        z_expected = math.sqrt(w ** 2 / w0 ** 2 - 1) * zR
        assert r.z0x[0] == pytest.approx(z_expected, rel=0.02)       # past the waist
        assert r.z0y[0] == pytest.approx(-z_expected, rel=0.02)      # before it

    def test_amplitude_and_global_phase_do_not_matter(self, xy):
        a = ba.analyze_slices(gauss(*xy, 33, 33, 4000, 4000), *xy, K0)
        b = ba.analyze_slices(7.5e-3 * np.exp(1.3j) * gauss(*xy, 33, 33, 4000, 4000), *xy, K0)
        for name in ("M2x", "wx", "Rx", "thx"):
            assert getattr(b, name)[0] == pytest.approx(getattr(a, name)[0], rel=1e-9, abs=1e-12)


class TestHermiteGauss:
    @staticmethod
    def hg(x, order, w0):
        from numpy.polynomial.hermite import hermval
        c = np.zeros(order + 1); c[order] = 1
        return hermval(math.sqrt(2) * x / w0, c) * np.exp(-(x / w0) ** 2)

    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    def test_m2_is_2n_plus_1(self, xy, order):
        x, y = xy
        E = self.hg(x, order, 30.0)[None, :] * np.exp(-(y[:, None] / 30.0) ** 2)
        r = ba.analyze_slices(E.astype(complex), x, y, K0)
        assert r.M2x[0] == pytest.approx(2 * order + 1, rel=0.01)
        assert r.M2y[0] == pytest.approx(1.0, abs=0.01)

    def test_m2_unchanged_by_curvature(self, xy):
        x, y = xy
        E = (self.hg(x, 2, 30.0)[None, :] * np.exp(-(y[:, None] / 30.0) ** 2)).astype(complex)
        E = E * np.exp(1j * K0 * (x[None, :] ** 2 / (2 * 3000.0) + y[:, None] ** 2 / (2 * -2000.0)))
        r = ba.analyze_slices(E, x, y, K0)
        assert r.M2x[0] == pytest.approx(5.0, rel=0.02) and r.M2y[0] == pytest.approx(1.0, abs=0.02)


class TestCentroidAndTilt:
    def test_shift_and_tilt(self, xy):
        # tilt on an exact FFT bin so there is no spectral leakage
        dk = 2 * math.pi / (N * D)
        kx, ky = 9 * dk, -5 * dk
        E = gauss(*xy, 40, 30, x0=17.0, y0=-23.0, kx=kx, ky=ky)
        r = ba.analyze_slices(E, *xy, K0)
        assert r.xbar[0] == pytest.approx(17.0, rel=1e-3) and r.ybar[0] == pytest.approx(-23.0, rel=1e-3)
        assert r.kxbar[0] == pytest.approx(kx, rel=1e-3) and r.kybar[0] == pytest.approx(ky, rel=1e-3)
        assert r.thx[0] == pytest.approx(kx / K0, rel=1e-3) and r.thy[0] == pytest.approx(ky / K0, rel=1e-3)
        # neither shift nor tilt change the quality or the width
        assert r.M2x[0] == pytest.approx(1.0, abs=0.01) and r.wx[0] == pytest.approx(40, rel=1e-3)

    def test_tilt_does_not_disturb_curvature(self, xy):
        dk = 2 * math.pi / (N * D)
        r = ba.analyze_slices(gauss(*xy, 40, 40, 5000, 5000, kx=6 * dk), *xy, K0)
        assert r.Rx[0] == pytest.approx(5000, rel=0.02) and r.M2x[0] == pytest.approx(1.0, abs=0.01)

    def test_air_referenced_values(self, xy):
        dk = 2 * math.pi / (N * D)
        r = ba.analyze_slices(gauss(*xy, 40, 40, 5000, -5000, kx=6 * dk), *xy, K0)
        a = ba.refer_to_air(r, IDX)
        assert a.Rx[0] == pytest.approx(r.Rx[0] / IDX) and a.Ry[0] == pytest.approx(r.Ry[0] / IDX)
        assert a.thx[0] == pytest.approx(r.thx[0] * IDX)
        assert a.M2x[0] == r.M2x[0] and a.wx[0] == r.wx[0] and a.xbar[0] == r.xbar[0]


class TestTimeDependence:
    def _pulse(self, xy, nt=9):
        env = np.exp(-(((np.arange(nt) - nt // 2) / 2.0) ** 2))
        return env[:, None, None] * gauss(*xy, 40, 30, 5000, -4000)[None], env

    def test_one_result_per_slice(self, xy):
        E, _ = self._pulse(xy)
        r = ba.analyze_slices(E, *xy, K0)
        assert len(r) == 9
        assert np.allclose(r.M2x, 1.0, atol=0.01) and np.allclose(r.wx, 40, rtol=1e-3)

    def test_power_threshold_skips_the_tails(self, xy):
        E, env = self._pulse(xy)
        r = ba.analyze_slices(E, *xy, K0, power_threshold=0.5)
        keep = env ** 2 >= 0.5
        assert np.array_equal(~np.isnan(r.M2x), keep) and keep.sum() < 9
        assert np.all(np.isnan(r.wx[~keep]))

    def test_dark_slice_is_skipped_not_an_error(self, xy):
        E, _ = self._pulse(xy)
        E[0] = 0
        r = ba.analyze_slices(E, *xy, K0)
        assert np.isnan(r.M2x[0]) and not np.isnan(r.M2x[4])

    def test_fluence_of_a_steady_beam_matches_a_slice(self, xy):
        E, _ = self._pulse(xy)
        f = ba.analyze_fluence(E, *xy, K0)
        s = ba.analyze_slices(E, *xy, K0)
        assert len(f) == 1
        for name in ("M2x", "M2y", "wx", "wy", "Rx", "Ry", "xbar", "thx"):
            assert getattr(f, name)[0] == pytest.approx(getattr(s, name)[4], rel=1e-6, abs=1e-9)

    def test_fluence_sees_a_wandering_beam(self, xy):
        # the centroid moves with time: each slice is a clean Gaussian, the
        # pulse as a whole is wider and has a worse M² in x
        E = np.stack([gauss(*xy, 30, 30, x0=x0) for x0 in (-30, 0, 30)])
        s = ba.analyze_slices(E, *xy, K0)
        f = ba.analyze_fluence(E, *xy, K0)
        assert np.allclose(s.M2x, 1.0, atol=0.01)
        assert f.wx[0] > 1.5 * s.wx[0] and f.xbar[0] == pytest.approx(0, abs=1e-6)

    def test_chunking_does_not_change_the_answer(self, xy, monkeypatch):
        E, _ = self._pulse(xy, nt=11)
        a = ba.analyze_fluence(E, *xy, K0)
        monkeypatch.setattr(ba, "_CHUNK", 3)
        b = ba.analyze_fluence(E, *xy, K0)
        for name in ("M2x", "wx", "Rx", "Ry"):
            assert getattr(b, name)[0] == pytest.approx(getattr(a, name)[0], rel=1e-9)


class TestRobustness:
    def test_noise_floor_option(self, xy):
        x, y = xy
        rng = np.random.default_rng(1)
        noise = 5e-2 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
        E = gauss(x, y, 25, 25) + noise
        raw = ba.analyze_slices(E, x, y, K0)
        cut = ba.analyze_slices(E, x, y, K0, noise_threshold=0.1)
        assert raw.wx[0] > 1.3 * 25                       # background swamps the width
        assert cut.wx[0] == pytest.approx(25, rel=0.15) and cut.M2x[0] < 0.5 * raw.M2x[0]
        # and the default really is "off"
        assert ba.analyze_slices(E, x, y, K0, noise_threshold=0.0).wx[0] == raw.wx[0]

    def test_noise_floor_in_fluence_mode(self, xy):
        x, y = xy
        rng = np.random.default_rng(2)
        E = np.stack([gauss(x, y, 25, 25) + 5e-2 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
                      for _ in range(4)])
        raw = ba.analyze_fluence(E, x, y, K0)
        cut = ba.analyze_fluence(E, x, y, K0, noise_threshold=0.1)
        assert raw.wx[0] > 1.3 * 25 and cut.wx[0] == pytest.approx(25, rel=0.15)

    def test_edge_warning(self, xy):
        ok = ba.analyze_slices(gauss(*xy, 40, 40), *xy, K0)
        wide = ba.analyze_slices(gauss(*xy, 140, 140), *xy, K0)
        assert not ok.edge_warning[0] and wide.edge_warning[0]

    def test_all_zero_field_gives_nan(self, xy):
        r = ba.analyze_slices(np.zeros((N, N), complex), *xy, K0)
        assert np.isnan(r.M2x[0]) and np.isnan(r.wx[0])
        assert np.isnan(ba.analyze_fluence(np.zeros((2, N, N), complex), *xy, K0).M2x[0])

    def test_2d_input_is_accepted(self, xy):
        assert len(ba.analyze_slices(gauss(*xy), *xy, K0)) == 1

    def test_grid_mismatch_is_reported(self, xy):
        with pytest.raises(ValueError, match="does not match"):
            ba.analyze_slices(np.ones((1, 8, 8), complex), *xy, K0)

    def test_k_axis_matches_fftshift(self):
        for n in (8, 9):
            k = ba._k_axis(n, 0.5)
            assert k[n // 2] == 0 and np.all(np.diff(k) > 0)


# ── A whole run directory ────────────────────────────────────────────────────

def _write_field(path, E):
    import h5py
    with h5py.File(path, "w") as f:
        f["real"] = E.real.astype(np.float32)
        f["imag"] = E.imag.astype(np.float32)


@pytest.fixture
def run_dir(tmp_path, xy):
    nt, lx = 5, N * D
    cfg = {"crystal": {"geometry": {"LX_mm": lx * 1e-3, "LY_mm": lx * 1e-3},
                       "degenerate": False,
                       "wavelengths": {"pump_um": 0.8, "signal_um": 1.5, "idler_um": 1.7},
                       "optics": {"np": 2.17, "ns": 2.13, "ni": 2.12}}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    import h5py
    with h5py.File(tmp_path / "time.h5", "w") as f:
        f["data"] = np.linspace(-1, 1, nt).astype(np.float32)
    for fname, w in (("pump_input_XY.h5", 40), ("pump_output.h5", 36),
                     ("signal_output.h5", 30), ("idler_output.h5", 28)):
        env = np.exp(-(((np.arange(nt) - nt // 2) / 1.5) ** 2))        # pulse in time
        _write_field(tmp_path / fname, env[:, None, None] * gauss(*xy, w, w, 8000, 8000)[None])
    return tmp_path


class TestRun:
    def test_analyzes_every_wave(self, run_dir):
        run = ba.analyze_run(str(run_dir))
        assert list(run.waves) == ["Input pump", "Pump", "Signal", "Idler"]
        assert run.t_ps[0] == pytest.approx(-1) and len(run.t_ps) == 5
        assert run.waves["Signal"].fluence.wx[0] == pytest.approx(30, rel=2e-3)
        assert run.waves["Idler"].slices.M2x[2] == pytest.approx(1.0, abs=0.02)

    def test_uses_each_waves_own_index_and_wavelength(self, run_dir):
        run = ba.analyze_run(str(run_dir))
        assert run.waves["Signal"].n == 2.13 and run.waves["Signal"].wavelength_um == 1.5
        # The test fields were built with k0 for n = 2.2; the analysis uses the
        # pump's own k0 (n = 2.17), and the radius read back scales as k0.
        assert run.waves["Pump"].fluence.Rx[0] == pytest.approx(8000 * 2.17 / IDX, rel=0.02)

    def test_degenerate_runs_have_no_idler(self, run_dir):
        cfg = json.loads((run_dir / "config.json").read_text())
        cfg["crystal"]["degenerate"] = True
        (run_dir / "config.json").write_text(json.dumps(cfg))
        assert "Idler" not in ba.analyze_run(str(run_dir)).waves

    def test_missing_files_are_skipped(self, run_dir):
        (run_dir / "pump_input_XY.h5").unlink()
        assert "Input pump" not in ba.analyze_run(str(run_dir)).waves

    def test_empty_directory_is_an_error(self, run_dir):
        for p in run_dir.glob("*.h5"):
            p.unlink()
        with pytest.raises(FileNotFoundError):
            ba.analyze_run(str(run_dir))

    def test_options_are_passed_through(self, run_dir):
        run = ba.analyze_run(str(run_dir), power_threshold=0.999)
        assert np.isnan(run.waves["Pump"].slices.M2x).sum() == 4      # only the peak survives

    def test_csv_fluence_and_slices(self, run_dir, tmp_path):
        run = ba.analyze_run(str(run_dir))
        p = tmp_path / "f.csv"
        ba.write_csv(str(p), run, "fluence")
        rows = list(csv.DictReader(open(p)))
        assert [r["wave"] for r in rows] == ["Input pump", "Pump", "Signal", "Idler"]
        assert float(rows[2]["w_x_um"]) == pytest.approx(30, rel=2e-3) and rows[2]["slice"] == "all"
        ba.write_csv(str(p), run, "slices")
        rows = list(csv.DictReader(open(p)))
        assert len(rows) == 4 * 5 and float(rows[1]["t_ps"]) == pytest.approx(-0.5)

    def test_csv_air_frame_converts_curvature_and_tilt(self, run_dir, tmp_path):
        run = ba.analyze_run(str(run_dir))
        a, b = tmp_path / "a.csv", tmp_path / "b.csv"
        ba.write_csv(str(a), run, "fluence", "crystal")
        ba.write_csv(str(b), run, "fluence", "air")
        ra, rb = list(csv.DictReader(open(a))), list(csv.DictReader(open(b)))
        n = run.waves["Pump"].n
        assert float(rb[1]["R_x_mm"]) == pytest.approx(float(ra[1]["R_x_mm"]) / n, rel=1e-6)
        assert rb[1]["frame"] == "air" and float(rb[1]["w_x_um"]) == float(ra[1]["w_x_um"])

    def test_csv_rejects_bad_options(self, run_dir, tmp_path):
        run = ba.analyze_run(str(run_dir))
        with pytest.raises(ValueError):
            ba.write_csv(str(tmp_path / "x.csv"), run, "bogus")


# ── Against the real engine ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine_beam(tmp_path_factory):
    """CPU engine on a 64×64 grid, CW (the focused-pump check needs a window
    larger than the beam but is cheap)."""
    from test_engine_cpu import _build_engine
    return _build_engine(tmp_path_factory, nt=1, nx=64, ny=64, nz=20)


@pytest.mark.engine
def test_engines_focused_pump_matches_the_gaussian_beam_formulas(engine_beam, tmp_path):
    """The engine's input pump is a focused Gaussian with its waist at the
    crystal centre. At the entry face the beam is converging (R < 0), at the
    exit face (linear propagation: the pump is weak) it is diverging by the
    same amount. Both have M² = 1 and the same width."""
    from test_engine_cpu import _params, _run
    exe_env = engine_beam
    w0, L_mm = 30.0, 5.0
    params = _params()
    params["pump"].update(power_W=1e-3, waist_um=w0, mode="focused-cw")
    params["length_mm"] = L_mm
    r, out = _run(exe_env, tmp_path, params)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    shutil.copy(tmp_path / "config.json", out / "config.json")      # the GUI runner does this too

    run = ba.analyze_run(str(out))
    pump_in, pump_out = run.waves["Input pump"].slices, run.waves["Pump"].slices
    n, lam = run.waves["Pump"].n, run.waves["Pump"].wavelength_um
    zR = math.pi * n * w0 ** 2 / lam
    fp = L_mm * 500.0                                # focus at the crystal centre
    w_expected = w0 * math.sqrt(1 + (fp / zR) ** 2)
    R_expected = fp + zR ** 2 / fp

    for res, sign in ((pump_in, -1), (pump_out, +1)):
        assert res.M2x[0] == pytest.approx(1.0, abs=0.02) and res.M2y[0] == pytest.approx(1.0, abs=0.02)
        assert res.wx[0] == pytest.approx(w_expected, rel=0.01)
        assert res.wy[0] == pytest.approx(w_expected, rel=0.01)
        assert res.w0x[0] == pytest.approx(w0, rel=0.02)
        assert res.Rx[0] == pytest.approx(sign * R_expected, rel=0.03)
        assert res.Ry[0] == pytest.approx(sign * R_expected, rel=0.03)
        assert res.z0x[0] == pytest.approx(sign * fp, rel=0.03)    # ±fp from the waist
        assert abs(res.xbar[0]) < 0.5 and abs(res.thx[0]) < 1e-6        # on axis, no tilt
        assert not res.edge_warning[0]
