"""End-to-end smoke tests of the C++/OpenMP engine. Build the engine in a
scratch copy (so the repo is untouched) on a tiny grid and run it. Skipped
when the toolchain (g++, make, FFTW3, HDF5) isn't available."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gui.core_py.config_builder import build_config
from gui.core_py.engine_runner import EXE_SUFFIX, mingw_bin_dir

pytestmark = pytest.mark.engine

ROOT = Path(__file__).resolve().parent.parent
MAKE = "mingw32-make" if sys.platform == "win32" else "make"


def _toolchain_env():
    env = os.environ.copy()
    mingw = mingw_bin_dir()
    if mingw:
        env["PATH"] = mingw + os.pathsep + env["PATH"]
    return env


def _build_engine(tmp_path_factory, nt):
    env = _toolchain_env()
    if not (shutil.which("g++", path=env["PATH"]) and shutil.which(MAKE, path=env["PATH"])):
        pytest.skip("g++ / make not available")
    work = tmp_path_factory.mktemp("engine_omp")
    shutil.copytree(ROOT / "engine_omp", work / "engine_omp",
                    ignore=shutil.ignore_patterns("twm_cpu", "twm_cpu.exe", "*.o"))
    r = subprocess.run([MAKE, "-C", str(work / "engine_omp"),
                        "NX=8", "NY=8", "NZ=20", f"NT={nt}"],
                       env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    # A toolchain is present, so a build failure is a real regression (or a
    # missing FFTW/HDF5 dev package), not something to skip silently.
    assert r.returncode == 0, f"CPU engine build failed:\n{r.stdout[-1500:]}{r.stderr[-1500:]}"
    exe = work / "engine_omp" / ("twm_cpu" + EXE_SUFFIX)
    assert exe.is_file()
    return exe, env


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    return _build_engine(tmp_path_factory, nt=1)


@pytest.fixture(scope="module")
def engine_pulsed(tmp_path_factory):
    return _build_engine(tmp_path_factory, nt=128)


def _params(**signal):
    p = {
        "process": "OPG", "crystal": "MgO:PPLN",
        "temperature": 27.0, "grating_um": 6.99, "length_mm": 5.0,
        "lx_mm": 0.5, "ly_mm": 0.5,
        "pump": {"lambda_um": 0.8, "power_W": 1.0, "waist_um": 30.0,
                 "fwhm_ps": 0.2, "mode": "focused-cw"},
        "signal": {"lambda_um": 1.5, "excitation": "noise", **signal},
        "idler": {"lambda_um": 1.5, "degenerate": False, "excitation": "noise"},
    }
    return p


def _run(engine, tmp_path, params):
    exe, env = engine
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(build_config(params)))
    out = tmp_path / "out"
    out.mkdir()
    env = {**env, "OMP_NUM_THREADS": "2"}
    r = subprocess.run([str(exe), str(cfg), str(out) + os.sep],
                       env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    return r, out


EXPECTED_OUTPUTS = {"pump_output.h5", "signal_output.h5", "idler_output.h5"}


def test_noise_seeded_run_succeeds_and_writes_outputs(engine, tmp_path):
    r, out = _run(engine, tmp_path, _params())
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert EXPECTED_OUTPUTS <= {p.name for p in out.iterdir()}


def test_injected_signal_run_succeeds(engine, tmp_path):
    r, out = _run(engine, tmp_path,
                  _params(excitation="inject", power_W=0.01, mode="focused-cw"))
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert EXPECTED_OUTPUTS <= {p.name for p in out.iterdir()}


def _pump_input_intensity(out):
    """(t [ps], on-axis |A_pump_in|^2) from the saved input pump field."""
    h5py = pytest.importorskip("h5py")
    import numpy as np
    with h5py.File(out / "time.h5") as f:
        t = np.asarray(f["data"]).ravel()
    with h5py.File(out / "pump_input_XY.h5") as f:
        a = np.asarray(f["real"]) + 1j * np.asarray(f["imag"])
    a = a.reshape(len(t), -1)                       # [NT, NY*NX]
    return t, (np.abs(a) ** 2).max(axis=1)           # peak over the transverse plane


def _fwhm(t, y):
    import numpy as np
    above = np.where(y >= y.max() / 2)[0]
    i0, i1 = above[0], above[-1]
    def cross(i, j):                                # linear interpolation of the half-max crossing
        return t[i] + (y.max() / 2 - y[i]) * (t[j] - t[i]) / (y[j] - y[i])
    return cross(i1, i1 + 1) - cross(i0 - 1, i0)


def test_pulse_fwhm_is_the_intensity_fwhm(engine_pulsed, tmp_path):
    params = _params()
    params["pump"].update(mode="waveplane-pulsed", fwhm_ps=0.2)
    params["t_window_ps"] = 1.6
    r, out = _run(engine_pulsed, tmp_path, params)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    t, inten = _pump_input_intensity(out)
    assert _fwhm(t, inten) == pytest.approx(0.2, rel=0.03)


def test_bad_config_path_fails_cleanly(engine, tmp_path):
    exe, env = engine
    r = subprocess.run([str(exe), str(tmp_path / "missing.json"), str(tmp_path) + os.sep],
                       env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert r.returncode != 0
