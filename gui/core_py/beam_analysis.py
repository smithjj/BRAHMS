"""
Beam analysis of the simulated output fields.

Second-moment beam quantities following Chapter 7.7 of A. V. Smith, "Crystal
Nonlinear Optics: with SNLO Examples" (pp. 299-303). This is a port of the
Msquared_class MATLAB code (``calculate`` and ``calculate_pulse_flat``):

    centroid          x̄, ȳ                      (Eq. 7.101)
    width             wx, wy  (4σ, 1/e² radius for a Gaussian)  (Eq. 7.100)
    angular centroid  k̄x, k̄y  → tilt θ = k̄/k    (Eq. 7.102)
    angular width     wkx, wky                   (Eq. 7.103)
    radius of curv.   Rx, Ry                     (Eqs. 7.107, 7.108)
    waist             w0x, w0y                   (Eq. 7.110)
    beam quality      M²x, M²y = w0·wk/2         (Eq. 7.111)

The curvature is removed three times and the radii added harmonically, which
copes with the large, non-spherical curvatures of tightly focused beams.

Conventions
-----------
* Fields are the engine's complex envelopes, shape ``[NT, NY, NX]`` (the x
  index varies fastest). Lengths are μm, wave numbers rad/μm, angles rad.
* The transverse grid is the engine's: ``x_i = (i - NX/2)·dx`` with
  ``dx = LX/NX``.
* ``k0`` is the wave number *in the medium*, 2π·n/λ_vacuum, so R, tilt and
  z0 are the values inside the crystal. ``refer_to_air`` converts them for a
  flat exit face.
* **R > 0 is a diverging beam, R < 0 a converging one** for the engine's
  phase convention (checked against the engine's focused Gaussian in the
  tests). A positive k̄ tilts toward +x.
* A plane/flat-phase beam has R = ±inf (reported as ``inf``).
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, fields, replace

import numpy as np

N_REMOVALS = 3                 # curvature-removal passes (as in the MATLAB code)
EDGE_WARN_FRACTION = 1e-3      # warn if more energy than this sits at the window edge
_CHUNK = 16                    # time slices handled at once (bounds memory)


def centered_axis(n: int, d: float) -> np.ndarray:
    """Engine transverse coordinates [μm]: x_i = (i - n/2)·d."""
    return (np.arange(n) - n / 2) * d


def _k_axis(n: int, d: float) -> np.ndarray:
    """Angular spatial frequencies [rad/μm] in fftshift order."""
    return 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, d))


# ── Results ──────────────────────────────────────────────────────────────────

@dataclass
class BeamResult:
    """Beam quantities, one entry per analysed slice (length 1 for the
    fluence-integrated analysis). Slices that were skipped hold NaN."""
    M2x: np.ndarray
    M2y: np.ndarray
    wx: np.ndarray            # second-moment width [μm]
    wy: np.ndarray
    w0x: np.ndarray           # extrapolated waist width [μm]
    w0y: np.ndarray
    xbar: np.ndarray          # centroid [μm]
    ybar: np.ndarray
    Rx: np.ndarray            # radius of curvature [μm]; <0 converging, ±inf flat
    Ry: np.ndarray
    kxbar: np.ndarray         # angular centroid [rad/μm]
    kybar: np.ndarray
    wkx: np.ndarray           # angular width [rad/μm]
    wky: np.ndarray
    z0x: np.ndarray           # signed distance of the plane from the waist [μm]
    z0y: np.ndarray
    thx: np.ndarray           # tilt kbar / k0 [rad]
    thy: np.ndarray
    edge_fraction: np.ndarray  # fraction of the energy in the outer border of the window

    def __len__(self) -> int:
        return len(self.M2x)

    @property
    def edge_warning(self) -> np.ndarray:
        """True where the beam reaches the window edge (periodic FFT grid:
        the moments are then unreliable)."""
        with np.errstate(invalid="ignore"):
            return self.edge_fraction > EDGE_WARN_FRACTION

    def row(self, i: int = 0) -> dict:
        return {f.name: float(getattr(self, f.name)[i]) for f in fields(self)}


def _empty(n: int) -> BeamResult:
    return BeamResult(**{f.name: np.full(n, np.nan) for f in fields(BeamResult)})


def refer_to_air(res: BeamResult, n: float) -> BeamResult:
    """Values seen after a flat exit face into air: a wavefront of radius R in
    a medium of index n has radius R/n in air, a ray at angle θ leaves at
    ≈ n·θ (paraxial), and the apparent distance to the waist shrinks by n.
    Widths, centroids and M² are unchanged."""
    return replace(res, Rx=res.Rx / n, Ry=res.Ry / n,
                   thx=res.thx * n, thy=res.thy * n,
                   kxbar=res.kxbar * n, kybar=res.kybar * n,
                   z0x=res.z0x / n, z0y=res.z0y / n)


# ── Core calculation ─────────────────────────────────────────────────────────

def _analyze_stack(E: np.ndarray, x: np.ndarray, y: np.ndarray, k0: float) -> dict:
    """Moments of the time-integrated (fluence) beam of E[T, NY, NX].
    With T = 1 this is the instantaneous per-slice analysis."""
    T, NY, NX = E.shape
    dx, dy = x[1] - x[0], y[1] - y[0]
    X = x[None, None, :]
    Y = y[None, :, None]
    kx = _k_axis(NX, dx)[None, :]
    ky = _k_axis(NY, dy)[:, None]

    E = np.array(E, dtype=np.complex128)                 # work on a copy
    chunks = [slice(i, i + _CHUNK) for i in range(0, T, _CHUNK)]

    Fxy = np.zeros((NY, NX))
    Fk = np.zeros((NY, NX))
    for c in chunks:
        Fxy += (E[c].real ** 2 + E[c].imag ** 2).sum(axis=0)
        Ek = np.fft.fftshift(np.fft.fft2(E[c], axes=(1, 2)), axes=(1, 2))
        Fk += (Ek.real ** 2 + Ek.imag ** 2).sum(axis=0)

    S, Sk = Fxy.sum(), Fk.sum()
    if not (S > 0 and np.isfinite(S)):
        return {f.name: np.nan for f in fields(BeamResult)}

    xbar = (Fxy * x[None, :]).sum() / S                                   # Eq. 7.101
    ybar = (Fxy * y[:, None]).sum() / S
    wx2 = 4 * (Fxy * (x[None, :] - xbar) ** 2).sum() / S                  # Eq. 7.100
    wy2 = 4 * (Fxy * (y[:, None] - ybar) ** 2).sum() / S
    kxbar = (Fk * kx).sum() / Sk                                          # Eq. 7.102
    kybar = (Fk * ky).sum() / Sk
    wkx2 = 4 * (Fk * (kx - kxbar) ** 2).sum() / Sk                        # Eq. 7.103
    wky2 = 4 * (Fk * (ky - kybar) ** 2).sum() / Sk

    b = max(2, NX // 16)
    by = max(2, NY // 16)
    edge = 1.0 - Fxy[by:-by, b:-b].sum() / S

    # Repeatedly remove the (spherical) curvature; the radii add as 1/R.
    invRx = invRy = 0.0
    if wx2 > 0 and wy2 > 0:
        for _ in range(N_REMOVALS):
            sx = sy = 0.0
            for c in chunks:
                e = E[c]
                dEdx = (np.roll(e, -1, axis=2) - np.roll(e, 1, axis=2)) / (2 * dx)
                dEdx[:, :, [0, -1]] = 0           # roll wraps around; there is ~no light here
                dEdy = (np.roll(e, -1, axis=1) - np.roll(e, 1, axis=1)) / (2 * dy)
                dEdy[:, [0, -1], :] = 0
                # i/k0 · Σ (x-x̄)(E E*' − c.c.) = −(2/k0) Σ (x-x̄) Im(E E*')   (Eq. 7.108)
                sx += (np.imag(e * np.conj(dEdx)) * (X - xbar)).sum()
                sy += (np.imag(e * np.conj(dEdy)) * (Y - ybar)).sum()
            Ax = -2.0 / k0 * sx / S
            Ay = -2.0 / k0 * sy / S
            ix, iy = 2 * Ax / wx2, 2 * Ay / wy2   # 1/R = 2A/w²   (Eq. 7.107)
            invRx += ix
            invRy += iy
            phase_x = np.exp(-0.5j * k0 * ix * (x - xbar) ** 2)
            phase_y = np.exp(-0.5j * k0 * iy * (y - ybar) ** 2)
            E *= phase_y[None, :, None] * phase_x[None, None, :]

    def waist_and_m2(w2, wk2, invR):
        w02 = w2 - w2 ** 2 * k0 ** 2 * invR ** 2 / wk2                    # Eq. 7.110
        w0 = np.sqrt(w02) if w02 >= 0 else np.nan
        return w0, w0 * np.sqrt(wk2) / 2                                  # Eq. 7.111

    w0x, M2x = waist_and_m2(wx2, wkx2, invRx)
    w0y, M2y = waist_and_m2(wy2, wky2, invRy)

    def radius(invR):
        return 1.0 / invR if invR != 0 else np.inf

    return dict(
        M2x=M2x, M2y=M2y, wx=np.sqrt(wx2), wy=np.sqrt(wy2), w0x=w0x, w0y=w0y,
        xbar=xbar, ybar=ybar, Rx=radius(invRx), Ry=radius(invRy),
        kxbar=kxbar, kybar=kybar, wkx=np.sqrt(wkx2), wky=np.sqrt(wky2),
        z0x=k0 ** 2 * wx2 * invRx / wkx2, z0y=k0 ** 2 * wy2 * invRy / wky2,
        thx=kxbar / k0, thy=kybar / k0, edge_fraction=edge)


def _apply_noise_floor(E: np.ndarray, fraction: float) -> np.ndarray:
    """Zero the pixels whose irradiance is below `fraction` of the peak."""
    if fraction <= 0:
        return E
    I = np.abs(E) ** 2
    return np.where(I >= fraction * I.max(), E, 0)


def _as_stack(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E)
    return E[None] if E.ndim == 2 else E


def _check_grid(E: np.ndarray, x, y):
    if E.ndim != 3 or E.shape[1] != len(y) or E.shape[2] != len(x):
        raise ValueError(f"field shape {E.shape} does not match grid ({len(y)} y × {len(x)} x)")


def analyze_slices(E: np.ndarray, x: np.ndarray, y: np.ndarray, k0: float,
                   power_threshold: float = 0.0, noise_threshold: float = 0.0) -> BeamResult:
    """Analyse every time slice of E[NT, NY, NX] on its own.

    power_threshold: skip (NaN) slices whose power is below this fraction of
        the peak power; the tails of a pulse carry no meaningful beam.
    noise_threshold: zero pixels below this fraction of the slice's peak
        irradiance before analysing (0 = off, like the MATLAB code).
    """
    E = _as_stack(E)
    _check_grid(E, x, y)
    power = (np.abs(E) ** 2).sum(axis=(1, 2))
    pmax = power.max() if power.size else 0.0
    out = _empty(E.shape[0])
    for i in range(E.shape[0]):
        if not (pmax > 0 and power[i] > 0 and power[i] >= power_threshold * pmax):
            continue
        r = _analyze_stack(_apply_noise_floor(E[i:i + 1], noise_threshold), x, y, k0)
        for name, v in r.items():
            getattr(out, name)[i] = v
    return out


def analyze_fluence(E: np.ndarray, x: np.ndarray, y: np.ndarray, k0: float,
                    noise_threshold: float = 0.0) -> BeamResult:
    """Analyse the pulse as a whole, from its time-integrated fluence
    (MATLAB ``calculate_pulse_flat``). One entry in the result.

    noise_threshold: zero pixels whose *fluence* is below this fraction of
        the peak fluence, at every time (0 = off).
    """
    E = _as_stack(E)
    _check_grid(E, x, y)
    if noise_threshold > 0:
        F = (np.abs(E) ** 2).sum(axis=0)
        E = np.where((F >= noise_threshold * F.max())[None], E, 0)
    out = _empty(1)
    for name, v in _analyze_stack(E, x, y, k0).items():
        getattr(out, name)[0] = v
    return out


# ── A whole simulation run ───────────────────────────────────────────────────

# (display name, file, wavelength key, index key)
WAVE_FILES = [
    ("Input pump", "pump_input_XY.h5", "pump_um",   "np"),
    ("Pump",       "pump_output.h5",   "pump_um",   "np"),
    ("Signal",     "signal_output.h5", "signal_um", "ns"),
    ("Idler",      "idler_output.h5",  "idler_um",  "ni"),
]


@dataclass
class WaveBeam:
    name: str
    n: float                   # refractive index of the wave
    wavelength_um: float       # vacuum wavelength
    slices: BeamResult         # one entry per time slice
    fluence: BeamResult        # one entry, whole pulse


@dataclass
class RunBeamAnalysis:
    t_ps: np.ndarray           # time of each slice (index if unknown)
    waves: dict                # name -> WaveBeam, in WAVE_FILES order


def load_complex(path: str) -> np.ndarray:
    """[NT, NY, NX] complex128 from an engine HDF5 file (datasets real/imag)."""
    import h5py
    with h5py.File(path, "r") as f:
        return f["real"][()].astype(np.float64) + 1j * f["imag"][()].astype(np.float64)


def load_vector(path: str) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        return f["data"][()].astype(np.float64).ravel()


def analyze_run(out_dir: str, power_threshold: float = 0.0, noise_threshold: float = 0.0,
                waves: "list[str] | None" = None) -> RunBeamAnalysis:
    """Analyse the field files in an engine output directory (needs
    ``config.json`` for the window size, indices and wavelengths). The idler
    is skipped for degenerate processes (SHG / degenerate OPG), where it is
    the same wave as the signal."""
    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    cr = cfg["crystal"]
    lx_um = cr["geometry"]["LX_mm"] * 1e3
    ly_um = cr["geometry"]["LY_mm"] * 1e3
    degenerate = bool(cr.get("degenerate", False))

    t_path = os.path.join(out_dir, "time.h5")
    t_ps = load_vector(t_path) if os.path.isfile(t_path) else None

    result: dict = {}
    for name, fname, wl_key, n_key in WAVE_FILES:
        if waves is not None and name not in waves:
            continue
        if name == "Idler" and degenerate:
            continue
        path = os.path.join(out_dir, fname)
        if not os.path.isfile(path):
            continue
        E = load_complex(path)
        nt, ny, nx = E.shape
        x = centered_axis(nx, lx_um / nx)
        y = centered_axis(ny, ly_um / ny)
        wl = cr["wavelengths"][wl_key]
        n = cr["optics"][n_key]
        k0 = 2 * np.pi * n / wl
        result[name] = WaveBeam(
            name=name, n=n, wavelength_um=wl,
            slices=analyze_slices(E, x, y, k0, power_threshold, noise_threshold),
            fluence=analyze_fluence(E, x, y, k0, noise_threshold))
        nt_found = nt
    if not result:
        raise FileNotFoundError(f"no field files found in {out_dir}")
    if t_ps is None or len(t_ps) != nt_found:
        t_ps = np.arange(nt_found, dtype=float)
    return RunBeamAnalysis(t_ps=t_ps, waves=result)


# ── CSV export ───────────────────────────────────────────────────────────────

CSV_COLUMNS = [            # (field, header, factor from internal units)
    ("M2x", "M2_x", 1), ("M2y", "M2_y", 1),
    ("wx", "w_x_um", 1), ("wy", "w_y_um", 1),
    ("w0x", "w0_x_um", 1), ("w0y", "w0_y_um", 1),
    ("xbar", "x_centroid_um", 1), ("ybar", "y_centroid_um", 1),
    ("Rx", "R_x_mm", 1e-3), ("Ry", "R_y_mm", 1e-3),
    ("thx", "tilt_x_mrad", 1e3), ("thy", "tilt_y_mrad", 1e3),
    ("wkx", "wk_x_rad_per_um", 1), ("wky", "wk_y_rad_per_um", 1),
    ("z0x", "z_from_waist_x_um", 1), ("z0y", "z_from_waist_y_um", 1),
    ("edge_fraction", "edge_energy_fraction", 1),
]


def write_csv(path: str, run: RunBeamAnalysis, mode: str = "fluence", frame: str = "crystal"):
    """Write the analysis to CSV. mode: "fluence" (one row per wave) or
    "slices" (one row per wave and time slice). frame: "crystal" or "air"."""
    if mode not in ("fluence", "slices") or frame not in ("crystal", "air"):
        raise ValueError("mode must be 'fluence' or 'slices'; frame 'crystal' or 'air'")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wave", "slice", "t_ps", "frame"] + [h for _, h, _ in CSV_COLUMNS])
        for wave in run.waves.values():
            res = wave.fluence if mode == "fluence" else wave.slices
            if frame == "air":
                res = refer_to_air(res, wave.n)
            for i in range(len(res)):
                t = "" if mode == "fluence" else f"{run.t_ps[i]:.6g}"
                idx = "all" if mode == "fluence" else i
                w.writerow([wave.name, idx, t, frame] +
                           [f"{getattr(res, name)[i] * fac:.8g}" for name, _, fac in CSV_COLUMNS])
