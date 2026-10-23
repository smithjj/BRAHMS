"""
Preflight estimates from a built engine config: the peak input irradiance of
each wave and the crystal's characteristic irradiance S0.

    S0 = eps0 c^3 n1 n2 n3 / (2 deff^2 w1 w2 L^2)
       = eps0 c   n1 n2 n3 lam1 lam2 / (8 pi^2 deff^2 L^2)

with waves 1 and 2 the two redder waves (for SHG both are the fundamental)
and wave 3 the bluest. The plain refractive indices are used (no beam-tilt
correction), and for QPM the effective coefficient is (2/pi)*deff, the same
factor the engines apply.
"""

import math

_EPS0_SI = 8.8541878128e-12      # F/m
_C_SI    = 299792458.0           # m/s
_UM2_PER_CM2 = 1e8               # 1 cm^2 = 1e8 um^2


def irradiance_scale(i_max: float):
    """(scale_factor, unit_string) so that i_max * scale_factor is of order 1-1000."""
    for threshold, scale, unit in [
        (1e12, 1e-12, "TW/cm²"),
        (1e9,  1e-9,  "GW/cm²"),
        (1e6,  1e-6,  "MW/cm²"),
        (1e3,  1e-3,  "kW/cm²"),
        (1.0,  1.0,   "W/cm²"),
        (1e-3, 1e3,   "mW/cm²"),
    ]:
        if i_max >= threshold:
            return scale, unit
    return 1e6, "μW/cm²"


def format_irradiance(i_W_cm2) -> str:
    if i_W_cm2 is None:
        return "—"
    if i_W_cm2 == 0:
        return "0 W/cm²"
    scale, unit = irradiance_scale(i_W_cm2)
    return f"{i_W_cm2 * scale:.3g} {unit}"


def characteristic_irradiance_W_cm2(cfg: dict):
    """S0 [W/cm²] for the config's crystal, or None if deff or L is zero."""
    cr  = cfg["crystal"]
    wl  = cr["wavelengths"]
    opt = cr["optics"]

    d = float(cr["nonlinear"]["deff_pm_V"]) * 1e-12            # m/V
    if "qpm" in cr:
        d *= 2.0 / math.pi
    L = float(cr["geometry"]["Lcr_mm"]) * 1e-3                 # m
    if d == 0 or L == 0:
        return None

    if cr["process"] == "SHG":
        n1 = n2 = opt["np"]; n3 = opt["ns"]
        lam1 = lam2 = wl["pump_um"]
    else:
        n1, n2, n3 = opt["ns"], opt["ni"], opt["np"]
        lam1, lam2 = wl["signal_um"], wl["idler_um"]

    s0 = (_EPS0_SI * _C_SI * n1 * n2 * n3 * (lam1 * 1e-6) * (lam2 * 1e-6)
          / (8.0 * math.pi ** 2 * d ** 2 * L ** 2))            # W/m²
    return s0 * 1e-4


def peak_irradiance_W_cm2(power_W: float, waist_um: float) -> float:
    """Peak irradiance 2P/(pi w^2) of a Gaussian beam at its waist [W/cm²]."""
    return 2.0 * power_W / (math.pi * waist_um ** 2) * _UM2_PER_CM2


def preflight(cfg: dict) -> dict:
    """Peak input irradiance of each wave and S0 for a built config.

    Returns {"S0_W_cm2": float | None,
             "waves": {"pump"|"signal"|"idler": {"lambda_um", "peak_W_cm2", "ratio"}}}.
    A vacuum-noise field has peak_W_cm2 = None (no defined input irradiance);
    ratio is peak / S0, or None when either is unavailable.
    """
    s0 = characteristic_irradiance_W_cm2(cfg)
    wl = cfg["crystal"]["wavelengths"]
    f  = cfg["fields"]

    waves = {}
    for name in ("pump", "signal", "idler"):
        blk = f[name]
        peak = None
        if not (name != "pump" and blk.get("noise", False)):
            peak = peak_irradiance_W_cm2(blk["power_W"], blk["waist_um"])
        waves[name] = {
            "lambda_um":  wl[f"{name}_um"],
            "peak_W_cm2": peak,
            "ratio":      (peak / s0) if (peak is not None and s0) else None,
        }
    return {"S0_W_cm2": s0, "waves": waves}


def preflight_summary(cfg: dict) -> str:
    """Log lines for the run log."""
    pf = preflight(cfg)
    lines = [f"  S₀      : {format_irradiance(pf['S0_W_cm2'])}  (characteristic irradiance)"]
    for name, w in pf["waves"].items():
        if name == "idler" and cfg["crystal"]["degenerate"]:
            continue
        if w["peak_W_cm2"] is None:
            continue
        r = "—" if w["ratio"] is None else f"{w['ratio']:.3g}"
        lines.append(f"  I_{name:<6s}: {format_irradiance(w['peak_W_cm2'])}  (I/S₀ = {r})")
    return "\n".join(lines)
