from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QAbstractButton, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QLabel,
    QSpinBox,
)

from ..core_py.config_builder import build_config
from ..core_py.preflight import format_irradiance, preflight

_ROWS = (("pump", "Pump"), ("signal", "Signal"), ("idler", "Idler"))

_TOOLTIP = (
    "Peak irradiance of each input wave at its beam waist, 2P/(πw²), with P the peak power.\n"
    "Vacuum-noise fields have no defined input irradiance.\n\n"
    "Characteristic irradiance of the crystal:\n"
    "  S₀ = ε₀c·n₁n₂n₃·λ₁λ₂ / (8π²·d_eff²·L²)\n"
    "with waves 1, 2 the two redder waves (both the fundamental for SHG).\n"
    "Plain refractive indices are used (no beam-tilt correction); for QPM,\n"
    "d_eff includes the 2/π factor. Estimates ignore depletion and absorption."
)


class PreflightPanel(QGroupBox):
    """Live estimate of each wave's peak input irradiance and the crystal's
    characteristic irradiance S₀, from the current input parameters."""

    def __init__(self, parent=None):
        super().__init__("Preflight", parent)
        self.setToolTip(_TOOLTIP)
        g = QGridLayout(self)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(3)

        for col, text in enumerate(("", "λ (μm)", "Peak irradiance", "I / S₀")):
            h = QLabel(text)
            h.setStyleSheet("color: gray;")
            g.addWidget(h, 0, col)

        self._cells = {}
        for r, (key, label) in enumerate(_ROWS, start=1):
            cells = [QLabel(label), QLabel("—"), QLabel("—"), QLabel("—")]
            for c, w in enumerate(cells):
                if c:
                    w.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                g.addWidget(w, r, c)
            self._cells[key] = cells

        self.lbl_s0 = QLabel("S₀ = —")
        self.lbl_s0.setToolTip(_TOOLTIP)
        g.addWidget(self.lbl_s0, len(_ROWS) + 1, 0, 1, 4)

        self.lbl_msg = QLabel()
        self.lbl_msg.setWordWrap(True)
        self.lbl_msg.setStyleSheet("color: #ff9800;")
        self.lbl_msg.hide()
        g.addWidget(self.lbl_msg, len(_ROWS) + 2, 0, 1, 4)

        self._params_fn = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.refresh)

    # ── display ────────────────────────────────────────────────────────
    def set_cfg(self, cfg: dict):
        pf = preflight(cfg)
        self.lbl_msg.hide()
        s0 = pf["S0_W_cm2"]
        self.lbl_s0.setText(f"S₀ = {format_irradiance(s0)}")
        degenerate = cfg["crystal"]["degenerate"]
        for key, cells in self._cells.items():
            w = pf["waves"][key]
            cells[1].setText(f"{w['lambda_um']:.4g}")
            cells[2].setText(format_irradiance(w["peak_W_cm2"]))
            cells[3].setText("—" if w["ratio"] is None else f"{w['ratio']:.3g}")
            hide = key == "idler" and degenerate
            for c in cells:
                c.setVisible(not hide)

    def set_error(self, msg: str):
        self.lbl_s0.setText("S₀ = —")
        for cells in self._cells.values():
            for c in cells[1:]:
                c.setText("—")
        self.lbl_msg.setText(msg)
        self.lbl_msg.show()

    # ── live updating ──────────────────────────────────────────────────
    def watch(self, root, params_fn):
        """Recompute (debounced) whenever an input widget under `root` changes.
        `params_fn()` returns the tab's params dict for build_config()."""
        self._params_fn = params_fn
        for sb in root.findChildren((QDoubleSpinBox, QSpinBox)):
            sb.valueChanged.connect(self._timer.start)
        for cb in root.findChildren(QComboBox):
            cb.currentIndexChanged.connect(self._timer.start)
        for b in root.findChildren(QAbstractButton):
            b.toggled.connect(self._timer.start)
        self.refresh()

    def refresh(self):
        if self._params_fn is None:
            return
        try:
            self.set_cfg(build_config(self._params_fn()))
        except Exception as exc:        # incomplete/invalid inputs: show why, don't crash
            self.set_error(str(exc))
