"""
Beam Analysis tab — beam quality and geometry of the simulated output beams.

For the input pump and each output wave: M², second-moment widths, radius of
curvature, centroid and tilt (see core_py/beam_analysis.py, after A. V. Smith,
"Crystal Nonlinear Optics", ch. 7.7).

Layout:
  top     — output directory, analysis options, CSV export
  middle  — table (one row per wave) for the whole pulse or one time slice,
            with a time slider / Play button for the single-slice view
  bottom  — one quantity versus time for all waves, with the selected time marked

Reads pump_input_XY.h5, pump/signal/idler_output.h5, time.h5 and config.json
from the engine output directory.
"""

from __future__ import annotations

import math
import os

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDoubleSpinBox,
    QFileDialog, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSlider, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core_py import beam_analysis as ba
from ..widgets.plot_canvas import PlotCanvas

_WAVE_COLORS = {
    "Input pump": "#ffd54f",
    "Pump":       "#ff6d00",
    "Signal":     "#00e676",
    "Idler":      "#40c4ff",
}

# (header, tooltip, BeamResult field, display factor, format)
_COLUMNS = [
    ("M²x",      "Beam quality in x (1 = diffraction-limited Gaussian)", "M2x", 1.0, "{:.3f}"),
    ("M²y",      "Beam quality in y", "M2y", 1.0, "{:.3f}"),
    ("wx (μm)",  "Second-moment width in x (4σ; the 1/e² radius for a Gaussian)", "wx", 1.0, "{:.4g}"),
    ("wy (μm)",  "Second-moment width in y", "wy", 1.0, "{:.4g}"),
    ("Rx (mm)",  "Radius of curvature in x. Negative = converging, positive = diverging", "Rx", 1e-3, "{:.4g}"),
    ("Ry (mm)",  "Radius of curvature in y", "Ry", 1e-3, "{:.4g}"),
    ("x̄ (μm)",   "Centroid in x", "xbar", 1.0, "{:.3f}"),
    ("ȳ (μm)",   "Centroid in y", "ybar", 1.0, "{:.3f}"),
    ("θx (mrad)", "Tilt (propagation angle) in x", "thx", 1e3, "{:.3f}"),
    ("θy (mrad)", "Tilt (propagation angle) in y", "thy", 1e3, "{:.3f}"),
    ("Edge energy", "Fraction of the energy in the outer border of the simulation "
                    "window. Above 0.1% the moments are unreliable", "edge_fraction", 1.0, "{:.1e}"),
]

# quantity plotted versus time: (label, y-axis label, x field, y field, factor)
_PLOTS = [
    ("Beam quality M²",     "M²",                      "M2x", "M2y", 1.0),
    ("Second-moment width", "w (μm)",                  "wx",  "wy",  1.0),
    ("Wavefront curvature", "curvature 1/R (m⁻¹)",     "Rx",  "Ry",  None),
    ("Centroid",            "centroid (μm)",           "xbar", "ybar", 1.0),
    ("Tilt",                "tilt (mrad)",             "thx", "thy", 1e3),
]


class BeamAnalysisTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._out_dir: str | None = None
        self._run: ba.RunBeamAnalysis | None = None
        self._stale = False                     # a directory is set but not analysed yet

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── Output directory ───────────────────────────────────────────
        row = QHBoxLayout()
        self.le_dir = QLineEdit()
        self.le_dir.setPlaceholderText("Output directory…")
        self.le_dir.setReadOnly(True)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        row.addWidget(self.le_dir)
        row.addWidget(btn_browse)
        root.addLayout(row)

        # ── Options ────────────────────────────────────────────────────
        box = QGroupBox("Analysis")
        bl = QHBoxLayout(box)

        bl.addWidget(QLabel("Result:"))
        self.cb_mode = QComboBox()
        self.cb_mode.addItem("Whole pulse (fluence)", "fluence")
        self.cb_mode.addItem("Single time slice", "slices")
        self.cb_mode.setToolTip(
            "Whole pulse: analyse the time-integrated fluence (one value per wave).\n"
            "Single time slice: analyse the instantaneous beam at the selected time.")
        bl.addWidget(self.cb_mode)

        bl.addWidget(QLabel("Reference:"))
        self.cb_frame = QComboBox()
        self.cb_frame.addItem("Inside the crystal", "crystal")
        self.cb_frame.addItem("After a flat exit face (air)", "air")
        self.cb_frame.setToolTip(
            "Curvature and tilt are measured inside the crystal. 'Air' converts them for a "
            "beam leaving through a flat exit face: R → R/n, θ → n·θ. Widths, centroids and "
            "M² do not change.")
        bl.addWidget(self.cb_frame)

        bl.addWidget(QLabel("Skip slices below"))
        self.sb_power = QDoubleSpinBox()
        self.sb_power.setRange(0.0, 99.0); self.sb_power.setDecimals(2)
        self.sb_power.setValue(1.0); self.sb_power.setSuffix(" % of peak power")
        self.sb_power.setToolTip(
            "Time slices with less power than this are not analysed (the wings of a pulse "
            "carry no meaningful beam). 0 analyses every slice.")
        bl.addWidget(self.sb_power)

        bl.addWidget(QLabel("Noise floor"))
        self.sb_noise = QDoubleSpinBox()
        self.sb_noise.setRange(0.0, 99.0); self.sb_noise.setDecimals(2)
        self.sb_noise.setValue(0.0); self.sb_noise.setSuffix(" % of peak")
        self.sb_noise.setSpecialValueText("off")
        self.sb_noise.setToolTip(
            "Ignore pixels dimmer than this fraction of the peak irradiance (fluence, for the "
            "whole-pulse analysis). Useful for noise-seeded signal/idler beams, whose weak "
            "background spreads over the whole window and inflates the second moments. "
            "'off' = analyse every pixel.")
        bl.addWidget(self.sb_noise)

        bl.addStretch()
        self.btn_csv = QPushButton("Export CSV…")
        self.btn_csv.setEnabled(False)
        bl.addWidget(self.btn_csv)
        root.addWidget(box)

        # ── Time slider / movie ────────────────────────────────────────
        trow = QHBoxLayout()
        trow.addWidget(QLabel("t-slice:"))
        self.sl_t = QSlider(Qt.Orientation.Horizontal)
        self.sl_t.setRange(0, 0)
        self.lbl_t = QLabel("—")
        self.lbl_t.setMinimumWidth(80)
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setToolTip("Loop through the time slices")
        self.sb_fps = QSpinBox()
        self.sb_fps.setRange(1, 60); self.sb_fps.setValue(10); self.sb_fps.setSuffix(" fps")
        trow.addWidget(self.sl_t, stretch=1)
        trow.addWidget(self.lbl_t)
        trow.addWidget(self.btn_play)
        trow.addWidget(self.sb_fps)
        root.addLayout(trow)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

        # ── Table ──────────────────────────────────────────────────────
        self.table = QTableWidget(0, 1 + len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(["Wave"] + [c[0] for c in _COLUMNS])
        for j, c in enumerate(_COLUMNS, start=1):
            self.table.horizontalHeaderItem(j).setToolTip(c[1])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(170)
        root.addWidget(self.table)

        self.lbl_note = QLabel("")
        self.lbl_note.setWordWrap(True)
        root.addWidget(self.lbl_note)

        # ── Plot vs time ───────────────────────────────────────────────
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Plot vs time:"))
        self.cb_plot = QComboBox()
        for label, *_ in _PLOTS:
            self.cb_plot.addItem(label)
        prow.addWidget(self.cb_plot)
        prow.addWidget(QLabel("(solid = x, dashed = y)"))
        prow.addStretch()
        root.addLayout(prow)
        self.canvas = PlotCanvas(nrows=1, ncols=1, figsize=(8, 3), dpi=90)
        root.addWidget(self.canvas, stretch=1)

        # ── Connections ────────────────────────────────────────────────
        self._opt_timer = QTimer(self)             # debounce the (re)analysis
        self._opt_timer.setSingleShot(True)
        self._opt_timer.setInterval(400)
        self._opt_timer.timeout.connect(self._recompute)
        self.sb_power.valueChanged.connect(lambda _=0: self._opt_timer.start())
        self.sb_noise.valueChanged.connect(lambda _=0: self._opt_timer.start())
        self.cb_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.cb_frame.currentIndexChanged.connect(self._refresh)
        self.cb_plot.currentIndexChanged.connect(self._draw_plot)
        self.sl_t.valueChanged.connect(self._refresh)
        self.btn_play.toggled.connect(self._on_play_toggled)
        self.sb_fps.valueChanged.connect(self._on_fps_changed)
        self.btn_csv.clicked.connect(self._export_csv)

        self._on_mode_changed()
        self._show_message("No data — run a simulation or browse to an output directory.")

    # ── Public API ──────────────────────────────────────────────────────

    def set_output_dir(self, out_dir: str):
        """Point at a new output directory. The analysis runs when the tab is
        first shown, so finishing a simulation does not stall the GUI."""
        self._out_dir = out_dir
        self.le_dir.setText(out_dir)
        self._stale = True
        if self.isVisible():
            self.refresh_if_needed()

    def load_output_dir(self, out_dir: str):
        """Point at `out_dir` and analyse it now."""
        self.set_output_dir(out_dir)
        self.refresh_if_needed()

    def refresh_if_needed(self):
        if self._stale:
            self._recompute()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_if_needed()

    # ── Analysis ────────────────────────────────────────────────────────

    def _recompute(self):
        self._opt_timer.stop()
        self._stale = False
        if not self._out_dir:
            return
        self._timer_stop()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._run = ba.analyze_run(
                self._out_dir,
                power_threshold=self.sb_power.value() / 100.0,
                noise_threshold=self.sb_noise.value() / 100.0)
            error = None
        except Exception as exc:                      # missing files, bad config, …
            self._run = None
            error = f"Could not analyse {self._out_dir}:\n{type(exc).__name__}: {exc}"
        finally:
            QApplication.restoreOverrideCursor()

        self.btn_csv.setEnabled(self._run is not None)
        if self._run is None:
            self.sl_t.setRange(0, 0)
            self._show_message(error)
            return

        # keep the selected time when only the options changed (same slice count)
        nt = len(self._run.t_ps)
        same_run = self.sl_t.maximum() == nt - 1 and nt > 1
        self.sl_t.blockSignals(True)
        self.sl_t.setRange(0, nt - 1)
        if not same_run:
            self.sl_t.setValue(self._peak_slice())
        self.sl_t.blockSignals(False)
        self._on_mode_changed()
        self._refresh()

    def _peak_slice(self) -> int:
        """Initial time: the middle of the analysed part of the pulse, taken
        from the input pump (a clean pulse centred in the window) if present."""
        wave = (self._run.waves.get("Input pump") or self._run.waves.get("Pump")
                or next(iter(self._run.waves.values())))
        valid = np.flatnonzero(~np.isnan(wave.slices.M2x))
        if valid.size == 0:
            return len(self._run.t_ps) // 2
        return int(valid[len(valid) // 2])

    # ── Display ─────────────────────────────────────────────────────────

    def _mode(self) -> str:
        return self.cb_mode.currentData()

    def _frame(self) -> str:
        return self.cb_frame.currentData()

    def _select(self, wave: ba.WaveBeam) -> "tuple[ba.BeamResult, int]":
        """The result and index shown in the table for `wave`."""
        res = wave.fluence if self._mode() == "fluence" else wave.slices
        if self._frame() == "air":
            res = ba.refer_to_air(res, wave.n)
        return res, (0 if self._mode() == "fluence" else self.sl_t.value())

    @staticmethod
    def _fmt(value: float, fmt: str, factor: float) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        if math.isinf(value):
            return "∞"
        text = fmt.format(value * factor)
        if text.startswith("-") and not text.strip("-0."):          # "-0.000" → "0.000"
            text = text[1:]
        return text

    def _refresh(self, *_):
        if self._run is None:
            return
        run = self._run
        it = min(self.sl_t.value(), len(run.t_ps) - 1)
        self.lbl_t.setText(f"{run.t_ps[it]:.3f} ps")

        self.table.setRowCount(len(run.waves))
        warnings, skipped = [], []
        for i, (name, wave) in enumerate(run.waves.items()):
            res, idx = self._select(wave)
            self.table.setItem(i, 0, self._item(name, color=_WAVE_COLORS.get(name)))
            if math.isnan(res.M2x[idx]):
                skipped.append(name)
            for j, (_, _, field, factor, fmt) in enumerate(_COLUMNS, start=1):
                value = float(getattr(res, field)[idx])
                item = self._item(self._fmt(value, fmt, factor))
                if field == "edge_fraction" and res.edge_warning[idx]:
                    item.setBackground(QBrush(QColor("#a65e00")))
                    warnings.append(name)
                self.table.setItem(i, j, item)

        notes = []
        if self._mode() == "slices" and skipped:
            notes.append("— : not analysed (below the power threshold, or no light) for "
                         + ", ".join(skipped) + ".")
        if warnings:
            notes.append("⚠ " + ", ".join(dict.fromkeys(warnings)) + ": a noticeable share of the "
                         "energy reaches the edge of the simulation window, so the moments (and "
                         "M²) are unreliable. Enlarge the window or set a noise floor.")
        notes.append("R < 0: converging, R > 0: diverging (∞: flat). Tilt is positive toward "
                     "+x / +y. " + ("Values as seen after a flat exit face into air."
                                    if self._frame() == "air" else "Values inside the crystal."))
        self.lbl_note.setText("\n".join(notes))
        self._draw_plot()

    @staticmethod
    def _item(text: str, color: "str | None" = None) -> QTableWidgetItem:
        it = QTableWidgetItem(text)
        it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if color:
            it.setForeground(QBrush(QColor(color)))
        return it

    def _show_message(self, text: str):
        self.table.setRowCount(0)
        self.lbl_note.setText(text)
        ax = self.canvas.get_ax(0)
        ax.cla()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="#555555",
                transform=ax.transAxes, fontsize=11)
        self.canvas.refresh()

    def _draw_plot(self, *_):
        if self._run is None:
            return
        label, ylabel, fx, fy, factor = _PLOTS[self.cb_plot.currentIndex()]
        ax = self.canvas.get_ax(0)
        ax.cla()
        t = self._run.t_ps
        for name, wave in self._run.waves.items():
            res = wave.slices
            if self._frame() == "air":
                res = ba.refer_to_air(res, wave.n)
            for field, style in ((fx, "-"), (fy, "--")):
                y = np.asarray(getattr(res, field), dtype=float)
                if factor is None:                                  # curvature 1/R [1/μm] → 1/m
                    with np.errstate(divide="ignore"):
                        y = 1e6 / y
                else:
                    y = y * factor
                ax.plot(t, y, style, color=_WAVE_COLORS.get(name), linewidth=1.3,
                        label=name if style == "-" else None)
        if self._mode() == "slices":
            ax.axvline(t[min(self.sl_t.value(), len(t) - 1)], color="#ffffff", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("t (ps)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(label, fontsize=10)
        if len(t) > 1:
            ax.set_xlim(float(t[0]), float(t[-1]))
        ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#3a3a3a", labelcolor="#ffffff")
        self.canvas.fig.tight_layout(pad=0.5)
        self.canvas.refresh()

    # ── Slots ───────────────────────────────────────────────────────────

    def _on_mode_changed(self, *_):
        slices = self._mode() == "slices"
        multi = self._run is not None and len(self._run.t_ps) > 1
        self.sl_t.setEnabled(slices and multi)
        self.btn_play.setEnabled(slices and multi)
        self.sb_fps.setEnabled(slices and multi)
        self.sb_power.setEnabled(True)
        if not slices:
            self._timer_stop()
        self._refresh()

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory", "")
        if d:
            self.load_output_dir(d)

    def _export_csv(self):
        if self._run is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export beam analysis", "beam_analysis.csv", "CSV files (*.csv)")
        if not path:
            return
        try:
            ba.write_csv(path, self._run, self._mode(), self._frame())
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))

    # ── Movie playback ──────────────────────────────────────────────────

    def _timer_stop(self):
        self._timer.stop()
        if self.btn_play.isChecked():
            self.btn_play.blockSignals(True)
            self.btn_play.setChecked(False)
            self.btn_play.blockSignals(False)
            self.btn_play.setText("▶ Play")

    def _on_play_toggled(self, playing: bool):
        if playing and self.sl_t.maximum() < 1:
            self.btn_play.setChecked(False)
            return
        self.btn_play.setText("⏸ Pause" if playing else "▶ Play")
        if playing:
            self._timer.start(int(1000 / self.sb_fps.value()))
        else:
            self._timer.stop()

    def _on_fps_changed(self):
        if self._timer.isActive():
            self._timer.start(int(1000 / self.sb_fps.value()))

    def _advance_frame(self):
        nxt = self.sl_t.value() + 1
        self.sl_t.setValue(nxt if nxt <= self.sl_t.maximum() else 0)
