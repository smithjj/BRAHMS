from PyQt6.QtCore import QLocale, QObject
from PyQt6.QtWidgets import QComboBox, QLabel

from ..core_py.config_builder import energy_from_peak_power, peak_power_from_energy
from .sci_spinbox import SciDoubleSpinBox

_LOCALE_C = QLocale(QLocale.Language.C)

POWER, ENERGY = "power", "energy"


class _LogStepSpinBox(SciDoubleSpinBox):
    """Scientific spinbox whose arrows step by 10 % of the value, since the
    entered quantity may be anywhere from fJ to MW."""

    def stepBy(self, steps: int):
        self.setValue(self.value() * (1.1 ** steps))


class PowerEnergyControl(QObject):
    """Peak power (W) or pulse energy (J) entry for one field.

    Owns the spinbox (`spin`), a selector (`cb_kind`, which stands in for the
    plain "Power (W)" label) and a read-only hint (`lbl_hint`) showing the
    derived peak power / energy. The widgets are placed by the caller. Energy
    is only offered for pulsed fields; CW always uses peak power.
    """

    def __init__(self, fwhm_spin, value_W=1.0, parent=None):
        super().__init__(parent)
        self._fwhm = fwhm_spin
        self._pulsed = False
        self._kind = POWER

        self.cb_kind = QComboBox()
        self.cb_kind.addItem("Peak power (W)", POWER)
        self.cb_kind.addItem("Pulse energy (J)", ENERGY)
        self.cb_kind.setToolTip(
            "Pulsed fields can be specified by pulse energy instead of peak power "
            "(Gaussian: E = 1.0645 · P_peak · FWHM, FWHM of the intensity).")

        self.spin = _LogStepSpinBox()
        self.spin.setLocale(_LOCALE_C)
        self.spin.setDecimals(30)       # keeps ~6 significant digits down to fJ energies
        self.spin.setRange(1e-15, 1e12)
        self.spin.setValue(value_W)

        self.lbl_hint = QLabel()
        self.lbl_hint.setStyleSheet("color: gray;")
        self.lbl_hint.hide()

        self.cb_kind.currentIndexChanged.connect(self._on_kind_changed)
        self.spin.valueChanged.connect(self.update_hint)
        fwhm_spin.valueChanged.connect(self.update_hint)

    # ── state ──────────────────────────────────────────────────────────
    def kind(self) -> str:
        return self._kind

    def set_pulsed(self, pulsed: bool):
        """CW fields can only use peak power; switching to CW reverts to it."""
        self._pulsed = pulsed
        self.cb_kind.setEnabled(pulsed)
        if not pulsed and self._kind == ENERGY:
            self.cb_kind.setCurrentIndex(0)    # converts the value back to W
        self.update_hint()

    def peak_power_W(self) -> float:
        v = self.spin.value()
        if self._kind == ENERGY:
            return peak_power_from_energy(v, self._fwhm.value())
        return v

    def params(self) -> dict:
        """The power/energy entries of a params dict for build_config()."""
        if self._kind == ENERGY and self._pulsed:
            return {"energy_J": self.spin.value(), "power_W": self.peak_power_W()}
        return {"power_W": self.spin.value()}

    # ── slots ──────────────────────────────────────────────────────────
    def _on_kind_changed(self):
        new = self.cb_kind.currentData()
        if new == self._kind:
            return
        fwhm = self._fwhm.value()
        v = self.spin.value()
        # Keep the same physical pulse: convert the displayed number.
        converted = energy_from_peak_power(v, fwhm) if new == ENERGY \
            else peak_power_from_energy(v, fwhm)
        self._kind = new
        self.spin.setValue(converted)
        self.update_hint()

    def update_hint(self, *_):
        if not self._pulsed:
            self.lbl_hint.hide()
            return
        fwhm = self._fwhm.value()
        v = self.spin.value()
        if self._kind == ENERGY:
            self.lbl_hint.setText(f"→ peak power {peak_power_from_energy(v, fwhm):.4g} W")
        else:
            self.lbl_hint.setText(f"→ pulse energy {energy_from_peak_power(v, fwhm):.4g} J")
        self.lbl_hint.show()
