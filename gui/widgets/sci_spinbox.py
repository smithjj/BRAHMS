from PyQt6.QtGui import QValidator
from PyQt6.QtWidgets import QDoubleSpinBox


class SciDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that shows small/large magnitudes in exponential
    notation (1e-4 rather than 0.000100) and accepts exponent input."""

    SIG_DIGITS = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setKeyboardTracking(False)

    def textFromValue(self, value: float) -> str:
        if value != 0 and (abs(value) < 1e-2 or abs(value) >= 1e5):
            mant, exp = f"{value:.{self.SIG_DIGITS - 1}e}".split("e")
            if "." in mant:
                mant = mant.rstrip("0").rstrip(".")
            return f"{mant}e{int(exp)}"
        text = f"{value:.{self.SIG_DIGITS}g}"
        return text

    def valueFromText(self, text: str) -> float:
        try:
            return float(text.strip())
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        s = text.strip()
        try:
            v = float(s)
        except ValueError:
            # partial input such as "", "-", "1e", "1e-" stays editable
            return (QValidator.State.Intermediate, text, pos)
        if self.minimum() <= v <= self.maximum():
            return (QValidator.State.Acceptable, text, pos)
        return (QValidator.State.Intermediate, text, pos)
