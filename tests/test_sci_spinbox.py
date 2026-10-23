import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QValidator
from PyQt6.QtTest import QTest

from gui.widgets.sci_spinbox import SciDoubleSpinBox


@pytest.fixture
def sb(qapp):
    box = SciDoubleSpinBox()
    box.setDecimals(12)
    box.setRange(0, 1e6)
    box.show()
    return box


@pytest.mark.parametrize("value,text", [
    (1e-4, "1e-4"), (2.5e-6, "2.5e-6"), (5e-3, "5e-3"),
    (0.025, "0.025"), (1.5, "1.5"), (25.0, "25"), (0.0, "0"),
    (2.5e5, "2.5e5"), (1e-12, "1e-12"),
])
def test_text_from_value(sb, value, text):
    assert sb.textFromValue(value) == text


@pytest.mark.parametrize("text,value", [
    ("1e-5", 1e-5), ("1E-5", 1e-5), ("2.5e-6", 2.5e-6),
    ("0.00001", 1e-5), ("  3 ", 3.0), ("4e3", 4000.0),
])
def test_value_from_text(sb, text, value):
    assert sb.valueFromText(text) == pytest.approx(value)


def test_unparseable_text_keeps_current_value(sb):
    sb.setValue(0.5)
    assert sb.valueFromText("garbage") == 0.5


@pytest.mark.parametrize("text,state", [
    ("1e-5", QValidator.State.Acceptable),
    ("0.5", QValidator.State.Acceptable),
    ("1e", QValidator.State.Intermediate),     # partial exponent, still being typed
    ("", QValidator.State.Intermediate),
    ("abc", QValidator.State.Intermediate),
    ("2e7", QValidator.State.Intermediate),    # above maximum
    ("-1", QValidator.State.Intermediate),     # below minimum
])
def test_validate(sb, text, state):
    assert sb.validate(text, len(text))[0] == state


def test_typing_exponent_and_pressing_enter_sets_value(sb):
    sb.setFocus()
    sb.selectAll()
    QTest.keyClicks(sb, "1e-5")
    QTest.keyClick(sb, Qt.Key.Key_Return)
    assert sb.value() == pytest.approx(1e-5)
    assert sb.text() == "1e-5"


def test_value_roundtrip_through_text(sb):
    for v in (1e-9, 3.3e-5, 0.0123, 12.5, 4.2e5):
        sb.setValue(v)
        assert sb.valueFromText(sb.text()) == pytest.approx(v, rel=1e-5)
