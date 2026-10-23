import pytest

from gui.core_py.crystal_db import CrystalDB

PRELOADED = {"MgO:PPLN", "MgO:sPPLT", "BBO"}

NEW = {
    "name": "TestCrystal",
    "type": "QPM-PPLN",
    "deff": 10.0,
    "formula_e": "sqrt(n0**2 + A*L**2)",
    "coeffs_e": {"n0": 2.0, "A": 0.01},
}


def test_preloaded_crystals_are_seeded(db):
    assert PRELOADED <= set(db.all_names())


def test_get_unknown_returns_none(db):
    assert db.get("nope") is None


def test_get_decodes_json_coefficients(db):
    cr = db.get("MgO:PPLN")
    assert isinstance(cr["coeffs_e"], dict) and cr["coeffs_e"]


def test_add_get_roundtrip(db):
    assert db.add(NEW)
    got = db.get("TestCrystal")
    assert got["deff"] == 10.0
    assert got["coeffs_e"] == {"n0": 2.0, "A": 0.01}
    assert got["preloaded"] == 0


def test_add_duplicate_name_does_not_overwrite(db):
    db.add(NEW)
    db.add({**NEW, "deff": 99.0})
    assert db.get("TestCrystal")["deff"] == 10.0


def test_update(db):
    db.add(NEW)
    assert db.update("TestCrystal", {"deff": 12.5, "coeffs_e": {"n0": 2.1, "A": 0.02}})
    got = db.get("TestCrystal")
    assert got["deff"] == 12.5 and got["coeffs_e"]["n0"] == 2.1


def test_delete_user_crystal(db):
    db.add(NEW)
    assert db.delete("TestCrystal")
    assert db.get("TestCrystal") is None


def test_cannot_delete_preloaded_or_missing(db):
    assert not db.delete("MgO:PPLN")
    assert db.get("MgO:PPLN") is not None
    assert not db.delete("does-not-exist")


def test_user_crystals_persist_across_reopen(tmp_path):
    path = tmp_path / "c.db"
    CrystalDB(path).add(NEW)
    assert CrystalDB(path).get("TestCrystal") is not None


def test_preloaded_rows_are_repaired_on_open(tmp_path):
    path = tmp_path / "c.db"
    d = CrystalDB(path)
    d.update("MgO:PPLN", {"deff": -1.0})
    assert CrystalDB(path).get("MgO:PPLN")["deff"] != -1.0


@pytest.mark.parametrize("name", sorted(PRELOADED))
def test_every_preloaded_crystal_has_working_sellmeier(db, name):
    sf = db.sellmeier(name, axis="e")
    assert sf.is_ready, sf.error
    n = float(sf.n(1.064, 25.0))
    assert 1.4 < n < 2.5


def test_ppln_extraordinary_index_is_physical(db):
    # Congruent/MgO:LN, extraordinary axis, 1.064 um: ~2.15
    assert float(db.sellmeier("MgO:PPLN", "e").n(1.064, 25.0)) == pytest.approx(2.15, abs=0.02)


def test_bbo_is_negative_uniaxial_with_both_axes(db):
    no = float(db.sellmeier("BBO", "o").n(1.064, 25.0))
    ne = float(db.sellmeier("BBO", "e").n(1.064, 25.0))
    assert no == pytest.approx(1.655, abs=0.01)
    assert no > ne


def test_sellmeier_accepts_dict_and_json_string_coeffs(db):
    sf = db.sellmeier({**NEW, "coeffs_e": '{"n0": 2.0, "A": 0.01}'}, axis="e")
    assert sf.is_ready
