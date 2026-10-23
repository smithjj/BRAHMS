from gui.core_py.bibtex_utils import bibtex_to_citation, parse_bibtex_fields

GAYER = """@article{Gayer2008,
  author = {O. Gayer and Z. Sacks and E. Galun and A. Arie},
  title = {Temperature and wavelength dependent refractive index},
  journal = {Applied Physics B},
  volume = {91},
  pages = {343--348},
  year = {2008}
}"""


def test_parse_fields_lowercases_keys():
    f = parse_bibtex_fields(GAYER)
    assert f["year"] == "2008"
    assert f["volume"] == "91"
    assert f["author"].startswith("O. Gayer")


def test_citation_multiple_authors():
    assert bibtex_to_citation(GAYER).startswith("Gayer et al.,")
    assert bibtex_to_citation(GAYER).endswith("(2008)")
    assert "91, 343" in bibtex_to_citation(GAYER)


def test_citation_single_author_has_no_et_al():
    s = "@article{x, author = {Smith, John}, journal = {Optica}, volume = {3}, pages = {10-12}, year = {2016}}"
    c = bibtex_to_citation(s)
    assert "et al." not in c and c.startswith("Smith")
    assert "3, 10" in c and c.endswith("(2016)")


def test_empty_and_yearless_input():
    assert bibtex_to_citation("") == ""
    assert bibtex_to_citation(None) == ""
    assert bibtex_to_citation("@article{x, author={A B}}") == ""


def test_year_only():
    assert bibtex_to_citation("@misc{x, year = {1999}}") == "(1999)"
