## Add tests for your customize modules here.

from atommovr.utils.customize import (
    SPECIES1NAME,
    SPECIES2NAME,
    SPECIES1COL,
    SPECIES2COL,
    NOATOMCOL,
    EDGECOL,
    ARROWCOL,
)

dclass = None


def test_species_names_are_strings():
    assert isinstance(SPECIES1NAME, str)
    assert isinstance(SPECIES2NAME, str)


def test_colors_are_defined():
    assert SPECIES1COL is not None
    assert SPECIES2COL is not None
    assert NOATOMCOL is not None
    assert EDGECOL is not None
    assert ARROWCOL is not None
