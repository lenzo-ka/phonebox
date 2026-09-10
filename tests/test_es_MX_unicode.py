"""Spanish Unicode normalization regressions."""

import unicodedata

import pytest

from phonebox import G2P
from phonebox.core.vectorizer import Vectorizer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("niña", "niña"),
        ("NIÑA", "niña"),
        ("pingüino", "pingüino"),
        ("PINGÜINO", "pingüino"),
        ("canción", "cancion"),
        ("CANCIÓN", "cancion"),
    ],
)
def test_spanish_xlit_has_explicit_nfc_and_nfd_outputs(text, expected):
    vectorizer = Vectorizer(locale="es_MX", phoneset_name="ipa")

    assert "".join(vectorizer.cook_letters(text, g2p=True)) == expected
    assert (
        "".join(vectorizer.cook_letters(unicodedata.normalize("NFD", text), g2p=True))
        == expected
    )


def test_public_g2p_treats_spanish_nfc_and_nfd_letters_identically(tmp_path):
    dictionary = tmp_path / "spanish-unicode.dict"
    dictionary.write_text("a A\nn N\nñ NY\nu U\nü W\n", encoding="utf-8")
    g2p = G2P.train(
        dictionary,
        locale="es_MX",
        phoneset="ipa",
        use_dict_fallback=False,
        verbose=False,
    )

    expected = {"ñ": ["NY"], "ü": ["W"]}
    for letter, phones in expected.items():
        variants = (
            letter,
            letter.upper(),
            unicodedata.normalize("NFD", letter),
            unicodedata.normalize("NFD", letter.upper()),
        )
        assert [g2p.pronounce(variant) for variant in variants] == [phones] * 4
