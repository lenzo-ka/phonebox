"""Spanish Unicode normalization regressions."""

import gzip
import json
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
        ("hacía", "hacía"),
        ("HACÍA", "hacía"),
        ("continúo", "continúo"),
        ("CONTINÚO", "continúo"),
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


def test_public_g2p_keeps_spanish_vowels_distinct_from_glides(tmp_path):
    dictionary = tmp_path / "spanish-hiatus.dict"
    dictionary.write_text(
        "hacia a s j a\n"
        "hacía a s i a\n"
        "continuo k o n t i n w o\n"
        "continúo k o n t i n u o\n",
        encoding="utf-8",
    )
    g2p = G2P.train(
        dictionary,
        locale="es_MX",
        phoneset="ipa",
        use_dict_fallback=False,
        verbose=False,
    )

    expected = {
        "hacia": ["a", "s", "j", "a"],
        "hacía": ["a", "s", "i", "a"],
        "continuo": ["k", "o", "n", "t", "i", "n", "w", "o"],
        "continúo": ["k", "o", "n", "t", "i", "n", "u", "o"],
    }
    for word, phones in expected.items():
        assert g2p.pronounce(word) == phones
        assert g2p.pronounce(word.upper()) == phones
        assert g2p.pronounce(unicodedata.normalize("NFD", word)) == phones


def test_spanish_snapshot_reload_keeps_new_policy(tmp_path):
    dictionary = tmp_path / "snapshot.dict"
    dictionary.write_text("hacia a s j a\nhacía a s i a\n", encoding="utf-8")
    model_path = tmp_path / "snapshot.g2p.gz"
    G2P.train(
        dictionary,
        locale="es_MX",
        phoneset="ipa",
        use_dict_fallback=False,
        output=model_path,
        verbose=False,
    )

    loaded = G2P(model=model_path, use_dict_fallback=False)
    assert "".join(loaded._dt.vectorizer.cook_letters("hacía", g2p=True)) == "hacía"


def test_snapshot_absent_spanish_cart_uses_known_previous_policy(tmp_path):
    dictionary = tmp_path / "legacy.dict"
    dictionary.write_text("hacia a s j a\nhacía a s i a\n", encoding="utf-8")
    model_path = tmp_path / "legacy.g2p.gz"
    G2P.train(
        dictionary,
        locale="es_MX",
        phoneset="ipa",
        use_dict_fallback=False,
        output=model_path,
        verbose=False,
    )
    with gzip.open(model_path, "rt", encoding="utf-8") as infile:
        rows = [json.loads(line) for line in infile]
    rows[-1]["metadata"].pop("letter_preprocessing")
    with gzip.open(model_path, "wt", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

    loaded = G2P(model=model_path, use_dict_fallback=False)
    vectorizer = loaded._dt.vectorizer
    assert "".join(vectorizer.cook_letters("hacía", g2p=True)) == "hacia"
    assert "".join(vectorizer.cook_letters("continúo", g2p=True)) == "continuo"
    assert "".join(vectorizer.cook_letters("canción", g2p=True)) == "cancion"
