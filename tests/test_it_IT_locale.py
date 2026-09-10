"""Italian locale: digraph joins and apostrophe stripping."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
import unicodedata

import pytest

from phonebox import G2P
from phonebox.bundler import bundle_g2p
from phonebox.core.vectorizer import Vectorizer
from phonebox.runner import G2PRunner

JOIN = "\u208a"


def test_it_joins_gn_gli_zz_sc():
    vec = Vectorizer(locale="it_IT", phoneset_name="ipa")
    assert vec.g2p_transliterator is not None
    assert vec.cook_letters("agnello", g2p=True) == [
        "a",
        f"g{JOIN}n",
        "e",
        f"l{JOIN}l",
        "o",
    ]
    assert vec.cook_letters("figlio", g2p=True) == ["f", "i", f"g{JOIN}l{JOIN}i", "o"]
    assert vec.cook_letters("pizza", g2p=True) == ["p", "i", f"z{JOIN}z", "a"]
    assert vec.cook_letters("scena", g2p=True) == [f"s{JOIN}c{JOIN}e", "n", "a"]
    assert vec.cook_letters("sciopero", g2p=True) == [
        f"s{JOIN}c{JOIN}i",
        "o",
        "p",
        "e",
        "r",
        "o",
    ]


def test_it_phonemic_accents_map_to_ipa_graphemes():
    vec = Vectorizer(locale="it_IT", phoneset_name="ipa")
    assert vec.cook_letters("caffè", g2p=True) == ["c", "a", "f", "f", "ɛ"]
    assert vec.cook_letters("abbandonò", g2p=True)[-1] == "ɔ"


@pytest.mark.parametrize(
    ("letter", "expected"),
    [
        ("à", "a"),
        ("á", "a"),
        ("ì", "i"),
        ("í", "i"),
        ("ù", "u"),
        ("ú", "u"),
        ("è", "ɛ"),
        ("é", "e"),
        ("ò", "ɔ"),
        ("ó", "o"),
    ],
)
def test_it_accent_policy_is_stable_across_case_and_normalization(letter, expected):
    vec = Vectorizer(locale="it_IT", phoneset_name="ipa")
    variants = (letter, letter.upper(), unicodedata.normalize("NFD", letter))
    assert ["".join(vec.cook_letters(value, g2p=True)) for value in variants] == [
        expected
    ] * 3


def test_public_g2p_folds_accented_aiu_to_trained_plain_spellings(tmp_path):
    dictionary = tmp_path / "italian-accents.dict"
    dictionary.write_text(
        "citta tʃ i t t a\ncosi k o z i\npiu p j u\nlagumina l a ɡ u m i n a\n",
        encoding="utf-8",
    )
    g2p = G2P.train(
        dictionary,
        locale="it_IT",
        phoneset="ipa",
        use_dict_fallback=False,
        verbose=False,
    )
    expected = {
        "città": ["tʃ", "i", "t", "t", "a"],
        "così": ["k", "o", "z", "i"],
        "più": ["p", "j", "u"],
        "lagúmina": ["l", "a", "ɡ", "u", "m", "i", "n", "a"],
    }
    for word, phones in expected.items():
        assert g2p.pronounce(word) == phones
        assert g2p.pronounce(word.upper()) == phones
        assert g2p.pronounce(unicodedata.normalize("NFD", word)) == phones


def test_italian_accented_words_match_runner_and_bundle(tmp_path):
    dictionary = tmp_path / "italian-accents-deployment.dict"
    dictionary.write_text(
        "citta tʃ i t t a\ncosi k o z i\npiu p j u\nlagumina l a ɡ u m i n a\n",
        encoding="utf-8",
    )
    g2p = G2P.train(
        dictionary,
        locale="it_IT",
        phoneset="ipa",
        use_dict_fallback=False,
        verbose=False,
    )
    cart_path = tmp_path / "italian-accents.cart"
    model_path = tmp_path / "italian-accents.g2p.gz"
    bundle_path = tmp_path / "italian-accents.py"
    g2p._dt.export(str(cart_path), include_exceptions=False)
    g2p._dt.export(str(model_path), include_exceptions=False)
    runner = G2PRunner(str(cart_path))
    bundle_g2p(str(model_path), str(bundle_path))

    expected = {
        "città": ["tʃ", "i", "t", "t", "a"],
        "così": ["k", "o", "z", "i"],
        "più": ["p", "j", "u"],
        "lagúmina": ["l", "a", "ɡ", "u", "m", "i", "n", "a"],
    }
    for word, phones in expected.items():
        for variant in (word, unicodedata.normalize("NFD", word)):
            assert g2p.pronounce(variant) == phones
            assert runner.pronounce(variant) == phones
            result = subprocess.run(
                [sys.executable, "-S", str(bundle_path), variant],
                capture_output=True,
                text=True,
                check=True,
            )
            assert result.stdout.strip().partition("\t")[2].split() == phones


def test_it_snapshot_and_known_legacy_policy_remain_distinct(tmp_path):
    dictionary = tmp_path / "italian-snapshot.dict"
    dictionary.write_text("citta k i t t a\n", encoding="utf-8")
    model_path = tmp_path / "italian.g2p.gz"
    G2P.train(
        dictionary,
        locale="it_IT",
        phoneset="ipa",
        use_dict_fallback=False,
        output=model_path,
        verbose=False,
    )

    current = G2P(model=model_path, use_dict_fallback=False)
    assert "".join(current._dt.vectorizer.cook_letters("città", g2p=True)) == "citta"

    with gzip.open(model_path, "rt", encoding="utf-8") as infile:
        rows = [json.loads(line) for line in infile]
    rows[-1]["metadata"].pop("letter_preprocessing")
    with gzip.open(model_path, "wt", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

    legacy = G2P(model=model_path, use_dict_fallback=False)
    assert "".join(legacy._dt.vectorizer.cook_letters("città", g2p=True)) == "città"
    assert "".join(legacy._dt.vectorizer.cook_letters("caffè", g2p=True)) == "caffɛ"


def test_it_apostrophe_stripped():
    vec = Vectorizer(locale="it_IT", phoneset_name="ipa")
    cooked = vec.cook_letters("dell'uragano", g2p=True)
    assert "'" not in "".join(cooked)
    # After elision removal, "ll" may still join as a digraph token.
    assert cooked[0] == "d"
    assert cooked[-1] == "o"
