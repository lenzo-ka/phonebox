"""Portable letter-preprocessing compiler and interpreter tests."""

from pathlib import Path

import pytest

from phonebox.portable_normalization import (
    PortableNormalizationError,
    apply_portable_preprocessing,
    compile_letter_preprocessing,
)


def _snapshot(rules, joins=(), **overrides):
    snapshot = {
        "version": 1,
        "source": {"norm_rules": None, "g2p_rules": rules},
        "join_char": "₊",
        "letter_joins": list(joins),
        "cased": False,
        "remove_accents": False,
        "filter_non_letters": False,
        "spelling_rewrites": {},
    }
    snapshot.update(overrides)
    return snapshot


SPANISH_RULES = r"""
:: NFC ;
\u00F1 > \uE001 ;
\u00D1 > \uE002 ;
\u00FC > \uE003 ;
\u00DC > \uE004 ;
:: NFD ;
:: [:M:] Remove ;
:: NFC ;
\uE001 > \u00F1 ;
\uE002 > \u00D1 ;
\uE003 > \u00FC ;
\uE004 > \u00DC ;
:: Null ;
:: [^-.'[:L:]] Remove ;
:: Any-Lower ;
"""


ITALIAN_RULES = r"""
:: NFC ;
\u00E8 > \u025B ;
\u00C8 > \u025B ;
\u00E9 > e ;
\u00C9 > e ;
\u00F2 > \u0254 ;
\u00D2 > \u0254 ;
\u00F3 > o ;
\u00D3 > o ;
:: Any-Lower ;
:: Null ;
:: [^-.[:L:]] Remove ;
"""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("MA\u00d1ANA", ["m", "a", "ñ", "a", "n", "a"]),
        ("MAN\u0303ANA", ["m", "a", "ñ", "a", "n", "a"]),
        ("PINGU\u0308INO", ["p", "i", "n", "g", "ü", "i", "n", "o"]),
        ("Canci\u00f3n!", ["c", "a", "n", "c", "i", "o", "n"]),
    ],
)
def test_spanish_expected_outputs(text, expected):
    program = compile_letter_preprocessing(_snapshot(SPANISH_RULES))
    assert apply_portable_preprocessing(text, program) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("CAFFÈ", ["c", "a", "f", "f", "ɛ"]),
        ("caffe\u0300", ["c", "a", "f", "f", "ɛ"]),
        ("ABBANDONÒ", ["a", "b", "b", "a", "n", "d", "o", "n", "ɔ"]),
        ("perché", ["p", "e", "r", "c", "h", "e"]),
        ("dell'uragano", list("delluragano")),
    ],
)
def test_italian_expected_outputs(text, expected):
    program = compile_letter_preprocessing(_snapshot(ITALIAN_RULES))
    assert apply_portable_preprocessing(text, program) == expected


def test_rewrites_are_simultaneous_and_joins_follow_them():
    snapshot = _snapshot(
        ":: NFC ;",
        joins=["c h"],
        spelling_rewrites={"x": "c", "c": "q"},
    )
    program = compile_letter_preprocessing(snapshot)
    assert apply_portable_preprocessing("xh", program) == ["c₊h"]


def test_uncased_flag_lowercases_each_scalar_independently():
    program = compile_letter_preprocessing(_snapshot(None, cased=False))
    assert apply_portable_preprocessing("ΑΣ", program) == ["α", "σ"]
    assert apply_portable_preprocessing("İΣ", program) == ["i", "σ"]


def test_icu_any_lower_remains_contextual():
    program = compile_letter_preprocessing(_snapshot(":: Any-Lower ;", cased=True))
    assert apply_portable_preprocessing("ΑΣ", program) == ["α", "ς"]
    assert apply_portable_preprocessing("İΣ", program) == ["i", "ς"]


@pytest.mark.parametrize("missing", ["norm_rules", "g2p_rules"])
def test_source_requires_both_rule_keys(missing):
    snapshot = _snapshot(None)
    del snapshot["source"][missing]
    with pytest.raises(PortableNormalizationError, match=missing):
        compile_letter_preprocessing(snapshot)


def test_consecutive_icu_replacements_are_one_simultaneous_pass():
    program = compile_letter_preprocessing(_snapshot("x > y; y > z;", cased=True))
    assert apply_portable_preprocessing("xy", program) == ["y", "z"]


def test_icu_mark_removal_removes_all_mark_categories_without_decomposition():
    program = compile_letter_preprocessing(_snapshot(":: [:M:] Remove;", cased=True))
    assert apply_portable_preprocessing("a\u0903", program) == ["a"]


def test_filter_flag_recomposes_preserved_marks_like_vectorizer():
    program = compile_letter_preprocessing(
        _snapshot(None, cased=True, filter_non_letters=True)
    )
    assert apply_portable_preprocessing("e\u0301", program) == ["é"]


@pytest.mark.parametrize(
    "rules",
    [
        ":: Latin ;",
        ":: Fullwidth-Halfwidth ;",
        "[:Pd:] > \\- ;",
        "a b > c ;",
        ":: [^a-z[:L:]] Remove ;",
        ". > x ;",
        "a > 'b' ;",
        "\\u0061b > c ;",
        ":: [^'-\\.[:L:]] Remove ;",
    ],
)
def test_unsupported_icu_invalidates_the_whole_program(rules):
    with pytest.raises(PortableNormalizationError, match="unsupported ICU rules"):
        compile_letter_preprocessing(_snapshot(rules))


def test_unknown_snapshot_version_is_rejected():
    with pytest.raises(PortableNormalizationError, match="version"):
        compile_letter_preprocessing(_snapshot(":: NFC ;", version=2))


def test_all_shipped_locale_g2p_rules_are_portable():
    locale_root = Path(__file__).parents[1] / "phonebox" / "config" / "locales"
    for rule_path in locale_root.glob("*/g2p.xlit"):
        rules = rule_path.read_text(encoding="utf-8")
        compile_letter_preprocessing(_snapshot(rules))
