"""Tests for train-split lexicon normalization policies."""

from phonebox.experiments.normalize import (
    normalize_it_collapse_open,
    normalize_it_spelling_gated,
    normalize_pt_citation_expand,
    normalize_pt_do_du,
    normalize_pt_surface_final,
)


def test_it_spelling_gated_keeps_open_with_accent():
    assert normalize_it_spelling_gated("caffè", ["k", "a", "f", "f", "ɛ"]) == [
        "k",
        "a",
        "f",
        "f",
        "ɛ",
    ]
    assert normalize_it_spelling_gated("bello", ["b", "ɛ", "l", "o"]) == [
        "b",
        "e",
        "l",
        "o",
    ]


def test_it_collapse_open_always():
    assert normalize_it_collapse_open("caffè", ["ɛ"]) == ["e"]


def test_pt_surface_final():
    assert normalize_pt_surface_final("bonito", ["b", "o", "n", "i", "t", "o"]) == [
        "b",
        "o",
        "n",
        "i",
        "t",
        "u",
    ]
    assert normalize_pt_surface_final("bonito", ["b", "o", "n", "i", "t", "u"]) == [
        "b",
        "o",
        "n",
        "i",
        "t",
        "u",
    ]


def test_pt_do_du():
    assert normalize_pt_do_du("quando", ["k", "w", "ɐ̃", "d", "o"]) == [
        "k",
        "w",
        "ɐ̃",
        "d",
        "u",
    ]


def test_pt_citation_expand_opposite():
    assert normalize_pt_citation_expand("bonito", ["b", "o", "n", "i", "t", "u"]) == [
        "b",
        "o",
        "n",
        "i",
        "t",
        "o",
    ]
