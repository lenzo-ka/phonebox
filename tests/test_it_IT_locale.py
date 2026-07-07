"""Italian locale: digraph joins and apostrophe stripping."""

from __future__ import annotations

from phonebox.core.vectorizer import Vectorizer

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


def test_it_apostrophe_stripped():
    vec = Vectorizer(locale="it_IT", phoneset_name="ipa")
    cooked = vec.cook_letters("dell'uragano", g2p=True)
    assert "'" not in "".join(cooked)
    # After elision removal, "ll" may still join as a digraph token.
    assert cooked[0] == "d"
    assert cooked[-1] == "o"
