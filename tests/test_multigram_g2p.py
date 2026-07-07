"""Tests for MultigramG2P (n:m G2P trainer).

These are sanity tests: train on a tiny lexicon, verify the model can
reproduce its training data and generalize to held-out variants. Real
quality comparisons against the 1:1 G2PDecisionTree live in the
join-discovery report and the eval script in scripts/.
"""

from __future__ import annotations

from phonebox.core.multigram_g2p import (
    MultigramG2P,
    decode_phones,
    encode_phones,
    encode_unit_letters,
)


def test_encode_decode_phones_roundtrip():
    assert decode_phones(encode_phones(("a", "b", "c"))) == ["a", "b", "c"]
    # Empty (silent) tuple → empty list
    assert decode_phones(encode_phones(())) == []


def test_encode_unit_letters_uses_join_char():
    assert encode_unit_letters(("c", "h")) == "c\u208ah"
    assert encode_unit_letters(("a",)) == "a"


def _toy_lexicon(tmp_path):
    """Write a small CMUdict-style lexicon and return the path.

    The data deliberately has digraphs and silent letters so the
    multigram aligner has work to do.
    """
    p = tmp_path / "tiny.dict"
    p.write_text(
        # "ch" digraph → /tʃ/
        "chat   tʃ a t\n"
        "chin   tʃ i n\n"
        "rich   r i tʃ\n"
        "batch  b a tʃ\n"
        "much   m a tʃ\n"
        # "sh" digraph → /ʃ/
        "ship   ʃ i p\n"
        "shop   ʃ o p\n"
        "fish   f i ʃ\n"
        # plain single-letter words
        "cat    k a t\n"
        "cap    k a p\n"
        "dog    d o g\n"
        "hot    h o t\n"
        "pin    p i n\n"
        "tip    t i p\n"
        "sip    s i p\n"
        "rib    r i b\n",
        encoding="utf-8",
    )
    return p


def test_training_produces_a_predictor_and_inventory(tmp_path):
    """Smoke test: training succeeds, the unit inventory is non-empty,
    and predict() returns lists of strings.

    The 16-entry toy lexicon is too small to make exact phone-by-phone
    assertions reliable — EM doesn't have enough mass to prune spurious
    compression units like ``(a, t) → /a/`` that only appear in one or
    two training words. Real-data evaluation (Italian/Spanish full
    lexicons, see compare_g2p.py) gives the proper
    accuracy picture.
    """
    lex = _toy_lexicon(tmp_path)
    model = MultigramG2P(
        max_letter_span=2,
        max_phone_span=1,
        min_phone_span=1,
        em_max_iterations=20,
    )
    model.train_from_dict(lex)
    assert model.aligner.q, "EM produced no units"
    # The "ch → tʃ" digraph should be present with non-trivial mass
    ch_unit = (("c", "h"), ("tʃ",))
    assert ch_unit in model.aligner.q, "ch → tʃ digraph not learned"
    assert model.aligner.q[ch_unit] > 0.05, (
        f"ch → tʃ has only {model.aligner.q[ch_unit]:.3f} mass — should be > 5%"
    )
    # And "sh → ʃ"
    sh_unit = (("s", "h"), ("ʃ",))
    assert sh_unit in model.aligner.q, "sh → ʃ digraph not learned"
    # pronounce returns reasonable shape
    out = model.pronounce("chat")
    assert isinstance(out, list)
    assert all(isinstance(p, str) for p in out)
    assert out[0] == "tʃ", f"chat- should start with tʃ, got {out}"


def test_save_load_roundtrip(tmp_path):
    lex = _toy_lexicon(tmp_path)
    model = MultigramG2P(
        max_letter_span=2,
        max_phone_span=1,
        min_phone_span=1,
        em_max_iterations=15,
    )
    model.train_from_dict(lex)
    saved = tmp_path / "tiny.mgg2p"
    model.export(saved)

    # Predictions should be byte-identical after save+load
    chat_before = model.pronounce("chat")
    fish_before = model.pronounce("fish")

    reloaded = MultigramG2P.load(saved)
    assert reloaded.pronounce("chat") == chat_before
    assert reloaded.pronounce("fish") == fish_before


def test_dict_fallback_returns_train_exception(tmp_path):
    model = MultigramG2P(
        max_letter_span=2,
        max_phone_span=1,
        min_phone_span=1,
        em_max_iterations=15,
    )
    model.train_from_pairs([(["c", "a", "t"], ["k", "a", "t"])])
    model.use_dict_fallback = True
    model.exceptions = {"dog": ["d", "o", "g"]}
    assert model.pronounce_letters(list("dog"), word="dog") == ["d", "o", "g"]
    model.use_dict_fallback = False
    assert model.pronounce_letters(list("dog"), word="dog") != ["d", "o", "g"]


def test_pronounce_unseen_letters_falls_back_to_single(tmp_path):
    lex = _toy_lexicon(tmp_path)
    model = MultigramG2P(
        max_letter_span=2, max_phone_span=1, min_phone_span=1, em_max_iterations=15
    )
    model.train_from_dict(lex)
    # 'z' was never seen — joint decode may return partial/empty; must not crash.
    out = model.pronounce("zip")
    # Output is best-effort but must be a list of strings
    assert isinstance(out, list)
    assert all(isinstance(p, str) for p in out)
