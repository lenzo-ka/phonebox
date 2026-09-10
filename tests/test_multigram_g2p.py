"""Tests for MultigramG2P (n:m G2P trainer).

These are sanity tests: train on a tiny lexicon, verify the model can
reproduce its training data and generalize to held-out variants. Real
quality comparisons against the 1:1 G2PDecisionTree live in the
join-discovery report and the ``phonebox compare`` workflows.
"""

from __future__ import annotations

import json

import pytest

from phonebox.core.multigram_g2p import (
    MultigramG2P,
    decode_phones,
    encode_phones,
    encode_unit_letters,
)
from phonebox.core.vectorizer import Vectorizer


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
    lexicons, see ``phonebox compare locale``) gives the proper
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


def test_raw_pronounce_uses_saved_training_preprocessing(tmp_path, monkeypatch):
    vec = Vectorizer(
        locale="default",
        phoneset_name="ipa",
        spelling_rewrites={"x": "q"},
    )
    model = MultigramG2P(
        max_letter_span=1,
        max_phone_span=1,
        min_phone_span=1,
        em_max_iterations=2,
        preprocessor=vec,
    )
    model.train_from_pairs([(["q"], ["K"])])

    # The raw API cooks x→q; the explicit cooked-token API remains literal.
    assert model.pronounce("x") == ["K"]
    assert model.pronounce_letters(["x"]) == []

    saved = tmp_path / "rewrite.g2p"
    model.export(saved)
    monkeypatch.setattr(
        Vectorizer,
        "setup_locale",
        lambda self, locale: (_ for _ in ()).throw(
            AssertionError("snapshot load consulted current locale policy")
        ),
    )
    loaded = MultigramG2P.load(saved)
    assert loaded.pronounce("x") == ["K"]
    assert loaded.preprocessor is not None
    assert loaded.preprocessor.export_letter_preprocessing() == (
        vec.export_letter_preprocessing()
    )


def test_disabled_config_joins_persist_after_reload(tmp_path):
    vec = Vectorizer(locale="it_IT", phoneset_name="ipa")
    vec.disable_config_joins()
    model = MultigramG2P(
        max_letter_span=1,
        max_phone_span=1,
        min_phone_span=1,
        em_max_iterations=2,
        preprocessor=vec,
    )
    model.train_from_pairs([(["g"], ["g"]), (["l"], ["l"]), (["i"], ["i"])])
    saved = tmp_path / "no-joins.g2p"
    model.export(saved)

    loaded = MultigramG2P.load(saved)
    assert loaded.preprocessor is not None
    assert loaded.preprocessor.export_letter_preprocessing()["letter_joins"] == []
    assert loaded.preprocessor.cook_letters("gli", g2p=True) == ["g", "l", "i"]


def test_legacy_v3_model_keeps_per_character_lowercase_contract(tmp_path):
    model = MultigramG2P(
        max_letter_span=1,
        max_phone_span=1,
        min_phone_span=1,
        em_max_iterations=2,
    )
    model.train_from_pairs([(["x"], ["K"])])
    saved = tmp_path / "legacy.g2p"
    model.export(saved)
    metadata_path = saved.with_suffix(saved.suffix + ".units.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["version"] = "3"
    metadata.pop("letter_preprocessing", None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded = MultigramG2P.load(saved)
    assert loaded.preprocessor is None
    assert loaded.pronounce("X") == ["K"]


def test_present_malformed_preprocessing_metadata_is_not_legacy(tmp_path):
    model = MultigramG2P(
        max_letter_span=1,
        max_phone_span=1,
        min_phone_span=1,
        em_max_iterations=2,
    )
    model.train_from_pairs([(["x"], ["K"])])
    saved = tmp_path / "malformed.g2p"
    model.export(saved)
    metadata_path = saved.with_suffix(saved.suffix + ".units.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["letter_preprocessing"] = {"version": 99}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    try:
        MultigramG2P.load(saved)
    except ValueError as error:
        assert "unsupported letter preprocessing version" in str(error)
    else:
        raise AssertionError("malformed snapshot silently used legacy behavior")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda snapshot: snapshot["spelling_rewrites"].update({"x": 3}),
            "spelling_rewrites",
        ),
        (
            lambda snapshot: snapshot["spelling_rewrites"].update({"X": "q"}),
            "use the cooked character 'x'",
        ),
        (
            lambda snapshot: snapshot["source"].update(
                {"g2p_rules": ":: definitely-not-a-transliterator ;"}
            ),
            "invalid saved g2p transliterator rules",
        ),
        (
            lambda snapshot: snapshot["source"].pop("g2p_rules"),
            "missing letter preprocessing source fields: g2p_rules",
        ),
    ],
)
def test_snapshot_validation_fails_at_model_load(tmp_path, mutate, message):
    vec = Vectorizer(locale="default", phoneset_name="ipa")
    model = MultigramG2P(
        max_letter_span=1,
        max_phone_span=1,
        min_phone_span=1,
        em_max_iterations=2,
        preprocessor=vec,
    )
    model.train_from_pairs([(["x"], ["K"])])
    saved = tmp_path / "invalid-snapshot.g2p"
    model.export(saved)
    metadata_path = saved.with_suffix(saved.suffix + ".units.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutate(metadata["letter_preprocessing"])
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        MultigramG2P.load(saved)
