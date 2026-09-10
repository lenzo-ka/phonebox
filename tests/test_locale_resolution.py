"""Shared locale identity and resource-selection contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phonebox import G2P
from phonebox.core.multigram_g2p import MultigramG2P
from phonebox.core.vectorizer import Vectorizer
from phonebox.exemplars import get_exemplars
from phonebox.locale_resolution import (
    canonical_locale,
    locale_candidates,
    orthographic_compatibility,
    resolve_locale,
)
from phonebox.locales import load_locale_defaults


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("it-it", "it_IT"),
        ("it-IT", "it_IT"),
        ("it_it", "it_IT"),
        ("it_IT", "it_IT"),
        ("IT_IT", "it_IT"),
        ("IT", "it"),
        ("zh-hant-tw", "zh_Hant_TW"),
        ("sr-Latn-RS", "sr_Latn_RS"),
        ("iw_IL", "he_IL"),
    ],
)
def test_canonical_locale_variants(source, expected):
    assert canonical_locale(source) == expected


@pytest.mark.parametrize("source", ["", "en__US", "english", "en_US_x"])
def test_canonical_locale_rejects_malformed_input(source):
    with pytest.raises(ValueError, match="locale"):
        canonical_locale(source)


def test_resolution_is_exact_then_likely_then_compatible():
    available = {"en_US", "en_IN", "es_MX", "it_IT"}
    compatibility = orthographic_compatibility()
    assert resolve_locale("EN_us", available, compatibility).match == "exact"
    assert resolve_locale("en", available, compatibility).resolved == "en_US"
    assert resolve_locale("it", available, compatibility).resolved == "it_IT"
    spanish = resolve_locale("es_ES", available, compatibility)
    assert (spanish.resolved, spanish.match) == ("es_MX", "compatible")
    assert resolve_locale("en_ZA", available, compatibility).resolved is None
    with pytest.raises(ValueError, match="ambiguous available"):
        resolve_locale("it", ["it_IT", "IT-it"])
    assert resolve_locale("en", ["fr_FR"], {"en": ["fr_FR"]}).resolved is None
    assert (
        resolve_locale(
            "sr_Latn_RS",
            ["sr_Cyrl_RS"],
            {"sr_Latn_RS": ["sr_Cyrl_RS"]},
        ).resolved
        is None
    )


def test_exemplars_keep_exact_bare_identity():
    assert get_exemplars("EN") == get_exemplars("en")
    assert get_exemplars("en") == get_exemplars("en_US")
    with pytest.raises(KeyError):
        get_exemplars("en_Latn_US")


def test_vectorizer_uses_resolved_policy_but_preserves_request():
    bare = Vectorizer(locale="IT", phoneset_name="ipa")
    assert bare.locale == "it"
    assert bare.policy_locale == "it_IT"
    assert bare.g2p_transliterator is not None
    exact = Vectorizer(locale="en_IN", phoneset_name="ipa")
    assert exact.locale == exact.policy_locale == "en_IN"
    compatible = Vectorizer(locale="es_ES", phoneset_name="ipa")
    assert compatible.locale == "es_ES"
    assert compatible.policy_locale == "es_MX"


def test_supplement_defaults_do_not_use_profile_compatibility():
    # es_ES has the same pinned orthographic profile as es_MX, whose defaults
    # split a locale-specific phone token. Phone defaults must not transfer.
    assert load_locale_defaults("es_ES") == load_locale_defaults("default")
    assert load_locale_defaults("it") == load_locale_defaults("it_IT")


def test_public_training_accepts_case_separator_alias(tmp_path):
    dictionary = tmp_path / "italian.dict"
    dictionary.write_text("caffe k a f f e\ncitta t i t t a\n", encoding="utf-8")
    model_path = tmp_path / "italian.g2p.gz"
    trained = G2P.train(
        dictionary,
        locale="IT-it",
        phoneset="ipa",
        output=model_path,
        use_dict_fallback=False,
        verbose=False,
    )
    assert trained.locale == "it_IT"
    assert trained._dt.vectorizer.locale == "it_IT"
    assert trained._dt.vectorizer.cook_letters("caffè", g2p=True)[-1] == "ɛ"
    loaded = G2P(model=model_path, use_dict_fallback=False)
    assert loaded._dt.vectorizer.policy_locale == "it_IT"


@pytest.mark.parametrize("attach", ["constructor", "setter"])
def test_multigram_roundtrips_compatible_policy_provenance(tmp_path, attach):
    vectorizer = Vectorizer(locale="ES-es", phoneset_name="ipa")
    assert vectorizer.policy_locale == "es_MX"
    if attach == "constructor":
        model = MultigramG2P(
            max_letter_span=1,
            max_phone_span=1,
            min_phone_span=1,
            em_max_iterations=2,
            preprocessor=vectorizer,
        )
    else:
        model = MultigramG2P(
            max_letter_span=1,
            max_phone_span=1,
            min_phone_span=1,
            em_max_iterations=2,
        )
    if attach == "setter":
        model.set_preprocessor(vectorizer)
    assert model.locale == "es_ES"
    assert model.phoneset_name == "ipa"
    model.train_from_pairs([(["a"], ["a"])])
    path = tmp_path / f"spanish-{attach}.g2p"
    model.export(path)
    loaded = MultigramG2P.load(path)
    assert loaded.preprocessor is not None
    assert loaded.preprocessor.locale == "es_ES"
    assert loaded.preprocessor.policy_locale == "es_MX"

    units_path = path.with_suffix(path.suffix + ".units.json")
    metadata = json.loads(units_path.read_text())
    metadata["policy_locale"] = 3
    units_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="malformed policy_locale"):
        MultigramG2P.load(path)


@pytest.mark.parametrize(
    ("locale", "policy", "liaison"),
    [
        ("fr", "fr_FR", "#"),
        ("fr_FR", "fr_FR", "#"),
        ("fr_CA", "default", None),
        ("fra", "default", None),
    ],
)
def test_liaison_follows_selected_policy(locale, policy, liaison):
    vectorizer = Vectorizer(locale=locale, phoneset_name="ipa")
    assert vectorizer.policy_locale == policy
    assert vectorizer.liaison_pad == liaison


def test_bare_model_candidates_use_likely_region_without_compatibility(
    tmp_path, monkeypatch
):
    expected = tmp_path / "it-it-ipa.g2p.gz"
    expected.write_bytes(b"sentinel")
    seen: dict[str, str | Path | None] = {}

    def fake_init(self: G2P, model: str | Path | None = None, **kwargs: object) -> None:
        seen["model"] = model

    monkeypatch.setattr(G2P, "__init__", fake_init)
    G2P.from_lang("IT", search_paths=[tmp_path], phoneset="ipa")
    assert seen["model"] == str(expected)
    assert locale_candidates("es_ES") == ("es_ES",)


def test_generated_resolution_metadata_is_compact_and_pinned():
    data = json.loads(
        (Path(__file__).parents[1] / "phonebox/config/exemplars.json").read_text()
    )
    assert data["format"] == 2
    assert data["generator"]["icukit"] == "0.4.0"
    assert data["locale_resolution"]["likely"]["en"] == "en_Latn_US"
