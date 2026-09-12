"""Source provenance, ranking and serialization controls for lexicon review."""

import json
import math
from typing import cast

import pytest

from phonebox.lexicon import parse_dict_entry, parse_dict_line
from phonebox.pronunciation_analysis import (
    PronunciationScorer,
    format_lexicon_review,
    review_lexicon,
)
from phonebox.pronunciation_scoring import PronunciationScore, ScoreMethod


class Scorer:
    def score_pronunciation_details(self, word, phones, method="geometric"):
        effective = tuple(p.rstrip("012") for p in phones)
        score = {"A": 0.9, "B": 0.1, "C": 0.1}[effective[0]]
        return PronunciationScore(
            score, score, math.log(score), 1, effective, True, cast(ScoreMethod, method)
        )


def test_shared_parser_retains_occurrence_and_legacy_view():
    raw = "WORD(7)\tA1 B # comment\n"
    entry = parse_dict_entry(raw, line_number=9)
    assert entry.to_dict() == {
        "word": "WORD",
        "phones": ["A1", "B"],
        "label": "WORD(7)",
        "variant": 7,
        "line_number": 9,
    }
    assert parse_dict_line(raw) == ("WORD", ["A1", "B"])
    assert parse_dict_entry(";;; comment") is None


def test_effective_dedup_preserves_all_origins_and_dense_stable_ranks():
    result = review_lexicon(
        Scorer(),
        ["word(7) B1\n", "word A1\n", "word(9) A2\n", "word(3) C1\n"],
        order="variants",
    )
    assert result.source_entries == 4 and result.unique_variants == 3
    assert [(r.entry, r.phones) for r in result.records] == [
        ("word", ("A",)),
        ("word(2)", ("B",)),
        ("word(3)", ("C",)),
    ]
    assert [origin.label for origin in result.records[0].origins] == ["word", "word(9)"]
    assert result.records[0].origins[1].phones == ("A2",)
    selected = review_lexicon(Scorer(), ["word B1", "word A1"], threshold=0.2)
    assert selected.records[0].rank == 2


def test_worst_ties_source_order_mapping_and_numeric_formats():
    result = review_lexicon(
        Scorer(), ["ABC C1", "word B1", "ABC A1"], phone_mapping={"C1": "B1"}
    )
    assert [r.word for r in result.records] == ["ABC", "word", "ABC"]
    envelope = json.loads(next(format_lexicon_review(result, format="json")))
    assert isinstance(envelope["records"][0]["score"], float)
    tsv = list(format_lexicon_review(result, header=False))
    assert tsv[0].split("\t")[:3] == ["0.1", "ABC(2)", "B"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "min"},
        {"threshold": float("nan")},
        {"limit": -1},
        {"order": "random"},
        {"phone_mapping": {}, "phone_transform": list},
    ],
)
def test_invalid_review_options(kwargs):
    with pytest.raises(ValueError):
        review_lexicon(Scorer(), ["word A"], **kwargs)


def test_product_underflow_orders_log_likelihood_and_dict_rejects_filters():
    class UnderflowScorer:
        def score_pronunciation_details(self, word, phones, method="product"):
            log = {"A": -1000.0, "B": -900.0}[phones[0]]
            return PronunciationScore(
                0.0, 0.0, log, 100, tuple(phones), True, "product"
            )

    result = review_lexicon(
        UnderflowScorer(), ["word A", "word B"], method="product", order="variants"
    )
    assert result.records[0].phones == ("B",)
    assert all(record.score == 0.0 for record in result.records)
    selected = review_lexicon(
        Scorer(), ["word B", "word A"], threshold=0.2, order="variants"
    )
    with pytest.raises(ValueError, match="unfiltered"):
        list(format_lexicon_review(selected, format="dict"))


def test_actual_train_review_cli_roundtrip_and_unix_sort(tmp_path):
    import subprocess

    from phonebox import G2P, train_g2p
    from phonebox.cli.main import main
    from phonebox.pronunciation_analysis import review_lexicon_file

    training = tmp_path / "training.dict"
    training.write_text(
        "cat K AE1 T\ncat(2) K AH1 T\nbat B AE1 T\n"
        "hat HH AE1 T\nmat M AE1 T\nrat R AE1 T\nfat F AE1 T\n",
        encoding="utf-8",
    )
    model = tmp_path / "model.g2p.gz"
    result = train_g2p(
        training, locale="en", phoneset="cmu", width=1, prune=False, output=model
    )
    source = tmp_path / "review.dict"
    source.write_text(
        "cat(9) K AH1 T\ncat K AE1 T\ncat(8) K AE2 T # same after explicit map\n",
        encoding="utf-8",
    )
    mapping = {"AE2": "AE1"}
    reviewed = review_lexicon_file(
        result.model, source, phone_mapping=mapping, order="variants"
    )
    assert reviewed.records[0].phones == ("K", "AE1", "T")
    assert reviewed.records[0].score > reviewed.records[1].score
    assert [origin.label for origin in reviewed.records[0].origins] == ["cat", "cat(8)"]
    loaded = review_lexicon_file(
        G2P(model=model), source, phone_mapping=mapping, order="variants"
    )
    assert loaded.to_dict() == reviewed.to_dict()
    mapfile = tmp_path / "map.json"
    mapfile.write_text(json.dumps(mapping))
    tsv = tmp_path / "review.tsv"
    args = [
        "dict",
        "review",
        str(source),
        "-m",
        str(model),
        "--phone-map",
        str(mapfile),
    ]
    assert main([*args, "--no-header", "-o", str(tsv)]) == 0
    rows = tsv.read_text().splitlines()
    sorted_rows = subprocess.check_output(
        ["sort", "-s", "-g", "-k1,1", str(tsv)], text=True
    ).splitlines()
    assert sorted_rows == rows
    assert float(rows[0].split("\t")[0]) < float(rows[1].split("\t")[0])
    dictionary = tmp_path / "ranked.dict"
    assert main([*args, "--format", "dict", "-o", str(dictionary)]) == 0
    cut = subprocess.check_output(["cut", "-f2,3", str(tsv)], text=True)
    assert set(cut.splitlines()) == set(dictionary.read_text().splitlines())
    assert [
        parse_dict_line(line)[1] for line in dictionary.read_text().splitlines()
    ] == [["K", "AE1", "T"], ["K", "AH1", "T"]]
    original = source.read_bytes()
    for alias in (source, tmp_path / "alias.dict"):
        if alias != source:
            alias.hardlink_to(source)
        assert main([*args, "-o", str(alias)]) == 2
        assert source.read_bytes() == original
    protected = tmp_path / "existing.dict"
    protected.write_text("keep")
    assert (
        main([*args, "--format", "dict", "--threshold", "0.5", "-o", str(protected)])
        == 2
    )
    assert protected.read_text() == "keep"


@pytest.mark.parametrize("remove_stress", [False, True])
def test_saved_stress_controls_effective_dedup_and_retains_source(
    tmp_path, remove_stress
):
    from phonebox import train_g2p

    training = tmp_path / "stress.dict"
    training.write_text("cat K AE1 T\n", encoding="utf-8")
    trained = train_g2p(
        training,
        locale="en",
        phoneset="cmu",
        width=1,
        remove_stress=remove_stress,
        prune=False,
    )
    result = review_lexicon(trained.model, ["cat K AE1 T", "cat(8) K AE2 T"])
    assert result.unique_variants == (1 if remove_stress else 2)
    origins = [origin for record in result.records for origin in record.origins]
    assert {origin.phones[1] for origin in origins} == {"AE1", "AE2"}
    if remove_stress:
        assert result.records[0].phones == ("K", "AE", "T")


def test_cli_bad_inputs_and_model_mapping_symlink_aliases_preserve_bytes(
    tmp_path, capsys
):
    from phonebox.cli.main import main

    source = tmp_path / "words.dict"
    model = tmp_path / "model.g2p.gz"
    mapping = tmp_path / "mapping.json"
    source.write_text("cat K AE1 T\n")
    model.write_bytes(b"not a model; alias must reject before loading it")
    mapping.write_text("{}")
    args = [
        "dict",
        "review",
        str(source),
        "-m",
        str(model),
        "--phone-map",
        str(mapping),
    ]
    for protected in (source, model, mapping):
        alias = tmp_path / "alias"
        alias.symlink_to(protected)
        before = protected.read_bytes()
        assert main([*args, "-o", str(alias)]) == 2
        assert "review output must differ" in capsys.readouterr().err
        assert protected.read_bytes() == before
        alias.unlink()
    output = tmp_path / "output.dict"
    output.write_text("keep")
    assert main([*args, "--format", "dict", "--order", "worst", "-o", str(output)]) == 2
    assert output.read_text() == "keep"
    assert main([*args, "-o", str(output)]) == 2
    assert output.read_text() == "keep"
    capsys.readouterr()


def test_empty_effective_pronunciation_is_reviewable_but_not_dictionary(
    tmp_path, capsys
):
    from phonebox import train_g2p
    from phonebox.cli.main import main

    training = tmp_path / "training.dict"
    training.write_text("a A1\n")
    model = tmp_path / "model.g2p.gz"
    trained = train_g2p(
        training,
        locale="en",
        phoneset="cmu",
        remove_stress=True,
        prune=False,
        output=model,
    )
    source = tmp_path / "empty-after-cooking.dict"
    source.write_text("a 1\n")
    result = review_lexicon(
        trained.model, source.read_text().splitlines(), order="variants"
    )
    assert result.records[0].phones == () and not result.records[0].supported
    assert (
        json.loads(next(format_lexicon_review(result, format="json")))["records"][0][
            "phones"
        ]
        == []
    )
    with pytest.raises(ValueError, match="effective pronunciations"):
        list(format_lexicon_review(result, format="dict"))
    output = tmp_path / "existing.dict"
    output.write_text("keep")
    assert (
        main(
            [
                "dict",
                "review",
                str(source),
                "-m",
                str(model),
                "--format",
                "dict",
                "-o",
                str(output),
            ]
        )
        == 2
    )
    assert output.read_text() == "keep"
    assert "effective pronunciations" in capsys.readouterr().err


def test_multigram_capability_is_rejected_clearly():
    from phonebox import MultigramG2P

    with pytest.raises(ValueError, match="multigram scoring is unsupported"):
        review_lexicon(cast(PronunciationScorer, MultigramG2P()), ["a A"])


@pytest.mark.parametrize("target", [[], None, 3])
def test_cli_invalid_mapping_preserves_output(tmp_path, capsys, target):
    from phonebox import train_g2p
    from phonebox.cli.main import main
    from phonebox.dictionary import phone_mapping_transform

    source = tmp_path / "words.dict"
    source.write_text("a A\n")
    model = tmp_path / "model.g2p.gz"
    train_g2p(source, locale="en", prune=False, output=model)
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({"A": target}))
    output = tmp_path / "review.tsv"
    output.write_text("keep")
    with pytest.raises(ValueError, match="phone mapping values"):
        phone_mapping_transform({"A": target})
    assert (
        main(
            [
                "dict",
                "review",
                str(source),
                "-m",
                str(model),
                "--phone-map",
                str(mapping),
                "-o",
                str(output),
            ]
        )
        == 2
    )
    assert output.read_text() == "keep"
    assert "phone mapping values" in capsys.readouterr().err


def test_cli_rejects_actual_multigram_sidecars(tmp_path, capsys):
    from phonebox import train_multigram
    from phonebox.cli.main import main

    source = tmp_path / "words.dict"
    source.write_text("a A\nb B\n")
    result = train_multigram(
        source, locale="en", em_iterations=1, output=tmp_path / "mg"
    )
    output = tmp_path / "review.tsv"
    output.write_text("keep")
    for model in (result.units_path, result.lm_path):
        assert (
            main(["dict", "review", str(source), "-m", str(model), "-o", str(output)])
            == 2
        )
        assert output.read_text() == "keep"
        assert "multigram scoring is unsupported" in capsys.readouterr().err


@pytest.mark.parametrize(
    "data",
    [
        {},
        [],
        None,
        3,
        {"model": None},
        {"model": []},
        {"model": {}},
        {"model": {"A": "bad"}},
        {"model": "A", "metadata": []},
    ],
)
def test_invalid_model_shapes_library_and_cli(tmp_path, capsys, data):
    from phonebox import G2P
    from phonebox.cli.main import main

    model = tmp_path / "invalid.json"
    model.write_text(json.dumps(data))
    source = tmp_path / "words.dict"
    source.write_text("a A\n")
    output = tmp_path / "review.tsv"
    output.write_text("keep")
    with pytest.raises(ValueError, match="Invalid CART"):
        G2P(model=model)
    assert (
        main(["dict", "review", str(source), "-m", str(model), "-o", str(output)]) == 2
    )
    assert output.read_text() == "keep"
    assert "Invalid CART" in capsys.readouterr().err


def test_renamed_multigram_model_library_and_cli(tmp_path, capsys):
    from phonebox import G2P, train_multigram
    from phonebox.cli.main import main

    source = tmp_path / "words.dict"
    source.write_text("a A\nb B\n")
    result = train_multigram(
        source, locale="en", em_iterations=1, output=tmp_path / "mg"
    )
    assert result.units_path is not None
    copied = tmp_path / "renamed.json"
    copied.write_bytes(result.units_path.read_bytes())
    linked = tmp_path / "linked.json"
    linked.symlink_to(result.units_path)
    output = tmp_path / "review.tsv"
    output.write_text("keep")
    for model in (copied, linked):
        with pytest.raises(ValueError, match="Invalid CART"):
            G2P(model=model)
        assert (
            main(["dict", "review", str(source), "-m", str(model), "-o", str(output)])
            == 2
        )
        assert output.read_text() == "keep"
        assert "Invalid CART" in capsys.readouterr().err


@pytest.mark.parametrize(
    "feature,names", [(0, []), (0, ["letter"]), ("letter", ["letter"])]
)
def test_official_cartlet_decision_export_load_score(tmp_path, feature, names):
    from cartlet import DecisionTree as Cartlet

    from phonebox import G2P

    tree = Cartlet(feature_names=names)
    tree.model = [feature, "=", "a", "A", {"B": 0.4, "C": 0.6}]
    model = tmp_path / "official.json"
    tree.export(str(model), metadata={"width": 1, "locale": "en_US", "phoneset": "ipa"})
    oracle = Cartlet()
    oracle.load_model(str(model))
    assert oracle.predict(["a"]) == "A"
    loaded = G2P(model=model)
    assert loaded.pronounce("a") == ["A"]
    assert loaded.score_pronunciation_details("a", ["A"]).score == 1.0
