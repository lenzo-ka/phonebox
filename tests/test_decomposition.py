"""Independent complete-path oracle and real CART training controls."""

import math
from collections import defaultdict

import pytest

from phonebox.core.decomposition import (
    decomposition_targets,
    prepare_decomposition_vectors,
)
from phonebox.core.g2p_model import G2PDecisionTree
from phonebox.core.multigram_align import MultigramAligner
from phonebox.training import train_g2p


def model():
    return G2PDecisionTree(
        locale=None, phoneset_name="ipa", cased=True, width=3, use_dict_fallback=False
    )


def aligner(q):
    a = MultigramAligner(max_letter_span=2, max_phone_span=2, min_unit_mass=0)
    a.q = q
    return a


def oracle(letters, phones, q, v):
    paths = []

    def visit(i, j, labels, probability):
        if i == len(letters) and j == len(phones):
            paths.append((labels, probability))
            return
        for (left, right), weight in q.items():
            if (
                tuple(letters[i : i + len(left)]) == left
                and tuple(phones[j : j + len(right)]) == right
            ):
                target = v.join_char.join(right) if right else v.epsilon
                visit(
                    i + len(left),
                    j + len(right),
                    labels + [target] + [v.epsilon] * (len(left) - 1),
                    probability * weight,
                )

    visit(0, 0, [], 1.0)
    total = sum(weight for _, weight in paths)
    result: list[defaultdict[str, float]] = [defaultdict(float) for _ in letters]
    for labels, weight in paths:
        assert v.uncook(labels) == phones
        for position, label in enumerate(labels):
            result[position][label] += weight / total
    return result


@pytest.mark.parametrize(
    "letters,phones",
    [("ab", ["A", "B"]), ("aba", ["A", "B"]), ("éβ", ["A"]), ("a", ["A", "B"])],
)
def test_posterior_matches_enumeration_and_preserves_position_mass(letters, phones):
    v = model().vectorizer
    q: dict[tuple[tuple[str, ...], tuple[str, ...]], float] = {}
    for i in range(len(letters)):
        for length in (1, 2):
            left = tuple(letters[i : i + length])
            if len(left) != length:
                continue
            for j in range(len(phones) + 1):
                for width in (0, 1, 2):
                    right = tuple(phones[j : j + width])
                    if len(right) == width:
                        q[left, right] = 0.1 + (len(q) % 7) / 20
    expected = oracle(list(letters), phones, q, v)
    actual = decomposition_targets(list(letters), phones, aligner(q), v)
    for got, want in zip(actual, expected, strict=True):
        assert got == pytest.approx(want, abs=1e-12)
        assert math.fsum(got.values()) == pytest.approx(1)


def test_merged_and_unmerged_paths_retain_likelihood_weights():
    m = model()
    a = aligner(
        {(("a", "b"), ("A", "B")): 0.5, (("a",), ("A",)): 0.25, (("b",), ("B",)): 0.25}
    )
    distributions = decomposition_targets(list("ab"), ["A", "B"], a, m.vectorizer)
    assert distributions[0] == pytest.approx({"A": 1 / 9, "A₊B": 8 / 9})
    assert distributions[1] == pytest.approx({"B": 1 / 9, "∅": 8 / 9})
    prepared = prepare_decomposition_vectors(
        [(list("ab"), ["A", "B"])] * 3, m.vectorizer, a
    )
    assert sum(prepared.counts) == pytest.approx(6)
    assert prepared.metadata["ambiguous_positions"] == 6


def test_long_low_probability_paths_do_not_underflow():
    a = aligner({(("a",), ("A",)): 1e-200})
    result = decomposition_targets(list("a" * 10), ["A"] * 10, a, model().vectorizer)
    assert result == [{"A": 1.0}] * 10


def test_unsupported_and_reserved_pairs_have_explicit_admission():
    m = model()
    a = aligner({(("a",), ("A",)): 0.5})
    assert decomposition_targets(list("b"), ["B"], a, m.vectorizer) is None
    prepared = prepare_decomposition_vectors(
        [(list("a"), ["A"]), (list("b"), ["B"]), (list("a"), ["A₊B"])], m.vectorizer, a
    )
    assert prepared.metadata["retained_entries"] == 1
    assert prepared.metadata["skip_reasons"] == {
        "unsupported_alignment": 1,
        "reserved_phone_syntax": 1,
    }


def test_real_tree_and_serialization_multiphone_and_silent(tmp_path):
    m = model()
    a = aligner({(("x",), ("K", "S")): 0.5, (("e",), ()): 0.5})
    metrics = m.train_decomposition_from_pairs(
        [(list("xe"), ["K", "S"])] * 10, aligner=a
    )
    assert metrics["decomposition"]["retained_entries"] == 10
    assert m.pronounce("xe") == ["K", "S"]
    path = tmp_path / "model.g2p.gz"
    m.export(str(path))
    loaded = G2PDecisionTree(model=str(path), use_dict_fallback=False)
    assert loaded.pronounce("xe") == ["K", "S"]


def test_explicit_source_pairs_build_exceptions_without_em(tmp_path):
    m = model()
    m.use_dict_fallback = True
    m.em = None
    a = aligner({(("a",), ("A",)): 0.5, (("a",), ("B",)): 0.5})
    m.train_decomposition_from_pairs(
        [(list("a"), ["A"]), (list("a"), ["B"])], aligner=a
    )
    assert m.exceptions["a"] in (["A"], ["B"])


def test_public_workflow_and_split_guard(tmp_path):
    dictionary = tmp_path / "lexicon.dict"
    dictionary.write_text("x\tK S\nxe\tK S\n")
    output = tmp_path / "model.g2p.gz"
    result = train_g2p(
        dictionary,
        locale="en",
        alignment_method="decomposition-posterior",
        prune=False,
        output=output,
        decomposition_iterations=3,
        use_dict_fallback=False,
    )
    assert result.alignments_path is None
    assert result.metrics["decomposition"]["supplied_entries"] == 2
    assert output.exists()
    with pytest.raises(ValueError, match="split spellings"):
        train_g2p(dictionary, locale="en", alignment_method="decomposition-posterior")


def test_actual_cli_matches_api_and_retains_decomposition_metadata(tmp_path):
    from phonebox.cli.main import main

    dictionary = tmp_path / "lexicon.dict"
    dictionary.write_text("x\tK S\nxe\tK S\n")
    output = tmp_path / "cli.g2p.gz"
    assert (
        main(
            [
                "train",
                "--locale",
                "en",
                "--lexicon",
                str(dictionary),
                "--output",
                str(output),
                "--alignment-method",
                "decomposition-posterior",
                "--no-prune",
                "--decomposition-iterations",
                "3",
            ]
        )
        == 0
    )
    loaded = G2PDecisionTree(model=str(output))
    assert (
        loaded.decomposition_metadata["alignment_method"] == "decomposition-posterior"
    )
    again = tmp_path / "again.g2p.gz"
    loaded.export(str(again))
    reloaded = G2PDecisionTree(model=str(again))
    assert reloaded.decomposition_metadata == loaded.decomposition_metadata
    api = train_g2p(
        dictionary,
        locale="en",
        alignment_method="decomposition-posterior",
        prune=False,
        decomposition_iterations=3,
    )
    assert loaded.pronounce("xe") == api.model.pronounce("xe")


def test_duplicate_pairs_reuse_posterior_and_spelling_contexts(monkeypatch):
    import phonebox.core.decomposition as module

    real = module.decomposition_targets
    calls = []

    def spy(*args):
        calls.append(args[:2])
        return real(*args)

    monkeypatch.setattr(module, "decomposition_targets", spy)
    m = model()
    a = aligner({(("a",), ("A",)): 0.5, (("a",), ("B",)): 0.5})
    prepared = prepare_decomposition_vectors(
        [(list("a"), ["A"])] * 4 + [(list("a"), ["B"])], m.vectorizer, a
    )
    assert len(calls) == 2
    assert prepared.metadata["context_spellings"] == 1
    assert sum(prepared.counts) == pytest.approx(5)
