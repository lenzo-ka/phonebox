"""Exact complete-path oracle and opt-in CART lattice integration."""

import math

import pytest

from phonebox.core.cart_lattice import CartDecompositionLattice
from phonebox.core.g2p_model import G2PDecisionTree
from phonebox.core.multigram_align import MultigramAligner

UNITS = [(("a", "b"), ("X",)), (("a",), ("A",)), (("b",), ("B",))]


def lattice(units=UNITS):
    return CartDecompositionLattice(units, join_char="₊", epsilon="∅")


def exhaustive(letters, distributions, units):
    paths = []

    def visit(position, phones, score):
        if position == len(letters):
            paths.append((score, phones))
            return
        for left, right in units:
            end = position + len(left)
            if tuple(letters[position:end]) != left:
                continue
            labels = ["₊".join(right) if right else "∅"] + ["∅"] * (len(left) - 1)
            probability = math.prod(
                distributions[position + offset].get(target, 0)
                for offset, target in enumerate(labels)
            )
            if probability:
                visit(end, phones + list(right), score + math.log(probability))

    visit(0, [], 0)
    return max(paths, key=lambda path: path[0])[1] if paths else None


@pytest.mark.parametrize(
    "anchor,last", [(0.6, 0.8), (0.9, 0.1), (0.3, 0.6), (0.7, 0.4)]
)
def test_context_selects_merged_or_unmerged_complete_path(anchor, last):
    distributions = [{"X": anchor, "A": 1 - anchor}, {"B": last, "∅": 1 - last}]
    assert lattice().decode(list("ab"), distributions) == exhaustive(
        list("ab"), distributions, UNITS
    )


def test_lattice_prevents_incoherent_greedy_combination():
    distributions = [{"X": 0.6, "A": 0.4}, {"B": 0.8, "∅": 0.2}]
    assert [max(d, key=d.get) for d in distributions] == ["X", "B"]
    assert lattice().decode(list("ab"), distributions) == ["A", "B"]


def test_fractional_leaf_roundoff_preserves_complete_path_ranking():
    distributions = [
        {"X": 0.6 * (1 - 5e-9), "A": 0.4 * (1 - 5e-9)},
        {"B": 0.8 * (1 - 2e-8), "∅": 0.2 * (1 - 2e-8)},
    ]
    assert lattice().decode(list("ab"), distributions) == ["A", "B"]


def test_empty_silent_unsupported_and_zero_probability():
    assert lattice().decode([], []) == []
    assert lattice().decode(["z"], ["∅"]) is None
    assert lattice().decode(list("ab"), ["X", "B"]) is None
    assert lattice([(("a",), ())]).decode(["a"], ["∅"]) == []


def test_inventory_is_owned_and_duplicate_order_is_stable():
    units = [[["a", "b"], ["X"]], [["a"], ["A"]], [["b"], ["B"]]]
    prepared = lattice(units + units)
    units[0][1][0] = "changed"
    assert len(prepared.units) == 3
    assert prepared.decode(
        list("ab"), [{"X": 0.5, "A": 0.5}, {"B": 0.5, "∅": 0.5}]
    ) == ["X"]


@pytest.mark.parametrize(
    "distribution", [{"X": float("nan")}, {"X": -1}, {"X": True}, {"X": 0.5}]
)
def test_malformed_distribution_is_rejected(distribution):
    with pytest.raises(ValueError):
        lattice().decode(list("ab"), [distribution, {"B": 1}])


def test_real_tree_multiphone_and_inventory_round_trip(tmp_path):
    model = G2PDecisionTree(
        locale=None, phoneset_name="ipa", width=3, cased=True, use_dict_fallback=False
    )
    aligner = MultigramAligner(max_letter_span=2, max_phone_span=2, min_unit_mass=0)
    aligner.q = {(("x",), ("K", "S")): 0.5, (("e",), ()): 0.5}
    model.train_decomposition_from_pairs(
        [(list("xe"), ["K", "S"])] * 10, aligner=aligner
    )
    prepared = model.prepare_decomposition_lattice()
    assert model.pronounce_lattice("xe", lattice=prepared) == ["K", "S"]
    path = tmp_path / "model.g2p.gz"
    model.export(str(path))
    loaded = G2PDecisionTree(model=str(path), use_dict_fallback=False)
    assert loaded.pronounce_lattice("xe") == ["K", "S"]
    assert loaded.decomposition_units == model.decomposition_units


def test_earlier_models_require_explicit_support_and_default_is_unchanged(monkeypatch):
    model = G2PDecisionTree(
        locale=None, phoneset_name="ipa", cased=True, use_dict_fallback=False
    )
    with pytest.raises(ValueError, match="inventory"):
        model.prepare_decomposition_lattice()
    monkeypatch.setattr(
        model,
        "_predict_distributions",
        lambda _: [{"X": 0.6, "A": 0.4}, {"B": 0.8, "∅": 0.2}],
    )
    assert model.pronounce_lattice(
        "ab", lattice=model.prepare_decomposition_lattice(UNITS)
    ) == ["A", "B"]


@pytest.mark.parametrize(
    "units", [["ab"], [[["a"], "B"]], [[[], ["A"]]], [[["a"], ["A₊B"]]]]
)
def test_malformed_inventory_is_rejected(units):
    with pytest.raises(ValueError):
        lattice(units)
