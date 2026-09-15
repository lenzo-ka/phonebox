"""The shared evaluation split never lets a test spelling into training."""

from phonebox.experiments.split import spelling_group, split_lexicon

PAIRS = [
    ("read", ["R", "IY", "D"]),
    ("read", ["R", "EH", "D"]),
    ("READ", ["R", "IY", "D"]),
    ("reed", ["R", "IY", "D"]),
    ("red", ["R", "EH", "D"]),
    ("lead", ["L", "IY", "D"]),
    ("lead", ["L", "EH", "D"]),
    ("Éclair", ["E", "K", "L", "EH", "R"]),
    ("éclair", ["E", "K", "L", "EH", "R"]),
    ("bass", ["B", "AE", "S"]),
    ("bass", ["B", "EY", "S"]),
    ("base", ["B", "EY", "S"]),
]


def test_variants_and_case_aliases_of_one_spelling_stay_on_one_side():
    for seed in range(8):
        test, train = split_lexicon(PAIRS, seed=seed, test_fraction=0.5, max_test=4)
        assert sorted(test + train) == sorted(PAIRS)
        assert {spelling_group(w) for w, _ in test}.isdisjoint(
            spelling_group(w) for w, _ in train
        )
        assert {w for w, _ in test}.isdisjoint(w for w, _ in train)


def test_limits_count_spelling_groups_and_are_deterministic():
    first = split_lexicon(PAIRS, seed=42, test_fraction=0.5, max_test=2)
    again = split_lexicon(PAIRS, seed=42, test_fraction=0.5, max_test=2)
    assert first == again
    test, _ = first
    assert len({spelling_group(w) for w, _ in test}) == 2
    # A pair-level shuffle could put both "bass" readings on opposite sides;
    # the grouped split cannot.
    assert not (
        any(w == "bass" for w, _ in test) and any(w == "bass" for w, _ in first[1])
    )
