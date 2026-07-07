"""Tests for the joint-multigram (n:m) EM aligner.

Validation strategy: feed it small handcrafted lexicons with known
ground-truth multigram patterns (a digraph, an n-to-many letter, a
silent-letter pattern) and check that the converged distribution
concentrates mass on the expected units.
"""

from __future__ import annotations

import math

from phonebox.core.multigram_align import MultigramAligner


def test_chxxx_digraph_discovered():
    """When ``ch`` consistently maps to a single phone, the EM should
    concentrate probability on the (('c', 'h'), ('S',)) unit and away
    from the spurious single-letter ('c',) and ('h',) decompositions.
    """
    # Tiny made-up lexicon: every ``ch`` becomes /S/, individual letters
    # behave standardly.
    pairs = [
        (list("chat"), ["S", "a", "t"]),
        (list("chin"), ["S", "i", "n"]),
        (list("rich"), ["r", "i", "S"]),
        (list("batch"), ["b", "a", "t", "S"]),
        (list("cat"), ["k", "a", "t"]),
        (list("cool"), ["k", "u", "l"]),
        (list("cap"), ["k", "a", "p"]),
        (list("hop"), ["h", "o", "p"]),
        (list("hot"), ["h", "o", "t"]),
        (list("rat"), ["r", "a", "t"]),
    ]
    aligner = MultigramAligner(
        max_letter_span=2,
        max_phone_span=1,
        min_phone_span=1,
        max_iterations=30,
        convergence_threshold=1e-5,
        verbose=False,
    )
    aligner.fit(pairs)
    cands = aligner.letter_join_candidates(min_letters=2)
    cand_dict = dict(cands)
    # The bigram ("c", "h") must dominate any other bigram
    assert ("c", "h") in cand_dict, "ch not discovered as a letter join"
    top = cands[0]
    assert top[0] == ("c", "h"), f"top join is {top[0]}, expected ('c','h')"


def test_xkernel_phone_join_discovered():
    """When letter ``x`` consistently maps to phones /k s/, the EM
    should concentrate on the (('x',), ('k', 's')) unit (a phone join).
    """
    pairs = [
        (list("exact"), ["e", "k", "s", "a", "k", "t"]),
        (list("axe"), ["a", "k", "s"]),
        (list("examine"), ["e", "k", "s", "a", "m", "i", "n"]),
        (list("taxi"), ["t", "a", "k", "s", "i"]),
        (list("ax"), ["a", "k", "s"]),
        (list("ox"), ["o", "k", "s"]),
        (list("box"), ["b", "o", "k", "s"]),
        (list("cat"), ["k", "a", "t"]),
        (list("apt"), ["a", "p", "t"]),
        (list("ant"), ["a", "n", "t"]),
    ]
    aligner = MultigramAligner(
        max_letter_span=1,
        max_phone_span=2,
        min_phone_span=1,
        max_iterations=30,
        convergence_threshold=1e-5,
    )
    aligner.fit(pairs)
    cands = aligner.phone_join_candidates(min_phones=2)
    cand_dict = dict(cands)
    assert ("k", "s") in cand_dict, "/k s/ not discovered as a phone join"


def test_silent_letter_unit_allowed_when_min_phone_span_zero():
    """When ``min_phone_span=0``, the EM must be able to learn at least
    ONE single-letter→empty-phone unit (silent letter). Note: which
    *specific* silent letter the EM converges to depends on the data —
    on a synthetic "_ame → _em" lexicon, both ``(m,) → ()`` (with e→m)
    and ``(e,) → ()`` (with m→m) are valid 1:1 alignments and EM picks
    one symmetry-breaking branch. We assert the weaker invariant: at
    least one silent-letter unit has substantial mass."""
    pairs = [
        (list("game"), ["g", "e", "m"]),
        (list("fame"), ["f", "e", "m"]),
        (list("name"), ["n", "e", "m"]),
        (list("same"), ["s", "e", "m"]),
        (list("cat"), ["k", "a", "t"]),
        (list("rat"), ["r", "a", "t"]),
    ]
    aligner = MultigramAligner(
        max_letter_span=1,
        max_phone_span=1,
        min_phone_span=0,
        max_iterations=20,
    )
    aligner.fit(pairs)
    silent_units = [(unit, p) for unit, p in aligner.q.items() if unit[1] == ()]
    assert silent_units, "no silent-letter units present"
    # The strongest silent unit should account for non-trivial mass
    top = max(silent_units, key=lambda x: x[1])
    assert top[1] > 0.05, f"strongest silent unit {top[0]} has only mass {top[1]:.4f}"


def test_loglikelihood_is_monotone_nondecreasing():
    """EM theory guarantees LL is non-decreasing; verify on a small lex."""
    pairs = [
        (list("at"), ["a", "t"]),
        (list("ate"), ["e", "t"]),  # silent e
        (list("cat"), ["k", "a", "t"]),
        (list("rat"), ["r", "a", "t"]),
        (list("bat"), ["b", "a", "t"]),
    ]
    aligner = MultigramAligner(
        max_letter_span=1, max_phone_span=1, min_phone_span=0, max_iterations=10
    )
    aligner.fit(pairs)
    history = aligner.loglik_history
    assert len(history) > 1
    # Allow a tiny epsilon for floating-point noise from per-column scaling
    for prev, curr in zip(history, history[1:]):
        assert curr >= prev - 1e-6, (
            f"LL decreased: prev={prev:.4f} curr={curr:.4f} (history={history})"
        )


def test_unit_probabilities_sum_to_one():
    """The q distribution must be a proper probability distribution
    after every M-step renormalisation."""
    pairs = [
        (list("cat"), ["k", "a", "t"]),
        (list("dog"), ["d", "o", "g"]),
    ]
    aligner = MultigramAligner(
        max_letter_span=2, max_phone_span=2, min_phone_span=1, max_iterations=5
    )
    aligner.fit(pairs)
    total = sum(aligner.q.values())
    assert math.isclose(total, 1.0, abs_tol=1e-9), (
        f"distribution sums to {total}, expected 1.0"
    )


def test_viterbi_align_recovers_known_digraph_unit():
    """After EM identifies ``c h → /S/`` as the dominant unit, the
    Viterbi best-path for "chat" should explicitly use that unit
    rather than the (suboptimal) (c, ε) + (h, S) two-step path.
    """
    pairs = [
        (list("chat"), ["S", "a", "t"]),
        (list("chin"), ["S", "i", "n"]),
        (list("rich"), ["r", "i", "S"]),
        (list("batch"), ["b", "a", "t", "S"]),
        (list("cat"), ["k", "a", "t"]),
        (list("cap"), ["k", "a", "p"]),
        (list("hot"), ["h", "o", "t"]),
        (list("hen"), ["h", "e", "n"]),
    ]
    aligner = MultigramAligner(
        max_letter_span=2, max_phone_span=1, min_phone_span=1, max_iterations=20
    )
    aligner.fit(pairs)
    units = aligner.viterbi_align(list("chat"), ["S", "a", "t"])
    assert units is not None, "no Viterbi path found"
    # Expected best path: [(c h, S), (a, a), (t, t)]
    assert units[0] == (("c", "h"), ("S",)), (
        f"first unit was {units[0]}, expected ('c h', 'S')"
    )
    assert len(units) == 3, f"expected 3 units, got {units}"


def test_segment_letters_uses_letter_marginal():
    """segment_letters must work without phone information. Trained on
    a 'ch is a digraph' lexicon, it should segment 'chap' as [ch, a, p]
    using marginalised letter probabilities.
    """
    pairs = [
        (list("chat"), ["S", "a", "t"]),
        (list("chin"), ["S", "i", "n"]),
        (list("rich"), ["r", "i", "S"]),
        (list("batch"), ["b", "a", "t", "S"]),
        (list("cat"), ["k", "a", "t"]),
        (list("cap"), ["k", "a", "p"]),
        (list("hot"), ["h", "o", "t"]),
        (list("hen"), ["h", "e", "n"]),
    ]
    aligner = MultigramAligner(
        max_letter_span=2, max_phone_span=1, min_phone_span=1, max_iterations=20
    )
    aligner.fit(pairs)
    # Unseen word "chap"
    seg = aligner.segment_letters(list("chap"))
    assert seg is not None
    assert seg[0] == ("c", "h"), f"first segment was {seg[0]}, expected ('c','h')"


def test_segment_letters_fallback_for_unseen_letters_returns_none():
    """If an unseen letter has no marginal mass at all, segmentation
    fails — callers fall back to single-letter units in that case.
    """
    pairs = [
        (list("cat"), ["k", "a", "t"]),
        (list("hat"), ["h", "a", "t"]),
    ]
    aligner = MultigramAligner(
        max_letter_span=2, max_phone_span=1, min_phone_span=1, max_iterations=5
    )
    aligner.fit(pairs)
    # 'z' was never seen
    seg = aligner.segment_letters(list("zoo"))
    assert seg is None
