"""
Joint-multigram (n:m) EM alignment.

This is the Bisani & Ney (2008) "joint-sequence model" — same idea as
Sequitur and the alignment step in Phonetisaurus. Where the existing
``EMAlign`` is 1:1 (one letter per phone, with epsilons on either side),
this aligner treats each alignment *unit* as a variable-size pair
``(letter-subseq, phone-subseq)``. EM iterates directly over the
distribution of units, so the high-probability multi-letter or multi-
phone units fall out as first-class output — these are exactly the
``letters`` / ``phones`` join candidates we want to feed back into the
locale ``config.json``.

The implementation is pure Python with per-entry alpha/beta tables. The
"sentences" here are single words (a handful of units), so alpha/beta run
in raw probability space — no per-column rescaling — and rely on float64's
range plus the ``EPS`` unit-probability floor rather than log-domain DP. For
the six European IPA lexicons we use as references, each EM iteration runs
in a few seconds for English-class lexicons and a few minutes for the
French/German 300k+ entry ones; convergence is usually 10-20 iterations.

Typical usage::

    pairs = [(list("chat"), ["ʃ", "a"]), (list("rich"), ["ɹ", "ɪ", "tʃ"]), ...]
    aligner = MultigramAligner(max_letter_span=3, max_phone_span=2)
    aligner.fit(pairs)
    for (letters, phones), prob in aligner.top_units(min_letters=2):
        print(letters, "->", phones, prob)

References:
    Bisani, M. & Ney, H. (2008). "Joint-sequence models for
    grapheme-to-phoneme conversion." Speech Communication 50:434-451.
"""

from __future__ import annotations

import math
import signal
from collections import defaultdict
from collections.abc import Iterable
from multiprocessing import Pool, cpu_count
from time import time

from ..constants import (
    CHUNKS_PER_WORKER,
    DEFAULT_MULTIGRAM_MIN_UNIT_MASS,
    MIN_CHUNK_SIZE,
    PARALLEL_ENTRY_THRESHOLD,
)
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


# A multigram unit is a pair (letter_tuple, phone_tuple). Both are
# Python tuples so they're hashable and interned-cheap by CPython for
# small strings, making dict lookups fast.
Unit = tuple[tuple[str, ...], tuple[str, ...]]

# Floor below which a unit probability is treated as zero, so dead
# transitions are skipped. alpha/beta accumulate raw products (no column
# rescaling); float64's ~1e-308 range covers the short single-word paths
# this aligner sees.
EPS = 1e-300


def _worker_init() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _enum_outgoing(
    letters: tuple[str, ...],
    phones: tuple[str, ...],
    max_l: int,
    max_p: int,
    min_p: int,
):
    N, M = len(letters), len(phones)
    for i in range(N + 1):
        for j in range(M + 1):
            if i == N and j == M:
                continue
            for l in range(1, max_l + 1):
                if i + l > N:
                    break
                a = letters[i : i + l]
                for m in range(min_p, max_p + 1):
                    if j + m > M:
                        break
                    b = phones[j : j + m]
                    yield i, j, l, m, (a, b)


def _forward(
    letters: tuple[str, ...],
    phones: tuple[str, ...],
    q: dict[Unit, float],
    max_l: int,
    max_p: int,
    min_p: int,
) -> tuple[list[list[float]], float]:
    N, M = len(letters), len(phones)
    alpha = [[0.0] * (M + 1) for _ in range(N + 1)]
    alpha[0][0] = 1.0
    # Visit cells in anti-diagonal order (i + j == diag): a unit advances i by
    # |a| >= 1 and/or j by |b|, always increasing i + j, so every cell is
    # finalized before any later cell reads from it.
    for diag in range(N + M + 1):
        cells = [
            (i, diag - i)
            for i in range(max(0, diag - M), min(N, diag) + 1)
            if 0 <= diag - i <= M
        ]
        for i, j in cells:
            if alpha[i][j] == 0.0:
                continue
            a_i = alpha[i][j]
            for l in range(1, max_l + 1):
                if i + l > N:
                    break
                a = letters[i : i + l]
                for m in range(min_p, max_p + 1):
                    if j + m > M:
                        break
                    b = phones[j : j + m]
                    prob = q.get((a, b), 0.0)
                    if prob > EPS:
                        alpha[i + l][j + m] += a_i * prob
    total = alpha[N][M]
    if total <= 0.0:
        return alpha, float("-inf")
    return alpha, math.log(total)


def _backward(
    letters: tuple[str, ...],
    phones: tuple[str, ...],
    q: dict[Unit, float],
    max_l: int,
    max_p: int,
    min_p: int,
) -> list[list[float]]:
    N, M = len(letters), len(phones)
    beta = [[0.0] * (M + 1) for _ in range(N + 1)]
    beta[N][M] = 1.0
    for diag in range(N + M - 1, -1, -1):
        cells = [
            (i, diag - i)
            for i in range(max(0, diag - M), min(N, diag) + 1)
            if 0 <= diag - i <= M
        ]
        for i, j in cells:
            total = 0.0
            for l in range(1, max_l + 1):
                if i + l > N:
                    break
                a = letters[i : i + l]
                for m in range(min_p, max_p + 1):
                    if j + m > M:
                        break
                    b = phones[j : j + m]
                    prob = q.get((a, b), 0.0)
                    if prob > EPS:
                        total += prob * beta[i + l][j + m]
            beta[i][j] = total
    return beta


def _expected_counts_for_pair(
    letters: tuple[str, ...],
    phones: tuple[str, ...],
    q: dict[Unit, float],
    max_l: int,
    max_p: int,
    min_p: int,
) -> tuple[float, bool, dict[Unit, float]]:
    alpha, ll_e = _forward(letters, phones, q, max_l, max_p, min_p)
    if ll_e == float("-inf"):
        return 0.0, True, {}
    beta = _backward(letters, phones, q, max_l, max_p, min_p)
    total = alpha[len(letters)][len(phones)]
    counts: dict[Unit, float] = defaultdict(float)
    for i, j, l, m, unit in _enum_outgoing(letters, phones, max_l, max_p, min_p):
        prob = q.get(unit, 0.0)
        if prob <= EPS:
            continue
        a_ij = alpha[i][j]
        if a_ij == 0.0:
            continue
        b_next = beta[i + l][j + m]
        if b_next == 0.0:
            continue
        counts[unit] += (a_ij * prob * b_next) / total
    return ll_e, False, dict(counts)


def _multigram_estep_chunk(
    args: tuple[
        list[tuple[tuple[str, ...], tuple[str, ...]]],
        dict[Unit, float],
        int,
        int,
        int,
    ],
) -> tuple[float, int, dict[Unit, float]]:
    chunk, q, max_l, max_p, min_p = args
    merged: dict[Unit, float] = defaultdict(float)
    ll = 0.0
    skipped = 0
    for letters, phones in chunk:
        ll_e, did_skip, counts = _expected_counts_for_pair(
            letters, phones, q, max_l, max_p, min_p
        )
        if did_skip:
            skipped += 1
            continue
        ll += ll_e
        for unit, mass in counts.items():
            merged[unit] += mass
    return ll, skipped, dict(merged)


def _viterbi_impl(
    letters: tuple[str, ...],
    phones: tuple[str, ...],
    q: dict[Unit, float],
    max_l: int,
    max_p: int,
    min_p: int,
) -> list[Unit] | None:
    N, M = len(letters), len(phones)
    neg_inf = float("-inf")
    delta = [[neg_inf] * (M + 1) for _ in range(N + 1)]
    back: list[list[tuple[int, int, Unit] | None]] = [
        [None] * (M + 1) for _ in range(N + 1)
    ]
    delta[0][0] = 0.0
    for diag in range(N + M + 1):
        cells = [
            (i, diag - i)
            for i in range(max(0, diag - M), min(N, diag) + 1)
            if 0 <= diag - i <= M
        ]
        for i, j in cells:
            d_ij = delta[i][j]
            if d_ij == neg_inf:
                continue
            for l in range(1, max_l + 1):
                if i + l > N:
                    break
                a = letters[i : i + l]
                for m in range(min_p, max_p + 1):
                    if j + m > M:
                        break
                    b = phones[j : j + m]
                    prob = q.get((a, b), 0.0)
                    if prob <= EPS:
                        continue
                    new_score = d_ij + math.log(prob)
                    if new_score > delta[i + l][j + m]:
                        delta[i + l][j + m] = new_score
                        back[i + l][j + m] = (i, j, (a, b))
    if delta[N][M] == neg_inf:
        return None
    out: list[Unit] = []
    i, j = N, M
    while (i, j) != (0, 0):
        entry = back[i][j]
        if entry is None:
            return None
        prev_i, prev_j, unit = entry
        out.append(unit)
        i, j = prev_i, prev_j
    out.reverse()
    return out


def _multigram_viterbi_chunk(
    args: tuple[
        list[tuple[tuple[str, ...], tuple[str, ...]]],
        dict[Unit, float],
        int,
        int,
        int,
    ],
) -> list[list[Unit] | None]:
    chunk, q, max_l, max_p, min_p = args
    return [
        _viterbi_impl(letters, phones, q, max_l, max_p, min_p)
        for letters, phones in chunk
    ]


def _prune_q(q: dict[Unit, float], min_mass: float) -> dict[Unit, float]:
    if min_mass <= 0.0:
        return q
    pruned = {u: p for u, p in q.items() if p >= min_mass}
    if not pruned:
        # Never let pruning empty q: an all-below-threshold table would break
        # the renormalisation below and leave the decoder with nothing. Keep
        # the single highest-mass unit as a floor.
        top = max(q.items(), key=lambda item: item[1])
        pruned = {top[0]: top[1]}
    total = sum(pruned.values())
    return {u: p / total for u, p in pruned.items()}


class MultigramAligner:
    """EM over the distribution of joint multigram units.

    Args:
        max_letter_span: maximum ``|a|`` for any unit ``(a, b)``.
            Default 2 — covers ``ch``, ``th``, ``ll``, etc.; raise to 3
            to allow ``sch``, ``eau``, ``tsch``, ``ng + h`` etc.
        max_phone_span: maximum ``|b|``. Default 2 — covers ``x → /k s/``
            and ``u → /j u/`` style phone joins.
        min_phone_span: 0 to allow silent letters (a unit may have an
            empty phone side), 1 to require at least one phone per unit.
            Default 0.
        max_iterations / convergence_threshold: standard EM stopping
            criteria. The threshold is on relative change in log-
            likelihood per iteration.
        parallel: Run the E-step (and ``viterbi_batch``) over entry
            chunks in a process pool when the lexicon is large enough.
            Off by default (same rationale as ``EMAlign.parallel``).
        num_workers: Worker count for ``parallel`` (default: cpu_count-1).
        min_unit_mass: After each M-step, drop units below this
            probability and renormalise (``0`` disables pruning).
    """

    def __init__(
        self,
        max_letter_span: int = 2,
        max_phone_span: int = 2,
        min_phone_span: int = 0,
        max_iterations: int = 30,
        convergence_threshold: float = 1e-4,
        verbose: bool = False,
        parallel: bool = False,
        num_workers: int | None = None,
        min_unit_mass: float = DEFAULT_MULTIGRAM_MIN_UNIT_MASS,
    ) -> None:
        if max_letter_span < 1:
            raise ValueError("max_letter_span must be >= 1")
        if max_phone_span < 0:
            raise ValueError("max_phone_span must be >= 0")
        if min_phone_span < 0 or min_phone_span > max_phone_span:
            raise ValueError("0 <= min_phone_span <= max_phone_span")
        self.max_l = max_letter_span
        self.max_p = max_phone_span
        self.min_p = min_phone_span
        self.max_iter = max_iterations
        self.conv = convergence_threshold
        self.verbose = verbose
        self.parallel = parallel
        self.num_workers = num_workers if num_workers else max(1, cpu_count() - 1)
        self.min_unit_mass = min_unit_mass

        # Unit probability table. Populated by `fit()`.
        self.q: dict[Unit, float] = {}
        self._loglik_history: list[float] = []

    # ----------------------------- internal helpers

    def _enum_outgoing(self, letters: tuple[str, ...], phones: tuple[str, ...]):
        return _enum_outgoing(letters, phones, self.max_l, self.max_p, self.min_p)

    def _initial_distribution(
        self, pairs: list[tuple[tuple[str, ...], tuple[str, ...]]]
    ) -> dict[Unit, float]:
        """Uniform-over-observed-units initialization.

        Every (letter_subseq, phone_subseq) pair that COULD appear in
        at least one entry's alignment gets probability ``1 / U`` where
        ``U`` is the total such count. This is a valid starting point
        for EM because every alignment has non-zero probability under
        it; concrete patterns then concentrate mass over iterations.
        """
        unit_set: set[Unit] = set()
        for letters, phones in pairs:
            for _, _, _, _, unit in self._enum_outgoing(letters, phones):
                unit_set.add(unit)
        n_units = len(unit_set)
        if n_units == 0:
            raise ValueError("no alignment units could be enumerated; check inputs")
        if self.verbose:
            logger.info(
                "MultigramAligner: initialised %d unique units (uniform 1/%d)",
                n_units,
                n_units,
            )
        return dict.fromkeys(unit_set, 1.0 / n_units)

    def _forward(
        self,
        letters: tuple[str, ...],
        phones: tuple[str, ...],
    ) -> tuple[list[list[float]], float]:
        return _forward(letters, phones, self.q, self.max_l, self.max_p, self.min_p)

    def _backward(
        self,
        letters: tuple[str, ...],
        phones: tuple[str, ...],
    ) -> list[list[float]]:
        return _backward(letters, phones, self.q, self.max_l, self.max_p, self.min_p)

    def _estep_serial(
        self, pair_list: list[tuple[tuple[str, ...], tuple[str, ...]]]
    ) -> tuple[float, int, dict[Unit, float]]:
        new_count: dict[Unit, float] = defaultdict(float)
        ll = 0.0
        skipped = 0
        for letters, phones in pair_list:
            ll_e, did_skip, counts = _expected_counts_for_pair(
                letters, phones, self.q, self.max_l, self.max_p, self.min_p
            )
            if did_skip:
                skipped += 1
                continue
            ll += ll_e
            for unit, mass in counts.items():
                new_count[unit] += mass
        return ll, skipped, dict(new_count)

    def _estep_parallel(
        self, pair_list: list[tuple[tuple[str, ...], tuple[str, ...]]]
    ) -> tuple[float, int, dict[Unit, float]]:
        n = len(pair_list)
        chunk_size = max(MIN_CHUNK_SIZE, n // (self.num_workers * CHUNKS_PER_WORKER))
        chunks = [pair_list[i : i + chunk_size] for i in range(0, n, chunk_size)]
        q_snap = self.q
        args_list = [
            (chunk, q_snap, self.max_l, self.max_p, self.min_p) for chunk in chunks
        ]
        new_count: dict[Unit, float] = defaultdict(float)
        ll = 0.0
        skipped = 0
        try:
            with Pool(processes=self.num_workers, initializer=_worker_init) as pool:
                for part_ll, part_sk, part_counts in pool.imap_unordered(
                    _multigram_estep_chunk, args_list
                ):
                    ll += part_ll
                    skipped += part_sk
                    for unit, mass in part_counts.items():
                        new_count[unit] += mass
        except KeyboardInterrupt:
            raise
        return ll, skipped, dict(new_count)

    def _m_step(self, new_count: dict[Unit, float]) -> None:
        grand_total = sum(new_count.values())
        if grand_total <= 0.0:
            raise RuntimeError(
                "EM iteration produced zero expected counts — "
                "all entries degenerate under current distribution"
            )
        self.q = {u: c / grand_total for u, c in new_count.items()}
        self.q = _prune_q(self.q, self.min_unit_mass)
        if hasattr(self, "_letter_marginal"):
            del self._letter_marginal

    # ----------------------------- public

    def fit(
        self,
        pairs: Iterable[tuple[Iterable[str], Iterable[str]]],
    ) -> MultigramAligner:
        """Run EM until convergence or max_iterations.

        ``pairs`` is any iterable of ``(letters, phones)`` where each
        side is an iterable of tokens. Tuples / lists / strings are all
        accepted; we coerce to tuples internally.
        """
        pair_list: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
            (tuple(L), tuple(P)) for L, P in pairs
        ]
        if not pair_list:
            raise ValueError("no input pairs")

        self.q = self._initial_distribution(pair_list)
        prev_ll: float | None = None
        self._loglik_history = []

        use_parallel = self.parallel and len(pair_list) > PARALLEL_ENTRY_THRESHOLD
        if self.verbose:
            logger.info(
                "MultigramAligner: %d entries, E-step %s, min_unit_mass=%g",
                len(pair_list),
                f"parallel ({self.num_workers} workers)" if use_parallel else "serial",
                self.min_unit_mass,
            )

        for it in range(self.max_iter):
            t_iter = time()
            if use_parallel:
                try:
                    ll, skipped, new_count = self._estep_parallel(pair_list)
                except Exception as exc:
                    if self.verbose:
                        logger.warning(
                            "parallel E-step failed (%s), falling back to serial", exc
                        )
                    ll, skipped, new_count = self._estep_serial(pair_list)
            else:
                ll, skipped, new_count = self._estep_serial(pair_list)

            self._m_step(new_count)
            self._loglik_history.append(ll)

            if self.verbose:
                logger.info(
                    "iter %d: LL=%.2f, %d units, %d skipped, %.1fs",
                    it + 1,
                    ll,
                    len(self.q),
                    skipped,
                    time() - t_iter,
                )

            if prev_ll is not None:
                denom = max(1.0, abs(prev_ll))
                if abs(ll - prev_ll) / denom < self.conv:
                    if self.verbose:
                        logger.info("converged after %d iterations", it + 1)
                    break
            prev_ll = ll

        return self

    def top_units(
        self,
        min_letters: int = 1,
        min_phones: int = 0,
        min_prob: float = 0.0,
    ) -> list[tuple[Unit, float]]:
        """Return units sorted by probability, optionally filtered to
        only the non-trivial (multi-letter or multi-phone) candidates.
        """
        out = [
            (u, p)
            for u, p in self.q.items()
            if p >= min_prob and len(u[0]) >= min_letters and len(u[1]) >= min_phones
        ]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def letter_join_candidates(
        self,
        min_letters: int = 2,
        min_count_fraction: float = 0.0,
    ) -> list[tuple[tuple[str, ...], float]]:
        """Letter-side join candidates: units whose letter-tuple is the
        target and whose total probability across phone-tuples is high.

        Aggregates across the phone side, so for ``c h → ʃ`` (high) and
        ``c h → k`` (medium) the candidate `("c","h")` gets summed mass.
        """
        agg: dict[tuple[str, ...], float] = defaultdict(float)
        for (letters, _phones), p in self.q.items():
            if len(letters) >= min_letters:
                agg[letters] += p
        out = [(k, v) for k, v in agg.items() if v >= min_count_fraction]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def phone_join_candidates(
        self,
        min_phones: int = 2,
        min_count_fraction: float = 0.0,
    ) -> list[tuple[tuple[str, ...], float]]:
        """Phone-side join candidates: symmetric to ``letter_join_candidates``."""
        agg: dict[tuple[str, ...], float] = defaultdict(float)
        for (_letters, phones), p in self.q.items():
            if len(phones) >= min_phones:
                agg[phones] += p
        out = [(k, v) for k, v in agg.items() if v >= min_count_fraction]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    @property
    def loglik_history(self) -> list[float]:
        """Per-iteration log-likelihood, for diagnostics."""
        return list(self._loglik_history)

    # ---------- decoding (used at training-time-to-units and at inference)

    def viterbi_align(
        self,
        letters: Iterable[str],
        phones: Iterable[str],
    ) -> list[Unit] | None:
        """Best multigram segmentation of (letters, phones) under ``self.q``."""
        return _viterbi_impl(
            tuple(letters),
            tuple(phones),
            self.q,
            self.max_l,
            self.max_p,
            self.min_p,
        )

    def viterbi_batch(
        self,
        pairs: Iterable[tuple[Iterable[str], Iterable[str]]],
        parallel: bool | None = None,
    ) -> list[list[Unit] | None]:
        """Viterbi-align many entries; optional parallel decode."""
        pair_list = [(tuple(L), tuple(P)) for L, P in pairs]
        if not pair_list:
            return []
        use_parallel = (self.parallel if parallel is None else parallel) and len(
            pair_list
        ) > PARALLEL_ENTRY_THRESHOLD
        if not use_parallel:
            return [
                _viterbi_impl(L, P, self.q, self.max_l, self.max_p, self.min_p)
                for L, P in pair_list
            ]
        n = len(pair_list)
        chunk_size = max(MIN_CHUNK_SIZE, n // (self.num_workers * CHUNKS_PER_WORKER))
        chunks = [pair_list[i : i + chunk_size] for i in range(0, n, chunk_size)]
        q_snap = self.q
        args_list = [
            (chunk, q_snap, self.max_l, self.max_p, self.min_p) for chunk in chunks
        ]
        out: list[list[Unit] | None] = []
        try:
            with Pool(processes=self.num_workers, initializer=_worker_init) as pool:
                for chunk_out in pool.imap(_multigram_viterbi_chunk, args_list):
                    out.extend(chunk_out)
        except KeyboardInterrupt:
            raise
        return out

    def segment_letters(
        self,
        letters: Iterable[str],
    ) -> list[tuple[str, ...]] | None:
        """Best-path segmentation of an unseen letter sequence using the
        letter-side marginal distribution.

        At training time we have both letters AND phones, so we can run
        ``viterbi_align`` with full information. At inference time we
        only have letters, so we segment using
        ``P(letter_subseq) = Σ_phone_subseq q(letter_subseq, phone_subseq)``
        — the marginal of the joint multigram distribution over the
        letter side. The resulting segmentation tells the downstream
        G2P which letter chunks to predict phone-subsequences for.

        Returns the ordered list of letter-tuple units, or ``None`` if
        no segmentation exists (caller should fall back to single-letter
        units). Note that ``None`` is rare: every single-letter unit
        ``((c,), some-phone-or-empty)`` typically has non-zero mass, so
        the fallback "every letter is its own segment" is always
        available.
        """
        L = tuple(letters)
        N = len(L)

        # Build letter-marginal lazily (cached on first call)
        if not hasattr(self, "_letter_marginal"):
            self._letter_marginal: dict[tuple[str, ...], float] = defaultdict(float)
            for (a, _b), p in self.q.items():
                self._letter_marginal[a] += p

        neg_inf = float("-inf")
        delta = [neg_inf] * (N + 1)
        back: list[tuple[int, tuple[str, ...]] | None] = [None] * (N + 1)
        delta[0] = 0.0
        for i in range(N + 1):
            if delta[i] == neg_inf:
                continue
            for l in range(1, self.max_l + 1):
                if i + l > N:
                    break
                a = L[i : i + l]
                p = self._letter_marginal.get(a, 0.0)
                if p <= EPS:
                    continue
                new_score = delta[i] + math.log(p)
                if new_score > delta[i + l]:
                    delta[i + l] = new_score
                    back[i + l] = (i, a)

        if delta[N] == neg_inf:
            return None

        out: list[tuple[str, ...]] = []
        i = N
        while i > 0:
            entry = back[i]
            if entry is None:
                return None
            prev_i, unit = entry
            out.append(unit)
            i = prev_i
        out.reverse()
        return out


__all__ = ["MultigramAligner", "Unit"]
