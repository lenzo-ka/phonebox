"""
N-gram language model over joint multigram units (stdlib-only).

Trained on Viterbi unit sequences from ``MultigramAligner``. Used by
``joint_decode`` to score full pronunciations globally instead of
greedy letter-only segmentation.

The prediction event set is the declared inference units plus EOS. SOS is
context-only. Add-k smoothing backs off from unseen contexts, and complete
paths include the EOS transition. Accuracy depends on the data and fixed
configuration; compare alternatives with reproducible held-out evaluation.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from collections.abc import Iterable

from ..constants import EPSILON, JOIN_CHAR

LETTER_JOIN = JOIN_CHAR
PHONE_JOIN = JOIN_CHAR
UNIT_ID_SEP = "\x1f"

SOS = "<s>"
EOS = "</s>"


def encode_unit_letters(letters: tuple[str, ...]) -> str:
    """Join a unit's letter tuple into the letter half of its id."""
    return LETTER_JOIN.join(letters)


def encode_phones(phones: tuple[str, ...]) -> str:
    """Join a unit's phone tuple into the phone half of its id.

    An empty phone side (a silent letter) encodes as the ``EPSILON`` sentinel
    so it round-trips distinctly from a missing field.
    """
    if not phones:
        return EPSILON
    return PHONE_JOIN.join(phones)


def unit_id(unit: tuple[tuple[str, ...], tuple[str, ...]]) -> str:
    """Encode a ``(letters, phones)`` unit as a single hashable string id."""
    letters, phones = unit
    return encode_unit_letters(letters) + UNIT_ID_SEP + encode_phones(phones)


def decode_unit_id(uid: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Inverse of ``unit_id``: split an id back into ``(letters, phones)``.

    An ``EPSILON`` or empty phone field decodes to an empty phone tuple.
    """
    letter_s, phone_s = uid.split(UNIT_ID_SEP, 1)
    letters = tuple(letter_s.split(LETTER_JOIN)) if letter_s else ()
    if phone_s == EPSILON or not phone_s:
        phones: tuple[str, ...] = ()
    else:
        phones = tuple(phone_s.split(PHONE_JOIN))
    return letters, phones


SUPPORTED_LM_ORDERS = range(1, 9)


def _log_smoothed_probability(count: int, total: int, k: float, size: int) -> float:
    """Preserve ordinary arithmetic; use log space beyond its safe range."""
    try:
        probability = (count + k) / (total + k * size)
        if sys.float_info.min <= probability <= 1.0:
            return math.log(probability)
    except OverflowError:
        # Serialized counts are Python integers and can exceed float range.
        pass

    def log_add(left: float, right: float) -> float:
        larger, smaller = max(left, right), min(left, right)
        return larger + math.log1p(math.exp(smaller - larger))

    log_k = math.log(k)
    numerator = log_add(math.log(count), log_k) if count else log_k
    denominator = log_add(math.log(total), log_k + math.log(size))
    return numerator - denominator


def validate_lm_order(order: int) -> int:
    """Validate the shared explicit order range for training and comparisons."""
    if type(order) is not int or order not in SUPPORTED_LM_ORDERS:
        raise ValueError(
            f"order must be an integer from 1 to {SUPPORTED_LM_ORDERS.stop - 1}"
        )
    return order


class MultigramLM:
    """Sparse add-k joint-unit LM with explicit orders 1–8 and suffix backoff."""

    VERSION = 3

    def __init__(self, order: int = 2, add_k: float = 0.1) -> None:
        validate_lm_order(order)
        if type(add_k) not in (int, float) or not math.isfinite(add_k) or add_k <= 0:
            raise ValueError("add_k must be finite and positive")
        self.order = order
        self.add_k = add_k
        self._counts: list[dict[tuple[str, ...], int]] = [{} for _ in range(order)]
        self._context_totals: list[dict[tuple[str, ...], int]] = [
            {} for _ in range(order)
        ]
        self._vocab: set[str] = set()
        self._trained = False

    @property
    def is_trained(self) -> bool:
        """Whether at least one nonempty unit sequence has been counted."""
        return self._trained

    @property
    def vocab_size(self) -> int:
        """Number of declared prediction units, including inference-only units."""
        return len(self._vocab)

    def train(
        self,
        unit_sequences: list[list[tuple[tuple[str, ...], tuple[str, ...]]]],
        *,
        vocabulary: Iterable[tuple[tuple[str, ...], tuple[str, ...]]] = (),
    ) -> None:
        """Count observed paths at each requested order, including terminal EOS.

        ``vocabulary`` declares additional decoder units, including silent-phone
        units absent from Viterbi paths. SOS occurs once as context only. Empty
        paths are skipped; each call replaces the previous counts and support.
        """
        self._counts = [{} for _ in range(self.order)]
        self._vocab = {unit_id(unit) for unit in vocabulary}
        for units in unit_sequences:
            if not units:
                continue
            ids = [unit_id(unit) for unit in units]
            self._vocab.update(ids)
            sequence = (SOS, *ids, EOS)
            for end in range(1, len(sequence)):
                for size in range(1, min(self.order, end + 1) + 1):
                    gram = sequence[end - size + 1 : end + 1]
                    counts = self._counts[size - 1]
                    counts[gram] = counts.get(gram, 0) + 1
        self._refresh_context_totals()

    def _refresh_context_totals(self) -> None:
        self._context_totals = []
        for counts in self._counts:
            totals: dict[tuple[str, ...], int] = defaultdict(int)
            for gram, count in counts.items():
                totals[gram[:-1]] += count
            self._context_totals.append(dict(totals))
        self._trained = bool(self._counts[0])

    def supports_unit(self, unit: tuple[tuple[str, ...], tuple[str, ...]]) -> bool:
        """Whether a unit belongs to the fixed prediction event vocabulary."""
        return unit_id(unit) in self._vocab

    def log_prob(
        self, unit: tuple[tuple[str, ...], tuple[str, ...]], history: list[str]
    ) -> float:
        """Log probability given prior unit ids; unseen contexts back off.

        A unit outside the fixed prediction support raises ValueError. Only
        the last ``order - 1`` history ids can affect the probability.
        """
        uid = unit_id(unit)
        if uid not in self._vocab:
            raise ValueError("unit is outside the declared LM prediction vocabulary")
        return self._log_prob_with_history(uid, history)

    def log_end_prob(self, history: list[str]) -> float:
        """Log probability of ending a sequence; neutral zero before training."""
        return self._log_prob_with_history(EOS, history)

    def _log_prob_with_history(self, uid: str, history: list[str]) -> float:
        if not self._trained:
            return 0.0
        context = (SOS, *history)
        return self._log_prob_id(uid, context)

    def _log_prob_id(self, uid: str, context: tuple[str, ...]) -> float:
        vocabulary_size = len(self._vocab) + 1  # EOS is predicted; SOS is not.
        for width in range(min(self.order - 1, len(context)), -1, -1):
            suffix = context[-width:] if width else ()
            total = self._context_totals[width].get(suffix, 0)
            if total:
                count = self._counts[width].get((*suffix, uid), 0)
                return _log_smoothed_probability(
                    count, total, self.add_k, vocabulary_size
                )
        return 0.0  # No observations: neutral score, as for log_end_prob.

    def to_dict(self) -> dict:
        """Serialize sparse requested-order counts; context totals are derived."""
        return {
            "version": self.VERSION,
            "order": self.order,
            "add_k": self.add_k,
            "counts": [
                [[list(gram), count] for gram, count in sorted(level.items())]
                for level in self._counts
            ],
            "events": sorted(self._vocab | {EOS}),
        }

    def _validate_counts(self) -> None:
        events = self._vocab | {EOS}
        for size, counts in enumerate(self._counts, 1):
            for gram, count in counts.items():
                if type(count) is not int or count <= 0:
                    raise ValueError("invalid multigram LM counts")
                if (
                    len(gram) != size
                    or gram[-1] not in events
                    or any(uid not in self._vocab for uid in gram[1:-1])
                    or (size > 1 and gram[0] not in self._vocab | {SOS})
                ):
                    raise ValueError("invalid multigram LM event or context")
            if size == 1:
                continue
            lower = self._counts[size - 2]
            expected_outgoing = {
                gram: count for gram, count in lower.items() if gram[-1] != EOS
            }
            if size == 2 and lower.get((EOS,), 0):
                expected_outgoing[(SOS,)] = lower[(EOS,)]
            if self._context_totals[size - 1] != expected_outgoing:
                raise ValueError("inconsistent multigram LM context counts")
            incoming: dict[tuple[str, ...], int] = defaultdict(int)
            for gram, count in counts.items():
                incoming[gram[1:]] += count
            expected_incoming = {
                gram: count for gram, count in lower.items() if gram[0] != SOS
            }
            if dict(incoming) != expected_incoming:
                raise ValueError("inconsistent multigram LM event counts")

    @classmethod
    def from_dict(cls, data: dict) -> MultigramLM:
        if not isinstance(data, dict) or data.get("version") != cls.VERSION:
            raise ValueError(
                "unsupported multigram LM scoring version; retrain and export"
            )
        events = data.get("events")
        if (
            not isinstance(events, list)
            or not all(isinstance(uid, str) for uid in events)
            or EOS not in events
            or SOS in events
            or len(set(events)) != len(events)
        ):
            raise ValueError("invalid multigram LM prediction events")
        inst = cls(order=data.get("order"), add_k=data.get("add_k"))
        levels = data.get("counts")
        if not isinstance(levels, list) or len(levels) != inst.order:
            raise ValueError("invalid multigram LM count levels")
        inst._vocab = set(events) - {EOS}
        for level_index, records in enumerate(levels):
            if not isinstance(records, list):
                raise ValueError("invalid multigram LM count records")
            for record in records:
                if (
                    not isinstance(record, list)
                    or len(record) != 2
                    or not isinstance(record[0], list)
                    or not all(isinstance(uid, str) for uid in record[0])
                ):
                    raise ValueError("invalid multigram LM count record")
                if type(record[1]) is not int or record[1] <= 0:
                    raise ValueError("invalid multigram LM counts")
                gram = tuple(record[0])
                if (
                    not gram
                    or len(gram) != level_index + 1
                    or gram in inst._counts[level_index]
                ):
                    raise ValueError("invalid or duplicate multigram LM ngram")
                inst._counts[level_index][gram] = record[1]
        inst._refresh_context_totals()
        inst._validate_counts()
        return inst


__all__ = [
    "MultigramLM",
    "SUPPORTED_LM_ORDERS",
    "validate_lm_order",
    "SOS",
    "EOS",
    "LETTER_JOIN",
    "PHONE_JOIN",
    "UNIT_ID_SEP",
    "encode_unit_letters",
    "encode_phones",
    "unit_id",
    "decode_unit_id",
]
