"""
N-gram language model over joint multigram units (stdlib-only).

Trained on Viterbi unit sequences from ``MultigramAligner``. Used by
``joint_decode`` to score full pronunciations globally instead of
greedy letter-only segmentation.

We experimented with backing this with the ``arpabo`` package (which gives
Good-Turing/Kneser-Ney/Katz smoothing + standard ARPA output). In our
specific regime — small unit vocab (~1k), very short "sentences" (~5
units), dense coverage — add-k with k=0.1 actually beats every arpabo
smoother by 1–4 pp PER across all six locales. The smoothers concentrate
too much mass on seen events for OOV units in inference-time ``q``.
Keeping the stdlib add-k implementation; switch to ARPA only if a
downstream tool demands the format.
"""

from __future__ import annotations

import math
from collections import defaultdict

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


class MultigramLM:
    """Add-k smoothed n-gram LM over multigram unit ids (order 1–3)."""

    def __init__(self, order: int = 2, add_k: float = 0.1) -> None:
        if order < 1 or order > 3:
            raise ValueError("order must be 1, 2, or 3")
        self.order = order
        self.add_k = add_k
        self.uni: dict[str, int] = defaultdict(int)
        self.bi: dict[tuple[str, str], int] = defaultdict(int)
        self.tri: dict[tuple[str, str, str], int] = defaultdict(int)
        self._vocab: set[str] = set()
        self._uni_total = 0
        self._trained = False

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained (has a non-empty vocabulary)."""
        return self._trained

    @property
    def vocab_size(self) -> int:
        """Number of distinct unit ids seen in training."""
        return len(self._vocab)

    def train(
        self, unit_sequences: list[list[tuple[tuple[str, ...], tuple[str, ...]]]]
    ) -> None:
        """Count n-grams from aligned unit paths (empty sequences skipped)."""
        self.uni.clear()
        self.bi.clear()
        self.tri.clear()
        self._vocab.clear()
        for units in unit_sequences:
            if not units:
                continue
            ids = [unit_id(u) for u in units]
            self._vocab.update(ids)
            seq = [SOS, *ids, EOS]
            for i, tok in enumerate(seq):
                self.uni[tok] += 1
                if i > 0:
                    self.bi[(seq[i - 1], tok)] += 1
                if i > 1:
                    self.tri[(seq[i - 2], seq[i - 1], tok)] += 1
        self._uni_total = sum(self.uni.values())
        self._trained = bool(self._vocab)

    def log_prob(
        self, unit: tuple[tuple[str, ...], tuple[str, ...]], history: list[str]
    ) -> float:
        """``log P(unit | last units in history)`` with backoff."""
        if not self._trained:
            return 0.0
        uid = unit_id(unit)
        ctx = [SOS, *history][-(self.order - 1) :]
        return self._log_prob_id(uid, tuple(ctx))

    def _log_prob_id(self, uid: str, ctx: tuple[str, ...]) -> float:
        # Katz-style backoff: use the highest order whose context was actually
        # observed (denominator d > 0); otherwise fall through to a shorter
        # context rather than condition on an unseen history. add-k smoothing
        # (k over a vocab of size v) keeps every probability non-zero.
        k = self.add_k
        v = max(len(self._vocab), 1)

        if self.order >= 3 and len(ctx) >= 2:
            c = self.tri.get((ctx[-2], ctx[-1], uid), 0)
            d = self.bi.get((ctx[-2], ctx[-1]), 0)
            if d > 0:
                return math.log((c + k) / (d + k * v))
        if self.order >= 2 and len(ctx) >= 1:
            c = self.bi.get((ctx[-1], uid), 0)
            d = self.uni.get(ctx[-1], 0)
            if d > 0:
                return math.log((c + k) / (d + k * v))
        c = self.uni.get(uid, 0)
        return math.log((c + k) / (self._uni_total + k * v))

    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "add_k": self.add_k,
            "uni": dict(self.uni),
            "bi": {"\t".join(k): v for k, v in self.bi.items()},
            "tri": {"\t".join(k): v for k, v in self.tri.items()},
            "vocab": sorted(self._vocab),
        }

    @classmethod
    def from_dict(cls, data: dict) -> MultigramLM:
        inst = cls(order=data["order"], add_k=data["add_k"])
        inst.uni = defaultdict(int, data["uni"])
        inst.bi = defaultdict(
            int,
            {tuple(k.split("\t")): v for k, v in data["bi"].items()},
        )
        inst.tri = defaultdict(
            int,
            {tuple(k.split("\t")): v for k, v in data["tri"].items()},
        )
        inst._vocab = set(data.get("vocab", []))
        inst._uni_total = sum(inst.uni.values())
        inst._trained = bool(inst._vocab)
        return inst


__all__ = [
    "MultigramLM",
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
