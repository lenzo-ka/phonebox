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


class MultigramLM:
    """Add-k smoothed n-gram LM over multigram unit ids (order 1–3)."""

    VERSION = 2

    def __init__(self, order: int = 2, add_k: float = 0.1) -> None:
        if order < 1 or order > 3:
            raise ValueError("order must be 1, 2, or 3")
        if not math.isfinite(add_k) or add_k <= 0:
            raise ValueError("add_k must be finite and positive")
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
        """Number of declared prediction units, including inference-only units."""
        return len(self._vocab)

    def train(
        self,
        unit_sequences: list[list[tuple[tuple[str, ...], tuple[str, ...]]]],
        *,
        vocabulary: Iterable[tuple[tuple[str, ...], tuple[str, ...]]] = (),
    ) -> None:
        """Count aligned paths over observed and declared inference units.

        ``vocabulary`` includes units available to the decoder even when no
        Viterbi training path selected them. Unknown prediction units raise
        ValueError; they are not assigned an implicit extra smoothing event.
        Empty training sequences are skipped.
        """
        self.uni.clear()
        self.bi.clear()
        self.tri.clear()
        self._vocab = {unit_id(unit) for unit in vocabulary}
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
        self._uni_total = sum(count for uid, count in self.uni.items() if uid != SOS)
        self._trained = bool(self._vocab)

    def supports_unit(self, unit: tuple[tuple[str, ...], tuple[str, ...]]) -> bool:
        """Whether a unit belongs to the fixed prediction event vocabulary."""
        return unit_id(unit) in self._vocab

    def log_prob(
        self, unit: tuple[tuple[str, ...], tuple[str, ...]], history: list[str]
    ) -> float:
        """``log P(unit | last units in history)`` with backoff."""
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
        ctx = [SOS, *history][-(self.order - 1) :]
        return self._log_prob_id(uid, tuple(ctx))

    def _log_prob_id(self, uid: str, ctx: tuple[str, ...]) -> float:
        # Katz-style backoff: use the highest order whose context was actually
        # observed (denominator d > 0); otherwise fall through to a shorter
        # context rather than condition on an unseen history. add-k smoothing
        # (k over a vocab of size v) keeps every probability non-zero.
        k = self.add_k
        v = len(self._vocab) + 1  # EOS is a predicted event; SOS is context-only.

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
            "version": self.VERSION,
            "order": self.order,
            "add_k": self.add_k,
            "uni": dict(self.uni),
            "bi": {"\t".join(k): v for k, v in self.bi.items()},
            "tri": {"\t".join(k): v for k, v in self.tri.items()},
            "events": sorted(self._vocab | {EOS}),
        }

    def _validate_counts(self) -> None:
        events = self._vocab | {EOS}
        contexts = self._vocab | {SOS}
        for counts in (self.uni, self.bi, self.tri):
            if any(type(count) is not int or count < 0 for count in counts.values()):
                raise ValueError("invalid multigram LM counts")
        outgoing: dict[str, int] = defaultdict(int)
        incoming: dict[str, int] = defaultdict(int)
        for key, count in self.bi.items():
            if len(key) != 2 or key[0] not in contexts or key[1] not in events:
                raise ValueError("invalid multigram LM bigram event or context")
            outgoing[key[0]] += count
            incoming[key[1]] += count
        for uid in contexts:
            if outgoing[uid] != self.uni.get(uid, 0):
                raise ValueError("inconsistent multigram LM context counts")
        for uid in events:
            if incoming[uid] != self.uni.get(uid, 0):
                raise ValueError("inconsistent multigram LM event counts")
        tri_outgoing: dict[tuple[str, str], int] = defaultdict(int)
        for trigram, count in self.tri.items():
            if (
                len(trigram) != 3
                or trigram[0] not in contexts
                or trigram[1] not in self._vocab
                or trigram[2] not in events
            ):
                raise ValueError("invalid multigram LM trigram event or context")
            tri_outgoing[(trigram[0], trigram[1])] += count
        for context in set(self.bi) | set(tri_outgoing):
            if context[1] != EOS and tri_outgoing[context] != self.bi.get(context, 0):
                raise ValueError("inconsistent multigram LM trigram context counts")

    @classmethod
    def from_dict(cls, data: dict) -> MultigramLM:
        if data.get("version") != cls.VERSION:
            raise ValueError(
                "unsupported multigram LM scoring version; retrain and export"
            )
        events = data.get("events")
        if not isinstance(events, list) or not all(
            isinstance(uid, str) for uid in events
        ):
            raise ValueError("invalid multigram LM prediction events")
        if EOS not in events or SOS in events or len(set(events)) != len(events):
            raise ValueError("invalid multigram LM prediction events")
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
        inst._vocab = set(events) - {EOS}
        if set(inst.uni) - {SOS, EOS} - inst._vocab:
            raise ValueError("multigram LM counts contain undeclared prediction events")
        inst._validate_counts()
        inst._uni_total = sum(count for uid, count in inst.uni.items() if uid != SOS)
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
