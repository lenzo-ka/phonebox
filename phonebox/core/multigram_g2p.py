"""
n:m grapheme-to-phoneme: joint multigram EM + unit n-gram Viterbi decode.

Training:
    1. Joint-multigram EM learns unit distribution ``q``.
    2. Viterbi-align training pairs into unit sequences.
    3. N-gram LM over units for inference.

Inference:
    Joint Viterbi over ``q`` + LM (global segmentation and phones).

References:
    Bisani, M. & Ney, H. (2008). "Joint-sequence models for
    grapheme-to-phoneme conversion." Speech Communication 50:434-451.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Any

from ..constants import (
    DEFAULT_MULTIGRAM_MIN_UNIT_MASS,
    DICT_ENCODING,
    EPSILON,
    FILE_ENCODING,
)
from ..dictionary import parse_dict_line
from ..utils.io import is_dict_comment
from ..utils.logging_config import get_logger
from .joint_decode import joint_decode
from .multigram_align import MultigramAligner
from .multigram_lm import (
    LETTER_JOIN,
    PHONE_JOIN,
    MultigramLM,
    encode_phones,
    encode_unit_letters,
)

logger = get_logger(__name__)

SILENT_TARGET = EPSILON


def decode_phones(target: str) -> list[str]:
    """Inverse of ``encode_phones``: target string back to phone list."""
    if target == SILENT_TARGET or not target:
        return []
    return target.split(PHONE_JOIN)


class MultigramG2P:
    """n:m G2P: EM unit model + unit n-gram LM + joint Viterbi decode."""

    VERSION = "3"

    def __init__(
        self,
        max_letter_span: int = 3,
        max_phone_span: int = 2,
        min_phone_span: int = 0,
        em_max_iterations: int = 15,
        em_convergence_threshold: float = 1e-4,
        lm_order: int = 2,
        decode_beam: int = 0,
        verbose: bool = False,
        parallel_align: bool = False,
        parallel_viterbi: bool = False,
        num_workers: int | None = None,
        min_unit_mass: float = DEFAULT_MULTIGRAM_MIN_UNIT_MASS,
    ) -> None:
        self.aligner = MultigramAligner(
            max_letter_span=max_letter_span,
            max_phone_span=max_phone_span,
            min_phone_span=min_phone_span,
            max_iterations=em_max_iterations,
            convergence_threshold=em_convergence_threshold,
            verbose=verbose,
            parallel=parallel_align,
            num_workers=num_workers,
            min_unit_mass=min_unit_mass,
        )
        self.lm = MultigramLM(order=lm_order)
        self.verbose = verbose
        self.parallel_viterbi = parallel_viterbi
        self.decode_beam = decode_beam
        self._max_l = max_letter_span
        self.use_dict_fallback = False
        self.exceptions: dict[str, list[str]] = {}
        self.locale: str | None = None
        self.phoneset_name: str | None = None

    # ----------- training

    def _load_dict(self, path: str | Path) -> list[tuple[list[str], list[str]]]:
        out: list[tuple[list[str], list[str]]] = []
        with open(path, encoding=DICT_ENCODING) as f:
            for line in f:
                if is_dict_comment(line):
                    continue
                parsed = parse_dict_line(line)
                if not parsed:
                    continue
                word, phones = parsed
                if not word or not phones:
                    continue
                out.append((list(word.lower()), list(phones)))
        if self.verbose:
            logger.info("MultigramG2P: loaded %d entries from %s", len(out), path)
        return out

    def train_from_pairs(
        self, pairs: list[tuple[list[str], list[str]]]
    ) -> dict[str, Any]:
        """Train EM and the unit n-gram LM."""
        if not pairs:
            raise ValueError("empty lexicon")

        if self.verbose:
            logger.info("MultigramG2P: fitting EM aligner...")
        t_em = time()
        self.aligner.fit(pairs)
        if self.verbose:
            logger.info("MultigramG2P: EM fit done in %.1fs", time() - t_em)

        if self.verbose:
            logger.info(
                "MultigramG2P: Viterbi-aligning %d entries (parallel=%s)",
                len(pairs),
                self.parallel_viterbi,
            )
        t_vit = time()
        units_list = self.aligner.viterbi_batch(pairs, parallel=self.parallel_viterbi)
        if self.verbose:
            logger.info("MultigramG2P: Viterbi done in %.1fs", time() - t_vit)

        aligned_units = [u for u in units_list if u is not None]
        if self.verbose:
            logger.info(
                "MultigramG2P: training unit LM (order=%d) on %d paths",
                self.lm.order,
                len(aligned_units),
            )
        t_lm = time()
        self.lm.train(aligned_units)
        if self.verbose:
            logger.info(
                "MultigramG2P: LM done in %.1fs (%d vocab)",
                time() - t_lm,
                self.lm.vocab_size,
            )

        return {
            "lm_vocab": self.lm.vocab_size,
            "aligned_entries": len(aligned_units),
            "skipped_entries": len(units_list) - len(aligned_units),
        }

    def train_from_dict(self, path: str | Path) -> dict[str, Any]:
        return self.train_from_pairs(self._load_dict(path))

    # ----------- inference

    def _lookup_exception(self, word: str) -> list[str] | None:
        if not self.use_dict_fallback:
            return None
        return self.exceptions.get(word.lower())

    def pronounce_letters(
        self, letters: list[str], *, word: str | None = None
    ) -> list[str]:
        """Predict phones for a pre-cooked letter-token sequence."""
        if word is not None:
            exception = self._lookup_exception(word)
            if exception is not None:
                return list(exception)
        if not self.lm.is_trained:
            raise RuntimeError("MultigramG2P not trained / loaded")
        if not letters:
            return []
        phones = joint_decode(
            letters,
            self.aligner.q,
            self.lm,
            self._max_l,
            beam=self.decode_beam,
        )
        return phones if phones is not None else []

    def pronounce(self, word: str) -> list[str]:
        """Predict phones for a raw word string (per-character split).

        Prefer ``pronounce_letters`` after ``Vectorizer.cook_letters``.
        """
        return self.pronounce_letters(list(word.lower()), word=word)

    # ----------- save/load

    def export(self, path: str | Path) -> None:
        if not self.lm.is_trained:
            raise RuntimeError("nothing to export — train first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        units_path = path.with_suffix(path.suffix + ".units.json")
        lm_path = path.with_suffix(path.suffix + ".lm.json")

        units_serialised = [
            [list(letters), list(phones), prob]
            for (letters, phones), prob in self.aligner.q.items()
        ]
        meta = {
            "version": self.VERSION,
            "max_letter_span": self._max_l,
            "max_phone_span": self.aligner.max_p,
            "min_phone_span": self.aligner.min_p,
            "lm_order": self.lm.order,
            "decode_beam": self.decode_beam,
            "units": units_serialised,
            "lm_path": lm_path.name,
            "use_dict_fallback": self.use_dict_fallback,
            "exceptions": self.exceptions if self.use_dict_fallback else {},
        }
        if self.locale:
            meta["locale"] = self.locale
        if self.phoneset_name:
            meta["phoneset_name"] = self.phoneset_name
        lm_path.write_text(
            json.dumps(self.lm.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding=FILE_ENCODING,
        )
        units_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding=FILE_ENCODING,
        )

    @classmethod
    def load(cls, path: str | Path) -> MultigramG2P:
        path = Path(path)
        units_path = path.with_suffix(path.suffix + ".units.json")
        meta = json.loads(units_path.read_text(encoding=FILE_ENCODING))
        version = meta.get("version", "1")
        if version in ("1", "2") and "lm_path" not in meta:
            raise ValueError(
                f"model version {version} used the removed tree/greedy decoder; "
                "retrain with MultigramG2P v3 and export again"
            )
        inst = cls(
            max_letter_span=meta["max_letter_span"],
            max_phone_span=meta["max_phone_span"],
            min_phone_span=meta["min_phone_span"],
            lm_order=meta.get("lm_order", 2),
            decode_beam=meta.get("decode_beam", 0),
        )
        inst.aligner.q = {(tuple(L), tuple(P)): prob for (L, P, prob) in meta["units"]}
        marg: dict[tuple[str, ...], float] = defaultdict(float)
        for (L, _P), prob in inst.aligner.q.items():
            marg[L] += prob
        inst.aligner._letter_marginal = marg

        lm_path = path.parent / meta["lm_path"]
        inst.lm = MultigramLM.from_dict(
            json.loads(lm_path.read_text(encoding=FILE_ENCODING))
        )
        if meta.get("use_dict_fallback"):
            inst.use_dict_fallback = True
            inst.exceptions = {
                str(k): list(v) for k, v in meta.get("exceptions", {}).items()
            }
        inst.locale = meta.get("locale")
        inst.phoneset_name = meta.get("phoneset_name")
        return inst


__all__ = [
    "MultigramG2P",
    "encode_unit_letters",
    "encode_phones",
    "decode_phones",
    "LETTER_JOIN",
    "PHONE_JOIN",
]
