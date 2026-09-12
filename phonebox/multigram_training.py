"""Dictionary-to-multigram training with saved preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core.multigram_g2p import MultigramG2P
from .core.vectorizer import Vectorizer
from .eval.g2p_compare import cook_pair, load_lexicon
from .utils.io import paths_refer_to_same_file


@dataclass(frozen=True)
class MultigramTrainingResult:
    """Trained model, admission metrics, and the two optional export files."""

    model: MultigramG2P
    metrics: dict[str, Any]
    max_letter_span: int
    max_phone_span: int
    units_path: Path | None = None
    lm_path: Path | None = None


def train_multigram(
    lexicon: str | Path,
    *,
    locale: str,
    phoneset: str = "ipa",
    output: str | Path | None = None,
    max_letter_span: int | None = None,
    max_phone_span: int | None = None,
    em_iterations: int = 15,
    lm_order: int = 2,
    decode_beam: int = 0,
    parallel_align: bool = False,
    no_config_joins: bool = False,
    spelling_rewrites: dict[str, str] | None = None,
    remove_stress: bool = False,
    verbose: bool = False,
) -> MultigramTrainingResult:
    """Cook a dictionary, train n:m G2P, and optionally export its sidecars.

    With no output, training does not write files. Raw-word inference replays
    the attached Vectorizer; train_from_pairs remains the cooked-token API.
    """
    lexicon = Path(lexicon)
    if not lexicon.is_file():
        raise FileNotFoundError(f"lexicon not found: {lexicon}")
    units_path = lm_path = None
    if output is not None:
        if not str(output).strip():
            raise ValueError("output must be a nonempty model stem")
        stem = Path(output)
        units_path, lm_path = MultigramG2P.export_paths(stem)
        for path in (stem, units_path, lm_path):
            if paths_refer_to_same_file(lexicon, path):
                raise ValueError(
                    "multigram output must differ from the input dictionary"
                )
        if paths_refer_to_same_file(units_path, lm_path):
            raise ValueError("multigram sidecars must be distinct files")
    vectorizer = Vectorizer(
        locale=locale,
        phoneset_name=phoneset,
        remove_stress=remove_stress,
        spelling_rewrites=spelling_rewrites,
    )
    if no_config_joins:
        vectorizer.disable_config_joins()
    config = vectorizer.multigram_config()
    max_l = (
        max_letter_span
        if max_letter_span is not None
        else config.get("max_letter_span", 2)
    )
    max_p = (
        max_phone_span
        if max_phone_span is not None
        else config.get("max_phone_span", 2)
    )
    if max_l <= 0 or max_p <= 0 or em_iterations <= 0:
        raise ValueError("spans and EM iteration count must be positive")
    pairs = []
    for word, phones in load_lexicon(lexicon):
        pair = cook_pair(vectorizer, word, phones)
        if pair is not None:
            pairs.append(pair)
    if not pairs:
        raise ValueError("no trainable pairs after cooking")
    model = MultigramG2P(
        max_letter_span=max_l,
        max_phone_span=max_p,
        em_max_iterations=em_iterations,
        lm_order=lm_order,
        decode_beam=decode_beam,
        parallel_align=parallel_align,
        parallel_viterbi=parallel_align,
        verbose=verbose,
        preprocessor=vectorizer,
    )
    metrics = model.train_from_pairs(pairs)
    if output is not None:
        model.export(output)
    return MultigramTrainingResult(model, metrics, max_l, max_p, units_path, lm_path)
