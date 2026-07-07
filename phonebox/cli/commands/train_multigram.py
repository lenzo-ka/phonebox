#!/usr/bin/env python
"""phonebox train-multigram — train and export a MultigramG2P (n:m) model."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


def setup_train_multigram_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "train-multigram",
        help="Train MultigramG2P (n:m joint Viterbi)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Train a MultigramG2P model from a pronunciation lexicon and export "
            "``<output>.units.json`` + ``<output>.lm.json`` sidecars. "
            "Pronounce with ``phonebox pronounce -m <output>`` when the sidecar "
            "is present."
        ),
    )
    parser.add_argument("--locale", required=True, help="Locale tag, e.g. it_IT")
    parser.add_argument(
        "--lexicon",
        required=True,
        help="Pronunciation dictionary TSV (word\\tphone phone ...)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Model stem (e.g. model.g2p.gz → model.g2p.gz.units.json)",
    )
    parser.add_argument("--phoneset", default="ipa")
    parser.add_argument("--max-letter-span", type=int, default=None)
    parser.add_argument("--max-phone-span", type=int, default=None)
    parser.add_argument("--em-iterations", type=int, default=15)
    parser.add_argument("--lm-order", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--decode-beam", type=int, default=0)
    parser.add_argument("--parallel-align", action="store_true")
    parser.add_argument(
        "--no-config-joins",
        action="store_true",
        help="Disable locale config.json letter/phone joins before training.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.set_defaults(func=handle_train_multigram)


def handle_train_multigram(args) -> int:
    from ...core.multigram_g2p import MultigramG2P
    from ...core.vectorizer import Vectorizer
    from ...eval.g2p_compare import cook_pair, load_lexicon

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("phonebox.train_multigram")

    lex = Path(args.lexicon)
    if not lex.is_file():
        print(f"Error: lexicon not found: {lex}", file=sys.stderr)
        return 2

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    vec = Vectorizer(
        locale=args.locale, phoneset_name=args.phoneset, remove_stress=False
    )
    if args.no_config_joins:
        vec.disable_config_joins()
    mg_cfg = vec.multigram_config()
    max_l = (
        args.max_letter_span
        if args.max_letter_span is not None
        else mg_cfg.get("max_letter_span", 2)
    )
    max_p = (
        args.max_phone_span
        if args.max_phone_span is not None
        else mg_cfg.get("max_phone_span", 2)
    )

    pairs: list[tuple[list[str], list[str]]] = []
    for word, phones in load_lexicon(lex):
        cooked = cook_pair(vec, word, phones)
        if cooked:
            pairs.append(cooked)

    if not pairs:
        print("Error: no trainable pairs after cooking", file=sys.stderr)
        return 1

    t0 = time.time()
    log.info(
        "training MultigramG2P locale=%s pairs=%d span=%d/%d em=%d",
        args.locale,
        len(pairs),
        max_l,
        max_p,
        args.em_iterations,
    )
    mg = MultigramG2P(
        max_letter_span=max_l,
        max_phone_span=max_p,
        em_max_iterations=args.em_iterations,
        lm_order=args.lm_order,
        decode_beam=args.decode_beam,
        verbose=args.verbose,
        parallel_align=args.parallel_align,
        parallel_viterbi=args.parallel_align,
    )
    mg.train_from_pairs(pairs)
    mg.locale = args.locale
    mg.phoneset_name = args.phoneset
    mg.export(out)
    log.info("exported %s (+ .units.json, .lm.json) in %.1fs", out, time.time() - t0)
    return 0
