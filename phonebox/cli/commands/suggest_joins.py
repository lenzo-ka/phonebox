#!/usr/bin/env python
"""phonebox suggest-joins — discover letter/phone joins from a lexicon.

Runs the joint-multigram (n:m) EM aligner from
``phonebox.core.multigram_align`` over a pronunciation lexicon and
emits high-probability multi-letter and multi-phone units as
``config.json``-ready suggestions.

Usage::

    phonebox suggest-joins \\
        --lexicon path/to/<lang>_ipa.tsv \\
        --max-letter-span 3 --max-phone-span 2 \\
        --min-prob 0.001 --top 50 \\
        -o path/to/joins.json

The output is a structured JSON report with two ranked lists:
``letter_joins`` (letter tuples whose total mass across phone sides is
high) and ``phone_joins`` (phone tuples whose total mass across letter
sides is high). Each entry carries the unit, its summed probability
mass, and the modal counterpart (most likely matching subsequence on
the other side, with confidence).

Sampling: large lexicons can be sub-sampled to keep wall-clock
manageable. Joint-multigram EM converges to the same high-mass joins
on samples of ~10-20k entries as on the full lexicon (the patterns are
high-frequency; rare entries don't move the modal distribution).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

from ...constants import DICT_ENCODING, FILE_ENCODING


def setup_suggest_joins_command(subparsers):
    parser = subparsers.add_parser(
        "suggest-joins",
        help="Discover letter/phone joins from a lexicon (joint-multigram EM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--lexicon",
        required=True,
        help="TSV lexicon (word\\tphone phone ...)",
    )
    parser.add_argument(
        "--locale",
        default=None,
        help="Locale tag (e.g. fr_FR): apply config.json multigram span defaults",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--max-letter-span",
        type=int,
        default=None,
        help="Maximum letters per unit (default: locale multigram config or 3).",
    )
    parser.add_argument(
        "--max-phone-span",
        type=int,
        default=None,
        help="Maximum phones per unit (default: locale multigram config or 2).",
    )
    parser.add_argument(
        "--min-phone-span",
        type=int,
        default=0,
        help="0 allows silent-letter units (default), 1 forbids them.",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.001,
        help="Drop units below this probability (default: 0.001 = 0.1%%).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Keep at most N top-ranked join candidates per category (default: 50).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Randomly sub-sample N entries from the lexicon (0 = no sampling, "
        "use all). Recommended: 10000-20000 for lexicons >100k entries.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="RNG seed for --sample (default: 42 for reproducibility).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="EM iteration ceiling (default: 15; convergence usually at 5-10).",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=1e-4,
        help="Relative LL-change for EM convergence (default: 1e-4).",
    )
    parser.add_argument(
        "--parallel-align",
        action="store_true",
        help="Parallel multigram EM E-step on large lexicons.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.set_defaults(func=handle_suggest_joins)


def _load_pairs(path: Path) -> list[tuple[list[str], list[str]]]:
    """Read a TSV lexicon. Strips ``(N)`` variants like parse_dict_line."""
    import re

    variant_re = re.compile(r"\(\d+\)$")
    pairs: list[tuple[list[str], list[str]]] = []
    with path.open(encoding=DICT_ENCODING) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith(("#", ";;;")):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            word = variant_re.sub("", parts[0])
            phones = parts[1].split()
            if not word or not phones:
                continue
            pairs.append((list(word), phones))
    return pairs


def handle_suggest_joins(args) -> int:
    # Late imports so `phonebox --help` is snappy.
    from ...core.multigram_align import MultigramAligner

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("phonebox.suggest_joins")

    lex_path = Path(args.lexicon)
    if not lex_path.is_file():
        print(f"Error: lexicon not found: {lex_path}", file=sys.stderr)
        return 2

    pairs = _load_pairs(lex_path)
    log.info("loaded %d entries from %s", len(pairs), lex_path)

    if args.sample and len(pairs) > args.sample:
        rng = random.Random(args.sample_seed)
        pairs = rng.sample(pairs, args.sample)
        log.info("sub-sampled to %d entries (seed=%d)", len(pairs), args.sample_seed)

    max_letter_span = args.max_letter_span
    max_phone_span = args.max_phone_span
    if args.locale:
        from ...core.vectorizer import Vectorizer

        mg_cfg = Vectorizer(locale=args.locale, phoneset_name="ipa").multigram_config()
        if max_letter_span is None:
            max_letter_span = mg_cfg.get("max_letter_span", 3)
        if max_phone_span is None:
            max_phone_span = mg_cfg.get("max_phone_span", 2)
    if max_letter_span is None:
        max_letter_span = 3
    if max_phone_span is None:
        max_phone_span = 2

    aligner = MultigramAligner(
        max_letter_span=max_letter_span,
        max_phone_span=max_phone_span,
        min_phone_span=args.min_phone_span,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        verbose=args.verbose,
        parallel=args.parallel_align,
    )
    log.info(
        "running EM (max_letter=%d, max_phone=%d, min_phone=%d, max_iter=%d)",
        max_letter_span,
        max_phone_span,
        args.min_phone_span,
        args.max_iterations,
    )
    aligner.fit(pairs)
    log.info(
        "EM finished after %d iterations; %d units survived above 0",
        len(aligner.loglik_history),
        sum(1 for p in aligner.q.values() if p > 0),
    )

    # ---- aggregate joins ----
    # The "join" concept (per phonebox config.json schema) is asymmetric:
    # a *letter join* is a letter-sequence that compresses to fewer
    # phones, e.g. ``(c, h) → (ʃ,)`` — 2 letters → 1 phone. A *phone
    # join* is the converse: ``(x,) → (k, s)`` — 1 letter → 2 phones.
    #
    # Symmetric multigram units (e.g. ``(c, o) → (k, o)``) have high
    # mass under the joint-EM (they're just the natural CV syllabic
    # chunks of the language), but they're NOT join candidates because
    # 1:1 unit alignment is what phonebox's EMAlign already does. So
    # we filter both aggregates to the asymmetric cases.
    letter_side: dict[tuple, dict[tuple, float]] = defaultdict(dict)
    phone_side: dict[tuple, dict[tuple, float]] = defaultdict(dict)
    for (L, P), prob in aligner.q.items():
        # Letter joins: letter-tuple longer than phone-tuple
        if len(L) > len(P):
            letter_side[L][P] = prob
        # Phone joins: phone-tuple longer than letter-tuple
        if len(P) > len(L):
            phone_side[P][L] = prob

    def _aggregate(side: dict) -> list[dict]:
        items = []
        for k, sub in side.items():
            total = sum(sub.values())
            if total < args.min_prob:
                continue
            modal_other, modal_mass = max(sub.items(), key=lambda x: x[1])
            confidence = modal_mass / total if total > 0 else 0.0
            items.append(
                {
                    "tokens": list(k),
                    "total_mass": total,
                    "modal_match": list(modal_other),
                    "modal_confidence": confidence,
                    "alternatives": [
                        {"tokens": list(other), "mass": mass}
                        for other, mass in sorted(sub.items(), key=lambda x: -x[1])[1:5]
                    ],
                }
            )
        items.sort(key=lambda x: -x["total_mass"])
        return items[: args.top]

    letter_joins = _aggregate(letter_side)
    phone_joins = _aggregate(phone_side)

    output = {
        "lexicon": str(lex_path),
        "n_entries": len(pairs),
        "settings": {
            "max_letter_span": args.max_letter_span,
            "max_phone_span": args.max_phone_span,
            "min_phone_span": args.min_phone_span,
            "min_prob": args.min_prob,
            "max_iterations": args.max_iterations,
            "iterations_run": len(aligner.loglik_history),
            "sample": args.sample if args.sample else None,
            "sample_seed": args.sample_seed if args.sample else None,
        },
        "loglik_history": aligner.loglik_history,
        "letter_joins": letter_joins,
        "phone_joins": phone_joins,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding=FILE_ENCODING,
    )
    log.info(
        "wrote %d letter joins and %d phone joins to %s",
        len(letter_joins),
        len(phone_joins),
        out_path,
    )
    return 0
