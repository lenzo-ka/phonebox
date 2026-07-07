#!/usr/bin/env python
"""Dump the top multigram units (n>1 letters or phones) for each locale.

Trains MultigramG2P on the train slice (same split as compare_g2p), then
prints the highest-mass n:m units that aren't 1:1. These are the units the
EM aligner found valuable — useful for sanity-checking / extending the
``join`` lists in the locale configs.

Example::

    export PHONEDECODING_LEXICON_DIR=…  # see compare_g2p_all
    python dump_units.py --locales pt_BR fr_FR --top 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from phonebox.core.vectorizer import Vectorizer
from phonebox.eval.g2p_compare import cook_pair, load_lexicon, train_multigram
from phonebox.experiments.split import split_lexicon

_LOCALES = [
    ("es_MX", "es_ipa.tsv"),
    ("fr_FR", "fr_ipa.tsv"),
    ("de_DE", "de_ipa.tsv"),
    ("en_US", "en_ipa.tsv"),
    ("pt_BR", "pt_ipa.tsv"),
    ("it_IT", "it_ipa.tsv"),
]


def _config_joins(locale: str) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
    """Return the existing letter and phone joins (as token tuples)."""
    import json
    from pathlib import Path as P

    cfg_path = (
        P(__file__).parent / "phonebox" / "config" / "locales" / locale / "config.json"
    )
    if not cfg_path.is_file():
        return set(), set()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    join = data.get("join", {})
    letters = {tuple(s.split()) for s in join.get("letters", [])}
    ipa = {tuple(s.split()) for s in join.get("ipa", [])}
    return letters, ipa


def dump(
    locale: str,
    lex_path: Path,
    *,
    top: int,
    em_iters: int,
    seed: int = 42,
    max_test: int = 2000,
) -> None:
    pairs = load_lexicon(lex_path)
    vec = Vectorizer(locale=locale, phoneset_name="ipa")
    mg_cfg = vec.multigram_config()
    max_l = mg_cfg.get("max_letter_span", 2)
    max_p = mg_cfg.get("max_phone_span", 2)
    _test_raw, train_raw = split_lexicon(pairs, seed=seed, max_test=max_test)
    train_cooked = []
    for w, ph in train_raw:
        cooked = cook_pair(vec, w, ph)
        if cooked:
            train_cooked.append(cooked)
    print(f"\n=== {locale} ({len(train_cooked)} train pairs) ===", flush=True)
    print(
        f"  multigram spans: letter={max_l} phone={max_p} (locale config)", flush=True
    )
    mg = train_multigram(
        train_cooked,
        max_l,
        max_p,
        em_iters,
        parallel_align=True,
        parallel_viterbi=True,
    )
    cfg_letters, cfg_phones = _config_joins(locale)

    units = [
        (L, P, prob)
        for (L, P), prob in mg.aligner.q.items()
        if (len(L) >= 2 or len(P) >= 2) and prob >= 1e-5
    ]
    units.sort(key=lambda x: -x[2])
    print(
        f"  joins in config: letters={sorted(cfg_letters)} phones={sorted(cfg_phones)}"
    )
    print(f"  top {top} multigram units (n>1):")
    print(f"    {'letters':<14} {'phones':<22} mass     letter_join? phone_join?")
    for L, P, prob in units[:top]:
        L_str = " ".join(L)
        P_str = " ".join(P)
        in_l = "yes" if L in cfg_letters else ""
        in_p = "yes" if P in cfg_phones else ""
        print(f"    {L_str:<14} {P_str:<22} {prob:7.4f} {in_l:<12} {in_p}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lexicon-dir", type=Path, default=None)
    ap.add_argument("--locales", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--em-iterations", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-test", type=int, default=2000)
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional: also tee the dump to this file as markdown.",
    )
    args = ap.parse_args()

    lex_dir = args.lexicon_dir or os.environ.get("PHONEDECODING_LEXICON_DIR")
    if not lex_dir:
        print("Set PHONEDECODING_LEXICON_DIR or --lexicon-dir", file=sys.stderr)
        return 2
    lex_root = Path(lex_dir)
    selected = set(args.locales) if args.locales else None

    import contextlib
    import io

    buffer = io.StringIO()
    for locale, lex_name in _LOCALES:
        if selected is not None and locale not in selected:
            continue
        with contextlib.redirect_stdout(_Tee(sys.stdout, buffer)):  # type: ignore[type-var]
            dump(
                locale,
                lex_root / lex_name,
                top=args.top,
                em_iters=args.em_iterations,
                seed=args.seed,
                max_test=args.max_test,
            )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# Top multigram units per locale\n\n"
            f"Train split matches compare_g2p (seed={args.seed}, max_test={args.max_test}). "
            f"Letter/phone spans from locale ``config.json`` multigram section. "
            f"EM iterations={args.em_iterations}. Config joins **on**.\n\n"
            "Only units where letter or phone side has length >= 2 are shown.\n\n"
            "```\n"
        )
        args.output.write_text(header + buffer.getvalue() + "```\n", encoding="utf-8")
    return 0


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for stream in self.streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


if __name__ == "__main__":
    raise SystemExit(main())
