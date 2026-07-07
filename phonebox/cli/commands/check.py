#!/usr/bin/env python
"""phonebox check — validate lexicon against a canonical phoneset.

Catches the bugs we keep stepping on the third time we use a language:

  * NFC vs NFD mismatches (silent killers: ``ã`` precomposed vs ``a + 0303``
    look identical but compare as different strings; common when the
    phoneset spec is hand-edited and the lexicon comes from a different
    pipeline). Detected by NFC-normalising both sides before comparison;
    if it would change the result, the mismatch is reported as fixable.
  * Phones in the lexicon missing from the canonical phoneset (these are
    "xenophones" the downstream consumer will mark as out-of-spec).
  * Phones in the canonical phoneset that the lexicon never uses (likely
    placeholders for English-loanword handling — informational, not an
    error).
  * Lexicon words that contain non-NFC characters.

    For letter/phone join discovery use ``phonebox suggest-joins`` (joint-
    multigram EM), not this command.

Usage::

    phonebox check \\
        --lexicon path/to/<lang>_lex.tsv \\
        --phoneset path/to/<lang>.phoneset.json

    # exit code 0 = clean (or warnings only), 1 = NFC/spec problems, 2 = bad inputs

The lexicon file is TSV with ``word\\tphone phone ...`` per line. The
phoneset file is a JSON array of allowed phone tokens. Both file
locations are arbitrary — pass whatever paths your project uses.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata as ud
from collections import Counter
from pathlib import Path

from ...constants import DICT_ENCODING, FILE_ENCODING


def setup_check_command(subparsers):
    parser = subparsers.add_parser(
        "check",
        help="Validate lexicon against canonical phoneset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--lexicon", required=True, help="Lexicon TSV file")
    parser.add_argument(
        "--phoneset",
        required=True,
        help="Canonical phoneset JSON (list of allowed phones)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing-from-spec phones as fatal (exit code 1 instead "
        "of warning).",
    )
    parser.add_argument(
        "--show-words",
        type=int,
        default=5,
        metavar="N",
        help="Show up to N example words for each lexicon→spec gap "
        "(default: 5; 0 to suppress).",
    )
    parser.set_defaults(func=handle_check)


def _normalise_phone(p: str) -> str:
    return ud.normalize("NFC", p)


def handle_check(args) -> int:
    lex_path = Path(args.lexicon)
    spec_path = Path(args.phoneset)

    if not lex_path.is_file():
        print(f"Error: lexicon not found: {lex_path}", file=sys.stderr)
        return 2
    if not spec_path.is_file():
        print(f"Error: phoneset spec not found: {spec_path}", file=sys.stderr)
        return 2

    try:
        spec_raw = json.loads(spec_path.read_text(encoding=FILE_ENCODING))
    except json.JSONDecodeError as e:
        print(f"Error: phoneset spec is not valid JSON: {e}", file=sys.stderr)
        return 2

    if not isinstance(spec_raw, list):
        print(
            f"Error: phoneset spec must be a JSON list of strings (got "
            f"{type(spec_raw).__name__})",
            file=sys.stderr,
        )
        return 2

    spec_phones = list(spec_raw)
    spec_nfc = {_normalise_phone(p): p for p in spec_phones}
    spec_raw_set = set(spec_phones)

    lex_phone_counts: Counter[str] = Counter()
    lex_phone_words: dict[str, list[str]] = {}
    word_not_nfc: list[str] = []
    n_entries = 0

    with lex_path.open(encoding=DICT_ENCODING) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith(("#", ";;;")):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            word, phones_str = parts[0], parts[1]
            n_entries += 1
            if ud.normalize("NFC", word) != word and len(word_not_nfc) < 20:
                word_not_nfc.append(word)
            for p in phones_str.split():
                lex_phone_counts[p] += 1
                if (
                    p not in spec_raw_set
                    and len(lex_phone_words.setdefault(p, [])) < args.show_words
                ):
                    lex_phone_words[p].append(word)

    lex_phones = set(lex_phone_counts)
    lex_nfc = {_normalise_phone(p): p for p in lex_phones}

    # 1) NFC/NFD mismatch detection
    nfc_only_in_spec: list[tuple[str, str]] = []
    for nfc_form, raw_lex in lex_nfc.items():
        if nfc_form in spec_nfc:
            raw_spec = spec_nfc[nfc_form]
            if raw_spec != raw_lex:
                nfc_only_in_spec.append((raw_spec, raw_lex))
    spec_nfc_set = set(spec_nfc)
    lex_nfc_set = set(lex_nfc)

    missing_in_spec = sorted(lex_nfc_set - spec_nfc_set)
    unused_in_spec = sorted(spec_nfc_set - lex_nfc_set)

    # Report
    print(f"=== phonebox check: {lex_path.name} vs {spec_path.name} ===")
    print(
        f"lexicon: {n_entries} entries, {len(lex_phones)} distinct phones; "
        f"spec: {len(spec_phones)} phones"
    )
    print()

    problems = 0

    if nfc_only_in_spec:
        print(
            f"!! NFC/NFD MISMATCH ({len(nfc_only_in_spec)} phones): "
            "spec uses a different normalisation form than the lexicon. "
            "These look identical but compare as different strings — "
            "downstream `phone in spec` checks will fail silently. "
            "Re-write the spec JSON in the lexicon's normalisation form."
        )
        for raw_spec, raw_lex in nfc_only_in_spec[:20]:
            cps_spec = " ".join(f"U+{ord(c):04X}" for c in raw_spec)
            cps_lex = " ".join(f"U+{ord(c):04X}" for c in raw_lex)
            print(f"   spec {raw_spec!r} [{cps_spec}]  vs  lex {raw_lex!r} [{cps_lex}]")
        problems += 1
        print()

    if missing_in_spec:
        print(
            f"!! {len(missing_in_spec)} phones used in lexicon but missing from "
            "canonical spec (these will be flagged as xenophones at "
            "supplement-time):"
        )
        for phone in missing_in_spec:
            raw = lex_nfc[phone]
            n = lex_phone_counts[raw]
            samples = lex_phone_words.get(raw, [])
            sample_str = f"  e.g. {', '.join(samples)}" if samples else ""
            print(f"   {raw!r:>10}  ({n} uses){sample_str}")
        problems += 1
        print()

    if unused_in_spec:
        print(
            f"-- {len(unused_in_spec)} phones in canonical spec that the lexicon "
            "never uses (probably reserved for loanwords / unused — informational):"
        )
        print(f"   {' '.join(repr(spec_nfc[p]) for p in unused_in_spec)}")
        print()

    if word_not_nfc:
        print(
            f"-- {len(word_not_nfc)}+ lexicon words are not in NFC (this is "
            "rarely a problem because phonebox NFC-normalises in "
            "letters_and_phones, but it's worth knowing):"
        )
        for w in word_not_nfc:
            print(f"   {w!r}")
        print()

    if problems == 0:
        print("OK: lexicon and spec agree, no NFC/NFD weirdness.")
        return 0

    if args.strict or nfc_only_in_spec:
        # NFC/NFD mismatches are always fatal — they're silent bugs.
        return 1

    # Reached only when not strict and no NFC issue: missing-from-spec is a
    # warning by default.
    return 0
