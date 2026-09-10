#!/usr/bin/env python
"""Dictionary operations: fetch, export-vectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ...constants import FILE_ENCODING
from ._common import add_vectorizer_args, require_file


def setup_dict_commands(subparsers):
    """Setup dictionary subcommands."""
    dict_parser = subparsers.add_parser(
        "dict",
        help="Dictionary operations (fetch, process, export-vectors)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Fetch and process pronunciation dictionaries.",
        epilog="""
Examples:
  # Download CMUdict
  phonebox dict fetch cmudict

  # Normalize and map phones, then deduplicate pronunciation variants
  phonebox dict process cmudict.dict -o processed.dict --phone-map phones.json

  # Export feature vectors for external training
  phonebox dict export-vectors cmudict.txt -o vectors.tsv
""",
    )
    dict_subparsers = dict_parser.add_subparsers(dest="dict_command")
    dict_parser.set_defaults(parser=dict_parser)

    # phonebox dict fetch
    fetch_parser = dict_subparsers.add_parser(
        "fetch",
        help="Fetch dictionary from source",
        description="Download CMUdict dictionary",
    )
    fetch_parser.add_argument("name", help="Dictionary name (e.g., cmudict)")
    fetch_parser.add_argument("--data-dir", help="Data directory")
    fetch_parser.set_defaults(func=handle_dict_fetch)

    process_parser = dict_subparsers.add_parser(
        "process",
        help="Normalize, map, and deduplicate a pronunciation dictionary",
    )
    process_parser.add_argument("dictionary", type=Path)
    process_parser.add_argument("-o", "--output", required=True, type=Path)
    process_parser.add_argument("--remove-stress", action="store_true")
    process_parser.add_argument("--lowercase", action="store_true")
    process_parser.add_argument(
        "--phone-map",
        type=Path,
        help="JSON object mapping a phone to one phone or a nonempty phone list",
    )
    process_parser.set_defaults(func=handle_dict_process)

    # phonebox dict export-vectors
    export_vectors_parser = dict_subparsers.add_parser(
        "export-vectors",
        help="Export vectorized pronunciation data",
        description="Convert pronunciation dictionary to feature vectors",
    )
    export_vectors_parser.add_argument("dictionary", help="Dictionary file")
    export_vectors_parser.add_argument(
        "-o", "--output", help="Output file (default: stdout)"
    )
    add_vectorizer_args(export_vectors_parser, cased=True, target_first=True)
    export_vectors_parser.set_defaults(func=handle_dict_export_vectors)


def handle_dict_fetch(args):
    """Handle 'phonebox dict fetch' command."""
    from ...dictionary import Dictionary

    data_dir = Path(args.data_dir) if args.data_dir else Path("data")

    print(f"Fetching dictionary: {args.name}", file=sys.stderr)
    try:
        dict_obj = Dictionary.fetch(args.name, data_dir=data_dir, verbose=True)
        print(f"Downloaded to: {dict_obj.path}", file=sys.stderr)
        Dictionary.create_manifest(data_dir, verbose=True)
        Dictionary.print_status(data_dir)
        return 0
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error fetching dictionary: {e}", file=sys.stderr)
        return 1


def handle_dict_process(args):
    """Handle ``phonebox dict process`` through the public Dictionary API."""
    from ...dictionary import Dictionary

    if (rc := require_file(args.dictionary, "dictionary")) is not None:
        return rc
    mapping = None
    if args.phone_map is not None:
        if (rc := require_file(args.phone_map, "phone map")) is not None:
            return rc
        try:
            mapping = json.loads(args.phone_map.read_text(encoding=FILE_ENCODING))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Invalid phone map: {exc}", file=sys.stderr)
            return 2
        if not isinstance(mapping, dict):
            print("Invalid phone map: expected a JSON object", file=sys.stderr)
            return 2
    try:
        Dictionary(args.dictionary).process(
            remove_stress=args.remove_stress,
            lowercase=args.lowercase,
            phone_mapping=mapping,
            output=args.output,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"Dictionary processing failed: {exc}", file=sys.stderr)
        return 2
    return 0


def handle_dict_export_vectors(args):
    """Handle 'phonebox dict export-vectors' command."""
    from ...constants import DICT_ENCODING
    from ...core.em_align import EMAlign
    from ...core.vectorizer import Vectorizer
    from ...utils.io import is_dict_comment, open_output

    vectorizer = Vectorizer(
        locale=args.locale,
        phoneset_name=args.phoneset,
        remove_stress=args.remove_stress,
        cased=args.cased,
        verbose=True,
        target_position="first" if args.target_first else "last",
    )

    if (rc := require_file(args.dictionary, "dictionary")) is not None:
        return rc

    em = EMAlign(vectorizer, verbose=True, parallel=False)

    print(f"Loading {args.dictionary}...", file=sys.stderr)
    with open(args.dictionary, encoding=DICT_ENCODING) as f:
        for line in f:
            if not is_dict_comment(line):
                em.add_line(line)

    print(f"Loaded {len(em.init_data):,} entries", file=sys.stderr)
    print("Aligning...", file=sys.stderr)
    em.align()

    print("Writing vectors...", file=sys.stderr)
    vector_count = 0
    with open_output(args.output) as outfile:
        for alignment in em.next_alignment():
            for vector in vectorizer.next_alignment_vector(alignment):
                print(vector, file=outfile)
                vector_count += 1

    print(
        f"Done: Exported {vector_count:,} vectors to {args.output or 'stdout'}",
        file=sys.stderr,
    )
    return 0
