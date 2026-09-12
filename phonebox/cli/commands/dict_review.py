"""Review and reorder a lexicon using a saved CART model's compatibility scores."""

from __future__ import annotations

import json
from argparse import BooleanOptionalAction
from pathlib import Path

from ...converter import G2P
from ...core.multigram_g2p import MultigramG2P
from ...pronunciation_analysis import format_lexicon_review, review_lexicon_file
from ...utils.io import open_output, paths_refer_to_same_file
from ._common import expected_input_errors


def setup_dict_review_command(subparsers) -> None:
    """Register the thin dictionary review adapter."""
    parser = subparsers.add_parser(
        "review",
        help="Score lexicon variants and reorder them",
        description=__doc__,
        epilog="Examples: phonebox dict review words.dict -m model.g2p.gz; "
        "phonebox dict review words.dict -m model.g2p.gz --format dict -o ranked.dict",
    )
    parser.add_argument("lexicon", type=Path)
    parser.add_argument("-m", "--model", required=True, type=Path)
    parser.add_argument(
        "-o", "--output", type=Path, help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--format", choices=("tsv", "json", "jsonl", "dict"), default="tsv"
    )
    parser.add_argument(
        "--order",
        choices=("worst", "variants"),
        default=None,
        help="Default: worst first; dictionary format defaults to variants",
    )
    parser.add_argument(
        "--method",
        choices=("geometric", "product"),
        default="geometric",
        help="Model compatibility scale (not correctness probability)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Keep scores strictly below threshold, retaining full ranks",
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum selected records, retaining full ranks"
    )
    parser.add_argument(
        "--no-header", action="store_true", help="Omit TSV column names"
    )
    parser.add_argument(
        "--number-senses",
        action=BooleanOptionalAction,
        default=True,
        help="Number ranked variants as word, word(2), ... (default: enabled)",
    )
    parser.add_argument(
        "--phone-map",
        type=Path,
        help="JSON literal phone mapping before saved model cooking",
    )
    parser.set_defaults(func=handle_dict_review)


@expected_input_errors
def handle_dict_review(args) -> int:
    """Load inputs, delegate review, and present data without changing sources."""
    units, lm = MultigramG2P.export_paths(args.model)
    if units.is_file() or args.model.name.endswith((".units.json", ".lm.json")):
        raise ValueError(
            "lexicon review supports CART models; multigram scoring is unsupported"
        )
    inputs = [args.lexicon, args.model, units, lm]
    if args.phone_map is not None:
        inputs.append(args.phone_map)
    if args.output is not None and any(
        paths_refer_to_same_file(path, args.output) for path in inputs
    ):
        raise ValueError(
            "review output must differ from lexicon, mapping, and model artifacts"
        )
    order = args.order or ("variants" if args.format == "dict" else "worst")
    if args.format == "dict" and order != "variants":
        raise ValueError("dictionary format requires variants order")
    if args.format == "dict" and (args.threshold is not None or args.limit is not None):
        raise ValueError("dictionary format requires an unfiltered lexicon")
    mapping = None
    if args.phone_map is not None:
        mapping = json.loads(args.phone_map.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise ValueError("phone mapping must be a JSON object")
    scorer = G2P(model=args.model, use_dict_fallback=False)
    result = review_lexicon_file(
        scorer,
        args.lexicon,
        method=args.method,
        phone_mapping=mapping,
        threshold=args.threshold,
        limit=args.limit,
        order=order,
    )
    lines = format_lexicon_review(
        result,
        format=args.format,
        header=not args.no_header,
        number_senses=args.number_senses,
    )
    # Run shared format validation before replacing a named destination.
    first = next(lines, None)
    with open_output(args.output) as stream:
        if first is not None:
            print(first, file=stream)
        for line in lines:
            print(line, file=stream)
    return 0
