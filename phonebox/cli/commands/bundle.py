"""Bundle command: create standalone G2P predictor."""

from __future__ import annotations

import argparse
import os
import sys

from ._common import expected_input_errors


def setup_bundle_command(subparsers):
    """Setup bundle command."""
    parser = subparsers.add_parser(
        "bundle",
        help="Bundle a decision-tree model as a standalone Python predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Bundle a decision-tree G2P model into a standalone executable.

Creates a self-contained .py file with the model embedded. Multigram models
are not supported by the standalone decision-tree predictor.

Examples:
  phonebox bundle model.g2p.gz -o g2p.py

Usage of bundled file:
  python g2p.py "Hello, world!"

Library usage:
  from g2p import G2PPredictor
  g2p = G2PPredictor.from_embedded()
  phones = g2p.pronounce_text('Hello, world!')""",
    )
    parser.add_argument("model", help="Decision-tree model (.g2p.gz, .cart, etc.)")
    parser.add_argument("-o", "--output", required=True, help="Output .py file")
    parser.set_defaults(func=handle_bundle)


@expected_input_errors
def handle_bundle(args):
    """Handle 'phonebox bundle' command."""
    from ...bundler import bundle_g2p

    try:
        print(f"Bundling -> {args.output}", file=sys.stderr)
        bundle_g2p(args.model, args.output)
        print(f"Done: Created {args.output}", file=sys.stderr)

        module_name = os.path.splitext(os.path.basename(args.output))[0]
        print("\nUsage:", file=sys.stderr)
        print(f'  CLI:     python {args.output} "Hello, world!"', file=sys.stderr)
        print(f"  Library: from {module_name} import G2PPredictor", file=sys.stderr)
        print("           g2p = G2PPredictor.from_embedded()", file=sys.stderr)
        print("           g2p.pronounce_text('Hello, world!')", file=sys.stderr)
        return 0
    except (ImportError, OSError, ValueError):
        raise
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
