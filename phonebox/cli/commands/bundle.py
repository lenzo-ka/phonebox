"""Bundle command: create standalone G2P predictor."""

from __future__ import annotations

import os
import sys
import tempfile


def setup_bundle_command(subparsers):
    """Setup bundle command."""
    parser = subparsers.add_parser(
        "bundle",
        help="Create standalone Python G2P predictor with embedded model",
        description="""Bundle G2P model into a standalone Python executable.

Creates a self-contained .py file with the model embedded.

Examples:
  phonebox bundle model.g2p.gz -o g2p.py

Usage of bundled file:
  python g2p.py "Hello, world!"

Library usage:
  from g2p import G2PPredictor
  g2p = G2PPredictor.from_embedded()
  phones = g2p.pronounce_text('Hello, world!')""",
    )
    parser.add_argument("model", help="Model file (.g2p.gz, .cart, etc.)")
    parser.add_argument("-o", "--output", required=True, help="Output .py file")
    parser.set_defaults(func=handle_bundle)


def handle_bundle(args):
    """Handle 'phonebox bundle' command."""
    from ...bundler import bundle_g2p
    from ...core.g2p_model import G2PDecisionTree

    model_path = args.model
    output_path = args.output

    # Convert to .cart format if needed (cartlet's bundler requires .cart)
    needs_conversion = not model_path.endswith(".cart")

    if needs_conversion:
        print(f"Converting {model_path} to .cart for bundling...", file=sys.stderr)

        dt = G2PDecisionTree(model=model_path)

        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as tmp:
            cart_path = tmp.name

        dt._cart.export(cart_path, store_distributions=dt._cart.store_distributions)
        model_path = cart_path

    try:
        print(f"Bundling -> {output_path}", file=sys.stderr)
        bundle_g2p(model_path, output_path)
        print(f"Done: Created {output_path}", file=sys.stderr)

        module_name = os.path.splitext(os.path.basename(output_path))[0]
        print("\nUsage:", file=sys.stderr)
        print(f'  CLI:     python {output_path} "Hello, world!"', file=sys.stderr)
        print(f"  Library: from {module_name} import G2PPredictor", file=sys.stderr)
        print("           g2p = G2PPredictor.from_embedded()", file=sys.stderr)
        print("           g2p.pronounce_text('Hello, world!')", file=sys.stderr)
        return 0
    finally:
        if needs_conversion:
            os.unlink(cart_path)
