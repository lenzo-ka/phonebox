"""
G2P model bundler - creates standalone Python executables with embedded models.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cartlet import bundle as cartlet_bundle

from .constants import FILE_ENCODING


def _ensure_cart_format(model_path: str) -> tuple[str, bool]:
    """
    Ensure model is in .cart format for bundling.

    Returns:
        (cart_path, needs_cleanup) - path to .cart file and whether to delete it
    """
    if model_path.endswith(".cart"):
        return model_path, False

    # Convert .g2p.gz or other formats to .cart
    from .core.g2p_model import G2PDecisionTree

    dt = G2PDecisionTree()
    dt.load_model(model_path)

    with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as tmp:
        cart_path = tmp.name

    dt._cart.export(cart_path, store_distributions=dt._cart.store_distributions)
    return cart_path, True


def bundle_g2p(model_path: str, output_path: str) -> None:
    """
    Bundle G2P model into a standalone Python executable.

    Args:
        model_path: Path to model file (.g2p.gz, .cart, etc.)
        output_path: Output `.py` file path
    """
    cart_path, cleanup = _ensure_cart_format(model_path)

    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
            tmp_path = tmp.name

        try:
            # library_only excludes cartlet's CLI; we provide our own G2P main.
            cartlet_bundle(cart_path, tmp_path, library_only=True)

            with open(tmp_path, encoding=FILE_ENCODING) as f:
                cart_code = f.read()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        g2p_template_path = Path(__file__).parent / "cart" / "g2p_predict.py"
        with open(g2p_template_path, encoding=FILE_ENCODING) as f:
            g2p_code = f.read()

        # Strip the file prelude; keep everything from class G2PPredictor onward.
        class_marker = "class G2PPredictor"
        if class_marker in g2p_code:
            idx = g2p_code.index(class_marker)
            g2p_code = g2p_code[idx:]

        output = cart_code + "\n\n\n" + g2p_code

        with open(output_path, "w", encoding=FILE_ENCODING) as f:
            f.write(output)
    finally:
        if cleanup:
            Path(cart_path).unlink(missing_ok=True)
