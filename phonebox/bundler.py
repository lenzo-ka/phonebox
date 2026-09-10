"""
G2P model bundler - creates standalone Python executables with embedded models.
"""

from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path

from cartlet import bundle as cartlet_bundle
from cartlet import read_cart_metadata

from .constants import FILE_ENCODING
from .portable_normalization import compile_metadata_preprocessing


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

    # Preserve the source metadata verbatim. Reconstructing it through
    # G2PDecisionTree.export would replace training-time settings with the
    # loader's defaults and could lose fields unknown to this version.
    structural_keys = {
        "class_labels",
        "feature_names",
        "feature_specs",
        "metadata",
        "model",
        "task",
    }
    source_metadata = {
        key: deepcopy(value)
        for key, value in dt._model_header.items()
        if key not in structural_keys
    }
    # Match G2PDecisionTree.load_model's field-by-field precedence: legacy flat
    # values remain fallbacks, while every nested value (including fields this
    # version does not know about) wins and survives conversion verbatim.
    source_metadata.update(deepcopy(dt._model_header.get("metadata", {})))
    try:
        dt._cart.export(
            cart_path,
            metadata=source_metadata,
            store_distributions=dt._cart.store_distributions,
            format=".cart",
        )
    except Exception:
        Path(cart_path).unlink(missing_ok=True)
        raise
    return cart_path, True


def bundle_g2p(model_path: str, output_path: str) -> None:
    """
    Bundle G2P model into a standalone Python executable.

    Args:
        model_path: Path to a decision-tree model (.g2p.gz, .cart, etc.)
        output_path: Output `.py` file path

    Raises:
        ValueError: The model is a multigram artifact, which is not supported.
    """
    model = Path(model_path)
    if model.with_suffix(model.suffix + ".units.json").is_file():
        raise ValueError(
            "standalone bundling supports decision-tree models only; "
            "the selected model has a MultigramG2P sidecar"
        )
    cart_path, cleanup = _ensure_cart_format(model_path)

    try:
        metadata = read_cart_metadata(cart_path)
        # Fail before writing when exact training behavior is outside the
        # deliberately small standard-library contract.
        compile_metadata_preprocessing(metadata)

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

        portable_path = Path(__file__).parent / "portable_normalization.py"
        with open(portable_path, encoding=FILE_ENCODING) as f:
            portable_code = f.read()

        normalize_path = Path(__file__).parent / "normalize.py"
        with open(normalize_path, encoding=FILE_ENCODING) as f:
            normalize_code = f.read()

        # Strip the file prelude; keep everything from class G2PPredictor onward.
        class_marker = "class G2PPredictor"
        if class_marker in g2p_code:
            idx = g2p_code.index(class_marker)
            g2p_code = g2p_code[idx:]

        # The generated file receives the exact compiler/interpreter source
        # used by G2PRunner, avoiding a second standalone implementation.
        output = (
            cart_code
            + "\n\n\n"
            + portable_code
            + "\n\n\n"
            + normalize_code
            + "\n\n\n"
            + g2p_code
        )

        with open(output_path, "w", encoding=FILE_ENCODING) as f:
            f.write(output)
    finally:
        if cleanup:
            Path(cart_path).unlink(missing_ok=True)
