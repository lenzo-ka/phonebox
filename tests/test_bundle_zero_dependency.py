"""A generated standalone bundle imports only the standard library and runs under -S."""

import ast
import subprocess
import sys
from pathlib import Path

from phonebox import G2P
from phonebox.bundler import bundle_g2p


def _imported_modules(source: str) -> set[str]:
    """Every module named by any import statement, at any nesting depth."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names.add("<relative>")
            elif node.module:
                names.add(node.module.partition(".")[0])
    return names


def test_generated_bundle_import_closure_is_standard_library_only(tmp_path):
    dictionary = tmp_path / "words.dict"
    dictionary.write_text(
        "cat K AE T\nbat B AE T\nfat F AE T\nhat HH AE T\nrat R AE T\n"
        "mat M AE T\nsat S AE T\npat P AE T\n",
        encoding="utf-8",
    )
    g2p = G2P.train(
        dictionary,
        locale="en",
        phoneset="cmu",
        prune=False,
        use_dict_fallback=False,
        verbose=False,
    )
    model_path = tmp_path / "model.g2p.gz"
    g2p._dt.export(str(model_path), include_exceptions=False)
    bundle_path = tmp_path / "g2p.py"
    bundle_g2p(str(model_path), str(bundle_path))

    imported = _imported_modules(bundle_path.read_text(encoding="utf-8"))
    assert imported, "bundle imports nothing, which cannot be a working predictor"
    foreign = imported - set(sys.stdlib_module_names)
    assert not foreign, (
        f"bundle imports outside the standard library: {sorted(foreign)}"
    )
    assert "__future__" not in foreign

    # No site-packages, no user site, no environment: the bundle must still run.
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(bundle_path), "cat"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip().partition("\t")[2].split() == ["K", "AE", "T"]
    assert not any(
        name in {"phonebox", "cartlet", "icukit", "icu"} for name in imported
    )


def test_import_closure_helper_sees_nested_and_relative_imports():
    source = Path(__file__).read_text(encoding="utf-8")
    assert {"ast", "subprocess", "sys", "pathlib", "phonebox"} <= _imported_modules(
        source
    )
    nested = "def f():\n    import numpy\n    from . import x\n"
    assert _imported_modules(nested) == {"numpy", "<relative>"}
