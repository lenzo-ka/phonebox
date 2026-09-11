"""Reproducible CART versus multigram evaluation on public CMUdict."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tempfile
import time
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from phonebox.constants import DOWNLOAD_TIMEOUT_SECONDS
from phonebox.core.vectorizer import Vectorizer
from phonebox.eval.g2p_compare import (
    cook_pair,
    evaluate,
    load_lexicon,
    predict_cooked_phones,
    train_baseline,
    train_multigram,
)
from phonebox.experiments.split import split_lexicon_by_key

CMUDICT_COMMIT = "74790861f652b15e4ac49015a90074ad62a27690"
CMUDICT_SHA256 = "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
CMUDICT_URL = (
    f"https://raw.githubusercontent.com/cmusphinx/cmudict/{CMUDICT_COMMIT}/cmudict.dict"
)
CMUDICT_LICENSE_URL = (
    f"https://github.com/cmusphinx/cmudict/blob/{CMUDICT_COMMIT}/LICENSE"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_cmudict(path: Path) -> Path:
    """Download the pinned public CMUdict file and verify its digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(  # noqa: S310
        CMUDICT_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS
    ) as response:
        path.write_bytes(response.read())
    validate_cmudict(path)
    return path


def validate_cmudict(path: Path) -> None:
    """Reject data that differs from the pinned CMUdict source file."""
    actual = sha256_file(path)
    if actual != CMUDICT_SHA256:
        raise ValueError(
            f"CMUdict SHA-256 mismatch: expected {CMUDICT_SHA256}, got {actual}"
        )


def load_cmudict(path: Path) -> list[tuple[str, list[str]]]:
    """Read CMUdict through Phonebox's shared pronunciation parser."""
    return [(word.lower(), phones) for word, phones in load_lexicon(path)]


def _artifact_bytes(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths)


def _training_accounting(
    cart: Any,
    train_cooked: list[tuple[list[str], list[str]]],
    multigram_metrics: dict[str, object],
    raw_entries: int,
) -> dict[str, Any]:
    """Describe candidate and admitted counts from the actual trainers."""
    cart_candidates = len(cart.em.seen)
    cart_retained = len(cart.em.init_data)
    return {
        "raw_entries_supplied_each": raw_entries,
        "G2PDecisionTree": {
            "unique_cooked_candidates": cart_candidates,
            "retained_entries": cart_retained,
            "skipped_after_dedup": cart_candidates - cart_retained,
        },
        "MultigramG2P": {
            "unique_cooked_candidates": len(train_cooked),
            "aligned_entries": multigram_metrics["aligned_entries"],
            "skipped_entries": multigram_metrics["skipped_entries"],
        },
    }


def _git_revision(root: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _code_fingerprint(root: Path) -> str:
    files = sorted(
        path
        for path in (root / "phonebox").rglob("*")
        if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".xlit"}
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_cmudict_comparison(
    lexicon: Path,
    *,
    seed: int = 1729,
    test_fraction: float = 0.1,
    max_test: int = 10000,
    em_iterations: int = 10,
    max_letter_span: int = 2,
    max_phone_span: int = 2,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train both models on identical word groups and return snapshot results."""
    validate_cmudict(lexicon)
    pairs = load_cmudict(lexicon)
    root = Path(__file__).resolve().parents[2]
    revision, dirty = _git_revision(root)
    conditions: list[dict[str, Any]] = []
    for remove_stress in (False, True):
        stress = "removed" if remove_stress else "preserved"
        vec = Vectorizer(
            locale="en_US", phoneset_name="cmu", remove_stress=remove_stress
        )

        def cooked_key(word: str, *, _vec: Vectorizer = vec) -> str:
            return "\u241f".join(_vec.cook_letters(word, g2p=True))

        test_raw, train_raw = split_lexicon_by_key(
            pairs,
            key=cooked_key,
            seed=seed,
            test_fraction=test_fraction,
            max_test=max_test,
        )
        train_keys = {cooked_key(word) for word, _ in train_raw}
        test_keys = {cooked_key(word) for word, _ in test_raw}
        if train_keys & test_keys:
            raise AssertionError("normalized word leaked across train/test split")
        train_lines = [f"{word}\t{' '.join(phones)}" for word, phones in train_raw]
        multigram_prepare_started = time.perf_counter()
        train_cooked: list[tuple[list[str], list[str]]] = []
        seen_cooked: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for word, phones in train_raw:
            cooked = cook_pair(vec, word, phones)
            if cooked is None:
                continue
            cooked_id = (tuple(cooked[0]), tuple(cooked[1]))
            if cooked_id not in seen_cooked:
                seen_cooked.add(cooked_id)
                train_cooked.append(cooked)
        multigram_prepare_seconds = time.perf_counter() - multigram_prepare_started
        grouped: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        display: dict[str, str] = {}
        for word, phones in test_raw:
            cooked = cook_pair(vec, word, phones)
            if cooked is not None:
                key = cooked_key(word)
                display.setdefault(key, word)
                grouped[key].add(tuple(cooked[1]))
        test_eval = [
            (display[key], list(sorted(gold)[0])) for key, gold in grouped.items()
        ]
        gold = {display[key]: variants for key, variants in grouped.items()}

        started = time.perf_counter()
        if progress is not None:
            progress(f"training CART ({stress} stress)")
        cart = train_baseline("en_US", "cmu", train_lines, remove_stress=remove_stress)
        cart_seconds = time.perf_counter() - started
        started = time.perf_counter()
        if progress is not None:
            progress(f"training multigram ({stress} stress)")
        multigram_result = train_multigram(
            train_cooked,
            max_letter_span,
            max_phone_span,
            em_iterations,
            preprocessor=vec,
        )
        multigram = multigram_result.model
        multigram_seconds = multigram_prepare_seconds + time.perf_counter() - started
        cart_metrics = evaluate(
            "cart",
            predict_cooked_phones(vec, cart.pronounce),
            test_eval,
            gold_variants=gold,
        )
        multigram_metrics = evaluate(
            "multigram", multigram.pronounce, test_eval, gold_variants=gold
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cart_path = out / "cart.g2p.gz"
            multigram_path = out / "multigram.g2p"
            cart.export(str(cart_path))
            multigram.export(multigram_path)
            cart_bytes = _artifact_bytes(list(out.glob("cart*")))
            multigram_bytes = _artifact_bytes(list(out.glob("multigram*")))
        conditions.append(
            {
                "stress": stress,
                "train_words": len(train_keys),
                "train_entries": len(train_raw),
                "test_words": len(grouped),
                "test_entries": len(test_raw),
                "training_accounting": _training_accounting(
                    cart, train_cooked, multigram_result.metrics, len(train_raw)
                ),
                "models": [
                    {
                        "model": "G2PDecisionTree",
                        "train_seconds": cart_seconds,
                        "train_time_scope": "load, cook, align, and fit",
                        "artifact_bytes": cart_bytes,
                        **cart_metrics,
                    },
                    {
                        "model": "MultigramG2P",
                        "train_seconds": multigram_seconds,
                        "train_time_scope": "cook, deduplicate, align, and fit",
                        "artifact_bytes": multigram_bytes,
                        **multigram_metrics,
                    },
                ],
            }
        )
    return {
        "schema_version": 1,
        "snapshot": True,
        "phonebox": {
            "revision": revision,
            "dirty": dirty,
            "code_sha256": _code_fingerprint(root),
        },
        "cmudict": {
            "repository": "https://github.com/cmusphinx/cmudict",
            "commit": CMUDICT_COMMIT,
            "file": "cmudict.dict",
            "sha256": CMUDICT_SHA256,
            "license": CMUDICT_LICENSE_URL,
        },
        "parameters": {
            "seed": seed,
            "test_fraction": test_fraction,
            "max_test_words": max_test,
            "em_iterations": em_iterations,
            "max_letter_span": max_letter_span,
            "max_phone_span": max_phone_span,
            "exceptions": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor() or None,
            "dependencies": {
                name: importlib.metadata.version(name) for name in ("cartlet", "icukit")
            },
        },
        "conditions": conditions,
    }


def write_results(path: Path, result: dict[str, Any]) -> None:
    """Write deterministic, machine-readable benchmark results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def render_markdown(result: dict[str, Any]) -> str:
    """Render the public snapshot document from structured results."""
    params = result["parameters"]
    source = result["cmudict"]
    lines = [
        "# CART and multigram G2P on CMUdict",
        "",
        "> These measurements are a reproducible snapshot, not a promise about future releases. "
        "Run the refresh command below after implementation changes.",
        "",
        "Phonebox exposes two trainable grapheme-to-phoneme models. "
        "`G2PDecisionTree` aligns one cooked grapheme position to one target position "
        "and learns CART decisions from spelling context. `MultigramG2P` learns joint "
        "n:m grapheme/phone units and decodes unit sequences with a language model. "
        "Both use `Vectorizer` for the same locale normalization, stress policy, and "
        "configured joins in this benchmark.",
        "",
        "## Reproduce",
        "",
        "To reproduce this exact snapshot, check out the recorded revision and install "
        "the recorded dependencies. Running `--refresh` from newer source creates a new "
        "snapshot rather than reproducing this one.",
        "",
        "```console",
        f"git checkout {result['phonebox']['revision']}",
        "python -m pip install -e '.[dev]' "
        f"'cartlet=={result['runtime']['dependencies']['cartlet']}'",
        "phonebox compare cmudict --refresh docs/cmudict-comparison.json",
        "phonebox compare cmudict --check docs/cmudict-comparison.json docs/CMUDICT_COMPARISON.md",
        "```",
        "",
        f"Data: [CMUdict]({source['repository']}) commit `{source['commit']}`, "
        f"`cmudict.dict` SHA-256 `{source['sha256']}`. Its "
        f"[license]({source['license']}) permits research and commercial use. "
        "CMUdict is Copyright Carnegie Mellon University, which requests "
        "acknowledgement of its origin.",
        "",
        f"Split: seed {params['seed']}, {params['test_fraction']:.0%} held out, "
        f"capped at {params['max_test_words']} normalized word identities. All variants "
        "of a cooked spelling remain on one side. Identical pronunciations after the "
        "selected phone mapping are deduplicated. Exceptions are disabled.",
        "The CART row uses the shared native, serial, unpruned benchmark helper. "
        "This is an explicit evaluation setting; the primary training workflow "
        "currently prunes by default. Exact helper behavior belongs to the recorded "
        "Phonebox source revision.",
        "",
        "WER is error against the first deterministic gold variant; WERv accepts any "
        "gold variant. PER uses the first variant and divides edits by its phone count. "
        "PERv selects the gold variant with "
        "the fewest phone edits for each word (ties use lexical phone order), then divides "
        "total edits by total phones in those references. It may exceed 100% when insertions "
        "outnumber reference phones. The policy is identical for both models. "
        "Artifact size includes every file required to reload the exported model.",
        "Training time includes each model's phone cooking, alignment, and fit. The "
        "CART export is gzip-compressed while multigram uses JSON sidecars, so size is "
        "the current complete export footprint rather than normalized complexity. "
        "Prediction errors and empty outputs are counted explicitly.",
        "",
        "## Results",
        "",
        "| Stress | Model | Source train words | Test words | Train s | Size bytes | WER% | WERv% | PER% | PERv% | Errors | Empty |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in result["conditions"]:
        for model in condition["models"]:
            lines.append(
                f"| {condition['stress']} | {model['model']} | "
                f"{condition['train_words']} | {condition['test_words']} | "
                f"{model['train_seconds']:.2f} | {model['artifact_bytes']} | "
                f"{model['wer_pct']:.2f} | {model['wer_relaxed_pct']:.2f} | "
                f"{model['per_reference_pct']:.2f} | {model['per_variant_pct']:.2f} | "
                f"{model['prediction_errors']} | {model['empty_predictions']} |"
            )
    lines.extend(["", "## Interpretation", ""])
    for condition in result["conditions"]:
        winner = min(condition["models"], key=lambda model: model["per_variant_pct"])
        lines.append(
            f"- With stress {condition['stress']}, {winner['model']} has the lower "
            f"variant-aware phone error ({winner['per_variant_pct']:.2f}%)."
        )
    lines.extend(
        [
            "- Variant-aware WER/PER credit alternate held-out pronunciations; compare "
            "them within the same stress condition.",
            "- Stress-preserved and stress-removed rows are different prediction tasks; "
            "their error rates are not direct measures of one task improving.",
            "- Training times are one observed run on the recorded environment, not a "
            "throughput guarantee.",
            "- Export byte totals reflect the current gzip CART and JSON-sidecar "
            "multigram formats, not normalized algorithm complexity.",
        ]
    )
    if all("training_accounting" in item for item in result["conditions"]):
        lines.extend(["", "## Training accounting", ""])
        for condition in result["conditions"]:
            accounting = condition["training_accounting"]
            cart = accounting["G2PDecisionTree"]
            multigram = accounting["MultigramG2P"]
            lines.extend(
                [
                    f"### Stress {condition['stress']}",
                    "",
                    f"Both training pipelines started from "
                    f"{accounting['raw_entries_supplied_each']} raw entries from the "
                    "same word-group split.",
                    "",
                    f"- CART saw {cart['unique_cooked_candidates']} distinct cooked "
                    f"candidates, retained {cart['retained_entries']}, and skipped "
                    f"{cart['skipped_after_dedup']} after deduplication because its 1:1 "
                    "alignment or combination cap could not admit them.",
                    f"- Multigram saw {multigram['unique_cooked_candidates']} distinct "
                    f"cooked candidates, aligned {multigram['aligned_entries']}, and "
                    f"skipped {multigram['skipped_entries']} during n:m training.",
                    "",
                ]
            )
    phonebox = result["phonebox"]
    runtime = result["runtime"]
    lines.extend(
        [
            "",
            "## Snapshot provenance",
            "",
            f"- Phonebox revision: `{phonebox['revision']}` "
            f"({'dirty' if phonebox['dirty'] else 'clean'})",
            f"- Workload code SHA-256: `{phonebox['code_sha256']}`",
            f"- Python: {runtime['implementation']} {runtime['python']}",
            f"- Platform: {runtime['platform']}",
            f"- Dependencies: {runtime['dependencies']}",
            f"- Multigram EM iterations: {params['em_iterations']}; spans "
            f"{params['max_letter_span']}:{params['max_phone_span']}",
            "",
        ]
    )
    return "\n".join(lines)
