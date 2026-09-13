"""Render measured benchmark results without mixing incompatible data splits."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

SYSTEM_LABELS = {
    "cart": "Phonebox CART",
    "multigram": "Phonebox n:m",
    "sequitur": "Sequitur",
    "phonetisaurus": "Phonetisaurus",
}


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be a finite nonnegative number") from error
    if not finite or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


def _count(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < (1 if positive else 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return value


def _validate_row(result: Mapping[str, Any]) -> None:
    if _count(result["schema_version"], "schema version") != 1:
        raise ValueError("unsupported benchmark result schema")
    dataset, metrics = result["dataset"], result["metrics"]
    if not isinstance(dataset["preparation"]["remove_stress"], bool):
        raise ValueError("remove_stress must be a boolean")
    for split in ("train", "dev", "test"):
        counts = dataset["counts"][split]
        for key in (
            "source_entries",
            "prepared_entries",
            "words",
            "duplicates_removed",
        ):
            _count(counts[key], f"{split} {key}")
        if (
            not 0
            < counts["words"]
            <= counts["prepared_entries"]
            <= counts["source_entries"]
        ):
            raise ValueError(f"inconsistent {split} population counts")
        if (
            counts["source_entries"] - counts["prepared_entries"]
            != counts["duplicates_removed"]
        ):
            raise ValueError(f"inconsistent {split} duplicate count")
        digest = dataset["prepared_sha256"][split]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid {split} prepared SHA-256")
    n_test = _count(metrics["n_test"], "evaluation population", positive=True)
    if n_test != dataset["counts"]["test"]["words"]:
        raise ValueError("evaluation population differs from test words")
    for key in ("empty_predictions", "prediction_errors"):
        if _count(metrics[key], key) > n_test:
            raise ValueError(f"{key} exceeds evaluation population")
    if _number(metrics["wer_relaxed_pct"], "WER") > 100:
        raise ValueError("WER must be between zero and 100")
    _number(metrics["per_variant_pct"], "PER")
    _count(result["training"]["model_bytes"], "model bytes")


def render_benchmark_report(results: Sequence[Mapping[str, Any]]) -> str:
    """Render aggregate JSON, rejecting duplicate rows or mismatched populations.

    A condition is a dataset name and its preparation policy. Within a condition,
    all systems must have exactly the same prepared train/dev/test digests and
    evaluation population. Missing systems are shown explicitly.
    """
    if not results:
        raise ValueError("at least one benchmark result is required")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    try:
        for result in results:
            _validate_row(result)
            dataset = result["dataset"]
            name = dataset["name"]
            if name not in {"cmudict", "french", "italian"}:
                raise ValueError(f"unknown benchmark dataset: {name!r}")
            if result["system"] not in SYSTEM_LABELS:
                raise ValueError("unknown benchmark system")
            key = json.dumps([name, dataset["preparation"]], sort_keys=True)
            groups.setdefault(key, []).append(result)

        lines = [
            "# Measured G2P comparisons",
            "",
            "Generated from aggregate benchmark JSON. See "
            "[protocol and reproduction instructions](REPRODUCIBLE_BENCHMARKS.md).",
            "",
            "WER accepts any reference pronunciation; PER uses the minimum-edit "
            "reference and its phone count. Missing predictions remain errors. "
            "Each condition uses identical prepared splits across systems. "
            "These are measured baseline runs, not historical-paper replications.",
            "",
        ]
        for key in sorted(groups):
            rows = groups[key]
            data = rows[0]["dataset"]
            systems: set[str] = set()
            for row in rows:
                if row["system"] in systems:
                    raise ValueError("duplicate system result within a condition")
                systems.add(row["system"])
                if row["dataset"]["prepared_sha256"] != data["prepared_sha256"]:
                    raise ValueError("cannot compare different prepared splits")
                if row["dataset"]["counts"] != data["counts"]:
                    raise ValueError("cannot compare different dataset populations")
            stress = "removed" if data["preparation"]["remove_stress"] else "preserved"
            title = {
                "cmudict": f"CMUdict, stress {stress}",
                "french": "French, medium resource",
                "italian": "Italian, low resource",
            }[data["name"]]
            lines += [f"## {title}", ""]
            counts = data["counts"]
            lines += [
                f"Train/dev/test words: {counts['train']['words']:,} / "
                f"{counts['dev']['words']:,} / {counts['test']['words']:,}.",
                "",
                "| System | WER (%) | PER (%) | Empty predictions | Train + export (s) | Predict (s) | Model bytes |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for row in sorted(
                rows, key=lambda item: list(SYSTEM_LABELS).index(item["system"])
            ):
                metrics, timing, training = (
                    row["metrics"],
                    row["timings"],
                    row["training"],
                )
                values = [
                    _number(metrics["wer_relaxed_pct"], "WER"),
                    _number(metrics["per_variant_pct"], "PER"),
                    _number(metrics["empty_predictions"], "empty predictions"),
                    _number(
                        _number(timing["training_seconds"], "training time")
                        + _number(timing["export_seconds"], "export time"),
                        "training plus export time",
                    ),
                    _number(timing["prediction_seconds"], "prediction time"),
                    _number(training["model_bytes"], "model bytes"),
                ]
                lines.append(
                    f"| {SYSTEM_LABELS[row['system']]} | {values[0]:.2f} | {values[1]:.2f} | "
                    f"{values[2]:.0f} | {values[3]:.2f} | {values[4]:.2f} | {values[5]:.0f} |"
                )
            absent = [
                label
                for system, label in SYSTEM_LABELS.items()
                if system not in systems
            ]
            lines += [""]
            if absent:
                lines += [f"Not measured in this artifact: {', '.join(absent)}.", ""]
            lines += [
                "Prepared split SHA-256:",
                "",
                *[
                    f"- {split}: `{data['prepared_sha256'][split]}`"
                    for split in ("train", "dev", "test")
                ],
                "",
            ]
        lines += [
            "## Scope",
            "",
            "Full configuration, source/dependency provenance, training admission "
            "counts, and error accounting are retained in the accompanying JSON. "
            "Train + export includes the recorded training procedure (including "
            "development selection where used) and export; external training "
            "already includes export. Shared-machine timing is descriptive. "
            "The 100-word Italian test is a small witness, not a precise ranking.",
            "",
        ]
        return "\n".join(lines)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"malformed benchmark result: {error}") from error
