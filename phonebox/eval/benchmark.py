"""Run fixed G2P systems on prepared splits without model-specific text changes.

External programs are optional executables, never Python runtime dependencies.
Their raw logs and artifacts stay in the caller's dedicated work directory.
Adapters invoke the authors' implementations as subprocesses; their algorithms
are not copied here and their separate software licenses apply.

Sequitur: Bisani and Ney (2008), joint-sequence models,
https://doi.org/10.1016/j.specom.2008.01.002;
https://github.com/sequitur-g2p/sequitur-g2p.

Phonetisaurus: Novak, Minematsu, and Hirose (2016), "Phonetisaurus: Exploring
grapheme-to-phoneme conversion with joint n-gram models in the WFST framework,"
Natural Language Engineering 22(6), 907–938;
https://www.cambridge.org/core/journals/natural-language-engineering/article/phonetisaurus-exploring-graphemetophoneme-conversion-with-joint-ngram-models-in-the-wfst-framework/F1160C3866842F0B707924EB30B8E753;
https://github.com/AdolfVonKleist/Phonetisaurus.

See docs/BENCHMARK_TOOLCHAINS.md and docs/REPRODUCIBLE_BENCHMARKS.md for pinned
toolchains, dataset citations, and license notices.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import time
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from itertools import chain
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any

from phonebox.constants import JOIN_CHAR
from phonebox.core.em_align import EMAlign
from phonebox.core.g2p_model import G2PDecisionTree
from phonebox.core.multigram_align import MultigramAligner
from phonebox.core.multigram_g2p import MultigramG2P
from phonebox.core.multigram_lm import MultigramLM
from phonebox.core.vectorizer import Vectorizer
from phonebox.eval.benchmark_data import _split_digest
from phonebox.eval.benchmark_systems import SYSTEMS
from phonebox.eval.cmudict_compare import _code_fingerprint, _git_revision, sha256_file
from phonebox.eval.g2p_compare import evaluate

if TYPE_CHECKING:
    from phonebox.eval.benchmark_data import PreparedDataset
    from phonebox.eval.benchmark_neural import NeuralSettings

_THREAD_VARIABLES = (
    "BLIS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_IDENTITY = {
    "version": 1,
    "source": {"norm_rules": None, "g2p_rules": None},
    "join_char": JOIN_CHAR,
    "letter_joins": [],
    "cased": True,
    "remove_accents": False,
    "filter_non_letters": False,
    "spelling_rewrites": {},
}


def _identity_vectorizer(phoneset: str) -> Vectorizer:
    vectorizer = Vectorizer(
        phoneset_name=phoneset,
        remove_stress=False,
        remove_accents=False,
        filter_non_letters=False,
        cased=True,
        letter_preprocessing=_IDENTITY,
    )
    vectorizer.phon_join_re = None
    vectorizer.liaison_pad = None
    return vectorizer


def _gold(pairs: list[tuple[str, list[str]]]):
    variants: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for word, phones in pairs:
        variants[word].add(tuple(phones))
    return [(word, list(min(phones))) for word, phones in variants.items()], dict(
        variants
    )


def _metrics(predict: Callable[[str], list[str]], pairs: list[tuple[str, list[str]]]):
    population, variants = _gold(pairs)
    return evaluate("benchmark", predict, population, gold_variants=variants)


def _validate_dataset(dataset: PreparedDataset) -> None:
    populations = []
    for name in ("train", "dev", "test"):
        pairs = getattr(dataset, name)
        if not pairs:
            raise ValueError(f"benchmark {name} split must be nonempty")
        words = set()
        for word, phones in pairs:
            if not isinstance(word, str) or not word or any(c.isspace() for c in word):
                raise ValueError(
                    "benchmark spellings must be nonempty whitespace-free strings"
                )
            if not phones or any(
                not isinstance(p, str) or not p or any(c.isspace() for c in p)
                for p in phones
            ):
                raise ValueError(
                    "benchmark phones must be nonempty whitespace-free strings"
                )
            if (
                unicodedata.normalize("NFC", word) != word
                or "\0" in word
                or any("\0" in phone for phone in phones)
            ):
                raise ValueError(
                    "benchmark inputs must have NFC spellings and no NUL characters"
                )
            words.add(word)
        if dataset.metadata.get("prepared_sha256", {}).get(name) != _split_digest(
            pairs
        ):
            raise ValueError(
                f"benchmark {name} split differs from recorded prepared hash"
            )
        counts = dataset.metadata.get("counts", {}).get(name, {})
        if counts.get("prepared_entries") != len(pairs) or counts.get("words") != len(
            words
        ):
            raise ValueError(f"benchmark {name} split differs from recorded counts")
        populations.append(words)
    if any(populations[i] & populations[j] for i in range(3) for j in range(i)):
        raise ValueError("benchmark spelling leaked across train/dev/test splits")


def _write_pairs(path: Path, pairs: list[tuple[str, list[str]]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for word, phones in pairs:
            stream.write(f"{word}\t{' '.join(phones)}\n")


def _write_words(path: Path, pairs: list[tuple[str, list[str]]]) -> set[str]:
    words = dict.fromkeys(word for word, _ in pairs)
    path.write_text("".join(word + "\n" for word in words), encoding="utf-8")
    return set(words)


def _executable(path: str | Path) -> Path:
    resolved = shutil.which(str(path))
    if resolved is None:
        raise ValueError(f"benchmark executable is unavailable: {Path(path).name}")
    return Path(resolved).resolve()


def _tool_identity(executable: Path, receipt: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "executable": executable.name,
        "sha256": sha256_file(executable),
        "version": None,
        "source_revision": None,
    }
    if receipt.exists():
        data = json.loads(
            receipt.read_text(encoding="utf-8"), parse_constant=_invalid_constant
        )
        if not isinstance(data, dict):
            raise ValueError("benchmark tool receipt must be an object")
        for field in ("version", "source_revision"):
            if data.get(field) is not None and not isinstance(data[field], str):
                raise ValueError(
                    f"benchmark tool receipt {field} must be a string or null"
                )
        if "build" in data and not isinstance(data["build"], dict):
            raise ValueError("benchmark tool receipt build must be an object")
        has_binding = "executable_sha256" in data.get("build", {})
        binding = data.get("build", {}).get("executable_sha256")
        if has_binding and binding != result["sha256"]:
            raise ValueError(
                "benchmark tool receipt executable hash differs from the binary"
            )
        for field in ("version", "source_revision", "build"):
            if field in data:
                result[field] = data[field]
        _validate_receipt(result)
        result["receipt_binary_binding_verified"] = has_binding
    else:
        result["receipt_binary_binding_verified"] = False
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"nonfinite JSON constant in benchmark provenance: {value}")


def _validate_receipt(value: Any) -> None:
    """Reject a known set of identity keys and any absolute path.

    This is a shape check: the fixed key set below and absolute POSIX or Windows
    paths in any string. An identity under an unlisted key, or a machine name
    written into prose, passes. Source URLs and relative filenames are allowed.
    """
    if isinstance(value, dict):
        forbidden = {
            "path",
            "paths",
            "hostname",
            "host",
            "username",
            "user",
            "command",
            "command_line",
            "cwd",
        }
        if any(key.lower() in forbidden for key in value):
            raise ValueError("benchmark provenance contains deployment identity fields")
        for item in value.values():
            _validate_receipt(item)
    elif isinstance(value, list):
        for item in value:
            _validate_receipt(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("benchmark provenance must contain finite numbers")
    elif isinstance(value, str):
        without_urls = re.sub(r"https?://[^\s]+", "", value)
        if (
            Path(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or re.search(r"(?:^|[\s=:(])(?:/|[A-Za-z]:[\\/]|~[/\\])", without_urls)
        ):
            raise ValueError(
                "benchmark provenance contains an absolute deployment path"
            )


def _run(command: list[str], directory: Path, label: str) -> Path:
    environment = dict(os.environ)
    environment.update(dict.fromkeys(_THREAD_VARIABLES, "1"))
    environment["LC_ALL"] = "C.UTF-8"
    output = directory / f"{label}.stdout"
    with (
        output.open("w", encoding="utf-8") as stdout,
        (directory / f"{label}.stderr").open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=directory,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode:
        raise ValueError(
            f"{label} failed with exit status {completed.returncode}; see work-directory logs"
        )
    return output


def _predictions(
    path: Path, words: set[str], phones: set[str], *, scored: bool = False
) -> dict[str, list[str]]:
    """Parse one-best output strictly; absent words remain absent for evaluation."""
    result = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != (3 if scored else 2):
                raise ValueError(f"malformed prediction record {number}")
            word = fields[0]
            if word not in words or word in result:
                raise ValueError(
                    f"unknown or duplicate prediction word at record {number}"
                )
            if scored:
                try:
                    score = float(fields[1])
                except ValueError as exc:
                    raise ValueError(
                        f"invalid prediction score at record {number}"
                    ) from exc
                if not math.isfinite(score):
                    raise ValueError(f"nonfinite prediction score at record {number}")
            predicted = fields[-1].split()
            if any(phone not in phones for phone in predicted):
                raise ValueError(f"unknown phone token in prediction record {number}")
            result[word] = predicted
    return result


def _cart_convergence(aligner: EMAlign) -> dict[str, Any]:
    history = aligner.alignment_history
    if not history:
        raise ValueError("CART alignment did not record convergence evidence")
    last = history[-1]
    converged = not last["changed"] or last["ratio"] < aligner.min_change_ratio
    if len(history) > aligner.max_iterations or (
        not converged and len(history) < aligner.max_iterations
    ):
        raise ValueError("CART alignment recorded an incomplete or over-cap trace")
    return {
        "criterion": "changed == 0 or changed-entry ratio < threshold",
        "threshold": aligner.min_change_ratio,
        "max_iterations": aligner.max_iterations,
        "iterations": len(history),
        "converged": converged,
        "cap_censored": not converged and len(history) >= aligner.max_iterations,
        "stop_reason": "converged" if converged else "iteration_limit",
        "history": history,
    }


def _multigram_convergence(aligner: MultigramAligner) -> dict[str, Any]:
    history = aligner.loglik_history
    if not history or not all(map(math.isfinite, history)):
        raise ValueError(
            "multigram alignment did not record finite convergence evidence"
        )
    relative = (
        abs(history[-1] - history[-2]) / max(1.0, abs(history[-2]))
        if len(history) > 1
        else None
    )
    converged = relative is not None and relative < aligner.conv
    if len(history) > aligner.max_iter or (
        not converged and len(history) < aligner.max_iter
    ):
        raise ValueError("multigram alignment recorded an incomplete or over-cap trace")
    return {
        "criterion": "absolute relative observed log-likelihood change < threshold",
        "threshold": aligner.conv,
        "max_iterations": aligner.max_iter,
        "iterations": len(history),
        "converged": converged,
        "cap_censored": not converged and len(history) >= aligner.max_iter,
        "stop_reason": "converged" if converged else "iteration_limit",
        "last_relative_change": relative,
        "history_semantics": "log-likelihood observed before each M-step, not a final-model rescore",
        "observed_pre_update_loglik_history": history,
    }


def _native(dataset: PreparedDataset, system: str, directory: Path):
    vectorizer = _identity_vectorizer(dataset.metadata.get("phoneset", "ipa"))
    # These checks catch reserved syntax/phone-join tokens before they silently
    # change a prepared dataset. The benchmark does not invent another parser.
    for word, phones in chain(dataset.train, dataset.dev, dataset.test):
        if (
            vectorizer.cook_letters(word, g2p=True) != list(word)
            or vectorizer.cook_phones(phones) != phones
            or vectorizer.uncook(phones) != phones
        ):
            raise ValueError(
                "prepared benchmark tokens change under Phonebox's identity preprocessing"
            )
    started = time.perf_counter()
    model: Any
    if system == "cart":
        settings = {
            "trainer": "native",
            "width": 7,
            "max_combinations": 5000,
            "max_iterations": 100,
            "parallel_align": False,
            "prune": False,
            "use_dict_fallback": False,
        }
        model = G2PDecisionTree(
            phoneset_name=vectorizer.phoneset_name,
            cased=True,
            remove_accents=False,
            filter_non_letters=False,
            trainer="native",
            width=7,
            max_combinations=5000,
            max_iterations=100,
            parallel_align=False,
            use_dict_fallback=False,
        )
        model.vectorizer = vectorizer
        lines = [f"{word}\t{' '.join(phones)}" for word, phones in dataset.train]
        for line, (word, phones) in zip(lines, dataset.train, strict=True):
            if vectorizer.letters_and_phones(line) != (list(word), phones):
                raise ValueError(
                    "prepared benchmark entry is not representable by the CART dictionary parser"
                )
        model.load_prondict(lines)
        model.align()
        model.train(prune=False)
        retained = len(model.em.init_data)
        accounting = {
            "convergence": _cart_convergence(model.em),
            "candidate_entries": len(model.em.seen),
            "retained_entries": retained,
            "skipped_entries": len(dataset.train) - retained,
        }
        settings["min_change_ratio"] = model.em.min_change_ratio
        artifacts = [directory / "model.g2p.gz"]
    else:
        settings = {
            "g2p_version": MultigramG2P.VERSION,
            "lm_version": MultigramLM.VERSION,
            "scoring": MultigramG2P.SCORING,
            "max_letter_span": 2,
            "max_phone_span": 2,
            "min_phone_span": 0,
            "em_max_iterations": 100,
            "lm_order": 2,
            "parallel_align": False,
            "parallel_viterbi": False,
            "use_dict_fallback": False,
        }
        model = MultigramG2P(
            max_letter_span=2,
            max_phone_span=2,
            min_phone_span=0,
            em_max_iterations=100,
            lm_order=2,
            parallel_align=False,
            parallel_viterbi=False,
            preprocessor=vectorizer,
        )
        metrics = model.train_from_pairs(
            [(list(word), phones) for word, phones in dataset.train]
        )
        accounting = {
            "convergence": _multigram_convergence(model.aligner),
            "retained_entries": metrics["aligned_entries"],
            "skipped_entries": metrics["skipped_entries"],
        }
        settings.update(
            {
                "em_convergence_threshold": model.aligner.conv,
                "min_unit_mass": model.aligner.min_unit_mass,
                "decode_beam": model.decode_beam,
                "lm_add_k": model.lm.add_k,
            }
        )
        artifacts = list(model.export_paths(directory / "model.g2p"))
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    # Model-only benchmarks exclude dictionary storage as well as lookup.
    model.exceptions = {}
    if system == "cart":
        model.export(str(directory / "model.g2p.gz"), include_exceptions=False)
    else:
        model.export(directory / "model.g2p")
    accounting["dictionary_entries"] = 0
    export_seconds = time.perf_counter() - started
    return (
        model.pronounce,
        settings,
        accounting,
        artifacts,
        training_seconds,
        export_seconds,
        {},
    )


def _sequitur_stop(output: Path) -> dict[str, Any]:
    """Read upstream stopping evidence; never infer convergence from exit status."""
    log = output.read_text(encoding="utf-8")
    if "iteration failed." in log:
        raise ValueError("Sequitur reported a failed training iteration")
    if "iteration converged." in log:
        reason = "converged"
    elif "maximum number of iterations reached." in log:
        reason = "iteration_limit"
    else:
        raise ValueError("Sequitur did not report its training stop reason")
    iterations = re.findall(r"^iteration: (\d+)$", log, re.MULTILINE)
    try:
        likelihoods = [
            float(value)
            for value in re.findall(r"^LL devel:[ \t]*(.*)$", log, re.MULTILINE)
        ]
    except ValueError as error:
        raise ValueError(
            "Sequitur training log has malformed development evidence"
        ) from error
    if (
        not iterations
        or len(likelihoods) != len(iterations)
        or not all(map(math.isfinite, likelihoods))
    ):
        raise ValueError("Sequitur training log lacks finite development evidence")
    return {
        "stop_reason": reason,
        "iterations": len(iterations),
        "last_dev_log_likelihoods": likelihoods[-2:],
    }


def _sequitur(
    dataset: PreparedDataset,
    directory: Path,
    executable: Path,
    min_iterations: int,
    max_iterations: int,
    extension_iterations: int,
):
    identity = _tool_identity(executable, Path(str(executable) + ".provenance.json"))
    train, dev = directory / "train.lex", directory / "dev.lex"
    _write_pairs(train, dataset.train)
    _write_pairs(dev, dataset.dev)
    dev_words = _write_words(directory / "dev.words", dataset.dev)
    inventory = {phone for _, phones in dataset.train for phone in phones}
    settings: dict[str, Any] = {
        "orders": [1, 2, 3],
        "min_iterations": min_iterations,
        "max_iterations": max_iterations,
        "extension_iterations": extension_iterations,
        "extension_policy": "fresh order restart if upstream iteration limit reached",
        "training_attempts": [],
        "selection": ["dev per_variant_pct", "dev wer_relaxed_pct", "lower order"],
    }
    candidates = []
    previous = None
    started = time.perf_counter()
    for order in (1, 2, 3):
        model = directory / f"model-{order}"
        command = [
            str(executable),
            "--encoding",
            "UTF-8",
            "--train",
            str(train),
            "--devel",
            str(dev),
            "--min-iterations",
            str(min_iterations),
            "--max-iterations",
            str(max_iterations),
            "--write-model",
            str(model),
        ]
        if previous is not None:
            command.extend(["--model", str(previous), "--ramp-up"])
        for cap in dict.fromkeys((max_iterations, extension_iterations)):
            if cap != max_iterations:
                model = directory / f"model-{order}-extended"
                command[command.index("--max-iterations") + 1] = str(cap)
                command[command.index("--write-model") + 1] = str(model)
            output = _run(command, directory, f"train-{order}-max-{cap}")
            evidence = _sequitur_stop(output)
            settings["training_attempts"].append(
                {"order": order, "max_iterations": cap, **evidence}
            )
            if not model.is_file():
                raise ValueError("Sequitur did not create its requested model")
            if evidence["stop_reason"] == "converged":
                break
        output = _run(
            [
                str(executable),
                "--encoding",
                "UTF-8",
                "--model",
                str(model),
                "--apply",
                str(directory / "dev.words"),
            ],
            directory,
            f"dev-{order}",
        )
        predictions = _predictions(output, dev_words, inventory)
        metrics = _metrics(predictions.__getitem__, dataset.dev)
        candidates.append(
            (
                metrics["per_variant_pct"],
                metrics["wer_relaxed_pct"],
                order,
                model,
                metrics,
            )
        )
        previous = model
    _, _, selected, model, _ = min(candidates, key=lambda item: item[:3])
    training_seconds = time.perf_counter() - started
    settings["selected_order"] = selected
    settings["iteration_limited_orders"] = [
        order
        for order in (1, 2, 3)
        if [
            attempt
            for attempt in settings["training_attempts"]
            if attempt["order"] == order
        ][-1]["stop_reason"]
        == "iteration_limit"
    ]
    settings["development_results"] = [
        {"order": candidate[2], "metrics": candidate[4]} for candidate in candidates
    ]
    words = _write_words(directory / "test.words", dataset.test)
    started = time.perf_counter()
    output = _run(
        [
            str(executable),
            "--encoding",
            "UTF-8",
            "--model",
            str(model),
            "--apply",
            str(directory / "test.words"),
        ],
        directory,
        "test",
    )
    predictions = _predictions(output, words, inventory)
    prediction_seconds = time.perf_counter() - started
    accounting = {
        "retained_entries": None,
        "skipped_entries": None,
        "retention_note": "upstream retention is not exposed by this adapter",
    }
    return (
        predictions.__getitem__,
        settings,
        accounting,
        [model],
        training_seconds,
        0.0,
        {"tool": identity, "external_prediction_seconds": prediction_seconds},
    )


def _phonetisaurus(dataset: PreparedDataset, directory: Path, prefix: Path):
    names = (
        "phonetisaurus-align",
        "estimate-ngram",
        "phonetisaurus-arpa2wfst",
        "phonetisaurus-g2pfst",
    )
    executables = {name: _executable(prefix / "bin" / name) for name in names}
    identities = {
        name: _tool_identity(executable, Path(str(executable) + ".provenance.json"))
        for name, executable in executables.items()
    }
    train, corpus, arpa, model = (
        directory / name
        for name in ("train.lex", "aligned.corpus", "model.arpa", "model.fst")
    )
    _write_pairs(train, dataset.train)
    settings = {
        "max_letter_span": 2,
        "max_phone_span": 2,
        "seq1_del": False,
        "seq2_del": True,
        "alignment_iterations": 11,
        "restrict": True,
        "grow": False,
        "penalize": True,
        "penalize_em": False,
        "lm_order": 8,
        "nbest": 1,
        "beam": 10000,
        "thresh": 99,
        "pmass": 0,
        "development_selection": None,
    }
    started = time.perf_counter()
    _run(
        [
            str(executables["phonetisaurus-align"]),
            f"--input={train}",
            f"--ofile={corpus}",
            "--seq1_del=false",
            "--seq2_del=true",
            "--seq1_max=2",
            "--seq2_max=2",
            "--iter=11",
            "--restrict=true",
            "--grow=false",
            "--penalize=true",
            "--penalize_em=false",
        ],
        directory,
        "align",
    )
    if not corpus.is_file() or not corpus.stat().st_size:
        raise ValueError("Phonetisaurus alignment did not produce a nonempty corpus")
    _run(
        [
            str(executables["estimate-ngram"]),
            "-o",
            "8",
            "-t",
            str(corpus),
            "-wl",
            str(arpa),
        ],
        directory,
        "estimate-ngram",
    )
    if not arpa.is_file() or not arpa.stat().st_size:
        raise ValueError("Phonetisaurus language-model estimation did not produce ARPA")
    _run(
        [
            str(executables["phonetisaurus-arpa2wfst"]),
            f"--lm={arpa}",
            f"--ofile={model}",
        ],
        directory,
        "arpa2wfst",
    )
    if not model.is_file() or not model.stat().st_size:
        raise ValueError("Phonetisaurus did not produce an FST model")
    training_seconds = time.perf_counter() - started
    words_path = directory / "test.words"
    words = _write_words(words_path, dataset.test)
    started = time.perf_counter()
    output = _run(
        [
            str(executables["phonetisaurus-g2pfst"]),
            f"--model={model}",
            f"--wordlist={words_path}",
            "--nbest=1",
            "--print_scores=false",
            "--beam=10000",
            "--thresh=99",
            "--pmass=0",
        ],
        directory,
        "test",
    )
    inventory = {phone for _, phones in dataset.train for phone in phones}
    predictions = _predictions(output, words, inventory)
    prediction_seconds = time.perf_counter() - started
    accounting = {
        "retained_entries": None,
        "skipped_entries": None,
        "retention_note": "upstream retention is not exposed by this adapter",
    }
    return (
        predictions.__getitem__,
        settings,
        accounting,
        [model],
        training_seconds,
        0.0,
        {"tools": identities, "external_prediction_seconds": prediction_seconds},
    )


def run_benchmark(
    dataset: PreparedDataset,
    system: str,
    work_dir: str | Path,
    *,
    sequitur_executable: str | Path | None = None,
    sequitur_min_iterations: int = 20,
    sequitur_max_iterations: int = 100,
    sequitur_extension_iterations: int = 200,
    phonetisaurus_prefix: str | Path | None = None,
    neural_python: str | Path | None = None,
    neural_device: str = "cpu",
    neural_settings: NeuralSettings | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train one fixed system and score every unique held-out spelling.

    ``work_dir`` must be empty or absent. Results are JSON serializable and omit
    deployment paths; model files and detailed subprocess logs stay on disk.
    Sequitur uses explicit min/max/extension iteration budgets (20/100/200).
    A capped order restarts from its original initialization at the extension
    cap; equal maximum and extension disables this diagnostic restart.
    Upstream stopping evidence is retained before any test decoding.
    External missing predictions count as errors and empty predictions through
    the same evaluator used by the native systems. Development data is used only
    for declared model selection; held-out test references never select settings.
    """
    if system not in SYSTEMS:
        raise ValueError(f"unknown benchmark system: {system}")
    budgets = (
        sequitur_min_iterations,
        sequitur_max_iterations,
        sequitur_extension_iterations,
    )
    if any(type(value) is not int or value < 1 for value in budgets) or not (
        sequitur_min_iterations
        < sequitur_max_iterations
        <= sequitur_extension_iterations
    ):
        raise ValueError(
            "Sequitur iteration budgets require 0 < min < max <= extension"
        )
    _validate_dataset(dataset)
    metadata = deepcopy(dataset.metadata)
    json.dumps(metadata, allow_nan=False)
    _validate_receipt(metadata)
    directory = Path(work_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("benchmark work directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    revision, dirty = _git_revision(root)
    source_hash = _code_fingerprint(root)
    if progress is not None:
        progress(f"Training {system} on {dataset.name}")
    if system == "sequitur":
        if sequitur_executable is None:
            raise ValueError("Sequitur requires an executable")
        fitted = _sequitur(
            dataset,
            directory,
            _executable(sequitur_executable),
            sequitur_min_iterations,
            sequitur_max_iterations,
            sequitur_extension_iterations,
        )
    elif system == "phonetisaurus":
        if phonetisaurus_prefix is None:
            raise ValueError("Phonetisaurus requires an installation prefix")
        fitted = _phonetisaurus(
            dataset, directory, Path(phonetisaurus_prefix).resolve()
        )
    elif system == "deepphonemizer":
        if neural_python is None:
            raise ValueError("DeepPhonemizer requires an isolated Python executable")
        from phonebox.eval.benchmark_neural import run_neural_training

        neural = run_neural_training(
            dataset,
            directory,
            python_executable=neural_python,
            device=neural_device,
            settings=neural_settings,
        )
        neural_predictions = json.loads(
            (directory / "neural-predictions.json").read_text(encoding="utf-8")
        )
        fitted = (
            neural_predictions.__getitem__,
            neural["settings"],
            neural["training"],
            [directory / "neural-model.pt"],
            neural["training_seconds"],
            neural["export_seconds"],
            {
                "neural_toolchain": neural["provenance"],
                "neural_test_diagnostics": neural["test_diagnostics"],
                "external_prediction_seconds": neural["prediction_seconds"],
            },
        )
    else:
        fitted = _native(dataset, system, directory)
    (
        predict,
        settings,
        accounting,
        artifacts,
        training_seconds,
        export_seconds,
        extra,
    ) = fitted
    if "external_prediction_seconds" in extra:
        prediction_seconds = extra.pop("external_prediction_seconds")
    else:
        predictions: dict[str, list[str] | Exception] = {}
        started = time.perf_counter()
        for word in dict.fromkeys(word for word, _ in dataset.test):
            try:
                predictions[word] = predict(word)
            except Exception as exc:
                predictions[word] = exc
        prediction_seconds = time.perf_counter() - started

        def cached_predict(word: str) -> list[str]:
            result = predictions[word]
            if isinstance(result, Exception):
                raise result
            return result

        predict = cached_predict
    metrics = _metrics(predict, dataset.test)
    for split, digest in metadata["prepared_sha256"].items():
        if _split_digest(getattr(dataset, split)) != digest:
            raise ValueError("benchmark dataset changed during the run")
    if source_hash != _code_fingerprint(root):
        raise ValueError("benchmark source changed during the run")
    return {
        "schema_version": 1,
        "dataset": {**metadata, "name": dataset.name},
        "system": system,
        "settings": {
            **settings,
            "threads": settings.get("threads", 1),
            "letter_preprocessing": deepcopy(_IDENTITY),
            "phone_mapping": None,
            "dictionary_lookup": False,
            "evaluation_scope": "held-out spellings; model only; dictionary lookup disabled",
            "remove_stress": False,
        },
        "metrics": metrics,
        "training": {
            "supplied_entries": len(dataset.train),
            **accounting,
            "model_bytes": sum(path.stat().st_size for path in artifacts),
        },
        "timings": {
            "training_seconds": training_seconds,
            "prediction_seconds": prediction_seconds,
            "export_seconds": export_seconds,
            "training_scope": "fit and declared development selection; external export included"
            if system in ("sequitur", "phonetisaurus")
            else "fit and full shared dev selection; export excluded"
            if system == "deepphonemizer"
            else "load/align/fit; export excluded",
        },
        "provenance": {
            "revision": revision,
            "dirty": dirty,
            "code_sha256": source_hash,
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("phonebox", "cartlet", "icukit")
            },
            **extra,
        },
    }
