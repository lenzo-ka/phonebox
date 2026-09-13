"""Run fixed G2P systems on prepared splits without model-specific text changes.

External programs are optional executables, never Python runtime dependencies.
Their raw logs and artifacts stay in the caller's dedicated work directory.
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
from phonebox.core.g2p_model import G2PDecisionTree
from phonebox.core.multigram_g2p import MultigramG2P
from phonebox.core.vectorizer import Vectorizer
from phonebox.eval.benchmark_data import _split_digest
from phonebox.eval.cmudict_compare import _code_fingerprint, _git_revision, sha256_file
from phonebox.eval.g2p_compare import evaluate

if TYPE_CHECKING:
    from phonebox.eval.benchmark_data import PreparedDataset

SYSTEMS = ("cart", "multigram", "sequitur", "phonetisaurus")
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
        for field in ("version", "source_revision", "build"):
            if field in data:
                result[field] = data[field]
        _validate_receipt(result)
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"nonfinite JSON constant in benchmark provenance: {value}")


def _validate_receipt(value: Any) -> None:
    """Reject deployment identities while allowing source URLs and relative filenames."""
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
            "max_iterations": 10,
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
            max_iterations=10,
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
            "candidate_entries": len(model.em.seen),
            "retained_entries": retained,
            "skipped_entries": len(dataset.train) - retained,
        }
        artifacts = [directory / "model.g2p.gz"]
    else:
        settings = {
            "max_letter_span": 2,
            "max_phone_span": 2,
            "min_phone_span": 0,
            "em_max_iterations": 10,
            "lm_order": 2,
            "parallel_align": False,
            "parallel_viterbi": False,
            "use_dict_fallback": False,
        }
        model = MultigramG2P(
            max_letter_span=2,
            max_phone_span=2,
            min_phone_span=0,
            em_max_iterations=10,
            lm_order=2,
            parallel_align=False,
            parallel_viterbi=False,
            preprocessor=vectorizer,
        )
        metrics = model.train_from_pairs(
            [(list(word), phones) for word, phones in dataset.train]
        )
        accounting = {
            "retained_entries": metrics["aligned_entries"],
            "skipped_entries": metrics["skipped_entries"],
        }
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


def _sequitur(dataset: PreparedDataset, directory: Path, executable: Path):
    identity = _tool_identity(executable, Path(str(executable) + ".provenance.json"))
    train, dev = directory / "train.lex", directory / "dev.lex"
    _write_pairs(train, dataset.train)
    _write_pairs(dev, dataset.dev)
    dev_words = _write_words(directory / "dev.words", dataset.dev)
    inventory = {phone for _, phones in dataset.train for phone in phones}
    settings = {
        "orders": [1, 2, 3],
        "min_iterations": 1,
        "max_iterations": 10,
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
            "1",
            "--max-iterations",
            "10",
            "--write-model",
            str(model),
        ]
        if previous is not None:
            command.extend(["--model", str(previous), "--ramp-up"])
        _run(command, directory, f"train-{order}")
        if not model.is_file():
            raise ValueError("Sequitur did not create its requested model")
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
    phonetisaurus_prefix: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train one fixed system and score every unique held-out spelling.

    ``work_dir`` must be empty or absent. Results are JSON serializable and omit
    deployment paths; model files and detailed subprocess logs stay on disk.
    External missing predictions count as errors and empty predictions through
    the same evaluator used by the native systems. Development data is used only
    for declared model selection; held-out test references never select settings.
    """
    if system not in SYSTEMS:
        raise ValueError(f"unknown benchmark system: {system}")
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
        fitted = _sequitur(dataset, directory, _executable(sequitur_executable))
    elif system == "phonetisaurus":
        if phonetisaurus_prefix is None:
            raise ValueError("Phonetisaurus requires an installation prefix")
        fitted = _phonetisaurus(
            dataset, directory, Path(phonetisaurus_prefix).resolve()
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
            "threads": 1,
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
