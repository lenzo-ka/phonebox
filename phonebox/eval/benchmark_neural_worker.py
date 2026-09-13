"""Private subprocess boundary for the optional author neural toolchain.

No Torch import occurs in ordinary Phonebox imports or command discovery.
The credited external source is verified before training; model code is not copied.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

from phonebox.eval.benchmark import _metrics
from phonebox.eval.benchmark_neural import (
    PATCH_SHA256,
    SOURCE_REVISION,
    NeuralSettings,
    _DevSelection,
)
from phonebox.eval.benchmark_neural_sources import SOURCE_HASHES
from phonebox.eval.cmudict_compare import sha256_file


def _installed_source() -> dict[str, Any]:
    import dp

    root = Path(dp.__file__).parent
    observed = {
        str(path.relative_to(root)): sha256_file(path) for path in root.rglob("*.py")
    }
    if observed != SOURCE_HASHES:
        raise ValueError(
            "DeepPhonemizer source does not match the pinned release and documented patch"
        )
    versions = {
        name: importlib.metadata.version(name)
        for name in ("deep-phonemizer", "torch", "numpy")
    }
    if {name: version.partition("+")[0] for name, version in versions.items()} != {
        "deep-phonemizer": "0.0.19",
        "torch": "2.5.1",
        "numpy": "1.26.4",
    }:
        raise ValueError(
            "Neural toolchain requires DeepPhonemizer0.0.19, Torch2.5.1, NumPy1.26.4"
        )
    return {
        "source_revision": SOURCE_REVISION,
        "source_version": "0.0.19",
        "patch_sha256": PATCH_SHA256,
        "installed_source_sha256": hashlib.sha256(
            json.dumps(observed, sort_keys=True).encode()
        ).hexdigest(),
        "dependencies": versions,
        "installed_distributions": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
        },
    }


def _predict(model, preprocessor, words: list[str], batch_size: int):
    """Use the author's predictor, preserving tokens and explicit failures."""
    from dp.model.predictor import Predictor

    predictor = Predictor(model, preprocessor)
    supported = []
    errors: dict[str, ValueError] = {}
    for word in words:
        decoded = predictor.text_tokenizer.decode(
            predictor.text_tokenizer(word, "g2p"), remove_special_tokens=True
        )
        if decoded != list(word):
            errors[word] = ValueError(
                "Unknown input symbol; whole-word prediction refused"
            )
        else:
            supported.append(word)
    model.eval()
    outputs = predictor(supported, "g2p", batch_size=batch_size)
    phones: dict[str, list[str]] = {}
    truncated = 0
    for output in outputs:
        if (
            predictor.phoneme_tokenizer.idx_to_token[
                predictor.phoneme_tokenizer.end_index
            ]
            not in output.phoneme_tokens
        ):
            truncated += 1
        if not all(math.isfinite(value) for value in output.token_probs):
            raise ValueError("Nonfinite neural token probability")
        phones[output.word] = [
            token
            for token in output.phoneme_tokens
            if token not in predictor.phoneme_tokenizer.special_tokens
        ]
    if set(phones) != set(supported):
        raise ValueError("Neural prediction population changed")

    def lookup(word: str) -> list[str]:
        if word in errors:
            raise errors[word]
        return phones[word]

    return (
        lookup,
        {"unsupported_inputs": len(errors), "truncated_predictions": truncated},
        phones,
    )


def run(directory: Path) -> dict[str, Any]:
    """Execute a fixed job; profile jobs do not load a test file."""
    identity = _installed_source()
    import numpy as np
    import torch
    import yaml
    from dp.model.model import (
        ModelType,
        create_model,
        load_checkpoint,
    )
    from dp.preprocess import preprocess
    from dp.preprocessing.text import Preprocessor
    from dp.training.trainer import Trainer
    from dp.utils.io import unpickle_binary

    job = json.loads((directory / "neural-job.json").read_text(encoding="utf-8"))
    settings = NeuralSettings(**job["settings"])
    torch.set_num_threads(settings.threads)
    torch.set_num_interop_threads(settings.threads)
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    device = job["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this process")
    torch.use_deterministic_algorithms(device == "cpu")
    train, dev = job["train"], job["dev"]
    text_symbols = sorted({char for word, _ in train for char in word})
    phone_symbols = sorted({phone for _, phones in train for phone in phones})
    if set(text_symbols + phone_symbols) & {"_", "<end>", "<g2p>"}:
        raise ValueError("Dataset symbol collides with an upstream special token")
    config = {
        "paths": {
            "data_dir": str(directory / "data"),
            "checkpoint_dir": str(directory / "checkpoints"),
        },
        "preprocessing": {
            "languages": ["g2p"],
            "text_symbols": text_symbols,
            "phoneme_symbols": phone_symbols,
            "lowercase": False,
            "char_repeats": 1,
            "n_val": len(dev),
        },
        "model": {
            "type": "autoreg_transformer",
            "d_model": settings.d_model,
            "d_fft": settings.d_fft,
            "layers": settings.layers,
            "heads": settings.heads,
            "dropout": settings.dropout,
        },
        "training": {
            "learning_rate": settings.learning_rate,
            "warmup_steps": settings.warmup_steps,
            "batch_size": settings.batch_size,
            "batch_size_val": settings.batch_size,
            "epochs": settings.max_epochs,
            "generate_steps": 10**12,
            "validate_steps": 10**12,
            "checkpoint_steps": 10**12,
            "n_generate_samples": 0,
            "store_phoneme_dict_in_model": False,
            "scheduler_plateau_factor": 0.5,
            "scheduler_plateau_patience": settings.plateau_patience,
        },
    }
    preprocessor = Preprocessor.from_config(config)
    encoded_dev = []
    for split, pairs in (("train", train), ("dev", dev)):
        for word, phones in pairs:
            text_ok = preprocessor.text_tokenizer.decode(
                preprocessor.text_tokenizer(word, "g2p"), remove_special_tokens=True
            ) == list(word)
            phones_ok = (
                preprocessor.phoneme_tokenizer.decode(
                    preprocessor.phoneme_tokenizer(phones, "g2p"),
                    remove_special_tokens=True,
                )
                == phones
            )
            if not phones_ok or (split == "train" and not text_ok):
                raise ValueError(
                    f"{split} tokenizer would lose an input or target token"
                )
            if split == "dev" and text_ok:
                encoded_dev.append(("g2p", word, phones))
    if not encoded_dev:
        raise ValueError("No fully encodable dev examples for upstream loader")
    config_path = directory / "neural-config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    started = time.perf_counter()
    preprocess(
        str(config_path),
        train_data=[("g2p", word, phones) for word, phones in train],
        val_data=encoded_dev,
        deduplicate_train_data=False,
    )
    if len(unpickle_binary(directory / "data/train_dataset.pkl")) != len(train):
        raise ValueError("Upstream preprocessing changed training population")
    model = create_model(ModelType.AUTOREG_TRANSFORMER, config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainer = Trainer(
        directory / "checkpoints",
        torch.device(device),
        rank=0,
        use_ddp=False,
        loss_type="cross_entropy",
    )
    history: list[dict[str, Any]] = []
    selection = _DevSelection(settings.warmup_steps, settings.stop_patience)
    stop_reason = "maximum epochs reached; convergence not established"
    checkpoint = {"config": config, "preprocessor": preprocessor}
    dev_words = list(dict.fromkeys(word for word, _ in dev))

    def epoch_complete(epoch, trained_model, state, optimizer, scheduler):
        nonlocal stop_reason
        actual_device = str(next(trained_model.parameters()).device)
        if next(trained_model.parameters()).device.type != device:
            raise ValueError("Actual training device differs from requested device")
        dev_started = time.perf_counter()
        lookup, diagnostic, _ = _predict(
            trained_model, preprocessor, dev_words, settings.batch_size
        )
        metrics = _metrics(lookup, dev)
        key = (metrics["per_variant_pct"], metrics["wer_relaxed_pct"])
        improved, should_stop = selection.observe(*key, state["optimizer_updates"])
        if improved:
            state["selected_epoch"] = epoch
            trainer._save_model(
                trained_model,
                None,
                state,
                directory / "checkpoints/best_model_no_optim.pt",
            )
        scheduler.step(key[0])
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record = {
            "epoch": epoch,
            "optimizer_updates": state["optimizer_updates"],
            "entries_visited": state["entries_visited"],
            "metrics": metrics,
            **diagnostic,
            "selected": improved,
            "nonimproving_after_warmup": selection.nonimproving,
            "actual_training_device": actual_device,
            "dev_seconds": time.perf_counter() - dev_started,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss
            if sys.platform == "darwin"
            else peak_rss * 1024,
        }
        if device == "mps":
            record["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
            record["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        history.append(record)
        (directory / "neural-history.json").write_text(
            json.dumps(history, allow_nan=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(record, allow_nan=False), flush=True)
        trained_model.train()
        if job["profile_only"]:
            stop_reason = "one-epoch compatibility/resource profile; no test decoding"
            return True
        if should_stop:
            stop_reason = "dev early stopping after warmup"
            return True
        return False

    try:
        trainer.train(
            model,
            checkpoint,
            store_phoneme_dict_in_model=False,
            epoch_callback=epoch_complete,
        )
    except ValueError:
        failure = {
            name: checkpoint.get(name, 0)
            for name in ("optimizer_updates", "entries_visited", "nonfinite_batches")
        }
        (directory / "neural-failed-accounting.json").write_text(
            json.dumps(failure), encoding="utf-8"
        )
        raise
    finally:
        trainer.writer.close()
    training_seconds = time.perf_counter() - started
    export_started = time.perf_counter()
    model, saved = load_checkpoint(
        str(directory / "checkpoints/best_model_no_optim.pt"), device=device
    )
    if "phoneme_dict" in saved:
        raise ValueError("Dictionary payload must be disabled")
    inference = {
        key: value
        for key, value in saved.items()
        if key not in ("optimizer", "phoneme_dict")
    }
    artifact = directory / "neural-model.pt"
    torch.save(inference, artifact)
    reloaded, reloaded_state = load_checkpoint(str(artifact), device=device)
    before, _, _ = _predict(
        model,
        saved["preprocessor"],
        dev_words[: settings.batch_size],
        settings.batch_size,
    )
    after, _, _ = _predict(
        reloaded,
        reloaded_state["preprocessor"],
        dev_words[: settings.batch_size],
        settings.batch_size,
    )
    for word in dev_words[: settings.batch_size]:
        if set(word) <= set(text_symbols) and before(word) != after(word):
            raise ValueError("Saved neural token prediction differs after reload")
    export_seconds = time.perf_counter() - export_started
    result = {
        "settings": {
            **settings.to_dict(),
            "model_type": "author autoregressive Transformer",
            "encoder_layers": settings.layers,
            "decoder_layers": settings.layers,
            "device": device,
            "deterministic_algorithms_requested": device == "cpu",
            "bit_deterministic_across_platforms": False,
            "sampler_seed": 42,
            "drop_last": False,
            "max_decode_steps": 100,
            "selection": "full shared dev PER/WER/earliest",
            "plateau_factor": 0.5,
        },
        "training": {
            "retained_entries": len(train),
            "optimizer_updates": history[-1]["optimizer_updates"],
            "entries_visited": history[-1]["entries_visited"],
            "nonfinite_batches": checkpoint["nonfinite_batches"],
            "dropped_entries": 0,
            "encoded_dev_loss_entries": len(encoded_dev),
            "dev_entries": len(dev),
            "completed_epochs": len(history),
            "selected_epoch": saved["selected_epoch"],
            "stop_reason": stop_reason,
            "parameter_count": parameter_count,
            "dictionary_entries": 0,
        },
        "training_seconds": training_seconds,
        "export_seconds": export_seconds,
        "provenance": {
            **identity,
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "history": history,
        "model_sha256": sha256_file(artifact),
        "model_bytes": artifact.stat().st_size,
        "profile_only": job["profile_only"],
    }
    if not job["profile_only"]:
        test = json.loads((directory / "neural-test.json").read_text(encoding="utf-8"))
        test_started = time.perf_counter()
        lookup, diagnostic, predictions = _predict(
            reloaded,
            reloaded_state["preprocessor"],
            list(dict.fromkeys(word for word, _ in test)),
            settings.batch_size,
        )
        result["prediction_seconds"] = time.perf_counter() - test_started
        result["metrics"] = _metrics(lookup, test)
        result["test_diagnostics"] = diagnostic
        (directory / "neural-predictions.json").write_text(
            json.dumps(predictions, ensure_ascii=False), encoding="utf-8"
        )
    (directory / "neural-result.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    run(Path(sys.argv[1]))
