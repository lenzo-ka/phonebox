"""Higher-order configuration reaches the public API, CLI and saved decoder."""

import subprocess
import sys

import pytest

from phonebox import train_multigram
from phonebox.core.multigram_g2p import MultigramG2P
from phonebox.core.multigram_lm import SUPPORTED_LM_ORDERS


def _lexicon(tmp_path):
    path = tmp_path / "tiny.dict"
    path.write_text("cat K AE T\ncap K AE P\n", encoding="utf-8")
    return path


def test_public_training_api_exports_requested_order_and_beam(tmp_path):
    result = train_multigram(
        _lexicon(tmp_path),
        locale="en",
        phoneset="cmu",
        output=tmp_path / "api.g2p",
        lm_order=8,
        decode_beam=4,
        em_iterations=2,
        max_letter_span=1,
        max_phone_span=1,
        no_config_joins=True,
    )
    assert result.model.lm.order == 8
    loaded = MultigramG2P.load(tmp_path / "api.g2p")
    assert loaded.lm.order == 8 and loaded.decode_beam == 4
    assert loaded.pronounce("cat") == ["K", "AE", "T"]


def test_cli_order_range_help_and_actual_export(tmp_path):
    command = [sys.executable, "-m", "phonebox", "train-multigram"]
    help_result = subprocess.run([*command, "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "{" + ",".join(map(str, SUPPORTED_LM_ORDERS)) + "}" in help_result.stdout
    assert "0 is exact" in help_result.stdout and "approximate" in help_result.stdout
    path = tmp_path / "cli.g2p"
    arguments = [
        "--locale",
        "en",
        "--phoneset",
        "cmu",
        "--lexicon",
        str(_lexicon(tmp_path)),
        "--output",
        str(path),
        "--lm-order",
        "8",
        "--decode-beam",
        "4",
        "--em-iterations",
        "2",
        "--max-letter-span",
        "1",
        "--max-phone-span",
        "1",
        "--no-config-joins",
    ]
    result = subprocess.run([*command, *arguments], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    loaded = MultigramG2P.load(path)
    assert loaded.lm.order == 8 and loaded.decode_beam == 4
    assert loaded.pronounce("cat") == ["K", "AE", "T"]


@pytest.mark.parametrize("option,value", [("--lm-order", "9"), ("--decode-beam", "-1")])
def test_cli_invalid_settings_preserve_existing_sidecars(tmp_path, option, value):
    path = tmp_path / "protected.g2p"
    sidecar, _ = MultigramG2P.export_paths(path)
    sidecar.write_text("preserve")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "phonebox",
            "train-multigram",
            "--locale",
            "en",
            "--lexicon",
            str(_lexicon(tmp_path)),
            "--output",
            str(path),
            option,
            value,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert sidecar.read_text() == "preserve"


@pytest.mark.parametrize("mode", ["locale", "sweep"])
def test_comparison_cli_exposes_the_shared_order_range_and_beam(mode):
    result = subprocess.run(
        [sys.executable, "-m", "phonebox", "compare", mode, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "{" + ",".join(map(str, SUPPORTED_LM_ORDERS)) + "}" in result.stdout
    assert "--decode-beam" in result.stdout and "approximate" in result.stdout


def test_sweep_validates_order_and_beam_before_reading_lexicons():
    from phonebox.eval.g2p_sweep import run_g2p_sweep

    with pytest.raises(ValueError, match="order must be an integer from 1 to 8"):
        run_g2p_sweep({}, locales=[], letter_spans=[2], lm_orders=[9])
    with pytest.raises(ValueError, match="decode beam must be a nonnegative integer"):
        run_g2p_sweep({}, locales=[], letter_spans=[2], lm_orders=[8], decode_beam=-1)


def test_sweep_forwards_order_and_beam_into_actual_training(tmp_path, monkeypatch):
    from phonebox.eval import g2p_sweep

    lexicon = _lexicon(tmp_path)
    observed = []
    train = g2p_sweep.train_multigram

    def record_training(*args, **kwargs):
        result = train(*args, **kwargs)
        observed.append((result.model.lm.order, result.model.decode_beam))
        return result

    monkeypatch.setattr(g2p_sweep, "train_multigram", record_training)
    rows = g2p_sweep.run_g2p_sweep(
        {"en": lexicon},
        locales=["en"],
        letter_spans=[1],
        lm_orders=[8],
        decode_beam=4,
        em_iterations=2,
        max_test=1,
    )
    assert observed == [(8, 4)]
    report = g2p_sweep.format_g2p_sweep(
        rows, letter_spans=[1], lm_orders=[8], decode_beam=4
    )
    assert "Decode beam: 4" in report
