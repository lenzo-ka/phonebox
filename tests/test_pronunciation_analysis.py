from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from phonebox import G2P
from phonebox.pronunciation_analysis import (
    find_low_scores,
    find_score_gaps,
    find_zeros,
    score_entries,
    score_pronunciations,
    strip_stress,
    triage_entries,
)


class FakeScorer:
    def __init__(self):
        self.calls = []

    def score_pronunciation(self, word, phones, method="geometric"):
        self.calls.append((word, phones, method))
        return {"AH0": 0.2, "EH1": 0.8}[phones[0]]


def test_score_api_preserves_phones_sorts_and_preserves_entry_fields():
    scorer = FakeScorer()
    scored = score_pronunciations(scorer, "read", ["AH0", "EH1"], method="harmonic")
    assert scored == {"EH1": 0.8, "AH0": 0.2}
    assert strip_stress("DH AH0") == ["DH", "AH"]
    assert scorer.calls == [
        ("read", ["AH0"], "harmonic"),
        ("read", ["EH1"], "harmonic"),
    ]
    assert list(
        score_entries(scorer, [{"word": "read", "freq": 3, "prons": ["AH0"]}])
    ) == [{"word": "read", "freq": 3, "prons": {"AH0": 0.2}}]


def test_structured_suspicion_results_and_english_triage_heuristics():
    entries = [
        {"word": "alpha", "prons": {"AE L": "0", "AA L": "0.2"}},
        {"word": "bravo", "prons": {"B R": "0.005", "B R AA": "0.2"}},
    ]
    zero = list(find_zeros(entries))
    assert zero[0].word == "alpha"
    assert zero[0].zero_pronunciations == ("AE L",)
    assert [result.word for result in find_low_scores(entries, 0.3)] == [
        "alpha",
        "bravo",
    ]
    gap = list(find_score_gaps(entries))
    assert [(result.word, result.ratio) for result in gap] == [
        ("alpha", float("inf")),
        ("bravo", 40.0),
    ]
    triage = triage_entries(
        [
            {"word": "the", "prons": {"DH AH": 0.01}},
            {"word": "macarthur", "prons": {"M AE K": 0.01}},
            {"word": "normal", "prons": {"N AO R": 0.8}},
        ]
    )
    assert triage["FUNCTION"][0].reason == "function word"
    assert triage["FOREIGN"][0].reason == "Irish/Scottish"
    assert triage["OK"][0].pronunciation == "N AO R"


def test_score_api_with_real_small_trained_model(tmp_path):
    dictionary = tmp_path / "train.dict"
    dictionary.write_text(
        "cat K AE T\ndog D AO G\nhat HH AE T\nbat B AE T\n"
        "rat R AE T\nmat M AE T\nsat S AE T\nfat F AE T\n",
        encoding="utf-8",
    )
    g2p = G2P.train(dictionary, locale="en_US", verbose=False)
    assert g2p.has_distributions
    assert score_pronunciations(g2p, "cat", ["K AE T", "K AH T"]) == {
        "K AE T": 1.0,
        "K AH T": 0.0,
    }


def test_score_api_preserves_stress_for_model(tmp_path):
    dictionary = tmp_path / "stress.dict"
    dictionary.write_text("cat K AE1 T\n", encoding="utf-8")
    g2p = G2P.train(dictionary, locale="en_US", remove_stress=False, verbose=False)
    assert score_pronunciations(g2p, "cat", ["K AE1 T"]) == {"K AE1 T": 1.0}


@pytest.mark.parametrize("command", ["score-prons", "find-suspicious"])
def test_cli_commands_expose_help(command):
    root = Path(__file__).parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "phonebox.cli.main", command, "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout


def test_find_suspicious_cli_preserves_zero_report(tmp_path):
    source = tmp_path / "scores.jsonl"
    source.write_text(
        json.dumps({"word": "abc", "prons": {"AE B": "0", "AH B": "0.2"}}) + "\n",
        encoding="utf-8",
    )
    root = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "phonebox.cli.main",
            "find-suspicious",
            str(source),
            "--zeros",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "abc:" in completed.stdout
    assert "AE B ← 0!" in completed.stdout


def test_find_suspicious_cli_requires_exactly_one_mode(tmp_path):
    source = tmp_path / "scores.jsonl"
    source.write_text('{"word":"abc","prons":{"AE":0}}\n', encoding="utf-8")
    base = [sys.executable, "-m", "phonebox.cli.main", "find-suspicious", str(source)]
    assert subprocess.run(base, capture_output=True, check=False).returncode == 2
    assert (
        subprocess.run(
            base + ["--zeros", "--triage"], capture_output=True, check=False
        ).returncode
        == 2
    )
    zero = subprocess.run(
        base + ["--low", "0"], text=True, capture_output=True, check=False
    )
    assert zero.returncode == 0
    assert "max score < 0.0" in zero.stdout
