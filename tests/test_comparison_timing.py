"""Stage timing must exclude work performed in later stages."""

from types import SimpleNamespace

import pytest

from phonebox.core.multigram_g2p import MultigramG2P
from phonebox.eval import g2p_compare, g2p_sweep


def lexicon(tmp_path):
    path = tmp_path / "tiny.dict"
    path.write_text("".join(f"a{x} A B\n" for x in "bcdefghijk"))
    return path


def controlled_stages(monkeypatch, module):
    clock = [0.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    train = module.train_multigram
    evaluate = module.evaluate
    prepare = MultigramG2P.prepare_predictor

    def train_model(*args, **kwargs):
        result = train(*args, **kwargs)
        clock[0] += 3
        return result

    def prepare_model(model):
        result = prepare(model)
        clock[0] += 5
        return result

    def evaluate_model(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        clock[0] += 7
        return result

    monkeypatch.setattr(module, "train_multigram", train_model)
    monkeypatch.setattr(module, "evaluate", evaluate_model)
    monkeypatch.setattr(MultigramG2P, "prepare_predictor", prepare_model)
    return clock


@pytest.mark.parametrize("load_existing", [False, True])
def test_compare_training_loading_preparation_and_evaluation_are_disjoint(
    tmp_path, monkeypatch, load_existing, capsys
):
    clock = controlled_stages(monkeypatch, g2p_compare)

    def baseline(*args, **kwargs):
        clock[0] += 4 if load_existing else 2
        return SimpleNamespace(pronounce=lambda word: ["A", "B"])

    monkeypatch.setattr(g2p_compare, "train_baseline", baseline)
    monkeypatch.setattr(g2p_compare, "load_baseline", baseline)
    existing = tmp_path / "existing.g2p"
    existing.touch()
    summary = g2p_compare.run_compare(
        lexicon=lexicon(tmp_path),
        locale="en_US",
        em_iterations=2,
        baseline_model=existing if load_existing else None,
        test_fraction=0.3,
    )
    rows = summary["results"]
    assert isinstance(rows, list)
    cart, multigram = rows
    assert cart["train_s"] == (0 if load_existing else 2)
    assert cart["load_s"] == (4 if load_existing else 0)
    assert cart["prep_s"] == 0 and cart["eval_s"] == 7
    assert multigram["train_s"] == 3
    assert multigram["load_s"] == 0
    assert multigram["prep_s"] == 5
    assert multigram["eval_s"] == 7
    g2p_compare.print_results_table(
        [(row["model"], row["train_s"], row) for row in rows]
    )
    output = capsys.readouterr().out
    assert all(field in output for field in ("train_s", "load_s", "prep_s", "eval_s"))
    assert "metric calculation" in output


def test_sweep_separates_stage_times_and_renders_them(tmp_path, monkeypatch):
    controlled_stages(monkeypatch, g2p_sweep)
    rows = g2p_sweep.run_g2p_sweep(
        {"en_US": lexicon(tmp_path)},
        locales=["en_US"],
        letter_spans=[1],
        lm_orders=[2],
        em_iterations=2,
        max_test=2,
    )
    metrics = rows["en_US"][(1, 2)]
    assert metrics["train_s"] == 3
    assert metrics["load_s"] == 0
    assert metrics["prep_s"] == 5
    assert metrics["eval_s"] == 7
    markdown = g2p_sweep.format_g2p_sweep(rows, letter_spans=[1], lm_orders=[2])
    assert "| 1 | 2 | 3.000 | 5.000 | 7.000 |" in markdown
    assert "metric calculation" in markdown
