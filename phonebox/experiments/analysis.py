"""Error analysis for G2P experiments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable


def collect_phone_substitutions(
    predict: Callable[[str], list[str]],
    test_set: Iterable[tuple[str, list[str]]],
    *,
    max_examples: int = 5,
) -> tuple[Counter[tuple[str, str]], list[tuple[str, list[str], list[str]]]]:
    """Count (gold, pred) phone substitutions and sample wrong words."""
    subs: Counter[tuple[str, str]] = Counter()
    wrong: list[tuple[str, list[str], list[str]]] = []
    for word, expected in test_set:
        pred = predict(word)
        if pred == expected:
            continue
        wrong.append((word, expected, pred))
        for i in range(min(len(expected), len(pred))):
            if expected[i] != pred[i]:
                subs[(expected[i], pred[i])] += 1
        if len(expected) != len(pred):
            subs[("__len__", str(len(expected)))] += 1
    wrong.sort(key=lambda t: (-sum(1 for a, b in zip(t[1], t[2]) if a != b), t[0]))
    return subs, wrong[:max_examples]


def format_substitution_table(
    subs: Counter[tuple[str, str]],
    *,
    top_n: int = 25,
    title: str = "",
) -> str:
    lines = []
    if title:
        lines.append(f"### {title}")
        lines.append("")
    lines.append("| gold | pred | count |")
    lines.append("|------|------|-------|")
    for (gold, pred), count in subs.most_common(top_n):
        if gold == "__len__":
            lines.append(f"| (length) | {pred} phones | {count} |")
        else:
            lines.append(f"| `{gold}` | `{pred}` | {count} |")
    if not subs:
        lines.append("| — | — | 0 |")
    lines.append("")
    return "\n".join(lines)


def audit_normalize_delta(
    train_raw: list[tuple[str, list[str]]],
    locale: str,
    policy: str,
) -> dict[str, int]:
    """Count how many train entries change under a normalize policy."""
    from phonebox.experiments.normalize import apply_train_normalize

    changed = 0
    phone_changes = 0
    for word, phones in train_raw:
        new_phones = apply_train_normalize(locale, policy, word, phones)
        if new_phones != phones:
            changed += 1
            phone_changes += sum(1 for a, b in zip(phones, new_phones) if a != b)
            phone_changes += abs(len(phones) - len(new_phones))
    return {
        "train_entries": len(train_raw),
        "entries_changed": changed,
        "phone_token_changes": phone_changes,
    }
