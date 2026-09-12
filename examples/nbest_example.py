#!/usr/bin/env python
"""Train a tiny distribution-enabled model and inspect N-best alternatives."""

from pathlib import Path
from tempfile import TemporaryDirectory

from phonebox import G2P, train_g2p


def main():
    with TemporaryDirectory() as directory:
        lexicon = Path(directory) / "words.dict"
        lexicon.write_text("read R IY1 D\nread(2) R EH1 D\n", encoding="utf-8")
        model = Path(directory) / "model.g2p.gz"
        train_g2p(
            lexicon,
            locale="en_US",
            phoneset="cmu",
            remove_stress=True,
            width=1,
            prune=False,
            store_distributions=True,
            output=model,
        )
        # Inspect model alternatives instead of the embedded exception lookup.
        g2p = G2P(model=model, use_dict_fallback=False)
        phones, confidences = g2p.pronounce_with_confidence("read")
        print("Best:", list(zip(phones, confidences, strict=True)))
        for rank, (candidate, score) in enumerate(g2p.pronounce_nbest("read", n=5), 1):
            print(f"{rank}: {' '.join(candidate)}; search score={score}")
        print(
            "Leaf confidence and N-best search scores are not calibrated correctness probabilities."
        )
        print(
            "Alternatives are model emissions, not a guarantee of valid pronunciations."
        )


if __name__ == "__main__":
    main()
