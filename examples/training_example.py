#!/usr/bin/env python
"""Train and reload a tiny CMU-tagged model; requires installed phonebox."""

from pathlib import Path
from tempfile import TemporaryDirectory

from phonebox import G2P, train_g2p


def main():
    with TemporaryDirectory() as directory:
        lexicon = Path(directory) / "words.dict"
        lexicon.write_text("cat K AE1 T\nbat B AE1 T\nhat HH AE1 T\n", encoding="utf-8")
        model = Path(directory) / "model.g2p.gz"
        result = train_g2p(
            lexicon,
            locale="en_US",
            phoneset="cmu",
            remove_stress=True,
            prune=False,
            output=model,
        )
        loaded = G2P(model=model)
        for word in ("cat", "bat", "hat"):
            assert loaded(word) == result.model.pronounce(word)
            print(f"{word}: {' '.join(loaded(word))}")
        print("Tiny in-sample demonstration; not an accuracy evaluation.")


if __name__ == "__main__":
    main()
