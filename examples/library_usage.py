#!/usr/bin/env python
"""Self-contained Dictionary processing and shared training API demonstration."""

from pathlib import Path
from tempfile import TemporaryDirectory

from phonebox import G2P, Dictionary


def main():
    with TemporaryDirectory() as directory:
        source = Path(directory) / "words.dict"
        source.write_text(
            "cat K AE1 T\ncat(2) K AE2 T\nbat B AE1 T\n", encoding="utf-8"
        )
        dictionary = Dictionary(source, locale="en_US")
        processed = dictionary.process(
            phoneset="cmu",
            remove_stress=True,
            output=Path(directory) / "processed.dict",
        )
        model = Path(directory) / "model.g2p.gz"
        trained = processed.train_g2p_model(
            locale="en_US",
            phoneset="cmu",
            prune=False,
            output=model,
        )
        g2p = G2P(model=model)
        assert g2p("cat") == trained.pronounce("cat")
        for word, phones in g2p.pronounce_batch(["cat", "bat"]):
            print(f"{word}: {' '.join(phones)}")
        print("Processing merged stress-equivalent variants; all files were temporary.")


if __name__ == "__main__":
    main()
