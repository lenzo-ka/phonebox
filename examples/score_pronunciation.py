#!/usr/bin/env python
"""Score complete ordered sequences with a tiny CMU-tagged CART model."""

from pathlib import Path
from tempfile import TemporaryDirectory

from phonebox import train_g2p


def main():
    with TemporaryDirectory() as directory:
        lexicon = Path(directory) / "words.dict"
        lexicon.write_text("read R IY1 D\nread(2) R EH1 D\n", encoding="utf-8")
        result = train_g2p(
            lexicon,
            locale="en_US",
            phoneset="cmu",
            remove_stress=True,
            width=1,
            prune=False,
        )
        for phones in (["R", "IY1", "D"], ["R", "EH1", "D"], ["D", "IY1", "R"]):
            details = result.model.score_pronunciation_details("read", phones)
            print(f"{' '.join(phones)}: {details.to_dict()}")
        print("Geometric model compatibility, not probability of correctness.")
        print(
            "Zero support requires manual review; valid pronunciations can be unsupported."
        )


if __name__ == "__main__":
    main()
