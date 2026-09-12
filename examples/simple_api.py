#!/usr/bin/env python
"""Load a caller-provided CART model: python examples/simple_api.py MODEL."""

import argparse

from phonebox import G2P


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Trained CART artifact, e.g. model.g2p.gz")
    args = parser.parse_args()
    g2p = G2P(model=args.model)
    print("hello:", " ".join(g2p("hello")))
    for word, phones in g2p.pronounce_batch(["world", "python"]):
        print(f"{word}: {' '.join(phones)}")


if __name__ == "__main__":
    main()
