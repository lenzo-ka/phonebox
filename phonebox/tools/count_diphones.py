#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

from phonebox.constants import FILE_ENCODING
from phonebox.utils.io import open_output


def consume(infile, accum):
    for line in infile:
        m = re.match(r"toomany\(.*?\):\s+([^\t]+)\t::\t([^\t]+)", line.rstrip())
        if not m:
            continue

        letters = m.group(1).split()
        for i in range(len(letters) - 1):
            accum[0][f"{letters[i]} {letters[i + 1]}"] += 1

        phones = m.group(2).split()
        for i in range(len(phones) - 1):
            accum[1][f"{phones[i]} {phones[i + 1]}"] += 1


def main():
    parser = argparse.ArgumentParser(description="Count diphones in logfile")
    parser.add_argument("files", nargs="*", type=str)
    parser.add_argument("-o", "--output", type=str, help="output to this file")

    args = parser.parse_args()

    accum: list[dict[str, int]] = [defaultdict(int), defaultdict(int)]

    if args.files:
        for path in args.files:
            with open(path, encoding=FILE_ENCODING) as infile:
                consume(infile, accum)
    else:
        consume(sys.stdin, accum)

    accum = [dict(sorted(x.items(), reverse=True, key=lambda x: x[1])) for x in accum]

    with open_output(args.output) as outfile:
        json.dump(accum, outfile, ensure_ascii=False, indent=None if args.output else 1)


if __name__ == "__main__":
    main()
