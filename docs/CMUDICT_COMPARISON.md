# CMUdict comparison workflows

The [current measured comparisons](G2P_BENCHMARKS.md) evaluate Phonebox CART,
Phonebox multigram and independent implementations on shared prepared data,
including CMUdict with stress preserved and removed. See the
[protocol and reproduction commands](REPRODUCIBLE_BENCHMARKS.md) for source
pins, training settings, development selection and error accounting.

The earlier [locale-aware two-way snapshot](archive/cmudict-489097f/README.md)
is retained as a historical record of revision `489097f`. Its multigram
scoring predates the current corrections; its conclusions are not current
model-selection advice.

## Different preprocessing, different experiments

The shared-data comparison uses identity grapheme/phone processing after the
declared normalization and stress mapping, and reserves separate train, dev
and test groups. The locale-aware workflow uses `Vectorizer(locale="en_US")`,
configured grapheme and phone joins, and a train/test split without a dev set.
For example, it joins `ch` into one grapheme unit and `K S` into one phone
unit. These joins change alignment admission and the units counted by phone
error rate. Scores from the two workflows are not controlled before/after
measurements of an implementation change.

## Run a separate locale-aware experiment

The library entry point is `phonebox.eval.cmudict_compare.run_cmudict_comparison`.
The CLI downloads and verifies the pinned public CMUdict source when no
`--lexicon` is supplied. Markdown defaults to the JSON's directory and basename
with a `.md` suffix; `--markdown` overrides it. Choose explicit output files for
a new experiment:

```console
phonebox compare cmudict --em-iterations 100 \
  --refresh .cache/benchmarks/locale-cmudict.json \
  --markdown .cache/benchmarks/locale-cmudict.md
phonebox compare cmudict --check \
  .cache/benchmarks/locale-cmudict.json .cache/benchmarks/locale-cmudict.md
```

The iteration argument is a maximum, not evidence of convergence. This workflow
records its own source and settings; it does not inherit the shared-data
benchmark's development-selection protocol. `--check` verifies that Markdown
matches JSON without rerunning training. It cannot establish that an older
snapshot measures the current implementation.
