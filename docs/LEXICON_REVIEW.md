# Review and reorder a pronunciation lexicon

Review measures compatibility with a trained CART model. Low or zero support is
an invitation to inspect a variant, not proof that it is erroneous, an acronym,
or a foreign word. Multigram candidate scoring is not supported by this workflow.

```bash
phonebox train --locale en --phoneset cmu --width 1 --no-prune \
  --lexicon words.dict -o model.g2p.gz
phonebox dict review words.dict -m model.g2p.gz -o review.tsv
phonebox dict review words.dict -m model.g2p.gz --format dict -o ranked.dict
phonebox dict review words.dict -m model.g2p.gz --format dict \
  --no-number-senses -o ranked-bare.dict
```

CMUdict is a documented public input: fetch it with `phonebox dict fetch cmudict`
and retain its downloaded license. Width1 is an explicit coarse spelling baseline;
a rich-context model trained on the reviewed words may memorize all variants and
assign them high support. Report model configuration and input population when
interpreting results. Deterministic or coarse emissions can also reject familiar
valid spellings, so zero support is evidence for manual review only. Neither
baseline scores nor in-sample scores are calibrated
probabilities of pronunciation correctness. This command does not train a new
model or automatically modify the input lexicon.

For a concrete CMUdict workflow, use the same saved stress setting for training
and review; retain `data/cmudict/LICENSE` alongside the downloaded dictionary:

```bash
phonebox dict fetch cmudict --data-dir data
phonebox train --locale en_US --phoneset cmu --width 1 --no-prune \
  --remove-stress --lexicon data/cmudict/cmudict.dict -o cmudict.g2p.gz
phonebox dict review data/cmudict/cmudict.dict -m cmudict.g2p.gz -o cmudict-review.tsv
phonebox dict review data/cmudict/cmudict.dict -m cmudict.g2p.gz \
  --format dict -o cmudict-ranked.dict
```

For example, saved stress removal merges source entries `be B IY1` and
`be(2) B IY0` into effective `be B IY`, retaining both origins in JSON.
A model can also rank source `record(2) R EH1 K ER0 D` first and emit
`record R EH K ER D`. That order expresses the particular model's compatibility;
it does not establish a preferred meaning or pronunciation. Preserve the input
and JSON provenance when interpreting or adopting a reordered dictionary.

## Data and ordering

Default output is TSV, worst first, with columns:

`score`, `entry`, `phones`, `word`, `rank`, `log_probability`, `status`, `source_lines`.

The first three fields support numeric review and copying dictionary entries:

```bash
phonebox dict review words.dict -m model.g2p.gz --no-header \
  | LC_ALL=C sort -s -g -k1,1 > worst.tsv
cut -f2,3 worst.tsv > selected.dict
```

`entry` defaults to bare/(2)/(3) labels assigned by descending within-word likelihood;
`phones` are the effective normalized public tokens used for scoring. JSON and
JSONL preserve source labels, variant suffixes, physical lines and original phones
in `origins`. Each review reads one input lexicon; origin line numbers refer
to that input, including skipped comment lines. A suffix records input numbering; it is not an inferred semantic
sense. Variants that become equivalent after an optional phone mapping and saved
model cooking merge, retaining every origin. Stress follows the saved model;
review does not silently strip stress. `--phone-map mapping.json` uses the same
literal mapping grammar as dictionary processing and runs before model cooking.

`--order variants` groups words by first source occurrence and sorts each word's
variants best first. Ties retain source order. `--threshold` selects scores strictly
below its value, and `--limit` caps the selected records. Ranks are assigned before
filtering: filtered TSV/JSON can contain rank gaps. Dense bare/(2)/(3) numbering
is guaranteed for the unfiltered population with numbering enabled only. Dictionary format implicitly
uses variants order and rejects threshold/limit or an explicit incompatible order;
it writes exactly two fields, without metadata mistaken for phone tokens.
`--no-number-senses` repeats the bare orthographic spelling for every variant;
`--number-senses` explicitly enables the default CMUdict-style suffixes. Both
settings apply to TSV, dictionary, JSON and JSONL labels without changing rank,
origins or order. Use `--order variants` (implicit for dictionary output) to keep
each spelling's most compatible pronunciation first. Numbering refers to ranked
pronunciation variants, not inferred semantic senses.
An empty effective pronunciation can still be reviewed in TSV/JSON, but
dictionary export rejects it rather than emitting an unreadable empty entry.

`--format json` writes one result envelope; `jsonl` writes one numeric record per
line. JSON numbers remain numbers, and unsupported log probability is null.
TSV uses an empty log field for unsupported sequences. `status` distinguishes
unsupported sequences from supported probabilities that underflow to zero.
Default geometric compatibility is `exp(log_probability / positions)`; product
is raw full-sequence probability. Ranking uses retained log values, so underflow
does not create false likelihood ties. If sorting product output yourself, sort
supported rows by the log column and keep unsupported rows first; sorting only
the displayed zero probabilities cannot reproduce that distinction. Floats are
written without fixed-decimal rounding that silently erases small values.

## Python API

```python
from phonebox import train_g2p
from phonebox.pronunciation_analysis import review_lexicon, format_lexicon_review

# Use a caller-provided dictionary; no fixed project layout is required.
trained = train_g2p("words.dict", locale="en", phoneset="cmu",
                    width=1, prune=False)
with open("words.dict", encoding="utf-8") as lines:
    result = review_lexicon(trained.model, lines, order="variants")
for line in format_lexicon_review(result, format="dict"):
    print(line)
# result.to_dict() retains numeric scores and source provenance.
# Set number_senses=False on the formatter or result/record.to_dict()
# to repeat bare spellings while retaining rank and origins.
```

`review_lexicon_file` is the file convenience wrapper. APIs return structured
records and do not print or write files. The CLI rejects output aliases of the
lexicon, mapping, or model artifacts before opening output. Bad inputs/options
produce status2 with a diagnostic. Use `phonebox dict review --help` for options.
Candidate JSONL `score-prons` and heuristic `find-suspicious` remain separate
adapters; their heuristics are optional human-review aids, not this workflow's
correctness classifier. See [scoring](PERFORMANCE_AND_SCORING.md) for exact model
semantics and [workflows](WORKFLOWS.md) for training and inference APIs.
