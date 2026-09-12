# Changelog

Phonebox is an alpha library. Each `0.X.0` release may break APIs and workflows;
patch releases within a minor line are intended to remain compatible.

## 0.2.0 — Unreleased

Changes since `v0.1.0`:

### Training and inference

- Score complete ordered CART pronunciation sequences, summing silent/joined
  emission paths with stable log probabilities and saved preprocessing. Share
  structured score details and numeric candidate JSON; geometric/product replace
  unordered phone aggregation. Restore native CART embedded model metadata.

- Unify dictionary training behind `train_g2p`, `train_g2p_from_config`,
  `G2P.train`, and `phonebox train`, with structured training results and shared
  CLI/config defaults. The primary workflow preserves stress, uses IPA-tagged
  phones, trains serially with the native CART trainer, and prunes with a 5%
  validation split by default. CMU dictionaries require `phoneset="cmu"`.
- Restore the dictionary content hash on CART model load and re-export using
  the same nested-metadata precedence as other model configuration.
- Save training-time preprocessing with CART and multigram models, including
  locale policy, transliteration rules, joins, scalar flags, and spelling
  rewrites. Current save/load paths preserve those choices.
- Replay portable preprocessing in standard-library-only decision-tree bundles;
  reject unsupported preprocessing and sidecar-only multigram inputs clearly.
  Bundling remains a decision-tree feature.
- Preserve Spanish Unicode equivalence and acute weak-vowel hiatus cues; fold
  Italian accented a/i/u in the stressless spelling policy.
- Share context width across alignment, vectorization, and prepared training;
  reject invalid dimensions and empty admitted training data with clear errors.
  Validate training config keys and protect input files from output aliases.
- Add `train_multigram` / `MultigramTrainingResult` and `discover_joins` /
  `JoinDiscoveryResult` as reusable, quiet workflow APIs with optional exports;
  their CLI commands delegate to the same operations.
- Validate n-best counts consistently in the library and CLI, including zero
  and negative counts; keep multigram inference's unsupported modes explicit.

### Licensing

- Change the code license from MIT at `v0.1.0` to BSD 2-Clause in post-tag
  commit `f807a46`. Packaged ICU/CLDR exemplar data additionally carries
  Unicode-3.0 with its complete notice.

- Correct CMUdict manifest licensing metadata to reference its upstream license.

### Locale and dictionary APIs

- Expose structured lexicon/inventory validation through `phonebox.validation`;
  retain all raw NFC-equivalent phone forms and validate CLI phoneset inputs.
- Add case-insensitive, hyphen/underscore locale canonicalization and shared
  resource resolution while preserving language, region, and script identities.
  Orthographic equivalence can share spelling policy, never trained models or
  phonetic supplement defaults.
- Package standard and auxiliary ICU exemplar inventories for 906 locales,
  with ICU-free readers/resolution, pinned development generation, source
  provenance, and the Unicode License v3 notice. Exemplars describe orthography,
  not phoneme inventories.
- Share dictionary-line parsing across training and analysis: arbitrary
  whitespace, numeric pronunciation variants, and inline comments are supported.
- Add general phone mapping/transforms to `Dictionary.process` and a thin
  `phonebox dict process` command. Deduplicate after mapping and optional
  phoneset-specific stress removal; emit dense pronunciation variant numbers.
  Reject aliased input/output paths before writing.

### Evaluation and tooling

- Add a current library/API/CLI workflow guide and correct removed-command,
  prepared-vector loading, held-out test, scoring, and context-demo examples.
- Use the native trainer by default in dictionary accuracy evaluation and base
  examples. Expose the optional sklearn backend through `phonebox[sklearn]` and
  CLI selection with clear missing-dependency errors. Remove the obsolete `icu`
  extra that installed PyICU alongside the mandatory icukit backend.
- Reuse the shared dictionary parser in accuracy loading, excluding numeric
  alternates while keeping inline comments out of phones.

- Replace operational repository scripts with reusable package APIs and thin
  CLI commands for evaluation, experiments, sweeps, units, pronunciation scoring,
  suspicious-entry analysis, and development exemplar generation.
- Make CLI `main(argv)` return a status without mutating process arguments;
  expose library pronunciation and normalization operations used by the CLI.
- Add a reproducible public CMUdict CART/multigram comparison with word-group
  splitting, stress controls, variant-aware accuracy and reference-normalized
  PER, model-specific training admission counts, full export footprints, and
  recorded source/data/dependency provenance. JSON renders the report and CI
  checks consistency; recorded metrics remain a historical snapshot.
- Replace machine-specific report paths and external-project environment names
  with portable artifact names and `PHONEBOX_LEXICON_DIR` / `PHONEBOX_MODEL_DIR`.

See [upgrade guidance](docs/RELEASING.md#upgrading-from-010) before migrating.

## 0.1.0

Initial public release (`v0.1.0`, classified Beta): decision-tree G2P, EM alignment,
multigram models, dictionary/model tooling, and standalone tree inference.
This tag used the MIT license. The later code-license change and
Unicode-licensed data addition are recorded under 0.2.0 above.
