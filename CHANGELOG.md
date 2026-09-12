# Changelog

Phonebox is an alpha library. Each `0.X.0` release may break APIs and workflows;
patch releases within a minor line are intended to remain compatible.

## 0.2.0 — Unreleased

Changes since `v0.1.0`:

### Training and inference

- Unify dictionary training behind `train_g2p`, `train_g2p_from_config`,
  `G2P.train`, and `phonebox train`, with structured training results and shared
  CLI/config defaults. The primary workflow preserves stress, uses IPA-tagged
  phones, trains serially with the native CART trainer, and prunes with a 5%
  validation split by default. CMU dictionaries require `phoneset="cmu"`.
- Save training-time preprocessing with CART and multigram models, including
  locale policy, transliteration rules, joins, scalar flags, and spelling
  rewrites. Current save/load paths preserve those choices.
- Replay portable preprocessing in standard-library-only decision-tree bundles;
  reject unsupported preprocessing and sidecar-only multigram inputs clearly.
  Bundling remains a decision-tree feature.
- Preserve Spanish Unicode equivalence and acute weak-vowel hiatus cues; fold
  Italian accented a/i/u in the stressless spelling policy.

### Locale and dictionary APIs

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

Initial tagged alpha baseline (`v0.1.0`): decision-tree G2P, EM alignment,
multigram models, dictionary/model tooling, and standalone tree inference.
The project license is BSD 2-Clause; 0.2.0 also packages Unicode-licensed data.
