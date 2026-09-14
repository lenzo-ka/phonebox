# Changelog

Phonebox is an alpha library. Each `0.X.0` release may break APIs and workflows;
patch releases within a minor line are intended to remain compatible.

## 0.3.0 — Unreleased

### Optional neural benchmark toolchain

- Add isolated, from-scratch DeepPhonemizer autoregressive training with the
  author's substantial configuration, shared whole-dev checkpoint selection,
  explicit convergence accounting, and model-only held-out metrics.
- Export the credited reproducible source patch through API/CLI; verify source,
  patch, dependency versions and interpreter receipts without adding Torch to
  ordinary runtime, development or CI installations.
- Provide a one-epoch resource profile whose training worker receives no test
  references, atomic
  phone inference, complete partial-batch admission, unknown-input failures and
  truncation counts. These tools alone do not claim measured neural accuracy.

### Cartlet 0.6 integration

- Require `cartlet>=0.6.0,<0.7.0` for base and sklearn installations, keeping the
  dependency within its compatible alpha minor line.
- Delegate model schema and tree validation to Cartlet, accepting both strict
  `<` and inclusive `<=` numerical nodes in model load/re-export workflows.
  Keep the G2P-specific requirement for phone-string leaves or distributions.
- Normalize upstream invalid-artifact errors consistently across library and
  CLI callers, preserving existing output files when lexicon review fails.
- Test numerical threshold boundaries through JSON, compressed G2P, and binary
  model round trips, including the lightweight runner.

**Upgrading:** this line uses Cartlet model format 2. Older binary and
unversioned JSON/JSONL/pickle artifacts are not automatically migrated. Retrain
or explicitly convert with the original writer release; regenerate standalone
bundles together with their models. See [migration notes](docs/RELEASING.md#upgrading-from-020).

### Multigram scoring

- Keep finite positive add-k values scorable at floating-point extremes using
  log-domain arithmetic when needed, without changing ordinary-range scores,
  smoothing defaults, or the saved model format.
- Separate training, model loading, predictor preparation and evaluation timing
  in locale comparisons and multigram sweeps; historical `train_s` incorrectly
  included evaluation. Report monotonic durations and label metric work explicitly.
- Add reusable multigram predictor snapshots that prepare candidate indexes and
  unit IDs once for repeated inference. CLI pronunciation and comparison reuse
  snapshots; later source-model mutation cannot alter an existing predictor.
- Support joint-unit LM orders 1–8 through sparse count tables shared by the
  library, training API and CLI. Keep order 2/add-k smoothing/exact search as
  defaults; document explicit approximate beam controls and higher-order costs.
- Score complete multigram sequences with the unit LM alone. Alignment q still
  defines candidate units, but is no longer multiplied into the LM score again.
  Save the explicit `unit-lm-with-eos` objective; older q-plus-LM models require
  retraining. This is an intentional modeling change, not a smoothing parameter.
- Include the trained end-of-sequence transition when choosing a complete
  pronunciation, correcting the previous prefix-only terminal ranking.
- Normalize the unit LM over its declared inference units and end-of-sequence
  event, with the start marker used only as context. Preserve alignment units
  absent from Viterbi paths, including silent-phone units.
- Version new multigram models as 7 and their LM count format as 3. Reject older
  artifacts with a clear retraining instruction; CART formats are unchanged.
- Replace an unsupported historical smoothing claim with the actual scoring
  contract and use the model version in newly rendered comparisons.

### Experimental CART decomposition training

- Add opt-in posterior decomposition scattering: marginalize q-supported gold
  joint-unit alignments and train the existing contextual CART on fractional
  multi-phone/epsilon target weights. Reuse duplicate-pair posteriors and
  spelling contexts; record admission, ambiguity, support and convergence.
- Keep ordinary per-position inference and epsilon training as defaults.
  Disable static locale joins in the new public workflow, reject internal
  row splits, and build dictionary corrections from admitted source pairs.
- Add an opt-in CART-scored decomposition lattice. `prepare_decomposition_lattice`
  and `pronounce_lattice` select a supported complete joint-unit path from
  contextual CART target probabilities with the shared exact Viterbi decoder,
  adding no q factor, beam, probability floor or sequence LM. Posterior-trained
  models save their validated unit inventory; earlier artifacts can supply
  explicit units. Fractional leaf roundoff is normalized without changing path
  rankings.
- Record the development-only outcomes: posterior scattering loses to hard
  projection in all four conditions; lattice decoding lowers posterior-tree
  PER in all four; retaining leaf alternatives improves lattice PER further
  with unchanged point predictions. No corpus gain or test claim is made.

### Training diagnostics

- Expose defensive, JSON-serializable CART alignment iteration history through
  `EMAlign.alignment_history`, recording actual changed counts and ratios per
  run so callers can distinguish convergence from reaching an iteration cap.

### Packaging and notices

- Ship the DeepPhonemizer MIT notice as `LICENSE-DEEPPHONEMIZER` in the wheel
  and source distribution, declare it in the license expression, and record in
  the third-party notices that the packaged neural benchmark module embeds a
  modification of that code. Include the tracked third-party patches, license
  texts and dependency locks in the source distribution; the packaging test
  now checks all of them.
- Clarify that the runtime requires only `icukit>=0.1.2`; the ICU backend pin
  that generated the exemplar data belongs to the `dev` extra.

### Documentation and evaluation

- Add shared public benchmark data preparation, model adapters, CLI commands, and
  validated aggregate reporting for CART, n:m, Sequitur, and Phonetisaurus.
  Pin dataset and toolchain sources, preserve pronunciation groups, and record
  split hashes, licenses, training admission, and prediction failures.
- Record the measured four-system comparison: sixteen aggregate result rows in
  `docs/benchmarks/` and the generated `docs/G2P_BENCHMARKS.md`, pinned by an
  artifact test to exact regeneration, shared split digests, disabled lookup,
  full accounting and path-free provenance. The DeepPhonemizer comparison is
  deferred to a later snapshot; the report labels it as not measured.

- Refresh the pinned CMUdict CART/multigram comparison with Cartlet 0.6.0,
  retaining the exact source, dependency, data, and split provenance.
- Explain the CART alignment workflow and epsilon scattering in the README,
  and cite the Black/Lenzo/Pagel and Pagel/Lenzo/Black letter-to-sound papers.

## 0.2.0 — 2026-09-12

Changes since `v0.1.0`:

### Release highlights

- Shared Python and CLI workflows for CART/multigram training, dictionary
  processing, and pronunciation review with numeric TSV/JSON results.
- Saved locale preprocessing, pinned ICU/CLDR orthographic inventories, and
  standard-library-only CART deployment bundles.
- Reproducible held-out CMUdict comparisons and explicit model-support scores.

**Upgrading:** this alpha minor release changes APIs, commands, and model
workflows. Primary training now defaults to IPA, preserved stress, native CART,
and pruning; select `cmu` explicitly for CMUdict. Retrain older models and see
[the migration guide](docs/RELEASING.md#upgrading-from-010).

### Training and inference

- Make ranked pronunciation sense numbering optional in lexicon review output:
  `--no-number-senses` and `number_senses=False` emit bare spellings across all
  formats, preserving ranks and source origins. Numbering remains enabled by default.

- Score complete ordered CART pronunciation sequences, summing silent/joined
  emission paths with stable log probabilities and saved preprocessing. Share
  structured score details and numeric candidate JSON; geometric/product replace
  unordered phone aggregation. Restore native CART embedded model metadata.
  Share the saved letter-vocabulary policy across prediction and score surfaces.

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

- Add shared lexicon review and variant ordering APIs plus `phonebox dict review`,
  with numeric TSV/JSON, stable likelihood ranks, effective-phone deduplication
  retaining all source origins, and unfiltered two-column dictionary output.

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
