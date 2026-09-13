# Archived locale-aware CMUdict snapshot

These [measurements](CMUDICT_COMPARISON.md) and their
[JSON source](cmudict-comparison.json) are preserved byte-for-byte from Phonebox
revision `489097f15c2596b1ae8f958ec8d81336697316ed`. They are historical evidence,
not accuracy claims for the current release. Their reproduction commands refer
to the repository layout at that recorded revision.

The multigram model in this snapshot predates the current end-event and
language-model normalization corrections and removal of the auxiliary q factor.
The snapshot also uses locale-aware grapheme/phone joins: phone error counts
cooked target atoms, which can combine multiple source phones. Its train/test
split has no separate development partition. Those choices differ from the
[current shared-data comparisons](../../G2P_BENCHMARKS.md), so their scores
cannot be interpreted as a controlled before/after improvement.

See [CMUdict workflows](../../CMUDICT_COMPARISON.md) for current usage and the
[shared-data protocol](../../REPRODUCIBLE_BENCHMARKS.md) for the release's
measured comparisons. The original pair passed its rendering check before archival. A committed
[SHA-256 manifest](manifest.json) and CI integrity test now preserve those exact
historical bytes and their source/protocol identity. New renderer changes do
not regenerate historical reports; current experiments retain `--check` for
their own JSON/Markdown pairs.
