# Historical runs: not the final matched comparison

- `frame_results`: the original four Dense-head experiments with independently
  generated validation speakers. CPU pilots under `cpu_pilot_archive` were
  excluded from the original final GPU comparison. Keep their identities separate.
- `results_provisional`: earlier pooled-final-layer linear baselines, with
  different normalization and checkpoint selection. They do not isolate the effect
  of replacing the current Dense head.

Use `../matched_loso` for the current reported four-configuration comparison.
Original numeric artifacts and fingerprints are preserved. Full historical weights
and earlier pooled features are available in the historical release archive.
