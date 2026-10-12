# results/

Every number any document in this repository quotes is in a file here, committed by the run
that produced it.

| File | What it is |
|---|---|
| `arm_dev_<date>.json` | The shipped configuration on the DEV arm (the fit/tuning data — diagnostic, never the headline). Includes `baselines_rebased` (like-for-like) and `per_sequence` (failure cases). The live wall calibrates from the newest one until a held-out file exists. |
| `arm_dev_<cell>_<date>.json` | Ablation cells and sweeps (`s4off`, `ztrail10`, a background or budget) — never the shipped configuration. |
| `bench_<date>.json` | The held-out (HOLD) run — the only file allowed to be called the headline. `bench_2026-10-10.json`: 27/54 detected, p50 38.0 min, read once at commit `5df7332` (`hold_access.log`) after the freeze (`constants.sha256`). |
| `verify_micro.json` | What `make verify` must reproduce byte-for-byte on the committed micro corpus. |
| `test_run_2026-10-12.txt` | The current receipt: `make verify` 833 passed, 1 skipped, MATCH, real 125 s; CI 821 / 13 (run 38228827405, commit b04d3ae). |
| `test_run_2026-10-10.txt` | The local `make test` summary line (832 passed, 1 skipped, full corpus) and the timed `make verify` run (real 129 s, MATCH), beside the CI count without the corpus (820 / 13). |
| `*.md` | Dated analysis notes, written when each measurement was taken and left as written. |

## Checkpoint names used in the dated notes

The notes refer to checkpoints of a plan written before the code, by day number of the build:

| Label | Date | Checkpoint |
|---|---|---|
| D3 | 2026-09-01 | go/no-go on the boundary statistic (`gonogo_2026-09-01.md`) |
| D10 | — | the S4 ablation gate: keep S4 only if it adds detections (`s4_ablation_2026-09-06.md`) |
| D11 | 2026-09-09 | the alert-budget checkpoint: what each confusion class costs of the budget (`ruling_d11_contribute.md`, `d11_fitted_2026-09-06.md`) |
| D31 | — | the demo wall's build date, before the constants freeze |
| D40 | — | the single held-out run, after the constants freeze |

The dates in the filenames are when each note was written; several checkpoints ran early or late,
and the notes say so.
