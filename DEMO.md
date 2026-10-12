# Demo & reproduce

This is the full walkthrough. For the 30-second version, see [`JUDGE.md`](JUDGE.md).

## Live

- Wall: `https://live.plumerank.edycu.dev`
- `GET /build-info` — real `cv2.getBuildInformation()`, instance id, AMI id, commit sha
- `GET /health`
- `/results` — `/build-info` in-page; the held-out operating curve, baseline race, sensitivity
  strip and ablation grid
- `GET /api/watchlist` — the SSE stream the Wall renders
- `GET /api/benchmark` — serves the committed `results/bench_2026-10-10.json` verbatim

## Reproduce — real run, measured just now

```bash
git clone https://github.com/edycutjong/plumerank && cd plumerank
make test      # offline suite + coverage + claim-consistency check
make verify    # clean-venv sanity check (does NOT reproduce the published headline)
               # measured 2026-10-10: real 129 s end to end (results/test_run_2026-10-10.txt)
```

Output of `make verify`'s final step (`scripts/bench.py --arm micro`), pasted verbatim; the full
target also rebuilds the `.venv` from `requirements-dev.lock` and runs the offline suite first:

```
========================================================================
REPRODUCED HERE
========================================================================
  what           : the micro corpus, recomputed live by the real OpenCV 5 pipeline
  corpus         : data/micro/ (10 windows x 31 frames)
  PlumeRank      : p50    n/a min   p95    n/a min   (2/5 detected)
  B1 CNN grid    : p50    n/a min   p95    n/a min   (1/5 detected)      (the SAME onnx, every frame)
  B2 frame-diff  : p50    n/a min   p95    n/a min   (0/5 detected)
  realised rate  : 1.57 alerts/camera-night on the 5 non-fire micro windows (target 2.00, uncapped)
  escalation     : 0.65% of frames touched by ONNX (top-2 of a 10-window roster; the published figure in block 2 is the same rule over a larger one)
  checked against: results/verify_micro.json -- MATCH (byte-for-byte)

========================================================================
PUBLISHED, NOT RECOMPUTED HERE
========================================================================
  source         : results/bench_2026-10-10.json, printed verbatim -- not recomputed here

  PlumeRank -- median time-to-detection @ 2.00 alerts/camera-night
    arm            : hold (54 sequences, 6 excluded from the 60 in manifest_hold.json, see data/excluded.json)
    PlumeRank      : p50   38.0 min   p95    n/a min
    B1 CNN grid    : p50    n/a min   p95    n/a min      (the SAME onnx, every frame)
    B2 frame-diff  : p50    n/a min   p95    n/a min
    realised rate  : 1.99 alerts/camera-night on the confusion set (target 2.00, uncapped)
    escalation     : 1.76% of frames touched by ONNX
    model calls    : PlumeRank 1 / camera-night   |   B1 3645 / camera-night
    t0 sensitivity : advantage not finite at all 11 offsets in [-5,+5]
    background     : mog2
    commit 5df73321059f2797c32d302676b813c81b87fd0c  instance local (not EC2)  wall clock 12m29s  2026-10-10
  -> results/bench_2026-10-10.json

  reproduce with: make fetch && make bench  (about 95 min)
```

Two lines of that printout, read literally: `model calls` is printed rounded to an integer (the
file's `model_invocations_per_camera_night.plumerank` is 1.424 — 75 CNN calls over 4,266 frames,
expressed per 81-frame camera-window; B1's 3,645 is its 191,970 calls over the same 4,266 frames,
same normalisation). `advantage not finite` means PlumeRank has a median at offsets where neither
baseline has one, so a difference in minutes is undefined — not that there is no advantage.

`make test` alone (the offline suite + coverage + `scripts/check_claims.py`): **832 passed, 1
skipped, 100% line coverage on `plumerank/`, ~2 min, no network, no credentials.**

### The pair that reproduces the published headline

```bash
make fetch   # ~35 min: the corpus, sha256-verified, resumable
make bench   # 12m29s on the arm64 machine that produced results/bench_2026-10-10.json (budget up to an hour on a small instance)
# the ablation grid is its own DEV run, quoted by the held-out file:
.venv/bin/python scripts/bench.py --arm dev --ablation   # twelve cells, a few hours, off the box
```

Nobody needs to run this to evaluate the submission — `results/` already carries every
artefact any document cites, and the live box runs the same scoring code: since the held-out
commit `5df7332`, `plumerank/` has changed only in docstrings and in how the evidence crop is
drawn (`EVIDENCE_RENDER_SCALE`, rendering only — nothing reaches a hazard, a rank or a
threshold), and `api/` only in how the wall is served (`git diff --stat 5df7332..HEAD --
plumerank/ api/`). The one pre-read fix to `plumerank/statistic.py` (the x86-64 DIS guard,
`09d816d`, 05:09Z) landed before `5df7332` and before the run that started at 05:37Z (HOLD read logged 05:41Z). It exists so the number
is checkable, not just assertable.

## The held-out receipt

The one HOLD read, as committed (`results/bench_2026-10-10.json`, `results/hold_access.log`):

| | |
|---|---|
| When / where | 2026-10-10T05:37Z, commit `5df7332`, local arm64 machine (`run.instance: local (not EC2)`), wall clock **12m29s** |
| Freeze | `results/constants.sha256` = `02752e86…`, committed before the read; bench.py refuses HOLD otherwise |
| Sequences | 54 measured (60 in the manifest, 6 excluded by `data/excluded.json`) |
| PlumeRank | **27/54 detected, p50 38.0 min**, mean of detected 13.3 min, 1.92 alerts/camera-window realised |
| Like-for-like baselines | B1 18/54, B2 20/54 — no median |
| As-deployed baselines | B1 2/54, B2 4/54 — no median |
| Escalation | 75 of 4,266 frames (1.76%) — within the 2% cap; 1.4 CNN calls per camera-window vs B1's 3,645 |
| Per-frame latency (that machine) | classical p50 6.5 ms / p95 8.5 ms; B1 CNN-grid p50 53.6 ms |
| Sensitivity | p50 32–38 min at ignition-clock offsets −5…0; no median at +1…+5 (`results/sensitivity_2026-10-10.csv`) |

## What a real-run receipt looks like for this project

There is no per-call provider bill here (no hosted LLM API, no metered generation) — the
"receipt" is the pipeline actually executing against real pixels and being checked against a
committed expectation, which is exactly what the block above shows. The AWS side has its own
receipt: one `t4g.small` (Graviton) instance, one S3 bucket (corpus: the DEV and CONFUSE
frames the box replays, 5,082 objects / 2,378,622,722 bytes under the `corpus/` prefix per `aws s3 ls s3://plumerank-corpus/corpus/ --recursive --summarize`, 2026-10-10),
CloudWatch metrics plus drift (±15%) and silence alarms on the alert budget — all funded by a $150 AWS compute
grant from this competition, not a personal card. The held-out run itself cost nothing: it ran
off the box, on the local machine named above.

## Honest limitations

See `README.md`'s "Honest limitations" section — identical list, not duplicated here to avoid
the two drifting apart.
