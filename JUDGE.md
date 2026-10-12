# For judges — the 30-second path

**The claim:** PlumeRank ranks which wildfire-camera tiles a dispatcher should open next,
calibrated to a fixed budget of 2 alerts per camera-window, and measures itself on median
minutes-after-ignition until the right camera reaches the top of that ranking. (A *camera-window*
is 81 minutes of footage on one camera — the length of a FIgLib sequence — not a 12-hour night;
the code calls it `camera_night`.)

## 1. Click these, in order

1. **The Wall** — `https://live.plumerank.edycu.dev` — no login. The live ranking: 29 development-set fire
   sequences and 33 fire-free confusion-set windows (the confusion set is marked on screen),
   replayed from real HPWREN footage through the real pipeline, on the frozen threshold the held-out
   run solved on the fire-free confusion set (§4).
2. **`/build-info`** — `https://live.plumerank.edycu.dev/build-info` — `GET /build-info`. Real
   `cv2.getBuildInformation()` output, the EC2 instance id and AMI id, and the git commit this
   box is running, read live off the box — nothing here is typed by hand.
3. **`/results`** — `https://live.plumerank.edycu.dev/results` — the same `/build-info` block rendered
   in-page, and the held-out evidence (baseline race, operating curve, sensitivity strip, ablation
   grid) from `results/bench_2026-10-10.json`.

## 2. The receipt

| | |
|---|---|
| Tests | **833 passed, 1 skipped locally (CI, without the corpus, runs 821 and skips 13)**, 100% line coverage on `plumerank/` |
| Property-based testing | **1,500 generated cases** (`tests/test_invariants.py`, `hypothesis`) — the fused hazard never leaves `[0, 1]` across 500 valid-input cases and never silently passes an invalid one through another 500; the alert budget can never be overspent under any of 500 generated call sequences |
| Static analysis | `ruff check .` and `mypy plumerank api --ignore-missing-imports` — both clean |
| Claim/code consistency | `scripts/check_claims.py`, run inside `make test` — 23 defended OpenCV 5 surfaces across 6 modules (18 analysis calls, 5 runtime/introspection), every API route, every dependency, every Makefile target and every file path this document and `README.md` name, checked mechanically against the real source on every test run |
| DEV arm (orientation only — fit data, never the claim) | 17/29 detected, p50 32.0 min; calibrated to 1.99 alerts/camera-window on the confusion set, 2.37 realised on DEV. Like-for-like baselines (B1/B2 through the same per-camera baseline): 14/29 and 10/29 (`results/arm_dev_2026-10-10.json`) |
| **HOLD arm — the headline** | **27/54 detected, median 38.0 min** at 2.0 alerts/camera-window, read once after the freeze. All 54 windows are aligned on ignition and ranked against each other, so top 4 is a fire-against-fire test for every system; 38.0 is a nearest-rank median, so it is the 27th and slowest detection (detected fires: median 10.0, mean 13.3). Paired p = 0.18 and 0.25 against the like-for-like baselines — a lead, not significant; the cost gap (1.4 vs 3,645 CNN calls per camera-window) is. Like-for-like B1 18/54, B2 20/54 — neither reaches a median. The median exists only at ignition-clock offsets −5…0 min (`results/bench_2026-10-10.json`, `results/sensitivity_2026-10-10.csv`) — see §4 |
| Compute | One AWS `t4g.small` (Graviton/ARM), one S3 bucket (the replayed corpus: 5,082 objects, 2,378,622,722 bytes per `aws s3 ls s3://plumerank-corpus/corpus/ --recursive --summarize`, 2026-10-10), CloudWatch metrics, two drift alarms (±15% around the 2.0 budget) and a heartbeat-silence alarm (`deploy/alarms.sh`). No Lambda, no managed vision API. |

## 3. Reproduce it yourself

```bash
git clone https://github.com/edycutjong/plumerank && cd plumerank
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.lock
make test      # ~2 min: the 833-test offline suite + coverage + the claim-consistency check above
make verify    # ≤3 min: rebuilds the venv from the lockfile, re-runs the offline suite, then
               # recomputes the micro-corpus metrics and checks them byte-for-byte against the
               # committed results/verify_micro.json. This is the fast sanity check — it does
               # NOT reproduce the published headline, and the command's own output says so.
```

The pair that **does** reproduce the held-out headline (`results/bench_2026-10-10.json`):
`make fetch && make bench` — ≈35 min corpus fetch, then the held-out evaluation (12m29s on the
arm64 machine that produced the published file; not yet timed on a small instance). The ablation grid is a separate DEV run,
`scripts/bench.py --arm dev --ablation` (twelve cells, about 65 min of per-cell wall clock in `results/ablation_2026-10-10.csv`, run off the box), which the
held-out file quotes verbatim. Nobody is asked to run that to evaluate this submission; it exists so the number is
checkable rather than assertable.

## 4. How the headline was produced — and how far to trust it

The held-out run happened once, on 2026-10-10, at commit `5df7332`:

- The constants were frozen first, in their own commit (`results/constants.sha256`,
  `02752e86…`); `scripts/bench.py` refuses to read a HOLD frame unless that hash matches.
- The single read is logged with its commit (`results/hold_access.log`), and every frame was
  checked against the fetch receipt before it counted.
- 54 sequences were measured: 60 in `data/manifest_hold.json`, 6 excluded by the published rule
  (`data/excluded.json`).
- `GET /api/benchmark` serves `results/bench_2026-10-10.json` verbatim; the live Wall runs on its
  threshold (0.849 — the same value the DEV run solved, because it depends only on the
  confusion set).

**How far to trust it.** 27 of 54 is exactly half, so the (nearest-rank) median is on a knife edge: re-scored
against an ignition clock shifted by −5…0 minutes it reads 32–38 min, and at +1…+5 minutes it
does not exist. The detection counts — 27 against 18 and 20 for the like-for-like baselines — are
the steadier comparison, but not a significant one at 54 windows (paired, exact McNemar p = 0.18 and
0.25); the cost gap — 1.4 CNN calls per camera-window against 3,645 — is. The run executed on a local arm64 machine, not the endpoint box.

## 5. Honest limitations

1. **Daylight only** — no IR hardware, no night detection.
2. **Cloud, not fog, is the real adversary, and the single statistics are weak** — the
   statistic (S3) this project was built around scores AUC 0.44 pre- vs post-ignition on DEV
   frames with a component (anti-predictive; S1 0.528 and S2 0.50 are near chance,
   `results/statistics_confound_2026-09-06.md`), yet S3 carries the largest fusion weight and no
   S3-off ablation has run. The per-camera baseline carries the ranking.
   Said here first, not found by you.
3. **One instance, no HA.** Every published number is committed in `results/` so the claim
   survives the box going down.
4. **No dispatch, no suppression, no upload path** — architecturally absent, not merely unused.
5. **Half the held-out fires are missed**, and the classifier escalation adds no detection on the
   DEV ablation grid (`results/ablation_2026-10-10.csv`; README limitation 8).

## 6. Links

- Demo video (3:34): https://youtu.be/QqX85JuNfVM
- Live endpoint: `https://live.plumerank.edycu.dev` (the same box also answers on its Elastic IP, `http://44.242.91.201`, the address used before the hostname existed)
- `/build-info` · `/health` · `/results`
- Repository: https://github.com/edycutjong/plumerank (this box's own `/build-info` names the exact commit it runs)
- Site and pitch deck: https://plumerank.edycu.dev · https://plumerank.edycu.dev/pitch/
- `README.md` (architecture + full testing detail) · `DEMO.md` (full reproduce walkthrough)
