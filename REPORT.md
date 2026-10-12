# PlumeRank — Technical Report

OpenCV AI Competition 2026, powered by AWS. See [`README.md`](README.md) for the pitch,
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the system diagram, and [`JUDGE.md`](JUDGE.md) for
the 30-second path. This document is the problem/users/evaluation/limitations write-up the
competition's required deliverables list asks for, in one place.

## 1. Problem

Wildfire camera networks (ALERTWildfire, HPWREN, Pyronear) now carry thousands of live tiles,
far more than the people on shift can watch at once. Published smoke detectors are scored on **per-frame
accuracy** (F1, precision, recall) or on time-to-detection with no cap on false alarms. Neither
number tells a dispatcher which of a thousand
tiles to open next, and none of it tells a procurement officer whether the system will survive
contact with a real shift before it gets muted for crying wolf.

The number that matters is different: **how many minutes after ignition does a human actually
look at the right camera, given she will tolerate only a fixed number of interruptions per
camera?** Raise sensitivity and you buy minutes at the cost of alarm fatigue; lower
it and the network goes quiet and late. Every deployed system lives somewhere on that curve.
Deployed systems rarely publish where they sit on it. PlumeRank does. (FIgLib's own reference
model, SmokeyNet, reports a 3.12-minute average time-to-detection at 89.84% precision and 76.45%
recall — Dewangan et al., 2022, arXiv:2112.08598, Table 2 — with no alert budget attached. Delay at
a fixed false-alarm rate is the classical quickest-change-detection criterion: Page, 1954; Lorden,
1971.)

**The unit.** The budget is expressed per *camera-window* (`camera_night` in the code, a name
that reads like a 12-hour night and is not one): one evaluation window — 81 minutes of footage
on one camera, the length of a FIgLib sequence. 2 alerts per 81-minute window is about 18 per
camera over a continuous 12-hour shift. A deployment would set a per-shift budget and re-solve the threshold for it.

## 2. Users

- **Primary — the dispatcher or camera watcher on a daytime shift** at a public fire agency
  command centre, verifying a queue of AI smoke alerts (CAL FIRE has routed such alerts from
  ALERTCalifornia's cameras to its dispatch centres since 2023). Not a data scientist, not a
  developer. The tool has to be legible at a glance, at the end of a long shift, on a projector,
  without training — which is why the Wall's landing band is "cameras,
  rank, evidence crop" and nothing else on the first screen.
- **Secondary — the network operations engineer** who sets the alert budget and answers for
  it. Served by the budget ledger and the CloudWatch drift alarms — the budget is a knob with
  telemetry, not a constant buried in a config file.
- **The population this stands for.** Radeloff et al., *PNAS* 2018: about 43 million US houses —
  roughly one in three — sit in the wildland–urban interface, and the interface is the
  fastest-growing land-use type in the country. Concretely: Paradise, California,
  2018 — 85 deaths, 18,804 structures lost, in a fire whose growth rate made every minute of
  detection latency worth acres of spread. It began around dawn, outside what this daylight-only
  build covers (§7); it is cited for the stakes of detection latency, not as a fire PlumeRank
  would have caught.

## 3. Architecture

Full diagram and data flow: [`ARCHITECTURE.md`](ARCHITECTURE.md). In one paragraph: a classical
OpenCV 5 stage (`plumerank/stabilise.py`, `segment.py`, `statistic.py`) measures four statistics
per foreground component on every frame; they are fused into a hazard, the top-2 ranked
cameras whose hazard clears an escalation floor (~2% of frames) get one ONNX tile score through
`cv2.dnn` folded in (`plumerank/escalate.py`), each camera is then scored against its own
trailing 15 minutes, and that hazard is thresholded to a published alert budget
(`plumerank/rank.py`); and the
ranked result is served by a four-route FastAPI app and a three-route Next.js static export on
one AWS `t4g.small` instance.

## 4. OpenCV 5 implementation

**Required by the competition** ("every entry must use OpenCV 5 for substantive image or video
analysis") and structural here, not decorative: **23 defended OpenCV 5 surfaces across 6
modules** (18 analysis calls, 5 runtime/introspection) (video, imgproc, geometry, bgsegm, dnn, core), mechanically counted from
the real source on every `make test` run by `scripts/check_claims.py` — never hand-typed, never
quoted from a plan. The full table with per-surface "what you would need without it" is in
`README.md`'s "The OpenCV 5 surfaces it uses" section.

The one trained component — a 96×96 ONNX tile classifier, reached through
`cv2.dnn.readNetFromONNX(ENGINE_NEW)` — is trained **only** on Pyro-SDIS (Apache-2.0, French
fire services) and never sees a FIgLib (Californian) pixel; a guard test asserts this at the
manifest level. The identical model is also baseline B1, run over every frame instead of the
~2% the classical stage escalates — the comparison is a controlled experiment rather than a
contest between differently-sized models, because the learned component is the same size on
both sides of it.

## 5. AWS deployment

One `t4g.small` (Graviton/ARM, burstable, `unlimited` credit mode) EC2 instance, one S3 bucket
holding the corpus and a git bundle of the source, CloudWatch carrying the realised alert rate
and escalation fraction as metrics — the alert rate measured on the wall's fire-free
confusion-set cameras only, the same population the threshold was solved on — with drift
alarms (±15% of the budget) and a heartbeat-silence alarm whose action reboots the instance
(`deploy/alarms.sh`, per-instance dimension; no notification target is configured yet). The wall
replays the confusion windows frame-for-frame as the benchmark does, so on a healthy box the
live fire-free rate reproduces the calibrated 1.99, and an IAM instance profile with read-only S3 access to the corpus
plus `cloudwatch:PutMetricData` scoped to the `PlumeRank` namespace. `deploy/bootstrap.sh` provisions the box from the S3
git bundle; `deploy/Caddyfile` reverse-proxies `/api/*`, `/health` and `/build-info` to Uvicorn
on the **same origin** as the static frontend, so production never makes a cross-origin
request the way a local two-port dev setup does.

`GET /build-info` exists specifically so a judge does not have to trust a sentence: it returns
`cv2.getBuildInformation()`, the running OpenCV version, the EC2 instance id and AMI id, and
the git commit sha, read live off the box.

## 6. Evaluation

**Ground truth is mechanical, not hand-labelled.** HPWREN FIgLib records each sequence's
ignition time, so time-to-detection needs zero labelling, no raters, and no institutional
partner. **Every system is calibrated to the same alert budget on the same confusion set**
before comparison (all land at 1.99 alerts/camera-window there), so the comparison happens at
the operating point a fire agency actually buys at, not at each system's most flattering one.
The like-for-like rows put B1 and B2 through the same per-camera baseline PlumeRank uses; on
DEV the realised rates then differ by system, and are listed.

**The headline — the held-out (HOLD) arm, read once after the constants were frozen:**

| Metric | HOLD, 54 sequences (`results/bench_2026-10-10.json`) |
|---|---|
| Detected | **27 / 54** |
| Median time-to-detection (misses = never) | **38.0 min** |
| Mean time-to-detection of the detected fires | 13.3 min |
| Alert rate | 1.99 alerts/camera-window on the confusion set (target 2.0); 1.92 realised on HOLD |
| Escalation fraction | 1.76% of frames — within the 2% design cap |
| **B1 through the same per-camera baseline as PlumeRank** | **18 / 54 detected — no median** |
| **B2 through the same per-camera baseline as PlumeRank** | **20 / 54 detected — no median** |
| B1 as deployed (same ONNX, every frame) | 2 / 54 detected — no median |
| B2 as deployed (frame differencing) | 4 / 54 detected — no median |

27 of 54 is exactly half, so the median is fragile: against an ignition clock shifted by −5…0
minutes it reads 32–38 min, and at +1…+5 minutes it does not exist
(`results/sensitivity_2026-10-10.csv`). The detection counts are the steadier comparison. The
ablation grid the HOLD file quotes (`results/ablation_2026-10-10.csv`) is a DEV run: escalation
on/off detects the same number of fires in five of six pairs, and GSOC with S4 off detects one more
DEV fire (18/29) than the shipped cell — disclosed, not acted on, because DEV is the tuning arm.

**The tuning arm — DEV (diagnostic, never the headline):**

| Metric | DEV, shipped configuration |
|---|---|
| Detected | 17 / 29 |
| Median time-to-detection | 32.0 min |
| Alert rate | 1.99 alerts/camera-window on the confusion set (target 2.0); 2.37 realised on DEV |
| Escalation fraction | 2.14% of frames — over the 2% design cap (`within_max: false` in the run file) |
| **B1 through the same per-camera baseline as PlumeRank** | **14 / 29 detected — no median** (2.51 realised on DEV) |
| **B2 through the same per-camera baseline as PlumeRank** | **10 / 29 detected — no median** (2.05 realised on DEV) |
| B1 as deployed (same ONNX, every frame) | 6 / 29 detected — no median (0.46 realised on DEV; threshold within 1e-6 of 1.0 — saturated probabilities) |
| B2 as deployed (frame differencing) | 3 / 29 detected — no median (1.84 realised on DEV) |

Source: `results/arm_dev_2026-10-10.json`. DEV is the arm the fusion weights were fit on — a
number from it is diagnostic, never a claim of generalisation.

**Failure cases — every DEV sequence, every system.** Minutes from ignition until the camera is
in the top 4 *and* has fired an alert; `miss` means it never did inside the window. From
`per_sequence` in `results/arm_dev_2026-10-10.json` — the same traces the aggregate is computed
from.

| Sequence | PlumeRank | B1 rebased | B2 rebased | B1 deployed | B2 deployed |
|---|---|---|---|---|---|
| `20240622_Border58Fire_om-w-mobo-c` | 8 | miss | miss | miss | miss |
| `20240625_DelMarFire_tdlln-n-mobo-c` | 38 | 5 | 0 | miss | miss |
| `20240701_McCainfire_ws-e-mobo-c` | miss | 29 | 7 | miss | miss |
| `20240724_GroveFire_tp-w-mobo-c` | 1 | miss | miss | miss | miss |
| `20240905_ValleyFire_cp-s-mobo-c` | 32 | miss | miss | miss | miss |
| `20240907_ChurchFire_ws-w-mobo-c` | miss | 14 | miss | miss | 29 |
| `20240907_FIRE_ws-s-mobo-c` | 19 | miss | miss | miss | miss |
| `20240908_BridgeFire_wilson-e-mobo-c` | 14 | miss | miss | miss | miss |
| `20241108_GardenFire_rm-s-mobo-c` | miss | 9 | miss | miss | miss |
| `20241113_Fire_dwpgm-s-mobo-c` | miss | 4 | 3 | 38 | miss |
| `20250107_PalisadesFire_69bravo-e-mobo-c` | 21 | miss | miss | miss | miss |
| `20250107_PalisadesFire_dwpgm-s-mobo-c` | miss | miss | miss | miss | miss |
| `20250123_Border2Fire_om-w-mobo-c` | miss | 11 | miss | 20 | miss |
| `20250123_GilmanFire_tdlln-s-mobo-c` | 34 | miss | miss | miss | miss |
| `20250514_PrescribedFire_hp-e-mobo-c` | 30 | miss | 3 | 21 | miss |
| `20250520_CrossingFire_hp-n-mobo-c` | 13 | 13 | 18 | miss | miss |
| `20250610_CahuillaFire_hp-n-mobo-c` | miss | 38 | 28 | miss | miss |
| `20250612_HighwayFire_hp-e-mobo-c` | 13 | 1 | 1 | miss | miss |
| `20250616_MissionFire_rm-s-mobo-c` | 12 | miss | 14 | miss | miss |
| `20250716_DaleBradfordFire_hp-n-mobo-c` | 7 | 19 | 1 | miss | miss |
| `20250724_PostaFire_ws-w-mobo-c` | miss | 14 | miss | miss | miss |
| `20251009_RanchoFire_bm-s-mobo-c` | miss | miss | miss | 11 | miss |
| `20251102_ScissorsFire_mp-n-mobo-c` | 8 | miss | miss | miss | 9 |
| `20260428_LiveOakFire_rm-s-mobo-c` | miss | 1 | miss | 3 | miss |
| `20260502_MissionFire_rm-s-mobo-c` | miss | miss | miss | miss | miss |
| `20260618_MissionFire_rm-s-mobo-c` | 17 | 12 | 0 | 24 | miss |
| `20260715_ThornFire_ws-n-mobo-c` | miss | miss | miss | miss | miss |
| `20260722_CreelmanFire_bm-s-mobo-c` | 9 | miss | miss | miss | miss |
| `20260820_ChiaFire_hp-n-mobo-c` | 11 | 23 | miss | miss | 28 |

What the rows say, plainly:

- **PlumeRank misses 12 of 29.** B1 through the same per-camera baseline catches 8 of
  those 12, so the two are partly complementary; PlumeRank's lead (17 vs 14) is a lead in
  coverage on this arm, not a dominance on every sequence, and B1-rebased is sometimes faster where
  both detect.
- **3 sequences are missed by all five systems** (`20250107_PalisadesFire_dwpgm-s-mobo-c`, `20260502_MissionFire_rm-s-mobo-c`, `20260715_ThornFire_ws-n-mobo-c`).
  These are the hardest cases on the arm and a starting point for error analysis.
- **DEV is in-sample.** The fusion weights were fit here; the held-out run is the test of whether
  any of this generalises.

**Failure cases — every held-out sequence, every system.** Same definition, from `per_sequence` in
`results/bench_2026-10-10.json` (the single held-out read). All 54 windows are aligned on their
ignition frame and ranked together, tick by tick (`scripts/bench.py`, `alert_trace`): from ignition
onward the top 4 is contested by 54 simultaneous fires, for every system alike.

| Sequence | PlumeRank | B1 rebased | B2 rebased | B1 deployed | B2 deployed |
|---|---|---|---|---|---|
| `20240728_Border65Fire_lp-s-mobo-c` | 10 | miss | 7 | miss | miss |
| `20240728_Border65Fire_om-e-mobo-c` | 28 | miss | 5 | miss | miss |
| `20240728_Border65Fire_pi-s-mobo-c` | miss | miss | 18 | miss | miss |
| `20240802_WarnersFire_sojr-s-mobo-c` | 16 | miss | miss | miss | miss |
| `20240813_PapaFire_rm-w-mobo-c` | miss | miss | 20 | miss | miss |
| `20240823_ProctorFire_sm-s-mobo-c` | miss | 3 | miss | 18 | miss |
| `20240825_TenajaFire_buff-n-mobo-c` | 5 | 15 | 9 | miss | miss |
| `20240903_NavajoFire_cp-n-mobo-c` | miss | 24 | 29 | miss | miss |
| `20240909_BridgeFire_marconi-n-mobo-c` | miss | miss | miss | miss | miss |
| `20240909_BridgeFire_starr-n-mobo-c` | 16 | miss | 2 | miss | miss |
| `20240909_BridgeFire_stgo-e-mobo-c` | miss | 37 | 14 | miss | 37 |
| `20241006_SaddlebackFire_stgo-w-mobo-c` | miss | 32 | miss | miss | miss |
| `20241007_Posta3Fire_pi-e-mobo-c` | 13 | miss | miss | miss | miss |
| `20241031_QuarryFire_sm-w-mobo-c` | miss | 13 | miss | miss | miss |
| `20241108_Pasqual4Fire_wc-n-mobo-c` | miss | 9 | miss | miss | miss |
| `20241108_ResortFire_bm-n-mobo-c` | miss | 25 | miss | miss | miss |
| `20241108_ResortFire_mg-n-mobo-c` | miss | 12 | 8 | miss | miss |
| `20241113_Fire_dwpgm-w-mobo-c` | miss | 3 | miss | miss | miss |
| `20250120_TedWilliamsFire_wc-w-mobo-m` | 29 | miss | miss | miss | miss |
| `20250122_BernardoFire_ch-s-mobo-c` | miss | miss | miss | miss | miss |
| `20250226_PennyFire_ml-w-mobo-c` | miss | miss | miss | miss | miss |
| `20250429_TrailFire_rm-w-mobo-c` | miss | 36 | 17 | miss | miss |
| `20250501_FelipeFire_ml-n-mobo-c` | miss | miss | miss | miss | miss |
| `20250516_AcademyFire_wc-n-mobo-c` | 3 | 5 | 17 | miss | miss |
| `20250530_HendersonFire_bh-w-mobo-c` | 35 | miss | miss | miss | miss |
| `20250617_MonteFire_cp-w-mobo-c` | 12 | miss | miss | miss | miss |
| `20250709_SteeleFire_lp-w-mobo-c` | miss | 15 | 5 | miss | miss |
| `20250709_SteeleFire_om-n-mobo-c` | 20 | miss | miss | miss | miss |
| `20250710_SpringsFire_lp-n-mobo-c` | 1 | 0 | miss | 24 | miss |
| `20250801_BernardoFire_bl-n-mobo-c` | miss | miss | miss | miss | miss |
| `20250823_VailFire_hp-w-mobo-c` | miss | 5 | miss | miss | miss |
| `20250905_LodgeFire_sm-n-mobo-c` | 2 | miss | miss | miss | 28 |
| `20250908_CochesFire_sm-n-mobo-c` | miss | miss | miss | miss | miss |
| `20250914_HenshawFire_mg-n-mobo-c` | 8 | miss | miss | miss | miss |
| `20251019_RanchFire_wc-w-mobo-m` | miss | miss | 6 | miss | miss |
| `20251102_ScissorsFire_mp-n-mobo-m` | 8 | miss | miss | miss | 9 |
| `20251102_ScissorsFire_vo-e-mobo-m` | miss | miss | miss | miss | miss |
| `20260128_Fire_wilson-s-mobo-c` | miss | miss | miss | miss | miss |
| `20260319_CrosleyFire_hp-w-mobo-c` | 38 | miss | miss | miss | miss |
| `20260319_CrosleyFire_mpo-w-mobo-c` | miss | 16 | 16 | miss | miss |
| `20260502_MissionFire_rm-e-mobo-c` | 24 | 2 | 25 | miss | miss |
| `20260519_TusilFire_mlo-s-mobo-c` | 1 | miss | miss | miss | miss |
| `20260622_BernardoFire_wc-w-mobo-c` | 5 | miss | 37 | miss | miss |
| `20260629_JunctionFire_bm-e-mobo-c` | 0 | miss | miss | miss | miss |
| `20260629_JunctionFire_mg-e-mobo-c` | 25 | miss | miss | miss | miss |
| `20260629_JunctionFire_vo-n-mobo-c` | miss | miss | miss | miss | miss |
| `20260629_JunctionFire_vo-w-mobo-c` | 26 | miss | miss | miss | miss |
| `20260722_CreelmanFire_cp-w-mobo-c` | 8 | miss | 15 | miss | miss |
| `20260722_CreelmanFire_mg-s-mobo-c` | 9 | miss | miss | miss | miss |
| `20260722_RainbowFire_bh-w-mobo-c` | miss | miss | 37 | miss | miss |
| `20260722_RainbowFire_rm-e-mobo-c` | 0 | miss | miss | miss | miss |
| `20260807_BeaverFire_lp-w-mobo-c` | 7 | 4 | 6 | miss | miss |
| `20260816_SorrentoFire_sdsc-e-mobo-c` | miss | miss | miss | miss | 21 |
| `20260817_Church2Fire_mlo-s-mobo-c` | 11 | miss | 10 | miss | miss |

What these rows say:

- **PlumeRank misses 27 of 54**; B1 through the same baseline catches 13 of those. The two
  overlap on only 5 detections and together catch 40 of 54 — mostly different fires. That is
  post hoc on the held-out arm: a hypothesis for combining them, not a result.
- **Paired, the lead is not significant**: 22 vs 13 against rebased B1 (exact McNemar p = 0.18),
  17 vs 10 against rebased B2 (p = 0.25); where both detect, rebased B1 is faster on 3 of 5 and
  rebased B2 on 5 of 10.
- **9 sequences are missed by all five systems**.

**How the headline was protected.** The constants were frozen and committed
(`results/constants.sha256`) before any held-out frame was read; `scripts/bench.py` refuses to
read HOLD otherwise, verifies every frame against the fetch receipt, and logs the single read with
its commit (`results/hold_access.log`). The live Wall runs on the frozen threshold the held-out
run solved on the fire-free confusion set. Reproduce: `make fetch && make bench` (`make verify`,
≤3 minutes, is a sanity check against the micro-corpus only and does not reproduce this number —
see `DEMO.md`).

**Sensitivity design.** Because the reported ignition time is itself soft, the headline is
designed to be re-scored at every offset from −5 to +5 minutes from the same cached score
matrix in the same run, so the sensitivity curve cannot drift from the headline it describes
(`scripts/bench.py --sensitivity`).

**Testing, as a form of evaluation in its own right.** 833 tests passed, 1 skipped locally (CI, without the corpus, runs 821 and skips 13), 100% line
coverage on `plumerank/`; 1,500 hypothesis-generated cases (`tests/test_invariants.py`)
property-testing that the fused hazard never leaves `[0, 1]` and the alert budget is never overspent
across 500 generated inputs or call sequences each; `scripts/check_claims.py` mechanically checks every route,
OpenCV symbol, dependency, Makefile target and file path every document names against the
real source, on every test run.

## 7. Limitations

See `README.md`'s "Honest limitations" section for the full list with evidence citations —
summarised: daylight-only (no IR hardware); **cloud, not fog, turned out to be the real
adversary**, and none of the four statistics is a good detector on its own (AUC 0.528 / 0.50 /
0.44 / 0.538 for S1–S4, pre- vs post-ignition on DEV, and the fused per-frame model beats the
single bit "a component exists" by only 0.0015 AUC in-sample; S3, the one the project was built around, is
anti-predictive yet carries the largest fusion weight, and no S3-off ablation has run yet);
the per-camera baseline is what carries the ranking; stabilisation registers each frame to its raw
predecessor, and refereed against the ground its warps of ≥ 1 px are real motion on one shaking
DEV camera (48 of 55) and otherwise — all 39 on CONFUSE — follow the moving sky while the ground
stays still (`results/registration_2026-10-10.json`) — a counterfactual replay without those
warps moves CONFUSE's uncapped crossings from 64 to 63 and the solved threshold from 0.8492 to
0.8482 (`results/registration_alerts_2026-10-10.json`);
the evidence crop (the largest moving component) contains the plume on 9 of 21 DEV first hits —
cloud on 7 (`results/evidence_first_hits_2026-10-10.json`); the novelty is a metric reframe rather than a new sensor;
one instance with no HA; no managed-cloud-vision-API baseline; and no dispatch, suppression,
or upload path, by architecture rather than oversight.

## 8. Responsible-use considerations

**False positive:** rolls an engine to nothing, pulls crews from a real incident, and at
network scale creates the alarm fatigue that gets a whole system muted — that is how these
tools die in the field. **False negative:** a fire grows undetected.

Safeguards, each a build item rather than a paragraph (verified by `tests/test_api.py` and
`tests/test_rank.py`, not merely asserted):

1. A **ranking aid for a human watcher**, with no dispatch path anywhere in the codebase.
2. **Never hides or suppresses** a camera — the full roster stays on screen; the system only
   reorders it.
3. The **alert budget is published, logged and tunable**, with CloudWatch drift alarms.
4. Every ranked entry ships its **evidence crop**, so a human can overrule in one glance.
5. **No new imagery is collected** — the wall replays public HPWREN camera footage (wildland and
   some developed areas at long range); the
   only frames written are small evidence crops and thumbnails of it, overwritten per replay.
6. A **persistent, non-dismissible limits strip** on every screen: daylight only; ranking aid,
   not dispatch; never suppresses a camera.

## 9. What's next

Error analysis on the three DEV sequences every system misses, and a test of whether
PlumeRank and B1-through-the-baseline combine (on the held-out arm they overlap on 5 detections and together catch 40 of 54 — §6); night/IR detection (needs infrared hardware this build does not have); live RTSP ingest
(replayed from disk today); a second camera network (Pyronear) to test whether the per-camera
baseline and the alert-budget calibration transfer across networks without refitting.
