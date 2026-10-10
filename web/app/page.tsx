"use client";

import Link from "next/link";
import { useState } from "react";
import { BudgetMeter } from "@/components/BudgetMeter";
import { CalibrationNote, calibrationKindOf } from "@/components/CalibrationNote";
import { Clock } from "@/components/Clock";
import { DegradedStrip, humaneReason } from "@/components/DegradedStrip";
import { EvidenceCard } from "@/components/EvidenceCard";
import { TopBar } from "@/components/TopBar";
import { Wall } from "@/components/Wall";
import { WatchList, confusionCamerasOf } from "@/components/WatchList";
import { DEV_RUN_FILE, DEV_SUMMARY } from "@/lib/devRun";
import { count, minutes, rate } from "@/lib/format";
import { useBenchmark } from "@/lib/useBenchmark";
import { useMediaQuery } from "@/lib/useMediaQuery";
import { useWatchlist } from "@/lib/useWatchlist";
import type { WatchRow } from "@/lib/types";

function str(v: unknown): string | null {
  return typeof v === "string" && v !== "" ? v : null;
}

/**
 * "/" -- The Wall. The thirty-second demo.
 *
 * Everything on the wall is live once the pipeline can run: the roster, the ranks,
 * the crops, the budget slots. There is no pre-rendered ranking anywhere on the judged
 * path -- the replay loop starts when you connect and the background model warms from
 * the pre-ignition frames FIgLib supplies, on screen. Until the held-out run publishes
 * its threshold the wall runs on the one the development run solved on the same
 * fire-free confusion set, and says so (CalibrationNote).
 */

/** Slots per camera when the stream does not publish its own. The operating design is
 *  2 (constants.ALERTS_PER_CAMERA_PER_NIGHT); it decides a badge, never a measurement. */
const DEFAULT_SLOTS_PER_CAMERA = 2;

/** One designed panel for the unavailable pipeline, in place of the wall, clock and
 *  budget -- three empty panels each promising "once it is running" read as a demo about
 *  to start, which it is not. */
function UnavailablePanel({ reason }: { reason: string }) {
  return (
    <section className="card" style={{ padding: "18px 20px", marginTop: 16 }} role="status">
      <p className="label" style={{ margin: "0 0 6px", color: "var(--alert)" }}>
        live pipeline unavailable on this box
      </p>
      <p style={{ color: "var(--ink-1)", margin: "0 0 8px", maxWidth: 760 }}>{reason}</p>
      <p style={{ color: "var(--ink-2)", fontSize: "var(--size-label)", margin: 0, maxWidth: 760 }}>
        Nothing is drawn in its place — no placeholder tiles, no cached rows. The committed
        development run is on <Link href="/results">/results</Link>, and the build this box
        is running is at <a href="/build-info">/build-info</a>.
      </p>
    </section>
  );
}

export default function WallPage() {
  const { meta, rows, frameIndex, warm, framesSeen, state, stale, error } = useWatchlist();
  const { data: benchmark, loading: benchmarkLoading } = useBenchmark();
  const [selected, setSelected] = useState<WatchRow | null>(null);

  const compact = useMediaQuery("(max-width: 1279px)");
  const topK = meta?.top_k ?? null;
  const perCamera = meta?.budget_per_camera ?? DEFAULT_SLOTS_PER_CAMERA;
  const threshold = typeof meta?.threshold === "number" ? meta.threshold : null;
  // api/main.py publishes its degraded flag ON the meta event, not only on /health, and
  // the stream stays open when it fires. Reading it is the only way this page can tell
  // an outage apart from an idle loop -- without it the wall captions a dead pipeline
  // "replay paused -- no viewers" while a viewer is looking at it.
  const pipelineDown = meta?.live_pipeline?.available === false;
  // The provenance caption names the held-out sequence count only when the published
  // run really is the held-out one. bench.py can write a dev-arm file (--arm dev), and
  // captioning the wall "measured on N HOLD sequences" off a DEV run's count would be
  // the exact confusion the caption exists to prevent.
  const publishedHold =
    benchmark !== null && (benchmark.run.arm === null || benchmark.run.arm === "hold");
  const holdSequences = publishedHold && benchmark !== null ? benchmark.run.n_sequences : null;
  // The count the exclusion list took out, carried beside the count that survived so that
  // no caption on this page can quote one without the other. bench.py publishes both
  // against the same arm, so holdSequences + holdExcluded is the manifest's own window
  // count -- an addition a reader can check, rather than a third figure that could
  // disagree with the two it summarises.
  const holdExcluded = publishedHold && benchmark !== null ? benchmark.exclusions.count : null;
  const holdP50 = publishedHold ? (benchmark?.headline.p50_minutes ?? null) : null;
  const confusionCameras = confusionCamerasOf(meta);
  const calibration = calibrationKindOf(meta);
  const dev = DEV_SUMMARY;

  return (
    <>
      <TopBar stream={state} />

      <main style={{ padding: "20px 16px 0" }}>
        {/* The landing band is deliberately SHORT. This route is the thirty-second
            demo, so the wall and the rank digit have to be on the first screen at
            1280 px: a hook that pushed them under the fold would be a landing page
            wearing the product as a footer. The two long-form notes -- the declined
            awards and the one-theme refusal -- were developer notes and are gone. */}
        <section className="landing">
          <div>
            <h1 className="display">
              Two alerts per camera per 81-minute window, hard budget. Which camera do you look at first?
            </h1>
            <p style={{ color: "var(--ink-1)", margin: 0 }}>
              PlumeRank ranks a wildfire-camera network so the camera most likely showing a
              new plume reaches the top of a dispatcher&apos;s watch list — under a hard
              budget of two alerts per camera&#8209;window, with the threshold solved on
              fire-free windows rather than picked by hand. (A camera&#8209;window is 81 minutes
              of footage on one camera, the length of a FIgLib sequence — not a 12-hour shift.)
              OpenCV 5 does the per-frame work: background modelling, optical flow and plume
              geometry, fused into one hazard and scored against each camera&apos;s own recent
              history.
            </p>
          </div>

          <div>
            {holdP50 !== null && benchmark !== null ? (
              <p style={{ margin: "0 0 8px" }}>
                <span className="mono num" style={{ fontSize: "var(--size-metric-s)", color: "var(--ink-0)" }}>
                  {minutes(holdP50)}
                </span>{" "}
                <span className="label" style={{ textTransform: "none", letterSpacing: 0 }}>
                  min median time-to-detection
                  {benchmark.budget.target === null
                    ? ""
                    : ` at ${benchmark.budget.target.toFixed(1)} alerts per camera-window`}
                  {holdSequences === null || holdExcluded === null
                    ? ", on the held-out sequences"
                    : `, on ${holdSequences} of ${holdSequences + holdExcluded} held-out sequences, ${holdExcluded} excluded`}
                </span>
              </p>
            ) : benchmarkLoading ? (
              // While /api/benchmark is in flight, hold the held-out sentence's shape: the
              // development-set fallback below is three times longer, and swapping it out
              // when the result landed shifted the make-verify row (Lighthouse CLS 0.126).
              <p style={{ margin: "0 0 8px" }} aria-busy="true">
                <span className="mono num" style={{ fontSize: "var(--size-metric-s)", color: "var(--ink-3)" }}>
                  …
                </span>{" "}
                <span className="label" style={{ textTransform: "none", letterSpacing: 0 }}>
                  min median time-to-detection at 2.0 alerts per camera-window, on the
                  held-out sequences — loading the committed result
                </span>
              </p>
            ) : (
              <p style={{ margin: "0 0 8px" }}>
                <span className="mono num" style={{ fontSize: "var(--size-metric-s)", color: "var(--ink-0)" }}>
                  {minutes(dev.p50)}
                </span>{" "}
                <span className="label" style={{ textTransform: "none", letterSpacing: 0 }}>
                  min median time-to-detection on the development set: {count(dev.detected)} of{" "}
                  {count(dev.n)} fires detected. Threshold solved to {rate(dev.calibrationRate)}{" "}
                  alerts per camera-window on the confusion set; {rate(dev.devRate)} realised on the
                  development set. In-sample — the constants were fit on these fires (
                  <span className="mono">{DEV_RUN_FILE}</span>). The held-out result has not been
                  run yet and is the only number this project will call its headline.
                </span>
              </p>
            )}
            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <code className="code">make verify</code>
              <span className="label" style={{ textTransform: "none", letterSpacing: 0, flex: "1 1 200px" }}>
                about 3 min — the tests plus a small committed sample, re-run. The development
                numbers and the held-out run each have <Link href="/results#reproduce">their own command</Link>.
              </span>
            </div>
            <p style={{ margin: "8px 0 0", fontSize: "var(--size-label)" }}>
              <Link href="/results">results and the build string</Link> ·{" "}
              <Link href="/judge">the 30-second path for judges</Link>
            </p>
          </div>
        </section>

        {pipelineDown ? (
          <UnavailablePanel
            reason={
              humaneReason(meta?.live_pipeline?.detail) ??
              "the box reports it cannot run the pipeline."
            }
          />
        ) : (
        <div className="wall-page" style={{ padding: "16px 0 0" }}>
          <div className="area-wall">
            <Wall
              meta={meta}
                rows={rows}
                topK={topK}
                stream={state}
              warm={warm}
              framesSeen={framesSeen}
              compact={compact}
              holdSequences={holdSequences}
              holdExcluded={holdExcluded}
            />
            <CalibrationNote kind={calibration} />
          </div>

          <div className="area-rail rail">
            <div className="rail-clock">
              <Clock meta={meta} frameIndex={frameIndex} />
            </div>
            <div className="show-mobile">
              <CalibrationNote kind={calibration} />
            </div>
            <div className="rail-list">
              <WatchList
                rows={rows}
                topK={topK}
                perCamera={perCamera}
                threshold={threshold}
                stream={state}
                confusionCameras={confusionCameras}
                onSelect={setSelected}
              />
            </div>
          </div>

          <div className="area-budget">
            <BudgetMeter meta={meta} rows={rows} benchmark={benchmark} />
          </div>
        </div>
        )}

        {/* BELOW the wall, not above it. The stream's verdict arrives after first paint,
            and a strip inserted above the wall pushed the whole wall 107 px down the
            screen (CLS 0.15 on this route). The wall's own header already says "live
            pipeline unavailable" in place, so the notice is not lost by moving the
            committed-results strip under it. */}
        {state === "down" && !pipelineDown ? (
          <div style={{ marginTop: 16 }}>
            <DegradedStrip
              commit={benchmark?.run.commit ?? null}
              file={benchmark ? str(benchmark.raw.results_file) : null}
              title="live stream not delivering"
              detail={
                (error ?? "the live stream is not delivering") +
                (stale
                  ? ". The rows shown are the last ones received and are NOT current. Reconnecting."
                  : ". Reconnecting.")
              }
            />
          </div>
        ) : null}

        <div style={{ height: 22 }} />
      </main>

      {selected ? <EvidenceCard row={selected} onClose={() => setSelected(null)} /> : null}
    </>
  );
}
