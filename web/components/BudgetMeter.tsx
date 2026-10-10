"use client";

import { useEffect, useRef } from "react";
import { calibrationKindOf } from "@/components/CalibrationNote";
import { DEV_RUN } from "@/lib/devRun";
import { rate, ABSENT } from "@/lib/format";
import type { Benchmark, WatchMeta, WatchRow } from "@/lib/types";

/**
 * DISCRETE COUNTABLE SLOTS (124 on the 62-camera wall) -- 2 per camera, grouped by camera. Never a percentage,
 * never a gauge, never an arc.
 *
 * A dispatcher's night contains a countable number of interruptions. "63% of budget
 * used" does not say that; sixty-two little pairs of boxes, of which eleven are filled,
 * says it without arithmetic and says it from across a control room. The grouping by
 * camera is the second half of the idea: the budget is spent PER CAMERA, so a meter
 * that pooled 124 slots into one bar would show a number the system does not actually
 * enforce.
 *
 * Two rates share one name in this project and this component publishes both, labelled,
 * because publishing either one unlabelled is how they get confused:
 *   - the SLOTS are budget-capped -- they are alerts that were actually spent tonight;
 *   - "realised (uncapped)" comes from /api/benchmark and counts threshold crossings
 *     WITHOUT the cap, because that is the quantity rank.solve_threshold was calibrated
 *     on. Capped counting makes every threshold satisfy the target and the solver
 *     degenerates.
 */

/** Slots per camera. The operating design is 2 (constants.ALERTS_PER_CAMERA_PER_NIGHT)
 *  and that is a shape, not a measurement -- but the server may still override it, so
 *  the meta value wins whenever the stream publishes one. */
const DEFAULT_SLOTS_PER_CAMERA = 2;

function Slot({ filled, justFilled }: { filled: boolean; justFilled: boolean }) {
  return (
    <span
      className={`slot${filled ? " slot--filled" : ""}${justFilled ? " slot--just-filled" : ""}`}
      aria-hidden="true"
    />
  );
}

export function BudgetMeter({
  meta,
  rows,
  benchmark,
}: {
  meta: WatchMeta | null;
  rows: WatchRow[];
  benchmark: Benchmark | null;
}) {
  const perCamera = meta?.budget_per_camera ?? DEFAULT_SLOTS_PER_CAMERA;
  const cameras = meta?.cameras ?? rows.map((r) => r.camera_id);

  const spentByCamera = new Map<string, number>();
  for (const r of rows) spentByCamera.set(r.camera_id, r.budget_spent);

  // Which slots filled since the previous tick -- the only thing the 160 ms scale-in is
  // allowed to fire on. Animating every filled slot on every tick would make the second
  // of the product's two animations mean "a tick happened", which is not worth a
  // channel.
  const previous = useRef<Map<string, number>>(new Map());
  // The first tick a viewer sees is not a fill. A judge who opens the page at 04:50 is
  // joining a night already in progress, so every slot the roster has already spent
  // would otherwise scale in at once -- which would make the product's second and last
  // animation mean "the page loaded", the exact thing the two-animation budget forbids.
  const seeded = useRef<boolean>(false);
  const justFilled = new Map<string, number>();
  if (seeded.current) {
    for (const cam of cameras) {
      const before = previous.current.get(cam) ?? 0;
      const now = spentByCamera.get(cam) ?? 0;
      if (now > before) justFilled.set(cam, before);
    }
  }
  useEffect(() => {
    previous.current = new Map(spentByCamera);
    if (rows.length > 0) seeded.current = true;
  });

  const totalSlots = cameras.length * perCamera;
  const totalSpent = cameras.reduce((acc, c) => acc + (spentByCamera.get(c) ?? 0), 0);

  const target = benchmark?.budget.target ?? meta?.budget_target ?? null;
  // Before the held-out file exists the wall runs on the development run's threshold
  // (calibration_kind "development"), so the calibration figure is that run's -- the same
  // compiled-in DEV payload /results shows, never a typed number.
  const kind = calibrationKindOf(meta);
  const realised =
    benchmark?.budget.realised_uncapped ??
    (kind === "development" ? DEV_RUN.budget.realised_uncapped : null);
  const source = benchmark ? "held-out run" : kind === "development" ? "development run" : null;
  const tolerance = benchmark?.budget.drift_tolerance ?? meta?.drift_tolerance ?? null;

  // What this pill can honestly say. `realised` is the OFFLINE calibration rate from the
  // committed results file -- solved once on the confusion set -- not a live measurement,
  // so it is labelled "calibration" and never "drift OK": until 2026-10-04 it read
  // "drift OK" in green, which looked like live monitoring and could never turn amber.
  // The band is the published tolerance, read as a fraction of the target, the same way
  // scripts/bench.py's two-sided calibration check reads it.
  const band = tolerance !== null && target !== null ? tolerance * target : null;
  const off = realised !== null && target !== null ? Math.abs(realised - target) : null;
  const bandText = tolerance === null ? "" : ` (band \u00b1${Math.round(tolerance * 100)}%)`;
  const bandPct = tolerance === null ? "band" : `\u00b1${Math.round(tolerance * 100)}%`;
  const driftClass =
    off === null || band === null ? "badge--spent" : off <= band ? "badge--ok" : "badge--alert";
  // The band is about the CONFUSION SET the threshold was solved on -- said in the pill,
  // because next to a hero quoting a higher realised rate on the fire footage, a bare
  // "within band" reads as a contradiction.
  const driftText =
    realised === null
      ? meta === null
        ? "calibration: connecting…"
        : `calibration not published${bandText}`
      : band === null
        ? "calibration band not published"
        : off !== null && off <= band
          ? `within ${bandPct} on the fire-free confusion set it was solved on${source ? ` (${source})` : ""}`
          : `outside ${bandPct} on the fire-free confusion set it was solved on${source ? ` (${source})` : ""}`;

  return (
    <section className="card" id="budget" style={{ padding: "14px 16px" }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
        <span className="label">alert budget</span>
        <span className="mono num" style={{ color: "var(--ink-0)", fontSize: "var(--size-lead)" }}>
          {totalSpent} / {totalSlots}
        </span>
        <span className="label" style={{ textTransform: "none", letterSpacing: 0 }}>
          slots spent this camera-window, {perCamera} per camera (budget-capped)
        </span>
      </div>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 8,
          marginTop: 12,
        }}
      >
        {cameras.length === 0 ? (
          <span className="label" style={{ textTransform: "none", letterSpacing: 0 }}>
            waiting for the camera roster — one pair of slots per camera.
          </span>
        ) : (
          cameras.map((cam) => {
            const spent = spentByCamera.get(cam) ?? 0;
            const firstNew = justFilled.get(cam);
            return (
              <span
                key={cam}
                title={`${cam}: ${spent} of ${perCamera} spent`}
                style={{ display: "flex", gap: 2 }}
              >
                {Array.from({ length: perCamera }, (_, i) => (
                  <Slot
                    key={i}
                    filled={i < spent}
                    justFilled={firstNew !== undefined && i >= firstNew && i < spent}
                  />
                ))}
              </span>
            );
          })
        )}
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          marginTop: 14,
          flexWrap: "wrap",
          fontSize: "var(--size-label)",
          color: "var(--ink-2)",
        }}
      >
        <span className="num">
          {realised === null
            ? `target ${target === null ? ABSENT : rate(target)} alerts per camera-window`
            : `${rate(realised)} alerts per camera-window on the fire-free confusion set (uncapped) / target ${
                target === null ? ABSENT : rate(target)
              }`}
        </span>
        <a href="/results#budget" className={`pill ${driftClass}`}>
          {driftText}
        </a>
      </div>
    </section>
  );
}
