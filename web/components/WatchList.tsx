"use client";

import { apiUrl } from "@/lib/api";
import { useFlip, usePrefersReducedMotion } from "@/lib/motion";
import type { StreamState, WatchMeta, WatchRow } from "@/lib/types";

/** Rows that load the per-tick evidence crop; the rest show the camera's thumbnail. */
const EVIDENCE_ROWS = 8;

/**
 * The hero, and the reason the rank digit is the largest element on the screen.
 *
 * A dashboard answers "how is the system doing?". A watch list answers "what do I look
 * at next?", so the row's first column is a rank and not a status light, and the row's
 * second column is the evidence crop -- inline, always, at every breakpoint. "A human
 * can overrule in one glance" is the product's second safeguard, and a safeguard that
 * requires a click is a safeguard nobody uses at 04:50.
 *
 * The full roster is rendered. A camera whose budget is spent still appears, with the
 * badge reading `budget spent`: the system reorders, it never suppresses, and a list
 * that dropped the row would be showing a different mechanism from the one that runs.
 */

/** A row is a confusion-set row when the server says so on the row, or when the
 *  manifest the wall was built from (meta.sequences[].arm) says its window is a CONFUSE
 *  night. Until 2026-10-04 only the row fields were read, the API never sends them, and
 *  the --fog styling never fired. The manifest arm is a label, not a statistic -- the
 *  frontend still never derives a class from S1-S4. */
function isFog(row: WatchRow, confusionCameras: ReadonlySet<string>): boolean {
  if (row.fog === true) return true;
  const cls = row.confusion_class;
  if (typeof cls === "string" && cls !== "" && cls !== "fire" && cls !== "smoke") return true;
  return confusionCameras.has(row.camera_id);
}

/** Cameras whose wall window comes from the CONFUSE manifest. */
export function confusionCamerasOf(meta: WatchMeta | null): ReadonlySet<string> {
  return new Set(
    (meta?.sequences ?? []).filter((s) => s.arm === "confuse").map((s) => s.camera_id),
  );
}

function rankClass(rank: number, topK: number | null): string {
  if (rank === 1) return "rank-digit rank-digit--one";
  if (topK !== null && rank <= topK) return "rank-digit rank-digit--top";
  return "rank-digit";
}

export function WatchList({
  rows,
  topK,
  perCamera,
  threshold,
  stream,
  confusionCameras,
  onSelect,
}: {
  rows: WatchRow[];
  topK: number | null;
  perCamera: number;
  /** The solved hazard threshold from the `meta` event. Null when the stream did not
   *  publish one, and the list then makes no claim about what is above it. */
  threshold: number | null;
  stream: StreamState;
  /** From confusionCamerasOf(meta). */
  confusionCameras: ReadonlySet<string>;
  onSelect: (row: WatchRow) => void;
}) {
  const reduced = usePrefersReducedMotion();
  const container = useFlip(rows.map((r) => r.camera_id), reduced);

  // "Nothing above the alert threshold" is a statement about the THRESHOLD, so it is
  // computed against the threshold. `alerted` is budget-capped -- a camera over the line
  // whose two slots are already spent carries alerted=false -- so a quiet-night caption
  // keyed off it would read "nothing above the alert threshold" on the loudest night the
  // wall can have, which is the one night the sentence must not be printed.
  // Both halves, because the two disagree at the fourth decimal: WatchEntry.to_json
  // rounds hazard to 4 dp while the threshold arrives at full precision, so a row the
  // server alerted on can compare below the line here. Requiring BOTH "nothing over the
  // threshold" and "nothing alerted" is what stops the list printing the quiet-night
  // caption directly above a row wearing an `alert` badge.
  const quiet =
    threshold !== null &&
    !rows.some((r) => r.hazard >= threshold) &&
    !rows.some((r) => r.alerted);

  return (
    <section className="card" style={{ overflow: "hidden" }}>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 10,
          padding: "12px 14px",
          borderBottom: "1px solid var(--hairline)",
        }}
      >
        <span className="label">watch list</span>
        <span className="label" style={{ textTransform: "none", letterSpacing: 0 }}>
          {rows.length === 0 ? "none removed" : `${rows.length} camera tiles, none removed`}
        </span>
      </div>
      {/* What the rank IS, said once above the rows: the score each row carries is how
          far the camera has departed from its own last 15 minutes, not a raw signal --
          a camera that is hazy all morning is not a fire. */}
      <p className="watchlist-caption">
        Ranked by departure from each camera&apos;s own recent history (its last 15
        minutes), not by raw signal.{threshold !== null ? ` A score at or above ${threshold.toFixed(2)} fires an alert.` : ""}
      </p>
      <p className="watchlist-legend">
        <span className="badge badge--alert">alert</span> fired this frame{" · "}
        <span className="badge badge--spent">budget spent</span> both slots used, still ranked{" · "}
        <span className="badge badge--signal">escalated</span> checked by the tile classifier{" · "}
        <span className="badge" style={{ color: "var(--fog)" }}>confusion set</span> a fire-free window
      </p>

      {rows.length === 0 ? (
        <p style={{ color: "var(--ink-2)", padding: "16px 14px", margin: 0 }}>
          {stream === "down"
            ? "the watch list is empty because the live stream is not delivering."
            : stream === "unavailable"
              ? "no ranking: the live pipeline is unavailable on this box."
              : "waiting for the first ranking."}
        </p>
      ) : (
        <>
          {/* Designed state: a quiet night is the normal case and must look normal.
              Not an empty-state illustration, not a green tick -- one line of --ink-2
              above a watch list that is still fully ranked. */}
          {quiet ? (
            <p style={{ color: "var(--ink-2)", padding: "10px 14px", margin: 0, fontSize: "var(--size-label)" }}>
              nothing above the alert threshold
            </p>
          ) : null}

          <div ref={container} className="watchlist-scroll">
            {rows.map((row) => {
              const fog = isFog(row, confusionCameras);
              const spent = row.budget_spent >= perCamera;
              // Evidence crops change URL every tick, so loading one per row made each viewer
              // fetch ~37 images a second. Only the rows a dispatcher reads first get the crop;
              // the rest show the thumbnail, whose URL changes only when it is rewritten.
              const thumb =
                (row.rank <= EVIDENCE_ROWS ? row.evidence_url : null) ?? row.frame_url ?? null;
              return (
                <div
                  key={row.camera_id}
                  data-flip-key={row.camera_id}
                  className={[
                    "row",
                    row.rank === 1 ? "row--one" : "",
                    row.escalated ? "row--escalating" : "",
                    fog ? "row--fog" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                >
                  <span className={`row-rank ${rankClass(row.rank, topK)}`}>{row.rank}</span>

                  <button
                    type="button"
                    className="row-thumb"
                    onClick={() => onSelect(row)}
                    style={{ padding: 0, background: "none", border: 0, cursor: "pointer" }}
                    aria-label={`open the evidence crop for ${row.camera_id}`}
                  >
                    {thumb ? (
                      <img
                        className="evidence-thumb"
                        src={apiUrl(thumb)}
                        alt=""
                        loading="lazy"
                        decoding="async"
                      />
                    ) : (
                      <span
                        className="evidence-thumb"
                        style={{
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 10,
                          color: "var(--ink-3)",
                          borderColor: "var(--hairline)",
                        }}
                      >
                        no crop
                      </span>
                    )}
                  </button>

                  <span className="row-main">
                    <span style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
                      <span className="row-name" title={row.camera_id}>{row.camera_id}</span>
                      {row.county ? <span className="row-meta">{row.county}</span> : null}
                      {row.alerted ? (
                        <span className="badge badge--alert">alert</span>
                      ) : spent ? (
                        <span className="badge badge--spent">budget spent</span>
                      ) : null}
                      {row.escalated ? <span className="badge badge--signal">escalated</span> : null}
                      {fog ? (
                        <span className="badge" style={{ color: "var(--fog)" }}>
                          {row.confusion_class ?? "confusion set, no fire"}
                        </span>
                      ) : null}
                    </span>
                    <span className="row-score">
                      <span className="mono num">score {row.hazard.toFixed(2)}</span>
                      <span>vs own last 15 min</span>
                    </span>
                    <span className="hazard-bar" style={{ position: "relative" }}>
                      <i style={{ width: `${Math.max(0, Math.min(1, row.hazard)) * 100}%` }} />
                      {threshold !== null ? (
                        <b
                          aria-hidden="true"
                          className="hazard-threshold"
                          style={{ left: `${Math.max(0, Math.min(1, threshold)) * 100}%` }}
                        />
                      ) : null}
                    </span>
                  </span>

                  {/* Colour is never the only channel: every statistic is a written name
                      and a number, so a red/green-blind dispatcher loses nothing. */}
                  <span
                    className="row-stats"
                    title="The raw plume statistics behind the score. Each is weak on its own (AUC 0.53, 0.50, 0.44 on the development set); the rank comes from the fused, per-camera score."
                  >
                    <span className="stat-chip">S1 anchor {row.s1.toFixed(2)}</span>
                    <span className="stat-chip">S2 aniso {row.s2.toFixed(2)}</span>
                    <span className="stat-chip">S3 diffuse {row.s3.toFixed(2)}</span>
                  </span>
                </div>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}
