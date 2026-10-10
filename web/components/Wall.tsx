"use client";

import { memo, useEffect, useState } from "react";

import { apiUrl } from "@/lib/api";
import type { StreamState, WatchMeta, WatchRow } from "@/lib/types";

/**
 * The full camera roster, 8 columns, and it never gets shorter.
 *
 * Low-ranked tiles are dimmed to 45%. They are NOT removed, and that is the whole
 * argument the wall exists to make: PlumeRank reorders attention, it does not suppress
 * a camera. A wall that dropped the quiet 36 would be showing a different system from
 * the one that runs, and the one safeguard a fire agency will ask about first is
 * exactly the one that removal would visually contradict.
 *
 * Below 1280 px the same tiles become a horizontally scrollable strip of the top 12;
 * below 768 px the wall is hidden by one CSS rule and the watch list IS the product.
 */

/** How many tiles the sub-1280 strip carries. A layout number, not a measurement. */
const STRIP_TILES = 12;

/**
 * A tile image that never goes blank between frames.
 *
 * Every tick rewrites ~8 thumbnails under a new ``?t=`` URL. A plain <img> given a new src
 * paints nothing until the new JPEG has decoded, so a quarter of the time some tiles were
 * blank (measured: 48 of 200 samples at 100 ms, up to 8 tiles). The new frame is decoded
 * off-screen first and swapped in only once it can paint; until then the previous frame
 * stays. A frame that fails to load leaves the previous one up rather than a hole.
 */
function TileImage({ src }: { src: string }) {
  const [shown, setShown] = useState(src);
  useEffect(() => {
    if (src === shown) return;
    let live = true;
    const next = new Image();
    next.src = src;
    next
      .decode()
      .then(() => {
        if (live) setShown(src);
      })
      .catch(() => {
        /* keep the previous frame */
      });
    return () => {
      live = false;
    };
  }, [src, shown]);
  return <img src={shown} alt="" decoding="async" />;
}

/**
 * One wall tile. Memoised on primitives: a tick re-ranks every row, but most tiles keep
 * their frame URL, rank and dimming, and re-rendering all 62 on every tick was most of
 * the wall's main-thread time on a throttled phone.
 */
const Tile = memo(function Tile({
  cam,
  src,
  rank,
  dim,
  warmWidth,
}: {
  cam: string;
  src: string | null;
  rank: number | null;
  dim: boolean;
  warmWidth: number | null;
}) {
  return (
    <div className={`tile${dim ? " tile--dim" : ""}`} title={cam}>
      {src ? <TileImage src={src} /> : <span className="tile-empty">no frame published</span>}
      {rank !== null ? <span className="tile-rank">{rank}</span> : null}
      <span className="tile-name">{cam}</span>
      {warmWidth !== null ? <span className="tile-warm" style={{ width: `${warmWidth * 100}%` }} /> : null}
    </div>
  );
});

export function Wall({
  meta,
  rows,
  topK,
  stream,
  warm,
  framesSeen,
  compact,
  holdSequences,
  holdExcluded,
}: {
  meta: WatchMeta | null;
  rows: WatchRow[];
  topK: number | null;
  stream: StreamState;
  warm: boolean | null;
  framesSeen: number | null;
  /** true for the < 1280 px strip. Set by the page, not sniffed here: the page owns
   *  the layout and a component that measured the viewport itself would be a second
   *  source of truth for the same breakpoint. */
  compact: boolean;
  /** How many held-out sequences the published headline was measured on, straight from
   *  /api/benchmark. The provenance caption names it rather than carrying a literal:
   *  the wall's whole job here is to say the demo is NOT the measurement, and a number
   *  typed into the caption could disagree with the one on /results. */
  holdSequences: number | null;
  /** How many the exclusion list dropped, from the same document. It is a SECOND prop
   *  and not a nicety: this caption is the line the demo video says out loud, and the
   *  surviving count on its own reads as sequences that went missing, while the manifest
   *  count on its own overstates a denominator the exclusion list has already reduced.
   *  Both or neither -- with neither, the caption drops the arithmetic rather than
   *  publishing half of it. */
  holdExcluded: number | null;
}) {
  const byCamera = new Map(rows.map((r) => [r.camera_id, r]));
  const confuseCount = (meta?.sequences ?? []).filter((s) => s.arm === "confuse").length;
  // The measured tick period when the server publishes one; the configured one otherwise.
  const measured = meta?.tick_seconds_measured;
  const configured = meta?.tick_seconds;
  const period =
    typeof measured === "number" && measured > 0
      ? measured
      : typeof configured === "number" && configured > 0
        ? configured
        : null;
  const tickText =
    period === null ? "replayed faster than real time" : `about one frame every ${period.toFixed(1)} s`;
  const roster = meta?.cameras ?? rows.map((r) => r.camera_id);

  const ordered = compact
    ? [...roster]
        .sort((a, b) => (byCamera.get(a)?.rank ?? 1e9) - (byCamera.get(b)?.rank ?? 1e9))
        .slice(0, STRIP_TILES)
    : roster;

  const warmupFrames = meta?.warmup_frames ?? null;
  // The server's own `warm` first: the background models are reset at every camera-night
  // wrap, so every lap re-warms and the tick says so. framesSeen against warmup_frames is
  // the fallback for a stream that does not; useWatchlist publishes null rather than a
  // count whenever it cannot honestly claim the loop started while this client watched.
  const seen = framesSeen;
  const warming =
    warm === false ||
    (warm === null && warmupFrames !== null && seen !== null && seen < warmupFrames);
  const warmProgress =
    warmupFrames !== null && warmupFrames > 0 && seen !== null
      ? Math.min(1, Math.max(0, seen / warmupFrames))
      : 0;

  // The three ways the wall can have nothing to draw are three DIFFERENT statements and
  // the page is not allowed to collapse them: the meta event has not arrived yet; it
  // arrived carrying api/main.py's own "live pipeline unavailable" flag; or the loop is
  // simply between ticks. Only the last one is "replay paused".
  const live = meta?.live_pipeline ?? null;
  const pipelineDown = live !== null && live.available === false;
  const paused = stream === "idle" && !pipelineDown;

  return (
    <section>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 10,
          marginBottom: 8,
          flexWrap: "wrap",
        }}
      >
        <span className="label">the wall</span>
        <span className="label" style={{ textTransform: "none", letterSpacing: 0 }}>
          {roster.length === 0
            ? "every camera on the roster, none removed"
            : compact
              ? `top ${ordered.length} of ${roster.length} camera tiles by rank — scroll sideways for more; none removed`
              : `all ${roster.length} camera tiles, none removed`}
        </span>
      </div>

      {roster.length === 0 ? (
        <p style={{ color: "var(--ink-2)", margin: 0 }}>
          {pipelineDown
            ? "No live ranking on this box."
            : meta === null
              ? "connecting to the live stream…"
              : "the stream arrived carrying no cameras."}
        </p>
      ) : (
        <div
          className={compact ? "wall-strip" : "wall-grid"}
          style={paused ? { opacity: 0.45 } : undefined}
        >
          {ordered.map((cam) => {
            const row = byCamera.get(cam);
            const src = row?.frame_url ?? row?.evidence_url ?? null;
            // topK null means the stream did not publish the cutoff, and the wall then
            // dims NOTHING rather than dimming by a guessed one. See the note below.
            const dim = topK !== null && (row === undefined || row.rank > topK);
            return (
              <Tile
                key={cam}
                cam={cam}
                src={src ? apiUrl(src) : null}
                rank={row ? row.rank : null}
                dim={dim}
                warmWidth={warming ? warmProgress : null}
              />
            );
          })}
        </div>
      )}

      {/* No second degraded box here: the empty-roster paragraph above already carries
          live_pipeline.detail, and the page renders the designed state-7 strip. Saying
          the same sentence three times is how a designed state turns into noise. */}
      {paused ? (
        <p style={{ color: "var(--ink-2)", fontSize: "var(--size-label)", marginTop: 8 }}>
          replay paused. The replay runs only while someone is watching and restarts on
          its own — reload the page if it does not resume within a few seconds.
        </p>
      ) : null}

      {warming ? (
        <p style={{ color: "var(--ink-2)", fontSize: "var(--size-label)", marginTop: 8 }}>
          warming background model — this is what a real camera start looks like
          {warmupFrames !== null && seen !== null ? ` (${seen} of ${warmupFrames} frames)` : ""}
        </p>
      ) : null}

      {topK === null && roster.length > 0 ? (
        <p style={{ color: "var(--ink-3)", fontSize: "var(--size-label)", marginTop: 8 }}>
          the stream did not publish its watch-list cutoff, so no tile is dimmed.
        </p>
      ) : null}

      {/* Provenance. The wall runs the DEV arm and says so, unprompted, right under the
          thing it is showing -- because the strongest attack on this project is
          "you demoed on the fire you tuned on", and answering it before it is asked is
          cheaper than answering it afterwards. The wall never reads HOLD. */}
      <p style={{ color: "var(--ink-3)", fontSize: "var(--size-label)", marginTop: 10 }}>
        Replaying the development fires (the constants were fit on them)
        {confuseCount > 0 ? ` and ${confuseCount} fire-free confusion-set windows` : ""} — one
        frame per minute of footage, {tickText} on screen. The headline {holdSequences === null ? "will be" : "is"} measured on{" "}
        {holdSequences === null || holdExcluded === null
          ? "the"
          : `${holdSequences} of the ${holdSequences + holdExcluded}`}{" "}
        held-out sequences the system has never seen
        {holdExcluded === null
          ? ""
          : ` — ${holdExcluded} excluded by the pre-stated criteria in data/excluded.json, listed with their reasons on /results`}
        .
      </p>
    </section>
  );
}
