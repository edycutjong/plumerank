"use client";

/**
 * The live wall's only data source: GET /api/watchlist, Server-Sent Events.
 *
 * The replay loop on the box is PROCESS-WIDE and on-demand -- it starts on the first
 * SSE connection and stops REPLAY_IDLE_STOP_SECONDS after the last one drops. That is
 * a correctness requirement rather than a convenience (t4g.small is burstable, and a
 * continuously running wall would throttle the endpoint mid-judging), and it has one
 * consequence this hook has to respect: every viewer is attached to the SAME stream,
 * so there is no session id to hold, nothing to clean up, and no per-viewer state that
 * could drift from what another judge is looking at.
 *
 * EventSource and not WebSocket: the traffic is one-directional, EventSource reconnects
 * by itself, and it survives Caddy's reverse proxy with `flush_interval -1` set. A
 * WebSocket would add an upgrade path to a public unattended box for no gain.
 */

import { useEffect, useRef, useState } from "react";
import { apiUrl, ROUTES } from "./api";
import type { StreamState, WatchMeta, WatchRow } from "./types";

/** How long a connected-but-silent stream may stay before the wall says so. The replay
 *  ticks about once a second, so 15 s of silence is not a slow frame, it is a stopped
 *  loop -- and the honest rendering of a stopped loop is the designed idle state, not a
 *  wall that keeps showing the last minute as though it were this one. */
const SILENCE_MS = 15_000;

/** Reconnect backoff for a stream the browser has given up on. EventSource retries a
 *  dropped connection by itself, but an HTTP error answer -- the 503 the box returns at
 *  its viewer cap -- closes it for good, and the page would sit on "stream down" until a
 *  reload. The hook reopens it: 1 s, doubling, capped at 30 s, reset on the next meta. */
const RECONNECT_MIN_MS = 1_000;
const RECONNECT_MAX_MS = 30_000;

export interface WatchlistState {
  meta: WatchMeta | null;
  rows: WatchRow[];
  frameIndex: number | null;
  /** The segmenter's own answer to "is the background model warm yet", when the tick
   *  carries it. `null` means the stream did not say, and the wall then falls back to
   *  framesSeen against meta.warmup_frames -- or, failing both, does not claim to know. */
  warm: boolean | null;
  /** Frames the background models have consumed, as the tick reports it (`frames_seen`),
   *  or null when that is not knowable.
   *
   *  The server sends it, with `warm`, on every tick: the segmenters are reset at every
   *  camera-night wrap (api/main.py `_reset_camera_night`), so every lap re-warms and only
   *  the server knows where in its warm-up each tick is. The local fallback below -- count
   *  from frame 0, give up after a wrap -- exists only for a stream that does not say. */
  framesSeen: number | null;
  state: StreamState;
  /** True when the rows on screen are the last ones received and the stream is not
   *  currently delivering. Rendered loudly; never allowed to be silent. */
  stale: boolean;
  error: string | null;
}

interface ParsedTick {
  rows: WatchRow[];
  frameIndex: number | null;
  warm: boolean | null;
  framesSeen: number | null;
}

function parseRows(payload: unknown): ParsedTick {
  // The API side may send the bare array WatchEntry.to_json() produces, or wrap it with
  // the frame index and the warm-up state. Both are read; nothing else is invented.
  const empty: ParsedTick = { rows: [], frameIndex: null, warm: null, framesSeen: null };
  if (Array.isArray(payload)) return { ...empty, rows: payload as WatchRow[] };
  if (typeof payload === "object" && payload !== null) {
    const doc = payload as Record<string, unknown>;
    const candidate = doc.rows ?? doc.entries ?? doc.watchlist ?? doc.watch_list;
    const idx = doc.frame_index ?? doc.minute ?? doc.tick;
    return {
      rows: Array.isArray(candidate) ? (candidate as WatchRow[]) : [],
      frameIndex: typeof idx === "number" ? idx : null,
      warm: typeof doc.warm === "boolean" ? doc.warm : null,
      framesSeen: typeof doc.frames_seen === "number" ? doc.frames_seen : null,
    };
  }
  return empty;
}

export function useWatchlist(): WatchlistState {
  const [meta, setMeta] = useState<WatchMeta | null>(null);
  const [rows, setRows] = useState<WatchRow[]>([]);
  const [frameIndex, setFrameIndex] = useState<number | null>(null);
  const [warm, setWarm] = useState<boolean | null>(null);
  const [framesSeen, setFramesSeen] = useState<number | null>(null);
  const [state, setState] = useState<StreamState>("connecting");
  const [error, setError] = useState<string | null>(null);
  const lastTick = useRef<number>(0);
  // Warm-up bookkeeping. `joinedAtStart` is false when the first frame this client saw
  // was not index 0 -- it attached to a loop already running, whose models are warm.
  // `wrapped` latches on the first frame index that goes backwards, which is one
  // completed lap of the window and therefore the end of any honest warm-up claim.
  const joinedAtStart = useRef<boolean | null>(null);
  const wrapped = useRef<boolean>(false);
  const previousIndex = useRef<number | null>(null);

  useEffect(() => {
    let source: EventSource | null = null;
    let retryTimer: number | null = null;
    let backoff = RECONNECT_MIN_MS;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      source = new EventSource(apiUrl(ROUTES.watchlist));
      const current = source;

      current.addEventListener("meta", (ev) => {
        try {
          const parsed = JSON.parse((ev as MessageEvent<string>).data) as WatchMeta;
          setMeta(parsed);
          setError(null);
          // The stream answered, so "connecting" is over either way. A meta carrying the
          // degraded flag will never be followed by a tick: that is its own state, and the
          // top bar must not keep saying "connecting" under a banner that says why not.
          if (parsed.live_pipeline?.available === false) setState("unavailable");
          else setState((prev) => (prev === "connecting" || prev === "down" || prev === "unavailable" ? "connected" : prev));
          // The silence clock starts at `meta`, not at the first tick. A stream that
          // hands over a camera roster and then never ticks is exactly the stopped-loop
          // case the idle state was designed for, and starting the clock at the first
          // tick would make that the one shape of silence the page cannot report.
          if (lastTick.current === 0) lastTick.current = Date.now();
          backoff = RECONNECT_MIN_MS;
        } catch (e) {
          // A meta event the page cannot parse is a broken contract, and it is shown as
          // one. Guessing a camera roster would put 62 invented tiles on a judge's screen.
          setError(`meta event was not JSON: ${String(e)}`);
          setState("down");
        }
      });

      const onTick = (ev: Event) => {
        try {
          const tick = parseRows(JSON.parse((ev as MessageEvent<string>).data));
          setRows(tick.rows);
          if (tick.frameIndex !== null) setFrameIndex(tick.frameIndex);
          setWarm(tick.warm);
          if (tick.frameIndex !== null) {
            if (joinedAtStart.current === null) joinedAtStart.current = tick.frameIndex === 0;
            if (previousIndex.current !== null && tick.frameIndex < previousIndex.current) {
              wrapped.current = true;
            }
            previousIndex.current = tick.frameIndex;
          }
          const countable = joinedAtStart.current === true && !wrapped.current;
          setFramesSeen(
            tick.framesSeen ?? (countable ? previousIndex.current : null),
          );
          lastTick.current = Date.now();
          setState("live");
          setError(null);
        } catch (e) {
          setError(`tick event was not JSON: ${String(e)}`);
          setState("down");
        }
      };
      current.addEventListener("tick", onTick);
      // A server that names nothing sends `message`. Accepted so a working stream is
      // never rendered as a dead one over an event-name disagreement.
      current.addEventListener("message", onTick);

      current.addEventListener("error", () => {
        // EventSource retries a dropped connection on its own, so this is not fatal -- but
        // it is also not silent. The page says the stream is down and keeps saying so until
        // a tick arrives, because "showing the last minute as though it were this one" is
        // the exact failure the degraded state exists to prevent.
        setState("down");
        setError(`no data from ${ROUTES.watchlist}`);
        // CLOSED means the browser will not retry (an HTTP error answer, e.g. the 503 at
        // the viewer cap). Reopen it ourselves, with backoff.
        if (current.readyState === EventSource.CLOSED && !disposed && retryTimer === null) {
          current.close();
          const wait = backoff;
          backoff = Math.min(RECONNECT_MAX_MS, backoff * 2);
          retryTimer = window.setTimeout(() => {
            retryTimer = null;
            connect();
          }, wait);
        }
      });
    };

    connect();

    const silence = window.setInterval(() => {
      if (lastTick.current === 0) return;
      // An unavailable pipeline never ticks by design; silence there is not a stopped loop.
      if (Date.now() - lastTick.current > SILENCE_MS) {
        setState((prev) => (prev === "unavailable" || prev === "down" ? prev : "idle"));
      }
    }, 2_000);

    return () => {
      disposed = true;
      window.clearInterval(silence);
      if (retryTimer !== null) window.clearTimeout(retryTimer);
      source?.close();
    };
  }, []);

  return {
    meta,
    rows,
    frameIndex,
    warm,
    framesSeen,
    state,
    stale: state !== "live" && rows.length > 0,
    error,
  };
}
