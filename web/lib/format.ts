/**
 * Formatting. Every function takes `number | null` and renders the null as an em dash
 * rather than as 0, "-" or "n/a": a missing measurement and a measurement of zero are
 * different claims, and this product is scored on not confusing them. (It was a literal
 * "--" until 2026-10-04, which read on screen as a rendering bug, not as "not yet".)
 */

/** The one string that means "the file did not publish this". */
export const ABSENT = "\u2014";

export function minutes(v: number | null, digits = 1): string {
  return v === null ? ABSENT : v.toFixed(digits);
}

export function rate(v: number | null, digits = 2): string {
  return v === null ? ABSENT : v.toFixed(digits);
}

/** Model calls: "3,645" and "0", as every document writes them; 1.4 keeps its decimal. */
export function calls(v: number | null): string {
  if (v === null) return ABSENT;
  if (v === 0) return "0";
  return v >= 100 ? Math.round(v).toLocaleString("en-US") : v.toFixed(1);
}

export function percent(v: number | null, digits = 1): string {
  return v === null ? ABSENT : `${(v * 100).toFixed(digits)}%`;
}

export function count(v: number | null): string {
  return v === null ? ABSENT : String(Math.round(v));
}

export function sha(v: string | null, chars = 12): string {
  return v === null ? ABSENT : v.slice(0, chars);
}

/** `T+MM:SS`, and negative offsets stay signed. A replay's pre-ignition frames are
 *  most of the window, so `T-12:00` has to be readable rather than clamped to zero --
 *  clamping would quietly turn 40 minutes of legitimate pre-ignition footage into a
 *  wall of `T+00:00` and hide the one thing the clock exists to show. */
export function signedClock(totalSeconds: number | null): string {
  if (totalSeconds === null || !Number.isFinite(totalSeconds)) return `T+${ABSENT}:${ABSENT}`;
  const sign = totalSeconds < 0 ? "-" : "+";
  const abs = Math.abs(Math.round(totalSeconds));
  const mm = String(Math.floor(abs / 60)).padStart(2, "0");
  const ss = String(abs % 60).padStart(2, "0");
  return `T${sign}${mm}:${ss}`;
}

/** Simulated wall time as HH:MM:SS from an epoch second, in UTC. UTC and not the
 *  viewer's zone: the replay's clock belongs to the camera, and rendering a Californian
 *  ignition in a judge's local timezone would silently move the fire in time. */
export function wallClock(epochSeconds: number | null): string {
  if (epochSeconds === null || !Number.isFinite(epochSeconds)) return `${ABSENT}:${ABSENT}:${ABSENT}`;
  const d = new Date(epochSeconds * 1000);
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  const ss = String(d.getUTCSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

export function duration(seconds: number | null): string {
  if (seconds === null) return ABSENT;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${String(s).padStart(2, "0")}s`;
}
