"use client";

import { minutes, rate, ABSENT } from "@/lib/format";
import type { Benchmark } from "@/lib/types";

/**
 * Budget (x) against median time-to-detection (y), one line per system.
 *
 * HAND-ROLLED INLINE SVG. There is no chart library in this repo, and that is a
 * decision with two halves. The visible half: about forty lines of path arithmetic
 * cost less than a dependency whose bundle a t4g.small has to serve to every judge.
 * The load-bearing half: this curve is the component that REPLACED the budget slider.
 * A dispatcher at 04:50 has no spare attention for a control, so the answer at every
 * budget is precomputed and drawn, and the one operating point the project is
 * calibrated to is the only annotated point on it.
 */

const W = 660;
const H = 280;
/* top padding leaves a clear band for the y-axis label, so it never lands on the
 * topmost gridline tick -- an axis label sitting on a number is a chart that has to be
 * decoded before it can be read. */
const PAD = { top: 34, right: 18, bottom: 40, left: 54 };

const STROKES = ["var(--signal)", "var(--ink-2)", "var(--fog)"];

export function OperatingCurve({ data }: { data: Benchmark }) {
  const series = data.operating.series;
  if (series.length === 0) {
    return (
      <p style={{ color: "var(--ink-3)", fontSize: "var(--size-label)" }}>
        operating curve not published yet — it appears here after the held-out run
      </p>
    );
  }

  const pts = series.flatMap((s) => s.points);
  // Every budget the file publishes is a tick, including the ones where no system has a
  // median -- a curve that silently starts at 2.0 hides that 0.5 and 1.0 were measured.
  const budgets = data.operating.budgets.length > 0 ? data.operating.budgets : pts.map((p) => p.budget);
  const xs = budgets;
  const ys = pts.map((p) => p.p50_minutes);
  // The y axis starts at zero, always. A truncated axis is how a two-minute difference
  // is made to look like a rout, and this chart's whole job is to be believed.
  const y1 = Math.max(...ys);

  // Budgets are spaced by index, not linearly: the published grid is geometric
  // (0.5, 1, 2, 4, 8), and a linear axis crushes the low end into one pixel.
  const index = new Map(xs.map((b, i) => [b, i]));
  const sx = (v: number) =>
    PAD.left + ((index.get(v) ?? 0) / (xs.length - 1 || 1)) * (W - PAD.left - PAD.right);
  const sy = (v: number) => H - PAD.bottom - (v / (y1 || 1)) * (H - PAD.top - PAD.bottom);

  const op = data.operating.operating_point;
  const opInRange = op !== null && index.has(op);
  const headlineKey = (series.find((s) => s.key === "plumerank") ?? series[0])?.key;

  return (
    <figure style={{ margin: 0 }}>
      <div className="scroll-x scroll-x--bare">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 560, maxWidth: 760, display: "block" }} role="img"
             aria-label="median time to detection against alert budget, one line per system">
          <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom}
                stroke="var(--hairline)" />
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--hairline)" />

          {[0, 0.5, 1].map((f) => (
            <g key={f}>
              <text x={PAD.left - 8} y={sy(y1 * f) + 4} textAnchor="end"
                    fill="var(--ink-2)" fontSize="13" fontFamily="var(--font-mono)">
                {(y1 * f).toFixed(0)}
              </text>
              <line x1={PAD.left} y1={sy(y1 * f)} x2={W - PAD.right} y2={sy(y1 * f)}
                    stroke="var(--hairline)" strokeDasharray="2 5" />
            </g>
          ))}

          {xs.map((b) => (
            <g key={b}>
              <line x1={sx(b)} y1={H - PAD.bottom} x2={sx(b)} y2={H - PAD.bottom + 5} stroke="var(--ink-3)" />
              <text x={sx(b)} y={H - PAD.bottom + 18} textAnchor="middle"
                    fill={b === op ? "var(--alert)" : "var(--ink-2)"} fontSize="13" fontFamily="var(--font-mono)">
                {b % 1 === 0 ? b.toFixed(0) : b.toFixed(1)}
              </text>
            </g>
          ))}

          {opInRange && op !== null ? (
            <line x1={sx(op)} y1={PAD.top} x2={sx(op)} y2={H - PAD.bottom}
                  stroke="var(--alert)" strokeWidth="1.5" strokeDasharray="4 4" />
          ) : null}

          {series.map((s, i) => (
            <g key={s.key}>
              {s.points.length > 1 ? (
                <polyline
                  fill="none"
                  stroke={STROKES[i % STROKES.length]}
                  strokeWidth={s.key === "plumerank" ? 2.5 : 1.5}
                  points={s.points.map((p) => `${sx(p.budget)},${sy(p.p50_minutes)}`).join(" ")}
                />
              ) : null}
              {s.points.map((p) => (
                <circle key={p.budget} cx={sx(p.budget)} cy={sy(p.p50_minutes)} r={s.key === "plumerank" ? 4 : 3}
                        fill={STROKES[i % STROKES.length]} />
              ))}
              {s.key === headlineKey
                ? s.points.map((p) => (
                    <text key={`l${p.budget}`} x={sx(p.budget) + (p.budget === xs[xs.length - 1] ? -7 : 7)}
                          y={sy(p.p50_minutes) - 9}
                          textAnchor={p.budget === xs[xs.length - 1] ? "end" : "start"}
                          fill="var(--ink-1)" fontSize="13" fontFamily="var(--font-mono)">
                      {minutes(p.p50_minutes)}
                    </text>
                  ))
                : null}
              {s.key === headlineKey
                ? s.missing.map((b) => (
                    <text key={`m${b}`} x={sx(b) + (b === xs[0] ? 4 : 0)} y={H - PAD.bottom - 22}
                          textAnchor={b === xs[0] ? "start" : "middle"}
                          fill="var(--ink-3)" fontSize="12">
                      no median
                    </text>
                  ))
                : null}
            </g>
          ))}

          <text x={(W + PAD.left) / 2} y={H - 4} textAnchor="middle" fill="var(--ink-2)" fontSize="13">
            alert budget (alerts per camera-window)
          </text>
          <text x={8} y={16} fill="var(--ink-2)" fontSize="13">
            median minutes to detection
          </text>
        </svg>
      </div>
      <figcaption style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 6 }}>
        {series.map((s, i) => (
          <span key={s.key} style={{ fontSize: "var(--size-label)", color: "var(--ink-2)" }}>
            <span aria-hidden="true" style={{
              display: "inline-block", width: 14, height: 2, marginRight: 6, verticalAlign: "middle",
              background: STROKES[i % STROKES.length],
            }} />
            {s.label}
            {s.missing.length > 0 ? ` (no median at ${s.missing.map((b) => rate(b, 1)).join(", ")})` : ""}
          </span>
        ))}
        {data.systems
          .filter((sys) => !series.some((s) => s.key === sys.key))
          .map((sys) => (
            <span key={sys.key} style={{ fontSize: "var(--size-label)", color: "var(--ink-3)" }}>
              {sys.label}: no median at any published budget
            </span>
          ))}
        <span style={{ fontSize: "var(--size-label)", color: "var(--ink-3)" }}>
          {op === null
            ? `operating point ${ABSENT}`
            : opInRange
              ? `dashed line: the operating point, ${rate(op, 1)} alerts per camera-window`
              : `operating point ${rate(op, 1)} lies outside the published budgets and is not marked`}
          {" · "}budgets evenly spaced; y axis starts at 0
        </span>
      </figcaption>
    </figure>
  );
}
