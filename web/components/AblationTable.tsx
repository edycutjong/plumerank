"use client";

import type { Benchmark } from "@/lib/types";

/**
 * {MOG2, LSBP, GSOC} x {S4 on/off} x {escalation on/off} on the DEV arm, as a plain
 * table.
 *
 * A plain table on purpose: this is what the freed hours bought, it sits inside the
 * criterion that carries 30% and breaks ties, and a reader comparing twelve cells wants
 * to read twelve numbers rather than decode twelve bars. The columns are whatever the
 * results file publishes -- the frontend does not know the grid's shape and has no
 * business asserting one.
 */
export function AblationTable({ data }: { data: Benchmark }) {
  const grid = data.ablation;
  if (!grid || grid.rows.length === 0) {
    return (
      <p style={{ color: "var(--ink-3)", fontSize: "var(--size-label)" }}>
        ablation grid not published yet — it is a separate twelve-cell run on the
        development set, quoted here by the held-out file once both exist
      </p>
    );
  }

  // Columns that hold one value on every row (commit, budget, seed, n...) are stated
  // once above the table instead of repeated twelve times: at 1440 px they pushed the
  // columns that differ -- the design cells and their outcomes -- behind a scrollbar.
  const constant = grid.columns
    .map((c, j) => ({ c, j, v: grid.rows[0]?.[j] }))
    .filter(({ j, v }) => grid.rows.every((r) => r[j] === v));
  const constantIdx = new Set(constant.map(({ j }) => j));
  const varying = grid.columns
    .map((c, j) => ({ c, j }))
    .filter(({ j }) => !constantIdx.has(j))
    .sort((a, b) => order(a.c) - order(b.c));

  return (
    <>
      {constant.length > 0 ? (
        <p className="mono" style={{ color: "var(--ink-2)", fontSize: "var(--size-label)", margin: "0 0 8px", overflowWrap: "anywhere" }}>
          same on every row:{" "}
          {constant.map(({ c, v }) => `${label(c)} ${c === "commit" ? String(v).slice(0, 7) : v}`).join(" · ")}
        </p>
      ) : null}
      <div className="scroll-x card">
        <table>
          <thead>
            <tr>
              {varying.map(({ c }) => (
                <th key={c}>{label(c)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {grid.rows.map((row, i) => (
              <tr key={i}>
                {varying.map(({ j }, k) => (
                  <td key={j} className={k === 0 ? undefined : "mono"} style={k === 0 ? undefined : { fontWeight: 400 }}>
                    {row[j]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

/** CSV column names, readable. The code's unit is `camera_night`; every page calls it a
 *  camera-window (81 minutes on one camera), so the header does too. */
function label(column: string): string {
  return column.replace(/camera_night/g, "camera-window").replace(/_/g, " ");
}

/** The design cells first, then what they produced; anything unknown keeps file order after. */
const PREFERRED = ["bg", "s4", "escalation", "n_detected", "n_missed", "p50_minutes", "p95_minutes"];
function order(column: string): number {
  const i = PREFERRED.indexOf(column);
  return i === -1 ? PREFERRED.length : i;
}
