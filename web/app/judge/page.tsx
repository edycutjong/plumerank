import Link from "next/link";
import { TopBar } from "@/components/TopBar";
import { DEV_RUN_FILE, DEV_SUMMARY } from "@/lib/devRun";
import { count, minutes } from "@/lib/format";

/**
 * "/judge" -- built for exactly one reader. No auth, no cookies, no rate limit, no
 * redirect, and no dependency on the live pipeline: this page is static (a server
 * component, no "use client", no data fetching) so it renders even when the wall
 * itself is reporting a degraded pipeline. JUDGE.md at the repo root is the same
 * content; keep the two in sync by hand -- scripts/check_claims.py checks both for
 * route/dependency/path consistency but does not diff them against each other.
 */

const CLAIM =
  "PlumeRank ranks which wildfire-camera tiles a dispatcher should open next, " +
  "calibrated to a fixed budget of 2 alerts per camera\u2011night (one 81-minute camera " +
  "window, the length of a FIgLib sequence), and measures itself on median minutes after " +
  "ignition until the right camera reaches the top of that ranking.";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={{ marginTop: 28 }}>
      <h2 style={{ fontSize: "var(--size-lead)", color: "var(--ink-0)", margin: "0 0 10px" }}>
        {title}
      </h2>
      {children}
    </section>
  );
}

export default function JudgePage() {
  return (
    <>
    <TopBar />
    <main style={{ padding: "20px 16px 48px", maxWidth: 820, margin: "0 auto" }}>
      <p className="label" style={{ margin: "0 0 6px" }}>
        for judges
      </p>
      <h1 className="display" style={{ fontSize: "var(--size-metric-s)", margin: "0 0 14px" }}>
        The 30&#8209;second path
      </h1>
      <p style={{ color: "var(--ink-1)", fontSize: "var(--size-lead)", margin: 0 }}>{CLAIM}</p>

      <Section title="1. Click these, in order">
        <ol className="steps" style={{ color: "var(--ink-1)", lineHeight: 1.7 }}>
          <li>
            <Link href="/" style={{ textDecoration: "underline" }}>
              The Wall
            </Link>{" "}
            — the live ranking, no login, replaying development fires and fire-free
            confusion-set windows. Its threshold is labelled on screen.
          </li>
          <li>
            <a href="/build-info" style={{ textDecoration: "underline" }}>
              /build-info
            </a>{" "}
            — real <code className="code-inline">cv2.getBuildInformation()</code> output, the
            instance id, the AMI id and the git commit this box is running, read live off the box.
          </li>
          <li>
            <Link href="/results" style={{ textDecoration: "underline" }}>
              /results
            </Link>{" "}
            — the build-info block in-page and the held-out headline: systems on one clock,
            operating curve, sensitivity strip and the ablation grid, from the committed run.
          </li>
        </ol>
      </Section>

      <Section title="2. The receipt">
        <ul className="bullets" style={{ color: "var(--ink-1)", lineHeight: 1.8 }}>
          <li>
            <strong>832 passed, 1 skipped locally (CI, without the corpus, runs 820 and skips 13)</strong>, 100% line coverage on <code className="code-inline">plumerank/</code>
          </li>
          <li>
            <strong>Property-based tests</strong> (<code className="code-inline">tests/test_invariants.py</code>,
            hypothesis, 1,500 generated cases) checking that the fused hazard stays in{" "}
            <code className="code-inline">[0, 1]</code> and the alert budget is never overspent
          </li>
          <li>
            <code className="code-inline">ruff check .</code> and <code className="code-inline">mypy</code> both clean;{" "}
            <code className="code-inline">scripts/check_claims.py</code> (inside <code className="code-inline">make test</code>)
            mechanically checks every route, dependency, Makefile target and OpenCV symbol the
            README and the other judge-facing documents name against the real source
          </li>
          <li>
            Development set (in-sample, never the claim): {count(DEV_SUMMARY.detected)}/
            {count(DEV_SUMMARY.n)} detected, p50 {minutes(DEV_SUMMARY.p50)} min (
            <span className="mono">{DEV_RUN_FILE}</span>; rates and baselines on{" "}
            <Link href="/results">/results</Link>)
          </li>
          <li>
            <strong>Held-out set — the headline: 27/54 detected, median 38.0 min</strong> at 2.0
            alerts per camera-night; like-for-like baselines 18/54 and 20/54, neither reaching a
            median (<span className="mono">results/bench_2026-10-10.json</span>) — see §4
          </li>
          <li>
            One AWS <code className="code-inline">t4g.small</code> (Graviton) and one S3 bucket. No
            Lambda, no managed vision API.
          </li>
        </ul>
      </Section>

      <Section title="3. Reproduce it yourself">
        <pre className="code" style={{ padding: 12, overflowX: "auto", whiteSpace: "pre" }}>
{`python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.lock
make test      # ~2 min: 832-test offline suite + coverage + the claim-consistency check
make verify    # ≤ 3 min: rebuilds the venv, re-runs the suite, recomputes the
               # micro-corpus metrics against the committed results. Does NOT
               # reproduce the published headline, and says so on screen.`}
        </pre>
        <p style={{ color: "var(--ink-2)", fontSize: "var(--size-label)" }}>
          The pair that reproduces the held-out headline itself:{" "}
          <code className="code-inline">make fetch</code> then <code className="code-inline">make bench</code> — about
          35 minutes of fetch, then the evaluation (12m29s on the machine that produced the
          published file). Nobody needs to run that to evaluate this submission.
        </p>
      </Section>

      <Section title="4. How the headline was produced — and how far to trust it">
        <p style={{ color: "var(--ink-1)" }}>
          The held-out run happened once, on 2026-10-10: the constants were frozen and committed
          first (<code className="code-inline">results/constants.sha256</code>), and the single read
          is logged with its commit (<code className="code-inline">results/hold_access.log</code>). 54
          sequences were measured — 60 in the manifest, 6 excluded by the published rule.
        </p>
        <p style={{ color: "var(--ink-1)" }}>
          27 of 54 is exactly half, so the median is on a knife edge: against an ignition clock
          shifted by −5 to 0 minutes it reads 32–38 minutes, and from +1 to +5 minutes it does not
          exist. The detection counts — 27 against 18 and 20 — are the steadier comparison.
        </p>
      </Section>

      <Section title="5. Honest limitations">
        <ul className="bullets" style={{ color: "var(--ink-1)", lineHeight: 1.8 }}>
          <li>Daylight only — no IR hardware, no night detection.</li>
          <li>
            Cloud, not fog, produced the most false alarms on the confusion set (measured
            before the per-camera baseline was adopted). A blind census found the &ldquo;fog&rdquo; class
            picks a marine layer no more often than chance (29% vs 29%), so it is mostly
            overcast; the threshold uses every confusion window, so only per-class claims change.
          </li>
          <li>
            The soft-boundary statistic the design started from (S3) is not predictive on its
            own — AUC 0.44 on the development frames where it is defined — yet it carries the
            largest fusion weight, and no S3-off ablation has run. The measured result comes from
            the per-camera baseline and the budget-calibrated ranking, not from that one signal.
          </li>
          <li>
            Stabilisation registers each frame to its raw predecessor, and under moving cloud
            its whole-frame estimate follows the sky: checked against the ground, 48 of DEV&rsquo;s 55
            warps of 1 px or more are one genuinely shaking camera, the rest follow cloud
            (results/registration_2026-10-10.json). Replaying the fire-free set without those
            warps moves its false alarms from 64 to 63.
          </li>
          <li>
            The evidence crop is the largest moving component, not a fire localisation: at each
            DEV fire&rsquo;s first rank-1 or alert tick it contains the plume on 9 of 21, cloud on 7
            (results/evidence_first_hits_2026-10-10.json). A cloud in the crop means &ldquo;look at the
            tile&rdquo;, not &ldquo;all clear&rdquo;.
          </li>
          <li>One instance, no HA — every published number is committed in the repository.</li>
          <li>No dispatch, no suppression, no upload path — architecturally absent.</li>
        </ul>
      </Section>

      <Section title="6. Links">
        <p style={{ color: "var(--ink-1)" }}>
          <Link href="/">Wall</Link> · <Link href="/results">/results</Link> ·{" "}
          <a href="/build-info">/build-info</a> · <a href="/health">/health</a> ·{" "}
          <a href="https://github.com/edycutjong/plumerank">source on GitHub</a> ·{" "}
          <a href="https://plumerank.edycu.dev/pitch/">pitch deck</a>
        </p>
      </Section>
    </main>
    </>
  );
}
