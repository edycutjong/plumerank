"""The submit-time gate: twelve checks, every one of them verifiable from outside.

A readiness checklist made of sentences ("the demo video is recorded", "the numbers are
consistent") is a checklist that passes at 23:30 on the last day whatever the truth is,
because the person ticking it is the person who wrote it. So there are no sentences here.
Every check below either fetches something over HTTPS, compares two files, or greps a tree
-- and each one names the exact artefact it read, so a failure says what to go and fix
rather than that something is wrong.

The twelve
----------
1.  ``live-url``               the deployed site answers 200
2.  ``build-info``             /build-info names OpenCV, a 5.x version, and this HEAD sha
3.  ``benchmark-matches``      /api/benchmark agrees with results/ on the headline
4.  ``reproduce-honest``       no MOCK / OFFLINE / --dry-run token on the reproduce path
5.  ``check-claims``           scripts/check_claims.py exits zero
6.  ``no-placeholders``        zero placeholder tokens anywhere in the submission text
7.  ``no-aws-key``             no access-key id anywhere in the tree
8.  ``no-stale-cost``          no superseded cost figure anywhere in the tree
9.  ``video-url``              the demo video URL answers 200
10. ``special-awards``         both optional-award boxes recorded, and both unticked
11. ``reproduce-both``         BOTH labelled commands, and no paragraph conflates them
12. ``eligibility-clause``     the eligibility text is pasted, with its retrieval date

Skipping is not passing
-----------------------
The live URL and the video URL do not exist until late. A check that cannot run reports
``SKIP`` loudly and the run exits **2 -- INCOMPLETE**, which is not zero: no wrapper, CI
job or human skimming the last line can read a skipped check as a green one. Only a run
where all twelve passed exits 0.

Exit codes: ``0`` ready, ``1`` a check failed, ``2`` a check could not run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

__all__ = ["Check", "Config", "Report", "run", "main", "PASS", "FAIL", "SKIP"]

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

#: Tokens are assembled from pieces rather than written out, because every one of them is
#: something this script greps the tree for -- and a scanner whose own source matches its
#: own pattern is a scanner that can never report a clean tree.
AWS_KEY_PREFIX = "AK" + "IA"
PLACEHOLDER_TOKEN = "NOT" + " " + "YET"
STALE_COSTS = ("$" + "15", "$" + "25")
MOCK_TOKENS = ("M" + "OCK", "OFF" + "LINE", "--dry" + "-run")

#: The 20-character key SHAPE, not the four-letter prefix. `deploy/launch.md` says the
#: prefix in prose in order to explain this very check, and `tests/test_deploy.py` carries
#: it inside a pattern; a substring grep fails on both and teaches the team to stop
#: writing down how their own secret scan works. A real leaked id is always prefix + 16.
AWS_KEY_RE = re.compile(AWS_KEY_PREFIX + r"[0-9A-Z]{16}")
PLACEHOLDER_RE = re.compile(re.escape(PLACEHOLDER_TOKEN))
#: Digit-bounded on the right: the fifteen-dollar figure must not match the grant's $150, which is the AWS compute grant
#: and a current figure, not a superseded cost.
STALE_COST_RE = re.compile(
    "|".join(re.escape(cost) + r"(?![0-9])" for cost in STALE_COSTS)
)

#: Case-INSENSITIVE and word-bounded, because a real mocking switch is spelled the way a
#: shell flag is spelled. An uppercase substring test passes `--mock` and `--offline`
#: cleanly, which are the two forms this check actually has to catch; the tokens are
#: uppercase above only so that this file does not match its own scan.
MOCK_COMMAND_RE = re.compile(
    "|".join(r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])" for token in MOCK_TOKENS),
    re.IGNORECASE,
)

#: The two optional featured paths on the event listing. Neither is entered, and "neither
#: is entered" has to be a fact on disk rather than a memory, or it is not checkable.
SPECIAL_AWARDS: tuple[str, ...] = ("Best Use of COOL", "Agentic Vision with OpenCV 5")

#: The two labels the reproduce block must carry. They are the whole point of the block:
#: one command reproduces the micro numbers here and now, the other is the 95-minute pair
#: that produced the published headline, and conflating them is the single most attackable
#: claim this project could make.
REPRODUCE_LABELS: tuple[str, ...] = ("REPRODUCED HERE", "PUBLISHED, NOT RECOMPUTED HERE")
REPRODUCE_COMMANDS: tuple[str, ...] = ("make verify", "make fetch", "make bench")

DOC_FILES: tuple[str, ...] = ("README.md", "DEMO.md", "REPORT.md")

TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".lock", ".cfg", ".ini",
        ".sh", ".service", ".html", ".css", ".js", ".jsx", ".ts", ".tsx", ".csv", ".log",
        ".env", ".conf",
    }
)
TEXT_NAMES: frozenset[str] = frozenset({"Makefile", "Caddyfile", "Dockerfile", ".gitignore"})
SKIP_DIRS: frozenset[str] = frozenset(
    {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".next", "dist", "corpus"}
)
MAX_SCAN_BYTES = 4_000_000


@dataclass
class Check:
    name: str
    status: str
    detail: str
    evidence: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass
class Config:
    """Submit-time inputs. Absent ones become SKIPs, never silent passes."""

    live_url: str | None = None
    video_url: str | None = None
    headline_fields: tuple[str, ...] = ("headline",)
    special_awards: tuple[str, ...] = SPECIAL_AWARDS
    timeout: float = 15.0

    @classmethod
    def load(cls, path: pathlib.Path | None) -> "Config":
        if path is None or not path.is_file():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            live_url=raw.get("live_url"),
            video_url=raw.get("video_url"),
            headline_fields=tuple(raw.get("headline_fields", ("headline",))),
            special_awards=tuple(raw.get("special_awards", SPECIAL_AWARDS)),
            timeout=float(raw.get("timeout", 15.0)),
        )


# ---------------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------------

Fetch = Callable[[str, float], "tuple[int, bytes]"]


def http_get(url: str, timeout: float = 15.0) -> tuple[int, bytes]:
    """One GET. Returns ``(status, body)``; a 4xx/5xx is a status, not an exception.

    ``urllib`` rather than ``httpx`` because this script has to run on the box during a
    deploy, where the test extras may not be installed -- and because a readiness gate
    that needs its own dependency installed correctly is a gate with a hole in it.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "plumerank-readiness/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()


# ---------------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------------


def iter_text_files(*roots: pathlib.Path | None) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for root in roots:
        if root is None or not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if SKIP_DIRS.intersection(path.parts):
                continue
            if path.suffix not in TEXT_SUFFIXES and path.name not in TEXT_NAMES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
    return files


def read_text(path: pathlib.Path) -> str:
    """The file, or its first ``MAX_SCAN_BYTES`` characters if it is enormous.

    Returning ``""`` for an oversized file -- which is what this did first -- makes the
    secret scan report a clean tree because it never looked: a key pasted at the end of a
    5 MB log would have passed. A prefix is scanned instead, so the cap bounds the work
    without ever turning "not read" into "nothing there".
    """
    with path.open(encoding="utf-8", errors="replace") as handle:
        return handle.read(MAX_SCAN_BYTES)


def grep_tree(roots: list[pathlib.Path | None], pattern: re.Pattern[str]) -> list[str]:
    """Every ``path:line: text`` hit, so a failure is actionable without a second grep."""
    hits: list[str] = []
    for path in iter_text_files(*roots):
        text = read_text(path)
        if not pattern.search(text):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{path}:{number}: {line.strip()[:100]}")
    return hits


_SHELL_COMMENT_RE = re.compile(r"(^|\s)#.*$")
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


def command_lines(root: pathlib.Path, submission: pathlib.Path | None) -> list[tuple[str, str]]:
    """Only the lines a reader would actually type or a target would actually run.

    Prose is excluded deliberately. An honest README says "there is no offline mode", and
    a scanner that failed on that sentence would push the documentation towards saying
    less about the thing it is proudest of.
    """
    out: list[tuple[str, str]] = []
    makefile = root / "Makefile"
    if makefile.is_file():
        for number, line in enumerate(read_text(makefile).splitlines(), start=1):
            if line.startswith("\t"):
                out.append((f"Makefile:{number}", line.strip()))
    sources = [root / name for name in DOC_FILES]
    if submission is not None and submission.is_dir():
        sources.extend(sorted(submission.rglob("*.md")))
    for path in sources:
        if not path.is_file():
            continue
        text = read_text(path)
        for block in _FENCE_RE.findall(text):
            for line in block.splitlines():
                if line.strip():
                    out.append((path.name, line.strip()))
    return out


def doc_text(root: pathlib.Path, submission: pathlib.Path | None) -> str:
    parts: list[str] = []
    for name in DOC_FILES:
        path = root / name
        if path.is_file():
            parts.append(read_text(path))
    if submission is not None and submission.is_dir():
        for path in sorted(submission.rglob("*.md")):
            parts.append(read_text(path))
    return "\n".join(parts)


def head_sha(root: pathlib.Path) -> str | None:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def newest_bench(root: pathlib.Path) -> pathlib.Path | None:
    # Exact name, as api/main.py:_benchmark_file: an ablation's bench_<date>_<cell>.json
    # sorts after bench_<date>.json and must never be read as the headline.
    candidates = sorted(
        p for p in (root / "results").glob("bench_*.json")
        if re.fullmatch(r"bench_\d{4}-\d{2}-\d{2}\.json", p.name)
    )
    return candidates[-1] if candidates else None


def dig(payload: object, dotted: str) -> object:
    cur = payload
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


_MISSING = object()


# ---------------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------------


def check_live_url(config: Config, fetch: Fetch) -> Check:
    if not config.live_url:
        return Check("live-url", SKIP, "no live URL supplied (--live-url or the config file)")
    try:
        status, _body = fetch(config.live_url, config.timeout)
    except Exception as exc:
        return Check("live-url", FAIL, f"{config.live_url} did not answer: {exc!r}")
    if status != 200:
        return Check("live-url", FAIL, f"{config.live_url} answered {status}")
    return Check("live-url", PASS, f"{config.live_url} answered 200")


def check_build_info(root: pathlib.Path, config: Config, fetch: Fetch) -> Check:
    if not config.live_url:
        return Check("build-info", SKIP, "no live URL supplied")
    sha = head_sha(root)
    if sha is None:
        return Check("build-info", FAIL, "cannot read git HEAD, so the deployed sha cannot be checked")
    url = config.live_url.rstrip("/") + "/build-info"
    try:
        status, body = fetch(url, config.timeout)
    except Exception as exc:
        return Check("build-info", FAIL, f"{url} did not answer: {exc!r}")
    if status != 200:
        return Check("build-info", FAIL, f"{url} answered {status}")

    text = body.decode("utf-8", errors="replace")
    misses: list[str] = []
    if "OpenCV" not in text:
        misses.append("the payload never says OpenCV")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return Check("build-info", FAIL, f"{url} is not JSON: {exc}")

    # The shape api/main.py actually serves: {"opencv": {"version": "5.x", ...}, ...}.
    # This read a flat top-level value until 2026-10-04 -- the same nested-vs-flat bug
    # c3a38bb fixed in the web tier -- so it failed against the real box while its own
    # fixture, written in a shape the server never sends, kept passing.
    opencv = payload.get("opencv")
    version = opencv.get("version") if isinstance(opencv, dict) else None
    if not isinstance(version, str) or not version.startswith("5."):
        misses.append(f"opencv.version is {version!r} -- the deployed OpenCV is not 5.x")
    commit = payload.get("commit")
    if commit != sha:
        misses.append(f"commit is {commit!r}, local HEAD is {sha!r}")
    if misses:
        return Check("build-info", FAIL, f"{url} disagrees with this checkout", misses)
    return Check("build-info", PASS, f"{url}: OpenCV 5.x at {sha[:12]}")


def check_benchmark_matches(root: pathlib.Path, config: Config, fetch: Fetch) -> Check:
    local = newest_bench(root)
    if local is None:
        return Check("benchmark-matches", FAIL, "no results/bench_<date>.json on disk to compare against")
    if not config.live_url:
        return Check("benchmark-matches", SKIP, "no live URL supplied")
    url = config.live_url.rstrip("/") + "/api/benchmark"
    try:
        status, body = fetch(url, config.timeout)
    except Exception as exc:
        return Check("benchmark-matches", FAIL, f"{url} did not answer: {exc!r}")
    if status != 200:
        return Check("benchmark-matches", FAIL, f"{url} answered {status}")

    on_disk = local.read_bytes()
    if body == on_disk:
        return Check("benchmark-matches", PASS, f"{url} is byte-identical to {local.name}")

    # Not byte-identical is not automatically wrong -- whitespace and key order survive a
    # round trip through nothing at all. The headline fields are what a judge reads, so
    # they are compared explicitly and the difference is named.
    try:
        served = json.loads(body.decode("utf-8", errors="replace"))
        stored = json.loads(on_disk.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return Check("benchmark-matches", FAIL, f"{url} or {local.name} is not JSON: {exc}")

    misses: list[str] = []
    for dotted in config.headline_fields:
        a, b = dig(served, dotted), dig(stored, dotted)
        if a is _MISSING or b is _MISSING:
            misses.append(f"headline field {dotted} is absent from the served payload or {local.name}")
        elif json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            misses.append(f"headline field {dotted} differs: served {a!r} vs {local.name} {b!r}")
    if misses:
        return Check("benchmark-matches", FAIL, f"{url} disagrees with {local.name}", misses)
    return Check(
        "benchmark-matches",
        PASS,
        f"{url} matches {local.name} on {len(config.headline_fields)} headline field(s); "
        f"the payloads differ only outside them",
    )


def check_reproduce_honest(root: pathlib.Path, submission: pathlib.Path | None) -> Check:
    lines = command_lines(root, submission)
    if not lines:
        return Check("reproduce-honest", FAIL, "no runnable command found in the Makefile or the docs")
    # The trailing shell comment is prose about the command, not part of it: "# offline
    # suite, no network, no credentials" describes `make test`, it does not switch a mock on.
    hits = [
        f"{where}: {line}"
        for where, line in lines
        if MOCK_COMMAND_RE.search(_SHELL_COMMENT_RE.sub("", line))
    ]
    if hits:
        return Check("reproduce-honest", FAIL, f"{len(hits)} command(s) carry a mocking token", hits)
    return Check("reproduce-honest", PASS, f"{len(lines)} command line(s), none of them mocked")


def check_claims_green(root: pathlib.Path) -> Check:
    script = root / "scripts" / "check_claims.py"
    if not script.is_file():
        return Check("check-claims", FAIL, "scripts/check_claims.py does not exist")
    # Run it, rather than importing and calling run(): the claim being verified is that
    # the COMMAND exits zero, and that is what make test runs.
    done = subprocess.run(
        [sys.executable, str(script), "--root", str(root), "--no-json"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if done.returncode != 0:
        misses = [
            line.strip()[2:]
            for line in done.stdout.splitlines()
            if line.startswith("          - ")
        ]
        return Check("check-claims", FAIL, f"check_claims.py exited {done.returncode}", misses[:12])
    # An exit code of zero is not the same as "every claim was checked". check_claims.py
    # skips its five document checks when no judge-facing document is on disk, and exits
    # zero having verified nothing a judge will read. That is the correct answer in week
    # three and the wrong one at submit time, so the skips are read out of the report
    # rather than inferred from the exit code.
    skipped = [
        line.strip()[len("[SKIP] ") :].strip()
        for line in done.stdout.splitlines()
        if line.strip().startswith("[" + SKIP + "]")
    ]
    if skipped:
        return Check(
            "check-claims",
            FAIL,
            f"check_claims.py exited zero with {len(skipped)} check(s) skipped -- "
            f"at submit time every judge-facing document must exist and be checked",
            skipped,
        )
    surface = next(
        (line.strip() for line in done.stdout.splitlines() if "SURFACES FOUND TODAY" in line),
        "surface count not printed",
    )
    return Check("check-claims", PASS, surface)


def check_no_placeholders(submission: pathlib.Path | None) -> Check:
    if submission is None or not submission.is_dir():
        return Check("no-placeholders", SKIP, "no submission directory to scan")
    hits = grep_tree([submission], PLACEHOLDER_RE)
    if hits:
        return Check("no-placeholders", FAIL, f"{len(hits)} placeholder(s) left in the submission text", hits)
    return Check("no-placeholders", PASS, f"no '{PLACEHOLDER_TOKEN}' anywhere under {submission.name}/")


def check_no_aws_key(root: pathlib.Path, submission: pathlib.Path | None) -> Check:
    hits = grep_tree([root, submission], AWS_KEY_RE)
    if hits:
        return Check("no-aws-key", FAIL, f"{len(hits)} possible access-key id(s) in the tree", hits)
    return Check("no-aws-key", PASS, "no access-key id anywhere in the tree")


def check_no_stale_cost(root: pathlib.Path, submission: pathlib.Path | None) -> Check:
    # web/out is the gitignored Next.js export: its RSC payload is full of dollar-digit-shaped
    # row references that are not money, and a stale figure in it can only have come from
    # a source file this scan already reads.
    export = root / "web" / "out"
    hits = [
        hit for hit in grep_tree([root, submission], STALE_COST_RE)
        if not pathlib.Path(hit.split(":", 1)[0]).is_relative_to(export)
    ]
    if hits:
        return Check(
            "no-stale-cost",
            FAIL,
            f"{len(hits)} superseded cost figure(s); every cost figure must be the current one",
            hits,
        )
    return Check("no-stale-cost", PASS, "no superseded cost figure in the tree")


def check_video_url(config: Config, fetch: Fetch) -> Check:
    if not config.video_url:
        return Check("video-url", SKIP, "no demo video URL supplied (--video-url or the config file)")
    try:
        status, _body = fetch(config.video_url, config.timeout)
    except Exception as exc:
        return Check("video-url", FAIL, f"{config.video_url} did not answer: {exc!r}")
    if status != 200:
        return Check("video-url", FAIL, f"{config.video_url} answered {status}")
    return Check("video-url", PASS, f"{config.video_url} answered 200")


_TICKED_RE = re.compile(r"\[\s*[xX]\s*\]")
_UNTICKED_RE = re.compile(r"\[\s*\]")


def check_special_awards(submission: pathlib.Path | None, config: Config) -> Check:
    if submission is None or not submission.is_dir():
        return Check("special-awards", SKIP, "no submission directory to scan")
    text_by_line: list[str] = []
    for path in sorted(submission.rglob("*")):
        if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES):
            text_by_line.extend(read_text(path).splitlines())

    misses: list[str] = []
    for award in config.special_awards:
        rows = [line for line in text_by_line if award.lower() in line.lower()]
        boxed = [line for line in rows if _TICKED_RE.search(line) or _UNTICKED_RE.search(line)]
        if not boxed:
            # An award nobody wrote down is an award whose state is a memory. The two
            # boxes are recorded as explicitly unticked so that "we did not enter" is a
            # thing on disk rather than a thing somebody recalls at 23:30.
            misses.append(f"'{award}' has no recorded checkbox in the submission text")
            continue
        ticked = [line.strip()[:100] for line in boxed if _TICKED_RE.search(line)]
        if ticked:
            misses.append(f"'{award}' is TICKED: {ticked[0]}")
    if misses:
        return Check("special-awards", FAIL, f"{len(misses)} award-box problem(s)", misses)
    return Check("special-awards", PASS, f"{len(config.special_awards)} award box(es) recorded, none ticked")


#: The one sentence this project must never contain. ``make verify`` runs the ~400-frame
#: micro corpus in three minutes; the published number is 54 held-out sequences (60 in the manifest) and takes
#: ``make fetch && make bench``, ~95 minutes. A judge who runs the three-minute command
#: expecting the headline gets a different p50 and concludes the headline does not
#: reproduce -- which turns the strongest asset in the project into its most attackable
#: claim. Having both labelled commands present is not enough -- a sentence elsewhere can
#: still collapse them -- so the prose is read as well as the block.
CONFLATION_SUBJECT: str = "make verify"
CONFLATION_OBJECTS: tuple[str, ...] = (
    "headline", "60-sequence", "60 sequence", "sixty-sequence", "held-out", "hold arm",
)
CONFLATION_VERBS: tuple[str, ...] = ("reproduc", "recomput", "regenerat")

#: A sentence that DENIES the conflation is the sentence we want to exist -- the README is
#: supposed to say `make verify` does not reproduce the headline, and a gate that failed on
#: that would push the documentation into silence about the distinction it exists to draw.
CONFLATION_DENIALS: tuple[str, ...] = (
    "not ", "n't", "never", "no ", "without", "rather than", "instead of",
)

#: A sentence that names the 95-minute pair as well is a sentence DRAWING the distinction,
#: not collapsing it -- "`make verify` recomputes the micro corpus and `make fetch && make
#: bench` is the one that reproduces the published headline" is the correct sentence, and
#: a gate that failed on it would be punishing the most careful writing in the document.
CONFLATION_DISAMBIGUATORS: tuple[str, ...] = ("make bench", "make fetch")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:])\s+")


def conflations(text: str) -> list[str]:
    """Sentences claiming the three-minute command reproduces the published headline.

    Paragraph first, because markdown prose wraps and a line-based reading would miss a
    claim split across two lines; then sentence, because the denial has to sit next to the
    claim it denies. "`make verify` recomputes the micro corpus. It does not reproduce the
    headline." must pass, and it does: neither sentence carries subject, object and verb
    together without a denial. A sentence naming the 95-minute pair as well passes for a
    different reason -- it is showing the reader both commands, which is the whole ask.
    """
    hits: list[str] = []
    for block in text.split("\n\n"):
        flat = " ".join(block.split())
        for sentence in _SENTENCE_SPLIT_RE.split(flat):
            low = sentence.lower()
            if CONFLATION_SUBJECT not in low:
                continue
            if not any(word in low for word in CONFLATION_OBJECTS):
                continue
            if not any(word in low for word in CONFLATION_VERBS):
                continue
            if any(word in low for word in CONFLATION_DENIALS):
                continue
            if any(word in low for word in CONFLATION_DISAMBIGUATORS):
                continue
            hits.append(sentence.strip()[:140])
    return hits


def check_reproduce_both(root: pathlib.Path, submission: pathlib.Path | None) -> Check:
    text = doc_text(root, submission)
    if not text.strip():
        return Check("reproduce-both", SKIP, "no README.md, DEMO.md or submission text yet")
    misses = [f"the reproduce block never says '{token}'" for token in REPRODUCE_COMMANDS + REPRODUCE_LABELS if token not in text]
    misses += [
        f"a paragraph describes '{CONFLATION_SUBJECT}' as producing the published "
        f"headline: {hit}"
        for hit in conflations(text)
    ]
    if misses:
        return Check("reproduce-both", FAIL, f"{len(misses)} defect(s) in the reproduce block", misses)
    return Check(
        "reproduce-both",
        PASS,
        "both commands present, both labelled, and no paragraph conflates them",
    )


_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def check_eligibility_clause(submission: pathlib.Path | None) -> Check:
    if submission is None or not submission.is_dir():
        return Check("eligibility-clause", SKIP, "no submission directory to scan")
    for path in sorted(submission.rglob("*")):
        if not path.is_file() or (path.suffix not in TEXT_SUFFIXES and path.name not in TEXT_NAMES):
            continue
        for block in read_text(path).split("\n\n"):
            low = block.lower()
            # All three in one block, not merely all three in the file: an eligibility
            # quote in one paragraph and a date in another is not a dated quote.
            dated = _ISO_DATE_RE.search(block)
            if "eligib" in low and ("retrieved" in low or "retrieval" in low) and dated:
                date = dated.group(0)
                return Check("eligibility-clause", PASS, f"{path.name}: eligibility clause retrieved {date}")
    return Check(
        "eligibility-clause",
        FAIL,
        "no block in the submission text carries the eligibility clause with a retrieval date",
    )


# ---------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------


@dataclass
class Report:
    checks: list[Check]

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def skipped(self) -> list[Check]:
        return [c for c in self.checks if c.status == SKIP]

    @property
    def exit_code(self) -> int:
        if self.failed:
            return 1
        return 2 if self.skipped else 0

    @property
    def verdict(self) -> str:
        if self.failed:
            return f"NOT READY -- {len(self.failed)} check(s) FAILED"
        if self.skipped:
            return (
                f"NOT READY -- {len(self.skipped)} check(s) could not run. "
                f"A skipped check is not a passed one."
            )
        return "READY -- all twelve checks passed"

    def to_json(self) -> dict:
        return {
            "checks": [c.to_json() for c in self.checks],
            "verdict": self.verdict,
            "exit_code": self.exit_code,
        }

    def render(self) -> str:
        width = max(len(c.name) for c in self.checks)
        lines = ["PlumeRank submission readiness", "=" * 78]
        for check in self.checks:
            lines.append(f"  [{check.status}] {check.name:<{width}}  {check.detail}")
            for item in check.evidence:
                lines.append(f"          - {item}")
        lines.append("-" * 78)
        lines.append(self.verdict)
        return "\n".join(lines)


def run(
    root: pathlib.Path,
    config: Config,
    submission: pathlib.Path | None = None,
    fetch: Fetch | None = None,
) -> Report:
    """Every check, in the order a reader would want to see them fail.

    ``fetch`` is a seam for the tests, which have no network, and it is resolved here
    rather than in the signature so that the default is looked up at call time. There is
    deliberately no command-line flag that swaps it: a switch that makes the gate stop
    talking to the deployed system is exactly the switch that would eventually be left on.
    """
    root = pathlib.Path(root)
    if fetch is None:
        fetch = http_get
    if submission is None:
        submission = root.parent / "submission"
    submission = submission if submission.is_dir() else None

    return Report(
        [
            check_live_url(config, fetch),
            check_build_info(root, config, fetch),
            check_benchmark_matches(root, config, fetch),
            check_reproduce_honest(root, submission),
            check_claims_green(root),
            check_no_placeholders(submission),
            check_no_aws_key(root, submission),
            check_no_stale_cost(root, submission),
            check_video_url(config, fetch),
            check_special_awards(submission, config),
            check_reproduce_both(root, submission),
            check_eligibility_clause(submission),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    default_root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path, default=default_root)
    parser.add_argument("--submission", type=pathlib.Path, default=None, help="the submission text directory")
    parser.add_argument("--config", type=pathlib.Path, default=None, help="JSON with live_url and video_url")
    parser.add_argument("--live-url", default=None)
    parser.add_argument("--video-url", default=None)
    parser.add_argument("--json", action="store_true", help="print the report as JSON as well")
    args = parser.parse_args(argv)

    submission = args.submission if args.submission is not None else args.root.parent / "submission"
    config_path = args.config if args.config is not None else submission / "readiness.json"
    config = Config.load(config_path)
    if args.live_url:
        config.live_url = args.live_url
    if args.video_url:
        config.video_url = args.video_url

    report = run(args.root, config, submission)
    print(report.render())
    if args.json:
        print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
