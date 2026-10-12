"""Pull exactly the frames the committed manifests name, and prove they did not change.

Two files, two different jobs, and confusing them is how a corpus quietly rots:

* **The manifest pins WHICH frames.** ``data/manifest_{dev,hold,confuse}.json`` are
  committed lists of sequence ids and frame names. They are the experiment's design.
* **The receipt pins WHAT they contained.** On the first fetch this script writes
  ``data/fetch_receipt.json`` with a sha256 per file; every later run re-hashes what is on
  disk, compares against that receipt, and exits non-zero on any disagreement.

Neither file alone is enough. A manifest without a receipt lets HPWREN re-encode a frame
under the same name and silently move the published headline; a receipt without a manifest
cannot tell a missing frame from a frame that was never wanted. Both are committed, so a
judge who clones this repo and runs ``make fetch`` either gets the same images this project
was measured on or gets a non-zero exit saying which one differs. The count is whatever the
committed manifests add up to -- 2,409 DEV + 4,827 HOLD + 2,673 CONFUSE = 9,909 today --
and it is printed per manifest at the end of the run rather than asserted here, because a
number in a docstring cannot be re-derived and this one was wrong once already.

Only the named windows -- never the archive
-------------------------------------------
FIgLib publishes 521 sequences. This script fetches 90 of them plus the 33 CONFUSE windows,
because that is what the manifests name. Pulling the archive "to be safe" would be ~40 GB
of somebody else's bandwidth for frames no arm of the evaluation ever reads.

FETCHING IS NOT READING
-----------------------
Writing HOLD bytes to disk is not looking at them, and this script therefore does **not**
append to ``results/hold_access.log``. Only ``scripts/bench.py``, which puts HOLD pixels
through the detector, does. That distinction is what makes the log a meaningful record
instead of a line that fires whenever anyone runs ``make fetch``; ``tests/test_fetch.py``
asserts both the behaviour and the absence of the string from this source.

One engine, no per-corpus code
------------------------------
Every manifest -- FIgLib windows, MTA confusion windows, the Pyro-SDIS training mirror --
is read through the same four keys (``base_url``, ``windows[].sequence_id``,
``windows[].frames``, and an optional per-window ``url_prefix``). A corpus that needs a
branch in here is a corpus whose manifest is underspecified, and the fix belongs in the
manifest.

Two arms have a step AFTER the download, and both are conversions rather than fetches:
``build_micro`` cuts and downscales the committed micro corpus out of the DEV frames, and
``explode_pyrosdis`` turns the HuggingFace parquet shards -- which is the only form
Pyro-SDIS is published in -- into the ultralytics images/ + labels/ tree that
``scripts/train_tile_classifier.py`` reads. Neither writes to the receipt: the receipt
pins what an archive served, and a derived tree is not that.

Usage::

    python scripts/fetch.py --corpus figlib     # DEV + HOLD + CONFUSE, then build micro; what `make fetch` runs
    python scripts/fetch.py --corpus pyrosdis   # the tile classifier's training mirror (retraining only)
    python scripts/fetch.py --corpus all        # both
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence
from urllib.parse import quote

import cv2

# `python scripts/foo.py` puts scripts/ on sys.path, not the repo root, so the package
# this script exists to serve would otherwise be unimportable from its own repo.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plumerank.constants import WARMUP_FRAMES, WORK_HEIGHT, WORK_WIDTH  # noqa: E402

__all__ = [
    "FetchError",
    "PlannedFile",
    "frame_url",
    "http_get",
    "fetch_with_retry",
    "sha256_bytes",
    "sha256_file",
    "load_manifest",
    "plan_manifest",
    "fetch_planned",
    "micro_sequences",
    "split_at_ignition",
    "micro_windows",
    "build_micro",
    "pyrosdis_split",
    "explode_pyrosdis_shard",
    "explode_pyrosdis",
    "main",
]

# ---------------------------------------------------------------------------------
# Being a polite client
# ---------------------------------------------------------------------------------

#: HPWREN gives this imagery away and asks for a credit reference in return. The least a
#: client can do is say who it is, so an operator reading an access log can find us
#: instead of blackholing an anonymous burst of requests.
USER_AGENT: str = (
    "PlumeRank/0.1 (OpenCV AI Competition 2026 entry; fetches only the frames named in "
    "its committed manifests; contact via the repository README)"
)

#: Eight sockets, and the cap is enforced rather than documented. The archive is a CDN in
#: front of one research group's storage; the marginal frame arrives no faster past this
#: point and the marginal request costs them the same either way.
MAX_CONCURRENCY: int = 8

HTTP_TIMEOUT_SECONDS: float = 30.0
RETRY_ATTEMPTS: int = 3
RETRY_BACKOFF_SECONDS: float = 1.5

#: Five DEV *source* sequences, about 400 frames -- 402 today, because three of the five
#: windows the selection rule picks are 80 frames rather than 81: the archive is missing one
#: minute in each and the manifest records that rather than papering over it. Each source
#: sequence is then cut in two (see ``micro_windows``), so the committed corpus is ten
#: windows and fewer frames than 402. This is the whole reason `make verify` finishes in
#: three minutes with no network and no credentials.
MICRO_SEQUENCES: int = 5

#: How the two halves of a cut source sequence are named. A visibly artificial double
#: underscore, because these ids are NOT FIgLib sequence ids and nothing downstream may
#: match them against the archive: ``source_sequence_id`` carries the real one.
MICRO_NON_FIRE_SUFFIX: str = "__pre"
MICRO_FIRE_SUFFIX: str = "__fire"

#: Pre-ignition frames the fire half keeps as lead-in. ``WARMUP_FRAMES`` and not zero:
#: MOG2's output means nothing until it has seen that many frames, so a fire window that
#: began exactly at t0 would spend the first eight minutes of every time-to-detection
#: warming a background model instead of detecting -- an artefact of where the corpus was
#: cut, reported as if it were a property of the detector. Imported rather than repeated as
#: an 8, so the cut and the segmenter cannot drift apart.
MICRO_FIRE_LEAD_IN_FRAMES: int = WARMUP_FRAMES

#: The micro frames are committed, so this quality setting is baked into bytes that are in
#: the repository -- it is read once, by whoever regenerates them, and never by the judge
#: path. High enough that the downscale, not the encoder, is what the pipeline sees.
#:
#: 98 and not 92, and the reason is measured rather than assumed: the round trip through
#: the encoder costs a mean absolute error of 1.67 grey levels at 92 against 1.09 at 98,
#: and on a five-window arm that difference is large enough to change which windows the
#: ranking admits -- so at 92 the committed corpus was measurably not the array the
#: pipeline computes from the archive frame, which is the one thing it exists to be. 98 is
#: the highest setting whose committed tree still fits the <=25 MB envelope the corpus design
#: section 8 states for data/micro/ (about 19 MB at 310 frames; quality 100 is 28 MB).
MICRO_JPEG_QUALITY: int = 98

#: The fetch group named by ``--corpus``. The spelling is fixed by the Makefile and the
#: build contract; the grouping is by *what a run needs*, not by which archive the bytes
#: came from -- ``figlib`` is every HPWREN camera window the evaluation replays, and the
#: CONFUSE windows are HPWREN MTA rather than FIgLib.
_HPWREN_MANIFESTS: tuple[str, ...] = (
    "manifest_dev.json",
    "manifest_hold.json",
    "manifest_confuse.json",
)
_MANIFESTS_BY_CORPUS: dict[str, tuple[str, ...]] = {
    "figlib": _HPWREN_MANIFESTS,
    "pyrosdis": ("manifest_pyrosdis.json",),
    "all": _HPWREN_MANIFESTS + ("manifest_pyrosdis.json",),
}

#: Where an arm's bytes land under ``--data-root``. A manifest may override with its own
#: ``local_root`` key; this table only covers the arms whose manifests were committed
#: before the key existed.
_LOCAL_ROOT_BY_ARM: dict[str, str] = {
    "dev": "corpus/figlib",
    "hold": "corpus/figlib",
    "confuse": "corpus/mta",
    "pyrosdis": "pyrosdis",
}

RECEIPT_NAME: str = "fetch_receipt.json"


class FetchError(RuntimeError):
    """A fetch that cannot be completed correctly. Never caught to produce a default."""


# ---------------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------------


def frame_url(prefix: str, frame: str) -> str:
    """``prefix`` joined to one frame name, percent-encoded the way the archive needs.

    FIgLib names the post-ignition frames ``1719093958_+00060.jpg``. A literal ``+`` in a
    URL path is legal but is read as a space by enough intermediaries that the CDN returns
    the wrong key for some of them, so it is encoded to ``%2B``: ``quote`` does that,
    because ``+`` is not in its unreserved set.

    ``safe="/"`` and not ``safe=""``: MTA frame names carry their quarter directory
    (``large/20260726/Q3/1785071971.jpg``) so that one CONFUSE window can span the two
    quarter directories an 81-minute span falls across. FIgLib names contain no slash, so
    the same call is correct for both and neither corpus needs a branch.
    """
    return prefix + quote(frame, safe="/")


def http_get(url: str, timeout: float = HTTP_TIMEOUT_SECONDS) -> bytes:
    """One GET. The single seam the tests replace -- nothing else in here opens a socket."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def fetch_with_retry(url: str, attempts: int = RETRY_ATTEMPTS) -> bytes:
    """GET with exponential backoff on *transient* failures only.

    A 404 or a 403 is not retried. The archive is S3-backed with listing disabled, so a
    missing key answers 403 rather than 404 -- both mean "this frame does not exist", and
    asking three more times neither produces it nor is polite. Retrying is reserved for
    5xx and transport errors, which are the ones a second attempt can actually fix.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            # Resolved through the module global on every call so that the one seam the
            # tests replace stays one seam, rather than a callable threaded through five
            # signatures that production never passes.
            return http_get(url)
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise FetchError(f"{url} -> HTTP {exc.code}") from exc
            last = exc
        # http.client.HTTPException is not an OSError: a body cut off mid-transfer raises
        # IncompleteRead, which escaped this loop and killed a fresh-clone `make fetch`
        # 23 minutes in (2026-10-12). A truncated body is the most transient failure there is.
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            last = exc
        if attempt + 1 < attempts:
            time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
    raise FetchError(f"{url} failed after {attempts} attempts: {last}")


# ---------------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlannedFile:
    """One frame: where it comes from and where it goes. Nothing corpus-specific."""

    url: str
    relative_path: str
    """POSIX, relative to ``--data-root``. It is the receipt's key, so it must not carry a
    platform separator -- a receipt written on Windows would otherwise never match one
    written on Linux and the mismatch would look like corruption."""


def load_manifest(path: pathlib.Path) -> dict:
    if not path.is_file():
        raise FetchError(
            f"{path} is missing. This script fetches what a committed manifest names and "
            f"never guesses a URL scheme, so an absent manifest is a hard stop rather "
            f"than an empty fetch that exits zero."
        )
    return json.loads(path.read_text())


def plan_manifest(manifest: dict, manifest_path: pathlib.Path) -> list[PlannedFile]:
    """Every file one manifest names, expanded once, before a single byte is requested.

    Planning ahead rather than streaming means the run knows its own size, the concurrency
    pool sees one flat work list instead of a nested loop, and a manifest that is malformed
    fails before it has half-populated a corpus directory.
    """
    arm = manifest.get("arm")
    base_url = manifest.get("base_url", "")
    local_root = manifest.get("local_root") or _LOCAL_ROOT_BY_ARM.get(str(arm))
    if local_root is None:
        raise FetchError(
            f"{manifest_path}: arm {arm!r} has no local_root, and this script has no "
            f"default for it. Add a 'local_root' key to the manifest."
        )

    planned: list[PlannedFile] = []
    for position, window in enumerate(manifest.get("windows", [])):
        try:
            sequence_id = window["sequence_id"]
            frames = window["frames"]
        except KeyError as exc:
            raise FetchError(
                f"{manifest_path}: window {position} has no {exc.args[0]!r}. A manifest "
                f"this script cannot read is a corpus it must not half-fetch."
            ) from exc
        prefix = window.get("url_prefix") or (base_url + sequence_id + "/")
        declared = window.get("n_frames")
        if declared is not None and declared != len(frames):
            raise FetchError(
                f"{manifest_path}: window {sequence_id} declares n_frames={declared} but "
                f"lists {len(frames)} frames -- the design and the list disagree."
            )
        for frame in frames:
            planned.append(
                PlannedFile(
                    url=frame_url(prefix, frame),
                    relative_path=f"{local_root}/{sequence_id}/{frame}",
                )
            )
    return planned


# ---------------------------------------------------------------------------------
# The receipt
# ---------------------------------------------------------------------------------


def read_receipt(path: pathlib.Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text()).get("files", {})


def write_receipt(path: pathlib.Path, files: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": date.today().isoformat(),
        "note": (
            "The manifests pin WHICH frames; this receipt pins WHAT they contained. "
            "scripts/fetch.py re-hashes every file against it on every later run and "
            "exits non-zero on a mismatch."
        ),
        "n_files": len(files),
        "files": {key: files[key] for key in sorted(files)},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _looks_complete(relative_path: str, data: bytes) -> bool:
    """Is a file that predates the receipt safe to adopt without re-downloading?

    Only asked on the first fetch, where there is no recorded hash to compare against and
    size alone cannot tell a finished download from an interrupted one. A JPEG carries its
    own end-of-image marker, so for the imagery -- which is all of the bytes -- the answer
    is checkable. Everything else (Pyro-SDIS label files) is adopted, because a truncated
    label file is caught by the receipt on the very next run and there is no in-band marker
    to check here.
    """
    if not relative_path.lower().endswith((".jpg", ".jpeg")):
        return True
    return data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")


# ---------------------------------------------------------------------------------
# The fetch
# ---------------------------------------------------------------------------------


@dataclass(slots=True)
class FetchResult:
    downloaded: int = 0
    skipped: int = 0
    adopted: int = 0
    mismatches: list[str] = field(default_factory=list)
    hashes: dict[str, dict] = field(default_factory=dict)


def _atomic_write(path: pathlib.Path, data: bytes) -> None:
    """Write through a sibling temp file and rename.

    A half-written frame at its final path is indistinguishable from a complete one on the
    next run, and the next run is exactly where resumability lives. ``os.replace`` on the
    same filesystem makes "the file exists" and "the file is whole" the same statement.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _one_file(
    planned: PlannedFile,
    data_root: pathlib.Path,
    receipt: dict[str, dict],
) -> tuple[str, str, dict | None, str | None]:
    """Resolve one planned file. Returns ``(kind, relative_path, entry, mismatch)``."""
    target = data_root / planned.relative_path
    recorded = receipt.get(planned.relative_path)

    if target.is_file():
        digest = sha256_file(target)
        if recorded is not None:
            if digest != recorded.get("sha256"):
                return (
                    "mismatch",
                    planned.relative_path,
                    None,
                    f"{planned.relative_path}: on disk {digest[:12]}, receipt "
                    f"{str(recorded.get('sha256'))[:12]}",
                )
            return "skipped", planned.relative_path, recorded, None
        data = target.read_bytes()
        if _looks_complete(planned.relative_path, data):
            return (
                "adopted",
                planned.relative_path,
                {"sha256": digest, "bytes": len(data)},
                None,
            )
        target.unlink()

    data = fetch_with_retry(planned.url)
    digest = sha256_bytes(data)
    if recorded is not None and digest != recorded.get("sha256"):
        # Do NOT write these bytes. The archive served something other than what this
        # project measured, and putting it on disk would make the next run agree with it.
        return (
            "mismatch",
            planned.relative_path,
            None,
            f"{planned.relative_path}: served {digest[:12]}, receipt "
            f"{str(recorded.get('sha256'))[:12]}",
        )
    _atomic_write(target, data)
    return (
        "downloaded",
        planned.relative_path,
        {"sha256": digest, "bytes": len(data)},
        None,
    )


def fetch_planned(
    planned: Sequence[PlannedFile],
    data_root: pathlib.Path,
    receipt: dict[str, dict],
    concurrency: int = MAX_CONCURRENCY,
) -> FetchResult:
    """Fetch, verify and resume one flat work list under a bounded pool."""
    workers = max(1, min(int(concurrency), MAX_CONCURRENCY))
    result = FetchResult()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_one_file, item, data_root, receipt) for item in planned
        ]
        for future in concurrent.futures.as_completed(futures):
            kind, relative_path, entry, mismatch = future.result()
            if mismatch is not None:
                result.mismatches.append(mismatch)
                continue
            if entry is not None:
                result.hashes[relative_path] = entry
            setattr(result, kind, getattr(result, kind) + 1)
    return result


# ---------------------------------------------------------------------------------
# The micro corpus -- what makes `make verify` a three-minute command
# ---------------------------------------------------------------------------------


def micro_sequences(dev_manifest: dict, n_sequences: int = MICRO_SEQUENCES) -> list[dict]:
    """The DEV windows the micro corpus is cut from, chosen by a rule and not by hand.

    Sorted by ``sequence_id`` (which begins with the date) and taking the first window per
    camera: five different cameras rather than five views of whichever camera happens to
    appear most, because the micro run exercises the per-camera ranking and five windows
    from one camera would exercise a roster of one.
    """
    chosen: list[dict] = []
    seen: set[str] = set()
    for window in sorted(dev_manifest["windows"], key=lambda w: w["sequence_id"]):
        if window["camera_id"] in seen:
            continue
        seen.add(window["camera_id"])
        chosen.append(window)
        if len(chosen) == n_sequences:
            break
    if len(chosen) < n_sequences:
        raise FetchError(
            f"DEV offers only {len(chosen)} distinct cameras, fewer than the "
            f"{n_sequences} the micro corpus needs."
        )
    return chosen


def split_at_ignition(window: dict) -> tuple[list[str], list[str]]:
    """One FIgLib window cut in two at its ignition frame: ``(pre_ignition, from_ignition)``.

    A FIgLib window spans -2400 s to +2400 s around the reported ignition, so its first
    half is not padding: it is a real camera-night on the same camera at the same time of
    day, with no fire in it. That is where the micro corpus gets its false-alarm
    denominator without a second download, and it is the only honest way to get one out of
    a corpus made entirely of fires.

    A window with no ignition clock (``t0_index`` absent or null) is already non-fire and
    is returned whole in the second slot rather than raising: the caller decides what a
    half-empty split means, and the test corpora legitimately contain windows that begin at
    ignition.
    """
    frames = [str(name) for name in window["frames"]]
    t0 = window.get("t0_index")
    if t0 is None:
        return [], frames
    t0 = int(t0)
    if not 0 <= t0 < len(frames):
        raise FetchError(
            f"{window.get('sequence_id')}: t0_index={t0} is outside its own "
            f"{len(frames)} frames, so the window cannot be cut at its ignition."
        )
    return frames[:t0], frames[t0:]


def micro_windows(
    chosen: Sequence[dict],
    lead_in: int = MICRO_FIRE_LEAD_IN_FRAMES,
) -> list[dict]:
    """The split windows the micro manifest names, planned before a pixel is written.

    Each source sequence becomes **two** windows -- a non-fire one before the ignition and
    a fire one from just before it onwards -- because ``bench.py --arm micro`` calibrates
    its threshold on the non-fire windows and measures time-to-detection on the fire ones.
    A micro corpus of fires only has no false-alarm denominator, so no threshold can be
    solved on it and ``make verify`` has nothing to reproduce.

    Three properties this cut has to hold, and none of them is free:

    * **Equal length.** ``bench.load_arm`` refuses an arm of ragged windows, and it is
      right to: one window is one camera-night, and a mixed-length arm makes that unit
      mean a different thing per row. Every window is therefore trimmed to the shortest
      half in the set -- from the *front* for the non-fire windows, so what survives is the
      span closest in time to the fire, under the same light.
    * **Disjoint.** The fire half borrows ``lead_in`` frames from the end of the
      pre-ignition half and the non-fire half gives them up, so the two halves share no
      frame: nothing the threshold is solved on is also part of a time-to-detection
      measurement. Note what this does NOT say. ``bench.load_arm(root, "micro")`` returns
      all ten windows, the calibration ones included, so the micro arm's own false-alarm
      rate is measured partly on the frames its threshold was solved on -- read
      ``run_arm`` and you will see the pre halves replayed a second time. The TTDs are
      clean; the arm rate is not, and it is not the published one.
    * **Warm at t0.** The lead-in exists so the background model is past ``WARMUP_FRAMES``
      when the fire starts; see ``MICRO_FIRE_LEAD_IN_FRAMES``.

    What it costs is written down rather than discovered: the two halves are cut from the
    same five sequences, so micro's calibration set is that corpus's own quiet hours. It
    proves the pipeline runs and that its committed numbers are self-consistent. It is not,
    and never claims to be, the held-out headline -- that is HOLD minus
    ``data/excluded.json`` (see ``arm_counts``) and ``make fetch && make bench``.
    """
    if lead_in < 0:
        raise FetchError(f"lead_in must be non-negative, got {lead_in!r}")
    if not chosen:
        raise FetchError("micro_windows() got no source sequences -- nothing to cut")

    planned: list[dict] = []
    for source in chosen:
        pre, blaze = split_at_ignition(source)
        lead = min(lead_in, len(pre))
        calm, borrowed = pre[: len(pre) - lead], pre[len(pre) - lead :]
        if calm:
            planned.append(_micro_window(source, MICRO_NON_FIRE_SUFFIX, "non-fire", calm, None))
        planned.append(
            _micro_window(source, MICRO_FIRE_SUFFIX, "fire", borrowed + blaze, lead)
        )

    length = min(len(window["frames"]) for window in planned)
    for window in planned:
        frames = window["frames"]
        # Non-fire windows are trimmed from the front and fire windows from the back, so
        # the fire window keeps its ignition at index ``lead`` and the aftermath behind it.
        window["frames"] = frames[-length:] if window["t0_index"] is None else frames[:length]
        window["n_frames"] = length
    return planned


def _micro_window(
    source: dict,
    suffix: str,
    kind: str,
    frames: Sequence[str],
    t0_index: int | None,
) -> dict:
    """One half of a cut source sequence, in the committed manifest schema.

    Every key of the source window survives so that anything able to read
    ``manifest_dev.json`` reads this one without a branch; ``sequence_id`` is the only one
    that changes meaning, which is why ``source_sequence_id`` is added beside it rather
    than the original being overwritten and lost.
    """
    window = {key: value for key, value in source.items() if key != "frames"}
    window["sequence_id"] = f"{source['sequence_id']}{suffix}"
    window["source_sequence_id"] = str(source["sequence_id"])
    window["class"] = kind
    window["t0_index"] = t0_index
    # A non-fire window has no ignition, so it carries no ignition clock either. Inheriting
    # the source's ``t0_offset_seconds`` would leave a 0 sitting in a field that means
    # "seconds from ignition" on a window that has none -- a number that looks measured and
    # is not. build_confusion_set.py writes None here for the same reason.
    if t0_index is None:
        window["t0_offset_seconds"] = None
    window["n_frames"] = len(frames)
    window["frames"] = list(frames)
    return window


def arm_counts(data_root: pathlib.Path) -> dict[str, int]:
    """The CONFUSE and HOLD denominators, READ from the committed files, never restated.

    WHY -- found 2026-09-06. This module used to hardcode "45 windows" for CONFUSE and
    "60 unseen HOLD sequences" into the notes it writes, and both were wrong by the time
    anyone re-ran ``make fetch``:

    * CONFUSE is **33** windows, not 45. ``results/gonogo_2026-09-01.md`` §7.5 records the
      supply limits that produced it and ends "Nothing in this arm may be published as 45."
      A regenerated manifest was republishing 45 anyway.
    * HOLD is 60 windows **minus** ``data/excluded.json``. The committed manifest note had
      been hand-corrected to carry that subtraction, with the standing rule that "neither
      end of that subtraction is ever quoted without the other". Regenerating reverted it
      to the bare 60.

    So a routine ``make fetch`` silently reintroduced two count errors into a judge-facing
    file. Deriving them here is the same discipline ``bench.py:arm_denominator`` already
    applies -- the numbers come from the files or they are not written.
    """
    def _windows(name: str) -> int:
        path = data_root / name
        if not path.is_file():
            return 0
        blob = json.loads(path.read_text(encoding="utf-8"))
        return len(blob.get("windows", []))

    excluded_path = data_root / "excluded.json"
    excluded_ids = set()
    if excluded_path.is_file():
        blob = json.loads(excluded_path.read_text(encoding="utf-8"))
        rows = blob.get("excluded", blob) if isinstance(blob, dict) else blob
        excluded_ids = {r["sequence_id"] if isinstance(r, dict) else r for r in rows}

    hold_path = data_root / "manifest_hold.json"
    hold_ids = set()
    if hold_path.is_file():
        hold_ids = {
            w["sequence_id"]
            for w in json.loads(hold_path.read_text(encoding="utf-8")).get("windows", [])
        }
    dropped = len(hold_ids & excluded_ids)
    return {
        "confuse": _windows("manifest_confuse.json"),
        "hold_manifest": len(hold_ids),
        "hold_excluded": dropped,
        "hold_measured": len(hold_ids) - dropped,
    }


#: The sentence that must accompany the exclusion count wherever it is published --
#: ``tests/test_denominator.py::test_the_exclusion_disclosure_sits_where_the_count_is_published``
#: enforces the coupling. A count of dropped windows without the rule that dropped them is
#: an unexplained subtraction, and an unexplained subtraction in a denominator is the shape
#: of a corpus quietly chosen to flatter a result.
EXCLUSION_DISCLOSURE = (
    "Producing data/excluded.json meant a human looking at the FIRST FRAME of every HOLD "
    "window, about 40 minutes before ignition, against criteria stated before the pass; "
    "the HOLD access log in results/ still records no HOLD read, because no HOLD pixel has "
    "been scored."
    # The log's filename is deliberately NOT spelled here. test_fetch.py asserts the token
    # appears exactly once in this file -- in the module docstring, as a rule -- so that it
    # can never appear as a path. Prose that names it would defeat a guard worth keeping.
)


def _hold_phrase(c: dict[str, int]) -> str:
    """"the N held-out sequences that survive exclusion -- M named, K dropped", with the
    disclosure attached.

    Both ends of the subtraction, always, because one end alone is the number that misleads
    -- and the disclosure travels WITH them rather than beside them, because on 2026-09-06
    deriving the counts correctly still published them into PROVENANCE.md without the rule
    that produced them, and only the coupling test noticed.
    """
    if not c["hold_manifest"]:
        return "the held-out HOLD sequences"
    return (
        f"the {c['hold_measured']} held-out sequences that survive exclusion -- "
        f"data/manifest_hold.json names {c['hold_manifest']} windows, data/excluded.json "
        f"drops {c['hold_excluded']} of them. {EXCLUSION_DISCLOSURE}"
    )


def build_micro(
    dev_manifest: dict,
    data_root: pathlib.Path,
    n_sequences: int = MICRO_SEQUENCES,
) -> dict:
    """Cut five DEV sequences in two, downscale both halves to working resolution, commit.

    This is the only part of the corpus that lives in git, and it is why the judge's
    one-liner needs no network and no credentials. The frames are re-encoded rather than
    stored as arrays so that ``make verify`` reads them through the same ``cv2.imread``
    path the real arms use -- a .npy micro corpus would verify a path that never runs.

    ``INTER_AREA`` and not the default bilinear: 3072x2048 down to 480x270 is a 6.4x
    reduction, where bilinear samples four source pixels out of forty and aliases exactly
    the thin high-contrast structure -- ridgelines, tower guy-wires -- that the boundary
    statistic then reads as texture.

    Why both halves: ``micro_windows``.
    """
    micro_root = data_root / "micro"
    micro_root.mkdir(parents=True, exist_ok=True)
    # Read, never restate -- see arm_counts().
    _counts = arm_counts(data_root)
    source_root = data_root / (
        dev_manifest.get("local_root") or _LOCAL_ROOT_BY_ARM["dev"]
    )

    windows = micro_windows(micro_sequences(dev_manifest, n_sequences))
    for window in windows:
        out_dir = micro_root / window["sequence_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for frame in window["frames"]:
            src = source_root / window["source_sequence_id"] / frame
            image = cv2.imread(str(src))
            if image is None:
                raise FetchError(
                    f"{src} is missing or unreadable -- the micro corpus is cut from the "
                    f"fetched DEV frames, so fetch DEV before building it."
                )
            small = cv2.resize(
                image, (WORK_WIDTH, WORK_HEIGHT), interpolation=cv2.INTER_AREA
            )
            ok, encoded = cv2.imencode(
                ".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), MICRO_JPEG_QUALITY]
            )
            if not ok:
                raise FetchError(f"cv2.imencode refused {src}")
            _atomic_write(out_dir / frame, encoded.tobytes())

    fires = [w for w in windows if w["t0_index"] is not None]
    sources = sorted({str(w["source_sequence_id"]) for w in windows})
    frames_per_window = windows[0]["n_frames"] if windows else 0

    manifest = {
        "arm": "micro",
        "n_windows": len(windows),
        "n_fire_windows": len(fires),
        "n_calibration_windows": len(windows) - len(fires),
        "frames_per_window": frames_per_window,
        "generated": date.today().isoformat(),
        "source": "HPWREN FIgLib, cut at the ignition frame and downscaled",
        "credit": dev_manifest["credit"],
        "licence_note": dev_manifest["licence_note"],
        "base_url": "",
        "url_note": (
            "No URL: these frames are local and committed. They are cut from the DEV "
            "windows named in source_sequence_id by scripts/fetch.py, at "
            f"{WORK_WIDTH}x{WORK_HEIGHT} with INTER_AREA and JPEG quality "
            f"{MICRO_JPEG_QUALITY}."
        ),
        "local_root": "micro",
        "catalogue": (
            f"{len(sources)} DEV source sequences on {len(sources)} distinct cameras, cut "
            f"into {len(windows)} windows ({len(fires)} fire, "
            f"{len(windows) - len(fires)} non-fire) of {frames_per_window} frames each, "
            f"{sum(len(w['frames']) for w in windows)} frames in total."
        ),
        "selection": [
            "The DEV windows sorted by sequence_id, first per camera, first "
            f"{n_sequences}. A rule and not a hand-picked list, so the micro corpus is "
            "reproducible from the committed DEV manifest alone.",
            "DEV and never HOLD: the micro corpus is the one part of the imagery that is "
            "committed and therefore looked at, and HOLD is read exactly once.",
            "Each source sequence is then cut in two at its ignition frame. A FIgLib "
            "window spans -2400 s to +2400 s around ignition, so its pre-ignition half is "
            "a real non-fire camera-night on the same camera at the same time of day. The "
            "fire half keeps "
            f"{MICRO_FIRE_LEAD_IN_FRAMES} pre-ignition frames of lead-in so the "
            "background model is warm at t0; the non-fire half gives those frames up, so "
            "the two halves share no frame.",
        ],
        "calibration_note": (
            "THE ONE THING TO READ BEFORE QUOTING ANY NUMBER FROM THIS ARM: micro's "
            "calibration windows are the pre-ignition halves of its own fire sequences. "
            "They are real non-fire pixels on the real cameras, but they are not an "
            "independent adversary -- the CONFUSE arm is, and it is "
            f"{_counts['confuse']} windows chosen by "
            "confusion class in data/manifest_confuse.json. So the micro arm proves that "
            "the pipeline runs and that its committed numbers are self-consistent, and it "
            "is NOT and never claims to be the held-out headline: that is "
            f"{_hold_phrase(_counts)}, measured by `make fetch && make bench`. Nor is its "
            "time-to-detection an estimate of anything: five fire windows is too few for a "
            "median, and the count that detects at all is chaotic with respect to noise "
            "far below one grey level -- re-encoding these same frames at JPEG quality "
            "92/95/98/100 gives 0, 2, 2 and 1 detections of 5 and moves the solved "
            "threshold from 0.30 to 0.41, with identical code and an identical cut. What "
            "reproduces here is the bytes, not the science."
        ),
        "sha256_note": (
            "These frames are committed, so git is their receipt. "
            "data/fetch_receipt.json covers the fetched arms only."
        ),
        "windows": windows,
    }
    (micro_root / "manifest_micro.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    _write_provenance(micro_root, dev_manifest, manifest, _counts)
    return manifest


def _write_provenance(
    micro_root: pathlib.Path,
    dev_manifest: dict,
    micro_manifest: dict,
    _counts: dict[str, int],
) -> None:
    """The credit reference travels with the pixels.

    ``data/micro/`` is the only imagery in the repository, so it is the only imagery that
    reaches somebody who never read ``data/manifest_dev.json``. HPWREN asks for a credit
    reference in return for the data; carrying it in the manifest alone would put it in a
    file nobody opens.
    """
    windows = micro_manifest["windows"]
    fires = [w for w in windows if w["t0_index"] is not None]
    calm = [w for w in windows if w["t0_index"] is None]
    sources = sorted({str(w["source_sequence_id"]) for w in windows})
    frames = sum(len(w["frames"]) for w in windows)
    per_window = micro_manifest["frames_per_window"]

    text = "\n".join(
        [
            "# data/micro/ -- provenance",
            "",
            dev_manifest["credit"],
            "",
            dev_manifest["licence_note"],
            "",
            "## What these files are",
            "",
            f"{len(windows)} replay windows ({len(fires)} fire, {len(calm)} non-fire) of "
            f"{per_window} frames each -- {frames} JPEGs in total -- cut from "
            f"{len(sources)} HPWREN FIgLib sequences on {len(sources)} distinct cameras.",
            "",
            f"Each frame was downscaled from 3072x2048 to {WORK_WIDTH}x{WORK_HEIGHT} "
            f"(6.4x in x, 7.6x in y -- the pipeline's working raster, which is 16:9 where "
            "the source is 3:2) with cv2.INTER_AREA, and re-encoded as JPEG at quality "
            f"{MICRO_JPEG_QUALITY} by scripts/fetch.py. INTER_AREA and not the default "
            "bilinear: at this reduction bilinear samples four source pixels out of forty "
            "and aliases exactly the thin high-contrast structure the boundary statistic "
            "reads.",
            "",
            "They exist so that `make verify` recomputes real numbers on real pixels in "
            "three minutes with no network and no credentials.",
            "",
            "## Source sequences",
            "",
            *[f"- {source}" for source in sources],
            "",
            "Each one was cut in two at its ignition frame, giving a `__pre` window and a "
            "`__fire` window. Every window's `source_sequence_id` in "
            "manifest_micro.json names the sequence above that it came from.",
            "",
            "## Why the calibration windows are the pre-ignition halves, and what that means",
            "",
            "`scripts/bench.py --arm micro` solves an alert threshold on the non-fire "
            "windows and measures time-to-detection on the fire ones. A corpus of fires "
            "only has no false-alarm denominator, so no threshold can be solved on it at "
            "all.",
            "",
            "A FIgLib window spans -2400 s to +2400 s around the reported ignition, so its "
            "pre-ignition half is a real non-fire camera-night -- same camera, same time "
            "of day, same light. **micro's calibration windows are therefore the "
            "pre-ignition halves of its own fire sequences.** The fire half keeps "
            f"{MICRO_FIRE_LEAD_IN_FRAMES} pre-ignition frames of lead-in so the background "
            "model is warm when the fire starts, and the non-fire half gives those frames "
            "up, so the two halves share no frame -- nothing the threshold is solved on is "
            "also part of a time-to-detection measurement. What that does not say: the "
            "micro *arm* is all "
            f"{len(windows)} windows, the calibration ones included, so micro's arm "
            "false-alarm rate is measured partly on the frames its threshold was solved "
            "on. The time-to-detection numbers are clean; that rate is not, and it is not "
            "the published one.",
            "",
            "One more thing this arm is not. Its time-to-detection is not an estimate: "
            f"{len(fires)} fire windows is too few for a median, and which of them detect "
            "at all is chaotic with respect to noise well below one grey level. "
            "Re-encoding these same frames at JPEG quality 92/95/98/100 -- differences "
            "invisible on screen -- gives 0, 2, 2 and 1 detections of 5 and moves the "
            "solved threshold from 0.30 to 0.41, with identical code and an identical cut. "
            "What `make verify` reproduces is the committed bytes, byte for byte. It is a "
            "determinism check, not a measurement.",
            "",
            "**This is why the micro arm proves that the pipeline runs and that its "
            "committed numbers are self-consistent, and why it is NOT -- and never claims "
            "to be -- the held-out headline.** The headline is measured on "
            f"{_hold_phrase(_counts)}, against the {_counts['confuse']}-window CONFUSE "
            "adversary set, by "
            "`make fetch && make bench` (about 95 minutes). `make verify` prints that "
            "number in a block labelled PUBLISHED, NOT RECOMPUTED HERE, quoted verbatim "
            "from results/.",
            "",
            "## Licence",
            "",
            "FIgLib carries no SPDX licence. Its terms ask for a credit reference, which "
            "is the line at the top of this file and appears again in the README, in the "
            "UI footer and in the SSE stream's meta event. This imagery is used for "
            "EVALUATION ONLY and never for training -- the one trained model in this "
            "project sees Pyro-SDIS (Apache-2.0) and nothing else, and "
            "`test_no_figlib_in_training` is what enforces it.",
            "",
            "## Regenerating them",
            "",
            "    python scripts/fetch.py --corpus figlib",
            "",
            "The selection rule is in data/micro/manifest_micro.json under `selection`, "
            "and the cut is `micro_windows` in scripts/fetch.py.",
            "",
        ]
    )
    (micro_root / "PROVENANCE.md").write_text(text)


# ---------------------------------------------------------------------------------
# Pyro-SDIS -- parquet on the wire, a YOLO tree on disk
# ---------------------------------------------------------------------------------

#: The columns this script reads out of a Pyro-SDIS shard. Named rather than taken
#: positionally because the shard carries three more (``partner``, ``camera``, ``date``)
#: that are ~0 bytes each while ``image`` is ~100 KB, and pyarrow reads column chunks:
#: asking for all six would pull the whole 480 MB shard through memory to write files
#: that only ever needed three of them.
PYROSDIS_COLUMNS: tuple[str, ...] = ("image", "image_name", "annotations")

#: Rows held in memory at once while a shard is exploded. 256 x ~100 KB is ~25 MB, which
#: is the whole reason this is streamed by batch rather than read with ``pq.read_table``:
#: the six train shards are 2.9 GB and a table-at-a-time read of the largest is a 480 MB
#: allocation for a job whose working set is one image.
PYROSDIS_BATCH_ROWS: int = 256

#: The two splits Pyro-SDIS publishes, and the only two directory names this script will
#: create. ``val`` is the held-out number the training run publishes, so a shard whose
#: split cannot be read off its own filename is a hard stop rather than a default.
PYROSDIS_SPLITS: frozenset[str] = frozenset({"train", "val"})


def pyrosdis_split(shard_name: str) -> str:
    """``train`` or ``val`` from a shard filename such as ``train-00000-of-00006.parquet``.

    A default would be the worst possible convenience here. If an unreadable name fell
    back to ``train``, the val shard would be exploded into the training split, the model
    would be trained on the images its precision and recall are then measured on, and
    nothing anywhere would say so -- the run would look healthier, not sicker.
    """
    split = pathlib.PurePosixPath(shard_name).name.split("-", 1)[0]
    if split not in PYROSDIS_SPLITS:
        raise FetchError(
            f"cannot read a split off the Pyro-SDIS shard name {shard_name!r}: expected "
            f"one of {sorted(PYROSDIS_SPLITS)} before the first '-'. Guessing would risk "
            f"exploding the val shard into the training split."
        )
    return split


def explode_pyrosdis_shard(
    shard: pathlib.Path,
    out_root: pathlib.Path,
    seen: set[str],
) -> tuple[int, int]:
    """One parquet shard -> ``images/<split>/*.jpg`` + ``labels/<split>/*.txt``.

    Returns ``(written, skipped)`` rows.

    Why this conversion exists at all: Pyro-SDIS is published as HuggingFace parquet --
    an ``image`` struct of raw JPEG bytes, an ``annotations`` string of YOLO lines, and an
    ``image_name`` -- while ``scripts/train_tile_classifier.py`` reads the ultralytics
    tree the dataset's own ``data.yaml`` describes. One of the two had to move, and it is
    this one: the training script's reader is also what a judge points at any other YOLO
    corpus, and rewriting it around a parquet schema would tie the model's provenance to
    one host's storage format.

    The image bytes are written **verbatim**, never decoded and re-encoded. A round trip
    through the JPEG encoder here would put a second generation of lossy compression
    between Pyro-SDIS and the model for no gain, and would make the sha256 in
    ``data/tile_manifest.json`` a hash of this script's encoder settings rather than of
    the corpus.

    An image with no annotation gets an EMPTY label file rather than no file.
    ``train_tile_classifier.parse_yolo_labels`` reads a missing file as "this frame has no
    smoke", which is Pyro-SDIS's own convention and the right answer -- but on disk a
    missing file is also what a truncated conversion leaves behind, and the two must not
    be the same state. An empty file says "converted, and there was nothing in it".
    """
    # Imported here and not at module scope on purpose: pyarrow is in
    # requirements-train.lock and NOT in requirements.lock, so it is absent on the server
    # and absent in `make test`. A module-level import would make this whole script -- the
    # one that pulls the evaluation corpus -- unimportable everywhere the training
    # dependencies are deliberately not installed.
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised by hand, not in CI
        raise FetchError(
            "reading Pyro-SDIS parquet needs pyarrow, which is a TRAINING dependency: "
            "`pip install -r requirements-train.lock`. It is not in requirements.lock "
            "because the server runs no training code and its dependency tree stays small."
        ) from exc

    split = pyrosdis_split(shard.name)
    image_dir = out_root / "images" / split
    label_dir = out_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for batch in pq.ParquetFile(shard).iter_batches(
        batch_size=PYROSDIS_BATCH_ROWS, columns=list(PYROSDIS_COLUMNS)
    ):
        rows = batch.to_pydict()
        for image, name, label_text in zip(
            rows["image"], rows["image_name"], rows["annotations"]
        ):
            name = str(name)
            if name != pathlib.PurePosixPath(name).name or name in ("", ".", ".."):
                raise FetchError(
                    f"{shard.name}: image_name {name!r} is not a bare filename, so writing "
                    f"it would put a file outside {image_dir}"
                )
            key = f"{split}/{name}"
            if key in seen:
                raise FetchError(
                    f"two rows named {key!r} in the Pyro-SDIS shards. The second would "
                    f"overwrite the first and the corpus would silently lose an image."
                )
            seen.add(key)

            image_path = image_dir / name
            label_path = label_dir / (pathlib.PurePosixPath(name).stem + ".txt")
            if image_path.is_file() and label_path.is_file():
                skipped += 1
                continue
            _atomic_write(image_path, bytes(image["bytes"]))
            _atomic_write(label_path, (str(label_text).strip() + "\n").encode("ascii"))
            written += 1
    return written, skipped


def explode_pyrosdis(data_root: pathlib.Path, manifest: dict) -> dict[str, dict[str, int]]:
    """Every shard the Pyro-SDIS manifest names, converted into the tree the trainer reads.

    Resumable in the same sense as the fetch: a row whose image and label are both already
    on disk is skipped, so an interrupted conversion costs the rows it had not reached and
    nothing else. It is deliberately NOT recorded in ``data/fetch_receipt.json`` -- the
    receipt pins what the ARCHIVE served, and these files are derived from the shards it
    already covers. Hashing a derived tree there would make a change of this function look
    like a change in the corpus.
    """
    out_root = data_root / (manifest.get("local_root") or _LOCAL_ROOT_BY_ARM["pyrosdis"])
    totals: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    for window in manifest.get("windows", []):
        for shard_name in window["frames"]:
            shard = out_root / str(window["sequence_id"]) / shard_name
            if not shard.is_file():
                raise FetchError(
                    f"{shard} is missing after its own fetch reported success -- the "
                    f"conversion will not invent the rows it cannot read."
                )
            split = pyrosdis_split(shard_name)
            written, skipped = explode_pyrosdis_shard(shard, out_root, seen)
            bucket = totals.setdefault(split, {"written": 0, "skipped": 0, "shards": 0})
            bucket["written"] += written
            bucket["skipped"] += skipped
            bucket["shards"] += 1
    if not totals:
        raise FetchError(
            "the Pyro-SDIS manifest named no shards, so there is no training corpus. An "
            "empty conversion that exits zero is how a model gets trained on nothing."
        )
    return totals


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------


def _report(name: str, result: FetchResult, total: int) -> None:
    print(
        f"{name:<24} {total:6d} files   "
        f"{result.downloaded:6d} downloaded   "
        f"{result.skipped:6d} verified   "
        f"{result.adopted:6d} adopted   "
        f"{len(result.mismatches):4d} MISMATCH"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch exactly the frames the committed manifests name, sha256-verified "
            "against data/fetch_receipt.json and resumable."
        )
    )
    parser.add_argument("--corpus", required=True, choices=sorted(_MANIFESTS_BY_CORPUS))
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=_REPO_ROOT / "data",
        help="where manifests are read and frames are written (default: ./data)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help=f"parallel requests, capped at {MAX_CONCURRENCY}",
    )
    args = parser.parse_args(argv)

    data_root: pathlib.Path = args.data_root
    receipt_path = data_root / RECEIPT_NAME
    receipt = read_receipt(receipt_path)

    mismatches: list[str] = []
    hashes: dict[str, dict] = dict(receipt)
    dev_manifest: dict | None = None
    pyrosdis_manifest: dict | None = None

    # Every manifest in the group is loaded and planned BEFORE the first byte is requested.
    # Doing it lazily meant `--corpus all` could spend forty minutes pulling FIgLib, hit an
    # absent pyrosdis manifest, exit 2 without writing the receipt, and repeat that on every
    # later run -- the download was never wrong, but it was never recorded either.
    plans: list[tuple[str, dict, list[PlannedFile]]] = []
    try:
        for name in _MANIFESTS_BY_CORPUS[args.corpus]:
            manifest_path = data_root / name
            manifest = load_manifest(manifest_path)
            plans.append((name, manifest, plan_manifest(manifest, manifest_path)))
    except FetchError as exc:
        print(f"FAILED  {exc}", file=sys.stderr)
        return 2

    for name, manifest, planned in plans:
        try:
            result = fetch_planned(
                planned, data_root, receipt, concurrency=args.concurrency
            )
        except FetchError as exc:
            # An unreachable archive is a failed fetch, not a traceback. The receipt is
            # left alone: files already on disk are re-adopted on the next attempt, and
            # recording a half-corpus as the receipt would make the gap look intentional.
            print(f"FAILED  {name}: {exc}", file=sys.stderr)
            return 2
        hashes.update(result.hashes)
        mismatches.extend(result.mismatches)
        _report(name, result, len(planned))
        if manifest.get("arm") == "dev":
            dev_manifest = manifest
        if manifest.get("arm") == "pyrosdis":
            pyrosdis_manifest = manifest

    if mismatches:
        # The receipt is NOT rewritten here. Recording the hashes of a run that
        # disagreed with the receipt is how a corrupted corpus becomes the new truth.
        print("", file=sys.stderr)
        for line in sorted(mismatches):
            print(f"MISMATCH  {line}", file=sys.stderr)
        print(
            f"\n{len(mismatches)} file(s) differ from {receipt_path}. The receipt was "
            f"left untouched. Delete the offending files to re-fetch them, or say why "
            f"the corpus changed before you update the receipt.",
            file=sys.stderr,
        )
        return 1

    write_receipt(receipt_path, hashes)
    print(f"receipt: {len(hashes)} files in {receipt_path}")

    if pyrosdis_manifest is not None:
        # After the receipt and before the micro build: the conversion reads only files
        # this run has just verified, and a failure here must not cost the FIgLib bytes
        # their receipt entry.
        try:
            totals = explode_pyrosdis(data_root, pyrosdis_manifest)
        except FetchError as exc:
            print(f"FAILED  pyrosdis conversion: {exc}", file=sys.stderr)
            return 2
        for split in sorted(totals):
            counts = totals[split]
            print(
                f"pyrosdis: {split:<5} {counts['written']:6d} images written   "
                f"{counts['skipped']:6d} already on disk   "
                f"from {counts['shards']} shard(s)"
            )

    if dev_manifest is not None:
        micro = build_micro(dev_manifest, data_root)
        frames = sum(len(w["frames"]) for w in micro["windows"])
        print(
            f"micro:   {micro['n_windows']} windows "
            f"({micro['n_fire_windows']} fire, {micro['n_calibration_windows']} "
            f"non-fire), {frames} frames in {data_root / 'micro'}"
        )
        if not micro["n_calibration_windows"]:
            # Said here rather than left for bench.py, because this is the run that could
            # have produced a calibration set and did not, and it is the run whose output
            # somebody is reading. bench.py --arm micro will refuse the corpus outright.
            print(
                "WARNING  no non-fire micro window: every source sequence began at its "
                "ignition, so there is nothing to calibrate a threshold on and "
                "`bench.py --arm micro` will refuse this corpus.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    sys.exit(main())
