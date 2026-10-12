"""Build the CONFUSE arm (45-window quota; 33 realised, see manifest_confuse.json): the false-alarm denominator the threshold is solved on.

A confusion set is not a bag of pretty non-fires. It is the *denominator* of the operating
point -- ``rank.solve_threshold`` bisects on this arm until the realised rate hits 2.0
alerts per camera-night, and every headline in the project is quoted at that point. So the
only interesting question about this arm is whether it is hard enough to be worth
calibrating on, and that question has three parts:

**1. Same cameras.** Every CONFUSE window sits on a camera that appears in
``manifest_hold.json``. A confusion set drawn from other cameras would calibrate the
threshold against a different lens, a different horizon and a different exposure schedule
from the one it is then applied to -- a straw man, and ``test_confuse_shares_cameras``
fails on it.

**2. Same hour.** Each window is centred within +/-45 minutes of the local time of day of
its camera's paired HOLD ignition. Marine layer at 06:00 and marine layer at 16:00 are
different physical objects with different illumination; matching the clock is what makes
"one camera-night" the same unit in both arms.

**3. Fog by a CALIBRATED rule, not by an assumed one.** The six confuser predicates decide
what this arm is made of, so the one that matters -- fog is 40% of it -- cannot be a pair of
cutlines somebody thought sounded right. An earlier revision's were exactly that
(``edge_energy <= 18.0 and sky_ground_ratio <= 1.05``) and they admit **zero** of the 60
frames a human hand-labelled as marine layer in ``spike/urls_fog.txt``. The fix was not to
move them until 18 windows appeared -- that is fitting the corpus to the quota -- but to
derive a predicate from that labelled set and report what it measures.
``calibrate_fog_predicate`` does it: five candidate statistics on the 60 labelled frames
against a camera- and clock-matched background sample drawn from the same six cameras,
ranked by AUC with a bootstrap interval. ``deck_fraction`` wins at 0.706 [0.609, 0.798];
``sky_smooth_fraction`` fails and is dropped; the two old ones are reported and discarded.
Each cutline is then the MEDIAN of the labelled class -- a rule fixed before the values were
read. The whole record, background frames included, goes into the manifest.

**And the limitation, stated here rather than discovered later:** a predicate calibrated on
60 frames from 6 cameras and applied to 45 is a SCREENING RULE, not a fog detector. At its
cutline it admits half the labelled fog class and a quarter of the background -- an
enrichment of about 2x. This arm is an adversary set. No window in it is certified fog.

**4. Hard, and hard by a stated measurement.** Picking which fog is the *difficult* fog is
the actual intellectual work here, and it is done by computation rather than by eye: every
candidate is ranked by ``statistic.s3_diffuse_boundary``'s raw value on its own largest
moving region -- the exact statistic that is supposed to reject fog, applied to fog. The
CONFUSE arm is therefore adversarially selected against this project's own discriminator,
which makes the solved threshold conservative and the published rate honest. The measured
value is written into the manifest beside every window, so the selection can be disputed
without re-reading the archive.

How the archive is walked, and why it is not walked
---------------------------------------------------
The MTA is ``https://cdn.hpwren.ucsd.edu/MTA/<camera>/large/<YYYYMMDD>/Q<n>/<epoch>.jpg``,
where the **file name** is a UTC epoch but the **directory** is San Diego local: the date
is the local date and ``Q<n>`` is the local three-hour block numbered from 1 at local
midnight, so n runs 1..8. See ``quarter_directory`` -- the 60 URLs in
``spike/urls_fog.txt`` all sit in the one band where the older UTC six-hour reading
happened to give the same answer, so they could not have caught this. The bucket has no
directory index: it answers ``AccessDenied`` to a bare directory, to ``?list-type=2`` and
to ``index.html`` alike, and the origin host 404s. So enumeration cannot list; it probes.
Frames land on a 60 s grid at a per-camera phase -- one camera writes 31 s past the
minute, another 55 s -- so the first slot of a window is searched across a whole minute to
discover that phase and every later slot is searched only within ``JITTER_SECONDS`` of it,
carrying the phase forward as it drifts. Some cameras in the HOLD roster are absent from
the MTA altogether, which is why a candidate window is cheaply screened on ``PROBE_SLOTS``
frames before anything commits to enumerating all 81. On a camera that IS present the
minute grid is complete: an 81-frame window on hp-s-mobo-c enumerated 81 of 81.

Degrading honestly
------------------
This script writes ``data/manifest_confuse.json`` once, at the end, or not at all. There is
no partial manifest: a half-built CONFUSE arm would still load, still solve a threshold and
still print a number, and that number would be wrong in a way nothing downstream could see.
An unreachable archive and an EMPTY selection both exit non-zero with the reason.

A selected window the archive cannot produce all 81 minutes of is neither shortened nor
fatal: it is dropped, its class is re-filled from the next-hardest candidate in the same
ranked pool, and the rejection is listed under ``incomplete_enumerations``. Shortening it
would make one row a shorter camera-night than the rest and quietly change what "per
camera-night" means; aborting spends a two-hour sweep on one absent JPEG. Nothing about
the predicate moves either way, which is why this is a rejection list and not a shortfall.

A class that comes up SHORT is different, and the policy changed: it is reduced, and the
deficit and its reason are written into the manifest. Raising there instead sounds stricter
and is not -- the only way to make a raising run succeed is to loosen a predicate until the
quota falls out, which is precisely the fabrication this file was rewritten to stop. A
42-window arm whose composition is stated is a denominator a reader can reason about. Quote
``n_windows`` and ``class_histogram``; ``class_quotas`` is the request, not the arm.

Usage::

    python scripts/build_confusion_set.py --calibrate --end-date 2026-09-01   # the derivation
    python scripts/build_confusion_set.py --end-date 2026-09-01               # the arm
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import pathlib
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import cv2
import numpy as np

# `python scripts/foo.py` puts scripts/ on sys.path, not the repo root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPTS = pathlib.Path(__file__).resolve().parent
for _path in (str(_REPO_ROOT), str(_SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# fetch.py owns the HTTP client and the polite-client settings. Importing it rather than
# copying them keeps the User-Agent and the retry policy in one place; the build contract
# fixes scripts/ at six files, so there is no seventh module for them to live in.
from fetch import (  # noqa: E402
    RETRY_ATTEMPTS,
    RETRY_BACKOFF_SECONDS,
    USER_AGENT,
    FetchError,
    fetch_with_retry,
)

from plumerank.constants import (  # noqa: E402
    CC_CONNECTIVITY,
    EPS,
    FRAMES_PER_WINDOW,
    IGNITION_FRAME_INDEX,
    MIN_COMPONENT_AREA_PX,
    MIN_COMPONENT_EXTENT_PX,
    MORPH_CLOSE_KERNEL,
    MORPH_OPEN_KERNEL,
    SECONDS_PER_FRAME,
    TRAIN_SEED,
    WORK_HEIGHT,
    WORK_WIDTH,
)
from plumerank.statistic import s3_diffuse_boundary  # noqa: E402
from plumerank.types import Component  # noqa: E402

__all__ = [
    "CLASS_QUOTAS",
    "Candidate",
    "Measurement",
    "background_sample",
    "build",
    "calibrate_fog_predicate",
    "classify",
    "deck_fraction",
    "enumerate_window",
    "frame_features",
    "frame_path",
    "furniture_mask",
    "hardness",
    "labelled_fog_frames",
    "local_minutes_of_day",
    "main",
    "measure",
    "minutes_apart",
    "paired_hold_windows",
    "probe_exists",
    "roi",
    "sky_greyness",
    "static_sky_mask",
]

# ---------------------------------------------------------------------------------
# The archive
# ---------------------------------------------------------------------------------

MTA_BASE_URL: str = "https://cdn.hpwren.ucsd.edu/MTA/"

#: HPWREN's cameras are in San Diego County. Every "time of day" in this file is local
#: there, because the thing being matched is the sun angle, not a UTC offset.
PACIFIC = ZoneInfo("America/Los_Angeles")

CREDIT: str = "Imagery: HPWREN, http://hpwren.ucsd.edu/"

# ---------------------------------------------------------------------------------
# Search budget
# ---------------------------------------------------------------------------------
#
# These govern how much of somebody else's archive this one-off script reads. They are not
# pipeline numbers -- nothing at inference time consults them -- but they do decide which
# windows exist, so every one of them is written into the manifest under
# ``search_budget``. results/constants.sha256 freezes what the detector computes;
# manifest_confuse.json freezes what the corpus is, and the two are different artefacts.

#: A camera writes on a 60 s grid at ITS OWN phase -- one camera lands 31 s past the
#: minute, another 55 s -- and a window's predicted epochs are built from a local wall
#: clock, which knows nothing about that. So the FIRST slot of every window is searched
#: across a whole minute to discover the phase; searching only the jitter would find
#: nothing on most cameras and this script would report an empty archive.
PHASE_SEARCH_SECONDS: int = 30

#: Seconds either side of a predicted epoch, once the phase is known. Measured on the live
#: archive: a missing slot is usually a frame one or two seconds off the grid rather than
#: an absent frame, and the phase drifts slowly across a day.
JITTER_SECONDS: int = 3

#: Slots probed to screen a candidate before committing to the full 81. Spread across the
#: window so a hole at either end is seen.
PROBE_SLOTS: int = 7
PROBE_MIN_PRESENT: int = 5

#: Sockets held open against the CDN while probing. Measured on the live archive: a HEAD
#: costs ~0.6 s of round trip whether or not the connection is reused -- keep-alive changes
#: nothing, so the walk is latency-bound and not bandwidth-bound. One request at a time
#: makes the sweep arithmetically impossible: enumerating one 81-frame window alone is 203
#: HEADs and 45 of them is the smallest thing this script does. Six is fetch.py's
#: politeness budget minus a margin, and it is a cap on a public academic service rather
#: than a tuning knob: raising it is a decision about somebody else's bandwidth, so it is
#: written into the manifest with the rest of the search budget.
MAX_PROBE_CONCURRENCY: int = 6

#: Days back from ``--end-date``, and the stride between them. Consecutive days are highly
#: correlated -- the same marine layer sits over the same camera for a week -- so the sweep
#: strides rather than taking the most recent N days.
CANDIDATE_DAYS: int = 60
CANDIDATE_DAY_STRIDE: int = 3

#: Offsets from the paired HOLD time of day, in minutes. All inside the +/-45 min the
#: contract allows, so a camera whose exact matching minute is missing from the archive
#: still has somewhere to go.
TOD_OFFSETS_MINUTES: tuple[int, ...] = (0, -30, 30)

#: Candidates measured per camera before giving up on it. Each one costs PROBE_SLOTS
#: probes plus up to PROBE_SLOTS image GETs.
MAX_CANDIDATES_PER_CAMERA: int = 6

# ---------------------------------------------------------------------------------
# The six classes, and the measurement that assigns each one
# ---------------------------------------------------------------------------------

CLASS_QUOTAS: dict[str, int] = {
    "fog_marine": 18,
    "dust": 8,
    "cloud_shadow": 7,
    "exposure_hunting": 5,
    "low_cloud": 4,
    "lens_artefact": 3,
}

#: Every threshold below is on the 0-255 luma scale at WORK_WIDTH x WORK_HEIGHT, inside the
#: ROI crop. Two of them -- the fog and low-cloud cutlines -- are CALIBRATED on the 60
#: hand-labelled marine-layer frames in ``spike/urls_fog.txt`` against a matched background
#: sample this file draws itself; see ``calibrate_fog_predicate``. The rest are a priori,
#: because there is no labelled set to fit them on, and the manifest says which is which
#: rather than presenting them all as though they had the same standing. Every feature
#: they are compared against is recorded per window, so a reader who disagrees with a
#: cutline can re-derive the whole assignment from the manifest without re-reading the CDN.

#: Fraction of the frame trimmed from each edge before ANY measurement is taken.
#:
#: This crop is not cosmetic. It is one of the two things the previous revision got wrong --
#: the other being the fog statistic itself -- and it is the whole of what was wrong with
#: lens_artefact, which fired on 88 of 108 candidates. An HPWREN frame carries a burned-in
#: caption bar of yellow monospace text across the top, an ALERT California mark in the top
#: left, an HPWREN mark in the bottom right, and -- on every camera in the roster -- a hard
#: vignette ring where the dome cuts the corners. All four are the highest-gradient objects
#: in the image and all four are in EVERY frame the camera has ever written, so a whole-frame
#: edge-energy statistic measures the caption's typeface and a whole-frame "static structure"
#: statistic fires on the vignette. 10% is measured, not guessed: the caption occupies the
#: top ~2.5% of the height and the marks and the vignette shoulder are inside 10% on all six
#: calibration cameras.
ROI_MARGIN_FRACTION: float = 0.10

#: Lens artefact. A fixed camera's sky is the one region that should carry no persistent
#: structure, so structure in the SKY that does not move over the window is on the glass --
#: EXCEPT that a tower, a mast or a vignette shoulder is also structure in the sky that does
#: not move, and those are scene furniture rather than an artefact of this window. So the
#: static mask is differenced against a per-camera FURNITURE reference built from the other
#: candidate windows on the same camera, days or weeks apart (see ``furniture_mask``):
#: anything that persists across a camera's whole sweep is the camera, and what is left is
#: what appeared. Clouds fail the static test because clouds move; the radio tower on
#: bl-n-mobo-c fails the novelty test because it is always there.
LENS_SKY_GRADIENT_MIN: float = 12.0
LENS_STATIC_TEMPORAL_STD_MAX: float = 2.0

#: Fraction of a camera's candidate windows a static-sky pixel must appear in before it is
#: called furniture. A half is the weakest claim that still means "usually": a droplet that
#: sat on the glass for one morning is in one window of six and survives as novel, while the
#: tower is in all six and does not.
FURNITURE_PRESENCE_FRACTION: float = 0.5

#: a priori, and the class most likely to come up short. There is no labelled set of lens
#: artefacts anywhere in this project, so this cutline cannot be calibrated the way the fog
#: cutline is, and the manifest records the shortfall rather than the cutline being walked
#: down until three windows appear.
LENS_NOVEL_STRUCTURE_MIN: float = 0.02

#: Auto-exposure hunting is a whole-frame brightness excursion that REVERSES: the shutter
#: overshoots, then corrects. The magnitude alone would not do -- probe frames are ~13 min
#: apart, and over thirteen minutes a rising sun moves whole-frame luma by more than this
#: while a marine layer sits under it, so a plain max-step test labels dawn fog as a broken
#: camera and spends the fog quota on it. Every atmospheric and solar process is monotone
#: over the span of one window; only the instrument goes up and then back down.
EXPOSURE_REVERSAL_MIN: float = 8.0

#: Dust is an obscuring layer that is brown. Fog is an obscuring layer that is grey. The
#: red-minus-blue difference and the saturation separate them; nothing else does.
DUST_WARMTH_MIN: float = 8.0
DUST_SATURATION_MIN: float = 40.0

#: Sky, horizon and ground are the top, middle and bottom third of the ROI. Thirds and not
#: a detected horizon: the horizon row moves with the camera's tilt and its own weather, and
#: a horizon detector that fails on the exact frames this file is trying to classify would
#: put the measurement band in a different place for fog than for anything else.
SKY_GROUND_BAND_DIVISOR: int = 3

# --- The deck statistic, and what it replaced -------------------------------------
#
# The previous revision defined fog as ``edge_energy <= 18`` and ``sky_ground_ratio <=
# 1.05``. Measured on the 60 hand-labelled fog frames, the conjunction admits zero of them,
# and sky_ground_ratio alone -- which never falls below 1.46 -- admits zero on its own. The
# sky/ground claim is not merely tight, it is inverted: these are ridge-top
# cameras looking DOWN on the marine layer, so the bright thing is the sky and the dark
# thing is the near foreground under the deck, on a clear morning as much as a foggy one.
# That was a specification error and no choice of cutline repairs it, so both statistics
# were re-measured, reported, and discarded rather than moved (``calibrate_fog_predicate``
# prints the AUCs; the manifest carries them).
#
# What replaced them is the thing a marine layer actually is in these frames: a BRIGHT,
# SMOOTH surface sitting where terrain would otherwise be. ``deck_fraction`` is the
# fraction of the sub-sky part of the ROI that is both, with "bright" and "smooth" written
# as fractions of the frame's own sky reference so that the statistic survives dawn, where
# the absolute luma of everything moves by a factor of three inside one window.

#: Percentile of the sky band taken as the frame's own brightness reference. The 90th and
#: not the mean: the sky band contains the top of the terrain on the tilted cameras, and a
#: mean would let that drag the reference down and inflate every deck fraction.
SKY_REFERENCE_PERCENTILE: int = 90

#: A deck pixel is at least this fraction of the sky reference. Marine layer is lit by the
#: same sky it sits under, so it comes back bright; terrain in the same band does not.
DECK_LUMA_FRACTION: float = 0.35

#: ...and carries at most this fraction of the sky reference in local gradient. A fog bank
#: has no texture of its own; chaparral, roads and rooftops in the same rows have a great
#: deal. Both parameters were chosen from a 20-point grid on the labelled set, which is a
#: fit, so ``calibrate_fog_predicate`` reports the neighbouring grid points as well: they
#: span AUC 0.65-0.71, which is how much of the headline AUC that choice is worth.
DECK_TEXTURE_FRACTION: float = 0.08

#: Box blur applied to the gradient magnitude before the texture test. Wide enough that a
#: single JPEG block edge inside a smooth deck does not disqualify its neighbourhood,
#: narrow enough that the deck's own leading edge stays a boundary and not a smear.
DECK_TEXTURE_BLUR_PX: int = 9

#: CALIBRATED -- the median ``deck_fraction`` of the 60 hand-labelled marine-layer frames,
#: measured by ``calibrate_fog_predicate`` and pasted here unrounded. The rule is "as much
#: deck as a median hand-labelled fog morning", which is a rule and not a knob: it was
#: fixed before any candidate was screened and it is recomputed by the test suite from the
#: frames on disk. AUC against a camera- and clock-matched background sample is 0.706, 95%
#: bootstrap CI [0.609, 0.798]; at this cutline 50% of the labelled fog class passes (that
#: is what a median is) against 23.3% of the background -- an enrichment of 2.1x, not a
#: detector.
FOG_DECK_FRACTION_MIN: float = 0.12847

#: CALIBRATED the same way: the median sky greyness of the same 60 frames. Greyness is
#: ``-(B - R) / (B + R)`` over the sky band, so a saturated blue sky is strongly negative
#: and a grey ceiling is near zero. AUC 0.674, 95% CI [0.578, 0.768]; 50% of labelled fog
#: against 23.3% of background. This is the LOW CLOUD cutline and not the fog one, and the
#: honest caveat is in the manifest: the labelled set is marine layer, low cloud has no
#: labelled set anywhere in this project, and the class is defined as the residual -- a sky
#: as grey as a median marine-layer morning with no bright deck under the horizon.
LOW_CLOUD_SKY_GREYNESS_MIN: float = -0.13140

#: Blur applied to both endpoint frames before differencing them. Wide enough that JPEG
#: blocking and sensor grain do not survive into the Otsu threshold, narrow enough that a
#: fog bank's leading edge does -- the edge is the only thing S3 is then asked about.
DIFF_BLUR_KERNEL_PX: int = 5

#: Cloud shadow: the air is clear, but a large soft luminance patch has moved across the
#: scene since the first probe. Measured on a heavily blurred frame so that terrain
#: texture, which does not move, contributes nothing.
SHADOW_LOWPASS_SIGMA_PX: float = 24.0
SHADOW_MOTION_MIN: float = 6.0

#: The lowpass is computed on a decimated image and not at working resolution. A sigma-24
#: Gaussian is a ~145-tap kernel per axis, which OpenCV runs in 13 ms per frame -- 5.5 s
#: across one selection run, and the slowest thing in this file by two orders of magnitude.
#: INTER_AREA decimation IS a box lowpass, so blurring the 1/8 image with sigma/8 carries
#: the same low-frequency field at 1/60th the cost. The factor divides both WORK dimensions
#: exactly, so no rounding enters the measurement.
SHADOW_LOWPASS_DECIMATION: int = 8

#: Order matters and is not alphabetical. The two instrument faults are checked first
#: because a camera with a droplet on the lens or a hunting shutter is that, whatever the
#: weather is doing behind it; then colour separates dust from the neutral layers; then
#: fog_marine takes the windows with a bright smooth deck under the horizon and low_cloud
#: takes the grey-ceiling residual, in that order because the deck statistic is the one with
#: a labelled set behind it and the residual should be what is left over from the calibrated
#: test rather than the other way round; cloud shadow is last because it is the only class
#: defined on a CLEAR scene and therefore cannot be confused with the others.
_CLASS_ORDER: tuple[str, ...] = (
    "lens_artefact",
    "exposure_hunting",
    "dust",
    "fog_marine",
    "low_cloud",
    "cloud_shadow",
)


# ---------------------------------------------------------------------------------
# Clock arithmetic
# ---------------------------------------------------------------------------------


def local_minutes_of_day(epoch: int) -> int:
    """Minutes past local midnight in San Diego, for a unix epoch."""
    moment = datetime.fromtimestamp(int(epoch), PACIFIC)
    return moment.hour * 60 + moment.minute


def minutes_apart(a: int, b: int) -> int:
    """Circular distance between two times of day, in minutes.

    Circular and not ``abs(a - b)``: 23:50 and 00:10 are twenty minutes apart, and the
    linear answer of 1420 would silently reject a legal pre-dawn pairing.
    """
    delta = abs(a - b) % (24 * 60)
    return min(delta, 24 * 60 - delta)


def frame_epoch(frame_name: str) -> int:
    """The unix epoch a frame name carries, for either archive.

    FIgLib names are ``<epoch>_<signed offset>.jpg`` and MTA names are ``<epoch>.jpg``, so
    the leading integer is the epoch in both and the tests can re-derive every time-of-day
    claim in the manifest from the frame lists alone.
    """
    stem = pathlib.PurePosixPath(frame_name).stem
    return int(stem.split("_")[0])


#: The archive bins a camera's day into EIGHT three-hour directories, not four six-hour
#: ones, and it bins them by San Diego local time rather than by UTC. Probed against the
#: live bucket rather than assumed: on hp-s-mobo-c, 2026-08-26, one frame per local hour
#: was found at every hour of the day under ``local_date/Q(local_hour // 3 + 1)``, and the
#: Q3/Q4 edge was walked minute by minute to 08:59 -> Q3, 09:00 -> Q4.
QUARTER_HOURS: int = 3


def quarter_directory(epoch: int) -> str:
    """``<YYYYMMDD>/Q<n>`` for one epoch, in San Diego local time.

    **Corrected against the live archive.** The 60 URLs in ``spike/urls_fog.txt`` were the
    only ground truth this file originally had, and they are not enough to pin the rule:
    every one of them falls in local hours 06-08, where ``UTC hour // 6 + 1`` and
    ``local hour // 3 + 1`` happen to agree on Q3. They disagree everywhere else, and the
    disagreement is silent -- a wrong quarter answers 403, which this file's own
    ``probe_exists`` reads as "the archive does not have that frame". The first run against
    the real bucket therefore reported a sparse archive with hour-long holes and no data
    at all after local noon, all of which was this function.

    The rule that fits every hour of a probed day is local: the date directory is the
    LOCAL date, and Q is the local three-hour block numbered from 1 at local midnight, so
    Q runs 1..8 and not 1..4. Local also makes the DST transition correct for free -- the
    quarter follows the camera's own clock, which is the clock the archive names its
    directories with.
    """
    moment = datetime.fromtimestamp(int(epoch), PACIFIC)
    return f"{moment:%Y%m%d}/Q{moment.hour // QUARTER_HOURS + 1}"


def frame_path(epoch: int) -> str:
    """The MTA key under ``large/``, which is what a manifest window stores as a frame."""
    return f"large/{quarter_directory(epoch)}/{int(epoch)}.jpg"


def frame_url(camera_id: str, epoch: int) -> str:
    return MTA_BASE_URL + camera_id + "/" + frame_path(epoch)


# ---------------------------------------------------------------------------------
# Walking an archive that will not list itself
# ---------------------------------------------------------------------------------


def probe_exists(url: str, timeout: float = 15.0) -> bool:
    """Does this key exist? A HEAD, because the answer is one bit and the file is 500 KB.

    A missing key answers 403 rather than 404 on this bucket (listing is disabled, so S3
    will not confirm absence), and both are treated as "no" on the first answer -- there
    is nothing transient about them and asking again is neither useful nor polite. A 5xx
    or a transport error is NOT treated as "no": it is retried on the same exponential
    backoff ``fetch_with_retry`` uses and then raised, because reading a flaky network as
    an empty archive is precisely how this script would quietly build a smaller corpus
    than it claims.

    The retry is not decoration. A full sweep is tens of thousands of HEADs over a couple
    of hours across six sockets, and one transient DNS failure -- observed, ``[Errno 8]
    nodename nor servname provided`` -- would otherwise discard the whole run at whatever
    minute it happened to land on.
    """
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": USER_AGENT}
    )
    last: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=timeout
            ) as response:
                return 200 <= response.status < 300
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                return False
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt + 1 < RETRY_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
    raise FetchError(f"HEAD {url} failed after {RETRY_ATTEMPTS} attempts: {last}")


def _jitter_order(limit: int) -> list[int]:
    """0, -1, +1, -2, +2, ... -- the predicted epoch first, then outward."""
    order = [0]
    for step in range(1, limit + 1):
        order.extend((-step, step))
    return order


def _slot_epochs(centre_epoch: int, n_frames: int) -> list[int]:
    """The predicted epoch of every slot in a window centred on ``centre_epoch``."""
    step = int(SECONDS_PER_FRAME)
    return [
        centre_epoch + (i - IGNITION_FRAME_INDEX) * step for i in range(n_frames)
    ]


def _chunks(items: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    """``items`` in fixed-size groups, order preserved."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _search_slot(camera_id: str, base_epoch: int, width: int) -> int | None:
    """The epoch nearest ``base_epoch`` that exists within ``width`` seconds, or None.

    Sequential inside one worker and concurrent across workers, rather than the reverse:
    "nearest wins" is an ordering constraint, and a slot that hits on its first probe --
    the common case once the phase is known -- must not have paid for six more.
    """
    for delta in _jitter_order(width):
        epoch = base_epoch + delta
        if probe_exists(frame_url(camera_id, epoch)):
            return epoch
    return None


def _first_existing(camera_id: str, epochs: Sequence[int]) -> int | None:
    """The first epoch of ``epochs`` that exists, probing ahead under a bounded pool.

    Used only for the full-minute phase discovery, where ``epochs`` is 61 long and the
    answer is uniformly distributed inside it, so probing strictly one at a time spends
    thirty seconds of wall clock to save a handful of HEADs. Chunked and not fully
    speculative, so a hit in the first chunk still costs six requests and not sixty-one:
    the result is identical to the sequential scan, only the wasted tail differs.
    """
    for chunk in _chunks(epochs, MAX_PROBE_CONCURRENCY):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunk)) as pool:
            hits = list(pool.map(lambda e: probe_exists(frame_url(camera_id, e)), chunk))
        for epoch, hit in zip(chunk, hits):
            if hit:
                return epoch
    return None


def enumerate_window(
    camera_id: str,
    centre_epoch: int,
    slots: Sequence[int],
    n_frames: int = FRAMES_PER_WINDOW,
    known: dict[int, int] | None = None,
    phase: int | None = None,
    min_present: int | None = None,
) -> dict[int, int]:
    """Which of ``slots`` exist, as ``{slot index: actual epoch}``.

    Two search widths, and the difference is the whole trick. Until a frame has been found
    the camera's phase is unknown, so the slot is searched across the full minute
    (``PHASE_SEARCH_SECONDS`` either side). After that the phase is carried forward and
    each further slot costs one HEAD in the common case, with the narrow jitter search
    reserved for the frames that really are missing -- 81 full-minute searches per window
    would be 5,000 requests at somebody else's expense for information already in hand.

    ``known`` seeds that phase from a candidate's screening probes, so the full
    enumeration does not re-discover what the screen already paid for. ``phase`` seeds it
    from a DIFFERENT window on the same camera, which is legitimate because the phase is a
    property of the camera and not of the day: a predicted epoch is always a whole local
    minute and the archive's offsets are whole hours, so the phase is just the second of
    the minute each camera writes at. Measured across two months of the live archive it
    moves by one or two seconds and no more -- hp-s writes at :31 in July and at :31 in
    September -- which is well inside ``JITTER_SECONDS``. Without it every candidate day
    on the sweep re-pays the full-minute discovery, and the sweep does not finish.

    ``min_present`` abandons a candidate as soon as it cannot reach that many slots. It is
    lossless -- the abandoned candidate would have been discarded by the same number -- and
    it caps an empty camera-day at three slots instead of seven. Two of the 45 HOLD cameras
    (ml-n-mobo-c, ml-w-mobo-c) are absent from the MTA entirely, so every one of their
    sixty candidate days runs this path.
    """
    predicted = _slot_epochs(centre_epoch, n_frames)
    found: dict[int, int] = dict(known or {})
    if found:
        first = min(found)
        phase = found[first] - predicted[first]

    pending = [index for index in slots if index not in found]

    # Phase discovery. The original walked slot by slot at full width until something was
    # found, and so does this: a camera-day can be missing its first slot and hold the
    # rest.
    while phase is None and pending:
        index = pending.pop(0)
        epoch = _first_existing(
            camera_id,
            [predicted[index] + d for d in _jitter_order(PHASE_SEARCH_SECONDS)],
        )
        if epoch is not None:
            found[index] = epoch
            phase = epoch - predicted[index]
        elif min_present is not None and len(found) + len(pending) < min_present:
            return found

    while pending:
        if min_present is not None and len(found) + len(pending) < min_present:
            break
        # Never probe further into the window than the point at which the candidate could
        # already be abandoned. A screening pass allows ``len(slots) - min_present``
        # misses, so once that many have accumulated the next miss ends the candidate and
        # a full chunk of six would have bought five answers nobody reads.
        width = MAX_PROBE_CONCURRENCY
        if min_present is not None:
            width = min(width, max(1, len(found) + len(pending) - min_present + 1))
        chunk = pending[:width]
        pending = pending[width:]
        assert phase is not None  # the loop above either set it or emptied ``pending``
        carried = phase
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunk)) as pool:
            hits = list(
                pool.map(
                    lambda i: _search_slot(
                        camera_id, predicted[i] + carried, JITTER_SECONDS
                    ),
                    chunk,
                )
            )
        for index, epoch in zip(chunk, hits):
            if epoch is not None:
                found[index] = epoch
                # Carry the drift forward from the latest slot resolved, as the sequential
                # walk did. Slots are handed out in increasing order, so this is the same
                # carry to within one chunk -- and the drift over one chunk is under a
                # second on every camera measured.
                phase = epoch - predicted[index]
    return found


# ---------------------------------------------------------------------------------
# Measuring a candidate
# ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Measurement:
    """What was measured on a candidate's probe frames, and what it implies."""

    features: dict[str, float]
    label: str | None
    """The class, or ``None`` for a window that is not any recognised confuser -- an
    ordinary clear afternoon has no business in a false-alarm denominator."""
    hardness: float
    """Raw S3 on the largest moving region. Higher means this window looks more like smoke
    to the statistic that is supposed to say it is not."""


def _decode(data: bytes) -> np.ndarray:
    """Archive bytes to a working-resolution BGR frame.

    Decoded and downscaled here rather than at full 3072x2048 because the classification
    below has to see what the pipeline sees: an edge-energy threshold measured at source
    resolution would be a different number by a factor of six.
    """
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FetchError("cv2.imdecode refused a frame the archive served as an image")
    return cv2.resize(image, (WORK_WIDTH, WORK_HEIGHT), interpolation=cv2.INTER_AREA)


def _largest_moving_component(first: np.ndarray, last: np.ndarray) -> Component | None:
    """The biggest thing that changed between two greyscale frames.

    Frame differencing and not ``segment.Segmenter``: the background subtractor needs
    WARMUP_FRAMES of history and this stage has seven frames spread over 81 minutes on
    purpose. The component is only ever used to place S3's boundary band, and Otsu on a
    blurred difference puts that band on the same soft edge the subtractor would.
    """
    kernel = (DIFF_BLUR_KERNEL_PX, DIFF_BLUR_KERNEL_PX)
    delta = cv2.absdiff(
        cv2.GaussianBlur(first, kernel, 0), cv2.GaussianBlur(last, kernel, 0)
    )
    _, mask = cv2.threshold(delta, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_OPEN_KERNEL, MORPH_OPEN_KERNEL)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=CC_CONNECTIVITY
    )
    best: Component | None = None
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < MIN_COMPONENT_AREA_PX:
            continue
        if w < MIN_COMPONENT_EXTENT_PX or h < MIN_COMPONENT_EXTENT_PX:
            continue
        if best is None or area > best.area:
            best = Component(
                x=x,
                y=y,
                w=w,
                h=h,
                area=area,
                cx=float(centroids[label][0]),
                cy=float(centroids[label][1]),
            )
    return best


def hardness(frames: Sequence[np.ndarray]) -> float:
    """Raw S3 on the largest moving region -- this project's own discriminator, on a non-fire.

    Deliberately the raw statistic and not ``rank.fuse``: the fusion weights are fitted on
    DEV, so ranking the confusion set with them would make the corpus a function of a fit
    and the fit a function of the corpus. ``S3_BAND_FRACTION`` is a priori and was frozen
    before the Sep 1 gate, so selecting on raw S3 borrows nothing that was learned.

    A window with nothing moving scores 0.0, which is the truth: it cannot fool anything.
    A window whose S3 is ``nan`` is a different statement -- the band geometry did not fit
    the component -- and it is raised rather than folded into that 0.0, because ranking the
    unmeasurable alongside the harmless is exactly how the softest windows in the archive
    would end up presented as the hardest. ``MIN_COMPONENT_EXTENT_PX`` should make it
    unreachable; if it fires, the filter and the statistic have drifted apart.
    """
    if len(frames) < 2:
        return 0.0
    # Inside the ROI, like every other measurement here, and for a reason specific to this
    # one: the burned-in caption carries a wall clock that changes on every frame, so the
    # largest thing that "moved" between the first and last probe frame of an uncropped
    # window is the timestamp, and its hard glyph edges make it the softest-boundary
    # nonsense a raw-S3 ranking could possibly be handed.
    first = roi(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY))
    last = roi(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY))
    component = _largest_moving_component(first, last)
    if component is None:
        return 0.0
    _, raw = s3_diffuse_boundary(last, component)
    if math.isnan(raw):
        raise FetchError(
            f"s3_diffuse_boundary returned nan on a {component.w}x{component.h} component "
            f"that cleared MIN_COMPONENT_EXTENT_PX={MIN_COMPONENT_EXTENT_PX}. No manifest "
            f"was written."
        )
    return float(raw)


def roi(frame: np.ndarray) -> np.ndarray:
    """The frame with ``ROI_MARGIN_FRACTION`` trimmed from every edge.

    Applied to every statistic in this file without exception, including the ones that were
    not obviously affected. A crop that some measurements see and others do not would make
    the recorded feature vector describe two different images, and the manifest promises a
    reader that they can re-derive the assignment from it.
    """
    dy = int(round(frame.shape[0] * ROI_MARGIN_FRACTION))
    dx = int(round(frame.shape[1] * ROI_MARGIN_FRACTION))
    return frame[dy : frame.shape[0] - dy, dx : frame.shape[1] - dx]


def _bands(height: int) -> tuple[slice, slice, slice]:
    """Sky, horizon and ground slices of an ROI of this height."""
    third = height // SKY_GROUND_BAND_DIVISOR
    return slice(0, third), slice(third, 2 * third), slice(2 * third, height)


def _gradient_magnitude(grey: np.ndarray) -> np.ndarray:
    """|Sobel_x| + |Sobel_y|, the L1 gradient this file measures texture with.

    L1 and not the Euclidean magnitude for one reason only: it is what ``statistic.py``'s
    S3 denominator uses, and two "gradient energy" numbers in one project that are not the
    same function of the same pixels is a trap for whoever reads them side by side.
    """
    return np.abs(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(
        cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    )


def sky_reference(grey_roi: np.ndarray) -> float:
    """The frame's own brightness scale, from its sky band.

    Every deck test is written as a fraction of this rather than in absolute DN. At dawn --
    which is when marine layer exists and therefore the only hour this arm is drawn from --
    whole-frame luma moves by a factor of three inside one 81-minute window, so an absolute
    cutline classifies the clock rather than the weather.
    """
    sky, _, _ = _bands(grey_roi.shape[0])
    return float(np.percentile(grey_roi[sky], SKY_REFERENCE_PERCENTILE))


def deck_fraction(frame: np.ndarray) -> float:
    """Fraction of the sub-sky ROI that is a bright, smooth surface: the deck statistic.

    THE CALIBRATED ONE. A marine layer in a ridge-top camera is not a loss of contrast over
    the whole frame -- the near foreground stays perfectly sharp -- it is a lit, textureless
    sheet lying across the rows where terrain belongs. Measuring that directly is what makes
    this statistic separate the labelled set best of the five measured: AUC 0.706, 95% CI
    [0.609, 0.798]. Whole-frame edge energy scores 0.386 -- 0.614 with fog as the lower
    class, so it is not signal-free -- but its published cutline of 18.0 sits below 59 of
    the 60 labelled frames, which is a different failure from being weak.

    Bright and smooth are both relative to ``sky_reference``; see there for why.
    """
    grey = roi(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    reference = sky_reference(grey) + EPS
    texture = cv2.blur(
        _gradient_magnitude(grey), (DECK_TEXTURE_BLUR_PX, DECK_TEXTURE_BLUR_PX)
    )
    _, horizon, ground = _bands(grey.shape[0])
    below = slice(horizon.start, ground.stop)
    lit = grey[below] >= DECK_LUMA_FRACTION * reference
    flat = texture[below] <= DECK_TEXTURE_FRACTION * reference
    return float((lit & flat).mean())


def sky_greyness(frame: np.ndarray) -> float:
    """``-(B - R) / (B + R)`` over the sky band: 0 for a grey ceiling, negative for blue sky.

    The ratio and not the raw ``B - R`` difference, because dawn changes both channels by
    more than the weather does and the difference then measures the sun's elevation. Signed
    so that "more like a ceiling" is "larger", which is the direction every other threshold
    in this file runs in.
    """
    band = roi(frame).astype(np.float32)
    sky, _, _ = _bands(band.shape[0])
    blue = band[sky][:, :, 0]
    red = band[sky][:, :, 2]
    return float(-((blue - red) / (blue + red + EPS)).mean())


def frame_features(frame: np.ndarray) -> dict[str, float]:
    """The five single-frame statistics, computed once so calibration cannot drift from use.

    ``calibrate_fog_predicate`` measures the labelled set with this function and ``measure``
    measures a candidate window with it. If the two had their own copies -- and the obvious
    shape for this file was to leave the calibration in a notebook and paste the cutlines --
    then the number in ``FOG_DECK_FRACTION_MIN`` would be a quantile of a distribution
    nothing in the shipped pipeline computes, and the test that recomputes it from
    ``spike/frames`` would be checking one function against itself.

    Three of the five decide nothing. They are here because the manifest promises that a
    reader can re-derive the assignment AND check the discarded alternatives from the same
    record.
    """
    grey = roi(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    reference = sky_reference(grey) + EPS
    texture = cv2.blur(
        _gradient_magnitude(grey), (DECK_TEXTURE_BLUR_PX, DECK_TEXTURE_BLUR_PX)
    )
    sky, horizon, ground = _bands(grey.shape[0])
    return {
        "deck_fraction": deck_fraction(frame),
        "sky_greyness": sky_greyness(frame),
        # Discarded, and recorded so the discard can be checked: the fraction of the SKY
        # band that is smooth. It sounds like a ceiling detector and measures nothing --
        # a clear sky is smooth too, and its AUC on the labelled set is 0.489.
        "sky_smooth_fraction": float(
            (texture[sky] <= DECK_TEXTURE_FRACTION * reference).mean()
        ),
        # The two the previous revision defined fog with. Kept, never consulted.
        "edge_energy": float(_gradient_magnitude(grey).mean()),
        "sky_ground_ratio": float(grey[sky].mean() / (grey[ground].mean() + EPS)),
    }


def static_sky_mask(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Sky pixels that carry structure and do not move across ``frames``.

    Separated from ``measure`` so that the same computation can be run over one candidate
    window (what is static this morning) and over a camera's whole sweep (what is static
    always). The difference between those two is the only thing that makes "lens artefact"
    mean anything on a camera with a tower in shot.
    """
    stack = np.stack(
        [roi(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)).astype(np.float32) for f in frames]
    )
    sky, _, _ = _bands(stack.shape[1])
    mean_frame = stack.mean(axis=0)[sky]
    structured = _gradient_magnitude(mean_frame) >= LENS_SKY_GRADIENT_MIN
    still = stack.std(axis=0)[sky] <= LENS_STATIC_TEMPORAL_STD_MAX
    return structured & still


def furniture_mask(masks: Sequence[np.ndarray]) -> np.ndarray:
    """What is static in at least ``FURNITURE_PRESENCE_FRACTION`` of a camera's windows.

    The caption bar, the vignette shoulder and the radio tower on bl-n-mobo-c are in every
    frame that camera has ever written, so they are in every window's static mask and they
    are in this one. ``ROI_MARGIN_FRACTION`` already removes the first two; the tower is
    inside the picture and only a cross-day reference can reach it.

    A majority and not an intersection: an intersection is decided by the single window with
    the weakest static mask, so one hazy morning on which the tower's gradient falls under
    LENS_SKY_GRADIENT_MIN would erase the tower from the reference and hand it back to every
    other window as a novel artefact.
    """
    if not masks:
        raise FetchError("a furniture reference needs at least one window's static mask")
    presence = np.mean(np.stack(masks).astype(np.float32), axis=0)
    return presence >= FURNITURE_PRESENCE_FRACTION


def measure(
    frames: Sequence[np.ndarray], furniture: np.ndarray | None = None
) -> Measurement:
    """Twelve numbers per candidate window, then the class they imply.

    Nine of the twelve decide something. The other three -- ``edge_energy``,
    ``sky_ground_ratio`` and ``sky_smooth_fraction`` -- are the statistics this predicate
    was measured AGAINST and found to be worse than, and they are recorded so that the
    comparison stays checkable from the manifest rather than from a commit message.

    ``furniture`` is the camera's cross-day static-sky reference. It is optional and not
    defaulted to "no furniture" silently: when it is absent every static sky pixel counts as
    novel, which is the OLD behaviour and is wrong on any camera with a mast, so callers
    that have a camera's sweep in hand are expected to pass one and ``_score_camera`` does.
    """
    if len(frames) < 2:
        raise FetchError("a candidate needs at least two probe frames to be measured")

    greys = [roi(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]

    means = [float(g.mean()) for g in greys]
    colour = [roi(f).astype(np.float32) for f in frames]
    warmth = float(np.mean([c[:, :, 2].mean() - c[:, :, 0].mean() for c in colour]))
    saturation = float(
        np.mean([roi(cv2.cvtColor(f, cv2.COLOR_BGR2HSV))[:, :, 1].mean() for f in frames])
    )

    # The smaller leg of each direction change, so a 30-up-then-2-down wobble scores 2 and
    # not 30: a reversal is only as big as the part of it that came back.
    diffs = [means[i] - means[i - 1] for i in range(1, len(means))]
    reversal = max(
        (
            min(abs(a), abs(b))
            for a, b in zip(diffs, diffs[1:])
            if a * b < 0.0
        ),
        default=0.0,
    )

    small = (
        greys[0].shape[1] // SHADOW_LOWPASS_DECIMATION,
        greys[0].shape[0] // SHADOW_LOWPASS_DECIMATION,
    )
    lowpass = [
        cv2.GaussianBlur(
            cv2.resize(g, small, interpolation=cv2.INTER_AREA),
            (0, 0),
            SHADOW_LOWPASS_SIGMA_PX / SHADOW_LOWPASS_DECIMATION,
        ).astype(np.float32)
        for g in greys
    ]
    shadow_motion = float(
        np.mean([np.abs(lp - lowpass[0]).mean() for lp in lowpass[1:]])
    )

    static = static_sky_mask(frames)
    novel = static if furniture is None else (static & ~furniture)

    # The MEDIAN across the probe frames and not the mean. The cutline was calibrated on
    # single frames, so the window-level number has to be a location estimate of that same
    # per-frame quantity; a mean would let one blown-out sunrise frame carry a window over
    # the cutline on its own.
    per_frame = [frame_features(f) for f in frames]
    median = {
        key: float(np.median([f[key] for f in per_frame])) for key in per_frame[0]
    }

    features = {
        "luma_mean": round(float(np.mean(means)), 3),
        "deck_fraction": round(median["deck_fraction"], 5),
        "sky_greyness": round(median["sky_greyness"], 5),
        "warmth_r_minus_b": round(warmth, 3),
        "saturation_mean": round(saturation, 3),
        "luma_reversal_max": round(reversal, 3),
        "shadow_motion": round(shadow_motion, 3),
        "sky_static_structure": round(float(static.mean()), 5),
        "sky_novel_structure": round(float(novel.mean()), 5),
        # Measured, recorded, and NOT used -- the two statistics the previous revision
        # defined fog with, plus the one this revision tried and threw away. They are in
        # the record so that a reader can see for themselves that the labelled fog frames
        # do not come near the published cutlines, rather than taking this file's word.
        "edge_energy": round(median["edge_energy"], 3),
        "sky_ground_ratio": round(median["sky_ground_ratio"], 4),
        "sky_smooth_fraction": round(median["sky_smooth_fraction"], 5),
    }
    return Measurement(
        features=features, label=classify(features), hardness=hardness(frames)
    )


def classify(features: dict[str, float]) -> str | None:
    """Which confuser class these features describe, or ``None``.

    Every branch is one comparison against one named threshold, evaluated in ``_CLASS_ORDER``.
    A decision tree with tuned depths would assign these labels better and would be
    unreviewable; a judge has to be able to read this function and disagree with it.
    """
    for label in _CLASS_ORDER:
        if label == "lens_artefact":
            if features["sky_novel_structure"] >= LENS_NOVEL_STRUCTURE_MIN:
                return label
        elif label == "exposure_hunting":
            if features["luma_reversal_max"] >= EXPOSURE_REVERSAL_MIN:
                return label
        elif label == "dust":
            if (
                features["warmth_r_minus_b"] >= DUST_WARMTH_MIN
                and features["saturation_mean"] >= DUST_SATURATION_MIN
            ):
                return label
        elif label == "fog_marine":
            if features["deck_fraction"] >= FOG_DECK_FRACTION_MIN:
                return label
        elif label == "low_cloud":
            if features["sky_greyness"] >= LOW_CLOUD_SKY_GREYNESS_MIN:
                return label
        elif label == "cloud_shadow":
            if features["shadow_motion"] >= SHADOW_MOTION_MIN:
                return label
    return None


def criterion_rule(label: str) -> str:
    """The predicate that put a window in its class, as text, for the manifest."""
    return {
        "lens_artefact": (
            f"sky_novel_structure >= {LENS_NOVEL_STRUCTURE_MIN} (fraction of sky pixels "
            f"with mean gradient >= {LENS_SKY_GRADIENT_MIN} and temporal std <= "
            f"{LENS_STATIC_TEMPORAL_STD_MAX} in THIS window, and not in this camera's "
            f"cross-day furniture reference)"
        ),
        "exposure_hunting": f"luma_reversal_max >= {EXPOSURE_REVERSAL_MIN}",
        "dust": (
            f"warmth_r_minus_b >= {DUST_WARMTH_MIN} and saturation_mean >= "
            f"{DUST_SATURATION_MIN}"
        ),
        "fog_marine": (
            f"deck_fraction >= {FOG_DECK_FRACTION_MIN} -- CALIBRATED as the median of the "
            f"60 hand-labelled marine-layer frames in spike/urls_fog.txt"
        ),
        "low_cloud": (
            f"deck_fraction < {FOG_DECK_FRACTION_MIN} and sky_greyness >= "
            f"{LOW_CLOUD_SKY_GREYNESS_MIN} -- the grey-ceiling residual of the calibrated "
            f"fog test; see calibration.low_cloud_note"
        ),
        "cloud_shadow": (
            f"deck_fraction < {FOG_DECK_FRACTION_MIN} and sky_greyness < "
            f"{LOW_CLOUD_SKY_GREYNESS_MIN} and shadow_motion >= {SHADOW_MOTION_MIN}"
        ),
    }[label]


# ---------------------------------------------------------------------------------
# Calibration -- where the two fog cutlines come from
# ---------------------------------------------------------------------------------
#
# The previous revision of this file could not fill the fog quota, and the correct reading
# of that was not "the cutlines are a little tight". Measured with this file's own
# statistics on the 60 hand-labelled marine-layer frames, edge_energy runs 12.6-60.1 against
# a cutline of <= 18.0 (1 frame of 60 clears it) and sky_ground_ratio runs 1.46-11.27
# against <= 1.05 (0 of 60): the conjunction that defined fog_marine admits ZERO of the 60
# frames a human confirmed as fog. The audit that first caught this measured the uncropped
# frame over whole windows and got 31.3-49.5 and 3.8-6.7; the crop moves the numbers and
# does not move the conclusion. Walking either cutline until 18 fog windows fell out would
# have been fitting the corpus to the quota, so the predicate was re-derived instead, from
# the one labelled set this project actually owns.
#
# The positive class is spike/urls_fog.txt: 60 frames a human picked and visually confirmed
# as marine layer, on six cameras. The negative class is drawn by this file, and it is NOT
# called "non-fog": it is the same six cameras at the SAME local minute of day on other
# days, which is exactly the population the candidate sweep below draws from. Some of it is
# genuinely foggy -- June on a coastal ridge usually is -- and that is why it is called a
# background sample and why every number measured against it is a LOWER bound on the
# predicate's separation rather than an estimate of it.

#: Day offsets tried, in order, when drawing a background frame for a labelled fog frame.
#: Cycled by index so the background spans the season instead of piling up at -7, where the
#: same weather system is usually still overhead.
CALIBRATION_DAY_OFFSETS: tuple[int, ...] = (
    -35, -28, -21, -14, -7, 7, 14, 21, 28, 35, -42, -49, 42, -56, -63,
)

#: How many of those offsets are cycled through as starting points before the list simply
#: continues. Ten, so that the ten "close" offsets are the ones a frame starts from and the
#: longer ones are the fallback for a camera with a gap in its archive.
CALIBRATION_OFFSET_CYCLE: int = 10

#: Bootstrap resamples behind the reported AUC interval. Enough that the 2.5/97.5
#: percentiles are stable to the third decimal, which is one more than is quoted.
CALIBRATION_BOOTSTRAP: int = 2000


def _auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Rank AUC with tie correction, written out rather than pulled from a library.

    ``requirements.lock`` is closed (build contract rule 6) and this is the whole of what
    sklearn would have been imported for. Ties are averaged rather than broken arbitrarily
    because ``deck_fraction`` is a fraction of a pixel count and genuinely ties at 0.0 on
    frames with no deck at all -- breaking those by list order would move the AUC by more
    than the effect being measured.
    """
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        raise FetchError("AUC needs at least one sample in each class")
    values = np.concatenate([pos, neg])
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.size:
        stop = start
        while stop + 1 < sorted_values.size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        if stop > start:
            ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    n_pos, n_neg = pos.size, neg.size
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _auc_interval(
    positive: Sequence[float], negative: Sequence[float], seed: int
) -> tuple[float, float]:
    """95% percentile-bootstrap interval on the AUC, resampling both classes.

    Both classes and not just the positive one: 60 against 60 is small enough that the
    background's own composition is a real source of the uncertainty, and an interval that
    holds the negatives fixed would be narrower than the thing it claims to describe --
    the same mistake caught in the Sep 1 go/no-go's first draft.
    """
    rng = np.random.default_rng(seed)
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    draws = [
        _auc(rng.choice(pos, pos.size), rng.choice(neg, neg.size))
        for _ in range(CALIBRATION_BOOTSTRAP)
    ]
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def labelled_fog_frames(spike_root: pathlib.Path) -> list[tuple[str, int, pathlib.Path]]:
    """``(camera, epoch, path)`` for every frame in ``spike/urls_fog.txt``.

    Read from the URL list rather than from ``ls spike/frames``, because that directory also
    holds the 60 smoke frames of the Sep 1 gate and they are named by the same epoch scheme.
    The URL list is the label.
    """
    listing = spike_root / "urls_fog.txt"
    if not listing.is_file():
        raise FetchError(f"{listing} is missing; it is the labelled set the fog cutline comes from")
    out: list[tuple[str, int, pathlib.Path]] = []
    for line in listing.read_text().split():
        url = line.strip()
        if not url:
            continue
        camera = url.split("/MTA/", 1)[1].split("/large/", 1)[0]
        epoch = int(pathlib.PurePosixPath(url).stem)
        path = spike_root / "frames" / f"{epoch}.jpg"
        if not path.is_file():
            raise FetchError(f"{path} is missing; the labelled fog pixels are not on disk")
        out.append((camera, epoch, path))
    return out


def background_sample(
    labelled: Sequence[tuple[str, int, pathlib.Path]], end: date
) -> list[dict]:
    """One matched background frame per labelled fog frame: same camera, same local minute.

    Same camera and same minute because those are the two confounds that would otherwise
    explain any separation found -- six cameras with six different lenses and horizons, and
    a dawn hour in which absolute brightness triples inside twenty minutes. Different DAY
    because the day is the only thing left to vary, and the weather is what varies with it.

    Days on which that camera has a labelled fog frame are skipped, so a background frame is
    never drawn from a morning already known to be foggy. Nothing else is excluded, and
    nothing here claims the result is fog-free.
    """
    fog_days: dict[str, set[date]] = {}
    for camera, epoch, _ in labelled:
        fog_days.setdefault(camera, set()).add(
            datetime.fromtimestamp(epoch, PACIFIC).date()
        )

    phase: dict[str, int] = {}
    drawn: list[dict] = []
    for index, (camera, epoch, _) in enumerate(labelled):
        local = datetime.fromtimestamp(epoch, PACIFIC)
        minutes = local.hour * 60 + local.minute
        rotation = index % CALIBRATION_OFFSET_CYCLE
        offsets = CALIBRATION_DAY_OFFSETS[rotation:] + CALIBRATION_DAY_OFFSETS[:rotation]
        for offset in offsets:
            day = local.date() + timedelta(days=offset)
            if day > end or day in fog_days[camera]:
                continue
            centre = candidate_centre_epoch(day, minutes, 0)
            if camera in phase:
                found = _search_slot(camera, centre + phase[camera], JITTER_SECONDS)
            else:
                found = _first_existing(
                    camera,
                    [centre + d for d in _jitter_order(PHASE_SEARCH_SECONDS)],
                )
                if found is not None:
                    phase[camera] = found - centre
            if found is not None:
                drawn.append(
                    {
                        "camera_id": camera,
                        "epoch": found,
                        "day_offset": offset,
                        "frame": frame_path(found),
                        "paired_fog_epoch": epoch,
                    }
                )
                break
    if len(drawn) < len(labelled) // 2:
        raise FetchError(
            f"the archive answered for only {len(drawn)} of {len(labelled)} background "
            f"draws; a cutline calibrated against that is calibrated against a hole in the "
            f"CDN. No manifest was written."
        )
    return drawn


def calibrate_fog_predicate(spike_root: pathlib.Path, end: date) -> dict:
    """Measure every candidate fog statistic on the labelled set, and report what separates.

    This is the derivation of ``FOG_DECK_FRACTION_MIN`` and ``LOW_CLOUD_SKY_GREYNESS_MIN``,
    and it returns the record that goes into the manifest verbatim. It is a function and not
    a comment because the alternative -- a cutline whose provenance is a sentence -- is
    exactly the thing this arm exists to be checkable about.

    The cutline rule is fixed before the numbers are looked at: **the median of the labelled
    fog class.** Not the quantile that makes 18 windows appear, and not the point that
    maximises anything -- a median is a rule a reader can restate in six words and check in
    one line, and it cannot be walked.
    """
    labelled = labelled_fog_frames(spike_root)
    drawn = background_sample(labelled, end)

    positive = [
        frame_features(_decode(path.read_bytes())) for _, _, path in labelled
    ]
    negative = [
        frame_features(
            _decode(fetch_with_retry(frame_url(row["camera_id"], row["epoch"])))
        )
        for row in drawn
    ]

    statistics: dict[str, dict] = {}
    for key in positive[0]:
        pos = [f[key] for f in positive]
        neg = [f[key] for f in negative]
        area = _auc(pos, neg)
        low, high = _auc_interval(pos, neg, TRAIN_SEED)
        cut = float(np.median(pos))
        statistics[key] = {
            "auc_fog_over_background": round(area, 4),
            "auc_95pc_bootstrap_ci": [round(low, 3), round(high, 3)],
            "separates": bool(low > 0.5 or high < 0.5),
            "fog_min_median_max": [
                round(float(np.min(pos)), 5),
                round(cut, 5),
                round(float(np.max(pos)), 5),
            ],
            "background_min_median_max": [
                round(float(np.min(neg)), 5),
                round(float(np.median(neg)), 5),
                round(float(np.max(neg)), 5),
            ],
            "median_of_labelled_fog": round(cut, 5),
            "background_fraction_at_or_above_that_median": round(
                float(np.mean(np.asarray(neg) >= cut)), 4
            ),
        }

    return {
        "positive_class": {
            "source": "spike/urls_fog.txt",
            "n_frames": len(labelled),
            "cameras": sorted({camera for camera, _, _ in labelled}),
            "label": (
                "hand-picked and visually confirmed as marine-layer fog by a human before "
                "this file existed. It is the only labelled fog in the project."
            ),
        },
        "background_class": {
            "n_frames": len(drawn),
            # Draws are made per labelled fog frame and are NOT deduplicated: two labelled
            # frames on the same camera at the same local minute on different days can
            # resolve to the same background frame, and on the shipped record two of them
            # do. Recorded because the bootstrap interval treats the negative class as that
            # many independent draws, and where this number is below n_frames the interval
            # is very slightly narrower than the sample it describes. Recomputable from
            # background_class.frames by anyone who doubts it.
            "n_distinct_frames": len({(row["camera_id"], row["epoch"]) for row in drawn}),
            "rule": (
                "For each labelled fog frame, one frame from the SAME camera at the SAME "
                "local minute of day on a different day, day offsets tried in the order "
                f"{list(CALIBRATION_DAY_OFFSETS)} rotated by the frame's index modulo "
                f"{CALIBRATION_OFFSET_CYCLE}, skipping any day that camera has a labelled "
                "fog frame on."
            ),
            "not_a_negative_class": (
                "This sample is UNLABELLED. It is the population the candidate sweep draws "
                "from, at dawn on coastal ridge cameras in June to August, and a large "
                "share of it is genuinely foggy. Every AUC below is therefore a LOWER "
                "bound on the statistic's separation, not an estimate of it."
            ),
            "frames": drawn,
        },
        "cutline_rule": (
            "Each cutline is the MEDIAN of the labelled fog class for that statistic -- "
            "fixed as a rule before the values were read, so that no cutline could be "
            "walked until a class quota fell out."
        ),
        "statistics": statistics,
        "chosen": {
            "fog_marine": "deck_fraction",
            "low_cloud": "sky_greyness",
        },
        "discarded": {
            "edge_energy": (
                "The previous revision's fog test, at edge_energy <= 18.0. Not one of the "
                "60 labelled fog frames reaches it. It does carry some signal in the "
                "opposite bookkeeping -- AUC 0.386, which is 0.614 with fog as the lower "
                "class -- but it is whole-frame, so on these cameras it is dominated by "
                "near-field terrain that a marine layer never touches, and no cutline "
                "reachable by the labelled set is reachable by anything else either. "
                "Discarded, not retuned."
            ),
            "sky_ground_ratio": (
                "The previous revision's fog/low-cloud split, at <= 1.05. The a priori "
                "claim was that a marine layer lifts the ground to the sky's brightness. "
                "These are ridge-top cameras looking DOWN on the layer: the sky is the "
                "bright thing and the near foreground under the deck is the dark thing, on "
                "a clear morning as much as a foggy one, and the labelled set never falls "
                "below 1.46. Worse than unreachable, the sign is inverted: fog measures "
                "HIGHER than the background sample (AUC 0.604), so the published rule was "
                "not merely tight, it pointed the wrong way. Discarded."
            ),
            "sky_smooth_fraction": (
                "Tried by this revision as a low-cloud ceiling detector and thrown away: a "
                "clear sky is smooth too. AUC 0.488, 95% CI [0.385, 0.599] -- the interval "
                "spans 0.5, so this statistic is not measured to separate anything and no "
                "cutline was taken from it."
            ),
        },
        "internal_parameter_note": (
            f"DECK_LUMA_FRACTION={DECK_LUMA_FRACTION} and "
            f"DECK_TEXTURE_FRACTION={DECK_TEXTURE_FRACTION} were chosen from a 20-point "
            "grid on this same labelled sample, so the deck AUC is in-sample and "
            "optimistic. Neighbouring grid points span roughly 0.65-0.71, which is the "
            "size of that optimism."
        ),
        "low_cloud_note": (
            "low_cloud has NO labelled set anywhere in this project. Its cutline is the "
            "median sky greyness of the MARINE-LAYER class, and the class is defined as the "
            "residual: a sky as grey as a median marine-layer morning, with no bright deck "
            "under the horizon. That is a defensible construction and it is not a "
            "measurement of low cloud."
        ),
        "limitation": (
            "A predicate calibrated on 60 frames from 6 cameras is a SCREENING RULE, not a "
            "fog detector, and the CONFUSE arm is an adversary set, not a ground truth. It "
            "is applied to 45 cameras, 39 of which contributed nothing to the calibration. "
            "Per-camera AUC on the six that did ranges from 0.28 to 0.86, so the rule does "
            "not transfer evenly even inside its own calibration set. Nothing downstream "
            "may describe a window in this arm as verified fog; what the arm supports is "
            "the statement that rank.solve_threshold was calibrated on non-fire windows "
            "enriched for fog, on the evaluation cameras, at the evaluation hour."
        ),
    }


# ---------------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    camera_id: str
    centre_epoch: int
    offset_minutes: int
    paired_hold_sequence_id: str
    paired_hold_t0_epoch: int

    @property
    def date(self) -> str:
        return f"{datetime.fromtimestamp(self.centre_epoch, PACIFIC):%Y%m%d}"

    @property
    def sequence_id(self) -> str:
        return f"confuse_{self.camera_id}_{self.date}"


@dataclass(slots=True)
class _Scored:
    candidate: Candidate
    measurement: Measurement
    probes: dict[int, int] = field(default_factory=dict)


def paired_hold_windows(hold_manifest: dict) -> dict[str, dict]:
    """One HOLD window per camera -- the one a CONFUSE window on that camera is paired to.

    A camera with several HOLD windows gets its earliest by sequence id, deterministically.
    Which one hardly matters for the sun angle; that the choice is a rule rather than a
    coin flip is what lets ``test_confuse_shares_cameras`` recompute the pairing.
    """
    paired: dict[str, dict] = {}
    for window in sorted(hold_manifest["windows"], key=lambda w: w["sequence_id"]):
        paired.setdefault(window["camera_id"], window)
    return paired


def hold_t0_epoch(window: dict) -> int:
    """The unix epoch of a HOLD window's ignition frame."""
    return frame_epoch(window["frames"][window["t0_index"]])


def candidate_days(end: date) -> list[date]:
    """The days swept, newest first. Reads the budget globals at call time on purpose --
    a default argument would freeze them at import and make the sweep untestable."""
    return [
        end - timedelta(days=CANDIDATE_DAY_STRIDE * i)
        for i in range(CANDIDATE_DAYS // CANDIDATE_DAY_STRIDE)
    ]


def candidate_centre_epoch(day: date, minutes_of_day: int, offset_minutes: int) -> int:
    """The epoch at which ``day`` reaches a given local time of day, plus an offset.

    Built from a localised ``datetime`` rather than by adding seconds to midnight UTC, so
    that a window straddling a DST boundary lands at the right sun angle instead of an hour
    off -- which for a marine-layer camera is the difference between fog and no fog.
    """
    total = minutes_of_day + offset_minutes
    midnight = datetime(day.year, day.month, day.day, tzinfo=PACIFIC)
    return int((midnight + timedelta(minutes=total)).timestamp())


def _figlib_fire_dates(manifests: Iterable[dict]) -> set[tuple[str, str]]:
    """``(camera, YYYYMMDD)`` pairs that FIgLib records a fire on.

    Excluded from the candidate sweep. It is a weak filter -- FIgLib catalogues 521 fires,
    not every fire -- and the manifest says so rather than claiming the arm is certified
    fire-free. The class predicates are the second filter: a plume does not look like
    marine layer to any of them.
    """
    dates: set[tuple[str, str]] = set()
    for manifest in manifests:
        for window in manifest.get("windows", []):
            dates.add((window["camera_id"], window["date"]))
    return dates


# ---------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------


def _probe_slot_indices(n_frames: int = FRAMES_PER_WINDOW, n: int = PROBE_SLOTS) -> list[int]:
    return [round(i * (n_frames - 1) / (n - 1)) for i in range(n)]


def _score_camera(
    camera_id: str,
    paired: dict,
    days: Sequence[date],
    excluded: set[tuple[str, str]],
) -> list[_Scored]:
    """Screen up to MAX_CANDIDATES_PER_CAMERA windows on one camera, then classify them.

    Two passes over the SAME frames, and the second one is the whole reason lens_artefact
    means anything. Pass one collects each surviving candidate's probe frames; between the
    passes the camera's furniture reference is built from all of them; pass two measures and
    classifies each candidate against it. Classifying inside pass one -- which is what this
    function used to do -- cannot see across days, so on bl-n-mobo-c it labelled the radio
    tower an artefact of whichever morning it happened to be looking at, and lens_artefact
    fired on 88 of 108 candidates.
    """
    t0_epoch = hold_t0_epoch(paired)
    minutes = local_minutes_of_day(t0_epoch)
    slots = _probe_slot_indices()
    collected: list[tuple[Candidate, dict[int, int], list[np.ndarray]]] = []
    #: Learned once from whichever candidate day first yields a frame, then reused for
    #: every later day on this camera. See ``enumerate_window``: the phase is the camera's
    #: second-of-the-minute, not the day's, and re-discovering it 60 times per camera is
    #: 60 full-minute searches spent on a number that does not change.
    phase: int | None = None

    for day in days:
        if len(collected) >= MAX_CANDIDATES_PER_CAMERA:
            break
        for offset in TOD_OFFSETS_MINUTES:
            if len(collected) >= MAX_CANDIDATES_PER_CAMERA:
                break
            centre = candidate_centre_epoch(day, minutes, offset)
            candidate = Candidate(
                camera_id=camera_id,
                centre_epoch=centre,
                offset_minutes=offset,
                paired_hold_sequence_id=paired["sequence_id"],
                paired_hold_t0_epoch=t0_epoch,
            )
            if (camera_id, candidate.date) in excluded:
                continue
            probes = enumerate_window(
                camera_id, centre, slots, phase=phase, min_present=PROBE_MIN_PRESENT
            )
            if probes and phase is None:
                first = min(probes)
                phase = probes[first] - _slot_epochs(centre, FRAMES_PER_WINDOW)[first]
            if len(probes) < PROBE_MIN_PRESENT:
                continue
            frames = [
                _decode(fetch_with_retry(frame_url(camera_id, probes[i])))
                for i in sorted(probes)
            ]
            collected.append((candidate, probes, frames))
            break  # one candidate per day per camera; the next offset is the same weather

    if not collected:
        return []
    # Pass two. Every candidate's static-sky mask contributes to the reference, INCLUDING
    # its own -- excluding it per candidate would make the reference a different image for
    # every window and the novelty fractions incomparable across a camera. With six windows
    # the self-contribution moves a pixel's presence by 1/6, which cannot on its own carry a
    # pixel over a half.
    furniture = furniture_mask([static_sky_mask(f) for _, _, f in collected])
    scored: list[_Scored] = []
    for candidate, probes, frames in collected:
        measurement = measure(frames, furniture=furniture)
        if measurement.label is None:
            continue
        scored.append(_Scored(candidate, measurement, probes))
    return scored


def _select(
    pool: dict[str, list[_Scored]],
    quotas: dict[str, int],
) -> tuple[dict[str, _Scored], list[dict]]:
    """Assign one camera to each slot of each class, filling as many slots as the pool allows.

    One window per camera is the constraint that makes this an assignment problem rather
    than six independent top-k lists, and the assignment is solved rather than approximated.
    The previous revision walked the classes scarcest-supply first and took each one's
    hardest unused camera. That is a greedy heuristic, and on the live sweep it measurably
    lost: 31 windows filled where the same pool supports 33, with fog_marine at 11 where 13
    were available, because five cameras that could only have supplied fog were spent on
    exposure_hunting slots that other cameras could have covered.

    Two windows is not the reason this was changed. The reason is what the shortfall then
    MEANS. A published "fog_marine short by 7" that is part supply and part the order the
    classes happened to be walked in is not a statement about the corpus, and this arm's
    whole value is that its composition is a statement about the corpus. Under a maximum
    matching a shortfall can only be supply: no assignment of these cameras to these slots
    fills more.

    Kuhn's augmenting-path algorithm, twenty lines, because that is all a 45-camera roster
    needs -- min-cost max-flow would let hardness enter the objective properly and is a
    great deal more code to review for a corpus this size. Cardinality is maximised exactly;
    hardness is preferred, not optimised, and the preference is expressed twice: each class
    offers its cameras hardest-first, and a camera contributes only its own hardest window
    in a class, since the others could never be chosen over it anyway. Classes are still
    offered slots scarcest-supply first, which no longer decides whether a class is filled
    -- the matching decides that -- but does decide which camera a class gets when several
    maximum matchings exist, and a rule is better there than an accident.

    Returns ``({sequence_id: scored}, shortfalls)``. A class the pool cannot fill is REDUCED
    and recorded, not raised on and not padded. The failure mode this arm was one audit away
    from was a cutline walked down until 18 fog windows appeared, and a rule that raises on a
    short class is exactly the pressure that produces it. A 33-window arm whose composition
    is stated is a denominator a reader can reason about; a 45-window arm with twelve windows
    admitted by a moved threshold is not.

    What the caller must then do is publish the realised composition rather than the quota.
    ``n_windows`` in the manifest is the count of windows written, ``class_quotas`` is what
    was asked for, ``class_histogram`` is what was got, and ``shortfalls`` says why they
    differ. Nothing downstream may quote the quota as if it were the arm.
    """
    by_class: dict[str, list[_Scored]] = {label: [] for label in quotas}
    for entries in pool.values():
        for scored in entries:
            label = scored.measurement.label
            if label in by_class:
                by_class[label].append(scored)
    for label in by_class:
        by_class[label].sort(
            key=lambda s: (-s.measurement.hardness, s.candidate.sequence_id)
        )

    # One entry per camera per class -- its hardest window in that class. A camera's
    # softer candidates in the same class are unreachable by construction (the camera can
    # hold at most one slot), so carrying them into the matching would only widen the
    # search.
    best: dict[str, dict[str, _Scored]] = {}
    for label, entries in by_class.items():
        hardest: dict[str, _Scored] = {}
        for scored in entries:
            hardest.setdefault(scored.candidate.camera_id, scored)
        best[label] = hardest

    order = sorted(quotas, key=lambda label: (len(by_class[label]), label))
    slots: list[str] = [label for label in order for _ in range(quotas[label])]
    held: dict[str, int] = {}

    def _augment(slot: int, seen: set[str]) -> bool:
        """Give ``slot`` a camera, displacing an earlier slot onto another of its own."""
        for camera_id in best[slots[slot]]:  # hardest first
            if camera_id in seen:
                continue
            seen.add(camera_id)
            if camera_id not in held or _augment(held[camera_id], seen):
                held[camera_id] = slot
                return True
        return False

    for slot in range(len(slots)):
        _augment(slot, set())

    chosen: dict[str, _Scored] = {}
    filled: dict[str, int] = {label: 0 for label in quotas}
    for camera_id, slot in held.items():
        scored = best[slots[slot]][camera_id]
        chosen[scored.candidate.sequence_id] = scored
        filled[slots[slot]] += 1

    shortfalls: list[dict] = []
    for label in order:
        short_by = quotas[label] - filled[label]
        if not short_by:
            continue
        shortfalls.append(
            {
                "class": label,
                "quota": quotas[label],
                "selected": filled[label],
                "short_by": short_by,
                "candidates_measured_in_this_class": len(by_class[label]),
                "cameras_measured_in_this_class": len(best[label]),
                "reason": (
                    f"The sweep measured {len(by_class[label])} candidate windows that "
                    f"satisfy the {label!r} predicate, on {len(best[label])} distinct "
                    f"cameras, and one window is taken per camera. {filled[label]} slots "
                    f"were filled, and that is the MAXIMUM over every assignment of these "
                    f"cameras to these slots -- no ordering fills more, so the deficit is "
                    f"the archive's and not the selector's. The class was reduced. The "
                    f"predicate was NOT moved: see criterion_thresholds and calibration."
                ),
            }
        )
    return chosen, shortfalls


# ---------------------------------------------------------------------------------
# Building the manifest
# ---------------------------------------------------------------------------------


def _window_json(scored: _Scored, epochs: dict[int, int]) -> dict:
    candidate = scored.candidate
    centre = epochs[IGNITION_FRAME_INDEX]
    paired_minutes = local_minutes_of_day(candidate.paired_hold_t0_epoch)
    local_minutes = local_minutes_of_day(centre)
    label = scored.measurement.label
    assert label is not None  # _score_camera drops unlabelled candidates
    return {
        "sequence_id": candidate.sequence_id,
        "camera_id": candidate.camera_id,
        "date": candidate.date,
        # No ignition, ever: that is what makes this arm a false-alarm denominator.
        # types.Sequence reads a null t0_index as "nothing ignites here".
        "t0_index": None,
        "t0_offset_seconds": None,
        "centre_index": IGNITION_FRAME_INDEX,
        "n_frames": len(epochs),
        "url_prefix": MTA_BASE_URL + candidate.camera_id + "/",
        "class": label,
        "criterion": {
            "rule": criterion_rule(label),
            "hardness_raw_s3": round(scored.measurement.hardness, 4),
            "hardness_note": (
                "Raw S3 (statistic.s3_diffuse_boundary) on the largest region that moved "
                "between the first and last probe frame. Candidates were ranked by it "
                "descending, so this arm is adversarial to PlumeRank's own discriminator."
            ),
            "features": scored.measurement.features,
        },
        "paired_hold_sequence_id": candidate.paired_hold_sequence_id,
        "paired_hold_t0_epoch": candidate.paired_hold_t0_epoch,
        "paired_hold_local_minutes": paired_minutes,
        "local_minutes": local_minutes,
        "local_minutes_delta": minutes_apart(local_minutes, paired_minutes),
        "frames": [frame_path(epochs[i]) for i in sorted(epochs)],
    }


def build(
    hold_manifest: dict,
    excluded: set[tuple[str, str]],
    end: date,
    quotas: dict[str, int] = CLASS_QUOTAS,
    calibration: dict | None = None,
) -> dict:
    """Screen, select, enumerate and assemble. Raises rather than returning something partial.

    Selection and enumeration interleave: a window the archive cannot deliver as 81
    consecutive minutes is dropped and its class re-filled from the same ranked pool. See
    the comment on the loop for why that is neither a shortfall nor a shortened window.

    ``calibration`` is the record ``calibrate_fog_predicate`` returned. It is required in
    practice and optional in the signature only so the selection machinery stays testable
    without the network; a manifest written without it says so in place of the record, and
    ``tests/test_confusion_set.py`` fails on the shipped file if it is absent.
    """
    paired = paired_hold_windows(hold_manifest)
    days = candidate_days(end)

    pool: dict[str, list[_Scored]] = {}
    for camera_id in sorted(paired):
        pool[camera_id] = _score_camera(camera_id, paired[camera_id], days, excluded)

    # Selection and enumeration are one loop, and that is the correction this function
    # most needed. A candidate is screened on PROBE_SLOTS frames and ranked on its
    # hardness; whether the other 74 minutes exist in the bucket is not known until the
    # full enumeration runs, and on the live MTA a few windows are missing a single minute
    # -- confuse_vo-e-mobo-m_20260901 came back 80 of 81. The three ways out of that are
    # not equal:
    #
    #   * shorten the window -- refused. It would be a shorter camera-night than every
    #     other row in the arm, and "alerts per camera-night" would stop being one unit.
    #   * abort the run -- what this function used to do, and it spends a two-hour sweep
    #     and several thousand of somebody else's HEAD requests on one absent JPEG.
    #   * reduce the class -- honest, but it pays a whole quota slot for a CDN gap rather
    #     than for anything about the corpus, and ``shortfalls`` would then mean two
    #     different things in one file.
    #
    # So the window is dropped and its class is re-filled from the same ranked pool: the
    # next-hardest candidate the archive can actually deliver whole. No threshold moves,
    # the ranking is untouched, and every rejection is written into the manifest under
    # ``incomplete_enumerations`` -- a reader can see exactly how many rows the archive,
    # rather than the predicate, decided. The loop terminates because ``rejected`` only
    # grows and each pass either accepts every window or rejects one.
    enumerated: dict[str, dict[int, int]] = {}
    incomplete: list[dict] = []
    rejected: set[str] = set()
    while True:
        available = {
            camera_id: [
                scored
                for scored in entries
                if scored.candidate.sequence_id not in rejected
            ]
            for camera_id, entries in pool.items()
        }
        chosen, shortfalls = _select(available, quotas)
        short: tuple[str, int] | None = None
        for sequence_id in sorted(chosen):
            if sequence_id in enumerated:
                continue
            scored = chosen[sequence_id]
            epochs = enumerate_window(
                scored.candidate.camera_id,
                scored.candidate.centre_epoch,
                list(range(FRAMES_PER_WINDOW)),
                known=scored.probes,
            )
            if len(epochs) != FRAMES_PER_WINDOW:
                short = (sequence_id, len(epochs))
                break
            enumerated[sequence_id] = epochs
        if short is None:
            break
        sequence_id, found = short
        rejected.add(sequence_id)
        incomplete.append(
            {
                "sequence_id": sequence_id,
                "class": chosen[sequence_id].measurement.label,
                "frames_found": found,
                "frames_required": FRAMES_PER_WINDOW,
                "reason": (
                    "The archive does not hold every minute of this window. It was "
                    "dropped and its class re-filled from the next-hardest candidate; "
                    "the window was NOT shortened and no threshold was moved."
                ),
            }
        )

    windows = [
        _window_json(chosen[sequence_id], enumerated[sequence_id])
        for sequence_id in sorted(chosen)
    ]

    if not windows:
        raise FetchError(
            "the sweep selected no windows at all. A class that comes up short is reduced "
            "and recorded, but an EMPTY arm is not a smaller denominator, it is no "
            "denominator: solve_threshold would bisect on nothing and return a bound. No "
            "manifest was written."
        )

    expected = sum(quotas.values()) - sum(s["short_by"] for s in shortfalls)
    if len(windows) != expected:
        raise FetchError(
            f"assembled {len(windows)} windows, expected {expected} "
            f"(quota {sum(quotas.values())} less {sum(s['short_by'] for s in shortfalls)} "
            f"recorded shortfall)"
        )

    histogram = {label: 0 for label in quotas}
    for window in windows:
        histogram[window["class"]] += 1

    # The clock record. It is computed rather than asserted because it is the caveat this
    # arm most needs and the one a reader is least likely to think of: the confusion set's
    # hour is set by HOLD, HOLD's hour is set by when fires start, and fires start in the
    # afternoon -- while the labelled set the fog predicate was calibrated on is DAWN
    # marine layer. Those two facts pull in opposite directions, and the numbers below are
    # how far apart they ended up. Nothing here is a defence of the arm; it is the
    # measurement a reader would otherwise have to reconstruct from 33 rows by hand.
    hold_hours: dict[str, int] = {}
    for window in paired.values():
        hour = local_minutes_of_day(hold_t0_epoch(window)) // 60
        hold_hours[f"{hour:02d}"] = hold_hours.get(f"{hour:02d}", 0) + 1
    early_cameras = sum(count for hour, count in hold_hours.items() if int(hour) < 10)
    # How many shipped windows clear the fog cutline but wear another class's label,
    # because an earlier predicate in _CLASS_ORDER fired first. Computed, because the
    # shortfall record says "candidate windows that satisfy the 'fog_marine' predicate"
    # and that phrase means "and no earlier class fired" -- a reader screening on
    # deck_fraction alone would count more and be entitled to an explanation.
    above_cutline: dict[str, int] = {}
    for window in windows:
        if (
            window["class"] != "fog_marine"
            and window["criterion"]["features"]["deck_fraction"] >= FOG_DECK_FRACTION_MIN
        ):
            above_cutline[window["class"]] = above_cutline.get(window["class"], 0) + 1

    early_fog = sum(
        1
        for window in windows
        if window["class"] == "fog_marine" and window["local_minutes"] < 10 * 60
    )

    return {
        "arm": "confuse",
        "n_windows": len(windows),
        "generated": date.today().isoformat(),
        "source": "HPWREN medium-term archive (MTA)",
        "credit": CREDIT,
        "licence_note": (
            "HPWREN imagery carries no SPDX licence and requires a credit reference. "
            "These windows contain no fire and are used ONLY as the false-alarm "
            "denominator that rank.solve_threshold is calibrated on."
        ),
        "base_url": MTA_BASE_URL,
        "url_note": (
            "Frame URL = window url_prefix + frame, with '+' percent-encoded as %2B. "
            "A frame name carries its own quarter directory because an 81-minute window "
            "can straddle two of them. The file name is a UTC epoch but the directory is "
            "San Diego local: the date is the local date and Q<n> is the local three-hour "
            "block numbered from 1 at local midnight, so n runs 1..8. Probed hour by hour "
            "against the live bucket; spike/urls_fog.txt alone cannot pin this, because "
            "all 60 of its URLs fall in the one band where a UTC six-hour reading gives "
            "the same answer."
        ),
        "local_root": "corpus/mta",
        "catalogue": (
            "The MTA publishes no directory index (the bucket answers AccessDenied to a "
            "bare directory, to ?list-type=2 and to index.html), so frames were "
            "enumerated by probing the 60 s grid with a +/-"
            f"{JITTER_SECONDS} s jitter search."
        ),
        "selection": [
            "Every window is on a camera that appears in manifest_hold.json and is "
            "centred within +/-45 min of the local time of day of that camera's paired "
            "HOLD ignition. A confusion set on other cameras or at another hour would be "
            "a straw man; tests/test_confusion_set.py and test_confuse_shares_cameras "
            "assert both from the frame epochs rather than from these fields.",
            "One window per camera, so the denominator spans the roster instead of "
            "concentrating on whichever camera happened to be foggiest.",
            "Within each class, candidates were ranked by raw S3 -- PlumeRank's own fog "
            "discriminator -- and each class offers its cameras hardest-first. The arm is "
            "adversarial to this project by construction, which makes the solved threshold "
            "conservative. Read that as a PREFERENCE and not a guarantee: see the next "
            "point.",
            "Cameras were assigned to class slots by a maximum-cardinality matching "
            "(Kuhn's augmenting paths), not by walking the classes in some order and "
            "taking each one's hardest unused camera. Cardinality is what is optimised "
            "exactly, so a class that comes up short is short on SUPPLY -- no assignment "
            "of these cameras to these slots fills more -- and the shortfall is a "
            "statement about the corpus rather than about the walk order. Hardness enters "
            "only as the order in which each class offers its cameras and as the "
            "scarcest-supply order in which slots are offered, which decides WHICH maximum "
            "matching is found when several exist. So an individual window is not "
            "guaranteed to be the hardest candidate its class had: the matching may "
            "displace a harder camera onto another of its own classes in order to fill one "
            "more slot. Where a shortfall record shows selected == "
            "cameras_measured_in_this_class, that class took every camera that could "
            "supply it and no hardness was traded away inside it.",
            "A selected window the archive could not deliver as 81 consecutive minutes was "
            "dropped and its class re-filled from the next-hardest candidate, never "
            "shortened. Every such rejection is listed under incomplete_enumerations, so "
            "the rows the CDN decided are separable from the rows the predicate decided.",
            "The fog predicate was CALIBRATED on this project's own labelled fog -- the 60 "
            "frames in spike/urls_fog.txt that a human picked and visually confirmed as "
            "marine layer -- against a camera- and clock-matched background sample drawn "
            "from the same six cameras on other days. The statistic that separates them is "
            "deck_fraction and the cutline is the median of the labelled class. The "
            "previous revision's a priori cutlines (edge_energy <= 18.0, sky_ground_ratio "
            "<= 1.05) were re-measured on that same labelled set, admitted zero of 60, and "
            "were discarded rather than moved. The full record, including the discarded "
            "statistics and the background frames, is under 'calibration'.",
            "A predicate calibrated on 60 frames from 6 cameras is a screening rule and not "
            "a fog detector, and this arm is an adversary set and not a ground truth. No "
            "window here is certified as fog; the class labels say which predicate admitted "
            "the window, and the measured features say on what evidence.",
            "Non-fire basis: candidate days exclude every (camera, date) FIgLib records a "
            "fire on, and every window additionally had to satisfy one of the six "
            "confuser predicates. FIgLib catalogues 521 fires and not every fire, so this "
            "is a filter and not a certification.",
        ],
        "class_quotas": dict(quotas),
        "class_histogram": histogram,
        "clock_note": {
            "paired_hold_t0_local_hour_histogram": dict(sorted(hold_hours.items())),
            "hold_cameras_igniting_before_10_local": early_cameras,
            "hold_cameras_total": len(paired),
            "fog_windows_selected_before_10_local": early_fog,
            "fog_windows_selected": histogram.get("fog_marine", 0),
            "note": (
                "Every window here is within +/-45 min of its camera's paired HOLD "
                "ignition, which is what stops the arm being a straw man -- and it is also "
                "the hardest constraint on the fog class. Fires ignite in the afternoon, so "
                "that is the hour HOLD fixes; coastal marine layer is a DAWN object, and it "
                "is the hour the labelled set in spike/urls_fog.txt was drawn from. The two "
                "counts above are how far apart those hours are. Read the fog_marine class "
                "accordingly: outside the early-morning windows it is not dawn marine "
                "layer, it is whatever satisfies a deck predicate calibrated on dawn marine "
                "layer at an hour marine layer has usually burnt off -- a bright, "
                "textureless surface lying where terrain belongs. That is a legitimate "
                "adversary for S3, which is what this arm is for, and it is NOT evidence "
                "about fog. Matching the clock and calling the class fog were always going "
                "to be in tension; the tension is published rather than resolved by "
                "quietly relaxing one of them."
            ),
        },
        "class_precedence_note": {
            "windows_above_the_fog_cutline_claimed_by_an_earlier_class": above_cutline,
            "note": (
                "classify() evaluates the six predicates in class_order, so 'satisfies the "
                "fog_marine predicate' means 'deck_fraction >= FOG_DECK_FRACTION_MIN AND no "
                "earlier class fired'. A reader who screens these rows on deck_fraction "
                "alone will therefore count MORE windows above the cutline than the "
                "fog_marine histogram shows, and the count above is exactly that "
                "difference, listed per class. The order is a priori and documented on "
                "_CLASS_ORDER -- a camera with a hunting shutter or a droplet on the glass "
                "is that, whatever the weather behind it is doing -- and it was NOT chosen "
                "to move windows between classes. It is recorded here because reordering it "
                "is the one remaining lever that could refill the fog class without "
                "touching a cutline, and a reader is entitled to see the size of that lever "
                "rather than discover it."
            ),
        },
        "shortfalls": shortfalls,
        "incomplete_enumerations": incomplete,
        "incomplete_enumerations_note": (
            "Windows the predicate selected and the ARCHIVE could not deliver whole. A "
            "window is 81 consecutive minutes or it is not a window, so each of these was "
            "dropped and its class re-filled from the next-hardest candidate in the same "
            "ranked pool. They are listed because the alternative -- silently taking the "
            "second choice -- would make the selection unre-derivable from the record. "
            "Distinct from shortfalls, which is a class the CORPUS could not fill."
        ),
        "composition_note": (
            "class_quotas is what was asked for and class_histogram is what the archive "
            "supplied. Where they differ, shortfalls carries the class, the deficit and the "
            "reason, and the reason is never a moved threshold: a class the sweep could not "
            "fill was REDUCED. Quote n_windows and class_histogram, never class_quotas."
        ),
        "criterion_thresholds": {
            "ROI_MARGIN_FRACTION": ROI_MARGIN_FRACTION,
            "LENS_SKY_GRADIENT_MIN": LENS_SKY_GRADIENT_MIN,
            "LENS_STATIC_TEMPORAL_STD_MAX": LENS_STATIC_TEMPORAL_STD_MAX,
            "LENS_NOVEL_STRUCTURE_MIN": LENS_NOVEL_STRUCTURE_MIN,
            "FURNITURE_PRESENCE_FRACTION": FURNITURE_PRESENCE_FRACTION,
            "EXPOSURE_REVERSAL_MIN": EXPOSURE_REVERSAL_MIN,
            "DUST_WARMTH_MIN": DUST_WARMTH_MIN,
            "DUST_SATURATION_MIN": DUST_SATURATION_MIN,
            "SKY_REFERENCE_PERCENTILE": SKY_REFERENCE_PERCENTILE,
            "DECK_LUMA_FRACTION": DECK_LUMA_FRACTION,
            "DECK_TEXTURE_FRACTION": DECK_TEXTURE_FRACTION,
            "DECK_TEXTURE_BLUR_PX": DECK_TEXTURE_BLUR_PX,
            "FOG_DECK_FRACTION_MIN": FOG_DECK_FRACTION_MIN,
            "LOW_CLOUD_SKY_GREYNESS_MIN": LOW_CLOUD_SKY_GREYNESS_MIN,
            "SHADOW_LOWPASS_SIGMA_PX": SHADOW_LOWPASS_SIGMA_PX,
            "SHADOW_LOWPASS_DECIMATION": SHADOW_LOWPASS_DECIMATION,
            "SHADOW_MOTION_MIN": SHADOW_MOTION_MIN,
            "SKY_GROUND_BAND_DIVISOR": SKY_GROUND_BAND_DIVISOR,
            "DIFF_BLUR_KERNEL_PX": DIFF_BLUR_KERNEL_PX,
            "class_order": list(_CLASS_ORDER),
            "provenance": {
                "FOG_DECK_FRACTION_MIN": (
                    "CALIBRATED on the 60 hand-labelled marine-layer frames in "
                    "spike/urls_fog.txt -- the median of that class. See calibration."
                ),
                "LOW_CLOUD_SKY_GREYNESS_MIN": (
                    "CALIBRATED the same way, on the same labelled class, which is marine "
                    "layer and not low cloud. See calibration.low_cloud_note."
                ),
                "everything_else": (
                    "a priori. Chosen from the physics of each process, never fitted, and "
                    "never moved to make a quota fall out. Of these, the lens_artefact and "
                    "cloud_shadow cutlines have no labelled set behind them at all and are "
                    "the weakest claims in this file."
                ),
            },
            "rules": {label: criterion_rule(label) for label in _CLASS_ORDER},
        },
        "calibration": calibration
        if calibration is not None
        else {
            "absent": (
                "This manifest was assembled without running calibrate_fog_predicate, so "
                "the two calibrated cutlines carry no record here. The shipped manifest is "
                "not built this way and tests/test_confusion_set.py fails if it is."
            )
        },
        "search_budget": {
            "PHASE_SEARCH_SECONDS": PHASE_SEARCH_SECONDS,
            "JITTER_SECONDS": JITTER_SECONDS,
            "PROBE_SLOTS": PROBE_SLOTS,
            "PROBE_MIN_PRESENT": PROBE_MIN_PRESENT,
            "MAX_PROBE_CONCURRENCY": MAX_PROBE_CONCURRENCY,
            "CANDIDATE_DAYS": CANDIDATE_DAYS,
            "CANDIDATE_DAY_STRIDE": CANDIDATE_DAY_STRIDE,
            "TOD_OFFSETS_MINUTES": list(TOD_OFFSETS_MINUTES),
            "MAX_CANDIDATES_PER_CAMERA": MAX_CANDIDATES_PER_CAMERA,
            "end_date": end.isoformat(),
        },
        "sha256_note": (
            "Per-file sha256 is not in this manifest and that is deliberate: hashing the "
            "frames requires downloading them, which is what scripts/fetch.py does. It "
            "writes data/fetch_receipt.json on the first fetch and verifies against it on "
            "every later run. The manifest pins WHICH frames; the receipt pins WHAT they "
            "contained."
        ),
        "windows": windows,
    }


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select 45 hard non-fire windows on the HOLD cameras at the HOLD hour and "
            "write data/manifest_confuse.json."
        )
    )
    parser.add_argument(
        "--data-root", type=pathlib.Path, default=_REPO_ROOT / "data"
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=date.today(),
        help="most recent day of the candidate sweep (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--spike-root", type=pathlib.Path, default=_REPO_ROOT / "spike"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "derive the fog cutlines from spike/urls_fog.txt and print the separation, "
            "then stop. This is the same computation the build runs; it is exposed so the "
            "derivation can be checked without a two-hour sweep."
        ),
    )
    args = parser.parse_args(argv)

    data_root: pathlib.Path = args.data_root
    out_path = data_root / "manifest_confuse.json"

    if args.calibrate:
        try:
            record = calibrate_fog_predicate(args.spike_root, args.end_date)
        except FetchError as exc:
            print(f"FAILED  {exc}", file=sys.stderr)
            return 1
        print(
            f"labelled fog: {record['positive_class']['n_frames']} frames on "
            f"{len(record['positive_class']['cameras'])} cameras"
        )
        print(f"background:   {record['background_class']['n_frames']} matched frames")
        print(f"{'statistic':22s} {'AUC':>7s} {'95% CI':>16s}  separates  median  bg>=median")
        for name, stat in record["statistics"].items():
            low, high = stat["auc_95pc_bootstrap_ci"]
            print(
                f"{name:22s} {stat['auc_fog_over_background']:7.3f} "
                f"[{low:6.3f},{high:6.3f}]  "
                f"{'yes' if stat['separates'] else 'NO ':>9s}  "
                f"{stat['median_of_labelled_fog']:7.4f}  "
                f"{stat['background_fraction_at_or_above_that_median']:9.3f}"
            )
        print(f"fog_marine cutline: deck_fraction >= {FOG_DECK_FRACTION_MIN}")
        print(f"low_cloud  cutline: sky_greyness  >= {LOW_CLOUD_SKY_GREYNESS_MIN}")
        return 0

    manifests: list[dict] = []
    for name in ("manifest_hold.json", "manifest_dev.json"):
        path = data_root / name
        if not path.is_file():
            print(f"FAILED  {path} is missing; it is what pins the cameras and the hour.", file=sys.stderr)
            return 2
        manifests.append(json.loads(path.read_text()))
    hold_manifest, dev_manifest = manifests

    try:
        # Calibrated FIRST, before a single candidate is screened. The order is the point:
        # the cutlines cannot have been influenced by what the sweep happened to find,
        # because the sweep has not run yet when they are fixed.
        calibration = calibrate_fog_predicate(args.spike_root, args.end_date)
        manifest = build(
            hold_manifest,
            _figlib_fire_dates((hold_manifest, dev_manifest)),
            args.end_date,
            calibration=calibration,
        )
    except FetchError as exc:
        print(f"FAILED  {exc}", file=sys.stderr)
        return 1

    # Written last, in one call, and only on the success path. A partial manifest is worse
    # than no manifest: it loads, it solves a threshold, and nothing downstream can tell.
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    histogram = manifest["class_histogram"]
    print(f"wrote {out_path}: {manifest['n_windows']} windows")
    for label in _CLASS_ORDER:
        quota = CLASS_QUOTAS[label]
        mark = "" if histogram[label] == quota else f"  (quota {quota}, SHORT)"
        print(f"  {label:<18} {histogram[label]:3d}{mark}")
    for shortfall in manifest["shortfalls"]:
        print(f"SHORT   {shortfall['class']}: {shortfall['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    sys.exit(main())
