# PlumeRank -- five targets, and no sixth.
#
# The Makefile holds NAMES, never logic: each target is one line of shell wrapping a
# script that already exists. A sixth target is a scope change,
# and scripts/check_claims.py asserts that every target the docs mention lives here.
#
#   make fetch       the evaluation corpus, sha256-verified, resumable (~80 min measured)
#   make bench       the full held-out evaluation                      (12m29s measured)
#   make test        offline suite, no network, no credentials         (~2 min)
#   make test-live   the two external-side-effect tests + determinism
#   make verify      THE JUDGE ONE-LINER: clean clone -> numbers       (<=3 min)
#
# `make fetch && make bench` is the pair that reproduces the published headline.
# `make verify` does NOT reproduce it and never claims to -- see README, ReproduceBlock.

PY  := .venv/bin/python
VENV := .venv

.PHONY: fetch bench test test-live verify

# FIgLib + MTA only: bench never reads the Pyro-SDIS training mirror (the classifier is
# committed as models/tile_classifier.onnx). Retraining: scripts/fetch.py --corpus pyrosdis.
fetch:
	$(PY) scripts/fetch.py --corpus figlib

bench:
	$(PY) scripts/bench.py --arm hold --budget 2.0 --sensitivity --seed 20261026

test:
	$(PY) -m pytest tests -m "not live" --cov=plumerank --cov-report=term-missing --cov-fail-under=100

test-live:
	$(PY) -m pytest tests -m live

verify:
	@python3 -m venv $(VENV)
	@$(VENV)/bin/pip install --quiet --disable-pip-version-check --upgrade pip
	@$(VENV)/bin/pip install --quiet --disable-pip-version-check -r requirements-dev.lock
	@$(VENV)/bin/python -m pytest tests -m "not live" -q
	@$(VENV)/bin/python scripts/bench.py --arm micro
