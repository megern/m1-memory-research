# Version 1.0 local workflow

This release completes the stated local workflow. It is not a claim of production certification, market adoption or scientific priority. Legacy entry points remain available.

## Install

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install .
```

The project requires Python 3.10+; the local release validation uses Python 3.14. Optional model training/inference is Apple Silicon only. Dependency installation can download packages, but runtime workflows use existing local inputs and no hosted compute.

## Verified research evidence catalog

```sh
m1-study-catalog /LOCAL/completed-study-export --output /LOCAL/reviews/research
```

Audits completed optimizer-study exports: receipt hashes, run coverage, protocol identity, finite measurements and boolean acceptance decisions. Retains negative gates and rejected artifacts. It checks shared file consistency, not original checkpoint arrays, measurement authenticity, novel algorithms or practical 70B fit.

## Review artifacts

New integrated workflows create a fresh directory containing `report.json`, a standalone `report.html`, and `manifest.json` with file hashes. Open report.html locally; it uses no external scripts, network service or remote fonts. Input text is HTML-escaped. Large arrays show their first 1000 records in HTML while the complete JSON remains available. Existing destinations are refused. Artifact hashes establish file consistency, not the truth or independent authenticity of claims.

## Exit outcomes and limitations

Investigate rejection/error outcomes rather than counting them as a clean result. Workflow review returns 1 on policy findings and 2 on scan/input errors. LLM regression review returns 1 when review is required. Static batch/catalog failures return 2 while retaining per-input diagnostics. SOC and data-model errors return 2. Nothing posts to LinkedIn, installs remediation packages, or automatically executes response actions.

Tests and installed entry points were exercised locally. Examples are synthetic or explicitly public research/data fixtures; user records, raw SMS/token rows, model weights and adapter weights are not release assets. All earlier failed scientific hypotheses and model interpretations remain documented.
