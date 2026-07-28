# Capstone: Gap Analysis & Improvement Plan

**Project selected:** Fraud Detection Model (Adey Innovations Inc.) — Weeks 5–6 project.

**Why this project:** it already has a real leakage-safe data pipeline, two distinct dataset shapes (rich behavioral e-commerce data vs. PCA-anonymized bank data), and a genuine cost trade-off (missed fraud vs. false alarms) that maps directly onto what a finance-sector reviewer evaluates. It was the strongest existing candidate to turn into a reliability-first, explainability-first portfolio piece.

## Gap Analysis Checklist

Assessed by cloning the repo, installing dependencies, and actually running the test suite and pipeline — not by reading the README's claims.

| Category | Question | Status (before this pass) | Status (after this pass) |
|---|---|---|---|
| Code Quality | Modular and well-organized? | Yes | Yes |
| | Type hints on functions? | Partial | Partial → improved in new `src/models/` code |
| | Clear project structure? | Yes | Yes |
| Testing | Unit tests for core functions? | Yes (36 tests) | Yes (52 tests) |
| | Do tests run automatically on push? | Yes (`.github/workflows/ci.yml` existed) | Yes, verified passing |
| Documentation | Is the README comprehensive? | Partial — described a pipeline that didn't actually run | Yes — rewritten to match reality, with real extracted metrics |
| | Docstrings on functions? | Partial | Improved in all new modules |
| Reproducibility | Can someone else run this project? | **No** — `main.py` called three scripts that didn't exist | **Yes** — verified end-to-end |
| | Dependencies in requirements.txt? | Yes | Yes |
| Visualization | Interactive way to explore results? | No | Yes — Streamlit dashboard |
| Business Impact | Problem clearly articulated? | Partial (narrative only) | Yes — quantified in README + dashboard |
| | Success metrics defined? | Partial | Yes — AUC-PR/ROC-AUC/F1 table with real numbers |

## What Was Actually Broken (found by running the code, not reading it)

1. `main.py` orchestrated `src/models/train_models.py`, `cross_validate.py`, `evaluate_models.py` — **none of which existed**. Running it crashed immediately.
2. The credit-card half of `scripts/pipeline.py` computed a train/test split but never wrote it to disk, so there was nothing for a training script to read even once one existed.
3. `main.py` invoked scripts by file path, which only puts that file's own directory on `sys.path` — every `from src....` import inside them would have raised `ModuleNotFoundError` even after the missing files were added.
4. `.gitignore` had a bare `models/` rule, which — because gitignore patterns without a leading slash match at any depth — also silently matched `src/models/`. The new source package would never have actually reached GitHub.

These four compound: fixing only #1 without #2–4 would have looked done locally and still failed for anyone else who cloned the repo, which is exactly the kind of gap a finance-sector reviewer would find first.

## Prioritized Improvement Plan

| # | Improvement | Addresses | Time estimate | Status |
|---|---|---|---|---|
| 1 | Fix the broken train → cross-validate → evaluate chain and the `.gitignore` bug hiding it | Critical reproducibility gap | ~3 hrs | Done |
| 2 | Reusable SHAP + built-in importance module (`src/explainability.py`) | Model Explainability deliverable | ~1.5 hrs | Done |
| 3 | Business-impact translation layer (`src/business_impact.py`) | Business Impact gap | ~1 hr | Done |
| 4 | Streamlit dashboard (metrics, prediction explorer, SHAP, $ impact) | Visualization gap | ~2.5 hrs | Done |
| 5 | README rewrite to the professional template with real numbers, gap analysis + technical report | Documentation gap | ~1.5 hrs | Done |

Selection rationale: items 1 and 4 were prioritized first because they were the two gaps a reviewer would notice within minutes (a broken Quick Start, no way to interact with results); items 2–3 were prioritized next because the dashboard depends on them; documentation came last since it needed real numbers from the working pipeline to be honest rather than aspirational.
