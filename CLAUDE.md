# Repository Instructions

## 1. Business Requirements
This repository is a fresh CMP4336 course project for hourly bike sharing demand forecasting using the Kaggle Bike Sharing Demand dataset.

The primary project behavior is:
- Load the Kaggle `train.csv`, `test.csv`, and `sampleSubmission.csv` files from local project data directories.
- Analyze how temporal, seasonal, and weather conditions relate to hourly rental demand.
- Predict the hourly total rental `count`.
- Produce report-ready visualizations for environmental impact and cyclical demand patterns.
- Train and compare reproducible baseline, linear, tree-based, and boosting models as phases are implemented.
- Evaluate forecasts primarily with RMSLE, while also reporting RMSE, MAE, and R2 where implemented.
- Generate Kaggle-ready submissions with exactly `datetime,count` columns and one prediction for every test row.
- Investigate the proposal's sequential-data objective through lag-feature experiments only when validation and test-time generation are leakage-safe.

Forecasting inventory (planned production behavior as phases land):
- Baseline: global mean and hour-of-day mean demand benchmarks.
- Linear model: Ridge trained against a `log1p(count)` target and inverted for predictions.
- Tree models: scikit-learn Random Forest and Gradient Boosting candidates.
- Boosting model: XGBoost as a later strong-model candidate, not an initial dependency requirement.
- Feature engineering: timestamp-derived calendar features, working-day/holiday context, seasonal/weather inputs, and cyclic time encodings.
- Submission generation: non-negative predictions written in Kaggle's required schema.
- Optional stretch strategy: separate `casual` and `registered` target models whose predictions are summed; neither target may be used as a feature for the other model or for a direct `count` model.

Scope control is strict:
- Do not use `casual` or `registered` as features for any model that predicts `count`; they sum exactly to the target and do not exist in `test.csv`.
- Do not introduce `count(t-1)` or another target lag as an ordinary feature unless inference can construct it without true future targets and validation proves the same recursive behavior.
- Do not add an application/API layer, remote service, external dataset, or automatic data downloader unless explicitly requested.
- Do not broaden the project beyond the proposal's environmental, temporal, sequential-analysis, modeling, evaluation, and submission objectives unless explicitly requested.
- Do not commit raw Kaggle CSV files, generated model artifacts, processed datasets, submission outputs, or machine-local environment files.
- Preserve the phase-based implementation strategy: build a tested, understandable forecasting pipeline before optional stretch work.

## 2. Technical Details
Package and runtime truth:
- Package manager: `pip` with a local virtual environment.
- Primary runtime manifest: `requirements.txt`.
- Package configuration: `pyproject.toml`, using the `src/` layout and editable install via `pip install -e .`.
- Primary environment setup commands: `python -m venv .venv`, `source .venv/bin/activate`, `pip install -r requirements.txt`, and `pip install -e .`.
- Primary test command once tests exist: `.venv/bin/python -m pytest`.
- Package import verification command: `.venv/bin/python -c "import bike_sharing; print(bike_sharing.__version__)"`.
- Treat `requirements.txt`, `pyproject.toml`, and `README.md` as the most reliable runtime truth for execution decisions.

Architecture and ownership:
- The repository is currently at Phase 2: in addition to the Phase 1 deliverables (package scaffold, raw-data loading, configuration validation, leakage preprocessing, and their automated tests), an executed EDA notebook lives at `notebooks/01_eda.ipynb` and five report figures are tracked under `reports/figures/01_count_distribution.png` through `05_correlation_heatmap.png`. Feature engineering and modeling modules are introduced in later phases.
- `src/bike_sharing/` is the importable package for reusable project logic.
- `config/config.yaml` owns paths, random seed, target name, excluded columns, and datetime configuration; feature flags may be added in later phases.
- `config/models.yaml`, once added, owns model hyperparameters rather than embedding experimental settings in scripts.
- `src/bike_sharing/config.py` loads and validates configuration, including required pipeline paths.
- `src/bike_sharing/data.py` loads local raw data and parses datetimes; time-aware split helpers may be added with modeling work.
- `src/bike_sharing/preprocessing.py` owns leakage-column removal and target transformations.
- `src/bike_sharing/features.py`, once added, owns reusable feature construction and must remove raw datetime only after deriving time features.
- `src/bike_sharing/models.py`, `train.py`, `evaluate.py`, and `predict.py`, once added, own estimator creation, fitting/validation, metrics, and prediction behavior.
- `scripts/` contains thin command-line orchestrators only; business logic belongs in `src/bike_sharing/`.
- `notebooks/` contains analysis and presentation work; notebooks must not silently redefine pipeline contracts.
- `tests/` owns automated contracts, especially leakage prevention, data schema, feature generation, metrics, and submission validity.

Runtime flow (target end-to-end pipeline):
1. The configuration loader resolves local paths, target, seed, datetime column, and forbidden feature columns.
2. Raw-data loading reads `data/raw/train.csv` and `data/raw/test.csv` and parses `datetime`.
3. Preprocessing and feature generation construct numeric/categorical model inputs while ensuring `casual`, `registered`, and `count` cannot enter the direct-count feature matrix.
4. Training fits baseline or requested candidate models using time-aware validation and a consistently applied target transformation where configured.
5. Evaluation calculates RMSLE as the primary score plus RMSE, MAE, and R2 where applicable, then writes comparable report output.
6. Prediction applies the trained model to processed test features, inverts any target transformation, clips negative demand predictions to zero, and preserves original test datetime order.
7. Submission generation writes `datetime,count` rows matching the Kaggle sample submission schema.
8. Optional sequential experiments model lags only with inference-realistic recursive generation and matching leakage-safe validation.

Repo drift and guardrails:
- `data/raw/` contains local Kaggle source files and is gitignored apart from `.gitkeep`; code must provide clear missing-data failures rather than assume tracked inputs.
- The observed Kaggle data contract is `train.shape == (10886, 12)` and `test.shape == (6493, 9)` before feature engineering.
- Train contains `casual`, `registered`, and `count`; test contains none of these target-derived fields.
- In the raw training data, `casual + registered == count`; enforce exclusion from direct-count features in configuration, preprocessing, and tests.
- The Kaggle split is time-structured: training rows cover days 1-19 of each month and test rows cover later days. Do not treat unknown test targets as available lag inputs.
- Preserve raw `datetime` for submission output even when it is removed from model features.
- Do not hardcode local absolute paths, secrets, notebook-only transformations, or untracked assumptions about artifact presence.
- Do not silently rename config keys, feature columns, artifact paths, metric keys, or submission columns once introduced.

## 3. Data Sources & Model Artifacts
* Kaggle Bike Sharing Demand is the primary and only required dataset source.
* Raw local data is expected under `data/raw/`: `train.csv`, `test.csv`, and `sampleSubmission.csv`.
* Raw Kaggle CSV files are local-only and gitignored; a clean clone requires the user to place downloaded data in `data/raw/`.
* The training dataset contains `datetime`, temporal/weather predictors, `casual`, `registered`, and primary target `count`.
* The test dataset contains only inference-time predictors and `datetime`; it does not contain `casual`, `registered`, or `count`.
* `casual` and `registered` are target-derived columns for the direct-count task and must never appear in its feature matrix.
* Processed artifacts, once introduced, belong under `data/interim/` and `data/processed/` and should be reproducible from local raw inputs.
* Saved trained estimators, once introduced, belong under `models/` and are generated/local artifacts rather than repository source files.
* Figures, metric reports, and Kaggle submissions, once introduced, belong under `reports/figures/`, `reports/metrics.json`, and `reports/submissions/` respectively.
* The proposal document is tracked under `docs/CMP4336-Project_Proposal-2103983-2103599-2102798.docx` and defines environmental impact, temporal visualization, and sequential-analysis goals.
* If a lag-based sequential experiment is run, its data-generation procedure, inference behavior, validation protocol, and comparison against the non-lag baseline must be documented.
* Do not commit private dataset variants, generated predictions, trained artifacts, virtual environments, local caches, secrets, or machine-specific paths.
* Do not add cloud storage, remote artifact fetch, or automatic dataset download behavior unless explicitly requested.

## 4. Strategy
Every agent working in this repository must follow this loop exactly:
Plan -> Wait for Approval -> Code -> Test -> Fix
Plan requirements (minimum for approval): - Which files will change and why. - Current behavior vs. target behavior. - Validation path: how the change will be verified.
Execution rules: - Always read the relevant module, schema, and config files before proposing changes. - Preserve module APIs and field names unless explicitly requested. - Prefer small, local, architecture-preserving edits over broad rewrites. - Large or ambiguous changes require an explicit plan and approval first. - Every behavioral change must include at least one concrete validation path: a pytest contract (data loading, leakage exclusion, preprocessing, feature engineering, metrics, or submission schema), an end-to-end notebook run with concrete output evidence (figures saved, metrics.json updated), or a verifiable file artifact (processed parquet shape, submission CSV with the exact `datetime,count` schema and 6493 rows).
## 5. Debugging Rules
When facing a bug, DO NOT GUESS. 1. Reproduce the problem. 2. Prove you reproduced it. 3. Find the root cause. 4. Fix it. 5. Prove you fixed it.
Repository-specific debugging rules: - Reproduce issues with tests, scripts, logs, stack traces, or artifact inspection. - Prove reproduction with concrete evidence before changing code. - Verify whether the failure belongs to: raw CSV loading, config resolution, leakage column exclusion, feature engineering output, time-aware split correctness, model fit or predict, evaluation metric computation, or submission file generation. - Do not patch around failures by swallowing exceptions or returning silent defaults - prove the root cause first. - Prefer focused tests or targeted reproductions over speculative edits. - Keep fixes minimal and local to the proven fault line. - After a fix, prove the outcome with the same reproduction path or a tighter automated test.
## 6. Coding Standards
Non-negotiable rules: - No emojis ever in code, logs, commits, or generated documentation. - Never use Turkish characters in variables, functions, or comments. - Avoid over-defensive programming. Do not add unnecessary try/except blocks, wrappers, or fallback branches without evidence. - Keep communication and README-style documentation short, direct, and human-readable. - Do not generate AI slop.
Repo-safe engineering standards: - Preserve the existing module boundaries between src/, scripts/, tests/, and config/. - Do not silently rename model files, artifact files, schema fields, or config keys. - Do not introduce broad refactors during feature work or bug work unless explicitly requested. - Prefer explicit, testable logic over abstraction-heavy wrappers and fallback-heavy control flow. - Preserve `src/bike_sharing` module boundaries: configuration in `config.py`, raw IO in `data.py`, leakage exclusion and target transforms in `preprocessing.py`. When new modules are added (`features.py`, `models.py`, `train.py`, `evaluate.py`, `predict.py`), each is single-responsibility and reusable; do not duplicate their logic inside scripts or notebooks. - `scripts/` may contain executable orchestrators that call into `src/` (e.g. future `train_model.py`, `generate_submission.py`) and reproducible artifact-building utilities (e.g. `_build_eda_notebook.py`); business, feature, modeling, or evaluation logic must not live in `scripts/`. - If you discover drift or contradictions, document them clearly in the task output instead of masking them.

## 7. Experiment Planning And Execution Log
+ Scope: this section applies to **modeling experiments and improvement plans** — sweeps, ablations, transform changes, hyperparameter studies, and anything else whose outcome is judged against a metric. Structural phase work (scaffolding, config setup, plain feature engineering, submission plumbing) follows the §4 strategy loop and uses the approved roadmap as its plan artifact; it does not require a separate `docs/experiments/` file.
+ This section defines how every such experiment / improvement plan is documented and how its execution is marked. Goal: every plan and its outcome live in one place (`docs/experiments/`) in a commit-traceable way, so future references to an experiment are `grep`-able.
+ Where the plan file goes, and what it is named:
+ All new plans are saved under `docs/experiments/`.
+ Filename format: `{YYYY-MM-DD}_{plan-name}.md` (kebab-case plan name). Example: `2026-06-10_log1p-vs-boxcox-sweep.md`, `2026-06-15_cyclic-hour-encoding.md`.
+ The date is the day the plan was **authored**, not the day it was executed. The filename stays fixed even if the plan spans multiple experiments over several days.
+ Do not touch code before the plan file exists. In the `Inspect -> Plan -> Code -> Test -> Fix` loop, the plan file is the artifact produced by the `Plan` step.
+ Required structure of the plan file:
+ A plan file is written **item by item**, **in logical order**, as a **narrative**: motivation -> hypothesis -> preconditions -> items -> expected outputs -> decision criteria.
+ Header block at the top of every plan file:
+   - **Date:** {YYYY-MM-DD}
+   - **Topic:** short title
+   - **Motivation:** which report section (`§X`) or which metric anomaly triggered this plan; link the baseline run id(s) so comparisons stay reproducible.
+   - **Hypothesis:** the proposition under test, expressed as a measurable claim (e.g. "`log1p` target outperforms Box-Cox on RMSLE by at least 0.005" or "cyclic hour encoding lowers Ridge RMSLE by at least 0.01 vs raw hour-of-day").
+   - **Preconditions:** code / config / cache state that must already be in place before the plan starts.
+ Then a numbered list of items (`## 1) ...`, `## 2) ...`). Template for each item:
+   - **Goal:** what will change (code / config / sweep parameter).
+   - **Files:** paths to touch (with current line numbers or function names where useful).
+   - **Steps:** sub-bullets, one logical operation per bullet (e.g. "add `data.train_outlier_sigma.BidderTotalBids: 3.0` to config", "run the sweep runner with a single trial and capture the log").
+   - **Test / verification:** which unit test gets added or updated; which full-training output is compared against which metric table.
+   - **Expected outcome:** decision criterion (e.g. how big a Q50 MAE delta counts as meaningful, where coverage should land relative to the target band).
+   - **DONE / DROPPED:** empty at authoring time; filled in during execution (see below).
+ Items are ordered by the **narrative**: dependencies before dependents, independents in parallel. The flow "test the hypothesis with a single trial -> if positive, expand to a sweep -> production decision" must always be visible — random ordering is not acceptable.
+ Execution / marking contract:
+ Each time an item is executed, write the outcome **into the same file**, **immediately under that item**. Template:
+   ```
+   **DONE (commit `<hash>`):** {one or two sentences: what changed, which behavior was gained, any remaining side-effect.}
+   - Metric / result: {small baseline-vs-experiment table if relevant}
+   - Run id: {always include — found under `models/training/{target}/{run_id}/`}
+   - Sweep JSON: {if applicable, path under `docs/experiments/...sweep_...json`}
+   - Decision: {shipped to production, shelved, or fed into another experiment}
+   ```
+ The `<hash>` placeholder must never be left in place; do not write DONE before the commit lands. If the execution required multiple commits, list all of them in order, comma-separated.
+ For abandonment, the marker becomes `DROPPED ({date}):` followed by a one-paragraph reason. No item is left open; every item is closed as either DONE or DROPPED.
+ If a new plan resolves a question raised in an earlier plan, the new file carries a `Corresponding plan: docs/experiments/{earlier-plan}.md` line near the top so the relationship is grep-able.
+ Any value landing in production config (`config/config.yaml` or, if introduced, `config/models.yaml`) as the outcome of a plan must carry an inline comment pointing back to the plan file (e.g. `# experiment: 2026-06-10_log1p-vs-boxcox-sweep.md - chose log1p`). This is how a config reader finds the rationale behind a value.
+ A plan file does NOT contain:
+ Speculative "might also try" lists above and beyond the concrete intent. A plan is the contract for **work happening now**, not a wishlist.
+ Re-summaries of already-closed plans. A cross-reference link is enough.
+ Pasted code blocks. A plan file is prose + bullets; code changes live in the commit.
+ Pre-flight (before creating a new plan file):
+ `grep` under `docs/experiments/` for a half-open plan on the same topic. If one exists, append a new item to that plan file — do not create a new one.
+ Record the current benchmark / baseline run id before the plan starts (write it in the **Motivation** section). This is what later makes statements like "experiment X is +0.4 Q50 MAE vs baseline" reproducible.
