# EM618 DT Load Balancing — simple Streamlit demo

The dashboard matches the corrected decision report: **8 DT transfers, 8 same-DT rephasings, 610 m**, with **J = 0.3335541699461108**. The maximum DT and phase loadings are 92.68% and 97.19% respectively.

## Run locally

Use Python 3.12 or later:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Keep `app.py`, `EM618_DT_Load_Balancing_Synthetic_Baseline.xlsx` and the complete `report_data/` folder together. Paths resolve beside the script, independently of the launch directory. For Streamlit deployment, select `app.py` as the entry point and install `requirements.txt`.

## Dashboard contents

Six tabs show the baseline, random-key encoding, live PSO, report solution, sensitivity checks and verified experiments. The demo uses three headline metrics and compact tables. Only Live PSO draws a chart: best feasible objective against iteration. Advanced settings, equations, objective components, consumer actions and experiment records are expandable. The report solution uses the original report settings; sidebar changes affect live experiments. Uploaded data is checked against the bundled dataset before report-specific recommendations are shown.

Checked against the final report on 11 October 2026: 40 numerical comparisons passed within the report's printed precision, covering baseline and recommended loading, all 16 actions, objective components, fixed-plan and archived re-optimized sensitivity results, tuning and validation summaries, and every recorded Figure 2 objective. The raw mean validation stopping count is 54.95, shown as 55.0 to one decimal in the report.

The experiment archive contains 60 tuning runs (12 configurations × 5 seeds), 20 validation runs, both reference convergence histories and computation metadata. Validation feasibility describes retention from a feasible initialization, rather than unseeded discovery.

## Reproduce the report runs

Leave the illustration-only recommendation warm-start unchecked. The default initialization uses a separately constructed feasible seed, not the final recommendation.

| Preset | Parameters | PSO result | Iterations |
| --- | --- | --- | --- |
| Validation run 1004 | 60 particles, inertia 0.9 → 0.4, c1 = c2 = 2 | J = 0.3691700423952783 | 64 |
| Recommendation origin: run 101 | 60 particles, inertia 0.7, c1 = c2 = 2 | J = 0.3639342845396221 | 45 |

Both use a 200-iteration cap, velocity limit 0.2 and patience 30. Local refinement defaults to off for the validation preset and on for the recommendation-origin preset. Nine accepted improvements from run 101 reproduce the recommendation and J = 0.3335541699461108. Figure 2's validation result and the final recommendation are different experiments, as explained in report Sections 5.2–5.3. The animated plot includes the final early-stopping iteration even when it falls between refresh intervals.

Objective normalization stays fixed to the original 90% baseline. Feasibility constraints include loading limits, action caps and prohibited reciprocal DT transfers. Excess-transfer repair removes the longest transfers first. Local search uses the repaired neighbourhood, a 200-pass cap and improvement tolerance 1e-9; PSO uses 1e-10.

## Sensitivity

At uniform +5% peak demand, the fixed recommendation reaches 97.31% maximum DT loading and **102.05% maximum phase loading**, violating the phase limit. Its phase-based uniform growth margin is approximately 2.89%. Interactive sensitivity evaluates the fixed plan. An archived scenario search that found no feasible result is not proof of infeasibility; cap comparisons are best-tested improvements, not exact shadow prices.

## Verify before pushing

```powershell
python -m unittest discover -s tests -v
git diff --check
git status --short
```

Include `app.py`, this README, `report_data/` and `tests/` in the revised commit, retaining the workbook and requirements already in the repository. Tests check the recommendation, reference PSO runs, local refinement and Streamlit rendering/execution.

This synthetic study supports decisions. Check physical connectivity, conductor capacity, voltage drop, protection and service routing before implementation.
