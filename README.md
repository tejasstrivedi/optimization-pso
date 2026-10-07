# EM618 DT Load Balancing – Streamlit PSO Simulator

## What this demo shows

- Baseline DT and R/Y/B loading
- Random-key encoding for discrete consumer states
- Live Particle Swarm Optimization (PSO)
- Feasible-particle count and convergence history
- Particle distance to global best
- DT loading of the current global-best particle
- Optional deterministic local-search refinement
- Exact final 6-transfer + 12-rephasing report solution
- Peak-demand and solar sensitivity demonstration

## Method terminology

- **Course method:** Particle Swarm Optimization (PSO)
- **Discrete representation:** random-key encoding
- **Final refinement:** deterministic single-move local search

Random-key encoding is not presented as a separate course algorithm. It is only the
mapping from continuous PSO positions to the discrete DT/phase choices.

## Files

Place these in the same folder:

- `app.py`
- `EM618_DT_Load_Balancing_Synthetic_Baseline.xlsx`

The workbook can also be uploaded inside the app.

## Install

Open Command Prompt / PowerShell in this folder:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

A browser window should open automatically. If not, copy the local URL printed in the terminal.

## Recommended viva demonstration

1. Open **Baseline** and point out DT-1 / DT-4 overloads and phase overload.
2. Open **Random-Key Encoding** and choose C0005.
3. Move the x slider to show how one continuous PSO key maps to a discrete DT/phase state.
4. Open **Live PSO**.
5. Use the report-selected settings:
   - swarm = 60
   - inertia = 0.9 -> 0.4
   - c1 = c2 = 2.0
   - Vmax = 0.20
   - maximum iterations = 200
   - early stopping = 30
6. Click **Run PSO** and explain:
   - each dot/candidate is one complete network configuration;
   - pbest = particle's personal best;
   - gbest = best configuration found by the swarm;
   - the random-key position is decoded to a feasible consumer state;
   - hard violations receive a penalty after light repair.
7. Open **Report Solution** to reproduce the exact 6-transfer / 12-phase-change recommendation.
8. Open **Sensitivity Demo** and set Peak Import growth to +5% to show why phase headroom is the near-binding issue.

## Important reproducibility note

The **Report Solution** tab recomputes the exact final consumer actions reported in the
assignment from the workbook.

The **Live PSO** tab is stochastic. Its convergence path and best solution can differ from
the historical run unless the exact original initialization, repair sequence and random
number stream are reproduced. This is normal for a metaheuristic and is why the report used
multiple independent seeds.

## Engineering limitation

This is a synthetic decision-support study, not a load-flow/protection package. Any real
consumer transfer requires field verification of physical connectivity, conductor capacity,
voltage drop, protection coordination and service routing.
