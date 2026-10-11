
from __future__ import annotations

import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# EM618 DT LOAD BALANCING - PSO DEMONSTRATOR
# Core course method: Particle Swarm Optimization (PSO)
# Discrete representation: random-key encoding
# Final refinement: deterministic single-move local search
# ============================================================

APP_TITLE = "Live PSO Demo"
DEFAULT_WORKBOOK = "EM618_DT_Load_Balancing_Synthetic_Baseline.xlsx"
APP_DIR = Path(__file__).resolve().parent
REPORT_DIR = APP_DIR / "report_data"

PHASES = ("R", "Y", "B")
PHASE_TO_INDEX = {"R": 0, "Y": 1, "B": 2, "RYB": 3}

# Report normalization for field effort:
# 3*20 transfers + 25 rephasings + 0.01*(20*140 m) = 113
REPORT_F_CHANGE_MAX = 113.0

# Exact final hybrid actions reported in the assignment.
# Consumers not listed here stay in their baseline state.
REPORT_FINAL_STATES: Dict[str, Tuple[str, str]] = {'C0003': ('DT-1', 'Y'), 'C0004': ('DT-1', 'B'), 'C0005': ('DT-2', 'B'), 'C0006': ('DT-4', 'Y'), 'C0007': ('DT-4', 'RYB'), 'C0010': ('DT-1', 'R'), 'C0039': ('DT-3', 'B'), 'C0045': ('DT-5', 'B'), 'C0046': ('DT-4', 'Y'), 'C0048': ('DT-4', 'B'), 'C0049': ('DT-5', 'Y'), 'C0054': ('DT-5', 'R'), 'C0056': ('DT-5', 'RYB'), 'C0091': ('DT-5', 'RYB'), 'C0100': ('DT-6', 'R'), 'C0138': ('DT-8', 'Y')}


@dataclass
class Model:
    consumers: pd.DataFrame
    dt_master: pd.DataFrame
    dt_names: List[str]
    dt_to_idx: Dict[str, int]
    capacities: np.ndarray

    state_lists: List[List[dict]]
    state_count: np.ndarray
    ctrl_consumers: np.ndarray
    ctrl_kcount: np.ndarray

    state_dt: np.ndarray
    state_phase: np.ndarray
    transfer_flag: np.ndarray
    rephase_flag: np.ndarray
    distance_m: np.ndarray

    p_contrib: np.ndarray
    q_contrib: np.ndarray
    existing_dt_idx: np.ndarray
    peak_s_kva: np.ndarray


@dataclass
class EvalConfig:
    max_transfers: int = 20
    max_rephases: int = 25
    preferred_loading: float = 90.0
    hard_dt_limit: float = 100.0
    hard_phase_limit: float = 100.0
    no_reciprocal: bool = True
    w_dt: float = 0.25
    w_phase: float = 0.25
    w_headroom: float = 0.25
    w_change: float = 0.25


@dataclass
class BaselineRefs:
    f_dt: float
    f_phase: float
    f_headroom: float


def yes(value) -> bool:
    return str(value).strip().lower() == "yes"


@st.cache_data(show_spinner=False)
def load_workbook_bytes(raw: bytes):
    xls = pd.ExcelFile(io.BytesIO(raw))
    consumers = pd.read_excel(xls, sheet_name="Consumer_Data")
    dt_master = pd.read_excel(xls, sheet_name="DT_Master")

    # Remove exported index columns if present.
    consumers = consumers.loc[:, consumers.columns.astype(str).str.lower() != "index"].copy()
    dt_master = dt_master.loc[:, dt_master.columns.astype(str).str.lower() != "index"].copy()

    return consumers, dt_master


def workbook_source():
    with st.sidebar.expander("Change dataset"):
        uploaded = st.file_uploader(
            "Upload the synthetic baseline workbook",
            type=["xlsx"],
            help="Use EM618_DT_Load_Balancing_Synthetic_Baseline.xlsx",
        )

    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name

    local = APP_DIR / DEFAULT_WORKBOOK
    if local.exists():
        return local.read_bytes(), local.name

    st.error(
        f"Workbook not found. Put `{DEFAULT_WORKBOOK}` beside app.py "
        "or upload it from the sidebar."
    )
    st.stop()


def add_state(
    state_list: List[dict],
    existing_dt: str,
    existing_phase: str,
    dt: str,
    phase: str,
    distance: float,
):
    """Add a unique feasible state."""
    key = (dt, phase)
    if any((s["dt"], s["phase"]) == key for s in state_list):
        return

    transfer = int(dt != existing_dt)
    rephase = int(dt == existing_dt and phase != existing_phase)

    state_list.append(
        {
            "dt": dt,
            "phase": phase,
            "transfer": transfer,
            "rephase": rephase,
            "distance": float(distance if transfer else 0.0),
            "label": f"{dt} / {phase}",
        }
    )


def build_model(consumers: pd.DataFrame, dt_master: pd.DataFrame) -> Model:
    required = {
        "Consumer_ID",
        "Existing_DT",
        "Connection_Type",
        "Existing_Phase",
        "Transferable",
        "Alternate_DT_1",
        "Alt1_Distance_m",
        "Alternate_DT_2",
        "Alt2_Distance_m",
        "Phase_Change_Allowed",
        "Peak_Import_P_kW",
        "Peak_Import_Q_kVAr",
        "Peak_Import_S_kVA",
        "Net_Day_P_kW",
        "Day_Q_kVAr",
    }
    missing = sorted(required - set(consumers.columns))
    if missing:
        raise ValueError(f"Missing Consumer_Data columns: {missing}")

    dt_names = dt_master["DT"].astype(str).tolist()
    dt_to_idx = {d: i for i, d in enumerate(dt_names)}
    capacities = dt_master["Capacity_kVA"].astype(float).to_numpy()

    state_lists: List[List[dict]] = []

    for _, row in consumers.iterrows():
        cid = str(row["Consumer_ID"])
        existing_dt = str(row["Existing_DT"])
        existing_phase = str(row["Existing_Phase"]).strip()
        connection_type = str(row["Connection_Type"]).strip()

        states: List[dict] = []

        # State 0 is ALWAYS the current connection.
        add_state(states, existing_dt, existing_phase, existing_dt, existing_phase, 0.0)

        # Same-DT rephasing only applies to eligible single-phase consumers.
        if connection_type == "Single Phase" and yes(row["Phase_Change_Allowed"]):
            for ph in PHASES:
                add_state(states, existing_dt, existing_phase, existing_dt, ph, 0.0)

        # Transfer states only use explicitly listed alternate DTs.
        if yes(row["Transferable"]):
            alternates = [
                (row.get("Alternate_DT_1"), row.get("Alt1_Distance_m")),
                (row.get("Alternate_DT_2"), row.get("Alt2_Distance_m")),
            ]
            for alt_dt, distance in alternates:
                if pd.isna(alt_dt) or str(alt_dt).strip() == "":
                    continue
                alt_dt = str(alt_dt).strip()
                dist = 0.0 if pd.isna(distance) else float(distance)

                if connection_type == "Three Phase":
                    add_state(states, existing_dt, existing_phase, alt_dt, "RYB", dist)
                else:
                    # When a single-phase service is physically transferred,
                    # the receiving phase is also a decision variable.
                    for ph in PHASES:
                        add_state(states, existing_dt, existing_phase, alt_dt, ph, dist)

        state_lists.append(states)

    n = len(consumers)
    max_k = max(len(s) for s in state_lists)

    state_count = np.array([len(s) for s in state_lists], dtype=int)
    ctrl_consumers = np.flatnonzero(state_count > 1)
    ctrl_kcount = state_count[ctrl_consumers]

    state_dt = np.zeros((n, max_k), dtype=int)
    state_phase = np.zeros((n, max_k), dtype=int)
    transfer_flag = np.zeros((n, max_k), dtype=int)
    rephase_flag = np.zeros((n, max_k), dtype=int)
    distance_m = np.zeros((n, max_k), dtype=float)

    # Contribution vector = 8 DTs * 3 phases = 24 positions.
    p_contrib = np.zeros((n, max_k, 2, len(dt_names) * 3), dtype=float)
    q_contrib = np.zeros_like(p_contrib)

    peak_p = consumers["Peak_Import_P_kW"].astype(float).to_numpy()
    peak_q = consumers["Peak_Import_Q_kVAr"].astype(float).to_numpy()
    solar_p = consumers["Net_Day_P_kW"].astype(float).to_numpy()
    solar_q = consumers["Day_Q_kVAr"].astype(float).to_numpy()

    existing_dt_idx = np.array([dt_to_idx[str(x)] for x in consumers["Existing_DT"]], dtype=int)
    peak_s_kva = consumers["Peak_Import_S_kVA"].astype(float).to_numpy()

    for i, states in enumerate(state_lists):
        for k, s in enumerate(states):
            dti = dt_to_idx[s["dt"]]
            ph_code = PHASE_TO_INDEX[s["phase"]]

            state_dt[i, k] = dti
            state_phase[i, k] = ph_code
            transfer_flag[i, k] = s["transfer"]
            rephase_flag[i, k] = s["rephase"]
            distance_m[i, k] = s["distance"]

            for scenario, (p_total, q_total) in enumerate(
                [(peak_p[i], peak_q[i]), (solar_p[i], solar_q[i])]
            ):
                if ph_code == 3:  # RYB
                    for ph in range(3):
                        flat = dti * 3 + ph
                        p_contrib[i, k, scenario, flat] = p_total / 3.0
                        q_contrib[i, k, scenario, flat] = q_total / 3.0
                else:
                    flat = dti * 3 + ph_code
                    p_contrib[i, k, scenario, flat] = p_total
                    q_contrib[i, k, scenario, flat] = q_total

    return Model(
        consumers=consumers,
        dt_master=dt_master,
        dt_names=dt_names,
        dt_to_idx=dt_to_idx,
        capacities=capacities,
        state_lists=state_lists,
        state_count=state_count,
        ctrl_consumers=ctrl_consumers,
        ctrl_kcount=ctrl_kcount,
        state_dt=state_dt,
        state_phase=state_phase,
        transfer_flag=transfer_flag,
        rephase_flag=rephase_flag,
        distance_m=distance_m,
        p_contrib=p_contrib,
        q_contrib=q_contrib,
        existing_dt_idx=existing_dt_idx,
        peak_s_kva=peak_s_kva,
    )


def decode_positions(positions: np.ndarray, model: Model) -> np.ndarray:
    """Continuous random keys -> discrete state index per consumer."""
    pop = positions.shape[0]
    assignments = np.zeros((pop, len(model.consumers)), dtype=int)

    discrete = np.floor(
        positions * model.ctrl_kcount[None, :]
    ).astype(int)
    discrete = np.minimum(discrete, model.ctrl_kcount[None, :] - 1)

    assignments[:, model.ctrl_consumers] = discrete
    return assignments


def assignment_to_position(assignment: np.ndarray, model: Model) -> np.ndarray:
    """Place each selected discrete state at the midpoint of its random-key interval."""
    idx = assignment[model.ctrl_consumers]
    return (idx + 0.5) / model.ctrl_kcount


def reciprocal_pair_count(assignment: np.ndarray, model: Model) -> int:
    directed = set()
    for i, k in enumerate(assignment):
        if model.transfer_flag[i, k] == 0:
            continue
        src = int(model.existing_dt_idx[i])
        dst = int(model.state_dt[i, k])
        directed.add((src, dst))

    count = 0
    seen = set()
    for edge in directed:
        rev = (edge[1], edge[0])
        pair = tuple(sorted((edge, rev)))
        if rev in directed and pair not in seen:
            seen.add(pair)
            count += 1
    return count


def raw_eval_batch(
    assignments: np.ndarray,
    model: Model,
    cfg: EvalConfig,
):
    """
    Vectorized network evaluation.

    Scenario index:
      0 = Peak Import
      1 = High Solar
    """
    assignments = np.asarray(assignments, dtype=int)
    if assignments.ndim == 1:
        assignments = assignments[None, :]

    pop, n = assignments.shape
    consumer_axis = np.arange(n)[None, :]

    selected_p = model.p_contrib[consumer_axis, assignments]  # pop,n,2,24
    selected_q = model.q_contrib[consumer_axis, assignments]

    p = selected_p.sum(axis=1).reshape(pop, 2, len(model.dt_names), 3)
    q = selected_q.sum(axis=1).reshape(pop, 2, len(model.dt_names), 3)

    s_phase = np.hypot(p, q)
    s_dt = s_phase.sum(axis=3)

    caps = model.capacities[None, None, :]
    phase_caps = (model.capacities / 3.0)[None, None, :, None]

    load_pct = 100.0 * s_dt / caps
    phase_load_pct = 100.0 * s_phase / phase_caps

    # F_DT: worst scenario population standard deviation of DT loading percentages.
    d_s = np.std(load_pct, axis=2, ddof=0)
    f_dt = np.max(d_s, axis=1)

    # F_phase: capacity-referenced phase imbalance:
    # 100 * max|S_phase - mean(S_R,S_Y,S_B)| / (C_i/3)
    mean_phase = s_phase.mean(axis=3, keepdims=True)
    phase_dev = np.max(np.abs(s_phase - mean_phase), axis=3)
    b = 100.0 * phase_dev / (model.capacities[None, None, :] / 3.0)
    f_phase = np.max(b, axis=(1, 2))

    # Headroom penalty against preferred loading target.
    h_s = np.sum(np.maximum(load_pct - cfg.preferred_loading, 0.0) ** 2, axis=2)
    f_headroom = np.max(h_s, axis=1)

    tf = model.transfer_flag[consumer_axis, assignments]
    rf = model.rephase_flag[consumer_axis, assignments]
    dist = model.distance_m[consumer_axis, assignments]

    n_t = tf.sum(axis=1)
    n_r = rf.sum(axis=1)
    d_transfer = (dist * tf).sum(axis=1)

    f_change = 3.0 * n_t + n_r + 0.01 * d_transfer

    reciprocal = np.array(
        [reciprocal_pair_count(a, model) for a in assignments],
        dtype=int,
    )

    # Hard constraint violation measure.
    dt_excess = np.maximum(load_pct - cfg.hard_dt_limit, 0.0)
    phase_excess = np.maximum(phase_load_pct - cfg.hard_phase_limit, 0.0)

    violation = (
        np.sum(dt_excess ** 2, axis=(1, 2))
        + np.sum(phase_excess ** 2, axis=(1, 2, 3))
        + 25.0 * np.maximum(n_t - cfg.max_transfers, 0) ** 2
        + 25.0 * np.maximum(n_r - cfg.max_rephases, 0) ** 2
    )

    if cfg.no_reciprocal:
        violation = violation + 100.0 * reciprocal

    feasible = (
        (np.max(load_pct, axis=(1, 2)) <= cfg.hard_dt_limit + 1e-9)
        & (np.max(phase_load_pct, axis=(1, 2, 3)) <= cfg.hard_phase_limit + 1e-9)
        & (n_t <= cfg.max_transfers)
        & (n_r <= cfg.max_rephases)
    )
    if cfg.no_reciprocal:
        feasible &= reciprocal == 0

    return {
        "p": p,
        "q": q,
        "s_phase": s_phase,
        "s_dt": s_dt,
        "load_pct": load_pct,
        "phase_load_pct": phase_load_pct,
        "f_dt": f_dt,
        "f_phase": f_phase,
        "f_headroom": f_headroom,
        "n_t": n_t,
        "n_r": n_r,
        "distance": d_transfer,
        "f_change": f_change,
        "reciprocal": reciprocal,
        "violation": violation,
        "feasible": feasible,
    }


def baseline_refs(model: Model, cfg: EvalConfig) -> BaselineRefs:
    baseline = np.zeros(len(model.consumers), dtype=int)
    r = raw_eval_batch(baseline, model, cfg)
    return BaselineRefs(
        f_dt=float(r["f_dt"][0]),
        f_phase=float(r["f_phase"][0]),
        f_headroom=float(r["f_headroom"][0]),
    )


def evaluate_batch(
    assignments: np.ndarray,
    model: Model,
    cfg: EvalConfig,
    refs: BaselineRefs,
):
    r = raw_eval_batch(assignments, model, cfg)

    weights = np.array([cfg.w_dt, cfg.w_phase, cfg.w_headroom, cfg.w_change], dtype=float)
    if weights.sum() <= 0:
        weights[:] = 0.25
    else:
        weights /= weights.sum()

    z_dt = r["f_dt"] / max(refs.f_dt, 1e-12)
    z_phase = r["f_phase"] / max(refs.f_phase, 1e-12)
    z_headroom = r["f_headroom"] / max(refs.f_headroom, 1e-12)
    z_change = r["f_change"] / REPORT_F_CHANGE_MAX

    j = (
        weights[0] * z_dt
        + weights[1] * z_phase
        + weights[2] * z_headroom
        + weights[3] * z_change
    )

    # Feasible solutions are compared on J.
    # Infeasible solutions are kept far outside feasible objective values.
    fitness = np.where(r["feasible"], j, 1000.0 + r["violation"])

    r.update(
        {
            "z_dt": z_dt,
            "z_phase": z_phase,
            "z_headroom": z_headroom,
            "z_change": z_change,
            "j": j,
            "fitness": fitness,
        }
    )
    return r


def repair_assignment(assignment: np.ndarray, model: Model, cfg: EvalConfig) -> np.ndarray:
    """
    Light deterministic repair BEFORE penalty evaluation.

    Repair covers:
      - excess number of transfers,
      - excess number of same-DT rephasings,
      - reciprocal pair transfers.

    Electrical DT/phase violations remain in the penalty term so the
    swarm can still learn from infeasible network states.
    """
    a = assignment.copy()

    # Action count repairs: revert high-distance / lower-impact actions first.
    transfer_idx = [i for i, k in enumerate(a) if model.transfer_flag[i, k] == 1]
    if len(transfer_idx) > cfg.max_transfers:
        # Prefer reverting longer-distance and smaller-kVA transfers.
        transfer_idx.sort(
            key=lambda i: (model.distance_m[i, a[i]], -model.peak_s_kva[i]),
            reverse=True,
        )
        for i in transfer_idx[: len(transfer_idx) - cfg.max_transfers]:
            a[i] = 0

    rephase_idx = [i for i, k in enumerate(a) if model.rephase_flag[i, k] == 1]
    if len(rephase_idx) > cfg.max_rephases:
        # Revert smaller-kVA phase changes first.
        rephase_idx.sort(key=lambda i: model.peak_s_kva[i])
        for i in rephase_idx[: len(rephase_idx) - cfg.max_rephases]:
            a[i] = 0

    if cfg.no_reciprocal:
        # Repeatedly remove one direction from each reciprocal pair.
        while True:
            edges = {}
            reciprocal = None

            for i, k in enumerate(a):
                if model.transfer_flag[i, k] == 0:
                    continue
                src = int(model.existing_dt_idx[i])
                dst = int(model.state_dt[i, k])
                edges.setdefault((src, dst), []).append(i)

            for edge in list(edges):
                rev = (edge[1], edge[0])
                if rev in edges:
                    reciprocal = (edge, rev)
                    break

            if reciprocal is None:
                break

            e1, e2 = reciprocal
            candidates = edges[e1] + edges[e2]
            # Revert the longest-distance transfer involved.
            revert_i = max(candidates, key=lambda i: model.distance_m[i, a[i]])
            a[revert_i] = 0

    return a


def repair_batch(assignments: np.ndarray, model: Model, cfg: EvalConfig) -> np.ndarray:
    return np.vstack([repair_assignment(a, model, cfg) for a in assignments])


def report_assignment(model: Model) -> np.ndarray:
    a = np.zeros(len(model.consumers), dtype=int)
    ids = model.consumers["Consumer_ID"].astype(str).tolist()
    id_to_row = {cid: i for i, cid in enumerate(ids)}

    missing = []
    for cid, (target_dt, target_phase) in REPORT_FINAL_STATES.items():
        if cid not in id_to_row:
            missing.append(cid)
            continue
        i = id_to_row[cid]
        match = None
        for k, s in enumerate(model.state_lists[i]):
            if s["dt"] == target_dt and s["phase"] == target_phase:
                match = k
                break
        if match is None:
            missing.append(f"{cid}->{target_dt}/{target_phase}")
        else:
            a[i] = match

    if missing:
        raise ValueError(
            "The workbook does not match the report action set. Missing states: "
            + ", ".join(missing)
        )
    return a


def matches_report_dataset(model: Model) -> bool:
    """Compare actual input tables, including powers and eligibility, not filenames."""
    consumers, transformers = load_workbook_bytes((APP_DIR / DEFAULT_WORKBOOK).read_bytes())
    return model.consumers.equals(consumers) and model.dt_master.equals(transformers)


def seed_assignment(model: Model) -> np.ndarray:
    """Load the retained independently constructed feasible seed by consumer ID."""
    states = json.loads((REPORT_DIR / "feasible_seed.json").read_text())
    assignment = np.zeros(len(model.consumers), dtype=int)
    for i, cid in enumerate(model.consumers["Consumer_ID"].astype(str)):
        destination, phase = states[cid]
        assignment[i] = next(k for k, state in enumerate(model.state_lists[i])
                             if (state["dt"], state["phase"]) == (destination, phase))
    return assignment


def find_feasible(model: Model, cfg: EvalConfig, refs: BaselineRefs,
                  start: np.ndarray) -> np.ndarray | None:
    """The bounded feasibility descent used to construct the report seed."""
    assignment = start.copy()
    rng = np.random.default_rng(1956)
    for _ in range(100):
        current = evaluate_batch(assignment, model, cfg, refs)
        if bool(current["feasible"][0]):
            return assignment
        candidates = []
        for i in model.ctrl_consumers:
            for k in range(model.state_count[i]):
                if k != assignment[i]:
                    candidate = assignment.copy()
                    candidate[i] = k
                    candidates.append(candidate)
        if not candidates:
            return None
        evaluated = evaluate_batch(np.array(candidates), model, cfg, refs)
        index = np.lexsort((evaluated["j"], evaluated["violation"]))[0]
        if evaluated["violation"][index] < current["violation"][0] - 1e-10:
            assignment = candidates[index]
            continue
        improved = None
        for batch in range(40):
            proposals = np.tile(assignment, (250, 1))
            count = min(2 + batch // 10, len(model.ctrl_consumers))
            for proposal in proposals:
                for i in rng.choice(model.ctrl_consumers, size=count, replace=False):
                    proposal[i] = rng.integers(model.state_count[i])
            evaluated = evaluate_batch(proposals, model, cfg, refs)
            index = int(np.argmin(evaluated["violation"]))
            if evaluated["violation"][index] < current["violation"][0] - 1e-10:
                improved = proposals[index].copy()
                break
        if improved is None:
            return None
        assignment = improved
    return None


def find_demo_warm_start(model: Model, cfg: EvalConfig, refs: BaselineRefs,
                         rng: np.random.Generator) -> np.ndarray:
    # Do not consume the PSO random stream while constructing its seed.
    # This seed was generated from the baseline, not from the final recommendation.
    if matches_report_dataset(model):
        saved = seed_assignment(model)
        if bool(evaluate_batch(saved, model, cfg, refs)["feasible"][0]):
            return saved
    baseline = np.zeros(len(model.consumers), dtype=int)
    seed_rng = np.random.default_rng(1956)
    for attempt in range(4):
        start = baseline.copy()
        if attempt:
            count = min(10, len(model.ctrl_consumers))
            for i in seed_rng.choice(model.ctrl_consumers, size=count, replace=False):
                start[i] = seed_rng.integers(model.state_count[i])
        feasible = find_feasible(model, cfg, refs, start)
        if feasible is not None:
            return feasible
    # A custom dataset or tighter limits may defeat bounded seed construction.
    # Let the penalized search proceed, without calling this starting point feasible.
    return baseline


def local_search(
    start: np.ndarray,
    model: Model,
    cfg: EvalConfig,
    refs: BaselineRefs,
    max_passes: int = 200,
):
    current = start.copy()
    cur_ev = evaluate_batch(current, model, cfg, refs)
    if not bool(cur_ev["feasible"][0]):
        return current, cur_ev, 0

    cur_j = float(cur_ev["j"][0])
    accepted = 0

    for _ in range(max_passes):
        candidates = []
        descriptors = []

        for i in model.ctrl_consumers:
            current_k = int(current[i])
            for k in range(model.state_count[i]):
                if k == current_k:
                    continue
                c = current.copy()
                c[i] = k
                c = repair_assignment(c, model, cfg)
                candidates.append(c)
                descriptors.append((i, k))

        if not candidates:
            break

        batch = np.vstack(candidates)
        ev = evaluate_batch(batch, model, cfg, refs)

        feasible_idx = np.flatnonzero(ev["feasible"])
        if len(feasible_idx) == 0:
            break

        best_local_idx = feasible_idx[np.argmin(ev["j"][feasible_idx])]
        best_j = float(ev["j"][best_local_idx])

        if best_j < cur_j - 1e-9:
            current = batch[best_local_idx].copy()
            cur_j = best_j
            accepted += 1
        else:
            break

    final_ev = evaluate_batch(current, model, cfg, refs)
    return current, final_ev, accepted


def run_pso(
    model: Model,
    cfg: EvalConfig,
    refs: BaselineRefs,
    swarm_size: int,
    max_iter: int,
    seed: int,
    inertia_start: float,
    inertia_end: float,
    c1: float,
    c2: float,
    vmax: float,
    patience: int,
    animate: bool,
    display_every: int,
    delay: float,
    use_exact_report_start: bool,
):
    rng = np.random.default_rng(seed)
    dim = len(model.ctrl_consumers)

    if use_exact_report_start:
        warm_assignment = report_assignment(model)
    else:
        warm_assignment = find_demo_warm_start(model, cfg, refs, rng)

    warm_pos = assignment_to_position(warm_assignment, model)

    positions = rng.random((swarm_size, dim))

    # Half the swarm starts around the feasible warm region.
    nearby = max(1, swarm_size // 2)
    positions[:nearby] = np.clip(
        warm_pos[None, :] + rng.normal(0.0, 0.18, size=(nearby, dim)),
        0.0,
        0.999999,
    )
    positions[0] = warm_pos

    velocities = rng.uniform(-0.05, 0.05, size=(swarm_size, dim))

    assignments = decode_positions(positions, model)
    assignments = repair_batch(assignments, model, cfg)
    positions = np.vstack([assignment_to_position(a, model) for a in assignments])

    ev = evaluate_batch(assignments, model, cfg, refs)

    pbest_pos = positions.copy()
    pbest_fit = ev["fitness"].copy()
    pbest_assignment = assignments.copy()

    g_idx = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_assignment = pbest_assignment[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])

    history = []
    stagnant = 0

    ui = {}
    if animate:
        ui["progress"] = st.progress(0.0)
        metric_cols = st.columns(4)
        ui["m_iter"] = metric_cols[0].empty()
        ui["m_best"] = metric_cols[1].empty()
        ui["m_feas"] = metric_cols[2].empty()
        ui["m_dist"] = metric_cols[3].empty()

        ui["conv"] = st.empty()

    def record(t, ev_now, pos_now):
        nonlocal history
        g_ev = evaluate_batch(gbest_assignment, model, cfg, refs)
        feasible_g = bool(g_ev["feasible"][0])
        best_j = float(g_ev["j"][0]) if feasible_g else np.nan

        distance = np.linalg.norm(pos_now - gbest_pos[None, :], axis=1) / max(math.sqrt(dim), 1.0)
        mean_distance = float(np.mean(distance))

        history.append(
            {
                "Iteration": t,
                "Best_J": best_j,
                "Feasible_particles": int(ev_now["feasible"].sum()),
                "Mean_distance_to_gbest": mean_distance,
            }
        )

        if animate and (t == 0 or t % display_every == 0 or t == max_iter or stagnant >= patience):
            ui["progress"].progress(min(t / max_iter, 1.0))
            ui["m_iter"].metric("Iteration", t)
            ui["m_best"].metric("Best feasible J", f"{best_j:.5f}" if np.isfinite(best_j) else "None")
            ui["m_feas"].metric("Feasible particles", f"{int(ev_now['feasible'].sum())}/{swarm_size}")
            ui["m_dist"].metric("Mean distance to gbest", f"{mean_distance:.4f}")

            hdf = pd.DataFrame(history)
            ui["conv"].line_chart(
                hdf.set_index("Iteration")[["Best_J"]],
                height=300,
            )

            if delay > 0:
                time.sleep(delay)

    record(0, ev, positions)

    for t in range(1, max_iter + 1):
        # Selected configuration uses linearly decreasing inertia.
        frac = (t - 1) / max(max_iter - 1, 1)
        w = inertia_start + frac * (inertia_end - inertia_start)

        r1 = rng.random((swarm_size, dim))
        r2 = rng.random((swarm_size, dim))

        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos[None, :] - positions)
        )
        velocities = np.clip(velocities, -vmax, vmax)

        positions = np.clip(positions + velocities, 0.0, 0.999999)

        assignments = decode_positions(positions, model)
        assignments = repair_batch(assignments, model, cfg)
        positions = np.vstack([assignment_to_position(a, model) for a in assignments])

        ev = evaluate_batch(assignments, model, cfg, refs)

        improved = ev["fitness"] < pbest_fit
        pbest_fit[improved] = ev["fitness"][improved]
        pbest_pos[improved] = positions[improved]
        pbest_assignment[improved] = assignments[improved]

        new_g_idx = int(np.argmin(pbest_fit))
        new_g_fit = float(pbest_fit[new_g_idx])

        if new_g_fit < gbest_fit - 1e-10:
            gbest_fit = new_g_fit
            gbest_pos = pbest_pos[new_g_idx].copy()
            gbest_assignment = pbest_assignment[new_g_idx].copy()
            stagnant = 0
        else:
            stagnant += 1

        record(t, ev, positions)

        if stagnant >= patience:
            break

    if animate:
        ui["progress"].progress(1.0)

    best_ev = evaluate_batch(gbest_assignment, model, cfg, refs)
    return {
        "assignment": gbest_assignment,
        "eval": best_ev,
        "history": pd.DataFrame(history),
        "iterations": int(history[-1]["Iteration"]),
        "warm_assignment": warm_assignment,
    }


def assignment_loading_table(
    assignment: np.ndarray,
    model: Model,
    cfg: EvalConfig,
    refs: BaselineRefs,
):
    ev = evaluate_batch(assignment, model, cfg, refs)
    peak = ev["load_pct"][0, 0]
    solar = ev["load_pct"][0, 1]
    peak_phase = ev["phase_load_pct"][0, 0]
    solar_phase = ev["phase_load_pct"][0, 1]

    return pd.DataFrame(
        {
            "DT": model.dt_names,
            "Capacity kVA": model.capacities,
            "Peak Import %": peak,
            "High Solar %": solar,
            "Peak R %": peak_phase[:, 0],
            "Peak Y %": peak_phase[:, 1],
            "Peak B %": peak_phase[:, 2],
            "High Solar max phase %": solar_phase.max(axis=1),
        }
    )


def action_table(assignment: np.ndarray, model: Model):
    rows = []
    c = model.consumers.reset_index(drop=True)

    for i, k in enumerate(assignment):
        if k == 0:
            continue

        state = model.state_lists[i][int(k)]
        existing_dt = str(c.loc[i, "Existing_DT"])
        existing_phase = str(c.loc[i, "Existing_Phase"])

        if state["dt"] != existing_dt:
            action = "DT Transfer"
        elif state["phase"] != existing_phase:
            action = "Same-DT Rephase"
        else:
            continue

        rows.append(
            {
                "Consumer": str(c.loc[i, "Consumer_ID"]),
                "Category": str(c.loc[i, "Consumer_Category"]),
                "Existing DT": existing_dt,
                "Existing Phase": existing_phase,
                "Recommended DT": state["dt"],
                "Recommended Phase": state["phase"],
                "Action": action,
                "Peak kVA": float(c.loc[i, "Peak_Import_S_kVA"]),
                "Distance m": state["distance"],
            }
        )

    return pd.DataFrame(rows)


def metric_block(ev):
    cols = st.columns(3)
    cols[0].metric("Objective J", f"{float(ev['j'][0]):.6f}")
    cols[1].metric("Max DT loading", f"{float(np.max(ev['load_pct'][0])):.2f}%")
    cols[2].metric("Max phase loading", f"{float(np.max(ev['phase_load_pct'][0])):.2f}%")
    feasible = "Feasible" if bool(ev["feasible"][0]) else "Infeasible"
    st.write(f"**{feasible}** · {int(ev['n_t'][0])} transfers · "
             f"{int(ev['n_r'][0])} same-DT rephasings · {float(ev['distance'][0]):.0f} m")
    with st.expander("Objective components"):
        st.write({name: round(float(ev[key][0]), 6) for name, key in
                  [("DT imbalance", "z_dt"), ("Phase imbalance", "z_phase"),
                   ("Headroom", "z_headroom"), ("Field effort", "z_change")]})


# ============================================================
# Streamlit UI
# ============================================================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Course method: Particle Swarm Optimization (PSO) | "
        "Discrete representation: random-key encoding | "
        "Final refinement: local search"
    )

    raw, workbook_name = workbook_source()

    try:
        consumers, dt_master = load_workbook_bytes(raw)
        model = build_model(consumers, dt_master)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    st.sidebar.caption(f"Dataset: {len(model.consumers)} consumers · {len(model.dt_names)} transformers")

    preset = st.sidebar.selectbox("Run preset", ["Validation run 1004", "Recommendation origin: run 101"])
    origin_run = preset == "Recommendation origin: run 101"
    with st.sidebar.expander("Advanced settings"):
        swarm_size = st.slider("Swarm size", 10, 100, 60, 10)
        max_iter = st.slider("Maximum iterations", 20, 300, 200, 10)
        seed = st.number_input("PSO random seed", min_value=0, max_value=999999,
                                       value=101 if origin_run else 1004, step=1, key=f"seed_{preset}")

        inertia_start = st.number_input("Initial inertia w", value=0.70 if origin_run else 0.90, step=0.05, format="%.2f", key=f"w0_{preset}")
        inertia_end = st.number_input("Final inertia w", value=0.70 if origin_run else 0.40, step=0.05, format="%.2f", key=f"w1_{preset}")
        c1 = st.number_input("c1", value=2.00, step=0.10, format="%.2f")
        c2 = st.number_input("c2", value=2.00, step=0.10, format="%.2f")
        vmax = st.number_input("Velocity clamp", value=0.20, step=0.05, format="%.2f")
        patience = st.slider("Early stopping patience", 5, 60, 30, 5)

        st.subheader("Operational constraints")
        max_transfers = st.slider("Maximum DT transfers", 1, 30, 20)
        max_rephases = st.slider("Maximum same-DT phase changes", 1, 40, 25)
        preferred_loading = st.slider("Preferred DT loading target (%)", 80, 98, 90)
        no_reciprocal = st.checkbox("Prohibit reciprocal DT-pair transfers", value=True)

        st.subheader("Objective weights")
        w_dt = st.slider("DT imbalance weight", 0.0, 1.0, 0.25, 0.05)
        w_phase = st.slider("Phase imbalance weight", 0.0, 1.0, 0.25, 0.05)
        w_headroom = st.slider("Headroom weight", 0.0, 1.0, 0.25, 0.05)
        w_change = st.slider("Field-change weight", 0.0, 1.0, 0.25, 0.05)

    cfg = EvalConfig(
        max_transfers=max_transfers,
        max_rephases=max_rephases,
        preferred_loading=float(preferred_loading),
        no_reciprocal=no_reciprocal,
        w_dt=w_dt,
        w_phase=w_phase,
        w_headroom=w_headroom,
        w_change=w_change,
    )
    # Freeze normalization at the original 90% target, as in the revised report.
    refs = baseline_refs(model, EvalConfig())

    st.write("Watch Particle Swarm Optimization improve transformer and phase loading. "
             "Click Run PSO to start; lower objective J is better.")

    animate = st.checkbox("Animate during the run", value=True)
    display_every, delay = 5, 0.0
    do_local = st.checkbox("Apply local-search refinement after PSO", value=origin_run, key=f"local_{preset}")
    if origin_run:
        st.caption("At report defaults: PSO J=0.363934; nine accepted local moves give J=0.333554.")
    else:
        st.caption("Figure 2: at report defaults, seed 1004 stops at iteration 64 with J=0.369170. "
                   "Leave local search off to show this validation result.")

    if st.button("Run PSO", type="primary"):
        with st.spinner("Running Particle Swarm Optimization..."):
            result = run_pso(
                model=model,
                cfg=cfg,
                refs=refs,
                swarm_size=swarm_size,
                max_iter=max_iter,
                seed=int(seed),
                inertia_start=float(inertia_start),
                inertia_end=float(inertia_end),
                c1=float(c1),
                c2=float(c2),
                vmax=float(vmax),
                patience=int(patience),
                animate=animate,
                display_every=int(display_every),
                delay=float(delay),
                use_exact_report_start=False,
            )

        if not bool(evaluate_batch(result["warm_assignment"], model, cfg, refs)["feasible"][0]):
            st.warning("The bounded initialization did not find a feasible seed under these inputs. "
                       "The run searched using constraint penalties; check its final feasibility.")
        pure_assignment = result["assignment"]
        pure_ev = result["eval"]

        st.info(f"PSO stopped after {result['iterations']} iterations.")
        if not bool(pure_ev["feasible"][0]):
            st.warning("No feasible solution was found in this run; these actions are not a recommendation.")
        st.markdown("### Best pure-PSO solution from this live run")
        metric_block(pure_ev)

        final_assignment = pure_assignment
        final_ev = pure_ev

        if do_local and bool(pure_ev["feasible"][0]):
            with st.spinner("Applying deterministic single-move local search..."):
                final_assignment, final_ev, accepted = local_search(
                    pure_assignment, model, cfg, refs
                )
            st.markdown("### After local-search refinement")
            st.caption(f"Accepted improving move-and-repair steps: {accepted}")
            metric_block(final_ev)

        if not animate:
            st.line_chart(result["history"].set_index("Iteration")[["Best_J"]], height=300)

        st.session_state["last_assignment"] = final_assignment
        st.session_state["last_history"] = result["history"]


    st.caption("Synthetic 8-transformer, 140-consumer study. Results depend on the selected settings.")


if __name__ == "__main__":
    main()
