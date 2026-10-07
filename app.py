
from __future__ import annotations

import io
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

APP_TITLE = "EM618 – DT Load Balancing PSO Simulator"
DEFAULT_WORKBOOK = "EM618_DT_Load_Balancing_Synthetic_Baseline.xlsx"

PHASES = ("R", "Y", "B")
PHASE_TO_INDEX = {"R": 0, "Y": 1, "B": 2, "RYB": 3}

# Report normalization for field effort:
# 3*20 transfers + 25 rephasings + 0.01*(20*140 m) = 113
REPORT_F_CHANGE_MAX = 113.0

# Exact final hybrid actions reported in the assignment.
# Consumers not listed here stay in their baseline state.
REPORT_FINAL_STATES: Dict[str, Tuple[str, str]] = {
    # 6 DT transfers
    "C0002": ("DT-4", "RYB"),
    "C0005": ("DT-2", "Y"),
    "C0049": ("DT-5", "Y"),
    "C0054": ("DT-5", "R"),
    "C0056": ("DT-5", "RYB"),
    "C0091": ("DT-5", "RYB"),

    # 12 same-DT phase changes
    "C0003": ("DT-1", "B"),
    "C0004": ("DT-1", "Y"),
    "C0006": ("DT-1", "R"),
    "C0008": ("DT-1", "B"),
    "C0010": ("DT-1", "R"),
    "C0012": ("DT-2", "B"),
    "C0029": ("DT-3", "B"),
    "C0045": ("DT-4", "B"),
    "C0046": ("DT-4", "B"),
    "C0048": ("DT-4", "Y"),
    "C0097": ("DT-6", "R"),
    "C0138": ("DT-8", "Y"),
}


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
    st.sidebar.subheader("Dataset")
    uploaded = st.sidebar.file_uploader(
        "Upload the synthetic baseline workbook",
        type=["xlsx"],
        help="Use EM618_DT_Load_Balancing_Synthetic_Baseline.xlsx",
    )

    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name

    local = Path(DEFAULT_WORKBOOK)
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
        for i in transfer_idx[cfg.max_transfers:]:
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


def find_demo_warm_start(
    model: Model,
    cfg: EvalConfig,
    refs: BaselineRefs,
    rng: np.random.Generator,
    target_low: float = 0.38,
    target_high: float = 0.52,
) -> np.ndarray:
    """
    Create a feasible, deliberately non-final warm start so the live
    simulator can visibly improve instead of starting at the report optimum.
    """
    base = report_assignment(model)
    best = None
    best_gap = float("inf")

    for _ in range(1200):
        a = base.copy()

        # Perturb several controllable consumers around the known feasible region.
        m = int(rng.integers(3, min(15, len(model.ctrl_consumers)) + 1))
        chosen = rng.choice(model.ctrl_consumers, size=m, replace=False)

        for i in chosen:
            # Mix reversion to baseline and another random feasible state.
            if rng.random() < 0.55:
                a[i] = 0
            else:
                a[i] = int(rng.integers(0, model.state_count[i]))

        a = repair_assignment(a, model, cfg)
        ev = evaluate_batch(a, model, cfg, refs)

        if bool(ev["feasible"][0]):
            j = float(ev["j"][0])
            if target_low <= j <= target_high:
                return a

            midpoint = (target_low + target_high) / 2
            gap = abs(j - midpoint)
            if j > float(evaluate_batch(base, model, cfg, refs)["j"][0]) and gap < best_gap:
                best = a.copy()
                best_gap = gap

    return best if best is not None else base


def local_search(
    start: np.ndarray,
    model: Model,
    cfg: EvalConfig,
    refs: BaselineRefs,
    max_passes: int = 40,
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

        chart_cols = st.columns(2)
        ui["conv"] = chart_cols[0].empty()
        ui["swarm"] = chart_cols[1].empty()
        ui["loading"] = st.empty()

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

        if animate and (t == 0 or t % display_every == 0 or t == max_iter):
            ui["progress"].progress(min(t / max_iter, 1.0))
            ui["m_iter"].metric("Iteration", t)
            ui["m_best"].metric("Best feasible J", f"{best_j:.5f}" if np.isfinite(best_j) else "None")
            ui["m_feas"].metric("Feasible particles", f"{int(ev_now['feasible'].sum())}/{swarm_size}")
            ui["m_dist"].metric("Mean distance to gbest", f"{mean_distance:.4f}")

            hdf = pd.DataFrame(history)
            ui["conv"].line_chart(
                hdf.set_index("Iteration")[["Best_J", "Mean_distance_to_gbest"]],
                height=300,
            )

            particle_df = pd.DataFrame(
                {
                    "Distance_to_gbest": distance,
                    "J_if_feasible": np.where(ev_now["feasible"], ev_now["j"], np.nan),
                }
            ).dropna()

            if len(particle_df):
                ui["swarm"].scatter_chart(
                    particle_df,
                    x="Distance_to_gbest",
                    y="J_if_feasible",
                    height=300,
                )
            else:
                ui["swarm"].info("No feasible particles in this snapshot.")

            load_df = assignment_loading_table(gbest_assignment, model, cfg, refs)
            ui["loading"].bar_chart(
                load_df.set_index("DT")[["Peak Import %", "High Solar %"]],
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
    cols = st.columns(6)
    cols[0].metric("J", f"{float(ev['j'][0]):.5f}")
    cols[1].metric("Z_DT", f"{float(ev['z_dt'][0]):.4f}")
    cols[2].metric("Z_phase", f"{float(ev['z_phase'][0]):.4f}")
    cols[3].metric("Z_headroom", f"{float(ev['z_headroom'][0]):.4f}")
    cols[4].metric("Z_change", f"{float(ev['z_change'][0]):.4f}")
    cols[5].metric("Feasible", "YES" if bool(ev["feasible"][0]) else "NO")

    cols2 = st.columns(5)
    cols2[0].metric("DT transfers", int(ev["n_t"][0]))
    cols2[1].metric("Phase changes", int(ev["n_r"][0]))
    cols2[2].metric("Transfer distance", f"{float(ev['distance'][0]):.0f} m")
    cols2[3].metric("Max DT loading", f"{float(np.max(ev['load_pct'][0])):.2f}%")
    cols2[4].metric("Max phase loading", f"{float(np.max(ev['phase_load_pct'][0])):.2f}%")


# ============================================================
# Streamlit UI
# ============================================================

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

st.sidebar.success(f"Loaded: {workbook_name}")

st.sidebar.subheader("Optimization controls")
swarm_size = st.sidebar.slider("Swarm size", 10, 100, 60, 10)
max_iter = st.sidebar.slider("Maximum iterations", 20, 300, 200, 10)
seed = st.sidebar.number_input("PSO random seed", min_value=0, max_value=999999, value=1019, step=1)

inertia_start = st.sidebar.number_input("Initial inertia w", value=0.90, step=0.05, format="%.2f")
inertia_end = st.sidebar.number_input("Final inertia w", value=0.40, step=0.05, format="%.2f")
c1 = st.sidebar.number_input("c1", value=2.00, step=0.10, format="%.2f")
c2 = st.sidebar.number_input("c2", value=2.00, step=0.10, format="%.2f")
vmax = st.sidebar.number_input("Velocity clamp", value=0.20, step=0.05, format="%.2f")
patience = st.sidebar.slider("Early stopping patience", 5, 60, 30, 5)

st.sidebar.subheader("Operational constraints")
max_transfers = st.sidebar.slider("Maximum DT transfers", 1, 30, 20)
max_rephases = st.sidebar.slider("Maximum same-DT phase changes", 1, 40, 25)
preferred_loading = st.sidebar.slider("Preferred DT loading target (%)", 80, 98, 90)
no_reciprocal = st.sidebar.checkbox("Prohibit reciprocal DT-pair transfers", value=True)

st.sidebar.subheader("Objective weights")
w_dt = st.sidebar.slider("DT imbalance weight", 0.0, 1.0, 0.25, 0.05)
w_phase = st.sidebar.slider("Phase imbalance weight", 0.0, 1.0, 0.25, 0.05)
w_headroom = st.sidebar.slider("Headroom weight", 0.0, 1.0, 0.25, 0.05)
w_change = st.sidebar.slider("Field-change weight", 0.0, 1.0, 0.25, 0.05)

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
refs = baseline_refs(model, cfg)

tabs = st.tabs(
    [
        "1. Baseline",
        "2. Random-Key Encoding",
        "3. Live PSO",
        "4. Report Solution",
        "5. Sensitivity Demo",
    ]
)

with tabs[0]:
    st.subheader("Baseline network")
    st.write(
        f"{len(model.consumers)} consumers, {len(model.dt_names)} DTs, "
        f"{len(model.ctrl_consumers)} consumers with at least one decision alternative."
    )

    baseline = np.zeros(len(model.consumers), dtype=int)
    b_ev = evaluate_batch(baseline, model, cfg, refs)
    metric_block(b_ev)

    st.markdown(
        rf"""
**Baseline normalization references calculated from the workbook**

- \(F_{{DT,0}} = {refs.f_dt:.4f}\)
- \(F_{{phase,0}} = {refs.f_phase:.4f}\)
- \(F_{{headroom,0}} = {refs.f_headroom:.4f}\)

The baseline has \(J=0.75\) under equal weights because the first three normalized
technical terms equal 1 and the change term equals 0. The baseline is nevertheless
**infeasible** because DT/phase hard limits are violated.
"""
    )

    btab = assignment_loading_table(baseline, model, cfg, refs)
    st.dataframe(
        btab.style.format(
            {
                "Capacity kVA": "{:.0f}",
                "Peak Import %": "{:.2f}",
                "High Solar %": "{:.2f}",
                "Peak R %": "{:.2f}",
                "Peak Y %": "{:.2f}",
                "Peak B %": "{:.2f}",
                "High Solar max phase %": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    st.bar_chart(
        btab.set_index("DT")[["Peak Import %", "High Solar %"]],
        height=350,
    )

with tabs[1]:
    st.subheader("How continuous PSO positions become discrete consumer decisions")
    st.latex(r"k_j = 1 + \left\lfloor x_j |K_j| \right\rfloor")
    st.write(
        r"PSO moves a continuous key $x_j\\in[0,1)$ . The interval is divided into "
        r"equal regions, one for each feasible state in consumer $j$'s set $K_j$."
    )

    ids = model.consumers["Consumer_ID"].astype(str).tolist()
    chosen_id = st.selectbox("Choose a consumer", ids, index=4)
    i = ids.index(chosen_id)
    states = model.state_lists[i]

    demo_df = []
    k = len(states)
    for idx, s in enumerate(states):
        demo_df.append(
            {
                "Discrete state": idx + 1,
                "Random-key interval": f"[{idx/k:.3f}, {(idx+1)/k:.3f})",
                "DT": s["dt"],
                "Phase": s["phase"],
                "Transfer?": "Yes" if s["transfer"] else "No",
                "Rephase?": "Yes" if s["rephase"] else "No",
                "Distance m": s["distance"],
            }
        )
    st.dataframe(pd.DataFrame(demo_df), use_container_width=True)

    x_demo = st.slider("Try random-key x", 0.0, 0.999, 0.55, 0.001)
    decoded = min(int(math.floor(x_demo * k)), k - 1)
    s = states[decoded]
    st.success(
        f"x={x_demo:.3f} decodes to State {decoded+1}: {s['dt']} / {s['phase']}"
    )

with tabs[2]:
    st.subheader("Live PSO convergence")

    st.latex(
        r"v_{pj}^{t+1}=w(t)v_{pj}^{t}+c_1r_1(pbest_{pj}-x_{pj}^{t})"
        r"+c_2r_2(gbest_j-x_{pj}^{t})"
    )
    st.latex(r"x_{pj}^{t+1}=x_{pj}^{t}+v_{pj}^{t+1}")

    st.info(
        "For a visible demonstration, the default mode starts near a feasible but deliberately "
        "non-final configuration. Because PSO is stochastic, a new run can follow a different "
        "trajectory from the historical run in the report."
    )

    animate = st.checkbox("Animate during the run", value=True)
    display_every = st.slider("Refresh display every N iterations", 1, 20, 5)
    delay = st.slider("Animation delay per refresh (seconds)", 0.0, 0.25, 0.02, 0.01)
    use_exact_report_start = st.checkbox(
        "Start from exact final report solution (reproduction mode)",
        value=False,
        help="If enabled, the best J may already start near the final report value, so convergence is less visually interesting.",
    )
    do_local = st.checkbox("Apply local-search refinement after PSO", value=True)

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
                use_exact_report_start=use_exact_report_start,
            )

        pure_assignment = result["assignment"]
        pure_ev = result["eval"]

        st.success(f"PSO stopped after {result['iterations']} iterations.")
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
            st.caption(f"Accepted improving single-consumer moves: {accepted}")
            metric_block(final_ev)

        st.markdown("### Final DT loading from this run")
        load = assignment_loading_table(final_assignment, model, cfg, refs)
        st.dataframe(
            load.style.format(
                {
                    "Capacity kVA": "{:.0f}",
                    "Peak Import %": "{:.2f}",
                    "High Solar %": "{:.2f}",
                    "Peak R %": "{:.2f}",
                    "Peak Y %": "{:.2f}",
                    "Peak B %": "{:.2f}",
                    "High Solar max phase %": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        st.markdown("### Consumer actions")
        actions = action_table(final_assignment, model)
        if len(actions):
            st.dataframe(
                actions.style.format({"Peak kVA": "{:.3f}", "Distance m": "{:.0f}"}),
                use_container_width=True,
            )
        else:
            st.write("No changes selected.")

        st.markdown("### Convergence history")
        st.dataframe(result["history"], use_container_width=True)

        st.session_state["last_assignment"] = final_assignment
        st.session_state["last_history"] = result["history"]

with tabs[3]:
    st.subheader("Exact final report recommendation")

    try:
        r_assignment = report_assignment(model)
        r_ev = evaluate_batch(r_assignment, model, cfg, refs)

        metric_block(r_ev)

        expected = {
            "J": 0.34128,
            "Transfers": 6,
            "Rephases": 12,
            "Distance": 473.0,
        }

        st.write(
            "This tab does not rerun PSO. It loads the exact consumer states reported in the "
            "assignment and independently recomputes their network metrics from the workbook."
        )

        comparison = pd.DataFrame(
            {
                "Metric": ["J", "DT transfers", "Phase changes", "Transfer distance (m)"],
                "Expected report value": [
                    expected["J"],
                    expected["Transfers"],
                    expected["Rephases"],
                    expected["Distance"],
                ],
                "Recomputed from workbook": [
                    float(r_ev["j"][0]),
                    int(r_ev["n_t"][0]),
                    int(r_ev["n_r"][0]),
                    float(r_ev["distance"][0]),
                ],
            }
        )
        st.dataframe(comparison, use_container_width=True)

        if (
            abs(float(r_ev["j"][0]) - expected["J"]) < 0.002
            and int(r_ev["n_t"][0]) == 6
            and int(r_ev["n_r"][0]) == 12
        ):
            st.success("The workbook reproduces the report solution within rounding tolerance.")
        else:
            st.warning(
                "The loaded workbook/settings differ from the report base case. "
                "Check the preferred-loading target, weights, constraints, or workbook version."
            )

        rload = assignment_loading_table(r_assignment, model, cfg, refs)
        st.dataframe(
            rload.style.format(
                {
                    "Capacity kVA": "{:.0f}",
                    "Peak Import %": "{:.2f}",
                    "High Solar %": "{:.2f}",
                    "Peak R %": "{:.2f}",
                    "Peak Y %": "{:.2f}",
                    "Peak B %": "{:.2f}",
                    "High Solar max phase %": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        baseline = np.zeros(len(model.consumers), dtype=int)
        bload = assignment_loading_table(baseline, model, cfg, refs)

        compare = pd.DataFrame(
            {
                "DT": model.dt_names,
                "Baseline Peak %": bload["Peak Import %"],
                "Report Solution Peak %": rload["Peak Import %"],
            }
        )
        st.bar_chart(
            compare.set_index("DT")[["Baseline Peak %", "Report Solution Peak %"]],
            height=350,
        )

        st.markdown("### Exact 6 transfers + 12 phase changes")
        acts = action_table(r_assignment, model)
        st.dataframe(
            acts.style.format({"Peak kVA": "{:.3f}", "Distance m": "{:.0f}"}),
            use_container_width=True,
        )

    except Exception as exc:
        st.exception(exc)

with tabs[4]:
    st.subheader("Sensitivity demo for the report solution")

    r_assignment = report_assignment(model)
    base_ev = evaluate_batch(r_assignment, model, cfg, refs)

    peak_growth = st.slider("Uniform Peak Import demand growth (%)", 0.0, 15.0, 5.0, 0.5)
    solar_growth = st.slider("Solar-output growth (%)", 0.0, 50.0, 20.0, 5.0)

    # Sensitivity is evaluated directly from the already-aggregated solution.
    # Peak demand scales P and Q proportionally.
    p_peak = base_ev["p"][0, 0] * (1 + peak_growth / 100.0)
    q_peak = base_ev["q"][0, 0] * (1 + peak_growth / 100.0)
    s_peak = np.hypot(p_peak, q_peak)
    dt_peak = s_peak.sum(axis=1)
    peak_load = 100 * dt_peak / model.capacities
    peak_phase = 100 * s_peak / (model.capacities[:, None] / 3.0)

    # For solar growth, only the net daytime active component changes.
    # We reconstruct consumer-level daytime P and add extra PV export for solar consumers.
    # This is a scenario illustration, not a rerun of the optimization.
    c = model.consumers
    extra_pv = np.where(
        c["Solar_Status"].astype(str).str.lower().eq("yes"),
        c["PV_Output_kW"].astype(float).to_numpy() * (solar_growth / 100.0),
        0.0,
    )

    # Aggregate extra PV using the report assignment's phase/DT states.
    delta_p = np.zeros((len(model.dt_names), 3), dtype=float)
    for i, k in enumerate(r_assignment):
        dti = model.state_dt[i, k]
        ph = model.state_phase[i, k]
        if ph == 3:
            delta_p[dti, :] += (-extra_pv[i] / 3.0)
        else:
            delta_p[dti, ph] += -extra_pv[i]

    p_solar = base_ev["p"][0, 1] + delta_p
    q_solar = base_ev["q"][0, 1].copy()
    s_solar = np.hypot(p_solar, q_solar)
    solar_load = 100 * s_solar.sum(axis=1) / model.capacities
    solar_phase = 100 * s_solar / (model.capacities[:, None] / 3.0)

    sensitivity_df = pd.DataFrame(
        {
            "Condition": [
                "Base report solution",
                f"Peak demand +{peak_growth:.1f}%",
                f"Solar output +{solar_growth:.0f}%",
            ],
            "Max DT loading %": [
                float(np.max(base_ev["load_pct"][0, 0])),
                float(np.max(peak_load)),
                float(np.max(solar_load)),
            ],
            "Max phase loading %": [
                float(np.max(base_ev["phase_load_pct"][0, 0])),
                float(np.max(peak_phase)),
                float(np.max(solar_phase)),
            ],
        }
    )
    st.dataframe(sensitivity_df.style.format("{:.2f}", subset=sensitivity_df.select_dtypes(include="number").columns), use_container_width=True)
    st.bar_chart(sensitivity_df.set_index("Condition"), height=350)

    if np.max(peak_phase) > 100:
        st.warning(
            f"At +{peak_growth:.1f}% Peak Import growth, the simplified phase-capacity "
            f"proxy reaches {np.max(peak_phase):.2f}% and violates the 100% model limit."
        )
    else:
        st.success("The selected Peak Import growth remains within the simplified phase-capacity limit.")

st.divider()
st.caption(
    "Important: This is a decision-support simulator for the synthetic EM618 study. "
    "A feasible mathematical consumer transfer is not a field authorization; actual implementation "
    "requires conductor, voltage-drop, protection, route and network-connectivity verification."
)
