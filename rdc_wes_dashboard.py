import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# Near-Solution Style WES Digital Twin Dashboard (6x12x4 = 576 Cells, 300 SKUs)
# Based on rdc_simulation_v7.py
# ==============================================================================

st.set_page_config(
    page_title="RDC Digital Twin WES Monitor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark-mode WES Control Center CSS
st.markdown("""
<style>
    .stApp { background-color: #0B0F19; color: #E2E8F0; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1E293B; }
    div[data-testid="stMetric"] {
        background-color: #131B2E;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
    }
    div[data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 0.85rem; }
    div[data-testid="stMetricValue"] { color: #38BDF8 !important; font-weight: 700; }
    .wes-header {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        border-left: 4px solid #06B6D4;
        padding: 14px 20px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    .wes-badge {
        display: inline-block;
        background: rgba(6, 182, 212, 0.15);
        color: #22D3EE;
        border: 1px solid #0891B2;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Constants from rdc_simulation_v7.py
NUM_AISLES = 6
BAYS_PER_SIDE = 12
NUM_TIERS = 4
AISLE_PITCH = 4.0
BAY_WIDTH = 1.2
WALK_SPEED = 45.7
PICK_TIME_GZ = 0.3
PICK_TIME_NON_GZ = 0.4

TOTAL_SKUS = 300
TOP_SKU_RATIO = 0.20
CELLS_PER_TOP_SKU = 3
CELLS_PER_OTHER_SKU = 1
WEIGHT_LIMIT_KG = 5.0

ORDER_BATCH_SIZE = 10
ORDERS_PER_WEEK = 300
EWMA_ALPHA = 0.50
SPC_SIGMA_MULTIPLIER = 3.0
SWAP_COST_MIN = 2.0
MAX_SWAPS_PER_WEEK = 2
NUM_WEEKS = 12
SHOCK_WEEK_1 = 5
SHOCK_WEEK_2 = 9


@dataclass
class Cell:
    aisle: int
    side: str
    bay: int
    tier: int
    sku_id: Optional[str] = None

    @property
    def is_golden_zone(self) -> bool:
        return self.tier in (2, 3)

    @property
    def x_coord(self) -> float:
        offset = -0.7 if self.side == 'L' else 0.7
        return (self.aisle - 1) * AISLE_PITCH + offset

    @property
    def y_coord(self) -> float:
        return (self.bay - 0.5) * BAY_WIDTH

    @property
    def ergo_pick_time_min(self) -> float:
        return PICK_TIME_GZ if self.is_golden_zone else PICK_TIME_NON_GZ


@dataclass
class SKU:
    sku_id: str
    category: str
    weight_kg: float
    is_esd: bool
    base_demand_weight: float
    allocated_cells: int = 1
    current_cells: List[Cell] = field(default_factory=list)


class VirtualWarehouse:
    def __init__(self):
        self.cells: List[Cell] = []
        for a in range(1, NUM_AISLES + 1):
            for s in ['L', 'R']:
                for b in range(1, BAYS_PER_SIDE + 1):
                    for t in range(1, NUM_TIERS + 1):
                        self.cells.append(Cell(aisle=a, side=s, bay=b, tier=t))

    def clear_all_slots(self):
        for c in self.cells:
            c.sku_id = None


def generate_hs8473_skus(seed: int) -> List[SKU]:
    categories = ["Mainboard", "GPU_Module", "RAM_DDR5", "SSD_NVMe", "Cooling_Unit", "Server_Connector"]
    top_n = int(TOTAL_SKUS * TOP_SKU_RATIO)
    other_n = TOTAL_SKUS - top_n
    top_weights = np.exp(-np.linspace(0, 1.5, top_n))
    top_weights = (top_weights / top_weights.sum()) * 0.80
    other_weights = np.exp(-np.linspace(0, 2.0, other_n))
    other_weights = (other_weights / other_weights.sum()) * 0.20
    all_weights = np.concatenate([top_weights, other_weights])
    rng = np.random.RandomState(seed)
    skus = []
    for i in range(TOTAL_SKUS):
        cat = categories[i % len(categories)]
        weight = round(rng.uniform(5.2, 8.5), 1) if rng.random() < 0.05 else round(rng.uniform(0.2, 3.5), 1)
        is_esd = True if (cat in ["Mainboard", "RAM_DDR5", "SSD_NVMe"] and rng.random() < 0.3) else False
        alloc = CELLS_PER_TOP_SKU if i < top_n else CELLS_PER_OTHER_SKU
        skus.append(SKU(
            sku_id=f"SKU_{i+1:03d}",
            category=cat,
            weight_kg=weight,
            is_esd=is_esd,
            base_demand_weight=all_weights[i],
            allocated_cells=alloc
        ))
    return skus


def get_gc2_sorted_cells(wh: VirtualWarehouse, near_aisles: Tuple[int, ...] = (1,)) -> List[Cell]:
    near_gz = sorted([c for c in wh.cells if c.aisle in near_aisles and c.is_golden_zone], key=lambda c: (c.bay, c.aisle, c.tier))
    far_gzw: List[Cell] = []
    for a in range(1, NUM_AISLES + 1):
        if a in near_aisles:
            continue
        gz_in_aisle = sorted([c for c in wh.cells if c.aisle == a and c.is_golden_zone], key=lambda c: (c.bay, c.tier))
        non_gz_in_aisle = sorted([c for c in wh.cells if c.aisle == a and not c.is_golden_zone], key=lambda c: (c.bay, c.tier))
        far_gzw.extend(gz_in_aisle + non_gz_in_aisle)
    near_non_gz = sorted([c for c in wh.cells if c.aisle in near_aisles and not c.is_golden_zone], key=lambda c: (c.bay, c.aisle, c.tier))
    return near_gz + far_gzw + near_non_gz


def assign_gc2_turnover_slotting(wh: VirtualWarehouse, skus: List[SKU], observed_counts: Dict[str, float], near_aisles: Tuple[int, ...] = (1,)):
    wh.clear_all_slots()
    for s in skus:
        s.current_cells = []
    sorted_skus = sorted(skus, key=lambda s: observed_counts.get(s.sku_id, s.base_demand_weight), reverse=True)
    gc2_cells = get_gc2_sorted_cells(wh, near_aisles)
    tier1_cells = sorted([c for c in gc2_cells if c.tier == 1], key=lambda c: (c.aisle, c.bay))
    remaining_skus = []
    for sku in sorted_skus:
        if sku.weight_kg > WEIGHT_LIMIT_KG:
            assigned = 0
            for c in tier1_cells:
                if c.sku_id is None:
                    c.sku_id = sku.sku_id
                    sku.current_cells.append(c)
                    assigned += 1
                    if assigned == sku.allocated_cells:
                        break
            if assigned < sku.allocated_cells:
                remaining_skus.append(sku)
        else:
            remaining_skus.append(sku)

    free_cells = [c for c in gc2_cells if c.sku_id is None]
    for sku in remaining_skus:
        for _ in range(sku.allocated_cells):
            if not free_cells:
                break
            chosen = free_cells[0]
            if sku.is_esd and chosen.tier == 3:
                same_loc_t2 = [c for c in free_cells if c.aisle == chosen.aisle and c.bay == chosen.bay and c.side == chosen.side and c.tier == 2]
                if same_loc_t2:
                    chosen = same_loc_t2[0]
            free_cells.remove(chosen)
            chosen.sku_id = sku.sku_id
            sku.current_cells.append(chosen)


def calculate_order_time_and_distance(order_skus: List[SKU]) -> Tuple[float, float, int, Dict[int, float]]:
    aisle_max_y: Dict[int, float] = {}
    pick_time_total = 0.0
    gz_pick_count = 0
    for sku in order_skus:
        if not sku.current_cells:
            continue
        best_cell = min(sku.current_cells, key=lambda c: (c.aisle, c.bay))
        a, y = best_cell.aisle, best_cell.y_coord
        if a not in aisle_max_y or y > aisle_max_y[a]:
            aisle_max_y[a] = y
        if best_cell.is_golden_zone:
            pick_time_total += PICK_TIME_GZ
            gz_pick_count += 1
        else:
            pick_time_total += PICK_TIME_NON_GZ
    if not aisle_max_y:
        return 0.0, 0.0, 0, {}
    max_aisle = max(aisle_max_y.keys())
    total_dist = 2.0 * ((max_aisle - 1) * AISLE_PITCH) + sum(2.0 * y for y in aisle_max_y.values())
    return total_dist, (total_dist / WALK_SPEED) + pick_time_total, gz_pick_count, aisle_max_y


def cell_pooled_expected_cost_min(cell: Cell, aisle_expected_max_y: Dict[int, float], expected_max_aisle: float) -> float:
    ergo_min = cell.ergo_pick_time_min
    extra_cross_dist = 2.0 * max(0.0, (cell.aisle - expected_max_aisle) * AISLE_PITCH)
    cur_aisle_max = aisle_expected_max_y.get(cell.aisle, 0.0)
    extra_in_aisle_dist = 2.0 * max(0.0, cell.y_coord - cur_aisle_max)
    return ergo_min + (1.0 / ORDER_BATCH_SIZE) * ((extra_cross_dist + extra_in_aisle_dist) / WALK_SPEED)


def estimate_routing_pool_profile(skus: List[SKU], ewma_demand: Dict[str, float]) -> Tuple[Dict[int, float], float]:
    aisle_items: Dict[int, List[Tuple[float, float]]] = {a: [] for a in range(1, NUM_AISLES + 1)}
    total_picks_per_week = ORDERS_PER_WEEK * ORDER_BATCH_SIZE
    for s in skus:
        if not s.current_cells:
            continue
        best_c = min(s.current_cells, key=lambda c: (c.aisle, c.bay))
        p_line = min(0.99, ewma_demand.get(s.sku_id, 0.0) / max(total_picks_per_week, 1.0))
        p_order = 1.0 - ((1.0 - p_line) ** ORDER_BATCH_SIZE)
        aisle_items[best_c.aisle].append((best_c.y_coord, p_order))
    aisle_exp_max_y: Dict[int, float] = {}
    aisle_any_prob: Dict[int, float] = {}
    for a in range(1, NUM_AISLES + 1):
        items = sorted(aisle_items[a], key=lambda x: x[0], reverse=True)
        prob_none_above, exp_y = 1.0, 0.0
        for y, p_ord in items:
            exp_y += y * prob_none_above * p_ord
            prob_none_above *= (1.0 - p_ord)
        aisle_exp_max_y[a] = exp_y
        aisle_any_prob[a] = 1.0 - prob_none_above
    prob_none_right, exp_max_a = 1.0, 1.0
    for a in range(NUM_AISLES, 0, -1):
        exp_max_a += (a - 1) * prob_none_right * aisle_any_prob[a]
        prob_none_right *= (1.0 - aisle_any_prob[a])
    return aisle_exp_max_y, exp_max_a


@st.cache_data(show_spinner="3D 디지털 트윈 시뮬레이션 트레이스 생성 중...")
def simulate_digital_twin_trace(seed: int = 42, k_horizon: int = 2):
    rng = np.random.RandomState(seed)
    skus_master = generate_hs8473_skus(seed=seed)
    base_weights = np.array([s.base_demand_weight for s in skus_master])
    base_weights /= base_weights.sum()

    initial_counts = {s.sku_id: 0.0 for s in skus_master}
    for _ in range(ORDERS_PER_WEEK * 4):
        chosen_idxs = rng.choice(TOTAL_SKUS, size=ORDER_BATCH_SIZE, replace=False, p=base_weights)
        for idx in chosen_idxs:
            initial_counts[skus_master[idx].sku_id] += 1.0

    wh_static = VirtualWarehouse()
    skus_static = generate_hs8473_skus(seed=seed)
    assign_gc2_turnover_slotting(wh_static, skus_static, initial_counts, near_aisles=(1,))

    wh_dynamic = VirtualWarehouse()
    skus_dynamic = generate_hs8473_skus(seed=seed)
    assign_gc2_turnover_slotting(wh_dynamic, skus_dynamic, initial_counts, near_aisles=(1,))

    ewma_demand = {s.sku_id: initial_counts[s.sku_id] / 4.0 for s in skus_dynamic}
    ewma_var = {s.sku_id: max(initial_counts[s.sku_id] / 4.0, 1.0) for s in skus_dynamic}
    skus_by_id = {s.sku_id: s for s in skus_dynamic}

    current_weights = base_weights.copy()
    moved_skus_set = set()
    weekly_snapshots = {}
    kpi_history = []

    for week in range(1, NUM_WEEKS + 1):
        if week == SHOCK_WEEK_1:
            current_weights[:20] *= 0.15
            current_weights[120:140] *= 8.0
            current_weights /= current_weights.sum()
        if week == SHOCK_WEEK_2:
            current_weights[20:30] *= 0.4
            current_weights[180:190] *= 3.5
            current_weights /= current_weights.sum()

        week_order_indices = [
            rng.choice(TOTAL_SKUS, size=ORDER_BATCH_SIZE, replace=False, p=current_weights)
            for _ in range(ORDERS_PER_WEEK)
        ]

        tot_picks = ORDERS_PER_WEEK * ORDER_BATCH_SIZE
        # Evaluate Static-GC2
        st_dist = st_time = 0.0
        st_gz = 0
        for idxs in week_order_indices:
            d, t, gz, _ = calculate_order_time_and_distance([skus_static[i] for i in idxs])
            st_dist += d
            st_time += t
            st_gz += gz

        # Evaluate Dynamic Quick-Fix
        dy_dist = dy_time = 0.0
        dy_gz = 0
        sample_route_aisles = {}
        aisle_pick_load = {a: 0 for a in range(1, NUM_AISLES + 1)}
        for o_i, idxs in enumerate(week_order_indices):
            o_skus = [skus_dynamic[i] for i in idxs]
            d, t, gz, a_max = calculate_order_time_and_distance(o_skus)
            dy_dist += d
            dy_time += t
            dy_gz += gz
            if o_i == 0:
                sample_route_aisles = a_max
            for s_obj in o_skus:
                if s_obj.current_cells:
                    bc = min(s_obj.current_cells, key=lambda c: (c.aisle, c.bay))
                    aisle_pick_load[bc.aisle] += 1

        kpi_history.append({
            "Week": week,
            "Static_GC2_Picks_Hr": (tot_picks / st_time) * 60.0,
            "Dynamic_Picks_Hr": (tot_picks / dy_time) * 60.0,
            "Dynamic_Order_Min": dy_time / ORDERS_PER_WEEK,
            "Dynamic_GZ_Pct": (dy_gz / tot_picks) * 100.0,
            "Dynamic_Dist_m": dy_dist / ORDERS_PER_WEEK,
            "Cumulative_Churn": len(moved_skus_set)
        })

        # Update EWMA & SPC
        week_counts = {s.sku_id: 0.0 for s in skus_dynamic}
        for idxs in week_order_indices:
            for idx in idxs:
                week_counts[skus_dynamic[idx].sku_id] += 1.0

        triggered_skus = []
        for s in skus_dynamic:
            sid = s.sku_id
            obs = week_counts[sid]
            prev_mean = ewma_demand[sid]
            ucl = prev_mean + SPC_SIGMA_MULTIPLIER * np.sqrt(ewma_var[sid])
            if obs > ucl:
                triggered_skus.append(sid)
            diff = obs - prev_mean
            ewma_demand[sid] = EWMA_ALPHA * obs + (1.0 - EWMA_ALPHA) * prev_mean
            ewma_var[sid] = (1.0 - EWMA_ALPHA) * ewma_var[sid] + EWMA_ALPHA * (diff ** 2)

        week_swaps = []
        if triggered_skus and week < NUM_WEEKS:
            aisle_max_y, expected_max_aisle = estimate_routing_pool_profile(skus_dynamic, ewma_demand)
            hot_candidates = sorted(triggered_skus, key=lambda sid: -ewma_demand.get(sid, 0.0))
            all_skus_by_coldness = sorted(skus_dynamic, key=lambda s: ewma_demand.get(s.sku_id, 0.0))
            swaps_this_week = 0
            used_skus = set()

            for hot_id in hot_candidates:
                if swaps_this_week >= MAX_SWAPS_PER_WEEK:
                    break
                if hot_id in used_skus:
                    continue
                hot = skus_by_id.get(hot_id)
                if hot is None or hot.weight_kg > WEIGHT_LIMIT_KG or not hot.current_cells:
                    continue
                hot_active_cell = min(hot.current_cells, key=lambda c: (c.aisle, c.bay))
                hot_cell_cost = cell_pooled_expected_cost_min(hot_active_cell, aisle_max_y, expected_max_aisle)

                best_partner, best_cold_cell, best_net_gain = None, None, 0.0
                for cold in all_skus_by_coldness:
                    if cold.sku_id == hot.sku_id or cold.sku_id in used_skus or cold.weight_kg > WEIGHT_LIMIT_KG:
                        continue
                    if ewma_demand.get(cold.sku_id, 0.0) >= ewma_demand.get(hot.sku_id, 0.0):
                        continue
                    cold_best_cell = min(cold.current_cells, key=lambda c: cell_pooled_expected_cost_min(c, aisle_max_y, expected_max_aisle))
                    hot_post_cells = [c for c in hot.current_cells if c is not hot_active_cell] + [cold_best_cell]
                    cold_post_cells = [c for c in cold.current_cells if c is not cold_best_cell] + [hot_active_cell]
                    hot_post_cost = cell_pooled_expected_cost_min(min(hot_post_cells, key=lambda c: (c.aisle, c.bay)), aisle_max_y, expected_max_aisle)
                    cold_pre_cost = cell_pooled_expected_cost_min(min(cold.current_cells, key=lambda c: (c.aisle, c.bay)), aisle_max_y, expected_max_aisle)
                    cold_post_cost = cell_pooled_expected_cost_min(min(cold_post_cells, key=lambda c: (c.aisle, c.bay)), aisle_max_y, expected_max_aisle)

                    weekly_saving = ewma_demand[hot.sku_id] * (hot_cell_cost - hot_post_cost) + ewma_demand[cold.sku_id] * (cold_pre_cost - cold_post_cost)
                    net_gain = (k_horizon * weekly_saving) - (2.0 * SWAP_COST_MIN)
                    if net_gain > best_net_gain:
                        best_net_gain, best_partner, best_cold_cell = net_gain, cold, cold_best_cell

                if best_partner is not None and best_cold_cell is not None:
                    week_swaps.append({
                        "hot_sku": hot.sku_id,
                        "cold_sku": best_partner.sku_id,
                        "from_loc": f"A{hot_active_cell.aisle}-{hot_active_cell.side}-B{hot_active_cell.bay}-T{hot_active_cell.tier}",
                        "to_loc": f"A{best_cold_cell.aisle}-{best_cold_cell.side}-B{best_cold_cell.bay}-T{best_cold_cell.tier}",
                        "net_gain_min": round(best_net_gain, 2)
                    })
                    hot_active_cell.sku_id = best_partner.sku_id
                    best_cold_cell.sku_id = hot.sku_id
                    hot.current_cells.remove(hot_active_cell)
                    hot.current_cells.append(best_cold_cell)
                    best_partner.current_cells.remove(best_cold_cell)
                    best_partner.current_cells.append(hot_active_cell)
                    used_skus.update([hot.sku_id, best_partner.sku_id])
                    moved_skus_set.update([hot.sku_id, best_partner.sku_id])
                    swaps_this_week += 1

        # Save cell grid snapshot
        cell_records = []
        for c in wh_dynamic.cells:
            demand_val = ewma_demand.get(c.sku_id, 0.0) if c.sku_id else 0.0
            cat_val = skus_by_id[c.sku_id].category if c.sku_id else "Buffer (Empty)"
            is_moved = (c.sku_id in moved_skus_set) if c.sku_id else False
            cell_records.append({
                "x": c.x_coord,
                "y": c.y_coord,
                "z": c.tier,
                "aisle": c.aisle,
                "side": c.side,
                "bay": c.bay,
                "tier": c.tier,
                "sku_id": c.sku_id or "EMPTY",
                "category": cat_val,
                "demand": demand_val,
                "is_gz": c.is_golden_zone,
                "is_moved": is_moved
            })

        weekly_snapshots[week] = {
            "cells": pd.DataFrame(cell_records),
            "swaps": week_swaps,
            "sample_route": sample_route_aisles,
            "aisle_load": aisle_pick_load
        }

    return weekly_snapshots, pd.DataFrame(kpi_history)


# ==============================================================================
# Sidebar Controls
# ==============================================================================
with st.sidebar:
    st.markdown("### 🎛️ WES 관제 파라미터")
    seed_val = st.selectbox("시뮬레이션 시드 (Seed)", [42, 43, 44, 45, 46], index=0)
    k_val = st.radio("투자회수 기준 (Horizon k주)", [1, 2, 3], index=1, horizontal=True)
    selected_week = st.slider("📅 타임라인 주차 (Week)", min_value=1, max_value=12, value=6, step=1)

    st.markdown("---")
    st.markdown("### 👁️ 디지털 트윈 레이어")
    show_empty = st.checkbox("여유 버퍼 슬롯 표시 (156셀)", value=False)
    highlight_swaps = st.checkbox("Quick-Fix 재배치 품목 강조", value=True)
    show_route = st.checkbox("Return 라우팅 동선 오버레이", value=True)

    st.markdown("---")
    if selected_week == 5:
        st.warning("⚡ **5주차 1차 수요 충격 발생!**\nC등급 20개 품목 주문량 8배 폭증")
    elif selected_week == 9:
        st.warning("⚡ **9주차 2차 계절 변동 발생!**\n신규 C등급 10개 품목 주문량 3.5배 급등")
    else:
        st.info("✅ **정상 운영 모니터링 중**\nShewhart 3-Sigma SPC 감시 가동")

# Run / Load Cached Simulation
snapshots, kpi_df = simulate_digital_twin_trace(seed=seed_val, k_horizon=k_val)
cur_snap = snapshots[selected_week]
cur_kpi = kpi_df[kpi_df["Week"] == selected_week].iloc[0]
prev_kpi = kpi_df[kpi_df["Week"] == max(1, selected_week - 1)].iloc[0]

# Top Header
st.markdown(f"""
<div class="wes-header">
    <span class="wes-badge">WES DIGITAL TWIN</span>
    <span class="wes-badge">6 AISLES × 12 BAYS × 4 TIERS (576 CELLS)</span>
    <h2 style="margin:6px 0 2px 0; color:#F8FAFC;">유럽 권역 RDC 동적 슬로팅(Dynamic Quick-Fix) 실시간 관제 대시보드</h2>
    <p style="margin:0; color:#94A3B8; font-size:0.9rem;">Shewhart 3σ 수요 급등 감지 및 Return 라우팅 풀링 기반 최소 비용 재배치 모니터링</p>
</div>
""", unsafe_allow_html=True)

# Top KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "시간당 피킹 처리량",
    f"{cur_kpi['Dynamic_Picks_Hr']:.2f} Picks/hr",
    f"{cur_kpi['Dynamic_Picks_Hr'] - cur_kpi['Static_GC2_Picks_Hr']:+.2f} vs Static-GC2"
)
c2.metric(
    "평균 주문 처리시간",
    f"{cur_kpi['Dynamic_Order_Min']:.3f} 분",
    f"{cur_kpi['Dynamic_Order_Min'] - prev_kpi['Dynamic_Order_Min']:+.3f} 분 (전주 대비)",
    delta_color="inverse"
)
c3.metric(
    "골든존(2·3단) 피킹 비율",
    f"{cur_kpi['Dynamic_GZ_Pct']:.1f} %",
    f"{cur_kpi['Dynamic_GZ_Pct'] - prev_kpi['Dynamic_GZ_Pct']:+.1f} %p"
)
c4.metric(
    "주문당 평균 이동거리",
    f"{cur_kpi['Dynamic_Dist_m']:.1f} m",
    f"{cur_kpi['Dynamic_Dist_m'] - prev_kpi['Dynamic_Dist_m']:+.1f} m",
    delta_color="inverse"
)
c5.metric(
    "누적 슬롯 재배치 (Churn)",
    f"{int(cur_kpi['Cumulative_Churn'])} / 300 SKU",
    f"전체 재고의 {cur_kpi['Cumulative_Churn']/300*100:.1f}%"
)

# Main Layout: Left 3D Digital Twin (7 cols) + Right Bottleneck & Trend (5 cols)
left_col, right_col = st.columns([7, 5])

with left_col:
    st.subheader(f"🏢 3D 창고 랙 히트맵 & 피킹 동선 (Week {selected_week})")
    df_cells = cur_snap["cells"]
    df_occ = df_cells[df_cells["sku_id"] != "EMPTY"].copy()
    df_emp = df_cells[df_cells["sku_id"] == "EMPTY"].copy()

    fig_3d = go.Figure()

    # Empty Buffer Cells
    if show_empty and not df_emp.empty:
        fig_3d.add_trace(go.Scatter3d(
            x=df_emp["x"], y=df_emp["y"], z=df_emp["z"],
            mode="markers",
            name="여유 버퍼 슬롯 (156셀)",
            marker=dict(size=4, color="rgba(148, 163, 184, 0.18)", symbol="square"),
            hoverinfo="skip"
        ))

    # Occupied SKU Cells Heatmap
    fig_3d.add_trace(go.Scatter3d(
        x=df_occ["x"], y=df_occ["y"], z=df_occ["z"],
        mode="markers",
        name="적재 셀 (EWMA 출고빈도)",
        marker=dict(
            size=6.5,
            color=df_occ["demand"],
            colorscale="Turbo",
            cmin=0,
            cmax=max(45.0, df_occ["demand"].max()),
            colorbar=dict(title="주간 출고량", x=0.02, len=0.65, tickfont=dict(color="#CBD5E1")),
            symbol="square",
            opacity=0.88
        ),
        text=[
            f"<b>{r.sku_id}</b> ({r.category})<br>"
            f"위치: {r.aisle}통로 {r.side}열 {r.bay}베이 {r.tier}단 {'[골든존]' if r.is_gz else ''}<br>"
            f"주간 EWMA 수요: {r.demand:.1f}건<br>"
            f"상태: {'🔄 Quick-Fix 재배치됨' if r.is_moved else '고정 배치'}"
            for r in df_occ.itertuples()
        ],
        hoverinfo="text"
    ))

    # Highlight Quick-Fix Swapped Cells
    if highlight_swaps:
        df_moved = df_occ[df_occ["is_moved"]]
        if not df_moved.empty:
            fig_3d.add_trace(go.Scatter3d(
                x=df_moved["x"], y=df_moved["y"], z=df_moved["z"],
                mode="markers",
                name="Quick-Fix 재배치 셀",
                marker=dict(
                    size=10,
                    color="rgba(0,0,0,0)",
                    line=dict(color="#22D3EE", width=4),
                    symbol="diamond"
                ),
                hoverinfo="skip"
            ))

    # Return Routing Path Overlay (Sample Order)
    if show_route and cur_snap["sample_route"]:
        r_aisles = cur_snap["sample_route"]
        rx, ry, rz = [0.0], [0.0], [0.5]
        for a in sorted(r_aisles.keys()):
            ax = (a - 1) * AISLE_PITCH
            max_y = r_aisles[a]
            rx.extend([ax, ax, ax])
            ry.extend([0.0, max_y, 0.0])
            rz.extend([0.5, 0.5, 0.5])
        rx.append(0.0)
        ry.append(0.0)
        rz.append(0.5)

        fig_3d.add_trace(go.Scatter3d(
            x=rx, y=ry, z=rz,
            mode="lines+markers",
            name="Return 피킹 동선",
            line=dict(color="#38BDF8", width=6, dash="dash"),
            marker=dict(size=4, color="#38BDF8")
        ))

    fig_3d.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=dict(l=0, r=0, t=10, b=0),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(color="#E2E8F0")),
        scene=dict(
            bgcolor="#0B0F19",
            xaxis=dict(title="통로 (Aisle 1~6)", tickvals=[i*AISLE_PITCH for i in range(6)], ticktext=[f"A{i+1}" for i in range(6)], gridcolor="#1E293B", color="#94A3B8"),
            yaxis=dict(title="베이 깊이 (m)", gridcolor="#1E293B", color="#94A3B8"),
            zaxis=dict(title="단 (Tier 1~4)", tickvals=[1, 2, 3, 4], ticktext=["1단", "2단(GZ)", "3단(GZ)", "4단"], gridcolor="#1E293B", color="#94A3B8"),
            camera=dict(eye=dict(x=1.55, y=-1.65, z=1.15))
        )
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with right_col:
    st.subheader("📊 통로별 피킹 부하 & 처리량 추이")

    # 1. Aisle Bottleneck Gauge Bar Chart
    aisle_load = cur_snap["aisle_load"]
    fig_bar = go.Figure(go.Bar(
        x=[f"Aisle {a}" for a in range(1, NUM_AISLES + 1)],
        y=[aisle_load[a] for a in range(1, NUM_AISLES + 1)],
        marker=dict(
            color=[aisle_load[a] for a in range(1, NUM_AISLES + 1)],
            colorscale=[[0, "#0284C7"], [0.5, "#06B6D4"], [1, "#F59E0B"]]
        ),
        text=[f"{aisle_load[a]}건" for a in range(1, NUM_AISLES + 1)],
        textposition="outside",
        textfont=dict(color="#E2E8F0")
    ))
    fig_bar.update_layout(
        title=dict(text=f"Week {selected_week} 통로별 피킹 작업 집중도", font=dict(size=14, color="#E2E8F0")),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=dict(l=10, r=10, t=35, b=10),
        height=230,
        xaxis=dict(color="#94A3B8", showgrid=False),
        yaxis=dict(color="#94A3B8", gridcolor="#1E293B", range=[0, max(aisle_load.values()) * 1.25])
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # 2. Weekly Throughput Comparison Line Chart
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=kpi_df["Week"], y=kpi_df["Dynamic_Picks_Hr"],
        mode="lines+markers", name="Dynamic Quick-Fix (제안)",
        line=dict(color="#22D3EE", width=3), marker=dict(size=7)
    ))
    fig_line.add_trace(go.Scatter(
        x=kpi_df["Week"], y=kpi_df["Static_GC2_Picks_Hr"],
        mode="lines+markers", name="Static-GC2 (정적 골든존)",
        line=dict(color="#94A3B8", width=2, dash="dot"), marker=dict(size=5)
    ))
    fig_line.add_vline(x=selected_week, line_width=2, line_dash="dash", line_color="#F59E0B")
    fig_line.update_layout(
        title=dict(text="주차별 시간당 피킹 처리량 비교 (Picks/hr)", font=dict(size=14, color="#E2E8F0")),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=dict(l=10, r=10, t=35, b=10),
        height=265,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color="#CBD5E1")),
        xaxis=dict(title="주차 (Week)", tickmode="linear", dtick=1, color="#94A3B8", gridcolor="#1E293B"),
        yaxis=dict(title="Picks/hr", color="#94A3B8", gridcolor="#1E293B")
    )
    st.plotly_chart(fig_line, use_container_width=True)

# Bottom Panel: Swap Event Log Table
st.subheader(f"🔄 Week {selected_week} Shewhart 3σ SPC 감지 및 슬롯 재배치(Quick-Fix) 실행 로그")
if cur_snap["swaps"]:
    swap_df = pd.DataFrame(cur_snap["swaps"])
    swap_df.columns = ["급등 품목 (Hot SKU)", "교체 저회전 품목 (Cold SKU)", "기존 위치", "신규 골든존 위치", "예상 순이익 (분)"]
    st.dataframe(swap_df, use_container_width=True, hide_index=True)
else:
    st.caption("해당 주차에는 손익분기점(net_gain > 0)을 초과하는 신규 재배치 이벤트가 발생하지 않았습니다.")
