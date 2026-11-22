import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# Near-Solution Style 3D Warehouse Digital Twin WES (Three.js + Streamlit)
# Real 3D Steel Racks, Pallets/Boxes, I/O Dock, Floor Lanes, & Moving Picker!
# Based on rdc_simulation_v7.py (6 Aisles x 12 Bays x 4 Tiers = 576 Cells, 300 SKUs)
# ==============================================================================

st.set_page_config(
    page_title="RDC 3D Digital Twin WES",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #070B14; color: #E2E8F0; }
    [data-testid="stSidebar"] { background-color: #0F172A; border-right: 1px solid #1E293B; }
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 0.82rem; }
    div[data-testid="stMetricValue"] { color: #38BDF8 !important; font-weight: 700; font-size: 1.45rem; }
    .wes-banner {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        border-left: 4px solid #06B6D4;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        background: rgba(6, 182, 212, 0.15);
        color: #22D3EE;
        border: 1px solid #0891B2;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
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


def calculate_order_time_and_distance(order_skus: List[SKU]) -> Tuple[float, float, int, List[dict]]:
    aisle_max_y: Dict[int, float] = {}
    pick_time_total = 0.0
    gz_pick_count = 0
    picked_targets = []
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
        picked_targets.append({
            "aisle": best_cell.aisle,
            "side": best_cell.side,
            "bay": best_cell.bay,
            "tier": best_cell.tier,
            "sku": sku.sku_id
        })
    if not aisle_max_y:
        return 0.0, 0.0, 0, []
    max_aisle = max(aisle_max_y.keys())
    total_dist = 2.0 * ((max_aisle - 1) * AISLE_PITCH) + sum(2.0 * y for y in aisle_max_y.values())
    return total_dist, (total_dist / WALK_SPEED) + pick_time_total, gz_pick_count, picked_targets


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


@st.cache_data(show_spinner="실제 3D 물류센터 디지털 트윈 데이터 생성 중...")
def run_3d_simulation_trace(seed: int = 42, k_horizon: int = 2):
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
    weekly_data = {}
    kpi_list = []

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
        st_time = 0.0
        for idxs in week_order_indices:
            _, t, _, _ = calculate_order_time_and_distance([skus_static[i] for i in idxs])
            st_time += t

        dy_dist = dy_time = 0.0
        dy_gz = 0
        sample_targets = []
        for o_i, idxs in enumerate(week_order_indices):
            o_skus = [skus_dynamic[i] for i in idxs]
            d, t, gz, targets = calculate_order_time_and_distance(o_skus)
            dy_dist += d
            dy_time += t
            dy_gz += gz
            if o_i == 0:
                sample_targets = sorted(targets, key=lambda x: (x["aisle"], x["bay"]))

        kpi_list.append({
            "Week": week,
            "Static_GC2_Picks_Hr": (tot_picks / st_time) * 60.0,
            "Dynamic_Picks_Hr": (tot_picks / dy_time) * 60.0,
            "Dynamic_Order_Min": dy_time / ORDERS_PER_WEEK,
            "Dynamic_GZ_Pct": (dy_gz / tot_picks) * 100.0,
            "Dynamic_Dist_m": dy_dist / ORDERS_PER_WEEK,
            "Cumulative_Churn": len(moved_skus_set)
        })

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

        cells_json = []
        for c in wh_dynamic.cells:
            if c.sku_id is None:
                continue
            s_obj = skus_by_id[c.sku_id]
            cells_json.append({
                "a": c.aisle,
                "s": c.side,
                "b": c.bay,
                "t": c.tier,
                "sku": c.sku_id,
                "cat": s_obj.category,
                "dem": round(ewma_demand.get(c.sku_id, 0.0), 1),
                "gz": c.is_golden_zone,
                "moved": (c.sku_id in moved_skus_set)
            })

        weekly_data[week] = {
            "cells": cells_json,
            "swaps": week_swaps,
            "targets": sample_targets
        }

    return weekly_data, pd.DataFrame(kpi_list)


def render_threejs_warehouse(week_payload: dict, week_num: int, picker_speed: float):
    payload_str = json.dumps(week_payload)
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      body {{ margin: 0; overflow: hidden; background: #070B14; font-family: 'Segoe UI', sans-serif; }}
      #hud {{
        position: absolute; top: 14px; left: 14px; z-index: 10;
        background: rgba(15, 23, 42, 0.85); border: 1px solid #1E293B;
        border-radius: 10px; padding: 12px 16px; color: #E2E8F0;
        backdrop-filter: blur(6px); pointer-events: none;
      }}
      #hud h4 {{ margin: 0 0 6px 0; color: #38BDF8; font-size: 14px; }}
      .legend-row {{ display: flex; align-items: center; font-size: 12px; margin-top: 4px; color: #CBD5E1; }}
      .dot {{ width: 11px; height: 11px; border-radius: 2px; margin-right: 8px; display: inline-block; }}
      #tooltip {{
        position: absolute; display: none; z-index: 20;
        background: rgba(15, 23, 42, 0.94); border: 1px solid #38BDF8;
        color: #F8FAFC; padding: 8px 12px; border-radius: 6px;
        font-size: 12px; pointer-events: none; box-shadow: 0 4px 12px rgba(0,0,0,0.5);
      }}
      #controls-hint {{
        position: absolute; bottom: 12px; right: 14px; z-index: 10;
        background: rgba(15, 23, 42, 0.78); color: #94A3B8;
        padding: 6px 12px; border-radius: 6px; font-size: 11px; border: 1px solid #1E293B;
      }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
      <div id="hud">
        <h4>🏭 RDC 3D DIGITAL TWIN — WEEK {week_num}</h4>
        <div class="legend-row"><span class="dot" style="background:#EF4444;"></span>초고회전 핫존 SKU (수요 급등)</div>
        <div class="legend-row"><span class="dot" style="background:#F59E0B;"></span>중·고회전 SKU (주력 출고)</div>
        <div class="legend-row"><span class="dot" style="background:#3B82F6;"></span>저회전 C등급 SKU</div>
        <div class="legend-row"><span class="dot" style="background:#22D3EE; box-shadow: 0 0 6px #22D3EE;"></span>Dynamic Quick-Fix 재배치 박스</div>
        <div class="legend-row"><span class="dot" style="background:#10B981;"></span>실시간 Return 라우팅 피커(작업자)</div>
      </div>
      <div id="tooltip"></div>
      <div id="controls-hint">🖱️ 좌클릭 드래그: 회전 | 우클릭 드래그: 이동 | 휠: 줌 인/아웃 | 박스 위 마우스 오버: 상세 스펙</div>

      <script>
        const DATA = {payload_str};
        const PICKER_SPEED = {picker_speed};

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x070B14);
        scene.fog = new THREE.FogExp2(0x070B14, 0.011);

        const camera = new THREE.PerspectiveCamera(46, window.innerWidth / window.innerHeight, 0.1, 300);
        camera.position.set(16, 22, 34);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(11, 1.5, 8);
        controls.enableDamping = true;
        controls.dampingFactor = 0.06;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x94A3B8, 0.85);
        scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xFFFFFF, 0.95);
        dirLight.position.set(25, 40, 25);
        dirLight.castShadow = true;
        scene.add(dirLight);

        const cyanLight = new THREE.PointLight(0x06B6D4, 0.7, 60);
        cyanLight.position.set(10, 12, 8);
        scene.add(cyanLight);

        // Warehouse Concrete Floor & Grid
        const floorGeo = new THREE.PlaneGeometry(46, 36);
        const floorMat = new THREE.MeshStandardMaterial({{ color: 0x0F172A, roughness: 0.85, metalness: 0.1 }});
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.position.set(11, 0, 8);
        floor.receiveShadow = true;
        scene.add(floor);

        const gridHelper = new THREE.GridHelper(44, 44, 0x1E293B, 0x162033);
        gridHelper.position.set(11, 0.01, 8);
        scene.add(gridHelper);

        // Coordinate Mapping:
        // World X = (aisle - 1) * 4.5 (Aisles 1..6)
        // World Z = bay * 1.45 (Bays 1..12, front cross-aisle at Z = 0)
        // World Y = tier height (Tier 1..4)
        const AISLE_STEP = 4.6;
        const BAY_STEP = 1.45;
        const TIER_HEIGHT = 1.05;

        // I/O Staging & Packing Dock Zone at Front (Z = -2.2)
        const dockGeo = new THREE.BoxGeometry(26, 0.15, 2.6);
        const dockMat = new THREE.MeshStandardMaterial({{ color: 0x1E293B, emissive: 0x0891B2, emissiveIntensity: 0.25 }});
        const dockMesh = new THREE.Mesh(dockGeo, dockMat);
        dockMesh.position.set(11.5, 0.075, -2.0);
        scene.add(dockMesh);

        // Cross-Aisle Yellow Safety Lines
        const lineMat = new THREE.MeshBasicMaterial({{ color: 0xEAB308 }});
        const crossLine = new THREE.Mesh(new THREE.PlaneGeometry(26, 0.12), lineMat);
        crossLine.rotation.x = -Math.PI / 2;
        crossLine.position.set(11.5, 0.02, -0.3);
        scene.add(crossLine);

        // Build Realistic Steel Pallet Racks (6 Aisles x Left/Right Racks x 12 Bays x 4 Tiers)
        const uprightMat = new THREE.MeshStandardMaterial({{ color: 0x38BDF8, metalness: 0.7, roughness: 0.3 }});
        const beamMat = new THREE.MeshStandardMaterial({{ color: 0xF97316, metalness: 0.5, roughness: 0.4 }});
        const gzBeamMat = new THREE.MeshStandardMaterial({{ color: 0x22D3EE, metalness: 0.6, roughness: 0.25 }});

        const uprightGeo = new THREE.BoxGeometry(0.09, NUM_TIERS * TIER_HEIGHT + 0.3, 0.09);
        const beamGeo = new THREE.BoxGeometry(0.75, 0.07, BAY_STEP * 0.96);

        for (let a = 1; a <= 6; a++) {{
          const aisleCenterX = (a - 1) * AISLE_STEP;
          // Center walkway line
          const walkStrip = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 12 * BAY_STEP), new THREE.MeshBasicMaterial({{ color: 0x131C31 }}));
          walkStrip.rotation.x = -Math.PI / 2;
          walkStrip.position.set(aisleCenterX, 0.015, 6.5 * BAY_STEP);
          scene.add(walkStrip);

          ['L', 'R'].forEach(side => {{
            const rackX = aisleCenterX + (side === 'L' ? -1.15 : 1.15);
            for (let b = 1; b <= 12; b++) {{
              const bayZ = b * BAY_STEP;
              // 4 vertical steel posts per bay
              [-0.35, 0.35].forEach(dx => {{
                [-BAY_STEP * 0.47, BAY_STEP * 0.47].forEach(dz => {{
                  const post = new THREE.Mesh(uprightGeo, uprightMat);
                  post.position.set(rackX + dx, (NUM_TIERS * TIER_HEIGHT + 0.3) / 2, bayZ + dz);
                  scene.add(post);
                }});
              }});
              // Horizontal shelf beams (Tier 2 & 3 highlighted as Golden Zone)
              for (let t = 1; t <= 4; t++) {{
                const shelf = new THREE.Mesh(beamGeo, (t === 2 || t === 3) ? gzBeamMat : beamMat);
                shelf.position.set(rackX, (t - 1) * TIER_HEIGHT + 0.12, bayZ);
                scene.add(shelf);
              }}
            }}
          }});
        }}

        // Populate Cardboard/Tote SKU Boxes inside Rack Cells
        const boxMeshes = [];
        const boxGeo = new THREE.BoxGeometry(0.62, 0.62, 0.95);

        DATA.cells.forEach(cell => {{
          const aisleCenterX = (cell.a - 1) * AISLE_STEP;
          const rackX = aisleCenterX + (cell.s === 'L' ? -1.15 : 1.15);
          const bayZ = cell.b * BAY_STEP;
          const tierY = (cell.t - 1) * TIER_HEIGHT + 0.48;

          let boxColor = 0x3B82F6; // Cold C-class
          if (cell.dem >= 25) boxColor = 0xEF4444; // Hot surge
          else if (cell.dem >= 8) boxColor = 0xF59E0B; // Medium

          const mat = new THREE.MeshStandardMaterial({{
            color: boxColor,
            roughness: 0.45,
            metalness: 0.15,
            emissive: cell.moved ? 0x06B6D4 : (cell.dem >= 25 ? 0x7F1D1D : 0x000000),
            emissiveIntensity: cell.moved ? 0.55 : 0.3
          }});

          const box = new THREE.Mesh(boxGeo, mat);
          box.position.set(rackX, tierY, bayZ);
          box.userData = cell;
          scene.add(box);
          boxMeshes.push(box);

          // Glowing Cyan Wireframe Outline for Quick-Fix Relocated SKUs
          if (cell.moved) {{
            const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(0.70, 0.70, 1.04));
            const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({{ color: 0x22D3EE }}));
            line.position.copy(box.position);
            scene.add(line);
          }}
        }});

        // Build Return-Routing Path & Animated Warehouse Picker (Worker + Cart)
        const waypoints = [new THREE.Vector3(0, 0.1, -1.6)];
        const aisleMaxBay = {{}};
        DATA.targets.forEach(t => {{
          if (!aisleMaxBay[t.aisle] || t.bay > aisleMaxBay[t.aisle]) {{
            aisleMaxBay[t.aisle] = t.bay;
          }}
        }});

        const sortedAisles = Object.keys(aisleMaxBay).map(Number).sort((a, b) => a - b);
        sortedAisles.forEach(a => {{
          const ax = (a - 1) * AISLE_STEP;
          const maxZ = aisleMaxBay[a] * BAY_STEP;
          waypoints.push(new THREE.Vector3(ax, 0.1, -0.3));
          waypoints.push(new THREE.Vector3(ax, 0.1, maxZ));
          waypoints.push(new THREE.Vector3(ax, 0.1, -0.3));
        }});
        waypoints.push(new THREE.Vector3(0, 0.1, -1.6));

        if (waypoints.length > 2) {{
          const pathGeo = new THREE.BufferGeometry().setFromPoints(waypoints);
          const pathLine = new THREE.Line(pathGeo, new THREE.LineBasicMaterial({{ color: 0x10B981 }}));
          scene.add(pathLine);
        }}

        // Picker Character + Picking Cart
        const pickerGroup = new THREE.Group();
        const bodyMesh = new THREE.Mesh(
          new THREE.CylinderGeometry(0.24, 0.28, 1.15, 16),
          new THREE.MeshStandardMaterial({{ color: 0x10B981, emissive: 0x059669, emissiveIntensity: 0.4 }})
        );
        bodyMesh.position.y = 0.75;
        const headMesh = new THREE.Mesh(
          new THREE.SphereGeometry(0.22, 16, 16),
          new THREE.MeshStandardMaterial({{ color: 0xFDE047 }})
        );
        headMesh.position.y = 1.55;
        const cartMesh = new THREE.Mesh(
          new THREE.BoxGeometry(0.55, 0.45, 0.75),
          new THREE.MeshStandardMaterial({{ color: 0x38BDF8, metalness: 0.6, roughness: 0.2 }})
        );
        cartMesh.position.set(0, 0.35, 0.55);
        pickerGroup.add(bodyMesh, headMesh, cartMesh);
        pickerGroup.position.copy(waypoints[0]);
        scene.add(pickerGroup);

        // Raycaster Tooltip on Hover
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        const tooltip = document.getElementById('tooltip');

        window.addEventListener('mousemove', (e) => {{
          mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
          mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
          raycaster.setFromCamera(mouse, camera);
          const hits = raycaster.intersectObjects(boxMeshes);
          if (hits.length > 0) {{
            const d = hits[0].object.userData;
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 14) + 'px';
            tooltip.style.top = (e.clientY + 14) + 'px';
            tooltip.innerHTML = `
              <b style="color:#38BDF8;">${{d.sku}}</b> (${{d.cat}})<br>
              위치: <b>${{d.a}}통로 ${{d.s}}열 ${{d.b}}베이 ${{d.t}}단</b> ${{d.gz ? '<span style="color:#22D3EE;">[골든존]</span>' : ''}}<br>
              주간 출고빈도: <b>${{d.dem}}건</b><br>
              상태: ${{d.moved ? '<span style="color:#22D3EE;">🔄 Dynamic Quick-Fix 재배치</span>' : '기존 배치'}}
            `;
          }} else {{
            tooltip.style.display = 'none';
          }}
        }});

        window.addEventListener('resize', () => {{
          camera.aspect = window.innerWidth / window.innerHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        // Animation Loop
        let wpIndex = 0;
        const clock = new THREE.Clock();

        function animate() {{
          requestAnimationFrame(animate);
          const dt = clock.getDelta();

          if (waypoints.length > 1 && PICKER_SPEED > 0) {{
            const target = waypoints[(wpIndex + 1) % waypoints.length];
            const dir = new THREE.Vector3().subVectors(target, pickerGroup.position);
            const dist = dir.length();
            const step = PICKER_SPEED * 5.0 * dt;
            if (dist <= step) {{
              pickerGroup.position.copy(target);
              wpIndex = (wpIndex + 1) % waypoints.length;
            }} else {{
              dir.normalize();
              pickerGroup.position.addScaledVector(dir, step);
              pickerGroup.lookAt(target.x, pickerGroup.position.y, target.z);
            }}
          }}

          controls.update();
          renderer.render(scene, camera);
        }}
        animate();
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=620, scrolling=False)


# ==============================================================================
# Sidebar & Dashboard Layout
# ==============================================================================
with st.sidebar:
    st.markdown("### 🏭 RDC 3D 관제 컨트롤")
    seed_val = st.selectbox("시뮬레이션 시드 (Seed)", [42, 43, 44, 45, 46], index=0)
    k_val = st.radio("투자회수 기준 (Horizon k주)", [1, 2, 3], index=1, horizontal=True)
    selected_week = st.slider("📅 타임라인 주차 (Week)", min_value=1, max_value=12, value=6, step=1)
    picker_speed = st.slider("🏃 피커(작업자) 이동 배속", min_value=0.0, max_value=3.0, value=1.2, step=0.2)

    st.markdown("---")
    if selected_week == 5:
        st.warning("⚡ **5주차 1차 수요 충격 발생!**\nC등급 20개 품목 주문 8배 폭증 → 빨간 박스 급증")
    elif selected_week == 9:
        st.warning("⚡ **9주차 2차 계절 변동 발생!**\n신규 C등급 10개 품목 주문 3.5배 급등")
    else:
        st.info("✅ **실시간 3D WES 가동 중**\n마우스 드래그로 랙 사이를 자유롭게 탐색하세요.")

weekly_data, kpi_df = run_3d_simulation_trace(seed=seed_val, k_horizon=k_val)
cur_week_data = weekly_data[selected_week]
cur_kpi = kpi_df[kpi_df["Week"] == selected_week].iloc[0]
prev_kpi = kpi_df[kpi_df["Week"] == max(1, selected_week - 1)].iloc[0]

st.markdown("""
<div class="wes-banner">
    <span class="badge">REAL-TIME 3D WES DIGITAL TWIN</span>
    <span class="badge">6 AISLES × 12 BAYS × 4 TIERS STEEL RACKS</span>
    <h3 style="margin:6px 0 2px 0; color:#F8FAFC;">유럽 권역 RDC 실제 3D 물류센터 디지털 트윈 관제 시뮬레이션</h3>
    <p style="margin:0; color:#94A3B8; font-size:0.86rem;">철제 팔레트 랙 · 골든존(2·3단) · I/O 도크 · Return 라우팅 피커 주행 · Dynamic Quick-Fix 슬롯 스왑 실시간 시각화</p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("시간당 피킹 처리량", f"{cur_kpi['Dynamic_Picks_Hr']:.2f} Picks/hr", f"{cur_kpi['Dynamic_Picks_Hr'] - cur_kpi['Static_GC2_Picks_Hr']:+.2f} vs Static-GC2")
m2.metric("평균 주문 처리시간", f"{cur_kpi['Dynamic_Order_Min']:.3f} 분", f"{cur_kpi['Dynamic_Order_Min'] - prev_kpi['Dynamic_Order_Min']:+.3f} 분", delta_color="inverse")
m3.metric("골든존(2·3단) 피킹 비율", f"{cur_kpi['Dynamic_GZ_Pct']:.1f} %", f"{cur_kpi['Dynamic_GZ_Pct'] - prev_kpi['Dynamic_GZ_Pct']:+.1f} %p")
m4.metric("주문당 평균 이동거리", f"{cur_kpi['Dynamic_Dist_m']:.1f} m", f"{cur_kpi['Dynamic_Dist_m'] - prev_kpi['Dynamic_Dist_m']:+.1f} m", delta_color="inverse")
m5.metric("누적 슬롯 재배치 (Churn)", f"{int(cur_kpi['Cumulative_Churn'])} / 300 SKU", f"전체의 {cur_kpi['Cumulative_Churn']/300*100:.1f}%")

# Render 3D Warehouse Scene
render_threejs_warehouse(cur_week_data, selected_week, picker_speed)

# Swap Execution Log Table
st.subheader(f"🔄 Week {selected_week} Shewhart 3σ 감지 및 Quick-Fix 슬롯 재배치 내역")
if cur_week_data["swaps"]:
    df_swaps = pd.DataFrame(cur_week_data["swaps"])
    df_swaps.columns = ["급등 품목 (Hot SKU)", "교체 저회전 품목 (Cold SKU)", "기존 위치", "신규 골든존 위치", "예상 순이익 (분)"]
    st.dataframe(df_swaps, use_container_width=True, hide_index=True)
else:
    st.caption("해당 주차에는 손익분기점(net_gain > 0)을 초과하는 신규 재배치 이벤트가 발생하지 않았습니다.")
