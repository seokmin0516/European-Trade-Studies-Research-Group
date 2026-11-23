import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# HS 8473 독일 RDC 3D 디지털 트윈 WES 시뮬레이터 (Clean Light UI 버전)
# - 설계계획서(수정판) 1,200㎡ 피킹 구역 구조 완비:
#   1) 선반+통로 (346㎡): 6통로 × 좌우 12베이 × 4단 (576셀), 통로폭 2.4m, 피치 4.0m, 깊이 14.4m
#   2) 앞쪽 순환통로 (72㎡, Front Cross-Aisle) — 뒤쪽 순환통로 없음 (Return 방식 정합)
#   3) P/D Point 카트 대기 버퍼 (150㎡)
#   4) 포장·검수·출하 준비 구역 (472㎡)
#   5) 안전통로·소방 대피 여유공간 (160㎡)
# - 작업 방식: Picker-to-Parts (인력 카트 도보 피킹 45.7 m/min, GZ 0.3분 vs Non-GZ 0.4분)
# - 비교 모델 4종 완비 (단일 뷰 전환 방식):
#   1. Random / 2. Static-W / 3. Static-GC2 (대조군) / 4. Dynamic Quick-Fix (제안 모델)
# ==============================================================================

st.set_page_config(
    page_title="유럽통상학 연구단",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 두 번째 이미지(CUBE SPACE) 스타일의 깔끔한 라이트 배경 + 인디고 포인트 테마 적용
# 상단 잘림 방지를 위해 padding-top을 3.6rem으로 충분히 확보
st.markdown("""
<style>
    .block-container {
        padding-top: 3.6rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
    }
    .stApp {
        background-color: #F4F6FB;
        color: #1E293B;
    }
    /* Streamlit 위젯 라벨 가독성 강화 */
    label[data-testid="stWidgetLabel"] p {
        color: #1E293B !important;
        font-weight: 700 !important;
        font-size: 13px !important;
    }
    div[data-testid="stRadio"] label p {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #5B6DF6;
        padding: 13px 20px;
        border-radius: 10px;
        margin-bottom: 14px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    .pill {
        background: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
        border-radius: 999px;
        padding: 3px 11px;
        font-size: 11.5px;
        font-weight: 700;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. 설계계획서 및 v7 코드 동일 상수 정의
# ==============================================================================
BASE_SEED = 42
NUM_WEEKS = 12
SHOCK_WEEK_1 = 5
SHOCK_WEEK_2 = 9

NUM_AISLES = 6
BAYS_PER_SIDE = 12
NUM_TIERS = 4
AISLE_PITCH = 4.0
BAY_WIDTH = 1.2

WALK_SPEED = 45.7         # 45.7 m/min (Petersen et al., 2005)
PICK_TIME_GZ = 0.3        # 0.3 min (18s) for Tier 2, 3
PICK_TIME_NON_GZ = 0.4    # 0.4 min (24s) for Tier 1, 4

TOTAL_SKUS = 300
TOP_SKU_RATIO = 0.20      # 상위 20% (60 SKUs) -> 3셀 (180셀)
CELLS_PER_TOP_SKU = 3
CELLS_PER_OTHER_SKU = 1   # 하위 80% (240 SKUs) -> 1셀 (240셀), 여유 버퍼 156셀
WEIGHT_LIMIT_KG = 5.0

ORDER_BATCH_SIZE = 10
ORDERS_PER_WEEK = 300
EWMA_ALPHA = 0.50
SPC_SIGMA_MULTIPLIER = 3.0
SWAP_COST_MIN = 2.0
MAX_SWAPS_PER_WEEK = 2


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


def generate_hs8473_skus(seed: int = BASE_SEED) -> List[SKU]:
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


def assign_random_slotting(wh: VirtualWarehouse, skus: List[SKU], seed: int):
    wh.clear_all_slots()
    for s in skus:
        s.current_cells = []
    available_cells = wh.cells.copy()
    rng = np.random.RandomState(seed + 100)
    rng.shuffle(available_cells)
    idx = 0
    for sku in skus:
        for _ in range(sku.allocated_cells):
            cell = available_cells[idx]
            cell.sku_id = sku.sku_id
            sku.current_cells.append(cell)
            idx += 1


def assign_w_storage_slotting(wh: VirtualWarehouse, skus: List[SKU], observed_counts: Dict[str, float]):
    wh.clear_all_slots()
    for s in skus:
        s.current_cells = []
    sorted_skus = sorted(skus, key=lambda s: observed_counts.get(s.sku_id, s.base_demand_weight), reverse=True)
    w_cells = sorted(wh.cells, key=lambda c: (c.aisle, c.bay, c.tier))
    idx = 0
    for sku in sorted_skus:
        for _ in range(sku.allocated_cells):
            cell = w_cells[idx]
            cell.sku_id = sku.sku_id
            sku.current_cells.append(cell)
            idx += 1


def get_gc2_sorted_cells(wh: VirtualWarehouse, near_aisles: Tuple[int, ...] = (1, 2)) -> List[Cell]:
    near_gz = sorted(
        [c for c in wh.cells if c.aisle in near_aisles and c.is_golden_zone],
        key=lambda c: (c.bay, c.aisle, c.tier)
    )
    far_gzw: List[Cell] = []
    for a in range(1, NUM_AISLES + 1):
        if a in near_aisles:
            continue
        gz_in_aisle = sorted([c for c in wh.cells if c.aisle == a and c.is_golden_zone], key=lambda c: (c.bay, c.tier))
        non_gz_in_aisle = sorted([c for c in wh.cells if c.aisle == a and not c.is_golden_zone], key=lambda c: (c.bay, c.tier))
        far_gzw.extend(gz_in_aisle + non_gz_in_aisle)
    near_non_gz = sorted(
        [c for c in wh.cells if c.aisle in near_aisles and not c.is_golden_zone],
        key=lambda c: (c.bay, c.aisle, c.tier)
    )
    return near_gz + far_gzw + near_non_gz


def assign_gc2_turnover_slotting(wh: VirtualWarehouse, skus: List[SKU],
                                 observed_counts: Dict[str, float],
                                 near_aisles: Tuple[int, ...] = (1, 2)):
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
                same_loc_t2 = [
                    c for c in free_cells
                    if c.aisle == chosen.aisle and c.bay == chosen.bay and c.side == chosen.side and c.tier == 2
                ]
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
        a = best_cell.aisle
        y = best_cell.y_coord

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
    cross_aisle_dist = 2.0 * ((max_aisle - 1) * AISLE_PITCH)
    in_aisle_dist = sum(2.0 * y for y in aisle_max_y.values())

    total_dist = cross_aisle_dist + in_aisle_dist
    travel_time = total_dist / WALK_SPEED
    total_time = travel_time + pick_time_total
    return total_dist, total_time, gz_pick_count, picked_targets


def cell_pooled_expected_cost_min(cell: Cell, aisle_expected_max_y: Dict[int, float],
                                  expected_max_aisle: float) -> float:
    ergo_min = cell.ergo_pick_time_min
    extra_cross_dist = 2.0 * max(0.0, (cell.aisle - expected_max_aisle) * AISLE_PITCH)
    cur_aisle_max = aisle_expected_max_y.get(cell.aisle, 0.0)
    extra_in_aisle_dist = 2.0 * max(0.0, cell.y_coord - cur_aisle_max)
    marginal_walk_min = (1.0 / ORDER_BATCH_SIZE) * ((extra_cross_dist + extra_in_aisle_dist) / WALK_SPEED)
    return ergo_min + marginal_walk_min


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
        prob_none_above = 1.0
        exp_y = 0.0
        for y, p_ord in items:
            exp_y += y * prob_none_above * p_ord
            prob_none_above *= (1.0 - p_ord)
        aisle_exp_max_y[a] = exp_y
        aisle_any_prob[a] = 1.0 - prob_none_above

    prob_none_right = 1.0
    exp_max_a = 1.0
    for a in range(NUM_AISLES, 0, -1):
        exp_max_a += (a - 1) * prob_none_right * aisle_any_prob[a]
        prob_none_right *= (1.0 - aisle_any_prob[a])

    return aisle_exp_max_y, exp_max_a


def serialize_warehouse_state(wh: VirtualWarehouse, skus_by_id: Dict[str, SKU], demand_dict: Dict[str, float], moved_set: set):
    cells_json = []
    a_cnt = b_cnt = c_cnt = empty_cnt = 0
    for c in wh.cells:
        if c.sku_id is None:
            empty_cnt += 1
            cells_json.append({
                "a": c.aisle, "s": c.side, "b": c.bay, "t": c.tier,
                "sku": "BUFFER", "cat": "스왑 버퍼 (빈 셀)", "dem": 0.0,
                "wt": 0.0, "esd": False,
                "grade": "EMPTY", "gz": c.is_golden_zone, "moved": False
            })
        else:
            s_obj = skus_by_id[c.sku_id]
            dem = round(demand_dict.get(c.sku_id, 0.0), 1)
            if dem >= 22.0:
                grade = "A"
                a_cnt += 1
            elif dem >= 6.0:
                grade = "B"
                b_cnt += 1
            else:
                grade = "C"
                c_cnt += 1
            cells_json.append({
                "a": c.aisle, "s": c.side, "b": c.bay, "t": c.tier,
                "sku": c.sku_id, "cat": s_obj.category, "dem": dem,
                "wt": s_obj.weight_kg, "esd": s_obj.is_esd,
                "grade": grade, "gz": c.is_golden_zone,
                "moved": (c.sku_id in moved_set)
            })
    return cells_json, {"a_cnt": a_cnt, "b_cnt": b_cnt, "c_cnt": c_cnt, "empty_cnt": empty_cnt}


# ==============================================================================
# 2. 전체 주차 시뮬레이션 실행 및 4개 모델 데이터 구축
# ==============================================================================
@st.cache_data(show_spinner="3D 디지털 트윈 시뮬레이션 데이터를 계산하고 있습니다...")
def build_full_simulation(seed: int = 42, k_horizon: int = 2, near_aisles: Tuple[int, ...] = (1, 2)):
    rng = np.random.RandomState(seed)
    skus_master = generate_hs8473_skus(seed=seed)
    base_weights = np.array([s.base_demand_weight for s in skus_master])
    base_weights /= base_weights.sum()

    initial_counts = {s.sku_id: 0.0 for s in skus_master}
    for _ in range(ORDERS_PER_WEEK * 4):
        chosen_idxs = rng.choice(TOTAL_SKUS, size=ORDER_BATCH_SIZE, replace=False, p=base_weights)
        for idx in chosen_idxs:
            initial_counts[skus_master[idx].sku_id] += 1.0

    wh_random = VirtualWarehouse()
    skus_random = generate_hs8473_skus(seed=seed)
    assign_random_slotting(wh_random, skus_random, seed=seed)
    skus_random_by_id = {s.sku_id: s for s in skus_random}

    wh_static_w = VirtualWarehouse()
    skus_static_w = generate_hs8473_skus(seed=seed)
    assign_w_storage_slotting(wh_static_w, skus_static_w, initial_counts)
    skus_static_w_by_id = {s.sku_id: s for s in skus_static_w}

    wh_static_gc2 = VirtualWarehouse()
    skus_static_gc2 = generate_hs8473_skus(seed=seed)
    assign_gc2_turnover_slotting(wh_static_gc2, skus_static_gc2, initial_counts, near_aisles=near_aisles)
    skus_static_gc2_by_id = {s.sku_id: s for s in skus_static_gc2}

    wh_dynamic = VirtualWarehouse()
    skus_dynamic = generate_hs8473_skus(seed=seed)
    assign_gc2_turnover_slotting(wh_dynamic, skus_dynamic, initial_counts, near_aisles=near_aisles)
    skus_dynamic_by_id = {s.sku_id: s for s in skus_dynamic}

    ewma_demand: Dict[str, float] = {s.sku_id: initial_counts[s.sku_id] / 4.0 for s in skus_dynamic}
    ewma_var: Dict[str, float] = {s.sku_id: max(initial_counts[s.sku_id] / 4.0, 1.0) for s in skus_dynamic}

    current_weights = base_weights.copy()
    moved_skus_set = set()
    weekly_snapshots = {}

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
        model_configs = [
            ("1. Random", wh_random, skus_random, skus_random_by_id),
            ("2. Static-W", wh_static_w, skus_static_w, skus_static_w_by_id),
            ("3. Static-GC2", wh_static_gc2, skus_static_gc2, skus_static_gc2_by_id),
            ("4. Dynamic Quick-Fix", wh_dynamic, skus_dynamic, skus_dynamic_by_id),
        ]

        week_metrics = {}
        week_targets = {}
        for m_name, _, sku_list, _ in model_configs:
            tot_dist = tot_time = 0.0
            tot_gz = 0
            sample_t = []
            for o_i, idxs in enumerate(week_order_indices):
                d, t, gz, targets = calculate_order_time_and_distance([sku_list[i] for i in idxs])
                tot_dist += d
                tot_time += t
                tot_gz += gz
                if o_i == 0:
                    sample_t = sorted(targets, key=lambda x: (x["aisle"], x["bay"]))

            picks_hr = round((tot_picks / tot_time) * 60.0, 2)
            avg_dist = round(tot_dist / ORDERS_PER_WEEK, 2)
            avg_time = round(tot_time / ORDERS_PER_WEEK, 3)
            gz_ratio = round((tot_gz / tot_picks) * 100.0, 2)

            week_metrics[m_name] = {
                "picks_hr": picks_hr,
                "dist_m": avg_dist,
                "time_min": avg_time,
                "gz_pct": gz_ratio
            }
            week_targets[m_name] = sample_t

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
                hot = skus_dynamic_by_id.get(hot_id)
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
                    if cell_pooled_expected_cost_min(cold_best_cell, aisle_max_y, expected_max_aisle) >= hot_cell_cost:
                        continue

                    hot_post_cells = [c for c in hot.current_cells if c is not hot_active_cell] + [cold_best_cell]
                    cold_post_cells = [c for c in cold.current_cells if c is not cold_best_cell] + [hot_active_cell]

                    hot_post_cost = cell_pooled_expected_cost_min(min(hot_post_cells, key=lambda c: (c.aisle, c.bay)), aisle_max_y, expected_max_aisle)
                    cold_pre_cost = cell_pooled_expected_cost_min(min(cold.current_cells, key=lambda c: (c.aisle, c.bay)), aisle_max_y, expected_max_aisle)
                    cold_post_cost = cell_pooled_expected_cost_min(min(cold_post_cells, key=lambda c: (c.aisle, c.bay)), aisle_max_y, expected_max_aisle)

                    weekly_saving = (
                        ewma_demand[hot.sku_id] * (hot_cell_cost - hot_post_cost)
                        + ewma_demand[cold.sku_id] * (cold_pre_cost - cold_post_cost)
                    )
                    net_gain = (k_horizon * weekly_saving) - (2.0 * SWAP_COST_MIN)
                    if net_gain > best_net_gain:
                        best_net_gain, best_partner, best_cold_cell = net_gain, cold, cold_best_cell

                if best_partner is not None and best_cold_cell is not None:
                    hot_active_cell.sku_id = best_partner.sku_id
                    best_cold_cell.sku_id = hot.sku_id
                    hot.current_cells.remove(hot_active_cell)
                    hot.current_cells.append(best_cold_cell)
                    best_partner.current_cells.remove(best_cold_cell)
                    best_partner.current_cells.append(hot_active_cell)
                    used_skus.update([hot.sku_id, best_partner.sku_id])
                    moved_skus_set.update([hot.sku_id, best_partner.sku_id])
                    swaps_this_week += 1

        models_3d = {}
        for m_name, wh_obj, _, s_map in model_configs:
            m_set = moved_skus_set if m_name == "4. Dynamic Quick-Fix" else set()
            c_json, counts = serialize_warehouse_state(wh_obj, s_map, ewma_demand, m_set)
            models_3d[m_name] = {
                "label": m_name,
                "cells": c_json,
                "targets": week_targets[m_name],
                "kpi": {
                    **week_metrics[m_name],
                    "churn": len(m_set),
                    "spc_triggered": len(triggered_skus),
                    **counts
                }
            }

        weekly_snapshots[week] = {
            "week": week,
            "models": models_3d
        }

    return weekly_snapshots


# ==============================================================================
# 3. Three.js 3D 디지털 트윈 WES 렌더러 (깔끔한 라이트 씬 + 기존 HUD 구조 유지)
# ==============================================================================
def render_rdc_digital_twin_3d(week_num: int, model_payload: dict, baseline_kpi: dict,
                               anim_speed: float, height: int = 660, is_proposed: bool = True):
    payload_str = json.dumps({
        "week": week_num,
        "label": model_payload["label"],
        "cells": model_payload["cells"],
        "targets": model_payload["targets"],
        "kpi": model_payload["kpi"],
        "diff_vs_gc2": round(model_payload["kpi"]["picks_hr"] - baseline_kpi["picks_hr"], 2)
    })

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      * {{ box-sizing: border-box; user-select: none; }}
      html, body {{
        margin: 0; padding: 0; width: 100%; height: 100%;
        overflow: hidden; background: #EAEFF7;
        font-family: 'Pretendard', 'Segoe UI', sans-serif;
      }}
      canvas {{
        display: block; width: 100% !important; height: 100% !important;
      }}

      /* 우측 WES 관제 HUD 패널 (기존 UI 형태 유지 + 깔끔한 화이트/슬레이트 톤 적용) */
      #hud-wrapper {{
        position: absolute; top: 14px; right: 14px; z-index: 20;
        display: flex; align-items: flex-start; gap: 6px;
      }}
      #hud-panel {{
        width: 245px;
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 12px 14px;
        color: #1E293B;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
        border: 1px solid #E2E8F0;
      }}
      #hud-panel.hidden {{ display: none; }}

      .hud-title-pill {{
        background: {'#5B6DF6' if is_proposed else '#64748B'}; color: #FFFFFF; text-align: center;
        font-weight: 700; font-size: 11.5px; padding: 5px 0;
        border-radius: 999px; margin-bottom: 8px; letter-spacing: 0.2px;
      }}
      .stat-pill {{
        display: flex; justify-content: space-between; align-items: center;
        background: #F1F5F9; color: #334155;
        border: 1px solid #E2E8F0;
        padding: 4.5px 11px; border-radius: 999px;
        font-size: 11px; margin-bottom: 5px; font-weight: 600;
      }}
      .stat-pill span.val {{ font-weight: 800; color: {'#4F46E5' if is_proposed else '#0F172A'}; font-size: 12px; }}

      .gauge-main {{
        text-align: center; margin: 7px 0 6px 0;
        background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 6px;
      }}
      .sub-gauges {{
        display: flex; justify-content: space-between; gap: 6px; margin-bottom: 8px;
      }}
      .sub-gauge-box {{
        flex: 1; text-align: center; background: #F8FAFC; border: 1px solid #E2E8F0;
        border-radius: 8px; padding: 5px 4px;
      }}

      .freq-btn {{
        width: 100%; background: #EEF2FF; color: #4338CA;
        font-weight: 700; font-size: 11.5px; text-align: center;
        padding: 4.5px 0; border-radius: 999px; margin: 5px 0 6px 0;
      }}
      .grade-row {{
        display: flex; justify-content: space-between; align-items: center;
        font-size: 11.5px; font-weight: 600; color: #334155;
        padding: 2.5px 5px;
      }}
      .grade-left {{ display: flex; align-items: center; gap: 7px; }}
      .circle {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}

      #icon-strip {{
        width: 44px;
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 7px 3px;
        display: flex; flex-direction: column; align-items: center; gap: 8px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.10);
      }}
      .icon-btn {{
        width: 36px; text-align: center; cursor: pointer;
        padding: 4px 2px; border-radius: 8px; color: #475569;
        font-size: 9px; font-weight: 700; transition: 0.15s;
      }}
      .icon-btn:hover, .icon-btn.active {{ background: #5B6DF6; color: #FFFFFF; }}
      .icon-emoji {{ font-size: 15px; display: block; margin-bottom: 1px; }}

      #tooltip {{
        position: absolute; display: none; z-index: 30;
        background: rgba(255, 255, 255, 0.97); border: 1.5px solid #5B6DF6;
        color: #1E293B; padding: 9px 12px; border-radius: 8px;
        font-size: 11.5px; pointer-events: none; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.15);
      }}
      #bottom-hint {{
        position: absolute; bottom: 12px; left: 14px; z-index: 15;
        background: rgba(255, 255, 255, 0.92); color: #334155;
        padding: 6px 12px; border-radius: 8px; font-size: 11px; font-weight: 600;
        border: 1px solid #CBD5E1; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
      }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
      <div id="hud-wrapper">
        <div id="hud-panel">
          <div class="hud-title-pill">{model_payload['label']} (W{week_num})</div>
          <div class="stat-pill"><span>피킹 구역 면적</span><span class="val">1,200㎡ (576셀)</span></div>
          <div class="stat-pill"><span>시간당 처리량</span><span class="val">{model_payload['kpi']['picks_hr']} Picks/hr</span></div>
          <div class="stat-pill"><span>주문당 평균 거리</span><span class="val">{model_payload['kpi']['dist_m']} m</span></div>
          <div class="stat-pill"><span>누적 재배치 품목</span><span class="val">{model_payload['kpi']['churn']} SKU</span></div>

          <div class="gauge-main">
            <svg width="115" height="68" viewBox="0 0 120 72">
              <path d="M 15 62 A 45 45 0 0 1 105 62" fill="none" stroke="#CBD5E1" stroke-width="10" stroke-linecap="round"/>
              <path d="M 15 62 A 45 45 0 0 1 105 62" fill="none" stroke="{'#5B6DF6' if is_proposed else '#64748B'}" stroke-width="10" stroke-linecap="round"
                    stroke-dasharray="141" stroke-dashoffset="{141 * (1 - model_payload['kpi']['gz_pct']/100.0):.1f}"/>
              <text x="60" y="54" text-anchor="middle" font-size="16" font-weight="800" fill="#0F172A">{model_payload['kpi']['gz_pct']}%</text>
              <text x="60" y="68" text-anchor="middle" font-size="9" font-weight="700" fill="#64748B">골든존(2·3단) 비율</text>
            </svg>
          </div>

          <div class="sub-gauges">
            <div class="sub-gauge-box">
              <div style="font-size:12.5px; font-weight:800; color:{'#10B981' if (model_payload['kpi']['picks_hr'] - baseline_kpi['picks_hr']) >= 0 else '#EF4444'};">{model_payload['kpi']['picks_hr'] - baseline_kpi['picks_hr']:+.2f}</div>
              <div style="font-size:8.5px; color:#64748B; font-weight:600;">vs Static-GC2</div>
            </div>
            <div class="sub-gauge-box">
              <div style="font-size:12.5px; font-weight:800; color:#EF4444;">{model_payload['kpi']['spc_triggered']}건</div>
              <div style="font-size:8.5px; color:#64748B; font-weight:600;">3σ SPC 감지</div>
            </div>
          </div>

          <div class="freq-btn">출고빈도 (ABC 등급)</div>
          <div class="grade-row">
            <div class="grade-left"><span class="circle" style="background:#F43F5E;"></span>A등급 (초고회전)</div>
            <span>{model_payload['kpi']['a_cnt']}셀</span>
          </div>
          <div class="grade-row">
            <div class="grade-left"><span class="circle" style="background:#5B6DF6;"></span>B등급 (중회전)</div>
            <span>{model_payload['kpi']['b_cnt']}셀</span>
          </div>
          <div class="grade-row">
            <div class="grade-left"><span class="circle" style="background:#94A3B8;"></span>C등급 (저회전)</div>
            <span>{model_payload['kpi']['c_cnt']}셀</span>
          </div>
          <div class="grade-row">
            <div class="grade-left"><span class="circle" style="background:#FFFFFF; border:1.5px solid #CBD5E1;"></span>스왑 버퍼(빈셀)</div>
            <span>{model_payload['kpi']['empty_cnt']}셀</span>
          </div>
        </div>

        <div id="icon-strip">
          <div class="icon-btn active" onclick="resetCam()"><span class="icon-emoji">🎥</span>Reset</div>
          <div class="icon-btn" onclick="topCam()"><span class="icon-emoji">🗺️</span>Layout</div>
          <div class="icon-btn" onclick="aisleCam()"><span class="icon-emoji">🚶</span>Picker</div>
          <div class="icon-btn" onclick="toggleHud()"><span class="icon-emoji">👁️</span>Hide</div>
        </div>
      </div>

      <div id="tooltip"></div>
      <div id="bottom-hint">👷 Picker-to-Parts (45.7 m/min) · Return 라우팅 · 앞쪽 순환통로 / P·D 버퍼 / 포장출하 구역</div>

      <script>
        const DATA = {payload_str};
        const ANIM_SPEED = {anim_speed};
        const IS_PROPOSED = {'true' if is_proposed else 'false'};

        function toggleHud() {{
          document.getElementById('hud-panel').classList.toggle('hidden');
        }}

        // 밝고 쾌적한 라이트 배경 씬
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xE8EEF6);
        scene.fog = new THREE.FogExp2(0xE8EEF6, 0.0055);

        const initW = Math.max(document.documentElement.clientWidth || window.innerWidth, 300);
        const initH = Math.max(document.documentElement.clientHeight || window.innerHeight, 300);

        const camera = new THREE.PerspectiveCamera(44, initW / initH, 0.1, 300);
        camera.position.set(14, 21, 31);

        const renderer = new THREE.WebGLRenderer({{
          antialias: true,
          powerPreference: 'high-performance',
          preserveDrawingBuffer: true
        }});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(initW, initH, false);
        renderer.shadowMap.enabled = true;
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(11.5, 1.5, 6.0);
        controls.enableDamping = true;
        controls.dampingFactor = 0.06;

        function resetCam() {{ camera.position.set(14, 21, 31); controls.target.set(11.5, 1.5, 6.0); }}
        function topCam() {{ camera.position.set(11.5, 36, 6.0); controls.target.set(11.5, 0, 5.5); }}
        function aisleCam() {{ camera.position.set(0.0, 3.0, -4.0); controls.target.set(0.0, 1.6, 10.0); }}

        scene.add(new THREE.AmbientLight(0xFFFFFF, 0.95));
        const dirLight = new THREE.DirectionalLight(0xFFFFFF, 0.65);
        dirLight.position.set(22, 42, 24);
        dirLight.castShadow = true;
        scene.add(dirLight);

        // 1. 전체 피킹 구역 바닥 (밝은 에폭시 물류센터 바닥)
        const floorCanvas = document.createElement('canvas');
        floorCanvas.width = 128; floorCanvas.height = 128;
        const fctx = floorCanvas.getContext('2d');
        fctx.fillStyle = '#DCE4F0';
        fctx.fillRect(0, 0, 128, 128);
        fctx.strokeStyle = '#CBD5E1';
        fctx.lineWidth = 2;
        for(let y=0; y<128; y+=16) {{
          for(let x=(y%32===0?0:8); x<128; x+=16) {{
            fctx.strokeRect(x, y, 16, 16);
          }}
        }}
        const floorTex = new THREE.CanvasTexture(floorCanvas);
        floorTex.wrapS = THREE.RepeatWrapping;
        floorTex.wrapT = THREE.RepeatWrapping;
        floorTex.repeat.set(28, 24);

        const floor = new THREE.Mesh(
          new THREE.PlaneGeometry(44, 38),
          new THREE.MeshStandardMaterial({{ map: floorTex, roughness: 0.85 }})
        );
        floor.rotation.x = -Math.PI / 2;
        floor.position.set(11.5, 0, 5.5);
        floor.receiveShadow = true;
        scene.add(floor);

        // 뒤쪽 막힌 벽면 (깔끔한 라이트 슬레이트)
        const backWall = new THREE.Mesh(
          new THREE.BoxGeometry(31, 4.2, 0.3),
          new THREE.MeshStandardMaterial({{ color: 0x94A3B8, roughness: 0.7 }})
        );
        backWall.position.set(11.5, 2.1, 17.5);
        scene.add(backWall);

        // 2. 앞쪽 순환통로 (72㎡ Front Cross-Aisle) - 인디고/슬레이트 화살표 레인
        const laneCanvas = document.createElement('canvas');
        laneCanvas.width = 512; laneCanvas.height = 64;
        const lctx = laneCanvas.getContext('2d');
        lctx.fillStyle = IS_PROPOSED ? '#EEF2FF' : '#E2E8F0';
        lctx.fillRect(0, 0, 512, 64);
        lctx.fillStyle = IS_PROPOSED ? '#5B6DF6' : '#94A3B8';
        lctx.fillRect(0, 0, 512, 6);
        lctx.fillRect(0, 58, 512, 6);
        lctx.font = 'bold 26px sans-serif';
        lctx.fillStyle = IS_PROPOSED ? '#4F46E5' : '#64748B';
        for(let x=30; x<512; x+=120) {{
          lctx.fillText('>>>', x, 42);
        }}
        const laneTex = new THREE.CanvasTexture(laneCanvas);
        laneTex.wrapS = THREE.RepeatWrapping;
        laneTex.repeat.set(4, 1);

        const mainLane = new THREE.Mesh(
          new THREE.PlaneGeometry(32, 2.4),
          new THREE.MeshBasicMaterial({{ map: laneTex }})
        );
        mainLane.rotation.x = -Math.PI / 2;
        mainLane.position.set(11.5, 0.02, -2.0);
        scene.add(mainLane);

        // 3. P/D Point 카트 대기 버퍼 (150㎡) & 포장·검수·출하 준비 구역 (472㎡)
        function createFloorZonePlate(text, subtext, x, z, w, d, bgColor, borderColor, titleColor) {{
          const c = document.createElement('canvas');
          c.width = 512; c.height = 256;
          const ctx = c.getContext('2d');
          ctx.fillStyle = bgColor;
          ctx.fillRect(0, 0, 512, 256);
          ctx.strokeStyle = borderColor;
          ctx.lineWidth = 10;
          ctx.strokeRect(6, 6, 500, 244);
          ctx.fillStyle = titleColor;
          ctx.font = 'bold 34px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(text, 256, 115);
          ctx.fillStyle = '#64748B';
          ctx.font = 'bold 22px sans-serif';
          ctx.fillText(subtext, 256, 165);
          const tex = new THREE.CanvasTexture(c);
          const plate = new THREE.Mesh(
            new THREE.PlaneGeometry(w, d),
            new THREE.MeshBasicMaterial({{ map: tex }})
          );
          plate.rotation.x = -Math.PI / 2;
          plate.position.set(x, 0.025, z);
          scene.add(plate);
        }}

        createFloorZonePlate('P/D Point 카트 대기 버퍼', '150㎡ · 피킹 시작/종료 기준점', 1.5, -6.2, 10.5, 4.4, '#FFFFFF', '#5B6DF6', '#3730A3');
        createFloorZonePlate('포장 · 검수 · 출하 준비 구역', '472㎡ · Packing & Outbound', 16.5, -6.2, 17.5, 4.4, '#FFFFFF', '#10B981', '#065F46');

        // 포장 작업대 시각화
        for (let px = 10; px <= 22; px += 4) {{
          const table = new THREE.Mesh(
            new THREE.BoxGeometry(2.2, 0.85, 1.1),
            new THREE.MeshStandardMaterial({{ color: 0x64748B, roughness: 0.4 }})
          );
          table.position.set(px, 0.43, -6.2);
          scene.add(table);
        }}

        // 4. 상단 구역 간판 (화이트 카드 + 인디고 테두리 스타일)
        function createZoneSign(text, x, y, z) {{
          const c = document.createElement('canvas');
          c.width = 512; c.height = 96;
          const ctx = c.getContext('2d');
          ctx.fillStyle = '#FFFFFF';
          ctx.strokeStyle = IS_PROPOSED ? '#5B6DF6' : '#94A3B8';
          ctx.lineWidth = 8;
          ctx.beginPath();
          ctx.roundRect(8, 8, 496, 80, 24);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = IS_PROPOSED ? '#3730A3' : '#334155';
          ctx.font = 'bold 34px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(text, 256, 60);
          const tex = new THREE.CanvasTexture(c);
          const sign = new THREE.Mesh(
            new THREE.BoxGeometry(6.2, 1.05, 0.25),
            new THREE.MeshBasicMaterial({{ map: tex }})
          );
          sign.position.set(x, y, z);
          scene.add(sign);
        }}

        const beamBar = new THREE.Mesh(
          new THREE.BoxGeometry(30, 0.30, 0.30),
          new THREE.MeshStandardMaterial({{ color: 0x94A3B8, metalness: 0.4, roughness: 0.4 }})
        );
        beamBar.position.set(11.5, 4.8, 0.4);
        scene.add(beamBar);

        createZoneSign('Aisle 1-2 : GZA Fast Zone', 2.3, 4.8, 0.65);
        createZoneSign('Aisle 3-4 : GZW Main Rack', 11.5, 4.8, 0.65);
        createZoneSign('Aisle 5-6 : GZW Reserve', 20.7, 4.8, 0.65);

        // 5. 6통로 × 12베이 × 4단 (576셀) 랙 및 토트박스 렌더링
        const AISLE_STEP = 4.6;
        const BAY_STEP = 1.35;
        const TIER_HEIGHT = 0.82;

        const postMat = new THREE.MeshStandardMaterial({{ color: 0x64748B, metalness: 0.4, roughness: 0.5 }});
        const postGeo = new THREE.BoxGeometry(0.08, 4 * TIER_HEIGHT + 0.15, 0.08);
        const boxGeo = new THREE.BoxGeometry(0.86, 0.70, 1.18);

        for (let a = 1; a <= 6; a++) {{
          const ax = (a - 1) * AISLE_STEP;
          ['L', 'R'].forEach(side => {{
            const rx = ax + (side === 'L' ? -1.05 : 1.05);
            for (let b = 1; b <= 12; b += 3) {{
              const bz = b * BAY_STEP - 0.6;
              const p1 = new THREE.Mesh(postGeo, postMat);
              p1.position.set(rx - 0.45, 1.7, bz);
              const p2 = new THREE.Mesh(postGeo, postMat);
              p2.position.set(rx + 0.45, 1.7, bz);
              scene.add(p1, p2);
            }}
          }});
        }}

        const boxMeshes = [];
        DATA.cells.forEach(cell => {{
          const ax = (cell.a - 1) * AISLE_STEP;
          const rx = ax + (cell.s === 'L' ? -1.05 : 1.05);
          const bz = cell.b * BAY_STEP;
          const ty = (cell.t - 1) * TIER_HEIGHT + 0.40;

          let mat;
          if (cell.grade === 'EMPTY') {{
            mat = new THREE.MeshStandardMaterial({{
              color: 0xFFFFFF, transparent: true, opacity: 0.35, roughness: 0.3
            }});
          }} else if (cell.grade === 'A') {{
            mat = new THREE.MeshStandardMaterial({{
              color: 0xF43F5E, roughness: 0.35,
              emissive: cell.moved ? 0x5B6DF6 : 0x881337,
              emissiveIntensity: cell.moved ? 0.35 : 0.12
            }});
          }} else if (cell.grade === 'B') {{
            mat = new THREE.MeshStandardMaterial({{
              color: 0x5B6DF6, roughness: 0.35,
              emissive: cell.moved ? 0x10B981 : 0x000000,
              emissiveIntensity: cell.moved ? 0.35 : 0.0
            }});
          }} else {{
            mat = new THREE.MeshStandardMaterial({{
              color: 0x94A3B8, roughness: 0.45
            }});
          }}

          const mesh = new THREE.Mesh(boxGeo, mat);
          mesh.position.set(rx, ty, bz);
          mesh.userData = cell;
          scene.add(mesh);
          if (cell.grade !== 'EMPTY') boxMeshes.push(mesh);

          if (cell.moved) {{
            const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(0.92, 0.76, 1.24));
            const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({{ color: 0x10B981 }}));
            line.position.copy(mesh.position);
            scene.add(line);
          }}
        }});

        // 6. 인력 피킹 작업자 + 다단 피킹 카트 3D 모델
        function createHumanPickerWithCart(vestColorHex) {{
          const workerGroup = new THREE.Group();
          const skinMat = new THREE.MeshStandardMaterial({{ color: 0xFDBA74, roughness: 0.5 }});
          const helmetMat = new THREE.MeshStandardMaterial({{ color: 0xFACC15, roughness: 0.3 }});
          const vestMat = new THREE.MeshStandardMaterial({{ color: vestColorHex, roughness: 0.4 }});
          const pantsMat = new THREE.MeshStandardMaterial({{ color: 0x334155, roughness: 0.7 }});
          const cartMat = new THREE.MeshStandardMaterial({{ color: 0x64748B, metalness: 0.5, roughness: 0.3 }});
          const toteMat = new THREE.MeshStandardMaterial({{ color: 0x5B6DF6, roughness: 0.4 }});

          const leftLeg = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.07, 0.75, 10), pantsMat);
          leftLeg.position.set(-0.11, 0.38, 0);
          const rightLeg = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.07, 0.75, 10), pantsMat);
          rightLeg.position.set(0.11, 0.38, 0);

          const torso = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.58, 0.24), vestMat);
          torso.position.set(0, 1.02, 0);

          const head = new THREE.Mesh(new THREE.SphereGeometry(0.15, 14, 14), skinMat);
          head.position.set(0, 1.46, 0);
          const helmet = new THREE.Mesh(new THREE.SphereGeometry(0.165, 14, 10, 0, Math.PI * 2, 0, Math.PI * 0.55), helmetMat);
          helmet.position.set(0, 1.49, 0);

          const leftArm = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.05, 0.55, 10), vestMat);
          leftArm.rotation.x = Math.PI / 3;
          leftArm.position.set(-0.26, 1.05, 0.24);
          const rightArm = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.05, 0.55, 10), vestMat);
          rightArm.rotation.x = Math.PI / 3;
          rightArm.position.set(0.26, 1.05, 0.24);

          const cartGroup = new THREE.Group();
          cartGroup.position.set(0, 0, 0.78);

          const baseTray = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.06, 0.85), cartMat);
          baseTray.position.set(0, 0.18, 0);
          const midTray = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.05, 0.85), cartMat);
          midTray.position.set(0, 0.65, 0);
          const handleBar = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.05, 0.05), cartMat);
          handleBar.position.set(0, 0.98, -0.4);

          [[-0.26, -0.38], [0.26, -0.38], [-0.26, 0.38], [0.26, 0.38]].forEach(p => {{
            const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.85), cartMat);
            pole.position.set(p[0], 0.58, p[1]);
            const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.05, 12), pantsMat);
            wheel.rotation.z = Math.PI / 2;
            wheel.position.set(p[0], 0.08, p[1]);
            cartGroup.add(pole, wheel);
          }});

          const tote1 = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.28, 0.36), toteMat);
          tote1.position.set(0, 0.82, -0.16);
          const tote2 = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.28, 0.36), toteMat);
          tote2.position.set(0, 0.82, 0.22);
          cartGroup.add(baseTray, midTray, handleBar, tote1, tote2);

          workerGroup.add(leftLeg, rightLeg, torso, head, helmet, leftArm, rightArm, cartGroup);
          workerGroup.userData = {{ leftLeg, rightLeg }};
          return workerGroup;
        }}

        const mainPicker = createHumanPickerWithCart(IS_PROPOSED ? 0xF97316 : 0x64748B);
        scene.add(mainPicker);

        const stagingPicker = createHumanPickerWithCart(0x10B981);
        stagingPicker.position.set(0, 0, -2.0);
        stagingPicker.rotation.y = Math.PI / 2;
        scene.add(stagingPicker);

        const waypoints = [new THREE.Vector3(0, 0, -1.8)];
        const aisleMaxBay = {{}};
        DATA.targets.forEach(t => {{
          if (!aisleMaxBay[t.aisle] || t.bay > aisleMaxBay[t.aisle]) aisleMaxBay[t.aisle] = t.bay;
        }});
        Object.keys(aisleMaxBay).map(Number).sort((a,b)=>a-b).forEach(a => {{
          const ax = (a - 1) * AISLE_STEP;
          const maxZ = aisleMaxBay[a] * BAY_STEP;
          waypoints.push(new THREE.Vector3(ax, 0, -1.8));
          waypoints.push(new THREE.Vector3(ax, 0, maxZ));
          waypoints.push(new THREE.Vector3(ax, 0, -1.8));
        }});
        waypoints.push(new THREE.Vector3(0, 0, -1.8));
        mainPicker.position.copy(waypoints[0]);

        if (waypoints.length > 2) {{
          const routeLine = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(waypoints.map(v => new THREE.Vector3(v.x, 0.05, v.z))),
            new THREE.LineBasicMaterial({{ color: IS_PROPOSED ? 0x4F46E5 : 0xF59E0B }})
          );
          scene.add(routeLine);
        }}

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
              <b style="color:#4F46E5;">${{d.sku}}</b> (${{d.cat}})<br>
              위치: <b>${{d.a}}통로 ${{d.s}}열 ${{d.b}}베이 ${{d.t}}단</b> ${{d.gz ? '<span style="color:#10B981; font-weight:700;">[골든존 0.3분]</span>' : '<span style="color:#64748B;">[일반존 0.4분]</span>'}}<br>
              물성: 중량 <b>${{d.wt}}kg</b> ${{d.esd ? '· <span style="color:#D97706; font-weight:700;">⚡ESD 민감부품</span>' : ''}}<br>
              주간 출고량: <b>${{d.dem}}건 (${{d.grade}}등급)</b><br>
              ${{d.moved ? '<span style="color:#10B981; font-weight:700;">🔄 Dynamic Quick-Fix 재배치 완료</span>' : '기본 할당 슬롯'}}
            `;
          }} else {{
            tooltip.style.display = 'none';
          }}
        }});

        let lastW = initW, lastH = initH;
        function handleResize() {{
          const w = document.documentElement.clientWidth || window.innerWidth;
          const h = document.documentElement.clientHeight || window.innerHeight;
          if (w > 0 && h > 0 && (w !== lastW || h !== lastH)) {{
            lastW = w;
            lastH = h;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h, false);
          }}
        }}
        window.addEventListener('resize', handleResize);

        let wpIdx = 0;
        let frontDir = 1;
        let walkPhase = 0;
        const clock = new THREE.Clock();

        function animate() {{
          requestAnimationFrame(animate);
          const dt = clock.getDelta();
          laneTex.offset.x -= dt * 0.35 * ANIM_SPEED;

          if (ANIM_SPEED > 0) {{
            walkPhase += dt * 9.5 * ANIM_SPEED;
            const swing = Math.sin(walkPhase) * 0.45;

            mainPicker.userData.leftLeg.rotation.x = swing;
            mainPicker.userData.rightLeg.rotation.x = -swing;
            stagingPicker.userData.leftLeg.rotation.x = -swing;
            stagingPicker.userData.rightLeg.rotation.x = swing;

            stagingPicker.position.x += frontDir * dt * 2.8 * ANIM_SPEED;
            if (stagingPicker.position.x > 22) {{ frontDir = -1; stagingPicker.rotation.y = -Math.PI / 2; }}
            if (stagingPicker.position.x < 1) {{ frontDir = 1; stagingPicker.rotation.y = Math.PI / 2; }}

            if (waypoints.length > 1) {{
              const target = waypoints[(wpIdx + 1) % waypoints.length];
              const diff = new THREE.Vector3().subVectors(target, mainPicker.position);
              const dist = diff.length();
              const step = dt * 3.4 * ANIM_SPEED;
              if (dist <= step) {{
                mainPicker.position.copy(target);
                wpIdx = (wpIdx + 1) % waypoints.length;
              }} else {{
                diff.normalize();
                mainPicker.position.addScaledVector(diff, step);
                mainPicker.lookAt(target.x, 0, target.z);
              }}
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
    components.html(html_code, height=height, scrolling=False)


# ==============================================================================
# 4. 상단 제어 바 & 3D 디지털 트윈 단일 뷰 대시보드
# ==============================================================================
st.markdown("""
<div class="top-bar">
  <div>
    <span class="pill">HS 8473 독일 RDC</span>
    <span class="pill">피킹 구역 1,200㎡ (576셀 · 300 SKU)</span>
    <strong style="color:#0F172A; font-size:16px; margin-left:6px;">유럽통상학 연구단</strong>
  </div>
  <div style="color:#64748B; font-size:12px; font-weight:600;">Picker-to-Parts (45.7 m/min) · Return 라우팅 · Dynamic Quick-Fix</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns([3.0, 3.0, 2.0, 2.0])
with c1:
    selected_week = st.slider("📅 시뮬레이션 주차 (5주차 1차 충격 / 9주차 2차 충격)", 1, 12, 6)
with c2:
    selected_model = st.radio(
        "🖥️ 3D 관제 모델 선택 (하나씩 보기)",
        [
            "4. Dynamic Quick-Fix (제안 모델)",
            "3. Static-GC2 (고정 골든존 대조군)",
            "2. Static-W (고정 구역 대조군)",
            "1. Random (무작위 배치 대조군)"
        ],
        index=0
    )
with c3:
    k_val = st.selectbox(
        "⏳ 재배치 효과 검증 기간",
        [1, 2, 3],
        index=1,
        format_func=lambda x: f"{x}주 안에 본전 회수 (k={x})",
        help="선반 위치를 바꿀 때 드는 작업 시간(비용)을 몇 주 안에 회수해야 재배치를 승인할지 정하는 기준입니다."
    )
    seed_val = st.selectbox(
        "🎲 주문 패턴 시나리오 번호",
        [42, 43, 44, 45, 46],
        index=0,
        format_func=lambda x: f"시나리오 #{x - 41} (Seed {x})",
        help="무작위 주문 발생 패턴을 고정하는 번호입니다. 번호를 바꾸면 다른 주문 상황을 테스트할 수 있습니다."
    )
with c4:
    anim_speed = st.slider("🚶 작업자 보행 배속", 0.0, 2.5, 1.0, 0.1)

weekly_snapshots = build_full_simulation(seed=seed_val, k_horizon=k_val, near_aisles=(1, 2))
cur_snap = weekly_snapshots[selected_week]
gc2_baseline_kpi = cur_snap["models"]["3. Static-GC2"]["kpi"]

model_key_map = {
    "4. Dynamic Quick-Fix (제안 모델)": "4. Dynamic Quick-Fix",
    "3. Static-GC2 (고정 골든존 대조군)": "3. Static-GC2",
    "2. Static-W (고정 구역 대조군)": "2. Static-W",
    "1. Random (무작위 배치 대조군)": "1. Random",
}
active_key = model_key_map[selected_model]

st.markdown('''
<style>
  iframe[title="streamlit_components_v1.components.v1.html"] {
    border-radius: 12px !important;
    border: 1px solid #CBD5E1 !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06) !important;
    display: block !important;
  }
</style>
''', unsafe_allow_html=True)

render_rdc_digital_twin_3d(
    selected_week,
    cur_snap["models"][active_key],
    gc2_baseline_kpi,
    anim_speed,
    height=650,
    is_proposed=(active_key == "4. Dynamic Quick-Fix")
)
