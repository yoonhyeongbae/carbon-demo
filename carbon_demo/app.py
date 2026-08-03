from __future__ import annotations

import hashlib
import io
import json
import math
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from folium.plugins import Fullscreen
from geopy.distance import geodesic
from ortools.linear_solver import pywraplp

try:
    import streamlit as st
    from streamlit_folium import st_folium
except Exception:  # local syntax/core testing without Streamlit
    st = None
    st_folium = None


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ASSET_DIR = APP_DIR / "assets"
REFERENCE_DIR = APP_DIR / "reference"

APP_BUILD = "xpress-parameter-ortools-full-lp-relaxation-v7.1"
APP_PACKAGE_ID = "20260803-2208-KST"
REFERENCE_LP_SHA256 = "efe0ec2e80a26b07dcbec47d2eaf74fb300cd63a5014e81e90147f9581ba4244"

REQUIRED_FILES = [
    "products.csv",
    "demand.csv",
    "raw_material_suppliers.csv",
    "assembly_locations.csv",
    "transport_parameters.csv",
    "markets.csv",
    "scenarios.csv",
    "poster_benchmark_cost_ratios.csv",
    "poster_benchmark_quartiles.csv",
    "xpress_model_metadata.csv",
]

MATERIALS = ["steel", "aluminum", "other", "battery"]
MATERIAL_INDEX = {m: i for i, m in enumerate(MATERIALS)}
MATERIAL_LABEL = {
    "steel": "철강",
    "aluminum": "알루미늄",
    "other": "기타 원자재",
    "battery": "배터리",
}
MATERIAL_COLOR = {
    "steel": "#e41a1c",
    "aluminum": "#ff9f1c",
    "other": "#238b45",
    "battery": "#2171b5",
    "finished": "#6a3d9a",
}
MODE_LABEL = {
    "line": "라인 생산",
    "modular": "모듈 활용 분산 생산",
}
SCENARIO_SHORT = {"S1": "시나리오 ①", "S2": "시나리오 ②", "S3": "시나리오 ③"}
QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]
TRANSPORT_COLOR = {1: "#1f78b4", 2: "#e31a1c", 3: "#999999", 4: "#666666"}
TRANSPORT_DASH = {1: None, 2: "3,7", 3: "8,5", 4: "1,5"}

BIG_M = 1_000_000.0
MIN_DISTANCE_KM = 50.0
MATERIAL_LOSS_RATE = 0.0
FLOW_TOL = 1e-6


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_default_tables() -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for name in REQUIRED_FILES:
        path = DATA_DIR / name
        if path.exists():
            tables[name] = pd.read_csv(path, encoding="utf-8-sig")
    return tables


def load_uploaded_tables(uploaded_files) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for uploaded in uploaded_files or []:
        name = Path(uploaded.name).name
        tables[name] = pd.read_csv(io.BytesIO(uploaded.getvalue()), encoding="utf-8-sig")
    return tables


def make_data_zip() -> bytes:
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in REQUIRED_FILES:
            path = DATA_DIR / name
            if path.exists():
                zf.write(path, arcname=f"data/{name}")
        ref = REFERENCE_DIR / "electric_car_modular.lp"
        if ref.exists():
            zf.write(ref, arcname="reference/electric_car_modular.lp")
    return memory.getvalue()


def validate_tables(tables: Mapping[str, pd.DataFrame]) -> List[str]:
    errors: List[str] = []
    missing = [name for name in REQUIRED_FILES if name not in tables]
    if missing:
        return ["필수 CSV 누락: " + ", ".join(missing)]

    required_columns = {
        "products.csv": {
            "xpress_product_index", "product_id", "vehicle_class", "trim", "product_name_ko",
            "battery_kwh", "vehicle_mass_kg", "nonbattery_mass_kg", "steel_kg", "aluminum_kg",
            "other_material_kg", "battery_mass_kg",
        },
        "demand.csv": {"product_id", "market_id", "demand_units"},
        "raw_material_suppliers.csv": {
            "material_id", "xpress_material_index", "supplier_id", "xpress_location_index",
            "location_name", "continent", "latitude", "longitude", "production_ef",
            "production_cost", "parameter_unit", "capacity", "capacity_unit",
        },
        "assembly_locations.csv": {
            "xpress_location_index", "plant_id", "location_name", "continent", "latitude",
            "longitude", "assembly_ef_kgco2_per_kg", "assembly_cost_eur_per_kg",
        },
        "transport_parameters.csv": {
            "xpress_mode_index", "transport_mode", "transport_mode_ko",
            "transport_cost_eur_per_unitkm", "transport_ef_kgco2_per_unitkm",
        },
        "markets.csv": {"market_id", "market_name", "latitude", "longitude", "minimum_distance_km"},
        "scenarios.csv": {
            "scenario_id", "scenario_name", "minimum_score", "apply_carbon_cap",
            "small_cap_kgco2_per_vehicle", "standard_cap_kgco2_per_vehicle",
        },
    }
    for filename, columns in required_columns.items():
        missing_columns = columns - set(tables[filename].columns)
        if missing_columns:
            errors.append(f"{filename}: 누락 컬럼 {sorted(missing_columns)}")

    if errors:
        return errors

    products = tables["products.csv"].copy()
    suppliers = tables["raw_material_suppliers.csv"].copy()
    plants = tables["assembly_locations.csv"].copy()
    transport = tables["transport_parameters.csv"].copy()

    if sorted(products["xpress_product_index"].astype(int).tolist()) != list(range(1, 7)):
        errors.append("products.csv: Xpress 제품 인덱스는 1~6이어야 합니다.")
    if sorted(plants["xpress_location_index"].astype(int).tolist()) != list(range(1, 25)):
        errors.append("assembly_locations.csv: Xpress 위치 인덱스는 1~24여야 합니다.")
    if sorted(transport["xpress_mode_index"].astype(int).tolist()) != [1, 2, 3, 4]:
        errors.append("transport_parameters.csv: Xpress 운송수단 인덱스는 1~4여야 합니다.")

    for material in MATERIALS:
        subset = suppliers[suppliers["material_id"] == material]
        indices = sorted(subset["xpress_location_index"].astype(int).tolist())
        if indices != list(range(1, 25)):
            errors.append(f"raw_material_suppliers.csv: {material}은 24개 위치를 모두 포함해야 합니다.")

    nonbattery_sum = products[["steel_kg", "aluminum_kg", "other_material_kg"]].sum(axis=1)
    if not np.allclose(nonbattery_sum, products["nonbattery_mass_kg"], atol=1e-9):
        errors.append("products.csv: 철강+알루미늄+기타 원자재 질량이 비배터리 질량과 다릅니다.")
    if not np.allclose(
        products["nonbattery_mass_kg"] + products["battery_mass_kg"],
        products["vehicle_mass_kg"],
        atol=1e-9,
    ):
        errors.append("products.csv: 비배터리 질량+배터리 질량이 차량 총질량과 다릅니다.")

    expected_battery = [50, 60, 70, 80, 90, 100]
    if products.sort_values("xpress_product_index")["battery_kwh"].astype(float).tolist() != expected_battery:
        errors.append("products.csv: Xpress 배터리 용량은 50, 60, 70, 80, 90, 100 kWh여야 합니다.")

    return errors


def ordered_tables(tables: Mapping[str, pd.DataFrame]):
    products = tables["products.csv"].sort_values("xpress_product_index").reset_index(drop=True)
    demand = tables["demand.csv"].copy()
    suppliers = tables["raw_material_suppliers.csv"].sort_values(
        ["xpress_material_index", "xpress_location_index"]
    ).reset_index(drop=True)
    plants = tables["assembly_locations.csv"].sort_values("xpress_location_index").reset_index(drop=True)
    transport = tables["transport_parameters.csv"].sort_values("xpress_mode_index").reset_index(drop=True)
    markets = tables["markets.csv"].copy()
    scenarios = tables["scenarios.csv"].copy()
    return products, demand, suppliers, plants, transport, markets, scenarios


def rounded_distance_km(lat1: float, lon1: float, lat2: float, lon2: float, minimum: float = MIN_DISTANCE_KM) -> float:
    distance = geodesic((float(lat1), float(lon1)), (float(lat2), float(lon2))).km
    return round(max(float(minimum), float(distance)), 2)


def carbon_cap_from_score(vehicle_class: str, score: float) -> float:
    score = min(80.0, max(0.0, float(score)))
    if vehicle_class == "small":
        low, high = 6000.0, 17000.0
    else:
        low, high = 12000.0, 21000.0
    return high - score / 80.0 * (high - low)


def subsidy_score(vehicle_class: str, emission_per_vehicle: float) -> float:
    if vehicle_class == "small":
        low, high = 6000.0, 17000.0
    else:
        low, high = 12000.0, 21000.0
    value = float(emission_per_vehicle)
    if value <= low:
        return 80.0
    if value >= high:
        return 0.0
    return 80.0 * (high - value) / (high - low)


# -----------------------------------------------------------------------------
# Sparse LP layout and matrix builder
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class IndexLayout:
    mode: str
    F: int = 6
    R: int = 4
    S: int = 24
    P: int = 24
    T: int = 4

    def __post_init__(self):
        if self.mode not in {"line", "modular"}:
            raise ValueError(self.mode)

    @property
    def off_rp(self) -> int:
        return 0

    @property
    def n_rp(self) -> int:
        return self.F * self.R * self.S

    @property
    def off_rt(self) -> int:
        return self.off_rp + self.n_rp

    @property
    def n_rt(self) -> int:
        return self.F * self.R * self.S * self.P * self.T

    @property
    def off_fp(self) -> int:
        return self.off_rt + self.n_rt

    @property
    def n_fp(self) -> int:
        return self.F * self.P

    @property
    def off_ft(self) -> int:
        return self.off_fp + self.n_fp

    @property
    def n_ft(self) -> int:
        return self.F * self.P * self.T

    @property
    def off_z1(self) -> int:
        return self.off_ft + self.n_ft

    @property
    def n_z1(self) -> int:
        return self.F * self.S * self.P

    @property
    def off_z2(self) -> int:
        return self.off_z1 + self.n_z1

    @property
    def n_z2(self) -> int:
        return self.F * self.S * self.P if self.mode == "modular" else 0

    @property
    def off_alpha(self) -> int:
        return self.off_z2 + self.n_z2

    @property
    def n_alpha(self) -> int:
        return self.n_rt

    @property
    def off_beta(self) -> int:
        return self.off_alpha + self.n_alpha

    @property
    def n_beta(self) -> int:
        return self.n_ft

    @property
    def n_vars(self) -> int:
        return self.off_beta + self.n_beta

    def rp(self, f: int, r: int, s: int) -> int:
        return self.off_rp + ((f * self.R + r) * self.S + s)

    def rt(self, f: int, r: int, s: int, p: int, t: int) -> int:
        return self.off_rt + ((((f * self.R + r) * self.S + s) * self.P + p) * self.T + t)

    def fp(self, f: int, p: int) -> int:
        return self.off_fp + f * self.P + p

    def ft(self, f: int, p: int, t: int) -> int:
        return self.off_ft + (f * self.P + p) * self.T + t

    def z1(self, f: int, s: int, p: int) -> int:
        return self.off_z1 + (f * self.S + s) * self.P + p

    def z2(self, f: int, s: int, p: int) -> int:
        if self.mode != "modular":
            raise ValueError("z2 exists only in modular mode")
        return self.off_z2 + (f * self.S + s) * self.P + p

    def alpha(self, f: int, r: int, s: int, p: int, t: int) -> int:
        return self.off_alpha + ((((f * self.R + r) * self.S + s) * self.P + p) * self.T + t)

    def beta(self, f: int, p: int, t: int) -> int:
        return self.off_beta + (f * self.P + p) * self.T + t


class LinearConstraintBuilder:
    """Compact row-wise storage used to build an OR-Tools MPSolver model.

    Coefficients are stored only until solve time. This avoids a SciPy dependency and
    keeps the Xpress-equivalent LP structure unchanged.
    """

    def __init__(self, n_vars: int):
        self.n_vars = int(n_vars)
        self.eq_cols: List[int] = []
        self.eq_data: List[float] = []
        self.eq_starts: List[int] = [0]
        self.eq_rhs: List[float] = []
        self.ub_cols: List[int] = []
        self.ub_data: List[float] = []
        self.ub_starts: List[int] = [0]
        self.ub_rhs: List[float] = []

    def add_eq(self, cols: Sequence[int], vals: Sequence[float], rhs: float):
        if len(cols) != len(vals):
            raise ValueError("equality columns and coefficients must have the same length")
        self.eq_cols.extend(int(c) for c in cols)
        self.eq_data.extend(float(v) for v in vals)
        self.eq_rhs.append(float(rhs))
        self.eq_starts.append(len(self.eq_cols))

    def add_le(self, cols: Sequence[int], vals: Sequence[float], rhs: float):
        if len(cols) != len(vals):
            raise ValueError("inequality columns and coefficients must have the same length")
        self.ub_cols.extend(int(c) for c in cols)
        self.ub_data.extend(float(v) for v in vals)
        self.ub_rhs.append(float(rhs))
        self.ub_starts.append(len(self.ub_cols))

    @property
    def equality_count(self) -> int:
        return len(self.eq_rhs)

    @property
    def inequality_count(self) -> int:
        return len(self.ub_rhs)

    @property
    def nonzero_count(self) -> int:
        return len(self.eq_data) + len(self.ub_data)

    def clear_coefficients(self) -> None:
        self.eq_cols.clear()
        self.eq_data.clear()
        self.eq_starts[:] = [0]
        self.eq_rhs.clear()
        self.ub_cols.clear()
        self.ub_data.clear()
        self.ub_starts[:] = [0]
        self.ub_rhs.clear()


@dataclass
class LPModel:
    layout: IndexLayout
    c: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    rows: LinearConstraintBuilder
    equality_count: int
    inequality_count: int
    matrix_nonzeros: int
    products: pd.DataFrame
    demand_values: np.ndarray
    suppliers: pd.DataFrame
    plants: pd.DataFrame
    transport: pd.DataFrame
    scenario: pd.Series
    raw_distances: np.ndarray
    final_distances: np.ndarray
    emission_rp: np.ndarray
    emission_rt: np.ndarray
    emission_fp: np.ndarray
    emission_ft: np.ndarray
    cap_application: str


@dataclass
class SolveResult:
    status: str
    message: str
    objective_value: Optional[float]
    wall_time_sec: float
    x: Optional[np.ndarray]
    model: LPModel
    ortools_status: int
    solver_name: str
    solver_version: str
    iterations: int


def build_xpress_relaxation_model(
    tables: Mapping[str, pd.DataFrame],
    production_mode: str,
    scenario_id: str,
    cap_application: str = "product_strict",
    score_override: Optional[float] = None,
) -> LPModel:
    products, demand, suppliers, plants, transport, markets, scenarios = ordered_tables(tables)
    layout = IndexLayout(production_mode)

    product_ids = products["product_id"].tolist()
    demand_map = demand.groupby("product_id")["demand_units"].sum().to_dict()
    demand_values = np.asarray([float(demand_map[p]) for p in product_ids], dtype=float)

    scenario_rows = scenarios[scenarios["scenario_id"] == scenario_id]
    if scenario_rows.empty:
        raise ValueError(f"scenario not found: {scenario_id}")
    scenario = scenario_rows.iloc[0].copy()
    if score_override is not None:
        score = min(80.0, max(0.0, float(score_override)))
        scenario["minimum_score"] = score
        scenario["apply_carbon_cap"] = 1
        scenario["small_cap_kgco2_per_vehicle"] = carbon_cap_from_score("small", score)
        scenario["standard_cap_kgco2_per_vehicle"] = carbon_cap_from_score("standard", score)
        scenario["scenario_name"] = f"보조금 점수 {score:g}점 민감도"

    supplier_cost = np.zeros((layout.R, layout.S), dtype=float)
    supplier_ef = np.zeros((layout.R, layout.S), dtype=float)
    supplier_capacity = np.zeros((layout.R, layout.S), dtype=float)
    for _, row in suppliers.iterrows():
        r = int(row["xpress_material_index"]) - 1
        s = int(row["xpress_location_index"]) - 1
        supplier_cost[r, s] = float(row["production_cost"])
        supplier_ef[r, s] = float(row["production_ef"])
        supplier_capacity[r, s] = float(row["capacity"])

    mode_cost = np.zeros(layout.T, dtype=float)
    mode_ef = np.zeros(layout.T, dtype=float)
    for _, row in transport.iterrows():
        t = int(row["xpress_mode_index"]) - 1
        mode_cost[t] = float(row["transport_cost_eur_per_unitkm"])
        mode_ef[t] = float(row["transport_ef_kgco2_per_unitkm"])

    raw_distances = np.zeros((layout.S, layout.P), dtype=float)
    for s in range(layout.S):
        for p in range(layout.P):
            raw_distances[s, p] = rounded_distance_km(
                plants.iloc[s]["latitude"], plants.iloc[s]["longitude"],
                plants.iloc[p]["latitude"], plants.iloc[p]["longitude"],
            )

    market = markets.iloc[0]
    minimum_market_distance = float(market.get("minimum_distance_km", MIN_DISTANCE_KM))
    final_distances = np.zeros(layout.P, dtype=float)
    for p in range(layout.P):
        final_distances[p] = rounded_distance_km(
            plants.iloc[p]["latitude"], plants.iloc[p]["longitude"],
            market["latitude"], market["longitude"], minimum_market_distance,
        )

    c = np.zeros(layout.n_vars, dtype=float)
    lb = np.zeros(layout.n_vars, dtype=float)
    ub = np.full(layout.n_vars, np.inf, dtype=float)
    ub[layout.off_alpha:layout.off_alpha + layout.n_alpha] = 1.0
    ub[layout.off_beta:layout.off_beta + layout.n_beta] = 1.0

    emission_rp = np.zeros(layout.n_rp, dtype=float)
    emission_rt = np.zeros(layout.n_rt, dtype=float)
    emission_fp = np.zeros(layout.n_fp, dtype=float)
    emission_ft = np.zeros(layout.n_ft, dtype=float)

    # Objective and carbon coefficients
    for f in range(layout.F):
        product = products.iloc[f]
        for r in range(layout.R):
            loss_multiplier = 1.0 / (1.0 - MATERIAL_LOSS_RATE) if r in {0, 1} else 1.0
            for s in range(layout.S):
                rp_idx = layout.rp(f, r, s)
                c[rp_idx] = supplier_cost[r, s]
                emission_rp[rp_idx - layout.off_rp] = supplier_ef[r, s] * loss_multiplier
                for p in range(layout.P):
                    distance = raw_distances[s, p]
                    for t in range(layout.T):
                        rt_idx = layout.rt(f, r, s, p, t)
                        c[rt_idx] = distance * mode_cost[t]
                        emission_rt[rt_idx - layout.off_rt] = distance * mode_ef[t]

        for p in range(layout.P):
            fp_idx = layout.fp(f, p)
            c[fp_idx] = float(product["nonbattery_mass_kg"]) * float(plants.iloc[p]["assembly_cost_eur_per_kg"])
            emission_fp[fp_idx - layout.off_fp] = (
                float(product["nonbattery_mass_kg"]) * float(plants.iloc[p]["assembly_ef_kgco2_per_kg"])
            )
            for t in range(layout.T):
                ft_idx = layout.ft(f, p, t)
                c[ft_idx] = (
                    float(product["vehicle_mass_kg"]) * final_distances[p] * mode_cost[t]
                )
                emission_ft[ft_idx - layout.off_ft] = (
                    float(product["vehicle_mass_kg"]) * final_distances[p] * mode_ef[t]
                )

    rows = LinearConstraintBuilder(layout.n_vars)

    # 1) Xpress transport-mode simplex: sum_t alpha = 1, sum_t beta = 1.
    for f in range(layout.F):
        for r in range(layout.R):
            for s in range(layout.S):
                for p in range(layout.P):
                    rows.add_eq(
                        [layout.alpha(f, r, s, p, t) for t in range(layout.T)],
                        [1.0] * layout.T,
                        1.0,
                    )
    for f in range(layout.F):
        for p in range(layout.P):
            rows.add_eq(
                [layout.beta(f, p, t) for t in range(layout.T)],
                [1.0] * layout.T,
                1.0,
            )

    # 2) Xpress complementary Big-M links. Variables are continuous in this LP relaxation.
    for f in range(layout.F):
        for r in range(layout.R):
            for s in range(layout.S):
                for p in range(layout.P):
                    for t in range(layout.T):
                        cols = [layout.rt(f, r, s, p, tau) for tau in range(layout.T) if tau != t]
                        vals = [1.0] * (layout.T - 1)
                        cols.append(layout.alpha(f, r, s, p, t))
                        vals.append(BIG_M)
                        rows.add_le(cols, vals, BIG_M)
    for f in range(layout.F):
        for p in range(layout.P):
            for t in range(layout.T):
                cols = [layout.ft(f, p, tau) for tau in range(layout.T) if tau != t]
                vals = [1.0] * (layout.T - 1)
                cols.append(layout.beta(f, p, t))
                vals.append(BIG_M)
                rows.add_le(cols, vals, BIG_M)

    # 3) Demand fulfillment.
    for f in range(layout.F):
        cols = [layout.ft(f, p, t) for p in range(layout.P) for t in range(layout.T)]
        rows.add_eq(cols, [1.0] * len(cols), demand_values[f])

    # 4) Battery production structure.
    if production_mode == "modular":
        # Exact uploaded modular LP structure: route kWh = 10*ZM + 5*ZS.
        for f in range(layout.F):
            for s in range(layout.S):
                for p in range(layout.P):
                    cols = [layout.rt(f, 3, s, p, t) for t in range(layout.T)]
                    vals = [1.0] * layout.T
                    cols.extend([layout.z1(f, s, p), layout.z2(f, s, p)])
                    vals.extend([-10.0, -5.0])
                    rows.add_eq(cols, vals, 0.0)
        # Product/plant battery requirement from the Xpress balance coefficients.
        for f in range(layout.F):
            battery_kwh = float(products.iloc[f]["battery_kwh"])
            for p in range(layout.P):
                cols = [layout.rt(f, 3, s, p, t) for s in range(layout.S) for t in range(layout.T)]
                vals = [1.0] * (layout.S * layout.T)
                cols.append(layout.fp(f, p))
                vals.append(-battery_kwh)
                rows.add_eq(cols, vals, 0.0)
    else:
        # Line-mode reconstruction from the Word/Xpress variable definition:
        # route kWh = pack kWh * ZL, and total pack-equivalent flow = FP.
        for f in range(layout.F):
            battery_kwh = float(products.iloc[f]["battery_kwh"])
            for s in range(layout.S):
                for p in range(layout.P):
                    cols = [layout.rt(f, 3, s, p, t) for t in range(layout.T)]
                    vals = [1.0] * layout.T
                    cols.append(layout.z1(f, s, p))
                    vals.append(-battery_kwh)
                    rows.add_eq(cols, vals, 0.0)
            for p in range(layout.P):
                cols = [layout.z1(f, s, p) for s in range(layout.S)] + [layout.fp(f, p)]
                vals = [1.0] * layout.S + [-1.0]
                rows.add_eq(cols, vals, 0.0)

    # 5) Steel, aluminum, and other-material balances at each assembly location.
    material_columns = ["steel_kg", "aluminum_kg", "other_material_kg"]
    for f in range(layout.F):
        for r, col in enumerate(material_columns):
            required_per_vehicle = float(products.iloc[f][col])
            for p in range(layout.P):
                cols = [layout.rt(f, r, s, p, t) for s in range(layout.S) for t in range(layout.T)]
                vals = [1.0] * (layout.S * layout.T)
                cols.append(layout.fp(f, p))
                vals.append(-required_per_vehicle)
                rows.add_eq(cols, vals, 0.0)

    # 6) Assembly quantity equals finished-vehicle shipment.
    for f in range(layout.F):
        for p in range(layout.P):
            cols = [layout.fp(f, p)] + [layout.ft(f, p, t) for t in range(layout.T)]
            vals = [1.0] + [-1.0] * layout.T
            rows.add_eq(cols, vals, 0.0)

    # 7) Supplier production equals all supplier-to-plant shipments.
    for f in range(layout.F):
        for r in range(layout.R):
            for s in range(layout.S):
                cols = [layout.rp(f, r, s)] + [
                    layout.rt(f, r, s, p, t) for p in range(layout.P) for t in range(layout.T)
                ]
                vals = [1.0] + [-1.0] * (layout.P * layout.T)
                rows.add_eq(cols, vals, 0.0)

    # 8) Xpress supplier capacity: each material-location pair <= 1,000,000.
    for r in range(layout.R):
        for s in range(layout.S):
            cols = [layout.rp(f, r, s) for f in range(layout.F)]
            rows.add_le(cols, [1.0] * layout.F, supplier_capacity[r, s])

    # 9) Carbon caps for S1/S3. S2 remains the exact cost-only uploaded Xpress structure.
    if int(scenario["apply_carbon_cap"]) == 1:
        product_rows: List[Tuple[List[int], List[float], float, str]] = []
        for f in range(layout.F):
            cols: List[int] = []
            vals: List[float] = []
            # RP production emissions
            start = f * layout.R * layout.S
            for local in range(layout.R * layout.S):
                idx = layout.off_rp + start + local
                coef = emission_rp[idx - layout.off_rp]
                if coef:
                    cols.append(idx); vals.append(coef)
            # RT transport emissions
            rt_start = f * layout.R * layout.S * layout.P * layout.T
            for local in range(layout.R * layout.S * layout.P * layout.T):
                idx = layout.off_rt + rt_start + local
                coef = emission_rt[idx - layout.off_rt]
                if coef:
                    cols.append(idx); vals.append(coef)
            # FP assembly emissions
            for p in range(layout.P):
                idx = layout.fp(f, p)
                cols.append(idx); vals.append(emission_fp[idx - layout.off_fp])
            # FT finished-transport emissions
            for p in range(layout.P):
                for t in range(layout.T):
                    idx = layout.ft(f, p, t)
                    cols.append(idx); vals.append(emission_ft[idx - layout.off_ft])
            vehicle_class = str(products.iloc[f]["vehicle_class"])
            cap = float(
                scenario["small_cap_kgco2_per_vehicle"]
                if vehicle_class == "small"
                else scenario["standard_cap_kgco2_per_vehicle"]
            )
            product_rows.append((cols, vals, cap * demand_values[f], vehicle_class))

        if cap_application == "product_strict":
            for cols, vals, rhs, _ in product_rows:
                rows.add_le(cols, vals, rhs)
        elif cap_application == "class_average":
            for vehicle_class in ["small", "standard"]:
                merged: Dict[int, float] = {}
                rhs = 0.0
                for cols, vals, product_rhs, cls in product_rows:
                    if cls != vehicle_class:
                        continue
                    rhs += product_rhs
                    for col, val in zip(cols, vals):
                        merged[col] = merged.get(col, 0.0) + val
                rows.add_le(list(merged.keys()), list(merged.values()), rhs)
        else:
            raise ValueError(f"unknown cap application: {cap_application}")

    return LPModel(
        layout=layout,
        c=c,
        lb=lb,
        ub=ub,
        rows=rows,
        equality_count=rows.equality_count,
        inequality_count=rows.inequality_count,
        matrix_nonzeros=rows.nonzero_count,
        products=products,
        demand_values=demand_values,
        suppliers=suppliers,
        plants=plants,
        transport=transport,
        scenario=scenario,
        raw_distances=raw_distances,
        final_distances=final_distances,
        emission_rp=emission_rp,
        emission_rt=emission_rt,
        emission_fp=emission_fp,
        emission_ft=emission_ft,
        cap_application=cap_application,
    )


def _create_ortools_lp_solver() -> Tuple[pywraplp.Solver, str]:
    """Create the same continuous LP with an OR-Tools backend.

    GLOP is the primary backend. CLP and PDLP are fallbacks for environments
    where a particular backend is unavailable.
    """
    for solver_name in ("GLOP", "CLP", "PDLP"):
        solver = pywraplp.Solver.CreateSolver(solver_name)
        if solver is not None:
            return solver, solver_name
    raise RuntimeError("OR-Tools LP solver is unavailable. Install the 'ortools' package.")


def _set_solver_time_limit(solver: pywraplp.Solver, time_limit_sec: int) -> None:
    milliseconds = max(1, int(float(time_limit_sec) * 1000.0))
    if hasattr(solver, "SetTimeLimit"):
        solver.SetTimeLimit(milliseconds)
    else:
        solver.set_time_limit(milliseconds)


def solve_lp_model(model: LPModel, time_limit_sec: int = 180) -> SolveResult:
    started = time.perf_counter()
    solver, solver_name = _create_ortools_lp_solver()
    _set_solver_time_limit(solver, time_limit_sec)
    try:
        solver.SetNumThreads(1)
    except Exception:
        pass

    infinity = solver.infinity()
    variables = []
    for i in range(model.layout.n_vars):
        lower = float(model.lb[i])
        upper = float(model.ub[i]) if np.isfinite(model.ub[i]) else infinity
        variables.append(solver.NumVar(lower, upper, ""))

    objective = solver.Objective()
    for idx in np.flatnonzero(model.c):
        objective.SetCoefficient(variables[int(idx)], float(model.c[int(idx)]))
    objective.SetMinimization()

    rows = model.rows
    for row_index, rhs in enumerate(rows.eq_rhs):
        constraint = solver.Constraint(float(rhs), float(rhs), "")
        start = rows.eq_starts[row_index]
        stop = rows.eq_starts[row_index + 1]
        for position in range(start, stop):
            constraint.SetCoefficient(
                variables[rows.eq_cols[position]], rows.eq_data[position]
            )

    for row_index, rhs in enumerate(rows.ub_rhs):
        constraint = solver.Constraint(-infinity, float(rhs), "")
        start = rows.ub_starts[row_index]
        stop = rows.ub_starts[row_index + 1]
        for position in range(start, stop):
            constraint.SetCoefficient(
                variables[rows.ub_cols[position]], rows.ub_data[position]
            )

    # OR-Tools now owns the coefficient matrix. Release Python-side coefficient lists
    # before Solve() to lower peak memory on Streamlit Community Cloud.
    rows.clear_coefficients()

    status_code = int(solver.Solve())
    wall = time.perf_counter() - started
    status_map = {
        int(pywraplp.Solver.OPTIMAL): "OPTIMAL",
        int(pywraplp.Solver.FEASIBLE): "FEASIBLE",
        int(pywraplp.Solver.INFEASIBLE): "INFEASIBLE",
        int(pywraplp.Solver.UNBOUNDED): "UNBOUNDED",
        int(pywraplp.Solver.ABNORMAL): "ABNORMAL",
        int(getattr(pywraplp.Solver, "MODEL_INVALID", 5)): "MODEL_INVALID",
        int(pywraplp.Solver.NOT_SOLVED): "NOT_SOLVED",
    }
    status = status_map.get(status_code, f"STATUS_{status_code}")
    has_solution = status_code in {
        int(pywraplp.Solver.OPTIMAL),
        int(pywraplp.Solver.FEASIBLE),
    }

    solution = None
    objective_value = None
    if has_solution:
        solution = np.fromiter(
            (variable.solution_value() for variable in variables),
            dtype=float,
            count=len(variables),
        )
        objective_value = float(objective.Value())

    try:
        iterations = int(solver.iterations())
    except Exception:
        iterations = 0
    try:
        solver_version = str(solver.SolverVersion())
    except Exception:
        solver_version = solver_name

    message = (
        f"{status} with OR-Tools {solver_version}; "
        f"variables={solver.NumVariables():,}, constraints={solver.NumConstraints():,}"
    )
    return SolveResult(
        status=status,
        message=message,
        objective_value=objective_value,
        wall_time_sec=wall,
        x=solution,
        model=model,
        ortools_status=status_code,
        solver_name=solver_name,
        solver_version=solver_version,
        iterations=iterations,
    )


# -----------------------------------------------------------------------------
# Result extraction and poster-style summaries
# -----------------------------------------------------------------------------
def _material_supplier_row(model: LPModel, r: int, s: int) -> pd.Series:
    return model.suppliers[
        (model.suppliers["xpress_material_index"].astype(int) == r + 1)
        & (model.suppliers["xpress_location_index"].astype(int) == s + 1)
    ].iloc[0]


def assign_flow_quartiles(df: pd.DataFrame, flow_col: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["quartile"] = pd.Series(dtype=str)
        return out
    ranks = out[flow_col].rank(method="first")
    q = min(4, len(out))
    labels = QUARTILE_LABELS[-q:]
    out["quartile"] = pd.qcut(ranks, q=q, labels=labels).astype(str)
    return out


def extract_solution(result: SolveResult) -> Dict:
    if result.x is None or result.status not in {"OPTIMAL", "FEASIBLE"}:
        return {
            "status": result.status,
            "message": result.message,
            "wall_time_sec": result.wall_time_sec,
            "objective_value": result.objective_value,
            "model": result.model,
        }

    model = result.model
    layout = model.layout
    x = result.x
    products = model.products
    plants = model.plants
    transport = model.transport.set_index("xpress_mode_index")

    rp = x[layout.off_rp:layout.off_rt].reshape(layout.F, layout.R, layout.S)
    rt = x[layout.off_rt:layout.off_fp].reshape(layout.F, layout.R, layout.S, layout.P, layout.T)
    fp = x[layout.off_fp:layout.off_ft].reshape(layout.F, layout.P)
    ft = x[layout.off_ft:layout.off_z1].reshape(layout.F, layout.P, layout.T)
    alpha = x[layout.off_alpha:layout.off_beta].reshape(layout.F, layout.R, layout.S, layout.P, layout.T)
    beta = x[layout.off_beta:].reshape(layout.F, layout.P, layout.T)

    supplier_records: List[Dict] = []
    for f, r, s in np.argwhere(rp > FLOW_TOL):
        product = products.iloc[f]
        supplier = _material_supplier_row(model, int(r), int(s))
        amount = float(rp[f, r, s])
        supplier_records.append({
            "product_id": product["product_id"],
            "product_name": product["product_name_ko"],
            "material_id": MATERIALS[r],
            "material_name": MATERIAL_LABEL[MATERIALS[r]],
            "supplier_index": s + 1,
            "supplier_location": supplier["location_name"],
            "production_amount": amount,
            "unit": "kWh" if r == 3 else "kg",
            "production_cost_eur": amount * float(supplier["production_cost"]),
            "production_emissions_kgco2": amount * float(supplier["production_ef"]) * (
                1.0 / (1.0 - MATERIAL_LOSS_RATE) if r in {0, 1} else 1.0
            ),
        })
    supplier_df = pd.DataFrame(supplier_records)

    raw_records: List[Dict] = []
    for f, r, s, p, t in np.argwhere(rt > FLOW_TOL):
        product = products.iloc[f]
        supplier = _material_supplier_row(model, int(r), int(s))
        plant = plants.iloc[p]
        mode = transport.loc[t + 1]
        amount = float(rt[f, r, s, p, t])
        distance = float(model.raw_distances[s, p])
        raw_records.append({
            "product_id": product["product_id"],
            "product_name": product["product_name_ko"],
            "material_id": MATERIALS[r],
            "material_name": MATERIAL_LABEL[MATERIALS[r]],
            "supplier_index": s + 1,
            "supplier_location": supplier["location_name"],
            "plant_index": p + 1,
            "plant_location": plant["location_name"],
            "transport_mode_index": t + 1,
            "transport_mode": mode["transport_mode"],
            "transport_mode_ko": mode["transport_mode_ko"],
            "flow_amount": amount,
            "flow_unit": "kWh-equivalent" if r == 3 else "kg",
            "distance_km": distance,
            "transport_cost_eur": amount * distance * float(mode["transport_cost_eur_per_unitkm"]),
            "transport_emissions_kgco2": amount * distance * float(mode["transport_ef_kgco2_per_unitkm"]),
            "alpha_relaxed": float(alpha[f, r, s, p, t]),
        })
    raw_df = pd.DataFrame(raw_records)

    plant_records: List[Dict] = []
    for f, p in np.argwhere(fp > FLOW_TOL):
        product = products.iloc[f]
        plant = plants.iloc[p]
        units = float(fp[f, p])
        plant_records.append({
            "product_id": product["product_id"],
            "product_name": product["product_name_ko"],
            "plant_index": p + 1,
            "plant_location": plant["location_name"],
            "assembled_vehicle_equivalents": units,
            "assembly_mass_kg": units * float(product["nonbattery_mass_kg"]),
            "assembly_cost_eur": units * float(product["nonbattery_mass_kg"]) * float(plant["assembly_cost_eur_per_kg"]),
            "assembly_emissions_kgco2": units * float(product["nonbattery_mass_kg"]) * float(plant["assembly_ef_kgco2_per_kg"]),
        })
    plant_df = pd.DataFrame(plant_records)

    final_records: List[Dict] = []
    for f, p, t in np.argwhere(ft > FLOW_TOL):
        product = products.iloc[f]
        plant = plants.iloc[p]
        mode = transport.loc[t + 1]
        units = float(ft[f, p, t])
        distance = float(model.final_distances[p])
        final_records.append({
            "product_id": product["product_id"],
            "product_name": product["product_name_ko"],
            "plant_index": p + 1,
            "plant_location": plant["location_name"],
            "market_name": "프랑스 시장",
            "transport_mode_index": t + 1,
            "transport_mode": mode["transport_mode"],
            "transport_mode_ko": mode["transport_mode_ko"],
            "vehicle_equivalents": units,
            "transport_mass_kg": units * float(product["vehicle_mass_kg"]),
            "distance_km": distance,
            "transport_cost_eur": units * float(product["vehicle_mass_kg"]) * distance * float(mode["transport_cost_eur_per_unitkm"]),
            "transport_emissions_kgco2": units * float(product["vehicle_mass_kg"]) * distance * float(mode["transport_ef_kgco2_per_unitkm"]),
            "beta_relaxed": float(beta[f, p, t]),
        })
    final_df = pd.DataFrame(final_records)

    production_cost = float(model.c[layout.off_rp:layout.off_rt] @ rp.ravel())
    raw_transport_cost = float(model.c[layout.off_rt:layout.off_fp] @ rt.ravel())
    assembly_cost = float(model.c[layout.off_fp:layout.off_ft] @ fp.ravel())
    finished_transport_cost = float(model.c[layout.off_ft:layout.off_z1] @ ft.ravel())

    production_emissions = float(model.emission_rp @ rp.ravel())
    raw_transport_emissions = float(model.emission_rt @ rt.ravel())
    assembly_emissions = float(model.emission_fp @ fp.ravel())
    final_transport_emissions = float(model.emission_ft @ ft.ravel())

    product_rows: List[Dict] = []
    for f in range(layout.F):
        product = products.iloc[f]
        production_e = float(model.emission_rp[f * layout.R * layout.S:(f + 1) * layout.R * layout.S] @ rp[f].ravel())
        raw_e = float(
            model.emission_rt[
                f * layout.R * layout.S * layout.P * layout.T:(f + 1) * layout.R * layout.S * layout.P * layout.T
            ] @ rt[f].ravel()
        )
        assy_e = float(model.emission_fp[f * layout.P:(f + 1) * layout.P] @ fp[f].ravel())
        final_e = float(model.emission_ft[f * layout.P * layout.T:(f + 1) * layout.P * layout.T] @ ft[f].ravel())
        total_e = production_e + raw_e + assy_e + final_e
        demand_units = float(model.demand_values[f])
        per_vehicle = total_e / demand_units
        scenario = model.scenario
        cap = np.nan
        if int(scenario["apply_carbon_cap"]) == 1:
            cap = float(
                scenario["small_cap_kgco2_per_vehicle"]
                if product["vehicle_class"] == "small"
                else scenario["standard_cap_kgco2_per_vehicle"]
            )
        product_rows.append({
            "product_id": product["product_id"],
            "product_name": product["product_name_ko"],
            "vehicle_class": product["vehicle_class"],
            "demand_units": demand_units,
            "total_emissions_kgco2": total_e,
            "emissions_per_vehicle_kgco2": per_vehicle,
            "subsidy_score": subsidy_score(str(product["vehicle_class"]), per_vehicle),
            "carbon_cap_kgco2_per_vehicle": cap,
            "cap_satisfied": bool(np.isnan(cap) or per_vehicle <= cap + 1e-5),
        })
    product_df = pd.DataFrame(product_rows)

    # Poster-style route quartiles are computed from positive part-supply route quantities.
    if raw_df.empty:
        route_agg = pd.DataFrame()
        quartile_df = pd.DataFrame({"quartile": QUARTILE_LABELS, "share_pct": [0.0] * 4})
    else:
        route_agg = raw_df.groupby(
            ["material_id", "material_name", "supplier_index", "supplier_location", "plant_index", "plant_location",
             "transport_mode_index", "transport_mode", "transport_mode_ko"],
            as_index=False,
        ).agg(
            flow_amount=("flow_amount", "sum"),
            distance_km=("distance_km", "first"),
            transport_cost_eur=("transport_cost_eur", "sum"),
            transport_emissions_kgco2=("transport_emissions_kgco2", "sum"),
        )
        route_agg = assign_flow_quartiles(route_agg, "flow_amount")
        total_flow = float(route_agg["flow_amount"].sum())
        quartile_df = (
            route_agg.groupby("quartile", as_index=False)["flow_amount"].sum()
            .assign(share_pct=lambda d: 100.0 * d["flow_amount"] / total_flow)
        )
        quartile_df = pd.DataFrame({"quartile": QUARTILE_LABELS}).merge(quartile_df, on="quartile", how="left").fillna(0.0)

    return {
        "status": result.status,
        "message": result.message,
        "scenario_id": str(model.scenario["scenario_id"]),
        "scenario_name": str(model.scenario["scenario_name"]),
        "production_mode": layout.mode,
        "production_mode_name": MODE_LABEL[layout.mode],
        "objective_value": float(result.objective_value),
        "wall_time_sec": result.wall_time_sec,
        "variable_count": layout.n_vars,
        "constraint_count": int(model.equality_count + model.inequality_count),
        "equality_count": int(model.equality_count),
        "inequality_count": int(model.inequality_count),
        "matrix_nonzeros": int(model.matrix_nonzeros),
        "solver_name": result.solver_name,
        "solver_version": result.solver_version,
        "solver_iterations": result.iterations,
        "continuous_variable_count": layout.n_vars,
        "integer_variable_count": 0,
        "binary_variable_count": 0,
        "cost_breakdown": {
            "원자재·배터리 생산비": production_cost,
            "부품 운송비": raw_transport_cost,
            "가공·조립비": assembly_cost,
            "완제품 운송비": finished_transport_cost,
        },
        "emission_breakdown": {
            "원자재·배터리 생산": production_emissions,
            "부품 운송": raw_transport_emissions,
            "가공·조립": assembly_emissions,
            "완제품 운송": final_transport_emissions,
        },
        "total_emissions_kgco2": production_emissions + raw_transport_emissions + assembly_emissions + final_transport_emissions,
        "supplier_summary": supplier_df,
        "raw_routes": raw_df,
        "route_aggregated": route_agg,
        "plant_summary": plant_df,
        "finished_routes": final_df,
        "product_summary": product_df,
        "quartile_summary": quartile_df,
        "plants": model.plants.copy(),
    }


def solve_case(
    tables: Mapping[str, pd.DataFrame],
    scenario_id: str,
    production_mode: str,
    cap_application: str,
    time_limit_sec: int,
    score_override: Optional[float] = None,
) -> Dict:
    model = build_xpress_relaxation_model(
        tables,
        production_mode=production_mode,
        scenario_id=scenario_id,
        cap_application=cap_application,
        score_override=score_override,
    )
    result = solve_lp_model(model, time_limit_sec=time_limit_sec)
    return extract_solution(result)


def results_zip(results: Mapping[Tuple[str, str], Dict]) -> bytes:
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as zf:
        summary_rows = []
        for (scenario_id, mode), result in results.items():
            summary_rows.append({
                "scenario_id": scenario_id,
                "production_mode": mode,
                "status": result.get("status"),
                "objective_value": result.get("objective_value"),
                "total_emissions_kgco2": result.get("total_emissions_kgco2"),
                "wall_time_sec": result.get("wall_time_sec"),
            })
            for key, filename in [
                ("product_summary", "product_summary.csv"),
                ("supplier_summary", "supplier_summary.csv"),
                ("raw_routes", "raw_routes.csv"),
                ("plant_summary", "plant_summary.csv"),
                ("finished_routes", "finished_routes.csv"),
                ("quartile_summary", "quartile_summary.csv"),
            ]:
                df = result.get(key)
                if isinstance(df, pd.DataFrame):
                    zf.writestr(
                        f"{scenario_id}_{mode}/{filename}",
                        df.to_csv(index=False).encode("utf-8-sig"),
                    )
        zf.writestr("all_case_summary.csv", pd.DataFrame(summary_rows).to_csv(index=False).encode("utf-8-sig"))
    return memory.getvalue()


# -----------------------------------------------------------------------------
# Mapping and charts
# -----------------------------------------------------------------------------
def build_supply_map(result: Dict, height: int = 480):
    plants = result["plants"]
    route_df = result.get("route_aggregated", pd.DataFrame()).copy()
    final_df = result.get("finished_routes", pd.DataFrame()).copy()

    supply_map = folium.Map(location=[35, 25], zoom_start=2, tiles="CartoDB positron")
    Fullscreen(position="topleft").add_to(supply_map)

    used_supplier_indices = set(route_df["supplier_index"].astype(int).tolist()) if not route_df.empty else set()
    used_plant_indices = set(route_df["plant_index"].astype(int).tolist()) if not route_df.empty else set()
    if not final_df.empty:
        used_plant_indices.update(final_df["plant_index"].astype(int).tolist())

    for idx in sorted(used_supplier_indices):
        row = plants.iloc[idx - 1]
        folium.CircleMarker(
            [row["latitude"], row["longitude"]],
            radius=5,
            color="#6baed6",
            fill=True,
            fill_opacity=0.9,
            tooltip=f"부품 생산 위치 {idx}: {row['location_name']}",
        ).add_to(supply_map)
    for idx in sorted(used_plant_indices):
        row = plants.iloc[idx - 1]
        folium.CircleMarker(
            [row["latitude"], row["longitude"]],
            radius=6,
            color="#31a354",
            fill=True,
            fill_opacity=0.9,
            tooltip=f"조립 위치 {idx}: {row['location_name']}",
        ).add_to(supply_map)

    market_lat, market_lon = 46.2276, 2.2137
    folium.Marker(
        [market_lat, market_lon],
        tooltip="프랑스 시장",
        icon=folium.Icon(color="orange", icon="shopping-cart", prefix="fa"),
    ).add_to(supply_map)

    width_by_quartile = {"Q1": 1.5, "Q2": 2.5, "Q3": 4.0, "Q4": 6.0}
    if not route_df.empty:
        for _, row in route_df.iterrows():
            s = plants.iloc[int(row["supplier_index"]) - 1]
            p = plants.iloc[int(row["plant_index"]) - 1]
            material = str(row["material_id"])
            t = int(row["transport_mode_index"])
            folium.PolyLine(
                [(s["latitude"], s["longitude"]), (p["latitude"], p["longitude"])],
                color=MATERIAL_COLOR.get(material, "#555555"),
                weight=width_by_quartile.get(str(row["quartile"]), 2.0),
                opacity=0.78,
                dash_array=TRANSPORT_DASH.get(t),
                tooltip=(
                    f"{row['material_name']} | {row['supplier_location']} → {row['plant_location']} | "
                    f"{row['transport_mode_ko']} | {row['flow_amount']:,.1f} | {row['quartile']}"
                ),
            ).add_to(supply_map)

    if not final_df.empty:
        final_agg = final_df.groupby(
            ["plant_index", "plant_location", "transport_mode_index", "transport_mode_ko"], as_index=False
        )["vehicle_equivalents"].sum()
        max_units = max(float(final_agg["vehicle_equivalents"].max()), 1.0)
        for _, row in final_agg.iterrows():
            p = plants.iloc[int(row["plant_index"]) - 1]
            weight = 1.5 + 4.5 * float(row["vehicle_equivalents"]) / max_units
            folium.PolyLine(
                [(p["latitude"], p["longitude"]), (market_lat, market_lon)],
                color=MATERIAL_COLOR["finished"],
                weight=weight,
                opacity=0.65,
                dash_array="6,5",
                tooltip=(
                    f"완제품 | {row['plant_location']} → 프랑스 | {row['transport_mode_ko']} | "
                    f"{row['vehicle_equivalents']:,.1f}대 등가량"
                ),
            ).add_to(supply_map)

    legend = """
    <div style="position: fixed; bottom: 24px; left: 24px; z-index:9999; background:white;
                border:1px solid #777; border-radius:6px; padding:8px 10px; font-size:12px;">
      <b>부품 공급망</b><br>
      <span style="color:#e41a1c">━</span> 철강 &nbsp;
      <span style="color:#ff9f1c">━</span> 알루미늄<br>
      <span style="color:#238b45">━</span> 기타 원자재 &nbsp;
      <span style="color:#2171b5">━</span> 배터리<br>
      <span style="color:#6a3d9a">┄</span> 완제품<br>
      선 굵기: Q1 &lt; Q2 &lt; Q3 &lt; Q4
    </div>
    """
    supply_map.get_root().html.add_child(folium.Element(legend))
    return supply_map


def cost_ratio_dataframe(results: Mapping[Tuple[str, str], Dict]) -> pd.DataFrame:
    rows = []
    for scenario_id in ["S1", "S2", "S3"]:
        line = results.get((scenario_id, "line"), {})
        modular = results.get((scenario_id, "modular"), {})
        if line.get("objective_value") and modular.get("objective_value"):
            rows.append({
                "scenario_id": scenario_id,
                "scenario_name": SCENARIO_SHORT[scenario_id],
                "line_cost": line["objective_value"],
                "modular_cost": modular["objective_value"],
                "modular_to_line_cost_ratio": modular["objective_value"] / line["objective_value"],
            })
    return pd.DataFrame(rows)


def quartile_comparison_table(results: Mapping[Tuple[str, str], Dict], scenario_id: str) -> pd.DataFrame:
    out = pd.DataFrame({"구간": QUARTILE_LABELS})
    for mode in ["line", "modular"]:
        result = results.get((scenario_id, mode))
        if not result or not isinstance(result.get("quartile_summary"), pd.DataFrame):
            out[MODE_LABEL[mode]] = np.nan
            continue
        q = result["quartile_summary"][["quartile", "share_pct"]].rename(
            columns={"quartile": "구간", "share_pct": MODE_LABEL[mode]}
        )
        out = out.merge(q, on="구간", how="left")
    return out


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def render_result_map(result: Dict, key: str, height: int = 430):
    if result.get("status") not in {"OPTIMAL", "FEASIBLE"}:
        st.warning(f"{result.get('status')}: {result.get('message')}")
        return
    supply_map = build_supply_map(result, height=height)
    st_folium(supply_map, width=None, height=height, key=key)


def render_solver_metrics(result: Dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총비용", f"€{result.get('objective_value', 0):,.0f}")
    c2.metric("총 탄소배출량", f"{result.get('total_emissions_kgco2', 0):,.0f} kg CO₂-eq")
    c3.metric("계산시간", f"{result.get('wall_time_sec', 0):.2f}초")
    c4.metric("양의 부품 경로", f"{len(result.get('route_aggregated', [])):,}")


def render_poster_scenario(results: Mapping[Tuple[str, str], Dict], scenario_id: str):
    st.markdown(f"### {SCENARIO_SHORT[scenario_id]} 결과")
    line = results.get((scenario_id, "line"))
    modular = results.get((scenario_id, "modular"))
    if not line or not modular:
        st.info("이 시나리오의 라인·모듈 두 결과를 모두 실행해야 포스터 형식으로 비교됩니다.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 라인 생산 방식 부품 공급망")
        render_result_map(line, f"map_{scenario_id}_line")
        render_solver_metrics(line)
    with c2:
        st.markdown("#### 모듈 활용 분산 생산 방식 부품 공급망")
        render_result_map(modular, f"map_{scenario_id}_modular")
        render_solver_metrics(modular)

    qtable = quartile_comparison_table(results, scenario_id)
    for col in [MODE_LABEL["line"], MODE_LABEL["modular"]]:
        if col in qtable:
            qtable[col] = qtable[col].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "-")
    st.markdown("#### 부품 운송량 사분위수 비율")
    st.dataframe(qtable, hide_index=True, use_container_width=True)

    if line.get("objective_value") and modular.get("objective_value"):
        ratio = modular["objective_value"] / line["objective_value"]
        st.caption(f"모듈/라인 총비용 비율: {ratio:.6f}")


def run_app():
    st.set_page_config(
        page_title="Xpress 기반 전기차 공급망 LP Relaxation",
        page_icon="🚗",
        layout="wide",
    )
    st.title("탄소배출 기반 전기차 공급망 최적화")
    st.caption(
        f"build: {APP_BUILD} · package: {APP_PACKAGE_ID} · "
        "Xpress 목적함수·후보집합·제약구조를 유지하고 모든 변수를 연속화한 LP relaxation"
    )

    defaults = load_default_tables()
    with st.sidebar:
        st.header("데이터")
        use_defaults = st.checkbox("패키지의 Xpress 기준 CSV 사용", value=True)
        uploads = st.file_uploader("수정 CSV 업로드", type="csv", accept_multiple_files=True)
        uploaded = load_uploaded_tables(uploads)
        tables = dict(defaults) if use_defaults else {}
        tables.update(uploaded)
        st.download_button(
            "Xpress 기준 CSV·LP 다운로드",
            make_data_zip(),
            file_name="xpress_reference_data_and_lp.zip",
            mime="application/zip",
        )
        st.divider()
        st.write("**LP relaxation 원칙**")
        st.write("RP·RT·FP·FT·ZM·ZS·ZL·α·β 모두 연속변수")
        st.write("α·β 범위: 0~1, Xpress Big-M 제약 유지")

    errors = validate_tables(tables)
    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    tabs = st.tabs([
        "1. Xpress 입력 데이터",
        "2. 최적화 실행",
        "3. 포스터형 최적화 결과",
        "4. 포스터 기준 비교",
        "5. 수학모형·코드 매핑",
    ])

    with tabs[0]:
        st.header("Xpress 기준 입력 데이터")
        st.info(
            "공급지와 조립지 후보는 Xpress와 동일한 24개 위치입니다. "
            "각 재질은 24개 위치를 모두 공급 후보로 가지므로 총 96개의 재질–공급위치 조합입니다."
        )
        display_names = {
            "products.csv": "제품 6종",
            "demand.csv": "프랑스 수요",
            "raw_material_suppliers.csv": "재질별 24개 공급지",
            "assembly_locations.csv": "조립 후보 24개",
            "transport_parameters.csv": "Xpress 운송수단 계수",
            "markets.csv": "수요지",
            "scenarios.csv": "시나리오",
            "xpress_model_metadata.csv": "모형 상수",
        }
        subtabs = st.tabs(list(display_names.values()))
        for sub, filename in zip(subtabs, display_names):
            with sub:
                st.dataframe(tables[filename], use_container_width=True, hide_index=True)

    with tabs[1]:
        st.header("최적화 실행")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            scenario_id = st.selectbox(
                "시나리오",
                ["S1", "S2", "S3"],
                format_func=lambda x: str(tables["scenarios.csv"].set_index("scenario_id").loc[x, "scenario_name"]),
            )
        with col2:
            production_mode = st.selectbox(
                "생산 방식",
                ["line", "modular"],
                format_func=lambda x: MODE_LABEL[x],
            )
        with col3:
            cap_application = st.selectbox(
                "탄소상한 적용 단위",
                ["class_average", "product_strict"],
                format_func=lambda x: "트림별 상한" if x == "product_strict" else "차급 수요가중 평균 상한",
            )
        with col4:
            time_limit = st.number_input("Solver 제한시간(초)", min_value=10, max_value=600, value=180, step=10)

        st.warning(
            "업로드된 Xpress LP가 직접 보증하는 정확한 기준 모형은 시나리오 ② 모듈 방식입니다. "
            "라인 방식과 시나리오 ①·③은 동일 Xpress 파라미터와 Word/포스터의 생산방식·탄소상한 정의를 결합한 재구성 모형입니다."
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("선택 조합 실행", type="primary", use_container_width=True):
                with st.spinner("Xpress 구조의 연속 LP를 OR-Tools GLOP으로 계산하는 중입니다..."):
                    try:
                        res = solve_case(
                            tables, scenario_id, production_mode, cap_application, int(time_limit)
                        )
                        st.session_state.setdefault("xpress_results", {})[(scenario_id, production_mode)] = res
                        st.success(f"{res['status']} · {res.get('wall_time_sec', 0):.2f}초")
                    except Exception as exc:
                        st.exception(exc)
        with b2:
            if st.button("포스터 6개 조합 순차 실행", use_container_width=True):
                progress = st.progress(0.0)
                status_box = st.empty()
                all_results = st.session_state.setdefault("xpress_results", {})
                cases = [(s, m) for s in ["S1", "S2", "S3"] for m in ["line", "modular"]]
                for i, (s, m) in enumerate(cases, start=1):
                    status_box.write(f"{SCENARIO_SHORT[s]} · {MODE_LABEL[m]} 계산 중 ({i}/6)")
                    try:
                        all_results[(s, m)] = solve_case(
                            tables, s, m, cap_application, int(time_limit)
                        )
                    except Exception as exc:
                        all_results[(s, m)] = {"status": "ERROR", "message": str(exc)}
                    progress.progress(i / len(cases))
                status_box.success("6개 조합 계산이 완료되었습니다.")

        st.markdown("### 보조금 점수 민감도")
        with st.expander("선택 생산방식의 점수 민감도 실행", expanded=False):
            c1, c2, c3 = st.columns(3)
            min_score = c1.number_input("최소 점수", 0.0, 80.0, 55.0, 1.0)
            max_score = c2.number_input("최대 점수", 0.0, 80.0, 70.0, 1.0)
            n_points = c3.number_input("점수 개수", 3, 7, 4, 1)
            if st.button("점수 민감도 계산"):
                rows = []
                scores = np.linspace(float(min_score), float(max_score), int(n_points))
                p = st.progress(0.0)
                for i, score in enumerate(scores, start=1):
                    try:
                        res = solve_case(
                            tables, "S1", production_mode, cap_application, int(time_limit), score_override=float(score)
                        )
                        rows.append({
                            "minimum_score": score,
                            "small_cap": carbon_cap_from_score("small", score),
                            "standard_cap": carbon_cap_from_score("standard", score),
                            "status": res.get("status"),
                            "total_cost_eur": res.get("objective_value"),
                            "total_emissions_kgco2": res.get("total_emissions_kgco2"),
                            "wall_time_sec": res.get("wall_time_sec"),
                        })
                    except Exception as exc:
                        rows.append({"minimum_score": score, "status": "ERROR", "message": str(exc)})
                    p.progress(i / len(scores))
                st.session_state["score_sensitivity"] = pd.DataFrame(rows)

    with tabs[2]:
        st.header("포스터형 최적화 결과")
        results = st.session_state.get("xpress_results", {})
        if not results:
            st.info("2번 탭에서 선택 조합 또는 6개 조합을 실행하세요.")
        else:
            for scenario in ["S1", "S2", "S3"]:
                render_poster_scenario(results, scenario)
                st.divider()

            st.markdown("### 분석 및 결론")
            ratio_df = cost_ratio_dataframe(results)
            if not ratio_df.empty:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.bar(ratio_df["scenario_name"], ratio_df["modular_to_line_cost_ratio"])
                ax.axhline(1.0, linewidth=1)
                ax.set_ylabel("모듈/라인 비용 비율")
                ax.set_title("시나리오별 비용 비율")
                ax.grid(axis="y", alpha=0.25)
                st.pyplot(fig, use_container_width=False)
                st.dataframe(ratio_df, hide_index=True, use_container_width=True)

            valid_results = {
                key: value for key, value in results.items()
                if value.get("status") in {"OPTIMAL", "FEASIBLE"}
            }
            if valid_results:
                st.download_button(
                    "전체 결과 ZIP 다운로드",
                    results_zip(valid_results),
                    file_name="xpress_lp_relaxation_results.zip",
                    mime="application/zip",
                )

            sensitivity = st.session_state.get("score_sensitivity")
            if isinstance(sensitivity, pd.DataFrame) and not sensitivity.empty:
                st.markdown("### 보조금 점수 민감도")
                st.dataframe(sensitivity, hide_index=True, use_container_width=True)
                valid = sensitivity[sensitivity["status"].isin(["OPTIMAL", "FEASIBLE"])]
                if not valid.empty:
                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    ax.plot(valid["minimum_score"], valid["total_cost_eur"], marker="o")
                    ax.set_xlabel("최소 보조금 점수")
                    ax.set_ylabel("총 공급망 비용 (€)")
                    ax.grid(alpha=0.25)
                    st.pyplot(fig, use_container_width=False)

            st.markdown("### 상세 결과")
            available = [key for key, value in results.items() if value.get("status") in {"OPTIMAL", "FEASIBLE"}]
            if available:
                chosen = st.selectbox(
                    "상세 조회 조합",
                    available,
                    format_func=lambda key: f"{SCENARIO_SHORT[key[0]]} · {MODE_LABEL[key[1]]}",
                )
                selected = results[chosen]
                subtabs = st.tabs(["차량별 결과", "공급지 생산량", "부품 경로", "조립지", "완제품 경로", "Solver 정보"])
                frames = [
                    selected.get("product_summary"), selected.get("supplier_summary"), selected.get("raw_routes"),
                    selected.get("plant_summary"), selected.get("finished_routes"),
                ]
                for tab, frame in zip(subtabs[:5], frames):
                    with tab:
                        if isinstance(frame, pd.DataFrame):
                            st.dataframe(frame, hide_index=True, use_container_width=True)
                with subtabs[5]:
                    st.json({
                        "status": selected.get("status"),
                        "solver": selected.get("solver_name"),
                        "solver_version": selected.get("solver_version"),
                        "solver_iterations": selected.get("solver_iterations"),
                        "variables": selected.get("variable_count"),
                        "continuous_variables": selected.get("continuous_variable_count"),
                        "integer_variables": selected.get("integer_variable_count"),
                        "binary_variables": selected.get("binary_variable_count"),
                        "constraints": selected.get("constraint_count"),
                        "matrix_nonzeros": selected.get("matrix_nonzeros"),
                        "wall_time_sec": selected.get("wall_time_sec"),
                    })

    with tabs[3]:
        st.header("포스터 기준 결과와 비교")
        poster_path = ASSET_DIR / "poster_reference.png"
        if poster_path.exists():
            st.image(str(poster_path), caption="2025 춘계산업공학회 포스터 기준 그림", use_container_width=True)

        benchmark_ratio = tables["poster_benchmark_cost_ratios.csv"].copy()
        benchmark_q = tables["poster_benchmark_quartiles.csv"].copy()
        st.markdown("### 포스터 비용 비율")
        st.dataframe(benchmark_ratio, hide_index=True, use_container_width=True)
        st.markdown("### 포스터 사분위수 비율")
        st.dataframe(benchmark_q, hide_index=True, use_container_width=True)

        results = st.session_state.get("xpress_results", {})
        calculated = cost_ratio_dataframe(results)
        if not calculated.empty:
            compare = benchmark_ratio.merge(
                calculated[["scenario_id", "modular_to_line_cost_ratio"]],
                on="scenario_id",
                how="left",
                suffixes=("_poster", "_saas"),
            )
            compare["absolute_difference"] = (
                compare["modular_to_line_cost_ratio_saas"] - compare["modular_to_line_cost_ratio_poster"]
            ).abs()
            st.markdown("### 재계산값–포스터 비교")
            st.dataframe(compare, hide_index=True, use_container_width=True)

    with tabs[4]:
        st.header("수학모형과 코드의 대응")
        st.markdown(
            r"""
**목적함수**

\[
\min Z=
\sum_{f,r,s} c^{RP}_{rs}RP_{frs}
+\sum_{f,r,s,p,t}c^{RT}_{spt}RT_{frspt}
+\sum_{f,p}c^{FP}_{fp}FP_{fp}
+\sum_{f,p,t}c^{FT}_{fpt}FT_{fpt}.
\]

**LP relaxation 정의역**

\[
RP,RT,FP,FT,ZM,ZS,ZL\ge0,\qquad 0\le\alpha,\beta\le1.
\]

**주요 제약**

\[
RP_{frs}=\sum_{p,t}RT_{frspt},
\]

\[
\sum_{s,t}RT_{frspt}=a_{fr}FP_{fp},
\]

\[
FP_{fp}=\sum_tFT_{fpt},\qquad \sum_{p,t}FT_{fpt}=D_f,
\]

\[
\sum_fRP_{frs}\le1{,}000{,}000.
\]

Xpress의 운송수단 선택식과 complementary Big-M 연결식도 그대로 유지합니다. 차이는 정수·이진 정의역만 연속구간으로 완화한 것입니다.
"""
        )
        st.markdown("### 코드 매핑")
        mapping = pd.DataFrame([
            ["RP", "IndexLayout.rp()", "공급지 생산량", "연속"],
            ["RT", "IndexLayout.rt()", "공급지→조립지 부품 운송량", "연속"],
            ["FP", "IndexLayout.fp()", "조립지별 완제품 생산량", "연속"],
            ["FT", "IndexLayout.ft()", "조립지→프랑스 완제품 운송량", "연속"],
            ["ZM/ZS 또는 ZL", "IndexLayout.z1()/z2()", "모듈 또는 팩 등가량", "연속"],
            ["α", "IndexLayout.alpha()", "부품 운송수단 선택 relaxation", "0~1 연속"],
            ["β", "IndexLayout.beta()", "완제품 운송수단 선택 relaxation", "0~1 연속"],
        ], columns=["수학 변수", "코드", "의미", "정의역"])
        st.dataframe(mapping, hide_index=True, use_container_width=True)

        st.markdown("### Xpress 구조 검증 기준")
        checks = pd.DataFrame([
            ["모듈 S2 변수 수", 119376, "IndexLayout('modular').n_vars"],
            ["모듈 S2 제약 수", 74694, "탄소상한이 없는 기본 Xpress 구조"],
            ["제품 수", 6, "products.csv"],
            ["재질 수", 4, "고정"],
            ["공급·조립 위치 수", 24, "assembly_locations.csv"],
            ["운송수단 수", 4, "transport_parameters.csv"],
            ["공급지당 용량", 1000000, "raw_material_suppliers.csv"],
            ["최소 거리", 50, "xpress_model_metadata.csv"],
        ], columns=["검사항목", "Xpress 기준", "코드/CSV"])
        st.dataframe(checks, hide_index=True, use_container_width=True)
        st.caption(f"참조 LP SHA-256: {REFERENCE_LP_SHA256}")


if __name__ == "__main__":
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install requirements.txt and run: streamlit run app.py")
    run_app()
