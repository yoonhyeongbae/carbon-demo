# DEPLOYMENT_MARKER: PDF_COUNTRY_ROUTE_LP_V8_7
from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import time
import zipfile
from array import array
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from ortools.linear_solver import pywraplp

try:
    import streamlit as st
except Exception:  # local syntax/core testing without Streamlit
    st = None


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ASSET_DIR = APP_DIR / "assets"
REFERENCE_DIR = APP_DIR / "reference"

APP_BUILD = "pdf-country-route-battery-stage-ui-v8.7"
APP_PACKAGE_ID = "20260804-v8.7-explained-tables-all-maps"
REFERENCE_LP_SHA256 = "efe0ec2e80a26b07dcbec47d2eaf74fb300cd63a5014e81e90147f9581ba4244"

REQUIRED_FILES = [
    "products.csv",
    "demand.csv",
    "raw_material_suppliers.csv",
    "assembly_locations.csv",
    "transport_parameters.csv",
    "country_transport_rules.csv",
    "material_parameters.csv",
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

# Direct land modes are used for Europe-Europe and same-continent routes.
# Composite modes represent origin inland + international sea/air + destination inland.
ROUTE_MODE_CODES = (
    "road",
    "rail",
    "sea_road",
    "sea_rail",
    "air_road",
    "air_rail",
)
ROUTE_MODE_LABEL = {
    "road": "도로",
    "rail": "철도",
    "sea_road": "해상+도로",
    "sea_rail": "해상+철도",
    "air_road": "항공+도로",
    "air_rail": "항공+철도",
}
TRANSPORT_COLOR = {
    1: "#d95f02",
    2: "#7570b3",
    3: "#1b9e77",
    4: "#66a61e",
    5: "#e7298a",
    6: "#e6ab02",
}
TRANSPORT_DASH = {1: None, 2: "11,6", 3: "4,7", 4: "12,5,3,5", 5: "1,6", 6: "1,4,9,4"}

MIN_DISTANCE_KM = 0.0
INTERNATIONAL_INLAND_LEG_KM = 50.0
FLOW_TOL = 1e-6
MAX_SENSITIVITY_POINTS = 5



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


# Cache only small immutable inputs and the reference ZIP. Limiting entries prevents
# an unbounded process-wide cache on Streamlit Community Cloud.
if st is not None:
    load_default_tables = st.cache_data(
        show_spinner=False, max_entries=1
    )(load_default_tables)
    make_data_zip = st.cache_data(
        show_spinner=False, max_entries=1
    )(make_data_zip)


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
            "transport_mode", "transport_mode_ko", "region_class",
            "transport_cost_eur_per_kgkm", "transport_ef_kgco2_per_kgkm",
        },
        "country_transport_rules.csv": {
            "location_name", "continent", "road_region_class", "rail_region_class",
            "allow_road", "allow_rail", "allow_sea", "allow_air",
        },
        "material_parameters.csv": {"material_id", "loss_rate"},
        "markets.csv": {"market_id", "market_name", "location_name", "continent", "latitude", "longitude", "minimum_distance_km"},
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
    rules = tables["country_transport_rules.csv"].copy()
    material_parameters = tables["material_parameters.csv"].copy()

    if sorted(products["xpress_product_index"].astype(int).tolist()) != list(range(1, 7)):
        errors.append("products.csv: 제품 인덱스는 1~6이어야 합니다.")
    if sorted(plants["xpress_location_index"].astype(int).tolist()) != list(range(1, 25)):
        errors.append("assembly_locations.csv: 위치 인덱스는 1~24여야 합니다.")

    for material in MATERIALS:
        subset = suppliers[suppliers["material_id"] == material]
        indices = sorted(subset["xpress_location_index"].astype(int).tolist())
        if indices != list(range(1, 25)):
            errors.append(f"raw_material_suppliers.csv: {material}은 24개 위치를 모두 포함해야 합니다.")

    if set(material_parameters["material_id"]) != set(MATERIALS):
        errors.append("material_parameters.csv: steel, aluminum, other, battery 네 재질이 모두 필요합니다.")
    loss = pd.to_numeric(material_parameters["loss_rate"], errors="coerce")
    if loss.isna().any() or (loss < 0).any() or (loss >= 1).any():
        errors.append("material_parameters.csv.loss_rate는 0 이상 1 미만이어야 합니다.")

    required_factor_keys = {
        ("sea", "world"), ("air", "world"),
        ("road", "France"), ("road", "Europe_ex_France"), ("road", "Asia"), ("road", "Americas"),
        ("rail", "France"), ("rail", "Europe_ex_France"), ("rail", "Asia"), ("rail", "Other"),
    }
    factor_keys = set(zip(transport["transport_mode"].astype(str), transport["region_class"].astype(str)))
    missing_factors = sorted(required_factor_keys - factor_keys)
    if missing_factors:
        errors.append(f"transport_parameters.csv: 필수 계수 누락 {missing_factors}")

    for col in ["allow_road", "allow_rail", "allow_sea", "allow_air"]:
        values = pd.to_numeric(rules[col], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            errors.append(f"country_transport_rules.csv.{col}: 0 또는 1만 허용됩니다.")
    required_locations = set(plants["location_name"].astype(str)) | set(suppliers["location_name"].astype(str)) | set(tables["markets.csv"]["location_name"].astype(str))
    missing_rules = sorted(required_locations - set(rules["location_name"].astype(str)))
    if missing_rules:
        errors.append("country_transport_rules.csv: 국가/위치 규칙 누락 " + ", ".join(missing_rules))

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
        errors.append("products.csv: 배터리 용량은 50, 60, 70, 80, 90, 100 kWh여야 합니다.")

    return errors


def ordered_tables(tables: Mapping[str, pd.DataFrame]):
    products = tables["products.csv"].sort_values("xpress_product_index").reset_index(drop=True)
    demand = tables["demand.csv"].copy()
    suppliers = tables["raw_material_suppliers.csv"].sort_values(
        ["xpress_material_index", "xpress_location_index"]
    ).reset_index(drop=True)
    plants = tables["assembly_locations.csv"].sort_values("xpress_location_index").reset_index(drop=True)
    transport = tables["transport_parameters.csv"].copy().reset_index(drop=True)
    country_rules = tables["country_transport_rules.csv"].copy().reset_index(drop=True)
    material_parameters = tables["material_parameters.csv"].copy().reset_index(drop=True)
    markets = tables["markets.csv"].copy()
    scenarios = tables["scenarios.csv"].copy()
    return products, demand, suppliers, plants, transport, country_rules, material_parameters, markets, scenarios


def rounded_distance_km(lat1: float, lon1: float, lat2: float, lon2: float, minimum: float = MIN_DISTANCE_KM) -> float:
    distance = geodesic((float(lat1), float(lon1)), (float(lat2), float(lon2))).km
    return round(max(float(minimum), float(distance)), 2)


@lru_cache(maxsize=4)
def cached_distance_matrices(
    plant_coordinates: Tuple[Tuple[float, float], ...],
    market_coordinate: Tuple[float, float],
    minimum_market_distance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    count = len(plant_coordinates)
    raw = np.zeros((count, count), dtype=np.float64)
    for s, (lat_s, lon_s) in enumerate(plant_coordinates):
        for p, (lat_p, lon_p) in enumerate(plant_coordinates):
            raw[s, p] = rounded_distance_km(lat_s, lon_s, lat_p, lon_p)

    final = np.zeros(count, dtype=np.float64)
    market_lat, market_lon = market_coordinate
    for p, (lat_p, lon_p) in enumerate(plant_coordinates):
        final[p] = rounded_distance_km(
            lat_p, lon_p, market_lat, market_lon, minimum_market_distance
        )

    raw.setflags(write=False)
    final.setflags(write=False)
    return raw, final


def _factor_maps(transport: pd.DataFrame) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    cost: Dict[Tuple[str, str], float] = {}
    ef: Dict[Tuple[str, str], float] = {}
    for _, row in transport.iterrows():
        key = (str(row["transport_mode"]), str(row["region_class"]))
        cost[key] = float(row["transport_cost_eur_per_kgkm"])
        ef[key] = float(row["transport_ef_kgco2_per_kgkm"])
    return cost, ef


def _factor_value(mapping: Mapping[Tuple[str, str], float], mode: str, region: str) -> float:
    for key in ((mode, region), (mode, "world"), (mode, "Other")):
        if key in mapping:
            return float(mapping[key])
    raise KeyError(f"운송계수 누락: mode={mode}, region={region}")


def _rule_map(country_rules: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    return {
        str(row["location_name"]): {
            "location_name": str(row["location_name"]),
            "continent": str(row["continent"]),
            "road_region_class": str(row["road_region_class"]),
            "rail_region_class": str(row["rail_region_class"]),
            "allow_road": int(row["allow_road"]),
            "allow_rail": int(row["allow_rail"]),
            "allow_sea": int(row["allow_sea"]),
            "allow_air": int(row["allow_air"]),
        }
        for _, row in country_rules.iterrows()
    }


def _route_allowed(origin: Mapping[str, object], destination: Mapping[str, object], route_code: str) -> bool:
    same_location = str(origin["location_name"]) == str(destination["location_name"])
    same_continent = str(origin["continent"]) == str(destination["continent"])
    both_europe = str(origin["continent"]) == "Europe" and str(destination["continent"]) == "Europe"

    if route_code == "road":
        return bool(origin["allow_road"] and destination["allow_road"] and (same_continent or both_europe))
    if route_code == "rail":
        return bool(origin["allow_rail"] and destination["allow_rail"] and (same_continent or both_europe))

    # Europe-Europe and within the same location are handled as direct land routes.
    if both_europe or same_location:
        return False

    international, land = route_code.split("_")
    return bool(
        origin[f"allow_{international}"]
        and destination[f"allow_{international}"]
        and origin[f"allow_{land}"]
        and destination[f"allow_{land}"]
    )


def _route_coefficient(
    distance_km: float,
    origin: Mapping[str, object],
    destination: Mapping[str, object],
    route_code: str,
    factor_cost: Mapping[Tuple[str, str], float],
    factor_ef: Mapping[Tuple[str, str], float],
) -> Tuple[float, float, float, float, float]:
    """Return cost/kg, kgCO2/kg, total km, inland km, international km.

    The PDF supplies mode/region emission factors but not exact port/airport leg
    distances. Direct land routes use the geodesic distance. Composite routes
    reserve a user-editable 50 km inland leg at each end and use the remaining
    distance as the sea/air main leg.
    """
    d = float(distance_km)
    if route_code in {"road", "rail"}:
        mode = route_code
        region_field = f"{mode}_region_class"
        origin_region = str(origin[region_field])
        destination_region = str(destination[region_field])
        unit_cost = 0.5 * (
            _factor_value(factor_cost, mode, origin_region)
            + _factor_value(factor_cost, mode, destination_region)
        )
        unit_ef = 0.5 * (
            _factor_value(factor_ef, mode, origin_region)
            + _factor_value(factor_ef, mode, destination_region)
        )
        return d * unit_cost, d * unit_ef, d, d, 0.0

    international_mode, land_mode = route_code.split("_")
    inland_each = min(INTERNATIONAL_INLAND_LEG_KM, d / 4.0)
    inland_total = 2.0 * inland_each
    international_distance = max(0.0, d - inland_total)
    region_field = f"{land_mode}_region_class"
    origin_region = str(origin[region_field])
    destination_region = str(destination[region_field])

    land_cost = inland_each * (
        _factor_value(factor_cost, land_mode, origin_region)
        + _factor_value(factor_cost, land_mode, destination_region)
    )
    land_ef = inland_each * (
        _factor_value(factor_ef, land_mode, origin_region)
        + _factor_value(factor_ef, land_mode, destination_region)
    )
    international_cost = international_distance * _factor_value(factor_cost, international_mode, "world")
    international_ef = international_distance * _factor_value(factor_ef, international_mode, "world")
    return (
        land_cost + international_cost,
        land_ef + international_ef,
        inland_total + international_distance,
        inland_total,
        international_distance,
    )


def build_route_matrices(
    plants: pd.DataFrame,
    market: pd.Series,
    transport: pd.DataFrame,
    country_rules: pd.DataFrame,
    raw_distances: np.ndarray,
    final_distances: np.ndarray,
) -> Dict[str, np.ndarray]:
    factor_cost, factor_ef = _factor_maps(transport)
    rules = _rule_map(country_rules)
    t_count = len(ROUTE_MODE_CODES)
    s_count = len(plants)
    p_count = len(plants)

    raw_allowed = np.zeros((s_count, p_count, t_count), dtype=bool)
    raw_cost = np.zeros((s_count, p_count, t_count), dtype=float)
    raw_ef = np.zeros((s_count, p_count, t_count), dtype=float)
    raw_total = np.zeros((s_count, p_count, t_count), dtype=float)
    raw_inland = np.zeros((s_count, p_count, t_count), dtype=float)
    raw_international = np.zeros((s_count, p_count, t_count), dtype=float)

    for s in range(s_count):
        origin = rules[str(plants.iloc[s]["location_name"])]
        for p in range(p_count):
            destination = rules[str(plants.iloc[p]["location_name"])]
            for t, code in enumerate(ROUTE_MODE_CODES):
                allowed = _route_allowed(origin, destination, code)
                raw_allowed[s, p, t] = allowed
                if allowed:
                    values = _route_coefficient(raw_distances[s, p], origin, destination, code, factor_cost, factor_ef)
                    raw_cost[s, p, t], raw_ef[s, p, t], raw_total[s, p, t], raw_inland[s, p, t], raw_international[s, p, t] = values

    final_allowed = np.zeros((p_count, t_count), dtype=bool)
    final_cost = np.zeros((p_count, t_count), dtype=float)
    final_ef = np.zeros((p_count, t_count), dtype=float)
    final_total = np.zeros((p_count, t_count), dtype=float)
    final_inland = np.zeros((p_count, t_count), dtype=float)
    final_international = np.zeros((p_count, t_count), dtype=float)
    destination = rules[str(market["location_name"])]
    for p in range(p_count):
        origin = rules[str(plants.iloc[p]["location_name"])]
        for t, code in enumerate(ROUTE_MODE_CODES):
            allowed = _route_allowed(origin, destination, code)
            final_allowed[p, t] = allowed
            if allowed:
                values = _route_coefficient(final_distances[p], origin, destination, code, factor_cost, factor_ef)
                final_cost[p, t], final_ef[p, t], final_total[p, t], final_inland[p, t], final_international[p, t] = values

    if not raw_allowed.any(axis=2).all():
        bad = np.argwhere(~raw_allowed.any(axis=2))
        raise ValueError(f"허용 운송경로가 없는 공급지-조립지 조합: {bad[:10].tolist()}")
    if not final_allowed.any(axis=1).all():
        bad = np.argwhere(~final_allowed.any(axis=1)).ravel().tolist()
        raise ValueError(f"허용 운송경로가 없는 조립지-프랑스 조합: {bad[:10]}")

    return {
        "raw_allowed": raw_allowed,
        "raw_cost_per_kg": raw_cost,
        "raw_ef_per_kg": raw_ef,
        "raw_total_km": raw_total,
        "raw_inland_km": raw_inland,
        "raw_international_km": raw_international,
        "final_allowed": final_allowed,
        "final_cost_per_kg": final_cost,
        "final_ef_per_kg": final_ef,
        "final_total_km": final_total,
        "final_inland_km": final_inland,
        "final_international_km": final_international,
    }



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
    T: int = len(ROUTE_MODE_CODES)

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
    def n_vars(self) -> int:
        return self.off_z2 + self.n_z2

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


class LinearConstraintBuilder:
    """Compact row-wise storage used to build an OR-Tools MPSolver model.

    Coefficients are stored only until solve time. This avoids an additional sparse-matrix dependency and
    keeps the route-based continuous LP coefficient storage memory-efficient.
    """

    def __init__(self, n_vars: int):
        self.n_vars = int(n_vars)
        # array('I') / array('d') stores coefficients compactly instead of
        # hundreds of thousands of boxed Python int/float objects.
        self.eq_cols = array("I")
        self.eq_data = array("d")
        self.eq_starts = array("I", [0])
        self.eq_rhs = array("d")
        self.ub_cols = array("I")
        self.ub_data = array("d")
        self.ub_starts = array("I", [0])
        self.ub_rhs = array("d")

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
        # Some Streamlit Community Cloud Python images expose array.array
        # without a .clear() method. Slice deletion works across supported
        # Python versions and releases the compact coefficient buffers.
        del self.eq_cols[:]
        del self.eq_data[:]
        del self.eq_starts[:]
        self.eq_starts.append(0)
        del self.eq_rhs[:]
        del self.ub_cols[:]
        del self.ub_data[:]
        del self.ub_starts[:]
        self.ub_starts.append(0)
        del self.ub_rhs[:]


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
    country_rules: pd.DataFrame
    material_parameters: pd.DataFrame
    material_loss_rates: np.ndarray
    scenario: pd.Series
    raw_distances: np.ndarray
    final_distances: np.ndarray
    raw_route_allowed: np.ndarray
    raw_route_cost_per_kg: np.ndarray
    raw_route_ef_per_kg: np.ndarray
    raw_route_total_km: np.ndarray
    raw_route_inland_km: np.ndarray
    raw_route_international_km: np.ndarray
    final_route_allowed: np.ndarray
    final_route_cost_per_kg: np.ndarray
    final_route_ef_per_kg: np.ndarray
    final_route_total_km: np.ndarray
    final_route_inland_km: np.ndarray
    final_route_international_km: np.ndarray
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


def build_pdf_route_lp_model(
    tables: Mapping[str, pd.DataFrame],
    production_mode: str,
    scenario_id: str,
    cap_application: str = "product_strict",
    score_override: Optional[float] = None,
) -> LPModel:
    (
        products, demand, suppliers, plants, transport, country_rules,
        material_parameters, markets, scenarios,
    ) = ordered_tables(tables)
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

    loss_map = material_parameters.set_index("material_id")["loss_rate"].astype(float).to_dict()
    material_loss_rates = np.asarray([float(loss_map[m]) for m in MATERIALS], dtype=float)

    market = markets.iloc[0]
    minimum_market_distance = float(market.get("minimum_distance_km", MIN_DISTANCE_KM))
    plant_coordinates = tuple(
        (float(row["latitude"]), float(row["longitude"]))
        for _, row in plants.iterrows()
    )
    raw_distances, final_distances = cached_distance_matrices(
        plant_coordinates,
        (float(market["latitude"]), float(market["longitude"])),
        minimum_market_distance,
    )
    route = build_route_matrices(
        plants, market, transport, country_rules, raw_distances, final_distances
    )

    c = np.zeros(layout.n_vars, dtype=float)
    lb = np.zeros(layout.n_vars, dtype=float)
    ub = np.full(layout.n_vars, np.inf, dtype=float)

    emission_rp = np.zeros(layout.n_rp, dtype=float)
    emission_rt = np.zeros(layout.n_rt, dtype=float)
    emission_fp = np.zeros(layout.n_fp, dtype=float)
    emission_ft = np.zeros(layout.n_ft, dtype=float)

    # Objective and PDF-based carbon coefficients.
    #
    # Battery interpretation differs by production mode:
    # - line: the finished battery pack is produced at the same indexed location as vehicle assembly.
    #         Battery RT is therefore restricted to one zero-distance internal transfer (s=p, k=road).
    # - modular: 10/5 kWh modules are produced at country s and may be transported to a different
    #            assembly country p. A completed pack is assembled at p. The module-production EF
    #            depends only on country s, while the modular pack-assembly EF depends only on country p.
    for f in range(layout.F):
        product = products.iloc[f]
        for r in range(layout.R):
            gross_multiplier = 1.0 / (1.0 - material_loss_rates[r])
            mass_per_flow_unit = (
                float(product["battery_mass_kg"]) / float(product["battery_kwh"])
                if r == MATERIAL_INDEX["battery"] else 1.0
            )
            for s in range(layout.S):
                rp_idx = layout.rp(f, r, s)
                c[rp_idx] = supplier_cost[r, s]
                emission_rp[rp_idx - layout.off_rp] = supplier_ef[r, s] * gross_multiplier
                for p in range(layout.P):
                    for t in range(layout.T):
                        rt_idx = layout.rt(f, r, s, p, t)

                        # In line production, the finished pack and vehicle are assembled at the same
                        # indexed country/location. Only a zero-distance internal battery flow is kept.
                        if r == MATERIAL_INDEX["battery"] and production_mode == "line":
                            if s != p or t != 0:
                                ub[rt_idx] = 0.0
                                continue
                            c[rt_idx] = 0.0
                            emission_rt[rt_idx - layout.off_rt] = 0.0
                            continue

                        if not route["raw_allowed"][s, p, t]:
                            ub[rt_idx] = 0.0
                            continue
                        c[rt_idx] = route["raw_cost_per_kg"][s, p, t] * mass_per_flow_unit
                        emission_rt[rt_idx - layout.off_rt] = route["raw_ef_per_kg"][s, p, t] * mass_per_flow_unit

        for p in range(layout.P):
            fp_idx = layout.fp(f, p)
            body_mass = float(product["nonbattery_mass_kg"])
            pack_mass = float(product["battery_mass_kg"]) if production_mode == "modular" else 0.0
            assembly_mass = body_mass + pack_mass

            # Common vehicle-body assembly uses the non-battery mass. Modular production additionally
            # assembles transported modules into a completed pack at country p. Because the source PDF
            # does not provide a separate module-to-pack factor, the existing country-specific assembly
            # cost/EF is used for that user-requested extension.
            c[fp_idx] = assembly_mass * float(plants.iloc[p]["assembly_cost_eur_per_kg"])
            emission_fp[fp_idx - layout.off_fp] = (
                assembly_mass * float(plants.iloc[p]["assembly_ef_kgco2_per_kg"])
            )
            for t in range(layout.T):
                ft_idx = layout.ft(f, p, t)
                if not route["final_allowed"][p, t]:
                    ub[ft_idx] = 0.0
                    continue
                c[ft_idx] = float(product["vehicle_mass_kg"]) * route["final_cost_per_kg"][p, t]
                emission_ft[ft_idx - layout.off_ft] = (
                    float(product["vehicle_mass_kg"]) * route["final_ef_per_kg"][p, t]
                )

    # ZL exists with an (f,s,p) index for compatibility with the historical layout, but line
    # production permits only its diagonal entries s=p. This is a fixed co-location rule, not a
    # binary decision variable.
    if production_mode == "line":
        for f in range(layout.F):
            for s in range(layout.S):
                for p in range(layout.P):
                    if s != p:
                        ub[layout.z1(f, s, p)] = 0.0

    rows = LinearConstraintBuilder(layout.n_vars)

    # 1) Exact market-demand fulfillment. Equality prevents both shortage and unexplained surplus.
    for f in range(layout.F):
        cols = [layout.ft(f, p, t) for p in range(layout.P) for t in range(layout.T)]
        rows.add_eq(cols, [1.0] * len(cols), demand_values[f])

    # 2) Battery production structure.
    if production_mode == "modular":
        # Stage 1: modules are produced at country/location s. The same country-specific battery EF
        # applies per kWh to both 10 kWh and 5 kWh modules; module size does not change the EF.
        for f in range(layout.F):
            for s in range(layout.S):
                for p in range(layout.P):
                    cols = [layout.rt(f, MATERIAL_INDEX["battery"], s, p, t) for t in range(layout.T)]
                    vals = [1.0] * layout.T
                    cols.extend([layout.z1(f, s, p), layout.z2(f, s, p)])
                    vals.extend([-10.0, -5.0])
                    rows.add_eq(cols, vals, 0.0)

        # Stage 2: all incoming modules are assembled into a completed pack at vehicle-assembly
        # location p. The completed pack capacity must equal vehicle battery demand B_f * FP_fp.
        for f in range(layout.F):
            battery_kwh = float(products.iloc[f]["battery_kwh"])
            for p in range(layout.P):
                cols = [
                    layout.rt(f, MATERIAL_INDEX["battery"], s, p, t)
                    for s in range(layout.S) for t in range(layout.T)
                ]
                vals = [1.0] * (layout.S * layout.T)
                cols.append(layout.fp(f, p))
                vals.append(-battery_kwh)
                rows.add_eq(cols, vals, 0.0)
    else:
        # Line production: completed battery-pack production and vehicle assembly are co-located.
        # Only the diagonal location pair s=p and the internal route k=0 may carry battery flow.
        for f in range(layout.F):
            battery_kwh = float(products.iloc[f]["battery_kwh"])
            for p in range(layout.P):
                internal_rt = layout.rt(f, MATERIAL_INDEX["battery"], p, p, 0)
                diagonal_zl = layout.z1(f, p, p)
                rows.add_eq([internal_rt, diagonal_zl], [1.0, -battery_kwh], 0.0)
                rows.add_eq([diagonal_zl, layout.fp(f, p)], [1.0, -1.0], 0.0)

    # 3) Receiving-side material balances at every assembly location.
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

    # 4) Sending-side assembly balance: assembled vehicles equal outgoing finished vehicles.
    for f in range(layout.F):
        for p in range(layout.P):
            cols = [layout.fp(f, p)] + [layout.ft(f, p, t) for t in range(layout.T)]
            vals = [1.0] + [-1.0] * layout.T
            rows.add_eq(cols, vals, 0.0)

    # 5) Sending-side supplier balance: net usable production equals all outgoing quantities.
    for f in range(layout.F):
        for r in range(layout.R):
            for s in range(layout.S):
                cols = [layout.rp(f, r, s)] + [
                    layout.rt(f, r, s, p, t) for p in range(layout.P) for t in range(layout.T)
                ]
                vals = [1.0] + [-1.0] * (layout.P * layout.T)
                rows.add_eq(cols, vals, 0.0)

    # 6) Supplier capacities.
    for r in range(layout.R):
        for s in range(layout.S):
            cols = [layout.rp(f, r, s) for f in range(layout.F)]
            rows.add_le(cols, [1.0] * layout.F, supplier_capacity[r, s])

    # 7) PDF-based carbon caps for S1/S3. S2 reports emissions but does not constrain them.
    if int(scenario["apply_carbon_cap"]) == 1:
        product_rows: List[Tuple[List[int], List[float], float, str]] = []
        for f in range(layout.F):
            cols: List[int] = []
            vals: List[float] = []
            start = f * layout.R * layout.S
            for local in range(layout.R * layout.S):
                idx = layout.off_rp + start + local
                coef = emission_rp[idx - layout.off_rp]
                if coef:
                    cols.append(idx); vals.append(coef)
            rt_start = f * layout.R * layout.S * layout.P * layout.T
            for local in range(layout.R * layout.S * layout.P * layout.T):
                idx = layout.off_rt + rt_start + local
                coef = emission_rt[idx - layout.off_rt]
                if coef:
                    cols.append(idx); vals.append(coef)
            for p in range(layout.P):
                idx = layout.fp(f, p)
                cols.append(idx); vals.append(emission_fp[idx - layout.off_fp])
            for p in range(layout.P):
                for t in range(layout.T):
                    idx = layout.ft(f, p, t)
                    coef = emission_ft[idx - layout.off_ft]
                    if coef:
                        cols.append(idx); vals.append(coef)
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
        country_rules=country_rules,
        material_parameters=material_parameters,
        material_loss_rates=material_loss_rates,
        scenario=scenario,
        raw_distances=raw_distances,
        final_distances=final_distances,
        raw_route_allowed=route["raw_allowed"],
        raw_route_cost_per_kg=route["raw_cost_per_kg"],
        raw_route_ef_per_kg=route["raw_ef_per_kg"],
        raw_route_total_km=route["raw_total_km"],
        raw_route_inland_km=route["raw_inland_km"],
        raw_route_international_km=route["raw_international_km"],
        final_route_allowed=route["final_allowed"],
        final_route_cost_per_kg=route["final_cost_per_kg"],
        final_route_ef_per_kg=route["final_ef_per_kg"],
        final_route_total_km=route["final_total_km"],
        final_route_inland_km=route["final_inland_km"],
        final_route_international_km=route["final_international_km"],
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


def solve_lp_model(
    model: LPModel,
    time_limit_sec: int = 180,
    objective_override: Optional[np.ndarray] = None,
) -> SolveResult:
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

    coefficients = model.c if objective_override is None else np.asarray(objective_override, dtype=float)
    objective = solver.Objective()
    for idx in np.flatnonzero(coefficients):
        objective.SetCoefficient(variables[int(idx)], float(coefficients[int(idx)]))
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


def total_emission_objective(model: LPModel) -> np.ndarray:
    obj = np.zeros(model.layout.n_vars, dtype=float)
    obj[model.layout.off_rp:model.layout.off_rt] = model.emission_rp
    obj[model.layout.off_rt:model.layout.off_fp] = model.emission_rt
    obj[model.layout.off_fp:model.layout.off_ft] = model.emission_fp
    obj[model.layout.off_ft:model.layout.off_z1] = model.emission_ft
    return obj


def diagnose_carbon_cap_infeasibility(
    tables: Mapping[str, pd.DataFrame],
    scenario_id: str,
    production_mode: str,
    cap_application: str,
    time_limit_sec: int = 90,
) -> Optional[Dict]:
    scenarios = tables["scenarios.csv"]
    target_rows = scenarios[scenarios["scenario_id"] == scenario_id]
    if target_rows.empty:
        return None
    target = target_rows.iloc[0]
    if int(target.get("apply_carbon_cap", 0)) != 1:
        return None

    tmp_tables = dict(tables)
    tmp_scenarios = scenarios.copy()
    tmp_scenarios.loc[tmp_scenarios["scenario_id"] == scenario_id, "apply_carbon_cap"] = 0
    tmp_tables["scenarios.csv"] = tmp_scenarios

    model: Optional[LPModel] = None
    result: Optional[SolveResult] = None
    try:
        model = build_pdf_route_lp_model(
            tmp_tables,
            production_mode=production_mode,
            scenario_id=scenario_id,
            cap_application=cap_application,
        )
        result = solve_lp_model(
            model,
            time_limit_sec=min(max(int(time_limit_sec), 20), 120),
            objective_override=total_emission_objective(model),
        )
        if result.status not in {"OPTIMAL", "FEASIBLE"}:
            return {
                "status": result.status,
                "message": "탄소상한을 제거한 최소배출 진단 문제도 해를 찾지 못했습니다.",
            }

        diag_solution = extract_solution(result)
        product_df = diag_solution.get("product_summary", pd.DataFrame()).copy()
        if product_df.empty:
            return {
                "status": "NO_PRODUCT_SUMMARY",
                "message": "최소배출 진단 결과에서 제품별 요약을 만들지 못했습니다.",
            }

        product_df = product_df[[
            "product_id", "product_name", "vehicle_class", "demand_units",
            "emissions_per_vehicle_kgco2", "subsidy_score"
        ]].copy()
        product_df["scenario_cap_kgco2_per_vehicle"] = product_df["vehicle_class"].map(
            lambda cls: float(target["small_cap_kgco2_per_vehicle"]) if cls == "small"
            else float(target["standard_cap_kgco2_per_vehicle"])
        )
        product_df["slack_kgco2_per_vehicle"] = (
            product_df["scenario_cap_kgco2_per_vehicle"] - product_df["emissions_per_vehicle_kgco2"]
        )
        product_df["cap_satisfied"] = product_df["slack_kgco2_per_vehicle"] >= -1e-6

        class_rows: List[Dict] = []
        for vehicle_class, grp in product_df.groupby("vehicle_class"):
            class_rows.append({
                "vehicle_class": vehicle_class,
                "minimum_weighted_avg_emissions_kgco2_per_vehicle": (
                    float((grp["emissions_per_vehicle_kgco2"] * grp["demand_units"]).sum())
                    / max(float(grp["demand_units"].sum()), 1.0)
                ),
                "scenario_cap_kgco2_per_vehicle": float(target["small_cap_kgco2_per_vehicle"]) if vehicle_class == "small"
                else float(target["standard_cap_kgco2_per_vehicle"]),
            })
        class_df = pd.DataFrame(class_rows)
        class_df["slack_kgco2_per_vehicle"] = (
            class_df["scenario_cap_kgco2_per_vehicle"]
            - class_df["minimum_weighted_avg_emissions_kgco2_per_vehicle"]
        )
        class_df["average_cap_satisfied"] = class_df["slack_kgco2_per_vehicle"] >= -1e-6

        return {
            "status": "DIAGNOSED",
            "message": "탄소상한을 제거하고 총배출량 최소화로 다시 풀어 산출한 이론적 최소배출 진단입니다.",
            "product_minimum_emissions": product_df,
            "class_average_minimum_emissions": class_df,
        }
    finally:
        if result is not None:
            result.x = None
        _release_model_memory(model)
        gc.collect()


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
        }

    model = result.model
    layout = model.layout
    x = result.x
    products = model.products
    plants = model.plants

    rp = x[layout.off_rp:layout.off_rt].reshape(layout.F, layout.R, layout.S)
    rt = x[layout.off_rt:layout.off_fp].reshape(layout.F, layout.R, layout.S, layout.P, layout.T)
    fp = x[layout.off_fp:layout.off_ft].reshape(layout.F, layout.P)
    ft = x[layout.off_ft:layout.off_z1].reshape(layout.F, layout.P, layout.T)

    supplier_records: List[Dict] = []
    for f, r, s in np.argwhere(rp > FLOW_TOL):
        product = products.iloc[f]
        supplier = _material_supplier_row(model, int(r), int(s))
        amount = float(rp[f, r, s])
        loss_rate = float(model.material_loss_rates[r])
        gross_amount = amount / (1.0 - loss_rate)
        supplier_records.append({
            "product_id": product["product_id"],
            "product_name": product["product_name_ko"],
            "material_id": MATERIALS[r],
            "material_name": MATERIAL_LABEL[MATERIALS[r]],
            "supplier_index": s + 1,
            "supplier_location": supplier["location_name"],
            "net_usable_production_amount": amount,
            "gross_production_amount_after_loss": gross_amount,
            "loss_rate": loss_rate,
            "unit": "kWh" if r == MATERIAL_INDEX["battery"] else "kg",
            "production_cost_eur": amount * float(supplier["production_cost"]),
            "production_ef": float(supplier["production_ef"]),
            "production_emissions_kgco2": gross_amount * float(supplier["production_ef"]),
        })
    supplier_df = pd.DataFrame(supplier_records)

    raw_records: List[Dict] = []
    for f, r, s, p, t in np.argwhere(rt > FLOW_TOL):
        # The line-mode diagonal battery flow is an internal co-located transfer, not a
        # country-to-country transport route. Exclude it from route maps and quartiles.
        if layout.mode == "line" and int(r) == MATERIAL_INDEX["battery"] and int(s) == int(p) and int(t) == 0:
            continue
        product = products.iloc[f]
        supplier = _material_supplier_row(model, int(r), int(s))
        plant = plants.iloc[p]
        amount = float(rt[f, r, s, p, t])
        mass_per_unit = (
            float(product["battery_mass_kg"]) / float(product["battery_kwh"])
            if r == MATERIAL_INDEX["battery"] else 1.0
        )
        transport_mass = amount * mass_per_unit
        idx = layout.rt(int(f), int(r), int(s), int(p), int(t))
        code = ROUTE_MODE_CODES[int(t)]
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
            "transport_mode": code,
            "transport_mode_ko": ROUTE_MODE_LABEL[code],
            "flow_amount": amount,
            "flow_unit": "kWh" if r == MATERIAL_INDEX["battery"] else "kg",
            "transport_mass_kg": transport_mass,
            "distance_km": float(model.raw_route_total_km[s, p, t]),
            "inland_distance_km": float(model.raw_route_inland_km[s, p, t]),
            "international_distance_km": float(model.raw_route_international_km[s, p, t]),
            "route_cost_eur_per_kg": float(model.raw_route_cost_per_kg[s, p, t]),
            "route_ef_kgco2_per_kg": float(model.raw_route_ef_per_kg[s, p, t]),
            "transport_cost_eur": amount * float(model.c[idx]),
            "transport_emissions_kgco2": amount * float(model.emission_rt[idx - layout.off_rt]),
        })
    raw_df = pd.DataFrame(raw_records)
    if not raw_df.empty:
        group_cols = ["product_id", "material_id", "supplier_index", "plant_index"]
        group_total = raw_df.groupby(group_cols)["flow_amount"].transform("sum")
        raw_df["mode_quantity_share_pct"] = 100.0 * raw_df["flow_amount"] / group_total

    plant_records: List[Dict] = []
    for f, p in np.argwhere(fp > FLOW_TOL):
        product = products.iloc[f]
        plant = plants.iloc[p]
        units = float(fp[f, p])
        body_mass = units * float(product["nonbattery_mass_kg"])
        pack_mass = units * float(product["battery_mass_kg"]) if layout.mode == "modular" else 0.0
        unit_cost = float(plant["assembly_cost_eur_per_kg"])
        unit_ef = float(plant["assembly_ef_kgco2_per_kg"])
        plant_records.append({
            "product_id": product["product_id"],
            "product_name": product["product_name_ko"],
            "plant_index": p + 1,
            "plant_location": plant["location_name"],
            "assembled_vehicle_equivalents": units,
            "body_assembly_mass_kg": body_mass,
            "modular_pack_assembly_mass_kg": pack_mass,
            "total_assembly_mass_kg": body_mass + pack_mass,
            "body_assembly_cost_eur": body_mass * unit_cost,
            "modular_pack_assembly_cost_eur": pack_mass * unit_cost,
            "total_assembly_cost_eur": (body_mass + pack_mass) * unit_cost,
            "body_assembly_emissions_kgco2": body_mass * unit_ef,
            "modular_pack_assembly_emissions_kgco2": pack_mass * unit_ef,
            "total_assembly_emissions_kgco2": (body_mass + pack_mass) * unit_ef,
        })
    plant_df = pd.DataFrame(plant_records)

    final_records: List[Dict] = []
    for f, p, t in np.argwhere(ft > FLOW_TOL):
        product = products.iloc[f]
        plant = plants.iloc[p]
        units = float(ft[f, p, t])
        idx = layout.ft(int(f), int(p), int(t))
        code = ROUTE_MODE_CODES[int(t)]
        final_records.append({
            "product_id": product["product_id"],
            "product_name": product["product_name_ko"],
            "plant_index": p + 1,
            "plant_location": plant["location_name"],
            "market_name": "프랑스 시장",
            "transport_mode_index": t + 1,
            "transport_mode": code,
            "transport_mode_ko": ROUTE_MODE_LABEL[code],
            "vehicle_equivalents": units,
            "transport_mass_kg": units * float(product["vehicle_mass_kg"]),
            "distance_km": float(model.final_route_total_km[p, t]),
            "inland_distance_km": float(model.final_route_inland_km[p, t]),
            "international_distance_km": float(model.final_route_international_km[p, t]),
            "route_cost_eur_per_kg": float(model.final_route_cost_per_kg[p, t]),
            "route_ef_kgco2_per_kg": float(model.final_route_ef_per_kg[p, t]),
            "transport_cost_eur": units * float(model.c[idx]),
            "transport_emissions_kgco2": units * float(model.emission_ft[idx - layout.off_ft]),
        })
    final_df = pd.DataFrame(final_records)
    if not final_df.empty:
        group_total = final_df.groupby(["product_id", "plant_index"])["vehicle_equivalents"].transform("sum")
        final_df["mode_quantity_share_pct"] = 100.0 * final_df["vehicle_equivalents"] / group_total

    production_cost = float(model.c[layout.off_rp:layout.off_rt] @ rp.ravel())
    raw_transport_cost = float(model.c[layout.off_rt:layout.off_fp] @ rt.ravel())
    assembly_cost = float(model.c[layout.off_fp:layout.off_ft] @ fp.ravel())
    finished_transport_cost = float(model.c[layout.off_ft:layout.off_z1] @ ft.ravel())

    production_emissions = float(model.emission_rp @ rp.ravel())
    raw_transport_emissions = float(model.emission_rt @ rt.ravel())
    assembly_emissions = float(model.emission_fp @ fp.ravel())
    final_transport_emissions = float(model.emission_ft @ ft.ravel())

    body_assembly_cost = 0.0
    modular_pack_assembly_cost = 0.0
    body_assembly_emissions = 0.0
    modular_pack_assembly_emissions = 0.0
    for f in range(layout.F):
        product = products.iloc[f]
        for p in range(layout.P):
            units = float(fp[f, p])
            if units <= FLOW_TOL:
                continue
            plant = plants.iloc[p]
            unit_cost = float(plant["assembly_cost_eur_per_kg"])
            unit_ef = float(plant["assembly_ef_kgco2_per_kg"])
            body_mass = units * float(product["nonbattery_mass_kg"])
            pack_mass = units * float(product["battery_mass_kg"]) if layout.mode == "modular" else 0.0
            body_assembly_cost += body_mass * unit_cost
            modular_pack_assembly_cost += pack_mass * unit_cost
            body_assembly_emissions += body_mass * unit_ef
            modular_pack_assembly_emissions += pack_mass * unit_ef

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
            transport_mass_kg=("transport_mass_kg", "sum"),
            distance_km=("distance_km", "first"),
            inland_distance_km=("inland_distance_km", "first"),
            international_distance_km=("international_distance_km", "first"),
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
        "transport_assignment": "deterministic continuous mode-specific quantities; no random draw",
        "cost_breakdown": {
            "원자재·배터리 생산비": production_cost,
            "부품·모듈 운송비": raw_transport_cost,
            "비배터리 차체 조립비": body_assembly_cost,
            "모듈 배터리팩 조립비": modular_pack_assembly_cost,
            "완제품 운송비": finished_transport_cost,
        },
        "emission_breakdown": {
            "원자재·배터리/모듈 생산": production_emissions,
            "부품·모듈 운송": raw_transport_emissions,
            "비배터리 차체 조립": body_assembly_emissions,
            "모듈 배터리팩 조립": modular_pack_assembly_emissions,
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



def _release_model_memory(model: Optional[LPModel]) -> None:
    if model is None:
        return
    try:
        model.rows.clear_coefficients()
        for name in (
            "c", "lb", "ub", "emission_rp", "emission_rt", "emission_fp", "emission_ft",
            "demand_values", "material_loss_rates", "final_distances",
        ):
            setattr(model, name, np.empty(0, dtype=float))
        model.raw_distances = np.empty((0, 0), dtype=float)
        for name in (
            "raw_route_allowed", "raw_route_cost_per_kg", "raw_route_ef_per_kg",
            "raw_route_total_km", "raw_route_inland_km", "raw_route_international_km",
        ):
            setattr(model, name, np.empty((0, 0, 0), dtype=float))
        for name in (
            "final_route_allowed", "final_route_cost_per_kg", "final_route_ef_per_kg",
            "final_route_total_km", "final_route_inland_km", "final_route_international_km",
        ):
            setattr(model, name, np.empty((0, 0), dtype=float))
    except Exception:
        pass



def solve_case(
    tables: Mapping[str, pd.DataFrame],
    scenario_id: str,
    production_mode: str,
    cap_application: str,
    time_limit_sec: int,
    score_override: Optional[float] = None,
) -> Dict:
    model: Optional[LPModel] = None
    result: Optional[SolveResult] = None
    try:
        model = build_pdf_route_lp_model(
            tables,
            production_mode=production_mode,
            scenario_id=scenario_id,
            cap_application=cap_application,
            score_override=score_override,
        )
        result = solve_lp_model(model, time_limit_sec=time_limit_sec)
        compact = extract_solution(result)
        if result.status == "INFEASIBLE" and score_override is None:
            try:
                diagnosis = diagnose_carbon_cap_infeasibility(
                    tables,
                    scenario_id=scenario_id,
                    production_mode=production_mode,
                    cap_application=cap_application,
                    time_limit_sec=min(max(int(time_limit_sec // 2), 30), 90),
                )
                if diagnosis:
                    compact["feasibility_diagnosis"] = diagnosis
            except Exception as exc:
                compact["feasibility_diagnosis"] = {
                    "status": "DIAGNOSTIC_ERROR",
                    "message": f"진단 계산 중 오류: {exc}",
                }
        # The session stores only compact summaries/nonzero routes, never the
        # 119,376-value solution vector or LP coefficient arrays.
        return compact
    finally:
        if result is not None:
            result.x = None
        _release_model_memory(model)
        result = None
        model = None
        gc.collect()


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
# Reusable explanatory UI helpers
# -----------------------------------------------------------------------------
def apply_global_font_scale() -> None:
    """Increase the visible SaaS typography by approximately 2 pt."""
    st.markdown(
        """
        <style>
        html, body, [class*="css"] { font-size: 18px; }
        .stMarkdown, .stDataFrame, .stSelectbox, .stNumberInput, .stTextInput,
        .stButton, .stCheckbox, .stRadio, .stMetric { font-size: 18px !important; }
        h1 { font-size: 2.7rem !important; }
        h2 { font-size: 2.15rem !important; }
        h3 { font-size: 1.75rem !important; }
        h4 { font-size: 1.35rem !important; }
        div[data-testid="stMetricValue"] { font-size: 2.15rem !important; }
        div[data-testid="stMetricLabel"] { font-size: 1.05rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def explain_table(
    title: str,
    row_meaning: str,
    column_meanings: Mapping[str, str],
    value_meaning: str,
) -> None:
    """Attach an explanation for the rows, columns and values of a table."""
    with st.expander(f"{title}: 행·열·값 읽는 방법", expanded=False):
        st.markdown(f"**행의 의미**: {row_meaning}")
        st.markdown("**열의 의미**")
        for name, meaning in column_meanings.items():
            st.markdown(f"- `{name}`: {meaning}")
        st.markdown(f"**값의 의미**: {value_meaning}")


def show_explained_dataframe(
    df: pd.DataFrame,
    title: str,
    row_meaning: str,
    column_meanings: Optional[Mapping[str, str]] = None,
    value_meaning: str = "각 값은 해당 행의 대상과 열의 지표가 만나는 실제 입력값 또는 최적화 결과값입니다.",
) -> None:
    st.dataframe(df, hide_index=True, use_container_width=True)
    meanings = dict(column_meanings or {str(c): f"{c} 항목" for c in df.columns})
    explain_table(title, row_meaning, meanings, value_meaning)


def render_formula_table(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    formula_columns: Sequence[int],
    row_explanation: str,
    value_explanation: str,
) -> None:
    """Render table-like rows while displaying formula cells with st.latex."""
    st.markdown(f"#### {title}")
    widths = [1.25] * len(headers)
    header_cols = st.columns(widths)
    for col, header in zip(header_cols, headers):
        col.markdown(f"**{header}**")
    st.divider()
    formula_set = set(formula_columns)
    for row in rows:
        cols = st.columns(widths)
        for idx, (col, value) in enumerate(zip(cols, row)):
            if idx in formula_set and str(value).strip():
                col.latex(str(value))
            else:
                col.markdown(str(value))
        st.divider()
    explain_table(
        title,
        row_explanation,
        {h: f"{h} 열의 정의와 해석" for h in headers},
        value_explanation,
    )


def render_map_legend_outside() -> None:
    """Render a readable legend outside Folium maps."""
    st.markdown("#### 공급망 지도 범례")
    st.markdown(
        """
        <div style="background:#fff;border:1px solid #aaa;border-radius:10px;padding:15px 18px;line-height:1.9;">
          <div><b>선 색상 = 운송되는 대상</b>&nbsp;&nbsp;
            <span style="display:inline-block;width:38px;border-top:7px solid #e41a1c;vertical-align:middle;"></span> 철강&nbsp;&nbsp;
            <span style="display:inline-block;width:38px;border-top:7px solid #ff9f1c;vertical-align:middle;"></span> 알루미늄&nbsp;&nbsp;
            <span style="display:inline-block;width:38px;border-top:7px solid #238b45;vertical-align:middle;"></span> 기타 원자재&nbsp;&nbsp;
            <span style="display:inline-block;width:38px;border-top:7px solid #2171b5;vertical-align:middle;"></span> 배터리/모듈&nbsp;&nbsp;
            <span style="display:inline-block;width:38px;border-top:7px solid #6a3d9a;vertical-align:middle;"></span> 완제품 차량
          </div>
          <div><b>선 모양 = 운송경로</b>&nbsp;&nbsp;
            <span style="display:inline-block;width:38px;border-top:4px solid #555;vertical-align:middle;"></span> 도로&nbsp;&nbsp;
            <span style="display:inline-block;width:38px;border-top:4px dashed #555;vertical-align:middle;"></span> 철도&nbsp;&nbsp;
            <span style="display:inline-block;width:38px;border-top:4px dotted #555;vertical-align:middle;"></span> 해상/항공 포함 복합운송
          </div>
          <div><b>선 굵기 = 운송량 사분위수</b>&nbsp;&nbsp;
            <span style="display:inline-block;width:34px;border-top:2.5px solid #333;vertical-align:middle;"></span> Q1&nbsp;&nbsp;
            <span style="display:inline-block;width:34px;border-top:5px solid #333;vertical-align:middle;"></span> Q2&nbsp;&nbsp;
            <span style="display:inline-block;width:34px;border-top:8px solid #333;vertical-align:middle;"></span> Q3&nbsp;&nbsp;
            <span style="display:inline-block;width:34px;border-top:11px solid #333;vertical-align:middle;"></span> Q4
          </div>
          <div><b>지도 기호</b>&nbsp;&nbsp; 🔵 파란 원 = 재질·배터리 생산/공급 위치&nbsp;&nbsp;
            🟢 초록 원 = 차량 또는 배터리팩 조립 위치&nbsp;&nbsp; 🛒 주황 아이콘 = 프랑스 수요시장</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "해상+도로는 생산지→출발항의 도로, 국제 해상운송, 도착항→조립지의 도로를 하나의 복합경로로 합산한 것입니다. "
        "예: 한국 모듈 공장→부산항(도로)→유럽 항만(해상)→독일 조립지(도로). 항공+철도도 같은 방식으로 국제 항공구간과 내륙 철도구간을 합친 경로입니다. "
        "지도 선은 전체 복합경로를 하나의 곡선으로 표시하며 실제 도로·항로의 정확한 형상은 아닙니다."
    )

# -----------------------------------------------------------------------------
# Mapping and charts
# -----------------------------------------------------------------------------
def _bezier_curve_points(start: Tuple[float, float], end: Tuple[float, float], bend: float, steps: int = 30):
    lat1, lon1 = float(start[0]), float(start[1])
    lat2, lon2 = float(end[0]), float(end[1])
    dx = lon2 - lon1
    dy = lat2 - lat1
    length = max(math.hypot(dx, dy), 1e-9)
    mid_lon = (lon1 + lon2) / 2.0
    mid_lat = (lat1 + lat2) / 2.0
    norm_x = -dy / length
    norm_y = dx / length
    control_lon = mid_lon + norm_x * bend * length
    control_lat = mid_lat + norm_y * bend * length
    points = []
    for i in range(steps + 1):
        t = i / steps
        lon = (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * control_lon + t ** 2 * lon2
        lat = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * control_lat + t ** 2 * lat2
        points.append((lat, lon))
    return points


def _route_bend(route_kind: str, transport_mode_index: int, material_id: str, supplier_index: int, plant_index: int) -> float:
    base_by_mode = {1: -0.08, 2: 0.08, 3: -0.16, 4: 0.16, 5: -0.24, 6: 0.24}
    material_adjust = {"steel": -0.02, "aluminum": 0.02, "other": -0.04, "battery": 0.04, "finished": 0.0}
    sign = 1.0 if ((supplier_index + plant_index) % 2 == 0) else -1.0
    base = base_by_mode.get(int(transport_mode_index), 0.1)
    if route_kind == "final":
        base = 0.6 * base
        sign = 1.0 if (plant_index % 2 == 0) else -1.0
        material_id = "finished"
    return sign * (base + material_adjust.get(str(material_id), 0.0))


def _add_route_label(supply_map, location: Tuple[float, float], transport_label: str, quartile: Optional[str] = None):
    label_text = transport_label if quartile is None else f"{transport_label} · {quartile}"
    html = f"""
    <div style="background: rgba(255,255,255,0.92); border: 1px solid #666; border-radius: 4px;
                padding: 1px 4px; font-size: 10px; white-space: nowrap; color: #222;">{label_text}</div>
    """
    import folium
    folium.Marker(
        location,
        icon=folium.DivIcon(html=html, icon_size=(140, 16), icon_anchor=(40, 8)),
    ).add_to(supply_map)


def build_supply_map(result: Dict, height: int = 620):
    # Heavy visualization libraries are imported only when the user asks for
    # one map. This materially lowers idle-process memory.
    import folium
    from folium.plugins import Fullscreen

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

    width_by_quartile = {"Q1": 2.5, "Q2": 5.0, "Q3": 8.0, "Q4": 11.0}
    if not route_df.empty:
        for _, row in route_df.iterrows():
            s = plants.iloc[int(row["supplier_index"]) - 1]
            p = plants.iloc[int(row["plant_index"]) - 1]
            material = str(row["material_id"])
            t = int(row["transport_mode_index"])
            quartile = str(row.get("quartile", ""))
            curve = _bezier_curve_points(
                (float(s["latitude"]), float(s["longitude"])),
                (float(p["latitude"]), float(p["longitude"])),
                _route_bend("raw", t, material, int(row["supplier_index"]), int(row["plant_index"])),
                steps=28,
            )
            folium.PolyLine(
                curve,
                color=MATERIAL_COLOR.get(material, "#555555"),
                weight=width_by_quartile.get(quartile, 4.0),
                opacity=0.82,
                dash_array=TRANSPORT_DASH.get(t),
                tooltip=(
                    f"{row['material_name']} | {row['supplier_location']} → {row['plant_location']} | "
                    f"{row['transport_mode_ko']} | {row['flow_amount']:,.1f} | {quartile}"
                ),
            ).add_to(supply_map)

    if not final_df.empty:
        final_agg = final_df.groupby(
            ["plant_index", "plant_location", "transport_mode_index", "transport_mode_ko"], as_index=False
        )["vehicle_equivalents"].sum()
        final_agg = assign_flow_quartiles(final_agg, "vehicle_equivalents")
        for _, row in final_agg.iterrows():
            p = plants.iloc[int(row["plant_index"]) - 1]
            quartile = str(row.get("quartile", ""))
            curve = _bezier_curve_points(
                (float(p["latitude"]), float(p["longitude"])),
                (market_lat, market_lon),
                _route_bend("final", int(row["transport_mode_index"]), "finished", int(row["plant_index"]), 999),
                steps=24,
            )
            folium.PolyLine(
                curve,
                color=MATERIAL_COLOR["finished"],
                weight=width_by_quartile.get(quartile, 4.0),
                opacity=0.72,
                dash_array=TRANSPORT_DASH.get(int(row["transport_mode_index"])),
                tooltip=(
                    f"완제품 | {row['plant_location']} → 프랑스 | {row['transport_mode_ko']} | "
                    f"{row['vehicle_equivalents']:,.1f}대 등가량 | {quartile}"
                ),
            ).add_to(supply_map)

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


def emissions_comparison_dataframe(results: Mapping[Tuple[str, str], Dict]) -> pd.DataFrame:
    rows=[]
    for scenario_id in ["S1","S2","S3"]:
        line=results.get((scenario_id,"line"),{})
        modular=results.get((scenario_id,"modular"),{})
        if line.get("status") in {"OPTIMAL","FEASIBLE"} and modular.get("status") in {"OPTIMAL","FEASIBLE"}:
            line_e=float(line.get("total_emissions_kgco2",0.0))
            mod_e=float(modular.get("total_emissions_kgco2",0.0))
            rows.append({
                "scenario_id":scenario_id,
                "scenario_name":SCENARIO_SHORT[scenario_id],
                "line_emissions_kgco2":line_e,
                "modular_emissions_kgco2":mod_e,
                "modular_to_line_emissions_ratio":mod_e/line_e if line_e else np.nan,
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
def render_result_map(result: Dict, key: str, height: int = 650):
    if result.get("status") not in {"OPTIMAL", "FEASIBLE"}:
        st.warning(f"{result.get('status')}: {result.get('message')}")
        return
    from streamlit_folium import st_folium

    supply_map = build_supply_map(result, height=height)
    st_folium(supply_map, width=None, height=height, key=key)
    del supply_map
    gc.collect()



def render_solver_metrics(result: Dict):
    status = str(result.get("status", "NOT_RUN"))
    if status not in {"OPTIMAL", "FEASIBLE"}:
        st.error(f"{status}: {result.get('message', '해를 찾지 못했습니다.')}")
        if status == "INFEASIBLE":
            st.caption(
                "INFEASIBLE은 변수·제약조건 수가 많아서 계산능력을 넘었다는 뜻이 아니라, 현재 입력값과 모든 제약조건을 동시에 만족하는 해가 없다고 solver가 판정했다는 뜻입니다. "
                "계산시간 초과나 수행 실패라면 보통 NOT_SOLVED, ABNORMAL 또는 시간제한 상태가 표시됩니다. S1·S2가 같은 규모에서 계산되고 S3만 실패한다면 강화된 탄소상한이 원인일 가능성이 높습니다."
            )
        diagnosis = result.get("feasibility_diagnosis")
        if isinstance(diagnosis, dict) and diagnosis.get("status") == "DIAGNOSED":
            st.caption(
                "현재 PDF 기반 국가별 생산·운송 배출계수와 손실률을 그대로 두면, "
                "시나리오 탄소상한을 만족하는 해가 존재하지 않을 수 있습니다. "
                "아래 표는 탄소상한을 제거한 뒤 총배출량을 최소화했을 때의 이론적 최저 배출량입니다."
            )
            with st.expander("왜 INFEASIBLE 인지 보기", expanded=False):
                st.markdown("**차종별 최소 평균배출량 vs 시나리오 상한**")
                class_df = diagnosis.get("class_average_minimum_emissions")
                if isinstance(class_df, pd.DataFrame):
                    st.dataframe(class_df, hide_index=True, use_container_width=True)
                st.markdown("**트림별 최소 배출량 vs 시나리오 상한**")
                product_df = diagnosis.get("product_minimum_emissions")
                if isinstance(product_df, pd.DataFrame):
                    st.dataframe(product_df, hide_index=True, use_container_width=True)
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총비용", f"€{float(result['objective_value']):,.0f}")
    c2.metric("총 탄소배출량", f"{float(result['total_emissions_kgco2']):,.0f} kg CO₂-eq")
    c3.metric("계산시간", f"{float(result.get('wall_time_sec', 0)):.2f}초")
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
        st.markdown("#### 라인 생산 방식")
        render_solver_metrics(line)
    with c2:
        st.markdown("#### 모듈 활용 분산 생산 방식")
        render_solver_metrics(modular)

    qtable = quartile_comparison_table(results, scenario_id)
    for col in [MODE_LABEL["line"], MODE_LABEL["modular"]]:
        if col in qtable:
            qtable[col] = qtable[col].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "-")
    st.markdown("#### 부품 운송량 사분위수 비율")
    show_explained_dataframe(
        qtable, f"{SCENARIO_SHORT[scenario_id]} 사분위수 비율",
        "Q1~Q4 각 행은 양의 부품 운송경로를 물량 크기로 4등분한 구간입니다.",
        value_meaning="각 생산방식 열의 값은 전체 부품 운송량 중 해당 Q구간이 차지하는 백분율입니다. Q4가 가장 큰 개별 흐름들입니다.",
    )

    if line.get("objective_value") and modular.get("objective_value"):
        ratio = modular["objective_value"] / line["objective_value"]
        st.caption(f"모듈/라인 총비용 비율: {ratio:.6f}")


def render_overview_tab(tables: Mapping[str, pd.DataFrame]):
    st.header("전기자동차 공급망·탄소 최적화 SaaS 개요")
    st.markdown(
        """
이 SaaS는 여러 국가에서 **소형·중형·대형 전기자동차의 미드·롱 트림**을 생산하여 프랑스 시장의 제품별 수요를 정확히 충족하도록
국가별 재질 생산량, 배터리 또는 모듈 생산량, 조립량, 운송경로별 물량을 결정합니다.
기업이 차량 종류별 프랑스 보조금 탄소기준을 충족할 수 있도록 생산·운송·조립·완제품 운송에서 발생하는 탄소배출량도 함께 계산하고,
시나리오 ①과 ③에서는 탄소배출량 상한을 제약조건으로 적용합니다. 목적함수는 주어진 수요와 탄소조건을 만족하는 공급망 중 **총비용이 가장 낮은 구조**를 찾는 것입니다.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        input_df = pd.DataFrame([
            ["제품·수요", "차량 종류/트림, 차량질량, 배터리용량, 재질소요량, 프랑스 수요량"],
            ["국가별 생산", "철강·알루미늄·기타 원자재·배터리/모듈 생산비, 생산배출계수, 생산용량"],
            ["국가별 조립", "차체 및 배터리팩 조립비와 조립배출계수"],
            ["운송", "국가별 허용수단, 도로·철도·해상·항공 비용/배출계수, 거리"],
            ["정책 시나리오", "보조금 점수에 대응하는 차량 1대당 탄소상한"],
        ], columns=["Input", "사용자가 입력하거나 CSV에서 불러오는 값"])
        st.markdown("### Input")
        show_explained_dataframe(input_df, "Input 표", "각 행은 최적화에 필요한 입력 데이터 그룹입니다.", value_meaning="오른쪽 값은 해당 입력 그룹에 포함되는 구체적인 필드입니다.")
    with c2:
        output_df = pd.DataFrame([
            ["최적 물량", "국가별 RP·RT·FP·FT와 배터리 모듈/팩 수량"],
            ["비용", "생산비·부품운송비·조립비·팩조립비·완제품운송비"],
            ["탄소배출량", "생산·유입운송·차체조립·팩조립·완제품운송"],
            ["정책 결과", "차량별/차급별 탄소상한 만족 여부와 보조금 점수"],
            ["공급망 시각화", "선 색상·모양·굵기로 표현한 모든 계산 조합의 지도"],
        ], columns=["Output", "SaaS가 계산하여 제공하는 결과"])
        st.markdown("### Output")
        show_explained_dataframe(output_df, "Output 표", "각 행은 SaaS가 출력하는 결과 그룹입니다.", value_meaning="오른쪽 값은 해당 결과가 포함하는 세부 지표입니다.")

    st.markdown("### 수학적 최적화 프레임워크의 일반적인 큰틀")
    st.markdown(
        """
1. **대안 정의**: 어떤 국가가 재질·배터리/모듈을 생산하고, 어떤 국가가 차량과 배터리팩을 조립할 수 있는지를 후보로 둡니다.  
2. **의사결정변수 설정**: 국가별 생산량, 공급지→조립지 운송량, 조립량, 조립지→프랑스 출하량을 변수로 둡니다.  
3. **목적함수 설정**: 모든 생산·운송·조립 비용의 합을 최소화합니다.  
4. **물량수지와 수요 제약**: 생산량과 출고량, 조립지 유입량과 필요량, 조립량과 출하량을 일치시키고 프랑스 수요를 정확히 충족합니다.  
5. **운영 제약**: 국가별 생산용량과 허용 운송경로를 반영합니다.  
6. **환경·정책 제약**: 차량 종류별 생산·운송·조립 탄소배출량을 합산하고 시나리오의 탄소상한을 만족시킵니다.  
7. **최적해 해석**: 비용, 탄소, 국가별 물량, 공급망 경로와 생산방식 차이를 비교합니다.
        """
    )

    st.markdown("### 라인 생산 방식: 3개 Stage")
    st.markdown(
        "각 Stage의 국가 후보는 고정된 한 국가가 아니라 CSV에 포함된 할당 가능한 국가들입니다. 최적화가 비용·탄소·용량·운송조건을 보고 국가별 물량을 배분합니다."
    )
    line_stages=pd.DataFrame([
        ["Stage 1", "재질 생산", "철강, 알루미늄, 철강·알루미늄·배터리를 제외한 기타 원자재를 여러 국가에서 생산", "RP, RT"],
        ["Stage 2", "차량·배터리팩 조립", "Stage 1 재질을 조립하고, 같은 위치에서 완성 배터리팩을 생산·결합하여 전기자동차를 완성", "FP, ZL, battery RP/RT"],
        ["Stage 3", "프랑스 시장", "완성 전기자동차를 프랑스로 보내 제품별 수요를 정확히 충족", "FT, 수요등식"],
    ],columns=["단계","Stage의 역할","사용자가 머릿속에 그릴 공정","관련 코드 변수"])
    show_explained_dataframe(line_stages,"라인 생산 Stage 표","각 행은 원재료 생산부터 프랑스 수요 충족까지의 공급망 Stage입니다.",value_meaning="각 셀은 Stage의 공정 역할과 이를 표현하는 코드 변수를 뜻합니다.")
    st.info(
        "현재 코드 구현과 일치합니다. 라인 방식은 배터리 생산지와 차량 조립지의 인덱스를 같게 강제하고(s=p), 완성팩의 국가 간 운송을 허용하지 않습니다. "
        "다만 Stage 2는 실제 공장 내부의 세부 순서를 따로 시뮬레이션하지 않고, 배터리 생산수지와 차량 조립수지를 결합한 네트워크 제약으로 표현합니다."
    )

    st.markdown("### 모듈 활용 분산 생산 방식: 3개 Stage")
    modular_stages=pd.DataFrame([
        ["Stage 1", "재질·모듈 생산", "철강, 알루미늄, 기타 원자재와 10kWh·5kWh 배터리 모듈을 여러 국가에서 생산", "RP, ZM, ZS, RT"],
        ["Stage 2", "차량·배터리팩 조립", "비배터리 재질을 조립하고 여러 생산국가에서 받은 모듈을 완성 배터리팩으로 조립한 뒤 전기자동차에 결합", "FP, battery RT"],
        ["Stage 3", "프랑스 시장", "완성 전기자동차를 프랑스로 보내 제품별 수요를 정확히 충족", "FT, 수요등식"],
    ],columns=["단계","Stage의 역할","사용자가 머릿속에 그릴 공정","관련 코드 변수"])
    show_explained_dataframe(modular_stages,"모듈 분산 생산 Stage 표","각 행은 모듈 생산부터 프랑스 수요 충족까지의 공급망 Stage입니다.",value_meaning="Stage 1의 모듈 생산국가와 Stage 2의 팩 조립국가는 서로 달라도 됩니다.")
    st.info(
        "현재 코드 구현과 일치합니다. 모듈 방식은 s와 p를 독립적으로 허용하고, 10/5kWh 모듈의 생산·운송수지와 p에서의 차량별 완성팩 용량 충족식을 적용합니다. "
        "또한 현재 코드에서는 모듈→팩 조립 배출량을 배터리팩이 실제로 조립되는 국가 p의 `assembly_ef_kgco2_per_kg`에 배터리 질량을 곱해 계산합니다."
    )

    comparison=pd.DataFrame([
        ["Stage 1의 배터리 형태", "완성팩을 Stage 2 위치에서 생산", "10/5kWh 모듈을 여러 국가에서 생산"],
        ["Stage 1과 Stage 2 국가", "배터리는 동일 위치(s=p)", "모듈 생산국가 s와 팩 조립국가 p가 달라도 됨"],
        ["배터리 국제운송", "완성팩 국제운송 없음", "모듈 상태 국제운송 가능"],
        ["팩 조립 배출계수", "별도 팩 조립 확장항 없음", "팩 조립국가 p의 국가별 조립 EF 적용"],
    ],columns=["핵심차이","라인 생산","모듈 활용 분산 생산"])
    st.markdown("### 두 방식의 핵심차이")
    show_explained_dataframe(comparison,"생산방식 비교 표","각 행은 두 생산방식을 구분하는 하나의 구조적 기준입니다.",value_meaning="라인과 모듈 열의 값은 동일 기준에 대한 각 방식의 구현 차이입니다.")


def run_app():
    st.set_page_config(
        page_title="PDF 기반 전기차 공급망 Route LP",
        page_icon="🚗",
        layout="wide",
    )
    apply_global_font_scale()
    st.title("탄소배출 기반 전기차 공급망 최적화")
    st.caption(
        f"build: {APP_BUILD} · package: {APP_PACKAGE_ID} · "
        "PDF 국가별 생산·운송 배출계수와 국가별 허용 복합운송경로를 적용한 연속 LP"
    )

    defaults = load_default_tables()
    with st.sidebar:
        st.header("데이터")
        use_defaults = st.checkbox("패키지의 PDF 기준 CSV 사용", value=True)
        uploads = st.file_uploader("수정 CSV 업로드", type="csv", accept_multiple_files=True)
        uploaded = load_uploaded_tables(uploads)
        tables = dict(defaults) if use_defaults else {}
        tables.update(uploaded)
        st.download_button(
            "PDF 기준 CSV·참조 LP 다운로드",
            make_data_zip(),
            file_name="pdf_route_data_and_reference_lp.zip",
            mime="application/zip",
        )
        st.divider()
        st.write("**LP relaxation 원칙**")
        st.write("RP·RT·FP·FT·ZM·ZS·ZL은 모두 비음이 아닌 연속변수")
        st.write("운송수단은 확률추첨하지 않고, 허용된 경로별 RT·FT 물량을 LP가 직접 배분")
        st.divider()
        if st.button("결과 메모리 초기화", use_container_width=True):
            for key in ("xpress_results", "score_sensitivity", "results_zip_bytes"):
                st.session_state.pop(key, None)
            gc.collect()
            st.success("세션 결과와 생성된 ZIP을 비웠습니다.")

    errors = validate_tables(tables)
    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    tabs = st.tabs([
        "1. SaaS·최적화 프레임워크 개요",
        "2. PDF·경로 입력 데이터",
        "3. 최적화 실행",
        "4. 포스터형 최적화 결과",
        "5. 수학모형·코드 매핑",
        "6. 포스터 기준 비교",
    ])

    with tabs[0]:
        render_overview_tab(tables)

    with tabs[1]:
        st.header("PDF 기반 탄소계수 및 국가별 운송경로 입력 데이터")
        st.info(
            "공급지와 조립지 후보는 기존 24개 위치를 유지합니다. 국가별 허용수단은 country_transport_rules.csv에서 여러 개를 동시에 허용할 수 있으며, 운송수단별 물량은 최적화변수 RT·FT로 결정됩니다."
        )
        display_names = {
            "products.csv": "제품 6종",
            "demand.csv": "프랑스 수요",
            "raw_material_suppliers.csv": "국가별 생산계수·비용",
            "assembly_locations.csv": "국가별 조립계수·비용",
            "transport_parameters.csv": "PDF 운송수단·지역별 계수",
            "country_transport_rules.csv": "국가별 허용 운송수단",
            "material_parameters.csv": "재질 손실률",
            "markets.csv": "수요지",
            "scenarios.csv": "시나리오",
            "xpress_model_metadata.csv": "모형 상수·가정",
        }
        table_guides = {
            "products.csv": ("각 행은 하나의 차량 트림입니다.", "값은 트림별 질량·배터리용량·재질소요량입니다."),
            "demand.csv": ("각 행은 제품-시장 수요 조합입니다.", "demand_units는 프랑스가 요구하는 차량 대수입니다."),
            "raw_material_suppliers.csv": ("각 행은 재질-공급국가 조합입니다.", "생산비·국가별 생산EF·용량 값이 해당 조합에 적용됩니다."),
            "assembly_locations.csv": ("각 행은 하나의 조립 후보국가/위치입니다.", "조립비와 조립EF는 그 위치에서 조립되는 질량에 적용됩니다."),
            "transport_parameters.csv": ("각 행은 운송수단-지역 조합입니다.", "비용과 EF는 kg·km 단위 운송에 적용됩니다."),
            "country_transport_rules.csv": ("각 행은 국가별 운송수단 허용규칙입니다.", "0은 불허, 1은 허용이며 최적화변수가 아닌 고정 입력입니다."),
            "material_parameters.csv": ("각 행은 하나의 재질입니다.", "loss_rate는 순사용량에서 총생산량으로 환산할 때 사용합니다."),
            "markets.csv": ("각 행은 최종 수요시장입니다.", "위도·경도와 최소거리 값은 완제품 운송거리에 사용됩니다."),
            "scenarios.csv": ("각 행은 하나의 정책 시나리오입니다.", "점수와 차급별 kg CO₂-eq/대 상한이 최적화 제약에 사용됩니다."),
            "xpress_model_metadata.csv": ("각 행은 모형의 상수 또는 가정입니다.", "값은 모형 크기·인덱스·기준 설정을 설명합니다."),
        }
        subtabs = st.tabs(list(display_names.values()))
        for sub, filename in zip(subtabs, display_names):
            with sub:
                row_guide, value_guide = table_guides[filename]
                show_explained_dataframe(
                    tables[filename], display_names[filename], row_guide,
                    value_meaning=value_guide,
                )

    with tabs[2]:
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
            "이 v8 모형은 기존 Xpress LP를 그대로 복제한 모형이 아닙니다. 물량수지·수요등식·생산방식 구조는 유지하지만, "
            "운송수단 이진 선택과 Big-M를 제거하고 PDF 기반 국가·지역별 복합경로 물량배분 LP로 수정했습니다."
        )
        st.info(
            "Community Cloud 메모리 보호: 계수는 compact array로 구성하고, solver 전달 후 즉시 해제하며, "
            "세션에는 전체 해 벡터/LPModel이 아닌 요약과 양의 경로만 저장합니다. 지도와 ZIP은 요청할 때 한 개씩 생성합니다."
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("선택 조합 실행", type="primary", use_container_width=True):
                with st.spinner("PDF 기반 국가별 복합운송 연속 LP를 OR-Tools GLOP으로 계산하는 중입니다..."):
                    try:
                        res = solve_case(
                            tables, scenario_id, production_mode, cap_application, int(time_limit)
                        )
                        st.session_state.setdefault("xpress_results", {})[(scenario_id, production_mode)] = res
                        st.session_state.pop("results_zip_bytes", None)
                        gc.collect()
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
                    st.session_state.pop("results_zip_bytes", None)
                    gc.collect()
                    progress.progress(i / len(cases))
                status_box.success("6개 조합 계산이 완료되었습니다. 결과에는 전체 LP/해 벡터가 저장되지 않습니다.")

        st.markdown("### 보조금 점수 민감도")
        with st.expander("선택 생산방식의 점수 민감도 실행", expanded=False):
            c1, c2, c3 = st.columns(3)
            min_score = c1.number_input("최소 점수", 0.0, 80.0, 55.0, 1.0)
            max_score = c2.number_input("최대 점수", 0.0, 80.0, 70.0, 1.0)
            n_points = c3.number_input("점수 개수", 3, MAX_SENSITIVITY_POINTS, 3, 1)
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
                    gc.collect()
                    p.progress(i / len(scores))
                st.session_state["score_sensitivity"] = pd.DataFrame(rows)

    with tabs[3]:
        st.header("포스터형 최적화 결과")
        results = st.session_state.get("xpress_results", {})
        if not results:
            st.info("3번 탭에서 선택 조합 또는 6개 조합을 실행하세요.")
        else:
            for scenario in ["S1", "S2", "S3"]:
                render_poster_scenario(results, scenario)
                st.divider()

            st.markdown("### 분석 및 결론")
            ratio_df = cost_ratio_dataframe(results)
            emission_df = emissions_comparison_dataframe(results)
            if not ratio_df.empty:
                st.markdown("#### 시나리오별 총비용 비교")
                cost_chart = ratio_df.set_index("scenario_name")[["line_cost", "modular_cost"]]
                st.bar_chart(cost_chart, use_container_width=True)
                show_explained_dataframe(
                    ratio_df, "비용 비교 결과",
                    "각 행은 하나의 정책 시나리오입니다.",
                    value_meaning="line_cost와 modular_cost는 생산방식별 총비용이며, 비율이 1보다 작으면 모듈 방식 비용이 더 낮습니다.",
                )
            if not emission_df.empty:
                st.markdown("#### 시나리오별 총탄소배출량 비교")
                emission_chart = emission_df.set_index("scenario_name")[["line_emissions_kgco2", "modular_emissions_kgco2"]]
                st.bar_chart(emission_chart, use_container_width=True)
                show_explained_dataframe(
                    emission_df, "탄소배출량 비교 결과",
                    "각 행은 두 생산방식 모두 feasible 해를 가진 하나의 시나리오입니다.",
                    value_meaning="배출량은 공급망 전체 kg CO₂-eq이며, 비율이 1보다 작으면 모듈 방식 배출량이 더 낮습니다.",
                )

            valid_results = {
                key: value for key, value in results.items()
                if value.get("status") in {"OPTIMAL", "FEASIBLE"}
            }

            if valid_results:
                st.markdown("### 계산된 모든 공급망 지도")
                render_map_legend_outside()
                st.caption("아래 지도는 현재 세션에서 최적해가 계산된 모든 시나리오×생산방식 조합입니다. INFEASIBLE 조합은 경로가 없으므로 지도에 표시되지 않습니다.")
                if st.checkbox("계산된 모든 공급망 지도 한꺼번에 표시", value=False, key="show_all_maps"):
                    ordered_keys=[k for k in [(s,m) for s in ["S1","S2","S3"] for m in ["line","modular"]] if k in valid_results]
                    for idx in range(0,len(ordered_keys),2):
                        cols=st.columns(2)
                        for col,key in zip(cols,ordered_keys[idx:idx+2]):
                            with col:
                                st.markdown(f"#### {SCENARIO_SHORT[key[0]]} · {MODE_LABEL[key[1]]}")
                                render_result_map(valid_results[key],f"all_map_{key[0]}_{key[1]}",height=560)
                st.markdown("#### 지도 해석")
                st.markdown(
                    "파란 원은 재질·배터리/모듈을 생산하는 공급 위치, 초록 원은 차량 또는 배터리팩이 조립되는 위치입니다. "
                    "하나의 위치가 두 역할을 모두 수행하면 파란 원과 초록 원이 같은 좌표에 겹칠 수 있습니다. 선에 마우스를 올리면 운송대상, 출발지·도착지, 운송수단, 물량과 Q구간을 확인할 수 있습니다."
                )
                st.markdown("### 결과 파일")
                if st.button("전체 결과 ZIP 생성", key="build_results_zip"):
                    with st.spinner("CSV 결과를 ZIP으로 묶는 중입니다..."):
                        st.session_state["results_zip_bytes"] = results_zip(valid_results)
                        gc.collect()
                zip_bytes = st.session_state.get("results_zip_bytes")
                if isinstance(zip_bytes, (bytes, bytearray)):
                    st.download_button(
                        "전체 결과 ZIP 다운로드",
                        data=zip_bytes,
                        file_name="pdf_country_route_lp_results.zip",
                        mime="application/zip",
                    )

            sensitivity = st.session_state.get("score_sensitivity")
            if isinstance(sensitivity, pd.DataFrame) and not sensitivity.empty:
                st.markdown("### 보조금 점수 민감도")
                st.dataframe(sensitivity, hide_index=True, use_container_width=True)
                valid = sensitivity[sensitivity["status"].isin(["OPTIMAL", "FEASIBLE"])]
                if not valid.empty:
                    st.line_chart(
                        valid.set_index("minimum_score")[["total_cost_eur"]],
                        use_container_width=True,
                    )

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

    with tabs[4]:
        st.header("수학모형·코드 매핑")
        st.info("이 탭은 현재 코드의 집합, 변수, 파라미터, 목적함수, 제약조건과 탄소계산을 설명합니다. 수학모형이 바뀔 때마다 이 탭도 함께 갱신합니다.")

        render_formula_table(
            "5.1 집합(Set) 정의",
            ["수식", "이름", "직관적 의미"],
            [
                (r"F", "제품 집합", "소형·중형·대형 차량의 미드·롱 트림"),
                (r"R", "재질 집합", "철강, 알루미늄, 기타 원자재, 배터리/모듈"),
                (r"S", "생산지 집합", "재질·배터리/모듈을 생산할 수 있는 국가/위치"),
                (r"P", "조립지 집합", "차체와 배터리팩을 조립할 수 있는 국가/위치"),
                (r"K", "운송경로 집합", "도로, 철도, 해상+도로, 해상+철도, 항공+도로, 항공+철도"),
                (r"G", "차급 집합", "small 및 standard 정책 차급"),
            ],
            [0],
            "각 행은 수학모형에서 반복 인덱스로 사용하는 하나의 집합입니다.",
            "수식은 집합기호이고, 이름과 의미는 그 기호에 포함되는 실제 대상을 설명합니다.",
        )

        render_formula_table(
            "5.2 결정변수와 변수종류",
            ["수식", "변수종류", "직관적 의미", "단위/정의역"],
            [
                (r"RP_{frs}", "연속변수", "제품 f용 재질 r을 생산지 s에서 생산하는 순사용 가능량", r"kg\;or\;kWh,\;\ge0"),
                (r"RT_{frspk}", "연속변수", "생산지 s에서 조립지 p로 경로 k를 통해 보내는 재질/모듈 물량", r"kg\;or\;kWh,\;\ge0"),
                (r"FP_{fp}", "연속변수", "조립지 p에서 완성하는 제품 f의 차량 등가량", r"vehicle\;equivalent,\;\ge0"),
                (r"FT_{fpk}", "연속변수", "조립지 p에서 프랑스로 경로 k를 통해 보내는 차량 등가량", r"vehicle\;equivalent,\;\ge0"),
                (r"ZM_{fsp}", "연속변수", "모듈 방식의 10kWh 모듈 등가개수", r"module\;equivalent,\;\ge0"),
                (r"ZS_{fsp}", "연속변수", "모듈 방식의 5kWh 모듈 등가개수", r"module\;equivalent,\;\ge0"),
                (r"ZL_{fpp}", "연속변수", "라인 방식에서 조립지 p와 동일위치에서 생산되는 완성팩 등가개수", r"pack\;equivalent,\;\ge0"),
            ],
            [0,3],
            "각 행은 solver가 값을 결정하는 하나의 의사결정변수입니다.",
            "현재 코드에는 연속변수만 있고 정수·이진 결정변수는 없습니다. 따라서 차량·모듈·팩 수량도 LP 연속 근사값입니다.",
        )
        classification=pd.DataFrame([
            ["연속변수","RP, RT, FP, FT, ZM, ZS, ZL","현재 solver가 실제 최적화"],
            ["정수변수","없음","차량·모듈·팩 개수를 정수로 강제하지 않음"],
            ["이진변수","없음","국가 선택 또는 단일 운송수단 선택을 0/1로 결정하지 않음"],
            ["고정 0/1 파라미터","A(raw), A(fin), delta(s,p)","경로 허용과 동일위치 여부를 미리 정한 입력값"],
        ],columns=["분류","해당 항목","현재 구현 의미"])
        show_explained_dataframe(classification,"변수종류 분류 표","각 행은 변수 또는 파라미터의 수학적 종류입니다.",value_meaning="고정 0/1 파라미터는 값이 0 또는 1이지만 solver가 선택하는 이진변수가 아닙니다.")

        render_formula_table(
            "5.3 주요 파라미터",
            ["수식", "직관적 의미", "단위/출처"],
            [
                (r"D_f", "제품 f의 프랑스 수요", "대, demand.csv"),
                (r"a_{fr}", "제품 f 한 대에 필요한 재질 r의 양", "kg/대 또는 kWh/대, products.csv"),
                (r"B_f", "제품 f의 완성 배터리팩 용량", "kWh/대"),
                (r"M_f^{bat}", "제품 f의 배터리 질량", "kg/대"),
                (r"M_f^{NB}", "제품 f의 비배터리 차체 질량", "kg/대"),
                (r"c^{RP}_{rs}", "재질/모듈의 국가별 생산비", "€/kg 또는 €/kWh"),
                (r"EF^{RP}_{rs}", "생산국가 s의 재질/모듈 생산 배출계수", "kg CO2-eq/kg 또는 /kWh"),
                (r"EF^{ASM}_{p}", "조립국가 p의 조립 배출계수", "kg CO2-eq/kg"),
                (r"A^{raw}_{spk},A^{fin}_{pk}", "운송경로 허용 여부", "0/1 고정 입력"),
                (r"\delta_{sp}", "생산지 s와 조립지 p의 동일위치 여부", "s=p이면 1"),
                (r"Cap_{rs}", "재질 r·생산지 s의 최대 생산용량", "kg 또는 kWh"),
                (r"\bar E_g^{cap}", "차급 g의 차량 1대당 탄소상한", "kg CO2-eq/대"),
            ],
            [0],
            "각 행은 의사결정변수의 비용·배출·용량·수요를 정하는 고정 입력 파라미터입니다.",
            "수식 셀은 기호, 의미 셀은 사용자 해석, 단위/출처 셀은 값의 측정단위와 CSV 위치입니다.",
        )

        st.markdown("### 5.4 목적함수")
        st.latex(r"\min Z=\sum c^{RP}RP+\sum c^{RT}RT+\sum c^{BODY}FP+\mathbf{1}_{mod}\sum c^{PACK}FP+\sum c^{FT}FT")
        objective_rows=[
            (r"\sum c^{RP}RP","재질·배터리/모듈 생산비","국가별 생산량 × 단위생산비"),
            (r"\sum c^{RT}RT","부품·모듈 운송비","운송질량 × 복합경로 비용"),
            (r"\sum c^{BODY}FP","비배터리 차체 조립비","비배터리 질량 × 조립국가 비용"),
            (r"\mathbf{1}_{mod}\sum c^{PACK}FP","모듈 방식 팩 조립비","배터리 질량 × 팩 조립국가 비용; 모듈 방식에만 적용"),
            (r"\sum c^{FT}FT","완제품 운송비","차량 전체질량 × 프랑스까지의 경로비용"),
        ]
        render_formula_table("목적함수 구성요소",["수식항","비용요소","사용자 해석"],objective_rows,[0],"각 행은 총비용 목적함수를 구성하는 하나의 비용항입니다.","모든 수식항을 합한 값이 solver가 최소화하는 총비용입니다.")

        st.markdown("### 5.5 제약조건")
        constraints=[
            (r"\sum_{p,k}FT_{fpk}=D_f","수요충족","제품별 프랑스 수요를 정확히 충족"),
            (r"RP_{frs}=\sum_{p,k}RT_{frspk}","공급지 수지","생산량과 모든 조립지로 나가는 출고량 일치"),
            (r"\sum_{s,k}RT_{frspk}=a_{fr}FP_{fp}","조립지 재질수지","들어온 비배터리 재질량과 차량생산 필요량 일치"),
            (r"FP_{fp}=\sum_kFT_{fpk}","조립지 출하수지","조립한 차량수와 프랑스로 보낸 차량수 일치"),
            (r"RT_{f,bat,s,p,k}=0\;(s\ne p)","라인 동일위치","라인 방식에서 배터리 생산지와 조립지 동일"),
            (r"RT_{f,bat,p,p,k_0}=B_fZL_{fpp},\;ZL_{fpp}=FP_{fp}","라인 팩수지","완성팩 용량과 조립차량수 일치"),
            (r"\sum_kRT_{f,bat,s,p,k}=10ZM_{fsp}+5ZS_{fsp}","모듈 생산수지","10/5kWh 모듈 생산량과 출고 kWh 일치"),
            (r"\sum_{s,k}RT_{f,bat,s,p,k}=B_fFP_{fp}","모듈 팩조립수지","p에 들어온 모듈 kWh와 완성차 팩용량 일치"),
            (r"\sum_fRP_{frs}\le Cap_{rs}","생산용량","국가별 공급능력 초과 금지"),
        ]
        render_formula_table("제약조건 식",["수식","제약 이름","직관적 설명"],constraints,[0],"각 행은 feasible 공급망이 반드시 만족해야 하는 하나의 제약식입니다.","등식은 좌우 물량을 정확히 맞추며, 부등식은 최대용량 또는 상한을 넘지 못하게 합니다.")

        st.markdown("### 5.6 탄소배출량 계산요소와 탄소상한")
        st.latex(r"C_f=C_f^{PROD}+C_f^{IN}+C_f^{BODY}+C_f^{PACK}+C_f^{OUT}")
        emission_rows=[
            (r"C_f^{PROD}","생산","철강·알루미늄·기타 원자재·배터리/모듈의 국가별 생산배출"),
            (r"C_f^{IN}","부품·모듈 유입운송","생산지에서 조립지까지의 재질/모듈 운송배출"),
            (r"C_f^{BODY}","비배터리 차체 조립","비배터리 질량 × 조립국가 p의 조립EF"),
            (r"C_f^{PACK}","모듈→완성팩 조립","모듈 방식만: 배터리질량 × 팩이 조립되는 국가 p의 조립EF"),
            (r"C_f^{OUT}","완제품 운송","완성차 전체질량 × 조립지에서 프랑스까지의 운송EF"),
        ]
        render_formula_table("탄소배출량 구성요소",["수식항","공정블록","포함되는 실제 배출"],emission_rows,[0],"각 행은 차량 f의 공급망 탄소발자국을 구성하는 하나의 공정블록입니다.","다섯 항을 모두 합한 값이 차량 f의 총 공급망 탄소배출량입니다.")
        st.latex(r"C_f^{PROD}=\sum_{r,s}\frac{EF^{RP}_{rs}}{1-L_r}RP_{frs}")
        st.markdown("철강과 알루미늄은 손실률 0.3 때문에 순사용량을 0.7로 나누어 총생산량을 계산합니다. 배터리/모듈에는 생산국가 s의 국가별 kg CO₂-eq/kWh가 적용됩니다.")
        st.latex(r"C_f^{IN}=\sum_{r,s,p,k}e^{IN}_{frspk}RT_{frspk}")
        st.markdown("비배터리 재질은 kg 물량을 직접 사용하고, 배터리/모듈 kWh는 제품별 kg/kWh로 환산한 후 운송거리와 운송수단별 EF를 곱합니다.")
        st.latex(r"C_f^{BODY}=\sum_pM_f^{NB}EF_p^{ASM}FP_{fp}")
        st.latex(r"C_f^{PACK}=\mathbf{1}_{mod}\sum_pM_f^{bat}EF_p^{ASM}FP_{fp}")
        st.markdown("현재 코드에서 모듈→팩 조립 배출계수는 배터리팩이 조립되는 국가 p의 `assembly_ef_kgco2_per_kg`입니다. 따라서 생산국가 s가 아니라 실제 팩 조립국가 p의 EF가 적용됩니다.")
        st.latex(r"C_f^{OUT}=\sum_{p,k}M_f^{veh}e^{OUT}_{pk}FT_{fpk}")

        st.markdown("#### 트림별 상한(product-strict)")
        st.latex(r"C_f\le E_{class(f)}^{cap}D_f\qquad\forall f")
        st.markdown("각 미드·롱 트림이 자기 상한을 개별적으로 만족해야 합니다. 한 트림이 초과하면 다른 트림의 낮은 배출량으로 보상할 수 없습니다. PDF의 개별 차량 버전 EC_version 점수화에 더 가까운 방식입니다.")
        st.markdown("#### 차급 평균상한(class-average)")
        st.latex(r"\sum_{f\in g}C_f\le\bar E_g^{cap}\sum_{f\in g}D_f")
        st.markdown("같은 차급의 여러 트림을 수요량으로 가중해 평균 배출량을 제한합니다. 한 트림이 상한을 초과해도 다른 저탄소 트림의 여유로 차급 평균이 기준 이하이면 통과할 수 있습니다. 이 방식은 PDF 직접 규칙이 아니라 포트폴리오 비교를 위해 SaaS가 추가한 옵션입니다.")
        source_df=pd.DataFrame([
            ["PDF 직접 근거","재질·배터리·조립·운송 배출항, 철강/알루미늄 손실률, 차량 버전별 점수곡선"],
            ["SaaS 확장","10/5kWh 모듈 네트워크, 라인 동일위치, 모듈 팩조립항, 차급 평균상한"],
        ],columns=["근거 구분","포함 내용"])
        show_explained_dataframe(source_df,"PDF 근거와 SaaS 확장 표","각 행은 모형 요소의 출처 구분입니다.",value_meaning="PDF 직접 근거와 연구 목적상 추가한 모델링 선택을 구분합니다.")

        st.markdown("### 5.7 코드 매핑")
        code_rows=[
            (r"s=p","라인 배터리 동일위치","battery RT upper bound + diagonal ZL"),
            (r"10ZM+5ZS","모듈 생산","ZM/ZS + battery RT"),
            (r"B_fFP_{fp}","p에서 완성팩 용량 충족","battery RT total equality"),
            (r"EF^{RP}_{rs}","국가별 생산EF","supplier_ef[r,s]"),
            (r"EF^{ASM}_{p}M_f^{bat}FP_{fp}","국가별 팩조립EF","assembly_ef[p] × battery_mass × FP"),
        ]
        render_formula_table("수식-코드 대응",["수식","수학적 역할","코드 구현"],code_rows,[0],"각 행은 하나의 수학적 구조와 이를 구현한 코드 위치입니다.","수식은 사용자에게 보이는 모델 의미이고 코드 구현은 실제 배열·제약 생성 방법입니다.")
        st.caption(f"역사적 Xpress 참조 LP SHA-256: {REFERENCE_LP_SHA256} · v8.7은 3-Stage 설명, 렌더링 수식표, 모든 지도 표시, 탄소 그래프와 INFEASIBLE 진단 설명을 반영합니다.")

    with tabs[5]:
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


if __name__ == "__main__":
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install requirements.txt and run: streamlit run app.py")
    run_app()
