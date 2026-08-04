# DEPLOYMENT_MARKER: PDF_COUNTRY_ROUTE_LP_V8_0
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

APP_BUILD = "pdf-country-route-lp-memory-safe-v8.2"
APP_PACKAGE_ID = "20260804-pdf-route-v8.2"
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
TRANSPORT_DASH = {1: None, 2: "8,5", 3: "3,7", 4: "8,5", 5: "1,5", 6: "1,5"}

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
                        if not route["raw_allowed"][s, p, t]:
                            ub[rt_idx] = 0.0
                            continue
                        c[rt_idx] = route["raw_cost_per_kg"][s, p, t] * mass_per_flow_unit
                        emission_rt[rt_idx - layout.off_rt] = route["raw_ef_per_kg"][s, p, t] * mass_per_flow_unit

        for p in range(layout.P):
            fp_idx = layout.fp(f, p)
            c[fp_idx] = float(product["nonbattery_mass_kg"]) * float(plants.iloc[p]["assembly_cost_eur_per_kg"])
            emission_fp[fp_idx - layout.off_fp] = (
                float(product["nonbattery_mass_kg"]) * float(plants.iloc[p]["assembly_ef_kgco2_per_kg"])
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

    rows = LinearConstraintBuilder(layout.n_vars)

    # 1) Exact market-demand fulfillment. Equality prevents both shortage and unexplained surplus.
    for f in range(layout.F):
        cols = [layout.ft(f, p, t) for p in range(layout.P) for t in range(layout.T)]
        rows.add_eq(cols, [1.0] * len(cols), demand_values[f])

    # 2) Battery production structure. Mode-specific RT quantities sum to the battery requirement.
    if production_mode == "modular":
        for f in range(layout.F):
            for s in range(layout.S):
                for p in range(layout.P):
                    cols = [layout.rt(f, 3, s, p, t) for t in range(layout.T)]
                    vals = [1.0] * layout.T
                    cols.extend([layout.z1(f, s, p), layout.z2(f, s, p)])
                    vals.extend([-10.0, -5.0])
                    rows.add_eq(cols, vals, 0.0)
        for f in range(layout.F):
            battery_kwh = float(products.iloc[f]["battery_kwh"])
            for p in range(layout.P):
                cols = [layout.rt(f, 3, s, p, t) for s in range(layout.S) for t in range(layout.T)]
                vals = [1.0] * (layout.S * layout.T)
                cols.append(layout.fp(f, p))
                vals.append(-battery_kwh)
                rows.add_eq(cols, vals, 0.0)
    else:
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
            _add_route_label(supply_map, curve[len(curve)//2], str(row["transport_mode_ko"]), quartile)

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
            _add_route_label(supply_map, curve[len(curve)//2], str(row["transport_mode_ko"]), quartile)

    legend = """
    <div style="position: fixed; bottom: 24px; left: 24px; z-index:9999; background:white;
                border:1px solid #777; border-radius:6px; padding:8px 10px; font-size:12px; line-height:1.45;">
      <b>부품/완제품 공급망 지도 범례</b><br>
      <span style="color:#e41a1c">━</span> 철강 &nbsp;
      <span style="color:#ff9f1c">━</span> 알루미늄 &nbsp;
      <span style="color:#238b45">━</span> 기타 원자재 &nbsp;
      <span style="color:#2171b5">━</span> 배터리 &nbsp;
      <span style="color:#6a3d9a">━</span> 완제품<br>
      <span style="display:inline-block; width:30px; border-top:3px solid #555;"></span> 도로 &nbsp;
      <span style="display:inline-block; width:30px; border-top:3px dashed #555;"></span> 철도 &nbsp;
      <span style="display:inline-block; width:30px; border-top:3px dashed #555;"></span> 해상+육상 &nbsp;
      <span style="display:inline-block; width:30px; border-top:3px dotted #555;"></span> 항공+육상<br>
      선 굵기: Q1(가장 얇음) &lt; Q2 &lt; Q3 &lt; Q4(가장 굵음)<br>
      곡선 라벨: 운송수단 종류 · 사분위수
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
def render_result_map(result: Dict, key: str, height: int = 620):
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
    st.dataframe(qtable, hide_index=True, use_container_width=True)

    if line.get("objective_value") and modular.get("objective_value"):
        ratio = modular["objective_value"] / line["objective_value"]
        st.caption(f"모듈/라인 총비용 비율: {ratio:.6f}")


def render_overview_tab(tables: Mapping[str, pd.DataFrame]):
    st.header("이 SaaS가 하는 일")
    st.markdown(
        """
이 앱은 **프랑스 전기차 보조금 탄소기준**과 **공급망 최적화**를 함께 다루는
**탄소·비용 기반 공급망 의사결정 SaaS 프로토타입**입니다.

사용자는 제품, 수요, 공급지, 조립지, 국가별 허용 운송수단, PDF 기반 배출계수, 시나리오를 입력하고,
앱은 **라인 생산 방식**과 **모듈 활용 분산 생산 방식**에 대해 총비용과 총탄소배출량을 최소화하는 공급망 구조를 계산합니다.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 1) 입력 데이터")
        input_df = pd.DataFrame([
            ["제품 정보", "전기차 6종의 질량, 배터리 용량, 차급"],
            ["수요 정보", "프랑스 시장의 제품별 수요"],
            ["공급지 정보", "국가별 생산비, 생산배출계수, 생산용량"],
            ["조립지 정보", "국가별 조립비, 조립배출계수"],
            ["운송 정보", "국가별 허용 운송수단과 운송계수"],
            ["시나리오", "현행 기준 / 기준 없음 / 강화 기준"],
        ], columns=["입력 항목", "설명"])
        st.dataframe(input_df, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("### 2) 주요 출력")
        output_df = pd.DataFrame([
            ["총비용", "생산·운송·조립 비용의 총합"],
            ["총탄소배출량", "생산·운송·조립 배출량의 총합"],
            ["공급망 구조", "공급지 → 조립지 → 프랑스 시장 흐름"],
            ["차량별 결과", "제품별 kg CO₂-eq/대, 보조금 점수, 상한 만족 여부"],
            ["지도 시각화", "운송수단·Q1~Q4 흐름 굵기·공급망 곡선"],
            ["포스터형 비교", "시나리오별 라인 생산 vs 모듈 분산 생산 비교"],
        ], columns=["출력 항목", "설명"])
        st.dataframe(output_df, hide_index=True, use_container_width=True)

    st.markdown("### 3) 최적화 프레임워크 큰틀")
    framework_df = pd.DataFrame([
        [1, "데이터 불러오기", "CSV 기반 제품·수요·공급지·조립지·운송·시나리오 데이터를 읽습니다."],
        [2, "허용 경로 생성", "국가별 운송수단 가용성과 거리행렬로 복합운송 경로를 구성합니다."],
        [3, "LP 모형 구성", "목적함수, 물량수지, 수요충족, 용량, 탄소상한 제약을 구성합니다."],
        [4, "최적화 계산", "OR-Tools 연속 LP로 비용 최소 공급망을 계산합니다."],
        [5, "시나리오 평가", "탄소상한 만족 여부, 비용, 배출량, 제품별 지표를 계산합니다."],
        [6, "결과 시각화", "포스터형 결과, 상세 표, 공급망 지도, 포스터 비교를 제공합니다."],
    ], columns=["단계", "모듈", "구체적 역할"])
    st.dataframe(framework_df, hide_index=True, use_container_width=True)

    st.markdown("### 4) 현재 앱에서 최적화하는 의사결정")
    decision_df = pd.DataFrame([
        ["RP", "각 재질을 어느 공급지에서 얼마나 생산할 것인가"],
        ["RT", "각 재질을 어느 공급지에서 어느 조립지로 어떤 운송경로로 얼마나 보낼 것인가"],
        ["FP", "각 조립지에서 제품을 얼마나 생산할 것인가"],
        ["FT", "완제품을 어느 조립지에서 프랑스 시장으로 어떤 운송경로로 얼마나 보낼 것인가"],
        ["ZM/ZS/ZL", "생산방식에 따른 배터리 모듈/팩 구조를 어떻게 충족할 것인가"],
    ], columns=["변수", "사용자 관점의 의미"])
    st.dataframe(decision_df, hide_index=True, use_container_width=True)

    st.info(
        "이 앱은 운송수단을 확률적으로 추첨하지 않습니다. 국가별로 허용된 여러 운송경로에 대해 LP가 연속 물량을 직접 배분합니다. "
        "즉, 운송수단 선택은 랜덤 선택이 아니라 비용·탄소·제약을 고려한 최적 물량 배분입니다."
    )


def run_app():
    st.set_page_config(
        page_title="PDF 기반 전기차 공급망 Route LP",
        page_icon="🚗",
        layout="wide",
    )
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
        subtabs = st.tabs(list(display_names.values()))
        for sub, filename in zip(subtabs, display_names):
            with sub:
                st.dataframe(tables[filename], use_container_width=True, hide_index=True)

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
            st.info("2번 탭에서 선택 조합 또는 6개 조합을 실행하세요.")
        else:
            for scenario in ["S1", "S2", "S3"]:
                render_poster_scenario(results, scenario)
                st.divider()

            st.markdown("### 분석 및 결론")
            ratio_df = cost_ratio_dataframe(results)
            if not ratio_df.empty:
                st.bar_chart(
                    ratio_df.set_index("scenario_name")[["modular_to_line_cost_ratio"]],
                    use_container_width=True,
                )
                st.dataframe(ratio_df, hide_index=True, use_container_width=True)

            valid_results = {
                key: value for key, value in results.items()
                if value.get("status") in {"OPTIMAL", "FEASIBLE"}
            }

            if valid_results:
                st.markdown("### 공급망 지도(요청 시 한 개만 생성)")
                map_choice = st.selectbox(
                    "지도 조합",
                    list(valid_results.keys()),
                    format_func=lambda key: f"{SCENARIO_SHORT[key[0]]} · {MODE_LABEL[key[1]]}",
                    key="lazy_map_choice",
                )
                if st.checkbox("선택한 공급망 지도 표시", value=False, key="show_one_map"):
                    render_result_map(
                        valid_results[map_choice],
                        f"lazy_map_{map_choice[0]}_{map_choice[1]}",
                        height=650,
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
        st.header("수학모형과 코드의 대응")
        st.info("이 5번 탭은 현재 구현된 수학적 모형과 코드 구현이 업데이트될 때마다 함께 수정되는 공식 설명 영역입니다.")

        st.markdown("### 5.1 목적함수")
        st.latex(r"""
        \min Z=
        \sum_{f,r,s} c^{RP}_{rs}RP_{frs}
        +\sum_{f,r,s,p,k}c^{RT}_{spk}RT_{frspk}
        +\sum_{f,p}c^{FP}_{fp}FP_{fp}
        +\sum_{f,p,k}c^{FT}_{pk}FT_{fpk}
        """)
        st.markdown("원자재·배터리 생산비, 부품 운송비, 조립비, 완제품 운송비의 총합을 최소화합니다.")

        st.markdown("### 5.2 변수 정의역과 허용 경로")
        st.latex(r"RP,RT,FP,FT,ZM,ZS,ZL\ge 0")
        st.latex(r"RT_{frspk}=0\;\text{if }A^{raw}_{spk}=0,\qquad FT_{fpk}=0\;\text{if }A^{fin}_{pk}=0")
        st.markdown("모든 변수는 연속변수이며, 허용되지 않은 운송경로는 상한을 0으로 두어 자동으로 사용되지 않게 합니다.")

        st.markdown("### 5.3 수요 충족과 물량 보존")
        st.latex(r"\sum_{p,k}FT_{fpk}=D_f")
        st.latex(r"FP_{fp}=\sum_k FT_{fpk}")
        st.latex(r"RP_{frs}=\sum_{p,k}RT_{frspk}")
        st.latex(r"\sum_{s,k}RT_{frspk}=a_{fr}FP_{fp}\qquad (r\in\{steel,aluminum,other\})")
        st.markdown("프랑스 수요는 등식으로 정확히 충족하며, 공급지 생산량=출고량, 조립지 유입량=필요량, 조립량=완제품 출하량의 구조를 강제합니다.")

        st.markdown("### 5.4 배터리 관련 생산방식 제약")
        st.markdown("**라인 생산 방식**")
        st.latex(r"\sum_k RT_{f,bat,spk}=B_f ZL_{fsp}")
        st.latex(r"\sum_s ZL_{fsp}=FP_{fp}")
        st.markdown(r"배터리 팩 전체를 한 번에 공급받는 구조입니다. 여기서 \(B_f\)는 차량 \(f\)의 배터리 용량(kWh)입니다.")
        st.markdown("**모듈 활용 분산 생산 방식**")
        st.latex(r"\sum_k RT_{f,bat,spk}=10\,ZM_{fsp}+5\,ZS_{fsp}")
        st.latex(r"\sum_{s,k}RT_{f,bat,spk}=B_f FP_{fp}")
        st.markdown("10kWh 메인 모듈과 5kWh 보조 모듈의 조합으로 배터리를 충족하는 구조입니다.")

        st.markdown("### 5.5 공급능력 제약")
        st.latex(r"\sum_f RP_{frs}\le Cap_{rs}")
        st.markdown("각 재질-공급지 조합의 총 생산량은 해당 공급지의 최대 생산가능량을 넘을 수 없습니다.")

        st.markdown("### 5.6 탄소배출량 계산과 상한")
        st.latex(r"C=C^{RP}+C^{RT}+C^{FP}+C^{FT}")
        st.latex(r"C^{RP}=\sum_{f,r,s}\frac{EF^{RP}_{rs}}{1-L_r}RP_{frs}")
        st.latex(r"C^{RT}=\sum_{f,r,s,p,k} e^{RT}_{spk} RT_{frspk}")
        st.latex(r"C^{FP}=\sum_{f,p} e^{FP}_{fp} FP_{fp}")
        st.latex(r"C^{FT}=\sum_{f,p,k} e^{FT}_{pk} FT_{fpk}")
        st.markdown(r"총 배출량은 생산, 공급지→조립지 운송, 조립, 조립지→시장 운송 배출량의 합입니다. 철강과 알루미늄에는 손실률 \(L_r=0.3\)이 반영됩니다.")
        st.markdown("**차급 평균 상한(class-average)**")
        st.latex(r"\sum_{f\in g} C_f \le \bar{E}^{cap}_g \sum_{f\in g} D_f \qquad (g\in\{small,standard\})")
        st.markdown("동일 차급 제품들의 수요가중 평균 배출량이 시나리오 상한 이하가 되도록 제약합니다.")
        st.markdown("**트림별 상한(product-strict)**")
        st.latex(r"C_f\le E^{cap}_f D_f")
        st.markdown("각 제품 트림별 배출량이 개별 탄소상한을 직접 만족하도록 제약합니다.")

        st.markdown("### 5.7 PDF 기반 운송계수 구조")
        st.latex(r"e^{RT}_{spk}=\sum_{\ell\in k} d_{spk\ell}EF_{\ell,region(\ell)}")
        st.markdown("유럽-유럽 구간은 도로/철도 직접운송을 사용합니다. 비유럽 국제구간은 해상 또는 항공 주운송에 출발·도착 내륙운송을 결합합니다.")

        st.markdown("### 5.8 코드 매핑")
        mapping = pd.DataFrame([
            ["RP_{frs}", "IndexLayout.rp()", "공급지의 순사용 가능 생산량", "연속"],
            ["RT_{frspk}", "IndexLayout.rt()", "허용된 복합경로별 공급지→조립지 물량", "연속"],
            ["FP_{fp}", "IndexLayout.fp()", "조립지별 완제품 생산량", "연속"],
            ["FT_{fpk}", "IndexLayout.ft()", "허용된 복합경로별 조립지→프랑스 물량", "연속"],
            ["ZM_{fsp}, ZS_{fsp}", "IndexLayout.z1()/z2() [modular]", "10kWh/5kWh 모듈 수량", "연속"],
            ["ZL_{fsp}", "IndexLayout.z1() [line]", "라인 생산용 배터리 팩 등가량", "연속"],
            ["A^{raw}_{spk}, A^{fin}_{pk}", "country_transport_rules.csv + build_route_matrices()", "국가별 운송수단 가용성", "고정 0/1 파라미터"],
            ["c^{RT}_{spk}, e^{RT}_{spk}", "build_route_matrices()", "복합운송 경로별 비용·배출계수", "파라미터"],
            ["EF^{RP}_{rs}, L_r", "raw_material_suppliers.csv + material_parameters.csv", "국가별 생산배출계수와 손실률", "파라미터"],
        ], columns=["수학 변수/파라미터", "코드·CSV", "의미", "정의역/유형"])
        st.dataframe(mapping, hide_index=True, use_container_width=True)

        st.markdown("### 5.9 구현된 제약식과 코드 위치")
        equations_df = pd.DataFrame([
            ["수요 충족", "sum FT = D", "build_pdf_route_lp_model() 1)", "프랑스 수요를 정확히 충족"],
            ["배터리 제약(모듈)", "sum RT = 10 ZM + 5 ZS", "build_pdf_route_lp_model() 2)", "배터리 모듈 구조 반영"],
            ["배터리 제약(라인)", "sum RT = B_f ZL, sum ZL = FP", "build_pdf_route_lp_model() 2)", "배터리 팩 구조 반영"],
            ["재질 수지", "sum RT = a_fr FP", "build_pdf_route_lp_model() 3)", "조립지의 유입량=필요량"],
            ["조립지 출고 수지", "FP = sum FT", "build_pdf_route_lp_model() 4)", "조립 수량=출고 수량"],
            ["공급지 출고 수지", "RP = sum RT", "build_pdf_route_lp_model() 5)", "생산량=출고량"],
            ["공급능력", "sum_f RP <= Cap", "build_pdf_route_lp_model() 6)", "공급지 최대 생산능력"],
            ["탄소상한", "product-strict 또는 class-average", "build_pdf_route_lp_model() 7)", "시나리오별 탄소기준"],
        ], columns=["제약식", "수식 요약", "코드 위치", "설명"])
        st.dataframe(equations_df, hide_index=True, use_container_width=True)

        st.markdown("### 5.10 PDF 기반 구현 기준")
        checks = pd.DataFrame([
            ["제품 수", 6, "products.csv"],
            ["재질 수", 4, "steel/aluminum/other/battery"],
            ["공급·조립 위치 수", 24, "assembly_locations.csv"],
            ["경로 대안 수", len(ROUTE_MODE_CODES), ", ".join(ROUTE_MODE_CODES)],
            ["철강 손실률", 0.3, "material_parameters.csv"],
            ["알루미늄 손실률", 0.3, "material_parameters.csv"],
            ["수요 충족", "등식", "sum FT = D"],
            ["확률적 수단 선택", "미사용", "운송수단별 물량을 LP가 직접 결정"],
        ], columns=["검사항목", "구현값", "근거/코드"])
        st.dataframe(checks, hide_index=True, use_container_width=True)
        st.caption(f"역사적 Xpress 참조 LP SHA-256: {REFERENCE_LP_SHA256} · 이 5번 탭은 수학적 모형이 업데이트될 때마다 함께 수정되는 공식 설명 탭입니다.")

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
