from __future__ import annotations

import io
import math
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import folium
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import Fullscreen
from geopy.distance import geodesic
from ortools.linear_solver import pywraplp
from streamlit_folium import st_folium


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ASSET_DIR = APP_DIR / "assets"
APP_BUILD = "strategic-lp-split-transport-v6.1"
APP_PACKAGE_ID = "20260803-2010-KST"

REQUIRED_FILES = [
    "products.csv",
    "demand.csv",
    "raw_material_suppliers.csv",
    "assembly_locations.csv",
    "transport_parameters.csv",
    "markets.csv",
    "scenarios.csv",
]
BENCHMARK_FILES = [
    "poster_benchmark_cost_ratios.csv",
    "poster_benchmark_quartiles.csv",
]
ALL_UPLOAD_FILES = REQUIRED_FILES + BENCHMARK_FILES

MATERIAL_LABEL = {
    "steel": "철강",
    "aluminum": "알루미늄",
    "other": "기타 원자재",
    "battery": "배터리",
}
MODE_LABEL = {"line": "라인 생산", "modular": "모듈 활용 분산 생산"}
TRANSPORT_LABEL = {"sea": "해상", "air": "항공", "road": "도로", "rail": "철도"}
MATERIAL_COLOR = {
    "steel": "#d73027",
    "aluminum": "#fc8d59",
    "other": "#91cf60",
    "battery": "#4575b4",
    "finished": "#542788",
}
TRANSPORT_DASH = {"sea": None, "air": "2,8", "road": "8,5", "rail": "1,5"}

# 첫 번째 PDF의 철강·알루미늄 탄소발자국 식에 제시된 손실률입니다.
MATERIAL_LOSS_RATE = 0.30
CAP_APPLICATION_LABEL = {
    "product_strict": "기본: 트림별 수요가중 평균 탄소상한",
    "class_average": "비교실험: 차급별 수요가중 평균 탄소상한",
}

st.set_page_config(
    page_title="탄소배출 기반 전기차 공급망 최적화 SaaS",
    page_icon="🚗",
    layout="wide",
)


def configure_matplotlib_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic", "AppleGothic"]:
        if candidate in installed:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


configure_matplotlib_font()


# -----------------------------------------------------------------------------
# Data I/O and validation
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def read_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_default_data() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for name in REQUIRED_FILES + BENCHMARK_FILES:
        path = DATA_DIR / name
        if path.exists():
            out[name] = read_csv_path(str(path))
    return out


def load_uploaded_data(files) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for uploaded in files or []:
        name = Path(uploaded.name).name
        out[name] = read_csv_bytes(uploaded.getvalue())
    return out


def merge_data_sources(defaults: Dict[str, pd.DataFrame], uploads: Dict[str, pd.DataFrame], use_defaults: bool):
    merged = dict(defaults) if use_defaults else {}
    merged.update(uploads)
    return merged


def make_template_zip() -> bytes:
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in REQUIRED_FILES + BENCHMARK_FILES:
            path = DATA_DIR / name
            if path.exists():
                zf.write(path, arcname=name)
    return memory.getvalue()


def validate_data(tables: Dict[str, pd.DataFrame]) -> List[str]:
    errors: List[str] = []
    missing = [f for f in ALL_UPLOAD_FILES if f not in tables]
    if missing:
        errors.append("업로드 필수 CSV 누락: " + ", ".join(missing))
        return errors

    required_cols = {
        "products.csv": {
            "product_id", "vehicle_class", "product_name_ko", "battery_kwh",
            "vehicle_mass_kg", "nonbattery_mass_kg", "steel_kg", "aluminum_kg",
            "other_material_kg", "battery_mass_kg", "main_module_count", "sub_module_count",
        },
        "demand.csv": {"product_id", "market_id", "demand_units"},
        "raw_material_suppliers.csv": {
            "material_id", "supplier_id", "location_name", "continent", "latitude", "longitude",
            "production_ef", "production_cost", "parameter_unit", "capacity", "active_default",
        },
        "assembly_locations.csv": {
            "plant_id", "location_name", "continent", "latitude", "longitude",
            "assembly_ef_kgco2_per_kg", "assembly_cost_eur_per_kg", "capacity_kg", "active_default",
        },
        "transport_parameters.csv": {
            "transport_mode", "region_class", "transport_ef_kgco2_per_kgkm",
            "transport_cost_eur_per_kgkm",
        },
        "markets.csv": {"market_id", "market_name", "continent", "latitude", "longitude"},
        "scenarios.csv": {
            "scenario_id", "scenario_name", "minimum_score", "apply_carbon_cap",
            "small_cap_kgco2_per_vehicle", "standard_cap_kgco2_per_vehicle",
        },
    }
    for filename, cols in required_cols.items():
        missing_cols = cols - set(tables[filename].columns)
        if missing_cols:
            errors.append(f"{filename}: 누락 컬럼 {sorted(missing_cols)}")

    if errors:
        return errors

    p = tables["products.csv"]
    d = tables["demand.csv"]
    if set(d["product_id"]) - set(p["product_id"]):
        errors.append("demand.csv에 products.csv에 없는 product_id가 있습니다.")

    numeric_checks = {
        "products.csv": ["battery_kwh", "vehicle_mass_kg", "nonbattery_mass_kg", "steel_kg", "aluminum_kg", "other_material_kg", "battery_mass_kg"],
        "demand.csv": ["demand_units"],
        "raw_material_suppliers.csv": ["latitude", "longitude", "production_ef", "production_cost", "capacity"],
        "assembly_locations.csv": ["latitude", "longitude", "assembly_ef_kgco2_per_kg", "assembly_cost_eur_per_kg", "capacity_kg"],
        "transport_parameters.csv": ["transport_ef_kgco2_per_kgkm", "transport_cost_eur_per_kgkm"],
    }
    signed_coordinate_cols = {"latitude", "longitude"}
    for filename, cols in numeric_checks.items():
        for c in cols:
            vals = pd.to_numeric(tables[filename][c], errors="coerce")
            if vals.isna().any():
                errors.append(f"{filename}.{c}: 숫자가 아닌 값 또는 빈 값이 있습니다.")
            elif c not in signed_coordinate_cols and (vals < 0).any():
                errors.append(f"{filename}.{c}: 음수 값이 있습니다.")

    product_mass = p["steel_kg"] + p["aluminum_kg"] + p["other_material_kg"]
    bad = ~np.isclose(product_mass, p["nonbattery_mass_kg"], rtol=0, atol=1e-6)
    if bad.any():
        errors.append("products.csv: steel+aluminum+other가 nonbattery_mass와 일치하지 않는 제품이 있습니다.")
    bad2 = ~np.isclose(p["nonbattery_mass_kg"] + p["battery_mass_kg"], p["vehicle_mass_kg"], rtol=0, atol=1e-6)
    if bad2.any():
        errors.append("products.csv: nonbattery+battery가 vehicle_mass와 일치하지 않는 제품이 있습니다.")

    for material in ["steel", "aluminum", "other", "battery"]:
        if material not in set(tables["raw_material_suppliers.csv"]["material_id"]):
            errors.append(f"raw_material_suppliers.csv: {material} 공급지가 없습니다.")
    return errors


# -----------------------------------------------------------------------------
# Transport and distance helpers
# -----------------------------------------------------------------------------
def route_region(origin_continent: str, destination_continent: str, destination_name: str, origin_name: str = "") -> str:
    if destination_name == "프랑스" and origin_continent == "Europe":
        return "France" if origin_name == "프랑스" else "Europe_ex_France"
    if origin_continent == destination_continent:
        if origin_continent == "Europe":
            return "Europe_ex_France"
        if origin_continent in {"Africa", "Asia", "Americas", "Oceania"}:
            return origin_continent
    return "Other"


def feasible_modes(origin_continent: str, destination_continent: str, destination_name: str) -> List[str]:
    # Word rule: routes linked to France can use all four modes within Europe,
    # while non-European routes use sea or air. The same geographic rule is
    # applied to supplier-to-assembly routes for an executable network model.
    if origin_continent == "Europe" and destination_continent == "Europe":
        return ["sea", "road", "rail", "air"]
    return ["sea", "air"]


def nondominated_modes(
    tp: pd.DataFrame,
    origin_continent: str,
    destination_continent: str,
    destination_name: str,
    origin_name: str = "",
) -> List[str]:
    """비용과 배출량이 모두 열등한 운송수단을 사전에 제거합니다.

    열등한 수단은 비용 최소화 목적함수와 탄소 상한 제약 하에서 어떤
    최적해에도 선택될 수 없으므로 제거해도 수학적 최적해는 변하지 않습니다.
    이 전처리는 모든 Word 후보지를 선택했을 때 MIP 이진변수 수를 크게 줄입니다.
    """
    candidates = feasible_modes(origin_continent, destination_continent, destination_name)
    region = route_region(origin_continent, destination_continent, destination_name, origin_name)
    values = {}
    for mode in candidates:
        try:
            values[mode] = get_transport_parameter(tp, mode, region)
        except KeyError:
            continue
    kept: List[str] = []
    for mode, (cost, ef) in values.items():
        dominated = False
        for other, (other_cost, other_ef) in values.items():
            if other == mode:
                continue
            weakly_better = other_cost <= cost + 1e-15 and other_ef <= ef + 1e-15
            strictly_better = other_cost < cost - 1e-15 or other_ef < ef - 1e-15
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            kept.append(mode)
    return kept or candidates


def get_transport_parameter(tp: pd.DataFrame, mode: str, region: str) -> Tuple[float, float]:
    if mode in {"sea", "air"}:
        row = tp[(tp["transport_mode"] == mode) & (tp["region_class"] == "world")]
    else:
        row = tp[(tp["transport_mode"] == mode) & (tp["region_class"] == region)]
        if row.empty and mode == "rail":
            row = tp[(tp["transport_mode"] == mode) & (tp["region_class"] == "Other")]
    if row.empty:
        raise KeyError(f"운송 파라미터 없음: mode={mode}, region={region}")
    r = row.iloc[0]
    return float(r["transport_cost_eur_per_kgkm"]), float(r["transport_ef_kgco2_per_kgkm"])


def distance_km(lat1, lon1, lat2, lon2) -> float:
    return float(geodesic((float(lat1), float(lon1)), (float(lat2), float(lon2))).km)


# -----------------------------------------------------------------------------
# Strategic linear optimization model (continuous flows and split transport)
# -----------------------------------------------------------------------------
class LpBuilder:
    """전략적 공급망 선형계획(LP)을 OR-Tools MPSolver로 생성한다.

    모든 의사결정변수는 연속변수이며, 운송수단별 물량분할을 허용한다.
    따라서 운송수단 선택 이진변수와 차량·팩·모듈 정수조건은 사용하지 않는다.
    GLOP을 우선 사용하고, 환경에 따라 CLP·PDLP·SCIP 순서로 대체한다.
    """

    def __init__(self):
        self.c: List[float] = []
        self.var_lb: List[float] = []
        self.var_ub: List[float] = []
        self.names: List[str] = []
        self.rows: List[Dict[int, float]] = []
        self.row_lb: List[float] = []
        self.row_ub: List[float] = []

    def add_var(self, lb: float, ub: float, kind: str, name: str) -> int:
        if kind != "C":
            raise ValueError(f"전략적 LP에서는 연속변수(C)만 허용합니다: {name}, kind={kind}")
        idx = len(self.c)
        self.c.append(0.0)
        self.var_lb.append(float(lb))
        self.var_ub.append(float(ub))
        self.names.append(name)
        return idx

    def add_obj(self, var: int, coefficient: float):
        self.c[var] += float(coefficient)

    def add_constraint(self, expression: Dict[int, float], lb: float = -np.inf, ub: float = np.inf):
        cleaned = {i: float(v) for i, v in expression.items() if abs(v) > 1e-14}
        self.rows.append(cleaned)
        self.row_lb.append(float(lb))
        self.row_ub.append(float(ub))

    @staticmethod
    def _create_solver():
        candidates = [
            ("GLOP", "OR-Tools GLOP LP"),
            ("CLP_LINEAR_PROGRAMMING", "OR-Tools CLP LP"),
            ("PDLP", "OR-Tools PDLP LP"),
            ("SCIP", "OR-Tools SCIP LP fallback"),
        ]
        for solver_id, label in candidates:
            solver = pywraplp.Solver.CreateSolver(solver_id)
            if solver is not None:
                return solver, label
        return None, "OR-Tools LP solver unavailable"

    def solve(self, time_limit_sec: int = 60):
        solver, backend = self._create_solver()
        if solver is None:
            return SimpleNamespace(
                status="SOLVER_NOT_CREATED",
                x=None,
                message="GLOP·CLP·PDLP·SCIP 중 사용할 수 있는 LP solver가 없습니다. ortools 설치를 확인하세요.",
                backend=backend,
                objective_value=None,
                best_bound=None,
                wall_time_sec=0.0,
                variable_count=len(self.c),
                integer_variable_count=0,
                binary_variable_count=0,
                constraint_count=len(self.rows),
            )

        solver.SetTimeLimit(max(1, int(time_limit_sec)) * 1000)
        try:
            solver.SetNumThreads(1)
        except Exception:
            pass

        infinity = solver.infinity()
        variables = []
        for lb, ub, name in zip(self.var_lb, self.var_ub, self.names):
            lower = -infinity if np.isneginf(lb) else float(lb)
            upper = infinity if np.isposinf(ub) else float(ub)
            variables.append(solver.NumVar(lower, upper, name))

        for row_index, (expression, lb, ub) in enumerate(zip(self.rows, self.row_lb, self.row_ub)):
            lower = -infinity if np.isneginf(lb) else float(lb)
            upper = infinity if np.isposinf(ub) else float(ub)
            constraint = solver.RowConstraint(lower, upper, f"constraint_{row_index}")
            for var_index, coefficient in expression.items():
                constraint.SetCoefficient(variables[var_index], float(coefficient))

        objective = solver.Objective()
        for var_index, coefficient in enumerate(self.c):
            if abs(coefficient) > 1e-14:
                objective.SetCoefficient(variables[var_index], float(coefficient))
        objective.SetMinimization()

        started = time.perf_counter()
        status_code = solver.Solve()
        wall_time_sec = time.perf_counter() - started
        status_map = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
            pywraplp.Solver.ABNORMAL: "ABNORMAL",
            pywraplp.Solver.MODEL_INVALID: "MODEL_INVALID",
            pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
        }
        status = status_map.get(status_code, f"UNKNOWN_{status_code}")
        common = dict(
            backend=backend,
            wall_time_sec=wall_time_sec,
            variable_count=len(self.c),
            integer_variable_count=0,
            binary_variable_count=0,
            constraint_count=len(self.rows),
        )

        if status in {"OPTIMAL", "FEASIBLE"}:
            values = np.array([var.solution_value() for var in variables], dtype=float)
            objective_value = float(objective.Value())
            return SimpleNamespace(
                status=status,
                x=values,
                message=f"{status} LP solution found",
                objective_value=objective_value,
                best_bound=objective_value if status == "OPTIMAL" else None,
                **common,
            )

        reason = {
            "INFEASIBLE": "선택된 공급·조립 용량과 탄소상한을 동시에 만족하는 연속 공급망 배분해가 존재하지 않습니다.",
            "NOT_SOLVED": "제한시간 안에 LP 해를 확정하지 못했습니다. 이는 infeasible 판정과 다릅니다.",
            "UNBOUNDED": "목적함수가 무한히 감소할 수 있어 모형 또는 입력값을 확인해야 합니다.",
            "MODEL_INVALID": "OR-Tools가 LP 모형을 유효하지 않은 것으로 판정했습니다.",
            "ABNORMAL": "LP Solver가 비정상 종료했습니다.",
        }.get(status, f"OR-Tools solve status: {status}")
        return SimpleNamespace(
            status=status,
            x=None,
            message=reason,
            objective_value=None,
            best_bound=None,
            **common,
        )


def add_term(expr: Dict[int, float], var: int, coefficient: float = 1.0):
    expr[var] = expr.get(var, 0.0) + float(coefficient)


def subsidy_score(vehicle_class: str, emission_per_vehicle: float) -> float:
    if vehicle_class == "small":
        low, high = 6000.0, 17000.0
    else:
        low, high = 12000.0, 21000.0
    if emission_per_vehicle <= low:
        return 80.0
    if emission_per_vehicle >= high:
        return 0.0
    return 80.0 * (high - emission_per_vehicle) / (high - low)


def selection_capacity_diagnostics(
    tables: Dict[str, pd.DataFrame],
    selected_products: List[str],
    selected_supplier_ids: List[str],
    selected_plant_ids: List[str],
) -> pd.DataFrame:
    products = tables["products.csv"]
    demand = tables["demand.csv"]
    suppliers = tables["raw_material_suppliers.csv"]
    plants = tables["assembly_locations.csv"]
    products = products[products["product_id"].isin(selected_products)].copy()
    demand_map = demand[demand["product_id"].isin(selected_products)].groupby("product_id")["demand_units"].sum().to_dict()
    suppliers = suppliers[suppliers["supplier_id"].isin(selected_supplier_ids)].copy()
    plants = plants[plants["plant_id"].isin(selected_plant_ids)].copy()
    rows = []
    material_cols = {"steel": "steel_kg", "aluminum": "aluminum_kg", "other": "other_material_kg"}
    product_map = products.set_index("product_id").to_dict("index")
    for material, col in material_cols.items():
        required = sum(float(product_map[f][col]) * float(demand_map.get(f, 0)) for f in product_map)
        available = float(suppliers.loc[suppliers["material_id"] == material, "capacity"].sum())
        rows.append({
            "검사항목": MATERIAL_LABEL[material] + " 공급용량",
            "필요량": required,
            "선택후보 용량": available,
            "단위": "kg",
            "판정": "충족" if available + 1e-6 >= required else "부족",
        })
    battery_required = sum(float(product_map[f]["battery_kwh"]) * float(demand_map.get(f, 0)) for f in product_map)
    battery_available = float(suppliers.loc[suppliers["material_id"] == "battery", "capacity"].sum())
    rows.append({
        "검사항목": "배터리 공급용량",
        "필요량": battery_required,
        "선택후보 용량": battery_available,
        "단위": "kWh",
        "판정": "충족" if battery_available + 1e-6 >= battery_required else "부족",
    })
    assembly_required = sum(float(product_map[f]["nonbattery_mass_kg"]) * float(demand_map.get(f, 0)) for f in product_map)
    assembly_available = float(plants["capacity_kg"].sum())
    rows.append({
        "검사항목": "가공·조립 용량",
        "필요량": assembly_required,
        "선택후보 용량": assembly_available,
        "단위": "kg",
        "판정": "충족" if assembly_available + 1e-6 >= assembly_required else "부족",
    })
    return pd.DataFrame(rows)


def optimistic_carbon_lower_bounds(
    tables: Dict[str, pd.DataFrame],
    selected_products: List[str],
    selected_supplier_ids: List[str],
    selected_plant_ids: List[str],
    scenario_id: str,
    loss_rate: float = MATERIAL_LOSS_RATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """공유 용량을 무시한 낙관적 탄소 하한입니다.

    이 하한조차 상한보다 크면 해당 선택은 수학적으로 확실히 infeasible입니다.
    하한이 상한보다 작다고 해서 반드시 feasible임을 보장하지는 않습니다.
    """
    products = tables["products.csv"]
    demand = tables["demand.csv"]
    suppliers = tables["raw_material_suppliers.csv"]
    plants = tables["assembly_locations.csv"]
    tp = tables["transport_parameters.csv"]
    markets = tables["markets.csv"]
    scenarios = tables["scenarios.csv"]
    products = products[products["product_id"].isin(selected_products)].copy()
    suppliers = suppliers[suppliers["supplier_id"].isin(selected_supplier_ids)].copy()
    plants = plants[plants["plant_id"].isin(selected_plant_ids)].copy()
    scenario = scenarios.loc[scenarios["scenario_id"] == scenario_id].iloc[0]
    demand_map = demand[demand["product_id"].isin(selected_products)].groupby("product_id")["demand_units"].sum().to_dict()
    market_id = str(demand.loc[demand["product_id"].isin(selected_products), "market_id"].iloc[0])
    market = markets.loc[markets["market_id"] == market_id].iloc[0]
    rows = []
    for _, product in products.iterrows():
        best = np.inf
        for _, plant in plants.iterrows():
            total = float(product["nonbattery_mass_kg"]) * float(plant["assembly_ef_kgco2_per_kg"])
            feasible = True
            for material, quantity_col in [
                ("steel", "steel_kg"),
                ("aluminum", "aluminum_kg"),
                ("other", "other_material_kg"),
                ("battery", "battery_mass_kg"),
            ]:
                sub = suppliers[suppliers["material_id"] == material]
                candidate_values = []
                for _, supplier in sub.iterrows():
                    modes = nondominated_modes(
                        tp, str(supplier["continent"]), str(plant["continent"]),
                        str(plant["location_name"]), str(supplier["location_name"])
                    )
                    dist = distance_km(supplier["latitude"], supplier["longitude"], plant["latitude"], plant["longitude"])
                    min_transport_ef = min(
                        get_transport_parameter(
                            tp, mode,
                            route_region(str(supplier["continent"]), str(plant["continent"]), str(plant["location_name"]), str(supplier["location_name"]))
                        )[1] for mode in modes
                    )
                    if material == "battery":
                        production = float(product["battery_kwh"]) * float(supplier["production_ef"])
                        transport = float(product["battery_mass_kg"]) * dist * min_transport_ef
                    else:
                        quantity = float(product[quantity_col])
                        factor = 1.0 / (1.0 - loss_rate) if material in {"steel", "aluminum"} else 1.0
                        production = quantity * float(supplier["production_ef"]) * factor
                        transport = quantity * dist * min_transport_ef
                    candidate_values.append(production + transport)
                if not candidate_values:
                    feasible = False
                    break
                total += min(candidate_values)
            if not feasible:
                continue
            final_modes = nondominated_modes(
                tp, str(plant["continent"]), str(market["continent"]),
                str(market["location_name"]), str(plant["location_name"])
            )
            final_region = route_region(str(plant["continent"]), str(market["continent"]), str(market["location_name"]), str(plant["location_name"]))
            final_dist = distance_km(plant["latitude"], plant["longitude"], market["latitude"], market["longitude"])
            final_ef = min(get_transport_parameter(tp, mode, final_region)[1] for mode in final_modes)
            total += float(product["vehicle_mass_kg"]) * final_dist * final_ef
            best = min(best, total)
        cap = np.nan
        if int(scenario["apply_carbon_cap"]) == 1:
            cap = float(scenario["small_cap_kgco2_per_vehicle"] if product["vehicle_class"] == "small" else scenario["standard_cap_kgco2_per_vehicle"])
        rows.append({
            "product_id": product["product_id"],
            "product_name": product["product_name_ko"],
            "vehicle_class": product["vehicle_class"],
            "demand_units": float(demand_map.get(product["product_id"], 0)),
            "optimistic_lower_bound_kgco2_per_vehicle": best,
            "carbon_cap_kgco2_per_vehicle": cap,
            "strict_product_possible": bool(np.isnan(cap) or best <= cap + 1e-6),
        })
    product_lb = pd.DataFrame(rows)
    class_rows = []
    for vehicle_class, grp in product_lb.groupby("vehicle_class"):
        total_demand = grp["demand_units"].sum()
        avg_lb = float((grp["optimistic_lower_bound_kgco2_per_vehicle"] * grp["demand_units"]).sum() / total_demand)
        cap = float(grp["carbon_cap_kgco2_per_vehicle"].dropna().iloc[0]) if grp["carbon_cap_kgco2_per_vehicle"].notna().any() else np.nan
        class_rows.append({
            "vehicle_class": vehicle_class,
            "class_name": "소형" if vehicle_class == "small" else "중형·대형",
            "optimistic_weighted_average_kgco2_per_vehicle": avg_lb,
            "carbon_cap_kgco2_per_vehicle": cap,
            "class_average_possible": bool(np.isnan(cap) or avg_lb <= cap + 1e-6),
        })
    return product_lb, pd.DataFrame(class_rows)


def solve_model(
    tables: Dict[str, pd.DataFrame],
    selected_products: List[str],
    selected_supplier_ids: List[str],
    selected_plant_ids: List[str],
    production_mode: str,
    scenario_id: str,
    cap_application: str = "product_strict",
    loss_rate: float = MATERIAL_LOSS_RATE,
    time_limit_sec: int = 300,
) -> Dict:
    products = tables["products.csv"].copy()
    demand = tables["demand.csv"].copy()
    suppliers = tables["raw_material_suppliers.csv"].copy()
    plants = tables["assembly_locations.csv"].copy()
    tp = tables["transport_parameters.csv"].copy()
    markets = tables["markets.csv"].copy()
    scenarios = tables["scenarios.csv"].copy()

    products = products[products["product_id"].isin(selected_products)].copy()
    demand = demand[demand["product_id"].isin(selected_products)].copy()
    suppliers = suppliers[suppliers["supplier_id"].isin(selected_supplier_ids)].copy()
    plants = plants[plants["plant_id"].isin(selected_plant_ids)].copy()

    if products.empty or demand.empty or suppliers.empty or plants.empty:
        return {"status": "INVALID_SELECTION", "message": "제품·공급지·조립지 선택을 확인하세요."}

    market_id = str(demand["market_id"].iloc[0])
    market_row = markets[markets["market_id"] == market_id]
    if market_row.empty:
        return {"status": "INVALID_MARKET", "message": f"markets.csv에 {market_id}가 없습니다."}
    market = market_row.iloc[0]
    scenario_row = scenarios[scenarios["scenario_id"] == scenario_id]
    if scenario_row.empty:
        return {"status": "INVALID_SCENARIO", "message": f"scenarios.csv에 {scenario_id}가 없습니다."}
    scenario = scenario_row.iloc[0]

    product_map = products.set_index("product_id").to_dict("index")
    demand_map = demand.groupby("product_id")["demand_units"].sum().astype(int).to_dict()
    supplier_map = suppliers.set_index("supplier_id").to_dict("index")
    plant_map = plants.set_index("plant_id").to_dict("index")
    material_suppliers = {
        m: suppliers[suppliers["material_id"] == m]["supplier_id"].tolist()
        for m in ["steel", "aluminum", "other", "battery"]
    }
    if any(not v for v in material_suppliers.values()):
        return {"status": "INVALID_SELECTION", "message": "선택된 후보 중 일부 원자재 공급지가 없습니다."}

    model = LpBuilder()
    F = list(product_map)
    P = list(plant_map)
    R = ["steel", "aluminum", "other", "battery"]
    material_mass_col = {
        "steel": "steel_kg", "aluminum": "aluminum_kg",
        "other": "other_material_kg", "battery": "battery_mass_kg",
    }

    fp: Dict[Tuple[str, str], int] = {}
    ft: Dict[Tuple[str, str, str], int] = {}
    rt: Dict[Tuple[str, str, str, str, str], int] = {}
    line_count: Dict[Tuple[str, str, str], int] = {}
    main_count: Dict[Tuple[str, str, str], int] = {}
    sub_count: Dict[Tuple[str, str, str], int] = {}
    raw_route_meta: Dict[Tuple[str, str, str, str, str], Dict] = {}
    final_route_meta: Dict[Tuple[str, str, str], Dict] = {}

    # Continuous assembly-equivalent and finished-goods flow variables
    for f in F:
        D = demand_map[f]
        for p in P:
            fp[f, p] = model.add_var(0, D, "C", f"FP__{f}__{p}")
            plant = plant_map[p]
            modes = nondominated_modes(tp, str(plant["continent"]), str(market["continent"]), str(market["location_name"]), str(plant["location_name"]))
            region = route_region(str(plant["continent"]), str(market["continent"]), str(market["location_name"]), str(plant["location_name"]))
            dist = distance_km(plant["latitude"], plant["longitude"], market["latitude"], market["longitude"])
            for t in modes:
                ft[f, p, t] = model.add_var(0, D, "C", f"FT__{f}__{p}__{t}")
                cost, ef = get_transport_parameter(tp, t, region)
                final_route_meta[f, p, t] = {"distance_km": dist, "cost": cost, "ef": ef, "region": region}
            expr = {fp[f, p]: -1.0}
            for t in modes:
                add_term(expr, ft[f, p, t])
            model.add_constraint(expr, 0.0, 0.0)

    # Raw-material route variables: each mode is an independent continuous flow, so mode splitting is allowed
    for f in F:
        D = demand_map[f]
        for r in R:
            max_flow = float(product_map[f][material_mass_col[r]]) * D
            for s_id in material_suppliers[r]:
                supplier = supplier_map[s_id]
                for p in P:
                    plant = plant_map[p]
                    modes = nondominated_modes(tp, str(supplier["continent"]), str(plant["continent"]), str(plant["location_name"]), str(supplier["location_name"]))
                    region = route_region(str(supplier["continent"]), str(plant["continent"]), str(plant["location_name"]), str(supplier["location_name"]))
                    dist = distance_km(supplier["latitude"], supplier["longitude"], plant["latitude"], plant["longitude"])
                    for t in modes:
                        key = (f, r, s_id, p, t)
                        rt[key] = model.add_var(0.0, max_flow, "C", "RT__" + "__".join(key))
                        cost, ef = get_transport_parameter(tp, t, region)
                        raw_route_meta[key] = {"distance_km": dist, "cost": cost, "ef": ef, "region": region}

    # Raw-material requirements at each assembly location
    for f in F:
        product = product_map[f]
        for p in P:
            for r in ["steel", "aluminum", "other"]:
                expr: Dict[int, float] = {fp[f, p]: -float(product[material_mass_col[r]])}
                for s_id in material_suppliers[r]:
                    supplier = supplier_map[s_id]
                    plant = plant_map[p]
                    modes = nondominated_modes(tp, str(supplier["continent"]), str(plant["continent"]), str(plant["location_name"]), str(supplier["location_name"]))
                    for t in modes:
                        add_term(expr, rt[f, r, s_id, p, t])
                model.add_constraint(expr, 0.0, 0.0)

    # Continuous battery-equivalent flow conditions for line or modular production
    if production_mode == "line":
        for f in F:
            product = product_map[f]
            D = demand_map[f]
            for p in P:
                balance_expr: Dict[int, float] = {fp[f, p]: -1.0}
                for s_id in material_suppliers["battery"]:
                    line_count[f, s_id, p] = model.add_var(0, D, "C", f"ZL__{f}__{s_id}__{p}")
                    add_term(balance_expr, line_count[f, s_id, p])
                    supplier = supplier_map[s_id]
                    plant = plant_map[p]
                    modes = nondominated_modes(tp, str(supplier["continent"]), str(plant["continent"]), str(plant["location_name"]), str(supplier["location_name"]))
                    expr = {line_count[f, s_id, p]: -float(product["battery_mass_kg"])}
                    for t in modes:
                        add_term(expr, rt[f, "battery", s_id, p, t])
                    model.add_constraint(expr, 0.0, 0.0)
                model.add_constraint(balance_expr, 0.0, 0.0)
    elif production_mode == "modular":
        for f in F:
            product = product_map[f]
            D = demand_map[f]
            main_mass = 10.0 * float(product["battery_mass_kg"]) / float(product["battery_kwh"])
            sub_mass = 5.0 * float(product["battery_mass_kg"]) / float(product["battery_kwh"])
            for p in P:
                main_balance: Dict[int, float] = {fp[f, p]: -int(product["main_module_count"])}
                sub_balance: Dict[int, float] = {fp[f, p]: -int(product["sub_module_count"])}
                for s_id in material_suppliers["battery"]:
                    main_count[f, s_id, p] = model.add_var(0, int(product["main_module_count"]) * D, "C", f"ZM__{f}__{s_id}__{p}")
                    sub_count[f, s_id, p] = model.add_var(0, max(1, int(product["sub_module_count"]) * D), "C", f"ZS__{f}__{s_id}__{p}")
                    add_term(main_balance, main_count[f, s_id, p])
                    add_term(sub_balance, sub_count[f, s_id, p])
                    supplier = supplier_map[s_id]
                    plant = plant_map[p]
                    modes = nondominated_modes(tp, str(supplier["continent"]), str(plant["continent"]), str(plant["location_name"]), str(supplier["location_name"]))
                    expr = {
                        main_count[f, s_id, p]: -main_mass,
                        sub_count[f, s_id, p]: -sub_mass,
                    }
                    for t in modes:
                        add_term(expr, rt[f, "battery", s_id, p, t])
                    model.add_constraint(expr, 0.0, 0.0)
                model.add_constraint(main_balance, 0.0, 0.0)
                model.add_constraint(sub_balance, 0.0, 0.0)
    else:
        return {"status": "INVALID_MODE", "message": production_mode}

    # Supplier capacities. Battery capacity is in kWh; all other capacities are in kg.
    for r in R:
        for s_id in material_suppliers[r]:
            expr: Dict[int, float] = {}
            for f in F:
                kwh_per_kg = float(product_map[f]["battery_kwh"]) / float(product_map[f]["battery_mass_kg"])
                for p in P:
                    supplier = supplier_map[s_id]
                    plant = plant_map[p]
                    modes = nondominated_modes(tp, str(supplier["continent"]), str(plant["continent"]), str(plant["location_name"]), str(supplier["location_name"]))
                    for t in modes:
                        add_term(expr, rt[f, r, s_id, p, t], kwh_per_kg if r == "battery" else 1.0)
            model.add_constraint(expr, ub=float(supplier_map[s_id]["capacity"]))

    # Assembly capacity in kg of battery-excluded vehicle mass
    for p in P:
        expr = {fp[f, p]: float(product_map[f]["nonbattery_mass_kg"]) for f in F}
        model.add_constraint(expr, ub=float(plant_map[p]["capacity_kg"]))

    # Market demand
    for f in F:
        expr: Dict[int, float] = {}
        for (ff, p, t), var in ft.items():
            if ff == f:
                add_term(expr, var)
        model.add_constraint(expr, float(demand_map[f]), float(demand_map[f]))

    # Objective and product carbon expressions
    product_emission_expr: Dict[str, Dict[int, float]] = {f: {} for f in F}
    coefficient_meta: Dict[Tuple[str, str, str, str, str], Dict[str, float]] = {}
    for key, var in rt.items():
        f, r, s_id, p, t = key
        product = product_map[f]
        supplier = supplier_map[s_id]
        meta = raw_route_meta[key]
        if r == "battery":
            production_unit_factor = float(product["battery_kwh"]) / float(product["battery_mass_kg"])
            prod_cost_coef = float(supplier["production_cost"]) * production_unit_factor
            prod_ef_coef = float(supplier["production_ef"]) * production_unit_factor
        else:
            prod_cost_coef = float(supplier["production_cost"])
            # 첫 번째 PDF: 철강·알루미늄 탄소발자국 = 사용량/(1-손실률 0.3) × 배출계수.
            loss_multiplier = 1.0 / (1.0 - loss_rate) if r in {"steel", "aluminum"} else 1.0
            prod_ef_coef = float(supplier["production_ef"]) * loss_multiplier
        tr_cost_coef = float(meta["cost"]) * float(meta["distance_km"])
        tr_ef_coef = float(meta["ef"]) * float(meta["distance_km"])
        assembly_ef_coef = float(plant_map[p]["assembly_ef_kgco2_per_kg"]) if r in {"steel", "aluminum", "other"} else 0.0
        model.add_obj(var, prod_cost_coef + tr_cost_coef)
        add_term(product_emission_expr[f], var, prod_ef_coef + tr_ef_coef + assembly_ef_coef)
        coefficient_meta[key] = {
            "prod_cost_coef": prod_cost_coef,
            "prod_ef_coef": prod_ef_coef,
            "tr_cost_coef": tr_cost_coef,
            "tr_ef_coef": tr_ef_coef,
            "assembly_ef_coef": assembly_ef_coef,
        }

    final_coefficient_meta: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for (f, p), var in fp.items():
        coef = float(product_map[f]["nonbattery_mass_kg"]) * float(plant_map[p]["assembly_cost_eur_per_kg"])
        model.add_obj(var, coef)

    for key, var in ft.items():
        f, p, t = key
        meta = final_route_meta[key]
        mass = float(product_map[f]["vehicle_mass_kg"])
        cost_coef = mass * float(meta["distance_km"]) * float(meta["cost"])
        ef_coef = mass * float(meta["distance_km"]) * float(meta["ef"])
        model.add_obj(var, cost_coef)
        add_term(product_emission_expr[f], var, ef_coef)
        final_coefficient_meta[key] = {"cost_coef": cost_coef, "ef_coef": ef_coef}

    if int(scenario["apply_carbon_cap"]) == 1:
        if cap_application == "product_strict":
            # PDF의 차량별 점수식을 각 트림에 개별 적용합니다.
            for f in F:
                cap = float(
                    scenario["small_cap_kgco2_per_vehicle"]
                    if product_map[f]["vehicle_class"] == "small"
                    else scenario["standard_cap_kgco2_per_vehicle"]
                )
                model.add_constraint(product_emission_expr[f], ub=cap * demand_map[f])
        elif cap_application == "class_average":
            # 포스터 결과 재현용: 같은 차급의 총 탄소배출량을 총수요로 나눈
            # 수요가중 평균이 차급 상한을 충족하도록 합니다.
            for vehicle_class in sorted({str(product_map[f]["vehicle_class"]) for f in F}):
                members = [f for f in F if str(product_map[f]["vehicle_class"]) == vehicle_class]
                expr: Dict[int, float] = {}
                total_demand = 0.0
                for f in members:
                    total_demand += float(demand_map[f])
                    for var, coef in product_emission_expr[f].items():
                        add_term(expr, var, coef)
                cap = float(
                    scenario["small_cap_kgco2_per_vehicle"]
                    if vehicle_class == "small"
                    else scenario["standard_cap_kgco2_per_vehicle"]
                )
                model.add_constraint(expr, ub=cap * total_demand)
        else:
            return {"status": "INVALID_CAP_APPLICATION", "message": cap_application}

    result = model.solve(time_limit_sec=int(time_limit_sec))
    status = result.status
    if status not in {"OPTIMAL", "FEASIBLE"}:
        return {
            "status": status,
            "message": str(result.message),
            "backend": result.backend,
            "solver_message": str(result.message),
            "cap_application": cap_application,
            "wall_time_sec": result.wall_time_sec,
            "variable_count": result.variable_count,
            "integer_variable_count": result.integer_variable_count,
            "binary_variable_count": result.binary_variable_count,
            "constraint_count": result.constraint_count,
        }

    x = result.x
    tol = 1e-5
    raw_records = []
    for key, var in rt.items():
        amount = float(x[var])
        if amount <= tol:
            continue
        f, r, s_id, p, t = key
        product = product_map[f]
        supplier = supplier_map[s_id]
        plant = plant_map[p]
        meta = raw_route_meta[key]
        coefs = coefficient_meta[key]
        raw_records.append({
            "product_id": f,
            "product_name": product["product_name_ko"],
            "material_id": r,
            "material_name": MATERIAL_LABEL[r],
            "supplier_id": s_id,
            "supplier_location": supplier["location_name"],
            "plant_id": p,
            "plant_location": plant["location_name"],
            "transport_mode": t,
            "transport_mode_ko": TRANSPORT_LABEL[t],
            "flow_kg": amount,
            "distance_km": meta["distance_km"],
            "production_cost_eur": amount * coefs["prod_cost_coef"],
            "transport_cost_eur": amount * coefs["tr_cost_coef"],
            "production_emissions_kgco2": amount * coefs["prod_ef_coef"],
            "transport_emissions_kgco2": amount * coefs["tr_ef_coef"],
        })

    assembly_records = []
    for (f, p), var in fp.items():
        units = float(x[var])
        if units <= tol:
            continue
        product = product_map[f]
        plant = plant_map[p]
        assembly_records.append({
            "product_id": f,
            "product_name": product["product_name_ko"],
            "plant_id": p,
            "plant_location": plant["location_name"],
            "assembled_units": units,
            "assembly_mass_kg": units * float(product["nonbattery_mass_kg"]),
            "assembly_cost_eur": units * float(product["nonbattery_mass_kg"]) * float(plant["assembly_cost_eur_per_kg"]),
            "assembly_emissions_kgco2": units * float(product["nonbattery_mass_kg"]) * float(plant["assembly_ef_kgco2_per_kg"]),
        })

    final_records = []
    for key, var in ft.items():
        units = float(x[var])
        if units <= tol:
            continue
        f, p, t = key
        product = product_map[f]
        plant = plant_map[p]
        meta = final_route_meta[key]
        coefs = final_coefficient_meta[key]
        final_records.append({
            "product_id": f,
            "product_name": product["product_name_ko"],
            "plant_id": p,
            "plant_location": plant["location_name"],
            "market_id": market_id,
            "market_name": market["market_name"],
            "transport_mode": t,
            "transport_mode_ko": TRANSPORT_LABEL[t],
            "vehicle_units": units,
            "transport_mass_kg": units * float(product["vehicle_mass_kg"]),
            "distance_km": meta["distance_km"],
            "transport_cost_eur": units * coefs["cost_coef"],
            "transport_emissions_kgco2": units * coefs["ef_coef"],
        })

    raw_df = pd.DataFrame(raw_records)
    assembly_df = pd.DataFrame(assembly_records)
    final_df = pd.DataFrame(final_records)
    cost_breakdown = {
        "생산비": float(raw_df["production_cost_eur"].sum()) if not raw_df.empty else 0.0,
        "원자재 운송비": float(raw_df["transport_cost_eur"].sum()) if not raw_df.empty else 0.0,
        "조립비": float(assembly_df["assembly_cost_eur"].sum()) if not assembly_df.empty else 0.0,
        "완제품 운송비": float(final_df["transport_cost_eur"].sum()) if not final_df.empty else 0.0,
    }
    emission_breakdown = {
        "원자재·배터리 생산": float(raw_df["production_emissions_kgco2"].sum()) if not raw_df.empty else 0.0,
        "원자재 운송": float(raw_df["transport_emissions_kgco2"].sum()) if not raw_df.empty else 0.0,
        "조립": float(assembly_df["assembly_emissions_kgco2"].sum()) if not assembly_df.empty else 0.0,
        "완제품 운송": float(final_df["transport_emissions_kgco2"].sum()) if not final_df.empty else 0.0,
    }

    product_rows = []
    for f in F:
        prod_raw = raw_df[raw_df["product_id"] == f] if not raw_df.empty else pd.DataFrame()
        prod_assy = assembly_df[assembly_df["product_id"] == f] if not assembly_df.empty else pd.DataFrame()
        prod_final = final_df[final_df["product_id"] == f] if not final_df.empty else pd.DataFrame()
        total_emissions = (
            (prod_raw["production_emissions_kgco2"].sum() + prod_raw["transport_emissions_kgco2"].sum() if not prod_raw.empty else 0)
            + (prod_assy["assembly_emissions_kgco2"].sum() if not prod_assy.empty else 0)
            + (prod_final["transport_emissions_kgco2"].sum() if not prod_final.empty else 0)
        )
        total_cost = (
            (prod_raw["production_cost_eur"].sum() + prod_raw["transport_cost_eur"].sum() if not prod_raw.empty else 0)
            + (prod_assy["assembly_cost_eur"].sum() if not prod_assy.empty else 0)
            + (prod_final["transport_cost_eur"].sum() if not prod_final.empty else 0)
        )
        per_vehicle = total_emissions / demand_map[f]
        cap = np.nan
        if int(scenario["apply_carbon_cap"]) == 1:
            cap = float(scenario["small_cap_kgco2_per_vehicle"] if product_map[f]["vehicle_class"] == "small" else scenario["standard_cap_kgco2_per_vehicle"])
        score = subsidy_score(str(product_map[f]["vehicle_class"]), per_vehicle)
        product_rows.append({
            "product_id": f,
            "product_name": product_map[f]["product_name_ko"],
            "vehicle_class": str(product_map[f]["vehicle_class"]),
            "demand_units": demand_map[f],
            "total_cost_eur": total_cost,
            "cost_per_vehicle_eur": total_cost / demand_map[f],
            "total_emissions_kgco2": total_emissions,
            "emissions_per_vehicle_kgco2": per_vehicle,
            "carbon_cap_kgco2_per_vehicle": cap,
            "subsidy_score": score,
            "scenario_minimum_score": float(scenario["minimum_score"]),
            "individual_eligible": (score + 1e-6 >= float(scenario["minimum_score"])) if int(scenario["apply_carbon_cap"]) == 1 else True,
        })
    product_summary = pd.DataFrame(product_rows)
    class_rows = []
    for vehicle_class, grp in product_summary.groupby("vehicle_class"):
        total_demand = float(grp["demand_units"].sum())
        total_emissions = float(grp["total_emissions_kgco2"].sum())
        weighted_average = total_emissions / total_demand
        cap = np.nan
        if int(scenario["apply_carbon_cap"]) == 1:
            cap = float(
                scenario["small_cap_kgco2_per_vehicle"]
                if vehicle_class == "small"
                else scenario["standard_cap_kgco2_per_vehicle"]
            )
        class_rows.append({
            "vehicle_class": vehicle_class,
            "class_name": "소형" if vehicle_class == "small" else "중형·대형",
            "demand_units": total_demand,
            "total_emissions_kgco2": total_emissions,
            "weighted_average_kgco2_per_vehicle": weighted_average,
            "carbon_cap_kgco2_per_vehicle": cap,
            "cap_satisfied": bool(np.isnan(cap) or weighted_average <= cap + 1e-6),
        })
    class_summary = pd.DataFrame(class_rows)
    if cap_application == "class_average" and not class_summary.empty:
        class_eligibility = class_summary.set_index("vehicle_class")["cap_satisfied"].to_dict()
        product_summary["eligible"] = product_summary["vehicle_class"].map(class_eligibility).fillna(True)
    else:
        product_summary["eligible"] = product_summary["individual_eligible"]

    supplier_summary = (
        raw_df.groupby(["material_id", "material_name", "supplier_id", "supplier_location"], as_index=False)
        .agg(flow_kg=("flow_kg", "sum"), production_cost_eur=("production_cost_eur", "sum"), production_emissions_kgco2=("production_emissions_kgco2", "sum"))
        if not raw_df.empty else pd.DataFrame()
    )
    plant_summary = (
        assembly_df.groupby(["plant_id", "plant_location"], as_index=False)
        .agg(assembled_units=("assembled_units", "sum"), assembly_mass_kg=("assembly_mass_kg", "sum"), assembly_cost_eur=("assembly_cost_eur", "sum"), assembly_emissions_kgco2=("assembly_emissions_kgco2", "sum"))
        if not assembly_df.empty else pd.DataFrame()
    )

    return {
        "status": status,
        "backend": result.backend,
        "solver_message": str(result.message),
        "lp_optimality_gap": (
            abs(float(result.objective_value) - float(result.best_bound))
            / max(1.0, abs(float(result.objective_value)))
            if result.objective_value is not None and result.best_bound is not None
            else np.nan
        ),
        "mip_gap": np.nan,
        "model_type": "Strategic continuous LP",
        "transport_splitting": True,
        "objective_value_eur": float(result.objective_value),
        "production_mode": production_mode,
        "cap_application": cap_application,
        "loss_rate": loss_rate,
        "wall_time_sec": result.wall_time_sec,
        "variable_count": result.variable_count,
        "integer_variable_count": result.integer_variable_count,
        "binary_variable_count": result.binary_variable_count,
        "constraint_count": result.constraint_count,
        "scenario_id": scenario_id,
        "scenario_name": scenario["scenario_name"],
        "raw_routes": raw_df,
        "assembly": assembly_df,
        "final_routes": final_df,
        "product_summary": product_summary,
        "class_summary": class_summary,
        "supplier_summary": supplier_summary,
        "plant_summary": plant_summary,
        "cost_breakdown": cost_breakdown,
        "emission_breakdown": emission_breakdown,
        "total_cost_eur": sum(cost_breakdown.values()),
        "total_emissions_kgco2": sum(emission_breakdown.values()),
        "tables_snapshot": {
            "suppliers": suppliers,
            "plants": plants,
            "market": market.to_dict(),
        },
    }


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def render_supply_chain_map(result: Dict):
    suppliers = result["tables_snapshot"]["suppliers"].set_index("supplier_id")
    plants = result["tables_snapshot"]["plants"].set_index("plant_id")
    market = result["tables_snapshot"]["market"]
    raw = result["raw_routes"]
    final = result["final_routes"]

    all_lats = list(suppliers["latitude"]) + list(plants["latitude"]) + [float(market["latitude"])]
    all_lons = list(suppliers["longitude"]) + list(plants["longitude"]) + [float(market["longitude"])]
    m = folium.Map(location=[np.mean(all_lats), np.mean(all_lons)], zoom_start=2, tiles="CartoDB positron")
    Fullscreen().add_to(m)

    used_supplier_ids = set(raw["supplier_id"]) if not raw.empty else set()
    used_plant_ids = set(pd.concat([raw["plant_id"], final["plant_id"]], ignore_index=True)) if (not raw.empty or not final.empty) else set()

    for sid in used_supplier_ids:
        r = suppliers.loc[sid]
        folium.CircleMarker(
            [r["latitude"], r["longitude"]], radius=6, color="#333333", fill=True,
            tooltip=f"공급지: {r['location_name']} ({MATERIAL_LABEL.get(r['material_id'], r['material_id'])})",
        ).add_to(m)
    for pid in used_plant_ids:
        r = plants.loc[pid]
        folium.CircleMarker(
            [r["latitude"], r["longitude"]], radius=7, color="#111111", fill=True, fill_color="#fdae61",
            tooltip=f"조립지: {r['location_name']}",
        ).add_to(m)
    folium.Marker(
        [market["latitude"], market["longitude"]],
        icon=folium.Icon(color="red", icon="star"),
        tooltip=f"수요지: {market['market_name']}",
    ).add_to(m)

    max_raw = float(raw["flow_kg"].max()) if not raw.empty else 1.0
    for _, r in raw.iterrows():
        s = suppliers.loc[r["supplier_id"]]
        p = plants.loc[r["plant_id"]]
        weight = 1.0 + 7.0 * math.sqrt(float(r["flow_kg"]) / max_raw)
        folium.PolyLine(
            [[s["latitude"], s["longitude"]], [p["latitude"], p["longitude"]]],
            color=MATERIAL_COLOR[r["material_id"]], weight=weight, opacity=0.65,
            dash_array=TRANSPORT_DASH.get(r["transport_mode"]),
            tooltip=(f"{r['product_name']} | {r['material_name']} | {r['supplier_location']} → {r['plant_location']} | "
                     f"{r['transport_mode_ko']} | {r['flow_kg']:,.0f} kg"),
        ).add_to(m)

    max_final = float(final["transport_mass_kg"].max()) if not final.empty else 1.0
    for _, r in final.iterrows():
        p = plants.loc[r["plant_id"]]
        weight = 2.0 + 7.0 * math.sqrt(float(r["transport_mass_kg"]) / max_final)
        folium.PolyLine(
            [[p["latitude"], p["longitude"]], [market["latitude"], market["longitude"]]],
            color=MATERIAL_COLOR["finished"], weight=weight, opacity=0.7,
            dash_array=TRANSPORT_DASH.get(r["transport_mode"]),
            tooltip=(f"완제품 {r['product_name']} | {r['plant_location']} → {r['market_name']} | "
                     f"{r['transport_mode_ko']} | {r['vehicle_units']:,.0f}대"),
        ).add_to(m)

    legend = """
    <div style='position:fixed; bottom:20px; left:20px; z-index:9999; background:white;
      padding:10px 12px; border:1px solid #777; border-radius:6px; font-size:12px;'>
      <b>공급망 지도 범례</b><br>
      <span style='color:#d73027'>━━</span> 철강 &nbsp;
      <span style='color:#fc8d59'>━━</span> 알루미늄<br>
      <span style='color:#91cf60'>━━</span> 기타 원자재 &nbsp;
      <span style='color:#4575b4'>━━</span> 배터리<br>
      <span style='color:#542788'>━━</span> 완제품 운송<br>
      선 굵기 = 물량, 선 패턴 = 운송수단
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    st_folium(m, use_container_width=True, height=650)


def pie_figure(values: Dict[str, float], title: str):
    clean = {k: v for k, v in values.items() if v > 0}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.pie(clean.values(), labels=clean.keys(), autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()
    return fig


def product_carbon_figure(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(df))
    ax.bar(x, df["emissions_per_vehicle_kgco2"], label="최적화 탄소발자국")
    if df["carbon_cap_kgco2_per_vehicle"].notna().any():
        ax.scatter(x, df["carbon_cap_kgco2_per_vehicle"], marker="_", s=500, linewidths=3, label="시나리오 상한")
    ax.set_xticks(x)
    ax.set_xticklabels(df["product_name"], rotation=25, ha="right")
    ax.set_ylabel("kg CO₂-eq / vehicle")
    ax.set_title("제품별 탄소발자국과 정책 상한")
    ax.legend()
    fig.tight_layout()
    return fig


def benchmark_cost_ratio_figure(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    pct = df["modular_to_line_cost_ratio"] * 100
    bars = ax.bar(df["scenario_name"], pct)
    ax.axhline(100, linestyle="--", linewidth=1)
    ax.set_ylabel("모듈러 비용 / 라인 비용 (%)")
    ax.set_title("포스터 기준: 시나리오별 비용 비율")
    ax.set_ylim(min(99.5, pct.min() - 0.05), 100.1)
    for b, v in zip(bars, pct):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.4f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def benchmark_quartile_figure(df: pd.DataFrame):
    pivot = df.pivot_table(index=["scenario_id", "production_mode"], columns="quartile", values="share_pct", aggfunc="sum").reset_index()
    order = [("S1", "line"), ("S1", "modular"), ("S2", "line"), ("S2", "modular"), ("S3", "line"), ("S3", "modular")]
    pivot["order"] = pivot.apply(lambda r: order.index((r["scenario_id"], r["production_mode"])), axis=1)
    pivot = pivot.sort_values("order")
    labels = [f"{s}\n{'라인' if m=='line' else '모듈러'}" for s, m in zip(pivot["scenario_id"], pivot["production_mode"])]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bottom = np.zeros(len(pivot))
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        vals = pivot[q].to_numpy()
        ax.bar(labels, vals, bottom=bottom, label=q)
        bottom += vals
    ax.set_ylabel("운송량 비중 (%)")
    ax.set_title("포스터 기준: 생산방식별 운송량 사분위수 분포")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    return fig


def benchmark_difference_figure(df: pd.DataFrame):
    pivot = df.pivot_table(index=["scenario_id", "quartile"], columns="production_mode", values="share_pct").reset_index()
    pivot["difference_pp"] = pivot["modular"] - pivot["line"]
    labels = [f"{s}-{q}" for s, q in zip(pivot["scenario_id"], pivot["quartile"])]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(labels, pivot["difference_pp"])
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("모듈러 - 라인 (%p)")
    ax.set_title("포스터 기준: 사분위수 수량 차이")
    fig.tight_layout()
    return fig


def result_zip(result: Dict) -> bytes:
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, obj in [
            ("product_summary.csv", result["product_summary"]),
            ("class_summary.csv", result["class_summary"]),
            ("supplier_summary.csv", result["supplier_summary"]),
            ("plant_summary.csv", result["plant_summary"]),
            ("raw_material_routes.csv", result["raw_routes"]),
            ("assembly_results.csv", result["assembly"]),
            ("finished_vehicle_routes.csv", result["final_routes"]),
        ]:
            zf.writestr(name, obj.to_csv(index=False).encode("utf-8-sig"))
    return memory.getvalue()


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main():
    st.title("탄소배출 기반 제품 보조금 제도하의 전기차 공급망 최적화")
    st.caption(
        "모든 의사결정변수를 연속화하고 운송수단별 물량분할을 허용한 전략적 공급망 LP — Google OR-Tools MPSolver"
    )
    st.caption(f"build: {APP_BUILD} · package: {APP_PACKAGE_ID}")

    with st.sidebar:
        st.header("CSV 입력: 방식 C")
        st.info(
            "내장 데이터를 계산에 사용하지 않습니다. 아래 9개 CSV를 매 실행 세션마다 모두 업로드해야 합니다."
        )
        uploads = st.file_uploader(
            "관련 CSV 9개를 한 번에 업로드",
            type=["csv"],
            accept_multiple_files=True,
            help="파일명은 템플릿과 정확히 같아야 합니다. 같은 이름을 중복 업로드하지 마세요.",
        )
        st.download_button(
            "입력 CSV 템플릿 ZIP 다운로드",
            data=make_template_zip(),
            file_name="carbon_supply_chain_input_csv.zip",
            mime="application/zip",
            use_container_width=True,
        )
        st.markdown("**업로드 필수 파일명 9개**")
        st.code("\n".join(ALL_UPLOAD_FILES), language=None)

    uploaded = load_uploaded_data(uploads)
    tables = uploaded

    uploaded_names = set(uploaded)
    required_names = set(ALL_UPLOAD_FILES)
    missing_names = sorted(required_names - uploaded_names)
    unexpected_names = sorted(uploaded_names - required_names)

    if unexpected_names:
        st.warning("모형에서 사용하지 않는 CSV 파일: " + ", ".join(unexpected_names))

    if missing_names:
        st.warning(
            f"CSV 업로드 대기 중: {len(uploaded_names & required_names)}/9개 완료. "
            "누락 파일: " + ", ".join(missing_names)
        )
        st.info("왼쪽 사이드바에서 9개 CSV를 모두 선택하면 입력 검증과 최적화 화면이 활성화됩니다.")
        st.stop()

    errors = validate_data(tables)
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    st.success("방식 C 입력 완료: 업로드한 9개 CSV만 사용합니다.")

    tab_input, tab_run, tab_output, tab_poster, tab_model = st.tabs([
        "1. 입력 CSV", "2. 최적화 실행", "3. 최적화 Output", "4. 포스터 그림", "5. 수학모형 구현"
    ])

    with tab_input:
        st.header("입력 CSV 확인")
        st.warning(
            "생산·배출계수·비용·제품·수요·정책값은 Word/PPT에서 추출했습니다. "
            "Word에 수치가 없는 위경도와 공급·조립 용량은 실행 가능한 데모를 위한 구현 기본값이며 CSV에서 수정할 수 있습니다."
        )
        display_names = {
            "products.csv": "제품 사양",
            "demand.csv": "프랑스 제품 수요",
            "raw_material_suppliers.csv": "원자재·배터리 공급지",
            "assembly_locations.csv": "가공·조립 후보지",
            "transport_parameters.csv": "운송수단 비용·배출계수",
            "markets.csv": "수요지",
            "scenarios.csv": "정책 시나리오",
        }
        subtabs = st.tabs([display_names[n] for n in REQUIRED_FILES])
        for sub, name in zip(subtabs, REQUIRED_FILES):
            with sub:
                st.dataframe(tables[name], use_container_width=True, hide_index=True)
                st.download_button(
                    f"{name} 다운로드",
                    data=tables[name].to_csv(index=False).encode("utf-8-sig"),
                    file_name=name,
                    mime="text/csv",
                    key=f"download_{name}",
                )

    with tab_run:
        st.header("최적화 설정")
        products = tables["products.csv"]
        suppliers = tables["raw_material_suppliers.csv"]
        plants = tables["assembly_locations.csv"]
        scenarios = tables["scenarios.csv"]

        product_options = dict(zip(products["product_name_ko"], products["product_id"]))
        selected_product_names = st.multiselect(
            "완제품 선택",
            list(product_options),
            default=list(product_options),
        )
        selected_products = [product_options[x] for x in selected_product_names]

        c1, c2, c3, c4 = st.columns([1.0, 1.3, 1.8, 0.9])
        with c1:
            production_mode = st.radio(
                "배터리 생산방식",
                options=["line", "modular"],
                format_func=lambda x: MODE_LABEL[x],
            )
        with c2:
            scenario_map = dict(zip(scenarios["scenario_name"], scenarios["scenario_id"]))
            scenario_name = st.selectbox("정책 시나리오", list(scenario_map))
            scenario_id = scenario_map[scenario_name]
        with c3:
            cap_application = st.selectbox(
                "탄소상한 적용 단위",
                options=["product_strict", "class_average"],
                format_func=lambda x: CAP_APPLICATION_LABEL[x],
                help=(
                    "기본 모드는 각 트림의 평균 탄소발자국에 상한을 적용합니다. "
                    "차급 평균 모드는 포스터 비교실험용 추가 가정입니다."
                ),
            )
        with c4:
            time_limit = st.number_input("Solver 제한시간(초)", min_value=30, max_value=1800, value=300, step=30)

        if cap_application == "class_average":
            st.info(
                "이 모드는 Word/PDF에 명시된 제약이 아니라 포스터 결과 비교를 위한 추가 실험입니다. "
                "동일 차급 내 트림 간 탄소배출량 상쇄가 가능합니다."
            )
        else:
            st.warning(
                "각 트림의 총배출량을 해당 트림 수요로 나눈 평균값이 차급별 상한 이하가 되도록 적용합니다. "
                "현재 전략적 LP의 기본 탄소제약입니다."
            )

        st.subheader("후보 공급지·조립지")
        st.caption("기본값은 Word/PDF에 수록된 모든 공급지와 가공·조립 후보입니다. 일부 후보만 선택하면 용량 또는 탄소상한 때문에 실제 infeasible이 될 수 있습니다.")
        supplier_ids = []
        cols = st.columns(4)
        for col, material in zip(cols, ["steel", "aluminum", "other", "battery"]):
            with col:
                sub = suppliers[suppliers["material_id"] == material]
                options = dict(zip(sub["location_name"], sub["supplier_id"]))
                default_names = list(options)
                chosen = st.multiselect(MATERIAL_LABEL[material], list(options), default=default_names, key=f"supplier_{material}")
                supplier_ids.extend(options[x] for x in chosen)

        plant_options = dict(zip(plants["location_name"], plants["plant_id"]))
        default_plants = list(plant_options)
        chosen_plants = st.multiselect("가공·조립 위치", list(plant_options), default=default_plants)
        plant_ids = [plant_options[x] for x in chosen_plants]

        st.info(
            "전략적 LP는 모든 물량을 연속적으로 배분하고 하나의 경로에서도 여러 운송수단으로 분할할 수 있습니다. "
            "실행 전 낙관적 feasibility 진단은 자동 수행하지 않으며, Solver의 OPTIMAL/INFEASIBLE 상태를 최종 판정으로 사용합니다."
        )

        run = st.button("전략적 LP 최적화 실행", type="primary", use_container_width=True)
        if run:
            progress = st.progress(5, text="1/4 입력과 후보지 선택을 확인하고 있습니다.")
            progress.progress(25, text="2/4 연속 공급·조립·운송 흐름변수와 제약조건을 생성합니다.")
            with st.spinner("OR-Tools 선형계획 공급망 모형을 계산하고 있습니다..."):
                progress.progress(65, text="3/4 LP Solver가 최소비용 연속 배분해를 계산하고 있습니다.")
                result = solve_model(
                    tables=tables,
                    selected_products=selected_products,
                    selected_supplier_ids=supplier_ids,
                    selected_plant_ids=plant_ids,
                    production_mode=production_mode,
                    scenario_id=scenario_id,
                    cap_application=cap_application,
                    loss_rate=MATERIAL_LOSS_RATE,
                    time_limit_sec=int(time_limit),
                )
            progress.progress(100, text="4/4 LP 결과 정리가 완료되었습니다.")
            st.session_state["optimization_result"] = result
            if result["status"] in {"OPTIMAL", "FEASIBLE"}:
                st.success(f"{result['status']} 전략적 LP 해를 찾았습니다. Solver: {result['backend']}")
                st.info("'3. 최적화 Output' 탭에서 지도·표·그림과 결과 CSV를 확인하세요.")
            else:
                st.error(f"Solver 상태: {result.get('status')} — {result.get('message', '')}")
                st.code(
                    "\n".join([
                        f"Backend: {result.get('backend', '')}",
                        f"Wall time: {result.get('wall_time_sec', 0):.2f} sec",
                        f"Continuous variables: {result.get('variable_count', 0):,}",
                        f"Transport splitting: enabled",
                        f"Constraints: {result.get('constraint_count', 0):,}",
                        f"Cap application: {CAP_APPLICATION_LABEL.get(result.get('cap_application'), result.get('cap_application'))}",
                    ]),
                    language=None,
                )
                if result.get("status") == "NOT_SOLVED":
                    st.warning("이는 infeasible 판정이 아닙니다. 제한시간을 늘리거나 후보지를 줄인 뒤 다시 실행하세요.")
                elif result.get("status") == "INFEASIBLE":
                    st.warning("연속 물량분할까지 허용한 LP가 INFEASIBLE이므로, 선택된 공급·조립 용량 또는 탄소상한을 동시에 만족할 수 없습니다.")

    with tab_output:
        st.header("최적화 Output")
        result = st.session_state.get("optimization_result")
        if not result or result.get("status") not in {"OPTIMAL", "FEASIBLE"}:
            st.info("먼저 '2. 최적화 실행' 탭에서 최적화를 실행하세요.")
        else:
            st.caption(
                f"{MODE_LABEL[result['production_mode']]} | {result['scenario_name']} | "
                f"{CAP_APPLICATION_LABEL.get(result['cap_application'], result['cap_application'])} | Solver {result['backend']}"
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("총 공급망 비용", f"€ {result['total_cost_eur']:,.0f}")
            c2.metric("총 탄소배출량", f"{result['total_emissions_kgco2']/1_000_000:,.2f} kt CO₂-eq")
            c3.metric("제품 평균 비용", f"€ {result['product_summary']['cost_per_vehicle_eur'].mean():,.0f}/대")
            c4.metric("평균 보조금 점수", f"{result['product_summary']['subsidy_score'].mean():.1f}/80")

            with st.expander("Solver 계산 정보", expanded=False):
                st.write({
                    "상태": result["status"],
                    "계산시간(초)": round(result.get("wall_time_sec", 0.0), 3),
                    "모형": result.get("model_type"),
                    "전체 연속변수": result.get("variable_count"),
                    "정수 변수": result.get("integer_variable_count"),
                    "이진 변수": result.get("binary_variable_count"),
                    "운송수단 물량분할": result.get("transport_splitting"),
                    "제약조건": result.get("constraint_count"),
                    "LP optimality gap": result.get("lp_optimality_gap"),
                })

            if result.get("cap_application") == "class_average":
                st.subheader("차급별 수요가중 평균 탄소상한 결과")
                st.dataframe(result["class_summary"], use_container_width=True, hide_index=True)

            st.subheader("제품별 정책 충족 결과")
            display = result["product_summary"].copy()
            display["eligible"] = display["eligible"].map({True: "충족", False: "미충족"})
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.pyplot(product_carbon_figure(result["product_summary"]), use_container_width=True)

            st.subheader("비용·탄소 구성 그림")
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(pie_figure(result["cost_breakdown"], "총비용 구성"), use_container_width=True)
            with col2:
                st.pyplot(pie_figure(result["emission_breakdown"], "총 탄소배출량 구성"), use_container_width=True)

            st.subheader("최적 공급망 지도")
            render_supply_chain_map(result)

            st.subheader("공급지·조립지 배분")
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**원자재 공급지별 물량**")
                st.dataframe(result["supplier_summary"], use_container_width=True, hide_index=True)
            with col4:
                st.markdown("**조립지별 생산량**")
                st.dataframe(result["plant_summary"], use_container_width=True, hide_index=True)

            with st.expander("상세 경로 결과"):
                st.markdown("**원자재·배터리 공급 경로**")
                st.dataframe(result["raw_routes"], use_container_width=True, hide_index=True)
                st.markdown("**완제품 운송 경로**")
                st.dataframe(result["final_routes"], use_container_width=True, hide_index=True)

            st.download_button(
                "전체 최적화 결과 CSV ZIP 다운로드",
                data=result_zip(result),
                file_name=f"optimization_result_{result['scenario_id']}_{result['production_mode']}.zip",
                mime="application/zip",
                use_container_width=True,
            )

    with tab_poster:
        st.header("세 번째 파일의 최종 결과 그림")
        st.info(
            "이 탭은 2025 춘계산업공학회 포스터에 기재된 그림과 수치를 그대로 보여주는 벤치마크 영역입니다. "
            "현재 업로드 CSV로 다시 계산한 최적화 결과와 구분해 해석하세요."
        )
        poster_path = ASSET_DIR / "poster_reference.png"
        if poster_path.exists():
            st.image(str(poster_path), caption="세 번째 파일 원본 포스터", use_container_width=True)
        else:
            st.warning("assets/poster_reference.png가 없습니다.")

        if all(name in tables for name in BENCHMARK_FILES):
            ratios = tables["poster_benchmark_cost_ratios.csv"]
            quartiles = tables["poster_benchmark_quartiles.csv"]
            st.subheader("포스터 결과 재구성 그림")
            st.pyplot(benchmark_cost_ratio_figure(ratios), use_container_width=True)
            st.pyplot(benchmark_quartile_figure(quartiles), use_container_width=True)
            st.pyplot(benchmark_difference_figure(quartiles), use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(ratios, use_container_width=True, hide_index=True)
            with c2:
                st.dataframe(quartiles, use_container_width=True, hide_index=True)
        else:
            st.warning("포스터 벤치마크 CSV가 없습니다.")

    with tab_model:
        st.header("현재 SaaS 전략적 LP 수학모형")
        st.markdown(
            """
**모형 성격**  
모든 의사결정변수를 연속변수로 두는 단일기간 전략적 선형계획(LP)입니다. 차량대수·배터리 팩·모듈 수는 물리적 정수개수가 아니라 분석기간의 `등가물량`으로 해석합니다. 공급지→조립지와 조립지→프랑스의 각 운송수단별 물량을 독립적으로 결정하므로 하나의 경로를 여러 운송수단으로 분할할 수 있습니다.

**목적함수**  
원자재·배터리 생산비 + 공급지→조립지 운송비 + 가공·조립비 + 조립지→프랑스 완제품 운송비를 최소화합니다.

**모든 결정변수는 연속변수**

1. `RT[f,r,s,p,t]`: 제품별·재질별 공급지→조립지 운송량(kg)
2. `FP[f,p]`: 조립지별 차량 생산 등가대수
3. `FT[f,p,t]`: 조립지→프랑스 완제품 운송 등가대수
4. `ZL[f,s,p]`: 라인 방식 완성 배터리 팩 등가수량
5. `ZM[f,s,p]`, `ZS[f,s,p]`: 모듈 방식 메인·보조 모듈 등가수량

**핵심 제약조건**

1. 원자재 수급: 조립지 도착 재질량 = 차량당 BOM × 조립 등가대수
2. 배터리 수급: 라인 팩 또는 메인·보조 모듈 등가물량 = 조립 등가대수에 필요한 배터리량
3. 공급지 용량: 모든 제품·조립지·운송수단으로 보낸 총량 ≤ 공급지 용량
4. 조립지 용량: 배터리 제외 차량질량 × 조립 등가대수 ≤ 조립지 처리용량
5. 완제품 흐름보존: 조립 등가대수 = 프랑스로 보낸 운송수단별 등가대수의 합
6. 수요충족: 모든 조립지·운송수단의 프랑스 도착 등가대수 합 = 제품별 수요
7. 탄소상한: 기본적으로 각 트림의 평균 탄소발자국 ≤ 해당 차급의 보조금 상한

**운송수단 물량분할**  
운송수단 선택 이진변수와 Big-M 제약은 제거했습니다. 따라서 비용과 탄소의 절충이 필요한 경우 같은 공급지–조립지 경로에서도 해상·철도·도로·항공 물량을 나누어 사용할 수 있습니다. 비용과 배출량이 모두 열등한 운송수단은 최적해를 바꾸지 않는 범위에서 사전 제거됩니다.

**해석 주의**  
이 결과는 개별 차량·팩·모듈의 실행계획이 아니라 대규모 연간 수요에 대한 전략적 연속 배분 최적값입니다. 원래 정수·이진조건을 가진 MILP와 비교할 경우 이 LP 목적함수값은 MILP 최소비용의 이론적 하한이 됩니다.
            """
        )


if __name__ == "__main__":
    main()
