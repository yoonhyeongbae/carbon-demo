from __future__ import annotations

import io
import math
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Tuple

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
APP_BUILD = "route-binary-thread1-progress-v4"

CAP_APPLICATION_LABEL = {
    "class_average": "포스터 재현: 차급별 수요가중 평균 탄소상한",
    "product_strict": "PDF 엄격 적용: 각 트림별 개별 차량 탄소상한",
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


ProgressCallback = Optional[Callable[[int, str, str], None]]


def emit_progress(
    callback: ProgressCallback,
    percent: int,
    stage: str,
    detail: str = "",
) -> None:
    """UI와 독립적으로 단계 진행률을 전달합니다.

    OR-Tools MPSolver의 Solve() 호출 내부에서는 실시간 노드 진행률을
    직접 받을 수 없으므로, 이 콜백은 입력 검증·모형 생성·Solver 실행·
    결과 정리의 작업 단계 진행률을 제공합니다.
    """
    if callback is not None:
        callback(max(0, min(100, int(percent))), str(stage), str(detail))


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


@st.cache_data(show_spinner=False)
def distance_km(lat1, lon1, lat2, lon2) -> float:
    return float(geodesic((float(lat1), float(lon1)), (float(lat2), float(lon2))).km)


# -----------------------------------------------------------------------------
# Mathematical optimization model (Word formulation operationalized)
# -----------------------------------------------------------------------------
class MilpBuilder:
    """Word의 혼합정수선형계획 모형을 OR-Tools MPSolver로 생성한다.

    SCIP를 먼저 사용하고, 배포 환경에서 SCIP가 없을 때만 CBC로 대체한다.
    나머지 수학모형 코드는 변수 인덱스를 사용하므로, 이 클래스가 OR-Tools
    변수 객체와 인덱스 사이의 연결을 담당한다.
    """

    def __init__(self):
        self.c: List[float] = []
        self.var_lb: List[float] = []
        self.var_ub: List[float] = []
        self.var_kind: List[str] = []
        self.names: List[str] = []
        self.rows: List[Dict[int, float]] = []
        self.row_lb: List[float] = []
        self.row_ub: List[float] = []

    def add_var(self, lb: float, ub: float, kind: str, name: str) -> int:
        if kind not in {"C", "I", "B"}:
            raise ValueError(f"지원하지 않는 변수 종류: {kind}")
        idx = len(self.c)
        self.c.append(0.0)
        self.var_lb.append(float(lb))
        self.var_ub.append(float(ub))
        self.var_kind.append(kind)
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
            ("SCIP", "OR-Tools SCIP"),
            ("CBC_MIXED_INTEGER_PROGRAMMING", "OR-Tools CBC"),
            ("CBC", "OR-Tools CBC"),
        ]
        for solver_id, label in candidates:
            solver = pywraplp.Solver.CreateSolver(solver_id)
            if solver is not None:
                return solver, label
        return None, "OR-Tools MIP solver unavailable"

    def solve(
        self,
        time_limit_sec: int = 60,
        mip_gap: float = 0.01,
        progress_callback: ProgressCallback = None,
    ):
        emit_progress(progress_callback, 64, "Solver 초기화", "SCIP 우선, CBC 대체")
        solver, backend = self._create_solver()
        if solver is None:
            return SimpleNamespace(
                status="SOLVER_NOT_CREATED",
                x=None,
                message="SCIP 또는 CBC MIP solver를 생성하지 못했습니다. ortools 설치를 확인하세요.",
                backend=backend,
                objective_value=None,
                best_bound=None,
                wall_time_sec=0.0,
                variable_count=len(self.c),
                integer_variable_count=sum(k in {"I", "B"} for k in self.var_kind),
                binary_variable_count=sum(k == "B" for k in self.var_kind),
                constraint_count=len(self.rows),
            )

        solver.SetTimeLimit(max(1, int(time_limit_sec)) * 1000)

        # Streamlit Community Cloud의 CPU 사용량을 낮추기 위해 Solver를
        # 명시적으로 단일 스레드로 제한합니다. 지원하지 않는 backend에서는
        # 예외를 무시하고 해당 Solver의 기본 동작을 사용합니다.
        try:
            solver.SetNumThreads(1)
        except Exception:
            pass

        # mip_gap은 상대 optimality gap입니다. 0.01은 incumbent와 best bound의
        # 상대 차이가 1% 이내이면 실용적으로 충분한 해로 종료할 수 있음을 뜻합니다.
        mip_gap = max(0.0, float(mip_gap))
        if "SCIP" in backend:
            try:
                solver.SetSolverSpecificParametersAsString(
                    f"limits/gap = {mip_gap:.12g}\n"
                    "presolving/maxrounds = 10\n"
                    "parallel/maxnthreads = 1"
                )
            except Exception:
                pass

        infinity = solver.infinity()
        variables = []

        emit_progress(
            progress_callback,
            68,
            "OR-Tools 변수 변환",
            f"전체 {len(self.c):,}개 · 정수/이진 {sum(k in {'I', 'B'} for k in self.var_kind):,}개",
        )
        for lb, ub, kind, name in zip(self.var_lb, self.var_ub, self.var_kind, self.names):
            lower = -infinity if np.isneginf(lb) else float(lb)
            upper = infinity if np.isposinf(ub) else float(ub)
            if kind == "C":
                var = solver.NumVar(lower, upper, name)
            elif kind == "B":
                var = solver.IntVar(max(0.0, lower), min(1.0, upper), name)
            else:
                var = solver.IntVar(lower, upper, name)
            variables.append(var)

        emit_progress(
            progress_callback,
            74,
            "OR-Tools 제약식 변환",
            f"전체 {len(self.rows):,}개",
        )
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

        emit_progress(
            progress_callback,
            82,
            "SCIP 분기한정 탐색",
            f"단일 스레드 · 제한시간 {int(time_limit_sec)}초 · 허용 gap {mip_gap * 100:.2f}%",
        )
        started = time.perf_counter()
        status_code = solver.Solve()
        wall_time_sec = time.perf_counter() - started
        emit_progress(
            progress_callback,
            94,
            "Solver 종료",
            f"경과시간 {wall_time_sec:.2f}초",
        )

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
            integer_variable_count=sum(k in {"I", "B"} for k in self.var_kind),
            binary_variable_count=sum(k == "B" for k in self.var_kind),
            constraint_count=len(self.rows),
            configured_mip_gap=mip_gap,
            solver_threads=1,
        )

        if status in {"OPTIMAL", "FEASIBLE"}:
            values = np.array([var.solution_value() for var in variables], dtype=float)
            try:
                best_bound = float(objective.BestBound())
            except Exception:
                best_bound = None
            return SimpleNamespace(
                status=status,
                x=values,
                message=f"{status} solution found",
                objective_value=float(objective.Value()),
                best_bound=best_bound,
                **common,
            )

        reason = {
            "INFEASIBLE": "선택된 후보·용량·탄소상한을 동시에 만족하는 해가 존재하지 않습니다.",
            "NOT_SOLVED": "제한시간 안에 실행 가능한 해를 찾지 못했습니다. 이는 infeasible 판정과 다릅니다.",
            "UNBOUNDED": "목적함수가 무한히 감소할 수 있어 모형 또는 입력값을 확인해야 합니다.",
            "MODEL_INVALID": "OR-Tools가 모형을 유효하지 않은 것으로 판정했습니다.",
            "ABNORMAL": "Solver가 비정상 종료했습니다.",
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
    progress_callback: ProgressCallback = None,
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
    total_steps = max(1, len(products) * max(1, len(plants)))
    completed_steps = 0
    emit_progress(progress_callback, 10, "탄소 하한 준비", "제품·공급지·조립지 조합을 준비합니다.")
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
            completed_steps += 1
            pct = 15 + int(75 * completed_steps / total_steps)
            emit_progress(
                progress_callback,
                pct,
                "낙관적 탄소 하한 계산",
                f"{product['product_name_ko']} · {completed_steps:,}/{total_steps:,} 조합",
            )
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
    emit_progress(progress_callback, 100, "feasibility 진단 완료", "낙관적 탄소 하한 계산을 마쳤습니다.")
    return product_lb, pd.DataFrame(class_rows)


def solve_model(
    tables: Dict[str, pd.DataFrame],
    selected_products: List[str],
    selected_supplier_ids: List[str],
    selected_plant_ids: List[str],
    production_mode: str,
    scenario_id: str,
    cap_application: str = "class_average",
    loss_rate: float = MATERIAL_LOSS_RATE,
    time_limit_sec: int = 300,
    mip_gap: float = 0.01,
    progress_callback: ProgressCallback = None,
) -> Dict:
    emit_progress(progress_callback, 2, "입력 검증", "업로드 CSV와 사용자 선택을 확인합니다.")
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

    emit_progress(progress_callback, 8, "모형 데이터 준비", "수요·공급지·조립지 인덱스를 구성합니다.")
    model = MilpBuilder()
    F = list(product_map)
    P = list(plant_map)
    R = ["steel", "aluminum", "other", "battery"]
    material_mass_col = {
        "steel": "steel_kg", "aluminum": "aluminum_kg",
        "other": "other_material_kg", "battery": "battery_mass_kg",
    }

    fp: Dict[Tuple[str, str], int] = {}
    ft: Dict[Tuple[str, str, str], int] = {}
    beta: Dict[Tuple[str, str, str], int] = {}
    rt: Dict[Tuple[str, str, str, str, str], int] = {}
    alpha: Dict[Tuple[str, str, str, str, str], int] = {}
    line_count: Dict[Tuple[str, str, str], int] = {}
    main_count: Dict[Tuple[str, str, str], int] = {}
    sub_count: Dict[Tuple[str, str, str], int] = {}
    raw_route_meta: Dict[Tuple[str, str, str, str, str], Dict] = {}
    final_route_meta: Dict[Tuple[str, str, str], Dict] = {}

    emit_progress(progress_callback, 14, "완제품 흐름 변수 생성", f"제품 {len(F)}종 · 조립지 {len(P)}곳")
    # Assembly and finished-goods variables
    for f in F:
        D = demand_map[f]
        for p in P:
            fp[f, p] = model.add_var(0, D, "I", f"FP__{f}__{p}")
            plant = plant_map[p]
            modes = nondominated_modes(tp, str(plant["continent"]), str(market["continent"]), str(market["location_name"]), str(plant["location_name"]))
            region = route_region(str(plant["continent"]), str(market["continent"]), str(market["location_name"]), str(plant["location_name"]))
            dist = distance_km(plant["latitude"], plant["longitude"], market["latitude"], market["longitude"])
            for t in modes:
                ft[f, p, t] = model.add_var(0, D, "I", f"FT__{f}__{p}__{t}")
                if len(modes) > 1:
                    beta[f, p, t] = model.add_var(0, 1, "B", f"BETA__{f}__{p}__{t}")
                cost, ef = get_transport_parameter(tp, t, region)
                final_route_meta[f, p, t] = {"distance_km": dist, "cost": cost, "ef": ef, "region": region}
            expr = {fp[f, p]: -1.0}
            for t in modes:
                add_term(expr, ft[f, p, t])
            model.add_constraint(expr, 0.0, 0.0)
            if len(modes) > 1:
                model.add_constraint({beta[f, p, t]: 1.0 for t in modes}, ub=1.0)
                for t in modes:
                    model.add_constraint({ft[f, p, t]: 1.0, beta[f, p, t]: -D}, ub=0.0)

    emit_progress(progress_callback, 25, "원자재 경로 변수 생성", "사용하지 않는 경로는 운송수단 이진변수를 선택하지 않도록 구성합니다.")
    # Raw-material route variables and transport-mode selection
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
                        if len(modes) > 1:
                            alpha[key] = model.add_var(0, 1, "B", "ALPHA__" + "__".join(key))
                        cost, ef = get_transport_parameter(tp, t, region)
                        raw_route_meta[key] = {"distance_km": dist, "cost": cost, "ef": ef, "region": region}
                    if len(modes) > 1:
                        # 강제 선택(=1)이 아니라 선택 가능(<=1)으로 둡니다.
                        # 따라서 경로의 총 운송량이 0이면 모든 alpha가 0일 수 있으며,
                        # 사용하지 않는 경로 때문에 불필요한 이진 선택이 발생하지 않습니다.
                        model.add_constraint({alpha[f, r, s_id, p, t]: 1.0 for t in modes}, ub=1.0)
                        for t in modes:
                            model.add_constraint({rt[f, r, s_id, p, t]: 1.0, alpha[f, r, s_id, p, t]: -max_flow}, ub=0.0)

    emit_progress(progress_callback, 42, "수급·생산방식 제약 생성", "원자재 균형과 라인/모듈 조건을 추가합니다.")
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

    # Battery line or modular conditions from the Word formulation
    if production_mode == "line":
        for f in F:
            product = product_map[f]
            D = demand_map[f]
            for p in P:
                balance_expr: Dict[int, float] = {fp[f, p]: -1.0}
                for s_id in material_suppliers["battery"]:
                    line_count[f, s_id, p] = model.add_var(0, D, "I", f"ZL__{f}__{s_id}__{p}")
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
                    main_count[f, s_id, p] = model.add_var(0, int(product["main_module_count"]) * D, "I", f"ZM__{f}__{s_id}__{p}")
                    sub_count[f, s_id, p] = model.add_var(0, max(1, int(product["sub_module_count"]) * D), "I", f"ZS__{f}__{s_id}__{p}")
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

    emit_progress(progress_callback, 52, "용량·수요 제약 생성", "공급지·조립지 용량과 프랑스 수요를 추가합니다.")
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

    emit_progress(progress_callback, 58, "목적함수·탄소상한 생성", "비용 최소화 목적함수와 정책 제약을 구성합니다.")
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

    result = model.solve(
        time_limit_sec=int(time_limit_sec),
        mip_gap=float(mip_gap),
        progress_callback=progress_callback,
    )
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
            "configured_mip_gap": result.configured_mip_gap,
            "solver_threads": result.solver_threads,
        }

    emit_progress(progress_callback, 96, "결과 정리", "최적 경로·비용·탄소배출량 표를 생성합니다.")
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

    emit_progress(progress_callback, 100, "최적화 완료", "결과가 최적화 Output 페이지에 저장되었습니다.")
    return {
        "status": status,
        "backend": result.backend,
        "solver_message": str(result.message),
        "mip_gap": (
            abs(float(result.objective_value) - float(result.best_bound))
            / max(
                1e-12,
                min(abs(float(result.objective_value)), abs(float(result.best_bound))),
            )
            if (
                result.objective_value is not None
                and result.best_bound is not None
                and float(result.objective_value) * float(result.best_bound) >= 0
            )
            else np.inf
            if result.objective_value is not None and result.best_bound is not None
            else np.nan
        ),
        "objective_value_eur": float(result.objective_value),
        "production_mode": production_mode,
        "cap_application": cap_application,
        "loss_rate": loss_rate,
        "wall_time_sec": result.wall_time_sec,
        "variable_count": result.variable_count,
        "integer_variable_count": result.integer_variable_count,
        "binary_variable_count": result.binary_variable_count,
        "constraint_count": result.constraint_count,
        "configured_mip_gap": result.configured_mip_gap,
        "solver_threads": result.solver_threads,
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
        "Word/PDF 수학모형을 Google OR-Tools MPSolver(SCIP 우선, CBC 대체)로 구현한 Streamlit SaaS"
    )
    st.caption(f"build: {APP_BUILD}")

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
    errors = validate_data(tables) if not missing_names else []
    data_ready = not missing_names and not errors

    if unexpected_names:
        st.warning("모형에서 사용하지 않는 CSV 파일: " + ", ".join(unexpected_names))
    if missing_names:
        st.warning(
            f"CSV 업로드 대기 중: {len(uploaded_names & required_names)}/9개 완료. "
            "누락 파일: " + ", ".join(missing_names)
        )
    elif errors:
        for error in errors:
            st.error(error)
    else:
        st.success("방식 C 입력 완료: 업로드한 9개 CSV만 사용합니다.")

    page = st.radio(
        "페이지",
        [
            "1. 입력 CSV",
            "2. 최적화 실행",
            "3. 최적화 Output",
            "4. 포스터 그림",
            "5. 수학모형 구현",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="page_navigation",
    )

    if page == "1. 입력 CSV":
        st.header("입력 CSV 확인")
        if not data_ready:
            st.info("왼쪽 사이드바에서 9개 CSV를 모두 업로드하고 오류를 수정하면 표가 활성화됩니다.")
            return

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
        selected_table = st.selectbox(
            "확인할 CSV",
            REQUIRED_FILES,
            format_func=lambda name: display_names[name],
        )
        st.dataframe(tables[selected_table], use_container_width=True, hide_index=True)
        st.download_button(
            f"{selected_table} 다운로드",
            data=tables[selected_table].to_csv(index=False).encode("utf-8-sig"),
            file_name=selected_table,
            mime="text/csv",
        )
        return

    if page == "2. 최적화 실행":
        st.header("최적화 설정")

        if not data_ready:
            st.markdown("#### 실행 작업")
            c1, c2 = st.columns(2)
            c1.button(
                "실행 전 feasibility 진단",
                disabled=True,
                use_container_width=True,
                help="CSV 9개 업로드와 검증이 완료되면 활성화됩니다.",
            )
            c2.button(
                "수학적 최적화 실행",
                disabled=True,
                use_container_width=True,
                help="CSV 9개 업로드와 검증이 완료되면 활성화됩니다.",
            )
            st.caption("버튼이 희미한 동안에는 입력 준비가 끝나지 않은 상태입니다.")
            return

        products = tables["products.csv"]
        suppliers = tables["raw_material_suppliers.csv"]
        plants = tables["assembly_locations.csv"]
        scenarios = tables["scenarios.csv"]

        product_options = dict(zip(products["product_name_ko"], products["product_id"]))
        scenario_map = dict(zip(scenarios["scenario_name"], scenarios["scenario_id"]))

        with st.form("optimization_form", clear_on_submit=False):
            # 두 실행 버튼을 폼의 최상단에 먼저 선언하여 사용자가 페이지를 열자마자
            # 수행 가능한 작업을 확인하도록 합니다. 설정 위젯은 같은 폼 안에 있으므로
            # 버튼을 누르면 아래의 현재 선택값이 한 번에 제출됩니다.
            st.markdown("#### 실행 작업")
            action_col1, action_col2 = st.columns(2)
            diagnostic_requested = action_col1.form_submit_button(
                "실행 전 feasibility 진단",
                use_container_width=True,
                help="용량 검사와 낙관적 탄소 하한을 계산합니다. 최적화와는 별도 작업입니다.",
            )
            optimization_requested = action_col2.form_submit_button(
                "수학적 최적화 실행",
                type="primary",
                use_container_width=True,
                help="아래 설정 전체를 제출하여 OR-Tools SCIP 최적화를 실행합니다.",
            )
            st.caption(
                "버튼은 즉시 표시됩니다. 계산을 시작한 뒤에는 단계 진행률에서 현재 작업을 확인할 수 있습니다."
            )

            selected_product_names = st.multiselect(
                "완제품 선택",
                list(product_options),
                default=list(product_options),
            )
            selected_products = [product_options[name] for name in selected_product_names]

            c1, c2, c3, c4, c5 = st.columns([1.0, 1.35, 1.75, 0.85, 0.85])
            with c1:
                production_mode = st.radio(
                    "배터리 생산방식",
                    options=["line", "modular"],
                    format_func=lambda value: MODE_LABEL[value],
                )
            with c2:
                scenario_name = st.selectbox("정책 시나리오", list(scenario_map))
                scenario_id = scenario_map[scenario_name]
            with c3:
                cap_application = st.selectbox(
                    "탄소상한 적용 단위",
                    options=["class_average", "product_strict"],
                    format_func=lambda value: CAP_APPLICATION_LABEL[value],
                    help=(
                        "포스터 재현 모드는 소형 및 중형·대형 차급별 수요가중 평균에 상한을 적용합니다. "
                        "PDF 엄격 모드는 각 트림의 차량 1대당 탄소발자국에 상한을 개별 적용합니다."
                    ),
                )
            with c4:
                time_limit = st.number_input(
                    "제한시간(초)", min_value=30, max_value=1800, value=300, step=30
                )
            with c5:
                mip_gap_pct = st.number_input(
                    "허용 MIP gap(%)",
                    min_value=0.0,
                    max_value=20.0,
                    value=1.0,
                    step=0.1,
                    help="incumbent와 best bound의 상대 optimality gap입니다.",
                )

            st.subheader("후보 공급지·조립지")
            st.caption(
                "모든 후보가 기본 선택됩니다. 운송량이 0인 경로는 운송수단 이진변수를 강제로 선택하지 않도록 구성했습니다."
            )
            supplier_ids: List[str] = []
            cols = st.columns(4)
            for col, material in zip(cols, ["steel", "aluminum", "other", "battery"]):
                with col:
                    sub = suppliers[suppliers["material_id"] == material]
                    options = dict(zip(sub["location_name"], sub["supplier_id"]))
                    chosen = st.multiselect(
                        MATERIAL_LABEL[material],
                        list(options),
                        default=list(options),
                        key=f"supplier_{material}",
                    )
                    supplier_ids.extend(options[name] for name in chosen)

            plant_options = dict(zip(plants["location_name"], plants["plant_id"]))
            chosen_plants = st.multiselect(
                "가공·조립 위치",
                list(plant_options),
                default=list(plant_options),
            )
            plant_ids = [plant_options[name] for name in chosen_plants]

        selection_ready = bool(
            selected_products
            and plant_ids
            and all(
                any(
                    supplier_id in supplier_ids
                    for supplier_id in suppliers.loc[
                        suppliers["material_id"] == material, "supplier_id"
                    ]
                )
                for material in ["steel", "aluminum", "other", "battery"]
            )
        )

        if not selection_ready and (diagnostic_requested or optimization_requested):
            st.error("제품, 네 재질의 공급지, 가공·조립 위치를 각각 하나 이상 선택하세요.")
            return

        if diagnostic_requested:
            st.subheader("실행 전 feasibility 진단 진행률")
            diag_progress = st.progress(0, text="진단 준비 중")
            diag_stage = st.empty()

            def diagnostic_progress(percent: int, stage: str, detail: str = "") -> None:
                label = stage if not detail else f"{stage} — {detail}"
                diag_progress.progress(percent, text=label)
                diag_stage.caption(label)

            emit_progress(diagnostic_progress, 2, "빠른 용량 검사", "공급·조립 총용량을 합산합니다.")
            capacity_diag = selection_capacity_diagnostics(
                tables, selected_products, supplier_ids, plant_ids
            )
            emit_progress(diagnostic_progress, 8, "탄소 하한 계산 시작", "최적화와 독립된 낙관적 진단입니다.")
            product_lb, class_lb = optimistic_carbon_lower_bounds(
                tables,
                selected_products,
                supplier_ids,
                plant_ids,
                scenario_id,
                MATERIAL_LOSS_RATE,
                progress_callback=diagnostic_progress,
            )
            st.session_state["feasibility_result"] = {
                "capacity": capacity_diag,
                "product_lb": product_lb,
                "class_lb": class_lb,
                "scenario_id": scenario_id,
            }
            diag_progress.progress(100, text="feasibility 진단 완료")
            diag_stage.success("진단 결과가 아래에 표시되었습니다.")

        feasibility_result = st.session_state.get("feasibility_result")
        if feasibility_result:
            with st.expander("최근 feasibility 진단 결과", expanded=True):
                st.markdown("**공급·조립 용량 필요량과 선택후보 용량**")
                st.dataframe(
                    feasibility_result["capacity"], use_container_width=True, hide_index=True
                )
                if not feasibility_result["product_lb"].empty:
                    st.markdown("**공유용량을 무시한 제품별 낙관적 탄소 하한**")
                    st.dataframe(
                        feasibility_result["product_lb"],
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.markdown("**차급별 수요가중 낙관적 탄소 하한**")
                    st.dataframe(
                        feasibility_result["class_lb"],
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "하한이 상한보다 크면 확실히 infeasible입니다. 하한이 상한보다 작아도 공유용량·정수조건 때문에 feasible을 보장하지는 않습니다."
                    )

        if optimization_requested:
            st.subheader("최적화 작업 단계 진행률")
            optimization_progress = st.progress(0, text="최적화 준비 중")
            optimization_stage = st.empty()

            def ui_progress(percent: int, stage: str, detail: str = "") -> None:
                label = stage if not detail else f"{stage} — {detail}"
                optimization_progress.progress(percent, text=label)
                if percent < 100:
                    optimization_stage.info(label)
                else:
                    optimization_stage.success(label)

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
                mip_gap=float(mip_gap_pct) / 100.0,
                progress_callback=ui_progress,
            )
            st.session_state["optimization_result"] = result

            if result["status"] in {"OPTIMAL", "FEASIBLE"}:
                optimization_progress.progress(100, text="최적화 완료")
                st.success(f"{result['status']} 해를 찾았습니다. Solver: {result['backend']}")
                st.info("'3. 최적화 Output' 페이지에서 지도·표·그림과 결과 CSV를 확인하세요.")
            else:
                optimization_progress.progress(100, text=f"Solver 종료: {result.get('status')}")
                st.error(f"Solver 상태: {result.get('status')} — {result.get('message', '')}")
                st.code(
                    "\n".join(
                        [
                            f"Backend: {result.get('backend', '')}",
                            f"Wall time: {result.get('wall_time_sec', 0):.2f} sec",
                            f"Variables: {result.get('variable_count', 0):,}",
                            f"Integer variables: {result.get('integer_variable_count', 0):,}",
                            f"Binary variables: {result.get('binary_variable_count', 0):,}",
                            f"Constraints: {result.get('constraint_count', 0):,}",
                            f"SCIP threads: {result.get('solver_threads', 1)}",
                            f"Configured MIP gap: {result.get('configured_mip_gap', float(mip_gap_pct) / 100):.4f}",
                        ]
                    ),
                    language=None,
                )
                if result.get("status") == "NOT_SOLVED":
                    st.warning(
                        "이는 infeasible 판정이 아닙니다. 제한시간을 늘리거나 후보지를 줄인 뒤 다시 실행하세요."
                    )
                elif result.get("status") == "INFEASIBLE":
                    st.warning(
                        "선택 후보의 용량과 탄소 하한을 확인하세요. 후보를 추가해도 탄소 하한이 상한보다 높으면 실제 infeasible입니다."
                    )
        return

    if page == "3. 최적화 Output":
        st.header("최적화 Output")
        result = st.session_state.get("optimization_result")
        if not result or result.get("status") not in {"OPTIMAL", "FEASIBLE"}:
            st.info("먼저 '2. 최적화 실행' 페이지에서 최적화를 실행하세요.")
            return

        st.caption(
            f"{MODE_LABEL[result['production_mode']]} | {result['scenario_name']} | "
            f"{CAP_APPLICATION_LABEL.get(result['cap_application'], result['cap_application'])} | "
            f"Solver {result['backend']} · 단일 스레드"
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 공급망 비용", f"€ {result['total_cost_eur']:,.0f}")
        c2.metric("총 탄소배출량", f"{result['total_emissions_kgco2']/1_000_000:,.2f} kt CO₂-eq")
        c3.metric("제품 평균 비용", f"€ {result['product_summary']['cost_per_vehicle_eur'].mean():,.0f}/대")
        c4.metric("평균 보조금 점수", f"{result['product_summary']['subsidy_score'].mean():.1f}/80")

        with st.expander("Solver 계산 정보", expanded=False):
            st.write(
                {
                    "상태": result["status"],
                    "계산시간(초)": round(result.get("wall_time_sec", 0.0), 3),
                    "전체 변수": result.get("variable_count"),
                    "정수 변수": result.get("integer_variable_count"),
                    "이진 변수": result.get("binary_variable_count"),
                    "제약조건": result.get("constraint_count"),
                    "SCIP 스레드": result.get("solver_threads", 1),
                    "설정 MIP gap": result.get("configured_mip_gap"),
                    "최종 MIP gap": result.get("mip_gap"),
                }
            )

        if result.get("cap_application") == "class_average":
            st.subheader("차급별 수요가중 평균 탄소상한 결과")
            st.dataframe(result["class_summary"], use_container_width=True, hide_index=True)

        st.subheader("제품별 정책 충족 결과")
        display = result["product_summary"].copy()
        display["eligible"] = display["eligible"].map({True: "충족", False: "미충족"})
        st.dataframe(display, use_container_width=True, hide_index=True)
        fig = product_carbon_figure(result["product_summary"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("비용·탄소 구성 그림")
        col1, col2 = st.columns(2)
        with col1:
            fig = pie_figure(result["cost_breakdown"], "총비용 구성")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with col2:
            fig = pie_figure(result["emission_breakdown"], "총 탄소배출량 구성")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if st.checkbox("최적 공급망 지도 생성", value=False):
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
        return

    if page == "4. 포스터 그림":
        st.header("세 번째 파일의 최종 결과 그림")
        st.info(
            "이 페이지는 2025 춘계산업공학회 포스터에 기재된 그림과 수치를 그대로 보여주는 벤치마크 영역입니다. "
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
            chart_choice = st.selectbox(
                "재구성할 포스터 그림",
                ["시나리오별 비용 비율", "운송량 사분위수", "라인 대비 모듈 차이"],
            )
            if chart_choice == "시나리오별 비용 비율":
                fig = benchmark_cost_ratio_figure(ratios)
            elif chart_choice == "운송량 사분위수":
                fig = benchmark_quartile_figure(quartiles)
            else:
                fig = benchmark_difference_figure(quartiles)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(ratios, use_container_width=True, hide_index=True)
            with c2:
                st.dataframe(quartiles, use_container_width=True, hide_index=True)
        else:
            st.warning("포스터 벤치마크 CSV가 없습니다. CSV 업로드 전에도 원본 이미지는 확인할 수 있습니다.")
        return

    if page == "5. 수학모형 구현":
        st.header("Word 수학적 최적화 모형의 코드 대응")
        st.markdown(
            """
**목적함수**  
원자재·배터리 생산비 + 공급지→조립지 운송비 + 가공·조립비 + 조립지→프랑스 운송비를 최소화합니다.

**주요 제약조건**

1. 공급량과 운송량의 흐름 보존
2. 공급지 및 조립지 최대 용량
3. 제품별 철강·알루미늄·기타 원자재 수급
4. 라인 생산: 완성 배터리 팩 공급 횟수와 조립대수의 일치
5. 모듈러 생산: 10 kWh 메인 모듈과 5 kWh 보조 모듈 개수의 일치
6. 프랑스 제품 수요 충족
7. 위치 간 운송수단 최대 하나 선택: `Σ_t α ≤ 1`
8. 미사용 경로: 운송량이 0이면 모든 운송수단 이진변수가 0일 수 있음
9. 포스터 재현 모드: 차급별 수요가중 평균 상한 / PDF 엄격 모드: 트림별 개별 상한

**Branch-and-bound와 LP relaxation**  
코드에서 별도의 반복문으로 직접 작성하지는 않지만, OR-Tools MPSolver가 SCIP/CBC에 정수·이진변수가 포함된 MIP를 전달하면 Solver 내부에서 LP relaxation, branch-and-bound 및 절단평면을 수행합니다.

**CPU 제한 설정**  
SCIP 및 MPSolver를 단일 스레드로 제한하여 Streamlit Community Cloud의 순간 CPU 사용량을 낮춥니다.

**MIP gap**  
현재 최선의 실행 가능 해(incumbent)와 best bound 사이의 상대 optimality gap입니다. 기본 1%는 두 값의 차이가 목적함수 기준 1% 이내이면 종료할 수 있음을 뜻합니다.

**단위 정합화**  
Word 모형의 배터리 운송변수는 kg, 생산비·배출계수는 kWh 기준이므로, 제품별 `battery_kWh / battery_mass_kg`를 이용해 코드에서 kg↔kWh를 변환합니다. 철강·알루미늄 생산 탄소배출량에는 첫 번째 PDF의 손실률 0.3을 반영하여 배출계수를 `1/(1-0.3)`배 적용합니다. 비용과 배출량이 모두 열등한 운송수단은 최적해를 바꾸지 않는 범위에서 사전 제거합니다.
            """
        )


if __name__ == "__main__":
    main()
