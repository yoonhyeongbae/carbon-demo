from __future__ import annotations

import gc
import io
import math
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import streamlit as st
from geopy.distance import geodesic
from ortools.linear_solver import pywraplp


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
APP_BUILD = "streamlit-cloud-summary-frontier-v5"

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
@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")


@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)
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


@st.cache_data(show_spinner=False, ttl=3600, max_entries=1)
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


@lru_cache(maxsize=4096)
def distance_km(lat1, lon1, lat2, lon2) -> float:
    """작은 스칼라 거리값만 프로세스 내 LRU 캐시에 보관합니다."""
    return float(geodesic((float(lat1), float(lon1)), (float(lat2), float(lon2))).km)


def carbon_cap_from_score(vehicle_class: str, score: float) -> float:
    """프랑스 보조금 선형 점수식에서 점수에 대응하는 차량 1대당 상한을 반환합니다."""
    score = min(80.0, max(0.0, float(score)))
    if vehicle_class == "small":
        low, high = 6000.0, 17000.0
    else:
        low, high = 12000.0, 21000.0
    return high - (score / 80.0) * (high - low)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def build_route_catalog(
    suppliers: pd.DataFrame,
    plants: pd.DataFrame,
    market_dict: Dict,
    tp: pd.DataFrame,
) -> Dict:
    """선택 후보의 거리·가능 운송수단·단위계수를 한 번만 계산합니다.

    Community Cloud에서 동일 후보로 진단과 최적화를 반복할 때 geodesic과
    pandas 필터링을 다시 수행하지 않도록 제한된 캐시에 저장합니다.
    """
    raw_modes: Dict[Tuple[str, str], Tuple[str, ...]] = {}
    raw_meta: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    final_modes: Dict[str, Tuple[str, ...]] = {}
    final_meta: Dict[Tuple[str, str], Dict[str, float]] = {}

    for supplier in suppliers.itertuples(index=False):
        for plant in plants.itertuples(index=False):
            modes = tuple(nondominated_modes(
                tp,
                str(supplier.continent),
                str(plant.continent),
                str(plant.location_name),
                str(supplier.location_name),
            ))
            raw_modes[(str(supplier.supplier_id), str(plant.plant_id))] = modes
            region = route_region(
                str(supplier.continent),
                str(plant.continent),
                str(plant.location_name),
                str(supplier.location_name),
            )
            dist = distance_km(
                float(supplier.latitude), float(supplier.longitude),
                float(plant.latitude), float(plant.longitude),
            )
            for mode in modes:
                cost, ef = get_transport_parameter(tp, mode, region)
                raw_meta[(str(supplier.supplier_id), str(plant.plant_id), mode)] = {
                    "distance_km": dist, "cost": cost, "ef": ef, "region": region,
                }

    for plant in plants.itertuples(index=False):
        modes = tuple(nondominated_modes(
            tp,
            str(plant.continent),
            str(market_dict["continent"]),
            str(market_dict["location_name"]),
            str(plant.location_name),
        ))
        final_modes[str(plant.plant_id)] = modes
        region = route_region(
            str(plant.continent),
            str(market_dict["continent"]),
            str(market_dict["location_name"]),
            str(plant.location_name),
        )
        dist = distance_km(
            float(plant.latitude), float(plant.longitude),
            float(market_dict["latitude"]), float(market_dict["longitude"]),
        )
        for mode in modes:
            cost, ef = get_transport_parameter(tp, mode, region)
            final_meta[(str(plant.plant_id), mode)] = {
                "distance_km": dist, "cost": cost, "ef": ef, "region": region,
            }

    return {
        "raw_modes": raw_modes,
        "raw_meta": raw_meta,
        "final_modes": final_modes,
        "final_meta": final_meta,
    }


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
        variable_count = len(self.c)
        integer_variable_count = sum(k in {"I", "B"} for k in self.var_kind)
        binary_variable_count = sum(k == "B" for k in self.var_kind)
        constraint_count = len(self.rows)
        if solver is None:
            return SimpleNamespace(
                status="SOLVER_NOT_CREATED", x=None,
                message="SCIP 또는 CBC MIP solver를 생성하지 못했습니다. ortools 설치를 확인하세요.",
                backend=backend, objective_value=None, best_bound=None, wall_time_sec=0.0,
                variable_count=variable_count, integer_variable_count=integer_variable_count,
                binary_variable_count=binary_variable_count, constraint_count=constraint_count,
                configured_mip_gap=float(mip_gap), solver_threads=1,
            )

        solver.SetTimeLimit(max(1, int(time_limit_sec)) * 1000)
        try:
            solver.SetNumThreads(1)
        except Exception:
            pass

        mip_gap = max(0.0, float(mip_gap))
        if "SCIP" in backend:
            try:
                solver.SetSolverSpecificParametersAsString(
                    f"limits/gap = {mip_gap:.12g}\n"
                    "parallel/maxnthreads = 1\n"
                    "presolving/maxrounds = 10"
                )
            except Exception:
                pass

        infinity = solver.infinity()
        variables = []
        emit_progress(
            progress_callback, 68, "OR-Tools 변수 변환",
            f"전체 {variable_count:,}개 · 정수/이진 {integer_variable_count:,}개",
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

        emit_progress(progress_callback, 74, "OR-Tools 제약식 변환", f"전체 {constraint_count:,}개")
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

        # OR-Tools 내부 모형이 완성된 뒤 Python 중간 행렬을 해제하여
        # Solver 탐색 중 메모리 이중 보유를 줄입니다.
        self.c.clear(); self.var_lb.clear(); self.var_ub.clear(); self.var_kind.clear()
        self.names.clear(); self.rows.clear(); self.row_lb.clear(); self.row_ub.clear()
        gc.collect()

        emit_progress(
            progress_callback, 82, "SCIP 분기한정 탐색",
            f"단일 스레드 · 제한시간 {int(time_limit_sec)}초 · 허용 gap {mip_gap * 100:.2f}%",
        )
        started = time.perf_counter()
        status_code = solver.Solve()
        wall_time_sec = time.perf_counter() - started
        emit_progress(progress_callback, 94, "Solver 종료", f"경과시간 {wall_time_sec:.2f}초")

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
            backend=backend, wall_time_sec=wall_time_sec,
            variable_count=variable_count, integer_variable_count=integer_variable_count,
            binary_variable_count=binary_variable_count, constraint_count=constraint_count,
            configured_mip_gap=mip_gap, solver_threads=1,
        )

        if status in {"OPTIMAL", "FEASIBLE"}:
            values = np.fromiter((var.solution_value() for var in variables), dtype=float, count=variable_count)
            try:
                best_bound = float(objective.BestBound())
            except Exception:
                best_bound = None
            return SimpleNamespace(
                status=status, x=values, message=f"{status} solution found",
                objective_value=float(objective.Value()), best_bound=best_bound, **common,
            )

        reason = {
            "INFEASIBLE": "선택된 후보·용량·탄소상한을 동시에 만족하는 해가 존재하지 않습니다.",
            "NOT_SOLVED": "제한시간 안에 실행 가능한 해를 찾지 못했습니다. 이는 infeasible 판정과 다릅니다.",
            "UNBOUNDED": "목적함수가 무한히 감소할 수 있어 모형 또는 입력값을 확인해야 합니다.",
            "MODEL_INVALID": "OR-Tools가 모형을 유효하지 않은 것으로 판정했습니다.",
            "ABNORMAL": "Solver가 비정상 종료했습니다.",
        }.get(status, f"OR-Tools solve status: {status}")
        return SimpleNamespace(
            status=status, x=None, message=reason,
            objective_value=None, best_bound=None, **common,
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
    products = products[products["product_id"].isin(selected_products)]
    demand_map = demand[demand["product_id"].isin(selected_products)].groupby("product_id")["demand_units"].sum().to_dict()
    suppliers = suppliers[suppliers["supplier_id"].isin(selected_supplier_ids)]
    plants = plants[plants["plant_id"].isin(selected_plant_ids)]
    product_map = products.set_index("product_id").to_dict("index")
    rows = []
    material_cols = {"steel": "steel_kg", "aluminum": "aluminum_kg", "other": "other_material_kg"}
    for material, col in material_cols.items():
        required = sum(float(product_map[f][col]) * float(demand_map.get(f, 0)) for f in product_map)
        available = float(suppliers.loc[suppliers["material_id"] == material, "capacity"].sum())
        margin = available - required
        rows.append({
            "검사항목": MATERIAL_LABEL[material] + " 공급용량", "필요량": required,
            "선택후보 용량": available, "여유량": margin, "단위": "kg",
            "판정": "충족" if margin >= -1e-6 else "부족",
            "쉬운 해석": "총용량만 보면 수요를 공급할 수 있습니다." if margin >= -1e-6 else "선택 공급지만으로는 필요한 물량을 공급할 수 없습니다.",
        })
    battery_required = sum(float(product_map[f]["battery_kwh"]) * float(demand_map.get(f, 0)) for f in product_map)
    battery_available = float(suppliers.loc[suppliers["material_id"] == "battery", "capacity"].sum())
    margin = battery_available - battery_required
    rows.append({
        "검사항목": "배터리 공급용량", "필요량": battery_required, "선택후보 용량": battery_available,
        "여유량": margin, "단위": "kWh", "판정": "충족" if margin >= -1e-6 else "부족",
        "쉬운 해석": "선택 배터리 공급지의 총 kWh 용량이 충분합니다." if margin >= -1e-6 else "선택 배터리 공급지의 총 kWh 용량이 부족합니다.",
    })
    assembly_required = sum(float(product_map[f]["nonbattery_mass_kg"]) * float(demand_map.get(f, 0)) for f in product_map)
    assembly_available = float(plants["capacity_kg"].sum())
    margin = assembly_available - assembly_required
    rows.append({
        "검사항목": "가공·조립 용량", "필요량": assembly_required, "선택후보 용량": assembly_available,
        "여유량": margin, "단위": "kg", "판정": "충족" if margin >= -1e-6 else "부족",
        "쉬운 해석": "선택 조립지의 총 처리용량이 충분합니다." if margin >= -1e-6 else "선택 조립지의 총 처리용량이 부족합니다.",
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
    """공유용량·정수조건을 무시한 낙관적 탄소 하한입니다."""
    products = tables["products.csv"]
    demand = tables["demand.csv"]
    suppliers = tables["raw_material_suppliers.csv"]
    plants = tables["assembly_locations.csv"]
    tp = tables["transport_parameters.csv"]
    markets = tables["markets.csv"]
    scenarios = tables["scenarios.csv"]
    products = products[products["product_id"].isin(selected_products)]
    suppliers = suppliers[suppliers["supplier_id"].isin(selected_supplier_ids)]
    plants = plants[plants["plant_id"].isin(selected_plant_ids)]
    scenario = scenarios.loc[scenarios["scenario_id"] == scenario_id].iloc[0]
    demand_map = demand[demand["product_id"].isin(selected_products)].groupby("product_id")["demand_units"].sum().to_dict()
    market_id = str(demand.loc[demand["product_id"].isin(selected_products), "market_id"].iloc[0])
    market = markets.loc[markets["market_id"] == market_id].iloc[0]
    catalog = build_route_catalog(suppliers, plants, market.to_dict(), tp)
    supplier_map = suppliers.set_index("supplier_id").to_dict("index")

    rows = []
    total_steps = max(1, len(products) * max(1, len(plants)))
    completed_steps = 0
    emit_progress(progress_callback, 10, "탄소 하한 준비", "캐시된 거리·운송계수를 사용합니다.")
    for product in products.itertuples(index=False):
        best = np.inf
        for plant in plants.itertuples(index=False):
            total = float(product.nonbattery_mass_kg) * float(plant.assembly_ef_kgco2_per_kg)
            feasible = True
            for material, quantity_attr in [
                ("steel", "steel_kg"), ("aluminum", "aluminum_kg"),
                ("other", "other_material_kg"), ("battery", "battery_mass_kg"),
            ]:
                candidate_values = []
                material_ids = suppliers.loc[suppliers["material_id"] == material, "supplier_id"].astype(str)
                for supplier_id in material_ids:
                    supplier = supplier_map[supplier_id]
                    modes = catalog["raw_modes"].get((supplier_id, str(plant.plant_id)), ())
                    if not modes:
                        continue
                    min_transport_ef = min(catalog["raw_meta"][(supplier_id, str(plant.plant_id), mode)]["ef"] for mode in modes)
                    dist = catalog["raw_meta"][(supplier_id, str(plant.plant_id), modes[0])]["distance_km"]
                    if material == "battery":
                        production = float(product.battery_kwh) * float(supplier["production_ef"])
                        transport = float(product.battery_mass_kg) * dist * min_transport_ef
                    else:
                        quantity = float(getattr(product, quantity_attr))
                        factor = 1.0 / (1.0 - loss_rate) if material in {"steel", "aluminum"} else 1.0
                        production = quantity * float(supplier["production_ef"]) * factor
                        transport = quantity * dist * min_transport_ef
                    candidate_values.append(production + transport)
                if not candidate_values:
                    feasible = False
                    break
                total += min(candidate_values)
            if feasible:
                modes = catalog["final_modes"].get(str(plant.plant_id), ())
                if modes:
                    min_final = min(
                        catalog["final_meta"][(str(plant.plant_id), mode)]["distance_km"]
                        * catalog["final_meta"][(str(plant.plant_id), mode)]["ef"]
                        for mode in modes
                    )
                    total += float(product.vehicle_mass_kg) * min_final
                    best = min(best, total)
            completed_steps += 1
            emit_progress(
                progress_callback, 15 + int(75 * completed_steps / total_steps),
                "낙관적 탄소 하한 계산",
                f"{product.product_name_ko} · {completed_steps:,}/{total_steps:,} 조합",
            )
        cap = np.nan
        if int(scenario["apply_carbon_cap"]) == 1:
            cap = float(scenario["small_cap_kgco2_per_vehicle"] if product.vehicle_class == "small" else scenario["standard_cap_kgco2_per_vehicle"])
        possible = bool(np.isnan(cap) or best <= cap + 1e-6)
        rows.append({
            "product_id": product.product_id, "product_name": product.product_name_ko,
            "vehicle_class": product.vehicle_class,
            "demand_units": float(demand_map.get(product.product_id, 0)),
            "optimistic_lower_bound_kgco2_per_vehicle": best,
            "carbon_cap_kgco2_per_vehicle": cap,
            "cap_margin_kgco2_per_vehicle": cap - best if not np.isnan(cap) else np.nan,
            "strict_product_possible": possible,
            "쉬운 해석": (
                "이 낙관적 하한도 상한을 넘으므로 개별 트림 기준에서는 확실히 불가능합니다."
                if not possible else
                "가장 유리한 조합에서는 상한 아래가 가능하지만, 공유용량·정수조건을 포함한 최종 가능 여부는 MILP가 결정합니다."
            ),
        })
    product_lb = pd.DataFrame(rows)
    class_rows = []
    for vehicle_class, grp in product_lb.groupby("vehicle_class"):
        total_demand = float(grp["demand_units"].sum())
        avg_lb = float((grp["optimistic_lower_bound_kgco2_per_vehicle"] * grp["demand_units"]).sum() / total_demand)
        cap = float(grp["carbon_cap_kgco2_per_vehicle"].dropna().iloc[0]) if grp["carbon_cap_kgco2_per_vehicle"].notna().any() else np.nan
        possible = bool(np.isnan(cap) or avg_lb <= cap + 1e-6)
        class_rows.append({
            "vehicle_class": vehicle_class,
            "class_name": "소형" if vehicle_class == "small" else "중형·대형",
            "optimistic_weighted_average_kgco2_per_vehicle": avg_lb,
            "carbon_cap_kgco2_per_vehicle": cap,
            "cap_margin_kgco2_per_vehicle": cap - avg_lb if not np.isnan(cap) else np.nan,
            "class_average_possible": possible,
            "쉬운 해석": (
                "차급 평균의 낙관적 하한도 상한을 넘으므로 차급 평균 모드에서 확실히 불가능합니다."
                if not possible else
                "차급 평균 하한은 상한 아래입니다. 다만 최종 feasible 여부는 공유용량과 정수조건을 포함한 MILP로 확인해야 합니다."
            ),
        })
    emit_progress(progress_callback, 100, "feasibility 진단 완료", "용량과 낙관적 탄소 하한을 계산했습니다.")
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
    time_limit_sec: int = 120,
    mip_gap: float = 0.01,
    progress_callback: ProgressCallback = None,
    score_override: Optional[float] = None,
    return_detail: bool = True,
) -> Dict:
    emit_progress(progress_callback, 2, "입력 검증", "업로드 CSV와 사용자 선택을 확인합니다.")
    products = tables["products.csv"]
    demand = tables["demand.csv"]
    suppliers = tables["raw_material_suppliers.csv"]
    plants = tables["assembly_locations.csv"]
    tp = tables["transport_parameters.csv"]
    markets = tables["markets.csv"]
    scenarios = tables["scenarios.csv"]

    products = products[products["product_id"].isin(selected_products)].copy()
    demand = demand[demand["product_id"].isin(selected_products)].copy()
    suppliers = suppliers[suppliers["supplier_id"].isin(selected_supplier_ids)].copy()
    plants = plants[plants["plant_id"].isin(selected_plant_ids)].copy()
    if products.empty or demand.empty or suppliers.empty or plants.empty:
        return {"status": "INVALID_SELECTION", "message": "제품·공급지·조립지 선택을 확인하세요."}

    market_id = str(demand["market_id"].iloc[0])
    market_row = markets[markets["market_id"] == market_id]
    scenario_row = scenarios[scenarios["scenario_id"] == scenario_id]
    if market_row.empty:
        return {"status": "INVALID_MARKET", "message": f"markets.csv에 {market_id}가 없습니다."}
    if scenario_row.empty:
        return {"status": "INVALID_SCENARIO", "message": f"scenarios.csv에 {scenario_id}가 없습니다."}
    market = market_row.iloc[0]
    scenario = scenario_row.iloc[0]

    if score_override is None:
        minimum_score = float(scenario["minimum_score"])
        apply_carbon_cap = bool(int(scenario["apply_carbon_cap"]))
        small_cap = float(scenario["small_cap_kgco2_per_vehicle"])
        standard_cap = float(scenario["standard_cap_kgco2_per_vehicle"])
        scenario_name_effective = str(scenario["scenario_name"])
    else:
        minimum_score = min(80.0, max(0.0, float(score_override)))
        apply_carbon_cap = True
        small_cap = carbon_cap_from_score("small", minimum_score)
        standard_cap = carbon_cap_from_score("standard", minimum_score)
        scenario_name_effective = f"보조금 점수 {minimum_score:g}점 민감도"

    product_map = products.set_index("product_id").to_dict("index")
    demand_map = demand.groupby("product_id")["demand_units"].sum().astype(int).to_dict()
    supplier_map = suppliers.set_index("supplier_id").to_dict("index")
    plant_map = plants.set_index("plant_id").to_dict("index")
    material_suppliers = {
        m: suppliers.loc[suppliers["material_id"] == m, "supplier_id"].astype(str).tolist()
        for m in ["steel", "aluminum", "other", "battery"]
    }
    if any(not ids for ids in material_suppliers.values()):
        return {"status": "INVALID_SELECTION", "message": "선택된 후보 중 일부 원자재 공급지가 없습니다."}

    emit_progress(progress_callback, 8, "경로 데이터 준비", "거리·운송수단·단위계수를 캐시에서 불러옵니다.")
    catalog = build_route_catalog(suppliers, plants, market.to_dict(), tp)
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
    for f in F:
        D = demand_map[f]
        for p in P:
            fp[f, p] = model.add_var(0, D, "I", f"FP__{f}__{p}")
            modes = catalog["final_modes"].get(p, ())
            if not modes:
                continue
            for t in modes:
                ft[f, p, t] = model.add_var(0, D, "I", f"FT__{f}__{p}__{t}")
                if len(modes) > 1:
                    beta[f, p, t] = model.add_var(0, 1, "B", f"BETA__{f}__{p}__{t}")
                final_route_meta[f, p, t] = catalog["final_meta"][(p, t)]
            expr = {fp[f, p]: -1.0}
            for t in modes:
                add_term(expr, ft[f, p, t])
            model.add_constraint(expr, 0.0, 0.0)
            if len(modes) > 1:
                model.add_constraint({beta[f, p, t]: 1.0 for t in modes}, ub=1.0)
                for t in modes:
                    model.add_constraint({ft[f, p, t]: 1.0, beta[f, p, t]: -D}, ub=0.0)

    emit_progress(progress_callback, 25, "원자재 경로 변수 생성", "미사용 경로의 운송수단 이진변수는 모두 0이 될 수 있습니다.")
    for f in F:
        D = demand_map[f]
        for r in R:
            max_flow = float(product_map[f][material_mass_col[r]]) * D
            for s_id in material_suppliers[r]:
                for p in P:
                    modes = catalog["raw_modes"].get((s_id, p), ())
                    if not modes:
                        continue
                    for t in modes:
                        key = (f, r, s_id, p, t)
                        rt[key] = model.add_var(0.0, max_flow, "C", "RT__" + "__".join(key))
                        if len(modes) > 1:
                            alpha[key] = model.add_var(0, 1, "B", "ALPHA__" + "__".join(key))
                        raw_route_meta[key] = catalog["raw_meta"][(s_id, p, t)]
                    if len(modes) > 1:
                        model.add_constraint({alpha[f, r, s_id, p, t]: 1.0 for t in modes}, ub=1.0)
                        for t in modes:
                            model.add_constraint({rt[f, r, s_id, p, t]: 1.0, alpha[f, r, s_id, p, t]: -max_flow}, ub=0.0)

    emit_progress(progress_callback, 42, "수급·생산방식 제약 생성", "원자재 균형과 라인/모듈 조건을 추가합니다.")
    for f in F:
        product = product_map[f]
        for p in P:
            for r in ["steel", "aluminum", "other"]:
                expr: Dict[int, float] = {fp[f, p]: -float(product[material_mass_col[r]])}
                for s_id in material_suppliers[r]:
                    for t in catalog["raw_modes"].get((s_id, p), ()):
                        add_term(expr, rt[f, r, s_id, p, t])
                model.add_constraint(expr, 0.0, 0.0)

    if production_mode == "line":
        for f in F:
            product = product_map[f]; D = demand_map[f]
            for p in P:
                balance_expr: Dict[int, float] = {fp[f, p]: -1.0}
                for s_id in material_suppliers["battery"]:
                    line_count[f, s_id, p] = model.add_var(0, D, "I", f"ZL__{f}__{s_id}__{p}")
                    add_term(balance_expr, line_count[f, s_id, p])
                    expr = {line_count[f, s_id, p]: -float(product["battery_mass_kg"])}
                    for t in catalog["raw_modes"].get((s_id, p), ()):
                        add_term(expr, rt[f, "battery", s_id, p, t])
                    model.add_constraint(expr, 0.0, 0.0)
                model.add_constraint(balance_expr, 0.0, 0.0)
    elif production_mode == "modular":
        for f in F:
            product = product_map[f]; D = demand_map[f]
            main_mass = 10.0 * float(product["battery_mass_kg"]) / float(product["battery_kwh"])
            sub_mass = 5.0 * float(product["battery_mass_kg"]) / float(product["battery_kwh"])
            for p in P:
                main_balance = {fp[f, p]: -int(product["main_module_count"])}
                sub_balance = {fp[f, p]: -int(product["sub_module_count"])}
                for s_id in material_suppliers["battery"]:
                    main_count[f, s_id, p] = model.add_var(0, int(product["main_module_count"]) * D, "I", f"ZM__{f}__{s_id}__{p}")
                    sub_count[f, s_id, p] = model.add_var(0, max(1, int(product["sub_module_count"]) * D), "I", f"ZS__{f}__{s_id}__{p}")
                    add_term(main_balance, main_count[f, s_id, p]); add_term(sub_balance, sub_count[f, s_id, p])
                    expr = {main_count[f, s_id, p]: -main_mass, sub_count[f, s_id, p]: -sub_mass}
                    for t in catalog["raw_modes"].get((s_id, p), ()):
                        add_term(expr, rt[f, "battery", s_id, p, t])
                    model.add_constraint(expr, 0.0, 0.0)
                model.add_constraint(main_balance, 0.0, 0.0)
                model.add_constraint(sub_balance, 0.0, 0.0)
    else:
        return {"status": "INVALID_MODE", "message": production_mode}

    emit_progress(progress_callback, 52, "용량·수요 제약 생성", "공급지·조립지 용량과 프랑스 수요를 추가합니다.")
    for r in R:
        for s_id in material_suppliers[r]:
            expr: Dict[int, float] = {}
            for f in F:
                kwh_per_kg = float(product_map[f]["battery_kwh"]) / float(product_map[f]["battery_mass_kg"])
                for p in P:
                    for t in catalog["raw_modes"].get((s_id, p), ()):
                        add_term(expr, rt[f, r, s_id, p, t], kwh_per_kg if r == "battery" else 1.0)
            model.add_constraint(expr, ub=float(supplier_map[s_id]["capacity"]))
    for p in P:
        model.add_constraint({fp[f, p]: float(product_map[f]["nonbattery_mass_kg"]) for f in F}, ub=float(plant_map[p]["capacity_kg"]))
    for f in F:
        expr: Dict[int, float] = {}
        for (ff, p, t), var in ft.items():
            if ff == f:
                add_term(expr, var)
        model.add_constraint(expr, float(demand_map[f]), float(demand_map[f]))

    emit_progress(progress_callback, 58, "목적함수·탄소상한 생성", "총비용 최소화 목적함수와 정책 제약을 구성합니다.")
    product_emission_expr: Dict[str, Dict[int, float]] = {f: {} for f in F}
    coefficient_meta: Dict[Tuple[str, str, str, str, str], Tuple[float, float, float, float]] = {}
    for key, var in rt.items():
        f, r, s_id, p, t = key
        product = product_map[f]; supplier = supplier_map[s_id]; meta = raw_route_meta[key]
        if r == "battery":
            factor = float(product["battery_kwh"]) / float(product["battery_mass_kg"])
            prod_cost_coef = float(supplier["production_cost"]) * factor
            prod_ef_coef = float(supplier["production_ef"]) * factor
        else:
            prod_cost_coef = float(supplier["production_cost"])
            loss_multiplier = 1.0 / (1.0 - loss_rate) if r in {"steel", "aluminum"} else 1.0
            prod_ef_coef = float(supplier["production_ef"]) * loss_multiplier
        tr_cost_coef = float(meta["cost"]) * float(meta["distance_km"])
        tr_ef_coef = float(meta["ef"]) * float(meta["distance_km"])
        assembly_ef_coef = float(plant_map[p]["assembly_ef_kgco2_per_kg"]) if r in {"steel", "aluminum", "other"} else 0.0
        model.add_obj(var, prod_cost_coef + tr_cost_coef)
        add_term(product_emission_expr[f], var, prod_ef_coef + tr_ef_coef + assembly_ef_coef)
        coefficient_meta[key] = (prod_cost_coef, prod_ef_coef, tr_cost_coef, tr_ef_coef)

    final_coefficient_meta: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
    for (f, p), var in fp.items():
        model.add_obj(var, float(product_map[f]["nonbattery_mass_kg"]) * float(plant_map[p]["assembly_cost_eur_per_kg"]))
    for key, var in ft.items():
        f, p, t = key; meta = final_route_meta[key]
        mass = float(product_map[f]["vehicle_mass_kg"])
        cost_coef = mass * float(meta["distance_km"]) * float(meta["cost"])
        ef_coef = mass * float(meta["distance_km"]) * float(meta["ef"])
        model.add_obj(var, cost_coef); add_term(product_emission_expr[f], var, ef_coef)
        final_coefficient_meta[key] = (cost_coef, ef_coef)

    if apply_carbon_cap:
        if cap_application == "product_strict":
            for f in F:
                cap = small_cap if str(product_map[f]["vehicle_class"]) == "small" else standard_cap
                model.add_constraint(product_emission_expr[f], ub=cap * demand_map[f])
        elif cap_application == "class_average":
            for vehicle_class in sorted({str(product_map[f]["vehicle_class"]) for f in F}):
                members = [f for f in F if str(product_map[f]["vehicle_class"]) == vehicle_class]
                expr: Dict[int, float] = {}; total_demand = 0.0
                for f in members:
                    total_demand += float(demand_map[f])
                    for var, coef in product_emission_expr[f].items():
                        add_term(expr, var, coef)
                cap = small_cap if vehicle_class == "small" else standard_cap
                model.add_constraint(expr, ub=cap * total_demand)
        else:
            return {"status": "INVALID_CAP_APPLICATION", "message": cap_application}

    result = model.solve(time_limit_sec=int(time_limit_sec), mip_gap=float(mip_gap), progress_callback=progress_callback)
    if result.status not in {"OPTIMAL", "FEASIBLE"}:
        return {
            "status": result.status, "message": str(result.message), "backend": result.backend,
            "solver_message": str(result.message), "cap_application": cap_application,
            "wall_time_sec": result.wall_time_sec, "variable_count": result.variable_count,
            "integer_variable_count": result.integer_variable_count,
            "binary_variable_count": result.binary_variable_count,
            "constraint_count": result.constraint_count, "configured_mip_gap": result.configured_mip_gap,
            "solver_threads": result.solver_threads,
        }

    x = result.x
    total_emissions_quick = sum(
        sum(float(coef) * float(x[var]) for var, coef in expr.items())
        for expr in product_emission_expr.values()
    )
    if not return_detail:
        return {
            "status": result.status, "backend": result.backend, "message": str(result.message),
            "total_cost_eur": float(result.objective_value),
            "total_emissions_kgco2": float(total_emissions_quick),
            "wall_time_sec": result.wall_time_sec, "configured_mip_gap": result.configured_mip_gap,
            "minimum_score": minimum_score, "small_cap": small_cap, "standard_cap": standard_cap,
        }

    emit_progress(progress_callback, 96, "결과 집계", "상세 행 대신 공급지·조립지·운송경로 집계표를 생성합니다.")
    tol = 1e-5
    cost_breakdown = {"생산비": 0.0, "원자재 운송비": 0.0, "조립비": 0.0, "완제품 운송비": 0.0}
    emission_breakdown = {"원자재·배터리 생산": 0.0, "원자재 운송": 0.0, "조립": 0.0, "완제품 운송": 0.0}
    product_acc = {f: {"cost": 0.0, "emissions": 0.0} for f in F}
    supplier_acc: Dict[Tuple, Dict] = {}
    supply_route_acc: Dict[Tuple, Dict] = {}
    plant_acc: Dict[Tuple, Dict] = {}
    finished_route_acc: Dict[Tuple, Dict] = {}

    for key, var in rt.items():
        amount = float(x[var])
        if amount <= tol:
            continue
        f, r, s_id, p, t = key
        product = product_map[f]; supplier = supplier_map[s_id]; plant = plant_map[p]
        prod_cost_coef, prod_ef_coef, tr_cost_coef, tr_ef_coef = coefficient_meta[key]
        production_cost = amount * prod_cost_coef
        transport_cost = amount * tr_cost_coef
        production_emissions = amount * prod_ef_coef
        transport_emissions = amount * tr_ef_coef
        battery_kwh = amount * float(product["battery_kwh"]) / float(product["battery_mass_kg"]) if r == "battery" else 0.0
        product_acc[f]["cost"] += production_cost + transport_cost
        product_acc[f]["emissions"] += production_emissions + transport_emissions
        cost_breakdown["생산비"] += production_cost; cost_breakdown["원자재 운송비"] += transport_cost
        emission_breakdown["원자재·배터리 생산"] += production_emissions; emission_breakdown["원자재 운송"] += transport_emissions

        skey = (r, s_id)
        srow = supplier_acc.setdefault(skey, {
            "material_id": r, "material_name": MATERIAL_LABEL[r], "supplier_id": s_id,
            "supplier_location": supplier["location_name"], "flow_kg": 0.0, "battery_kwh": 0.0,
            "production_cost_eur": 0.0, "transport_cost_eur": 0.0,
            "production_emissions_kgco2": 0.0, "transport_emissions_kgco2": 0.0,
        })
        for name, value in [("flow_kg", amount), ("battery_kwh", battery_kwh),
                            ("production_cost_eur", production_cost), ("transport_cost_eur", transport_cost),
                            ("production_emissions_kgco2", production_emissions), ("transport_emissions_kgco2", transport_emissions)]:
            srow[name] += value

        meta = raw_route_meta[key]
        rkey = (r, s_id, p, t)
        rrow = supply_route_acc.setdefault(rkey, {
            "material_id": r, "material_name": MATERIAL_LABEL[r],
            "supplier_id": s_id, "supplier_location": supplier["location_name"],
            "plant_id": p, "plant_location": plant["location_name"],
            "transport_mode": t, "transport_mode_ko": TRANSPORT_LABEL[t],
            "distance_km": float(meta["distance_km"]), "flow_kg": 0.0, "battery_kwh": 0.0,
            "production_cost_eur": 0.0, "transport_cost_eur": 0.0,
            "production_emissions_kgco2": 0.0, "transport_emissions_kgco2": 0.0,
        })
        for name, value in [("flow_kg", amount), ("battery_kwh", battery_kwh),
                            ("production_cost_eur", production_cost), ("transport_cost_eur", transport_cost),
                            ("production_emissions_kgco2", production_emissions), ("transport_emissions_kgco2", transport_emissions)]:
            rrow[name] += value

    for (f, p), var in fp.items():
        units = float(x[var])
        if units <= tol:
            continue
        product = product_map[f]; plant = plant_map[p]
        mass = units * float(product["nonbattery_mass_kg"])
        cost = mass * float(plant["assembly_cost_eur_per_kg"])
        emissions = mass * float(plant["assembly_ef_kgco2_per_kg"])
        product_acc[f]["cost"] += cost; product_acc[f]["emissions"] += emissions
        cost_breakdown["조립비"] += cost; emission_breakdown["조립"] += emissions
        prow = plant_acc.setdefault(p, {
            "plant_id": p, "plant_location": plant["location_name"], "assembled_units": 0.0,
            "assembly_mass_kg": 0.0, "assembly_cost_eur": 0.0, "assembly_emissions_kgco2": 0.0,
        })
        prow["assembled_units"] += units; prow["assembly_mass_kg"] += mass
        prow["assembly_cost_eur"] += cost; prow["assembly_emissions_kgco2"] += emissions

    for key, var in ft.items():
        units = float(x[var])
        if units <= tol:
            continue
        f, p, t = key; product = product_map[f]; plant = plant_map[p]
        cost_coef, ef_coef = final_coefficient_meta[key]; meta = final_route_meta[key]
        cost = units * cost_coef; emissions = units * ef_coef
        transport_mass = units * float(product["vehicle_mass_kg"])
        product_acc[f]["cost"] += cost; product_acc[f]["emissions"] += emissions
        cost_breakdown["완제품 운송비"] += cost; emission_breakdown["완제품 운송"] += emissions
        rkey = (p, market_id, t)
        row = finished_route_acc.setdefault(rkey, {
            "plant_id": p, "plant_location": plant["location_name"],
            "market_id": market_id, "market_name": market["market_name"],
            "transport_mode": t, "transport_mode_ko": TRANSPORT_LABEL[t],
            "distance_km": float(meta["distance_km"]), "vehicle_units": 0.0,
            "transport_mass_kg": 0.0, "transport_cost_eur": 0.0,
            "transport_emissions_kgco2": 0.0,
        })
        row["vehicle_units"] += units; row["transport_mass_kg"] += transport_mass
        row["transport_cost_eur"] += cost; row["transport_emissions_kgco2"] += emissions

    supplier_summary = pd.DataFrame(supplier_acc.values())
    supply_route_summary = pd.DataFrame(supply_route_acc.values())
    plant_summary = pd.DataFrame(plant_acc.values())
    finished_route_summary = pd.DataFrame(finished_route_acc.values())
    for df in [supplier_summary, supply_route_summary]:
        if not df.empty:
            df["total_cost_eur"] = df["production_cost_eur"] + df["transport_cost_eur"]
            df["total_emissions_kgco2"] = df["production_emissions_kgco2"] + df["transport_emissions_kgco2"]

    product_rows = []
    for f in F:
        vehicle_class = str(product_map[f]["vehicle_class"])
        total_emissions = float(product_acc[f]["emissions"])
        total_cost = float(product_acc[f]["cost"])
        per_vehicle = total_emissions / demand_map[f]
        cap = small_cap if vehicle_class == "small" else standard_cap
        cap_value = cap if apply_carbon_cap else np.nan
        score = subsidy_score(vehicle_class, per_vehicle)
        product_rows.append({
            "product_id": f, "product_name": product_map[f]["product_name_ko"],
            "vehicle_class": vehicle_class, "demand_units": demand_map[f],
            "total_cost_eur": total_cost, "cost_per_vehicle_eur": total_cost / demand_map[f],
            "total_emissions_kgco2": total_emissions,
            "emissions_per_vehicle_kgco2": per_vehicle,
            "carbon_cap_kgco2_per_vehicle": cap_value,
            "cap_margin_kgco2_per_vehicle": cap_value - per_vehicle if apply_carbon_cap else np.nan,
            "subsidy_score": score, "scenario_minimum_score": minimum_score,
            "individual_eligible": (score + 1e-6 >= minimum_score) if apply_carbon_cap else True,
        })
    product_summary = pd.DataFrame(product_rows)
    class_rows = []
    for vehicle_class, grp in product_summary.groupby("vehicle_class"):
        total_demand = float(grp["demand_units"].sum())
        total_emissions = float(grp["total_emissions_kgco2"].sum())
        weighted_average = total_emissions / total_demand
        cap = small_cap if vehicle_class == "small" else standard_cap
        cap_value = cap if apply_carbon_cap else np.nan
        class_rows.append({
            "vehicle_class": vehicle_class, "class_name": "소형" if vehicle_class == "small" else "중형·대형",
            "demand_units": total_demand, "total_emissions_kgco2": total_emissions,
            "weighted_average_kgco2_per_vehicle": weighted_average,
            "carbon_cap_kgco2_per_vehicle": cap_value,
            "cap_margin_kgco2_per_vehicle": cap_value - weighted_average if apply_carbon_cap else np.nan,
            "cap_satisfied": bool(not apply_carbon_cap or weighted_average <= cap + 1e-4),
        })
    class_summary = pd.DataFrame(class_rows)
    if cap_application == "class_average" and not class_summary.empty:
        eligibility = class_summary.set_index("vehicle_class")["cap_satisfied"].to_dict()
        product_summary["eligible"] = product_summary["vehicle_class"].map(eligibility).fillna(True)
    else:
        product_summary["eligible"] = product_summary["individual_eligible"]

    cap_validation_warning = ""
    if apply_carbon_cap:
        if cap_application == "product_strict":
            violations = product_summary[product_summary["cap_margin_kgco2_per_vehicle"] < -0.05]
        else:
            violations = class_summary[class_summary["cap_margin_kgco2_per_vehicle"] < -0.05]
        if not violations.empty:
            cap_validation_warning = "Solver 결과와 사후 집계 상한 사이에 유의한 차이가 있습니다. 단위·제약·수치오차를 점검하세요."

    total_cost_eur = float(sum(cost_breakdown.values()))
    total_emissions_kgco2 = float(sum(emission_breakdown.values()))
    emit_progress(progress_callback, 100, "최적화 완료", "집계 결과가 Output 페이지에 저장되었습니다.")
    return {
        "status": result.status, "backend": result.backend, "solver_message": str(result.message),
        "mip_gap": (
            abs(float(result.objective_value) - float(result.best_bound)) /
            max(1e-12, min(abs(float(result.objective_value)), abs(float(result.best_bound))))
            if result.objective_value is not None and result.best_bound is not None and float(result.objective_value) * float(result.best_bound) >= 0
            else np.nan
        ),
        "objective_value_eur": float(result.objective_value),
        "objective_reconciliation_eur": float(result.objective_value) - total_cost_eur,
        "production_mode": production_mode, "cap_application": cap_application,
        "loss_rate": loss_rate, "wall_time_sec": result.wall_time_sec,
        "variable_count": result.variable_count, "integer_variable_count": result.integer_variable_count,
        "binary_variable_count": result.binary_variable_count, "constraint_count": result.constraint_count,
        "configured_mip_gap": result.configured_mip_gap, "solver_threads": result.solver_threads,
        "scenario_id": scenario_id, "scenario_name": scenario_name_effective,
        "minimum_score": minimum_score, "apply_carbon_cap": apply_carbon_cap,
        "small_cap": small_cap, "standard_cap": standard_cap,
        "product_summary": product_summary, "class_summary": class_summary,
        "supplier_summary": supplier_summary, "supply_route_summary": supply_route_summary,
        "plant_summary": plant_summary, "finished_route_summary": finished_route_summary,
        "cost_breakdown": cost_breakdown, "emission_breakdown": emission_breakdown,
        "total_cost_eur": total_cost_eur, "total_emissions_kgco2": total_emissions_kgco2,
        "cap_validation_warning": cap_validation_warning,
        "selected_products": list(selected_products),
        "selected_supplier_ids": list(selected_supplier_ids),
        "selected_plant_ids": list(selected_plant_ids),
        "time_limit_sec": int(time_limit_sec),
        "tables_snapshot": {"suppliers": suppliers, "plants": plants, "market": market.to_dict()},
    }


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def render_supply_chain_map(result: Dict):
    # Folium은 지도 요청 시에만 import하여 기본 페이지의 메모리 점유를 줄입니다.
    import folium
    from folium.plugins import Fullscreen
    from streamlit_folium import st_folium

    suppliers = result["tables_snapshot"]["suppliers"].set_index("supplier_id")
    plants = result["tables_snapshot"]["plants"].set_index("plant_id")
    market = result["tables_snapshot"]["market"]
    raw = result["supply_route_summary"]
    final = result["finished_route_summary"]
    all_lats = list(suppliers["latitude"]) + list(plants["latitude"]) + [float(market["latitude"])]
    all_lons = list(suppliers["longitude"]) + list(plants["longitude"]) + [float(market["longitude"])]
    m = folium.Map(location=[np.mean(all_lats), np.mean(all_lons)], zoom_start=2, tiles="CartoDB positron")
    Fullscreen().add_to(m)
    used_supplier_ids = set(raw["supplier_id"]) if not raw.empty else set()
    used_plant_ids = set(raw["plant_id"]) | set(final["plant_id"]) if (not raw.empty or not final.empty) else set()
    for sid in used_supplier_ids:
        r = suppliers.loc[sid]
        folium.CircleMarker([r["latitude"], r["longitude"]], radius=6, color="#333333", fill=True,
            tooltip=f"공급지: {r['location_name']} ({MATERIAL_LABEL.get(r['material_id'], r['material_id'])})").add_to(m)
    for pid in used_plant_ids:
        r = plants.loc[pid]
        folium.CircleMarker([r["latitude"], r["longitude"]], radius=7, color="#111111", fill=True,
            fill_color="#fdae61", tooltip=f"조립지: {r['location_name']}").add_to(m)
    folium.Marker([market["latitude"], market["longitude"]], icon=folium.Icon(color="red", icon="star"),
        tooltip=f"수요지: {market['market_name']}").add_to(m)
    max_raw = float(raw["flow_kg"].max()) if not raw.empty else 1.0
    for _, r in raw.iterrows():
        s = suppliers.loc[r["supplier_id"]]; p = plants.loc[r["plant_id"]]
        weight = 1.0 + 7.0 * math.sqrt(float(r["flow_kg"]) / max_raw)
        folium.PolyLine([[s["latitude"], s["longitude"]], [p["latitude"], p["longitude"]]],
            color=MATERIAL_COLOR[r["material_id"]], weight=weight, opacity=0.65,
            dash_array=TRANSPORT_DASH.get(r["transport_mode"]),
            tooltip=(f"{r['material_name']} | {r['supplier_location']} → {r['plant_location']} | "
                     f"{r['transport_mode_ko']} | {r['flow_kg']:,.0f} kg")).add_to(m)
    max_final = float(final["transport_mass_kg"].max()) if not final.empty else 1.0
    for _, r in final.iterrows():
        p = plants.loc[r["plant_id"]]
        weight = 2.0 + 7.0 * math.sqrt(float(r["transport_mass_kg"]) / max_final)
        folium.PolyLine([[p["latitude"], p["longitude"]], [market["latitude"], market["longitude"]]],
            color=MATERIAL_COLOR["finished"], weight=weight, opacity=0.7,
            dash_array=TRANSPORT_DASH.get(r["transport_mode"]),
            tooltip=(f"완제품 | {r['plant_location']} → {r['market_name']} | "
                     f"{r['transport_mode_ko']} | {r['vehicle_units']:,.0f}대")).add_to(m)
    st_folium(m, use_container_width=True, height=650)
    del m
    gc.collect()


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


def run_score_sensitivity(
    tables: Dict[str, pd.DataFrame],
    base_result: Dict,
    scores: List[float],
    per_point_time_limit: int,
    sensitivity_gap: float,
    progress_callback: ProgressCallback = None,
) -> pd.DataFrame:
    """점수별로 독립 MILP를 풀어 비용·배출량의 정책 민감도 점을 만듭니다."""
    rows = []
    total = max(1, len(scores))
    for index, score in enumerate(scores, start=1):
        emit_progress(
            progress_callback,
            int(5 + 90 * (index - 1) / total),
            "보조금 점수 민감도",
            f"{score:g}점 계산 {index}/{total}",
        )
        result = solve_model(
            tables=tables,
            selected_products=base_result["selected_products"],
            selected_supplier_ids=base_result["selected_supplier_ids"],
            selected_plant_ids=base_result["selected_plant_ids"],
            production_mode=base_result["production_mode"],
            scenario_id=base_result["scenario_id"],
            cap_application=base_result["cap_application"],
            loss_rate=base_result["loss_rate"],
            time_limit_sec=int(per_point_time_limit),
            mip_gap=float(sensitivity_gap),
            score_override=float(score),
            return_detail=False,
        )
        rows.append({
            "minimum_score": float(score),
            "small_cap_kgco2_per_vehicle": carbon_cap_from_score("small", score),
            "standard_cap_kgco2_per_vehicle": carbon_cap_from_score("standard", score),
            "status": result.get("status"),
            "total_supply_chain_cost_eur": result.get("total_cost_eur", np.nan),
            "total_emissions_kgco2": result.get("total_emissions_kgco2", np.nan),
            "wall_time_sec": result.get("wall_time_sec", np.nan),
        })
        gc.collect()
    emit_progress(progress_callback, 100, "민감도 완료", f"{len(scores)}개 점수의 MILP 계산을 마쳤습니다.")
    return pd.DataFrame(rows)


def result_zip(result: Dict) -> bytes:
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, obj in [
            ("product_summary.csv", result["product_summary"]),
            ("class_summary.csv", result["class_summary"]),
            ("supplier_summary.csv", result["supplier_summary"]),
            ("supply_route_summary.csv", result["supply_route_summary"]),
            ("plant_summary.csv", result["plant_summary"]),
            ("finished_route_summary.csv", result["finished_route_summary"]),
        ]:
            zf.writestr(name, obj.to_csv(index=False).encode("utf-8-sig"))
    return memory.getvalue()


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main():
    st.title("탄소배출 기반 제품 보조금 제도하의 전기차 공급망 최적화")
    st.caption("Word/PDF 수학모형을 Google OR-Tools MPSolver(SCIP 우선, CBC 대체)로 구현한 Streamlit SaaS")
    st.caption(f"build: {APP_BUILD}")
    st.caption("Community Cloud 절약 설정: 단일 Solver 스레드 · 제한된 캐시 · 요약 결과만 보관 · 지도/민감도는 요청 시에만 계산")

    with st.sidebar:
        st.header("CSV 입력: 방식 C")
        st.info("내장 데이터를 계산에 사용하지 않습니다. 아래 9개 CSV를 매 실행 세션마다 모두 업로드해야 합니다.")
        uploads = st.file_uploader(
            "관련 CSV 9개를 한 번에 업로드", type=["csv"], accept_multiple_files=True,
            help="파일명은 템플릿과 정확히 같아야 합니다.",
        )
        st.download_button(
            "입력 CSV 템플릿 ZIP 다운로드", data=make_template_zip(),
            file_name="carbon_supply_chain_input_csv.zip", mime="application/zip", use_container_width=True,
        )
        st.markdown("**업로드 필수 파일명 9개**")
        st.code("\n".join(ALL_UPLOAD_FILES), language=None)
        if st.button("세션 결과·메모리 정리", use_container_width=True):
            for key in ["optimization_result", "feasibility_result", "score_sensitivity_result"]:
                st.session_state.pop(key, None)
            gc.collect()
            st.success("현재 세션의 계산 결과를 정리했습니다.")

    tables = load_uploaded_data(uploads)
    uploaded_names = set(tables)
    required_names = set(ALL_UPLOAD_FILES)
    missing_names = sorted(required_names - uploaded_names)
    unexpected_names = sorted(uploaded_names - required_names)
    errors = validate_data(tables) if not missing_names else []
    data_ready = not missing_names and not errors
    if unexpected_names:
        st.warning("모형에서 사용하지 않는 CSV 파일: " + ", ".join(unexpected_names))
    if missing_names:
        st.warning(f"CSV 업로드 대기 중: {len(uploaded_names & required_names)}/9개 완료. 누락 파일: " + ", ".join(missing_names))
    elif errors:
        for error in errors:
            st.error(error)
    else:
        st.success("방식 C 입력 완료: 업로드한 9개 CSV만 사용합니다.")

    page = st.radio(
        "페이지",
        ["1. 입력 CSV", "2. 최적화 실행", "3. 최적화 Output", "4. 포스터 그림", "5. 수학모형 구현"],
        horizontal=True, label_visibility="collapsed", key="page_navigation",
    )

    if page == "1. 입력 CSV":
        st.header("입력 CSV 확인")
        if not data_ready:
            st.info("왼쪽 사이드바에서 9개 CSV를 모두 업로드하고 오류를 수정하면 표가 활성화됩니다.")
            return
        st.warning("생산·배출계수·비용·제품·수요·정책값은 Word/PPT에서 추출했습니다. Word에 수치가 없는 위경도와 공급·조립 용량은 실행용 구현값이며 CSV에서 수정할 수 있습니다.")
        display_names = {
            "products.csv": "제품 사양", "demand.csv": "프랑스 제품 수요",
            "raw_material_suppliers.csv": "원자재·배터리 공급지", "assembly_locations.csv": "가공·조립 후보지",
            "transport_parameters.csv": "운송수단 비용·배출계수", "markets.csv": "수요지", "scenarios.csv": "정책 시나리오",
        }
        selected_table = st.selectbox("확인할 CSV", REQUIRED_FILES, format_func=lambda name: display_names[name])
        st.dataframe(tables[selected_table], use_container_width=True, hide_index=True)
        st.download_button(f"{selected_table} 다운로드", data=tables[selected_table].to_csv(index=False).encode("utf-8-sig"),
            file_name=selected_table, mime="text/csv")
        return

    if page == "2. 최적화 실행":
        st.header("최적화 설정")
        if not data_ready:
            st.markdown("#### 실행 작업")
            c1, c2 = st.columns(2)
            c1.button("실행 전 feasibility 진단", disabled=True, use_container_width=True)
            c2.button("수학적 최적화 실행", disabled=True, use_container_width=True)
            st.caption("희미한 버튼은 CSV 입력 준비가 끝나지 않았다는 뜻입니다.")
            return

        products = tables["products.csv"]; suppliers = tables["raw_material_suppliers.csv"]
        plants = tables["assembly_locations.csv"]; scenarios = tables["scenarios.csv"]
        product_options = dict(zip(products["product_name_ko"], products["product_id"]))
        scenario_map = dict(zip(scenarios["scenario_name"], scenarios["scenario_id"]))

        with st.form("optimization_form", clear_on_submit=False):
            st.markdown("#### 실행 작업")
            b1, b2 = st.columns(2)
            diagnostic_requested = b1.form_submit_button("실행 전 feasibility 진단", use_container_width=True)
            optimization_requested = b2.form_submit_button("수학적 최적화 실행", type="primary", use_container_width=True)
            st.caption("버튼은 먼저 표시되며, 누를 때 아래 선택값이 한 번에 제출됩니다.")

            selected_product_names = st.multiselect("완제품 선택", list(product_options), default=list(product_options))
            selected_products = [product_options[name] for name in selected_product_names]
            c1, c2, c3, c4, c5 = st.columns([1.0, 1.35, 1.75, 0.85, 0.85])
            with c1:
                production_mode = st.radio("배터리 생산방식", ["line", "modular"], format_func=lambda value: MODE_LABEL[value])
            with c2:
                scenario_name = st.selectbox("정책 시나리오", list(scenario_map)); scenario_id = scenario_map[scenario_name]
            with c3:
                cap_application = st.selectbox("탄소상한 적용 단위", ["class_average", "product_strict"],
                    format_func=lambda value: CAP_APPLICATION_LABEL[value])
            with c4:
                time_limit = st.number_input("제한시간(초)", min_value=30, max_value=600, value=120, step=30,
                    help="Community Cloud 기본 권장값은 120초입니다.")
            with c5:
                mip_gap_pct = st.number_input("허용 MIP gap(%)", min_value=0.0, max_value=20.0, value=1.0, step=0.1)

            st.subheader("후보 공급지·조립지")
            st.caption("모든 Word/PDF 후보가 기본 선택됩니다. 설정 변경은 폼 제출 전까지 앱 전체 재실행을 유발하지 않습니다.")
            supplier_ids: List[str] = []
            cols = st.columns(4)
            for col, material in zip(cols, ["steel", "aluminum", "other", "battery"]):
                with col:
                    sub = suppliers[suppliers["material_id"] == material]
                    options = dict(zip(sub["location_name"], sub["supplier_id"]))
                    chosen = st.multiselect(MATERIAL_LABEL[material], list(options), default=list(options), key=f"supplier_{material}")
                    supplier_ids.extend(options[name] for name in chosen)
            plant_options = dict(zip(plants["location_name"], plants["plant_id"]))
            chosen_plants = st.multiselect("가공·조립 위치", list(plant_options), default=list(plant_options))
            plant_ids = [plant_options[name] for name in chosen_plants]

        selection_ready = bool(selected_products and plant_ids and all(
            any(sid in supplier_ids for sid in suppliers.loc[suppliers["material_id"] == material, "supplier_id"])
            for material in ["steel", "aluminum", "other", "battery"]
        ))
        if not selection_ready and (diagnostic_requested or optimization_requested):
            st.error("제품, 네 재질의 공급지, 가공·조립 위치를 각각 하나 이상 선택하세요.")
            return

        if diagnostic_requested:
            st.subheader("실행 전 feasibility 진단")
            st.info(
                "이 진단은 ① 총 공급·조립 용량이 충분한지, ② 각 제품이 가장 유리한 공급망을 독립적으로 사용할 때에도 탄소상한을 넘는지 확인합니다. "
                "두 검사를 통과해도 공유용량과 정수조건 때문에 최종 feasible이 보장되지는 않으며, 최종 판정은 MILP가 합니다."
            )
            prog = st.progress(0, text="진단 준비 중"); stage = st.empty()
            def diag_progress(percent: int, label: str, detail: str = ""):
                text = label if not detail else f"{label} — {detail}"; prog.progress(percent, text=text); stage.caption(text)
            capacity = selection_capacity_diagnostics(tables, selected_products, supplier_ids, plant_ids)
            product_lb, class_lb = optimistic_carbon_lower_bounds(
                tables, selected_products, supplier_ids, plant_ids, scenario_id, MATERIAL_LOSS_RATE, diag_progress
            )
            st.session_state["feasibility_result"] = {"capacity": capacity, "product_lb": product_lb, "class_lb": class_lb,
                "scenario_id": scenario_id, "cap_application": cap_application}
            prog.progress(100, text="feasibility 진단 완료"); stage.success("아래 결과를 확인하세요.")

        feasibility_result = st.session_state.get("feasibility_result")
        if feasibility_result:
            with st.expander("최근 feasibility 진단 결과와 읽는 방법", expanded=True):
                capacity = feasibility_result["capacity"]
                capacity_ok = bool((capacity["판정"] == "충족").all())
                if capacity_ok:
                    st.success("1단계 용량검사 통과: 선택한 공급지와 조립지의 합계 용량은 총수요보다 큽니다.")
                else:
                    st.error("1단계 용량검사 실패: '부족' 항목은 해당 후보를 추가하거나 용량값을 높여야 합니다.")
                st.dataframe(capacity, use_container_width=True, hide_index=True)
                st.markdown("**2단계 낙관적 탄소 하한**")
                st.caption("각 제품이 다른 제품과 용량을 공유하지 않고 가장 낮은 배출 경로를 독점한다고 가정한 매우 유리한 값입니다.")
                st.dataframe(feasibility_result["product_lb"], use_container_width=True, hide_index=True)
                st.dataframe(feasibility_result["class_lb"], use_container_width=True, hide_index=True)
                if feasibility_result["cap_application"] == "product_strict":
                    possible = bool(feasibility_result["product_lb"]["strict_product_possible"].all())
                else:
                    possible = bool(feasibility_result["class_lb"]["class_average_possible"].all())
                if not possible:
                    st.error("낙관적 하한도 상한을 넘는 항목이 있으므로 현재 탄소상한 적용방식에서는 확실히 INFEASIBLE입니다.")
                elif capacity_ok:
                    st.warning("기본검사는 통과했습니다. 이것은 feasible 보장이 아니며, 공유용량·정수 모듈·운송수단 선택을 포함한 MILP 실행이 필요합니다.")

        if optimization_requested:
            st.subheader("최적화 작업 단계 진행률")
            progress = st.progress(0, text="최적화 준비 중"); stage = st.empty()
            def ui_progress(percent: int, label: str, detail: str = ""):
                text = label if not detail else f"{label} — {detail}"; progress.progress(percent, text=text)
                (stage.info if percent < 100 else stage.success)(text)
            result = solve_model(
                tables, selected_products, supplier_ids, plant_ids, production_mode, scenario_id,
                cap_application, MATERIAL_LOSS_RATE, int(time_limit), float(mip_gap_pct) / 100.0, ui_progress,
            )
            st.session_state["optimization_result"] = result
            st.session_state.pop("score_sensitivity_result", None)
            if result["status"] in {"OPTIMAL", "FEASIBLE"}:
                progress.progress(100, text="최적화 완료")
                st.success(f"{result['status']} 해를 찾았습니다. Solver: {result['backend']}")
                st.info("'3. 최적화 Output' 페이지에서 집계표·그래프·결과 ZIP을 확인하세요.")
            else:
                progress.progress(100, text=f"Solver 종료: {result.get('status')}")
                st.error(f"Solver 상태: {result.get('status')} — {result.get('message', '')}")
                if result.get("status") == "NOT_SOLVED":
                    st.warning("이는 infeasible 판정이 아닙니다. 제한시간 또는 gap을 조정하세요.")
        return

    if page == "3. 최적화 Output":
        st.header("최적화 Output")
        result = st.session_state.get("optimization_result")
        if not result or result.get("status") not in {"OPTIMAL", "FEASIBLE"}:
            st.info("먼저 '2. 최적화 실행' 페이지에서 최적화를 실행하세요.")
            return
        st.caption(f"{MODE_LABEL[result['production_mode']]} | {result['scenario_name']} | {CAP_APPLICATION_LABEL[result['cap_application']]} | Solver {result['backend']} · 1 thread")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 공급망 비용", f"€ {result['total_cost_eur']:,.0f}")
        c2.metric("총 탄소배출량", f"{result['total_emissions_kgco2']/1_000_000:,.2f} kt CO₂-eq")
        c3.metric("제품 평균 비용", f"€ {result['product_summary']['cost_per_vehicle_eur'].mean():,.0f}/대")
        c4.metric("평균 보조금 점수", f"{result['product_summary']['subsidy_score'].mean():.1f}/80")

        with st.expander("총비용 계산방법", expanded=True):
            st.markdown(r"""
총비용은 다음 네 항의 합입니다.

1. **원자재·배터리 생산비**: 철강·알루미늄·기타는 `공급량(kg) × 생산비(€/kg)`, 배터리는 `공급량(kg) × 제품별 kWh/kg × 생산비(€/kWh)`
2. **공급지→조립지 운송비**: `운송량(kg) × 거리(km) × 운송비(€/kg-km)`
3. **가공·조립비**: `조립대수 × 배터리 제외 차량질량(kg/대) × 조립비(€/kg)`
4. **조립지→프랑스 완제품 운송비**: `차량대수 × 차량 총질량(kg/대) × 거리(km) × 운송비(€/kg-km)`
            """)
            cost_table = pd.DataFrame({"비용항목": list(result["cost_breakdown"]), "비용_EUR": list(result["cost_breakdown"].values())})
            st.dataframe(cost_table, use_container_width=True, hide_index=True)
            st.caption(f"목적함수값과 사후 집계 차이: € {result['objective_reconciliation_eur']:,.6f}")

        if result.get("cap_validation_warning"):
            st.error(result["cap_validation_warning"])
        if result.get("apply_carbon_cap"):
            if result["cap_application"] == "class_average":
                st.info("차급 평균 모드에서는 개별 롱레인지 트림이 상한보다 높아도 같은 차급의 수요가중 평균이 상한 이하이면 모형 제약을 충족합니다.")
            else:
                st.info("트림별 엄격 모드에서는 각 제품의 차량 1대당 탄소발자국이 상한 이하이어야 합니다. 유의한 초과가 보이면 단위·집계·수치허용오차를 점검해야 합니다.")
        else:
            st.info("현재 시나리오는 탄소상한을 적용하지 않으므로 최적 탄소발자국이 정책 기준선보다 높을 수 있습니다.")

        if result["cap_application"] == "class_average":
            st.subheader("차급별 수요가중 평균 탄소상한 결과")
            st.dataframe(result["class_summary"], use_container_width=True, hide_index=True)
        st.subheader("제품별 정책 결과")
        display = result["product_summary"].copy(); display["eligible"] = display["eligible"].map({True: "충족", False: "미충족"})
        st.dataframe(display, use_container_width=True, hide_index=True)
        fig = product_carbon_figure(result["product_summary"]); st.pyplot(fig, use_container_width=True); plt.close(fig); gc.collect()

        st.subheader("비용·탄소 구성")
        col1, col2 = st.columns(2)
        with col1:
            fig = pie_figure(result["cost_breakdown"], "총비용 구성"); st.pyplot(fig, use_container_width=True); plt.close(fig)
        with col2:
            fig = pie_figure(result["emission_breakdown"], "총 탄소배출량 구성"); st.pyplot(fig, use_container_width=True); plt.close(fig)
        gc.collect()

        st.subheader("공급지·조립지·운송경로 집계")
        st.markdown("**원자재·배터리 공급지별 총물량**")
        st.dataframe(result["supplier_summary"], use_container_width=True, hide_index=True)
        st.markdown("**공급지→조립지 경로별 총물량**")
        st.dataframe(result["supply_route_summary"], use_container_width=True, hide_index=True)
        st.markdown("**조립지별 총생산량**")
        st.dataframe(result["plant_summary"], use_container_width=True, hide_index=True)
        st.markdown("**조립지→프랑스 완제품 운송경로별 총물량**")
        st.dataframe(result["finished_route_summary"], use_container_width=True, hide_index=True)
        st.caption("제품별 상세 행은 메모리를 줄이기 위해 보관하지 않고, 공급지·조립지·운송수단 기준으로 합산합니다.")

        if st.checkbox("최적 공급망 지도 생성", value=False):
            render_supply_chain_map(result)

        st.subheader("보조금 점수에 따른 비용·탄소 민감도")
        st.info("MIP 최적해는 점수에 따라 불연속적으로 바뀔 수 있으므로 진정한 연속함수는 아닙니다. 선택한 점수에서 MILP를 각각 풀고 점들을 선으로 연결해 추세를 보여줍니다.")
        with st.form("score_sensitivity_form"):
            a, b, c, d = st.columns(4)
            min_score = a.number_input("최소 점수", 0.0, 80.0, 40.0, 5.0)
            max_score = b.number_input("최대 점수", 0.0, 80.0, 80.0, 5.0)
            points = int(c.number_input("계산 점 개수", 3, 7, 5, 1))
            point_limit = int(d.number_input("점당 제한시간(초)", 15, 60, 30, 5))
            run_frontier = st.form_submit_button("점수 민감도 계산", use_container_width=True)
        if run_frontier:
            if max_score <= min_score:
                st.error("최대 점수는 최소 점수보다 커야 합니다.")
            else:
                scores = np.linspace(float(min_score), float(max_score), points).tolist()
                pbar = st.progress(0, text="점수 민감도 준비 중"); ptxt = st.empty()
                def frontier_progress(percent: int, label: str, detail: str = ""):
                    text = label if not detail else f"{label} — {detail}"; pbar.progress(percent, text=text); ptxt.caption(text)
                frontier = run_score_sensitivity(
                    tables, result, scores, point_limit,
                    max(0.02, float(result["configured_mip_gap"])), frontier_progress,
                )
                st.session_state["score_sensitivity_result"] = frontier
        frontier = st.session_state.get("score_sensitivity_result")
        if isinstance(frontier, pd.DataFrame) and not frontier.empty:
            st.dataframe(frontier, use_container_width=True, hide_index=True)
            feasible = frontier[frontier["status"].isin(["OPTIMAL", "FEASIBLE"])].copy()
            if not feasible.empty:
                feasible = feasible.sort_values("minimum_score")
                st.markdown("**총 공급망 비용 변화**")
                fig, ax = plt.subplots(figsize=(8, 4.2))
                ax.plot(feasible["minimum_score"], feasible["total_supply_chain_cost_eur"] / 1_000_000, marker="o")
                ax.set_xlabel("최소 보조금 점수")
                ax.set_ylabel("총비용 (백만 €)")
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.markdown("**총 탄소배출량 변화**")
                fig, ax = plt.subplots(figsize=(8, 4.2))
                ax.plot(feasible["minimum_score"], feasible["total_emissions_kgco2"] / 1_000_000, marker="o")
                ax.set_xlabel("최소 보조금 점수")
                ax.set_ylabel("총배출량 (kt CO₂-eq)")
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                gc.collect()

        st.download_button("전체 최적화 결과 CSV ZIP 다운로드", data=result_zip(result),
            file_name=f"optimization_result_{result['scenario_id']}_{result['production_mode']}.zip",
            mime="application/zip", use_container_width=True)
        return

    if page == "4. 포스터 그림":
        st.header("세 번째 파일의 최종 결과 그림")
        st.info("이 페이지는 포스터에 기재된 원본 수치의 벤치마크 영역이며, 현재 업로드 CSV 최적화 결과와 구분합니다.")
        poster_path = ASSET_DIR / "poster_reference.png"
        if poster_path.exists():
            st.image(str(poster_path), caption="세 번째 파일 원본 포스터", use_container_width=True)
        else:
            st.warning("assets/poster_reference.png가 없습니다.")
        if all(name in tables for name in BENCHMARK_FILES):
            ratios = tables["poster_benchmark_cost_ratios.csv"]; quartiles = tables["poster_benchmark_quartiles.csv"]
            choice = st.selectbox("재구성할 포스터 그림", ["시나리오별 비용 비율", "운송량 사분위수", "라인 대비 모듈 차이"])
            fig = benchmark_cost_ratio_figure(ratios) if choice == "시나리오별 비용 비율" else benchmark_quartile_figure(quartiles) if choice == "운송량 사분위수" else benchmark_difference_figure(quartiles)
            st.pyplot(fig, use_container_width=True); plt.close(fig); gc.collect()
        return

    if page == "5. 수학모형 구현":
        st.header("Word 수학적 최적화 모형 ↔ SaaS 매핑")
        mapping = pd.DataFrame([
            ["완제품 집합 f", "소형/중형/대형 × 미드/롱 6종", "products.csv · 완제품 선택", "product_id, product_map"],
            ["원자재 r", "철강·알루미늄·기타·배터리", "raw_material_suppliers.csv", "material_suppliers[r]"],
            ["공급지 s", "재질별 국가 후보", "후보 공급지 multiselect", "supplier_id"],
            ["조립지 p", "가공·조립 국가 후보", "가공·조립 위치 multiselect", "plant_id"],
            ["운송수단 t", "해상·항공·도로·철도", "transport_parameters.csv", "rt/ft와 alpha/beta"],
            ["수요 D_f", "프랑스 모델별 수요", "demand.csv", "demand_map"],
            ["원자재 흐름", "공급지→조립지 kg", "공급경로 집계표", "rt[f,r,s,p,t]"],
            ["완제품 생산", "조립지별 차량 대수", "조립지별 생산량", "fp[f,p]"],
            ["완제품 운송", "조립지→프랑스 차량 대수", "완제품 운송경로 집계", "ft[f,p,t]"],
            ["라인 배터리", "완성 팩 공급횟수=조립대수", "라인 생산 선택", "line_count"],
            ["모듈 배터리", "10kWh 메인·5kWh 보조 모듈 개수", "모듈 활용 분산 생산", "main_count, sub_count"],
            ["목적함수", "생산+공급운송+조립+완제품운송 비용 최소", "총비용/비용 구성", "model.add_obj"],
            ["탄소제약", "보조금 점수에 대응하는 탄소상한", "시나리오·상한 적용단위", "product_emission_expr"],
        ], columns=["Word 요소", "Word 의미", "SaaS 입력·출력", "코드 객체"])
        st.dataframe(mapping, use_container_width=True, hide_index=True)
        st.markdown(r"""
### 목적함수 대응
\[
\min Z=C_{prod}+C_{raw\ transport}+C_{assembly}+C_{finished\ transport}
\]

### 제약조건 대응
1. 원자재 생산량–운송량 및 제품 생산량–완제품 운송량 흐름 보존
2. 재질별 공급지 최대용량과 조립지 처리용량
3. 제품별 철강·알루미늄·기타 원자재 질량 수급
4. 라인 생산의 완성 배터리 팩 공급횟수
5. 모듈 생산의 10 kWh·5 kWh 모듈 개수
6. 프랑스의 제품별 수요 충족
7. 한 경로에서 운송수단 최대 하나 선택, 미사용 경로는 모든 이진변수 0 허용
8. 정책 시나리오의 탄소상한

### Word 밖에서 실행을 위해 명시한 구현 가정
- 국가 중심 좌표의 WGS-84 측지거리
- 공급지→조립지에도 유럽/비유럽 운송수단 규칙을 적용
- 포스터 재현용 차급 평균 상한과 PDF 엄격 트림별 상한을 선택 가능하게 분리
- 철강·알루미늄 손실률 0.3 반영
- 비용·배출량이 모두 열등한 운송수단 사전 제거
- Community Cloud 계산설정인 1 thread, 시간제한, MIP gap은 수학적 의사결정모형이 아니라 풀이 설정
        """)
        st.info("현재 SaaS는 Word의 변수·목적함수·핵심 제약을 직접 대응시키되, Word에 명시되지 않은 거리·용량·상한 집계방식은 별도 구현 가정으로 표시합니다.")
        return


if __name__ == "__main__":
    main()
