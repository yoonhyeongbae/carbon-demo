from io import StringIO
import math
import hashlib
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from ortools.linear_solver import pywraplp

st.set_page_config(page_title="공급망 기반 탄소배출량 최적화 데모", layout="wide")

# ============================================================
# 1. 기본 상수 / 가정 데이터
# ============================================================
REQUIRED_COLS = [
    "scope",
    "site",
    "activity_group",
    "option_name",
    "baseline_amount",
    "min_amount",
    "max_amount",
    "unit",
    "emission_factor_tco2_per_unit",
    "cost_per_unit",
    "description",
]

OPTIONAL_COLS = [
    "usage_location",
    "procurement_region",
    "route_id",
    "origin",
    "destination",
    "quantity_ton",
    "distance_km",
    "transport_mode",
]

SCOPE_LIST = ["Scope1", "Scope2", "Scope3"]

# 노드(지도용) - 임의 가정
NODES_DF = pd.DataFrame([
    {"node": "Plant_A", "node_type": "Plant", "lat": 37.2636, "lon": 127.0286},       # 수원 근처
    {"node": "Plant_B", "node_type": "Plant", "lat": 35.1796, "lon": 129.0756},       # 부산
    {"node": "Warehouse_Seoul", "node_type": "Warehouse", "lat": 37.5665, "lon": 126.9780},
    {"node": "Warehouse_Daejeon", "node_type": "Warehouse", "lat": 36.3504, "lon": 127.3845},
    {"node": "Supplier_Incheon", "node_type": "Supplier", "lat": 37.4563, "lon": 126.7052},
    {"node": "Supplier_Pohang", "node_type": "Supplier", "lat": 36.0190, "lon": 129.3435},
    {"node": "Port_Busan", "node_type": "Port", "lat": 35.1053, "lon": 129.0403},
    {"node": "Customer_Daegu", "node_type": "Customer", "lat": 35.8714, "lon": 128.6014},
    {"node": "Customer_Gwangju", "node_type": "Customer", "lat": 35.1595, "lon": 126.8526},
])

NODE_TYPE_COLOR = {
    "Plant": "red",
    "Warehouse": "blue",
    "Supplier": "green",
    "Port": "purple",
    "Customer": "orange",
}

PLANT_NODES = ["Plant_A", "Plant_B"]
WAREHOUSE_NODES = ["Warehouse_Seoul", "Warehouse_Daejeon"]
SUPPLIER_NODES = ["Supplier_Incheon", "Supplier_Pohang"]
CUSTOMER_NODES = ["Customer_Daegu", "Customer_Gwangju"]
ALL_NODE_NAMES = NODES_DF["node"].tolist()

SITE_DEFAULTS = {
    "Plant_A": {"boiler_demand": 120.0, "fleet_demand": 40.0, "power_demand": 180.0},
    "Plant_B": {"boiler_demand": 90.0, "fleet_demand": 35.0, "power_demand": 160.0},
}

# Scope 1 가정 데이터
SCOPE1_BOILER_OPTIONS = {
    "Natural_Gas": {"ef": 0.22, "cost": 55, "unit": "MWh", "desc": "보일러 열원"},
    "LPG": {"ef": 0.27, "cost": 68, "unit": "MWh", "desc": "보일러 열원"},
    "Biomethane": {"ef": 0.06, "cost": 92, "unit": "MWh", "desc": "보일러 열원"},
}

SCOPE1_FLEET_OPTIONS = {
    "Diesel_Fleet": {"ef": 0.27, "cost": 60, "unit": "MWh", "desc": "사내차량 연료"},
    "LNG_Fleet": {"ef": 0.18, "cost": 72, "unit": "MWh", "desc": "사내차량 연료"},
    "EV_Fleet": {"ef": 0.05, "cost": 95, "unit": "MWh", "desc": "사내차량 전동화"},
}

# Scope 2 가정 데이터
GRID_REGION_META = {
    "Region_A": {"ef": 0.48, "cost": 98},
    "Region_B": {"ef": 0.38, "cost": 106},
}
SOLAR_META = {"ef": 0.02, "cost": 135}

# Scope 3 가정 데이터
TRANSPORT_MODE_META = {
    "Truck": {"ef": 0.00018, "cost": 1.20, "unit": "ton-km"},
    "Rail": {"ef": 0.00006, "cost": 1.55, "unit": "ton-km"},
    "Ship": {"ef": 0.00004, "cost": 1.50, "unit": "ton-km"},
}

TRANSPORT_MODE_COLOR = {
    "Truck": "blue",
    "Rail": "green",
    "Ship": "purple",
}

# 컬럼 설명
GUIDE_DF = pd.DataFrame(
    [
        ["scope", "문자", "Scope1 / Scope2 / Scope3", "배출 범주"],
        ["site", "문자", "Plant_A", "사업장명"],
        ["activity_group", "문자", "Boiler_Heat / Fleet_Fuel / Electricity_Procurement_... / Flow_...", "같은 기능을 수행하는 대체 가능한 선택지 묶음"],
        ["option_name", "문자", "Natural_Gas / Grid_Region_A / Truck", "개별 선택지 이름"],
        ["baseline_amount", "숫자", "100", "현재 기준안 사용량 또는 물량"],
        ["min_amount", "숫자", "0", "최적화 후 최소 사용량"],
        ["max_amount", "숫자", "140", "최적화 후 최대 사용량"],
        ["unit", "문자", "MWh / ton-km", "사용 단위"],
        ["emission_factor_tco2_per_unit", "숫자", "0.22", "단위당 배출계수(tCO2/unit)"],
        ["cost_per_unit", "숫자", "55", "단위당 비용"],
        ["description", "문자", "사업장 보일러 열원", "행 설명용 텍스트"],
    ],
    columns=["컬럼명", "자료형", "예시", "설명"]
)

ACTIVITY_GROUP_GUIDE_DF = pd.DataFrame(
    [
        ["Boiler_Heat", "보일러 열원 활동", "Natural_Gas / LPG / Biomethane"],
        ["Fleet_Fuel", "사내차량 연료 활동", "Diesel_Fleet / LNG_Fleet / EV_Fleet"],
        ["Electricity_Procurement_<Location>", "전력 조달 활동", "Grid_Region_A / Solar_Onsite"],
        ["Flow_RouteX_Origin_to_Destination", "공급망 이동 활동", "Truck / Rail / Ship"],
    ],
    columns=["activity_group", "설명", "예시 option_name"]
)

# 기본 내장 샘플 CSV (업로드 안 해도 결과 확인용)
DEFAULT_SAMPLE_TEXT = """scope,site,activity_group,option_name,baseline_amount,min_amount,max_amount,unit,emission_factor_tco2_per_unit,cost_per_unit,description,usage_location,procurement_region,route_id,origin,destination,quantity_ton,distance_km,transport_mode
Scope1,Plant_A,Boiler_Heat,Natural_Gas,120,0,120,MWh,0.22,55,사업장 보일러 열원,,,,,,,
Scope1,Plant_A,Fleet_Fuel,Diesel_Fleet,40,0,40,MWh,0.27,60,사내차량 연료,,,,,,,
Scope2,Plant_A,Electricity_Procurement_Plant_A,Grid_Region_A,180,0,180,MWh,0.48,98,사업장 전력 사용,Plant_A,Region_A,,,,,
Scope3,Plant_A,Flow_R1_Supplier_Incheon_to_Plant_A,Truck,23085,0,23085,ton-km,0.00018,1.20,기본 공급망 이동,,,R1,Supplier_Incheon,Plant_A,120,192.38,Truck
"""

# ============================================================
# 2. 유틸 함수
# ============================================================
def get_node_row(node_name: str):
    row = NODES_DF[NODES_DF["node"] == node_name]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return r * c


def find_nearest_node(lat, lon):
    best_node = None
    best_dist = None
    for _, row in NODES_DF.iterrows():
        d = haversine_km(lat, lon, row["lat"], row["lon"])
        if best_dist is None or d < best_dist:
            best_dist = d
            best_node = row["node"]
    return best_node


def ensure_optional_columns(df: pd.DataFrame):
    df = df.copy()
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = ""
    return df


def validate_input_df(df: pd.DataFrame):
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        return False, f"필수 컬럼이 없습니다: {', '.join(missing_cols)}", None

    df = df.copy()

    numeric_cols = [
        "baseline_amount",
        "min_amount",
        "max_amount",
        "emission_factor_tco2_per_unit",
        "cost_per_unit",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_cols].isnull().any().any():
        return False, "숫자 컬럼에 비어 있거나 숫자가 아닌 값이 있습니다.", None

    if (df["min_amount"] > df["max_amount"]).any():
        return False, "min_amount가 max_amount보다 큰 행이 있습니다.", None

    if (df[numeric_cols] < 0).any().any():
        return False, "숫자 컬럼에 음수가 있습니다.", None

    group_cols = ["site", "scope", "activity_group", "unit"]
    grp = df.groupby(group_cols, as_index=False).agg(
        demand=("baseline_amount", "sum"),
        min_total=("min_amount", "sum"),
        max_total=("max_amount", "sum"),
    )

    bad_rows = grp[(grp["demand"] < grp["min_total"]) | (grp["demand"] > grp["max_total"])]
    if not bad_rows.empty:
        return False, "일부 activity_group에서 기준 수요가 min/max 범위를 만족하지 않습니다.", None

    return True, "입력 데이터 검증 완료", df


def make_choice_rows(scope, site, activity_group, demand, candidate_options, baseline_option, option_catalog, description, extra_cols=None):
    rows = []
    extra_cols = extra_cols or {}

    candidate_options = [opt for opt in candidate_options if opt in option_catalog]
    if not candidate_options:
        return rows

    if baseline_option not in candidate_options:
        baseline_option = candidate_options[0]

    for opt in candidate_options:
        meta = option_catalog[opt]
        row = {
            "scope": scope,
            "site": site,
            "activity_group": activity_group,
            "option_name": opt,
            "baseline_amount": float(demand) if opt == baseline_option else 0.0,
            "min_amount": 0.0,
            "max_amount": float(demand),
            "unit": meta["unit"],
            "emission_factor_tco2_per_unit": meta["ef"],
            "cost_per_unit": meta["cost"],
            "description": description,
        }
        row.update(extra_cols)
        rows.append(row)
    return rows


def build_scope2_rows(site, usage_location, procurement_region, demand, candidate_methods, baseline_method):
    candidate_methods = [m for m in candidate_methods if m in ["Grid", "Solar_Onsite"]]
    if not candidate_methods:
        return []

    if baseline_method not in candidate_methods:
        baseline_method = candidate_methods[0]

    rows = []

    option_map = {}
    if "Grid" in candidate_methods:
        option_map["Grid"] = {
            "name": f"Grid_{procurement_region}",
            "ef": GRID_REGION_META[procurement_region]["ef"],
            "cost": GRID_REGION_META[procurement_region]["cost"],
            "unit": "MWh",
            "desc": f"{usage_location} 전력 사용 / {procurement_region} 지역 전력망",
        }

    if "Solar_Onsite" in candidate_methods:
        option_map["Solar_Onsite"] = {
            "name": f"Solar_Onsite_{site}",
            "ef": SOLAR_META["ef"],
            "cost": SOLAR_META["cost"],
            "unit": "MWh",
            "desc": f"{usage_location} 전력 사용 / 자가 태양광",
        }

    activity_group = f"Electricity_Procurement_{usage_location}"

    for method in candidate_methods:
        meta = option_map[method]
        rows.append({
            "scope": "Scope2",
            "site": site,
            "activity_group": activity_group,
            "option_name": meta["name"],
            "baseline_amount": float(demand) if method == baseline_method else 0.0,
            "min_amount": 0.0,
            "max_amount": float(demand),
            "unit": meta["unit"],
            "emission_factor_tco2_per_unit": meta["ef"],
            "cost_per_unit": meta["cost"],
            "description": meta["desc"],
            "usage_location": usage_location,
            "procurement_region": procurement_region,
            "route_id": "",
            "origin": "",
            "destination": "",
            "quantity_ton": "",
            "distance_km": "",
            "transport_mode": "",
        })
    return rows


def build_scope3_rows(site, route_id, origin, destination, quantity_ton, candidate_modes, baseline_mode):
    origin_row = get_node_row(origin)
    dest_row = get_node_row(destination)
    if origin_row is None or dest_row is None:
        return []

    if origin == destination:
        return []

    distance_km = haversine_km(origin_row["lat"], origin_row["lon"], dest_row["lat"], dest_row["lon"])
    ton_km = float(quantity_ton) * float(distance_km)

    candidate_modes = [m for m in candidate_modes if m in TRANSPORT_MODE_META]
    if not candidate_modes:
        return []

    if baseline_mode not in candidate_modes:
        baseline_mode = candidate_modes[0]

    rows = []
    activity_group = f"Flow_{route_id}_{origin}_to_{destination}"

    for mode in candidate_modes:
        meta = TRANSPORT_MODE_META[mode]
        rows.append({
            "scope": "Scope3",
            "site": site,
            "activity_group": activity_group,
            "option_name": mode,
            "baseline_amount": ton_km if mode == baseline_mode else 0.0,
            "min_amount": 0.0,
            "max_amount": ton_km,
            "unit": meta["unit"],
            "emission_factor_tco2_per_unit": meta["ef"],
            "cost_per_unit": meta["cost"],
            "description": f"{origin} → {destination} 공급망 이동",
            "usage_location": "",
            "procurement_region": "",
            "route_id": route_id,
            "origin": origin,
            "destination": destination,
            "quantity_ton": float(quantity_ton),
            "distance_km": float(distance_km),
            "transport_mode": mode,
        })
    return rows


def optimize_total_emissions(df: pd.DataFrame, budget_increase_pct: float = 5.0):
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        return {"status": "SOLVER_NOT_CREATED"}

    x = {}
    for idx, row in df.iterrows():
        x[idx] = solver.NumVar(
            float(row["min_amount"]),
            float(row["max_amount"]),
            f"x_{idx}_{row['scope']}_{row['activity_group']}_{row['option_name']}"
        )

    group_cols = ["site", "scope", "activity_group", "unit"]
    grouped = df.groupby(group_cols)["baseline_amount"].sum().reset_index()

    for _, g in grouped.iterrows():
        mask = (
            (df["site"] == g["site"])
            & (df["scope"] == g["scope"])
            & (df["activity_group"] == g["activity_group"])
            & (df["unit"] == g["unit"])
        )
        idxs = df.index[mask]
        solver.Add(sum(x[i] for i in idxs) == float(g["baseline_amount"]))

    baseline_total_cost = float((df["baseline_amount"] * df["cost_per_unit"]).sum())
    max_allowed_cost = baseline_total_cost * (1 + budget_increase_pct / 100.0)

    solver.Add(
        sum(float(df.loc[i, "cost_per_unit"]) * x[i] for i in df.index)
        <= max_allowed_cost
    )

    objective = solver.Objective()
    for i in df.index:
        objective.SetCoefficient(x[i], float(df.loc[i, "emission_factor_tco2_per_unit"]))
    objective.SetMinimization()

    status = solver.Solve()

    status_map = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    status_label = status_map.get(status, f"UNKNOWN_{status}")

    if status_label not in ["OPTIMAL", "FEASIBLE"]:
        return {"status": status_label}

    result_df = df.copy()
    result_df["optimized_amount"] = [x[i].solution_value() for i in df.index]
    result_df["delta_amount"] = result_df["optimized_amount"] - result_df["baseline_amount"]

    result_df["baseline_emissions_tco2"] = (
        result_df["baseline_amount"] * result_df["emission_factor_tco2_per_unit"]
    )
    result_df["optimized_emissions_tco2"] = (
        result_df["optimized_amount"] * result_df["emission_factor_tco2_per_unit"]
    )

    result_df["baseline_cost"] = result_df["baseline_amount"] * result_df["cost_per_unit"]
    result_df["optimized_cost"] = result_df["optimized_amount"] * result_df["cost_per_unit"]

    scope_summary = result_df.groupby("scope", as_index=False).agg(
        baseline_amount=("baseline_amount", "sum"),
        optimized_amount=("optimized_amount", "sum"),
        baseline_emissions_tco2=("baseline_emissions_tco2", "sum"),
        optimized_emissions_tco2=("optimized_emissions_tco2", "sum"),
        baseline_cost=("baseline_cost", "sum"),
        optimized_cost=("optimized_cost", "sum"),
    )
    scope_summary["reduction_tco2"] = (
        scope_summary["baseline_emissions_tco2"] - scope_summary["optimized_emissions_tco2"]
    )
    scope_summary["reduction_pct"] = (
        100 * scope_summary["reduction_tco2"] / scope_summary["baseline_emissions_tco2"]
    )

    activity_summary = result_df.groupby(
        ["scope", "activity_group", "unit"], as_index=False
    ).agg(
        baseline_amount=("baseline_amount", "sum"),
        optimized_amount=("optimized_amount", "sum"),
        baseline_emissions_tco2=("baseline_emissions_tco2", "sum"),
        optimized_emissions_tco2=("optimized_emissions_tco2", "sum"),
        baseline_cost=("baseline_cost", "sum"),
        optimized_cost=("optimized_cost", "sum"),
    )
    activity_summary["reduction_tco2"] = (
        activity_summary["baseline_emissions_tco2"] - activity_summary["optimized_emissions_tco2"]
    )

    total = {
        "baseline_emissions_tco2": float(result_df["baseline_emissions_tco2"].sum()),
        "optimized_emissions_tco2": float(result_df["optimized_emissions_tco2"].sum()),
        "baseline_cost": float(result_df["baseline_cost"].sum()),
        "optimized_cost": float(result_df["optimized_cost"].sum()),
    }
    total["reduction_tco2"] = (
        total["baseline_emissions_tco2"] - total["optimized_emissions_tco2"]
    )
    total["reduction_pct"] = (
        100 * total["reduction_tco2"] / total["baseline_emissions_tco2"]
        if total["baseline_emissions_tco2"] > 0 else 0
    )

    return {
        "status": status_label,
        "detail": result_df,
        "scope_summary": scope_summary,
        "activity_summary": activity_summary,
        "total": total,
    }


def build_signature(df: pd.DataFrame) -> str:
    csv_text = df.to_csv(index=False)
    return hashlib.md5(csv_text.encode("utf-8")).hexdigest()


def summarize_routes(df: pd.DataFrame, optimized=False):
    if "route_id" not in df.columns:
        return pd.DataFrame()

    route_df = df.copy()
    route_df = route_df[route_df["route_id"].astype(str).str.strip() != ""].copy()
    if route_df.empty:
        return pd.DataFrame()

def get_dominant_mode_from_mix(mix_text: str):
    """
    optimized_mix 예: 'Truck:12000.0, Rail:8000.0'
    가장 많이 배정된 운송수단명을 반환
    """
    if mix_text is None or str(mix_text).strip() == "":
        return ""

    best_mode = ""
    best_value = -1.0

    parts = str(mix_text).split(",")
    for part in parts:
        part = part.strip()
        if ":" not in part:
            continue
        mode, value = part.split(":", 1)
        mode = mode.strip()
        try:
            value = float(value.strip())
        except ValueError:
            continue

        if value > best_value:
            best_value = value
            best_mode = mode

    return best_mode

    
    if "baseline_emissions_tco2" not in route_df.columns:
        route_df["baseline_emissions_tco2"] = (
            route_df["baseline_amount"] * route_df["emission_factor_tco2_per_unit"]
        )

    if "optimized_emissions_tco2" not in route_df.columns:
        if "optimized_amount" in route_df.columns:
            route_df["optimized_emissions_tco2"] = (
                route_df["optimized_amount"] * route_df["emission_factor_tco2_per_unit"]
            )
        else:
            route_df["optimized_emissions_tco2"] = route_df["baseline_emissions_tco2"]

    agg_col = "optimized_emissions_tco2" if optimized else "baseline_emissions_tco2"

    summary = route_df.groupby(["route_id", "site", "origin", "destination"], as_index=False).agg(
        quantity_ton=("quantity_ton", "first"),
        distance_km=("distance_km", "first"),
        emissions_tco2=(agg_col, "sum"),
    )

    def get_baseline_mode(g):
        rows = g[g["baseline_amount"] > 0]
        if rows.empty:
            return ""
        return rows.iloc[0]["transport_mode"]

    def get_opt_mix(g):
        if "optimized_amount" not in g.columns:
            return ""
        out = []
        for _, r in g.iterrows():
            if r["optimized_amount"] > 1e-9:
                out.append(f"{r['transport_mode']}:{r['optimized_amount']:.1f}")
        return ", ".join(out)

    mode_info = []
    for rid, g in route_df.groupby("route_id"):
        mode_info.append({
            "route_id": rid,
            "baseline_mode": get_baseline_mode(g),
            "optimized_mix": get_opt_mix(g),
        })
    mode_info = pd.DataFrame(mode_info)

    summary = summary.merge(mode_info, on="route_id", how="left")

    node_lookup = NODES_DF.set_index("node")
    summary["source_lon"] = summary["origin"].map(node_lookup["lon"])
    summary["source_lat"] = summary["origin"].map(node_lookup["lat"])
    summary["target_lon"] = summary["destination"].map(node_lookup["lon"])
    summary["target_lat"] = summary["destination"].map(node_lookup["lat"])

    return summary


def build_node_points(df: pd.DataFrame, result_df=None):
    node_names = set()

    if "site" in df.columns:
        node_names.update(df["site"].dropna().astype(str).tolist())
    if "usage_location" in df.columns:
        node_names.update([x for x in df["usage_location"].dropna().astype(str).tolist() if x != ""])
    if "origin" in df.columns:
        node_names.update([x for x in df["origin"].dropna().astype(str).tolist() if x != ""])
    if "destination" in df.columns:
        node_names.update([x for x in df["destination"].dropna().astype(str).tolist() if x != ""])

    points = NODES_DF[NODES_DF["node"].isin(node_names)].copy()
    if points.empty:
        return points

    baseline_map = {}
    optimized_map = {}

    work_df = result_df if result_df is not None else df
    work_df = work_df.copy()

    if "baseline_emissions_tco2" not in work_df.columns and "baseline_amount" in work_df.columns:
        work_df["baseline_emissions_tco2"] = (
            pd.to_numeric(work_df["baseline_amount"], errors="coerce").fillna(0)
            * pd.to_numeric(work_df["emission_factor_tco2_per_unit"], errors="coerce").fillna(0)
        )

    if "optimized_emissions_tco2" not in work_df.columns and "optimized_amount" in work_df.columns:
        work_df["optimized_emissions_tco2"] = (
            pd.to_numeric(work_df["optimized_amount"], errors="coerce").fillna(0)
            * pd.to_numeric(work_df["emission_factor_tco2_per_unit"], errors="coerce").fillna(0)
        )

    for _, r in work_df[work_df["scope"].isin(["Scope1", "Scope2"])].iterrows():
        node_name = r["site"]
        if r.get("usage_location", "") not in ["", None] and r["scope"] == "Scope2":
            node_name = r["usage_location"]

        baseline_map[node_name] = baseline_map.get(node_name, 0.0) + float(r.get("baseline_emissions_tco2", 0.0))
        optimized_map[node_name] = optimized_map.get(node_name, 0.0) + float(r.get("optimized_emissions_tco2", 0.0))

    points["baseline_tco2"] = points["node"].map(baseline_map).fillna(0.0)
    points["optimized_tco2"] = points["node"].map(optimized_map).fillna(0.0)

    if result_df is None:
        points["info"] = points.apply(
            lambda r: f"{r['node_type']} | baseline={r['baseline_tco2']:.2f} tCO2",
            axis=1
        )
    else:
        points["info"] = points.apply(
            lambda r: f"{r['node_type']} | baseline={r['baseline_tco2']:.2f} tCO2 | optimized={r['optimized_tco2']:.2f} tCO2",
            axis=1
        )
    return points


def make_folium_map(points_df: pd.DataFrame, routes_df: pd.DataFrame = None, result_mode=False):
    center_lat = float(NODES_DF["lat"].mean())
    center_lon = float(NODES_DF["lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

    # 노드 표시
    for _, r in points_df.iterrows():
        color = NODE_TYPE_COLOR.get(r["node_type"], "gray")
        popup_text = f"{r['node']} ({r['node_type']})<br>{r.get('info', '')}"
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.85,
            popup=folium.Popup(popup_text, max_width=350),
            tooltip=r["node"],
        ).add_to(m)

    # 경로 표시
    if routes_df is not None and not routes_df.empty:
        routes_df = routes_df.copy()

        # 물량 기준 선 굵기 계산
        # 최소 2, 최대 14 정도로 제한
        max_qty = routes_df["quantity_ton"].fillna(0).max()
        if max_qty <= 0:
            max_qty = 1.0

        for _, r in routes_df.iterrows():
            if pd.isna(r["source_lat"]) or pd.isna(r["target_lat"]):
                continue

            quantity = float(r.get("quantity_ton", 0.0))

            # 물량 비례 선 굵기
            weight = 2 + 12 * (quantity / max_qty)
            weight = max(2, min(14, weight))

            if result_mode:
                # 결과 지도에서는 optimized_mix에서 가장 큰 운송수단을 대표색으로 사용
                dominant_mode = get_dominant_mode_from_mix(r.get("optimized_mix", ""))
                if dominant_mode == "":
                    dominant_mode = r.get("baseline_mode", "")
                color = TRANSPORT_MODE_COLOR.get(dominant_mode, "gray")

                tooltip_text = (
                    f"{r['origin']} → {r['destination']} | "
                    f"물량={r['quantity_ton']:.1f} ton | "
                    f"거리={r['distance_km']:.1f} km | "
                    f"baseline={r.get('baseline_route_emissions', 0):.2f} tCO2 | "
                    f"optimized={r.get('emissions_tco2', 0):.2f} tCO2 | "
                    f"reduction={r.get('reduction_tco2', 0):.2f} tCO2 | "
                    f"optimized_mix={r.get('optimized_mix', '')}"
                )
            else:
                # 입력 지도에서는 현재 운송수단 색 사용
                baseline_mode = r.get("baseline_mode", "")
                color = TRANSPORT_MODE_COLOR.get(baseline_mode, "gray")

                tooltip_text = (
                    f"{r['origin']} → {r['destination']} | "
                    f"물량={r['quantity_ton']:.1f} ton | "
                    f"거리={r['distance_km']:.1f} km | "
                    f"baseline={r.get('emissions_tco2', 0):.2f} tCO2 | "
                    f"current={baseline_mode}"
                )

            folium.PolyLine(
                locations=[
                    [r["source_lat"], r["source_lon"]],
                    [r["target_lat"], r["target_lon"]],
                ],
                color=color,
                weight=weight,
                opacity=0.8,
                tooltip=tooltip_text,
            ).add_to(m)

    return m


# ============================================================
# 3. 제목
# ============================================================
st.title("공급망 기반 탄소배출량 최적화 데모")
st.write(
    "사용자가 Scope 1/2/3 구조를 직접 선택하고, 공급망 이동 경로와 운송수단을 구성한 뒤, "
    "총 탄소배출량을 최소화하는 데모입니다."
)

tab1, tab2, tab3 = st.tabs(["공급망 구조 및 입력", "Scope별 최적화", "결과 및 보고서"])

# ============================================================
# 4. 탭 1: 공급망 구조 및 입력
# ============================================================
with tab1:
    st.header("1. 공급망 구조 및 입력")

    selected_scopes = st.multiselect("활성 Scope 선택", SCOPE_LIST, default=SCOPE_LIST)

    if not selected_scopes:
        st.warning("최소 하나의 Scope를 선택하세요.")
        st.stop()

    st.subheader("고급 옵션: 통합 CSV 업로드")
    st.caption("업로드를 하지 않으면 아래 UI에서 구성한 기본 구조 데이터를 사용합니다.")
    uploaded_file = st.file_uploader("통합 CSV 업로드 (선택)", type=["csv"])

    ui_rows = []

    # ----------------------------
    # Scope 1
    # ----------------------------
    if "Scope1" in selected_scopes:
        st.markdown("---")
        st.subheader("Scope 1 - 사업장 내부 설비/연료 구조")
        scope1_sites = st.multiselect("Scope1 대상 사업장 선택", PLANT_NODES, default=["Plant_A"], key="scope1_sites")

        for site in scope1_sites:
            defaults = SITE_DEFAULTS.get(site, {"boiler_demand": 100.0, "fleet_demand": 30.0})

            with st.expander(f"{site} - Scope1 설정", expanded=False):
                include_boiler = st.checkbox("보일러 사용", value=True, key=f"{site}_boiler_on")
                if include_boiler:
                    boiler_demand = st.number_input(
                        f"{site} 보일러 열수요 (MWh)",
                        min_value=0.0,
                        value=float(defaults["boiler_demand"]),
                        step=10.0,
                        key=f"{site}_boiler_demand",
                    )
                    boiler_candidates = st.multiselect(
                        f"{site} 보일러 열원 후보",
                        list(SCOPE1_BOILER_OPTIONS.keys()),
                        default=list(SCOPE1_BOILER_OPTIONS.keys()),
                        key=f"{site}_boiler_candidates",
                    )
                    if not boiler_candidates:
                        boiler_candidates = ["Natural_Gas"]
                    baseline_boiler = st.selectbox(
                        f"{site} 현재 보일러 열원",
                        boiler_candidates,
                        key=f"{site}_baseline_boiler",
                    )

                    ui_rows.extend(
                        make_choice_rows(
                            scope="Scope1",
                            site=site,
                            activity_group="Boiler_Heat",
                            demand=boiler_demand,
                            candidate_options=boiler_candidates,
                            baseline_option=baseline_boiler,
                            option_catalog=SCOPE1_BOILER_OPTIONS,
                            description=f"{site} 보일러 열원",
                            extra_cols={c: "" for c in OPTIONAL_COLS},
                        )
                    )

                include_fleet = st.checkbox("사내차량 사용", value=True, key=f"{site}_fleet_on")
                if include_fleet:
                    fleet_demand = st.number_input(
                        f"{site} 사내차량 에너지 수요 (MWh)",
                        min_value=0.0,
                        value=float(defaults["fleet_demand"]),
                        step=5.0,
                        key=f"{site}_fleet_demand",
                    )
                    fleet_candidates = st.multiselect(
                        f"{site} 차량 연료 후보",
                        list(SCOPE1_FLEET_OPTIONS.keys()),
                        default=list(SCOPE1_FLEET_OPTIONS.keys()),
                        key=f"{site}_fleet_candidates",
                    )
                    if not fleet_candidates:
                        fleet_candidates = ["Diesel_Fleet"]
                    baseline_fleet = st.selectbox(
                        f"{site} 현재 차량 연료",
                        fleet_candidates,
                        key=f"{site}_baseline_fleet",
                    )

                    ui_rows.extend(
                        make_choice_rows(
                            scope="Scope1",
                            site=site,
                            activity_group="Fleet_Fuel",
                            demand=fleet_demand,
                            candidate_options=fleet_candidates,
                            baseline_option=baseline_fleet,
                            option_catalog=SCOPE1_FLEET_OPTIONS,
                            description=f"{site} 사내차량 연료",
                            extra_cols={c: "" for c in OPTIONAL_COLS},
                        )
                    )

    # ----------------------------
    # Scope 2
    # ----------------------------
    if "Scope2" in selected_scopes:
        st.markdown("---")
        st.subheader("Scope 2 - 전력 사용 위치 / 조달 지역 / 조달 방식")
        scope2_sites = st.multiselect("Scope2 대상 사업장 선택", PLANT_NODES, default=["Plant_A"], key="scope2_sites")

        for site in scope2_sites:
            defaults = SITE_DEFAULTS.get(site, {"power_demand": 150.0})

            with st.expander(f"{site} - Scope2 설정", expanded=False):
                usage_location = st.selectbox(
                    f"{site} 전력 사용 위치",
                    [site] + WAREHOUSE_NODES,
                    key=f"{site}_scope2_location",
                )
                procurement_region = st.selectbox(
                    f"{site} 조달 지역",
                    ["Region_A", "Region_B"],
                    key=f"{site}_scope2_region",
                )
                power_demand = st.number_input(
                    f"{site} 전력 수요 (MWh)",
                    min_value=0.0,
                    value=float(defaults["power_demand"]),
                    step=10.0,
                    key=f"{site}_scope2_demand",
                )
                method_candidates = st.multiselect(
                    f"{site} 전력 조달 방식 후보",
                    ["Grid", "Solar_Onsite"],
                    default=["Grid", "Solar_Onsite"],
                    key=f"{site}_scope2_candidates",
                )
                if not method_candidates:
                    method_candidates = ["Grid"]
                baseline_method = st.selectbox(
                    f"{site} 현재 전력 조달 방식",
                    method_candidates,
                    key=f"{site}_scope2_baseline",
                )

                ui_rows.extend(
                    build_scope2_rows(
                        site=site,
                        usage_location=usage_location,
                        procurement_region=procurement_region,
                        demand=power_demand,
                        candidate_methods=method_candidates,
                        baseline_method=baseline_method,
                    )
                )

    # ----------------------------
    # Scope 3
    # ----------------------------
    if "Scope3" in selected_scopes:
        st.markdown("---")
        st.subheader("Scope 3 - 공급망 이동 흐름 선택")
        st.caption("아래 버튼을 누른 뒤 지도에서 노드를 클릭하면 출발지/도착지가 자동으로 선택됩니다. 최대 50개 경로까지 생성할 수 있습니다.")

        scope3_sites = st.multiselect("Scope3 관련 사업장 선택", PLANT_NODES, default=["Plant_A"], key="scope3_sites")

        num_routes = st.number_input("추가할 공급망 경로 수", min_value=1, max_value=50, value=2, step=1)

        # 세션 상태 초기화
        if "pending_pick" not in st.session_state:
            st.session_state["pending_pick"] = None

        for i in range(int(num_routes)):
            if f"route_{i}_site" not in st.session_state:
                st.session_state[f"route_{i}_site"] = scope3_sites[0] if scope3_sites else "Plant_A"
            if f"route_{i}_origin" not in st.session_state:
                st.session_state[f"route_{i}_origin"] = SUPPLIER_NODES[0]
            if f"route_{i}_destination" not in st.session_state:
                st.session_state[f"route_{i}_destination"] = PLANT_NODES[0]
            if f"route_{i}_quantity" not in st.session_state:
                st.session_state[f"route_{i}_quantity"] = 120.0 if i == 0 else 80.0
            if f"route_{i}_modes" not in st.session_state:
                st.session_state[f"route_{i}_modes"] = ["Truck", "Rail"]
            if f"route_{i}_baseline_mode" not in st.session_state:
                st.session_state[f"route_{i}_baseline_mode"] = "Truck"

        st.write("### 노드 클릭 지도")
        map_points_for_input = NODES_DF.copy()

        # 현재 선택된 경로를 지도에 미리 표시
        preview_routes = []
        for i in range(int(num_routes)):
            origin = st.session_state.get(f"route_{i}_origin", "")
            destination = st.session_state.get(f"route_{i}_destination", "")
            quantity = st.session_state.get(f"route_{i}_quantity", 0.0)
            if origin and destination and origin != destination:
                ori = get_node_row(origin)
                dst = get_node_row(destination)
                if ori and dst:
                    preview_routes.append({
                        "route_id": f"R{i+1}",
                        "origin": origin,
                        "destination": destination,
                        "quantity_ton": float(quantity),
                        "distance_km": haversine_km(ori["lat"], ori["lon"], dst["lat"], dst["lon"]),
                        "source_lat": ori["lat"],
                        "source_lon": ori["lon"],
                        "target_lat": dst["lat"],
                        "target_lon": dst["lon"],
                        "baseline_mode": st.session_state.get(f"route_{i}_baseline_mode", ""),
                        "emissions_tco2": 0.0,
                        "label": f"R{i+1}",
                        "info": f"{origin} → {destination} | 물량={float(quantity):.1f} ton",
                    })
        preview_routes_df = pd.DataFrame(preview_routes) if preview_routes else pd.DataFrame()

        input_map = make_folium_map(map_points_for_input, preview_routes_df, result_mode=False)
        map_result = st_folium(input_map, width=None, height=500, key="supply_chain_input_map")

        if st.session_state["pending_pick"] is not None and map_result and map_result.get("last_clicked"):
            clicked = map_result["last_clicked"]
            nearest_node = find_nearest_node(clicked["lat"], clicked["lng"])

            pick_type, idx_str = st.session_state["pending_pick"].split("|")
            idx = int(idx_str)

            if pick_type == "origin":
                st.session_state[f"route_{idx}_origin"] = nearest_node
            elif pick_type == "destination":
                st.session_state[f"route_{idx}_destination"] = nearest_node

            st.session_state["pending_pick"] = None
            st.rerun()

        if st.session_state["pending_pick"] is not None:
            pick_type, idx_str = st.session_state["pending_pick"].split("|")
            if pick_type == "origin":
                st.info(f"경로 {int(idx_str)+1}의 출발지를 선택 중입니다. 지도에서 노드를 클릭하세요.")
            else:
                st.info(f"경로 {int(idx_str)+1}의 도착지를 선택 중입니다. 지도에서 노드를 클릭하세요.")

        for i in range(int(num_routes)):
            with st.expander(f"경로 {i+1} 설정", expanded=(i == 0)):
                st.selectbox(
                    f"경로 {i+1} 관련 사업장",
                    scope3_sites if scope3_sites else PLANT_NODES,
                    key=f"route_{i}_site",
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"현재 출발지: **{st.session_state[f'route_{i}_origin']}**")
                    if st.button(f"경로 {i+1} 출발지 지도에서 선택", key=f"pick_origin_{i}"):
                        st.session_state["pending_pick"] = f"origin|{i}"
                        st.rerun()

                with c2:
                    st.write(f"현재 도착지: **{st.session_state[f'route_{i}_destination']}**")
                    if st.button(f"경로 {i+1} 도착지 지도에서 선택", key=f"pick_destination_{i}"):
                        st.session_state["pending_pick"] = f"destination|{i}"
                        st.rerun()

                # 수동 보정용 선택박스
                st.selectbox(
                    f"경로 {i+1} 출발지(수동 수정 가능)",
                    ALL_NODE_NAMES,
                    key=f"route_{i}_origin",
                )
                destination_candidates = [n for n in ALL_NODE_NAMES if n != st.session_state[f"route_{i}_origin"]]
                if st.session_state[f"route_{i}_destination"] == st.session_state[f"route_{i}_origin"]:
                    st.session_state[f"route_{i}_destination"] = destination_candidates[0]
                st.selectbox(
                    f"경로 {i+1} 도착지(수동 수정 가능)",
                    destination_candidates,
                    key=f"route_{i}_destination",
                )

                st.number_input(
                    f"경로 {i+1} 물량 (ton)",
                    min_value=1.0,
                    value=float(st.session_state[f"route_{i}_quantity"]),
                    step=10.0,
                    key=f"route_{i}_quantity",
                )

                st.multiselect(
                    f"경로 {i+1} 운송수단 후보",
                    list(TRANSPORT_MODE_META.keys()),
                    default=st.session_state[f"route_{i}_modes"],
                    key=f"route_{i}_modes",
                )
                if not st.session_state[f"route_{i}_modes"]:
                    st.session_state[f"route_{i}_modes"] = ["Truck"]

                current_modes = st.session_state[f"route_{i}_modes"]
                current_baseline = st.session_state.get(f"route_{i}_baseline_mode", current_modes[0])
                if current_baseline not in current_modes:
                    st.session_state[f"route_{i}_baseline_mode"] = current_modes[0]
                st.selectbox(
                    f"경로 {i+1} 현재 운송수단",
                    current_modes,
                    key=f"route_{i}_baseline_mode",
                )

        # UI 기반 Scope3 데이터 생성
        for i in range(int(num_routes)):
            related_site = st.session_state[f"route_{i}_site"]
            origin = st.session_state[f"route_{i}_origin"]
            destination = st.session_state[f"route_{i}_destination"]
            quantity_ton = st.session_state[f"route_{i}_quantity"]
            candidate_modes = st.session_state[f"route_{i}_modes"]
            baseline_mode = st.session_state[f"route_{i}_baseline_mode"]

            ui_rows.extend(
                build_scope3_rows(
                    site=related_site,
                    route_id=f"R{i+1}",
                    origin=origin,
                    destination=destination,
                    quantity_ton=quantity_ton,
                    candidate_modes=candidate_modes,
                    baseline_mode=baseline_mode,
                )
            )

    # ----------------------------
    # 데이터 생성 / 업로드 우선순위
    # ----------------------------
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.info("현재 업로드한 CSV 파일을 사용 중입니다.")
    else:
        if ui_rows:
            raw_df = pd.DataFrame(ui_rows)
            st.info("현재 UI에서 구성한 공급망 데이터를 사용 중입니다.")
        else:
            raw_df = pd.read_csv(StringIO(DEFAULT_SAMPLE_TEXT))
            st.info("현재 기본 내장 샘플 데이터를 사용 중입니다.")

    raw_df = ensure_optional_columns(raw_df)

    is_valid, msg, cleaned_df = validate_input_df(raw_df)
    if not is_valid:
        st.error(msg)
        st.stop()
    else:
        st.success(msg)

    current_signature = build_signature(cleaned_df)
    if st.session_state.get("data_signature") != current_signature:
        st.session_state["data_signature"] = current_signature
        st.session_state.pop("opt_result", None)

    st.subheader("입력 컬럼 설명")
    st.dataframe(GUIDE_DF, use_container_width=True)

    st.subheader("activity_group 설명")
    st.dataframe(ACTIVITY_GROUP_GUIDE_DF, use_container_width=True)

    st.subheader("현재 구성된 공급망 데이터")
    st.dataframe(cleaned_df, use_container_width=True)

# ============================================================
# 5. 탭 2: Scope별 최적화
# ============================================================
with tab2:
    st.header("2. Scope별 최적화")
    sub1, sub2, sub3, sub4 = st.tabs(["Scope 1", "Scope 2", "Scope 3", "통합 최적화 실행"])

    with sub1:
        st.subheader("Scope 1 데이터")
        s1 = cleaned_df[cleaned_df["scope"] == "Scope1"]
        if s1.empty:
            st.info("현재 Scope 1 데이터가 없습니다.")
        else:
            st.dataframe(s1, use_container_width=True)

    with sub2:
        st.subheader("Scope 2 데이터")
        s2 = cleaned_df[cleaned_df["scope"] == "Scope2"]
        if s2.empty:
            st.info("현재 Scope 2 데이터가 없습니다.")
        else:
            st.dataframe(s2, use_container_width=True)

    with sub3:
        st.subheader("Scope 3 데이터")
        s3 = cleaned_df[cleaned_df["scope"] == "Scope3"]
        if s3.empty:
            st.info("현재 Scope 3 데이터가 없습니다.")
        else:
            st.dataframe(s3, use_container_width=True)

    with sub4:
        st.subheader("통합 최적화 실행")
        st.write("현재 구성한 Scope 1/2/3 전체를 대상으로 총 탄소배출량을 최소화합니다.")

        baseline_total_emissions = float(
            (cleaned_df["baseline_amount"] * cleaned_df["emission_factor_tco2_per_unit"]).sum()
        )
        baseline_total_cost = float(
            (cleaned_df["baseline_amount"] * cleaned_df["cost_per_unit"]).sum()
        )

        c1, c2 = st.columns(2)
        c1.metric("기준 총 탄소배출량 (tCO2)", f"{baseline_total_emissions:.2f}")
        c2.metric("기준 총 비용", f"{baseline_total_cost:.2f}")

        budget_increase_pct = st.slider(
            "허용 총비용 증가율 (%)",
            min_value=0,
            max_value=30,
            value=5,
            step=1
        )

        run_opt = st.button("통합 최적화 실행", key="run_integrated_optimization")

        if run_opt:
            result = optimize_total_emissions(cleaned_df, budget_increase_pct)
            st.session_state["opt_result"] = result

            st.subheader("최적화 결과 상태")
            st.write(f"상태: **{result['status']}**")

            if result["status"] not in ["OPTIMAL", "FEASIBLE"]:
                st.error("최적해를 찾지 못했습니다. 입력 데이터와 제약조건을 확인하세요.")
            else:
                st.success("최적화를 완료했습니다. 결과 및 보고서 탭에서 상세 결과를 확인하세요.")

# ============================================================
# 6. 탭 3: 결과 및 보고서
# ============================================================
with tab3:
    st.header("3. 결과 및 보고서")

    if "opt_result" not in st.session_state:
        st.info("먼저 'Scope별 최적화' 탭에서 통합 최적화를 실행하세요.")
    else:
        result = st.session_state["opt_result"]

        if result["status"] not in ["OPTIMAL", "FEASIBLE"]:
            st.error("최적해가 없어 결과를 표시할 수 없습니다.")
        else:
            total = result["total"]
            scope_summary = result["scope_summary"]
            activity_summary = result["activity_summary"]
            detail = ensure_optional_columns(result["detail"])

            st.subheader("총괄 결과")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("기준 총 배출량", f"{total['baseline_emissions_tco2']:.2f}")
            c2.metric("최적 총 배출량", f"{total['optimized_emissions_tco2']:.2f}")
            c3.metric("총 감축량", f"{total['reduction_tco2']:.2f}")
            c4.metric("총 감축률 (%)", f"{total['reduction_pct']:.2f}")

            c5, c6 = st.columns(2)
            c5.metric("기준 총 비용", f"{total['baseline_cost']:.2f}")
            c6.metric("최적 총 비용", f"{total['optimized_cost']:.2f}")

            st.subheader("Scope별 분석")
            st.dataframe(scope_summary, use_container_width=True)

            scope_chart = scope_summary.set_index("scope")[["baseline_emissions_tco2", "optimized_emissions_tco2"]]
            st.bar_chart(scope_chart)

            st.subheader("활동그룹(activity_group)별 분석")
            st.dataframe(activity_summary, use_container_width=True)

            st.subheader("상세 결과")
            st.dataframe(detail, use_container_width=True)

            st.subheader("공급망 결과 지도")
            result_points = build_node_points(cleaned_df, detail)
            result_routes = summarize_routes(detail, optimized=True)

            if not result_routes.empty:
                baseline_route_summary = summarize_routes(detail, optimized=False)[["route_id", "emissions_tco2"]].rename(
                    columns={"emissions_tco2": "baseline_route_emissions"}
                )
                result_routes = result_routes.merge(baseline_route_summary, on="route_id", how="left")
                result_routes["reduction_tco2"] = result_routes["baseline_route_emissions"] - result_routes["emissions_tco2"]

                result_map = make_folium_map(result_points, result_routes, result_mode=True)
                st_folium(result_map, width=None, height=550, key="result_supply_chain_map")
            else:
                st.info("지도에 표시할 Scope 3 이동 경로가 없습니다.")

            st.subheader("보고서 다운로드")
            scope_csv = scope_summary.to_csv(index=False).encode("utf-8-sig")
            activity_csv = activity_summary.to_csv(index=False).encode("utf-8-sig")
            detail_csv = detail.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="Scope 요약 CSV 다운로드",
                data=scope_csv,
                file_name="scope_summary_report.csv",
                mime="text/csv"
            )
            st.download_button(
                label="활동그룹 요약 CSV 다운로드",
                data=activity_csv,
                file_name="activity_summary_report.csv",
                mime="text/csv"
            )
            st.download_button(
                label="상세 결과 CSV 다운로드",
                data=detail_csv,
                file_name="detailed_optimization_report.csv",
                mime="text/csv"
            )
