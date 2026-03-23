import math
import hashlib
import pandas as pd
import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
from ortools.linear_solver import pywraplp

st.set_page_config(page_title="공급망 기반 탄소배출량 최적화 데모", layout="wide")

# ============================================================
# 1. 기본 설정
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
    "site_lat",
    "site_lon",
    "route_id",
    "route_scope",
    "sequence_no",
    "origin",
    "destination",
    "origin_lat",
    "origin_lon",
    "destination_lat",
    "destination_lon",
    "flow_amount",
    "flow_unit",
    "distance_km",
    "transport_mode",
    "usage_location",
    "procurement_region",
]

SCOPE_LIST = ["Scope1", "Scope2", "Scope3"]

# ------------------------------------------------------------
# Scope 1: 보일러 / 내부 차량 이동
# ------------------------------------------------------------
SCOPE1_BOILER_OPTIONS = {
    "Natural_Gas": {"ef": 0.22, "cost": 55, "unit": "MWh", "desc": "보일러 열원"},
    "LPG": {"ef": 0.27, "cost": 68, "unit": "MWh", "desc": "보일러 열원"},
    "Biomethane": {"ef": 0.06, "cost": 92, "unit": "MWh", "desc": "보일러 열원"},
}

SCOPE1_INTERNAL_VEHICLE_OPTIONS = {
    "Diesel_Internal_Truck": {"ef": 0.00045, "cost": 1.35, "unit": "ton-km", "desc": "사업장 내부 차량 이동"},
    "LNG_Internal_Truck": {"ef": 0.00030, "cost": 1.55, "unit": "ton-km", "desc": "사업장 내부 차량 이동"},
    "EV_Internal_Truck": {"ef": 0.00008, "cost": 1.90, "unit": "ton-km", "desc": "사업장 내부 차량 이동"},
}

# ------------------------------------------------------------
# Scope 2: 전기 owner -> 사업장(client)
# ------------------------------------------------------------
GRID_REGION_META = {
    "Region_A": {"ef": 0.48, "cost": 98},
    "Region_B": {"ef": 0.38, "cost": 106},
}
SOLAR_META = {"ef": 0.02, "cost": 135}

# ------------------------------------------------------------
# Scope 3: 외부 공급망 이동
# ------------------------------------------------------------
SCOPE3_TRANSPORT_OPTIONS = {
    "Truck": {"ef": 0.00018, "cost": 1.20, "unit": "ton-km", "desc": "외부 공급망 이동"},
    "Rail": {"ef": 0.00006, "cost": 1.55, "unit": "ton-km", "desc": "외부 공급망 이동"},
    "Ship": {"ef": 0.00004, "cost": 1.50, "unit": "ton-km", "desc": "외부 공급망 이동"},
}

TRANSPORT_MODE_COLOR = {
    "Diesel_Internal_Truck": "blue",
    "LNG_Internal_Truck": "green",
    "EV_Internal_Truck": "purple",
    "Grid": "orange",
    "Solar_Onsite": "darkgreen",
    "Truck": "blue",
    "Rail": "green",
    "Ship": "purple",
}

NODE_TYPE_COLOR = {
    "Plant": "red",
    "Scope2Owner": "orange",
    "S1Owner": "black",
    "S1Client": "cadetblue",
    "S3Owner": "black",
    "S3Client": "purple",
}

PLANT_DEFAULTS = [
    {"lat": 37.5665, "lon": 126.9780},   # Seoul
    {"lat": 35.1796, "lon": 129.0756},   # Busan
    {"lat": 36.3504, "lon": 127.3845},   # Daejeon
]

SITE_DEFAULTS = {
    "plant1": {"boiler_demand": 120.0, "power_demand": 180.0},
    "plant2": {"boiler_demand": 90.0, "power_demand": 160.0},
    "plant3": {"boiler_demand": 70.0, "power_demand": 130.0},
}

GUIDE_DF = pd.DataFrame(
    [
        ["scope", "문자", "Scope1 / Scope2 / Scope3", "배출 범주"],
        ["site", "문자", "plant1", "사업장명"],
        ["activity_group", "문자", "Boiler_Heat / Scope1_InternalRoute / Scope2_PowerRoute / Scope3_ExternalRoute", "같은 기능의 대체 선택지 묶음"],
        ["option_name", "문자", "Natural_Gas / Grid / Truck", "개별 선택지 이름"],
        ["baseline_amount", "숫자", "100", "현재 기준 사용량"],
        ["min_amount", "숫자", "0", "최적화 후 최소값"],
        ["max_amount", "숫자", "140", "최적화 후 최대값"],
        ["unit", "문자", "MWh / ton-km", "사용 단위"],
        ["emission_factor_tco2_per_unit", "숫자", "0.22", "단위당 배출계수"],
        ["cost_per_unit", "숫자", "55", "단위당 비용"],
        ["site_lat/site_lon", "숫자", "37.56 / 126.97", "사업장 위치"],
        ["origin_lat/origin_lon", "숫자", "31.23 / 121.47", "경로 출발 위치"],
        ["destination_lat/destination_lon", "숫자", "37.56 / 126.97", "경로 도착 위치"],
    ],
    columns=["컬럼명", "자료형", "예시", "설명"]
)

ACTIVITY_GROUP_GUIDE_DF = pd.DataFrame(
    [
        ["Boiler_Heat", "사업장 위치와 동일한 Scope1 고정배출 구조", "Natural_Gas / LPG / Biomethane"],
        ["S1_Internal_<plant>_<k>", "사업장 내부 차량 공급망 구조", "Diesel_Internal_Truck / LNG_Internal_Truck / EV_Internal_Truck"],
        ["S2_Power_<plant>", "전기 owner -> 사업장(client) 구조", "Grid / Solar_Onsite"],
        ["S3_External_<plant>_<k>", "외부 공급망 이동 구조", "Truck / Rail / Ship"],
    ],
    columns=["activity_group", "설명", "예시 option_name"]
)

# ============================================================
# 2. 유틸 함수
# ============================================================
def blank_optional_dict():
    return {c: "" for c in OPTIONAL_COLS}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return r * c


def ensure_optional_columns(df: pd.DataFrame):
    df = df.copy()
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = ""
    return df


def auto_s1_owner_name(site, seq_no):
    return f"{site}_s1_owner{seq_no}"


def auto_s1_client_name(site, seq_no):
    return f"{site}_s1_client{seq_no}"


def auto_s2_owner_name(site):
    return f"{site}_scope2_owner"


def auto_s3_owner_name(site, seq_no):
    return f"{site}_s3_owner{seq_no}"


def auto_s3_client_name(site, seq_no):
    return f"{site}_s3_client{seq_no}"


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


def build_scope1_boiler_rows(site, site_lat, site_lon, demand, candidate_options, baseline_option):
    rows = []
    if not candidate_options:
        candidate_options = ["Natural_Gas"]
    if baseline_option not in candidate_options:
        baseline_option = candidate_options[0]

    for opt in candidate_options:
        meta = SCOPE1_BOILER_OPTIONS[opt]
        row = {
            "scope": "Scope1",
            "site": site,
            "activity_group": "Boiler_Heat",
            "option_name": opt,
            "baseline_amount": float(demand) if opt == baseline_option else 0.0,
            "min_amount": 0.0,
            "max_amount": float(demand),
            "unit": meta["unit"],
            "emission_factor_tco2_per_unit": meta["ef"],
            "cost_per_unit": meta["cost"],
            "description": f"{site} 보일러 열원",
        }
        row.update(blank_optional_dict())
        row["site_lat"] = site_lat
        row["site_lon"] = site_lon
        rows.append(row)

    return rows


def build_scope1_internal_route_rows(
    site,
    site_lat,
    site_lon,
    seq_no,
    owner_lat,
    owner_lon,
    client_lat,
    client_lon,
    flow_amount_ton,
    candidate_modes,
    baseline_mode,
):
    if owner_lat is None or owner_lon is None or client_lat is None or client_lon is None:
        return []
    if owner_lat == client_lat and owner_lon == client_lon:
        return []

    if not candidate_modes:
        candidate_modes = ["Diesel_Internal_Truck"]
    if baseline_mode not in candidate_modes:
        baseline_mode = candidate_modes[0]

    distance_km = haversine_km(owner_lat, owner_lon, client_lat, client_lon)
    ton_km = float(flow_amount_ton) * float(distance_km)

    rows = []
    for mode in candidate_modes:
        meta = SCOPE1_INTERNAL_VEHICLE_OPTIONS[mode]
        row = {
            "scope": "Scope1",
            "site": site,
            "activity_group": f"S1_Internal_{site}_{seq_no}",
            "option_name": mode,
            "baseline_amount": ton_km if mode == baseline_mode else 0.0,
            "min_amount": 0.0,
            "max_amount": ton_km,
            "unit": meta["unit"],
            "emission_factor_tco2_per_unit": meta["ef"],
            "cost_per_unit": meta["cost"],
            "description": f"{site} 내부 차량 이동 {seq_no}",
        }
        row.update(blank_optional_dict())
        row["site_lat"] = site_lat
        row["site_lon"] = site_lon
        row["route_id"] = f"{site}_S1R{seq_no}"
        row["route_scope"] = "Scope1_Internal"
        row["sequence_no"] = seq_no
        row["origin"] = auto_s1_owner_name(site, seq_no)
        row["destination"] = auto_s1_client_name(site, seq_no)
        row["origin_lat"] = owner_lat
        row["origin_lon"] = owner_lon
        row["destination_lat"] = client_lat
        row["destination_lon"] = client_lon
        row["flow_amount"] = float(flow_amount_ton)
        row["flow_unit"] = "ton"
        row["distance_km"] = float(distance_km)
        row["transport_mode"] = mode
        rows.append(row)

    return rows


def build_scope2_power_rows(
    site,
    site_lat,
    site_lon,
    owner_lat,
    owner_lon,
    procurement_region,
    demand_mwh,
    candidate_methods,
    baseline_method,
):
    if not candidate_methods:
        candidate_methods = ["Grid"]
    if baseline_method not in candidate_methods:
        baseline_method = candidate_methods[0]

    rows = []
    for method in candidate_methods:
        if method == "Grid":
            ef = GRID_REGION_META[procurement_region]["ef"]
            cost = GRID_REGION_META[procurement_region]["cost"]
            owner_x = owner_lat if owner_lat is not None else site_lat
            owner_y = owner_lon if owner_lon is not None else site_lon
        else:
            ef = SOLAR_META["ef"]
            cost = SOLAR_META["cost"]
            owner_x = site_lat
            owner_y = site_lon

        row = {
            "scope": "Scope2",
            "site": site,
            "activity_group": f"S2_Power_{site}",
            "option_name": method,
            "baseline_amount": float(demand_mwh) if method == baseline_method else 0.0,
            "min_amount": 0.0,
            "max_amount": float(demand_mwh),
            "unit": "MWh",
            "emission_factor_tco2_per_unit": ef,
            "cost_per_unit": cost,
            "description": f"{site} 전력 조달",
        }
        row.update(blank_optional_dict())
        row["site_lat"] = site_lat
        row["site_lon"] = site_lon
        row["usage_location"] = site
        row["procurement_region"] = procurement_region
        row["route_id"] = f"{site}_S2"
        row["route_scope"] = "Scope2_Power"
        row["sequence_no"] = 1
        row["origin"] = auto_s2_owner_name(site)
        row["destination"] = site
        row["origin_lat"] = owner_x
        row["origin_lon"] = owner_y
        row["destination_lat"] = site_lat
        row["destination_lon"] = site_lon
        row["flow_amount"] = float(demand_mwh)
        row["flow_unit"] = "MWh"
        row["distance_km"] = float(haversine_km(owner_x, owner_y, site_lat, site_lon))
        row["transport_mode"] = method
        rows.append(row)

    return rows


def build_scope3_external_rows(
    site,
    site_lat,
    site_lon,
    seq_no,
    owner_lat,
    owner_lon,
    client_lat,
    client_lon,
    flow_amount_ton,
    candidate_modes,
    baseline_mode,
):
    if owner_lat is None or owner_lon is None or client_lat is None or client_lon is None:
        return []
    if owner_lat == client_lat and owner_lon == client_lon:
        return []

    if not candidate_modes:
        candidate_modes = ["Truck"]
    if baseline_mode not in candidate_modes:
        baseline_mode = candidate_modes[0]

    distance_km = haversine_km(owner_lat, owner_lon, client_lat, client_lon)
    ton_km = float(flow_amount_ton) * float(distance_km)

    rows = []
    for mode in candidate_modes:
        meta = SCOPE3_TRANSPORT_OPTIONS[mode]
        row = {
            "scope": "Scope3",
            "site": site,
            "activity_group": f"S3_External_{site}_{seq_no}",
            "option_name": mode,
            "baseline_amount": ton_km if mode == baseline_mode else 0.0,
            "min_amount": 0.0,
            "max_amount": ton_km,
            "unit": meta["unit"],
            "emission_factor_tco2_per_unit": meta["ef"],
            "cost_per_unit": meta["cost"],
            "description": f"{site} 외부 공급망 이동 {seq_no}",
        }
        row.update(blank_optional_dict())
        row["site_lat"] = site_lat
        row["site_lon"] = site_lon
        row["route_id"] = f"{site}_S3R{seq_no}"
        row["route_scope"] = "Scope3_External"
        row["sequence_no"] = seq_no
        row["origin"] = auto_s3_owner_name(site, seq_no)
        row["destination"] = auto_s3_client_name(site, seq_no)
        row["origin_lat"] = owner_lat
        row["origin_lon"] = owner_lon
        row["destination_lat"] = client_lat
        row["destination_lon"] = client_lon
        row["flow_amount"] = float(flow_amount_ton)
        row["flow_unit"] = "ton"
        row["distance_km"] = float(distance_km)
        row["transport_mode"] = mode
        rows.append(row)

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
        pd.to_numeric(result_df["baseline_amount"], errors="coerce").fillna(0)
        * pd.to_numeric(result_df["emission_factor_tco2_per_unit"], errors="coerce").fillna(0)
    )
    result_df["optimized_emissions_tco2"] = (
        pd.to_numeric(result_df["optimized_amount"], errors="coerce").fillna(0)
        * pd.to_numeric(result_df["emission_factor_tco2_per_unit"], errors="coerce").fillna(0)
    )

    result_df["baseline_cost"] = (
        pd.to_numeric(result_df["baseline_amount"], errors="coerce").fillna(0)
        * pd.to_numeric(result_df["cost_per_unit"], errors="coerce").fillna(0)
    )
    result_df["optimized_cost"] = (
        pd.to_numeric(result_df["optimized_amount"], errors="coerce").fillna(0)
        * pd.to_numeric(result_df["cost_per_unit"], errors="coerce").fillna(0)
    )

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
    total["reduction_tco2"] = total["baseline_emissions_tco2"] - total["optimized_emissions_tco2"]
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


def get_dominant_mode_from_mix(mix_text: str):
    if mix_text is None or str(mix_text).strip() == "":
        return ""
    best_mode = ""
    best_value = -1.0
    for part in str(mix_text).split(","):
        part = part.strip()
        if ":" not in part:
            continue
        mode, value = part.split(":", 1)
        try:
            value = float(value.strip())
        except ValueError:
            continue
        if value > best_value:
            best_value = value
            best_mode = mode.strip()
    return best_mode


def summarize_routes(df: pd.DataFrame, optimized=False):
    if "route_id" not in df.columns:
        return pd.DataFrame()

    route_df = df.copy()
    route_df = route_df[route_df["route_id"].astype(str).str.strip() != ""].copy()
    if route_df.empty:
        return pd.DataFrame()

    num_cols = [
        "flow_amount", "distance_km",
        "origin_lat", "origin_lon",
        "destination_lat", "destination_lon",
        "baseline_amount", "emission_factor_tco2_per_unit",
        "sequence_no"
    ]
    for col in num_cols:
        if col in route_df.columns:
            route_df[col] = pd.to_numeric(route_df[col], errors="coerce")

    if "baseline_emissions_tco2" not in route_df.columns:
        route_df["baseline_emissions_tco2"] = (
            route_df["baseline_amount"].fillna(0)
            * pd.to_numeric(route_df["emission_factor_tco2_per_unit"], errors="coerce").fillna(0)
        )

    if "optimized_emissions_tco2" not in route_df.columns:
        if "optimized_amount" in route_df.columns:
            route_df["optimized_emissions_tco2"] = (
                pd.to_numeric(route_df["optimized_amount"], errors="coerce").fillna(0)
                * pd.to_numeric(route_df["emission_factor_tco2_per_unit"], errors="coerce").fillna(0)
            )
        else:
            route_df["optimized_emissions_tco2"] = route_df["baseline_emissions_tco2"]

    agg_col = "optimized_emissions_tco2" if optimized else "baseline_emissions_tco2"

    summary = route_df.groupby(
        ["route_id", "route_scope", "sequence_no", "site", "origin", "destination",
         "origin_lat", "origin_lon", "destination_lat", "destination_lon"],
        as_index=False
    ).agg(
        flow_amount=("flow_amount", "first"),
        flow_unit=("flow_unit", "first"),
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
            amt = float(r["optimized_amount"])
            if amt > 1e-9:
                out.append(f"{r['transport_mode']}:{amt:.1f}")
        return ", ".join(out)

    mode_info = []
    for rid, g in route_df.groupby("route_id"):
        mode_info.append({
            "route_id": rid,
            "baseline_mode": get_baseline_mode(g),
            "optimized_mix": get_opt_mix(g),
        })
    mode_info = pd.DataFrame(mode_info)

    return summary.merge(mode_info, on="route_id", how="left")


def build_node_points(df: pd.DataFrame, result_df=None, plant_defs=None):
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

    baseline_map = {}
    optimized_map = {}

    for _, r in work_df[work_df["scope"].isin(["Scope1", "Scope2"])].iterrows():
        node_name = r["site"]
        baseline_map[node_name] = baseline_map.get(node_name, 0.0) + float(r.get("baseline_emissions_tco2", 0.0))
        optimized_map[node_name] = optimized_map.get(node_name, 0.0) + float(r.get("optimized_emissions_tco2", 0.0))

    points = []

    if plant_defs:
        for p in plant_defs:
            if p["lat"] is None or p["lon"] is None:
                continue
            info = f"Plant | baseline={baseline_map.get(p['name'], 0.0):.2f} tCO2"
            if result_df is not None:
                info += f" | optimized={optimized_map.get(p['name'], 0.0):.2f} tCO2"
            points.append({
                "node": p["name"],
                "node_type": "Plant",
                "lat": p["lat"],
                "lon": p["lon"],
                "info": info,
            })

    route_df = df[df["route_id"].astype(str).str.strip() != ""].copy()
    for _, r in route_df.iterrows():
        try:
            o_lat = float(r["origin_lat"])
            o_lon = float(r["origin_lon"])
            d_lat = float(r["destination_lat"])
            d_lon = float(r["destination_lon"])
        except Exception:
            continue

        if r["route_scope"] == "Scope1_Internal":
            otype, dtype = "S1Owner", "S1Client"
        elif r["route_scope"] == "Scope2_Power":
            otype, dtype = "Scope2Owner", "Plant"
        else:
            otype, dtype = "S3Owner", "S3Client"

        points.append({
            "node": str(r["origin"]),
            "node_type": otype,
            "lat": o_lat,
            "lon": o_lon,
            "info": f"{otype} | {r['origin']}",
        })
        points.append({
            "node": str(r["destination"]),
            "node_type": dtype,
            "lat": d_lat,
            "lon": d_lon,
            "info": f"{dtype} | {r['destination']}",
        })

    if not points:
        return pd.DataFrame(columns=["node", "node_type", "lat", "lon", "info"])

    return pd.DataFrame(points).drop_duplicates(subset=["node", "lat", "lon", "node_type"])


def get_route_style(route_scope: str):
    """
    scope별 선 스타일을 더 직관적으로 구분
    - Scope1 내부차량: 점선(dotted)
    - Scope2 전력: 긴 대시(dashed)
    - Scope3 외부공급망: 실선(solid)
    """
    if route_scope == "Scope1_Internal":
        return {
            "dash_array": "2, 14",
            "prefix": "S1",
            "weight_scale": 0.95,
        }
    if route_scope == "Scope2_Power":
        return {
            "dash_array": "14, 10",
            "prefix": "S2",
            "weight_scale": 0.80,
        }
    return {
        "dash_array": None,
        "prefix": "S3",
        "weight_scale": 1.15,
    }
    

def add_sequence_label(m, lat, lon, text, border_color):
    html = f"""
    <div style="
        font-size: 11px;
        font-weight: bold;
        color: black;
        background: white;
        border: 2px solid {border_color};
        border-radius: 10px;
        padding: 2px 6px;
        white-space: nowrap;
        text-align: center;
        box-shadow: 0 0 3px rgba(0,0,0,0.35);
    ">{text}</div>
    """
    folium.Marker(
        [lat, lon],
        icon=folium.DivIcon(html=html)
    ).add_to(m)


def interpolate_point(lat1, lon1, lat2, lon2, ratio=0.94):
    lat = lat1 + (lat2 - lat1) * ratio
    lon = lon1 + (lon2 - lon1) * ratio
    return lat, lon


def add_endpoint_arrow(m, lat1, lon1, lat2, lon2, color):
    arrow_lat, arrow_lon = interpolate_point(lat1, lon1, lat2, lon2, ratio=0.94)
    html = f"""
    <div style="
        font-size: 18px;
        font-weight: bold;
        color: {color};
        text-shadow: 0 0 2px white, 0 0 2px white;
        transform: translate(-50%, -50%);
    ">▶</div>
    """
    folium.Marker(
        [arrow_lat, arrow_lon],
        icon=folium.DivIcon(html=html)
    ).add_to(m)


def add_map_legend(m):
    legend_html = """
    <div style="
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        background: white;
        border: 2px solid #555;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        min-width: 220px;
    ">
      <div style="font-weight: bold; margin-bottom: 8px;">지도 범례</div>

      <div style="margin-bottom: 6px; font-weight: bold;">선 종류 (Scope)</div>
      <div style="display:flex; align-items:center; margin-bottom:4px;">
        <span style="display:inline-block; width:42px; border-top:4px dotted #444; margin-right:8px;"></span>
        <span>Scope1 내부차량</span>
      </div>
      <div style="display:flex; align-items:center; margin-bottom:4px;">
        <span style="display:inline-block; width:42px; border-top:4px dashed #444; margin-right:8px;"></span>
        <span>Scope2 전력</span>
      </div>
      <div style="display:flex; align-items:center; margin-bottom:8px;">
        <span style="display:inline-block; width:42px; border-top:4px solid #444; margin-right:8px;"></span>
        <span>Scope3 외부공급망</span>
      </div>

      <div style="margin-bottom: 6px; font-weight: bold;">색상 의미</div>
      <div style="margin-bottom:2px;">선 색상 = 운송수단 / 조달방식</div>
      <div style="margin-bottom:2px;">선 굵기 = 물량 크기</div>
      <div style="margin-bottom:2px;">▶ = 이동 방향(끝점)</div>
      <div>S1-1 / S3-2 = 순서 라벨</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

def make_folium_map(points_df: pd.DataFrame, routes_df: pd.DataFrame = None, result_mode=False):
    if points_df is None or points_df.empty:
        m = folium.Map(location=[20, 110], zoom_start=2, tiles="CartoDB positron")
        add_map_legend(m)
        return m

    center_lat = float(points_df["lat"].mean())
    center_lon = float(points_df["lon"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=3, tiles="CartoDB positron")

    for _, r in points_df.iterrows():
        color = NODE_TYPE_COLOR.get(r["node_type"], "gray")
        popup_text = f"{r['node']} ({r['node_type']})<br>{r.get('info', '')}"
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.85,
            popup=folium.Popup(popup_text, max_width=350),
            tooltip=r["node"],
        ).add_to(m)

    if isinstance(routes_df, pd.DataFrame) and not routes_df.empty:
        routes_df = routes_df.copy()
        max_flow = pd.to_numeric(routes_df["flow_amount"], errors="coerce").fillna(0).max()
        if max_flow <= 0:
            max_flow = 1.0

        for _, r in routes_df.iterrows():
            if pd.isna(r["origin_lat"]) or pd.isna(r["destination_lat"]):
                continue

            flow_amount = float(r.get("flow_amount", 0.0))
            style = get_route_style(r["route_scope"])

            base_weight = 2 + 12 * (flow_amount / max_flow)
            weight = base_weight * style["weight_scale"]
            weight = max(2, min(16, weight))

            if result_mode:
                dominant_mode = get_dominant_mode_from_mix(r.get("optimized_mix", ""))
                if dominant_mode == "":
                    dominant_mode = r.get("baseline_mode", "")
                color = TRANSPORT_MODE_COLOR.get(dominant_mode, "gray")
                tooltip_text = (
                    f"{r['route_scope']} | {r['origin']} → {r['destination']} | "
                    f"flow={r['flow_amount']:.1f} {r.get('flow_unit', '')} | "
                    f"distance={r['distance_km']:.1f} km | "
                    f"baseline={r.get('baseline_route_emissions', 0):.2f} tCO2 | "
                    f"optimized={r.get('emissions_tco2', 0):.2f} tCO2 | "
                    f"reduction={r.get('reduction_tco2', 0):.2f} tCO2 | "
                    f"optimized_mix={r.get('optimized_mix', '')}"
                )
            else:
                baseline_mode = r.get("baseline_mode", "")
                color = TRANSPORT_MODE_COLOR.get(baseline_mode, "gray")
                tooltip_text = (
                    f"{r['route_scope']} | {r['origin']} → {r['destination']} | "
                    f"flow={r['flow_amount']:.1f} {r.get('flow_unit', '')} | "
                    f"distance={r['distance_km']:.1f} km | "
                    f"baseline={r.get('emissions_tco2', 0):.2f} tCO2 | "
                    f"current={baseline_mode}"
                )

            locations = [
                [r["origin_lat"], r["origin_lon"]],
                [r["destination_lat"], r["destination_lon"]],
            ]

            folium.PolyLine(
                locations=locations,
                color=color,
                weight=weight,
                opacity=0.85,
                tooltip=tooltip_text,
                dash_array=style["dash_array"],
            ).add_to(m)

            # 화살표는 끝점 근처에만 1개 표시
            add_endpoint_arrow(
                m,
                float(r["origin_lat"]),
                float(r["origin_lon"]),
                float(r["destination_lat"]),
                float(r["destination_lon"]),
                color
            )

            # Scope1 내부차량 / Scope3 외부공급망만 순서 라벨 표시
            if r["route_scope"] in ["Scope1_Internal", "Scope3_External"]:
                mid_lat = (float(r["origin_lat"]) + float(r["destination_lat"])) / 2
                mid_lon = (float(r["origin_lon"]) + float(r["destination_lon"])) / 2
                seq_label = f"{style['prefix']}-{int(r['sequence_no'])}"
                add_sequence_label(m, mid_lat, mid_lon, seq_label, color)

    add_map_legend(m)
    return m

def build_default_sample_df():
    plant_defs = [
        {"name": "plant1", "lat": 37.5665, "lon": 126.9780},
        {"name": "plant2", "lat": 35.1796, "lon": 129.0756},
    ]
    rows = []

    rows.extend(build_scope1_boiler_rows("plant1", 37.5665, 126.9780, 120.0, ["Natural_Gas", "LPG", "Biomethane"], "Natural_Gas"))
    rows.extend(build_scope1_internal_route_rows("plant1", 37.5665, 126.9780, 1, 37.5610, 126.9750, 37.5700, 126.9850, 25.0, ["Diesel_Internal_Truck", "EV_Internal_Truck"], "Diesel_Internal_Truck"))
    rows.extend(build_scope1_internal_route_rows("plant1", 37.5665, 126.9780, 2, 37.5580, 126.9700, 37.5675, 126.9920, 18.0, ["Diesel_Internal_Truck", "LNG_Internal_Truck"], "LNG_Internal_Truck"))
    rows.extend(build_scope2_power_rows("plant1", 37.5665, 126.9780, 37.4563, 126.7052, "Region_A", 180.0, ["Grid", "Solar_Onsite"], "Grid"))
    rows.extend(build_scope3_external_rows("plant1", 37.5665, 126.9780, 1, 31.2304, 121.4737, 37.5665, 126.9780, 120.0, ["Truck", "Rail", "Ship"], "Ship"))
    rows.extend(build_scope3_external_rows("plant1", 37.5665, 126.9780, 2, 35.6762, 139.6503, 37.5665, 126.9780, 80.0, ["Truck", "Rail", "Ship"], "Rail"))

    rows.extend(build_scope1_boiler_rows("plant2", 35.1796, 129.0756, 90.0, ["Natural_Gas", "LPG"], "LPG"))
    rows.extend(build_scope1_internal_route_rows("plant2", 35.1796, 129.0756, 1, 35.1700, 129.0600, 35.1900, 129.0850, 18.0, ["Diesel_Internal_Truck", "LNG_Internal_Truck"], "LNG_Internal_Truck"))
    rows.extend(build_scope2_power_rows("plant2", 35.1796, 129.0756, 35.1200, 129.0400, "Region_B", 160.0, ["Grid", "Solar_Onsite"], "Grid"))
    rows.extend(build_scope3_external_rows("plant2", 35.1796, 129.0756, 1, 35.6762, 139.6503, 35.1796, 129.0756, 80.0, ["Truck", "Rail", "Ship"], "Ship"))

    return pd.DataFrame(rows), plant_defs


# ============================================================
# 3. 세션 상태 초기화
# ============================================================
if "pending_pick" not in st.session_state:
    st.session_state["pending_pick"] = None

# ============================================================
# 4. 제목
# ============================================================
st.title("공급망 기반 탄소배출량 최적화 데모")
st.write(
    "사업장 위치를 지도에서 선택하고, Scope 1은 사업장 위치와 동일하게 연결하며, "
    "Scope 1 내부 차량 공급망 / Scope 2 전기 owner→사업장 / Scope 3 외부 공급망을 "
    "사업장별로 병렬 구성하는 데모입니다."
)

tab1, tab2, tab3 = st.tabs(["공급망 구조 및 입력", "Scope별 최적화", "결과 및 보고서"])

# ============================================================
# 5. 탭 1
# ============================================================
with tab1:
    st.header("1. 공급망 구조 및 입력")

    selected_scopes = st.multiselect("활성 Scope 선택", SCOPE_LIST, default=SCOPE_LIST)
    if not selected_scopes:
        st.warning("최소 하나의 Scope를 선택하세요.")
        st.stop()

    st.subheader("고급 옵션: 통합 CSV 업로드")
    st.caption("업로드를 하지 않으면 아래 UI에서 구성한 데이터 또는 기본 샘플 데이터를 사용합니다.")
    uploaded_file = st.file_uploader("통합 CSV 업로드 (선택)", type=["csv"])

    ui_rows = []
    plant_defs = []

    if uploaded_file is None:
        st.markdown("---")
        st.subheader("사업장 병렬 구성")
        num_plants = st.number_input("사업장 수", min_value=1, max_value=10, value=2, step=1)

        for i in range(int(num_plants)):
            plant_name = f"plant{i+1}"
            default_lat = PLANT_DEFAULTS[i]["lat"] if i < len(PLANT_DEFAULTS) else None
            default_lon = PLANT_DEFAULTS[i]["lon"] if i < len(PLANT_DEFAULTS) else None

            if f"plant_{i}_lat" not in st.session_state:
                st.session_state[f"plant_{i}_lat"] = default_lat
            if f"plant_{i}_lon" not in st.session_state:
                st.session_state[f"plant_{i}_lon"] = default_lon

            plant_defs.append({
                "index": i,
                "name": plant_name,
                "lat": st.session_state[f"plant_{i}_lat"],
                "lon": st.session_state[f"plant_{i}_lon"],
            })

        for p in plant_defs:
            i = p["index"]
            if f"plant_{i}_s1_route_count" not in st.session_state:
                st.session_state[f"plant_{i}_s1_route_count"] = 1
            if f"plant_{i}_s3_route_count" not in st.session_state:
                st.session_state[f"plant_{i}_s3_route_count"] = 1

        preview_points = []
        preview_routes = []

        for p in plant_defs:
            if p["lat"] is not None and p["lon"] is not None:
                preview_points.append({
                    "node": p["name"],
                    "node_type": "Plant",
                    "lat": p["lat"],
                    "lon": p["lon"],
                    "info": f"Plant | {p['name']}",
                })

            if "Scope2" in selected_scopes:
                if f"{p['name']}_scope2_owner_lat" not in st.session_state:
                    st.session_state[f"{p['name']}_scope2_owner_lat"] = None
                if f"{p['name']}_scope2_owner_lon" not in st.session_state:
                    st.session_state[f"{p['name']}_scope2_owner_lon"] = None
                if f"{p['name']}_scope2_method_candidates" not in st.session_state:
                    st.session_state[f"{p['name']}_scope2_method_candidates"] = ["Grid", "Solar_Onsite"]
                if f"{p['name']}_scope2_baseline" not in st.session_state:
                    st.session_state[f"{p['name']}_scope2_baseline"] = "Grid"

                s2_owner_lat = st.session_state[f"{p['name']}_scope2_owner_lat"]
                s2_owner_lon = st.session_state[f"{p['name']}_scope2_owner_lon"]
                if s2_owner_lat is not None and s2_owner_lon is not None:
                    preview_points.append({
                        "node": auto_s2_owner_name(p["name"]),
                        "node_type": "Scope2Owner",
                        "lat": s2_owner_lat,
                        "lon": s2_owner_lon,
                        "info": f"Scope2Owner | {auto_s2_owner_name(p['name'])}",
                    })
                    if p["lat"] is not None and p["lon"] is not None:
                        preview_routes.append({
                            "route_id": f"{p['name']}_S2",
                            "route_scope": "Scope2_Power",
                            "sequence_no": 1,
                            "origin": auto_s2_owner_name(p["name"]),
                            "destination": p["name"],
                            "origin_lat": s2_owner_lat,
                            "origin_lon": s2_owner_lon,
                            "destination_lat": p["lat"],
                            "destination_lon": p["lon"],
                            "flow_amount": float(st.session_state.get(f"{p['name']}_scope2_demand", SITE_DEFAULTS.get(p["name"], {"power_demand": 150.0})["power_demand"])),
                            "flow_unit": "MWh",
                            "distance_km": haversine_km(s2_owner_lat, s2_owner_lon, p["lat"], p["lon"]),
                            "baseline_mode": st.session_state[f"{p['name']}_scope2_baseline"],
                            "emissions_tco2": 0.0,
                        })

            if "Scope1" in selected_scopes:
                s1_count = int(st.session_state[f"plant_{p['index']}_s1_route_count"])
                for j in range(s1_count):
                    if f"{p['name']}_s1_{j}_owner_lat" not in st.session_state:
                        st.session_state[f"{p['name']}_s1_{j}_owner_lat"] = None
                    if f"{p['name']}_s1_{j}_owner_lon" not in st.session_state:
                        st.session_state[f"{p['name']}_s1_{j}_owner_lon"] = None
                    if f"{p['name']}_s1_{j}_client_lat" not in st.session_state:
                        st.session_state[f"{p['name']}_s1_{j}_client_lat"] = None
                    if f"{p['name']}_s1_{j}_client_lon" not in st.session_state:
                        st.session_state[f"{p['name']}_s1_{j}_client_lon"] = None
                    if f"{p['name']}_s1_{j}_flow" not in st.session_state:
                        st.session_state[f"{p['name']}_s1_{j}_flow"] = 20.0
                    if f"{p['name']}_s1_{j}_modes" not in st.session_state:
                        st.session_state[f"{p['name']}_s1_{j}_modes"] = ["Diesel_Internal_Truck", "EV_Internal_Truck"]
                    if f"{p['name']}_s1_{j}_baseline" not in st.session_state:
                        st.session_state[f"{p['name']}_s1_{j}_baseline"] = "Diesel_Internal_Truck"

                    o_lat = st.session_state[f"{p['name']}_s1_{j}_owner_lat"]
                    o_lon = st.session_state[f"{p['name']}_s1_{j}_owner_lon"]
                    c_lat = st.session_state[f"{p['name']}_s1_{j}_client_lat"]
                    c_lon = st.session_state[f"{p['name']}_s1_{j}_client_lon"]

                    if o_lat is not None and o_lon is not None:
                        preview_points.append({
                            "node": auto_s1_owner_name(p["name"], j + 1),
                            "node_type": "S1Owner",
                            "lat": o_lat,
                            "lon": o_lon,
                            "info": f"S1Owner | {auto_s1_owner_name(p['name'], j+1)}",
                        })
                    if c_lat is not None and c_lon is not None:
                        preview_points.append({
                            "node": auto_s1_client_name(p["name"], j + 1),
                            "node_type": "S1Client",
                            "lat": c_lat,
                            "lon": c_lon,
                            "info": f"S1Client | {auto_s1_client_name(p['name'], j+1)}",
                        })

                    if o_lat is not None and o_lon is not None and c_lat is not None and c_lon is not None:
                        preview_routes.append({
                            "route_id": f"{p['name']}_S1R{j+1}",
                            "route_scope": "Scope1_Internal",
                            "sequence_no": j + 1,
                            "origin": auto_s1_owner_name(p["name"], j + 1),
                            "destination": auto_s1_client_name(p["name"], j + 1),
                            "origin_lat": o_lat,
                            "origin_lon": o_lon,
                            "destination_lat": c_lat,
                            "destination_lon": c_lon,
                            "flow_amount": float(st.session_state[f"{p['name']}_s1_{j}_flow"]),
                            "flow_unit": "ton",
                            "distance_km": haversine_km(o_lat, o_lon, c_lat, c_lon),
                            "baseline_mode": st.session_state[f"{p['name']}_s1_{j}_baseline"],
                            "emissions_tco2": 0.0,
                        })

            if "Scope3" in selected_scopes:
                s3_count = int(st.session_state[f"plant_{p['index']}_s3_route_count"])
                for j in range(s3_count):
                    if f"{p['name']}_s3_{j}_owner_lat" not in st.session_state:
                        st.session_state[f"{p['name']}_s3_{j}_owner_lat"] = None
                    if f"{p['name']}_s3_{j}_owner_lon" not in st.session_state:
                        st.session_state[f"{p['name']}_s3_{j}_owner_lon"] = None
                    if f"{p['name']}_s3_{j}_client_lat" not in st.session_state:
                        st.session_state[f"{p['name']}_s3_{j}_client_lat"] = None
                    if f"{p['name']}_s3_{j}_client_lon" not in st.session_state:
                        st.session_state[f"{p['name']}_s3_{j}_client_lon"] = None
                    if f"{p['name']}_s3_{j}_flow" not in st.session_state:
                        st.session_state[f"{p['name']}_s3_{j}_flow"] = 120.0 if j == 0 else 80.0
                    if f"{p['name']}_s3_{j}_modes" not in st.session_state:
                        st.session_state[f"{p['name']}_s3_{j}_modes"] = ["Truck", "Rail", "Ship"]
                    if f"{p['name']}_s3_{j}_baseline" not in st.session_state:
                        st.session_state[f"{p['name']}_s3_{j}_baseline"] = "Ship"

                    o_lat = st.session_state[f"{p['name']}_s3_{j}_owner_lat"]
                    o_lon = st.session_state[f"{p['name']}_s3_{j}_owner_lon"]
                    c_lat = st.session_state[f"{p['name']}_s3_{j}_client_lat"]
                    c_lon = st.session_state[f"{p['name']}_s3_{j}_client_lon"]

                    if o_lat is not None and o_lon is not None:
                        preview_points.append({
                            "node": auto_s3_owner_name(p["name"], j + 1),
                            "node_type": "S3Owner",
                            "lat": o_lat,
                            "lon": o_lon,
                            "info": f"S3Owner | {auto_s3_owner_name(p['name'], j+1)}",
                        })
                    if c_lat is not None and c_lon is not None:
                        preview_points.append({
                            "node": auto_s3_client_name(p["name"], j + 1),
                            "node_type": "S3Client",
                            "lat": c_lat,
                            "lon": c_lon,
                            "info": f"S3Client | {auto_s3_client_name(p['name'], j+1)}",
                        })

                    if o_lat is not None and o_lon is not None and c_lat is not None and c_lon is not None:
                        preview_routes.append({
                            "route_id": f"{p['name']}_S3R{j+1}",
                            "route_scope": "Scope3_External",
                            "sequence_no": j + 1,
                            "origin": auto_s3_owner_name(p["name"], j + 1),
                            "destination": auto_s3_client_name(p["name"], j + 1),
                            "origin_lat": o_lat,
                            "origin_lon": o_lon,
                            "destination_lat": c_lat,
                            "destination_lon": c_lon,
                            "flow_amount": float(st.session_state[f"{p['name']}_s3_{j}_flow"]),
                            "flow_unit": "ton",
                            "distance_km": haversine_km(o_lat, o_lon, c_lat, c_lon),
                            "baseline_mode": st.session_state[f"{p['name']}_s3_{j}_baseline"],
                            "emissions_tco2": 0.0,
                        })

        preview_points_df = (
            pd.DataFrame(preview_points).drop_duplicates(subset=["node", "lat", "lon", "node_type"])
            if preview_points else
            pd.DataFrame(columns=["node", "node_type", "lat", "lon", "info"])
        )
        preview_routes_df = pd.DataFrame(preview_routes)

        st.write("### 지도 선택기")
        st.caption("위 버튼을 누른 뒤 아래 지도에서 위치를 클릭해.")
        pick_map = make_folium_map(preview_points_df, preview_routes_df, result_mode=False)
        map_result = st_folium(pick_map, width=None, height=600, key="structure_picker_map")

        if st.session_state["pending_pick"] is not None and map_result and map_result.get("last_clicked"):
            clicked = map_result["last_clicked"]
            parts = st.session_state["pending_pick"].split("|")
            kind = parts[0]

            if kind == "plant":
                plant_idx = int(parts[1])
                st.session_state[f"plant_{plant_idx}_lat"] = clicked["lat"]
                st.session_state[f"plant_{plant_idx}_lon"] = clicked["lng"]

            elif kind == "s2_owner":
                plant_name = parts[1]
                st.session_state[f"{plant_name}_scope2_owner_lat"] = clicked["lat"]
                st.session_state[f"{plant_name}_scope2_owner_lon"] = clicked["lng"]

            elif kind == "s1_owner":
                plant_name = parts[1]
                route_idx = int(parts[2])
                st.session_state[f"{plant_name}_s1_{route_idx}_owner_lat"] = clicked["lat"]
                st.session_state[f"{plant_name}_s1_{route_idx}_owner_lon"] = clicked["lng"]

            elif kind == "s1_client":
                plant_name = parts[1]
                route_idx = int(parts[2])
                st.session_state[f"{plant_name}_s1_{route_idx}_client_lat"] = clicked["lat"]
                st.session_state[f"{plant_name}_s1_{route_idx}_client_lon"] = clicked["lng"]

            elif kind == "s3_owner":
                plant_name = parts[1]
                route_idx = int(parts[2])
                st.session_state[f"{plant_name}_s3_{route_idx}_owner_lat"] = clicked["lat"]
                st.session_state[f"{plant_name}_s3_{route_idx}_owner_lon"] = clicked["lng"]

            elif kind == "s3_client":
                plant_name = parts[1]
                route_idx = int(parts[2])
                st.session_state[f"{plant_name}_s3_{route_idx}_client_lat"] = clicked["lat"]
                st.session_state[f"{plant_name}_s3_{route_idx}_client_lon"] = clicked["lng"]

            st.session_state["pending_pick"] = None
            st.rerun()

        if st.session_state["pending_pick"] is not None:
            parts = st.session_state["pending_pick"].split("|")
            kind = parts[0]
            if kind == "plant":
                st.info(f"사업장 {int(parts[1]) + 1} 위치를 지도에서 클릭하세요.")
            elif kind == "s2_owner":
                st.info(f"{parts[1]}의 Scope2 owner 위치를 지도에서 클릭하세요.")
            elif kind == "s1_owner":
                st.info(f"{parts[1]}의 Scope1 내부 차량 경로 {int(parts[2]) + 1} owner 위치를 지도에서 클릭하세요.")
            elif kind == "s1_client":
                st.info(f"{parts[1]}의 Scope1 내부 차량 경로 {int(parts[2]) + 1} client 위치를 지도에서 클릭하세요.")
            elif kind == "s3_owner":
                st.info(f"{parts[1]}의 Scope3 외부 경로 {int(parts[2]) + 1} owner 위치를 지도에서 클릭하세요.")
            elif kind == "s3_client":
                st.info(f"{parts[1]}의 Scope3 외부 경로 {int(parts[2]) + 1} client 위치를 지도에서 클릭하세요.")

        st.markdown("---")
        st.subheader("사업장별 병렬 설정")

        plant_defs = []
        for i in range(int(num_plants)):
            plant_name = f"plant{i+1}"
            plant_defs.append({
                "index": i,
                "name": plant_name,
                "lat": st.session_state[f"plant_{i}_lat"],
                "lon": st.session_state[f"plant_{i}_lon"],
            })

        for p in plant_defs:
            defaults = SITE_DEFAULTS.get(
                p["name"],
                {"boiler_demand": 80.0, "power_demand": 140.0}
            )

            with st.expander(f"{p['name']} 설정", expanded=(p["index"] == 0)):
                st.write(f"사업장 좌표: {p['lat']}, {p['lon']}")
                if st.button(f"{p['name']} 위치 지도에서 선택", key=f"pick_plant_{p['index']}"):
                    st.session_state["pending_pick"] = f"plant|{p['index']}"
                    st.rerun()

                if "Scope1" in selected_scopes:
                    st.write("#### Scope 1")
                    st.caption("Scope1 위치는 사업장 위치와 동일하게 적용됩니다.")

                    boiler_demand = st.number_input(
                        f"{p['name']} 보일러 열수요 (MWh)",
                        min_value=0.0,
                        value=float(defaults["boiler_demand"]),
                        step=10.0,
                        key=f"{p['name']}_boiler_demand",
                    )
                    boiler_candidates = st.multiselect(
                        f"{p['name']} 보일러 열원 후보",
                        list(SCOPE1_BOILER_OPTIONS.keys()),
                        default=list(SCOPE1_BOILER_OPTIONS.keys()),
                        key=f"{p['name']}_boiler_candidates",
                    )
                    if not boiler_candidates:
                        boiler_candidates = ["Natural_Gas"]
                    baseline_boiler = st.selectbox(
                        f"{p['name']} 현재 보일러 열원",
                        boiler_candidates,
                        key=f"{p['name']}_baseline_boiler",
                    )

                    ui_rows.extend(
                        build_scope1_boiler_rows(
                            p["name"], p["lat"], p["lon"],
                            boiler_demand,
                            boiler_candidates,
                            baseline_boiler,
                        )
                    )

                    st.write("##### Scope1 내부 차량 공급망 구조")
                    s1_count = st.number_input(
                        f"{p['name']} 내부 차량 경로 수",
                        min_value=0,
                        max_value=10,
                        value=int(st.session_state[f'plant_{p["index"]}_s1_route_count']),
                        step=1,
                        key=f"plant_{p['index']}_s1_route_count",
                    )

                    for j in range(int(s1_count)):
                        st.write(f"**내부 차량 경로 {j+1}**")
                        st.write(f"자동 owner 이름: **{auto_s1_owner_name(p['name'], j+1)}**")
                        st.write(f"자동 client 이름: **{auto_s1_client_name(p['name'], j+1)}**")

                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(
                                f"owner 좌표: {st.session_state.get(f'{p['name']}_s1_{j}_owner_lat')}, "
                                f"{st.session_state.get(f'{p['name']}_s1_{j}_owner_lon')}"
                            )
                            if st.button(f"{p['name']} 내부경로 {j+1} owner 지도에서 찍기", key=f"btn_{p['name']}_s1_owner_{j}"):
                                st.session_state["pending_pick"] = f"s1_owner|{p['name']}|{j}"
                                st.rerun()

                        with c2:
                            st.write(
                                f"client 좌표: {st.session_state.get(f'{p['name']}_s1_{j}_client_lat')}, "
                                f"{st.session_state.get(f'{p['name']}_s1_{j}_client_lon')}"
                            )
                            if st.button(f"{p['name']} 내부경로 {j+1} client 지도에서 찍기", key=f"btn_{p['name']}_s1_client_{j}"):
                                st.session_state["pending_pick"] = f"s1_client|{p['name']}|{j}"
                                st.rerun()

                        flow_amount = st.number_input(
                            f"{p['name']} 내부경로 {j+1} 물량 (ton)",
                            min_value=1.0,
                            value=float(st.session_state.get(f"{p['name']}_s1_{j}_flow", 20.0)),
                            step=5.0,
                            key=f"{p['name']}_s1_{j}_flow",
                        )
                        mode_candidates = st.multiselect(
                            f"{p['name']} 내부경로 {j+1} 차량 후보",
                            list(SCOPE1_INTERNAL_VEHICLE_OPTIONS.keys()),
                            default=st.session_state.get(f"{p['name']}_s1_{j}_modes", ["Diesel_Internal_Truck", "EV_Internal_Truck"]),
                            key=f"{p['name']}_s1_{j}_modes",
                        )
                        if not mode_candidates:
                            mode_candidates = ["Diesel_Internal_Truck"]
                            st.session_state[f"{p['name']}_s1_{j}_modes"] = mode_candidates

                        baseline_mode = st.selectbox(
                            f"{p['name']} 내부경로 {j+1} 현재 차량",
                            st.session_state[f"{p['name']}_s1_{j}_modes"],
                            key=f"{p['name']}_s1_{j}_baseline",
                        )

                        ui_rows.extend(
                            build_scope1_internal_route_rows(
                                site=p["name"],
                                site_lat=p["lat"],
                                site_lon=p["lon"],
                                seq_no=j + 1,
                                owner_lat=st.session_state.get(f"{p['name']}_s1_{j}_owner_lat"),
                                owner_lon=st.session_state.get(f"{p['name']}_s1_{j}_owner_lon"),
                                client_lat=st.session_state.get(f"{p['name']}_s1_{j}_client_lat"),
                                client_lon=st.session_state.get(f"{p['name']}_s1_{j}_client_lon"),
                                flow_amount_ton=flow_amount,
                                candidate_modes=mode_candidates,
                                baseline_mode=baseline_mode,
                            )
                        )

                if "Scope2" in selected_scopes:
                    st.write("#### Scope 2")
                    st.caption("Scope2 client 위치는 사업장 위치와 동일합니다. 전기 owner 위치만 별도로 지도에서 선택합니다.")

                    s2_owner_lat = st.session_state.get(f"{p['name']}_scope2_owner_lat")
                    s2_owner_lon = st.session_state.get(f"{p['name']}_scope2_owner_lon")
                    st.write(f"Scope2 owner 좌표: {s2_owner_lat}, {s2_owner_lon}")
                    if st.button(f"{p['name']} Scope2 owner 지도에서 선택", key=f"btn_{p['name']}_scope2_owner"):
                        st.session_state["pending_pick"] = f"s2_owner|{p['name']}"
                        st.rerun()

                    procurement_region = st.selectbox(
                        f"{p['name']} 조달 지역",
                        ["Region_A", "Region_B"],
                        key=f"{p['name']}_scope2_region",
                    )
                    power_demand = st.number_input(
                        f"{p['name']} 전력 수요 (MWh)",
                        min_value=0.0,
                        value=float(st.session_state.get(f"{p['name']}_scope2_demand", defaults["power_demand"])),
                        step=10.0,
                        key=f"{p['name']}_scope2_demand",
                    )
                    method_candidates = st.multiselect(
                        f"{p['name']} 전력 조달 방식 후보",
                        ["Grid", "Solar_Onsite"],
                        default=st.session_state.get(f"{p['name']}_scope2_method_candidates", ["Grid", "Solar_Onsite"]),
                        key=f"{p['name']}_scope2_method_candidates",
                    )
                    if not method_candidates:
                        method_candidates = ["Grid"]
                        st.session_state[f"{p['name']}_scope2_method_candidates"] = method_candidates

                    baseline_method = st.selectbox(
                        f"{p['name']} 현재 전력 조달 방식",
                        st.session_state[f"{p['name']}_scope2_method_candidates"],
                        key=f"{p['name']}_scope2_baseline",
                    )

                    ui_rows.extend(
                        build_scope2_power_rows(
                            site=p["name"],
                            site_lat=p["lat"],
                            site_lon=p["lon"],
                            owner_lat=s2_owner_lat,
                            owner_lon=s2_owner_lon,
                            procurement_region=procurement_region,
                            demand_mwh=power_demand,
                            candidate_methods=method_candidates,
                            baseline_method=baseline_method,
                        )
                    )

                if "Scope3" in selected_scopes:
                    st.write("#### Scope 3")
                    s3_count = st.number_input(
                        f"{p['name']} 외부 공급망 경로 수",
                        min_value=0,
                        max_value=50,
                        value=int(st.session_state[f'plant_{p["index"]}_s3_route_count']),
                        step=1,
                        key=f"plant_{p['index']}_s3_route_count",
                    )

                    for j in range(int(s3_count)):
                        st.write(f"**외부 공급망 경로 {j+1}**")
                        st.write(f"자동 owner 이름: **{auto_s3_owner_name(p['name'], j+1)}**")
                        st.write(f"자동 client 이름: **{auto_s3_client_name(p['name'], j+1)}**")

                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(
                                f"owner 좌표: {st.session_state.get(f'{p['name']}_s3_{j}_owner_lat')}, "
                                f"{st.session_state.get(f'{p['name']}_s3_{j}_owner_lon')}"
                            )
                            if st.button(f"{p['name']} 외부경로 {j+1} owner 지도에서 찍기", key=f"btn_{p['name']}_s3_owner_{j}"):
                                st.session_state["pending_pick"] = f"s3_owner|{p['name']}|{j}"
                                st.rerun()

                        with c2:
                            st.write(
                                f"client 좌표: {st.session_state.get(f'{p['name']}_s3_{j}_client_lat')}, "
                                f"{st.session_state.get(f'{p['name']}_s3_{j}_client_lon')}"
                            )
                            if st.button(f"{p['name']} 외부경로 {j+1} client 지도에서 찍기", key=f"btn_{p['name']}_s3_client_{j}"):
                                st.session_state["pending_pick"] = f"s3_client|{p['name']}|{j}"
                                st.rerun()

                        flow_amount = st.number_input(
                            f"{p['name']} 외부경로 {j+1} 물량 (ton)",
                            min_value=1.0,
                            value=float(st.session_state.get(f"{p['name']}_s3_{j}_flow", 120.0 if j == 0 else 80.0)),
                            step=10.0,
                            key=f"{p['name']}_s3_{j}_flow",
                        )
                        mode_candidates = st.multiselect(
                            f"{p['name']} 외부경로 {j+1} 운송수단 후보",
                            list(SCOPE3_TRANSPORT_OPTIONS.keys()),
                            default=st.session_state.get(f"{p['name']}_s3_{j}_modes", ["Truck", "Rail", "Ship"]),
                            key=f"{p['name']}_s3_{j}_modes",
                        )
                        if not mode_candidates:
                            mode_candidates = ["Truck"]
                            st.session_state[f"{p['name']}_s3_{j}_modes"] = mode_candidates

                        baseline_mode = st.selectbox(
                            f"{p['name']} 외부경로 {j+1} 현재 운송수단",
                            st.session_state[f"{p['name']}_s3_{j}_modes"],
                            key=f"{p['name']}_s3_{j}_baseline",
                        )

                        ui_rows.extend(
                            build_scope3_external_rows(
                                site=p["name"],
                                site_lat=p["lat"],
                                site_lon=p["lon"],
                                seq_no=j + 1,
                                owner_lat=st.session_state.get(f"{p['name']}_s3_{j}_owner_lat"),
                                owner_lon=st.session_state.get(f"{p['name']}_s3_{j}_owner_lon"),
                                client_lat=st.session_state.get(f"{p['name']}_s3_{j}_client_lat"),
                                client_lon=st.session_state.get(f"{p['name']}_s3_{j}_client_lon"),
                                flow_amount_ton=flow_amount,
                                candidate_modes=mode_candidates,
                                baseline_mode=baseline_mode,
                            )
                        )

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        raw_df = ensure_optional_columns(raw_df)
        st.info("현재 업로드한 CSV 파일을 사용 중입니다.")

        plant_map = {}
        for _, r in raw_df.iterrows():
            if str(r.get("site", "")).strip() == "":
                continue
            try:
                lat = float(r.get("site_lat", ""))
                lon = float(r.get("site_lon", ""))
            except Exception:
                continue
            plant_map[str(r["site"])] = {"name": str(r["site"]), "lat": lat, "lon": lon}
        plant_defs_for_map = list(plant_map.values())

    else:
        if ui_rows:
            raw_df = pd.DataFrame(ui_rows)
            raw_df = ensure_optional_columns(raw_df)
            st.info("현재 UI에서 구성한 공급망 데이터를 사용 중입니다.")
            plant_defs_for_map = plant_defs
        else:
            raw_df, plant_defs_for_map = build_default_sample_df()
            raw_df = ensure_optional_columns(raw_df)
            st.info("현재 기본 내장 샘플 데이터를 사용 중입니다.")

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

    st.subheader("입력 공급망 지도")
    input_points = build_node_points(cleaned_df, plant_defs=plant_defs_for_map)
    input_routes = summarize_routes(cleaned_df, optimized=False)

    if isinstance(input_routes, pd.DataFrame) and not input_routes.empty:
        input_map = make_folium_map(input_points, input_routes, result_mode=False)
    else:
        input_map = make_folium_map(input_points, None, result_mode=False)
    st_folium(input_map, width=None, height=650, key="input_supply_chain_map")

# ============================================================
# 6. 탭 2
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
            (pd.to_numeric(cleaned_df["baseline_amount"], errors="coerce").fillna(0) *
             pd.to_numeric(cleaned_df["emission_factor_tco2_per_unit"], errors="coerce").fillna(0)).sum()
        )
        baseline_total_cost = float(
            (pd.to_numeric(cleaned_df["baseline_amount"], errors="coerce").fillna(0) *
             pd.to_numeric(cleaned_df["cost_per_unit"], errors="coerce").fillna(0)).sum()
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
# 7. 탭 3
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

            baseline_scope_view = scope_summary[
                ["scope", "baseline_amount", "baseline_emissions_tco2", "baseline_cost"]
            ].rename(
                columns={
                    "scope": "Scope",
                    "baseline_amount": "Baseline Activity",
                    "baseline_emissions_tco2": "Baseline Emissions (tCO2)",
                    "baseline_cost": "Baseline Cost",
                }
            )

            optimized_scope_view = scope_summary[
                ["scope", "optimized_amount", "optimized_emissions_tco2", "optimized_cost"]
            ].rename(
                columns={
                    "scope": "Scope",
                    "optimized_amount": "Optimized Activity",
                    "optimized_emissions_tco2": "Optimized Emissions (tCO2)",
                    "optimized_cost": "Optimized Cost",
                }
            )

            comparison_scope_view = scope_summary[
                ["scope", "baseline_emissions_tco2", "optimized_emissions_tco2", "reduction_tco2", "reduction_pct"]
            ].rename(
                columns={
                    "scope": "Scope",
                    "baseline_emissions_tco2": "Baseline Emissions (tCO2)",
                    "optimized_emissions_tco2": "Optimized Emissions (tCO2)",
                    "reduction_tco2": "Reduction (tCO2)",
                    "reduction_pct": "Reduction (%)",
                }
            )

            col_b, col_o = st.columns(2)
            with col_b:
                st.markdown("**Baseline 기준**")
                st.dataframe(baseline_scope_view, use_container_width=True)

            with col_o:
                st.markdown("**Optimized 기준**")
                st.dataframe(optimized_scope_view, use_container_width=True)

            st.markdown("**Baseline vs Optimized 비교**")
            st.dataframe(comparison_scope_view, use_container_width=True)

            scope_chart = scope_summary.set_index("scope")[["baseline_emissions_tco2", "optimized_emissions_tco2"]]
            st.bar_chart(scope_chart)

            st.subheader("활동그룹(activity_group)별 분석")
            st.dataframe(activity_summary, use_container_width=True)

            st.subheader("상세 결과")
            st.dataframe(detail, use_container_width=True)

            st.subheader("공급망 결과 지도")
            plant_map = {}
            for _, r in detail[detail["site_lat"].astype(str).str.strip() != ""].iterrows():
                try:
                    plant_map[r["site"]] = {
                        "name": r["site"],
                        "lat": float(r["site_lat"]),
                        "lon": float(r["site_lon"]),
                    }
                except Exception:
                    pass
            result_plant_defs = list(plant_map.values())

            result_points = build_node_points(cleaned_df, result_df=detail, plant_defs=result_plant_defs)
            result_routes = summarize_routes(detail, optimized=True)

            if isinstance(result_routes, pd.DataFrame) and not result_routes.empty:
                baseline_route_summary = summarize_routes(detail, optimized=False)[["route_id", "emissions_tco2"]].rename(
                    columns={"emissions_tco2": "baseline_route_emissions"}
                )
                result_routes = result_routes.merge(baseline_route_summary, on="route_id", how="left")
                result_routes["reduction_tco2"] = result_routes["baseline_route_emissions"] - result_routes["emissions_tco2"]

                result_map = make_folium_map(result_points, result_routes, result_mode=True)
                st_folium(result_map, width=None, height=700, key="result_supply_chain_map")
            else:
                st.info("지도에 표시할 경로가 없습니다.")

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
