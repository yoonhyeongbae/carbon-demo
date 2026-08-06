from __future__ import annotations

"""
Flexible EV supply-chain LP (v10.0)

Key additions
-------------
1. Dynamic product structure based on item_catalog.csv + product_bom.csv.
2. Every model-data CSV is supplied by the user; no model dataset is embedded in app.py.
3. Users define a new raw material/intermediate item in the data tab, upload its BOM/supplier/process CSVs, then activate it by checkbox.
4. Checkboxes include/exclude raw materials and intermediate goods from the optimization model.
5. Stage-separated maps: Stage 1 production, Stage 2 inbound/assembly, Stage 3 France market.

Run:
    streamlit run app.py

Dependencies:
    streamlit pandas numpy geopy ortools folium streamlit-folium altair
"""

import gc
import hashlib
import io
import json
import math
import re
import time
import zipfile
from array import array
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from ortools.linear_solver import pywraplp

try:
    import streamlit as st
except Exception:
    st = None


APP_BUILD = "external-csv-flexible-product-structure-stage-maps-v10.0"
APP_PACKAGE_ID = "20260806-v10.0-external-csv-custom-items-stage-maps"
SESSION_TABLES_KEY = "user_tables_v10"
SESSION_RESULTS_KEY = "optimization_results_v10"

REQUIRED_FILES = [
    "products.csv",
    "product_bom.csv",
    "item_catalog.csv",
    "item_suppliers.csv",
    "stage2_item_processes.csv",
    "demand.csv",
    "assembly_locations.csv",
    "transport_parameters.csv",
    "country_transport_rules.csv",
    "markets.csv",
    "scenarios.csv",
    "model_metadata.csv",
]

ADDON_BOM_COLUMNS = {
    "product_id", "quantity_per_vehicle", "quantity_unit", "mass_per_unit_kg"
}
ADDON_SUPPLIER_COLUMNS = {
    "supplier_id", "location_index", "location_name", "continent", "latitude", "longitude",
    "stage1_cost_per_unit", "stage1_ef_kgco2_per_unit", "parameter_unit", "capacity",
    "capacity_unit", "active_default"
}
ADDON_PROCESS_COLUMNS = {
    "location_index", "location_name", "process_name_ko", "process_cost_per_unit",
    "process_ef_kgco2_per_unit", "parameter_unit", "active_default"
}

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
TRANSPORT_DASH = {
    1: None,
    2: "11,6",
    3: "4,7",
    4: "12,5,3,5",
    5: "1,6",
    6: "1,4,9,4",
}
FINISHED_COLOR = "#6a3d9a"
ASSEMBLY_COLOR = "#31a354"
MARKET_COLOR = "orange"
INTERNAL_ROUTE_INDEX = 0
INTERNATIONAL_INLAND_LEG_KM = 50.0
FLOW_TOL = 1e-7
OBJECTIVE_TARGET_MAX_COEFFICIENT = 1_000_000.0
GLOP_PARAMETER_TEXT = """
use_scaling: true
primal_feasibility_tolerance: 1e-8
dual_feasibility_tolerance: 1e-9
solution_feasibility_tolerance: 1e-7
"""
MODE_LABEL = {"line": "라인 생산", "modular": "모듈 활용 분산 생산"}
SCENARIO_SHORT = {"S1": "시나리오 ①", "S2": "시나리오 ②", "S3": "시나리오 ③"}
SCENARIO_POLICY_SCORE = {"S1": None, "S2": 60.0, "S3": 65.0}


# -----------------------------------------------------------------------------
# Data loading, upload override, and validation
# -----------------------------------------------------------------------------
def _csv_from_bytes(data: bytes, filename: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"{filename} 읽기 실패: {exc}") from exc


def parse_full_data_uploads(uploaded_files: Optional[Sequence]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Read required CSVs from individual uploads or ZIP packages.

    Only files whose basename exactly matches REQUIRED_FILES are loaded. A ZIP may contain
    subfolders; basenames are used. Later uploads replace earlier files with the same name.
    """
    tables: Dict[str, pd.DataFrame] = {}
    messages: List[str] = []
    if not uploaded_files:
        return tables, messages
    for uploaded in uploaded_files:
        filename = str(getattr(uploaded, "name", "")).split("/")[-1]
        try:
            uploaded.seek(0)
            raw = uploaded.read()
            if filename.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
                    for member in archive.namelist():
                        basename = member.replace("\\", "/").split("/")[-1]
                        if basename in REQUIRED_FILES and not member.endswith("/"):
                            tables[basename] = _csv_from_bytes(archive.read(member), basename)
                            messages.append(f"ZIP에서 읽음: {basename}")
            elif filename in REQUIRED_FILES:
                tables[filename] = _csv_from_bytes(raw, filename)
                messages.append(f"CSV에서 읽음: {filename}")
            else:
                messages.append(f"무시됨: {filename} — 필수 CSV 이름 또는 ZIP이 아닙니다.")
        except Exception as exc:
            messages.append(f"오류: {filename} — {exc}")
    return tables, messages


def read_single_csv(uploaded, label: str) -> Optional[pd.DataFrame]:
    if uploaded is None:
        return None
    uploaded.seek(0)
    return _csv_from_bytes(uploaded.read(), label)


def make_tables_zip(tables: Mapping[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_FILES:
            if name in tables:
                archive.writestr(name, tables[name].to_csv(index=False).encode("utf-8-sig"))
        archive.writestr(
            "README_CURRENT_SESSION.txt",
            (
                "This ZIP contains the CSV tables currently loaded in the Streamlit session.\n"
                "No model dataset is embedded in app.py. Upload all required CSV files or a ZIP at the next session.\n"
            ).encode("utf-8"),
        )
    return output.getvalue()


def _upsert_frame(base: pd.DataFrame, incoming: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if incoming.empty:
        return base.copy()
    missing = [key for key in keys if key not in incoming.columns]
    if missing:
        raise ValueError(f"병합키 누락: {missing}")
    if base.empty:
        return incoming.copy().reset_index(drop=True)
    key_tuples = set(tuple(str(row[key]) for key in keys) for _, row in incoming.iterrows())
    keep_mask = [tuple(str(row[key]) for key in keys) not in key_tuples for _, row in base.iterrows()]
    return pd.concat([base.loc[keep_mask], incoming], ignore_index=True, sort=False)


def _save_tables_to_session(tables: Mapping[str, pd.DataFrame]) -> None:
    st.session_state[SESSION_TABLES_KEY] = {name: frame.copy() for name, frame in tables.items()}
    st.session_state.pop(SESSION_RESULTS_KEY, None)


def _normalized_item_id(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", str(value).strip().lower()).strip("_")
    if not cleaned:
        raise ValueError("item_id는 영문 소문자, 숫자, 밑줄을 포함해야 합니다.")
    return cleaned


def item_data_status(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if not tables or "item_catalog.csv" not in tables:
        return pd.DataFrame()
    catalog = tables["item_catalog.csv"].copy()
    products = tables.get("products.csv", pd.DataFrame())
    bom = tables.get("product_bom.csv", pd.DataFrame())
    suppliers = tables.get("item_suppliers.csv", pd.DataFrame())
    processes = tables.get("stage2_item_processes.csv", pd.DataFrame())
    plants = tables.get("assembly_locations.csv", pd.DataFrame())
    product_count = len(products)
    location_count = len(plants)
    rows = []
    for _, item in catalog.sort_values("item_index").iterrows():
        item_id = str(item["item_id"])
        bom_rows = bom[bom.get("item_id", pd.Series(dtype=str)).astype(str) == item_id] if "item_id" in bom else pd.DataFrame()
        supplier_rows = suppliers[suppliers.get("item_id", pd.Series(dtype=str)).astype(str) == item_id] if "item_id" in suppliers else pd.DataFrame()
        process_rows = processes[processes.get("item_id", pd.Series(dtype=str)).astype(str) == item_id] if "item_id" in processes else pd.DataFrame()
        needs_process = int(item.get("has_stage2_process", 0)) == 1
        ready = (
            bom_rows["product_id"].astype(str).nunique() == product_count
            and not supplier_rows.empty
            and (not needs_process or process_rows["location_index"].astype(int).nunique() == location_count)
        )
        rows.append({
            "item_id": item_id,
            "품목명": str(item.get("item_name_ko", item_id)),
            "유형": str(item.get("item_type", "")),
            "BOM 제품수": bom_rows["product_id"].astype(str).nunique() if not bom_rows.empty else 0,
            "필요 제품수": product_count,
            "생산지 행수": len(supplier_rows),
            "Stage 2 공정 행수": len(process_rows),
            "필요 공정 행수": location_count if needs_process else 0,
            "최적화 사용 준비": ready,
        })
    return pd.DataFrame(rows)


def apply_item_addon(
    tables: Mapping[str, pd.DataFrame],
    item_id: str,
    bom_addon: pd.DataFrame,
    supplier_addon: pd.DataFrame,
    process_addon: Optional[pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    out = {name: frame.copy() for name, frame in tables.items()}
    catalog = out["item_catalog.csv"]
    match = catalog[catalog["item_id"].astype(str) == str(item_id)]
    if len(match) != 1:
        raise ValueError(f"item_catalog.csv에서 item_id={item_id}를 정확히 한 행으로 먼저 추가해야 합니다.")
    item_index = int(match.iloc[0]["item_index"])
    needs_process = int(match.iloc[0]["has_stage2_process"]) == 1

    missing = ADDON_BOM_COLUMNS - set(bom_addon.columns)
    if missing:
        raise ValueError(f"제품 BOM 추가 CSV 누락 열: {sorted(missing)}")
    missing = ADDON_SUPPLIER_COLUMNS - set(supplier_addon.columns)
    if missing:
        raise ValueError(f"생산지 추가 CSV 누락 열: {sorted(missing)}")
    if needs_process and process_addon is None:
        raise ValueError("이 품목은 Stage 2 추가 공정이 있으므로 공정 CSV가 필요합니다.")
    if process_addon is not None:
        missing = ADDON_PROCESS_COLUMNS - set(process_addon.columns)
        if missing:
            raise ValueError(f"Stage 2 공정 추가 CSV 누락 열: {sorted(missing)}")

    bom_in = bom_addon.copy()
    bom_in["item_id"] = item_id
    if "source_basis" not in bom_in:
        bom_in["source_basis"] = "user item add-on CSV"
    bom_cols = list(out["product_bom.csv"].columns)
    for col in bom_cols:
        if col not in bom_in:
            bom_in[col] = np.nan
    bom_in = bom_in[bom_cols]

    supplier_in = supplier_addon.copy()
    supplier_in["item_id"] = item_id
    supplier_in["item_index"] = item_index
    if "source_basis" not in supplier_in:
        supplier_in["source_basis"] = "user item add-on CSV"
    supplier_cols = list(out["item_suppliers.csv"].columns)
    for col in supplier_cols:
        if col not in supplier_in:
            supplier_in[col] = np.nan
    supplier_in = supplier_in[supplier_cols]

    out["product_bom.csv"] = _upsert_frame(out["product_bom.csv"], bom_in, ["product_id", "item_id"])
    out["item_suppliers.csv"] = _upsert_frame(out["item_suppliers.csv"], supplier_in, ["item_id", "location_index"])

    if process_addon is not None:
        process_in = process_addon.copy()
        process_in["item_id"] = item_id
        if "source_basis" not in process_in:
            process_in["source_basis"] = "user item add-on CSV"
        process_cols = list(out["stage2_item_processes.csv"].columns)
        for col in process_cols:
            if col not in process_in:
                process_in[col] = np.nan
        process_in = process_in[process_cols]
        out["stage2_item_processes.csv"] = _upsert_frame(
            out["stage2_item_processes.csv"], process_in, ["item_id", "location_index"]
        )
    return out


def delete_custom_item(tables: Mapping[str, pd.DataFrame], item_id: str) -> Dict[str, pd.DataFrame]:
    out = {name: frame.copy() for name, frame in tables.items()}
    catalog = out["item_catalog.csv"]
    row = catalog[catalog["item_id"].astype(str) == str(item_id)]
    if row.empty:
        return out
    if int(row.iloc[0].get("mandatory", 0)) == 1:
        raise ValueError("필수 품목은 삭제할 수 없습니다.")
    out["item_catalog.csv"] = catalog[catalog["item_id"].astype(str) != str(item_id)].reset_index(drop=True)
    for filename in ["product_bom.csv", "item_suppliers.csv", "stage2_item_processes.csv"]:
        frame = out[filename]
        if "item_id" in frame:
            out[filename] = frame[frame["item_id"].astype(str) != str(item_id)].reset_index(drop=True)
    return out


def validate_tables(tables: Mapping[str, pd.DataFrame]) -> List[str]:
    errors: List[str] = []
    missing = [name for name in REQUIRED_FILES if name not in tables]
    if missing:
        return ["필수 CSV 누락: " + ", ".join(missing)]

    required_columns = {
        "products.csv": {
            "product_index", "product_id", "vehicle_class", "trim", "product_name_ko",
            "battery_kwh", "battery_mass_kg", "reference_vehicle_mass_kg",
        },
        "product_bom.csv": {
            "product_id", "item_id", "quantity_per_vehicle", "quantity_unit", "mass_per_unit_kg",
        },
        "item_catalog.csv": {
            "item_index", "item_id", "item_name_ko", "item_type", "flow_unit",
            "default_enabled", "mandatory", "has_stage2_process", "stage1_process_name_ko",
            "stage2_process_name_ko", "color_hex", "loss_rate",
        },
        "item_suppliers.csv": {
            "item_id", "item_index", "supplier_id", "location_index", "location_name",
            "continent", "latitude", "longitude", "stage1_cost_per_unit",
            "stage1_ef_kgco2_per_unit", "parameter_unit", "capacity", "capacity_unit",
            "active_default",
        },
        "stage2_item_processes.csv": {
            "item_id", "location_index", "location_name", "process_name_ko",
            "process_cost_per_unit", "process_ef_kgco2_per_unit", "parameter_unit",
            "active_default",
        },
        "demand.csv": {"product_id", "market_id", "demand_units"},
        "assembly_locations.csv": {
            "location_index", "plant_id", "location_name", "continent", "latitude", "longitude",
            "assembly_ef_kgco2_per_kg", "assembly_cost_eur_per_kg",
        },
        "transport_parameters.csv": {
            "transport_mode", "transport_mode_ko", "region_class",
            "transport_cost_eur_per_kgkm", "transport_ef_kgco2_per_kgkm",
        },
        "country_transport_rules.csv": {
            "location_name", "continent", "road_region_class", "rail_region_class",
            "allow_road", "allow_rail", "allow_sea", "allow_air",
        },
        "markets.csv": {
            "market_id", "market_name", "location_name", "continent", "latitude", "longitude",
            "minimum_distance_km",
        },
        "scenarios.csv": {
            "scenario_id", "scenario_name", "minimum_score", "apply_carbon_cap",
            "small_cap_kgco2_per_vehicle", "standard_cap_kgco2_per_vehicle",
        },
        "model_metadata.csv": {"parameter_name", "parameter_value", "unit", "description"},
    }
    for filename, columns in required_columns.items():
        missing_columns = columns - set(tables[filename].columns)
        if missing_columns:
            errors.append(f"{filename}: 누락 열 {sorted(missing_columns)}")
    if errors:
        return errors

    products = tables["products.csv"].copy()
    catalog = tables["item_catalog.csv"].copy()
    bom = tables["product_bom.csv"].copy()
    suppliers = tables["item_suppliers.csv"].copy()
    processes = tables["stage2_item_processes.csv"].copy()
    plants = tables["assembly_locations.csv"].copy()

    if products["product_id"].astype(str).duplicated().any():
        errors.append("products.csv: product_id는 중복될 수 없습니다.")
    if catalog["item_id"].astype(str).duplicated().any():
        errors.append("item_catalog.csv: item_id는 중복될 수 없습니다.")
    if catalog["item_index"].astype(int).duplicated().any():
        errors.append("item_catalog.csv: item_index는 중복될 수 없습니다.")
    if plants["location_index"].astype(int).duplicated().any():
        errors.append("assembly_locations.csv: location_index는 중복될 수 없습니다.")

    catalog_ids = set(catalog["item_id"].astype(str))
    product_ids = set(products["product_id"].astype(str))
    if not set(bom["item_id"].astype(str)).issubset(catalog_ids):
        errors.append("product_bom.csv: item_catalog.csv에 없는 item_id가 있습니다.")
    if not set(bom["product_id"].astype(str)).issubset(product_ids):
        errors.append("product_bom.csv: products.csv에 없는 product_id가 있습니다.")
    if not set(suppliers["item_id"].astype(str)).issubset(catalog_ids):
        errors.append("item_suppliers.csv: item_catalog.csv에 없는 item_id가 있습니다.")
    if not set(processes["item_id"].astype(str)).issubset(catalog_ids):
        errors.append("stage2_item_processes.csv: item_catalog.csv에 없는 item_id가 있습니다.")

    numeric_checks = [
        (bom, "quantity_per_vehicle", 0.0, None, "product_bom.csv"),
        (bom, "mass_per_unit_kg", 0.0, None, "product_bom.csv"),
        (catalog, "loss_rate", 0.0, 1.0, "item_catalog.csv"),
        (suppliers, "stage1_cost_per_unit", 0.0, None, "item_suppliers.csv"),
        (suppliers, "stage1_ef_kgco2_per_unit", 0.0, None, "item_suppliers.csv"),
        (suppliers, "capacity", 0.0, None, "item_suppliers.csv"),
        (processes, "process_cost_per_unit", 0.0, None, "stage2_item_processes.csv"),
        (processes, "process_ef_kgco2_per_unit", 0.0, None, "stage2_item_processes.csv"),
    ]
    for frame, col, low, high, filename in numeric_checks:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any() or (values < low).any() or (high is not None and (values >= high).any()):
            upper_text = "" if high is None else f" 및 {high} 미만"
            errors.append(f"{filename}.{col}: {low} 이상{upper_text}의 숫자여야 합니다.")

    for item_id in catalog_ids:
        item_bom = bom[bom["item_id"].astype(str) == item_id]
        if set(item_bom["product_id"].astype(str)) != product_ids:
            errors.append(f"product_bom.csv: {item_id}는 모든 제품에 대해 정확히 한 행이 필요합니다.")
        item_sup = suppliers[suppliers["item_id"].astype(str) == item_id]
        if item_sup.empty:
            errors.append(f"item_suppliers.csv: {item_id} 공급자 데이터가 없습니다.")

    location_ids = set(plants["location_index"].astype(int))
    bad_supplier_locations = set(suppliers["location_index"].astype(int)) - location_ids
    if bad_supplier_locations:
        errors.append(f"item_suppliers.csv: 조립지 목록에 없는 location_index {sorted(bad_supplier_locations)}")

    process_items = set(
        catalog.loc[pd.to_numeric(catalog["has_stage2_process"], errors="coerce").fillna(0).astype(int) == 1, "item_id"].astype(str)
    )
    for item_id in process_items:
        item_proc = processes[processes["item_id"].astype(str) == item_id]
        if set(item_proc["location_index"].astype(int)) != location_ids:
            errors.append(
                f"stage2_item_processes.csv: {item_id}는 모든 조립지에 대한 공정계수가 필요합니다."
            )

    battery_items = catalog[catalog["item_type"].astype(str) == "battery"]
    if len(battery_items) != 1:
        errors.append("item_catalog.csv: item_type=battery인 항목은 정확히 1개여야 합니다.")
    elif int(battery_items.iloc[0]["mandatory"]) != 1:
        errors.append("item_catalog.csv: 배터리는 mandatory=1이어야 합니다.")

    rules = tables["country_transport_rules.csv"]
    for col in ["allow_road", "allow_rail", "allow_sea", "allow_air"]:
        values = pd.to_numeric(rules[col], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            errors.append(f"country_transport_rules.csv.{col}: 0 또는 1만 허용됩니다.")

    return errors


def ordered_tables(tables: Mapping[str, pd.DataFrame]):
    return (
        tables["products.csv"].sort_values("product_index").reset_index(drop=True),
        tables["product_bom.csv"].copy(),
        tables["item_catalog.csv"].sort_values("item_index").reset_index(drop=True),
        tables["item_suppliers.csv"].sort_values(["item_index", "location_index"]).reset_index(drop=True),
        tables["stage2_item_processes.csv"].copy(),
        tables["demand.csv"].copy(),
        tables["assembly_locations.csv"].sort_values("location_index").reset_index(drop=True),
        tables["transport_parameters.csv"].copy().reset_index(drop=True),
        tables["country_transport_rules.csv"].copy().reset_index(drop=True),
        tables["markets.csv"].copy(),
        tables["scenarios.csv"].copy(),
    )


# -----------------------------------------------------------------------------
# Distance and route coefficients
# -----------------------------------------------------------------------------
def rounded_distance_km(lat1: float, lon1: float, lat2: float, lon2: float, minimum: float = 0.0) -> float:
    value = geodesic((float(lat1), float(lon1)), (float(lat2), float(lon2))).km
    return round(max(float(minimum), float(value)), 2)


@lru_cache(maxsize=8)
def cached_distance_matrices(
    coordinates: Tuple[Tuple[float, float], ...],
    market_coordinate: Tuple[float, float],
    minimum_market_distance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(coordinates)
    raw = np.zeros((n, n), dtype=float)
    for o, (lat_o, lon_o) in enumerate(coordinates):
        for a, (lat_a, lon_a) in enumerate(coordinates):
            raw[o, a] = rounded_distance_km(lat_o, lon_o, lat_a, lon_a)
    final = np.zeros(n, dtype=float)
    for a, (lat_a, lon_a) in enumerate(coordinates):
        final[a] = rounded_distance_km(
            lat_a, lon_a, market_coordinate[0], market_coordinate[1], minimum_market_distance
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
    if same_location:
        return route_code == "road"
    if route_code == "road":
        return bool(origin["allow_road"] and destination["allow_road"] and (same_continent or both_europe))
    if route_code == "rail":
        return bool(origin["allow_rail"] and destination["allow_rail"] and (same_continent or both_europe))
    if both_europe:
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
    d = float(distance_km)
    if str(origin["location_name"]) == str(destination["location_name"]):
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if route_code in {"road", "rail"}:
        region_field = f"{route_code}_region_class"
        unit_cost = 0.5 * (
            _factor_value(factor_cost, route_code, str(origin[region_field]))
            + _factor_value(factor_cost, route_code, str(destination[region_field]))
        )
        unit_ef = 0.5 * (
            _factor_value(factor_ef, route_code, str(origin[region_field]))
            + _factor_value(factor_ef, route_code, str(destination[region_field]))
        )
        return d * unit_cost, d * unit_ef, d, d, 0.0
    international_mode, land_mode = route_code.split("_")
    inland_each = min(INTERNATIONAL_INLAND_LEG_KM, d / 4.0)
    inland_total = 2.0 * inland_each
    international_distance = max(0.0, d - inland_total)
    region_field = f"{land_mode}_region_class"
    land_cost = inland_each * (
        _factor_value(factor_cost, land_mode, str(origin[region_field]))
        + _factor_value(factor_cost, land_mode, str(destination[region_field]))
    )
    land_ef = inland_each * (
        _factor_value(factor_ef, land_mode, str(origin[region_field]))
        + _factor_value(factor_ef, land_mode, str(destination[region_field]))
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
    n = len(plants)
    t_count = len(ROUTE_MODE_CODES)
    raw_allowed = np.zeros((n, n, t_count), dtype=bool)
    raw_cost = np.zeros((n, n, t_count), dtype=float)
    raw_ef = np.zeros((n, n, t_count), dtype=float)
    raw_total = np.zeros((n, n, t_count), dtype=float)
    raw_inland = np.zeros((n, n, t_count), dtype=float)
    raw_international = np.zeros((n, n, t_count), dtype=float)
    for o in range(n):
        origin = rules[str(plants.iloc[o]["location_name"])]
        for a in range(n):
            destination = rules[str(plants.iloc[a]["location_name"])]
            for t, code in enumerate(ROUTE_MODE_CODES):
                allowed = _route_allowed(origin, destination, code)
                raw_allowed[o, a, t] = allowed
                if allowed:
                    values = _route_coefficient(raw_distances[o, a], origin, destination, code, factor_cost, factor_ef)
                    raw_cost[o, a, t], raw_ef[o, a, t], raw_total[o, a, t], raw_inland[o, a, t], raw_international[o, a, t] = values

    final_allowed = np.zeros((n, t_count), dtype=bool)
    final_cost = np.zeros((n, t_count), dtype=float)
    final_ef = np.zeros((n, t_count), dtype=float)
    final_total = np.zeros((n, t_count), dtype=float)
    final_inland = np.zeros((n, t_count), dtype=float)
    final_international = np.zeros((n, t_count), dtype=float)
    destination = rules[str(market["location_name"])]
    for a in range(n):
        origin = rules[str(plants.iloc[a]["location_name"])]
        for t, code in enumerate(ROUTE_MODE_CODES):
            allowed = _route_allowed(origin, destination, code)
            final_allowed[a, t] = allowed
            if allowed:
                values = _route_coefficient(final_distances[a], origin, destination, code, factor_cost, factor_ef)
                final_cost[a, t], final_ef[a, t], final_total[a, t], final_inland[a, t], final_international[a, t] = values
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


# -----------------------------------------------------------------------------
# Dynamic LP layout and solver
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class IndexLayout:
    mode: str
    F: int
    R: int
    O: int
    A: int
    T: int

    def __post_init__(self):
        if self.mode not in {"line", "modular"}:
            raise ValueError(self.mode)

    @property
    def off_p(self) -> int:
        return 0

    @property
    def n_p(self) -> int:
        return self.F * self.R * self.O

    @property
    def off_tin(self) -> int:
        return self.off_p + self.n_p

    @property
    def n_tin(self) -> int:
        return self.F * self.R * self.O * self.A * self.T

    @property
    def off_a(self) -> int:
        return self.off_tin + self.n_tin

    @property
    def n_a(self) -> int:
        return self.F * self.A

    @property
    def off_tmarket(self) -> int:
        return self.off_a + self.n_a

    @property
    def n_tmarket(self) -> int:
        return self.F * self.A * self.T

    @property
    def off_m10(self) -> int:
        return self.off_tmarket + self.n_tmarket

    @property
    def n_m10(self) -> int:
        return self.F * self.O * self.A if self.mode == "modular" else 0

    @property
    def off_m5(self) -> int:
        return self.off_m10 + self.n_m10

    @property
    def n_m5(self) -> int:
        return self.F * self.O * self.A if self.mode == "modular" else 0

    @property
    def n_vars(self) -> int:
        return self.off_m5 + self.n_m5

    def p(self, v: int, r: int, o: int) -> int:
        return self.off_p + ((v * self.R + r) * self.O + o)

    def tin(self, v: int, r: int, o: int, a: int, t: int) -> int:
        return self.off_tin + ((((v * self.R + r) * self.O + o) * self.A + a) * self.T + t)

    def assembly(self, v: int, a: int) -> int:
        return self.off_a + v * self.A + a

    def tmarket(self, v: int, a: int, t: int) -> int:
        return self.off_tmarket + (v * self.A + a) * self.T + t

    def m10(self, v: int, o: int, a: int) -> int:
        if self.mode != "modular":
            raise ValueError("m10 exists only in modular mode")
        return self.off_m10 + (v * self.O + o) * self.A + a

    def m5(self, v: int, o: int, a: int) -> int:
        if self.mode != "modular":
            raise ValueError("m5 exists only in modular mode")
        return self.off_m5 + (v * self.O + o) * self.A + a


class LinearConstraintBuilder:
    def __init__(self, n_vars: int):
        self.n_vars = int(n_vars)
        self.eq_cols = array("I")
        self.eq_data = array("d")
        self.eq_starts = array("I", [0])
        self.eq_rhs = array("d")
        self.ub_cols = array("I")
        self.ub_data = array("d")
        self.ub_starts = array("I", [0])
        self.ub_rhs = array("d")

    def add_eq(self, cols: Sequence[int], vals: Sequence[float], rhs: float):
        self.eq_cols.extend(int(c) for c in cols)
        self.eq_data.extend(float(v) for v in vals)
        self.eq_rhs.append(float(rhs))
        self.eq_starts.append(len(self.eq_cols))

    def add_le(self, cols: Sequence[int], vals: Sequence[float], rhs: float):
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

    def clear_coefficients(self):
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
class FlexibleLPModel:
    layout: IndexLayout
    c: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    rows: LinearConstraintBuilder
    products: pd.DataFrame
    items: pd.DataFrame
    bom: pd.DataFrame
    suppliers: pd.DataFrame
    processes: pd.DataFrame
    plants: pd.DataFrame
    market: pd.Series
    scenario: pd.Series
    demand_values: np.ndarray
    item_ids: Tuple[str, ...]
    item_index: Dict[str, int]
    battery_r: int
    bom_quantity: np.ndarray
    mass_per_unit: np.ndarray
    selected_vehicle_mass: np.ndarray
    nonbattery_mass: np.ndarray
    supplier_cost: np.ndarray
    supplier_ef: np.ndarray
    supplier_capacity: np.ndarray
    process_cost: np.ndarray
    process_ef: np.ndarray
    route: Dict[str, np.ndarray]
    emission_p: np.ndarray
    emission_tin: np.ndarray
    emission_a: np.ndarray
    emission_tmarket: np.ndarray
    selected_country_map: Dict[str, Tuple[str, ...]]
    structure_signature: str
    equality_count: int
    inequality_count: int
    matrix_nonzeros: int


@dataclass
class SolveResult:
    status: str
    message: str
    objective_value: Optional[float]
    wall_time_sec: float
    x: Optional[np.ndarray]
    model: FlexibleLPModel
    solver_name: str
    solver_version: str
    iterations: int


def _normalize_active_items(catalog: pd.DataFrame, active_item_ids: Optional[Sequence[str]]) -> List[str]:
    ordered = catalog.sort_values("item_index")["item_id"].astype(str).tolist()
    mandatory = set(catalog.loc[pd.to_numeric(catalog["mandatory"], errors="coerce").fillna(0).astype(int) == 1, "item_id"].astype(str))
    if active_item_ids is None:
        selected = set(catalog.loc[pd.to_numeric(catalog["default_enabled"], errors="coerce").fillna(0).astype(int) == 1, "item_id"].astype(str))
    else:
        selected = {str(v) for v in active_item_ids}
    selected |= mandatory
    return [item for item in ordered if item in selected]


def _normalize_country_map(
    active_items: Sequence[str],
    suppliers: pd.DataFrame,
    plants: pd.DataFrame,
    selected_country_map: Optional[Mapping[str, Sequence[str]]],
) -> Dict[str, Tuple[str, ...]]:
    all_plants = tuple(plants["location_name"].astype(str).tolist())
    result: Dict[str, Tuple[str, ...]] = {}
    for item_id in active_items:
        available = tuple(
            suppliers.loc[
                (suppliers["item_id"].astype(str) == item_id)
                & (pd.to_numeric(suppliers["active_default"], errors="coerce").fillna(0).astype(int) == 1),
                "location_name",
            ].astype(str).tolist()
        )
        requested = None if selected_country_map is None else selected_country_map.get(item_id)
        chosen = tuple(name for name in available if requested is None or name in {str(v) for v in requested})
        if not chosen:
            raise ValueError(f"{item_id}: 허용 생산지를 최소 1개 선택해야 합니다.")
        result[item_id] = chosen
    requested_assembly = None if selected_country_map is None else selected_country_map.get("assembly")
    result["assembly"] = tuple(
        name for name in all_plants if requested_assembly is None or name in {str(v) for v in requested_assembly}
    )
    if not result["assembly"]:
        raise ValueError("조립지를 최소 1개 선택해야 합니다.")
    return result


def _structure_signature(active_items: Sequence[str], selected_country_map: Mapping[str, Sequence[str]]) -> str:
    payload = json.dumps(
        {"items": list(active_items), "countries": {k: list(v) for k, v in selected_country_map.items()}},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_flexible_lp_model(
    tables: Mapping[str, pd.DataFrame],
    production_mode: str,
    scenario_id: str,
    active_item_ids: Optional[Sequence[str]] = None,
    selected_country_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> FlexibleLPModel:
    (
        products, bom_all, catalog_all, suppliers_all, processes_all, demand, plants,
        transport, country_rules, markets, scenarios,
    ) = ordered_tables(tables)

    active_items = _normalize_active_items(catalog_all, active_item_ids)
    items = catalog_all[catalog_all["item_id"].astype(str).isin(active_items)].copy()
    items = items.sort_values("item_index").reset_index(drop=True)
    item_ids = tuple(items["item_id"].astype(str).tolist())
    item_index = {item_id: i for i, item_id in enumerate(item_ids)}
    battery_ids = items.loc[items["item_type"].astype(str) == "battery", "item_id"].astype(str).tolist()
    if len(battery_ids) != 1:
        raise ValueError("활성 제품구조에는 배터리 항목이 정확히 1개 있어야 합니다.")
    battery_id = battery_ids[0]
    battery_r = item_index[battery_id]

    bom = bom_all[bom_all["item_id"].astype(str).isin(active_items)].copy()
    suppliers = suppliers_all[suppliers_all["item_id"].astype(str).isin(active_items)].copy()
    processes = processes_all[processes_all["item_id"].astype(str).isin(active_items)].copy()
    selected_map = _normalize_country_map(active_items, suppliers, plants, selected_country_map)

    F, R, O, A, T = len(products), len(items), len(plants), len(plants), len(ROUTE_MODE_CODES)
    layout = IndexLayout(production_mode, F, R, O, A, T)
    product_ids = products["product_id"].astype(str).tolist()
    demand_map = demand.groupby("product_id")["demand_units"].sum().to_dict()
    demand_values = np.asarray([float(demand_map[pid]) for pid in product_ids], dtype=float)

    bom_quantity = np.zeros((F, R), dtype=float)
    mass_per_unit = np.zeros((F, R), dtype=float)
    for v, product_id in enumerate(product_ids):
        for r, item_id in enumerate(item_ids):
            row = bom[(bom["product_id"].astype(str) == product_id) & (bom["item_id"].astype(str) == item_id)]
            if len(row) != 1:
                raise ValueError(f"product_bom.csv: product={product_id}, item={item_id} 행이 정확히 1개여야 합니다.")
            bom_quantity[v, r] = float(row.iloc[0]["quantity_per_vehicle"])
            mass_per_unit[v, r] = float(row.iloc[0]["mass_per_unit_kg"])

    selected_vehicle_mass = np.sum(bom_quantity * mass_per_unit, axis=1)
    nonbattery_mask = np.asarray([str(v) != "battery" for v in items["item_type"].astype(str)], dtype=bool)
    nonbattery_mass = np.sum((bom_quantity * mass_per_unit)[:, nonbattery_mask], axis=1)

    supplier_cost = np.zeros((R, O), dtype=float)
    supplier_ef = np.zeros((R, O), dtype=float)
    supplier_capacity = np.zeros((R, O), dtype=float)
    active_supplier = np.zeros((R, O), dtype=bool)
    location_names = plants["location_name"].astype(str).tolist()
    for r, item_id in enumerate(item_ids):
        chosen = set(selected_map[item_id])
        for o, location_name in enumerate(location_names):
            row = suppliers[(suppliers["item_id"].astype(str) == item_id) & (suppliers["location_index"].astype(int) == o + 1)]
            if row.empty:
                continue
            supplier_cost[r, o] = float(row.iloc[0]["stage1_cost_per_unit"])
            supplier_ef[r, o] = float(row.iloc[0]["stage1_ef_kgco2_per_unit"])
            supplier_capacity[r, o] = float(row.iloc[0]["capacity"])
            active_supplier[r, o] = location_name in chosen

    process_cost = np.zeros((R, A), dtype=float)
    process_ef = np.zeros((R, A), dtype=float)
    for r, item in items.iterrows():
        if int(item["has_stage2_process"]) != 1:
            continue
        item_id = str(item["item_id"])
        for a in range(A):
            row = processes[(processes["item_id"].astype(str) == item_id) & (processes["location_index"].astype(int) == a + 1)]
            if row.empty:
                raise ValueError(f"stage2_item_processes.csv: {item_id}, location_index={a+1} 누락")
            process_cost[r, a] = float(row.iloc[0]["process_cost_per_unit"])
            process_ef[r, a] = float(row.iloc[0]["process_ef_kgco2_per_unit"])

    active_assembly = np.asarray([name in set(selected_map["assembly"]) for name in location_names], dtype=bool)
    market = markets.iloc[0]
    coordinates = tuple((float(row["latitude"]), float(row["longitude"])) for _, row in plants.iterrows())
    raw_distances, final_distances = cached_distance_matrices(
        coordinates,
        (float(market["latitude"]), float(market["longitude"])),
        float(market.get("minimum_distance_km", 0.0)),
    )
    route = build_route_matrices(plants, market, transport, country_rules, raw_distances, final_distances)

    scenario_rows = scenarios[scenarios["scenario_id"].astype(str) == str(scenario_id)]
    if scenario_rows.empty:
        raise ValueError(f"scenario not found: {scenario_id}")
    scenario = scenario_rows.iloc[0].copy()
    score = float(scenario.get("minimum_score", 0.0))
    if int(scenario.get("apply_carbon_cap", 0)) == 1:
        fleet_cap = sum(
            carbon_cap_from_score(str(products.iloc[v]["vehicle_class"]), score) * demand_values[v]
            for v in range(F)
        )
    else:
        fleet_cap = np.nan
    scenario["fleet_total_cap_kgco2"] = fleet_cap

    c = np.zeros(layout.n_vars, dtype=float)
    lb = np.zeros(layout.n_vars, dtype=float)
    ub = np.full(layout.n_vars, np.inf, dtype=float)
    emission_p = np.zeros(layout.n_p, dtype=float)
    emission_tin = np.zeros(layout.n_tin, dtype=float)
    emission_a = np.zeros(layout.n_a, dtype=float)
    emission_tmarket = np.zeros(layout.n_tmarket, dtype=float)

    loss_rates = items["loss_rate"].astype(float).to_numpy()
    for v in range(F):
        for r in range(R):
            gross_multiplier = 1.0 / (1.0 - float(loss_rates[r]))
            for o in range(O):
                idx = layout.p(v, r, o)
                c[idx] = supplier_cost[r, o] * gross_multiplier
                emission_p[idx - layout.off_p] = supplier_ef[r, o] * gross_multiplier
                if not active_supplier[r, o]:
                    ub[idx] = 0.0
                for a in range(A):
                    for t in range(T):
                        tidx = layout.tin(v, r, o, a, t)
                        if not active_supplier[r, o] or not active_assembly[a]:
                            ub[tidx] = 0.0
                            continue
                        is_battery = r == battery_r
                        if is_battery and production_mode == "line":
                            if o != a or t != INTERNAL_ROUTE_INDEX:
                                ub[tidx] = 0.0
                                continue
                            c[tidx] = 0.0
                            emission_tin[tidx - layout.off_tin] = 0.0
                            continue
                        if o == a:
                            if t != INTERNAL_ROUTE_INDEX:
                                ub[tidx] = 0.0
                                continue
                            c[tidx] = 0.0
                            emission_tin[tidx - layout.off_tin] = 0.0
                            continue
                        if not route["raw_allowed"][o, a, t]:
                            ub[tidx] = 0.0
                            continue
                        mass = mass_per_unit[v, r]
                        c[tidx] = route["raw_cost_per_kg"][o, a, t] * mass
                        emission_tin[tidx - layout.off_tin] = route["raw_ef_per_kg"][o, a, t] * mass

        for a in range(A):
            aidx = layout.assembly(v, a)
            if not active_assembly[a]:
                ub[aidx] = 0.0
            generic_cost = nonbattery_mass[v] * float(plants.iloc[a]["assembly_cost_eur_per_kg"])
            generic_ef = nonbattery_mass[v] * float(plants.iloc[a]["assembly_ef_kgco2_per_kg"])
            extra_cost = float(np.dot(bom_quantity[v, :], process_cost[:, a]))
            extra_ef = float(np.dot(bom_quantity[v, :], process_ef[:, a]))
            c[aidx] = generic_cost + extra_cost
            emission_a[aidx - layout.off_a] = generic_ef + extra_ef
            for t in range(T):
                fidx = layout.tmarket(v, a, t)
                if not active_assembly[a] or not route["final_allowed"][a, t]:
                    ub[fidx] = 0.0
                    continue
                c[fidx] = selected_vehicle_mass[v] * route["final_cost_per_kg"][a, t]
                emission_tmarket[fidx - layout.off_tmarket] = (
                    selected_vehicle_mass[v] * route["final_ef_per_kg"][a, t]
                )

    rows = LinearConstraintBuilder(layout.n_vars)

    # Demand and vehicle-flow balance.
    for v in range(F):
        cols = [layout.tmarket(v, a, t) for a in range(A) for t in range(T)]
        rows.add_eq(cols, [1.0] * len(cols), demand_values[v])
        for a in range(A):
            cols = [layout.assembly(v, a)] + [layout.tmarket(v, a, t) for t in range(T)]
            rows.add_eq(cols, [1.0] + [-1.0] * T, 0.0)

    # Item requirements at each assembly location.
    for v in range(F):
        for r in range(R):
            requirement = bom_quantity[v, r]
            for a in range(A):
                cols = [layout.tin(v, r, o, a, t) for o in range(O) for t in range(T)]
                cols.append(layout.assembly(v, a))
                vals = [1.0] * (O * T) + [-requirement]
                rows.add_eq(cols, vals, 0.0)

    # Production equals all outbound item flow.
    for v in range(F):
        for r in range(R):
            for o in range(O):
                cols = [layout.p(v, r, o)] + [layout.tin(v, r, o, a, t) for a in range(A) for t in range(T)]
                rows.add_eq(cols, [1.0] + [-1.0] * (A * T), 0.0)

    # Supplier capacity.
    for r in range(R):
        for o in range(O):
            rows.add_le(
                [layout.p(v, r, o) for v in range(F)],
                [1.0] * F,
                supplier_capacity[r, o],
            )

    # Modular 10/5-kWh composition. It is a continuous aggregate model.
    if production_mode == "modular":
        for v in range(F):
            for o in range(O):
                for a in range(A):
                    cols = [layout.tin(v, battery_r, o, a, t) for t in range(T)]
                    cols.extend([layout.m10(v, o, a), layout.m5(v, o, a)])
                    vals = [1.0] * T + [-10.0, -5.0]
                    rows.add_eq(cols, vals, 0.0)

    # Company-wide fleet carbon-footprint cap.
    if int(scenario.get("apply_carbon_cap", 0)) == 1:
        cols: List[int] = []
        vals: List[float] = []
        for idx, coef in enumerate(emission_p):
            if coef:
                cols.append(layout.off_p + idx); vals.append(float(coef))
        for idx, coef in enumerate(emission_tin):
            if coef:
                cols.append(layout.off_tin + idx); vals.append(float(coef))
        for idx, coef in enumerate(emission_a):
            if coef:
                cols.append(layout.off_a + idx); vals.append(float(coef))
        for idx, coef in enumerate(emission_tmarket):
            if coef:
                cols.append(layout.off_tmarket + idx); vals.append(float(coef))
        rows.add_le(cols, vals, float(fleet_cap))

    signature = _structure_signature(active_items, selected_map)
    return FlexibleLPModel(
        layout=layout,
        c=c,
        lb=lb,
        ub=ub,
        rows=rows,
        products=products,
        items=items,
        bom=bom,
        suppliers=suppliers,
        processes=processes,
        plants=plants,
        market=market,
        scenario=scenario,
        demand_values=demand_values,
        item_ids=item_ids,
        item_index=item_index,
        battery_r=battery_r,
        bom_quantity=bom_quantity,
        mass_per_unit=mass_per_unit,
        selected_vehicle_mass=selected_vehicle_mass,
        nonbattery_mass=nonbattery_mass,
        supplier_cost=supplier_cost,
        supplier_ef=supplier_ef,
        supplier_capacity=supplier_capacity,
        process_cost=process_cost,
        process_ef=process_ef,
        route=route,
        emission_p=emission_p,
        emission_tin=emission_tin,
        emission_a=emission_a,
        emission_tmarket=emission_tmarket,
        selected_country_map={k: tuple(v) for k, v in selected_map.items()},
        structure_signature=signature,
        equality_count=rows.equality_count,
        inequality_count=rows.inequality_count,
        matrix_nonzeros=rows.nonzero_count,
    )


def _create_solver() -> Tuple[pywraplp.Solver, str]:
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("OR-Tools GLOP를 사용할 수 없습니다. requirements.txt에 ortools를 추가하세요.")
    return solver, "GLOP"


def solve_lp_model(model: FlexibleLPModel, time_limit_sec: int = 180) -> SolveResult:
    started = time.perf_counter()
    solver, solver_name = _create_solver()
    try:
        solver.SetNumThreads(1)
    except Exception:
        pass
    try:
        solver.SetSolverSpecificParametersAsString(GLOP_PARAMETER_TEXT)
    except Exception:
        pass
    solver.SetTimeLimit(max(1, int(time_limit_sec * 1000)))

    coefficients = model.c
    nonzero = np.abs(coefficients[np.nonzero(coefficients)])
    max_abs = float(nonzero.max()) if nonzero.size else 1.0
    scale = max(1.0, max_abs / OBJECTIVE_TARGET_MAX_COEFFICIENT)
    scaled = coefficients / scale
    infinity = solver.infinity()
    variables = [
        solver.NumVar(float(model.lb[i]), float(model.ub[i]) if np.isfinite(model.ub[i]) else infinity, "")
        for i in range(model.layout.n_vars)
    ]
    objective = solver.Objective()
    for idx in np.flatnonzero(scaled):
        objective.SetCoefficient(variables[int(idx)], float(scaled[int(idx)]))
    objective.SetMinimization()

    rows = model.rows
    for row_idx, rhs in enumerate(rows.eq_rhs):
        con = solver.Constraint(float(rhs), float(rhs), "")
        for pos in range(rows.eq_starts[row_idx], rows.eq_starts[row_idx + 1]):
            con.SetCoefficient(variables[rows.eq_cols[pos]], rows.eq_data[pos])
    for row_idx, rhs in enumerate(rows.ub_rhs):
        con = solver.Constraint(-infinity, float(rhs), "")
        for pos in range(rows.ub_starts[row_idx], rows.ub_starts[row_idx + 1]):
            con.SetCoefficient(variables[rows.ub_cols[pos]], rows.ub_data[pos])
    rows.clear_coefficients()

    code = int(solver.Solve())
    status_map = {
        int(pywraplp.Solver.OPTIMAL): "OPTIMAL",
        int(pywraplp.Solver.FEASIBLE): "FEASIBLE",
        int(pywraplp.Solver.INFEASIBLE): "INFEASIBLE",
        int(pywraplp.Solver.UNBOUNDED): "UNBOUNDED",
        int(pywraplp.Solver.ABNORMAL): "ABNORMAL",
        int(getattr(pywraplp.Solver, "MODEL_INVALID", 5)): "MODEL_INVALID",
        int(pywraplp.Solver.NOT_SOLVED): "NOT_SOLVED",
    }
    status = status_map.get(code, f"STATUS_{code}")
    has_solution = status in {"OPTIMAL", "FEASIBLE"}
    x = None
    objective_value = None
    if has_solution:
        x = np.fromiter((var.solution_value() for var in variables), dtype=float, count=len(variables))
        objective_value = float(np.dot(model.c, x))
    try:
        version = str(solver.SolverVersion())
    except Exception:
        version = solver_name
    try:
        iterations = int(solver.iterations())
    except Exception:
        iterations = 0
    message = (
        f"{status}; solver={version}; variables={solver.NumVariables():,}; "
        f"constraints={solver.NumConstraints():,}; objective_scale={scale:.8g}"
    )
    return SolveResult(
        status=status,
        message=message,
        objective_value=objective_value,
        wall_time_sec=time.perf_counter() - started,
        x=x,
        model=model,
        solver_name=solver_name,
        solver_version=version,
        iterations=iterations,
    )


# -----------------------------------------------------------------------------
# Solution extraction and summaries
# -----------------------------------------------------------------------------
def _positive(value: float) -> bool:
    return float(value) > FLOW_TOL


def extract_solution(result: SolveResult) -> Dict:
    model = result.model
    out: Dict = {
        "status": result.status,
        "message": result.message,
        "objective_value": result.objective_value,
        "wall_time_sec": result.wall_time_sec,
        "solver_name": result.solver_name,
        "solver_version": result.solver_version,
        "solver_iterations": result.iterations,
        "variable_count": model.layout.n_vars,
        "constraint_count": model.equality_count + model.inequality_count,
        "matrix_nonzeros": model.matrix_nonzeros,
        "scenario_id": str(model.scenario["scenario_id"]),
        "scenario_name": str(model.scenario["scenario_name"]),
        "production_mode": model.layout.mode,
        "production_mode_name": MODE_LABEL[model.layout.mode],
        "active_item_ids": list(model.item_ids),
        "active_item_names": model.items["item_name_ko"].astype(str).tolist(),
        "selected_country_map": {k: list(v) for k, v in model.selected_country_map.items()},
        "structure_signature": model.structure_signature,
        "plants": model.plants,
        "market": model.market,
        "item_catalog": model.items,
    }
    if result.x is None or result.status not in {"OPTIMAL", "FEASIBLE"}:
        return out

    x = result.x
    L = model.layout
    production_rows: List[Dict] = []
    inbound_rows: List[Dict] = []
    assembly_rows: List[Dict] = []
    market_rows: List[Dict] = []
    module_rows: List[Dict] = []

    item_meta = model.items.set_index("item_id")
    product_ids = model.products["product_id"].astype(str).tolist()
    location_names = model.plants["location_name"].astype(str).tolist()

    for v, product in model.products.iterrows():
        pid = str(product["product_id"])
        for r, item_id in enumerate(model.item_ids):
            item = item_meta.loc[item_id]
            gross_multiplier = 1.0 / (1.0 - float(item["loss_rate"]))
            for o in range(L.O):
                amount = float(x[L.p(v, r, o)])
                if _positive(amount):
                    production_rows.append({
                        "product_id": pid,
                        "product_name_ko": product["product_name_ko"],
                        "item_id": item_id,
                        "item_name_ko": item["item_name_ko"],
                        "item_type": item["item_type"],
                        "origin_index": o + 1,
                        "origin_location": location_names[o],
                        "net_output_amount": amount,
                        "gross_input_amount_after_loss": amount * gross_multiplier,
                        "flow_unit": item["flow_unit"],
                        "stage1_cost_eur": amount * model.supplier_cost[r, o] * gross_multiplier,
                        "stage1_emissions_kgco2": amount * model.supplier_ef[r, o] * gross_multiplier,
                        "stage1_process": item["stage1_process_name_ko"],
                    })
                for a in range(L.A):
                    for t in range(L.T):
                        flow = float(x[L.tin(v, r, o, a, t)])
                        if not _positive(flow):
                            continue
                        mass = flow * model.mass_per_unit[v, r]
                        inbound_rows.append({
                            "product_id": pid,
                            "product_name_ko": product["product_name_ko"],
                            "item_id": item_id,
                            "item_name_ko": item["item_name_ko"],
                            "item_type": item["item_type"],
                            "origin_index": o + 1,
                            "origin_location": location_names[o],
                            "assembly_index": a + 1,
                            "assembly_location": location_names[a],
                            "transport_mode_index": t + 1,
                            "transport_mode": ROUTE_MODE_CODES[t],
                            "transport_mode_ko": ROUTE_MODE_LABEL[ROUTE_MODE_CODES[t]],
                            "flow_amount": flow,
                            "flow_unit": item["flow_unit"],
                            "transport_mass_kg": mass,
                            "distance_km": float(model.route["raw_total_km"][o, a, t]),
                            "transport_cost_eur": flow * model.c[L.tin(v, r, o, a, t)],
                            "transport_emissions_kgco2": flow * model.emission_tin[L.tin(v, r, o, a, t) - L.off_tin],
                            "internal_flow": bool(o == a),
                        })

        for a in range(L.A):
            vehicles = float(x[L.assembly(v, a)])
            if _positive(vehicles):
                generic_cost_per_vehicle = model.nonbattery_mass[v] * float(model.plants.iloc[a]["assembly_cost_eur_per_kg"])
                generic_ef_per_vehicle = model.nonbattery_mass[v] * float(model.plants.iloc[a]["assembly_ef_kgco2_per_kg"])
                extra_cost_per_vehicle = float(np.dot(model.bom_quantity[v, :], model.process_cost[:, a]))
                extra_ef_per_vehicle = float(np.dot(model.bom_quantity[v, :], model.process_ef[:, a]))
                assembly_rows.append({
                    "product_id": pid,
                    "product_name_ko": product["product_name_ko"],
                    "assembly_index": a + 1,
                    "assembly_location": location_names[a],
                    "vehicle_equivalents": vehicles,
                    "selected_nonbattery_mass_kg_per_vehicle": model.nonbattery_mass[v],
                    "selected_vehicle_mass_kg": model.selected_vehicle_mass[v],
                    "body_assembly_cost_eur": vehicles * generic_cost_per_vehicle,
                    "body_assembly_emissions_kgco2": vehicles * generic_ef_per_vehicle,
                    "additional_item_process_cost_eur": vehicles * extra_cost_per_vehicle,
                    "additional_item_process_emissions_kgco2": vehicles * extra_ef_per_vehicle,
                    "total_stage2_cost_eur": vehicles * (generic_cost_per_vehicle + extra_cost_per_vehicle),
                    "total_stage2_emissions_kgco2": vehicles * (generic_ef_per_vehicle + extra_ef_per_vehicle),
                })
            for t in range(L.T):
                vehicles_out = float(x[L.tmarket(v, a, t)])
                if not _positive(vehicles_out):
                    continue
                market_rows.append({
                    "product_id": pid,
                    "product_name_ko": product["product_name_ko"],
                    "assembly_index": a + 1,
                    "assembly_location": location_names[a],
                    "market_id": model.market["market_id"],
                    "market_name": model.market["market_name"],
                    "transport_mode_index": t + 1,
                    "transport_mode": ROUTE_MODE_CODES[t],
                    "transport_mode_ko": ROUTE_MODE_LABEL[ROUTE_MODE_CODES[t]],
                    "vehicle_equivalents": vehicles_out,
                    "selected_vehicle_mass_kg": model.selected_vehicle_mass[v],
                    "distance_km": float(model.route["final_total_km"][a, t]),
                    "transport_cost_eur": vehicles_out * model.c[L.tmarket(v, a, t)],
                    "transport_emissions_kgco2": vehicles_out * model.emission_tmarket[L.tmarket(v, a, t) - L.off_tmarket],
                })

    if L.mode == "modular":
        battery_id = model.item_ids[model.battery_r]
        for v, product in model.products.iterrows():
            for o in range(L.O):
                for a in range(L.A):
                    n10 = float(x[L.m10(v, o, a)])
                    n5 = float(x[L.m5(v, o, a)])
                    if _positive(n10) or _positive(n5):
                        module_rows.append({
                            "product_id": product["product_id"],
                            "product_name_ko": product["product_name_ko"],
                            "item_id": battery_id,
                            "origin_location": location_names[o],
                            "assembly_location": location_names[a],
                            "module_10kwh_count": n10,
                            "module_5kwh_count": n5,
                            "total_capacity_kwh": 10.0 * n10 + 5.0 * n5,
                        })

    production_df = pd.DataFrame(production_rows)
    inbound_df = pd.DataFrame(inbound_rows)
    assembly_df = pd.DataFrame(assembly_rows)
    market_df = pd.DataFrame(market_rows)
    module_df = pd.DataFrame(module_rows)

    if production_df.empty:
        battery_production_cost = battery_production_em = 0.0
        nonbattery_production_cost = nonbattery_production_em = 0.0
    else:
        battery_mask = production_df["item_type"].astype(str).eq("battery")
        battery_production_cost = float(production_df.loc[battery_mask, "stage1_cost_eur"].sum())
        battery_production_em = float(production_df.loc[battery_mask, "stage1_emissions_kgco2"].sum())
        nonbattery_production_cost = float(production_df.loc[~battery_mask, "stage1_cost_eur"].sum())
        nonbattery_production_em = float(production_df.loc[~battery_mask, "stage1_emissions_kgco2"].sum())

    # Stage classification follows the user-facing framework. In line mode, completed-pack
    # production is co-located with vehicle assembly and therefore shown in Stage 2. In modular
    # mode, module production occurs at Stage 1 and the separate module-to-pack coefficient at
    # Stage 2 remains zero unless the user supplies a custom process item.
    if model.layout.mode == "line":
        stage1_cost = nonbattery_production_cost
        stage1_em = nonbattery_production_em
        stage2_battery_cost = battery_production_cost
        stage2_battery_em = battery_production_em
    else:
        stage1_cost = nonbattery_production_cost + battery_production_cost
        stage1_em = nonbattery_production_em + battery_production_em
        stage2_battery_cost = 0.0
        stage2_battery_em = 0.0

    inbound_cost = float(inbound_df.get("transport_cost_eur", pd.Series(dtype=float)).sum())
    inbound_em = float(inbound_df.get("transport_emissions_kgco2", pd.Series(dtype=float)).sum())
    body_cost = float(assembly_df.get("body_assembly_cost_eur", pd.Series(dtype=float)).sum())
    body_em = float(assembly_df.get("body_assembly_emissions_kgco2", pd.Series(dtype=float)).sum())
    extra_cost = float(assembly_df.get("additional_item_process_cost_eur", pd.Series(dtype=float)).sum())
    extra_em = float(assembly_df.get("additional_item_process_emissions_kgco2", pd.Series(dtype=float)).sum())
    market_cost = float(market_df.get("transport_cost_eur", pd.Series(dtype=float)).sum())
    market_em = float(market_df.get("transport_emissions_kgco2", pd.Series(dtype=float)).sum())

    total_emissions = stage1_em + stage2_battery_em + inbound_em + body_em + extra_em + market_em
    supply_chain_cost = stage1_cost + stage2_battery_cost + inbound_cost + body_cost + extra_cost + market_cost
    fleet_cap = float(model.scenario.get("fleet_total_cap_kgco2", np.nan))

    product_rows = []
    for v, product in model.products.iterrows():
        pid = str(product["product_id"])
        demand = float(model.demand_values[v])
        product_cost = 0.0
        product_em = 0.0
        for df, cost_col, em_col in [
            (production_df, "stage1_cost_eur", "stage1_emissions_kgco2"),
            (inbound_df, "transport_cost_eur", "transport_emissions_kgco2"),
            (assembly_df, "total_stage2_cost_eur", "total_stage2_emissions_kgco2"),
            (market_df, "transport_cost_eur", "transport_emissions_kgco2"),
        ]:
            if not df.empty:
                subset = df[df["product_id"].astype(str) == pid]
                product_cost += float(subset[cost_col].sum())
                product_em += float(subset[em_col].sum())
        reference_cap = (
            carbon_cap_from_score(str(product["vehicle_class"]), float(model.scenario["minimum_score"]))
            if int(model.scenario.get("apply_carbon_cap", 0)) == 1 else np.nan
        )
        product_rows.append({
            "product_id": pid,
            "product_name_ko": product["product_name_ko"],
            "vehicle_class": product["vehicle_class"],
            "demand_units": demand,
            "reference_vehicle_mass_kg": product["reference_vehicle_mass_kg"],
            "selected_structure_vehicle_mass_kg": model.selected_vehicle_mass[v],
            "mass_change_kg": model.selected_vehicle_mass[v] - float(product["reference_vehicle_mass_kg"]),
            "total_cost_eur": product_cost,
            "total_emissions_kgco2": product_em,
            "emissions_per_vehicle_kgco2": product_em / demand if demand else np.nan,
            "reference_cap_kgco2_per_vehicle": reference_cap,
            "reference_cap_met_individually": True if not np.isfinite(reference_cap) else product_em / demand <= reference_cap + 1e-6,
        })

    structure_rows = []
    all_catalog = model.items
    for v, product in model.products.iterrows():
        for r, item in all_catalog.iterrows():
            structure_rows.append({
                "product_id": product["product_id"],
                "product_name_ko": product["product_name_ko"],
                "item_id": item["item_id"],
                "item_name_ko": item["item_name_ko"],
                "item_type": item["item_type"],
                "quantity_per_vehicle": model.bom_quantity[v, r],
                "flow_unit": item["flow_unit"],
                "mass_kg_per_vehicle": model.bom_quantity[v, r] * model.mass_per_unit[v, r],
                "stage1_process": item["stage1_process_name_ko"],
                "stage2_process": item["stage2_process_name_ko"],
            })

    out.update({
        "production_summary": production_df,
        "inbound_routes": inbound_df,
        "assembly_summary": assembly_df,
        "market_routes": market_df,
        "module_summary": module_df,
        "product_summary": pd.DataFrame(product_rows),
        "product_structure_summary": pd.DataFrame(structure_rows),
        "cost_breakdown": {
            "Stage 1 원자재·중간재 생산(모듈 방식은 배터리모듈 포함)": stage1_cost,
            "Stage 1→2 조립지 유입 운송": inbound_cost,
            "Stage 2 라인 배터리팩 생산·조립 / 모듈→팩 기본 0": stage2_battery_cost,
            "Stage 2 비배터리 차체 중간가공·차량 조립": body_cost,
            "Stage 2 선택 품목 추가 조립공정": extra_cost,
            "Stage 3 프랑스 시장 출시 운송": market_cost,
        },
        "emission_breakdown": {
            "Stage 1 원자재·중간재 생산(모듈 방식은 배터리모듈 포함)": stage1_em,
            "Stage 1→2 조립지 유입 운송": inbound_em,
            "Stage 2 라인 배터리팩 생산·조립 / 모듈→팩 기본 0": stage2_battery_em,
            "Stage 2 비배터리 차체 중간가공·차량 조립": body_em,
            "Stage 2 선택 품목 추가 조립공정": extra_em,
            "Stage 3 프랑스 시장 출시 운송": market_em,
        },
        "battery_process_definition": (
            "라인 방식은 완성 배터리팩 생산비·탄소발자국을 동일 조립지의 Stage 2로 분류합니다. "
            "모듈 방식은 10/5 kWh 모듈 생산을 Stage 1에 포함하고, 조립지의 별도 모듈→팩 "
            "공정비·탄소발자국은 기본값 0입니다."
        ),
        "supply_chain_cost_eur": supply_chain_cost,
        "total_emissions_kgco2": total_emissions,
        "objective_reconstruction_gap_eur": float(result.objective_value or 0.0) - supply_chain_cost,
        "fleet_total_cap_kgco2": fleet_cap,
        "fleet_cap_slack_kgco2": fleet_cap - total_emissions if np.isfinite(fleet_cap) else np.nan,
        "fleet_cap_utilization_pct": 100.0 * total_emissions / fleet_cap if np.isfinite(fleet_cap) and fleet_cap else np.nan,
        "fleet_cap_met": True if not np.isfinite(fleet_cap) else total_emissions <= fleet_cap + 1e-5,
    })
    return out


def solve_case(
    tables: Mapping[str, pd.DataFrame],
    scenario_id: str,
    production_mode: str,
    time_limit_sec: int,
    active_item_ids: Sequence[str],
    selected_country_map: Mapping[str, Sequence[str]],
) -> Dict:
    model = build_flexible_lp_model(
        tables,
        production_mode=production_mode,
        scenario_id=scenario_id,
        active_item_ids=active_item_ids,
        selected_country_map=selected_country_map,
    )
    result = solve_lp_model(model, time_limit_sec=time_limit_sec)
    return extract_solution(result)


# -----------------------------------------------------------------------------
# Stage-separated mapping
# -----------------------------------------------------------------------------
def _bezier_curve_points(start: Tuple[float, float], end: Tuple[float, float], bend: float, steps: int = 30):
    lat1, lon1 = float(start[0]), float(start[1])
    lat2, lon2 = float(end[0]), float(end[1])
    dx, dy = lon2 - lon1, lat2 - lat1
    length = max(math.hypot(dx, dy), 1e-9)
    mid_lon, mid_lat = (lon1 + lon2) / 2.0, (lat1 + lat2) / 2.0
    control_lon = mid_lon + (-dy / length) * bend * length
    control_lat = mid_lat + (dx / length) * bend * length
    points = []
    for i in range(steps + 1):
        t = i / steps
        lon = (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * control_lon + t ** 2 * lon2
        lat = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * control_lat + t ** 2 * lat2
        points.append((lat, lon))
    return points


def _flow_width(values: pd.Series, value: float) -> float:
    positives = pd.to_numeric(values, errors="coerce").dropna()
    positives = positives[positives > 0]
    if positives.empty or positives.nunique() == 1:
        return 5.0
    q1, q2, q3 = positives.quantile([0.25, 0.5, 0.75]).tolist()
    if value <= q1:
        return 2.5
    if value <= q2:
        return 5.0
    if value <= q3:
        return 8.0
    return 11.0


def build_stage_map(result: Dict, stage: int):
    import folium
    from folium.plugins import Fullscreen

    plants = result["plants"]
    market = result["market"]
    item_catalog = result["item_catalog"].set_index("item_id")
    fmap = folium.Map(location=[35, 25], zoom_start=2, tiles="CartoDB positron")
    Fullscreen(position="topleft").add_to(fmap)

    if stage == 1:
        production = result.get("production_summary", pd.DataFrame()).copy()
        if result.get("production_mode") == "line" and not production.empty:
            production = production[production["item_type"].astype(str) != "battery"]
        if production.empty:
            folium.Marker([35, 25], tooltip="Stage 1 양의 생산량 없음").add_to(fmap)
            return fmap
        agg = production.groupby(
            ["origin_index", "origin_location", "item_id", "item_name_ko", "flow_unit"], as_index=False
        ).agg(
            quantity=("net_output_amount", "sum"),
            cost_eur=("stage1_cost_eur", "sum"),
            emissions_kgco2=("stage1_emissions_kgco2", "sum"),
        )
        for _, row in agg.iterrows():
            loc = plants.iloc[int(row["origin_index"]) - 1]
            color = str(item_catalog.loc[str(row["item_id"]), "color_hex"])
            folium.CircleMarker(
                [loc["latitude"], loc["longitude"]],
                radius=max(5.0, min(13.0, 5.0 + math.log10(max(1.0, float(row["quantity"]))))),
                color=color,
                fill=True,
                fill_opacity=0.82,
                tooltip=(
                    f"Stage 1 생산지: {row['origin_location']}<br>"
                    f"품목: {row['item_name_ko']}<br>"
                    f"생산량: {row['quantity']:,.2f} {row['flow_unit']}<br>"
                    f"비용: €{row['cost_eur']:,.0f}<br>"
                    f"탄소발자국: {row['emissions_kgco2']:,.0f} kg CO₂-eq"
                ),
            ).add_to(fmap)

    elif stage == 2:
        inbound = result.get("inbound_routes", pd.DataFrame()).copy()
        assembly = result.get("assembly_summary", pd.DataFrame()).copy()
        if not inbound.empty:
            external = inbound[~inbound["internal_flow"].astype(bool)].copy()
            if not external.empty:
                agg = external.groupby(
                    ["origin_index", "origin_location", "assembly_index", "assembly_location",
                     "item_id", "item_name_ko", "transport_mode_index", "transport_mode_ko"],
                    as_index=False,
                ).agg(
                    flow_amount=("flow_amount", "sum"),
                    transport_mass_kg=("transport_mass_kg", "sum"),
                    transport_cost_eur=("transport_cost_eur", "sum"),
                    transport_emissions_kgco2=("transport_emissions_kgco2", "sum"),
                )
                for _, row in agg.iterrows():
                    origin = plants.iloc[int(row["origin_index"]) - 1]
                    destination = plants.iloc[int(row["assembly_index"]) - 1]
                    color = str(item_catalog.loc[str(row["item_id"]), "color_hex"])
                    bend = 0.08 * (1 if int(row["transport_mode_index"]) % 2 else -1)
                    curve = _bezier_curve_points(
                        (origin["latitude"], origin["longitude"]),
                        (destination["latitude"], destination["longitude"]),
                        bend,
                    )
                    folium.PolyLine(
                        curve,
                        color=color,
                        weight=_flow_width(agg["transport_mass_kg"], float(row["transport_mass_kg"])),
                        opacity=0.78,
                        dash_array=TRANSPORT_DASH.get(int(row["transport_mode_index"])),
                        tooltip=(
                            f"{row['item_name_ko']} | {row['origin_location']} → {row['assembly_location']}<br>"
                            f"운송: {row['transport_mode_ko']}<br>질량: {row['transport_mass_kg']:,.1f} kg<br>"
                            f"비용: €{row['transport_cost_eur']:,.0f}<br>"
                            f"탄소발자국: {row['transport_emissions_kgco2']:,.0f} kg CO₂-eq"
                        ),
                    ).add_to(fmap)
                    folium.CircleMarker(
                        [origin["latitude"], origin["longitude"]], radius=4, color=color,
                        fill=True, fill_opacity=0.8, tooltip=f"생산지: {row['origin_location']} · {row['item_name_ko']}"
                    ).add_to(fmap)
        if not assembly.empty:
            agg_a = assembly.groupby(["assembly_index", "assembly_location"], as_index=False).agg(
                vehicles=("vehicle_equivalents", "sum"),
                stage2_cost=("total_stage2_cost_eur", "sum"),
                stage2_emissions=("total_stage2_emissions_kgco2", "sum"),
                additional_process_cost=("additional_item_process_cost_eur", "sum"),
                additional_process_emissions=("additional_item_process_emissions_kgco2", "sum"),
            )
            for _, row in agg_a.iterrows():
                loc = plants.iloc[int(row["assembly_index"]) - 1]
                folium.CircleMarker(
                    [loc["latitude"], loc["longitude"]], radius=8, color=ASSEMBLY_COLOR,
                    fill=True, fill_opacity=0.9,
                    tooltip=(
                        f"Stage 2 조립지: {row['assembly_location']}<br>"
                        f"차량 조립량: {row['vehicles']:,.1f}대<br>"
                        f"Stage 2 비용: €{row['stage2_cost']:,.0f}<br>"
                        f"Stage 2 탄소발자국: {row['stage2_emissions']:,.0f} kg CO₂-eq<br>"
                        f"선택 품목 추가공정 비용: €{row['additional_process_cost']:,.0f}"
                    ),
                ).add_to(fmap)
        if result.get("production_mode") == "line":
            battery_prod = result.get("production_summary", pd.DataFrame())
            if not battery_prod.empty:
                battery_prod = battery_prod[battery_prod["item_type"].astype(str) == "battery"]
                for location_name, group in battery_prod.groupby("origin_location"):
                    row = plants[plants["location_name"].astype(str) == str(location_name)].iloc[0]
                    folium.Marker(
                        [row["latitude"], row["longitude"]],
                        icon=folium.Icon(color="blue", icon="bolt", prefix="fa"),
                        tooltip=f"동일 조립지의 완성 배터리팩 생산·조립: {location_name}",
                    ).add_to(fmap)

    elif stage == 3:
        routes = result.get("market_routes", pd.DataFrame()).copy()
        folium.Marker(
            [float(market["latitude"]), float(market["longitude"])],
            icon=folium.Icon(color=MARKET_COLOR, icon="shopping-cart", prefix="fa"),
            tooltip=str(market["market_name"]),
        ).add_to(fmap)
        if not routes.empty:
            agg = routes.groupby(
                ["assembly_index", "assembly_location", "transport_mode_index", "transport_mode_ko"],
                as_index=False,
            ).agg(
                vehicles=("vehicle_equivalents", "sum"),
                cost_eur=("transport_cost_eur", "sum"),
                emissions_kgco2=("transport_emissions_kgco2", "sum"),
            )
            for _, row in agg.iterrows():
                loc = plants.iloc[int(row["assembly_index"]) - 1]
                folium.CircleMarker(
                    [loc["latitude"], loc["longitude"]], radius=7, color=ASSEMBLY_COLOR,
                    fill=True, fill_opacity=0.88, tooltip=f"조립지: {row['assembly_location']}"
                ).add_to(fmap)
                bend = 0.09 * (1 if int(row["transport_mode_index"]) % 2 else -1)
                curve = _bezier_curve_points(
                    (loc["latitude"], loc["longitude"]),
                    (float(market["latitude"]), float(market["longitude"])),
                    bend,
                )
                folium.PolyLine(
                    curve, color=FINISHED_COLOR,
                    weight=_flow_width(agg["vehicles"], float(row["vehicles"])),
                    opacity=0.78,
                    dash_array=TRANSPORT_DASH.get(int(row["transport_mode_index"])),
                    tooltip=(
                        f"완성 전기자동차 | {row['assembly_location']} → 프랑스 시장<br>"
                        f"운송: {row['transport_mode_ko']}<br>차량: {row['vehicles']:,.1f}대<br>"
                        f"비용: €{row['cost_eur']:,.0f}<br>탄소발자국: {row['emissions_kgco2']:,.0f} kg CO₂-eq"
                    ),
                ).add_to(fmap)
    else:
        raise ValueError(stage)
    return fmap


def render_stage_map(result: Dict, stage: int, key: str, height: int = 560):
    if result.get("status") not in {"OPTIMAL", "FEASIBLE"}:
        st.warning(f"{result.get('status')}: {result.get('message')}")
        return
    from streamlit_folium import st_folium
    fmap = build_stage_map(result, stage)
    st_folium(fmap, width=None, height=height, key=key)
    del fmap
    gc.collect()


# -----------------------------------------------------------------------------
# Streamlit UI helpers
# -----------------------------------------------------------------------------
def show_dataframe(df: pd.DataFrame, title: str, caption: Optional[str] = None):
    st.markdown(f"### {title}")
    if caption:
        st.caption(caption)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_item_selection(catalog: pd.DataFrame) -> List[str]:
    st.markdown("### 제품구조 구성품 선택")
    st.info(
        "체크된 원자재·중간재만 제품 BOM과 공급망에 포함됩니다. 항목을 해제하면 해당 질량과 공정이 제거되며, "
        "다른 재료로 자동 대체되지 않습니다. 배터리는 전기자동차 모형의 필수 항목입니다."
    )
    active: List[str] = []
    categories = [
        ("raw_material", "원자재"),
        ("intermediate", "중간재"),
        ("battery", "배터리"),
    ]
    for item_type, label in categories:
        subset = catalog[catalog["item_type"].astype(str) == item_type]
        if subset.empty:
            continue
        st.markdown(f"**{label}**")
        cols = st.columns(min(3, len(subset)))
        for idx, (_, row) in enumerate(subset.iterrows()):
            item_id = str(row["item_id"])
            mandatory = int(row["mandatory"]) == 1
            default = bool(int(row["default_enabled"])) or mandatory
            with cols[idx % len(cols)]:
                checked = st.checkbox(
                    str(row["item_name_ko"]),
                    value=default,
                    disabled=mandatory,
                    key=f"item_enabled::{item_id}",
                    help=f"Stage 1: {row['stage1_process_name_ko']} / Stage 2: {row['stage2_process_name_ko']}",
                )
                if checked or mandatory:
                    active.append(item_id)
    return active


def render_country_selection(
    catalog: pd.DataFrame,
    suppliers: pd.DataFrame,
    plants: pd.DataFrame,
    active_item_ids: Sequence[str],
) -> Dict[str, List[str]]:
    selected: Dict[str, List[str]] = {}
    item_lookup = catalog.set_index("item_id")
    labels = [str(item_lookup.loc[item_id, "item_name_ko"]) for item_id in active_item_ids] + ["차량 조립지"]
    keys = list(active_item_ids) + ["assembly"]
    with st.expander("품목별 생산지·차량 조립지 선택", expanded=False):
        st.caption(
            "각 품목은 서로 다른 생산지 집합을 가질 수 있습니다. 사용자가 허용한 국가 안에서 Solver가 물량을 배분합니다."
        )
        tabs = st.tabs(labels)
        for tab, key in zip(tabs, keys):
            with tab:
                if key == "assembly":
                    available = plants["location_name"].astype(str).tolist()
                else:
                    available = suppliers.loc[
                        (suppliers["item_id"].astype(str) == key)
                        & (pd.to_numeric(suppliers["active_default"], errors="coerce").fillna(0).astype(int) == 1),
                        "location_name",
                    ].astype(str).tolist()
                b1, b2 = st.columns(2)
                prefix = f"country::{key}::"
                if b1.button("모두 선택", key=f"all::{key}", use_container_width=True):
                    for name in available:
                        st.session_state[prefix + name] = True
                if b2.button("모두 해제", key=f"none::{key}", use_container_width=True):
                    for name in available:
                        st.session_state[prefix + name] = False
                chosen: List[str] = []
                grid = st.columns(4)
                for i, name in enumerate(available):
                    state_key = prefix + name
                    if state_key not in st.session_state:
                        st.session_state[state_key] = True
                    with grid[i % 4]:
                        if st.checkbox(name, key=state_key):
                            chosen.append(name)
                st.caption(f"선택: {len(chosen)} / {len(available)}개")
                selected[key] = chosen
    return selected


def product_structure_preview(
    tables: Mapping[str, pd.DataFrame], active_item_ids: Sequence[str]
) -> pd.DataFrame:
    products = tables["products.csv"].sort_values("product_index")
    catalog = tables["item_catalog.csv"].set_index("item_id")
    bom = tables["product_bom.csv"]
    rows = []
    for _, product in products.iterrows():
        pid = str(product["product_id"])
        selected_mass = 0.0
        parts = []
        for item_id in active_item_ids:
            row = bom[(bom["product_id"].astype(str) == pid) & (bom["item_id"].astype(str) == item_id)]
            if row.empty:
                continue
            quantity = float(row.iloc[0]["quantity_per_vehicle"])
            mass = quantity * float(row.iloc[0]["mass_per_unit_kg"])
            selected_mass += mass
            parts.append(f"{catalog.loc[item_id, 'item_name_ko']} {quantity:g} {row.iloc[0]['quantity_unit']}")
        reference = float(product["reference_vehicle_mass_kg"])
        rows.append({
            "product_id": pid,
            "차량": product["product_name_ko"],
            "선택 제품구조": ", ".join(parts),
            "선택 구조 질량(kg)": selected_mass,
            "기준 질량(kg)": reference,
            "질량 변화(kg)": selected_mass - reference,
        })
    return pd.DataFrame(rows)


def render_solver_metrics(result: Dict):
    status = str(result.get("status", "NOT_RUN"))
    if status not in {"OPTIMAL", "FEASIBLE"}:
        st.error(f"{status}: {result.get('message', '해를 찾지 못했습니다.')}")
        if status == "INFEASIBLE":
            st.caption("현재 제품구조, 생산지·조립지, 생산용량, 물량수지와 탄소발자국 상한을 동시에 만족하는 공급망이 없습니다.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("최적 공급망 비용", f"€{float(result.get('objective_value', 0.0)):,.0f}")
    c2.metric("회사 전체 탄소발자국", f"{float(result.get('total_emissions_kgco2', 0.0)):,.0f} kg CO₂-eq")
    score = SCENARIO_POLICY_SCORE.get(str(result.get("scenario_id")))
    c3.metric("적용 보조금 점수", "상한 없음" if score is None else f"{score:.0f}점")
    c4.metric("활성 품목 수", f"{len(result.get('active_item_ids', []))}개")
    if np.isfinite(float(result.get("fleet_total_cap_kgco2", np.nan))):
        st.caption(
            f"회사 전체 상한 {float(result['fleet_total_cap_kgco2']):,.0f} kg CO₂-eq · "
            f"이용률 {float(result['fleet_cap_utilization_pct']):.2f}% · "
            f"잔여 {float(result['fleet_cap_slack_kgco2']):,.0f} kg CO₂-eq · "
            f"충족 {'예' if result.get('fleet_cap_met') else '아니오'}"
        )


def comparison_dataframe(results: Mapping[Tuple[str, str], Dict]) -> pd.DataFrame:
    rows = []
    for scenario in ["S1", "S2", "S3"]:
        for mode in ["line", "modular"]:
            result = results.get((scenario, mode), {})
            if result.get("status") not in {"OPTIMAL", "FEASIBLE"}:
                continue
            rows.append({
                "scenario_id": scenario,
                "시나리오": SCENARIO_SHORT[scenario],
                "production_mode": mode,
                "생산방식": MODE_LABEL[mode],
                "총비용(EUR)": float(result.get("objective_value", 0.0)),
                "회사 전체 탄소발자국(kgCO2-eq)": float(result.get("total_emissions_kgco2", 0.0)),
                "상한 이용률(%)": float(result.get("fleet_cap_utilization_pct", np.nan)),
                "활성 품목": ", ".join(result.get("active_item_names", [])),
                "제품구조 서명": result.get("structure_signature"),
            })
    return pd.DataFrame(rows)


def render_overview_tab(tables: Optional[Mapping[str, pd.DataFrame]] = None):
    st.header("사용자 업로드 데이터 기반 전기자동차 공급망 최적화 SaaS")
    st.info(
        "본 SaaS의 목적함수는 Stage 1 품목 생산, Stage 2 조립지 유입 운송·차량 조립, "
        "Stage 3 프랑스 시장 출시 운송의 총비용 최소화입니다. S2·S3에서는 회사 전체 "
        "탄소발자국 상한이 제약조건으로 적용됩니다."
    )
    st.markdown("## 데이터 운영 원칙")
    st.markdown(
        "- 최적화 데이터는 `app.py`에 내장되어 있지 않습니다. 2번 탭에서 필수 CSV 12개 또는 ZIP을 업로드해야 합니다.\n"
        "- 철강·알루미늄·기타 원자재·배터리 등 기존 품목도 업로드 CSV에서 정의됩니다.\n"
        "- 새 품목은 2번 탭의 품목 정의 폼에 이름과 속성을 입력하고, BOM·생산지·추가 공정 CSV를 업로드하여 추가합니다.\n"
        "- 데이터가 완성된 새 품목은 3번 탭에 체크박스로 자동 표시됩니다.\n"
        "- 품목 체크는 제품구조 범위를 정하며, Solver는 체크된 품목의 생산지·조립지·운송수단과 물량을 최적화합니다."
    )
    st.markdown("## Stage 구조")
    stages = pd.DataFrame([
        ["Stage 1 생산지", "투입 원료 → 원자재·중간재 생산", "선택된 각 품목의 생산량·비용·탄소발자국·생산용량"],
        ["Stage 2 조립지", "품목 운송 → 중간가공·차체 조립·추가 품목공정", "품목별 유입량, 조립국가, 조립비·탄소발자국"],
        ["Stage 3 프랑스 시장", "완성 전기자동차 운송 → 시장 출시", "선택 제품구조 질량과 프랑스 수요 반영"],
    ], columns=["단계", "핵심 흐름", "최적화 반영"])
    st.dataframe(stages, use_container_width=True, hide_index=True)
    st.markdown("## 제품구조 변경 해석")
    st.warning(
        "품목을 제외해도 대체재가 자동으로 증가하지 않습니다. 새 품목을 추가하면 해당 BOM 질량이 차량질량에 추가됩니다. "
        "기능적으로 동등한 설계를 비교하려면 사용자가 BOM CSV에서 제외·대체 품목의 수량을 함께 조정해야 합니다."
    )
    if tables:
        status = item_data_status(tables)
        if not status.empty:
            st.markdown("## 현재 세션의 품목")
            st.dataframe(status, use_container_width=True, hide_index=True)


def render_input_tab(tables: Mapping[str, pd.DataFrame]):
    st.header("사용자 CSV 데이터 추가·관리")
    st.info(
        "모든 최적화 데이터는 사용자가 업로드합니다. 최초 실행 시 제공된 `base_csv` 폴더의 CSV 12개 또는 "
        "`base_csv_upload.zip`을 업로드한 뒤 적용하세요."
    )

    st.markdown("## 2.1 필수 CSV 또는 ZIP 업로드")
    uploads = st.file_uploader(
        "필수 데이터 파일",
        type=["csv", "zip"],
        accept_multiple_files=True,
        key="v10_full_data_uploads",
        help="ZIP 내부의 하위 폴더는 허용됩니다. 필수 파일명과 일치하는 CSV만 읽습니다.",
    )
    if st.button("업로드 데이터 적용", type="primary", key="v10_apply_full_data"):
        parsed, messages = parse_full_data_uploads(uploads)
        for message in messages:
            st.caption(message)
        merged = {name: frame.copy() for name, frame in tables.items()}
        merged.update(parsed)
        missing = [name for name in REQUIRED_FILES if name not in merged]
        if missing:
            st.error("필수 CSV가 부족합니다: " + ", ".join(missing))
        else:
            _save_tables_to_session(merged)
            st.success("사용자 데이터를 세션에 적용했습니다.")
            st.rerun()

    if tables:
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "현재 세션 CSV ZIP 다운로드",
                data=make_tables_zip(tables),
                file_name="current_ev_supply_chain_data.zip",
                mime="application/zip",
                use_container_width=True,
            )
        with c2:
            if st.button("업로드 데이터와 결과 초기화", use_container_width=True, key="v10_reset_all"):
                st.session_state.pop(SESSION_TABLES_KEY, None)
                st.session_state.pop(SESSION_RESULTS_KEY, None)
                st.rerun()
    else:
        st.warning("아직 적용된 데이터가 없습니다. 필수 CSV 12개 또는 ZIP을 업로드하세요.")
        return

    errors = validate_tables(tables)
    if errors:
        st.warning("현재 데이터에는 아래 검증 문제가 있습니다. 새 품목 정의만 먼저 추가한 경우 관련 CSV를 적용하면 해소됩니다.")
        for error in errors:
            st.write(f"- {error}")
    else:
        st.success("현재 데이터가 최적화 스키마와 참조 무결성 검증을 통과했습니다.")

    st.markdown("## 2.2 새 원자재·중간재 정의")
    st.caption(
        "여기서는 품목의 이름과 공정 속성만 정의합니다. 정의 후 아래 2.3에서 차량별 BOM, 생산지, "
        "필요한 경우 Stage 2 추가 공정 CSV를 업로드해야 최적화에 사용할 수 있습니다."
    )
    with st.form("v10_item_definition_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        item_id_input = c1.text_input("item_id", value="rare_earth", help="영문 소문자·숫자·밑줄. 예: rare_earth")
        item_name = c2.text_input("화면 표시 품목명", value="희토류 원자재")
        item_type = c3.selectbox("품목 유형", ["raw_material", "intermediate"], format_func=lambda x: {"raw_material":"원자재", "intermediate":"중간재"}[x])
        c4, c5, c6 = st.columns(3)
        flow_unit = c4.text_input("흐름 단위", value="kg")
        loss_rate = c5.number_input("Stage 1 손실률", min_value=0.0, max_value=0.999, value=0.03, step=0.01, format="%.4f")
        color_hex = c6.color_picker("지도 색상", value="#8c510a")
        stage1_name = st.text_input("Stage 1 공정명", value="희토류 원료 → 희토류 원자재")
        has_stage2 = st.checkbox("별도의 Stage 2 추가 공정이 있음", value=True)
        stage2_name = st.text_input("Stage 2 추가 공정명", value="희토류 자석·구동계 부품 조립")
        default_enabled = st.checkbox("3번 탭에서 기본 체크 상태", value=False)
        submitted = st.form_submit_button("품목 정의 추가 또는 갱신", type="primary")
    if submitted:
        try:
            item_id = _normalized_item_id(item_id_input)
            if not item_name.strip():
                raise ValueError("화면 표시 품목명을 입력하세요.")
            updated = {name: frame.copy() for name, frame in tables.items()}
            catalog = updated["item_catalog.csv"].copy()
            existing = catalog[catalog["item_id"].astype(str) == item_id]
            item_index = int(existing.iloc[0]["item_index"]) if not existing.empty else int(pd.to_numeric(catalog["item_index"], errors="coerce").max()) + 1
            row = {
                "item_index": item_index,
                "item_id": item_id,
                "item_name_ko": item_name.strip(),
                "item_type": item_type,
                "flow_unit": flow_unit.strip() or "kg",
                "default_enabled": int(default_enabled),
                "mandatory": 0,
                "has_stage2_process": int(has_stage2),
                "stage1_process_name_ko": stage1_name.strip(),
                "stage2_process_name_ko": stage2_name.strip() if has_stage2 else "차량 일반 조립공정에 포함",
                "color_hex": color_hex,
                "loss_rate": float(loss_rate),
                "source_basis": "user definition form",
            }
            incoming = pd.DataFrame([row])
            updated["item_catalog.csv"] = _upsert_frame(catalog, incoming, ["item_id"])
            _save_tables_to_session(updated)
            st.success(f"품목 정의를 저장했습니다: {item_id} / {item_name}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    status = item_data_status(tables)
    if not status.empty:
        st.dataframe(status, use_container_width=True, hide_index=True)

    st.markdown("## 2.3 선택 품목의 BOM·생산지·추가 공정 CSV 적용")
    catalog = tables["item_catalog.csv"].sort_values("item_index")
    item_options = catalog["item_id"].astype(str).tolist()
    selected_item = st.selectbox(
        "데이터를 적용할 품목",
        item_options,
        format_func=lambda item_id: f"{item_id} — {catalog.set_index('item_id').loc[item_id, 'item_name_ko']}",
        key="v10_addon_item",
    )
    selected_row = catalog[catalog["item_id"].astype(str) == selected_item].iloc[0]
    needs_process = int(selected_row["has_stage2_process"]) == 1
    st.caption(
        "추가 CSV에는 item_id와 item_index를 넣지 않아도 됩니다. 이 화면에서 선택한 품목의 값이 자동으로 붙습니다. "
        "같은 제품·생산지 행이 이미 있으면 새 행으로 갱신됩니다."
    )
    c1, c2, c3 = st.columns(3)
    bom_upload = c1.file_uploader("제품별 BOM 추가 CSV", type="csv", key="v10_bom_addon")
    supplier_upload = c2.file_uploader("생산지 추가 CSV", type="csv", key="v10_supplier_addon")
    process_upload = c3.file_uploader(
        "Stage 2 공정 추가 CSV" + (" (필수)" if needs_process else " (선택)"),
        type="csv", key="v10_process_addon"
    )
    if st.button("선택 품목 CSV 적용", type="primary", key="v10_apply_addon"):
        try:
            if bom_upload is None or supplier_upload is None:
                raise ValueError("제품별 BOM 추가 CSV와 생산지 추가 CSV는 필수입니다.")
            bom_addon = read_single_csv(bom_upload, "제품별 BOM 추가 CSV")
            supplier_addon = read_single_csv(supplier_upload, "생산지 추가 CSV")
            process_addon = read_single_csv(process_upload, "Stage 2 공정 추가 CSV") if process_upload is not None else None
            updated = apply_item_addon(tables, selected_item, bom_addon, supplier_addon, process_addon)
            _save_tables_to_session(updated)
            st.success(f"{selected_item} 관련 CSV를 적용했습니다. 3번 탭의 제품구조 체크박스에서 활성화하세요.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.markdown("## 2.4 사용자 추가 품목 삭제")
    removable = catalog[pd.to_numeric(catalog["mandatory"], errors="coerce").fillna(0).astype(int) == 0]
    if not removable.empty:
        remove_id = st.selectbox("삭제할 품목", removable["item_id"].astype(str).tolist(), key="v10_remove_item")
        if st.button("품목 정의와 관련 데이터 삭제", key="v10_delete_item"):
            try:
                _save_tables_to_session(delete_custom_item(tables, remove_id))
                st.success(f"{remove_id}를 삭제했습니다.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.markdown("## 2.5 현재 입력 테이블 확인")
    display = {
        "item_catalog.csv": "품목 카탈로그",
        "product_bom.csv": "제품별 BOM",
        "item_suppliers.csv": "품목별 생산지 데이터",
        "stage2_item_processes.csv": "품목별 Stage 2 추가 공정",
        "products.csv": "차량 종류",
        "demand.csv": "프랑스 수요",
        "assembly_locations.csv": "차량 조립지",
        "transport_parameters.csv": "운송수단 계수",
        "country_transport_rules.csv": "국가별 운송 허용규칙",
        "markets.csv": "시장",
        "scenarios.csv": "정책 시나리오",
        "model_metadata.csv": "모형 메타데이터",
    }
    subtabs = st.tabs(list(display.values()))
    for tab, filename in zip(subtabs, display):
        with tab:
            st.dataframe(tables[filename], use_container_width=True, hide_index=True)


def render_results_tab(results: Mapping[Tuple[str, str], Dict]):
    st.header("최적화 결과")
    if not results:
        st.info("3번 탭에서 최적화를 실행하세요.")
        return
    for scenario in ["S1", "S2", "S3"]:
        st.markdown(f"## {SCENARIO_SHORT[scenario]}")
        cols = st.columns(2)
        for col, mode in zip(cols, ["line", "modular"]):
            with col:
                st.markdown(f"### {MODE_LABEL[mode]}")
                result = results.get((scenario, mode))
                if not result:
                    st.info("미실행")
                else:
                    render_solver_metrics(result)
        st.divider()

    available = [key for key, value in results.items() if value.get("status") in {"OPTIMAL", "FEASIBLE"}]
    if not available:
        return
    st.markdown("## Stage 1·2·3 공급망 지도")
    chosen = st.selectbox(
        "지도 조회 조합",
        available,
        format_func=lambda key: f"{SCENARIO_SHORT[key[0]]} · {MODE_LABEL[key[1]]}",
        key="stage_map_result_choice",
    )
    stage_label = st.radio(
        "지도 단계",
        ["Stage 1 생산지", "Stage 2 조립지", "Stage 3 프랑스 시장"],
        horizontal=True,
        key="stage_map_stage_choice",
    )
    stage = {"Stage 1 생산지": 1, "Stage 2 조립지": 2, "Stage 3 프랑스 시장": 3}[stage_label]
    selected = results[chosen]
    if stage == 1:
        st.caption("원자재·중간재 생산지를 품목별로 표시합니다. 라인 방식의 배터리팩 생산은 동일 조립지 활동이므로 Stage 2에 표시합니다.")
    elif stage == 2:
        st.caption("생산지에서 조립지로 들어오는 품목별 운송경로와 차량 조립지를 표시합니다.")
    else:
        st.caption("조립지에서 프랑스 시장으로 이동하는 완성 전기자동차 경로를 표시합니다.")
    render_stage_map(selected, stage, key=f"stage_map_{chosen[0]}_{chosen[1]}_{stage}")

    st.markdown("## 상세 결과")
    detail_tabs = st.tabs([
        "제품구조", "차량별 결과", "Stage 1 생산", "Stage 1→2 운송",
        "Stage 2 조립", "배터리모듈", "Stage 3 시장 출시", "Solver 정보",
    ])
    frames = [
        selected.get("product_structure_summary", pd.DataFrame()),
        selected.get("product_summary", pd.DataFrame()),
        selected.get("production_summary", pd.DataFrame()),
        selected.get("inbound_routes", pd.DataFrame()),
        selected.get("assembly_summary", pd.DataFrame()),
        selected.get("module_summary", pd.DataFrame()),
        selected.get("market_routes", pd.DataFrame()),
    ]
    for tab, frame in zip(detail_tabs[:7], frames):
        with tab:
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                st.dataframe(frame, use_container_width=True, hide_index=True)
            else:
                st.info("표시할 양의 결과가 없습니다.")
    with detail_tabs[7]:
        st.json({
            key: selected.get(key) for key in [
                "status", "message", "solver_name", "solver_version", "solver_iterations",
                "variable_count", "constraint_count", "matrix_nonzeros", "wall_time_sec",
                "structure_signature", "active_item_ids", "selected_country_map",
                "fleet_total_cap_kgco2", "fleet_cap_utilization_pct", "fleet_cap_met",
                "objective_reconstruction_gap_eur",
            ]
        })


def render_analysis_tab(results: Mapping[Tuple[str, str], Dict]):
    st.header("분석 및 결론")
    comparison = comparison_dataframe(results)
    if comparison.empty:
        st.info("분석할 OPTIMAL 또는 FEASIBLE 결과가 없습니다.")
        return
    show_dataframe(comparison, "시나리오·생산방식 비교")
    st.markdown("## 총비용 비교")
    st.bar_chart(comparison.set_index(["시나리오", "생산방식"])[["총비용(EUR)"]])
    st.markdown("## 회사 전체 탄소발자국 비교")
    st.bar_chart(comparison.set_index(["시나리오", "생산방식"])[["회사 전체 탄소발자국(kgCO2-eq)"]])

    st.markdown("## 제품구조 변경 해석")
    st.markdown(
        "- 사용자가 추가한 품목을 체크하면 해당 품목의 Stage 1 생산·운송과 선택적으로 정의한 Stage 2 추가 공정이 공급망에 포함됩니다.\n"
        "- 기존 품목을 해제하면 해당 품목의 생산·운송·질량이 제거됩니다. 다른 재료는 자동으로 대체되지 않습니다.\n"
        "- 선택 제품구조의 질량이 Stage 2 일반 조립과 Stage 3 완성차 운송비·탄소발자국에 반영됩니다.\n"
        "- 서로 다른 제품구조를 비교할 때는 `제품구조 서명`, 활성 품목, BOM 및 입력 데이터 버전이 같은지 확인해야 합니다."
    )
    st.warning("사용자 업로드 데이터의 출처·단위·시스템 경계를 검증한 뒤 결과를 의사결정에 사용하세요.")


def run_app():
    st.set_page_config(page_title="사용자 데이터 기반 EV 공급망 LP", page_icon="🚗", layout="wide")
    st.title("프랑스 전기차 보조금 탄소발자국 상한 대응 공급망 비용 최적화")
    st.caption(
        f"build: {APP_BUILD} · 전체 CSV 사용자 업로드 · 사용자 품목 정의·추가 · 제품구조 체크박스 · Stage 1/2/3 분리 지도"
    )

    tables: Dict[str, pd.DataFrame] = {
        name: frame.copy() for name, frame in st.session_state.get(SESSION_TABLES_KEY, {}).items()
    }

    with st.sidebar:
        st.header("세션 상태")
        if tables:
            st.success(f"{len(tables)}개 CSV가 세션에 로드됨")
            errors = validate_tables(tables) if all(name in tables for name in REQUIRED_FILES) else ["필수 CSV 누락"]
            if errors:
                st.warning(f"검증 문제 {len(errors)}개")
            else:
                st.success("최적화 준비 데이터 검증 통과")
        else:
            st.info("2번 탭에서 CSV 또는 ZIP을 업로드하세요.")
        if st.button("최적화 결과만 초기화", use_container_width=True, key="v10_reset_results"):
            st.session_state.pop(SESSION_RESULTS_KEY, None)
            gc.collect()
            st.success("결과를 초기화했습니다.")

    tabs = st.tabs([
        "1. SaaS 최적화 프레임워크 개요",
        "2. 사용자 데이터 추가·관리",
        "3. 최적화 실행",
        "4. 최적화 결과",
        "5. 분석 및 결론",
    ])

    with tabs[0]:
        render_overview_tab(tables if tables else None)

    with tabs[1]:
        render_input_tab(tables)

    with tabs[2]:
        st.header("최적화 실행")
        if not tables or any(name not in tables for name in REQUIRED_FILES):
            st.info("2번 탭에서 필수 CSV 12개 또는 ZIP을 먼저 업로드하세요.")
        else:
            data_errors = validate_tables(tables)
            if data_errors:
                st.error("입력 데이터 검증 문제를 해결해야 최적화를 실행할 수 있습니다.")
                for error in data_errors:
                    st.write(f"- {error}")
            else:
                active_item_ids = render_item_selection(tables["item_catalog.csv"].sort_values("item_index"))
                active_errors = []
                ready = item_data_status(tables).set_index("item_id") if not item_data_status(tables).empty else pd.DataFrame()
                for item_id in active_item_ids:
                    if item_id not in ready.index or not bool(ready.loc[item_id, "최적화 사용 준비"]):
                        active_errors.append(item_id)
                if active_errors:
                    st.error("선택된 품목의 BOM·생산지·공정 데이터가 완전하지 않습니다: " + ", ".join(active_errors))
                preview = product_structure_preview(tables, active_item_ids)
                show_dataframe(
                    preview,
                    "선택 제품구조 미리보기",
                    "품목 추가·제거에 따른 질량 변화입니다. 자동 대체재는 적용되지 않습니다.",
                )
                selected_country_map = render_country_selection(
                    tables["item_catalog.csv"], tables["item_suppliers.csv"], tables["assembly_locations.csv"], active_item_ids
                )
                valid_selection = not active_errors and all(selected_country_map.get(key) for key in [*active_item_ids, "assembly"])
                if not valid_selection:
                    st.error("모든 활성 품목과 차량 조립지에 최소 1개 국가를 선택하고 품목 데이터를 완성해야 합니다.")

                c1, c2, c3 = st.columns(3)
                scenario_id = c1.selectbox(
                    "정책 시나리오", ["S1", "S2", "S3"],
                    format_func=lambda value: str(tables["scenarios.csv"].set_index("scenario_id").loc[value, "scenario_name"]),
                )
                production_mode = c2.selectbox("생산방식", ["line", "modular"], format_func=lambda value: MODE_LABEL[value])
                time_limit = c3.number_input("Solver 제한시간(초)", min_value=10, max_value=600, value=180, step=10)

                st.info(
                    "품목 체크와 허용국가는 사용자가 정하는 모형 범위입니다. Solver는 선택된 제품구조 안에서 "
                    "생산지·조립지·운송수단별 연속 물량을 배분하여 총비용을 최소화합니다."
                )

                def execute(s: str, mode: str, status_box, prefix: str = "") -> Dict:
                    score = SCENARIO_POLICY_SCORE[s]
                    if score is None:
                        status_box.info(f"{prefix}탄소발자국 상한 없는 S1 최소비용 공급망을 계산합니다.")
                    else:
                        status_box.info(f"{prefix}{score:.0f}점의 회사 전체 탄소발자국 상한에서 최소비용 공급망을 계산합니다.")
                    try:
                        output = solve_case(
                            tables,
                            scenario_id=s,
                            production_mode=mode,
                            time_limit_sec=int(time_limit),
                            active_item_ids=active_item_ids,
                            selected_country_map=selected_country_map,
                        )
                        if output.get("status") == "OPTIMAL":
                            status_box.success(f"{prefix}OPTIMAL 해를 찾았습니다.")
                        elif output.get("status") == "FEASIBLE":
                            status_box.warning(f"{prefix}FEASIBLE 해를 찾았지만 최적성은 증명되지 않았습니다.")
                        else:
                            status_box.error(f"{prefix}{output.get('status')}: {output.get('message')}")
                        return output
                    except Exception as exc:
                        status_box.error(f"{prefix}ERROR: {exc}")
                        return {
                            "status": "ERROR", "message": str(exc), "scenario_id": s,
                            "production_mode": mode, "active_item_ids": list(active_item_ids),
                        }

                b1, b2 = st.columns(2)
                with b1:
                    if st.button("선택 조합 실행", type="primary", use_container_width=True, disabled=not valid_selection):
                        box = st.empty()
                        result = execute(scenario_id, production_mode, box)
                        st.session_state.setdefault(SESSION_RESULTS_KEY, {})[(scenario_id, production_mode)] = result
                with b2:
                    if st.button("3개 시나리오 × 2개 생산방식 실행", use_container_width=True, disabled=not valid_selection):
                        results = st.session_state.setdefault(SESSION_RESULTS_KEY, {})
                        progress = st.progress(0.0)
                        box = st.empty()
                        cases = [(s, mode) for s in ["S1", "S2", "S3"] for mode in ["line", "modular"]]
                        for i, (s, mode) in enumerate(cases, start=1):
                            prefix = f"[{i}/6] {SCENARIO_SHORT[s]} · {MODE_LABEL[mode]}: "
                            results[(s, mode)] = execute(s, mode, box, prefix)
                            progress.progress(i / len(cases))
                            gc.collect()
                        box.success("6개 조합 계산을 완료했습니다.")

    with tabs[3]:
        render_results_tab(st.session_state.get(SESSION_RESULTS_KEY, {}))

    with tabs[4]:
        render_analysis_tab(st.session_state.get(SESSION_RESULTS_KEY, {}))


if __name__ == "__main__":
    if st is None:
        raise RuntimeError("Streamlit이 설치되어 있지 않습니다.")
    run_app()
