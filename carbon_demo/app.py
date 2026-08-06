from __future__ import annotations

# Flexible EV supply-chain LP (v19.0)
# Every selected item follows the same line/modular algebra.
# Line: Stage-1 origin equals Stage-2 assembly for every selected item.
# Modular: two item-specific module sizes are produced off-site and processed at Stage 2.



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


APP_BUILD = "all-item-two-level-module-efficiency-v19.0"
APP_PACKAGE_ID = "20260807-v19.0-dynamic-item-modules-line-colocation"
SESSION_TABLES_KEY = "user_tables_v19"
SESSION_RESULTS_KEY = "optimization_results_v19"
SESSION_SELECTED_ITEMS_KEY = "selected_item_ids_v19"
SESSION_SELECTED_TRANSPORT_MODES_KEY = "selected_transport_modes_v19"

REQUIRED_FILES = [
    "products.csv",
    "product_bom.csv",
    "item_catalog.csv",
    "item_suppliers.csv",
    "stage2_item_processes.csv",
    "module_parameters.csv",
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
ASSEMBLY_COLOR = "#111827"
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
MODE_LABEL = {"line": "전 품목 라인 생산", "modular": "전 품목 2수준 모듈 분산 생산"}
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
    st.session_state.pop(SESSION_SELECTED_ITEMS_KEY, None)
    for key in list(st.session_state.keys()):
        if str(key).startswith(("v11_item_enabled::", "country::stage1::", "country::stage2::", "transport_mode::")):
            st.session_state.pop(key, None)
    st.session_state.pop(SESSION_SELECTED_TRANSPORT_MODES_KEY, None)


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
    missing_files = [name for name in REQUIRED_FILES if name not in tables]
    if missing_files:
        return ["필수 CSV 누락: " + ", ".join(missing_files)]

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
        "module_parameters.csv": {
            "item_id", "large_module_size", "small_module_size", "module_unit",
            "large_stage1_cost_factor", "small_stage1_cost_factor",
            "large_stage1_ef_factor", "small_stage1_ef_factor",
            "large_stage2_cost_factor", "small_stage2_cost_factor",
            "large_stage2_ef_factor", "small_stage2_ef_factor",
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
    modules = tables["module_parameters.csv"].copy()
    plants = tables["assembly_locations.csv"].copy()

    if products["product_id"].astype(str).duplicated().any():
        errors.append("products.csv: product_id는 중복될 수 없습니다.")
    if catalog["item_id"].astype(str).duplicated().any():
        errors.append("item_catalog.csv: item_id는 중복될 수 없습니다.")
    if modules["item_id"].astype(str).duplicated().any():
        errors.append("module_parameters.csv: item_id는 중복될 수 없습니다.")
    if plants["location_index"].astype(int).duplicated().any():
        errors.append("assembly_locations.csv: location_index는 중복될 수 없습니다.")

    catalog_ids = set(catalog["item_id"].astype(str))
    product_ids = set(products["product_id"].astype(str))
    for frame, filename in [(bom,"product_bom.csv"),(suppliers,"item_suppliers.csv"),(processes,"stage2_item_processes.csv"),(modules,"module_parameters.csv")]:
        if not set(frame["item_id"].astype(str)).issubset(catalog_ids):
            errors.append(f"{filename}: item_catalog.csv에 없는 item_id가 있습니다.")
    if set(modules["item_id"].astype(str)) != catalog_ids:
        errors.append("module_parameters.csv: item_catalog.csv의 모든 품목에 대해 정확히 한 행이 필요합니다.")
    if not set(bom["product_id"].astype(str)).issubset(product_ids):
        errors.append("product_bom.csv: products.csv에 없는 product_id가 있습니다.")

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
    for col in [
        "large_module_size", "small_module_size",
        "large_stage1_cost_factor", "small_stage1_cost_factor",
        "large_stage1_ef_factor", "small_stage1_ef_factor",
        "large_stage2_cost_factor", "small_stage2_cost_factor",
        "large_stage2_ef_factor", "small_stage2_ef_factor",
    ]:
        numeric_checks.append((modules, col, 0.0, None, "module_parameters.csv"))
    for frame, col, low, high, filename in numeric_checks:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any() or (values <= low if col in {"large_module_size","small_module_size"} else values < low).any() or (high is not None and (values >= high).any()):
            errors.append(f"{filename}.{col}: 유효한 0 이상의 숫자여야 합니다.")

    for item_id in catalog_ids:
        item_bom = bom[bom["item_id"].astype(str) == item_id]
        if set(item_bom["product_id"].astype(str)) != product_ids or len(item_bom) != len(product_ids):
            errors.append(f"product_bom.csv: {item_id}는 모든 제품에 대해 정확히 한 행이 필요합니다.")
        if suppliers[suppliers["item_id"].astype(str) == item_id].empty:
            errors.append(f"item_suppliers.csv: {item_id} 공급자 데이터가 없습니다.")

    location_ids = set(plants["location_index"].astype(int))
    for item_id in catalog_ids:
        item_proc = processes[processes["item_id"].astype(str) == item_id]
        if set(item_proc["location_index"].astype(int)) != location_ids:
            errors.append(f"stage2_item_processes.csv: {item_id}는 모든 조립지에 대한 공정계수가 필요합니다.")

    # Every BOM quantity must be represented exactly by the fixed large/small module recipe.
    module_by_item = modules.set_index("item_id")
    for _, row in bom.iterrows():
        item_id = str(row["item_id"])
        q = float(row["quantity_per_vehicle"])
        large = float(module_by_item.loc[item_id, "large_module_size"])
        small = float(module_by_item.loc[item_id, "small_module_size"])
        if large <= 0 or small <= 0 or large <= small:
            errors.append(f"module_parameters.csv: {item_id}는 large_module_size > small_module_size > 0이어야 합니다.")
            continue
        n_large = math.floor((q + 1e-10) / large)
        remainder = q - n_large * large
        n_small = remainder / small
        if abs(n_small - round(n_small)) > 1e-7:
            errors.append(
                f"모듈 조합 불일치: product={row['product_id']}, item={item_id}, quantity={q}는 "
                f"large={large}, small={small}의 정수 조합으로 정확히 표현되지 않습니다."
            )

    battery_items = catalog[catalog["item_type"].astype(str) == "battery"]
    if len(battery_items) != 1 or int(battery_items.iloc[0]["mandatory"]) != 1:
        errors.append("item_catalog.csv: mandatory=1인 battery 항목이 정확히 1개여야 합니다.")

    rules = tables["country_transport_rules.csv"]
    for col in ["allow_road", "allow_rail", "allow_sea", "allow_air"]:
        values = pd.to_numeric(rules[col], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            errors.append(f"country_transport_rules.csv.{col}: 0 또는 1만 허용됩니다.")

    transport_costs = pd.to_numeric(tables["transport_parameters.csv"]["transport_cost_eur_per_kgkm"], errors="coerce")
    if transport_costs.isna().any() or (transport_costs < 0).any() or (transport_costs > 10.0).any():
        errors.append("transport_parameters.csv: 운송비는 0~10 EUR/(kg·km) 범위여야 합니다.")
    return errors


def ordered_tables(tables: Mapping[str, pd.DataFrame]):
    return (
        tables["products.csv"].sort_values("product_index").reset_index(drop=True),
        tables["product_bom.csv"].copy(),
        tables["item_catalog.csv"].sort_values("item_index").reset_index(drop=True),
        tables["item_suppliers.csv"].sort_values(["item_index", "location_index"]).reset_index(drop=True),
        tables["stage2_item_processes.csv"].copy(),
        tables["module_parameters.csv"].copy(),
        tables["demand.csv"].copy(),
        tables["assembly_locations.csv"].sort_values("location_index").reset_index(drop=True),
        tables["transport_parameters.csv"].copy().reset_index(drop=True),
        tables["country_transport_rules.csv"].copy().reset_index(drop=True),
        tables["markets.csv"].copy(),
        tables["scenarios.csv"].copy(),
        tables["model_metadata.csv"].copy(),
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


@lru_cache(maxsize=16)
def cached_selected_distance_matrices(
    origin_coordinates: Tuple[Tuple[float, float], ...],
    assembly_coordinates: Tuple[Tuple[float, float], ...],
    market_coordinate: Tuple[float, float],
    minimum_market_distance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Distances for compact Stage-1 origin and Stage-2 assembly index sets."""
    raw = np.zeros((len(origin_coordinates), len(assembly_coordinates)), dtype=float)
    for o, (lat_o, lon_o) in enumerate(origin_coordinates):
        for a, (lat_a, lon_a) in enumerate(assembly_coordinates):
            raw[o, a] = rounded_distance_km(lat_o, lon_o, lat_a, lon_a)
    final = np.zeros(len(assembly_coordinates), dtype=float)
    for a, (lat_a, lon_a) in enumerate(assembly_coordinates):
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
    if route_code == "road":
        return bool(origin["allow_road"] and destination["allow_road"] and (same_continent or both_europe))
    if route_code == "rail":
        return bool(origin["allow_rail"] and destination["allow_rail"] and (same_continent or both_europe))
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
    origins: pd.DataFrame,
    assemblies: pd.DataFrame,
    market: pd.Series,
    transport: pd.DataFrame,
    country_rules: pd.DataFrame,
    raw_distances: np.ndarray,
    final_distances: np.ndarray,
    route_mode_codes: Sequence[str],
    selected_external_modes: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Build route coefficients only for the active transport-mode index set.

    ``road`` remains as the internal zero-distance slot required by the uploaded
    v8.20 battery structure. When the user disables external road transport, road
    is still available only for same-location internal movement.
    """
    factor_cost, factor_ef = _factor_maps(transport)
    rules = _rule_map(country_rules)
    route_codes = tuple(str(code) for code in route_mode_codes)
    external_modes = {str(code) for code in selected_external_modes}
    O, A, T = len(origins), len(assemblies), len(route_codes)

    raw_allowed = np.zeros((O, A, T), dtype=bool)
    raw_cost = np.zeros((O, A, T), dtype=float)
    raw_ef = np.zeros((O, A, T), dtype=float)
    raw_total = np.zeros((O, A, T), dtype=float)
    raw_inland = np.zeros((O, A, T), dtype=float)
    raw_international = np.zeros((O, A, T), dtype=float)
    for o in range(O):
        origin = rules[str(origins.iloc[o]["location_name"])]
        for a in range(A):
            destination = rules[str(assemblies.iloc[a]["location_name"])]
            same_location = str(origin["location_name"]) == str(destination["location_name"])
            for t, code in enumerate(route_codes):
                if same_location and code == "road":
                    # Keep the uploaded model's t=0 internal slot even when external road is unchecked.
                    allowed = True
                elif code not in external_modes:
                    allowed = False
                else:
                    allowed = _route_allowed(origin, destination, code)
                raw_allowed[o, a, t] = allowed
                if allowed:
                    values = _route_coefficient(
                        raw_distances[o, a], origin, destination, code, factor_cost, factor_ef
                    )
                    (raw_cost[o, a, t], raw_ef[o, a, t], raw_total[o, a, t],
                     raw_inland[o, a, t], raw_international[o, a, t]) = values

    final_allowed = np.zeros((A, T), dtype=bool)
    final_cost = np.zeros((A, T), dtype=float)
    final_ef = np.zeros((A, T), dtype=float)
    final_total = np.zeros((A, T), dtype=float)
    final_inland = np.zeros((A, T), dtype=float)
    final_international = np.zeros((A, T), dtype=float)
    destination = rules[str(market["location_name"])]
    for a in range(A):
        origin = rules[str(assemblies.iloc[a]["location_name"])]
        same_location = str(origin["location_name"]) == str(destination["location_name"])
        for t, code in enumerate(route_codes):
            if same_location and code == "road":
                allowed = True
            elif code not in external_modes:
                allowed = False
            else:
                allowed = _route_allowed(origin, destination, code)
            final_allowed[a, t] = allowed
            if allowed:
                values = _route_coefficient(
                    final_distances[a], origin, destination, code, factor_cost, factor_ef
                )
                (final_cost[a, t], final_ef[a, t], final_total[a, t],
                 final_inland[a, t], final_international[a, t]) = values

    return {
        "route_mode_codes": route_codes,
        "selected_external_modes": tuple(str(v) for v in selected_external_modes),
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
    """Dynamic LP variable layout for all selected items.

    P: Stage-1 output; Tin: Stage-1→2 flow; A: vehicle assembly; Tmarket: finished vehicle flow.
    Modular mode additionally creates ML/MS for every selected item, vehicle, origin and assembly.
    """
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
    def off_p(self): return 0
    @property
    def n_p(self): return self.F*self.R*self.O
    @property
    def off_tin(self): return self.off_p+self.n_p
    @property
    def n_tin(self): return self.F*self.R*self.O*self.A*self.T
    @property
    def off_a(self): return self.off_tin+self.n_tin
    @property
    def n_a(self): return self.F*self.A
    @property
    def off_tmarket(self): return self.off_a+self.n_a
    @property
    def n_tmarket(self): return self.F*self.A*self.T
    @property
    def off_ml(self): return self.off_tmarket+self.n_tmarket
    @property
    def n_ml(self): return self.F*self.R*self.O*self.A if self.mode=="modular" else 0
    @property
    def off_ms(self): return self.off_ml+self.n_ml
    @property
    def n_ms(self): return self.F*self.R*self.O*self.A if self.mode=="modular" else 0
    @property
    def n_vars(self): return self.off_ms+self.n_ms

    def p(self,v,r,o): return self.off_p+((v*self.R+r)*self.O+o)
    def tin(self,v,r,o,a,t): return self.off_tin+((((v*self.R+r)*self.O+o)*self.A+a)*self.T+t)
    def assembly(self,v,a): return self.off_a+v*self.A+a
    def tmarket(self,v,a,t): return self.off_tmarket+(v*self.A+a)*self.T+t
    def ml(self,v,r,o,a):
        if self.mode!="modular": raise ValueError("ml exists only in modular mode")
        return self.off_ml+(((v*self.R+r)*self.O+o)*self.A+a)
    def ms(self,v,r,o,a):
        if self.mode!="modular": raise ValueError("ms exists only in modular mode")
        return self.off_ms+(((v*self.R+r)*self.O+o)*self.A+a)
    # compatibility aliases
    def z1(self,v,o,a): return self.ml(v,0,o,a)
    def z2(self,v,o,a): return self.ms(v,0,o,a)
    def m10(self,v,o,a): return self.z1(v,o,a)
    def m5(self,v,o,a): return self.z2(v,o,a)


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
    module_parameters: pd.DataFrame
    plants: pd.DataFrame
    origins: pd.DataFrame
    assemblies: pd.DataFrame
    market: pd.Series
    scenario: pd.Series
    demand_values: np.ndarray
    item_ids: Tuple[str,...]
    item_index: Dict[str,int]
    battery_r: int
    bom_quantity: np.ndarray
    mass_per_unit: np.ndarray
    selected_vehicle_mass: np.ndarray
    supplier_cost: np.ndarray
    supplier_ef: np.ndarray
    supplier_capacity: np.ndarray
    process_cost_per_kg: np.ndarray
    process_ef_per_kg: np.ndarray
    final_assembly_cost_per_kg: np.ndarray
    final_assembly_ef_per_kg: np.ndarray
    final_assembly_share: float
    large_module_size: np.ndarray
    small_module_size: np.ndarray
    large_module_count: np.ndarray
    small_module_count: np.ndarray
    module_factors: Dict[str,np.ndarray]
    route: Dict[str,np.ndarray]
    emission_p: np.ndarray
    emission_tin: np.ndarray
    emission_a: np.ndarray
    emission_tmarket: np.ndarray
    emission_ml: np.ndarray
    emission_ms: np.ndarray
    selected_country_map: Dict[str,Tuple[str,...]]
    selected_transport_modes: Tuple[str,...]
    route_mode_codes: Tuple[str,...]
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
    processes: pd.DataFrame,
    plants: pd.DataFrame,
    selected_country_map: Optional[Mapping[str, Sequence[str]]],
) -> Dict[str, Tuple[str, ...]]:
    """Normalize item-specific Stage 1 and Stage 2 country selections.

    Keys are ``stage1::<item_id>`` and ``stage2::<item_id>``. The common vehicle
    assembly set is the intersection of all selected items' Stage 2 sets and is
    stored under ``assembly``.
    """
    all_plants = tuple(plants["location_name"].astype(str).tolist())
    result: Dict[str, Tuple[str, ...]] = {}
    common_stage2 = set(all_plants)
    for item_id in active_items:
        stage1_available = tuple(
            suppliers.loc[
                (suppliers["item_id"].astype(str) == item_id)
                & (pd.to_numeric(suppliers["active_default"], errors="coerce").fillna(0).astype(int) == 1),
                "location_name",
            ].astype(str).tolist()
        )
        stage2_available = tuple(
            processes.loc[
                (processes["item_id"].astype(str) == item_id)
                & (pd.to_numeric(processes["active_default"], errors="coerce").fillna(0).astype(int) == 1),
                "location_name",
            ].astype(str).tolist()
        )
        if not stage1_available:
            raise ValueError(f"{item_id}: Stage 1 생산지 데이터가 없습니다.")
        if not stage2_available:
            raise ValueError(f"{item_id}: Stage 2 공정국가 데이터가 없습니다.")

        req1 = None if selected_country_map is None else selected_country_map.get(f"stage1::{item_id}")
        req2 = None if selected_country_map is None else selected_country_map.get(f"stage2::{item_id}")
        chosen1 = tuple(name for name in stage1_available if req1 is None or name in {str(v) for v in req1})
        chosen2 = tuple(name for name in stage2_available if req2 is None or name in {str(v) for v in req2})
        if not chosen1:
            raise ValueError(f"{item_id}: Stage 1 생산지를 최소 1개 선택해야 합니다.")
        if not chosen2:
            raise ValueError(f"{item_id}: Stage 2 조립지를 최소 1개 선택해야 합니다.")
        result[f"stage1::{item_id}"] = chosen1
        result[f"stage2::{item_id}"] = chosen2
        common_stage2 &= set(chosen2)

    result["assembly"] = tuple(name for name in all_plants if name in common_stage2)
    if not result["assembly"]:
        raise ValueError(
            "선택된 모든 원료의 Stage 2 국가에 공통으로 포함되는 차량 조립지가 없습니다. "
            "각 원료의 Stage 2 국가선택에서 최소 1개의 공통 국가를 남기세요."
        )
    return result


def _normalize_transport_modes(selected_transport_modes: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if selected_transport_modes is None:
        chosen = set(ROUTE_MODE_CODES)
    else:
        chosen = {str(v) for v in selected_transport_modes}
    invalid = sorted(chosen - set(ROUTE_MODE_CODES))
    if invalid:
        raise ValueError("지원하지 않는 운송수단: " + ", ".join(invalid))
    ordered = tuple(code for code in ROUTE_MODE_CODES if code in chosen)
    if not ordered:
        raise ValueError("외부 운송수단을 최소 1개 선택해야 합니다.")
    return ordered


def _structure_signature(
    active_items: Sequence[str],
    selected_country_map: Mapping[str, Sequence[str]],
    selected_transport_modes: Sequence[str],
) -> str:
    payload = json.dumps(
        {
            "items": list(active_items),
            "countries": {k: list(v) for k, v in selected_country_map.items()},
            "transport_modes": list(selected_transport_modes),
        },
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
    selected_transport_modes: Optional[Sequence[str]] = None,
) -> FlexibleLPModel:
    """Build a dynamic all-item line/modular LP.

    Line: every selected item has origin==assembly and no external Stage-1→2 transport.
    Modular: every selected item uses fixed large/small module recipes, origin!=assembly,
    and mode-specific efficiency factors from module_parameters.csv.
    """
    (
        products,bom_all,catalog_all,suppliers_all,processes_all,module_all,demand,plants,
        transport,country_rules,markets,scenarios,metadata,
    )=ordered_tables(tables)

    active_items=_normalize_active_items(catalog_all,active_item_ids)
    items=catalog_all[catalog_all["item_id"].astype(str).isin(active_items)].sort_values("item_index").reset_index(drop=True)
    item_ids=tuple(items["item_id"].astype(str).tolist())
    item_index={k:i for i,k in enumerate(item_ids)}
    battery_ids=items.loc[items["item_type"].astype(str)=="battery","item_id"].astype(str).tolist()
    if len(battery_ids)!=1: raise ValueError("활성 제품구조에는 배터리가 정확히 1개 있어야 합니다.")
    battery_r=item_index[battery_ids[0]]
    bom=bom_all[bom_all["item_id"].astype(str).isin(active_items)].copy()
    suppliers=suppliers_all[suppliers_all["item_id"].astype(str).isin(active_items)].copy()
    processes=processes_all[processes_all["item_id"].astype(str).isin(active_items)].copy()
    module_parameters=module_all[module_all["item_id"].astype(str).isin(active_items)].copy()
    selected_map=_normalize_country_map(active_items,suppliers,processes,plants,selected_country_map)
    external_modes=_normalize_transport_modes(selected_transport_modes)
    route_mode_codes=tuple(["road"]+[m for m in external_modes if m!="road"])

    all_locations=plants["location_name"].astype(str).tolist()
    origin_union=set().union(*(set(selected_map[f"stage1::{i}"]) for i in active_items))
    origin_names=[n for n in all_locations if n in origin_union]
    base_assemblies=list(selected_map["assembly"])
    if production_mode=="line":
        assembly_names=[a for a in base_assemblies if all(a in set(selected_map[f"stage1::{i}"]) for i in active_items)]
        if not assembly_names:
            raise ValueError("라인 생산은 모든 선택 품목의 Stage 1 생산지와 Stage 2 조립지가 동일해야 합니다. 공통 국가가 없습니다.")
    else:
        assembly_names=[a for a in base_assemblies if all(any(o!=a for o in selected_map[f"stage1::{i}"]) for i in active_items)]
        if not assembly_names:
            raise ValueError("모듈 생산은 모든 선택 품목에 대해 생산지와 조립지가 달라야 합니다. 가능한 조립국가가 없습니다.")
    selected_map=dict(selected_map); selected_map["assembly"]=tuple(assembly_names)
    origins=plants[plants["location_name"].astype(str).isin(origin_names)].copy().reset_index(drop=True)
    assemblies=plants[plants["location_name"].astype(str).isin(assembly_names)].copy().reset_index(drop=True)

    F,R,O,A,T=len(products),len(items),len(origins),len(assemblies),len(route_mode_codes)
    layout=IndexLayout(production_mode,F,R,O,A,T)
    product_ids=products["product_id"].astype(str).tolist()
    demand_map=demand.groupby("product_id")["demand_units"].sum().to_dict()
    demand_values=np.asarray([float(demand_map[p]) for p in product_ids])

    bom_quantity=np.zeros((F,R)); mass_per_unit=np.zeros((F,R))
    for v,pid in enumerate(product_ids):
        for r,iid in enumerate(item_ids):
            row=bom[(bom["product_id"].astype(str)==pid)&(bom["item_id"].astype(str)==iid)]
            if len(row)!=1: raise ValueError(f"BOM 행 오류: {pid}/{iid}")
            bom_quantity[v,r]=float(row.iloc[0]["quantity_per_vehicle"])
            mass_per_unit[v,r]=float(row.iloc[0]["mass_per_unit_kg"])
    selected_vehicle_mass=np.sum(bom_quantity*mass_per_unit,axis=1)

    module_parameters=module_parameters.set_index("item_id").loc[list(item_ids)].reset_index()
    large_size=module_parameters["large_module_size"].astype(float).to_numpy()
    small_size=module_parameters["small_module_size"].astype(float).to_numpy()
    large_count=np.zeros((F,R)); small_count=np.zeros((F,R))
    for v in range(F):
        for r in range(R):
            q=bom_quantity[v,r]
            nl=math.floor((q+1e-10)/large_size[r]); rem=q-nl*large_size[r]; ns=round(rem/small_size[r])
            if abs(q-(nl*large_size[r]+ns*small_size[r]))>1e-7:
                raise ValueError(f"모듈 조합 불일치: {product_ids[v]}/{item_ids[r]}")
            large_count[v,r]=nl; small_count[v,r]=ns
    factor_names=[
        "large_stage1_cost_factor","small_stage1_cost_factor","large_stage1_ef_factor","small_stage1_ef_factor",
        "large_stage2_cost_factor","small_stage2_cost_factor","large_stage2_ef_factor","small_stage2_ef_factor",
    ]
    module_factors={n:module_parameters[n].astype(float).to_numpy() for n in factor_names}

    origin_names=origins["location_name"].astype(str).tolist(); assembly_names2=assemblies["location_name"].astype(str).tolist()
    supplier_cost=np.zeros((R,O)); supplier_ef=np.zeros((R,O)); supplier_capacity=np.zeros((R,O)); active_stage1=np.zeros((R,O),bool)
    for r,iid in enumerate(item_ids):
        chosen=set(selected_map[f"stage1::{iid}"])
        for o,name in enumerate(origin_names):
            row=suppliers[(suppliers["item_id"].astype(str)==iid)&(suppliers["location_name"].astype(str)==name)]
            if row.empty: continue
            supplier_cost[r,o]=float(row.iloc[0]["stage1_cost_per_unit"]); supplier_ef[r,o]=float(row.iloc[0]["stage1_ef_kgco2_per_unit"])
            supplier_capacity[r,o]=float(row.iloc[0]["capacity"]); active_stage1[r,o]=name in chosen

    process_cost=np.zeros((R,A)); process_ef=np.zeros((R,A))
    for r,iid in enumerate(item_ids):
        for a,name in enumerate(assembly_names2):
            row=processes[(processes["item_id"].astype(str)==iid)&(processes["location_name"].astype(str)==name)]
            if row.empty: raise ValueError(f"Stage 2 공정계수 누락: {iid}/{name}")
            process_cost[r,a]=float(row.iloc[0]["process_cost_per_unit"]); process_ef[r,a]=float(row.iloc[0]["process_ef_kgco2_per_unit"])

    final_assembly_share=0.15
    if "parameter_name" in metadata.columns:
        row=metadata[metadata["parameter_name"].astype(str)=="final_vehicle_assembly_share"]
        if not row.empty: final_assembly_share=float(row.iloc[0]["parameter_value"])
    final_assembly_cost=assemblies["assembly_cost_eur_per_kg"].astype(float).to_numpy()
    final_assembly_ef=assemblies["assembly_ef_kgco2_per_kg"].astype(float).to_numpy()

    market=markets.iloc[0]
    raw_dist,final_dist=cached_selected_distance_matrices(
        tuple((float(x.latitude),float(x.longitude)) for x in origins.itertuples()),
        tuple((float(x.latitude),float(x.longitude)) for x in assemblies.itertuples()),
        (float(market["latitude"]),float(market["longitude"])),float(market.get("minimum_distance_km",0.0)))
    route=build_route_matrices(origins,assemblies,market,transport,country_rules,raw_dist,final_dist,
                               route_mode_codes=route_mode_codes,selected_external_modes=external_modes)
    scenario=scenarios[scenarios["scenario_id"].astype(str)==str(scenario_id)].iloc[0].copy()
    score=float(scenario.get("minimum_score",0)); fleet_cap=np.nan
    if int(scenario.get("apply_carbon_cap",0))==1:
        fleet_cap=sum(carbon_cap_from_score(str(products.iloc[v]["vehicle_class"]),score)*demand_values[v] for v in range(F))
    scenario["fleet_total_cap_kgco2"]=fleet_cap

    c=np.zeros(layout.n_vars); lb=np.zeros(layout.n_vars); ub=np.full(layout.n_vars,np.inf)
    ep=np.zeros(layout.n_p); et=np.zeros(layout.n_tin); ea=np.zeros(layout.n_a); efinal=np.zeros(layout.n_tmarket)
    eml=np.zeros(layout.n_ml); ems=np.zeros(layout.n_ms)
    loss=items["loss_rate"].astype(float).to_numpy()

    # Final vehicle integration is common to both modes; item-specific Stage-2 processing is separate.
    for v in range(F):
        for a in range(A):
            idx=layout.assembly(v,a)
            c[idx]=selected_vehicle_mass[v]*final_assembly_cost[a]*final_assembly_share
            ea[idx-layout.off_a]=selected_vehicle_mass[v]*final_assembly_ef[a]*final_assembly_share
            for t in range(T):
                fidx=layout.tmarket(v,a,t)
                if not route["final_allowed"][a,t]: ub[fidx]=0; continue
                c[fidx]=selected_vehicle_mass[v]*route["final_cost_per_kg"][a,t]
                efinal[fidx-layout.off_tmarket]=selected_vehicle_mass[v]*route["final_ef_per_kg"][a,t]

    for v in range(F):
        for r in range(R):
            gross=1/(1-loss[r]); flow_mass=mass_per_unit[v,r]
            for o in range(O):
                pidx=layout.p(v,r,o)
                if not active_stage1[r,o]: ub[pidx]=0
                if production_mode=="line":
                    c[pidx]=supplier_cost[r,o]; ep[pidx-layout.off_p]=supplier_ef[r,o]*gross
                # Modular production cost/emission is carried by module variables, not P.
                for a in range(A):
                    same=origin_names[o]==assembly_names2[a]
                    for t in range(T):
                        tidx=layout.tin(v,r,o,a,t)
                        if not active_stage1[r,o]: ub[tidx]=0; continue
                        if production_mode=="line":
                            if not same or t!=INTERNAL_ROUTE_INDEX: ub[tidx]=0; continue
                            c[tidx]=0; et[tidx-layout.off_tin]=0
                        else:
                            if same or not route["raw_allowed"][o,a,t]: ub[tidx]=0; continue
                            c[tidx]=route["raw_cost_per_kg"][o,a,t]*flow_mass
                            et[tidx-layout.off_tin]=route["raw_ef_per_kg"][o,a,t]*flow_mass
                    if production_mode=="modular":
                        il=layout.ml(v,r,o,a); is_=layout.ms(v,r,o,a)
                        if (not active_stage1[r,o]) or same:
                            ub[il]=0; ub[is_]=0; continue
                        mass_l=large_size[r]*flow_mass; mass_s=small_size[r]*flow_mass
                        c[il]=large_size[r]*supplier_cost[r,o]*module_factors["large_stage1_cost_factor"][r] + mass_l*process_cost[r,a]*module_factors["large_stage2_cost_factor"][r]
                        c[is_]=small_size[r]*supplier_cost[r,o]*module_factors["small_stage1_cost_factor"][r] + mass_s*process_cost[r,a]*module_factors["small_stage2_cost_factor"][r]
                        eml[il-layout.off_ml]=large_size[r]*supplier_ef[r,o]*gross*module_factors["large_stage1_ef_factor"][r] + mass_l*process_ef[r,a]*module_factors["large_stage2_ef_factor"][r]
                        ems[is_-layout.off_ms]=small_size[r]*supplier_ef[r,o]*gross*module_factors["small_stage1_ef_factor"][r] + mass_s*process_ef[r,a]*module_factors["small_stage2_ef_factor"][r]

    # Line item-specific Stage-2 processing is charged on the vehicle assembly variable.
    if production_mode=="line":
        for v in range(F):
            for a in range(A):
                idx=layout.assembly(v,a)
                for r in range(R):
                    mass=bom_quantity[v,r]*mass_per_unit[v,r]
                    c[idx]+=mass*process_cost[r,a]
                    ea[idx-layout.off_a]+=mass*process_ef[r,a]

    rows=LinearConstraintBuilder(layout.n_vars)
    # 1 demand
    for v in range(F): rows.add_eq([layout.tmarket(v,a,t) for a in range(A) for t in range(T)],[1]*(A*T),demand_values[v])
    # 2 item structure, line or modular
    origin_by_name={n:o for o,n in enumerate(origin_names)}
    if production_mode=="line":
        for v in range(F):
            for r in range(R):
                q=bom_quantity[v,r]
                for a,name in enumerate(assembly_names2):
                    o=origin_by_name.get(name)
                    if o is None or not active_stage1[r,o]:
                        rows.add_eq([layout.assembly(v,a)],[1],0); continue
                    rows.add_eq([layout.tin(v,r,o,a,INTERNAL_ROUTE_INDEX),layout.assembly(v,a)],[1,-q],0)
    else:
        for v in range(F):
            for r in range(R):
                for o in range(O):
                    for a in range(A):
                        cols=[layout.tin(v,r,o,a,t) for t in range(T)]+[layout.ml(v,r,o,a),layout.ms(v,r,o,a)]
                        rows.add_eq(cols,[1]*T+[-large_size[r],-small_size[r]],0)
                for a in range(A):
                    rows.add_eq([layout.ml(v,r,o,a) for o in range(O)]+[layout.assembly(v,a)],[1]*O+[-large_count[v,r]],0)
                    rows.add_eq([layout.ms(v,r,o,a) for o in range(O)]+[layout.assembly(v,a)],[1]*O+[-small_count[v,r]],0)
    # 3 assembly output
    for v in range(F):
        for a in range(A): rows.add_eq([layout.assembly(v,a)]+[layout.tmarket(v,a,t) for t in range(T)],[1]+[-1]*T,0)
    # 4 production balance
    for v in range(F):
        for r in range(R):
            for o in range(O): rows.add_eq([layout.p(v,r,o)]+[layout.tin(v,r,o,a,t) for a in range(A) for t in range(T)],[1]+[-1]*(A*T),0)
    # 5 capacity
    for r in range(R):
        for o in range(O): rows.add_le([layout.p(v,r,o) for v in range(F)],[1]*F,supplier_capacity[r,o])
    # 6 carbon cap
    if int(scenario.get("apply_carbon_cap",0))==1:
        cols=[]; vals=[]
        for off,arr in [(layout.off_p,ep),(layout.off_tin,et),(layout.off_a,ea),(layout.off_tmarket,efinal),(layout.off_ml,eml),(layout.off_ms,ems)]:
            for j,val in enumerate(arr):
                if val: cols.append(off+j); vals.append(float(val))
        rows.add_le(cols,vals,float(fleet_cap))

    signature=_structure_signature(active_items,selected_map,external_modes)
    return FlexibleLPModel(
        layout,c,lb,ub,rows,products,items,bom,suppliers,processes,module_parameters,plants,origins,assemblies,market,scenario,
        demand_values,item_ids,item_index,battery_r,bom_quantity,mass_per_unit,selected_vehicle_mass,
        supplier_cost,supplier_ef,supplier_capacity,process_cost,process_ef,final_assembly_cost,final_assembly_ef,final_assembly_share,
        large_size,small_size,large_count,small_count,module_factors,route,ep,et,ea,efinal,eml,ems,
        {k:tuple(v) for k,v in selected_map.items()},tuple(external_modes),tuple(route_mode_codes),signature,
        rows.equality_count,rows.inequality_count,rows.nonzero_count,
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
    m=result.model
    out={
        "status":result.status,"message":result.message,"objective_value":result.objective_value,
        "wall_time_sec":result.wall_time_sec,"solver_name":result.solver_name,"solver_version":result.solver_version,
        "solver_iterations":result.iterations,"variable_count":m.layout.n_vars,
        "constraint_count":m.equality_count+m.inequality_count,"matrix_nonzeros":m.matrix_nonzeros,
        "scenario_id":str(m.scenario["scenario_id"]),"scenario_name":str(m.scenario["scenario_name"]),
        "production_mode":m.layout.mode,"production_mode_name":MODE_LABEL[m.layout.mode],
        "model_definition":"all selected items: line colocation / fixed two-level modular recipe",
        "active_item_ids":list(m.item_ids),"active_item_names":m.items["item_name_ko"].astype(str).tolist(),
        "selected_country_map":{k:list(v) for k,v in m.selected_country_map.items()},
        "selected_transport_modes":list(m.selected_transport_modes),"route_mode_codes":list(m.route_mode_codes),
        "active_index_sizes":{"F_vehicles":m.layout.F,"R_materials":m.layout.R,"O_stage1_origins":m.layout.O,"A_stage2_assemblies":m.layout.A,"T_transport_slots":m.layout.T},
        "structure_signature":m.structure_signature,"plants":m.plants,"origins":m.origins,"assemblies":m.assemblies,
        "market":m.market,"item_catalog":m.items,"module_parameters":m.module_parameters,
    }
    if result.x is None or result.status not in {"OPTIMAL","FEASIBLE"}: return out
    x=result.x; L=m.layout; meta=m.items.set_index("item_id"); onames=m.origins["location_name"].astype(str).tolist(); anames=m.assemblies["location_name"].astype(str).tolist()
    prod=[]; inbound=[]; assembly=[]; stage2=[]; market=[]; modules=[]

    # Modular rows first, because production and Stage-2 item cost are carried by ML/MS.
    modular_prod_cost={}; modular_prod_em={}; modular_proc_cost={}; modular_proc_em={}
    if L.mode=="modular":
        for v,product in m.products.iterrows():
            for r,iid in enumerate(m.item_ids):
                for o in range(L.O):
                    for a in range(L.A):
                        for kind,idx,size,cost_s1,ef_s1,cost_s2,ef_s2 in [
                            ("대형",L.ml(v,r,o,a),m.large_module_size[r],"large_stage1_cost_factor","large_stage1_ef_factor","large_stage2_cost_factor","large_stage2_ef_factor"),
                            ("소형",L.ms(v,r,o,a),m.small_module_size[r],"small_stage1_cost_factor","small_stage1_ef_factor","small_stage2_cost_factor","small_stage2_ef_factor"),
                        ]:
                            n=float(x[idx])
                            if not _positive(n): continue
                            flow=n*size; mass=flow*m.mass_per_unit[v,r]; gross=1/(1-float(meta.loc[iid,"loss_rate"]))
                            s1c=flow*m.supplier_cost[r,o]*m.module_factors[cost_s1][r]
                            s1e=flow*m.supplier_ef[r,o]*gross*m.module_factors[ef_s1][r]
                            s2c=mass*m.process_cost_per_kg[r,a]*m.module_factors[cost_s2][r]
                            s2e=mass*m.process_ef_per_kg[r,a]*m.module_factors[ef_s2][r]
                            key1=(v,r,o); key2=(v,r,a)
                            modular_prod_cost[key1]=modular_prod_cost.get(key1,0)+s1c; modular_prod_em[key1]=modular_prod_em.get(key1,0)+s1e
                            modular_proc_cost[key2]=modular_proc_cost.get(key2,0)+s2c; modular_proc_em[key2]=modular_proc_em.get(key2,0)+s2e
                            modules.append({"product_id":product["product_id"],"product_name_ko":product["product_name_ko"],"item_id":iid,"item_name_ko":meta.loc[iid,"item_name_ko"],"origin_location":onames[o],"assembly_location":anames[a],"module_level":kind,"module_size":size,"module_unit":meta.loc[iid,"flow_unit"],"module_count_equivalent":n,"total_flow_amount":flow,"stage1_module_cost_eur":s1c,"stage1_module_emissions_kgco2":s1e,"stage2_module_process_cost_eur":s2c,"stage2_module_process_emissions_kgco2":s2e})

    for v,product in m.products.iterrows():
        pid=str(product["product_id"])
        for r,iid in enumerate(m.item_ids):
            item=meta.loc[iid]; gross=1/(1-float(item["loss_rate"]))
            for o in range(L.O):
                amount=float(x[L.p(v,r,o)])
                if _positive(amount):
                    if L.mode=="line": basec=amount*m.supplier_cost[r,o]; basee=amount*m.supplier_ef[r,o]*gross
                    else: basec=modular_prod_cost.get((v,r,o),0); basee=modular_prod_em.get((v,r,o),0)
                    prod.append({"product_id":pid,"product_name_ko":product["product_name_ko"],"item_id":iid,"item_name_ko":item["item_name_ko"],"item_type":item["item_type"],"origin_index":int(m.origins.iloc[o]["location_index"]),"origin_model_index":o+1,"origin_location":onames[o],"net_output_amount":amount,"gross_input_amount_after_loss":amount*gross,"flow_unit":item["flow_unit"],"stage1_cost_eur":basec,"stage1_emissions_kgco2":basee,"stage1_process":item["stage1_process_name_ko"]})
                for a in range(L.A):
                    for t in range(L.T):
                        flow=float(x[L.tin(v,r,o,a,t)])
                        if not _positive(flow): continue
                        mass=flow*m.mass_per_unit[v,r]
                        inbound.append({"product_id":pid,"product_name_ko":product["product_name_ko"],"item_id":iid,"item_name_ko":item["item_name_ko"],"item_type":item["item_type"],"origin_index":int(m.origins.iloc[o]["location_index"]),"origin_model_index":o+1,"origin_location":onames[o],"assembly_index":int(m.assemblies.iloc[a]["location_index"]),"assembly_model_index":a+1,"assembly_location":anames[a],"transport_mode_index":ROUTE_MODE_CODES.index(m.route_mode_codes[t])+1,"transport_mode":m.route_mode_codes[t],"transport_mode_ko":ROUTE_MODE_LABEL[m.route_mode_codes[t]],"flow_amount":flow,"flow_unit":item["flow_unit"],"transport_mass_kg":mass,"distance_km":float(m.route["raw_total_km"][o,a,t]),"transport_cost_eur":flow*m.c[L.tin(v,r,o,a,t)],"transport_emissions_kgco2":flow*m.emission_tin[L.tin(v,r,o,a,t)-L.off_tin],"internal_flow":bool(onames[o]==anames[a])})
        for a in range(L.A):
            vehicles=float(x[L.assembly(v,a)])
            if _positive(vehicles):
                finalc=vehicles*m.selected_vehicle_mass[v]*m.final_assembly_cost_per_kg[a]*m.final_assembly_share
                finale=vehicles*m.selected_vehicle_mass[v]*m.final_assembly_ef_per_kg[a]*m.final_assembly_share
                itemc=0; iteme=0
                for r,iid in enumerate(m.item_ids):
                    masspv=m.bom_quantity[v,r]*m.mass_per_unit[v,r]
                    if L.mode=="line": pc=vehicles*masspv*m.process_cost_per_kg[r,a]; pe=vehicles*masspv*m.process_ef_per_kg[r,a]
                    else: pc=modular_proc_cost.get((v,r,a),0); pe=modular_proc_em.get((v,r,a),0)
                    itemc+=pc; iteme+=pe
                    stage2.append({"product_id":pid,"product_name_ko":product["product_name_ko"],"item_id":iid,"item_name_ko":meta.loc[iid,"item_name_ko"],"assembly_index":int(m.assemblies.iloc[a]["location_index"]),"assembly_model_index":a+1,"assembly_location":anames[a],"vehicle_equivalents":vehicles,"quantity_per_vehicle":m.bom_quantity[v,r],"flow_unit":meta.loc[iid,"flow_unit"],"stage2_processed_mass_kg":vehicles*masspv,"stage2_process":meta.loc[iid,"stage2_process_name_ko"],"stage2_cost_eur":pc,"stage2_emissions_kgco2":pe})
                assembly.append({"product_id":pid,"product_name_ko":product["product_name_ko"],"assembly_index":int(m.assemblies.iloc[a]["location_index"]),"assembly_model_index":a+1,"assembly_location":anames[a],"vehicle_equivalents":vehicles,"selected_vehicle_mass_kg":m.selected_vehicle_mass[v],"item_process_cost_eur":itemc,"item_process_emissions_kgco2":iteme,"final_vehicle_assembly_cost_eur":finalc,"final_vehicle_assembly_emissions_kgco2":finale,"total_stage2_cost_eur":itemc+finalc,"total_stage2_emissions_kgco2":iteme+finale})
            for t in range(L.T):
                qty=float(x[L.tmarket(v,a,t)])
                if not _positive(qty): continue
                market.append({"product_id":pid,"product_name_ko":product["product_name_ko"],"assembly_index":int(m.assemblies.iloc[a]["location_index"]),"assembly_model_index":a+1,"assembly_location":anames[a],"market_id":m.market["market_id"],"market_name":m.market["market_name"],"transport_mode_index":ROUTE_MODE_CODES.index(m.route_mode_codes[t])+1,"transport_mode":m.route_mode_codes[t],"transport_mode_ko":ROUTE_MODE_LABEL[m.route_mode_codes[t]],"vehicle_equivalents":qty,"selected_vehicle_mass_kg":m.selected_vehicle_mass[v],"distance_km":float(m.route["final_total_km"][a,t]),"transport_cost_eur":qty*m.c[L.tmarket(v,a,t)],"transport_emissions_kgco2":qty*m.emission_tmarket[L.tmarket(v,a,t)-L.off_tmarket]})

    production_df=pd.DataFrame(prod); inbound_df=pd.DataFrame(inbound); assembly_df=pd.DataFrame(assembly); stage2_df=pd.DataFrame(stage2); market_df=pd.DataFrame(market); module_df=pd.DataFrame(modules)
    s1c=float(production_df.get("stage1_cost_eur",pd.Series(dtype=float)).sum()); s1e=float(production_df.get("stage1_emissions_kgco2",pd.Series(dtype=float)).sum())
    inc=float(inbound_df.get("transport_cost_eur",pd.Series(dtype=float)).sum()); ine=float(inbound_df.get("transport_emissions_kgco2",pd.Series(dtype=float)).sum())
    itemc=float(stage2_df.get("stage2_cost_eur",pd.Series(dtype=float)).sum()); iteme=float(stage2_df.get("stage2_emissions_kgco2",pd.Series(dtype=float)).sum())
    finalc=float(assembly_df.get("final_vehicle_assembly_cost_eur",pd.Series(dtype=float)).sum()); finale=float(assembly_df.get("final_vehicle_assembly_emissions_kgco2",pd.Series(dtype=float)).sum())
    outc=float(market_df.get("transport_cost_eur",pd.Series(dtype=float)).sum()); oute=float(market_df.get("transport_emissions_kgco2",pd.Series(dtype=float)).sum())
    totalc=s1c+inc+itemc+finalc+outc; totale=s1e+ine+iteme+finale+oute
    product_rows=[]
    for v,product in m.products.iterrows():
        pid=str(product["product_id"]); d=float(m.demand_values[v]); pc=pe=0
        for df,cc,ee in [(production_df,"stage1_cost_eur","stage1_emissions_kgco2"),(inbound_df,"transport_cost_eur","transport_emissions_kgco2"),(assembly_df,"total_stage2_cost_eur","total_stage2_emissions_kgco2"),(market_df,"transport_cost_eur","transport_emissions_kgco2")]:
            if not df.empty:
                sub=df[df["product_id"].astype(str)==pid]; pc+=float(sub[cc].sum()); pe+=float(sub[ee].sum())
        cap=carbon_cap_from_score(str(product["vehicle_class"]),float(m.scenario["minimum_score"])) if int(m.scenario.get("apply_carbon_cap",0))==1 else np.nan
        product_rows.append({"product_id":pid,"product_name_ko":product["product_name_ko"],"vehicle_class":product["vehicle_class"],"demand_units":d,"reference_vehicle_mass_kg":product["reference_vehicle_mass_kg"],"selected_structure_vehicle_mass_kg":m.selected_vehicle_mass[v],"mass_change_kg":m.selected_vehicle_mass[v]-float(product["reference_vehicle_mass_kg"]),"total_cost_eur":pc,"total_emissions_kgco2":pe,"emissions_per_vehicle_kgco2":pe/d if d else np.nan,"reference_cap_kgco2_per_vehicle":cap,"reference_cap_met_individually":True if not np.isfinite(cap) else pe/d<=cap+1e-6})
    structure=[]
    for v,p in m.products.iterrows():
        for r,item in m.items.iterrows():
            structure.append({"product_id":p["product_id"],"product_name_ko":p["product_name_ko"],"item_id":item["item_id"],"item_name_ko":item["item_name_ko"],"item_type":item["item_type"],"quantity_per_vehicle":m.bom_quantity[v,r],"flow_unit":item["flow_unit"],"mass_kg_per_vehicle":m.bom_quantity[v,r]*m.mass_per_unit[v,r],"large_module_size":m.large_module_size[r],"small_module_size":m.small_module_size[r],"large_modules_per_vehicle":m.large_module_count[v,r],"small_modules_per_vehicle":m.small_module_count[v,r],"stage1_process":item["stage1_process_name_ko"],"stage2_process":item["stage2_process_name_ko"]})
    fleet=float(m.scenario.get("fleet_total_cap_kgco2",np.nan))
    out.update({"production_summary":production_df,"inbound_routes":inbound_df,"assembly_summary":assembly_df,"stage2_item_process_summary":stage2_df,"market_routes":market_df,"module_summary":module_df,"product_summary":pd.DataFrame(product_rows),"product_structure_summary":pd.DataFrame(structure),"cost_breakdown":{"Stage 1 품목 생산/모듈 생산":s1c,"Stage 1→2 운송":inc,"Stage 2 품목 가공·조립":itemc,"Stage 2 최종 차량 조립":finalc,"Stage 2→3 시장 출시 운송":outc},"emission_breakdown":{"Stage 1 품목 생산/모듈 생산":s1e,"Stage 1→2 운송":ine,"Stage 2 품목 가공·조립":iteme,"Stage 2 최종 차량 조립":finale,"Stage 2→3 시장 출시 운송":oute},"supply_chain_cost_eur":totalc,"total_emissions_kgco2":totale,"objective_reconstruction_gap_eur":float(result.objective_value or 0)-totalc,"fleet_total_cap_kgco2":fleet,"fleet_cap_slack_kgco2":fleet-totale if np.isfinite(fleet) else np.nan,"fleet_cap_utilization_pct":100*totale/fleet if np.isfinite(fleet) and fleet else np.nan,"fleet_cap_met":True if not np.isfinite(fleet) else totale<=fleet+1e-5,"line_all_item_colocation":L.mode=="line","modular_all_item_separation":L.mode=="modular","modular_same_country_positive_flow_count":int(inbound_df.get("internal_flow",pd.Series(dtype=bool)).astype(bool).sum()) if L.mode=="modular" and not inbound_df.empty else 0})
    return out



def solve_case(
    tables: Mapping[str, pd.DataFrame],
    scenario_id: str,
    production_mode: str,
    time_limit_sec: int,
    active_item_ids: Sequence[str],
    selected_country_map: Mapping[str, Sequence[str]],
    selected_transport_modes: Sequence[str],
) -> Dict:
    model = build_flexible_lp_model(
        tables,
        production_mode=production_mode,
        scenario_id=scenario_id,
        active_item_ids=active_item_ids,
        selected_country_map=selected_country_map,
        selected_transport_modes=selected_transport_modes,
    )
    result = solve_lp_model(model, time_limit_sec=time_limit_sec)
    return extract_solution(result)


# -----------------------------------------------------------------------------
# Stage-separated mapping
# -----------------------------------------------------------------------------
def _bezier_curve_points(
    start: Tuple[float, float],
    end: Tuple[float, float],
    bend: float,
    steps: int = 44,
):
    """Return a quadratic Bézier route used as a visually separated lane.

    ``bend`` is dimensionless relative to the start/end distance. Different items receive
    different bends, so routes sharing the same origin and destination do not hide one another.
    """
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
        return 4.0
    low, high = float(positives.min()), float(positives.max())
    ratio = (math.sqrt(max(value, 0.0)) - math.sqrt(low)) / max(1e-12, math.sqrt(high) - math.sqrt(low))
    return float(2.2 + max(0.0, min(1.0, ratio)) * 6.2)


def _production_radius(values: pd.Series, value: float) -> float:
    positives = pd.to_numeric(values, errors="coerce").dropna()
    positives = positives[positives > 0]
    if positives.empty or positives.nunique() == 1:
        return 9.0
    low, high = float(positives.min()), float(positives.max())
    ratio = (math.sqrt(max(value, 0.0)) - math.sqrt(low)) / max(1e-12, math.sqrt(high) - math.sqrt(low))
    return float(6.0 + max(0.0, min(1.0, ratio)) * 13.0)


def _assembly_radius(values: pd.Series, value: float) -> float:
    positives = pd.to_numeric(values, errors="coerce").dropna()
    positives = positives[positives > 0]
    if positives.empty or positives.nunique() == 1:
        return 10.0
    low, high = float(positives.min()), float(positives.max())
    ratio = (math.sqrt(max(value, 0.0)) - math.sqrt(low)) / max(1e-12, math.sqrt(high) - math.sqrt(low))
    return float(7.0 + max(0.0, min(1.0, ratio)) * 12.0)


def _offset_marker_coordinate(
    lat: float,
    lon: float,
    rank: int,
    count: int,
    radius_deg: float = 0.34,
) -> Tuple[float, float]:
    if count <= 1:
        return float(lat), float(lon)
    angle = 2.0 * math.pi * (rank / count)
    lat_offset = radius_deg * math.sin(angle)
    lon_scale = max(0.35, math.cos(math.radians(float(lat))))
    lon_offset = radius_deg * math.cos(angle) / lon_scale
    return float(lat) + lat_offset, float(lon) + lon_offset


def _point_on_polyline(points: Sequence[Tuple[float, float]], fraction: float) -> Tuple[float, float, float]:
    fraction = max(0.0, min(1.0, float(fraction)))
    if len(points) < 2:
        lat, lon = points[0]
        return float(lat), float(lon), 0.0
    pos = fraction * (len(points) - 1)
    idx = min(len(points) - 2, max(0, int(math.floor(pos))))
    alpha = pos - idx
    p0, p1 = points[idx], points[idx + 1]
    lat = float(p0[0]) + alpha * (float(p1[0]) - float(p0[0]))
    lon = float(p0[1]) + alpha * (float(p1[1]) - float(p0[1]))
    mean_lat = math.radians((float(p0[0]) + float(p1[0])) / 2.0)
    dx = (float(p1[1]) - float(p0[1])) * math.cos(mean_lat)
    dy = -(float(p1[0]) - float(p0[0]))
    angle = math.degrees(math.atan2(dy, dx))
    return lat, lon, angle


def _add_direction_arrow(
    target,
    points: Sequence[Tuple[float, float]],
    color: str,
    tooltip: str,
    fraction: float,
    size: int = 20,
):
    import folium
    if len(points) < 2:
        return
    lat, lon, angle = _point_on_polyline(points, fraction)
    html = (
        f'<div style="width:{size}px;height:{size}px;line-height:{size}px;text-align:center;'
        f'color:{color};font-size:{size}px;font-weight:900;'
        'text-shadow:-2px -2px 2px white,2px -2px 2px white,-2px 2px 2px white,2px 2px 2px white;'
        f'transform:rotate({angle:.2f}deg);transform-origin:center center;pointer-events:none;">➤</div>'
    )
    folium.Marker(
        [lat, lon],
        icon=folium.DivIcon(html=html, icon_size=(size, size), icon_anchor=(size // 2, size // 2)),
        tooltip=tooltip,
    ).add_to(target)


def _add_route_line(
    target,
    points: Sequence[Tuple[float, float]],
    color: str,
    weight: float,
    dash_array: Optional[str],
    tooltip: str,
    arrival_tooltip: str,
):
    """Draw a white route casing, colored lane, and only three clear arrows.

    The previous repeated arrow text covered nearby colored routes. Three spaced arrows preserve
    direction while keeping each item lane readable.
    """
    import folium
    folium.PolyLine(
        list(points), color="#ffffff", weight=float(weight) + 3.2, opacity=0.86,
        dash_array=None, interactive=False,
    ).add_to(target)
    line = folium.PolyLine(
        list(points), color=color, weight=float(weight), opacity=0.88,
        dash_array=dash_array, tooltip=tooltip,
    ).add_to(target)
    for fraction, size in ((0.36, 17), (0.64, 17), (0.86, 23)):
        _add_direction_arrow(target, points, color, arrival_tooltip, fraction=fraction, size=size)
    return line


def _active_item_table(result: Dict, visible_item_ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
    catalog = result.get("item_catalog", pd.DataFrame()).copy()
    if catalog.empty:
        return catalog
    order = {item_id: idx for idx, item_id in enumerate(result.get("active_item_ids", []))}
    catalog["_order"] = catalog["item_id"].astype(str).map(order).fillna(10_000)
    if visible_item_ids is not None:
        visible = {str(v) for v in visible_item_ids}
        catalog = catalog[catalog["item_id"].astype(str).isin(visible)]
    return catalog.sort_values(["_order", "item_index"], kind="stable").reset_index(drop=True)


def _add_stage1_layer(
    result: Dict,
    target,
    visible_item_ids: Optional[Sequence[str]] = None,
    show_empty_marker: bool = True,
):
    import folium
    plants = result["plants"]
    full_catalog = _active_item_table(result, None)
    catalog = _active_item_table(result, visible_item_ids)
    if catalog.empty or full_catalog.empty:
        return
    item_catalog = catalog.set_index("item_id")
    item_order = catalog["item_id"].astype(str).tolist()
    full_order = full_catalog["item_id"].astype(str).tolist()
    global_rank = {v: i for i, v in enumerate(full_order)}
    production = result.get("production_summary", pd.DataFrame()).copy()
    if production.empty:
        if show_empty_marker:
            folium.Marker([35, 25], tooltip="Stage 1 양의 생산량 없음").add_to(target)
        return
    production = production[production["item_id"].astype(str).isin(item_order)]
    if production.empty:
        return
    agg = production.groupby(
        ["origin_index", "origin_location", "item_id", "item_name_ko", "flow_unit"], as_index=False
    ).agg(
        quantity=("net_output_amount", "sum"),
        cost_eur=("stage1_cost_eur", "sum"),
        emissions_kgco2=("stage1_emissions_kgco2", "sum"),
    )
    item_values = {item_id: group["quantity"] for item_id, group in agg.groupby("item_id", sort=False)}
    for origin_index, location_group in agg.groupby("origin_index", sort=False):
        rows = location_group.copy()
        rows["_rank"] = rows["item_id"].astype(str).map(global_rank)
        rows = rows.sort_values("_rank")
        loc = plants.iloc[int(origin_index) - 1]
        for _, row in rows.iterrows():
            item_id = str(row["item_id"])
            color = str(item_catalog.loc[item_id, "color_hex"])
            marker_lat, marker_lon = _offset_marker_coordinate(
                float(loc["latitude"]), float(loc["longitude"]),
                global_rank[item_id], len(full_order), radius_deg=0.42
            )
            folium.CircleMarker(
                [marker_lat, marker_lon],
                radius=_production_radius(item_values[item_id], float(row["quantity"])),
                color=color, weight=2.5, fill=True, fill_color=color, fill_opacity=0.80,
                tooltip=(
                    f"Stage 1 생산지: {row['origin_location']}<br>"
                    f"원료·원자재: {row['item_name_ko']}<br>"
                    f"생산량: {row['quantity']:,.2f} {row['flow_unit']}<br>"
                    f"비용: €{row['cost_eur']:,.0f}<br>"
                    f"탄소발자국: {row['emissions_kgco2']:,.0f} kg CO₂-eq<br>"
                    "원의 크기: 동일 원료 내 국가별 생산량"
                ),
            ).add_to(target)


def _add_stage2_layer(result: Dict, target):
    """Show only actual vehicle-assembly locations using dark navy, not material green."""
    import folium
    plants = result["plants"]
    assembly = result.get("assembly_summary", pd.DataFrame()).copy()
    if assembly.empty:
        return
    agg = assembly.groupby(["assembly_index", "assembly_location"], as_index=False).agg(
        vehicles=("vehicle_equivalents", "sum"),
        stage2_cost=("total_stage2_cost_eur", "sum"),
        stage2_emissions=("total_stage2_emissions_kgco2", "sum"),
    )
    for _, row in agg.iterrows():
        loc = plants.iloc[int(row["assembly_index"]) - 1]
        folium.CircleMarker(
            [float(loc["latitude"]), float(loc["longitude"])],
            radius=_assembly_radius(agg["vehicles"], float(row["vehicles"])),
            color="#ffffff", weight=3.5, fill=True, fill_color=ASSEMBLY_COLOR, fill_opacity=0.96,
            tooltip=(
                f"Stage 2 차량 조립지: {row['assembly_location']}<br>"
                f"차량 조립량: {row['vehicles']:,.1f}대<br>"
                f"Stage 2 비용: €{row['stage2_cost']:,.0f}<br>"
                f"Stage 2 탄소발자국: {row['stage2_emissions']:,.0f} kg CO₂-eq<br>"
                "진한 남색 원 = 차량 조립지"
            ),
        ).add_to(target)


def _add_stage3_layer(result: Dict, target):
    import folium
    market = result["market"]
    folium.Marker(
        [float(market["latitude"]), float(market["longitude"])],
        icon=folium.Icon(color=MARKET_COLOR, icon="shopping-cart", prefix="fa"),
        tooltip=f"Stage 3 프랑스 시장: {market['market_name']}",
    ).add_to(target)


def _route_bend(item_rank: int, item_count: int, mode_index: int) -> float:
    item_center = item_rank - (item_count - 1) / 2.0
    mode_center = (int(mode_index) - 1) - (len(ROUTE_MODE_CODES) - 1) / 2.0
    return float(item_center * 0.095 + mode_center * 0.008)


def _add_stage12_route_layer(
    result: Dict,
    route_target,
    visible_item_ids: Optional[Sequence[str]] = None,
    show_origin_markers: bool = True,
    show_stage2_markers: bool = True,
):
    import folium
    plants = result["plants"]
    full_catalog = _active_item_table(result, None)
    catalog = _active_item_table(result, visible_item_ids)
    if catalog.empty or full_catalog.empty:
        return
    item_catalog = catalog.set_index("item_id")
    item_order = catalog["item_id"].astype(str).tolist()
    full_order = full_catalog["item_id"].astype(str).tolist()
    rank_map = {v: i for i, v in enumerate(full_order)}
    inbound = result.get("inbound_routes", pd.DataFrame()).copy()
    if show_origin_markers:
        _add_stage1_layer(result, route_target, item_order, show_empty_marker=False)
    if not inbound.empty:
        external = inbound[
            (~inbound["internal_flow"].astype(bool))
            & inbound["item_id"].astype(str).isin(item_order)
        ].copy()
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
                item_id = str(row["item_id"])
                item_rank = rank_map[item_id]
                color = str(item_catalog.loc[item_id, "color_hex"])
                # Each material uses its own visual departure/arrival lane around the real locations.
                origin_lat, origin_lon = _offset_marker_coordinate(
                    float(origin["latitude"]), float(origin["longitude"]),
                    item_rank, len(full_order), radius_deg=0.42,
                )
                dest_lat, dest_lon = _offset_marker_coordinate(
                    float(destination["latitude"]), float(destination["longitude"]),
                    item_rank, len(full_order), radius_deg=0.55,
                )
                bend = _route_bend(item_rank, len(full_order), int(row["transport_mode_index"]))
                curve = _bezier_curve_points(
                    (origin_lat, origin_lon), (dest_lat, dest_lon), bend,
                )
                tooltip = (
                    f"Stage 1→2 | {row['item_name_ko']}<br>"
                    f"{row['origin_location']} → {row['assembly_location']}<br>"
                    f"운송수단: {row['transport_mode_ko']}<br>"
                    f"운송질량: {row['transport_mass_kg']:,.1f} kg<br>"
                    f"운송비용: €{row['transport_cost_eur']:,.0f}<br>"
                    f"운송 탄소발자국: {row['transport_emissions_kgco2']:,.0f} kg CO₂-eq"
                )
                _add_route_line(
                    route_target, curve, color=color,
                    weight=_flow_width(agg["transport_mass_kg"], float(row["transport_mass_kg"])),
                    dash_array=TRANSPORT_DASH.get(int(row["transport_mode_index"])),
                    tooltip=tooltip,
                    arrival_tooltip=f"Stage 2 도착: {row['assembly_location']} · {row['item_name_ko']}",
                )
                folium.CircleMarker(
                    [dest_lat, dest_lon], radius=4.0, color="#ffffff", weight=2,
                    fill=True, fill_color=color, fill_opacity=0.95,
                    tooltip=f"Stage 2 원료 도착점: {row['assembly_location']} · {row['item_name_ko']}",
                ).add_to(route_target)
    if show_stage2_markers:
        _add_stage2_layer(result, route_target)


def _add_stage23_route_layer(
    result: Dict,
    route_target,
    show_stage2_markers: bool = True,
    show_market_marker: bool = True,
):
    plants = result["plants"]
    market = result["market"]
    routes = result.get("market_routes", pd.DataFrame()).copy()
    if show_market_marker:
        _add_stage3_layer(result, route_target)
    if routes.empty:
        return
    agg = routes.groupby(
        ["assembly_index", "assembly_location", "transport_mode_index", "transport_mode_ko"],
        as_index=False,
    ).agg(
        vehicles=("vehicle_equivalents", "sum"),
        cost_eur=("transport_cost_eur", "sum"),
        emissions_kgco2=("transport_emissions_kgco2", "sum"),
    )
    if show_stage2_markers:
        _add_stage2_layer(result, route_target)
    mode_count = max(1, agg["transport_mode_index"].nunique())
    for _, row in agg.iterrows():
        loc = plants.iloc[int(row["assembly_index"]) - 1]
        mode_rank = sorted(agg["transport_mode_index"].unique()).index(row["transport_mode_index"])
        bend = (mode_rank - (mode_count - 1) / 2.0) * 0.085
        curve = _bezier_curve_points(
            (float(loc["latitude"]), float(loc["longitude"])),
            (float(market["latitude"]), float(market["longitude"])), bend,
        )
        tooltip = (
            "Stage 2→3 | 완성 전기자동차<br>"
            f"{row['assembly_location']} → {market['market_name']}<br>"
            f"운송수단: {row['transport_mode_ko']}<br>"
            f"차량: {row['vehicles']:,.1f}대<br>"
            f"운송비용: €{row['cost_eur']:,.0f}<br>"
            f"운송 탄소발자국: {row['emissions_kgco2']:,.0f} kg CO₂-eq"
        )
        _add_route_line(
            route_target, curve, color=FINISHED_COLOR,
            weight=_flow_width(agg["vehicles"], float(row["vehicles"])),
            dash_array=TRANSPORT_DASH.get(int(row["transport_mode_index"])),
            tooltip=tooltip, arrival_tooltip=f"Stage 3 도착: {market['market_name']}",
        )


def _add_map_legend(
    fmap,
    result: Dict,
    view_code: int,
    visible_item_ids: Optional[Sequence[str]] = None,
):
    import folium
    catalog = _active_item_table(result, visible_item_ids)
    item_rows = []
    for _, row in catalog.iterrows():
        item_rows.append(
            f"<div><span style='display:inline-block;width:13px;height:13px;border-radius:50%;"
            f"background:{row['color_hex']};border:1px solid #444;margin-right:6px;'></span>"
            f"{row['item_name_ko']}</div>"
        )
    item_html = "".join(item_rows)
    html = f"""
    <div style="position:fixed;bottom:24px;left:42px;z-index:9999;background:white;
                border:2px solid #666;border-radius:7px;padding:10px 13px;font-size:12px;
                max-height:330px;overflow:auto;box-shadow:0 1px 6px rgba(0,0,0,.28);">
      <b>표시 원료·원자재 색상</b><br>{item_html}
      <hr style='margin:7px 0'>
      <div><span style='color:{ASSEMBLY_COLOR};font-size:18px'>●</span> Stage 2 차량 조립지(진한 남색)</div>
      <div><span style='color:{FINISHED_COLOR};font-size:18px'>━</span> Stage 2→3 완성차 운송</div>
      <div><span style='font-size:17px;font-weight:900'>➤</span> 화살표 방향 = 도착지</div>
      <div>흰 외곽선 = 겹치는 경로 구분</div>
      <div>Stage 1 원 크기 = 동일 원료 내 생산량</div>
      <div>Stage 2 원 크기 = 차량 조립량</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(html))


def build_stage_map(
    result: Dict,
    view_code: int,
    visible_item_ids: Optional[Sequence[str]] = None,
):
    import folium
    from folium.plugins import Fullscreen
    fmap = folium.Map(location=[35, 25], zoom_start=2, tiles="CartoDB positron")
    Fullscreen(position="topleft").add_to(fmap)
    catalog = _active_item_table(result, visible_item_ids)
    visible = catalog["item_id"].astype(str).tolist()

    if view_code == 0:
        for _, item in catalog.iterrows():
            iid, name = str(item["item_id"]), str(item["item_name_ko"])
            g1 = folium.FeatureGroup(name=f"Stage 1 생산 · {name}", show=True).add_to(fmap)
            _add_stage1_layer(result, g1, [iid], show_empty_marker=False)
            g12 = folium.FeatureGroup(name=f"Stage 1→2 운송 · {name}", show=True).add_to(fmap)
            _add_stage12_route_layer(result, g12, [iid], show_origin_markers=False, show_stage2_markers=False)
        g2 = folium.FeatureGroup(name="Stage 2 차량 조립지", show=True).add_to(fmap)
        g23 = folium.FeatureGroup(name="Stage 2→3 완성차 운송", show=True).add_to(fmap)
        g3 = folium.FeatureGroup(name="Stage 3 프랑스 시장", show=True).add_to(fmap)
        _add_stage2_layer(result, g2)
        _add_stage23_route_layer(result, g23, show_stage2_markers=False, show_market_marker=False)
        _add_stage3_layer(result, g3)
        folium.LayerControl(collapsed=False, position="topright").add_to(fmap)
    elif view_code == 1:
        for _, item in catalog.iterrows():
            iid, name = str(item["item_id"]), str(item["item_name_ko"])
            group = folium.FeatureGroup(name=name, show=True).add_to(fmap)
            _add_stage1_layer(result, group, [iid])
        folium.LayerControl(collapsed=False, position="topright").add_to(fmap)
    elif view_code == 12:
        for _, item in catalog.iterrows():
            iid, name = str(item["item_id"]), str(item["item_name_ko"])
            group = folium.FeatureGroup(name=f"{name} 경로", show=True).add_to(fmap)
            _add_stage12_route_layer(result, group, [iid], show_origin_markers=True, show_stage2_markers=False)
        g2 = folium.FeatureGroup(name="Stage 2 차량 조립지", show=True).add_to(fmap)
        _add_stage2_layer(result, g2)
        folium.LayerControl(collapsed=False, position="topright").add_to(fmap)
    elif view_code == 2:
        _add_stage2_layer(result, fmap)
    elif view_code == 23:
        _add_stage23_route_layer(result, fmap, show_stage2_markers=True, show_market_marker=True)
    elif view_code == 3:
        _add_stage3_layer(result, fmap)
    else:
        raise ValueError(f"지원하지 않는 지도 보기: {view_code}")

    _add_map_legend(fmap, result, view_code, visible)
    return fmap


def render_stage_map(
    result: Dict,
    view_code: int,
    key: str,
    visible_item_ids: Optional[Sequence[str]] = None,
    height: int = 620,
):
    if result.get("status") not in {"OPTIMAL", "FEASIBLE"}:
        st.warning(f"{result.get('status')}: {result.get('message')}")
        return
    from streamlit_folium import st_folium
    fmap = build_stage_map(result, view_code, visible_item_ids=visible_item_ids)
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
    st.markdown("### 2.2 제품구조 원료 선택")
    st.info(
        "업로드 데이터에는 철강·알루미늄·기타 원자재·희토류·구리·플라스틱·배터리의 Stage 1 및 Stage 2 데이터가 존재합니다. "
        "이번 최적화에 포함할 원료만 선택합니다. 기본값은 철강·알루미늄·기타 원자재·배터리이며 희토류·구리·플라스틱은 해제 상태입니다."
    )
    active: List[str] = []
    cols = st.columns(min(5, max(1, len(catalog))))
    for idx, (_, row) in enumerate(catalog.sort_values("item_index").iterrows()):
        item_id = str(row["item_id"])
        mandatory = int(row["mandatory"]) == 1
        default = bool(int(row["default_enabled"])) or mandatory
        key = f"v19_item_enabled::{item_id}"
        if key not in st.session_state:
            st.session_state[key] = default
        with cols[idx % len(cols)]:
            checked = st.checkbox(
                str(row["item_name_ko"]), key=key, disabled=mandatory,
                help=f"Stage 1: {row['stage1_process_name_ko']} / Stage 2: {row['stage2_process_name_ko']}",
            )
            if checked or mandatory:
                active.append(item_id)
    previous = list(st.session_state.get(SESSION_SELECTED_ITEMS_KEY, []))
    st.session_state[SESSION_SELECTED_ITEMS_KEY] = list(active)
    if previous and previous != list(active):
        st.session_state.pop(SESSION_RESULTS_KEY, None)
    selected_names = catalog.loc[catalog["item_id"].astype(str).isin(active), "item_name_ko"].astype(str).tolist()
    st.caption("선택된 원료: " + ", ".join(selected_names))
    return active


def _render_country_grid(available: Sequence[str], prefix: str, all_key: str, none_key: str) -> List[str]:
    available = list(dict.fromkeys(str(v) for v in available))
    b1, b2 = st.columns(2)
    if b1.button("24개국 모두 선택", key=all_key, use_container_width=True):
        for name in available:
            st.session_state[prefix + name] = True
    if b2.button("모두 해제", key=none_key, use_container_width=True):
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
    return chosen


def render_stage_country_selection(
    catalog: pd.DataFrame,
    suppliers: pd.DataFrame,
    processes: pd.DataFrame,
    plants: pd.DataFrame,
    active_item_ids: Sequence[str],
) -> Dict[str, List[str]]:
    st.markdown("### 선택 원료별 Stage 1·Stage 2 국가 선택")
    selected: Dict[str, List[str]] = {}
    item_lookup = catalog.set_index("item_id")
    all_plant_order = plants["location_name"].astype(str).tolist()
    for item_id in active_item_ids:
        item_name = str(item_lookup.loc[item_id, "item_name_ko"])
        with st.expander(f"{item_name} — Stage 1·2 국가", expanded=False):
            t1, t2 = st.tabs(["Stage 1 생산지", "Stage 2 중간가공·조립지"])
            with t1:
                available1_set = set(suppliers.loc[
                    (suppliers["item_id"].astype(str) == item_id)
                    & (pd.to_numeric(suppliers["active_default"], errors="coerce").fillna(0).astype(int) == 1),
                    "location_name",
                ].astype(str).tolist())
                available1 = [name for name in all_plant_order if name in available1_set]
                selected[f"stage1::{item_id}"] = _render_country_grid(
                    available1, f"country::stage1::{item_id}::",
                    f"all::stage1::{item_id}", f"none::stage1::{item_id}",
                )
            with t2:
                available2_set = set(processes.loc[
                    (processes["item_id"].astype(str) == item_id)
                    & (pd.to_numeric(processes["active_default"], errors="coerce").fillna(0).astype(int) == 1),
                    "location_name",
                ].astype(str).tolist())
                available2 = [name for name in all_plant_order if name in available2_set]
                selected[f"stage2::{item_id}"] = _render_country_grid(
                    available2, f"country::stage2::{item_id}::",
                    f"all::stage2::{item_id}", f"none::stage2::{item_id}",
                )

    common = set(all_plant_order)
    for item_id in active_item_ids:
        common &= set(selected.get(f"stage2::{item_id}", []))
    selected["assembly"] = [name for name in all_plant_order if name in common]
    if selected["assembly"]:
        st.success(
            f"선택 원료 전체에 공통으로 허용된 Stage 2 차량 조립국가: {len(selected['assembly'])}개 — "
            + ", ".join(selected["assembly"])
        )
    else:
        st.error("선택 원료들의 Stage 2 국가 교집합이 비어 있습니다. 최소 1개의 공통 조립국가를 남기세요.")
    return selected




def render_transport_mode_selection() -> List[str]:
    st.markdown("### 허용 외부 운송수단 선택")
    saved = st.session_state.get(SESSION_SELECTED_TRANSPORT_MODES_KEY, list(ROUTE_MODE_CODES))
    selected: List[str] = []
    cols = st.columns(3)
    for i, code in enumerate(ROUTE_MODE_CODES):
        key = f"transport_mode::{code}"
        if key not in st.session_state:
            st.session_state[key] = code in set(saved)
        with cols[i % 3]:
            if st.checkbox(ROUTE_MODE_LABEL[code], key=key):
                selected.append(code)
    st.session_state[SESSION_SELECTED_TRANSPORT_MODES_KEY] = list(selected)
    if selected:
        st.success("허용 외부 운송수단: " + ", ".join(ROUTE_MODE_LABEL[v] for v in selected))
    else:
        st.error("외부 운송수단을 최소 1개 선택하세요.")
    return selected


def active_index_preview(
    active_item_ids: Sequence[str],
    selected_country_map: Mapping[str, Sequence[str]],
    selected_transport_modes: Sequence[str],
) -> pd.DataFrame:
    origin_union = set()
    for item_id in active_item_ids:
        origin_union.update(str(v) for v in selected_country_map.get(f"stage1::{item_id}", []))
    assembly_set = list(selected_country_map.get("assembly", []))
    route_slots = ["road (내부 t=0)"] + [v for v in selected_transport_modes if v != "road"]
    return pd.DataFrame([
        ["R", "선택 원료·원자재", len(active_item_ids), ", ".join(active_item_ids)],
        ["O", "Stage 1 생산국가 합집합", len(origin_union), ", ".join(sorted(origin_union))],
        ["A", "선택 원료 Stage 2 국가의 교집합", len(assembly_set), ", ".join(assembly_set)],
        ["T", "모형 운송 슬롯", len(route_slots), ", ".join(route_slots)],
    ], columns=["인덱스", "적용범위", "개수", "활성값"])

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
    with st.expander("수학적 최적화 모형 구현 확인", expanded=False):
        st.markdown(
            "`build_flexible_lp_model()`은 2번 탭의 활성 품목 R, 3번 탭의 Stage 1 국가 O, "
            "공통 Stage 2 국가 A, 허용 운송수단 T로 `IndexLayout`을 다시 생성합니다. "
            "목적함수·탄소계수·7개 제약군은 이 활성 인덱스에 대해서만 생성됩니다. 전 품목 모듈 방식에서는 모든 활성 품목의 Stage 1 국가와 차량 조립국가가 달라야 합니다."
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
        "원료를 체크 해제해도 다른 원료가 자동으로 증가하지 않습니다. 희토류·구리·플라스틱을 체크하면 해당 BOM 질량과 "
        "Stage 1 생산·운송, Stage 2 조립 및 Stage 3 완성차 운송 항이 동일한 기준모형 인덱스에 추가됩니다. "
        "기능적으로 동등한 제품구조를 비교하려면 기타 원자재와 추가 품목의 BOM 질량을 함께 조정해야 합니다."
    )
    if tables:
        status = item_data_status(tables)
        if not status.empty:
            st.markdown("## 현재 세션의 품목")
            st.dataframe(status, use_container_width=True, hide_index=True)


def render_input_tab(tables: Mapping[str, pd.DataFrame]):
    st.header("사용자 CSV 데이터 및 제품구조 원료 선택")

    st.markdown("## 2.1 필수 CSV 또는 ZIP 업로드")
    uploads = st.file_uploader(
        "필수 데이터 파일", type=["csv", "zip"], accept_multiple_files=True,
        key="v19_full_data_uploads",
        help="필수 CSV 13개 또는 base_csv_upload_v19.zip을 업로드하세요. 파일명은 정확히 일치해야 합니다.",
    )
    if st.button("업로드 데이터 적용", type="primary", key="v19_apply_full_data"):
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
                "현재 세션 CSV ZIP 다운로드", data=make_tables_zip(tables),
                file_name="current_ev_supply_chain_data_v19.zip", mime="application/zip",
                use_container_width=True,
            )
        with c2:
            if st.button("업로드 데이터와 결과 초기화", use_container_width=True, key="v19_reset_all"):
                st.session_state.pop(SESSION_TABLES_KEY, None)
                st.session_state.pop(SESSION_RESULTS_KEY, None)
                st.session_state.pop(SESSION_SELECTED_ITEMS_KEY, None)
                st.session_state.pop(SESSION_SELECTED_TRANSPORT_MODES_KEY, None)
                st.rerun()
    else:
        st.warning("아직 적용된 데이터가 없습니다. 필수 CSV 13개 또는 ZIP을 업로드하세요.")
        return

    errors = validate_tables(tables)
    if errors:
        st.error("현재 데이터가 검증을 통과하지 못했습니다.")
        for error in errors:
            st.write(f"- {error}")
        return
    st.success("활성화 가능한 전체 품목의 BOM·Stage 1·Stage 2 데이터 검증을 통과했습니다.")

    active_item_ids = render_item_selection(tables["item_catalog.csv"].sort_values("item_index"))

    st.markdown("### 2.3 선택 품목의 Stage 1·Stage 2·모듈 계수 확인")
    catalog = tables["item_catalog.csv"].set_index("item_id")
    suppliers = tables["item_suppliers.csv"]
    processes = tables["stage2_item_processes.csv"]
    for item_id in active_item_ids:
        item_name = str(catalog.loc[item_id, "item_name_ko"])
        with st.expander(f"{item_name} ({item_id}) 계수", expanded=False):
            c1, c2 = st.columns(2)
            stage1 = suppliers[suppliers["item_id"].astype(str) == item_id].copy()
            stage2 = processes[processes["item_id"].astype(str) == item_id].copy()
            with c1:
                st.markdown("**Stage 1 생산계수**")
                st.caption(
                    f"국가 {stage1['location_name'].nunique()}개 · 비용범위 "
                    f"{stage1['stage1_cost_per_unit'].min():,.4g}~{stage1['stage1_cost_per_unit'].max():,.4g} · "
                    f"배출계수범위 {stage1['stage1_ef_kgco2_per_unit'].min():,.4g}~{stage1['stage1_ef_kgco2_per_unit'].max():,.4g}"
                )
                st.dataframe(stage1, use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**Stage 2 허용국가·공정 참고정보**")
                if str(catalog.loc[item_id, "item_type"]) == "battery":
                    st.caption(
                        f"국가 {stage2['location_name'].nunique()}개 · 배터리의 실제 제조·조립 비용·배출계수는 "
                        "assembly_locations.csv에서 Stage 1 배터리 생산지 RP에 포함됩니다. 이 표는 "
                        "배터리의 Stage 2 허용국가와 공정명을 확인하는 데 사용합니다."
                    )
                else:
                    st.caption(
                        f"국가 {stage2['location_name'].nunique()}개 · 이 표의 active_default와 위치정보는 "
                        "원료별 Stage 2 허용국가 집합을 구성합니다. 실제 비배터리 조립 비용·배출량은 "
                        "assembly_locations.csv의 국가별 계수를 활성 비배터리 총질량에 적용합니다."
                    )
                st.dataframe(stage2, use_container_width=True, hide_index=True)
            st.markdown("**대형·소형 모듈 및 효율계수**")
            module_row = tables["module_parameters.csv"][tables["module_parameters.csv"]["item_id"].astype(str) == item_id]
            st.dataframe(module_row, use_container_width=True, hide_index=True)

    st.markdown("## 2.4 현재 입력 테이블 확인")
    display = {
        "item_catalog.csv": "품목 카탈로그", "product_bom.csv": "제품별 BOM",
        "item_suppliers.csv": "품목별 Stage 1 생산지 데이터",
        "stage2_item_processes.csv": "품목별 Stage 2 공정 데이터",
        "module_parameters.csv": "품목별 대형·소형 모듈 파라미터",
        "products.csv": "차량 종류", "demand.csv": "프랑스 수요",
        "assembly_locations.csv": "후보국가·위치", "transport_parameters.csv": "운송수단 계수",
        "country_transport_rules.csv": "국가별 운송 허용규칙", "markets.csv": "시장",
        "scenarios.csv": "정책 시나리오", "model_metadata.csv": "모형 메타데이터",
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
    st.markdown("## 전체 및 단계별 공급망 지도")
    st.info(
        "Stage 1·2·3 위치 지도와 두 운송구간 지도를 분리했습니다. 선택 원료마다 고유한 색상을 사용하며, "
        "Stage 1 원의 크기는 동일 원료 내 국가별 생산량, Stage 2 원의 크기는 차량 조립량을 나타냅니다. "
        "운송선은 원료별로 서로 다른 곡선 lane과 흰 외곽선을 사용하며, 세 개의 화살표가 이동방향을 나타냅니다."
    )
    chosen = st.selectbox(
        "지도 조회 조합",
        available,
        format_func=lambda key: f"{SCENARIO_SHORT[key[0]]} · {MODE_LABEL[key[1]]}",
        key="stage_map_result_choice_v19",
    )
    map_options = [
        "전체 공급망 전과정: Stage 1→2→3",
        "Stage 1 생산지",
        "Stage 1→2 원료·원자재 운송경로",
        "Stage 2 차량 조립지",
        "Stage 2→3 완성차 운송경로",
        "Stage 3 프랑스 시장",
    ]
    stage_label = st.radio(
        "지도 단계",
        map_options,
        horizontal=False,
        key="stage_map_stage_choice_v19",
    )
    view_code = {
        map_options[0]: 0,
        map_options[1]: 1,
        map_options[2]: 12,
        map_options[3]: 2,
        map_options[4]: 23,
        map_options[5]: 3,
    }[stage_label]
    selected = results[chosen]
    captions = {
        0: "Stage 1 생산, Stage 1→2 이동, Stage 2 조립, Stage 2→3 이동 및 프랑스 시장을 통합 표시합니다.",
        1: "원료·원자재별 색상과 생산량에 비례한 원 크기로 Stage 1 생산지만 표시합니다.",
        12: "Stage 1 생산지에서 Stage 2 차량 조립지로 이동하는 품목별 경로와 방향 화살표를 표시합니다.",
        2: "Stage 2에서 실제로 차량을 조립하는 국가만 표시합니다. 원의 크기는 조립 차량대수입니다.",
        23: "Stage 2 차량 조립지에서 Stage 3 프랑스 시장으로 이동하는 완성차 경로와 방향 화살표를 표시합니다.",
        3: "최종 Stage 3 시장인 프랑스 위치만 표시합니다.",
    }
    st.caption(captions[view_code])
    visible_item_ids = list(selected.get("active_item_ids", []))
    if view_code in {0, 1, 12}:
        catalog_map = selected.get("item_catalog", pd.DataFrame()).set_index("item_id")
        visible_item_ids = st.multiselect(
            "지도에 표시할 원료·원자재",
            options=list(selected.get("active_item_ids", [])),
            default=list(selected.get("active_item_ids", [])),
            format_func=lambda item_id: str(catalog_map.loc[item_id, "item_name_ko"]) if item_id in catalog_map.index else str(item_id),
            key=f"map_items_v19_{chosen[0]}_{chosen[1]}_{view_code}",
            help="경로가 겹칠 때 특정 원료만 선택하면 해당 공급경로를 분리해서 확인할 수 있습니다.",
        )
        if not visible_item_ids:
            st.warning("지도에 표시할 원료를 최소 1개 선택하세요.")
            return
    render_stage_map(
        selected, view_code,
        key=f"stage_map_v19_{chosen[0]}_{chosen[1]}_{view_code}_{'_'.join(visible_item_ids)}",
        visible_item_ids=visible_item_ids,
    )

    st.markdown("## 상세 결과")
    detail_tabs = st.tabs([
        "제품구조", "차량별 결과", "Stage 1 생산", "Stage 1→2 운송",
        "Stage 2 원료별 공정", "Stage 2 조립지 합계", "선택 품목 대형·소형 모듈", "Stage 3 시장 출시", "Solver 정보",
    ])
    frames = [
        selected.get("product_structure_summary", pd.DataFrame()),
        selected.get("product_summary", pd.DataFrame()),
        selected.get("production_summary", pd.DataFrame()),
        selected.get("inbound_routes", pd.DataFrame()),
        selected.get("stage2_item_process_summary", pd.DataFrame()),
        selected.get("assembly_summary", pd.DataFrame()),
        selected.get("module_summary", pd.DataFrame()),
        selected.get("market_routes", pd.DataFrame()),
    ]
    for tab, frame in zip(detail_tabs[:8], frames):
        with tab:
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                st.dataframe(frame, use_container_width=True, hide_index=True)
            else:
                st.info("표시할 양의 결과가 없습니다.")
    with detail_tabs[8]:
        st.json({
            key: selected.get(key) for key in [
                "status", "message", "solver_name", "solver_version", "solver_iterations",
                "variable_count", "constraint_count", "matrix_nonzeros", "wall_time_sec",
                "structure_signature", "active_item_ids", "selected_country_map",
                "selected_transport_modes", "route_mode_codes", "active_index_sizes",
                "fleet_total_cap_kgco2", "fleet_cap_utilization_pct", "fleet_cap_met",
                "objective_reconstruction_gap_eur", "line_all_item_colocation",
                "modular_all_item_separation",
                "modular_same_country_positive_flow_count",
            ]
        })



def render_comparison_bar_chart(
    comparison: pd.DataFrame,
    value_column: str,
    heading: str,
    y_title: str,
):
    """Render a dependency-light HTML bar chart.

    This function intentionally does not call st.bar_chart and does not create a pandas
    MultiIndex. It therefore avoids the Streamlit/Pandas KeyError seen in older deployments.
    """
    import html

    st.markdown(f"## {heading}")
    required = {"시나리오", "생산방식", value_column}
    missing = sorted(required - set(comparison.columns))
    if missing:
        st.warning(f"차트에 필요한 열이 없습니다: {', '.join(missing)}")
        return

    chart_data = comparison.loc[:, ["시나리오", "생산방식", value_column]].copy()
    chart_data[value_column] = pd.to_numeric(chart_data[value_column], errors="coerce")
    chart_data = chart_data.dropna(subset=[value_column]).reset_index(drop=True)
    if chart_data.empty:
        st.info("차트에 표시할 숫자 결과가 없습니다.")
        return

    max_value = float(chart_data[value_column].max())
    if not np.isfinite(max_value) or max_value <= 0:
        max_value = 1.0
    mode_colors = {"라인 생산": "#1f77b4", "전 품목 2수준 모듈 분산 생산": "#ff7f0e"}
    rows = []
    for _, row in chart_data.iterrows():
        label = f"{row['시나리오']} · {row['생산방식']}"
        value = float(row[value_column])
        width = max(0.8, min(100.0, 100.0 * value / max_value))
        color = mode_colors.get(str(row["생산방식"]), "#6b7280")
        formatted = f"{value:,.2f}"
        rows.append(
            "<div style='margin:9px 0 13px 0;'>"
            f"<div style='display:flex;justify-content:space-between;gap:12px;font-size:13px;'>"
            f"<span>{html.escape(label)}</span><b>{html.escape(formatted)}</b></div>"
            "<div style='height:22px;background:#eef1f5;border-radius:5px;overflow:hidden;border:1px solid #d6dbe3;'>"
            f"<div style='height:100%;width:{width:.3f}%;background:{color};'></div></div></div>"
        )
    chart_html = (
        "<div style='border:1px solid #d9dde5;border-radius:8px;padding:12px 15px;background:white;'>"
        f"<div style='font-size:12px;color:#555;margin-bottom:8px;'>{html.escape(y_title)}</div>"
        + "".join(rows)
        + "</div>"
    )
    st.markdown(chart_html, unsafe_allow_html=True)

def render_analysis_tab(results: Mapping[Tuple[str, str], Dict]):
    st.header("분석 및 결론")
    comparison = comparison_dataframe(results)
    if comparison.empty:
        st.info("분석할 OPTIMAL 또는 FEASIBLE 결과가 없습니다.")
        return
    show_dataframe(comparison, "시나리오·생산방식 비교")
    render_comparison_bar_chart(
        comparison,
        value_column="총비용(EUR)",
        heading="총비용 비교",
        y_title="총 공급망 비용(EUR)",
    )
    render_comparison_bar_chart(
        comparison,
        value_column="회사 전체 탄소발자국(kgCO2-eq)",
        heading="회사 전체 탄소발자국 비교",
        y_title="회사 전체 탄소발자국(kg CO₂-eq)",
    )

    st.markdown("## 라인 생산 대비 전 품목 모듈 분산 생산 차이")
    delta_rows = []
    for scenario in ["S1", "S2", "S3"]:
        line = results.get((scenario, "line"), {})
        modular = results.get((scenario, "modular"), {})
        if line.get("status") not in {"OPTIMAL", "FEASIBLE"} or modular.get("status") not in {"OPTIMAL", "FEASIBLE"}:
            continue
        modular_inbound = modular.get("inbound_routes", pd.DataFrame())
        if isinstance(modular_inbound, pd.DataFrame) and not modular_inbound.empty:
            external = modular_inbound.loc[~modular_inbound["internal_flow"].astype(bool)].copy()
            external_mass = float(external.get("transport_mass_kg", pd.Series(dtype=float)).sum())
            external_items = int(external.get("item_id", pd.Series(dtype=str)).astype(str).nunique())
            external_routes = int(len(external))
        else:
            external_mass = 0.0
            external_items = 0
            external_routes = 0
        delta_rows.append({
            "시나리오": SCENARIO_SHORT[scenario],
            "전 품목 모듈-라인 비용차(EUR)": float(modular["objective_value"]) - float(line["objective_value"]),
            "전 품목 모듈-라인 탄소차이(kgCO2-eq)": float(modular["total_emissions_kgco2"]) - float(line["total_emissions_kgco2"]),
            "모듈 방식 외부 운송질량(kg)": external_mass,
            "외부 운송 품목 수": external_items,
            "양의 원료·중간재 경로 수": external_routes,
        })
    if delta_rows:
        st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)
        st.caption(
            "전 품목 2수준 모듈 분산 생산에서는 선택된 모든 품목이 module_parameters.csv의 대형·소형 모듈 조합을 사용하며, "
            "Stage 1 모듈 생산국가와 Stage 2 차량 조립국가는 서로 달라야 합니다."
        )

    st.markdown("## 제품구조 변경 해석")
    st.markdown(
        "- 사용자가 추가한 품목을 체크하면 동일한 RP·RT·FP·FT 수식에서 원료 인덱스 R의 한 항으로 추가됩니다.\n"
        "- 기존 품목을 해제하면 해당 품목의 생산·운송·질량이 제거됩니다. 다른 재료는 자동으로 대체되지 않습니다.\n"
        "- 선택 제품구조의 비배터리 질량은 기준모형의 FP 조립항에, 전체 질량은 FT 완성차 운송항에 반영됩니다.\n"
        "- 서로 다른 제품구조를 비교할 때는 `제품구조 서명`, 활성 품목, BOM 및 입력 데이터 버전이 같은지 확인해야 합니다."
    )
    st.warning("사용자 업로드 데이터의 출처·단위·시스템 경계를 검증한 뒤 결과를 의사결정에 사용하세요.")


def run_app():
    st.set_page_config(page_title="사용자 데이터 기반 EV 공급망 LP", page_icon="🚗", layout="wide")
    st.title("프랑스 전기차 보조금 탄소발자국 상한 대응 공급망 비용 최적화")

    tables: Dict[str, pd.DataFrame] = {
        name: frame.copy() for name, frame in st.session_state.get(SESSION_TABLES_KEY, {}).items()
    }

    tabs = st.tabs([
        "1. SaaS 최적화 프레임워크 개요", "2. 사용자 데이터·원료 선택",
        "3. 최적화 실행", "4. 최적화 결과", "5. 분석 및 결론",
    ])

    with tabs[0]:
        render_overview_tab(tables if tables else None)

    with tabs[1]:
        render_input_tab(tables)

    with tabs[2]:
        st.header("최적화 실행")
        if st.button("최적화 결과 초기화", key="v19_reset_results_main"):
            st.session_state.pop(SESSION_RESULTS_KEY, None)
            gc.collect()
            st.success("최적화 결과를 초기화했습니다.")
        if not tables or any(name not in tables for name in REQUIRED_FILES):
            st.info("2번 탭에서 필수 CSV 13개 또는 ZIP을 먼저 업로드하세요.")
        else:
            data_errors = validate_tables(tables)
            if data_errors:
                st.error("입력 데이터 검증 문제를 해결해야 최적화를 실행할 수 있습니다.")
                for error in data_errors:
                    st.write(f"- {error}")
            else:
                default_items = _normalize_active_items(tables["item_catalog.csv"], None)
                active_item_ids = list(st.session_state.get(SESSION_SELECTED_ITEMS_KEY, default_items))
                catalog = tables["item_catalog.csv"].sort_values("item_index")
                active_item_ids = [i for i in catalog["item_id"].astype(str).tolist() if i in set(active_item_ids)]
                battery_ids = catalog.loc[catalog["item_type"].astype(str) == "battery", "item_id"].astype(str).tolist()
                for battery_id in battery_ids:
                    if battery_id not in active_item_ids:
                        active_item_ids.append(battery_id)
                names = catalog.set_index("item_id").loc[active_item_ids, "item_name_ko"].astype(str).tolist()
                st.success("2번 탭에서 선택된 원료: " + ", ".join(names))

                preview = product_structure_preview(tables, active_item_ids)
                show_dataframe(preview, "선택 제품구조 미리보기", "기본구조는 철강·알루미늄·기타 원자재·배터리이며 희토류·구리·플라스틱은 선택사항입니다.")

                selected_country_map = render_stage_country_selection(
                    tables["item_catalog.csv"], tables["item_suppliers.csv"],
                    tables["stage2_item_processes.csv"], tables["assembly_locations.csv"],
                    active_item_ids,
                )
                selected_transport_modes = render_transport_mode_selection()
                required_keys = [f"stage1::{i}" for i in active_item_ids] + [f"stage2::{i}" for i in active_item_ids] + ["assembly"]
                valid_selection = (
                    all(bool(selected_country_map.get(key)) for key in required_keys)
                    and bool(selected_transport_modes)
                )
                show_dataframe(
                    active_index_preview(active_item_ids, selected_country_map, selected_transport_modes),
                    "현재 최적화에 적용되는 인덱스",
                    "목적함수·탄소발자국 합계·물량수지는 이 활성 인덱스에 대해서만 생성됩니다.",
                )

                c1, c2, c3 = st.columns(3)
                scenario_id = c1.selectbox(
                    "정책 시나리오", ["S1", "S2", "S3"],
                    format_func=lambda value: str(tables["scenarios.csv"].set_index("scenario_id").loc[value, "scenario_name"]),
                )
                production_mode = c2.selectbox("생산방식", ["line", "modular"], format_func=lambda value: MODE_LABEL[value])
                time_limit = c3.number_input("Solver 제한시간(초)", min_value=10, max_value=600, value=180, step=10)

                base_assemblies = list(selected_country_map.get("assembly", []))
                if production_mode == "line":
                    feasible = [a for a in base_assemblies if all(a in selected_country_map.get(f"stage1::{i}", []) for i in active_item_ids)]
                    if not feasible:
                        valid_selection = False
                        st.error("전 품목 라인 생산 조건(origin = assembly)을 만족하는 공통 국가가 없습니다.")
                else:
                    feasible = [a for a in base_assemblies if all(any(o != a for o in selected_country_map.get(f"stage1::{i}", [])) for i in active_item_ids)]
                    if not feasible:
                        valid_selection = False
                        st.error("전 품목 모듈 생산 조건(origin ≠ assembly)을 만족하는 공통 차량 조립국가가 없습니다.")

                st.info(
                    "원료 포함 여부와 원료별 Stage 1·2 허용국가는 사용자가 정하는 모형 범위입니다. Solver는 "
                    "그 범위에서 생산국가, 공통 조립국가, 운송수단과 연속 물량을 배분하여 총비용을 최소화합니다."
                )

                def execute(s: str, mode: str, status_box, prefix: str = "") -> Dict:
                    score = SCENARIO_POLICY_SCORE[s]
                    if score is None:
                        status_box.info(f"{prefix}탄소발자국 상한 없는 S1 최소비용 공급망을 계산합니다.")
                    else:
                        status_box.info(f"{prefix}{score:.0f}점의 회사 전체 탄소발자국 상한에서 최소비용 공급망을 계산합니다.")
                    try:
                        output = solve_case(
                            tables, scenario_id=s, production_mode=mode, time_limit_sec=int(time_limit),
                            active_item_ids=active_item_ids, selected_country_map=selected_country_map,
                            selected_transport_modes=selected_transport_modes,
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
                        return {"status": "ERROR", "message": str(exc), "scenario_id": s,
                                "production_mode": mode, "active_item_ids": list(active_item_ids),
                                "selected_transport_modes": list(selected_transport_modes)}

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
