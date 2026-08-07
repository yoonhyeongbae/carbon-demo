"""Public Python API for the EV carbon supply-chain optimizer.

This module intentionally contains no GUI dependency. It wraps the validated v21.1
optimization engine so ERP/MES integration code can call the same solver used by the
Flet desktop application.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

import legacy_core as core


class _NamedBytesIO(io.BytesIO):
    def __init__(self, payload: bytes, name: str):
        super().__init__(payload)
        self.name = name


@dataclass(frozen=True)
class OptimizationRequest:
    scenario_id: str = "S1"
    production_mode: str = "line"
    time_limit_sec: int = 180
    active_item_ids: Tuple[str, ...] = field(default_factory=tuple)
    selected_country_map: Mapping[str, Sequence[str]] = field(default_factory=dict)
    selected_transport_modes: Tuple[str, ...] = field(default_factory=lambda: tuple(core.ROUTE_MODE_CODES))


@dataclass
class OptimizationResult:
    data: Dict

    @property
    def status(self) -> str:
        return str(self.data.get("status", "UNKNOWN"))

    @property
    def objective_value(self) -> Optional[float]:
        value = self.data.get("objective_value")
        return None if value is None else float(value)

    @property
    def total_emissions_kgco2(self) -> Optional[float]:
        value = self.data.get("total_emissions_kgco2")
        return None if value is None else float(value)

    def export_zip_bytes(self) -> bytes:
        return result_to_zip_bytes(self.data)


def load_input_files(paths: Sequence[str | Path]) -> Dict[str, pd.DataFrame]:
    uploads = []
    for raw_path in paths:
        path = Path(raw_path)
        uploads.append(_NamedBytesIO(path.read_bytes(), path.name))
    tables, messages = core.parse_full_data_uploads(uploads)
    errors = [message for message in messages if message.startswith("오류:")]
    if errors:
        raise ValueError("\n".join(errors))
    return tables


def load_input_directory(directory: str | Path) -> Dict[str, pd.DataFrame]:
    root = Path(directory)
    return load_input_files([root / name for name in core.REQUIRED_FILES if (root / name).exists()])


def load_input_zip(path: str | Path) -> Dict[str, pd.DataFrame]:
    return load_input_files([Path(path)])


def validate_input(tables: Mapping[str, pd.DataFrame]) -> list[str]:
    return core.validate_tables(tables)


def default_active_items(tables: Mapping[str, pd.DataFrame]) -> list[str]:
    return core._normalize_active_items(tables["item_catalog.csv"], None)


def default_country_map(
    tables: Mapping[str, pd.DataFrame],
    active_item_ids: Sequence[str],
    production_mode: str,
) -> Dict[str, list[str]]:
    catalog = tables["item_catalog.csv"]
    suppliers = tables["item_suppliers.csv"]
    processes = tables["stage2_item_processes.csv"]
    plants = tables["assembly_locations.csv"]
    order = plants["location_name"].astype(str).tolist()
    selected: Dict[str, list[str]] = {}
    for item_id in active_item_ids:
        s1 = set(suppliers.loc[
            (suppliers["item_id"].astype(str) == str(item_id))
            & (pd.to_numeric(suppliers["active_default"], errors="coerce").fillna(0).astype(int) == 1),
            "location_name",
        ].astype(str))
        s2 = set(processes.loc[
            (processes["item_id"].astype(str) == str(item_id))
            & (pd.to_numeric(processes["active_default"], errors="coerce").fillna(0).astype(int) == 1),
            "location_name",
        ].astype(str))
        if production_mode == "line":
            linked = [name for name in order if name in s1 and name in s2]
            selected[f"stage1::{item_id}"] = linked
            selected[f"stage2::{item_id}"] = list(linked)
        else:
            selected[f"stage1::{item_id}"] = [name for name in order if name in s1]
            selected[f"stage2::{item_id}"] = [name for name in order if name in s2]
    common = set(order)
    for item_id in active_item_ids:
        common &= set(selected.get(f"stage2::{item_id}", []))
    selected["assembly"] = [name for name in order if name in common]
    return selected


def optimize(tables: Mapping[str, pd.DataFrame], request: OptimizationRequest) -> OptimizationResult:
    errors = validate_input(tables)
    if errors:
        raise ValueError("입력 데이터 검증 실패:\n- " + "\n- ".join(errors))
    active = list(request.active_item_ids) or default_active_items(tables)
    country_map = dict(request.selected_country_map) or default_country_map(
        tables, active, request.production_mode
    )
    result = core.solve_case(
        tables=tables,
        scenario_id=request.scenario_id,
        production_mode=request.production_mode,
        time_limit_sec=int(request.time_limit_sec),
        active_item_ids=active,
        selected_country_map=country_map,
        selected_transport_modes=list(request.selected_transport_modes),
    )
    return OptimizationResult(result)


def optimize_all(
    tables: Mapping[str, pd.DataFrame],
    active_item_ids: Optional[Sequence[str]] = None,
    country_maps: Optional[Mapping[str, Mapping[str, Sequence[str]]]] = None,
    selected_transport_modes: Sequence[str] = core.ROUTE_MODE_CODES,
    time_limit_sec: int = 180,
) -> Dict[Tuple[str, str], OptimizationResult]:
    active = list(active_item_ids or default_active_items(tables))
    maps = country_maps or {
        mode: default_country_map(tables, active, mode) for mode in ("line", "modular")
    }
    outputs: Dict[Tuple[str, str], OptimizationResult] = {}
    for scenario in ("S1", "S2", "S3"):
        for mode in ("line", "modular"):
            outputs[(scenario, mode)] = optimize(
                tables,
                OptimizationRequest(
                    scenario_id=scenario,
                    production_mode=mode,
                    time_limit_sec=time_limit_sec,
                    active_item_ids=tuple(active),
                    selected_country_map=maps[mode],
                    selected_transport_modes=tuple(selected_transport_modes),
                ),
            )
    return outputs


def result_to_zip_bytes(result: Mapping) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        frame_keys = [
            "product_structure_summary", "product_summary", "production_summary",
            "inbound_routes", "stage2_item_process_summary", "assembly_summary",
            "module_summary", "market_routes",
        ]
        for key in frame_keys:
            frame = result.get(key)
            if isinstance(frame, pd.DataFrame):
                archive.writestr(f"{key}.csv", frame.to_csv(index=False).encode("utf-8-sig"))
        metadata = {}
        for key, value in result.items():
            if isinstance(value, pd.DataFrame):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value
            elif isinstance(value, (list, tuple, dict)):
                try:
                    json.dumps(value, ensure_ascii=False)
                    metadata[key] = value
                except TypeError:
                    metadata[key] = str(value)
            else:
                metadata[key] = str(value)
        archive.writestr(
            "solver_metadata.json",
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        )
    return output.getvalue()


__all__ = [
    "OptimizationRequest", "OptimizationResult", "load_input_files", "load_input_directory",
    "load_input_zip", "validate_input", "default_active_items", "default_country_map",
    "optimize", "optimize_all", "result_to_zip_bytes",
]
