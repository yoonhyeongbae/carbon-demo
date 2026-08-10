from __future__ import annotations

"""Apply the EV Carbon Optimizer v0.5.7 result-map UI hotfix to a v0.5.6 DEV_READY folder.

This patch intentionally DOES NOT modify dynamic_structure_core.py, legacy_core.py,
ev_optimizer_api.py, CSV data, the OR-Tools objective, decision variables, constraints,
or scenario caps. It changes only result/analysis UI behavior and map geometry density.

Run this script from the EV_Carbon_Optimizer_v0.5.6_DEV_READY folder.
"""

from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parent
# Allow the patch script to live either in project root or under a downloaded patch folder.
if not (ROOT / "main.py").exists():
    # command-line first argument can explicitly point to the EVCO project root
    if len(sys.argv) > 1:
        ROOT = Path(sys.argv[1]).expanduser().resolve()

MAIN = ROOT / "main.py"
MAP_LOGIC = ROOT / "map_logic.py"

if not MAIN.exists() or not MAP_LOGIC.exists():
    raise SystemExit(
        "main.py / map_logic.py를 찾을 수 없습니다.\n"
        "사용법: python APPLY_V057_MAP_UI_PATCH.py C:\\EV_Carbon_Optimizer_v0.5.6_DEV_READY"
    )

main = MAIN.read_text(encoding="utf-8")
logic = MAP_LOGIC.read_text(encoding="utf-8")

if "EVCO-0.5.6-USER-GUIDE-CSV-OVERRIDE-SCOPE" not in main:
    raise SystemExit(
        "이 패치는 EVCO v0.5.6 USER-GUIDE-CSV-OVERRIDE-SCOPE 기준입니다. "
        "현재 main.py BUILD_ID를 확인하세요."
    )

# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------
for path in (MAIN, MAP_LOGIC):
    backup = path.with_suffix(path.suffix + ".v056_backup")
    if not backup.exists():
        shutil.copy2(path, backup)

# ---------------------------------------------------------------------------
# 1) Results-map shared controls: one common legend + stage/material filters.
# ---------------------------------------------------------------------------
needle = """        self.scenario_map_filter_area = ft.Row(wrap=True, spacing=8)\n        self.scenario_map_stack_area = ft.Column(spacing=14)\n"""
replacement = """        self.scenario_map_filter_area = ft.Row(wrap=True, spacing=8)\n        # v0.5.7: shared controls for all scenario/mode maps.  The legend is mounted\n        # once directly below the scenario/mode checkboxes instead of inside map #1.\n        self.scenario_map_legend_area = ft.Column(spacing=6)\n        self.scenario_map_stage_filter_area = ft.Row(wrap=True, spacing=10)\n        self.scenario_map_item_filter_area = ft.Row(wrap=True, spacing=8, run_spacing=5)\n        self.map_stage_enabled: set[str] = {\"stage1\", \"stage12\", \"stage2\", \"stage23\"}\n        self.map_visible_item_ids: set[str] = set()\n        self._map_material_filter_initialized = False\n        self.scenario_map_stack_area = ft.Column(spacing=14)\n"""
if needle not in main:
    raise SystemExit("main.py에서 scenario_map_filter_area 초기화 블록을 찾지 못했습니다.")
main = main.replace(needle, replacement, 1)

# ---------------------------------------------------------------------------
# 2) Put common legend immediately below scenario/mode checkboxes, then filters.
# ---------------------------------------------------------------------------
needle = """                    self.scenario_map_filter_area,\n                    self.scenario_map_stack_area,\n"""
replacement = """                    self.scenario_map_filter_area,\n                    self.scenario_map_legend_area,\n                    self.scenario_map_stage_filter_area,\n                    self.scenario_map_item_filter_area,\n                    self.scenario_map_stack_area,\n"""
if needle not in main:
    raise SystemExit("결과 탭의 지도 컨트롤 배치 블록을 찾지 못했습니다.")
main = main.replace(needle, replacement, 1)

# ---------------------------------------------------------------------------
# 3) Remove duplicated Scope chart from Analysis only.  Results-tab Scope remains.
# ---------------------------------------------------------------------------
pattern = re.compile(
    r"\n\s*scope_frame\s*=\s*self\._scope_proxy_dataframe\(\)\s*\n"
    r"\s*if\s+not\s+scope_frame\.empty:\s*\n"
    r"\s*visual_controls\.append\(self\._scope_grouped_bar_block\(scope_frame\)\)\s*\n"
)
main, n_scope = pattern.subn(
    "\n        # v0.5.7: Scope 1+2 / Scope 3 chart is shown only in Tab 5 Results.\n",
    main,
    count=1,
)
if n_scope != 1:
    raise SystemExit(f"분석 탭 중복 Scope 그래프 블록 치환 실패: {n_scope}")

# ---------------------------------------------------------------------------
# 4) Add shared map stage/material filtering methods before scenario callback.
#    We filter a shallow result copy, so the existing validated map renderer can
#    stay untouched and the OR-Tools result object itself is never mutated.
# ---------------------------------------------------------------------------
anchor = "    def on_scenario_map_filter_changed(self, e) -> None:\n"
if anchor not in main:
    raise SystemExit("on_scenario_map_filter_changed 메서드를 찾지 못했습니다.")

methods = r'''    def _map_first_success_result(self):
        for scenario in ("S1", "S2", "S3"):
            for mode in ("line", "modular"):
                result = self.state.results.get((scenario, mode), {})
                if result.get("status") in VALID_STATUSES:
                    return result
        return None

    def _map_common_legend_control(self, result) -> ft.Control:
        rows: list[ft.Control] = []
        catalog = result.get("item_catalog", pd.DataFrame()) if result else pd.DataFrame()
        active = [str(x) for x in (result.get("active_item_ids", []) if result else [])]
        if isinstance(catalog, pd.DataFrame) and not catalog.empty and "item_id" in catalog.columns:
            cat = catalog.copy()
            cat["item_id"] = cat["item_id"].astype(str)
            if active:
                cat = cat[cat["item_id"].isin(active)]
                order = {item_id: i for i, item_id in enumerate(active)}
                cat["_order"] = cat["item_id"].map(order).fillna(9999)
                cat = cat.sort_values("_order", kind="stable")
            for _, row in cat.iterrows():
                color = str(row.get("color_hex", "#666666"))
                name = str(row.get("item_name_ko", row["item_id"]))
                rows.append(
                    ft.Row(
                        tight=True,
                        spacing=5,
                        controls=[
                            ft.Container(width=12, height=12, bgcolor=color, border_radius=6,
                                         border=ft.Border.all(1, ft.Colors.GREY_700)),
                            ft.Text(name, size=13),
                        ],
                    )
                )
        rows.extend([
            ft.Row(tight=True, spacing=5, controls=[
                ft.Container(width=12, height=12, bgcolor="#111827", border_radius=6),
                ft.Text("Stage 2 차량 조립지", size=13),
            ]),
            ft.Row(tight=True, spacing=5, controls=[
                ft.Text("━", color="#6a3d9a", weight=ft.FontWeight.BOLD),
                ft.Text("실선 = Stage 2→3 완성차 운송", size=13),
            ]),
            ft.Row(tight=True, spacing=5, controls=[
                ft.Text("┄", color=ft.Colors.GREY_800, weight=ft.FontWeight.BOLD),
                ft.Text("점선/파선 = Stage 1→2 원료·원자재 운송 (운송수단별 선형 구분)", size=13),
            ]),
            ft.Text("➤ 화살표 방향 = 도착지 · 흰 외곽선 = 겹치는 경로 구분", size=12, color=ft.Colors.GREY_700),
            ft.Text("Stage 1 원 크기 = 동일 원료 내 생산량 · Stage 2 원 크기 = 차량 조립량", size=12, color=ft.Colors.GREY_700),
        ])
        return ft.Container(
            padding=10,
            border=ft.Border.all(1, ft.Colors.GREY_300),
            border_radius=8,
            content=ft.Column(
                spacing=5,
                controls=[
                    ft.Text("지도 범례", weight=ft.FontWeight.BOLD),
                    ft.Row(wrap=True, spacing=14, run_spacing=6, controls=rows),
                ],
            ),
        )

    def _refresh_shared_map_filters(self, success_results) -> None:
        first = success_results[0][1] if success_results else self._map_first_success_result()
        if not first:
            self.scenario_map_legend_area.controls = []
            self.scenario_map_stage_filter_area.controls = []
            self.scenario_map_item_filter_area.controls = []
            return

        # The shared legend is deliberately placed directly below scenario/mode checkboxes.
        self.scenario_map_legend_area.controls = [self._map_common_legend_control(first)]

        stage_specs = [
            ("stage1", "Stage 1 생산지"),
            ("stage12", "Stage 1→2 원료·원자재 운송"),
            ("stage2", "Stage 2 차량 조립지"),
            ("stage23", "Stage 2→3 완성차 운송"),
        ]
        self.scenario_map_stage_filter_area.controls = [
            ft.Checkbox(
                label=label,
                value=code in self.map_stage_enabled,
                data=code,
                on_change=self.on_shared_map_stage_filter_changed,
            )
            for code, label in stage_specs
        ]

        catalog = first.get("item_catalog", pd.DataFrame())
        active = [str(x) for x in first.get("active_item_ids", [])]
        if not self._map_material_filter_initialized:
            self.map_visible_item_ids = set(active)
            self._map_material_filter_initialized = True
        else:
            self.map_visible_item_ids &= set(active)

        item_controls: list[ft.Control] = []
        if isinstance(catalog, pd.DataFrame) and not catalog.empty and "item_id" in catalog.columns:
            cat = catalog.copy()
            cat["item_id"] = cat["item_id"].astype(str)
            cat = cat[cat["item_id"].isin(active)]
            order = {item_id: i for i, item_id in enumerate(active)}
            cat["_order"] = cat["item_id"].map(order).fillna(9999)
            cat = cat.sort_values("_order", kind="stable")
            for _, row in cat.iterrows():
                item_id = str(row["item_id"])
                color = str(row.get("color_hex", "#666666"))
                label = str(row.get("item_name_ko", item_id))
                item_controls.append(
                    ft.Container(
                        padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                        border=ft.Border.all(1, ft.Colors.GREY_300),
                        border_radius=6,
                        content=ft.Row(
                            tight=True,
                            spacing=4,
                            controls=[
                                ft.Container(width=10, height=10, bgcolor=color, border_radius=5),
                                ft.Checkbox(
                                    label=label,
                                    value=item_id in self.map_visible_item_ids,
                                    data=item_id,
                                    on_change=self.on_shared_map_item_filter_changed,
                                ),
                            ],
                        ),
                    )
                )
        self.scenario_map_item_filter_area.controls = [
            ft.Text("지도에 표시할 원료·원자재", weight=ft.FontWeight.BOLD),
            *item_controls,
        ]

    def on_shared_map_stage_filter_changed(self, e) -> None:
        code = str(e.control.data)
        if bool(e.control.value):
            self.map_stage_enabled.add(code)
        else:
            self.map_stage_enabled.discard(code)
        self._refresh_scenario_map_stack()
        self.page.update()

    def on_shared_map_item_filter_changed(self, e) -> None:
        item_id = str(e.control.data)
        if bool(e.control.value):
            self.map_visible_item_ids.add(item_id)
        else:
            self.map_visible_item_ids.discard(item_id)
        self._refresh_scenario_map_stack()
        self.page.update()

    @staticmethod
    def _empty_like(frame):
        return frame.iloc[0:0].copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()

    def _filtered_map_result(self, result):
        """Return a map-only shallow copy respecting shared Stage/material filters."""
        filtered = dict(result)
        visible = set(self.map_visible_item_ids)

        catalog = result.get("item_catalog", pd.DataFrame())
        if isinstance(catalog, pd.DataFrame) and not catalog.empty and "item_id" in catalog.columns:
            filtered["item_catalog"] = catalog[catalog["item_id"].astype(str).isin(visible)].copy()
        filtered["active_item_ids"] = [
            str(iid) for iid in result.get("active_item_ids", []) if str(iid) in visible
        ]

        production = result.get("production_summary", pd.DataFrame())
        if "stage1" not in self.map_stage_enabled:
            filtered["production_summary"] = self._empty_like(production)
        elif isinstance(production, pd.DataFrame) and not production.empty and "item_id" in production.columns:
            filtered["production_summary"] = production[
                production["item_id"].astype(str).isin(visible)
            ].copy()

        inbound = result.get("inbound_routes", pd.DataFrame())
        if "stage12" not in self.map_stage_enabled:
            filtered["inbound_routes"] = self._empty_like(inbound)
        elif isinstance(inbound, pd.DataFrame) and not inbound.empty and "item_id" in inbound.columns:
            filtered["inbound_routes"] = inbound[
                inbound["item_id"].astype(str).isin(visible)
            ].copy()

        assembly = result.get("assembly_summary", pd.DataFrame())
        if "stage2" not in self.map_stage_enabled:
            filtered["assembly_summary"] = self._empty_like(assembly)

        market_routes = result.get("market_routes", pd.DataFrame())
        if "stage23" not in self.map_stage_enabled:
            filtered["market_routes"] = self._empty_like(market_routes)
            filtered["market"] = None

        return filtered

'''
main = main.replace(anchor, methods + anchor, 1)

# ---------------------------------------------------------------------------
# 5) Refresh the shared legend/filters before building the selected maps, and
#    pass a filtered result copy.  Per-map legends are disabled to avoid repeat UI.
# ---------------------------------------------------------------------------
needle = """        valid_results = [entry for entry in all_success if entry[0] in self.map_compare_enabled]\n        if not all_success:\n"""
replacement = """        self._refresh_shared_map_filters(all_success)\n        valid_results = [entry for entry in all_success if entry[0] in self.map_compare_enabled]\n        if not all_success:\n"""
if needle not in main:
    raise SystemExit("_refresh_scenario_map_stack의 valid_results 블록을 찾지 못했습니다.")
main = main.replace(needle, replacement, 1)

needle = """            for (scenario, mode), result in valid_results:\n                subtitle = (\n"""
replacement = """            for (scenario, mode), result in valid_results:\n                map_result = self._filtered_map_result(result)\n                subtitle = (\n"""
if needle not in main:
    raise SystemExit("지도 반복 렌더링 블록을 찾지 못했습니다.")
main = main.replace(needle, replacement, 1)

needle = """                        result,\n                        title=f\"{core.SCENARIO_SHORT[scenario]} · {UI_MODE_LABEL[mode]}\",\n                        subtitle=subtitle,\n                        height=380,\n                        show_legend=(len(controls) == 0),\n"""
replacement = """                        map_result,\n                        title=f\"{core.SCENARIO_SHORT[scenario]} · {UI_MODE_LABEL[mode]}\",\n                        subtitle=subtitle,\n                        height=380,\n                        show_legend=False,\n"""
if needle not in main:
    raise SystemExit("build_static_map_panel 호출 블록을 찾지 못했습니다.")
main = main.replace(needle, replacement, 1)

# ---------------------------------------------------------------------------
# 6) Lightweight route geometry.  This also reduces the invisible hover-marker
#    population in the existing v0.5.6 renderer because it samples route points.
# ---------------------------------------------------------------------------
logic, n_steps = re.subn(r"steps:\s*int\s*=\s*44", "steps: int = 18", logic, count=1)
if n_steps != 1:
    raise SystemExit(f"map_logic bezier steps 치환 실패: {n_steps}")

old_arrows = """        arrows = [\n            point_on_polyline(curve, fraction)\n            for fraction in (0.36, 0.64, 0.86)\n        ]\n"""
new_arrows = """        # v0.5.7 performance: one direction marker per route is enough visually.\n        arrows = [point_on_polyline(curve, 0.78)]\n"""
if old_arrows not in logic:
    raise SystemExit("Stage 1→2 arrow block을 찾지 못했습니다.")
logic = logic.replace(old_arrows, new_arrows, 1)

old = '                "arrows": [point_on_polyline(curve, f) for f in (0.36, 0.64, 0.86)],\n'
new = '                "arrows": [point_on_polyline(curve, 0.78)],\n'
if old not in logic:
    raise SystemExit("Stage 2→3 arrow block을 찾지 못했습니다.")
logic = logic.replace(old, new, 1)

# Add a small visible source marker without changing BUILD_ID (START_DEV verifier stays compatible).
main = main.replace(
    'APP_TITLE = "프랑스 전기자동차 보조금 탄소발자국 상한 대응 공급망 비용 최적화"',
    'APP_TITLE = "프랑스 전기자동차 보조금 탄소발자국 상한 대응 공급망 비용 최적화"\nMAP_UI_PATCH = "v0.5.7-stage-material-filter-performance"',
    1,
)

MAIN.write_text(main, encoding="utf-8")
MAP_LOGIC.write_text(logic, encoding="utf-8")

print("PASS: EVCO v0.5.7 map/result UI patch applied")
print("- Tab 6 duplicated Scope chart removed")
print("- common legend moved directly below scenario/mode map checkboxes")
print("- Stage 1 / Stage 1→2 / Stage 2 / Stage 2→3 shared checkboxes added")
print("- per-material map checkboxes restored")
print("- route curve points reduced 45→19 and arrows 3→1")
print("- OR-Tools model/data files were not changed")
print("Backups:", MAIN.with_suffix(MAIN.suffix + '.v056_backup'), MAP_LOGIC.with_suffix(MAP_LOGIC.suffix + '.v056_backup'))
