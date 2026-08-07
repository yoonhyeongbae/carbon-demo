from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import zipfile
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

import flet as ft
import numpy as np
import pandas as pd

import legacy_core as core
from ev_optimizer_api import default_country_map, result_to_zip_bytes

APP_TITLE = "프랑스 전기차 보조금 탄소발자국 상한 대응 공급망 비용 최적화"
VALID_STATUSES = {"OPTIMAL", "FEASIBLE"}


class NamedBytesIO(io.BytesIO):
    def __init__(self, payload: bytes, name: str):
        super().__init__(payload)
        self.name = name


@dataclass
class AppState:
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    results: Dict[Tuple[str, str], Dict] = field(default_factory=dict)
    active_item_ids: list[str] = field(default_factory=list)
    selected_transport_modes: list[str] = field(default_factory=lambda: list(core.ROUTE_MODE_CODES))
    country_maps: Dict[str, Dict[str, list[str]]] = field(default_factory=dict)


def fmt_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.6g}"
    return str(value)


def df_table(df: pd.DataFrame, max_rows: int = 200) -> ft.Control:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ft.Text("표시할 데이터가 없습니다.", color=ft.Colors.GREY_700)
    view = df.head(max_rows).copy()
    cols = [str(c) for c in view.columns]
    table = ft.DataTable(
        columns=[ft.DataColumn(label=ft.Text(c, weight=ft.FontWeight.BOLD, size=12)) for c in cols],
        rows=[
            ft.DataRow(cells=[ft.DataCell(ft.Text(fmt_value(row[c]), size=11, selectable=True)) for c in cols])
            for _, row in view.iterrows()
        ],
        column_spacing=18,
        data_row_min_height=38,
        data_row_max_height=58,
    )
    note = []
    if len(df) > max_rows:
        note.append(ft.Text(f"화면에는 처음 {max_rows:,}행만 표시합니다. 저장되는 CSV에는 전체 {len(df):,}행이 포함됩니다.", size=11, color=ft.Colors.GREY_700))
    return ft.Column([ft.Row([table], scroll=ft.ScrollMode.AUTO), *note], spacing=4)


def info_box(text: str, color: str = ft.Colors.BLUE_50) -> ft.Container:
    return ft.Container(
        content=ft.Text(text, size=13),
        bgcolor=color,
        border_radius=8,
        padding=12,
        border=ft.Border.all(1, ft.Colors.GREY_300),
    )


def section(title: str, *controls: ft.Control) -> ft.Column:
    return ft.Column([ft.Text(title, size=19, weight=ft.FontWeight.BOLD), *controls], spacing=8)


def svg_data_uri(svg: str) -> str:
    payload = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{payload}"


def _location_lookup(result: Mapping) -> Dict[str, Tuple[float, float]]:
    points: Dict[str, Tuple[float, float]] = {}
    plants = result.get("plants", pd.DataFrame())
    if isinstance(plants, pd.DataFrame):
        for _, r in plants.iterrows():
            points[str(r.get("location_name"))] = (float(r.get("latitude", 0)), float(r.get("longitude", 0)))
    market = result.get("market", {})
    if isinstance(market, Mapping) and market:
        points[str(market.get("market_name", market.get("location_name", "France")))] = (
            float(market.get("latitude", 46.2276)), float(market.get("longitude", 2.2137))
        )
    return points


def build_offline_svg_map(result: Mapping, view_code: int, visible_item_ids: Sequence[str]) -> str:
    """Generate an entirely local SVG supply-chain map; no web tile/API is requested."""
    width, height = 1200, 620
    left, top, plot_w, plot_h = 70, 45, 1060, 500
    locations = _location_lookup(result)
    visible = set(str(v) for v in visible_item_ids)
    catalog = result.get("active_item_table", pd.DataFrame())
    color_map = {}
    name_map = {}
    if isinstance(catalog, pd.DataFrame):
        for _, row in catalog.iterrows():
            iid = str(row.get("item_id"))
            color_map[iid] = str(row.get("color_hex", "#2563eb"))
            name_map[iid] = str(row.get("item_name_ko", iid))

    def xy(lat: float, lon: float) -> Tuple[float, float]:
        x = left + (float(lon) + 180.0) / 360.0 * plot_w
        y = top + (90.0 - float(lat)) / 180.0 * plot_h
        return x, y

    def loc_xy(name: str) -> Tuple[float, float] | None:
        point = locations.get(str(name))
        return None if point is None else xy(*point)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    out.append('<rect width="100%" height="100%" fill="#f8fafc"/>')
    out.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="14" fill="#eef4f8" stroke="#b8c4cf"/>')
    # coordinate grid makes the offline map geographically interpretable without external tiles.
    for lon in range(-150, 181, 30):
        x, _ = xy(0, lon)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#d9e2e8" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{top+plot_h+20}" text-anchor="middle" font-size="10" fill="#64748b">{lon}°</text>')
    for lat in range(-60, 91, 30):
        _, y = xy(lat, 0)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#d9e2e8" stroke-width="1"/>')
        out.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#64748b">{lat}°</text>')

    inbound = result.get("inbound_routes", pd.DataFrame())
    market_routes = result.get("market_routes", pd.DataFrame())
    prod = result.get("production_summary", pd.DataFrame())
    assy = result.get("assembly_summary", pd.DataFrame())

    if view_code in {0, 12} and isinstance(inbound, pd.DataFrame) and not inbound.empty:
        for _, row in inbound.iterrows():
            iid = str(row.get("item_id"))
            if iid not in visible:
                continue
            p1 = loc_xy(str(row.get("origin_location"))); p2 = loc_xy(str(row.get("assembly_location")))
            if not p1 or not p2:
                continue
            x1,y1=p1; x2,y2=p2; mx=(x1+x2)/2; my=min(y1,y2)-max(18,abs(x2-x1)*0.06)
            color = color_map.get(iid, "#2563eb")
            out.append(f'<path d="M{x1:.1f},{y1:.1f} Q{mx:.1f},{my:.1f} {x2:.1f},{y2:.1f}" fill="none" stroke="white" stroke-width="6" opacity="0.9"/>')
            out.append(f'<path d="M{x1:.1f},{y1:.1f} Q{mx:.1f},{my:.1f} {x2:.1f},{y2:.1f}" fill="none" stroke="{color}" stroke-width="3" opacity="0.85"/>')
    if view_code in {0, 23} and isinstance(market_routes, pd.DataFrame) and not market_routes.empty:
        for _, row in market_routes.iterrows():
            p1 = loc_xy(str(row.get("assembly_location"))); p2 = loc_xy(str(row.get("market_name")))
            if not p1 or not p2:
                continue
            x1,y1=p1; x2,y2=p2; mx=(x1+x2)/2; my=(y1+y2)/2-30
            out.append(f'<path d="M{x1:.1f},{y1:.1f} Q{mx:.1f},{my:.1f} {x2:.1f},{y2:.1f}" fill="none" stroke="#6a3d9a" stroke-width="4" opacity="0.8"/>')

    if view_code in {0,1,12} and isinstance(prod, pd.DataFrame) and not prod.empty:
        for _, row in prod.iterrows():
            iid = str(row.get("item_id"))
            if iid not in visible:
                continue
            name = str(row.get("origin_location")); p=loc_xy(name)
            if not p: continue
            x,y=p; flow=float(row.get("stage1_output", row.get("production_amount", 1)) or 1)
            r=max(5,min(15,4+math.sqrt(max(flow,0))*0.05))
            color = color_map.get(iid, "#2563eb")
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" fill-opacity="0.75" stroke="white" stroke-width="2"><title>{name} · {name_map.get(iid,iid)} · {flow:,.2f}</title></circle>')
    if view_code in {0,2,23} and isinstance(assy, pd.DataFrame) and not assy.empty:
        for _, row in assy.iterrows():
            name=str(row.get("assembly_location")); p=loc_xy(name)
            if not p: continue
            x,y=p; q=float(row.get("vehicle_equivalents",1) or 1); r=max(6,min(18,5+math.sqrt(max(q,0))*0.08))
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#111827" fill-opacity="0.8" stroke="white" stroke-width="2"><title>Stage 2 조립지 · {name} · {q:,.2f}</title></circle>')
    if view_code in {0,23}:
        for name,(lat,lon) in locations.items():
            if "france" in name.lower() or "프랑" in name:
                x,y=xy(lat,lon); out.append(f'<rect x="{x-6:.1f}" y="{y-6:.1f}" width="12" height="12" fill="#f59e0b" stroke="white" stroke-width="2"><title>Stage 3 시장 · {name}</title></rect>')

    out.append('<text x="70" y="585" font-size="12" fill="#475569">오프라인 좌표 지도: 경·위도 격자 + 공급망 노드/경로. 외부 타일, API, CDN을 호출하지 않습니다.</text>')
    lx=610; ly=575
    for iid in visible_item_ids:
        label = name_map.get(str(iid), str(iid)); color = color_map.get(str(iid), "#2563eb")
        out.append(f'<circle cx="{lx}" cy="{ly}" r="5" fill="{color}"/><text x="{lx+9}" y="{ly+4}" font-size="10" fill="#334155">{label}</text>')
        lx += 105
    out.append('</svg>')
    return "".join(out)


class DesktopApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.state = AppState()
        self.page.title = APP_TITLE
        self.page.padding = 12
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.window.width = 1540
        self.page.window.height = 940
        self.status = ft.Text("필수 CSV 13개 또는 ZIP을 불러오세요.", size=12, color=ft.Colors.BLUE_700)
        self.item_column = ft.Column(spacing=3)
        self.coefficient_area = ft.Column()
        self.input_table_area = ft.Column()
        self.country_areas = {"line": ft.Column(), "modular": ft.Column()}
        self.transport_column = ft.Column()
        self.index_preview = ft.Column()
        self.result_summary = ft.Column()
        self.result_choice = ft.Dropdown(width=420, on_select=self.refresh_result_details)
        self.map_stage = ft.Dropdown(width=420, value="0", options=[
            ft.DropdownOption(key="0", text="전체 공급망 전과정: Stage 1→2→3"),
            ft.DropdownOption(key="1", text="Stage 1 생산지"),
            ft.DropdownOption(key="12", text="Stage 1→2 원료·원자재 운송경로"),
            ft.DropdownOption(key="2", text="Stage 2 차량 조립지"),
            ft.DropdownOption(key="23", text="Stage 2→3 완성차 운송경로"),
        ], on_select=self.refresh_map)
        self.map_items_column = ft.Column(spacing=2)
        self.map_area = ft.Column()
        self.detail_area = ft.Column()
        self.analysis_page = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
        self.scenario_dd = ft.Dropdown(width=350, value="S1", options=[ft.DropdownOption(key=s, text=s) for s in ("S1","S2","S3")])
        self.mode_dd = ft.Dropdown(width=350, value="line", options=[
            ft.DropdownOption(key="line", text=core.MODE_LABEL["line"]),
            ft.DropdownOption(key="modular", text=core.MODE_LABEL["modular"]),
        ], on_select=lambda e: self.refresh_optimization_controls())
        self.time_limit = ft.TextField(label="Solver 제한시간(초)", value="180", width=220)
        self.tabs = ft.Tabs(length=5, expand=True, content=ft.Column(expand=True, controls=[
            ft.TabBar(tabs=[ft.Tab(label="1. 개요"), ft.Tab(label="2. 입력 데이터"), ft.Tab(label="3. 최적화"), ft.Tab(label="4. 결과"), ft.Tab(label="5. 분석")]),
            ft.TabBarView(expand=True, controls=[self.build_overview(), self.build_input_page(), self.build_optimization_page(), self.build_results_page(), self.analysis_page]),
        ]))
        self.page.add(ft.Column([ft.Text(APP_TITLE, size=26, weight=ft.FontWeight.BOLD), self.status, self.tabs], expand=True))
        self.refresh_overview()
        self.refresh_analysis()

    def snack(self, text: str, error: bool=False):
        self.page.show_dialog(ft.SnackBar(ft.Text(text), bgcolor=ft.Colors.RED_700 if error else ft.Colors.GREEN_700))

    def build_overview(self):
        self.overview_area=ft.Column(scroll=ft.ScrollMode.AUTO, expand=True); return self.overview_area

    def refresh_overview(self):
        stages=pd.DataFrame([
            ["Stage 1 생산지","투입 원료 → 원자재·중간재 생산","선택된 각 품목의 생산량·비용·탄소발자국·생산용량"],
            ["Stage 2 조립지","품목 운송 → 중간가공·차체 조립·추가 품목공정","품목별 유입량, 조립국가, 조립비·탄소발자국"],
            ["Stage 3 프랑스 시장","완성 전기자동차 운송 → 시장 출시","선택 제품구조 질량과 프랑스 수요 반영"],
        ], columns=["단계","핵심 흐름","최적화 반영"])
        controls=[ft.Text("사용자 업로드 데이터 기반 전기자동차 공급망 최적화",size=24,weight=ft.FontWeight.BOLD),
                  section("Stage 구조",df_table(stages)),
                  section("제품구조 변경 해석",info_box("원료를 체크 해제해도 다른 원료가 자동으로 증가하지 않습니다. 추가 품목을 체크하면 해당 BOM 질량과 Stage 1 생산·운송, Stage 2 조립 및 Stage 3 완성차 운송 항이 동일한 기준모형 인덱스에 추가됩니다.",ft.Colors.AMBER_50))]
        if self.state.tables:
            status=core.item_data_status(self.state.tables)
            if not status.empty: controls.append(section("현재 세션의 품목",df_table(status)))
        self.overview_area.controls=controls

    def build_input_page(self):
        self.input_page=ft.Column(scroll=ft.ScrollMode.AUTO,expand=True)
        self.input_page.controls=[
            ft.Text("사용자 CSV 데이터 및 제품구조 원료 선택",size=24,weight=ft.FontWeight.BOLD),
            section("2.1 필수 CSV 또는 ZIP 업로드",info_box("필수 CSV 13개 또는 ZIP을 선택하세요. 파일명은 기존 Streamlit 버전과 동일하게 정확히 일치해야 합니다."),
                    ft.Row([ft.Button("CSV/ZIP 불러오기",icon=ft.Icons.FOLDER_OPEN,on_click=self.pick_inputs),ft.Button("현재 입력 CSV ZIP 저장",icon=ft.Icons.DOWNLOAD,on_click=self.save_input_zip),ft.Button("입력·결과 초기화",icon=ft.Icons.DELETE_OUTLINE,on_click=self.reset_all)])),
            section("2.2 제품구조 원료 선택",self.item_column),
            section("2.3 선택 품목의 Stage 1·Stage 2·모듈 계수 확인",self.coefficient_area),
            section("2.4 현재 입력 테이블 확인",self.input_table_area),
        ]
        return self.input_page

    async def pick_inputs(self,e=None):
        files=await ft.FilePicker().pick_files(dialog_title="필수 CSV 13개 또는 ZIP 선택",allow_multiple=True,file_type=ft.FilePickerFileType.CUSTOM,allowed_extensions=["csv","zip"],with_data=True)
        if not files:return
        uploads=[]
        for f in files:
            payload=f.bytes
            if payload is None and f.path:
                payload=open(f.path,"rb").read()
            uploads.append(NamedBytesIO(payload or b"",f.name))
        parsed,messages=core.parse_full_data_uploads(uploads)
        merged={name:frame.copy() for name,frame in self.state.tables.items()}; merged.update(parsed)
        missing=[name for name in core.REQUIRED_FILES if name not in merged]
        if missing:
            self.status.value="필수 CSV가 부족합니다: "+", ".join(missing); self.status.color=ft.Colors.RED_700
        else:
            errors=core.validate_tables(merged)
            if errors:
                self.status.value="입력 데이터 검증 실패: "+" | ".join(errors[:3]); self.status.color=ft.Colors.RED_700
            else:
                self.state.tables=merged; self.state.results.clear(); self.state.active_item_ids=core._normalize_active_items(merged["item_catalog.csv"],None)
                self.state.country_maps={mode:default_country_map(merged,self.state.active_item_ids,mode) for mode in ("line","modular")}
                self.status.value="사용자 데이터 적용 및 검증 완료"; self.status.color=ft.Colors.GREEN_700
                self.refresh_after_data()
        if messages:self.status.value += " · " + messages[-1]
        self.page.update()

    async def save_input_zip(self,e=None):
        if not self.state.tables:
            self.snack("먼저 입력 데이터를 불러오세요.",True);return
        path=await ft.FilePicker().save_file(dialog_title="현재 입력 CSV ZIP 저장",file_name="current_ev_supply_chain_data_v21.zip",file_type=ft.FilePickerFileType.CUSTOM,allowed_extensions=["zip"],src_bytes=core.make_tables_zip(self.state.tables))
        if path:self.status.value=f"입력 ZIP 저장 완료: {path}";self.page.update()

    def reset_all(self,e=None):
        self.state=AppState(); self.status.value="입력 데이터와 최적화 결과를 초기화했습니다.";self.status.color=ft.Colors.BLUE_700;self.refresh_after_data();self.page.update()

    def refresh_after_data(self):
        self.refresh_overview();self.refresh_items();self.refresh_coefficients();self.refresh_input_tables();self.refresh_optimization_controls();self.refresh_results();self.refresh_analysis()

    def refresh_items(self):
        self.item_column.controls=[]
        if not self.state.tables:
            self.item_column.controls=[info_box("아직 적용된 데이터가 없습니다.")];return
        catalog=self.state.tables["item_catalog.csv"].sort_values("item_index")
        checks=[]
        for _,row in catalog.iterrows():
            iid=str(row["item_id"]); mandatory=int(row["mandatory"])==1
            cb=ft.Checkbox(label=str(row["item_name_ko"]),value=iid in self.state.active_item_ids,disabled=mandatory,data=iid,on_change=self.item_changed)
            checks.append(cb)
        self.item_column.controls=[ft.Row(checks,wrap=True)]

    def item_changed(self,e):
        iid=str(e.control.data); active=set(self.state.active_item_ids)
        if e.control.value: active.add(iid)
        else: active.discard(iid)
        catalog=self.state.tables["item_catalog.csv"].sort_values("item_index")
        for _,r in catalog.iterrows():
            if int(r["mandatory"])==1:active.add(str(r["item_id"]))
        self.state.active_item_ids=[i for i in catalog["item_id"].astype(str).tolist() if i in active]
        self.state.results.clear();self.state.country_maps={m:default_country_map(self.state.tables,self.state.active_item_ids,m) for m in ("line","modular")};self.refresh_coefficients();self.refresh_optimization_controls();self.refresh_results();self.refresh_analysis();self.page.update()

    def refresh_coefficients(self):
        self.coefficient_area.controls=[]
        if not self.state.tables:return
        catalog=self.state.tables["item_catalog.csv"].set_index("item_id");sup=self.state.tables["item_suppliers.csv"];proc=self.state.tables["stage2_item_processes.csv"];mod=self.state.tables["module_parameters.csv"]
        for iid in self.state.active_item_ids:
            item_name=str(catalog.loc[iid,"item_name_ko"]);s1=sup[sup["item_id"].astype(str)==iid].copy();s2=proc[proc["item_id"].astype(str)==iid].copy();mr=mod[mod["item_id"].astype(str)==iid].copy()
            panel=ft.ExpansionPanel(header=ft.ListTile(title=ft.Text(f"{item_name} ({iid}) 계수")),content=ft.Container(padding=12,content=ft.Column([section("Stage 1 생산계수",df_table(s1)),section("Stage 2 허용국가·공정 참고정보",df_table(s2)),section("대형·소형 모듈 및 효율계수",df_table(mr))])))
            self.coefficient_area.controls.append(ft.ExpansionPanelList(controls=[panel]))

    def refresh_input_tables(self):
        self.input_table_area.controls=[]
        if not self.state.tables:return
        for name in core.REQUIRED_FILES:
            self.input_table_area.controls.append(ft.ExpansionPanelList(controls=[ft.ExpansionPanel(header=ft.ListTile(title=ft.Text(name)),content=ft.Container(padding=12,content=df_table(self.state.tables[name]))) ]))

    def build_country_panel(self,mode:str)->ft.Control:
        area=self.country_areas[mode]
        return section("라인 생산 국가선택" if mode=="line" else "모듈 생산 국가선택",area)

    def build_optimization_page(self):
        self.optimization_page=ft.Column(scroll=ft.ScrollMode.AUTO,expand=True)
        self.optimization_page.controls=[ft.Text("최적화 실행",size=24,weight=ft.FontWeight.BOLD),ft.Row([ft.Button("최적화 결과 초기화",on_click=self.reset_results)]),self.index_preview,self.build_country_panel("line"),self.build_country_panel("modular"),section("허용 외부 운송수단 선택",self.transport_column),ft.Row([self.scenario_dd,self.mode_dd,self.time_limit],wrap=True),ft.Row([ft.Button("선택 조합 실행",icon=ft.Icons.PLAY_ARROW,on_click=self.run_selected),ft.Button("3개 시나리오 × 2개 생산방식 실행",icon=ft.Icons.PLAY_CIRCLE,on_click=self.run_all)]),self.result_summary]
        return self.optimization_page

    def reset_results(self,e=None):
        self.state.results.clear();self.status.value="최적화 결과를 초기화했습니다.";self.refresh_results();self.refresh_analysis();self.page.update()

    def _update_assembly(self,mode:str):
        cmap=self.state.country_maps.setdefault(mode,{})
        order=self.state.tables["assembly_locations.csv"]["location_name"].astype(str).tolist();common=set(order)
        for iid in self.state.active_item_ids:common &= set(cmap.get(f"stage2::{iid}",[]))
        cmap["assembly"]=[name for name in order if name in common]

    def country_changed(self,e):
        mode,item,key,name=e.control.data;cmap=self.state.country_maps.setdefault(mode,{})
        current=list(cmap.get(key,[]));s=set(current)
        if e.control.value:s.add(name)
        else:s.discard(name)
        order=self.state.tables["assembly_locations.csv"]["location_name"].astype(str).tolist();cmap[key]=[n for n in order if n in s]
        if mode=="line":
            cmap[f"stage1::{item}"]=list(cmap[key]);cmap[f"stage2::{item}"]=list(cmap[key])
        self._update_assembly(mode);self.refresh_index_preview();self.page.update()

    def refresh_country_area(self,mode:str):
        area=self.country_areas[mode];area.controls=[]
        if not self.state.tables:return
        cmap=self.state.country_maps[mode];catalog=self.state.tables["item_catalog.csv"].set_index("item_id")
        for iid in self.state.active_item_ids:
            if mode=="line":keys=[f"stage1::{iid}"];labels=["동일 생산·가공·조립국가"]
            else:keys=[f"stage1::{iid}",f"stage2::{iid}"];labels=["Stage 1 모듈 생산지","Stage 2 가공·차량 조립지"]
            cols=[]
            default=default_country_map(self.state.tables,[iid],mode)
            for key,label in zip(keys,labels):
                available=default.get(key,[]);selected=set(cmap.get(key,[]));checks=[ft.Checkbox(label=name,value=name in selected,data=(mode,iid,key,name),on_change=self.country_changed) for name in available]
                cols.append(section(label,ft.Row(checks,wrap=True)))
            area.controls.append(ft.ExpansionPanelList(controls=[ft.ExpansionPanel(header=ft.ListTile(title=ft.Text(str(catalog.loc[iid,"item_name_ko"]))),content=ft.Container(padding=10,content=ft.Column(cols))) ]))
        label="라인 생산 공통국가" if mode=="line" else "모듈 생산 공통 Stage 2 차량 조립국가"
        area.controls.append(info_box(f"{label}: {len(cmap.get('assembly',[]))}개 — "+", ".join(cmap.get("assembly",[]))))

    def transport_changed(self,e):
        code=str(e.control.data);s=set(self.state.selected_transport_modes)
        if e.control.value:s.add(code)
        else:s.discard(code)
        self.state.selected_transport_modes=[c for c in core.ROUTE_MODE_CODES if c in s];self.refresh_index_preview();self.page.update()

    def refresh_transport(self):
        self.transport_column.controls=[ft.Row([ft.Checkbox(label=core.ROUTE_MODE_LABEL[c],value=c in self.state.selected_transport_modes,data=c,on_change=self.transport_changed) for c in core.ROUTE_MODE_CODES],wrap=True)]

    def refresh_optimization_controls(self):
        if not self.state.tables:
            for a in self.country_areas.values():a.controls=[]
            self.transport_column.controls=[];self.index_preview.controls=[info_box("2번 탭에서 필수 CSV를 먼저 불러오세요.")];return
        if not self.state.country_maps:self.state.country_maps={m:default_country_map(self.state.tables,self.state.active_item_ids,m) for m in ("line","modular")}
        self.refresh_country_area("line");self.refresh_country_area("modular");self.refresh_transport();self.refresh_index_preview()

    def refresh_index_preview(self):
        if not self.state.tables:return
        controls=[ft.Text("선택 제품구조 미리보기",size=18,weight=ft.FontWeight.BOLD),df_table(core.product_structure_preview(self.state.tables,self.state.active_item_ids))]
        for mode in ("line","modular"):
            cmap=self.state.country_maps[mode];preview=core.active_index_preview(self.state.active_item_ids,cmap,self.state.selected_transport_modes);controls.extend([ft.Text(f"{core.MODE_LABEL[mode]} 활성 인덱스",weight=ft.FontWeight.BOLD),df_table(preview)])
        self.index_preview.controls=controls

    def selection_valid(self,mode:str)->tuple[bool,str]:
        if not self.state.tables:return False,"입력 데이터가 없습니다."
        cmap=self.state.country_maps.get(mode,{});required=[f"stage1::{i}" for i in self.state.active_item_ids]+[f"stage2::{i}" for i in self.state.active_item_ids]+["assembly"]
        if not all(cmap.get(k) for k in required):return False,"원료별 Stage 1·2 및 공통 조립국가를 최소 1개 선택하세요."
        if not self.state.selected_transport_modes:return False,"외부 운송수단을 최소 1개 선택하세요."
        assemblies=list(cmap.get("assembly",[]))
        if mode=="line":
            feasible=[a for a in assemblies if all(a in cmap.get(f"stage1::{i}",[]) for i in self.state.active_item_ids)]
            if not feasible:return False,"전 품목 라인 생산 조건(origin = assembly)을 만족하는 공통 국가가 없습니다."
        else:
            feasible=[a for a in assemblies if all(any(o!=a for o in cmap.get(f"stage1::{i}",[])) for i in self.state.active_item_ids)]
            if not feasible:return False,"전 품목 모듈 생산 조건(origin ≠ assembly)을 만족하는 공통 차량 조립국가가 없습니다."
        return True,""

    async def solve_one(self,scenario:str,mode:str)->Dict:
        valid,message=self.selection_valid(mode)
        if not valid:return {"status":"ERROR","message":message,"scenario_id":scenario,"production_mode":mode}
        try:limit=max(10,min(600,int(float(self.time_limit.value or "180"))))
        except ValueError:limit=180
        return await asyncio.to_thread(core.solve_case,self.state.tables,scenario,mode,limit,self.state.active_item_ids,self.state.country_maps[mode],self.state.selected_transport_modes)

    async def run_selected(self,e=None):
        if not self.state.tables:self.snack("먼저 필수 CSV를 불러오세요.",True);return
        scenario=str(self.scenario_dd.value or "S1");mode=str(self.mode_dd.value or "line");self.status.value=f"{scenario} · {core.MODE_LABEL[mode]} 계산 중...";self.page.update()
        result=await self.solve_one(scenario,mode);self.state.results[(scenario,mode)]=result;self.status.value=f"{scenario} · {core.MODE_LABEL[mode]}: {result.get('status')}";self.status.color=ft.Colors.GREEN_700 if result.get("status") in VALID_STATUSES else ft.Colors.RED_700;self.refresh_results();self.refresh_analysis();self.page.update()

    async def run_all(self,e=None):
        if not self.state.tables:self.snack("먼저 필수 CSV를 불러오세요.",True);return
        cases=[(s,m) for s in ("S1","S2","S3") for m in ("line","modular")]
        for idx,(s,m) in enumerate(cases,1):
            self.status.value=f"[{idx}/6] {core.SCENARIO_SHORT[s]} · {core.MODE_LABEL[m]} 계산 중...";self.page.update();self.state.results[(s,m)]=await self.solve_one(s,m)
        self.status.value="6개 조합 계산을 완료했습니다.";self.status.color=ft.Colors.GREEN_700;self.refresh_results();self.refresh_analysis();self.page.update()

    def metric_cards(self,result:Mapping)->ft.Control:
        if result.get("status") not in VALID_STATUSES:return info_box(f"{result.get('status')}: {result.get('message','해를 찾지 못했습니다.')}",ft.Colors.RED_50)
        vals=[("Solver",result.get("status")),("총비용(EUR)",f"{float(result.get('objective_value',0)):,.2f}"),("총 탄소발자국(kgCO2-eq)",f"{float(result.get('total_emissions_kgco2',0)):,.2f}"),("상한 이용률",fmt_value(result.get("fleet_cap_utilization_pct"))+"%")]
        return ft.Row([ft.Container(width=245,padding=12,border=ft.Border.all(1,ft.Colors.GREY_300),border_radius=8,content=ft.Column([ft.Text(k,size=11,color=ft.Colors.GREY_700),ft.Text(str(v),size=17,weight=ft.FontWeight.BOLD)])) for k,v in vals],wrap=True)

    def build_results_page(self):
        self.results_page=ft.Column(scroll=ft.ScrollMode.AUTO,expand=True)
        self.results_page.controls=[ft.Text("최적화 결과",size=24,weight=ft.FontWeight.BOLD),self.result_summary,ft.Row([self.result_choice,ft.Button("선택 결과 ZIP 저장",icon=ft.Icons.DOWNLOAD,on_click=self.save_selected_result)],wrap=True),self.detail_area,section("전체 및 단계별 공급망 지도",info_box("기존 CARTO/Leaflet 온라인 배경지도 대신 완전 로컬 SVG 좌표 지도를 사용합니다. 공급망 노드, 원료별 경로, Stage 2→3 경로 및 원료 색상은 유지하며 인터넷 호출이 없습니다."),ft.Row([self.result_choice,self.map_stage],wrap=True),self.map_items_column,self.map_area)]
        return self.results_page

    def _result_key_text(self,key):return f"{core.SCENARIO_SHORT[key[0]]} · {core.MODE_LABEL[key[1]]}"

    def refresh_results(self):
        controls=[]
        if not self.state.results:controls=[info_box("3번 탭에서 최적화를 실행하세요.")]
        else:
            for s in ("S1","S2","S3"):
                controls.append(ft.Text(core.SCENARIO_SHORT[s],size=18,weight=ft.FontWeight.BOLD))
                for m in ("line","modular"):
                    r=self.state.results.get((s,m));controls.append(ft.Text(core.MODE_LABEL[m],weight=ft.FontWeight.BOLD));controls.append(info_box("미실행") if not r else self.metric_cards(r))
        self.result_summary.controls=controls
        valid=[k for k,v in self.state.results.items() if v.get("status") in VALID_STATUSES]
        self.result_choice.options=[ft.DropdownOption(key=f"{k[0]}|{k[1]}",text=self._result_key_text(k)) for k in valid]
        if valid and (not self.result_choice.value or self.result_choice.value not in [f"{k[0]}|{k[1]}" for k in valid]):self.result_choice.value=f"{valid[0][0]}|{valid[0][1]}"
        self.refresh_result_details(update_page=False);self.refresh_map(update_page=False)

    def _selected_result(self):
        if not self.result_choice.value:return None,None
        s,m=str(self.result_choice.value).split("|",1);return (s,m),self.state.results.get((s,m))

    def refresh_result_details(self,e=None,update_page=True):
        key,result=self._selected_result();self.detail_area.controls=[]
        if not result:return
        detail_keys=[("product_structure_summary","제품구조"),("product_summary","차량별 결과"),("production_summary","Stage 1 생산"),("inbound_routes","Stage 1→2 운송"),("stage2_item_process_summary","Stage 2 원료별 공정"),("assembly_summary","Stage 2 조립지 합계"),("module_summary","모듈 결과"),("market_routes","Stage 3 시장 출시"),("solver","Solver 정보")]
        dd=ft.Dropdown(width=380,value="product_summary",options=[ft.DropdownOption(key=k,text=t) for k,t in detail_keys])
        def changed(evt):self.render_detail_table(result,str(dd.value))
        dd.on_select=changed;self.detail_area.controls=[dd];self.render_detail_table(result,"product_summary",append=True)
        if update_page:self.refresh_map(update_page=False);self.page.update()

    def render_detail_table(self,result:Mapping,detail_key:str,append:bool=False):
        if not append:self.detail_area.controls=self.detail_area.controls[:1]
        if detail_key=="solver":
            fields=["status","message","solver_name","solver_version","solver_iterations","variable_count","constraint_count","matrix_nonzeros","wall_time_sec","structure_signature","active_item_ids","selected_country_map","selected_transport_modes","route_mode_codes","active_index_sizes","fleet_total_cap_kgco2","fleet_cap_utilization_pct","fleet_cap_met","objective_reconstruction_gap_eur","line_all_item_colocation","modular_all_item_separation","modular_same_country_positive_flow_count"]
            self.detail_area.controls.append(ft.Text(json.dumps({f:result.get(f) for f in fields},ensure_ascii=False,indent=2,default=str),selectable=True,font_family="Consolas"))
        else:self.detail_area.controls.append(df_table(result.get(detail_key,pd.DataFrame()),max_rows=500))
        self.page.update()

    def refresh_map(self,e=None,update_page=True):
        key,result=self._selected_result();self.map_area.controls=[];self.map_items_column.controls=[]
        if not result:return
        view_code=int(self.map_stage.value or 0);catalog=result.get("active_item_table",pd.DataFrame());ids=catalog.get("item_id",pd.Series(dtype=str)).astype(str).tolist() if isinstance(catalog,pd.DataFrame) else []
        visible=ids
        if view_code in {0,1,12}:
            checks=[ft.Checkbox(label=str(catalog.loc[catalog["item_id"].astype(str)==iid,"item_name_ko"].iloc[0]) if not catalog.empty else iid,value=True,data=iid,on_change=self.map_item_changed) for iid in ids]
            self.map_items_column.controls=[ft.Row(checks,wrap=True)]
        svg=build_offline_svg_map(result,view_code,visible)
        self.map_area.controls=[ft.InteractiveViewer(content=ft.Image(src=svg_data_uri(svg),width=1200,height=620),min_scale=0.5,max_scale=5,boundary_margin=40)]
        if update_page:self.page.update()

    def map_item_changed(self,e=None):
        key,result=self._selected_result()
        if not result:return
        visible=[c.data for row in self.map_items_column.controls if isinstance(row,ft.Row) for c in row.controls if isinstance(c,ft.Checkbox) and c.value]
        if not visible:self.map_area.controls=[info_box("지도에 표시할 원료를 최소 1개 선택하세요.",ft.Colors.AMBER_50)]
        else:self.map_area.controls=[ft.InteractiveViewer(content=ft.Image(src=svg_data_uri(build_offline_svg_map(result,int(self.map_stage.value or 0),visible)),width=1200,height=620),min_scale=0.5,max_scale=5,boundary_margin=40)]
        self.page.update()

    async def save_selected_result(self,e=None):
        key,result=self._selected_result()
        if not result or not key:return
        s,m=key
        path=await ft.FilePicker().save_file(dialog_title="최적화 결과 ZIP 저장",file_name=f"EV_optimization_{s}_{m}.zip",file_type=ft.FilePickerFileType.CUSTOM,allowed_extensions=["zip"],src_bytes=result_to_zip_bytes(result))
        if path:self.status.value=f"결과 ZIP 저장 완료: {path}";self.status.color=ft.Colors.GREEN_700;self.page.update()

    def _bar_chart(self,comparison:pd.DataFrame,col:str,title:str)->ft.Control:
        if comparison.empty or col not in comparison:return ft.Text("표시할 결과가 없습니다.")
        values=pd.to_numeric(comparison[col],errors="coerce").fillna(0);max_v=max(float(values.max()),1.0);rows=[ft.Text(title,size=18,weight=ft.FontWeight.BOLD)]
        for (_,row),value in zip(comparison.iterrows(),values):
            label=f"{row['시나리오']} · {row['생산방식']}";rows.append(ft.Row([ft.Text(label,width=330,size=12),ft.ProgressBar(value=max(0,min(1,float(value)/max_v)),width=520),ft.Text(f"{float(value):,.2f}",width=150)]))
        return ft.Column(rows)

    def refresh_analysis(self):
        controls=[ft.Text("분석 및 결론",size=24,weight=ft.FontWeight.BOLD)];comparison=core.comparison_dataframe(self.state.results)
        if comparison.empty:self.analysis_page.controls=controls+[info_box("분석할 OPTIMAL 또는 FEASIBLE 결과가 없습니다.")];return
        controls += [section("시나리오·생산방식 비교",df_table(comparison)),self._bar_chart(comparison,"총비용(EUR)","총비용 비교"),self._bar_chart(comparison,"회사 전체 탄소발자국(kgCO2-eq)","회사 전체 탄소발자국 비교")]
        delta=[]
        for s in ("S1","S2","S3"):
            line=self.state.results.get((s,"line"),{});mod=self.state.results.get((s,"modular"),{})
            if line.get("status") not in VALID_STATUSES or mod.get("status") not in VALID_STATUSES:continue
            inbound=mod.get("inbound_routes",pd.DataFrame());external=inbound.loc[~inbound["internal_flow"].astype(bool)].copy() if isinstance(inbound,pd.DataFrame) and not inbound.empty else pd.DataFrame()
            delta.append({"시나리오":core.SCENARIO_SHORT[s],"전 품목 모듈-라인 비용차(EUR)":float(mod["objective_value"])-float(line["objective_value"]),"전 품목 모듈-라인 탄소차이(kgCO2-eq)":float(mod["total_emissions_kgco2"])-float(line["total_emissions_kgco2"]),"모듈 방식 외부 운송질량(kg)":float(external.get("transport_mass_kg",pd.Series(dtype=float)).sum()) if not external.empty else 0.0,"외부 운송 품목 수":int(external.get("item_id",pd.Series(dtype=str)).astype(str).nunique()) if not external.empty else 0,"양의 원료·중간재 경로 수":int(len(external))})
        if delta:controls.append(section("라인 생산 대비 전 품목 모듈 분산 생산 차이",df_table(pd.DataFrame(delta))))
        controls.append(section("제품구조 변경 해석",info_box("사용자가 추가 품목을 체크하면 동일한 RP·RT·FP·FT 수식의 원료 인덱스 R에 추가됩니다. 기존 품목을 해제하면 해당 품목의 생산·운송·질량이 제거되고 다른 재료가 자동 대체되지 않습니다. 결과 사용 전 입력 데이터의 출처·단위·시스템 경계를 검증해야 합니다.",ft.Colors.AMBER_50)))
        self.analysis_page.controls=controls


def main(page:ft.Page):
    DesktopApp(page)


if __name__=="__main__":
    ft.run(main)
