from __future__ import annotations

"""Offline/public-domain supply-chain map renderer for EV Carbon Optimizer.

The module deliberately does not create a TileLayer and does not contact CARTO,
OpenStreetMap tile servers, or any other hosted basemap provider.  Country
polygons are read from the bundled Natural Earth 1:110m Admin-0 shapefile.

Natural Earth Terms of Use:
https://www.naturalearthdata.com/about/terms-of-use/
"""

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence
import math

import flet as ft
try:
    import flet_map as ftm
except Exception:  # pragma: no cover
    ftm = None

import map_logic as logic

try:
    import map_view as _legacy
except Exception:  # pragma: no cover
    _legacy = None

try:
    import shapefile  # pyshp
except Exception:  # pragma: no cover
    shapefile = None

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "natural_earth"
SHP_PATH = ASSET_DIR / "ne_110m_admin_0_countries.shp"


def _coord(lat: float, lon: float):
    if _legacy is not None and hasattr(_legacy, "_coord"):
        return _legacy._coord(float(lat), float(lon))
    return ftm.MapLatitudeLongitude(latitude=float(lat), longitude=float(lon))


def _safe_color(value, default="#d9dee3"):
    text = str(value or "").strip()
    return text if text else default


@lru_cache(maxsize=1)
def _natural_earth_polygons():
    if ftm is None or shapefile is None or not SHP_PATH.exists():
        return []
    reader = shapefile.Reader(str(SHP_PATH), encoding="utf-8")
    polygons = []
    for shape in reader.shapes():
        points = list(shape.points)
        if len(points) < 3:
            continue
        starts = list(shape.parts) + [len(points)]
        for i in range(len(starts) - 1):
            ring = points[starts[i]:starts[i + 1]]
            if len(ring) < 3:
                continue
            # 1:110m is already coarse.  Additional point decimation keeps
            # Flutter rendering light when six scenario maps are visible.
            if len(ring) > 180:
                step = max(1, len(ring) // 180)
                ring = ring[::step]
            coords = [_coord(y, x) for x, y in ring]
            try:
                polygons.append(
                    ftm.PolygonMarker(
                        coordinates=coords,
                        color="#eef1f4",
                        border_color="#c7cdd3",
                        border_stroke_width=0.7,
                    )
                )
            except TypeError:
                polygons.append(ftm.PolygonMarker(coordinates=coords, color="#eef1f4"))
    return polygons


def _natural_earth_layer():
    polys = _natural_earth_polygons()
    try:
        return ftm.PolygonLayer(
            polygons=polys,
            polygon_culling=True,
            simplification_tolerance=0.35,
            use_alternative_rendering=True,
        )
    except TypeError:
        try:
            return ftm.PolygonLayer(
                polygons=polys,
                polygon_culling=True,
                simplification_tolerance=0.35,
            )
        except TypeError:
            return ftm.PolygonLayer(polygons=polys)


def _route_polyline(route: Mapping):
    if _legacy is not None and hasattr(_legacy, "_route_polyline"):
        return _legacy._route_polyline(route)
    points = list(route.get("points", []))
    kwargs = dict(
        coordinates=[_coord(lat, lon) for lat, lon in points],
        color=_safe_color(route.get("color"), "#666666"),
        stroke_width=float(route.get("width", 2.5)),
        border_color="#ffffff",
        border_stroke_width=1.5,
    )
    return ftm.PolylineMarker(**kwargs)


def _stage1_circle(node: Mapping):
    if _legacy is not None and hasattr(_legacy, "_stage1_circle"):
        return _legacy._stage1_circle(node)
    return ftm.CircleMarker(
        coordinates=_coord(node["latitude"], node["longitude"]),
        radius=float(node.get("radius", 8)),
        color=_safe_color(node.get("color"), "#e53935"),
        border_color="#ffffff",
        border_stroke_width=1.5,
    )


def _stage2_circle(node: Mapping):
    if _legacy is not None and hasattr(_legacy, "_stage2_circle"):
        return _legacy._stage2_circle(node)
    return ftm.CircleMarker(
        coordinates=_coord(node["latitude"], node["longitude"]),
        radius=float(node.get("radius", 9)),
        color=getattr(logic, "ASSEMBLY_COLOR", "#111827"),
        border_color="#ffffff",
        border_stroke_width=1.7,
    )


def _arrow_marker(route: Mapping):
    arrows = list(route.get("arrows", []))
    if not arrows:
        return None
    arrow = arrows[-1]
    if _legacy is not None and hasattr(_legacy, "_arrow_marker"):
        try:
            return _legacy._arrow_marker(route, arrow, 21.0)
        except Exception:
            pass
    if isinstance(arrow, Mapping):
        lat = float(arrow.get("latitude", arrow.get("lat", 0)))
        lon = float(arrow.get("longitude", arrow.get("lon", 0)))
    else:
        lat, lon = map(float, arrow[:2])
    tooltip = str(route.get("tooltip", route.get("label", "운송 경로 · 화살표 방향 = 도착지")))
    return ftm.Marker(
        coordinates=_coord(lat, lon),
        width=25,
        height=25,
        content=ft.Text("➤", size=20, color=_safe_color(route.get("color"), "#555555"), tooltip=tooltip),
    )


def _market_marker(market: Mapping | None):
    if not market:
        return None
    try:
        lat = float(market["latitude"])
        lon = float(market["longitude"])
    except Exception:
        return None
    name = str(market.get("location", market.get("location_name", "프랑스 시장")))
    return ftm.Marker(
        coordinates=_coord(lat, lon),
        width=38,
        height=38,
        content=ft.Icon(ft.Icons.SHOPPING_CART, color="#f28e2b", size=28, tooltip=f"Stage 3 프랑스 시장: {name}"),
    )


def _filter_payload(payload: dict, visible_stages: set[str] | None):
    if visible_stages is None:
        return payload
    enabled = set(visible_stages)
    if "stage1" not in enabled:
        payload["stage1_nodes"] = []
    if "stage12" not in enabled:
        payload["stage12_routes"] = []
    if "stage2" not in enabled:
        payload["stage2_nodes"] = []
    if "stage23" not in enabled:
        payload["stage23_routes"] = []
        payload["stage3_node"] = None
    return payload


def _bounds(payload: Mapping):
    pts = []
    for key in ("stage1_nodes", "stage2_nodes"):
        for node in payload.get(key, []):
            pts.append((float(node["latitude"]), float(node["longitude"])))
    for key in ("stage12_routes", "stage23_routes"):
        for route in payload.get(key, []):
            line = list(route.get("points", []))
            if line:
                pts.extend([tuple(line[0]), tuple(line[-1])])
    market = payload.get("stage3_node")
    if market:
        try:
            pts.append((float(market["latitude"]), float(market["longitude"])))
        except Exception:
            pass
    return pts


def _active_catalog(result: Mapping, visible_item_ids=None):
    try:
        return logic.active_item_table(result, visible_item_ids)
    except Exception:
        return result.get("item_catalog")


def build_map_legend(result: Mapping, visible_item_ids=None, visible_stages=None):
    enabled = set(visible_stages or ("stage1", "stage12", "stage2", "stage23"))
    rows = []
    catalog = _active_catalog(result, visible_item_ids)
    if catalog is not None and hasattr(catalog, "empty") and not catalog.empty:
        for _, row in catalog.iterrows():
            rows.append(
                ft.Row(
                    tight=True,
                    spacing=4,
                    controls=[
                        ft.Container(width=10, height=10, bgcolor=_safe_color(row.get("color_hex")), border_radius=5),
                        ft.Text(str(row.get("item_name_ko", row.get("item_id", ""))), size=11),
                    ],
                )
            )
    if "stage2" in enabled:
        rows.append(ft.Row(tight=True, spacing=4, controls=[ft.Container(width=10, height=10, bgcolor=getattr(logic, "ASSEMBLY_COLOR", "#111827"), border_radius=5), ft.Text("Stage 2 차량 조립지", size=11)]))
    if "stage12" in enabled:
        rows.append(ft.Text("┄ Stage 1→2 원료·원자재 운송", size=11))
    if "stage23" in enabled:
        rows.append(ft.Text("━ Stage 2→3 완성차 운송", size=11, color=getattr(logic, "FINISHED_COLOR", "#6a3d9a")))
    return ft.Container(
        padding=8,
        border=ft.Border.all(1, ft.Colors.GREY_300),
        border_radius=8,
        content=ft.Column(
            spacing=4,
            controls=[
                ft.Text("지도 범례", weight=ft.FontWeight.BOLD),
                ft.Row(wrap=True, spacing=14, run_spacing=4, controls=rows),
                ft.Text("➤ 화살표 = 도착방향 · 배경지도: Natural Earth 1:110m (public domain)", size=10.5, color=ft.Colors.GREY_700),
            ],
        ),
    )


def build_static_map_panel(
    page: ft.Page,
    result: Mapping,
    *,
    title: str = "",
    subtitle: str = "",
    height: int = 420,
    show_legend: bool = False,
    visible_item_ids=None,
    visible_stages=None,
    lightweight: bool = True,
):
    if ftm is None:
        return ft.Text("flet-map 패키지가 설치되지 않았습니다.", color=ft.Colors.RED_700)
    payload = logic.build_map_payload(result, view_code=0, visible_item_ids=visible_item_ids)
    payload = _filter_payload(payload, set(visible_stages) if visible_stages is not None else None)

    polylines = [_route_polyline(r) for r in list(payload.get("stage12_routes", [])) + list(payload.get("stage23_routes", []))]
    circles = [_stage1_circle(n) for n in payload.get("stage1_nodes", [])] + [_stage2_circle(n) for n in payload.get("stage2_nodes", [])]
    markers = [m for m in [_arrow_marker(r) for r in list(payload.get("stage12_routes", [])) + list(payload.get("stage23_routes", []))] if m is not None]
    mm = _market_marker(payload.get("stage3_node"))
    if mm is not None:
        markers.append(mm)

    layers = [_natural_earth_layer()]
    if polylines:
        try:
            layers.append(ftm.PolylineLayer(polylines=polylines, simplification_tolerance=0.35, culling_margin=6))
        except TypeError:
            layers.append(ftm.PolylineLayer(polylines=polylines))
    if circles:
        layers.append(ftm.CircleLayer(circles=circles))
    if markers:
        layers.append(ftm.MarkerLayer(markers=markers, rotate=False))
    try:
        layers.append(ftm.SimpleAttribution(text="Made with Natural Earth · public domain"))
    except Exception:
        pass

    pts = _bounds(payload)
    camera_fit = None
    if pts:
        try:
            camera_fit = ftm.CameraFit(coordinates=[_coord(lat, lon) for lat, lon in pts], padding=ft.Padding.all(45), max_zoom=4.5)
        except Exception:
            camera_fit = None
    kwargs = dict(
        expand=True,
        bgcolor="#dfe7ec",
        initial_center=_coord(28.0, 20.0),
        initial_zoom=2.2,
        min_zoom=1.5,
        max_zoom=12.0,
        layers=layers,
    )
    if camera_fit is not None:
        kwargs["initial_camera_fit"] = camera_fit
    try:
        kwargs["keep_alive"] = True
        fmap = ftm.Map(**kwargs)
    except TypeError:
        kwargs.pop("keep_alive", None)
        fmap = ftm.Map(**kwargs)

    controls = []
    if title:
        controls.append(ft.Text(title, size=16, weight=ft.FontWeight.BOLD))
    if subtitle:
        controls.append(ft.Text(subtitle, size=11.5, color=ft.Colors.GREY_700))
    if show_legend:
        controls.append(build_map_legend(result, visible_item_ids=visible_item_ids, visible_stages=visible_stages))
    controls.append(ft.Container(height=height, border=ft.Border.all(1, ft.Colors.GREY_300), border_radius=8, clip_behavior=ft.ClipBehavior.HARD_EDGE, content=fmap))
    return ft.Container(padding=7, border=ft.Border.all(1, ft.Colors.GREY_200), border_radius=9, content=ft.Column(spacing=5, controls=controls))


class SupplyChainMapPanel:
    """Compatibility panel used by App._set_map_result().

    The six-scenario comparison UI has the shared Stage/material filters.  This
    compatibility control keeps the selected-result map usable without any
    hosted map tiles.
    """

    def __init__(self, page: ft.Page):
        self.page = page
        self.result = None
        self.map_host = ft.Container()
        self.root = ft.Column(spacing=8, controls=[self.map_host])

    def set_result(self, result: Mapping | None):
        self.result = result
        if not result:
            self.map_host.content = ft.Text("표시할 지도 결과가 없습니다.", color=ft.Colors.GREY_700)
        else:
            self.map_host.content = build_static_map_panel(self.page, result, height=500, show_legend=True)
        try:
            self.page.update()
        except Exception:
            pass


__all__ = ["SupplyChainMapPanel", "build_static_map_panel", "build_map_legend"]
