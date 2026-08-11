from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile
import re

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd().resolve()

# Allow double-click / call from the patch directory while the EVCO project is
# its parent or while the script itself has been copied into the project root.
if not (ROOT / "main.py").exists():
    candidates = [HERE, HERE.parent, HERE.parent.parent, HERE.parent.parent.parent]
    ROOT = next((p.resolve() for p in candidates if (p / "main.py").exists()), ROOT)

MAIN = ROOT / "main.py"
MAP_LOGIC = ROOT / "map_logic.py"
REQ = ROOT / "requirements-dev.txt"
PYPROJECT = ROOT / "pyproject.toml"

if not MAIN.exists():
    raise SystemExit(
        "main.py를 찾을 수 없습니다. 이 업데이트 폴더를 "
        "EV_Carbon_Optimizer_v0.5.6_DEV_READY 폴더 안에 복사한 후 다시 실행하세요."
    )

print("EV Carbon Optimizer v0.5.8 updater")
print("Project root:", ROOT)

# ---------------------------------------------------------------------------
# 1) Apply the verified v0.5.7 shared Stage/material map-filter patch first.
# ---------------------------------------------------------------------------
main_text = MAIN.read_text(encoding="utf-8")
if "MAP_UI_PATCH = \"v0.5.7-stage-material-filter-performance\"" not in main_text:
    v057 = HERE / "APPLY_V057_MAP_UI_PATCH.py"
    if not v057.exists():
        raise SystemExit("APPLY_V057_MAP_UI_PATCH.py가 업데이트 패키지에 없습니다.")
    print("Applying v0.5.7 Stage/material filter patch...")
    subprocess.run([sys.executable, str(v057)], cwd=str(ROOT), check=True)
else:
    print("v0.5.7 Stage/material filter patch already present.")

# ---------------------------------------------------------------------------
# 2) Backups.
# ---------------------------------------------------------------------------
backup_dir = ROOT / "_backup_before_v058"
backup_dir.mkdir(exist_ok=True)
for p in (MAIN, MAP_LOGIC, REQ, PYPROJECT):
    if p.exists() and not (backup_dir / p.name).exists():
        shutil.copy2(p, backup_dir / p.name)

# ---------------------------------------------------------------------------
# 3) Install the offline Natural Earth map module.
# ---------------------------------------------------------------------------
src_map = HERE / "natural_earth_map.py"
if not src_map.exists():
    raise SystemExit("natural_earth_map.py가 업데이트 패키지에 없습니다.")
shutil.copy2(src_map, ROOT / "natural_earth_map.py")

# ---------------------------------------------------------------------------
# 4) Download and bundle Natural Earth 1:110m Admin-0 Countries locally.
#    Natural Earth data is public domain under its official Terms of Use.
# ---------------------------------------------------------------------------
asset_dir = ROOT / "assets" / "natural_earth"
asset_dir.mkdir(parents=True, exist_ok=True)
shp = asset_dir / "ne_110m_admin_0_countries.shp"
if not shp.exists():
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    zpath = asset_dir / "ne_110m_admin_0_countries.zip"
    print("Downloading Natural Earth 1:110m Admin-0 Countries...")
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(asset_dir)
    try:
        zpath.unlink()
    except OSError:
        pass
if not shp.exists():
    raise SystemExit("Natural Earth shapefile 준비에 실패했습니다.")

terms = asset_dir / "NATURAL_EARTH_TERMS.txt"
terms.write_text(
    "Natural Earth Terms of Use\n"
    "Official source: https://www.naturalearthdata.com/about/terms-of-use/\n"
    "Data used: Natural Earth Admin 0 – Countries, 1:110m, version 5.1.1\n"
    "Official data page: https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/\n\n"
    "Natural Earth states that all raster and vector map data found on its "
    "website are public domain and may be modified, electronically disseminated, "
    "printed, and used for personal, educational, and commercial purposes. "
    "Permission and attribution are not required. EVCO nevertheless displays "
    "'Made with Natural Earth' for transparency.\n",
    encoding="utf-8",
)

# ---------------------------------------------------------------------------
# 5) Point every user-visible supply-chain map import at the local-vector map.
# ---------------------------------------------------------------------------
main_text = MAIN.read_text(encoding="utf-8")
main_text = main_text.replace("from map_view import ", "from natural_earth_map import ")

if "MAP_BASEMAP = \"Natural Earth 1:110m local vector (public domain)\"" not in main_text:
    anchor = 'MAP_UI_PATCH = "v0.5.7-stage-material-filter-performance"'
    if anchor in main_text:
        main_text = main_text.replace(
            anchor,
            anchor + '\nMAP_BASEMAP = "Natural Earth 1:110m local vector (public domain)"',
            1,
        )

old_build = "EVCO-0.5.6-USER-GUIDE-CSV-OVERRIDE-SCOPE"
new_build = "EVCO-0.5.8-NATURAL-EARTH-SHARED-MAP-FILTERS"
main_text = main_text.replace(old_build, new_build)
MAIN.write_text(main_text, encoding="utf-8")

# Keep verifier / launcher build strings synchronized without changing their logic.
for name in ("BUILD_ID", "BUILD_ID.txt", "VERIFY_BUILD.py", "dev_supervisor.py", "START_DEV.bat"):
    p = ROOT / name
    if not p.exists() or p.is_dir():
        continue
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = p.read_text(encoding="cp949")
    if old_build in text:
        text = text.replace(old_build, new_build)
        p.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# 6) Add PyShp.  No hosted-tile dependency is added.
# ---------------------------------------------------------------------------
if REQ.exists():
    req = REQ.read_text(encoding="utf-8")
    if not re.search(r"(?im)^pyshp(?:[<>=!~].*)?$", req):
        req = req.rstrip() + "\npyshp>=2.3,<3\n"
        REQ.write_text(req, encoding="utf-8")

if PYPROJECT.exists():
    text = PYPROJECT.read_text(encoding="utf-8")
    if "pyshp" not in text.lower():
        # pyproject formats differed across EVCO packages. requirements-dev.txt
        # is authoritative for START_DEV, so only add when a simple dependency
        # array can be recognized safely.
        text2, n = re.subn(
            r'(dependencies\s*=\s*\[)',
            r'\1\n    "pyshp>=2.3,<3",',
            text,
            count=1,
        )
        if n:
            PYPROJECT.write_text(text2, encoding="utf-8")

# ---------------------------------------------------------------------------
# 7) Release notes / audit markers.
# ---------------------------------------------------------------------------
(ROOT / "MAP_LICENSE_AND_DATA_v0.5.8.md").write_text(
    "# EVCO v0.5.8 map data\n\n"
    "- Hosted CARTO / OpenStreetMap tile layers: **not used by the v0.5.8 user-visible supply-chain map**.\n"
    "- Basemap: Natural Earth Admin 0 – Countries, 1:110m, version 5.1.1, bundled locally.\n"
    "- Natural Earth legal status: public domain under the official Terms of Use.\n"
    "- Terms: https://www.naturalearthdata.com/about/terms-of-use/\n"
    "- Data: https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/\n"
    "- The app displays 'Made with Natural Earth' although credit is not required.\n"
    "- Natural Earth default Admin-0 data depicts de facto boundaries; boundary presentation is a separate geopolitical/accuracy issue from copyright licensing.\n\n"
    "## Performance\n"
    "The v0.5.7 prerequisite patch reduces Bézier route geometry from 44 to 18 steps and direction arrows from three to one per route. The v0.5.8 renderer uses local polygons, PolygonLayer viewport culling/simplification and Map keep_alive, and does not create a hosted TileLayer.\n",
    encoding="utf-8",
)

(ROOT / "CHANGES_v0.5.8.md").write_text(
    "# EV Carbon Optimizer v0.5.8\n\n"
    "- Restored shared Stage 1 / Stage 1→2 / Stage 2 / Stage 2→3 checkboxes for the six result maps.\n"
    "- Restored shared raw-material/material map checkboxes.\n"
    "- Common legend sits immediately below scenario/production-mode checkboxes.\n"
    "- Removed duplicate Scope proxy chart from Analysis; Results copy remains.\n"
    "- Replaced CARTO/OSM hosted basemap with bundled Natural Earth 1:110m Admin-0 public-domain vector polygons.\n"
    "- Reduced route geometry/arrows and enabled lightweight local rendering to reduce map pan/zoom lag.\n"
    "- OR-Tools mathematical model and optimization data semantics are unchanged by this map update.\n",
    encoding="utf-8",
)

# Syntax check files changed by this updater.
for p in (ROOT / "main.py", ROOT / "map_logic.py", ROOT / "natural_earth_map.py"):
    compile(p.read_text(encoding="utf-8"), str(p), "exec")

print("\nPASS: EVCO v0.5.8 update applied")
print("- shared Stage/material result-map filters enabled")
print("- CARTO/OSM hosted map tiles replaced by local Natural Earth polygons")
print("- Natural Earth asset:", shp)
print("- OR-Tools core not modified by v0.5.8 updater")
print("\nRun START_DEV.bat. On the first run, requirements-dev.txt will install pyshp.")
