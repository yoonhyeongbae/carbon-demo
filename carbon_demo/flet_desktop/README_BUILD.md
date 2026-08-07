# EV Carbon Optimizer - Flet desktop build

This directory is a desktop replacement for the Streamlit UI. The optimization engine is the v21.1 `app.py` copied as `legacy_core.py` at build time.

## Developer run

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install "flet[all]==0.86.5" numpy==2.3.2 pandas==2.3.2 geopy==2.4.1 ortools==9.15.6755
flet run .
```

## Windows build

```powershell
flet build windows . --python-version 3.13 --artifact ev_carbon_optimizer
```

The end-user build contains the Python runtime and dependencies and does not require Python, pip, Streamlit, a browser, or Internet access at runtime. The offline SVG map intentionally replaces CartoDB/Folium online tiles.
