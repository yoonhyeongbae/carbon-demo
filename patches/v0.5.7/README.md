# EV Carbon Optimizer v0.5.7 map/result UI patch

Target: `EVCO-0.5.6-USER-GUIDE-CSV-OVERRIDE-SCOPE` (`EV_Carbon_Optimizer_v0.5.6_DEV_READY`).

## Requested changes

1. Remove the duplicated Scope 1+2 / Scope 3 chart from Tab 6 Analysis. The same chart remains in Tab 5 Results.
2. Move the shared map legend directly below the six scenario/production-mode checkboxes.
3. Restore shared map filters:
   - Stage 1 production nodes
   - Stage 1→2 raw-material/material transport
   - Stage 2 vehicle assembly nodes
   - Stage 2→3 finished-vehicle transport
   - each active material/raw-material by checkbox and color
4. Reduce map interaction load:
   - quadratic route geometry from 44 steps to 18 steps (45→19 points per route)
   - direction arrows from 3 per route to 1
   - material/stage filtering is applied before maps are constructed, so hidden layers do not create map controls
   - common legend is constructed once, not repeated inside map 1
5. The OR-Tools mathematical model is not changed.

## How to apply

1. Keep your complete `EV_Carbon_Optimizer_v0.5.6_DEV_READY` folder.
2. Download `APPLY_V057_MAP_UI_PATCH.py` and `APPLY_V057_MAP_UI_PATCH.bat` from this folder.
3. Run:

```bat
APPLY_V057_MAP_UI_PATCH.bat C:\EV_Carbon_Optimizer_v0.5.6_DEV_READY
```

4. The patch creates backups:
   - `main.py.v056_backup`
   - `map_logic.py.v056_backup`
5. Run the usual `START_DEV.bat` in the EVCO project folder.

## Result-map control order

1. Scenario / production-mode checkboxes
2. Shared map legend
3. Stage checkboxes
4. Material/raw-material checkboxes
5. Interactive maps

All filters apply to every checked scenario×production-mode map.

## Model integrity

The patch does not modify `dynamic_structure_core.py`, `legacy_core.py`, `ev_optimizer_api.py`, the 18 CSV input files, decision-variable definitions, objective coefficients, constraints, or scenario carbon caps.
