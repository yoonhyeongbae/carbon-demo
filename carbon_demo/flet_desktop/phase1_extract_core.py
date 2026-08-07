from pathlib import Path
import ast

SOURCE = Path('../app.py')
TARGET = Path('legacy_core.py')

REMOVE_FUNCTIONS = {
    '_active_item_table', '_add_direction_arrow', '_add_map_legend', '_add_route_line',
    '_add_stage12_route_layer', '_add_stage1_layer', '_add_stage23_route_layer',
    '_add_stage2_layer', '_add_stage3_layer', '_assembly_radius', '_bezier_curve_points',
    '_flow_width', '_market_record', '_offset_marker_coordinate', '_point_on_polyline',
    '_production_radius', '_render_country_grid', '_route_bend', '_save_tables_to_session',
    'build_stage_map', 'render_analysis_tab', 'render_comparison_bar_chart',
    'render_input_tab', 'render_item_selection', 'render_map_legend_above',
    'render_overview_tab', 'render_results_tab', 'render_solver_metrics',
    'render_stage_country_selection', 'render_stage_map', 'render_transport_mode_selection',
    'run_app', 'show_dataframe',
}

src = SOURCE.read_text(encoding='utf-8')
tree = ast.parse(src)
ranges = []
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in REMOVE_FUNCTIONS:
        ranges.append((node.lineno, node.end_lineno))
    elif isinstance(node, ast.Try):
        text = '\n'.join(src.splitlines()[node.lineno - 1:node.end_lineno])
        if 'import streamlit as st' in text:
            ranges.append((node.lineno, node.end_lineno))
    elif isinstance(node, ast.If):
        text = '\n'.join(src.splitlines()[node.lineno - 1:node.end_lineno])
        if '__name__' in text and 'run_app' in text:
            ranges.append((node.lineno, node.end_lineno))

lines = src.splitlines(keepends=True)
keep = [True] * len(lines)
for start, end in ranges:
    for i in range(start - 1, end):
        keep[i] = False
TARGET.write_text(''.join(line for i, line in enumerate(lines) if keep[i]), encoding='utf-8')
print(f'Wrote {TARGET}')
