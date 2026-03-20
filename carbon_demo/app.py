from pathlib import Path
import pandas as pd
import streamlit as st
from ortools.linear_solver import pywraplp

st.set_page_config(page_title="탄소배출량 계산 및 최적화 데모", layout="wide")

REQUIRED_COLS = [
    "site",
    "source",
    "baseline_use_mwh",
    "min_use_mwh",
    "max_use_mwh",
    "emission_factor_tco2_per_mwh",
    "cost_per_mwh",
]


def validate_input_df(df: pd.DataFrame):
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        return False, f"필수 컬럼이 없습니다: {', '.join(missing_cols)}"

    numeric_cols = [
        "baseline_use_mwh",
        "min_use_mwh",
        "max_use_mwh",
        "emission_factor_tco2_per_mwh",
        "cost_per_mwh",
    ]

    # 숫자형 변환
    df = df.copy()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_cols].isnull().any().any():
        return False, "숫자 컬럼에 비어 있거나 숫자가 아닌 값이 있습니다."

    if (df["min_use_mwh"] > df["max_use_mwh"]).any():
        return False, "min_use_mwh가 max_use_mwh보다 큰 행이 있습니다."

    if (df[numeric_cols] < 0).any().any():
        return False, "숫자 컬럼에 음수가 있습니다."

    return True, "입력 데이터 검증 완료"


def optimize_emissions(df: pd.DataFrame, budget_increase_pct: float = 5.0):
    """
    단순 선형 최적화 모형
    목적: 총 탄소배출량 최소화
    제약:
    1) 사업장별 총 사용량은 기준안과 동일
    2) 각 에너지원 사용량은 min/max 범위 내
    3) 총비용은 기준 총비용의 (1 + 허용증가율) 이하
    """
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        return {"status": "SOLVER_NOT_CREATED"}

    x = {}
    for idx, row in df.iterrows():
        x[idx] = solver.NumVar(
            float(row["min_use_mwh"]),
            float(row["max_use_mwh"]),
            f"x_{idx}_{row['site']}_{row['source']}"
        )

    # 사업장별 총 수요 유지
    site_demands = df.groupby("site")["baseline_use_mwh"].sum()
    for site, demand in site_demands.items():
        idxs = df.index[df["site"] == site]
        solver.Add(sum(x[i] for i in idxs) == float(demand))

    # 총비용 증가 제한
    baseline_total_cost = float((df["baseline_use_mwh"] * df["cost_per_mwh"]).sum())
    max_allowed_cost = baseline_total_cost * (1 + budget_increase_pct / 100.0)
    solver.Add(
        sum(float(df.loc[i, "cost_per_mwh"]) * x[i] for i in df.index)
        <= max_allowed_cost
    )

    # 목적함수: 총 탄소배출량 최소화
    objective = solver.Objective()
    for i in df.index:
        objective.SetCoefficient(x[i], float(df.loc[i, "emission_factor_tco2_per_mwh"]))
    objective.SetMinimization()

    status = solver.Solve()

    status_map = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    status_label = status_map.get(status, f"UNKNOWN_{status}")

    if status_label not in ["OPTIMAL", "FEASIBLE"]:
        return {"status": status_label}

    result_df = df.copy()
    result_df["optimized_use_mwh"] = [x[i].solution_value() for i in df.index]
    result_df["baseline_emissions_tco2"] = (
        result_df["baseline_use_mwh"] * result_df["emission_factor_tco2_per_mwh"]
    )
    result_df["optimized_emissions_tco2"] = (
        result_df["optimized_use_mwh"] * result_df["emission_factor_tco2_per_mwh"]
    )
    result_df["baseline_cost"] = result_df["baseline_use_mwh"] * result_df["cost_per_mwh"]
    result_df["optimized_cost"] = result_df["optimized_use_mwh"] * result_df["cost_per_mwh"]
    result_df["delta_use_mwh"] = result_df["optimized_use_mwh"] - result_df["baseline_use_mwh"]

    site_summary = result_df.groupby("site", as_index=False).agg(
        baseline_use_mwh=("baseline_use_mwh", "sum"),
        optimized_use_mwh=("optimized_use_mwh", "sum"),
        baseline_emissions_tco2=("baseline_emissions_tco2", "sum"),
        optimized_emissions_tco2=("optimized_emissions_tco2", "sum"),
        baseline_cost=("baseline_cost", "sum"),
        optimized_cost=("optimized_cost", "sum"),
    )
    site_summary["emissions_reduction_tco2"] = (
        site_summary["baseline_emissions_tco2"] - site_summary["optimized_emissions_tco2"]
    )
    site_summary["cost_increase"] = site_summary["optimized_cost"] - site_summary["baseline_cost"]

    total = {
        "baseline_use_mwh": float(site_summary["baseline_use_mwh"].sum()),
        "optimized_use_mwh": float(site_summary["optimized_use_mwh"].sum()),
        "baseline_emissions_tco2": float(site_summary["baseline_emissions_tco2"].sum()),
        "optimized_emissions_tco2": float(site_summary["optimized_emissions_tco2"].sum()),
        "baseline_cost": float(site_summary["baseline_cost"].sum()),
        "optimized_cost": float(site_summary["optimized_cost"].sum()),
    }
    total["reduction_tco2"] = (
        total["baseline_emissions_tco2"] - total["optimized_emissions_tco2"]
    )
    total["reduction_pct"] = (
        100 * total["reduction_tco2"] / total["baseline_emissions_tco2"]
        if total["baseline_emissions_tco2"] > 0
        else 0
    )

    return {
        "status": status_label,
        "detail": result_df,
        "site_summary": site_summary,
        "total": total,
    }


st.title("탄소배출량 계산 및 최적화 데모")
st.write("이 단계에서는 사용자가 업로드한 CSV 또는 샘플 CSV를 바탕으로 기준안 계산과 OR-Tools 최적화를 수행합니다.")

# ----------------------------
# 1. 데이터 입력
# ----------------------------
st.subheader("1. 데이터 입력")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

BASE_DIR = Path(__file__).resolve().parent
sample_csv_path = BASE_DIR / "data" / "sample_energy_options.csv"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.info("현재 업로드한 CSV 파일을 사용 중입니다.")
else:
    df = pd.read_csv(sample_csv_path)
    st.info("현재 샘플 CSV 데이터를 사용 중입니다. 사용자가 CSV를 업로드하면 그 파일로 계산합니다.")

is_valid, msg = validate_input_df(df)
if not is_valid:
    st.error(msg)
    st.stop()
else:
    st.success(msg)

# ----------------------------
# 2. 기준안 계산
# ----------------------------
df["baseline_emissions_tco2"] = (
    df["baseline_use_mwh"] * df["emission_factor_tco2_per_mwh"]
)
df["baseline_cost"] = df["baseline_use_mwh"] * df["cost_per_mwh"]

site_summary_baseline = df.groupby("site", as_index=False).agg(
    baseline_use_mwh=("baseline_use_mwh", "sum"),
    baseline_emissions_tco2=("baseline_emissions_tco2", "sum"),
    baseline_cost=("baseline_cost", "sum"),
)

total_use = float(site_summary_baseline["baseline_use_mwh"].sum())
total_emissions = float(site_summary_baseline["baseline_emissions_tco2"].sum())
total_cost = float(site_summary_baseline["baseline_cost"].sum())

st.subheader("2. 기준안 요약")
col1, col2, col3 = st.columns(3)
col1.metric("총 사용량 (MWh)", f"{total_use:.2f}")
col2.metric("총 탄소배출량 (tCO2)", f"{total_emissions:.2f}")
col3.metric("총 비용", f"{total_cost:.2f}")

st.subheader("원본 입력 데이터")
st.dataframe(df[[
    "site",
    "source",
    "baseline_use_mwh",
    "min_use_mwh",
    "max_use_mwh",
    "emission_factor_tco2_per_mwh",
    "cost_per_mwh",
]], use_container_width=True)

st.subheader("사업장별 기준안 합계")
st.dataframe(site_summary_baseline, use_container_width=True)

baseline_chart = site_summary_baseline.set_index("site")[["baseline_emissions_tco2"]]
st.subheader("사업장별 기준 탄소배출량")
st.bar_chart(baseline_chart)

# ----------------------------
# 3. 최적화 조건
# ----------------------------
st.subheader("3. 최적화 조건")
budget_increase_pct = st.slider(
    "허용 총비용 증가율 (%)",
    min_value=0,
    max_value=30,
    value=5,
    step=1
)

run_opt = st.button("최적화 실행")

# ----------------------------
# 4. 최적화 결과
# ----------------------------
if run_opt:
    result = optimize_emissions(df, budget_increase_pct=budget_increase_pct)

    st.subheader("4. 최적화 결과 상태")
    st.write(f"상태: **{result['status']}**")

    if result["status"] not in ["OPTIMAL", "FEASIBLE"]:
        st.error("최적해를 찾지 못했습니다. 입력 데이터와 제약조건을 확인하세요.")
    else:
        total = result["total"]
        detail = result["detail"]
        site_summary = result["site_summary"]

        st.subheader("최적안 요약")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("최적 총 배출량 (tCO2)", f"{total['optimized_emissions_tco2']:.2f}")
        c2.metric("배출량 감소 (tCO2)", f"{total['reduction_tco2']:.2f}")
        c3.metric("배출량 감소율 (%)", f"{total['reduction_pct']:.2f}")
        c4.metric("최적 총 비용", f"{total['optimized_cost']:.2f}")

        st.subheader("사업장별 기준안 vs 최적안 비교")
        st.dataframe(site_summary, use_container_width=True)

        compare_chart = site_summary.set_index("site")[
            ["baseline_emissions_tco2", "optimized_emissions_tco2"]
        ]
        st.subheader("사업장별 배출량 비교")
        st.bar_chart(compare_chart)

        st.subheader("에너지원별 상세 최적화 결과")
        st.dataframe(detail[[
            "site",
            "source",
            "baseline_use_mwh",
            "optimized_use_mwh",
            "delta_use_mwh",
            "emission_factor_tco2_per_mwh",
            "baseline_emissions_tco2",
            "optimized_emissions_tco2",
            "baseline_cost",
            "optimized_cost",
        ]], use_container_width=True)
