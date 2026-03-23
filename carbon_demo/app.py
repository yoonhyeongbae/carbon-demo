from io import StringIO
import pandas as pd
import streamlit as st
from ortools.linear_solver import pywraplp

st.set_page_config(page_title="공급망 기반 탄소배출량 최적화 데모", layout="wide")

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
REQUIRED_COLS = [
    "scope",
    "site",
    "activity_group",
    "option_name",
    "baseline_amount",
    "min_amount",
    "max_amount",
    "unit",
    "emission_factor_tco2_per_unit",
    "cost_per_unit",
    "description",
]

SCOPE_LIST = ["Scope1", "Scope2", "Scope3"]

# --------------------------------------------------
# 공급망 구조 템플릿
# 사용자가 UI에서 이 템플릿을 선택하면 해당 activity_group만 최적화 대상에 포함됨
# --------------------------------------------------
STRUCTURE_TEMPLATES = {
    "구조 X - 전범위 통합형": {
        "description": "직접배출 + 구매전력 + 원자재 운송 + 원자재 조달을 모두 포함하는 구조",
        "groups": [
            "Boiler_Heat",
            "Electricity_Procurement",
            "Inbound_Transport",
            "Material_Sourcing",
        ],
        "nodes": [
            "x1: Boiler_Heat (보일러/연료 열원)",
            "x2: Electricity_Procurement (전력 조달)",
            "x3: Inbound_Transport (원자재 운송)",
            "x4: Material_Sourcing (원자재 조달)",
        ],
    },
    "구조 Y - 조달 중심형": {
        "description": "직접배출 + 구매전력 + 원자재 조달 중심 구조 (운송 제외)",
        "groups": [
            "Boiler_Heat",
            "Electricity_Procurement",
            "Material_Sourcing",
        ],
        "nodes": [
            "y1: Boiler_Heat (보일러/연료 열원)",
            "y2: Electricity_Procurement (전력 조달)",
            "y3: Material_Sourcing (원자재 조달)",
        ],
    },
    "구조 Z - 물류 중심형": {
        "description": "직접배출 + 구매전력 + 원자재 운송 중심 구조 (원자재 조달 제외)",
        "groups": [
            "Boiler_Heat",
            "Electricity_Procurement",
            "Inbound_Transport",
        ],
        "nodes": [
            "z1: Boiler_Heat (보일러/연료 열원)",
            "z2: Electricity_Procurement (전력 조달)",
            "z3: Inbound_Transport (원자재 운송)",
        ],
    },
}

# --------------------------------------------------
# activity_group 설명
# --------------------------------------------------
ACTIVITY_GROUP_GUIDE_DF = pd.DataFrame(
    [
        ["Boiler_Heat", "보일러/열원 활동", "Natural_Gas, LPG, Biomethane 등 같은 열원 대체 선택지 묶음"],
        ["Electricity_Procurement", "전력 조달 활동", "Grid, Green_PPA, Solar_Onsite 등 전력 조달 방식 묶음"],
        ["Inbound_Transport", "원자재 운송 활동", "Truck, Rail, Ship 등 원자재 운송 수단 묶음"],
        ["Material_Sourcing", "원자재 조달 활동", "Standard_Steel, Recycled_Steel, Green_Steel 등 조달 대안 묶음"],
    ],
    columns=["activity_group", "설명", "예시 option_name"]
)

# --------------------------------------------------
# 입력 컬럼 설명
# --------------------------------------------------
GUIDE_DF = pd.DataFrame(
    [
        ["scope", "문자", "Scope1 / Scope2 / Scope3", "배출 범주"],
        ["site", "문자", "Plant_A", "사업장명"],
        ["activity_group", "문자", "Boiler_Heat", "같은 기능을 하는 대체 가능한 선택지 묶음"],
        ["option_name", "문자", "Natural_Gas", "개별 선택지 이름"],
        ["baseline_amount", "숫자", "100", "현재 기준안 사용량 또는 물량"],
        ["min_amount", "숫자", "20", "최적화 후 최소 사용량"],
        ["max_amount", "숫자", "140", "최적화 후 최대 사용량"],
        ["unit", "문자", "MWh / ton / ton-km", "사용 단위"],
        ["emission_factor_tco2_per_unit", "숫자", "0.22", "단위당 배출계수(tCO2/unit)"],
        ["cost_per_unit", "숫자", "55", "단위당 비용"],
        ["description", "문자", "사업장 보일러 열원", "행 설명용 텍스트"],
    ],
    columns=["컬럼명", "자료형", "예시", "설명"]
)

# --------------------------------------------------
# 앱 내부 기본 샘플 데이터 (파일 업로드 없어도 바로 결과 가능)
# --------------------------------------------------
SAMPLE_DATA_TEXT = {
    "샘플 1 - 균형형": """scope,site,activity_group,option_name,baseline_amount,min_amount,max_amount,unit,emission_factor_tco2_per_unit,cost_per_unit,description
Scope1,Plant_A,Boiler_Heat,Natural_Gas,100,20,140,MWh,0.22,55,사업장 보일러 열원
Scope1,Plant_A,Boiler_Heat,LPG,20,0,60,MWh,0.27,68,사업장 보일러 열원
Scope1,Plant_A,Boiler_Heat,Biomethane,0,0,80,MWh,0.06,92,사업장 보일러 열원
Scope2,Plant_A,Electricity_Procurement,Grid,180,50,220,MWh,0.45,100,구매 전력
Scope2,Plant_A,Electricity_Procurement,Green_PPA,20,0,140,MWh,0.05,130,구매 전력
Scope2,Plant_A,Electricity_Procurement,Solar_Onsite,0,0,60,MWh,0.02,145,구매 전력
Scope3,Plant_A,Inbound_Transport,Truck,260,50,320,ton-km,0.00018,1.20,원자재 운송
Scope3,Plant_A,Inbound_Transport,Rail,40,0,220,ton-km,0.00006,1.60,원자재 운송
Scope3,Plant_A,Inbound_Transport,Ship,0,0,100,ton-km,0.00004,1.55,원자재 운송
Scope3,Plant_A,Material_Sourcing,Standard_Steel,90,30,120,ton,1.90,650,원자재 구매
Scope3,Plant_A,Material_Sourcing,Recycled_Steel,10,0,80,ton,1.10,720,원자재 구매
Scope3,Plant_A,Material_Sourcing,Green_Steel,0,0,40,ton,0.90,790,원자재 구매
""",
    "샘플 2 - 친환경 유연형": """scope,site,activity_group,option_name,baseline_amount,min_amount,max_amount,unit,emission_factor_tco2_per_unit,cost_per_unit,description
Scope1,Plant_A,Boiler_Heat,Natural_Gas,70,10,100,MWh,0.22,55,사업장 보일러 열원
Scope1,Plant_A,Boiler_Heat,LPG,10,0,30,MWh,0.27,68,사업장 보일러 열원
Scope1,Plant_A,Boiler_Heat,Biomethane,40,0,140,MWh,0.06,88,사업장 보일러 열원
Scope2,Plant_A,Electricity_Procurement,Grid,120,20,160,MWh,0.45,100,구매 전력
Scope2,Plant_A,Electricity_Procurement,Green_PPA,50,0,200,MWh,0.05,125,구매 전력
Scope2,Plant_A,Electricity_Procurement,Solar_Onsite,30,0,120,MWh,0.02,140,구매 전력
Scope3,Plant_A,Inbound_Transport,Truck,180,30,220,ton-km,0.00018,1.20,원자재 운송
Scope3,Plant_A,Inbound_Transport,Rail,80,0,260,ton-km,0.00006,1.55,원자재 운송
Scope3,Plant_A,Inbound_Transport,Ship,40,0,180,ton-km,0.00004,1.50,원자재 운송
Scope3,Plant_A,Material_Sourcing,Standard_Steel,60,10,80,ton,1.90,650,원자재 구매
Scope3,Plant_A,Material_Sourcing,Recycled_Steel,25,0,120,ton,1.10,710,원자재 구매
Scope3,Plant_A,Material_Sourcing,Green_Steel,15,0,90,ton,0.90,770,원자재 구매
""",
    "샘플 3 - 제약이 빡빡한 형": """scope,site,activity_group,option_name,baseline_amount,min_amount,max_amount,unit,emission_factor_tco2_per_unit,cost_per_unit,description
Scope1,Plant_A,Boiler_Heat,Natural_Gas,110,80,120,MWh,0.22,55,사업장 보일러 열원
Scope1,Plant_A,Boiler_Heat,LPG,10,0,20,MWh,0.27,68,사업장 보일러 열원
Scope1,Plant_A,Boiler_Heat,Biomethane,0,0,10,MWh,0.06,95,사업장 보일러 열원
Scope2,Plant_A,Electricity_Procurement,Grid,190,160,210,MWh,0.45,100,구매 전력
Scope2,Plant_A,Electricity_Procurement,Green_PPA,10,0,25,MWh,0.05,132,구매 전력
Scope2,Plant_A,Electricity_Procurement,Solar_Onsite,0,0,15,MWh,0.02,150,구매 전력
Scope3,Plant_A,Inbound_Transport,Truck,280,240,300,ton-km,0.00018,1.20,원자재 운송
Scope3,Plant_A,Inbound_Transport,Rail,20,0,40,ton-km,0.00006,1.65,원자재 운송
Scope3,Plant_A,Inbound_Transport,Ship,0,0,10,ton-km,0.00004,1.60,원자재 운송
Scope3,Plant_A,Material_Sourcing,Standard_Steel,95,85,100,ton,1.90,650,원자재 구매
Scope3,Plant_A,Material_Sourcing,Recycled_Steel,5,0,15,ton,1.10,730,원자재 구매
Scope3,Plant_A,Material_Sourcing,Green_Steel,0,0,5,ton,0.90,810,원자재 구매
"""
}

# --------------------------------------------------
# 함수들
# --------------------------------------------------
def validate_input_df(df: pd.DataFrame):
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        return False, f"필수 컬럼이 없습니다: {', '.join(missing_cols)}", None

    df = df.copy()

    numeric_cols = [
        "baseline_amount",
        "min_amount",
        "max_amount",
        "emission_factor_tco2_per_unit",
        "cost_per_unit",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_cols].isnull().any().any():
        return False, "숫자 컬럼에 비어 있거나 숫자가 아닌 값이 있습니다.", None

    if (df["min_amount"] > df["max_amount"]).any():
        return False, "min_amount가 max_amount보다 큰 행이 있습니다.", None

    if (df[numeric_cols] < 0).any().any():
        return False, "숫자 컬럼에 음수가 있습니다.", None

    group_cols = ["site", "scope", "activity_group", "unit"]
    grp = df.groupby(group_cols, as_index=False).agg(
        demand=("baseline_amount", "sum"),
        min_total=("min_amount", "sum"),
        max_total=("max_amount", "sum"),
    )

    bad_rows = grp[(grp["demand"] < grp["min_total"]) | (grp["demand"] > grp["max_total"])]
    if not bad_rows.empty:
        return False, "일부 activity_group에서 기준 수요가 min/max 범위를 만족하지 않습니다.", None

    return True, "입력 데이터 검증 완료", df


def optimize_total_emissions(df: pd.DataFrame, budget_increase_pct: float = 5.0):
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        return {"status": "SOLVER_NOT_CREATED"}

    x = {}
    for idx, row in df.iterrows():
        x[idx] = solver.NumVar(
            float(row["min_amount"]),
            float(row["max_amount"]),
            f"x_{idx}_{row['scope']}_{row['activity_group']}_{row['option_name']}"
        )

    group_cols = ["site", "scope", "activity_group", "unit"]
    grouped = df.groupby(group_cols)["baseline_amount"].sum().reset_index()

    for _, g in grouped.iterrows():
        mask = (
            (df["site"] == g["site"])
            & (df["scope"] == g["scope"])
            & (df["activity_group"] == g["activity_group"])
            & (df["unit"] == g["unit"])
        )
        idxs = df.index[mask]
        solver.Add(sum(x[i] for i in idxs) == float(g["baseline_amount"]))

    baseline_total_cost = float((df["baseline_amount"] * df["cost_per_unit"]).sum())
    max_allowed_cost = baseline_total_cost * (1 + budget_increase_pct / 100.0)

    solver.Add(
        sum(float(df.loc[i, "cost_per_unit"]) * x[i] for i in df.index)
        <= max_allowed_cost
    )

    objective = solver.Objective()
    for i in df.index:
        objective.SetCoefficient(x[i], float(df.loc[i, "emission_factor_tco2_per_unit"]))
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
    result_df["optimized_amount"] = [x[i].solution_value() for i in df.index]
    result_df["delta_amount"] = result_df["optimized_amount"] - result_df["baseline_amount"]

    result_df["baseline_emissions_tco2"] = (
        result_df["baseline_amount"] * result_df["emission_factor_tco2_per_unit"]
    )
    result_df["optimized_emissions_tco2"] = (
        result_df["optimized_amount"] * result_df["emission_factor_tco2_per_unit"]
    )

    result_df["baseline_cost"] = result_df["baseline_amount"] * result_df["cost_per_unit"]
    result_df["optimized_cost"] = result_df["optimized_amount"] * result_df["cost_per_unit"]

    scope_summary = result_df.groupby("scope", as_index=False).agg(
        baseline_amount=("baseline_amount", "sum"),
        optimized_amount=("optimized_amount", "sum"),
        baseline_emissions_tco2=("baseline_emissions_tco2", "sum"),
        optimized_emissions_tco2=("optimized_emissions_tco2", "sum"),
        baseline_cost=("baseline_cost", "sum"),
        optimized_cost=("optimized_cost", "sum"),
    )
    scope_summary["reduction_tco2"] = (
        scope_summary["baseline_emissions_tco2"] - scope_summary["optimized_emissions_tco2"]
    )
    scope_summary["reduction_pct"] = (
        100 * scope_summary["reduction_tco2"] / scope_summary["baseline_emissions_tco2"]
    )

    activity_summary = result_df.groupby(
        ["scope", "activity_group", "unit"], as_index=False
    ).agg(
        baseline_amount=("baseline_amount", "sum"),
        optimized_amount=("optimized_amount", "sum"),
        baseline_emissions_tco2=("baseline_emissions_tco2", "sum"),
        optimized_emissions_tco2=("optimized_emissions_tco2", "sum"),
        baseline_cost=("baseline_cost", "sum"),
        optimized_cost=("optimized_cost", "sum"),
    )
    activity_summary["reduction_tco2"] = (
        activity_summary["baseline_emissions_tco2"] - activity_summary["optimized_emissions_tco2"]
    )

    total = {
        "baseline_emissions_tco2": float(result_df["baseline_emissions_tco2"].sum()),
        "optimized_emissions_tco2": float(result_df["optimized_emissions_tco2"].sum()),
        "baseline_cost": float(result_df["baseline_cost"].sum()),
        "optimized_cost": float(result_df["optimized_cost"].sum()),
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
        "scope_summary": scope_summary,
        "activity_summary": activity_summary,
        "total": total,
    }


# --------------------------------------------------
# 제목
# --------------------------------------------------
st.title("공급망 기반 탄소배출량 최적화 데모")
st.write("사용자가 공급망 구조 템플릿을 선택하고, 기본 내장 샘플 데이터 또는 업로드 CSV를 기반으로 총 탄소배출량을 최소화하는 데모입니다.")

tab1, tab2, tab3 = st.tabs(["공급망 구조 및 입력", "Scope별 최적화", "결과 및 보고서"])

# --------------------------------------------------
# Tab 1
# --------------------------------------------------
with tab1:
    st.header("1. 공급망 구조 및 입력")

    selected_scopes = st.multiselect(
        "분석할 Scope 선택",
        SCOPE_LIST,
        default=SCOPE_LIST
    )

    structure_template_name = st.selectbox(
        "공급망 구조 템플릿 선택",
        list(STRUCTURE_TEMPLATES.keys())
    )
    selected_template = STRUCTURE_TEMPLATES[structure_template_name]

    st.subheader("선택된 공급망 구조 설명")
    st.info(selected_template["description"])

    st.write("현재 선택된 공급망 구조 노드")
    for node in selected_template["nodes"]:
        st.markdown(f"- {node}")

    st.subheader("입력 방식")
    sample_choice = st.selectbox("샘플 데이터 선택", list(SAMPLE_DATA_TEXT.keys()))
    uploaded_file = st.file_uploader("사용자 CSV 업로드", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.info("현재 업로드한 CSV 파일을 사용 중입니다.")
    else:
        raw_df = pd.read_csv(StringIO(SAMPLE_DATA_TEXT[sample_choice]))
        st.info(f"현재 {sample_choice} 기본 내장 데이터를 사용 중입니다.")

    is_valid, msg, cleaned_df = validate_input_df(raw_df)
    if not is_valid:
        st.error(msg)
        st.stop()
    else:
        st.success(msg)

    filtered_df = cleaned_df[
        cleaned_df["scope"].isin(selected_scopes)
        & cleaned_df["activity_group"].isin(selected_template["groups"])
    ].copy()

    if filtered_df.empty:
        st.warning("현재 선택한 Scope/구조 템플릿에 해당하는 데이터가 없습니다.")
        st.stop()

    current_signature = (
        str(pd.util.hash_pandas_object(filtered_df, index=True).sum())
        + "|"
        + structure_template_name
        + "|"
        + ",".join(selected_scopes)
    )

    if st.session_state.get("data_signature") != current_signature:
        st.session_state["data_signature"] = current_signature
        st.session_state.pop("opt_result", None)

    st.subheader("입력 컬럼 설명")
    st.dataframe(GUIDE_DF, use_container_width=True)

    st.subheader("activity_group 설명")
    st.dataframe(ACTIVITY_GROUP_GUIDE_DF, use_container_width=True)

    st.subheader("단위 설명")
    st.markdown(
        """
        - **MWh**: 전기/연료/열 에너지 사용량  
        - **ton**: 자재 구매량  
        - **ton-km**: 운송 물량 × 운송 거리  
        - **emission_factor_tco2_per_unit**: 각 단위 1개당 탄소배출량  
        - **cost_per_unit**: 각 단위 1개당 비용  
        """
    )

    st.subheader("현재 구조 템플릿이 반영된 공급망 데이터")
    st.dataframe(filtered_df, use_container_width=True)

# --------------------------------------------------
# Tab 2
# --------------------------------------------------
with tab2:
    st.header("2. Scope별 최적화")
    sub1, sub2, sub3, sub4 = st.tabs(["Scope 1", "Scope 2", "Scope 3", "통합 최적화 실행"])

    with sub1:
        st.subheader("Scope 1 데이터")
        s1 = filtered_df[filtered_df["scope"] == "Scope1"]
        if s1.empty:
            st.info("선택된 Scope 1 데이터가 없습니다.")
        else:
            st.dataframe(s1, use_container_width=True)

    with sub2:
        st.subheader("Scope 2 데이터")
        s2 = filtered_df[filtered_df["scope"] == "Scope2"]
        if s2.empty:
            st.info("선택된 Scope 2 데이터가 없습니다.")
        else:
            st.dataframe(s2, use_container_width=True)

    with sub3:
        st.subheader("Scope 3 데이터")
        s3 = filtered_df[filtered_df["scope"] == "Scope3"]
        if s3.empty:
            st.info("선택된 Scope 3 데이터가 없습니다.")
        else:
            st.dataframe(s3, use_container_width=True)

    with sub4:
        st.subheader("통합 최적화 실행")
        st.write("선택된 Scope 및 공급망 구조 템플릿 전체를 대상으로 총 탄소배출량을 최소화합니다.")

        baseline_total_emissions = float(
            (filtered_df["baseline_amount"] * filtered_df["emission_factor_tco2_per_unit"]).sum()
        )
        baseline_total_cost = float(
            (filtered_df["baseline_amount"] * filtered_df["cost_per_unit"]).sum()
        )

        c1, c2 = st.columns(2)
        c1.metric("기준 총 탄소배출량 (tCO2)", f"{baseline_total_emissions:.2f}")
        c2.metric("기준 총 비용", f"{baseline_total_cost:.2f}")

        budget_increase_pct = st.slider(
            "허용 총비용 증가율 (%)",
            min_value=0,
            max_value=30,
            value=5,
            step=1
        )

        run_opt = st.button("통합 최적화 실행", key="run_integrated_optimization")

        if run_opt:
            result = optimize_total_emissions(filtered_df, budget_increase_pct)
            st.session_state["opt_result"] = result
            st.session_state["selected_scopes"] = selected_scopes
            st.session_state["selected_template_name"] = structure_template_name

            st.subheader("최적화 결과 상태")
            st.write(f"상태: **{result['status']}**")

            if result["status"] not in ["OPTIMAL", "FEASIBLE"]:
                st.error("최적해를 찾지 못했습니다. 입력 데이터와 제약조건을 확인하세요.")
            else:
                st.success("최적화를 완료했습니다. 결과 및 보고서 탭에서 상세 결과를 확인하세요.")

# --------------------------------------------------
# Tab 3
# --------------------------------------------------
with tab3:
    st.header("3. 결과 및 보고서")

    if "opt_result" not in st.session_state:
        st.info("먼저 'Scope별 최적화' 탭에서 통합 최적화를 실행하세요.")
    else:
        result = st.session_state["opt_result"]

        if result["status"] not in ["OPTIMAL", "FEASIBLE"]:
            st.error("최적해가 없어 결과를 표시할 수 없습니다.")
        else:
            total = result["total"]
            scope_summary = result["scope_summary"]
            activity_summary = result["activity_summary"]
            detail = result["detail"]

            st.subheader("적용된 구조")
            st.write(f"선택된 공급망 구조 템플릿: **{st.session_state.get('selected_template_name', '-')}**")

            st.subheader("총괄 결과")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("기준 총 배출량", f"{total['baseline_emissions_tco2']:.2f}")
            c2.metric("최적 총 배출량", f"{total['optimized_emissions_tco2']:.2f}")
            c3.metric("총 감축량", f"{total['reduction_tco2']:.2f}")
            c4.metric("총 감축률 (%)", f"{total['reduction_pct']:.2f}")

            c5, c6 = st.columns(2)
            c5.metric("기준 총 비용", f"{total['baseline_cost']:.2f}")
            c6.metric("최적 총 비용", f"{total['optimized_cost']:.2f}")

            st.subheader("Scope별 분석")
            st.dataframe(scope_summary, use_container_width=True)

            scope_chart = scope_summary.set_index("scope")[["baseline_emissions_tco2", "optimized_emissions_tco2"]]
            st.bar_chart(scope_chart)

            st.subheader("활동그룹(activity_group)별 분석")
            st.dataframe(activity_summary, use_container_width=True)

            st.subheader("상세 결과")
            st.dataframe(detail, use_container_width=True)

            st.subheader("보고서 다운로드")
            scope_csv = scope_summary.to_csv(index=False).encode("utf-8-sig")
            activity_csv = activity_summary.to_csv(index=False).encode("utf-8-sig")
            detail_csv = detail.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="Scope 요약 CSV 다운로드",
                data=scope_csv,
                file_name="scope_summary_report.csv",
                mime="text/csv"
            )
            st.download_button(
                label="활동그룹 요약 CSV 다운로드",
                data=activity_csv,
                file_name="activity_summary_report.csv",
                mime="text/csv"
            )
            st.download_button(
                label="상세 결과 CSV 다운로드",
                data=detail_csv,
                file_name="detailed_optimization_report.csv",
                mime="text/csv"
            )
