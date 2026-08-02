# 전기차 보조금 공급망 최적화 SaaS — OR-Tools 버전

## 최적화 엔진

- Google OR-Tools `pywraplp.MPSolver` (`ortools==9.15.6755`)
- 우선 solver: `SCIP`
- 대체 solver: `CBC`
- 기존 `GLOP`은 연속 LP 전용이므로 이 Word MILP 모형에는 사용하지 않습니다.

## 입력 방식

이 버전은 **방식 C** 전용입니다. 계산에 내장 기본 CSV를 사용하지 않으며, 앱을 열 때 아래 9개 CSV를 모두 업로드해야 합니다.

1. `products.csv`
2. `demand.csv`
3. `raw_material_suppliers.csv`
4. `assembly_locations.csv`
5. `transport_parameters.csv`
6. `markets.csv`
7. `scenarios.csv`
8. `poster_benchmark_cost_ratios.csv`
9. `poster_benchmark_quartiles.csv`

`data/` 폴더의 CSV는 앱의 템플릿 ZIP 다운로드 기능에만 사용됩니다.

## 파일 배치

```text
repository-root/
├── packages.txt
└── carbon_demo/
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── UPLOAD_CHECKLIST.txt
    ├── data/
    │   └── 9개 CSV
    └── assets/
        └── poster_reference.png
```

## 로컬 실행

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r carbon_demo/requirements.txt
streamlit run carbon_demo/app.py
```

Linux/macOS에서는 가상환경 활성화 명령을 `source .venv/bin/activate`로 사용합니다.
