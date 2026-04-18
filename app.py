from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = Path("panel.parquet")
GEOJSON_PATH = Path("moscow_districts.geojson")

METRIC_OPTIONS = {
    "cagr": "CAGR за период",
    "pct_change": "Рост, %",
    "abs_change": "Абсолютное изменение цены за м²",
    "base_price": "Цена за м² в базовом квартале",
    "current_price": "Цена за м² в конечном квартале",
}


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def parse_quarter(value: object) -> pd.Period | pd.NaT:
    if value is None or pd.isna(value):
        return pd.NaT
    try:
        return pd.Period(str(value), freq="Q")
    except Exception:
        return pd.NaT


def quarter_distance(base_quarter: str, end_quarter: str) -> int:
    base = parse_quarter(base_quarter)
    end = parse_quarter(end_quarter)
    if pd.isna(base) or pd.isna(end):
        return -1
    return int(end.ordinal - base.ordinal)


@st.cache_data(show_spinner=False)
def load_panel(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def validate_panel(df: pd.DataFrame) -> pd.DataFrame:
    required = {"district_name", "price_per_m2", "n_deals"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"В panel.parquet отсутствуют обязательные колонки: {missing}")

    out = df.copy()
    if "quarter" in out.columns:
        out["quarter_parsed"] = out["quarter"].apply(parse_quarter)
    elif {"year", "q_num"}.issubset(out.columns):
        year = pd.to_numeric(out["year"], errors="coerce")
        q_num = pd.to_numeric(out["q_num"], errors="coerce")
        out["quarter_parsed"] = pd.PeriodIndex(
            year=year.astype("Int64"),
            quarter=q_num.astype("Int64"),
            freq="Q",
        )
    else:
        raise ValueError("Нужны либо колонка quarter, либо пара year + q_num")

    if out["quarter_parsed"].isna().all():
        raise ValueError("Не удалось распарсить кварталы из panel.parquet")

    out = out.dropna(subset=["district_name", "quarter_parsed"]).copy()
    out["quarter_label"] = out["quarter_parsed"].astype(str)
    out["quarter_idx"] = out["quarter_parsed"].map(lambda p: p.ordinal).astype("Int64")
    out["quarter_dt"] = out["quarter_parsed"].dt.start_time
    out["price_per_m2"] = pd.to_numeric(out["price_per_m2"], errors="coerce")
    out["n_deals"] = pd.to_numeric(out["n_deals"], errors="coerce")
    out["district_name"] = out["district_name"].astype("string").str.strip()

    out = out.sort_values(["quarter_idx", "district_name"]).copy()
    return out


def _candidate_name_columns(df: pd.DataFrame) -> Iterable[str]:
    preferred = [
        "district_name",
        "name",
        "NAME",
        "district",
        "district_ru",
        "adm_name",
        "mun_name",
    ]
    available = [c for c in preferred if c in df.columns]
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and col not in available:
            available.append(col)
    return available


def harmonize_district_names(gdf: gpd.GeoDataFrame, panel_districts: pd.Series) -> tuple[gpd.GeoDataFrame, str]:
    panel_norm = set(_normalize_text(panel_districts.dropna()))
    best_col = None
    best_overlap = -1

    for col in _candidate_name_columns(gdf):
        overlap = len(set(_normalize_text(gdf[col].dropna())) & panel_norm)
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = col

    if best_col is None:
        raise ValueError("В GeoJSON не найдено строковых колонок для join")

    gdf_out = gdf.copy()
    gdf_out["join_key"] = _normalize_text(gdf_out[best_col])
    return gdf_out, best_col


@st.cache_data(show_spinner=False)
def load_geojson(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("GeoJSON не содержит геометрий")
    return gdf


def compute_coverage(df: pd.DataFrame, base_idx: int, end_idx: int) -> pd.DataFrame:
    in_range = df[(df["quarter_idx"] >= base_idx) & (df["quarter_idx"] <= end_idx)].copy()
    observed = in_range.groupby("district_name", dropna=False)["quarter_idx"].nunique().rename("observed_quarters")
    coverage = observed.reset_index()
    expected = end_idx - base_idx + 1
    coverage["expected_quarters"] = expected
    coverage["coverage_ratio"] = np.where(
        expected > 0,
        coverage["observed_quarters"] / expected,
        np.nan,
    )
    coverage["full_coverage_flag"] = (coverage["observed_quarters"] == expected).astype(bool)
    return coverage


def build_interval_snapshot(
    df: pd.DataFrame,
    base_quarter: str,
    end_quarter: str,
    min_base_deals: int,
    min_end_deals: int,
    require_both_ends: bool,
    require_full_coverage: bool,
) -> tuple[pd.DataFrame, dict[str, int], int]:
    delta_quarters = quarter_distance(base_quarter, end_quarter)
    if delta_quarters <= 0:
        raise ValueError(
            f"Некорректный интервал: базовый квартал {base_quarter}, конечный {end_quarter}. "
            "Конечный квартал должен быть строго позже базового."
        )

    base_frame = (
        df[df["quarter_label"] == base_quarter][["district_name", "price_per_m2", "n_deals"]]
        .drop_duplicates(subset=["district_name"], keep="last")
        .rename(columns={"price_per_m2": "base_price", "n_deals": "base_deals"})
    )
    end_frame = (
        df[df["quarter_label"] == end_quarter][["district_name", "price_per_m2", "n_deals"]]
        .drop_duplicates(subset=["district_name"], keep="last")
        .rename(columns={"price_per_m2": "current_price", "n_deals": "current_deals"})
    )

    snapshot = pd.merge(base_frame, end_frame, on="district_name", how="outer")
    snapshot["has_base_price"] = snapshot["base_price"].notna().astype(bool)
    snapshot["has_end_price"] = snapshot["current_price"].notna().astype(bool)
    snapshot["has_data_both_ends"] = (snapshot["has_base_price"] & snapshot["has_end_price"]).astype(bool)

    snapshot["base_deals"] = pd.to_numeric(snapshot["base_deals"], errors="coerce")
    snapshot["current_deals"] = pd.to_numeric(snapshot["current_deals"], errors="coerce")
    snapshot["passes_deals_filter"] = (
        snapshot["base_deals"].fillna(0).ge(min_base_deals)
        & snapshot["current_deals"].fillna(0).ge(min_end_deals)
    ).astype(bool)

    coverage = compute_coverage(
        df=df,
        base_idx=parse_quarter(base_quarter).ordinal,
        end_idx=parse_quarter(end_quarter).ordinal,
    )
    snapshot = snapshot.merge(coverage, on="district_name", how="left")
    snapshot["expected_quarters"] = snapshot["expected_quarters"].fillna(delta_quarters + 1).astype("Int64")
    snapshot["observed_quarters"] = snapshot["observed_quarters"].fillna(0).astype("Int64")
    snapshot["coverage_ratio"] = snapshot["coverage_ratio"].fillna(0.0)
    snapshot["full_coverage_flag"] = snapshot["full_coverage_flag"].fillna(False).astype(bool)

    growth_ratio = snapshot["current_price"] / snapshot["base_price"]
    valid_growth = (
        snapshot["has_data_both_ends"]
        & snapshot["base_price"].gt(0)
        & snapshot["current_price"].gt(0)
    ).astype(bool)

    snapshot["abs_change"] = snapshot["current_price"] - snapshot["base_price"]
    snapshot["pct_change"] = np.where(valid_growth, (growth_ratio - 1) * 100, np.nan)
    snapshot["cagr"] = np.where(
        valid_growth,
        ((growth_ratio ** (4 / delta_quarters)) - 1) * 100,
        np.nan,
    )

    snapshot["join_key"] = _normalize_text(snapshot["district_name"])

    snapshot["passes_endpoint_filter"] = snapshot["has_data_both_ends"].astype(bool)
    if not require_both_ends:
        snapshot["passes_endpoint_filter"] = True

    snapshot["passes_coverage_filter"] = snapshot["full_coverage_flag"].astype(bool)
    if not require_full_coverage:
        snapshot["passes_coverage_filter"] = True

    snapshot["has_reliable_data"] = (
        snapshot["passes_endpoint_filter"]
        & snapshot["passes_deals_filter"]
        & snapshot["passes_coverage_filter"]
    ).astype(bool)

    total_districts = int(df["district_name"].nunique())
    diagnostics = {
        "total_districts_panel": total_districts,
        "with_base_price": int(snapshot["has_base_price"].sum()),
        "with_end_price": int(snapshot["has_end_price"].sum()),
        "with_both_prices": int(snapshot["has_data_both_ends"].sum()),
        "pass_min_deals": int(snapshot["passes_deals_filter"].sum()),
        "full_coverage": int(snapshot["full_coverage_flag"].sum()),
    }

    return snapshot, diagnostics, delta_quarters


def build_map_frame(snapshot: pd.DataFrame, gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, list[str]]:
    warnings: list[str] = []
    map_gdf, join_col = harmonize_district_names(gdf, snapshot["district_name"])

    map_gdf = map_gdf.drop_duplicates(subset=["join_key"], keep="first").copy()
    snap = snapshot.drop_duplicates(subset=["join_key"], keep="first").copy()

    merged = map_gdf.merge(snap, on="join_key", how="left", suffixes=("", "_snap"))
    merged["district_name"] = merged["district_name"].fillna(merged[join_col].astype("string"))

    bool_cols = [
        "has_reliable_data",
        "has_data_both_ends",
        "passes_deals_filter",
        "full_coverage_flag",
    ]
    for col in bool_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(False).astype(bool)

    panel_keys = set(snap["join_key"].dropna())
    geo_keys = set(map_gdf["join_key"].dropna())
    missing_in_geo = snap[~snap["join_key"].isin(geo_keys)]["district_name"].dropna().unique().tolist()
    only_geo = map_gdf[~map_gdf["join_key"].isin(panel_keys)][join_col].dropna().astype(str).tolist()
    if missing_in_geo:
        warnings.append(f"В GeoJSON отсутствуют {len(missing_in_geo)} район(а/ов), присутствующие в panel.parquet.")
    if only_geo:
        warnings.append(f"В GeoJSON есть {len(only_geo)} район(а/ов), отсутствующих в panel.parquet.")

    return merged, warnings


def draw_data_quality_summary(diag: dict[str, int], active_count: int) -> None:
    st.subheader("Диагностика покрытия и отбора данных")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Районов в панели", diag["total_districts_panel"])
    c2.metric("Есть цена в базовом", diag["with_base_price"])
    c3.metric("Есть цена в конечном", diag["with_end_price"])
    c4.metric("Есть в обоих кварталах", diag["with_both_prices"])

    d1, d2, d3 = st.columns(3)
    d1.metric("Прошли min deals", diag["pass_min_deals"])
    d2.metric("Полное покрытие периода", diag["full_coverage"])
    d3.metric("Участвуют в метрике", active_count)


def draw_choropleth(
    map_gdf: gpd.GeoDataFrame,
    selected_metric_col: str,
    metric_label: str,
    base_quarter: str,
    end_quarter: str,
    show_gray_unreliable: bool,
) -> None:
    gdf_wgs = map_gdf.to_crs(4326)
    reliable = gdf_wgs[gdf_wgs["has_reliable_data"] & gdf_wgs[selected_metric_col].notna()].copy()
    unreliable = gdf_wgs[~(gdf_wgs["has_reliable_data"] & gdf_wgs[selected_metric_col].notna())].copy()

    fig = go.Figure()

    if not reliable.empty:
        reliable["tooltip_cov"] = reliable.apply(
            lambda r: f"{int(r['observed_quarters'])}/{int(r['expected_quarters'])}", axis=1
        )
        geojson_rel = json.loads(reliable[["join_key", "geometry"]].to_json())
        ch = px.choropleth_mapbox(
            reliable,
            geojson=geojson_rel,
            locations="join_key",
            featureidkey="properties.join_key",
            color=selected_metric_col,
            color_continuous_scale="RdYlGn",
            mapbox_style="carto-positron",
            center={"lat": 55.75, "lon": 37.62},
            zoom=8,
            opacity=0.8,
            custom_data=[
                "district_name",
                "base_price",
                "current_price",
                "cagr",
                "pct_change",
                "abs_change",
                "base_deals",
                "current_deals",
                "tooltip_cov",
                "coverage_ratio",
            ],
        )
        fig.add_traces(ch.data)
        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"База: {base_quarter}<br>"
                f"Конец: {end_quarter}<br>"
                "Цена в базе: %{customdata[1]:,.0f}<br>"
                "Цена в конце: %{customdata[2]:,.0f}<br>"
                "CAGR: %{customdata[3]:.2f}%<br>"
                "Рост: %{customdata[4]:.2f}%<br>"
                "Абс. изменение: %{customdata[5]:,.0f}<br>"
                "Сделки в базе: %{customdata[6]:.0f}<br>"
                "Сделки в конце: %{customdata[7]:.0f}<br>"
                "Покрытие: %{customdata[8]}<br>"
                "coverage_ratio: %{customdata[9]:.2f}<extra></extra>"
            )
        )

    if show_gray_unreliable and not unreliable.empty:
        geojson_unrel = json.loads(unreliable[["join_key", "geometry"]].to_json())
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=geojson_unrel,
                locations=unreliable["join_key"],
                z=[1] * len(unreliable),
                featureidkey="properties.join_key",
                colorscale=[[0, "#BDBDBD"], [1, "#BDBDBD"]],
                marker_opacity=0.6,
                marker_line_width=0.2,
                showscale=False,
                name="Нет надежных данных",
                hovertext=unreliable["district_name"].fillna("Неизвестный район"),
                hovertemplate="<b>%{hovertext}</b><br>Нет надежных данных<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Карта: {metric_label}",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        mapbox_style="carto-positron",
        mapbox_zoom=8,
        mapbox_center={"lat": 55.75, "lon": 37.62},
    )
    st.plotly_chart(fig, use_container_width=True)


def draw_rank_tables(snapshot: pd.DataFrame, selected_metric_col: str, metric_label: str) -> None:
    ranked = snapshot[snapshot["has_reliable_data"] & snapshot[selected_metric_col].notna()].copy()
    if ranked.empty:
        st.info("Нет районов, прошедших фильтры, для построения Top/Bottom таблиц.")
        return

    ranked = ranked.sort_values(selected_metric_col, ascending=False)
    top = ranked.head(10)
    bottom = ranked.tail(10).sort_values(selected_metric_col)

    cols = [
        "district_name",
        "base_price",
        "current_price",
        "abs_change",
        "pct_change",
        "cagr",
        "base_deals",
        "current_deals",
        "coverage_ratio",
    ]

    rename = {
        "district_name": "district_name",
        "base_price": "base_price",
        "current_price": "current_price",
        "abs_change": "abs_change",
        "pct_change": "pct_change",
        "cagr": "cagr",
        "base_deals": "base_deals",
        "current_deals": "current_deals",
        "coverage_ratio": "coverage_ratio",
    }

    st.subheader(f"Top / Bottom 10 по метрике: {metric_label}")
    c1, c2 = st.columns(2)
    c1.markdown("**Top-10**")
    c1.dataframe(top[cols].rename(columns=rename), use_container_width=True)
    c2.markdown("**Bottom-10**")
    c2.dataframe(bottom[cols].rename(columns=rename), use_container_width=True)


def draw_district_timeseries(
    panel: pd.DataFrame,
    district_name: str,
    base_quarter: str,
    end_quarter: str,
    district_snapshot: pd.Series | None,
) -> None:
    district_df = panel[panel["district_name"] == district_name].dropna(subset=["price_per_m2"]).copy()
    district_df = district_df.sort_values("quarter_idx")
    if district_df.empty:
        st.info("Для выбранного района нет временного ряда price_per_m2.")
        return

    fig = px.line(
        district_df,
        x="quarter_dt",
        y="price_per_m2",
        markers=True,
        labels={"quarter_dt": "quarter", "price_per_m2": "price_per_m2"},
        title=f"Динамика price_per_m2: {district_name}",
    )
    fig.add_vline(x=parse_quarter(base_quarter).start_time, line_dash="dash", line_color="green")
    fig.add_vline(x=parse_quarter(end_quarter).start_time, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    if district_snapshot is not None:
        st.caption(
            " | ".join(
                [
                    f"CAGR: {district_snapshot.get('cagr', np.nan):.2f}%",
                    f"Рост: {district_snapshot.get('pct_change', np.nan):.2f}%",
                    f"Абс. изменение: {district_snapshot.get('abs_change', np.nan):,.0f}",
                    f"coverage_ratio: {district_snapshot.get('coverage_ratio', np.nan):.2f}",
                ]
            )
        )


def main() -> None:
    st.set_page_config(page_title="Москва: цены на квартиры по районам (v2)", layout="wide")
    st.title("Анализ цен на квартиры по районам Москвы — Interval v2")

    if not DATA_PATH.exists() or not GEOJSON_PATH.exists():
        st.error("Нужны два файла в корне проекта: panel.parquet и moscow_districts.geojson")
        st.stop()

    panel_raw = load_panel(DATA_PATH)
    panel = validate_panel(panel_raw)
    gdf = load_geojson(GEOJSON_PATH)

    quarter_options = sorted(panel["quarter_label"].dropna().unique().tolist())
    if len(quarter_options) < 2:
        st.error("Для interval-анализа нужно минимум 2 квартала в panel.parquet")
        st.stop()

    default_base = max(0, len(quarter_options) // 4)
    default_end = min(len(quarter_options) - 1, max(default_base + 1, len(quarter_options) - 1))

    with st.sidebar:
        st.header("Параметры интервала")
        base_quarter = st.selectbox("Базовый квартал", quarter_options, index=default_base)
        end_quarter = st.selectbox("Конечный квартал", quarter_options, index=default_end)
        metric_key = st.selectbox(
            "Метрика",
            options=list(METRIC_OPTIONS.keys()),
            format_func=lambda k: METRIC_OPTIONS[k],
            index=0,
        )
        min_base_deals = st.number_input("Минимум сделок в базовом квартале", min_value=0, max_value=1000, value=5)
        min_end_deals = st.number_input("Минимум сделок в конечном квартале", min_value=0, max_value=1000, value=5)
        require_both_ends = st.checkbox("Показывать только районы с данными и в базе, и в конце", value=True)
        require_full_coverage = st.checkbox("Требовать полное покрытие внутри периода", value=False)
        show_gray_unreliable = st.checkbox("Показывать районы без надежных данных серым", value=True)

    try:
        snapshot, diagnostics, delta_q = build_interval_snapshot(
            df=panel,
            base_quarter=base_quarter,
            end_quarter=end_quarter,
            min_base_deals=int(min_base_deals),
            min_end_deals=int(min_end_deals),
            require_both_ends=require_both_ends,
            require_full_coverage=require_full_coverage,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    metric_col = {
        "cagr": "cagr",
        "pct_change": "pct_change",
        "abs_change": "abs_change",
        "base_price": "base_price",
        "current_price": "current_price",
    }[metric_key]
    metric_label = METRIC_OPTIONS[metric_key]

    snapshot["metric_value"] = snapshot[metric_col]
    snapshot["has_metric_value"] = snapshot["metric_value"].notna().astype(bool)
    snapshot["has_reliable_data"] = (snapshot["has_reliable_data"] & snapshot["has_metric_value"]).astype(bool)

    active_count = int(snapshot["has_reliable_data"].sum())
    draw_data_quality_summary(diagnostics, active_count)

    st.info(
        "CAGR считается как ((P_end / P_base)^(4 / delta_quarters) - 1) × 100. "
        "coverage_ratio = observed_quarters / expected_quarters. "
        "Серый цвет на карте означает отсутствие надежных данных для выбранного интервала."
    )

    map_gdf, warnings = build_map_frame(snapshot, gdf)
    for msg in warnings:
        st.warning(msg)

    draw_choropleth(
        map_gdf=map_gdf,
        selected_metric_col="metric_value",
        metric_label=metric_label,
        base_quarter=base_quarter,
        end_quarter=end_quarter,
        show_gray_unreliable=show_gray_unreliable,
    )

    draw_rank_tables(snapshot=snapshot, selected_metric_col="metric_value", metric_label=metric_label)

    st.subheader("Временной ряд выбранного района")
    district_options = sorted(snapshot["district_name"].dropna().unique().tolist())
    selected_district = st.selectbox("Район", options=district_options)
    district_row = snapshot[snapshot["district_name"] == selected_district]
    district_snapshot = district_row.iloc[0] if not district_row.empty else None
    draw_district_timeseries(
        panel=panel,
        district_name=selected_district,
        base_quarter=base_quarter,
        end_quarter=end_quarter,
        district_snapshot=district_snapshot,
    )

    st.caption(f"Интервал: {base_quarter} → {end_quarter} (Δ {delta_q} кварталов)")


if __name__ == "__main__":
    main()
