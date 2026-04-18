from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH = Path("panel.parquet")
GEOJSON_PATH = Path("moscow_districts.geojson")
WINDOW_OPTIONS = list(range(4, 33, 4))

METRIC_CONFIG = {
    "price_per_m2": {
        "label": "Текущая цена за м²",
        "is_window_metric": False,
        "format": ",.0f",
    },
    "abs_change": {
        "label": "Абсолютное изменение (₽/м²)",
        "is_window_metric": True,
        "format": ",.0f",
    },
    "pct_change": {
        "label": "Изменение (%)",
        "is_window_metric": True,
        "format": ",.2f",
    },
    "CAGR": {
        "label": "CAGR (%)",
        "is_window_metric": True,
        "format": ",.2f",
    },
}


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


@st.cache_data(show_spinner=False)
def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "district_name" not in df.columns:
        raise ValueError("В panel.parquet отсутствует колонка district_name")

    if "quarter_ts" in df.columns:
        df["quarter_dt"] = pd.to_datetime(df["quarter_ts"], errors="coerce")
    else:
        df["quarter_dt"] = pd.to_datetime(df.get("quarter"), errors="coerce")

    if df["quarter_dt"].isna().all():
        raise ValueError("Не удалось распарсить кварталы: quarter_ts/quarter")

    df = df.sort_values(["quarter_dt", "district_name"]).copy()
    df["quarter_label"] = df["quarter_dt"].dt.to_period("Q").astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_geo(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("GeoJSON не содержит объектов")
    return gdf


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
    object_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    for col in object_cols:
        if col not in available:
            available.append(col)
    return available


def resolve_geo_join_column(gdf: gpd.GeoDataFrame, panel_districts: pd.Series) -> str:
    panel_norm = set(_normalize_text(panel_districts.dropna()))
    best_col = None
    best_overlap = -1

    for col in _candidate_name_columns(gdf):
        overlap = len(set(_normalize_text(gdf[col].dropna())) & panel_norm)
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = col

    if best_col is None or best_overlap <= 0:
        raise ValueError(
            "Не удалось подобрать колонку для связки GeoJSON с district_name. "
            "Добавьте district_name в geojson или скорректируйте названия."
        )

    return best_col


def get_metric_column(metric_key: str, window_q: int) -> str:
    if metric_key == "price_per_m2":
        return "price_per_m2"
    if metric_key == "abs_change":
        return f"abs_change_{window_q}q"
    if metric_key == "pct_change":
        return f"pct_change_{window_q}q"
    if metric_key == "CAGR":
        return f"cagr_{window_q}q"
    raise KeyError(f"Неизвестная метрика: {metric_key}")


def format_metric_series(series: pd.Series, metric_key: str) -> pd.Series:
    fmt = METRIC_CONFIG[metric_key]["format"]
    return series.map(lambda x: f"{x:{fmt}}" if pd.notna(x) else "—")


def build_snapshot(df: pd.DataFrame, quarter_label: str, window_q: int, min_deals: int, metric_key: str) -> pd.DataFrame:
    metric_col = get_metric_column(metric_key, window_q)
    lag_price_col = f"price_lag_{window_q}q"
    lag_deals_col = f"n_deals_lag_{window_q}q"

    required = ["district_name", "quarter_label", "n_deals", "price_per_m2", metric_col]
    optional = [lag_price_col, lag_deals_col]

    missing_req = [col for col in required if col not in df.columns]
    if missing_req:
        raise ValueError(f"В panel.parquet отсутствуют нужные колонки: {missing_req}")

    snapshot = df[df["quarter_label"] == quarter_label].copy()
    snapshot = snapshot[snapshot["n_deals"].fillna(0) >= min_deals]

    for col in optional:
        if col not in snapshot.columns:
            snapshot[col] = pd.NA

    snapshot = snapshot.rename(
        columns={
            metric_col: "metric_value",
            "price_per_m2": "current_price_per_m2",
            lag_price_col: "base_price_per_m2",
            "n_deals": "n_deals_current",
            lag_deals_col: "n_deals_base",
        }
    )
    return snapshot


def build_map_frame(snapshot: pd.DataFrame, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geo_join_col = resolve_geo_join_column(gdf, snapshot["district_name"])
    map_gdf = gdf.copy()
    map_gdf["join_key"] = _normalize_text(map_gdf[geo_join_col])

    snap = snapshot.copy()
    snap["join_key"] = _normalize_text(snap["district_name"])

    merged = map_gdf.merge(snap, on="join_key", how="left")
    merged["district_name"] = merged["district_name"].fillna(merged[geo_join_col].astype("string"))
    return merged


def draw_choropleth(map_gdf: gpd.GeoDataFrame, metric_key: str, metric_label: str) -> None:
    map_gdf = map_gdf.to_crs(4326)
    map_gdf["metric_label"] = format_metric_series(map_gdf["metric_value"], metric_key)
    map_gdf["curr_label"] = format_metric_series(map_gdf["current_price_per_m2"], "price_per_m2")
    map_gdf["base_label"] = format_metric_series(map_gdf["base_price_per_m2"], "price_per_m2")
    map_gdf["deals_curr_label"] = map_gdf["n_deals_current"].fillna(0).astype("Int64").astype(str)
    map_gdf["deals_base_label"] = map_gdf["n_deals_base"].fillna(0).astype("Int64").astype(str)

    fig = px.choropleth_mapbox(
        map_gdf,
        geojson=map_gdf.geometry,
        locations=map_gdf.index,
        color="metric_value",
        color_continuous_scale="RdYlGn",
        mapbox_style="carto-positron",
        zoom=8,
        center={"lat": 55.75, "lon": 37.62},
        opacity=0.75,
        hover_name="district_name",
        custom_data=[
            "metric_label",
            "curr_label",
            "base_label",
            "deals_curr_label",
            "deals_base_label",
        ],
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            f"{metric_label}: %{{customdata[0]}}<br>"
            "Текущая цена: %{customdata[1]}<br>"
            "Базовая цена: %{customdata[2]}<br>"
            "Сделок (текущий): %{customdata[3]}<br>"
            "Сделок (база): %{customdata[4]}<extra></extra>"
        )
    )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)


def draw_top_bottom(snapshot: pd.DataFrame, metric_key: str, metric_label: str) -> None:
    ranked = snapshot.dropna(subset=["metric_value"]).copy()
    if ranked.empty:
        st.info("Нет данных для ранжирования по выбранным параметрам.")
        return

    ranked = ranked.sort_values("metric_value", ascending=False)
    top10 = ranked.head(10)
    bottom10 = ranked.tail(10).sort_values("metric_value", ascending=True)

    display_cols = [
        "district_name",
        "metric_value",
        "current_price_per_m2",
        "base_price_per_m2",
        "n_deals_current",
        "n_deals_base",
    ]
    rename_cols = {
        "district_name": "Район",
        "metric_value": metric_label,
        "current_price_per_m2": "Текущая цена, ₽/м²",
        "base_price_per_m2": "Базовая цена, ₽/м²",
        "n_deals_current": "Сделок (текущий)",
        "n_deals_base": "Сделок (база)",
    }

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Топ-10")
        st.dataframe(top10[display_cols].rename(columns=rename_cols), use_container_width=True)
    with col2:
        st.subheader("Боттом-10")
        st.dataframe(bottom10[display_cols].rename(columns=rename_cols), use_container_width=True)


def draw_district_line(df: pd.DataFrame, district_name: str) -> None:
    district_df = df[df["district_name"] == district_name].sort_values("quarter_dt").copy()
    if district_df.empty:
        st.info("Нет временного ряда для выбранного района.")
        return

    fig = px.line(
        district_df,
        x="quarter_dt",
        y="price_per_m2",
        markers=True,
        title=f"Динамика цены за м²: {district_name}",
        labels={"quarter_dt": "Квартал", "price_per_m2": "₽/м²"},
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Карта цен по районам Москвы", layout="wide")
    st.title("Интерактивная карта цен на квартиры по районам Москвы")

    if not DATA_PATH.exists() or not GEOJSON_PATH.exists():
        st.error(
            "Не найдены входные файлы panel.parquet и/или moscow_districts.geojson в текущей папке."
        )
        st.stop()

    df = load_panel(DATA_PATH)
    gdf = load_geo(GEOJSON_PATH)

    quarter_options = df["quarter_label"].dropna().sort_values().unique().tolist()
    if not quarter_options:
        st.error("В panel.parquet отсутствуют валидные кварталы.")
        st.stop()

    with st.sidebar:
        st.header("Параметры анализа")
        selected_quarter = st.selectbox("Конечный квартал", quarter_options, index=len(quarter_options) - 1)
        window_q = st.slider("Окно анализа (кварталы)", min_value=4, max_value=32, step=4, value=8)
        metric_key = st.selectbox("Метрика", list(METRIC_CONFIG.keys()), format_func=lambda x: METRIC_CONFIG[x]["label"])
        min_deals = st.slider("Минимум сделок (текущий квартал)", min_value=0, max_value=200, step=1, value=5)

    metric_label = METRIC_CONFIG[metric_key]["label"]
    snapshot = build_snapshot(df, selected_quarter, window_q, min_deals, metric_key)

    if snapshot.empty:
        st.warning("Для выбранных параметров нет данных. Попробуйте снизить фильтр по сделкам.")
        st.stop()

    st.subheader(f"Карта: {metric_label} (окно {window_q} кв., квартал {selected_quarter})")
    map_gdf = build_map_frame(snapshot, gdf)
    draw_choropleth(map_gdf, metric_key, metric_label)

    st.subheader("Сводные таблицы")
    draw_top_bottom(snapshot, metric_key, metric_label)

    st.subheader("Временной ряд района")
    available_districts = sorted(snapshot["district_name"].dropna().unique().tolist())
    selected_district = st.selectbox("Район", available_districts)
    draw_district_line(df, selected_district)


if __name__ == "__main__":
    main()
