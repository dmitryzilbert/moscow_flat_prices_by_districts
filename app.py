from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def build_snapshot(
    df: pd.DataFrame, quarter_label: str, window_q: int, min_deals: int, metric_key: str
) -> tuple[pd.DataFrame, list[str]]:
    metric_col = get_metric_column(metric_key, window_q)
    lag_price_col = f"price_lag_{window_q}q"
    lag_deals_col = f"n_deals_lag_{window_q}q"
    warnings: list[str] = []

    required = ["district_name", "quarter_label", "n_deals", "price_per_m2"]
    optional = [lag_price_col, lag_deals_col]

    missing_req = [col for col in required if col not in df.columns]
    if missing_req:
        raise ValueError(f"В panel.parquet отсутствуют нужные колонки: {missing_req}")

    snapshot = df[df["quarter_label"] == quarter_label].copy()

    for col in optional:
        if col not in snapshot.columns:
            snapshot[col] = pd.NA

    if metric_col not in snapshot.columns:
        snapshot[metric_col] = pd.NA
        warnings.append(
            f"В panel.parquet нет колонки {metric_col}. Метрика '{METRIC_CONFIG[metric_key]['label']}' недоступна для выбранного окна."
        )

    snapshot = snapshot.rename(
        columns={
            metric_col: "metric_value",
            "price_per_m2": "current_price_per_m2",
            lag_price_col: "base_price_per_m2",
            "n_deals": "n_deals_current",
            lag_deals_col: "n_deals_base",
        }
    )
    snapshot["has_reliable_data"] = (
        snapshot["n_deals_current"].fillna(0).ge(min_deals)
        & snapshot["metric_value"].notna()
        & snapshot["current_price_per_m2"].notna()
    )
    return snapshot, warnings


def build_map_frame(snapshot: pd.DataFrame, gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, list[str]]:
    warnings: list[str] = []
    geo_join_col = resolve_geo_join_column(gdf, snapshot["district_name"])
    map_gdf = gdf.copy()
    map_gdf["join_key"] = _normalize_text(map_gdf[geo_join_col])

    snap = snapshot.copy()
    snap["join_key"] = _normalize_text(snap["district_name"])

    merged = map_gdf.merge(snap, on="join_key", how="left")
    merged["district_name"] = merged["district_name"].fillna(merged[geo_join_col].astype("string"))
    merged["has_reliable_data"] = merged["has_reliable_data"].fillna(False)

    geo_keys = set(map_gdf["join_key"].dropna().tolist())
    panel_missing_geo = snap[~snap["join_key"].isin(geo_keys)]["district_name"].dropna().unique().tolist()
    if panel_missing_geo:
        preview = ", ".join(sorted(panel_missing_geo)[:5])
        suffix = "..." if len(panel_missing_geo) > 5 else ""
        warnings.append(
            f"В GeoJSON отсутствуют {len(panel_missing_geo)} район(а/ов) из данных: {preview}{suffix}"
        )
    return merged, warnings


def draw_choropleth(map_gdf: gpd.GeoDataFrame, metric_key: str, metric_label: str) -> None:
    map_gdf = map_gdf.to_crs(4326)
    reliable = map_gdf[map_gdf["has_reliable_data"]].copy()
    unreliable = map_gdf[~map_gdf["has_reliable_data"]].copy()

    fig = go.Figure()
    if not reliable.empty:
        reliable["metric_label"] = format_metric_series(reliable["metric_value"], metric_key)
        reliable["curr_label"] = format_metric_series(reliable["current_price_per_m2"], "price_per_m2")
        reliable["base_label"] = format_metric_series(reliable["base_price_per_m2"], "price_per_m2")
        reliable["deals_curr_label"] = reliable["n_deals_current"].fillna(0).astype("Int64").astype(str)
        reliable["deals_base_label"] = reliable["n_deals_base"].fillna(0).astype("Int64").astype(str)

        base_fig = px.choropleth_mapbox(
            reliable,
            geojson=reliable.geometry,
            locations=reliable.index,
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
        fig.add_traces(base_fig.data)
        fig.update_traces(
            selector=dict(type="choroplethmapbox"),
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                f"{metric_label}: %{{customdata[0]}}<br>"
                "Текущая цена: %{customdata[1]}<br>"
                "Базовая цена: %{customdata[2]}<br>"
                "Сделок (текущий): %{customdata[3]}<br>"
                "Сделок (база): %{customdata[4]}<extra></extra>"
            ),
        )

    if not unreliable.empty:
        unreliable["district_name"] = unreliable["district_name"].fillna("Неизвестный район")
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=unreliable.geometry,
                locations=unreliable.index,
                z=[1] * len(unreliable),
                colorscale=[[0, "#BDBDBD"], [1, "#BDBDBD"]],
                marker_opacity=0.7,
                marker_line_width=0.3,
                showscale=False,
                hovertext=unreliable["district_name"],
                customdata=[["Нет надежных данных"]] * len(unreliable),
                hovertemplate="<b>%{hovertext}</b><br>%{customdata[0]}<extra></extra>",
                name="Нет надежных данных",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scattermapbox(
                lat=[None],
                lon=[None],
                mode="markers",
                marker=dict(size=12, color="#BDBDBD"),
                name="Нет надежных данных",
                showlegend=True,
            )
        )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        mapbox_style="carto-positron",
        mapbox_zoom=8,
        mapbox_center={"lat": 55.75, "lon": 37.62},
        legend=dict(title="Легенда"),
    )
    st.plotly_chart(fig, use_container_width=True)


def draw_top_bottom(snapshot: pd.DataFrame, metric_key: str, metric_label: str) -> None:
    ranked = snapshot[snapshot["has_reliable_data"]].dropna(subset=["metric_value"]).copy()
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
    district_df = district_df.dropna(subset=["quarter_dt", "price_per_m2"])
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
    snapshot, snapshot_warnings = build_snapshot(df, selected_quarter, window_q, min_deals, metric_key)
    for msg in snapshot_warnings:
        st.warning(msg)

    if snapshot.empty:
        st.warning("Для выбранных параметров нет данных. Попробуйте снизить фильтр по сделкам.")
        st.stop()

    st.subheader(f"Карта: {metric_label} (окно {window_q} кв., квартал {selected_quarter})")
    map_gdf, map_warnings = build_map_frame(snapshot, gdf)
    for msg in map_warnings:
        st.warning(msg)
    unreliable_count = int((~snapshot["has_reliable_data"]).sum())
    if unreliable_count > 0:
        st.warning(
            f"Для {unreliable_count} район(а/ов) в выбранном квартале недостаточно данных: они отмечены серым цветом."
        )
    draw_choropleth(map_gdf, metric_key, metric_label)
    st.caption(
        "Рост считается относительно цены window_q кварталов назад: "
        "abs_change = current - base; pct_change = (current / base - 1) × 100; "
        "CAGR = ((current / base)^(4/window_q) - 1) × 100."
    )

    st.subheader("Сводные таблицы")
    draw_top_bottom(snapshot, metric_key, metric_label)
    export_cols = [
        "district_name",
        "metric_value",
        "current_price_per_m2",
        "base_price_per_m2",
        "n_deals_current",
        "n_deals_base",
        "has_reliable_data",
    ]
    export_table = snapshot[export_cols].rename(
        columns={
            "district_name": "Район",
            "metric_value": metric_label,
            "current_price_per_m2": "Текущая цена, ₽/м²",
            "base_price_per_m2": "Базовая цена, ₽/м²",
            "n_deals_current": "Сделок (текущий)",
            "n_deals_base": "Сделок (база)",
            "has_reliable_data": "Надежные данные",
        }
    )
    st.download_button(
        label="Скачать текущую таблицу (CSV)",
        data=export_table.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"snapshot_{selected_quarter}_{metric_key}_{window_q}q.csv",
        mime="text/csv",
    )

    st.subheader("Временной ряд района")
    available_districts = sorted(snapshot["district_name"].dropna().unique().tolist())
    selected_district = st.selectbox("Район", available_districts)
    draw_district_line(df, selected_district)


if __name__ == "__main__":
    main()
