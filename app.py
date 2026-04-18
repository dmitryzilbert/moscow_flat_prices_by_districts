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

QUARTER_MODE_OPTIONS = {
    "strict": "Strict exact quarter",
    "carry_forward": "Use last available observation up to selected quarter",
}


def _normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace("ё", "е", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )


def parse_quarter(value: object) -> pd.Period | pd.NaT:
    if value is None or pd.isna(value):
        return pd.NaT
    try:
        raw = str(value).strip().upper()
        normalized = (
            raw.replace(" ", "")
            .replace("-", "")
            .replace("/", "")
            .replace("_", "")
            .replace("КВ", "Q")
        )
        if len(normalized) >= 6 and normalized[0] == "Q" and normalized[1] in "1234":
            normalized = f"{normalized[2:6]}Q{normalized[1]}"
        return pd.Period(normalized, freq="Q")
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
        valid = year.notna() & q_num.isin([1, 2, 3, 4])
        quarter_str = (
            year.astype("Int64").astype("string")
            + "Q"
            + q_num.astype("Int64").astype("string")
        )
        out["quarter_parsed"] = pd.Series(pd.NaT, index=out.index, dtype="period[Q-DEC]")
        out.loc[valid, "quarter_parsed"] = quarter_str[valid].apply(parse_quarter)
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
    if best_overlap <= 0:
        raise ValueError("Не найдено пересечений названий районов между panel и GeoJSON. Проверьте нейминг.")

    gdf_out = gdf.copy()
    gdf_out["join_key"] = _normalize_text(gdf_out[best_col])
    return gdf_out, best_col


@st.cache_data(show_spinner=False)
def load_geojson(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("GeoJSON не содержит геометрий")
    return gdf


def _build_quarter_frame(df: pd.DataFrame, selected_idx: int, mode: str, side: str) -> pd.DataFrame:
    if mode == "strict":
        part = df[df["quarter_idx"] == selected_idx].copy()
    else:
        # Carry-forward should use the latest *available* price observation,
        # not simply the latest row by quarter (which may contain NaN price).
        part = df[df["quarter_idx"] <= selected_idx].copy()
        part = part[part["price_per_m2"].notna()].copy()

    part = part.sort_values(["district_name", "quarter_idx"]).drop_duplicates(
        subset=["district_name"], keep="last"
    )
    return part[["district_name", "quarter_label", "quarter_idx", "price_per_m2", "n_deals"]].rename(
        columns={
            "quarter_label": f"effective_{side}_quarter",
            "quarter_idx": f"effective_{side}_idx",
            "price_per_m2": "base_price" if side == "base" else "current_price",
            "n_deals": "base_deals" if side == "base" else "current_deals",
        }
    )


def compute_effective_coverage(panel: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    work = snapshot[["district_name", "effective_base_idx", "effective_end_idx"]].copy()
    work["expected_quarters"] = np.where(
        work["effective_base_idx"].notna() & work["effective_end_idx"].notna(),
        (work["effective_end_idx"] - work["effective_base_idx"] + 1),
        np.nan,
    )

    observed_source = panel[panel["price_per_m2"].notna()][["district_name", "quarter_idx"]].drop_duplicates().copy()
    obs_join = observed_source.merge(
        work[["district_name", "effective_base_idx", "effective_end_idx"]],
        on="district_name",
        how="inner",
    )
    in_effective_range = obs_join[
        obs_join["quarter_idx"].ge(obs_join["effective_base_idx"])
        & obs_join["quarter_idx"].le(obs_join["effective_end_idx"])
    ]
    observed = (
        in_effective_range.groupby("district_name", dropna=False)["quarter_idx"]
        .nunique()
        .rename("observed_quarters")
        .reset_index()
    )

    coverage = work.merge(observed, on="district_name", how="left")
    coverage["observed_quarters"] = coverage["observed_quarters"].fillna(0)
    coverage["coverage_ratio"] = np.where(
        coverage["expected_quarters"].gt(0),
        coverage["observed_quarters"] / coverage["expected_quarters"],
        np.nan,
    )
    coverage["full_coverage_flag"] = (
        coverage["expected_quarters"].gt(0) & coverage["observed_quarters"].eq(coverage["expected_quarters"])
    )

    coverage["expected_quarters"] = coverage["expected_quarters"].astype("Int64")
    coverage["observed_quarters"] = coverage["observed_quarters"].astype("Int64")
    coverage["full_coverage_flag"] = coverage["full_coverage_flag"].astype(bool)
    return coverage


def build_interval_snapshot(
    df: pd.DataFrame,
    base_quarter: str,
    end_quarter: str,
    min_base_deals: int,
    min_end_deals: int,
    require_both_ends: bool,
    require_full_coverage: bool,
    quarter_mode: str,
) -> tuple[pd.DataFrame, dict[str, int], int]:
    base_period = parse_quarter(base_quarter)
    end_period = parse_quarter(end_quarter)
    if pd.isna(base_period) or pd.isna(end_period):
        raise ValueError("Не удалось распарсить базовый или конечный квартал.")

    selected_delta = int(end_period.ordinal - base_period.ordinal)
    if selected_delta <= 0:
        raise ValueError(
            f"Некорректный интервал: базовый квартал {base_quarter}, конечный {end_quarter}. "
            "Конечный квартал должен быть строго позже базового."
        )

    base_frame = _build_quarter_frame(df, base_period.ordinal, quarter_mode, side="base")
    end_frame = _build_quarter_frame(df, end_period.ordinal, quarter_mode, side="end")

    all_districts = pd.DataFrame({"district_name": df["district_name"].dropna().unique()})
    snapshot = all_districts.merge(base_frame, on="district_name", how="left").merge(
        end_frame, on="district_name", how="left"
    )

    snapshot["selected_base_quarter"] = base_quarter
    snapshot["selected_end_quarter"] = end_quarter

    snapshot["effective_base_quarter"] = snapshot["effective_base_quarter"].astype("string")
    snapshot["effective_end_quarter"] = snapshot["effective_end_quarter"].astype("string")
    snapshot["effective_base_idx"] = snapshot["effective_base_idx"].astype("Int64")
    snapshot["effective_end_idx"] = snapshot["effective_end_idx"].astype("Int64")

    snapshot["has_base_price"] = snapshot["base_price"].notna().astype(bool)
    snapshot["has_end_price"] = snapshot["current_price"].notna().astype(bool)
    snapshot["has_data_both_ends"] = (snapshot["has_base_price"] & snapshot["has_end_price"]).astype(bool)

    snapshot["effective_delta_quarters"] = (
        snapshot["effective_end_idx"] - snapshot["effective_base_idx"]
    ).astype("Int64")
    snapshot["valid_effective_interval"] = (
        snapshot["effective_delta_quarters"].gt(0) & snapshot["has_data_both_ends"]
    ).astype(bool)

    snapshot["base_deals"] = pd.to_numeric(snapshot["base_deals"], errors="coerce")
    snapshot["current_deals"] = pd.to_numeric(snapshot["current_deals"], errors="coerce")
    snapshot["passes_deals_filter"] = (
        snapshot["base_deals"].fillna(0).ge(min_base_deals)
        & snapshot["current_deals"].fillna(0).ge(min_end_deals)
    ).astype(bool)

    coverage = compute_effective_coverage(panel=df, snapshot=snapshot)
    snapshot = snapshot.merge(
        coverage[["district_name", "expected_quarters", "observed_quarters", "coverage_ratio", "full_coverage_flag"]],
        on="district_name",
        how="left",
    )
    snapshot["expected_quarters"] = snapshot["expected_quarters"].astype("Int64")
    snapshot["observed_quarters"] = snapshot["observed_quarters"].fillna(0).astype("Int64")
    # Keep NaN when effective interval is not defined or non-positive.
    snapshot["coverage_ratio"] = pd.to_numeric(snapshot["coverage_ratio"], errors="coerce")
    snapshot["full_coverage_flag"] = snapshot["full_coverage_flag"].fillna(False).astype(bool)

    growth_ratio = snapshot["current_price"] / snapshot["base_price"]
    valid_growth = (
        snapshot["has_data_both_ends"]
        & snapshot["base_price"].gt(0)
        & snapshot["current_price"].gt(0)
        & snapshot["effective_delta_quarters"].gt(0)
    ).astype(bool)

    snapshot["abs_change"] = snapshot["current_price"] - snapshot["base_price"]
    snapshot["pct_change"] = np.where(valid_growth, (growth_ratio - 1) * 100, np.nan)
    snapshot["cagr"] = np.where(
        valid_growth,
        ((growth_ratio ** (4 / snapshot["effective_delta_quarters"])) - 1) * 100,
        np.nan,
    )

    snapshot["join_key"] = _normalize_text(snapshot["district_name"])

    snapshot["passes_endpoint_filter"] = (
        snapshot["has_data_both_ends"] & snapshot["valid_effective_interval"]
    ).astype(bool)
    if not require_both_ends:
        snapshot["passes_endpoint_filter"] = pd.Series(True, index=snapshot.index, dtype=bool)

    snapshot["passes_coverage_filter"] = snapshot["full_coverage_flag"].astype(bool)
    if not require_full_coverage:
        snapshot["passes_coverage_filter"] = pd.Series(True, index=snapshot.index, dtype=bool)

    snapshot["has_reliable_data"] = (
        snapshot["passes_endpoint_filter"]
        & snapshot["passes_deals_filter"]
        & snapshot["passes_coverage_filter"]
    ).astype(bool)

    exact_base = df[df["quarter_idx"] == base_period.ordinal][["district_name", "price_per_m2"]]
    exact_end = df[df["quarter_idx"] == end_period.ordinal][["district_name", "price_per_m2"]]
    exact_pair = exact_base.merge(exact_end, on="district_name", how="inner")
    exact_pair_count = int(
        exact_pair[(exact_pair["price_per_m2_x"].notna()) & (exact_pair["price_per_m2_y"].notna())]["district_name"].nunique()
    )

    total_districts = int(df["district_name"].nunique())
    missing_base = int((~snapshot["has_base_price"]).sum())
    missing_end = int((snapshot["has_base_price"] & ~snapshot["has_end_price"]).sum())
    invalid_effective = int((snapshot["has_data_both_ends"] & ~snapshot["valid_effective_interval"]).sum())
    with_both = int((snapshot["has_data_both_ends"] & snapshot["valid_effective_interval"]).sum())

    diagnostics = {
        "total_districts_panel": total_districts,
        "with_base_price": int(snapshot["has_base_price"].sum()),
        "with_end_price": int(snapshot["has_end_price"].sum()),
        "with_both_prices": int(snapshot["has_data_both_ends"].sum()),
        "pass_min_deals": int(snapshot["passes_deals_filter"].sum()),
        "full_coverage": int(snapshot["full_coverage_flag"].sum()),
        "exact_pair_count": exact_pair_count,
        "carry_forward_only_pair_count": max(with_both - exact_pair_count, 0),
        "excluded_no_data_before_base": missing_base,
        "excluded_no_data_before_end": missing_end,
        "excluded_nonpositive_effective_delta": invalid_effective,
    }

    return snapshot, diagnostics, selected_delta


def build_map_frame(snapshot: pd.DataFrame, gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, list[str]]:
    warnings: list[str] = []
    try:
        map_gdf, join_col = harmonize_district_names(gdf, snapshot["district_name"])
    except ValueError as exc:
        raise ValueError(f"Ошибка merge panel ↔ GeoJSON: {exc}") from exc

    map_gdf = map_gdf.drop_duplicates(subset=["join_key"], keep="first").copy()
    snap = snapshot.drop_duplicates(subset=["join_key"], keep="first").copy()

    merged = map_gdf.merge(snap, on="join_key", how="left", suffixes=("", "_snap"))
    merged["district_name"] = merged["district_name"].fillna(merged[join_col].astype("string"))

    bool_cols = [
        "has_reliable_data",
        "has_data_both_ends",
        "passes_deals_filter",
        "full_coverage_flag",
        "valid_effective_interval",
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

    e1, e2, e3 = st.columns(3)
    e1.metric("Точная пара кварталов", diag["exact_pair_count"])
    e2.metric("Пара только через carry-forward", diag["carry_forward_only_pair_count"])
    e3.metric("Исключены: нет данных до base", diag["excluded_no_data_before_base"])

    f1, f2 = st.columns(2)
    f1.metric("Исключены: нет данных до end", diag["excluded_no_data_before_end"])
    f2.metric("Исключены: effective end ≤ base", diag["excluded_nonpositive_effective_delta"])


def draw_choropleth(
    map_gdf: gpd.GeoDataFrame,
    selected_metric_col: str,
    metric_label: str,
    show_gray_unreliable: bool,
) -> None:
    gdf_wgs = map_gdf.to_crs(4326)
    reliable = gdf_wgs[gdf_wgs["has_reliable_data"] & gdf_wgs[selected_metric_col].notna()].copy()
    unreliable = gdf_wgs[~(gdf_wgs["has_reliable_data"] & gdf_wgs[selected_metric_col].notna())].copy()

    fig = go.Figure()

    if not reliable.empty:
        reliable["tooltip_cov"] = reliable.apply(
            lambda r: (
                f"{int(r['observed_quarters'])}/{int(r['expected_quarters'])}"
                if pd.notna(r["expected_quarters"]) and r["expected_quarters"] > 0
                else "n/a"
            ),
            axis=1,
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
                "selected_base_quarter",
                "selected_end_quarter",
                "effective_base_quarter",
                "effective_end_quarter",
                "effective_delta_quarters",
                "base_price",
                "current_price",
                "cagr",
                "pct_change",
                "abs_change",
                "base_deals",
                "current_deals",
                "coverage_ratio",
                "tooltip_cov",
            ],
        )
        fig.add_traces(ch.data)
        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Selected base: %{customdata[1]}<br>"
                "Selected end: %{customdata[2]}<br>"
                "Effective base: %{customdata[3]}<br>"
                "Effective end: %{customdata[4]}<br>"
                "Effective delta quarters: %{customdata[5]}<br>"
                "Цена в базе: %{customdata[6]:,.0f}<br>"
                "Цена в конце: %{customdata[7]:,.0f}<br>"
                "CAGR: %{customdata[8]:.2f}%<br>"
                "Рост: %{customdata[9]:.2f}%<br>"
                "Абс. изменение: %{customdata[10]:,.0f}<br>"
                "Сделки в базе: %{customdata[11]:.0f}<br>"
                "Сделки в конце: %{customdata[12]:.0f}<br>"
                "Coverage ratio: %{customdata[13]:.2f}<br>"
                "Покрытие (obs/exp): %{customdata[14]}<extra></extra>"
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
        "selected_base_quarter",
        "selected_end_quarter",
        "effective_base_quarter",
        "effective_end_quarter",
        "effective_delta_quarters",
        "base_price",
        "current_price",
        "abs_change",
        "pct_change",
        "cagr",
        "base_deals",
        "current_deals",
        "coverage_ratio",
    ]

    st.subheader(f"Top / Bottom 10 по метрике: {metric_label}")
    c1, c2 = st.columns(2)
    c1.markdown("**Top-10**")
    c1.dataframe(top[cols], use_container_width=True)
    c2.markdown("**Bottom-10**")
    c2.dataframe(bottom[cols], use_container_width=True)


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

    selected_base = parse_quarter(base_quarter)
    selected_end = parse_quarter(end_quarter)
    if pd.notna(selected_base):
        fig.add_vline(x=selected_base.start_time, line_dash="dash", line_color="green")
    if pd.notna(selected_end):
        fig.add_vline(x=selected_end.start_time, line_dash="dash", line_color="red")

    if district_snapshot is not None:
        eff_base_q = district_snapshot.get("effective_base_quarter")
        eff_end_q = district_snapshot.get("effective_end_quarter")
        eff_base = parse_quarter(eff_base_q)
        eff_end = parse_quarter(eff_end_q)

        marker_x = []
        marker_y = []
        marker_text = []
        if pd.notna(eff_base) and pd.notna(district_snapshot.get("base_price")):
            marker_x.append(eff_base.start_time)
            marker_y.append(float(district_snapshot.get("base_price")))
            marker_text.append("effective base")
        if pd.notna(eff_end) and pd.notna(district_snapshot.get("current_price")):
            marker_x.append(eff_end.start_time)
            marker_y.append(float(district_snapshot.get("current_price")))
            marker_text.append("effective end")

        if marker_x:
            fig.add_trace(
                go.Scatter(
                    x=marker_x,
                    y=marker_y,
                    mode="markers+text",
                    text=marker_text,
                    textposition="top center",
                    marker={"size": 12, "symbol": "diamond", "color": "black"},
                    name="effective points",
                )
            )

    st.plotly_chart(fig, use_container_width=True)

    if district_snapshot is not None:
        st.caption(
            " | ".join(
                [
                    f"Selected: {district_snapshot.get('selected_base_quarter')} → {district_snapshot.get('selected_end_quarter')}",
                    f"Effective: {district_snapshot.get('effective_base_quarter')} → {district_snapshot.get('effective_end_quarter')}",
                    f"CAGR: {district_snapshot.get('cagr', np.nan):.2f}%",
                    f"Рост: {district_snapshot.get('pct_change', np.nan):.2f}%",
                    f"Абс. изменение: {district_snapshot.get('abs_change', np.nan):,.0f}",
                    f"coverage_ratio: {district_snapshot.get('coverage_ratio', np.nan):.2f}",
                ]
            )
        )


def main() -> None:
    st.set_page_config(page_title="Москва: цены на квартиры по районам (v2)", layout="wide")
    st.title("Анализ цен на квартиры по районам Москвы — Interval v2.1")

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
        quarter_mode = st.radio(
            "Режим обработки кварталов",
            options=list(QUARTER_MODE_OPTIONS.keys()),
            format_func=lambda k: QUARTER_MODE_OPTIONS[k],
            index=1,
        )
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
        snapshot, diagnostics, selected_delta_q = build_interval_snapshot(
            df=panel,
            base_quarter=base_quarter,
            end_quarter=end_quarter,
            min_base_deals=int(min_base_deals),
            min_end_deals=int(min_end_deals),
            require_both_ends=require_both_ends,
            require_full_coverage=require_full_coverage,
            quarter_mode=quarter_mode,
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

    if quarter_mode == "strict":
        st.info(
            "Strict mode: используются только точные совпадения quarter == selected quarter. "
            "CAGR считается по выбранному интервалу только если обе точки существуют и effective end > effective base."
        )
    else:
        st.info(
            "Carry-forward mode: для каждого района берется последнее доступное наблюдение не позже выбранного квартала. "
            "CAGR считается по фактическому effective interval между найденными effective quarter."
        )

    map_gdf, warnings = build_map_frame(snapshot, gdf)
    for msg in warnings:
        st.warning(msg)

    draw_choropleth(
        map_gdf=map_gdf,
        selected_metric_col="metric_value",
        metric_label=metric_label,
        show_gray_unreliable=show_gray_unreliable,
    )

    draw_rank_tables(snapshot=snapshot, selected_metric_col="metric_value", metric_label=metric_label)

    st.subheader("Временной ряд выбранного района")
    district_options = sorted(snapshot["district_name"].dropna().unique().tolist())
    if not district_options:
        st.info("Нет районов для выбора в блоке временного ряда.")
        st.caption(f"Selected интервал: {base_quarter} → {end_quarter} (Δ {selected_delta_q} кварталов)")
        return
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

    st.caption(f"Selected интервал: {base_quarter} → {end_quarter} (Δ {selected_delta_q} кварталов)")


if __name__ == "__main__":
    main()
