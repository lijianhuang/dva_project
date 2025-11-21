import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import plotly.express as px
import json
import time
import itertools
import numpy as np
import shap
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_mesh_shap_explanation(model_name: str):
    """Load mesh-level SHAP Explanation (.pkl) for a given model."""
    model_key = model_name.strip().lower()
    
    # e.g. BASE_DIR / "mesh_xgboost_val_shap.pkl"
    shap_path = BASE_DIR / f"mesh_{model_key}_val_shap.pkl"

    if not shap_path.exists():
        st.warning(f"SHAP pickle not found: {shap_path}")
        return None

    with shap_path.open("rb") as f:
        shap_exp = pickle.load(f)

    return shap_exp

def st_shap(plot, height=300):
    """Render a SHAP JS plot (e.g., force_plot) in Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)



st.set_page_config(page_title="Japan Housing Dashboard", layout="wide")
st.title("🏙️ Japan Housing Dashboard — Mesh-250 m Dynamic Visualization")

# --------------------------
# City definitions
# --------------------------
TOKYO_WARDS = {f"131{str(i).zfill(2)}" for i in range(1,24)}  # 13101..13123
SENDAI_WARDS = {"04101","04102","04103","04104","04105"}

# fallback bounding boxes (min_lat, max_lat, min_lon, max_lon)
BBOXES = {
    "Tokyo":  (35.40, 35.92, 139.40, 139.95),
    "Sendai": (38.15, 38.40, 140.70, 141.05),
}

CENTER = {
    "Tokyo":  dict(latitude=35.68, longitude=139.76, zoom=9),
    "Sendai": dict(latitude=38.27, longitude=140.87, zoom=10),
}

# =====================================================
# 1) Load or create placeholder mesh data
# =====================================================
@st.cache_data
def load_predictions():
    try:
        df_pred = pd.read_csv("model_predictions_viz.csv")
    except Exception as e:
        st.error(f"❌ Failed to load model_predictions_viz.csv: {e}")
        df_pred = pd.DataFrame()
    return df_pred

df_pred = load_predictions()



@st.cache_data
def load_mesh_data():

    gdf = gpd.read_file("mesh250_all_quarters.geojson")

    # Ensure numeric types
    for col in [
        "mesh_mean_ppsqm", "mesh_median_ppsqm", "mesh_transaction_count",
        "mesh_avg_age", "mesh_avg_area", "PriceIndex"
    ]:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    # Normalize MunicipalityCode if present
    if "MunicipalityCode" in gdf.columns:
        gdf["MunicipalityCode"] = gdf["MunicipalityCode"].astype(str).str.zfill(5)

    return gdf

gdf = load_mesh_data()

# Load model metrics
df_results = pd.read_csv("model_results.csv")

# Filter only Mesh level models
mesh_models = df_results[df_results["Level"].str.lower() == "Mesh"]

if not mesh_models.empty:
    best_model_mesh = mesh_models.loc[mesh_models["test_rmse"].idxmin()]
    best_model_name = best_model_mesh["Model"]
    best_model_rmse = best_model_mesh["test_rmse"]
else:
    best_model_name = "N/A"
    best_model_rmse = float("nan")

# Coverage (based on mesh_quarterly_price_index.csv)
# =====================================================
try:
    price_index_df = pd.read_csv("mesh_quarterly_price_index.csv")
    # Unique meshes that have valid price index values
    coverage = price_index_df['Latitude'].notnull().sum()
    total = len(price_index_df["Latitude"])
    coverage_pct = (coverage / total * 100) if total > 0 else 0
except Exception as e:
    st.warning(f"⚠️ Unable to load mesh_quarterly_price_index.csv ({e})")
    coverage, total, coverage_pct = 0, 0, 0


#Sidebar Controls

st.sidebar.title("🏠 Japan Housing Forecast")
st.sidebar.write(f"**Data Coverage (Mesh250m)**")
st.sidebar.write(f"{coverage_pct:.1f}({coverage}/{total})%")
st.sidebar.caption("Mesh-250 m granularity · Tokyo · Sendai")

city = st.sidebar.selectbox("Select City", ["Tokyo", "Sendai"])
# Dynamically populate model list
model_list = sorted(df_pred["Model"].unique())
model = st.sidebar.selectbox("Select Model", model_list)
quarters = sorted(gdf["PeriodKey"].unique())
period = st.sidebar.select_slider("Select Quarter", options=quarters, value=quarters[0])
animate = st.sidebar.checkbox("Auto-play Animation", value=False)
metric = st.sidebar.selectbox(
    "Color by Metric",
    [
        "PriceIndex",
        "mesh_mean_ppsqm",
        "mesh_median_ppsqm",
        "mesh_transaction_count",
        "mesh_avg_age",
        "mesh_avg_area",
    ],
    index=0,
)

show_shap = st.sidebar.checkbox("Show SHAP Panel", value=True)
show_leaderboard = st.sidebar.checkbox("Show Leaderboard", value=True)



# Display leaderboard

if show_leaderboard:
    st.subheader("Model Leaderboard")

    # Load model results
    df_results = pd.read_csv("model_results.csv")

    # Select level (Ward or Mesh)
    levels = df_results["Level"].unique().tolist()[::-1]
    level_choice = st.radio("Select Evaluation Level", levels, horizontal=True)

    # Filter for level
    filtered = df_results[df_results["Level"] == level_choice]

    # Sort by best R² descending
    filtered = filtered.sort_values(by="test_r2", ascending=False)

    # Pretty format numbers
    numeric_cols = ["test_mae", "test_rmse", "test_r2"]
    for col in numeric_cols:
        filtered[col] = filtered[col].apply(lambda x: f"{x:,.3f}" if pd.notnull(x) else "-")

    # Display table
    st.dataframe(
        filtered[["Model", "test_mae", "test_rmse", "test_r2"]],
        hide_index=True,
        width='stretch',
    )

    # Highlight top model
    top_model = filtered.iloc[0]["Model"]
    top_r2 = filtered.iloc[0]["test_r2"]
    st.success(f"🏆 **Top Model:** {top_model} (R² = {top_r2})")


def filter_city(frame: gpd.GeoDataFrame, city_name: str) -> gpd.GeoDataFrame:
    """Filter meshes to Tokyo or Sendai using MunicipalityCode when available, else bbox."""
    if "MunicipalityCode" in frame.columns:
        if city_name == "Tokyo":
            return frame[frame["MunicipalityCode"].isin(TOKYO_WARDS)]
        else:
            return frame[frame["MunicipalityCode"].isin(SENDAI_WARDS)]
    # bbox fallback
    lat_min, lat_max, lon_min, lon_max = BBOXES[city_name]
    return frame[
        (frame["Latitude"] >= lat_min) & (frame["Latitude"] <= lat_max) &
        (frame["Longitude"] >= lon_min) & (frame["Longitude"] <= lon_max)
    ]


# Render one map frame for (city, period, model)
def render_period(city_name: str, period_key: str, model_name: str):
    # Filter base mesh data for that quarter + city
    frame = gdf[gdf["PeriodKey"] == period_key].copy()
    frame = filter_city(frame, city_name)

    # Merge predictions for same model + quarter
    preds = df_pred[
        (df_pred["Model"].str.strip().str.lower() == model_name.strip().lower())
        & (df_pred["PeriodKey"] == period_key)
    ].copy()

    # Clean up Mesh ID formatting on both sides
    if "Mesh250m" in frame.columns:
        frame["Mesh250m"] = (
            frame["Mesh250m"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.extract(r"(\d+)", expand=False)
        )

    if "Mesh250m" in preds.columns:
        preds["Mesh250m"] = (
            preds["Mesh250m"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.extract(r"(\d+)", expand=False)
        )

    # Merge using both Mesh250m + PeriodKey if available
    if "Mesh250m" in preds.columns and "Mesh250m" in frame.columns:
        frame = frame.merge(
            preds[["Mesh250m", "PeriodKey", "Predicted"]],
            on=["Mesh250m", "PeriodKey"],
            how="left"
        )
    if frame.empty:
        st.warning(f"No data for {city_name} ({model_name}) in {period_key}")
        return

    # Format tooltip values (2 dp)
    for col in [
        "PriceIndex", "mesh_median_ppsqm",
        "mesh_mean_ppsqm", "mesh_avg_age",
        "mesh_avg_area", "Predicted"
    ]:
        if col in frame.columns:
            frame[f"{col}_str"] = frame[col].apply(
                lambda x: f"{x:,.2f}" if pd.notnull(x) else "-"
            )

    geojson = json.loads(frame.to_json())
    max_val = frame[metric].max() or 1e-6  # Avoid div-by-zero

    # Green (cheap) → Red (expensive)
    color_expr = f"[255*(properties.{metric}/{max_val}), 255*(1 - properties.{metric}/{max_val}), 0]"

    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        stroked=False,
        filled=True,
        opacity=0.8,
        get_fill_color=color_expr,
        pickable=True,
    )

    vs = pdk.ViewState(**CENTER[city_name])

    tooltip = {
        "html": (
            "<b>Mesh ID:</b> {Mesh250m}<br>"
            "<b>Quarter:</b> {PeriodKey}<br>"
            "<b>Predicted Median ¥/m²:</b> {Predicted_str}<br>"
            "<b>Actual Median ¥/m²:</b> {mesh_median_ppsqm_str}<br>"
            "<b>Mean ¥/m²:</b> {mesh_mean_ppsqm_str}<br>"
            "<b>Avg Age:</b> {mesh_avg_age_str}<br>"
            "<b>Avg Area:</b> {mesh_avg_area_str}<br>"
            "<b>Price Index:</b> {PriceIndex_str}"
        ),
        "style": {"backgroundColor": "white", "color": "black"}
    }

    deck = pdk.Deck(layers=[layer], initial_view_state=vs, tooltip=tooltip)
    st.pydeck_chart(deck)


# Main Layout — Map + Right Panels
map_col, right_col = st.columns([1.6, 1])

st.subheader("Mesh-250 m Dynamic Map")
st.caption("Color by selected metric; hover for mesh-level stats.")
if animate:
    placeholder = st.empty()
    for q in quarters:
        with placeholder.container():
            st.markdown(f"🕒 Quarter: `{q}` — {city} — Color by `{metric}`")
            render_period(city, q, model)
        time.sleep(1.2)
else:
    st.markdown(f"### 🕒 Quarter: `{period}` — {city} — Color by `{metric}`")
    render_period(city, period, model)
left_col, right_col = st.columns([0.7, 1])

with left_col:
    if show_shap:
        model_name = model.strip().lower()
        genre = st.radio(
        "Choose your plot",["Bar Plot","BeeSwarm Plot"],index=None)

        st.subheader(f"{city}: {model} Global Feature importance")
        try:
            if genre == "Bar Plot":
                st.image(f"mesh_{model_name}_val_{city}_bar.png", width='stretch')
            else:
                st.image(f"mesh_{model_name}_val_{city}_beeswarm.png", width='stretch')
        except:
            st.warning(f"SHAP for {city}: {model} is not available in this version. Support will be added in a future update.")
with right_col:
        #  Interactive dependence plot for top 5 features (Mesh only)
        shap_exp_global = load_mesh_shap_explanation(model)
        if shap_exp_global is not None:
            st.markdown("### Feature-level SHAP Dependence (Mesh-250m)")

            # Compute global importance (mean |SHAP|) to get top 5 features
            shap_vals = getattr(shap_exp_global, "values", shap_exp_global)
            abs_mean = np.mean(np.abs(shap_vals), axis=0)
            top5_idx = np.argsort(-abs_mean)[:5]
            top5_features = [shap_exp_global.feature_names[i] for i in top5_idx]

            feature_choice = st.selectbox(
                "Select one of the top 5 features",
                top5_features,
                key="dep_feature",
            )

            interaction_choice = "None"
            # Prepare feature matrix from the SHAP Explanation
            X_df = pd.DataFrame(
                shap_exp_global.data,
                columns=shap_exp_global.feature_names,
            )

            # Draw dependence plot using classic SHAP API
            plt.clf()
            if interaction_choice == "None":
                shap.dependence_plot(
                    feature_choice,
                    shap_vals,
                    X_df,
                    show=False,
                )
            fig_dep = plt.gcf()
            st.pyplot(fig_dep, clear_figure=True)




# Local SHAP Explanations for Mesh-250m
st.markdown("---")
st.subheader("🔍 Local SHAP Explanations — Mesh-250 m")

shap_exp_local = load_mesh_shap_explanation(model)
if shap_exp_local is None:
    st.info("No mesh-level SHAP pickle available for this model yet.")
else:

    shap_vals = getattr(shap_exp_local, "values", shap_exp_local)
    n_obs = shap_vals.shape[0]

    idx = st.slider(
        "Select observation index",
        min_value=0,
        max_value=n_obs - 1,
        value=0,
    )

    # Compute base value and prediction for this observation
    base_value = shap_exp_local.base_values[idx] if np.ndim(shap_exp_local.base_values) > 0 else shap_exp_local.base_values
    contrib = shap_vals[idx, :].sum()
    pred = base_value + contrib
    st.markdown("#### SHAP Force Plot")
    try:
        force_plot = shap.force_plot(
            base_value,
            shap_vals[idx, :],
            shap_exp_local.data[idx, :],
            feature_names=shap_exp_local.feature_names,
        )
        st_shap(force_plot, height=250)
    except Exception as e:
        st.warning(f"Could not render force plot: {e}")


    left_col, right_col2 = st.columns([1, 1])

    #Left: force + waterfall
    with left_col:

        st.markdown("#### SHAP Waterfall Plot")
        try:
            fig_wf, ax_wf = plt.subplots()
            shap.plots.waterfall(shap_exp_local[idx], show=False)
            st.pyplot(fig_wf, clear_figure=True)
        except Exception as e:
            st.warning(f"Could not render waterfall plot: {e}")

    #Right: top-5 explanation
    with right_col2:
        st.markdown("#### SHAP Value Impact for Selected Mesh")

        # st.markdown(
        #     f"**Base prediction:** {base_value:,.0f} units  \n"
        #     f"**Final prediction:** {pred:,.0f} units  \n"
        #     f"**Total SHAP adjustment:** {contrib:+,.0f} units"
        # )

        # Top 5 features by absolute SHAP for this observation
        abs_vals = np.abs(shap_vals[idx, :])
        top5_idx_local = np.argsort(-abs_vals)[:5]
        st.markdown(" ")
        st.markdown(" ")

        st.markdown("##### Top 5 contributing features")

        for j in top5_idx_local:
            feat_name = shap_exp_local.feature_names[j]
            v = shap_vals[idx, j]
            direction = "positive impact, and increases" if v >= 0 else "negative impact, and decreases"
            st.write(
                f"- **{feat_name}** has a {direction} the predicted value by "
                f"${v:+,.2f} ."
            )

