# ============================================================
# Streamlit Biomarker & Symptom Explorer (Plotly Hover + Brand)
# UPDATED (fixes title/legend overlap + readable sidebar text):
#   - Global Plotly layout: title BELOW legend (no overlap)
#   - Legend placed at top, outside plot area
#   - Extra top margin + consistent spacing on every figure
#   - Sidebar dark + text visible; main page stone + text dark
#   - Unit-split plots (never mix units on same y-axis)
#
# Files expected:
#   - lab_data.csv
#   - events.csv
#   - reference_ranges.csv
#
# requirements.txt:
#   streamlit
#   pandas
#   numpy
#   plotly
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# 1) Brand Colors
# ----------------------------
brand_colors = {
    "primary": "#405A51",
    "secondary": "#607663",
    "accent": "#A25B4C",
    "background": "#F2EEE2",
}

category_colors = {
    "immune": brand_colors["accent"],
    "neuro": brand_colors["primary"],
    "metabolic": brand_colors["secondary"],
}

# ----------------------------
# 2) Page Config + Styling
# ----------------------------
st.set_page_config(layout="wide", page_title="Biomarker Dashboard")

st.markdown(
    f"<h1 style='font-family:sans-serif; color:{brand_colors['primary']}; margin-bottom:0;'>🧬 Biomarker & Symptom Explorer</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# Main stone background + dark text; dark sidebar with light text
st.markdown(
    f"""
    <style>
      .stApp {{
        background-color: {brand_colors['background']};
      }}

      /* MAIN PAGE TEXT */
      .stMarkdown, .stText, .stCaption, p, span {{
        color: {brand_colors['primary']} !important;
      }}
      h2, h3 {{
        color: {brand_colors['primary']} !important;
      }}

      /* SIDEBAR */
      section[data-testid="stSidebar"] {{
        background-color: {brand_colors['primary']};
      }}
      section[data-testid="stSidebar"] h1,
      section[data-testid="stSidebar"] h2,
      section[data-testid="stSidebar"] h3,
      section[data-testid="stSidebar"] p,
      section[data-testid="stSidebar"] span,
      section[data-testid="stSidebar"] label {{
        color: {brand_colors['background']} !important;
      }}
      section[data-testid="stSidebar"] summary {{
        color: {brand_colors['background']} !important;
        font-weight: 650;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# 3) Plotly Styling Helper (NO OVERLAP)
# ----------------------------
def style_plotly(fig, title_text=None):
    """
    Fixes title/legend overlap by:
      - Legend at very top (outside plotting region)
      - Title below legend
      - Big top margin
    """
    if title_text is not None:
        fig.update_layout(title=dict(text=title_text))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=brand_colors["background"],
        plot_bgcolor=brand_colors["background"],
        font=dict(color=brand_colors["primary"]),
        # Put legend ABOVE title
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.16,          # higher so it has its own space
            xanchor="left",
            x=0,
            font=dict(color=brand_colors["primary"]),
            title_text=""
        ),
        # Put title BELOW legend, aligned left
        title=dict(
            font=dict(color=brand_colors["primary"], size=16),
            x=0,
            xanchor="left",
            y=1.08,          # below legend but still in top area
            yanchor="top"
        ),
        hoverlabel=dict(
            bgcolor=brand_colors["background"],
            font=dict(color=brand_colors["primary"]),
            bordercolor=brand_colors["secondary"],
        ),
        # Extra top margin gives the legend+title space
        margin=dict(l=30, r=20, t=130, b=55),
    )

    fig.update_xaxes(
        tickfont=dict(color=brand_colors["primary"]),
        title_font=dict(color=brand_colors["primary"]),
        gridcolor="rgba(64,90,81,0.12)",
        zerolinecolor="rgba(64,90,81,0.12)",
    )
    fig.update_yaxes(
        tickfont=dict(color=brand_colors["primary"]),
        title_font=dict(color=brand_colors["primary"]),
        gridcolor="rgba(64,90,81,0.12)",
        zerolinecolor="rgba(64,90,81,0.12)",
    )
    return fig

# ----------------------------
# 4) Load Data
# ----------------------------
labs_df = pd.read_csv("lab_data.csv", parse_dates=["date"])
events_df = pd.read_csv("events.csv", parse_dates=["start_date", "end_date"])

# ----------------------------
# 5) Normalize Units (important for unit-split)
# ----------------------------
unit_map = {
    "pg/ml": "pg/mL",
    "pg/ ml": "pg/mL",
    "pg per ml": "pg/mL",
    "ng/ml": "ng/mL",
    "ng/ ml": "ng/mL",
    "mg/dl": "mg/dL",
    "ug/dl": "ug/dL",
    "µg/dl": "ug/dL",
    "uiu/ml": "uIU/mL",
    "miu/l": "mIU/L",
    "ratio": "ratio",
    "npq": "NPQ",
}

def normalize_unit(u):
    if pd.isna(u) or str(u).strip() == "":
        return "Unknown"
    key = str(u).strip()
    return unit_map.get(key.lower(), key)

if "unit" not in labs_df.columns:
    labs_df["unit"] = "Unknown"
labs_df["unit"] = labs_df["unit"].apply(normalize_unit)

# ----------------------------
# 6) Standard reference ranges from reference_ranges.csv
# ----------------------------
ref_df = pd.read_csv("reference_ranges.csv")
ref_df["biomarker"] = ref_df["biomarker"].astype(str).str.strip()
ref_df["low"] = pd.to_numeric(ref_df["low"], errors="coerce")
ref_df["high"] = pd.to_numeric(ref_df["high"], errors="coerce")
default_ref_ranges = dict(zip(ref_df["biomarker"], list(zip(ref_df["low"], ref_df["high"]))))

# ----------------------------
# 7) Sidebar Controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    min_date = labs_df["date"].min().date()
    max_date = labs_df["date"].max().date()
    date_range = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    all_categories = sorted(labs_df["category"].dropna().unique())
    selected_categories = st.multiselect(
        "Select categories",
        all_categories,
        default=list(all_categories)
    )

    available_biomarker_df = labs_df[labs_df["category"].isin(selected_categories)]
    available_biomarkers = sorted(available_biomarker_df["biomarker"].dropna().unique())

    selected_biomarkers = st.multiselect(
        "Select biomarkers (overlay)",
        available_biomarkers,
        default=available_biomarkers[:10] if len(available_biomarkers) >= 10 else available_biomarkers
    )

    available_units = sorted(available_biomarker_df["unit"].dropna().unique())
    selected_units = st.multiselect(
        "Units to display",
        options=available_units,
        default=list(available_units)
    )

    with st.expander("Reference Ranges"):
        show_standard_ref = st.checkbox("Show standard ranges (reference_ranges.csv)", value=True)
        show_custom_ref = st.checkbox("Enable custom ranges (sliders)", value=False)

        custom_scope = st.radio(
            "Custom ranges apply to:",
            ["Overlay biomarkers only", "All biomarkers in selected categories"],
            index=0
        )
        custom_biomarker_list = selected_biomarkers if custom_scope == "Overlay biomarkers only" else available_biomarkers

        custom_ref_ranges = {}
        if show_custom_ref:
            if len(custom_biomarker_list) > 40:
                st.warning("Large panel detected. Choose a subset for custom sliders to keep the UI fast.")
                slider_biomarkers = st.multiselect(
                    "Choose biomarkers to show sliders for",
                    options=custom_biomarker_list,
                    default=custom_biomarker_list[:15]
                )
            else:
                slider_biomarkers = custom_biomarker_list

            for bm in slider_biomarkers:
                low, high = default_ref_ranges.get(bm, (None, None))

                if low is None or high is None or pd.isna(low) or pd.isna(high):
                    bm_vals = pd.to_numeric(labs_df[labs_df["biomarker"] == bm]["value"], errors="coerce").dropna()
                    if len(bm_vals) > 0:
                        vmin, vmax = float(bm_vals.min()), float(bm_vals.max())
                        pad = (vmax - vmin) * 0.25 if vmax > vmin else max(1.0, abs(vmin) * 0.25)
                        low, high = vmin - pad, vmax + pad
                    else:
                        low, high = 0.0, 1.0

                min_bound = float(low) - abs(float(low)) * 1.0 - 1.0
                max_bound = float(high) + abs(float(high)) * 1.0 + 1.0

                custom_ref_ranges[bm] = st.slider(
                    f"{bm} custom range",
                    min_value=min_bound,
                    max_value=max_bound,
                    value=(float(low), float(high))
                )

    show_out_of_range = st.checkbox("Show only out-of-range values?", value=False)

    st.subheader("Category Plots")
    show_category_plots = st.checkbox("Show separate plots for each selected category (unit-split)", value=True)

    st.subheader("Individual Biomarker Plots")
    individual_bio_selected = st.multiselect(
        "Select biomarkers for separate plots",
        options=available_biomarkers,
        default=[]
    )

# ----------------------------
# 8) Date window + range helpers
# ----------------------------
start_win = pd.Timestamp(date_range[0])
end_win = pd.Timestamp(date_range[1])

def range_for_biomarker(bm: str):
    if show_custom_ref and (bm in custom_ref_ranges):
        low, high = custom_ref_ranges.get(bm, (None, None))
        if low is not None and high is not None:
            return (low, high)

    if show_standard_ref and (bm in default_ref_ranges):
        low, high = default_ref_ranges.get(bm, (None, None))
        if pd.notna(low) and pd.notna(high):
            return (float(low), float(high))

    return (None, None)

def is_out_of_range(row):
    low, high = range_for_biomarker(row["biomarker"])
    if low is None or high is None:
        return False
    try:
        v = float(row["value"])
    except Exception:
        return False
    return (v < low) or (v > high)

# ----------------------------
# 9) Filter labs (overlay selection) + apply unit filter
# ----------------------------
filtered_labs = labs_df[
    (labs_df["date"] >= start_win) &
    (labs_df["date"] <= end_win) &
    (labs_df["category"].isin(selected_categories)) &
    (labs_df["biomarker"].isin(selected_biomarkers)) &
    (labs_df["unit"].isin(selected_units))
].copy()

filtered_labs["value"] = pd.to_numeric(filtered_labs["value"], errors="coerce")
filtered_labs = filtered_labs.dropna(subset=["value"])

filtered_labs["out_of_range"] = filtered_labs.apply(is_out_of_range, axis=1)
if show_out_of_range:
    filtered_labs = filtered_labs[filtered_labs["out_of_range"]].copy()

filtered_labs["date_str"] = filtered_labs["date"].dt.strftime("%Y-%m-%d")
filtered_labs["value_str"] = filtered_labs["value"].map(lambda x: f"{x:.4g}")

# ----------------------------
# 10) Events Timeline
# ----------------------------
st.subheader("🩺 Symptoms / Infections Timeline")

events_clean = events_df.copy()
events_clean["start_date"] = pd.to_datetime(events_clean["start_date"], errors="coerce")
events_clean["end_date"] = pd.to_datetime(events_clean["end_date"], errors="coerce")
events_clean["end_date"] = events_clean["end_date"].fillna(events_clean["start_date"])

events_clean["type_norm"] = events_clean["type"].astype(str).str.strip().str.lower()
infection_keywords = ["infection", "covid", "flu", "uri", "viral", "bacterial", "virus"]
events_clean["is_infection"] = events_clean["type_norm"].apply(lambda t: any(k in t for k in infection_keywords))
events_clean["type_label"] = np.where(events_clean["is_infection"], "Infection", "Symptom")

events_filtered = events_clean[
    (events_clean["end_date"] >= start_win) &
    (events_clean["start_date"] <= end_win)
].dropna(subset=["start_date", "end_date"]).copy()

if events_filtered.empty:
    st.info("No events in the selected date window.")
else:
    fig_evt = px.timeline(
        events_filtered,
        x_start="start_date",
        x_end="end_date",
        y="name",
        color="type_label",
        color_discrete_map={
            "Symptom": brand_colors["secondary"],
            "Infection": brand_colors["accent"],
        },
        hover_data={"start_date": True, "end_date": True, "type": True},
    )
    fig_evt.update_yaxes(autorange="reversed", title=None)
    fig_evt.update_xaxes(range=[start_win, end_win], tickformat="%Y-%m-%d")
    fig_evt.update_layout(height=max(420, 40 * len(events_filtered)))
    fig_evt = style_plotly(fig_evt, title_text="Symptoms / Infections (hover for details)")
    st.plotly_chart(fig_evt, use_container_width=True)

# ----------------------------
# 11) Biomarker Overlay — unit-split
# ----------------------------
st.subheader("🧪 Biomarker Trends (Overlay — split by unit)")

if filtered_labs.empty:
    st.warning("No lab data in the selected window / filters.")
else:
    plot_df = filtered_labs.sort_values(["date", "biomarker"]).copy()
    for u in sorted(plot_df["unit"].unique()):
        df_u = plot_df[plot_df["unit"] == u].copy()
        if df_u.empty:
            continue

        fig_bio = px.line(
            df_u,
            x="date",
            y="value",
            color="biomarker",
            line_group="biomarker",
            markers=True,
            hover_data={
                "biomarker": True,
                "category": True,
                "unit": True,
                "draw_id": True,
                "date_str": True,
                "value_str": True,
                "out_of_range": True,
            },
        )
        fig_bio.update_xaxes(range=[start_win, end_win], tickformat="%Y-%m-%d")
        fig_bio.update_layout(height=560, yaxis_title=f"Value ({u})")

        # Out-of-range markers
        oor = df_u[df_u["out_of_range"]].copy()
        if not oor.empty:
            fig_bio.add_trace(
                go.Scatter(
                    x=oor["date"],
                    y=oor["value"],
                    mode="markers",
                    name="Out of range",
                    marker=dict(symbol="x", size=10, color=brand_colors["accent"]),
                )
            )

        fig_bio = style_plotly(fig_bio, title_text=f"Overlay — Unit: {u}")
        st.plotly_chart(fig_bio, use_container_width=True)

# ----------------------------
# 12) Category Plots — unit-split
# ----------------------------
if show_category_plots:
    st.subheader("🧩 Category Plots (Unit-split)")

    cat_df_all = labs_df[
        (labs_df["date"] >= start_win) &
        (labs_df["date"] <= end_win) &
        (labs_df["category"].isin(selected_categories)) &
        (labs_df["unit"].isin(selected_units))
    ].copy()

    cat_df_all["value"] = pd.to_numeric(cat_df_all["value"], errors="coerce")
    cat_df_all = cat_df_all.dropna(subset=["value"])
    cat_df_all["date_str"] = cat_df_all["date"].dt.strftime("%Y-%m-%d")
    cat_df_all["value_str"] = cat_df_all["value"].map(lambda x: f"{x:.4g}")
    cat_df_all["out_of_range"] = cat_df_all.apply(is_out_of_range, axis=1)

    if show_out_of_range:
        cat_df_all = cat_df_all[cat_df_all["out_of_range"]].copy()

    if cat_df_all.empty:
        st.info("No category data in the selected window / filters.")
    else:
        for cat in selected_categories:
            cat_df = cat_df_all[cat_df_all["category"] == cat].copy()
            if cat_df.empty:
                continue

            st.markdown(f"### {cat}")

            for u in sorted(cat_df["unit"].unique()):
                df_u = cat_df[cat_df["unit"] == u].copy()
                if df_u.empty:
                    continue

                fig_cat = px.line(
                    df_u.sort_values(["date", "biomarker"]),
                    x="date",
                    y="value",
                    color="biomarker",
                    line_group="biomarker",
                    markers=True,
                    hover_data={
                        "biomarker": True,
                        "category": True,
                        "unit": True,
                        "draw_id": True,
                        "date_str": True,
                        "value_str": True,
                        "out_of_range": True,
                    },
                )
                fig_cat.update_xaxes(range=[start_win, end_win], tickformat="%Y-%m-%d")
                fig_cat.update_layout(height=480, yaxis_title=f"Value ({u})")

                oor = df_u[df_u["out_of_range"]].copy()
                if not oor.empty:
                    fig_cat.add_trace(
                        go.Scatter(
                            x=oor["date"],
                            y=oor["value"],
                            mode="markers",
                            name="Out of range",
                            marker=dict(symbol="x", size=10, color=brand_colors["accent"]),
                        )
                    )

                fig_cat = style_plotly(fig_cat, title_text=f"{cat} — Unit: {u}")
                st.plotly_chart(fig_cat, use_container_width=True)

# ----------------------------
# 13) Individual Biomarker Plots
# ----------------------------
if individual_bio_selected:
    st.subheader("📊 Individual Biomarker Plots")

    for bm in individual_bio_selected:
        bm_df = labs_df[
            (labs_df["biomarker"] == bm) &
            (labs_df["category"].isin(selected_categories)) &
            (labs_df["date"] >= start_win) &
            (labs_df["date"] <= end_win) &
            (labs_df["unit"].isin(selected_units))
        ].copy()

        bm_df["value"] = pd.to_numeric(bm_df["value"], errors="coerce")
        bm_df = bm_df.dropna(subset=["value"])
        if bm_df.empty:
            continue

        bm_df = bm_df.sort_values("date")
        bm_df["date_str"] = bm_df["date"].dt.strftime("%Y-%m-%d")
        bm_df["value_str"] = bm_df["value"].map(lambda x: f"{x:.4g}")
        bm_df["out_of_range"] = bm_df.apply(is_out_of_range, axis=1)
        unit_here = bm_df["unit"].iloc[0] if "unit" in bm_df.columns else "Unknown"

        fig_one = px.line(
            bm_df,
            x="date",
            y="value",
            markers=True,
            hover_data={
                "biomarker": True,
                "category": True,
                "unit": True,
                "draw_id": True,
                "date_str": True,
                "value_str": True,
                "out_of_range": True,
            },
        )
        fig_one.update_xaxes(range=[start_win, end_win], tickformat="%Y-%m-%d")
        fig_one.update_layout(height=360, showlegend=False, yaxis_title=f"Value ({unit_here})")

        low, high = range_for_biomarker(bm)
        if low is not None and high is not None:
            fig_one.add_hrect(
                y0=low,
                y1=high,
                fillcolor=brand_colors["secondary"],
                opacity=0.12,
                line_width=0
            )

        oor_one = bm_df[bm_df["out_of_range"]]
        if not oor_one.empty:
            fig_one.add_trace(
                go.Scatter(
                    x=oor_one["date"],
                    y=oor_one["value"],
                    mode="markers",
                    marker=dict(symbol="x", size=10, color=brand_colors["accent"]),
                    name="Out of range",
                )
            )

        fig_one = style_plotly(fig_one, title_text=f"{bm} ({unit_here})")
        st.plotly_chart(fig_one, use_container_width=True)


