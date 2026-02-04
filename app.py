# ============================================================
# Streamlit Biomarker & Symptom Explorer (Option B: Plotly Hover)
# - Interactive hover tooltips for events + biomarker datapoints
# - Stone main background + dark sidebar contrast
# - Shared date window across ALL plots
#
# IMPORTANT:
#   Add plotly to requirements.txt on GitHub:
#     streamlit
#     pandas
#     numpy
#     matplotlib
#     plotly
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
    "primary": "#405A51",    # Dark green
    "secondary": "#607663",  # Medium green
    "accent": "#A25B4C",     # Reddish accent
    "background": "#F2EEE2", # Stone / neutral
}

# Optional: category -> brand-ish colors (extend as you add categories)
category_colors = {
    "immune": brand_colors["accent"],
    "neuro": brand_colors["primary"],
    "metabolic": brand_colors["secondary"],
}

def style_plotly(fig):
    fig.update_layout(
        template="plotly_white",  # <-- key: stops dark-mode Plotly defaults
        paper_bgcolor=brand_colors["background"],
        plot_bgcolor=brand_colors["background"],
        font=dict(color=brand_colors["primary"]),
        title=dict(font=dict(color=brand_colors["primary"])),
        legend=dict(font=dict(color=brand_colors["primary"])),
        hoverlabel=dict(
            bgcolor=brand_colors["background"],
            font=dict(color=brand_colors["primary"]),
            bordercolor=brand_colors["secondary"],
        ),
        margin=dict(l=20, r=20, t=40, b=40),
    )
    fig.update_xaxes(
        tickfont=dict(color=brand_colors["primary"]),
        titlefont=dict(color=brand_colors["primary"]),
        gridcolor="rgba(64,90,81,0.12)",
        zerolinecolor="rgba(64,90,81,0.12)",
    )
    fig.update_yaxes(
        tickfont=dict(color=brand_colors["primary"]),
        titlefont=dict(color=brand_colors["primary"]),
        gridcolor="rgba(64,90,81,0.12)",
        zerolinecolor="rgba(64,90,81,0.12)",
    )
    return fig


# ----------------------------
# 2) Page Config + Styling
# ----------------------------
st.set_page_config(layout="wide", page_title="Biomarker Dashboard")

st.markdown(
    f"""
    <style>
      /* ---------- MAIN APP (stone background + dark text) ---------- */
      .stApp {{
        background-color: {brand_colors['background']};
        color: {brand_colors['primary']};
      }}

      /* Make main-page text readable (Streamlit can default to white) */
      div[data-testid="stAppViewContainer"] {{
        color: {brand_colors['primary']} !important;
      }}
      div[data-testid="stAppViewContainer"] p,
      div[data-testid="stAppViewContainer"] span,
      div[data-testid="stAppViewContainer"] li,
      div[data-testid="stAppViewContainer"] label {{
        color: {brand_colors['primary']} !important;
      }}

      /* Main headers */
      div[data-testid="stAppViewContainer"] h1,
      div[data-testid="stAppViewContainer"] h2,
      div[data-testid="stAppViewContainer"] h3 {{
        color: {brand_colors['primary']} !important;
        font-weight: 650;
      }}

      /* ---------- SIDEBAR (dark background + readable controls) ---------- */
      section[data-testid="stSidebar"] {{
        background-color: {brand_colors['primary']};
      }}

      /* Sidebar headers */
      section[data-testid="stSidebar"] h1,
      section[data-testid="stSidebar"] h2,
      section[data-testid="stSidebar"] h3 {{
        color: {brand_colors['background']} !important;
        font-weight: 650;
      }}

      /* Sidebar labels (keep white) */
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] p,
      section[data-testid="stSidebar"] span {{
        color: #FFFFFF !important;
      }}

      /* Sidebar expander titles */
      section[data-testid="stSidebar"] summary {{
        color: {brand_colors['background']} !important;
        font-weight: 650;
      }}

      /* Sidebar input boxes: make them light so white text doesn't disappear */
      section[data-testid="stSidebar"] input,
      section[data-testid="stSidebar"] textarea {{
        background-color: {brand_colors['background']} !important;
        color: {brand_colors['primary']} !important;
      }}

      /* Dropdown / multiselect selected text */
      section[data-testid="stSidebar"] [data-baseweb="select"] * {{
        color: {brand_colors['primary']} !important;
      }}
      section[data-testid="stSidebar"] [data-baseweb="select"] {{
        background-color: {brand_colors['background']} !important;
      }}

      /* ---------- PLOTLY: force readable text + stone background ---------- */
      .js-plotly-plot, .plotly, .plot-container {{
        color: {brand_colors['primary']} !important;
      }}
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------------------
# 3) Load Data
# ----------------------------
labs_df = pd.read_csv("lab_data.csv", parse_dates=["date"])
events_df = pd.read_csv("events.csv", parse_dates=["start_date", "end_date"])

# ----------------------------
# 4) Standard Reference Ranges (team-set defaults)
# ----------------------------
default_ref_ranges = {
    "CRP": (0.0, 3.0),
    "IL6": (0.0, 5.0),
    "Ferritin": (20, 300),
    "Cortisol": (5, 20),
    "BDNF": (15, 30),
    "Serotonin": (100, 200),
    "Glucose": (70, 100),
    "Insulin": (2, 15),
}

# ----------------------------
# 5) Sidebar Controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    # Date window (global)
    min_date = labs_df["date"].min().date()
    max_date = labs_df["date"].max().date()
    date_range = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    # Category selection
    all_categories = sorted(labs_df["category"].dropna().unique())
    selected_categories = st.multiselect(
        "Select categories",
        all_categories,
        default=list(all_categories)
    )

    # Biomarker selection based on categories
    available_biomarkers = sorted(
        labs_df[labs_df["category"].isin(selected_categories)]["biomarker"].dropna().unique()
    )

    selected_biomarkers = st.multiselect(
        "Select biomarkers (overlay)",
        available_biomarkers,
        default=available_biomarkers[:6] if len(available_biomarkers) >= 6 else available_biomarkers
    )

    overlay_option = st.selectbox(
        "Overlay option",
        ["All biomarkers", "Biomarkers within selected categories", "Individual categories"]
    )

    # Reference ranges
    with st.expander("Reference Ranges"):
        show_standard_ref = st.checkbox("Show standard ranges (when possible)", value=True)
        show_custom_ref = st.checkbox("Enable custom ranges", value=False)

        custom_ref_ranges = {}
        if show_custom_ref:
            for bm in selected_biomarkers:
                if bm in default_ref_ranges:
                    low, high = default_ref_ranges[bm]
                    custom_ref_ranges[bm] = st.slider(
                        f"{bm} custom range",
                        min_value=float(low * 0.5),
                        max_value=float(high * 2.0),
                        value=(float(low), float(high))
                    )
                else:
                    custom_ref_ranges[bm] = (None, None)
        else:
            for bm in selected_biomarkers:
                custom_ref_ranges[bm] = (None, None)

    # Out-of-range filter
    show_out_of_range = st.checkbox("Show only out-of-range values?", value=False)

    # Individual plots selection
    st.subheader("Individual Biomarker Plots")
    individual_bio_selected = st.multiselect(
        "Select biomarkers for separate plots",
        options=available_biomarkers,
        default=[]
    )

# ----------------------------
# 6) Filter Data + Compute Out-of-Range
# ----------------------------
start_win = pd.Timestamp(date_range[0])
end_win = pd.Timestamp(date_range[1])

filtered_labs = labs_df[
    (labs_df["date"] >= start_win) &
    (labs_df["date"] <= end_win) &
    (labs_df["category"].isin(selected_categories)) &
    (labs_df["biomarker"].isin(selected_biomarkers))
].copy()

def range_for_biomarker(bm: str):
    """Prefer custom range (if enabled and set), else standard range (if present)."""
    if show_custom_ref:
        low, high = custom_ref_ranges.get(bm, (None, None))
        if low is not None and high is not None:
            return (low, high)
    if bm in default_ref_ranges:
        return default_ref_ranges[bm]
    return (None, None)

def is_out_of_range(row):
    low, high = range_for_biomarker(row["biomarker"])
    if low is None or high is None:
        return False  # unknown range => don't label as out-of-range
    return (row["value"] < low) or (row["value"] > high)

filtered_labs["out_of_range"] = filtered_labs.apply(is_out_of_range, axis=1)

if show_out_of_range:
    filtered_labs = filtered_labs[filtered_labs["out_of_range"]].copy()

# Helpful hover fields
filtered_labs["date_str"] = filtered_labs["date"].dt.strftime("%Y-%m-%d")
filtered_labs["value_str"] = filtered_labs["value"].astype(float).map(lambda x: f"{x:.4g}")

# ----------------------------
# 7) Events Timeline (Plotly hover)
# ----------------------------
st.subheader("🩺 Symptoms / Infections Timeline")

events_clean = events_df.copy()
events_clean["start_date"] = pd.to_datetime(events_clean["start_date"], errors="coerce")
events_clean["end_date"] = pd.to_datetime(events_clean["end_date"], errors="coerce")
events_clean["end_date"] = events_clean["end_date"].fillna(events_clean["start_date"])

events_clean["type_norm"] = (
    events_clean["type"]
    .astype(str)
    .str.strip()
    .str.lower()
)

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
        hover_data={
            "start_date": True,
            "end_date": True,
            "type": True,
            "type_label": False,
            "is_infection": False,
            "type_norm": False,
        },
    )

    fig_evt.update_yaxes(autorange="reversed", title=None)
    fig_evt.update_xaxes(range=[start_win, end_win], tickformat="%Y-%m-%d")
    fig_evt.update_layout(
        height=max(320, 36 * len(events_filtered)),
        margin=dict(l=15, r=15, t=40, b=40),
        plot_bgcolor=brand_colors["background"],
        paper_bgcolor=brand_colors["background"],
        legend_title_text="",
        font=dict(color=brand_colors["primary"]),
    )

    st.plotly_chart(fig_evt, use_container_width=True)

# ----------------------------
# 8) Biomarker Trends (Plotly hover)
# ----------------------------
st.subheader("🧪 Biomarker Trends")

if filtered_labs.empty:
    st.warning("No lab data in the selected window / filters.")
else:
    plot_df = filtered_labs.sort_values(["date", "biomarker"]).copy()

    # How to color/group traces:
    # - "All biomarkers": color by biomarker
    # - "Biomarkers within selected categories": color by category (trace still separated by biomarker)
    # - "Individual categories": color by category (same as above; labels include category)
    if overlay_option == "All biomarkers":
        color_col = "biomarker"
        line_group_col = "biomarker"
        hover_cols = ["biomarker", "category", "unit", "draw_id", "date_str", "value_str", "out_of_range"]
    else:
        color_col = "category"
        line_group_col = "biomarker"
        hover_cols = ["biomarker", "category", "unit", "draw_id", "date_str", "value_str", "out_of_range"]

    # Base line chart
    fig_bio = px.line(
        plot_df,
        x="date",
        y="value",
        color=color_col,
        line_group=line_group_col,
        markers=True,
        hover_data={c: True for c in hover_cols},
    )

    # Brand styling
    fig_bio.update_xaxes(range=[start_win, end_win], tickformat="%Y-%m-%d")
    fig_bio.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=40, b=40),
        plot_bgcolor=brand_colors["background"],
        paper_bgcolor=brand_colors["background"],
        legend_title_text="",
        font=dict(color=brand_colors["primary"]),
    )

    # Apply category colors if we are coloring by category
    if color_col == "category":
        # Plotly discrete color map expects exact category labels
        fig_bio.update_traces()  # no-op; keeps structure intact
        fig_bio.for_each_trace(
            lambda tr: tr.update(
                line=dict(color=category_colors.get(tr.name, None)),
                marker=dict(color=category_colors.get(tr.name, None))
            ) if tr.name in category_colors else None
        )

    # If "All biomarkers", we *don't* force a small palette—Plotly will assign distinct colors.
    # (This helps with many biomarkers; otherwise colors will repeat.)

    # ---- Out-of-range highlight layer (accent X markers) ----
    oor = plot_df[plot_df["out_of_range"]].copy()
    if not oor.empty:
        fig_bio.add_trace(
            go.Scatter(
                x=oor["date"],
                y=oor["value"],
                mode="markers",
                name="Out of range",
                marker=dict(symbol="x", size=10, color=brand_colors["accent"]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Category: %{customdata[1]}<br>"
                    "Value: %{customdata[2]} %{customdata[3]}<br>"
                    "Date: %{customdata[4]}<br>"
                    "Draw: %{customdata[5]}<br>"
                    "<extra></extra>"
                ),
                customdata=np.stack(
                    [
                        oor["biomarker"].astype(str),
                        oor["category"].astype(str),
                        oor["value_str"].astype(str),
                        oor["unit"].astype(str),
                        oor["date_str"].astype(str),
                        oor.get("draw_id", pd.Series([""] * len(oor))).astype(str),
                    ],
                    axis=1,
                ),
            )
        )

    # ---- Reference range band (only when ONE biomarker is selected) ----
    # Reason: Different biomarkers have different ranges; a single shared y-axis band would be misleading.
    only_one = (plot_df["biomarker"].nunique() == 1)
    if only_one and show_standard_ref:
        bm = plot_df["biomarker"].iloc[0]
        low, high = range_for_biomarker(bm)
        if low is not None and high is not None:
            fig_bio.add_hrect(
                y0=low,
                y1=high,
                fillcolor=brand_colors["secondary"],
                opacity=0.12,
                line_width=0,
                annotation_text="Reference range",
                annotation_position="top left",
            )

    st.plotly_chart(fig_bio, use_container_width=True)

    if (plot_df["biomarker"].nunique() > 1) and show_standard_ref:
        st.caption(
            "Note: Reference range shading is shown only when a single biomarker is selected "
            "(ranges differ across biomarkers, so a single shared band would be misleading)."
        )

# ----------------------------
# 9) Individual Biomarker Plots (Plotly hover + per-biomarker reference band)
# ----------------------------
if individual_bio_selected:
    st.subheader("📊 Individual Biomarker Plots")

    for bm in individual_bio_selected:
        bm_df = labs_df[
            (labs_df["biomarker"] == bm) &
            (labs_df["category"].isin(selected_categories)) &
            (labs_df["date"] >= start_win) &
            (labs_df["date"] <= end_win)
        ].copy()

        if bm_df.empty:
            continue

        bm_df = bm_df.sort_values("date")
        bm_df["date_str"] = bm_df["date"].dt.strftime("%Y-%m-%d")
        bm_df["value_str"] = bm_df["value"].astype(float).map(lambda x: f"{x:.4g}")
        bm_df["out_of_range"] = bm_df.apply(is_out_of_range, axis=1)

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
        fig_one.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=40, b=40),
            plot_bgcolor=brand_colors["background"],
            paper_bgcolor=brand_colors["background"],
            showlegend=False,
            font=dict(color=brand_colors["primary"]),
            title=dict(text=f"{bm}", x=0.01),
        )

        # Brand line color
        fig_one.update_traces(line=dict(color=brand_colors["primary"]), marker=dict(color=brand_colors["primary"]))

        # Out-of-range X markers
        oor_one = bm_df[bm_df["out_of_range"]]
        if not oor_one.empty:
            fig_one.add_trace(
                go.Scatter(
                    x=oor_one["date"],
                    y=oor_one["value"],
                    mode="markers",
                    name="Out of range",
                    marker=dict(symbol="x", size=10, color=brand_colors["accent"]),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Value: %{customdata[1]} %{customdata[2]}<br>"
                        "Date: %{customdata[3]}<br>"
                        "Draw: %{customdata[4]}<br>"
                        "<extra></extra>"
                    ),
                    customdata=np.stack(
                        [
                            oor_one["biomarker"].astype(str),
                            oor_one["value_str"].astype(str),
                            oor_one["unit"].astype(str),
                            oor_one["date_str"].astype(str),
                            oor_one.get("draw_id", pd.Series([""] * len(oor_one))).astype(str),
                        ],
                        axis=1,
                    ),
                )
            )

        # Reference range band (per biomarker)
        if show_standard_ref:
            low, high = range_for_biomarker(bm)
            if low is not None and high is not None:
                fig_one.add_hrect(
                    y0=low,
                    y1=high,
                    fillcolor=brand_colors["secondary"],
                    opacity=0.12,
                    line_width=0
                )

        st.plotly_chart(fig_one, use_container_width=True)

