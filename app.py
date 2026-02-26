# ============================================================
# Streamlit Biomarker & Symptom Explorer (Plotly Hover + Brand)
# FULL app.py (legend collision FIXED)
#
# Key fixes:
#   ✅ Dynamic legend placement:
#       - small number of traces -> top horizontal legend
#       - large number of traces -> right vertical legend (outside plot)
#       - optional auto-hide legend when huge
#   ✅ Brand styling: stone main background + dark sidebar + readable text
#   ✅ Unit-split plots (never mix units on the same y-axis)
#   ✅ Standard reference ranges from reference_ranges.csv
#   ✅ Custom reference ranges via sliders (optional)
#   ✅ Hover tooltips for events and biomarkers
#
# Files expected in repo:
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

# ----------------------------
# 2) Page Config + Styling
# ----------------------------
st.set_page_config(layout="wide", page_title="Biomarker Dashboard")

st.markdown(
    f"<h1 style='font-family:sans-serif; color:{brand_colors['primary']}; margin-bottom:0;'>🧬 Biomarker & Symptom Explorer</h1>",
    unsafe_allow_html=True,
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
      h2, h3, h4 {{
        color: {brand_colors['primary']} !important;
      }}

      /* SIDEBAR */
      section[data-testid="stSidebar"] {{
        background-color: {brand_colors['primary']};
      }}
      section[data-testid="stSidebar"] h1,
      section[data-testid="stSidebar"] h2,
      section[data-testid="stSidebar"] h3,
      section[data-testid="stSidebar"] h4,
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
    unsafe_allow_html=True,
)

# ----------------------------
# 3) Plotly Styling Helper (dynamic legend to prevent collisions)
# ----------------------------
def style_plotly(fig, title_text=None, max_top_legend_items=10, hide_legend_over=80):
    """
    Fix legend collisions by switching legend layout based on number of traces.
    - <= max_top_legend_items: top horizontal legend
    - >  max_top_legend_items: right-side vertical legend (outside plot)
    - >  hide_legend_over: hide legend entirely (hover + sidebar selections are enough)
    """
    if title_text is not None:
        fig.update_layout(title_text=title_text)

    n_traces = len(getattr(fig, "data", []))

    # Base styling
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=brand_colors["background"],
        plot_bgcolor=brand_colors["background"],
        font=dict(color=brand_colors["primary"]),
        title=dict(
            x=0,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(size=16, color=brand_colors["primary"]),
        ),
        hoverlabel=dict(
            bgcolor=brand_colors["background"],
            font=dict(color=brand_colors["primary"]),
            bordercolor=brand_colors["secondary"],
        ),
    )

    # Optional: hide legend completely when too many traces
    if hide_legend_over is not None and n_traces > hide_legend_over:
        fig.update_layout(showlegend=False, margin=dict(l=40, r=20, t=90, b=60))
    else:
        if n_traces <= max_top_legend_items:
            # Small legend: top horizontal
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color=brand_colors["primary"]),
                ),
                margin=dict(l=40, r=20, t=120, b=60),
            )
        else:
            # Big legend: right vertical, outside plot
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=1.02,  # outside plot area
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color=brand_colors["primary"]),
                    itemsizing="constant",
                ),
                margin=dict(l=40, r=300, t=90, b=60),  # reserve space for legend
            )

    # Axes styling
    fig.update_xaxes(
        tickfont=dict(color=brand_colors["primary"]),
        title_font=dict(color=brand_colors["primary"]),
        gridcolor="rgba(64,90,81,0.12)",
        zerolinecolor="rgba(64,90,81,0.12)",
        tickformat="%Y-%m-%d",
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
ref_df = pd.read_csv("reference_ranges.csv")

# ----------------------------
# 5) Normalize Units
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
# 6) Standard Reference Ranges
# ----------------------------
ref_df["biomarker"] = ref_df["biomarker"].astype(str).str.strip()
ref_df["low"] = pd.to_numeric(ref_df["low"], errors="coerce")
ref_df["high"] = pd.to_numeric(ref_df["high"], errors="coerce")
default_ref_ranges = dict(zip(ref_df["biomarker"], list(zip(ref_df["low"], ref_df["high"]))))

# ----------------------------
# 7) Sidebar Controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    # Date range
    min_date = labs_df["date"].min().date()
    max_date = labs_df["date"].max().date()
    date_range = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )

    # Categories
    all_categories = sorted(labs_df["category"].dropna().unique())
    selected_categories = st.multiselect(
        "Select categories",
        all_categories,
        default=list(all_categories),
    )

    # Biomarkers available in selected categories
    available_biomarker_df = labs_df[labs_df["category"].isin(selected_categories)]
    available_biomarkers = sorted(available_biomarker_df["biomarker"].dropna().unique())

    selected_biomarkers = st.multiselect(
        "Select biomarkers (overlay)",
        available_biomarkers,
        default=available_biomarkers[:10] if len(available_biomarkers) >= 10 else available_biomarkers,
    )

    # Units
    available_units = sorted(available_biomarker_df["unit"].dropna().unique())
    selected_units = st.multiselect(
        "Units to display",
        options=available_units,
        default=list(available_units),
    )

    with st.expander("Reference Ranges"):
        show_standard_ref = st.checkbox("Show standard ranges (reference_ranges.csv)", value=True)
        show_custom_ref = st.checkbox("Enable custom ranges (sliders)", value=False)

        custom_scope = st.radio(
            "Custom ranges apply to:",
            ["Overlay biomarkers only", "All biomarkers in selected categories"],
            index=0,
        )
        custom_biomarker_list = selected_biomarkers if custom_scope == "Overlay biomarkers only" else available_biomarkers

        custom_ref_ranges = {}
        if show_custom_ref:
            # prevent sidebar blow-up for huge panels
            if len(custom_biomarker_list) > 40:
                st.warning("Large panel detected. Choose a subset for custom sliders to keep the UI fast.")
                slider_biomarkers = st.multiselect(
                    "Choose biomarkers to show sliders for",
                    options=custom_biomarker_list,
                    default=custom_biomarker_list[:15],
                )
            else:
                slider_biomarkers = custom_biomarker_list

            for bm in slider_biomarkers:
                low, high = default_ref_ranges.get(bm, (None, None))

                # fallback if no standard ref range exists
                if low is None or high is None or pd.isna(low) or pd.isna(high):
                    bm_vals = pd.to_numeric(
                        labs_df[labs_df["biomarker"] == bm]["value"],
                        errors="coerce"
                    ).dropna()
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
                    value=(float(low), float(high)),
                )

    show_out_of_range = st.checkbox("Show only out-of-range values?", value=False)

    st.subheader("Category Plots")
    show_category_plots = st.checkbox("Show separate plots for each selected category (unit-split)", value=True)

    st.subheader("Individual Biomarker Plots")
    individual_bio_selected = st.multiselect(
        "Select biomarkers for separate plots",
        options=available_biomarkers,
        default=[],
    )

    st.subheader("Legend Behavior")
    max_top_legend_items = st.number_input("Top legend if <= this many lines", value=10, min_value=0, step=1)
    hide_legend_over = st.number_input("Hide legend if > this many lines", value=80, min_value=0, step=5)

# ----------------------------
# Tabs (Explorer + Event Lens)
# ----------------------------
tab_explorer, tab_event_lens = st.tabs(["📈 Explorer", "🎯 Event Lens"])

# ----------------------------
# 8) Date Window + Range Helpers
# ----------------------------
start_win = pd.Timestamp(date_range[0])
end_win = pd.Timestamp(date_range[1])

def range_for_biomarker(bm: str):
    if show_custom_ref and (bm in custom_ref_ranges):
        low, high = custom_ref_ranges.get(bm, (None, None))
        if low is not None and high is not None:
            return (float(low), float(high))

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
# 9) Filter Labs (overlay selection) + units
# ----------------------------
filtered_labs = labs_df[
    (labs_df["date"] >= start_win)
    & (labs_df["date"] <= end_win)
    & (labs_df["category"].isin(selected_categories))
    & (labs_df["biomarker"].isin(selected_biomarkers))
    & (labs_df["unit"].isin(selected_units))
].copy()

filtered_labs["value"] = pd.to_numeric(filtered_labs["value"], errors="coerce")
filtered_labs = filtered_labs.dropna(subset=["value"])

filtered_labs["out_of_range"] = filtered_labs.apply(is_out_of_range, axis=1)
if show_out_of_range:
    filtered_labs = filtered_labs[filtered_labs["out_of_range"]].copy()

filtered_labs["date_str"] = filtered_labs["date"].dt.strftime("%Y-%m-%d")
filtered_labs["value_str"] = filtered_labs["value"].map(lambda x: f"{x:.4g}")

with tab_explorer:
                # ----------------------------
                # 10) Events Timeline (Symptoms/Infections/Medications)
                # ----------------------------
                st.subheader("🩺 Symptoms / Infections Timeline")
                
                events_clean = events_df.copy()
                events_clean["start_date"] = pd.to_datetime(events_clean["start_date"], errors="coerce")
                events_clean["end_date"] = pd.to_datetime(events_clean["end_date"], errors="coerce")
                events_clean["end_date"] = events_clean["end_date"].fillna(events_clean["start_date"])
                
                # keep the original type values visible in hover
                events_clean["type_norm"] = events_clean["type"].astype(str).str.strip().str.lower()
                
                def label_event_type(t: str):
                    t = (t or "").lower()
                    if "infection" in t or "covid" in t or "flu" in t or "uri" in t or "viral" in t:
                        return "Infection"
                    if "med" in t or "taper" in t or "vaccine" in t or "antiviral" in t:
                        return "Medication"
                    return "Symptom"
                
                events_clean["type_label"] = events_clean["type_norm"].apply(label_event_type)
                
                events_filtered = events_clean[
                    (events_clean["end_date"] >= start_win)
                    & (events_clean["start_date"] <= end_win)
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
                            "Medication": brand_colors["primary"],
                        },
                        hover_data={"type": True, "start_date": True, "end_date": True},
                    )
                    fig_evt.update_yaxes(autorange="reversed", title=None)
                    fig_evt.update_xaxes(range=[start_win, end_win])
                    fig_evt.update_layout(height=max(420, 40 * len(events_filtered)))
                    fig_evt = style_plotly(
                        fig_evt,
                        title_text="Clinical Events (hover for details)",
                        max_top_legend_items=int(max_top_legend_items),
                        hide_legend_over=int(hide_legend_over),
                    )
                    st.plotly_chart(fig_evt, use_container_width=True)
                
                # ----------------------------
                # 11) Biomarker Overlay — unit split
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
                            color="biomarker",  # IMPORTANT: one line per biomarker
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
                
                        fig_bio.update_xaxes(range=[start_win, end_win])
                        fig_bio.update_layout(height=560, yaxis_title=f"Value ({u})")
                
                        # Mark out-of-range points
                        oor = df_u[df_u["out_of_range"]].copy()
                        if not oor.empty:
                            fig_bio.add_trace(
                                go.Scatter(
                                    x=oor["date"],
                                    y=oor["value"],
                                    mode="markers",
                                    name="Out of range",
                                    marker=dict(symbol="x", size=10, color=brand_colors["accent"]),
                                    hovertemplate="Out of range<br>%{x|%Y-%m-%d}<br>%{y}<extra></extra>",
                                )
                            )
                
                        fig_bio = style_plotly(
                            fig_bio,
                            title_text=f"Overlay — Unit: {u}",
                            max_top_legend_items=int(max_top_legend_items),
                            hide_legend_over=int(hide_legend_over),
                        )
                        st.plotly_chart(fig_bio, use_container_width=True)
                
                # ----------------------------
                # 12) Category Plots — unit split
                # ----------------------------
                if show_category_plots:
                    st.subheader("🧩 Category Plots (Unit-split)")
                
                    cat_df_all = labs_df[
                        (labs_df["date"] >= start_win)
                        & (labs_df["date"] <= end_win)
                        & (labs_df["category"].isin(selected_categories))
                        & (labs_df["unit"].isin(selected_units))
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
                                    color="biomarker",  # IMPORTANT: one line per biomarker
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
                                fig_cat.update_xaxes(range=[start_win, end_win])
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
                                            hovertemplate="Out of range<br>%{x|%Y-%m-%d}<br>%{y}<extra></extra>",
                                        )
                                    )
                
                                fig_cat = style_plotly(
                                    fig_cat,
                                    title_text=f"{cat} — Unit: {u}",
                                    max_top_legend_items=int(max_top_legend_items),
                                    hide_legend_over=int(hide_legend_over),
                                )
                                st.plotly_chart(fig_cat, use_container_width=True)
                
                # ----------------------------
                # 13) Individual Biomarker Plots
                # ----------------------------
                if individual_bio_selected:
                    st.subheader("📊 Individual Biomarker Plots")
                
                    for bm in individual_bio_selected:
                        bm_df = labs_df[
                            (labs_df["biomarker"] == bm)
                            & (labs_df["category"].isin(selected_categories))
                            & (labs_df["date"] >= start_win)
                            & (labs_df["date"] <= end_win)
                            & (labs_df["unit"].isin(selected_units))
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
                        fig_one.update_xaxes(range=[start_win, end_win])
                        fig_one.update_layout(height=360, showlegend=False, yaxis_title=f"Value ({unit_here})")

                        fig.add_vrect(x0=B_start, x1=B_end, fillcolor=brand_colors["secondary"], opacity=0.10, line_width=0)
                        # Reference band
                        low, high = range_for_biomarker(bm)
                        if low is not None and high is not None:
                            fig_one.add_hrect(
                                y0=low,
                                y1=high,
                                fillcolor=brand_colors["secondary"],
                                opacity=0.12,
                                line_width=0,
                            )
                
                        # Out-of-range markers
                        oor_one = bm_df[bm_df["out_of_range"]]
                        if not oor_one.empty:
                            fig_one.add_trace(
                                go.Scatter(
                                    x=oor_one["date"],
                                    y=oor_one["value"],
                                    mode="markers",
                                    marker=dict(symbol="x", size=10, color=brand_colors["accent"]),
                                    name="Out of range",
                                    hovertemplate="Out of range<br>%{x|%Y-%m-%d}<br>%{y}<extra></extra>",
                                )
                            )
                
                        fig_one = style_plotly(
                            fig_one,
                            title_text=f"{bm} ({unit_here})",
                            max_top_legend_items=int(max_top_legend_items),
                            hide_legend_over=int(hide_legend_over),
                        )
                        st.plotly_chart(fig_one, use_container_width=True)



with tab_event_lens:
    st.subheader("🎯 Event Lens — trends around a clinical event")
    st.caption("Descriptive trends only (no clinical interpretation). Compare biomarker values during an event vs a baseline window.")

    # Clean events
    events_clean2 = events_df.copy()
    events_clean2["start_date"] = pd.to_datetime(events_clean2["start_date"], errors="coerce")
    events_clean2["end_date"] = pd.to_datetime(events_clean2["end_date"], errors="coerce")
    events_clean2["end_date"] = events_clean2["end_date"].fillna(events_clean2["start_date"])
    events_clean2 = events_clean2.dropna(subset=["start_date", "end_date"])

    if events_clean2.empty:
        st.info("No events available.")
    else:
        # 1) Comparison mode + choose Event A (and optionally Event B)
        event_names = events_clean2["name"].astype(str).tolist()
        
        compare_mode = st.radio(
            "Comparison mode",
            ["Event vs Baseline (30 days before)", "Event A vs Event B"],
            index=0,
            horizontal=True,
        )
        
        colA, colB = st.columns(2)
        with colA:
            event_A_name = st.selectbox("Event A", options=event_names, index=0)
        
        with colB:
            if compare_mode == "Event A vs Event B":
                default_idx = 1 if len(event_names) > 1 else 0
                event_B_name = st.selectbox("Event B", options=event_names, index=default_idx)
            else:
                event_B_name = None

        # 2) Define windows
        baseline_days = 30  # fixed per your request
        baseline_gap_days = st.slider("Gap before Event A start (days)", 0, 30, 7, 1)
        agg_method = st.radio("Summary statistic", options=["Median", "Mean"], index=0, horizontal=True)
        
        # Optional: restrict by categories/units
        all_cats = sorted(labs_df["category"].dropna().unique())
        cats_for_event = st.multiselect("Categories to include", all_cats, default=all_cats)

        units_for_event = sorted(labs_df["unit"].dropna().unique())
        units_selected_event = st.multiselect("Units to include", units_for_event, default=units_for_event)

       # Get Event A window
        evA = events_clean2[events_clean2["name"].astype(str) == str(event_A_name)].iloc[0]
        A_start = pd.Timestamp(evA["start_date"])
        A_end = pd.Timestamp(evA["end_date"])
        
        # Define Window B: either baseline-before-A OR Event B window
        if compare_mode == "Event A vs Event B" and event_B_name:
            evB = events_clean2[events_clean2["name"].astype(str) == str(event_B_name)].iloc[0]
            B_start = pd.Timestamp(evB["start_date"])
            B_end = pd.Timestamp(evB["end_date"])
        else:
            B_end = A_start - pd.Timedelta(days=baseline_gap_days)
            B_start = B_end - pd.Timedelta(days=baseline_days)

        # Show window info
        c1, c2, c3 = st.columns(3)
        c1.metric("Window A", f"{A_start.date()} → {A_end.date()}")
        if compare_mode == "Event A vs Event B":
            c2.metric("Window B", f"{B_start.date()} → {B_end.date()}")
            c3.metric("Mode", "Event vs Event")
        else:
            c2.metric("Baseline (30d)", f"{B_start.date()} → {B_end.date()}")
            c3.metric("Mode", "Event vs Baseline")
            
        # Pull lab rows
        df = labs_df.copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df = df[(df["category"].isin(cats_for_event)) & (df["unit"].isin(units_selected_event))].copy()

        df_A = df[(df["date"] >= A_start) & (df["date"] <= A_end)].copy()
        df_B = df[(df["date"] >= B_start) & (df["date"] <= B_end)].copy()
        
    if df_A.empty or df_B.empty:
        st.warning("Not enough lab data in one or both windows to compute changes.")
    else:
        agg_fn = np.median if agg_method == "Median" else np.mean
    
        def summarize(window_df, label):
            g = window_df.groupby(["biomarker", "unit", "category"])["value"]
            out = g.apply(agg_fn).reset_index().rename(columns={"value": f"{label}_{agg_method.lower()}"})
            out[f"n_{label}"] = g.size().values
            return out
    
        sumA = summarize(df_A, "A")
        sumB = summarize(df_B, "B")
    
        merged = pd.merge(sumA, sumB, on=["biomarker", "unit", "category"], how="inner")
    
        Acol = f"A_{agg_method.lower()}"
        Bcol = f"B_{agg_method.lower()}"
    
        merged["delta"] = merged[Acol] - merged[Bcol]
        merged["pct_change"] = np.where(
            merged[Bcol].abs() > 1e-12,
            100.0 * (merged["delta"] / merged[Bcol]),
            np.nan
        )
        merged["abs_delta"] = merged["delta"].abs()
    
        st.markdown("#### Observed changes (A vs B)")
    
        sort_mode = st.selectbox(
            "Rank by",
            ["Largest % increase", "Largest % decrease", "Largest absolute delta"],
            index=0
        )
    
        if sort_mode == "Largest % increase":
            merged_view = merged.sort_values("pct_change", ascending=False)
        elif sort_mode == "Largest % decrease":
            merged_view = merged.sort_values("pct_change", ascending=True)
        else:
            merged_view = merged.sort_values("abs_delta", ascending=False)
    
        top_n = st.slider("Show top N biomarkers", 5, 80, 25, 5)
        merged_view = merged_view.head(top_n)
    
        display_cols = [
            "biomarker", "category", "unit",
            Bcol, Acol, "delta", "pct_change",
            "n_B", "n_A"
        ]
    
        st.dataframe(
            merged_view[display_cols].style.format({
                Bcol: "{:.4g}",
                Acol: "{:.4g}",
                "delta": "{:.4g}",
                "pct_change": "{:.2f}",
            }),
            use_container_width=True
        )
            
        st.dataframe(
           merged_view[display_cols].style.format({
                    bcol: "{:.4g}",
                    ecol: "{:.4g}",
                    "abs_change": "{:.4g}",
                    "pct_change": "{:.2f}",
                }),
                use_container_width=True
            )     

        st.markdown("#### Plot of changes (top biomarkers)")
        metric = st.selectbox("Plot metric", ["Percent change", "Delta"], index=0)
        plot_n = st.slider("How many biomarkers to plot", 5, 40, 15, 5)
            
            plot_df = merged_view.head(plot_n).copy()
            ycol = "pct_change" if metric == "Percent change" else "delta"
            ytitle = "% change (A vs B)" if metric == "Percent change" else "Delta (A - B)"
            
            fig_delta = px.bar(
                plot_df,
                x="biomarker",
                y=ycol,
                color="category",
                hover_data={"unit": True, Bcol: True, Acol: True, "delta": True, "pct_change": True, "n_B": True, "n_A": True},
            )
            fig_delta.update_layout(height=450, yaxis_title=ytitle)
            fig_delta = style_plotly(fig_delta, title_text=f"Top changes — {metric}", max_top_legend_items=10, hide_legend_over=80)
            st.plotly_chart(fig_delta, use_container_width=True)

            st.markdown("#### Visualize top biomarkers (time series)")
            plot_top = st.checkbox("Plot top biomarkers", value=True)

            if plot_top:
                biomarker_list = merged_view["biomarker"].unique().tolist()
                plot_df = df[df["biomarker"].isin(biomarker_list)].copy().sort_values(["unit", "biomarker", "date"])

                for u in sorted(plot_df["unit"].unique()):
                    df_u = plot_df[plot_df["unit"] == u].copy()
                    if df_u.empty:
                        continue

                    fig = px.line(
                        df_u,
                        x="date",
                        y="value",
                        color="biomarker",
                        markers=True,
                        hover_data={"biomarker": True, "category": True, "unit": True, "value": True, "date": True},
                    )
                   fig.add_vrect(
                        x0=A_start, x1=A_end,
                        fillcolor=brand_colors["accent"],
                        opacity=0.12,
                        line_width=0
                    )
                    fig.update_layout(height=520, yaxis_title=f"Value ({u})")
                    fig = style_plotly(fig, title_text=f"Top changes — Unit: {u}", max_top_legend_items=10, hide_legend_over=80)
                    st.plotly_chart(fig, use_container_width=True)


