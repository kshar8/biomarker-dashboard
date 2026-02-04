# ============================================================
# Streamlit Biomarker & Symptom Explorer (Plotly Hover + Brand)
# UPDATE:
#   - Standard reference ranges from reference_ranges.csv
#   - Custom ranges via sliders (overlay-only OR all biomarkers in selected categories)
#   - Biomarker Trends: ALWAYS 1 line per biomarker (legend = biomarker)
#   - NEW: Separate category plots (each category gets its own plot with its biomarkers)
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

category_colors = {
    "immune": brand_colors["accent"],
    "neuro": brand_colors["primary"],
    "metabolic": brand_colors["secondary"],
}

# ----------------------------
# 2) Page Config + Minimal Styling
# ----------------------------
st.set_page_config(layout="wide", page_title="Biomarker Dashboard")

st.markdown(
    f"<h1 style='font-family:sans-serif; color:{brand_colors['primary']};'>🧬 Biomarker & Symptom Explorer</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

st.markdown(
    f"""
    <style>
      .stApp {{
        background-color: {brand_colors['background']};
      }}

      section[data-testid="stSidebar"] {{
        background-color: {brand_colors['primary']};
      }}

      section[data-testid="stSidebar"] h1,
      section[data-testid="stSidebar"] h2,
      section[data-testid="stSidebar"] h3 {{
        color: {brand_colors['background']} !important;
        font-weight: 650;
      }}

      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] p,
      section[data-testid="stSidebar"] span {{
        color: #FFFFFF !important;
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
# 3) Plotly Styling Helper
# ----------------------------
def style_plotly(fig):
    fig.update_layout(
        template="plotly_white",
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
        margin=dict(l=20, r=20, t=50, b=45),
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

# Standard reference ranges from CSV
ref_df = pd.read_csv("reference_ranges.csv")
ref_df["biomarker"] = ref_df["biomarker"].astype(str).str.strip()
ref_df["low"] = pd.to_numeric(ref_df["low"], errors="coerce")
ref_df["high"] = pd.to_numeric(ref_df["high"], errors="coerce")
default_ref_ranges = dict(zip(ref_df["biomarker"], list(zip(ref_df["low"], ref_df["high"]))))

# ----------------------------
# 5) Sidebar Controls
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
            st.caption(f"Custom sliders available for: {len(custom_biomarker_list)} biomarkers")

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
                    bm_vals = labs_df[labs_df["biomarker"] == bm]["value"]
                    if bm_vals.notna().any():
                        vmin, vmax = float(bm_vals.min()), float(bm_vals.max())
                        pad = (vmax - vmin) * 0.25 if vmax > vmin else max(1.0, abs(vmin) * 0.25)
                        low, high = vmin - pad, vmax + pad
                    else:
                        low, high = 0.0, 1.0

                # safer bounds
                min_bound = float(low) - abs(float(low)) * 1.0 - 1.0
                max_bound = float(high) + abs(float(high)) * 1.0 + 1.0

                custom_ref_ranges[bm] = st.slider(
                    f"{bm} custom range",
                    min_value=min_bound,
                    max_value=max_bound,
                    value=(float(low), float(high))
                )

    show_out_of_range = st.checkbox("Show only out-of-range values?", value=False)

    st.subheader("Individual Biomarker Plots")
    individual_bio_selected = st.multiselect(
        "Select biomarkers for separate plots",
        options=available_biomarkers,
        default=[]
    )

    st.subheader("Category Plots")
    show_category_plots = st.checkbox("Show separate plots for each selected category", value=True)

# ----------------------------
# 6) Date window + range helpers
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
    return (row["value"] < low) or (row["value"] > high)

# ----------------------------
# 7) Filter labs (overlay selection)
# ----------------------------
filtered_labs = labs_df[
    (labs_df["date"] >= start_win) &
    (labs_df["date"] <= end_win) &
    (labs_df["category"].isin(selected_categories)) &
    (labs_df["biomarker"].isin(selected_biomarkers))
].copy()

filtered_labs["out_of_range"] = filtered_labs.apply(is_out_of_range, axis=1)
if show_out_of_range:
    filtered_labs = filtered_labs[filtered_labs["out_of_range"]].copy()

filtered_labs["date_str"] = filtered_labs["date"].dt.strftime("%Y-%m-%d")
filtered_labs["value_str"] = pd.to_numeric(filtered_labs["value"], errors="coerce").map(lambda x: f"{x:.4g}")

# ----------------------------
# 8) Events Timeline (Plotly hover)
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
    fig_evt.update_layout(height=max(340, 36 * len(events_filtered)), legend_title_text="")
    fig_evt = style_plotly(fig_evt)
    st.plotly_chart(fig_evt, use_container_width=True)

# ----------------------------
# 9) Biomarker Trends (1 line per biomarker, legend = biomarker)
# ----------------------------
st.subheader("🧪 Biomarker Trends (Overlay)")

if filtered_labs.empty:
    st.warning("No lab data in the selected window / filters.")
else:
    plot_df = filtered_labs.sort_values(["date", "biomarker"]).copy()

    fig_bio = px.line(
        plot_df,
        x="date",
        y="value",
        color="biomarker",          # IMPORTANT: 1 line per biomarker
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
    fig_bio.update_layout(
        height=580,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    # Out-of-range highlight layer
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

    fig_bio = style_plotly(fig_bio)
    st.plotly_chart(fig_bio, use_container_width=True)

# ----------------------------
# 9B) NEW: Separate plots per category (biomarkers within that category)
# ----------------------------
if show_category_plots:
    st.subheader("🧩 Category Plots (Biomarkers within each category)")

    # Use the full lab dataset for those selected categories, but keep the date window
    cat_df_all = labs_df[
        (labs_df["date"] >= start_win) &
        (labs_df["date"] <= end_win) &
        (labs_df["category"].isin(selected_categories))
    ].copy()

    if cat_df_all.empty:
        st.info("No lab data for selected categories in the date window.")
    else:
        # Build one plot per category
        for cat in selected_categories:
            cat_df = cat_df_all[cat_df_all["category"] == cat].copy()
            if cat_df.empty:
                continue

            # Optional: limit to selected biomarkers only? (currently shows ALL in that category)
            # If you want it limited, uncomment the next line:
            # cat_df = cat_df[cat_df["biomarker"].isin(selected_biomarkers)]

            cat_df["date_str"] = cat_df["date"].dt.strftime("%Y-%m-%d")
            cat_df["value_str"] = pd.to_numeric(cat_df["value"], errors="coerce").map(lambda x: f"{x:.4g}")
            cat_df["out_of_range"] = cat_df.apply(is_out_of_range, axis=1)

            fig_cat = px.line(
                cat_df.sort_values(["date", "biomarker"]),
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
                title=f"{cat} biomarkers",
            )

            fig_cat.update_xaxes(range=[start_win, end_win], tickformat="%Y-%m-%d")
            fig_cat.update_layout(
                height=460,
                legend_title_text="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )

            # Out-of-range highlight for this category
            oor_cat = cat_df[cat_df["out_of_range"]].copy()
            if not oor_cat.empty:
                fig_cat.add_trace(
                    go.Scatter(
                        x=oor_cat["date"],
                        y=oor_cat["value"],
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
                                oor_cat["biomarker"].astype(str),
                                oor_cat["value_str"].astype(str),
                                oor_cat["unit"].astype(str),
                                oor_cat["date_str"].astype(str),
                                oor_cat.get("draw_id", pd.Series([""] * len(oor_cat))).astype(str),
                            ],
                            axis=1,
                        ),
                    )
                )

            fig_cat = style_plotly(fig_cat)
            st.plotly_chart(fig_cat, use_container_width=True)

# ----------------------------
# 10) Individual Biomarker Plots
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
        bm_df["value_str"] = pd.to_numeric(bm_df["value"], errors="coerce").map(lambda x: f"{x:.4g}")
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
            title=bm
        )

        fig_one.update_xaxes(range=[start_win, end_win], tickformat="%Y-%m-%d")
        fig_one.update_layout(height=340, showlegend=False)

        fig_one.update_traces(line=dict(color=brand_colors["primary"]), marker=dict(color=brand_colors["primary"]))

        # Reference range band (per biomarker) if available
        low, high = range_for_biomarker(bm)
        if low is not None and high is not None:
            fig_one.add_hrect(
                y0=low,
                y1=high,
                fillcolor=brand_colors["secondary"],
                opacity=0.12,
                line_width=0
            )

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
                )
            )

        fig_one = style_plotly(fig_one)
        st.plotly_chart(fig_one, use_container_width=True)
