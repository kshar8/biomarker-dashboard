# ----------------------------
# Streamlit Biomarker & Symptom Explorer - Brand Colors
# FIXED: aligned x-axes visually across plots (consistent margins)
# ----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# ----------------------------
# 1️⃣ Brand Colors
# ----------------------------
brand_colors = {
    "primary": "#405A51",    # Dark green
    "secondary": "#607663",  # Medium green
    "accent": "#A25B4C",     # Reddish accent
    "background": "#F2EEE2", # Light neutral background
}

# Optional category colors (extend later)
category_colors = {
    "immune": brand_colors["accent"],
    "neuro": brand_colors["primary"],
    "metabolic": brand_colors["secondary"],
}

# ----------------------------
# 2️⃣ Page Config + Styling (Stone main, dark sidebar)
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
    /* Main app background */
    .stApp {{
        background-color: {brand_colors['background']};
    }}

    /* Main page headers */
    div[data-testid="stAppViewContainer"] h1,
    div[data-testid="stAppViewContainer"] h2,
    div[data-testid="stAppViewContainer"] h3 {{
        color: {brand_colors['primary']} !important;
        font-weight: 650;
    }}

    /* Sidebar background */
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

    /* Sidebar labels / text */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {{
        color: #FFFFFF !important;
    }}

    /* Expander titles */
    section[data-testid="stSidebar"] summary {{
        color: {brand_colors['background']} !important;
        font-weight: 650;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# 3️⃣ Load Data
# ----------------------------
labs_df = pd.read_csv("lab_data.csv", parse_dates=["date"])
events_df = pd.read_csv("events.csv", parse_dates=["start_date", "end_date"])

# ----------------------------
# 4️⃣ Default Reference Ranges (team-set defaults)
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
# 5️⃣ Sidebar Controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    # Date range (global x-axis window)
    min_date = labs_df["date"].min().date()
    max_date = labs_df["date"].max().date()
    date_range = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    # Categories & biomarkers
    all_categories = sorted(labs_df["category"].dropna().unique())
    selected_categories = st.multiselect(
        "Select categories",
        all_categories,
        default=list(all_categories)
    )

    available_biomarkers = sorted(
        labs_df[labs_df["category"].isin(selected_categories)]["biomarker"].dropna().unique()
    )

    selected_biomarkers = st.multiselect(
        "Select biomarkers for overlay",
        available_biomarkers,
        default=available_biomarkers[:5]
    )

    overlay_option = st.selectbox(
        "Overlay option",
        ["All biomarkers", "Biomarkers within selected categories", "Individual categories"]
    )

    # Reference range controls
    with st.expander("Reference Ranges"):
        show_standard_ref = st.checkbox("Show standard ranges", value=True)
        show_custom_ref = st.checkbox("Enable custom ranges", value=False)

        custom_ref_ranges = {}
        if show_custom_ref:
            for bm in selected_biomarkers:
                if bm in default_ref_ranges:
                    min_val, max_val = default_ref_ranges[bm]
                    custom_ref_ranges[bm] = st.slider(
                        f"{bm} custom range",
                        min_value=float(min_val * 0.5),
                        max_value=float(max_val * 2.0),
                        value=(float(min_val), float(max_val))
                    )
                else:
                    custom_ref_ranges[bm] = (None, None)
        else:
            for bm in selected_biomarkers:
                custom_ref_ranges[bm] = (None, None)

    with st.expander("Annotations"):
        show_annotations = st.checkbox("Show min/max annotations only", value=False)

    show_out_of_range = st.checkbox("Show only out-of-range values?", value=False)

    st.subheader("Individual Biomarker Plots")
    individual_bio_selected = st.multiselect(
        "Select biomarkers for separate plots",
        options=available_biomarkers,
        default=[]
    )

# ----------------------------
# 5.5️⃣ Shared axis + shared layout helpers (FORCE ALIGNMENT)
# ----------------------------
start_win = pd.Timestamp(date_range[0])
end_win = pd.Timestamp(date_range[1])

def enforce_shared_time_axis(ax):
    ax.set_xlim(start_win, end_win)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.tick_params(axis="x", rotation=45)

LEFT_MARGIN = 0.12
RIGHT_MARGIN = 0.98
TOP_MARGIN = 0.92
BOTTOM_MARGIN = 0.28

def enforce_shared_layout(fig, ax):
    enforce_shared_time_axis(ax)
    fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN, top=TOP_MARGIN, bottom=BOTTOM_MARGIN)

# ----------------------------
# 6️⃣ Filter Lab Data
# ----------------------------
filtered_labs = labs_df[
    (labs_df["category"].isin(selected_categories)) &
    (labs_df["biomarker"].isin(selected_biomarkers)) &
    (labs_df["date"] >= start_win) &
    (labs_df["date"] <= end_win)
].copy()

def get_range_for_biomarker(bm: str):
    if show_custom_ref and bm in custom_ref_ranges and custom_ref_ranges[bm] != (None, None):
        return custom_ref_ranges[bm]
    if bm in default_ref_ranges:
        return default_ref_ranges[bm]
    return (None, None)

if show_out_of_range:
    def out_of_range(row):
        low, high = get_range_for_biomarker(row["biomarker"])
        if low is None or high is None:
            return True
        return (row["value"] < low) or (row["value"] > high)

    filtered_labs = filtered_labs[filtered_labs.apply(out_of_range, axis=1)]

# ----------------------------
# 7️⃣ Timeline: Symptoms / Infections (Improved + aligned x-axis + not squished)
# ----------------------------
st.subheader("🩺 Symptoms / Infections Timeline")

events_clean = events_df.copy()
events_clean["start_date"] = pd.to_datetime(events_clean["start_date"], errors="coerce")
events_clean["end_date"] = pd.to_datetime(events_clean["end_date"], errors="coerce")
events_clean["end_date"] = events_clean["end_date"].fillna(events_clean["start_date"])

events_clean["type_norm"] = events_clean["type"].astype(str).str.strip().str.lower()
infection_keywords = ["infection", "covid", "flu", "uri", "viral", "bacterial", "virus"]
events_clean["is_infection"] = events_clean["type_norm"].apply(lambda t: any(k in t for k in infection_keywords))

events_filtered = events_clean[
    (events_clean["end_date"] >= start_win) &
    (events_clean["start_date"] <= end_win)
].dropna(subset=["start_date", "end_date"]).copy()

n_events = len(events_filtered)
fig_height = max(2.8, n_events * 0.55)

fig_event, ax_event = plt.subplots(figsize=(14, fig_height))
ax_event.set_facecolor(brand_colors["background"])
ax_event.set_yticks(range(n_events))
ax_event.set_yticklabels([""] * n_events)
ax_event.set_title("Symptoms / Infections Timeline", fontsize=12, color=brand_colors["primary"])

for i, event in enumerate(events_filtered.to_dict(orient="records")):
    color = brand_colors["accent"] if event["is_infection"] else brand_colors["secondary"]
    ax_event.barh(
        y=i,
        width=event["end_date"] - event["start_date"],
        left=event["start_date"],
        height=0.65,
        color=color
    )
    ax_event.text(
        event["start_date"],
        i,
        f"  {event['name']}",
        va="center",
        ha="left",
        fontsize=9,
        color=brand_colors["primary"]
    )

ax_event.spines["top"].set_visible(False)
ax_event.spines["right"].set_visible(False)
ax_event.spines["left"].set_visible(False)
ax_event.tick_params(axis="y", length=0)

symptom_patch = mpatches.Patch(color=brand_colors["secondary"], label="Symptom")
infection_patch = mpatches.Patch(color=brand_colors["accent"], label="Infection")
ax_event.legend(handles=[symptom_patch, infection_patch], loc="upper right", fontsize=8, framealpha=0.85)

enforce_shared_layout(fig_event, ax_event)
st.pyplot(fig_event, use_container_width=True)

# ----------------------------
# 8️⃣ Biomarker Overlay Plot (aligned x-axis + aligned margins)
# ----------------------------
st.subheader("🧪 Biomarker Trends")

fig_bio, ax_bio = plt.subplots(figsize=(14, 5))
ax_bio.set_facecolor(brand_colors["background"])

fallback_colors = [brand_colors["primary"], brand_colors["secondary"], brand_colors["accent"]]

def add_ref_band(ax, bm_data, bm):
    if bm_data.empty:
        return

    if show_standard_ref and bm in default_ref_ranges:
        low, high = default_ref_ranges[bm]
        ax.fill_between(bm_data["date"], low, high, alpha=0.08, color=brand_colors["primary"])

    if show_custom_ref:
        low, high = custom_ref_ranges.get(bm, (None, None))
        if low is not None and high is not None:
            ax.fill_between(bm_data["date"], low, high, alpha=0.12, color=brand_colors["secondary"])

def add_minmax_annotations(ax, bm_data):
    if not show_annotations or bm_data.empty:
        return
    for idx in [bm_data["value"].idxmin(), bm_data["value"].idxmax()]:
        row = bm_data.loc[idx]
        ax.annotate(
            f"{row['value']}",
            (row["date"], row["value"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=brand_colors["primary"]
        )

def plot_series(ax, bm, bm_data, color, label):
    if bm_data.empty:
        return
    bm_data = bm_data.sort_values("date")
    add_ref_band(ax, bm_data, bm)
    ax.plot(bm_data["date"], bm_data["value"], marker="o", label=label, color=color)
    add_minmax_annotations(ax, bm_data)

if overlay_option == "All biomarkers":
    for i, bm in enumerate(sorted(filtered_labs["biomarker"].unique())):
        bm_data = filtered_labs[filtered_labs["biomarker"] == bm]
        plot_series(ax_bio, bm, bm_data, fallback_colors[i % len(fallback_colors)], bm)

elif overlay_option == "Biomarkers within selected categories":
    for i, cat in enumerate(selected_categories):
        cat_df = filtered_labs[filtered_labs["category"] == cat]
        if cat_df.empty:
            continue
        cat_color = category_colors.get(cat, fallback_colors[i % len(fallback_colors)])
        for bm in sorted(cat_df["biomarker"].unique()):
            bm_data = cat_df[cat_df["biomarker"] == bm]
            plot_series(ax_bio, bm, bm_data, cat_color, f"{bm} ({cat})")

elif overlay_option == "Individual categories":
    for i, cat in enumerate(selected_categories):
        cat_df = filtered_labs[filtered_labs["category"] == cat]
        if cat_df.empty:
            continue
        cat_color = category_colors.get(cat, fallback_colors[i % len(fallback_colors)])
        for bm in sorted(cat_df["biomarker"].unique()):
            bm_data = cat_df[cat_df["biomarker"] == bm]
            plot_series(ax_bio, bm, bm_data, cat_color, f"{bm} [{cat}]")

ax_bio.set_ylabel("Value", color=brand_colors["primary"])
ax_bio.legend(fontsize=7, ncols=2)

enforce_shared_layout(fig_bio, ax_bio)
st.pyplot(fig_bio, use_container_width=True)

# ----------------------------
# 9️⃣ Individual Biomarker Plots (aligned x-axis + aligned margins)
# ----------------------------
if individual_bio_selected:
    st.subheader("📊 Individual Biomarker Plots")

    for bm in individual_bio_selected:
        bm_data = labs_df[
            (labs_df["biomarker"] == bm) &
            (labs_df["category"].isin(selected_categories)) &
            (labs_df["date"] >= start_win) &
            (labs_df["date"] <= end_win)
        ].copy()

        if bm_data.empty:
            continue

        bm_data = bm_data.sort_values("date")

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.set_facecolor(brand_colors["background"])
        ax.plot(bm_data["date"], bm_data["value"], marker="o", color=brand_colors["primary"])

        if show_standard_ref and bm in default_ref_ranges:
            low, high = default_ref_ranges[bm]
            ax.fill_between(bm_data["date"], low, high, alpha=0.08, color=brand_colors["primary"])

        if show_custom_ref:
            low, high = custom_ref_ranges.get(bm, (None, None))
            if low is not None and high is not None:
                ax.fill_between(bm_data["date"], low, high, alpha=0.12, color=brand_colors["secondary"])

        if show_annotations and not bm_data.empty:
            for idx in [bm_data["value"].idxmin(), bm_data["value"].idxmax()]:
                row = bm_data.loc[idx]
                ax.annotate(
                    f"{row['value']}",
                    (row["date"], row["value"]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=brand_colors["primary"]
                )

        unit = bm_data["unit"].iloc[0] if "unit" in bm_data.columns and bm_data["unit"].notna().any() else ""
        ax.set_title(f"{bm}", fontsize=11, color=brand_colors["primary"])
        ax.set_ylabel(unit, color=brand_colors["primary"])

        enforce_shared_layout(fig, ax)
        st.pyplot(fig, use_container_width=True)
