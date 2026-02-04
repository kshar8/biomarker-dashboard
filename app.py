# ----------------------------
# Streamlit Biomarker & Symptom Explorer - Brand Colors (vNext)
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

# Optional category colors (extend later as you add categories)
category_colors = {
    "immune": brand_colors["accent"],
    "neuro": brand_colors["primary"],
    "metabolic": brand_colors["secondary"],
}

# ----------------------------
# 2️⃣ Page Config + Styling
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
        background-color: {brand_colors['background']};
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

    # Date range
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

    # Overlay option
    overlay_option = st.selectbox(
        "Overlay option",
        ["All biomarkers", "Biomarkers within selected categories", "Individual categories"]
    )

    # Reference range controls
    with st.expander("Reference Ranges"):
        show_standard_ref = st.checkbox("Show standard ranges", value=True)
        show_custom_ref = st.checkbox("Show custom ranges", value=False)

        custom_ref_ranges = {}
        if show_custom_ref:
            for bm in selected_biomarkers:
                if bm in default_ref_ranges:
                    min_val, max_val = default_ref_ranges[bm]
                    user_range = st.slider(
                        f"{bm} custom range",
                        min_value=float(min_val * 0.5),
                        max_value=float(max_val * 2.0),
                        value=(float(min_val), float(max_val))
                    )
                    custom_ref_ranges[bm] = user_range
                else:
                    custom_ref_ranges[bm] = (None, None)
        else:
            # still populate keys for downstream code
            for bm in selected_biomarkers:
                custom_ref_ranges[bm] = (None, None)

    # Annotations
    with st.expander("Annotations"):
        show_annotations = st.checkbox("Show min/max annotations only", value=False)

    # Out-of-range filter (based on custom if enabled, else standard)
    show_out_of_range = st.checkbox("Show only out-of-range values?", value=False)

    # Individual biomarker plots
    st.subheader("Individual Biomarker Plots")
    individual_bio_selected = st.multiselect(
        "Select biomarkers for separate plots",
        options=available_biomarkers,
        default=[]
    )

# ----------------------------
# 6️⃣ Filter Lab Data
# ----------------------------
filtered_labs = labs_df[
    (labs_df["category"].isin(selected_categories)) &
    (labs_df["biomarker"].isin(selected_biomarkers)) &
    (labs_df["date"] >= pd.Timestamp(date_range[0])) &
    (labs_df["date"] <= pd.Timestamp(date_range[1]))
].copy()

# Choose which ranges define out-of-range filtering
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
# 7️⃣ Timeline: Symptoms / Infections
# ----------------------------
st.subheader("🩺 Symptoms / Infections Timeline")

fig_event, ax_event = plt.subplots(figsize=(14, 1.7))
ax_event.set_facecolor(brand_colors["background"])
ax_event.set_yticks([])
ax_event.set_title("Symptoms / Infections Timeline", fontsize=12, color=brand_colors["primary"])

# Filter events to date window (so timeline matches selected dates)
events_filtered = events_df[
    (events_df["end_date"] >= pd.Timestamp(date_range[0])) &
    (events_df["start_date"] <= pd.Timestamp(date_range[1]))
].copy()

for i, event in enumerate(events_filtered.to_dict(orient="records")):
    color = brand_colors["accent"] if str(event["type"]).lower() == "infection" else brand_colors["secondary"]
    ax_event.barh(
        y=i,
        width=event["end_date"] - event["start_date"],
        left=event["start_date"],
        height=0.7,
        color=color
    )
    ax_event.text(
        event["start_date"] + (event["end_date"] - event["start_date"]) / 2,
        i,
        event["name"],
        ha="center",
        va="center",
        fontsize=8,
        color=brand_colors["primary"]
    )

# show date ticks
ax_event.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45, ha="right")

# Legend (explicit patches so it's always correct)
symptom_patch = mpatches.Patch(color=brand_colors["secondary"], label="Symptom")
infection_patch = mpatches.Patch(color=brand_colors["accent"], label="Infection")
ax_event.legend(handles=[symptom_patch, infection_patch], loc="upper right", fontsize=8, framealpha=0.8)

plt.tight_layout()
st.pyplot(fig_event, use_container_width=True)

# ----------------------------
# 8️⃣ Biomarker Overlay Plot (ALL overlay modes implemented)
# ----------------------------
st.subheader("🧪 Biomarker Trends")

fig_bio, ax_bio = plt.subplots(figsize=(14, 5))
ax_bio.set_facecolor(brand_colors["background"])

fallback_colors = [brand_colors["primary"], brand_colors["secondary"], brand_colors["accent"]]

def add_ref_band(bm_data: pd.DataFrame, bm: str):
    if bm_data.empty:
        return

    # standard reference range
    if show_standard_ref and bm in default_ref_ranges:
        low, high = default_ref_ranges[bm]
        ax_bio.fill_between(bm_data["date"], low, high, alpha=0.08, color=brand_colors["primary"])

    # custom reference range
    if show_custom_ref:
        low, high = custom_ref_ranges.get(bm, (None, None))
        if low is not None and high is not None:
            ax_bio.fill_between(bm_data["date"], low, high, alpha=0.12, color=brand_colors["secondary"])

def add_minmax_annotations(bm_data: pd.DataFrame):
    if not show_annotations or bm_data.empty:
        return
    for idx in [bm_data["value"].idxmin(), bm_data["value"].idxmax()]:
        row = bm_data.loc[idx]
        ax_bio.annotate(
            f"{row['value']}",
            (row["date"], row["value"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=brand_colors["primary"]
        )

def plot_series(bm: str, bm_data: pd.DataFrame, color: str, label: str):
    if bm_data.empty:
        return
    bm_data = bm_data.sort_values("date")
    add_ref_band(bm_data, bm)
    ax_bio.plot(bm_data["date"], bm_data["value"], marker="o", label=label, color=color)
    add_minmax_annotations(bm_data)

if overlay_option == "All biomarkers":
    for i, bm in enumerate(sorted(filtered_labs["biomarker"].unique())):
        bm_data = filtered_labs[filtered_labs["biomarker"] == bm]
        color = fallback_colors[i % len(fallback_colors)]
        plot_series(bm, bm_data, color=color, label=bm)

elif overlay_option == "Biomarkers within selected categories":
    for i, cat in enumerate(selected_categories):
        cat_df = filtered_labs[filtered_labs["category"] == cat]
        if cat_df.empty:
            continue
        cat_color = category_colors.get(cat, fallback_colors[i % len(fallback_colors)])
        for bm in sorted(cat_df["biomarker"].unique()):
            bm_data = cat_df[cat_df["biomarker"] == bm]
            plot_series(bm, bm_data, color=cat_color, label=f"{bm} ({cat})")

elif overlay_option == "Individual categories":
    for i, cat in enumerate(selected_categories):
        cat_df = filtered_labs[filtered_labs["category"] == cat]
        if cat_df.empty:
            continue
        cat_color = category_colors.get(cat, fallback_colors[i % len(fallback_colors)])
        for bm in sorted(cat_df["biomarker"].unique()):
            bm_data = cat_df[cat_df["biomarker"] == bm]
            plot_series(bm, bm_data, color=cat_color, label=f"{bm} [{cat}]")

ax_bio.set_ylabel("Value", color=brand_colors["primary"])
ax_bio.legend(fontsize=7, ncols=2)
ax_bio.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

st.pyplot(fig_bio, use_container_width=True)

# ----------------------------
# 9️⃣ Individual Biomarker Plots
# ----------------------------
if individual_bio_selected:
    st.subheader("📊 Individual Biomarker Plots")

    for bm in individual_bio_selected:
        bm_data = labs_df[
            (labs_df["biomarker"] == bm) &
            (labs_df["category"].isin(selected_categories)) &
            (labs_df["date"] >= pd.Timestamp(date_range[0])) &
            (labs_df["date"] <= pd.Timestamp(date_range[1]))
        ].copy()

        if bm_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.set_facecolor(brand_colors["background"])

        bm_data = bm_data.sort_values("date")
        ax.plot(bm_data["date"], bm_data["value"], marker="o", color=brand_colors["primary"])

        # Reference ranges
        if show_standard_ref and bm in default_ref_ranges:
            low, high = default_ref_ranges[bm]
            ax.fill_between(bm_data["date"], low, high, alpha=0.08, color=brand_colors["primary"])

        if show_custom_ref:
            low, high = custom_ref_ranges.get(bm, (None, None))
            if low is not None and high is not None:
                ax.fill_between(bm_data["date"], low, high, alpha=0.12, color=brand_colors["secondary"])

        # Min/Max annotations
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

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)
