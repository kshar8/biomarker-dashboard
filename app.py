# ----------------------------
# Streamlit Biomarker & Symptom Explorer - Brand Colors
# ----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from datetime import datetime

# ----------------------------
# 1️⃣ Brand Colors
# ----------------------------
brand_colors = {
    "primary": "#405A51",    # Dark green
    "secondary": "#607663",  # Medium green
    "accent": "#A25B4C",     # Reddish accent
    "background": "#F2EEE2", # Light neutral background
}

category_colors = {
    "immune": brand_colors["accent"],
    "neuro": brand_colors["primary"],
    "metabolic": brand_colors["secondary"],
}

# ----------------------------
# 2️⃣ Page Config
# ----------------------------
st.set_page_config(layout="wide", page_title="Biomarker Dashboard")
st.markdown(
    f"<h1 style='font-family:sans-serif; color:{brand_colors['primary']};'>🧬 Biomarker & Symptom Explorer</h1>", 
    unsafe_allow_html=True
)
st.markdown("---")

# Sidebar brand styling
st.markdown(
    f"""
    <style>
    .sidebar .sidebar-content {{
        background-color: {brand_colors['background']};
    }}
    .sidebar .sidebar-content h2, .sidebar .sidebar-content h3 {{
        color: {brand_colors['primary']};
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
# 4️⃣ Default Reference Ranges
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
    date_range = st.slider("Select date range", min_value=min_date, max_value=max_date, value=(min_date, max_date))

    # Categories & biomarkers
    all_categories = labs_df["category"].unique()
    selected_categories = st.multiselect("Select categories", all_categories, default=list(all_categories))
    available_biomarkers = labs_df[labs_df["category"].isin(selected_categories)]["biomarker"].unique()
    selected_biomarkers = st.multiselect("Select biomarkers for overlay", available_biomarkers, default=available_biomarkers[:5])

    # Overlay option
    overlay_option = st.selectbox("Overlay option", ["All biomarkers", "Biomarkers within selected categories", "Individual categories"])

    # Reference range expanders
    with st.expander("Reference Ranges"):
        show_standard_ref = st.checkbox("Show standard ranges", value=True)
        show_custom_ref = st.checkbox("Show custom ranges", value=True)

        custom_ref_ranges = {}
        for bm in selected_biomarkers:
            if bm in default_ref_ranges:
                min_val, max_val = default_ref_ranges[bm]
                user_range = st.slider(f"{bm} custom range", min_value=float(min_val*0.5), max_value=float(max_val*2), value=(float(min_val), float(max_val)))
                custom_ref_ranges[bm] = user_range
            else:
                custom_ref_ranges[bm] = (None, None)

    # Annotations
    with st.expander("Annotations"):
        show_annotations = st.checkbox("Show max/min annotations?", value=True)

    # Out-of-range filter
    show_out_of_range = st.checkbox("Show only out-of-range values?", value=False)

    # Individual biomarker plots
    st.subheader("Individual Biomarker Plots")
    individual_bio_selected = st.multiselect("Select biomarkers for separate plots", options=available_biomarkers, default=[])

# ----------------------------
# 6️⃣ Filter Lab Data
# ----------------------------
filtered_labs = labs_df[
    (labs_df["category"].isin(selected_categories)) &
    (labs_df["biomarker"].isin(selected_biomarkers)) &
    (labs_df["date"] >= pd.Timestamp(date_range[0])) &
    (labs_df["date"] <= pd.Timestamp(date_range[1]))
].copy()

if show_out_of_range:
    def out_of_range(row):
        low, high = custom_ref_ranges.get(row["biomarker"], (None, None))
        if low is not None and high is not None:
            return row["value"] < low or row["value"] > high
        return True
    filtered_labs = filtered_labs[filtered_labs.apply(out_of_range, axis=1)]

# ----------------------------
# 7️⃣ Timeline: Symptoms / Infections
# ----------------------------
st.subheader("🩺 Symptoms / Infections Timeline")
fig_event, ax_event = plt.subplots(figsize=(14,1.5))
ax_event.set_ylim(0, len(events_df))
ax_event.set_yticks([])
ax_event.set_xticks([])
ax_event.set_facecolor(brand_colors["background"])
ax_event.set_title("Symptoms / Infections Timeline", fontsize=12, color=brand_colors["primary"])

for i, event in enumerate(events_df.to_dict(orient="records")):
    start = mdates.date2num(event["start_date"])
    end = mdates.date2num(event["end_date"])
    color = brand_colors["accent"] if event["type"].lower() == "infection" else brand_colors["secondary"]
    ax_event.barh(i, width=end-start, left=start, height=0.8, color=color)
    ax_event.text(start + (end-start)/2, i, event["name"], ha="center", va="center", fontsize=8, color=brand_colors["primary"])

# Legend
symptom_patch = mpatches.Patch(color=brand_colors["secondary"], label="Symptom")
infection_patch = mpatches.Patch(color=brand_colors["accent"], label="Infection")
ax_event.legend(handles=[symptom_patch, infection_patch], loc="upper right", fontsize=8, framealpha=0.7)

plt.tight_layout()
st.pyplot(fig_event, use_container_width=True)

# ----------------------------
# 8️⃣ Biomarker Overlay Plot
# ----------------------------
st.subheader("🧪 Biomarker Trends")
fig_bio, ax_bio = plt.subplots(figsize=(14,5))
ax_bio.set_facecolor(brand_colors["background"])
colors = [brand_colors["primary"], brand_colors["secondary"], brand_colors["accent"]]

def plot_biomarkers_overlay(df, overlay_option):
    if overlay_option == "All biomarkers":
        for i, bm in enumerate(df["biomarker"].unique()):
            bm_data = df[df["biomarker"]==bm]
            ax_bio.plot(bm_data["date"], bm_data["value"], marker="o", label=bm, color=colors[i % len(colors)])
            if show_annotations and len(bm_data)>0:
                for idx in [bm_data["value"].idxmax(), bm_data["value"].idxmin()]:
                    row = bm_data.loc[idx]
                    ax_bio.annotate(f"{row['value']}\n{row['date'].date()}",
                                    (row['date'], row['value']), xytext=(0,10),
                                    textcoords="offset points", ha='center', fontsize=7)
            if show_standard_ref and bm in default_ref_ranges:
                low, high = default_ref_ranges[bm]
                ax_bio.fill_between(bm_data["date"], low, high, alpha=0.1, color=brand_colors["primary"])
            if show_custom_ref and bm in custom_ref_ranges:
                low, high = custom_ref_ranges[bm]
                ax_bio.fill_between(bm_data["date"], low, high, alpha=0.15, color=brand_colors["secondary"])

    # Can expand for other overlay options similarly...

plot_biomarkers_overlay(filtered_labs, overlay_option)
ax_bio.set_ylabel("Value", color=brand_colors["primary"])
ax_bio.legend(fontsize=7)
ax_bio.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig_bio, use_container_width=True)

# ----------------------------
# 9️⃣ Individual Biomarker Plots
# ----------------------------
if individual_bio_selected:
    st.subheader("📊 Individual Biomarker Plots")
    for bm in individual_bio_selected:
        bm_data = filtered_labs[filtered_labs["biomarker"]==bm]
        if bm_data.empty:
            continue
        fig, ax = plt.subplots(figsize=(12,3))
        ax.set_facecolor(brand_colors["background"])
        ax.plot(bm_data["date"], bm_data["value"], marker="o", label=bm, color=brand_colors["primary"])

        # Reference ranges
        if show_standard_ref and bm in default_ref_ranges:
            low, high = default_ref_ranges[bm]
            ax.fill_between(bm_data["date"], low, high, alpha=0.1, color=brand_colors["primary"])
        if show_custom_ref and bm in custom_ref_ranges:
            low, high = custom_ref_ranges[bm]
            ax.fill_between(bm_data["date"], low, high, alpha=0.15, color=brand_colors["secondary"])

        # Max/min annotations
        if show_annotations and len(bm_data) > 0:
            for idx in [bm_data["value"].idxmax(), bm_data["value"].idxmin()]:
                row = bm_data.loc[idx]
                ax.annotate(f"{row['value']}\n{row['date'].date()}",
                            (row['date'], row['value']),
                            xytext=(0,10), textcoords="offset points",
                            ha='center', fontsize=7)

        ax.set_title(f"{bm} ({bm_data['category'].iloc[0]})", fontsize=10, color=brand_colors["primary"])
        ax.set_ylabel(f"{bm_data['unit'].iloc[0]}", color=brand_colors["primary"])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)




