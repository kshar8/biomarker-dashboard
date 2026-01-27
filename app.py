# ----------------------------
# Streamlit Biomarker Explorer (Fixed)
# ----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

st.set_page_config(layout="wide")
st.title("Biomarker & Symptom Explorer")

# ----------------------------
# 1️⃣ Load CSVs
# ----------------------------
labs_df = pd.read_csv("lab_data.csv", parse_dates=["date"])
events_df = pd.read_csv("events.csv", parse_dates=["start_date", "end_date"])
events = events_df.to_dict(orient="records")

# ----------------------------
# 2️⃣ Reference ranges
# ----------------------------
ref_ranges = {
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
# 3️⃣ Sidebar controls
# ----------------------------
categories = labs_df["category"].unique()
category_choice = st.sidebar.selectbox("Select category", categories)

available_biomarkers = labs_df[labs_df["category"] == category_choice]["biomarker"].unique()
biomarker_choice = st.sidebar.multiselect(
    "Select biomarkers", available_biomarkers, default=available_biomarkers[:3]
)

overlay = st.sidebar.checkbox("Overlay biomarkers?", value=False)
show_ref = st.sidebar.checkbox("Show reference ranges?", value=True)

# ----------------------------
# 4️⃣ Date slider (fixed)
# ----------------------------
min_date = labs_df["date"].min().to_pydatetime()
max_date = labs_df["date"].max().to_pydatetime()

date_range = st.sidebar.slider(
    "Select date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)
)

# ----------------------------
# 5️⃣ Filter data
# ----------------------------
filtered_df = labs_df[
    (labs_df["category"] == category_choice)
    & (labs_df["biomarker"].isin(biomarker_choice))
    & (labs_df["date"] >= pd.Timestamp(date_range[0]))
    & (labs_df["date"] <= pd.Timestamp(date_range[1]))
]

# ----------------------------
# 6️⃣ Event Track Plot
# ----------------------------
st.subheader("Symptoms / Infections Timeline")
fig_event, ax_event = plt.subplots(figsize=(12, 1.5))
ax_event.set_ylim(0, len(events))
ax_event.set_yticks([])
ax_event.set_xticks([])
ax_event.set_title("Symptoms / Infections Timeline")

for i, event in enumerate(events):
    start = mdates.date2num(event["start_date"].to_pydatetime())
    end = mdates.date2num(event["end_date"].to_pydatetime())
    color = "#ff9999" if event["type"] == "infection" else "#ffe066"
    ax_event.barh(i, width=end - start, left=start, height=0.8, color=color)
    ax_event.text(start + (end - start) / 2, i, event["name"], ha="center", va="center", fontsize=9)

ax_event.barh([], [], color="#ffe066", label="Symptom")
ax_event.barh([], [], color="#ff9999", label="Infection")
ax_event.legend(loc="upper right", fontsize=9, framealpha=0.7)
plt.tight_layout()
st.pyplot(fig_event)

# ----------------------------
# 7️⃣ Biomarker Plot(s)
# ----------------------------
st.subheader("Biomarker Trends")

if overlay:
    # All selected biomarkers on one plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for bm in biomarker_choice:
        bm_data = filtered_df[filtered_df["biomarker"] == bm]
        ax.plot(bm_data["date"], bm_data["value"], marker="o", label=bm)
        if show_ref and bm in ref_ranges:
            low, high = ref_ranges[bm]
            ax.fill_between(bm_data["date"], low, high, alpha=0.05)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
else:
    # Separate subplots for each biomarker
    biomarker_list = biomarker_choice
    fig_height = 2 * len(biomarker_list)
    fig, axes = plt.subplots(len(biomarker_list), 1, figsize=(12, fig_height), sharex=True)
    axes = np.atleast_1d(axes)

    for i, bm in enumerate(biomarker_list):
        ax = axes[i]
        bm_data = filtered_df[filtered_df["biomarker"] == bm]
        ax.plot(bm_data["date"], bm_data["value"], marker="o", label=bm)
        if show_ref and bm in ref_ranges:
            low, high = ref_ranges[bm]
            y_min, y_max = bm_data["value"].min(), bm_data["value"].max()
            ref_low = max(low, y_min - (y_max - y_min) * 0.1)
            ref_high = min(high, y_max + (y_max - y_min) * 0.1)
            if ref_low < ref_high:
                ax.fill_between(bm_data["date"], ref_low, ref_high, color="green", alpha=0.1, label="Reference Range")
        ax.set_ylabel(f"{bm} ({bm_data['unit'].iloc[0]})")
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

