import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

st.set_page_config(layout="wide", page_title="Biomarker Dashboard")
st.title("Biomarker & Symptom Explorer (Matplotlib)")

# ----------------------------
# Load Data
# ----------------------------
labs_df = pd.read_csv("lab_data.csv", parse_dates=["date"])
events_df = pd.read_csv("events.csv", parse_dates=["start_date", "end_date"])

# ----------------------------
# Default reference ranges
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
# Sidebar Controls
# ----------------------------
st.sidebar.header("Controls")

# Date range slider
min_date = labs_df["date"].min().date()
max_date = labs_df["date"].max().date()
date_range = st.sidebar.slider(
    "Select date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)
)

# Category selection
all_categories = labs_df["category"].unique()
selected_categories = st.sidebar.multiselect(
    "Select categories", all_categories, default=list(all_categories)
)

# Biomarker selection
available_biomarkers = labs_df[labs_df["category"].isin(selected_categories)]["biomarker"].unique()
selected_biomarkers = st.sidebar.multiselect(
    "Select biomarkers", available_biomarkers, default=available_biomarkers[:5]
)

# Show reference ranges
show_ref = st.sidebar.checkbox("Show reference ranges?", value=True)

# Custom reference ranges
custom_ref_ranges = {}
st.sidebar.subheader("Custom Reference Ranges")
for bm in selected_biomarkers:
    if bm in default_ref_ranges:
        min_val, max_val = default_ref_ranges[bm]
        user_range = st.sidebar.slider(
            f"{bm} range", min_value=float(min_val*0.5), max_value=float(max_val*2),
            value=(min_val, max_val)
        )
        custom_ref_ranges[bm] = user_range
    else:
        custom_ref_ranges[bm] = (None, None)

# ----------------------------
# Filter lab data
# ----------------------------
filtered_labs = labs_df[
    (labs_df["category"].isin(selected_categories)) &
    (labs_df["biomarker"].isin(selected_biomarkers)) &
    (labs_df["date"] >= pd.to_datetime(date_range[0])) &
    (labs_df["date"] <= pd.to_datetime(date_range[1]))
].copy()

# ----------------------------
# Event track plot
# ----------------------------
st.subheader("Symptoms / Infections Timeline")
fig_event, ax_event = plt.subplots(figsize=(12, 1.5))
ax_event.set_ylim(0, len(events_df))
ax_event.set_yticks([])
ax_event.set_xticks([])
ax_event.set_title("Symptoms / Infections Timeline")

for i, event in enumerate(events_df.to_dict(orient="records")):
    start = mdates.date2num(event["start_date"].to_pydatetime())
    end = mdates.date2num(event["end_date"].to_pydatetime())
    color = "#ff9999" if event["type"] == "infection" else "#ffe066"
    ax_event.barh(i, width=end-start, left=start, height=0.8, color=color)
    ax_event.text(start + (end-start)/2, i, event["name"], ha="center", va="center", fontsize=9)

ax_event.barh([], [], color="#ffe066", label="Symptom")
ax_event.barh([], [], color="#ff9999", label="Infection")
ax_event.legend(loc="upper right", fontsize=9, framealpha=0.7)
plt.tight_layout()
st.pyplot(fig_event)

# ----------------------------
# Biomarker overlay plot
# ----------------------------
st.subheader("Biomarker Trends")

fig, ax = plt.subplots(figsize=(12,6))

for bm in selected_biomarkers:
    bm_data = filtered_labs[filtered_labs["biomarker"]==bm]
    ax.plot(bm_data["date"], bm_data["value"], marker='o', label=bm)
    
    # Reference range shading
    if show_ref and bm in custom_ref_ranges:
        low, high = custom_ref_ranges[bm]
        ax.fill_between(bm_data["date"], low, high, alpha=0.1)

ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.legend()
ax.grid(True)
plt.tight_layout()
st.pyplot(fig)



