# ----------------------------
# Streamlit Biomarker Explorer v5 - Standard & Custom Reference Ranges
# ----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

st.set_page_config(layout="wide", page_title="Biomarker Dashboard")
st.title("Biomarker & Symptom Explorer")

# ----------------------------
# 1️⃣ Load Data
# ----------------------------
labs_df = pd.read_csv("lab_data.csv", parse_dates=["date"])
events_df = pd.read_csv("events.csv", parse_dates=["start_date", "end_date"])

# ----------------------------
# 2️⃣ Standard Reference Ranges
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
# 3️⃣ Sidebar Controls
# ----------------------------
st.sidebar.header("Controls")

# Date range
min_date = labs_df["date"].min().date()
max_date = labs_df["date"].max().date()
date_range = st.sidebar.slider(
    "Select date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)
)

# Categories and biomarkers
all_categories = labs_df["category"].unique()
selected_categories = st.sidebar.multiselect(
    "Select categories", all_categories, default=list(all_categories)
)

available_biomarkers = labs_df[labs_df["category"].isin(selected_categories)]["biomarker"].unique()
selected_biomarkers = st.sidebar.multiselect(
    "Select biomarkers", available_biomarkers, default=available_biomarkers[:5]
)

# Overlay option
overlay_option = st.sidebar.selectbox(
    "Overlay option",
    options=[
        "All biomarkers",
        "Biomarkers within selected categories",
        "Individual categories"
    ]
)

# Reference ranges toggles
st.sidebar.subheader("Reference Ranges")
show_standard_ref = st.sidebar.checkbox("Show standard reference ranges?", value=True)
show_custom_ref = st.sidebar.checkbox("Show custom reference ranges?", value=True)

# Show annotations toggle
show_annotations = st.sidebar.checkbox("Show annotations for max/min?", value=True)

# Out-of-range filter
show_out_of_range = st.sidebar.checkbox("Show only out-of-range values?", value=False)

# Custom ranges
custom_ref_ranges = {}
st.sidebar.subheader("Custom Reference Ranges")
for bm in selected_biomarkers:
    if bm in default_ref_ranges:
        min_val, max_val = default_ref_ranges[bm]
        user_range = st.sidebar.slider(
            f"{bm} custom range",
            min_value=float(min_val*0.5),
            max_value=float(max_val*2),
            value=(float(min_val), float(max_val))
        )
        custom_ref_ranges[bm] = user_range
    else:
        custom_ref_ranges[bm] = (None, None)

# ----------------------------
# 4️⃣ Filter Lab Data
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
# 5️⃣ Event Track
# ----------------------------
st.subheader("Symptoms / Infections Timeline")

# Ensure datetime
events_df["start_date"] = pd.to_datetime(events_df["start_date"])
events_df["end_date"] = pd.to_datetime(events_df["end_date"])

fig_event, ax_event = plt.subplots(figsize=(12, 1.5))
ax_event.set_ylim(0, len(events_df))
ax_event.set_yticks([])
ax_event.set_xticks([])
ax_event.set_title("Symptoms / Infections Timeline")

for i, event in enumerate(events_df.to_dict(orient="records")):
    start = mdates.date2num(event["start_date"])
    end = mdates.date2num(event["end_date"])
    color = "#ff9999" if event["type"].lower() == "infection" else "#ffe066"
    ax_event.barh(i, width=end-start, left=start, height=0.8, color=color)
    ax_event.text(start + (end-start)/2, i, event["name"], ha="center", va="center", fontsize=9)

ax_event.barh([], [], color="#ffe066", label="Symptom")
ax_event.barh([], [], color="#ff9999", label="Infection")
ax_event.legend(loc="upper right", fontsize=9, framealpha=0.7)
plt.tight_layout()
st.pyplot(fig_event)

# ----------------------------
# 6️⃣ Biomarker Trends
# ----------------------------
st.subheader("Biomarker Trends")

def plot_biomarkers(df, overlay_option):
    if overlay_option == "All biomarkers":
        fig, ax = plt.subplots(figsize=(12,6))
        for bm in df["biomarker"].unique():
            bm_data = df[df["biomarker"]==bm]
            ax.plot(bm_data["date"], bm_data["value"], marker="o", label=bm)
            
            # Annotations for max/min
            if show_annotations and len(bm_data) > 0:
                max_idx = bm_data["value"].idxmax()
                min_idx = bm_data["value"].idxmin()
                for idx in [max_idx, min_idx]:
                    row = bm_data.loc[idx]
                    ax.annotate(f"{row['value']}\n{row['date'].date()}",
                                (row['date'], row['value']),
                                textcoords="offset points", xytext=(0,10),
                                ha='center', fontsize=8)
            
            # Reference ranges
            if show_standard_ref and bm in default_ref_ranges:
                low, high = default_ref_ranges[bm]
                ax.fill_between(bm_data["date"], low, high, alpha=0.1, color="blue", label="Standard range")
            if show_custom_ref and bm in custom_ref_ranges:
                low, high = custom_ref_ranges[bm]
                ax.fill_between(bm_data["date"], low, high, alpha=0.15, color="green", label="Custom range")
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    elif overlay_option == "Biomarkers within selected categories":
        fig, ax = plt.subplots(figsize=(12,6))
        for cat in selected_categories:
            cat_data = df[df["category"]==cat]
            for bm in cat_data["biomarker"].unique():
                bm_data = cat_data[cat_data["biomarker"]==bm]
                ax.plot(bm_data["date"], bm_data["value"], marker="o", label=f"{bm} ({cat})")
                
                if show_annotations and len(bm_data) > 0:
                    max_idx = bm_data["value"].idxmax()
                    min_idx = bm_data["value"].idxmin()
                    for idx in [max_idx, min_idx]:
                        row = bm_data.loc[idx]
                        ax.annotate(f"{row['value']}\n{row['date'].date()}",
                                    (row['date'], row['value']),
                                    textcoords="offset points", xytext=(0,10),
                                    ha='center', fontsize=8)
                
                if show_standard_ref and bm in default_ref_ranges:
                    low, high = default_ref_ranges[bm]
                    ax.fill_between(bm_data["date"], low, high, alpha=0.1, color="blue")
                if show_custom_ref and bm in custom_ref_ranges:
                    low, high = custom_ref_ranges[bm]
                    ax.fill_between(bm_data["date"], low, high, alpha=0.15, color="green")
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    else:  # Individual categories
        fig, axes = plt.subplots(len(selected_categories),1, figsize=(12, 3*len(selected_categories)), sharex=True)
        axes = np.atleast_1d(axes)
        for i, cat in enumerate(selected_categories):
            ax = axes[i]
            cat_data = df[df["category"]==cat]
            for bm in cat_data["biomarker"].unique():
                bm_data = cat_data[cat_data["biomarker"]==bm]
                ax.plot(bm_data["date"], bm_data["value"], marker="o", label=bm)
                
                if show_annotations and len(bm_data) > 0:
                    max_idx = bm_data["value"].idxmax()
                    min_idx = bm_data["value"].idxmin()
                    for idx in [max_idx, min_idx]:
                        row = bm_data.loc[idx]
                        ax.annotate(f"{row['value']}\n{row['date'].date()}",
                                    (row['date'], row['value']),
                                    textcoords="offset points", xytext=(0,10),
                                    ha='center', fontsize=8)
                
                if show_standard_ref and bm in default_ref_ranges:
                    low, high = default_ref_ranges[bm]
                    ax.fill_between(bm_data["date"], low, high, alpha=0.1, color="blue")
                if show_custom_ref and bm in custom_ref_ranges:
                    low, high = custom_ref_ranges[bm]
                    ax.fill_between(bm_data["date"], low, high, alpha=0.15, color="green")
            
            ax.set_ylabel(cat)
            ax.legend(fontsize=8)
        
        axes[-1].set_xlabel("Date")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

# Call the plotting function
plot_biomarkers(filtered_labs, overlay_option)

# ----------------------------
# 7️⃣ Detailed Table for Hover-like inspection
# ----------------------------
st.subheader("Detailed Lab Data")
st.dataframe(filtered_labs.reset_index(drop=True))
