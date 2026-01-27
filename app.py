# ----------------------------
# Streamlit Biomarker Dashboard v6 - Clean UI, Tabs, and Polished Layout
# ----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ----------------------------
# 1️⃣ Page Config
# ----------------------------
st.set_page_config(layout="wide", page_title="Biomarker Dashboard")
st.markdown("<h1 style='font-family:sans-serif; color:#2F4F4F;'>🧬 Biomarker & Symptom Explorer</h1>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------
# 2️⃣ Load Data
# ----------------------------
labs_df = pd.read_csv("lab_data.csv", parse_dates=["date"])
events_df = pd.read_csv("events.csv", parse_dates=["start_date", "end_date"])

# ----------------------------
# 3️⃣ Standard Reference Ranges
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
# 4️⃣ Sidebar Controls
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
    selected_biomarkers = st.multiselect("Select biomarkers", available_biomarkers, default=available_biomarkers[:5])

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

# ----------------------------
# 5️⃣ Filter Lab Data
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
# 6️⃣ Tabs: Events & Biomarkers
# ----------------------------
tab1, tab2 = st.tabs(["🦠 Symptoms / Events", "📊 Biomarker Trends"])

# ----------------------------
# Tab 1: Events
# ----------------------------
with tab1:
    st.subheader("Symptoms / Infections Timeline")
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
# Tab 2: Biomarkers
# ----------------------------
with tab2:
    st.subheader("Biomarker Trends")

    def plot_biomarkers(df, overlay_option):
        if overlay_option == "All biomarkers":
            fig, ax = plt.subplots(figsize=(12,6))
            for bm in df["biomarker"].unique():
                bm_data = df[df["biomarker"]==bm]
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

        else:
            st.info("Other overlay options coming soon!")  # placeholder

    plot_biomarkers(filtered_labs, overlay_option)

    st.subheader("Detailed Lab Data")
    # Highlight out-of-range values
    def highlight_out_of_range(row):
        low, high = custom_ref_ranges.get(row["biomarker"], (None, None))
        if low is not None and high is not None:
            return ["background-color: yellow" if (row["value"] < low or row["value"] > high) else "" for _ in row]
        return [""]*len(row)

    st.dataframe(filtered_labs.style.apply(highlight_out_of_range, axis=1))
