# ----------------------------
# Streamlit Biomarker Explorer v5 - Plotly Interactive Overlay
# ----------------------------

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(layout="wide", page_title="Biomarker Dashboard")
st.title("Biomarker & Symptom Explorer (Interactive)")

# ----------------------------
# 1️⃣ Load Data
# ----------------------------
labs_df = pd.read_csv("lab_data.csv", parse_dates=["date"])
events_df = pd.read_csv("events.csv", parse_dates=["start_date", "end_date"])

# ----------------------------
# 2️⃣ Default Reference Ranges
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
min_date = labs_df["date"].min()
max_date = labs_df["date"].max()
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
        min_val_f = float(min_val)
        max_val_f = float(max_val)
        user_range = st.sidebar.slider(
            f"{bm} range",
            min_value=min_val_f * 0.5,
            max_value=max_val_f * 2.0,
            value=(min_val_f, max_val_f)
        )
        custom_ref_ranges[bm] = user_range
    else:
        custom_ref_ranges[bm] = (None, None)

# Out-of-range filter
show_out_of_range = st.sidebar.checkbox("Show only out-of-range values?", value=False)

# ----------------------------
# 4️⃣ Filter Lab Data
# ----------------------------
filtered_labs = labs_df[
    (labs_df["category"].isin(selected_categories)) &
    (labs_df["biomarker"].isin(selected_biomarkers)) &
    (labs_df["date"] >= date_range[0]) &
    (labs_df["date"] <= date_range[1])
].copy()

if show_out_of_range:
    def out_of_range(row):
        low, high = custom_ref_ranges.get(row["biomarker"], (None, None))
        if low is not None and high is not None:
            return row["value"] < low or row["value"] > high
        return True
    filtered_labs = filtered_labs[filtered_labs.apply(out_of_range, axis=1)]

# ----------------------------
# 5️⃣ Event Track Plot (Matplotlib)
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
    ax_event.barh(i, width=end - start, left=start, height=0.8, color=color)
    ax_event.text(start + (end-start)/2, i, event["name"], ha="center", va="center", fontsize=9)

ax_event.barh([], [], color="#ffe066", label="Symptom")
ax_event.barh([], [], color="#ff9999", label="Infection")
ax_event.legend(loc="upper right", fontsize=9, framealpha=0.7)
plt.tight_layout()
st.pyplot(fig_event)

# ----------------------------
# 6️⃣ Interactive Biomarker Overlay (Plotly)
# ----------------------------
st.subheader("Interactive Biomarker Trends")

fig = go.Figure()

for bm in selected_biomarkers:
    bm_data = filtered_labs[filtered_labs["biomarker"] == bm]
    fig.add_trace(
        go.Scatter(
            x=bm_data["date"],
            y=bm_data["value"],
            mode="lines+markers",
            name=bm,
            hovertemplate='Date: %{x}<br>Value: %{y}<extra></extra>'
        )
    )
    # Optional reference range shading
    if show_ref and bm in custom_ref_ranges:
        low, high = custom_ref_ranges[bm]
        fig.add_trace(
            go.Scatter(
                x=bm_data["date"],
                y=[high]*len(bm_data),
                fill=None,
                mode="lines",
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bm_data["date"],
                y=[low]*len(bm_data),
                fill='tonexty',
                mode="lines",
                line=dict(color='rgba(0,0,0,0)'),
                fillcolor='rgba(100,100,200,0.1)',
                showlegend=False
            )
        )

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Value",
    legend_title="Biomarker",
    hovermode="x unified",
    template="plotly_white",
    height=600
)

st.plotly_chart(fig, use_container_width=True)
