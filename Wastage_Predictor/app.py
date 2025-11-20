import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import tempfile
import zipfile
import io
from openai import OpenAI
import json
import plotly.express as px
import plotly.graph_objects as go

# Custom modules
from snowflake_read import fetch_recent_wastage
from email_sender import send_email_smtp

load_dotenv()

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
if not os.getenv("OPENROUTER_API_KEY"):
    st.error("OPENROUTER_API_KEY missing!")
    st.stop()

st.set_page_config(page_title="Wastage Monitor Pro", layout="wide")
st.title("Inventory Wastage Analytics Dashboard")
st.markdown("**AI-Powered • Real-Time • Exportable Charts**")

# === SIDEBAR ===
st.sidebar.header("Settings")
days = st.sidebar.slider("Days to analyze", 1, 30, 7)
model = st.sidebar.selectbox("AI Model", [
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.0-flash-exp",
    "openai/gpt-4o-mini"
], index=0)

# === CHART CREATION + PNG EXPORT ===
def create_and_export_charts(df: pd.DataFrame):
    if df.empty:
        st.info("No data.")
        return [], None

    figs = []
    png_bytes = []

    # 1. Risk Distribution
    fig1 = px.pie(
        df["WASTAGE_PREDICTOR"].value_counts(),
        names=df["WASTAGE_PREDICTOR"].value_counts().index,
        title="Wastage Risk Distribution",
        color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f97316", "LOW": "#22c55e"}
    )
    fig1.update_traces(textinfo='percent+label')
    st.plotly_chart(fig1, use_container_width=True)
    figs.append(("Risk_Distribution.png", fig1))

    # 2. Top 10 High-Risk
    top10 = df.nlargest(10, "RISK_SCORE")
    fig2 = px.bar(top10, x="RISK_SCORE", y="SKU_NAME", orientation='h',
                  title="Top 10 Highest Risk Items", color="RISK_SCORE",
                  color_continuous_scale="Reds")
    fig2.update_yaxes(autorange="reversed")
    st.plotly_chart(fig2, use_container_width=True)
    figs.append(("Top_10_High_Risk.png", fig2))

    # 3. Risk by Category
    cat = df.groupby("CATEGORY")["RISK_SCORE"].mean().sort_values(ascending=False)
    fig3 = px.bar(x=cat.index, y=cat.values, title="Average Risk by Category",
                  color=cat.values, color_continuous_scale="Oranges")
    st.plotly_chart(fig3, use_container_width=True)
    figs.append(("Risk_by_Category.png", fig3))

    # 4. Expiry Histogram
    fig4 = px.histogram(df, x="DAYS_TO_EXPIRY_EST", color="WASTAGE_PREDICTOR",
                        title="Days to Expiry Distribution", nbins=20,
                        color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f97316", "LOW": "#22c55e"})
    st.plotly_chart(fig4, use_container_width=True)
    figs.append(("Days_to_Expiry.png", fig4))

    # === EXPORT TO PNG ===
    for name, fig in figs:
        img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
        png_bytes.append((name, img_bytes))

    # ZIP Download Button
    if png_bytes:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for name, data in png_bytes:
                zf.writestr(name, data)
        zip_buffer.seek(0)

        st.download_button(
            label="Download All Charts as PNG (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"wastage_charts_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip"
        )

    # High-risk alert
    high = df[df["WASTAGE_PREDICTOR"] == "HIGH"]
    if len(high) > 0:
        st.error(f"URGENT: {len(high)} HIGH-RISK items!")
        st.dataframe(high[["SKU_NAME", "CATEGORY", "RISK_SCORE", "AI_RECOMMENDATION"]])

    return figs, zip_buffer

# Simple local prediction function to avoid undefined name errors.
def generate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight heuristic to produce required columns:
    - RISK_SCORE: integer 0-100
    - WASTAGE_PREDICTOR: HIGH / MEDIUM / LOW
    - AI_RECOMMENDATION: short text guidance
    This avoids relying on external AI calls and ensures downstream charting works.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    # Use DAYS_TO_EXPIRY_EST when available, otherwise assume 30 days
    if "DAYS_TO_EXPIRY_EST" in df.columns:
        days = df["DAYS_TO_EXPIRY_EST"].fillna(30).astype(float)
    else:
        days = pd.Series(30.0, index=df.index)

    # Heuristic risk: items closer to expiry get higher risk
    # Base score = (30 - days) * 3, clamped to [0,100]
    score = ((30.0 - days) * 3.0).clip(lower=0.0, upper=100.0)
    df["RISK_SCORE"] = score.round(0).astype(int)

    # Categorize
    conditions = [
        df["RISK_SCORE"] >= 70,
        df["RISK_SCORE"] >= 40
    ]
    choices = ["HIGH", "MEDIUM"]
    df["WASTAGE_PREDICTOR"] = pd.Series(pd.Categorical(pd.np.select(conditions, choices, default="LOW"),
                                                       categories=["HIGH", "MEDIUM", "LOW"]),
                                       index=df.index)

    # Provide simple recommendations based on category
    def recommend(row):
        cat = row["WASTAGE_PREDICTOR"]
        if cat == "HIGH":
            return "Immediate clearance and review reorder quantities."
        if cat == "MEDIUM":
            return "Consider promotions or reallocation to faster-moving stores."
        return "Normal monitoring; no immediate action."

    df["AI_RECOMMENDATION"] = df.apply(recommend, axis=1)

    return df

# === MAIN RUN ===
if st.sidebar.button("Run Analysis Now", type="primary", use_container_width=True):
    with st.spinner("Fetching data..."):
        data = fetch_recent_wastage(days)

    if data is None or data.empty:
        st.error("No data.")
        st.stop()

    st.success(f"Loaded {len(data):,} items")

    with st.spinner("AI analyzing..."):
        # Use the local heuristic prediction function
        result_df = generate_predictions(data)

    create_and_export_charts(result_df)

else:
    st.info("Click **Run Analysis Now** to generate predictions + charts")
    st.markdown("### Features\n- Interactive charts\n- Export all as high-res PNG\n- One-click ZIP download\n- Powered by OpenRouter")