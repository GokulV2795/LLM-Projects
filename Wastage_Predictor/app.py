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
from datetime import datetime, timedelta
import base64

# Your modules
from snowflake_read import fetch_recent_wastage
from email_sender import send_email_smtp

load_dotenv()

# OpenRouter
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("OPENROUTER_API_KEY missing → https://openrouter.ai/keys")
    st.stop()

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

st.set_page_config(page_title="Wastage Monitor Pro", layout="wide")
st.title("Inventory Wastage Predictor & Auto-Email Report")
st.markdown("**Uses your real Snowflake data • Trend Charts • Auto-Email with PNGs + CSV**")

# Sidebar
days = st.sidebar.slider("Days to Analyze", 3, 30, 14)
model = st.sidebar.selectbox("AI Model", ["meta-llama/llama-3.3-70b-instruct", "google/gemini-2.0-flash-exp"], index=0)

# === AI PREDICTION ===
def predict_wastage(df: pd.DataFrame):
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["EXPIRY_DATE"] = pd.to_datetime(df["EXPIRY_DATE"])

    # Calculate real metrics
    df["DAYS_TO_EXPIRY"] = (df["EXPIRY_DATE"] - datetime.now()).dt.days
    df["STOCK_DAYS_COVER"] = df["CLOSING_INVENTORY"] / (df["DAILY_SALES_RATE"].replace(0, 0.1))
    df["WASTAGE_RISK"] = (
        (df["DAYS_TO_EXPIRY"] < 7).astype(int) * 0.5 +
        (df["STOCK_DAYS_COVER"] > 30).astype(int) * 0.3 +
        (df["WASTAGE_UNITS_RATE"] > 0.1).astype(int) * 0.2
    )

    # AI Enhancement
    sample = df.head(30).to_dict("records")
    prompt = f"""Analyze these inventory items and return JSON array with predictions.

Rules:
- HIGH: <7 days to expiry OR very slow moving
- MEDIUM: 7–14 days
- LOW: >14 days and good turnover

Return only JSON:
[
  {{"SKU_ID": "ABC123", "WASTAGE_PREDICTOR": "HIGH", "RISK_SCORE": 0.92, "DAYS_TO_EXPIRY_EST": 4, "AI_RECOMMENDATION": "Donate or 70% off"}}
]

Data:
{json.dumps(sample, default=str)}
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1].split("```")[0]
        preds = json.loads(content)
        pred_df = pd.DataFrame(preds if isinstance(preds, list) else preds.get("predictions", []))
        df = df.merge(pred_df, on="SKU_ID", how="left")
    except:
        pass  # fallback to rule-based

    # Final columns
    df["WASTAGE_PREDICTOR"] = df["WASTAGE_PREDICTOR"].fillna(
        pd.cut(df["WASTAGE_RISK"], bins=[0, 0.4, 0.7, 1], labels=["LOW", "MEDIUM", "HIGH"])
    )
    df["RISK_SCORE"] = df["RISK_SCORE"].fillna(df["WASTAGE_RISK"])
    df["DAYS_TO_EXPIRY_EST"] = df["DAYS_TO_EXPIRY_EST"].fillna(df["DAYS_TO_EXPIRY"])
    df["AI_RECOMMENDATION"] = df["AI_RECOMMENDATION"].fillna("Monitor stock")

    return df

# === CHARTS + PNG + EMAIL ===
def generate_report_and_email(df: pd.DataFrame):
    figs = []

    # 1. Wastage Trend (7-day moving avg)
    trend = df.groupby("DATE")["WASTED_UNITS"].sum().reset_index()
    trend["7D_MA"] = trend["WASTED_UNITS"].rolling(7, min_periods=1).mean()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=trend["DATE"], y=trend["WASTED_UNITS"], name="Daily Wasted"))
    fig1.add_trace(go.Scatter(x=trend["DATE"], y=trend["7D_MA"], name="7-Day Avg", line=dict(dash="dash")))
    fig1.update_layout(title="Wastage Trend Over Time", xaxis_title="Date", yaxis_title="Units Wasted")
    st.plotly_chart(fig1, use_container_width=True)
    figs.append(("01_Wastage_Trend.png", fig1))

    # 2. Expiry Countdown
    exp = df[df["DAYS_TO_EXPIRY"] < 15].sort_values("DAYS_TO_EXPIRY")
    fig2 = px.bar(exp.head(15), x="SKU_NAME", y="DAYS_TO_EXPIRY", color="WASTAGE_PREDICTOR",
                  title="Items Expiring Soon (<15 days)")
    st.plotly_chart(fig2, use_container_width=True)
    figs.append(("02_Expiring_Soon.png", fig2))

    # 3. Risk Distribution
    fig3 = px.pie(df["WASTAGE_PREDICTOR"].value_counts(), names=df["WASTAGE_PREDICTOR"].value_counts().index,
                  title="Current Risk Distribution")
    st.plotly_chart(fig3, use_container_width=True)
    figs.append(("03_Risk_Distribution.png", fig3))

    # 4. Top 10 Slow Moving
    slow = df.nlargest(10, "STOCK_DAYS_COVER")
    fig4 = px.bar(slow, x="SKU_NAME", y="STOCK_DAYS_COVER", title="Slowest Moving Items (Days Cover)")
    st.plotly_chart(fig4, use_container_width=True)
    figs.append(("04_Slow_Moving.png", fig4))

    # High risk alert
    high = df[df["WASTAGE_PREDICTOR"] == "HIGH"]
    st.error(f"URGENT: {len(high)} HIGH-RISK ITEMS!")
    st.dataframe(high[["SKU_NAME", "CATEGORY", "LOCATION", "DAYS_TO_EXPIRY_EST", "AI_RECOMMENDATION"]])

    # === AUTO EMAIL WITH PNGs + CSV ===
    if st.button("Send Full Report via Email", type="primary"):
        with st.spinner("Preparing report..."):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as zf:
                # Add charts
                for name, fig in figs:
                    png = fig.to_image(format="png", width=1400, height=700, scale=2)
                    zf.writestr(name, png)
                # Add CSV
                csv_bytes = df.to_csv(index=False).encode()
                zf.writestr("Wastage_Report_Data.csv", csv_bytes)
            buffer.seek(0)

            # Email
            manager = os.getenv("WAREHOUSE_MANAGER_EMAIL")
            if not manager:
                st.error("WAREHOUSE_MANAGER_EMAIL not set!")
            else:
                try:
                    send_email_smtp(
                        subject=f"Wastage Alert: {len(high)} High-Risk Items ({datetime.now():%Y-%m-%d})",
                        body=f"""
Inventory Wastage Report

• Analysis Period: Last {days} days
• HIGH Risk Items: {len(high)}
• Total Items: {len(df):,}

Attached:
- 4 trend & risk charts (PNG)
- Full data (CSV)

Automated by Wastage Monitor Pro.
                        """.strip(),
                        to=manager,
                        attachments=[("Wastage_Report_Full.zip", buffer.getvalue(), "application/zip")]
                    )
                    st.success(f"Full report emailed to {manager}")
                except Exception as e:
                    st.error(f"Email failed: {e}")

# === MAIN ===
if st.sidebar.button("Run Analysis Now", type="primary"):
    with st.spinner("Fetching data..."):
        data = fetch_recent_wastage(days)

    if data is None or data.empty:
        st.error("No data from Snowflake.")
        st.stop()

    st.success(f"Analyzing {len(data):,} records...")
    result = predict_wastage(data)
    generate_report_and_email(result)

else:
    st.info("Click **Run Analysis Now** → Get charts + auto-email report")
    st.markdown("### Features\n- Real trend charts\n- Uses your exact columns\n- Auto-email with PNGs + CSV\n- No setup needed")