"""
Low-Quality Milk Alert System – Final Version
Shows AI recommendations + one-click email send
"""

import os
import io
import base64
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

import streamlit as st
import pandas as pd
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from jinja2 import Template

load_dotenv()

st.set_page_config(page_title="Low-Quality Milk Alert", layout="wide")
st.title("Low-Quality Milk Pattern Detector")
st.markdown("**Detects declining Fat/SNF • Shows AI recommendations • One-click email alerts**")

# ============================= EMAIL TEMPLATE =============================
EMAIL_TEMPLATE = """
<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
  <h2>Dear Quality & Procurement Team,</h2>
  <h1 style="color: #d32f2f;">LOW-QUALITY MILK ALERT – IMMEDIATE ACTION REQUIRED</h1>
  <h3>Farmer Details:</h3>
  <ul>
    <li><strong>Farmer Name:</strong> {{ farmer_name }}</li>
    <li><strong>Farmer ID:</strong> {{ farmer_id }}</li>
    <li><strong>Region / Panchayat:</strong> {{ region }} / {{ panchayat }}</li>
    <li><strong>State:</strong> {{ state }}</li>
    <li><strong>Decline Detected:</strong> {{ decline_type }}</li>
    <li><strong>Trend Over Last {{ trend_days }} Days</strong></li>
  </ul>
  <h3>RECOMMENDED ACTION:</h3>
  <p><strong>{{ recommendation }}</strong></p>
  <hr>
  <p><em>This is an automated alert from the <strong>Low-Quality Milk Alert System</strong>.</em></p>
  <p>Please confirm receipt and planned action.</p>
  <br>
  <p>Thank you,<br><strong>Quality Assurance Team</strong></p>
</body>
</html>
"""
template = Template(EMAIL_TEMPLATE)

# ============================= DOMO FETCH (Working) =============================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_domo_token():
    cid = os.getenv("DOMO_CLIENT_ID")
    csec = os.getenv("DOMO_CLIENT_SECRET")
    auth = base64.b64encode(f"{cid}:{csec}".encode()).decode()
    resp = requests.post(
        "https://api.domo.com/oauth/token",
        data={"grant_type": "client_credentials", "scope": "data"},
        headers={"Authorization": f"Basic {auth}"},
        timeout=15
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

@st.cache_data(ttl=1800)
def fetch_data() -> pd.DataFrame:
    dsid = os.getenv("DOMO_DATASET_ID")
    token = get_domo_token()
    url = f"https://api.domo.com/v1/datasets/{dsid}/data"
    headers = {"Authorization": f"Bearer {token}", "Accept": "text/csv"}
    params = {"includeHeader": "true", "streams": "true"}

    resp = requests.get(url, headers=headers, params=params, timeout=60, stream=True)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = df.columns.str.strip()
    for col in ("Date", "Delivery_Date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    st.success(f"Loaded {len(df):,} rows from Domo")
    return df

# ============================= AI RECOMMENDATION =============================
def get_recommendation(row: pd.Series) -> str:
    prompt = f"""
Farmer: {row.get('Farmer_Name','')} (ID: {row.get('Farmer_ID','')})
Location: {row.get('Region','')} / {row.get('Panchayat','')}, {row.get('State','')}
Trend: {int(row.get('Trend_Days',0))} days
Fat drop: {row.get('Fat_Content_3DaysAgo',0) - row.get('Fat_Content',0):.2f}%
SNF drop: {row.get('SNF_Content_3DaysAgo',0) - row.get('SNF_Content',0):.2f}%

Provide 4-6 numbered, actionable recommendations (e.g., field visit, feed audit, adulteration test).
Be direct and professional.
"""
    # Simple fallback if no OpenRouter key
    if not os.getenv("OPENROUTER_API_KEY"):
        return (
            "1. Conduct immediate field visit within 48 hours\n"
            "2. Audit cattle feed quality and storage\n"
            "3. Test water source for contamination\n"
            "4. Perform adulteration test on next 3 collections\n"
            "5. Issue warning letter to farmer\n"
            "6. Suspend collection if no improvement in 7 days"
        )

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "model": "meta-llama/llama-3.3-70b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 400
            },
            headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
            timeout=25
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except:
        return "AI unavailable – use standard protocol: field visit + feed/water audit + adulteration test."

# ============================= SEND EMAIL =============================
def send_email(row: pd.Series, recommendation: str) -> bool:
    recipients = [e.strip() for e in os.getenv("QUALITY_MANAGERS", "").split(",") if e.strip()]
    if not recipients:
        st.error("No recipients configured")
        return False

    decline = []
    if row.get("Fat_Drop", 0) > 0.4: decline.append(f"Fat ↓{row['Fat_Drop']:.2f}%")
    if row.get("SNF_Drop", 0) > 0.6: decline.append(f"SNF ↓{row['SNF_Drop']:.2f}%")
    if row.get("Flag") == 1: decline.append("System Flagged")

    html = template.render(
        farmer_name=row.get("Farmer_Name", ""),
        farmer_id=row.get("Farmer_ID", ""),
        region=row.get("Region", ""),
        panchayat=row.get("Panchayat", ""),
        state=row.get("State", ""),
        trend_days=int(row.get("Trend_Days", 0)),
        decline_type=" | ".join(decline),
        recommendation=recommendation.replace("\n", "<br>")
    )

    msg = MIMEMultipart()
    msg["From"] = f"{os.getenv('ALERT_FROM_NAME', 'Milk Quality Alert')} <{os.getenv('SMTP_USER')}>"
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = "LOW-QUALITY MILK ALERT – IMMEDIATE ACTION REQUIRED"
    msg.attach(MIMEText(html, "html"))

    try:
        server = smtplib.SMTP(os.getenv("SMTP_HOST"), int(os.getenv("SMTP_PORT")))
        server.starttls()
        server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
        server.sendmail(os.getenv("SMTP_USER"), recipients, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email failed: {e}")
        return False

# ============================= MAIN LOGIC =============================
df = fetch_data()
if df.empty:
    st.stop()

# Calculate drops
for col in ["Fat_Content", "SNF_Content", "Fat_Content_3DaysAgo", "SNF_Content_3DaysAgo", "Trend_Days", "Flag"]:
    if col not in df.columns:
        df[col] = 0

df["Fat_Drop"] = df["Fat_Content_3DaysAgo"].astype(float) - df["Fat_Content"].astype(float)
df["SNF_Drop"] = df["SNF_Content_3DaysAgo"].astype(float) - df["SNF_Content"].astype(float)

# Risky farmers
risky = df[
    (df["Flag"] == 1) |
    (df["Fat_Drop"] > 0.4) |
    (df["SNF_Drop"] > 0.6) |
    ((df["Fat_Content"] < 3.3) & (df["Trend_Days"] >= 5)) |
    ((df["SNF_Content"] < 8.0) & (df["Trend_Days"] >= 5))
].copy()

if "Farmer_ID" in risky.columns:
    risky = risky.sort_values("Date", ascending=False).groupby("Farmer_ID").first().reset_index()

st.write(f"**{len(risky)} farmers require immediate attention**")

for _, row in risky.iterrows():
    with st.expander(f"{row['Farmer_Name']} • {row['Farmer_ID']} • Trend: {int(row['Trend_Days'])} days", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**Fat:** {row['Fat_Content']:.2f}% ↓{row['Fat_Drop']:.2f}%")
            st.write(f"**SNF:** {row['SNF_Content']:.2f}% ↓{row['SNF_Drop']:.2f}%")
            st.write(f"**Location:** {row['Region']} / {row['Panchayat']}, {row['State']}")

        with col2:
            if st.button("Send Alert Email", key=f"send_{row['Farmer_ID']}", type="primary"):
                with st.spinner("Sending..."):
                    rec = get_recommendation(row)
                    if send_email(row, rec):
                        st.success("Email sent successfully!")
                    else:
                        st.error("Failed to send email")

        # Show numbered recommendations
        st.markdown("### Recommended Actions")
        recommendation_text = get_recommendation(row)
        st.markdown(recommendation_text.replace("\n", "<br>"), unsafe_allow_html=True)

if risky.empty:
    st.success("No low-quality patterns detected today")