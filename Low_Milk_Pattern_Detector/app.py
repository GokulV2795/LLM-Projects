import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime, timedelta
from openai import OpenAI
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from jinja2 import Template

load_dotenv()

st.set_page_config(page_title="Low-Quality Milk Alert", layout="wide")
st.title("Low-Quality Milk Pattern Detector")
st.markdown("**Detects farmers with declining Fat/SNF • Sends professional alerts**")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

# === DOMO DATA LOADER ===
@st.cache_data(ttl=1800)
def load_data():
    cid, csec, dsid = os.getenv("DOMO_CLIENT_ID"), os.getenv("DOMO_CLIENT_SECRET"), os.getenv("DOMO_DATASET_ID")
    if not all([cid, csec, dsid]):
        st.error("Domo credentials missing")
        return pd.DataFrame()

    # Get access token
    auth_url = "https://api.domo.com/oauth/token"
    auth_payload = {"grant_type": "client_credentials", "client_id": cid, "client_secret": csec, "scope": "dataset"}
    auth_resp = requests.post(auth_url, json=auth_payload)
    
    if auth_resp.status_code != 200:
        st.error("Domo authentication failed")
        return pd.DataFrame()
    
    token = auth_resp.json()["access_token"]
    
    # Query dataset
    query_url = f"https://api.domo.com/v1/datasets/{dsid}/query"
    headers = {"Authorization": f"Bearer {token}"}
    query_payload = {"sql": "SELECT * FROM table ORDER BY Date DESC"}
    query_resp = requests.post(query_url, json=query_payload, headers=headers)
    
    if query_resp.status_code != 200:
        st.error("Failed to fetch data from Domo")
        return pd.DataFrame()
    
    df = pd.DataFrame(query_resp.json()["rows"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"])
    return df

# === EMAIL TEMPLATE (Exact format you showed) ===
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
  <p>Please confirm receipt and planned action (field visit, training, suspension, etc.).</p>

  <br>
  <p>Thank you,<br>
  <strong>Quality Assurance Team</strong></p>
</body>
</html>
"""
template = Template(EMAIL_TEMPLATE)

# === DETECT LOW-QUALITY PATTERNS ===
def detect_low_quality_patterns(df: pd.DataFrame):
    df = df.copy()
    
    # Rule 1: Significant drop in Fat or SNF compared to 3 days ago
    df["Fat_Drop"] = df["Fat_Content_3DaysAgo"] - df["Fat_Content"]
    df["SNF_Drop"] = df["SNF_Content_3DaysAgo"] - df["SNF_Content"]

    # Rule 2: Already flagged + sustained low values
    df["Low_Fat"] = df["Fat_Content"] < 3.3
    df["Low_SNF"] = df["SNF_Content"] < 8.0

    # High-risk farmers
    risky = df[
        (df["Flag"] == 1) |
        (df["Fat_Drop"] > 0.4) |
        (df["SNF_Drop"] > 0.6) |
        (df["Low_Fat"] & df["Trend_Days"] >= 5) |
        (df["Low_SNF"] & df["Trend_Days"] >= 5)
    ].copy()

    # Group by farmer and take latest record
    latest_risky = risky.sort_values("Date").groupby("Farmer_ID").tail(1)
    return latest_risky

# === AI RECOMMENDATION ===
def generate_recommendation(row: pd.Series) -> str:
    prompt = f"""
Farmer: {row['Farmer_Name']} (ID: {row['Farmer_ID']})
Location: {row['Region']} / {row['Panchayat']}, {row['State']}
Trend Duration: {int(row['Trend_Days'])} days

Current Values:
• Fat: {row['Fat_Content']:.2f}% (was {row['Fat_Content_3DaysAgo']:.2f} → drop of {row['Fat_Content_3DaysAgo'] - row['Fat_Content']:.2f})
• SNF: {row['SNF_Content']:.2f}% (was {row['SNF_Content_3DaysAgo']:.2f} → drop of {row['SNF_Content_3DaysAgo'] - row['SNF_Content']:.2f})

This farmer has been flagged for sustained low quality.

Write a clear, professional recommendation in 2–4 sentences.
Possible actions: field visit, cattle feed audit, water quality check, adulteration test, temporary suspension.
"""

    try:
        resp = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        return resp.choices[0].message.content.strip()
    except:
        return f"Urgent field visit required for {row['Farmer_Name']} due to sustained decline in Fat (-{row['Fat_Content_3DaysAgo'] - row['Fat_Content']:.2f}%) and SNF for {row['Trend_Days']} days."

# === SEND EMAIL ===
def send_alert(row: pd.Series, recommendation: str):
    recipients = [e.strip() for e in os.getenv("QUALITY_MANAGERS", "").split(",") if e.strip()]
    if not recipients:
        st.error("No recipients")
        return False

    decline = []
    if row["Fat_Drop"] > 0.4: decline.append(f"Fat ↓{row['Fat_Drop']:.2f}%")
    if row["SNF_Drop"] > 0.6: decline.append(f"SNF ↓{row['SNF_Drop']:.2f}%")
    if row["Flag"] == 1: decline.append("System Flagged")

    html = template.render(
        farmer_name=row["Farmer_Name"],
        farmer_id=row["Farmer_ID"],
        region=row["Region"],
        panchayat=row["Panchayat"],
        state=row["State"],
        trend_days=int(row["Trend_Days"]),
        decline_type=" | ".join(decline),
        recommendation=recommendation
    )

    msg = MIMEMultipart()
    msg["From"] = f"{os.getenv('ALERT_FROM_NAME')} <{os.getenv('SMTP_USER')}>"
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
        st.error(f"Failed: {e}")
        return False

# === MAIN ===
df = load_data()
if df.empty:
    st.stop()

risky_farmers = detect_low_quality_patterns(df)

st.write(f"**{len(risky_farmers)} farmers detected with declining milk quality**")

if not risky_farmers.empty:
    for _, row in risky_farmers.iterrows():
        with st.expander(f"{row['Farmer_Name']} • {row['Farmer_ID']} • Trend: {int(row['Trend_Days'])} days"):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**Fat:** {row['Fat_Content']:.2f}% (↓{row['Fat_Content_3DaysAgo'] - row['Fat_Content']:.2f})")
                st.write(f"**SNF:** {row['SNF_Content']:.2f}% (↓{row['SNF_Content_3DaysAgo'] - row['SNF_Content']:.2f})")
            with col2:
                st.write(f"**Region:** {row['Region']}")
                st.write(f"**Panchayat:** {row['Panchayat']}")
            with col3:
                if st.button("Send Alert", key=row['Farmer_ID']):
                    with st.spinner("Sending..."):
                        rec = generate_recommendation(row)
                        if send_alert(row, rec):
                            st.success(f"Alert sent for {row['Farmer_Name']}!")
            st.markdown("**AI Recommendation:**")
            st.write(generate_recommendation(row))
else:
    st.success("No low-quality patterns detected today")

# Auto-send all
if st.sidebar.checkbox("Auto-Send All Alerts"):
    if st.sidebar.button("Send All Now", type="primary"):
        sent = 0
        for _, row in risky_farmers.iterrows():
            rec = generate_recommendation(row)
            if send_alert(row, rec):
                sent += 1
        st.success(f"Sent {sent} automated alerts!")