"""
Low-Quality Milk Alert System
Detects declining Fat/SNF trends in milk quality data from Domo.
Generates AI-powered recommendations and sends formatted email alerts.
"""

import os
import base64
import json
import io
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

# Page setup
st.set_page_config(page_title="Low-Quality Milk Alert", layout="wide")
st.title("Low-Quality Milk Pattern Detector")
st.markdown("**Identifies farmers with dropping Fat/SNF levels • Triggers actionable email alerts**")

# AI config (OpenRouter fallback to rule-based)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = "https://openrouter.ai/api/v1" if OPENROUTER_API_KEY else None

# HTML email template for alerts
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

# AI recommendation via OpenRouter (with fallback)
def get_ai_recommendation(prompt: str, model: str = "gpt-4o-mini") -> str:
    if not OPENROUTER_BASE or not OPENROUTER_API_KEY:
        return (
            "Conduct an urgent field visit and audit feed/water quality. "
            "Perform adulteration tests; consider temporary suspension if issues persist."
        )
    
    url = f"{OPENROUTER_BASE}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 250
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices and isinstance(choices, list) and len(choices) > 0:
            content = choices[0].get("message", {}).get("content", "")
            return content.strip()
        
        # Fallback message
        return (
            "Urgent field visit recommended: audit feed and water quality, test for adulteration. "
            "Escalate to procurement if no improvement in 7 days."
        )
    except Exception:
        # Fallback message
        return (
            "Urgent field visit recommended: audit feed and water quality, test for adulteration. "
            "Escalate to procurement if no improvement in 7 days."
        )

# Domo authentication (OAuth client credentials with retries)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_domo_token() -> str:
    client_id = os.getenv("DOMO_CLIENT_ID")
    client_secret = os.getenv("DOMO_CLIENT_SECRET")
    if not all([client_id, client_secret]):
        raise ValueError("Missing DOMO_CLIENT_ID or DOMO_CLIENT_SECRET in .env")
    
    auth_url = "https://api.domo.com/oauth/token"
    auth_data = {"grant_type": "client_credentials", "scope": "data"}
    basic_auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    
    resp = requests.post(
        auth_url, 
        data=auth_data, 
        headers={"Authorization": f"Basic {basic_auth}"}, 
        timeout=15
    )
    resp.raise_for_status()
    
    token_data = resp.json()
    token = token_data.get("access_token")
    if not token:
        raise ValueError("No access_token in Domo response")
    
    return token

# Load dataset from Domo (corrected endpoint: /data for CSV export)
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_domo_data() -> pd.DataFrame:
    dataset_id = os.getenv("DOMO_DATASET_ID")
    if not dataset_id:
        st.error("DOMO_DATASET_ID missing in .env")
        return pd.DataFrame()
    
    try:
        token = fetch_domo_token()
    except Exception as e:
        st.error(f"Domo authentication error: {e}")
        return pd.DataFrame()
    
    # Corrected: Use /data endpoint for row export as CSV
    data_url = f"https://api.domo.com/v1/datasets/{dataset_id}/data"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "text/csv"
    }
    params = {
        "includeHeader": "true",  # Include column headers in CSV
        "streams": "true"         # Stream for large datasets
    }
    
    try:
        resp = requests.get(data_url, headers=headers, params=params, timeout=60, stream=True)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 404:
            st.error(f"Dataset not found (ID: {dataset_id}). Confirm ID in Domo and 'data' scope in OAuth app.")
        elif resp.status_code == 403:
            st.error("Access denied (403). Verify OAuth scopes include 'data' and dataset permissions.")
        else:
            st.error(f"HTTP {resp.status_code}: {resp.text[:500]}...")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Dataset fetch failed: {e}")
        return pd.DataFrame()
    
    # Parse streamed CSV into DataFrame
    try:
        csv_stream = io.StringIO(resp.text)
        df = pd.read_csv(csv_stream)
        
        # Normalize column names (strip spaces, replace multiples with underscores)
        df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
        
        st.success(f"Successfully loaded {len(df):,} rows from Domo dataset.")
    except Exception as e:
        st.error(f"CSV parsing failed: {e}")
        st.text(resp.text[:1000])  # Debug: Show sample response
        return pd.DataFrame()
    
    # Safely parse date columns if present
    for col in ("Date", "Delivery_Date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Optional: Filter to recent data (e.g., last 90 days) in Pandas
    if "Date" in df.columns and not df["Date"].empty:
        cutoff = datetime.now() - timedelta(days=90)
        df = df[df["Date"] >= cutoff]
    
    return df

# Identify risky farmers based on quality drops or flags
def identify_risky_farmers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    df_work = df.copy()
    
    # Initialize missing columns to defaults
    for col in ["Fat_Content", "SNF_Content", "Fat_Content_3DaysAgo", "SNF_Content_3DaysAgo", "Trend_Days", "Flag"]:
        if col not in df_work.columns:
            df_work[col] = 0
    
    # Calculate drops and low thresholds
    df_work["Fat_Drop"] = (
        pd.to_numeric(df_work["Fat_Content_3DaysAgo"], errors="coerce").fillna(0) - 
        pd.to_numeric(df_work["Fat_Content"], errors="coerce").fillna(0)
    )
    df_work["SNF_Drop"] = (
        pd.to_numeric(df_work["SNF_Content_3DaysAgo"], errors="coerce").fillna(0) - 
        pd.to_numeric(df_work["SNF_Content"], errors="coerce").fillna(0)
    )
    df_work["Low_Fat"] = pd.to_numeric(df_work["Fat_Content"], errors="coerce") < 3.3
    df_work["Low_SNF"] = pd.to_numeric(df_work["SNF_Content"], errors="coerce") < 8.0
    
    # Flag risky patterns
    risky_mask = (
        (pd.to_numeric(df_work.get("Flag", 0), errors="coerce") == 1) |
        (df_work["Fat_Drop"] > 0.4) |
        (df_work["SNF_Drop"] > 0.6) |
        (df_work["Low_Fat"] & (pd.to_numeric(df_work.get("Trend_Days", 0), errors="coerce") >= 5)) |
        (df_work["Low_SNF"] & (pd.to_numeric(df_work.get("Trend_Days", 0), errors="coerce") >= 5))
    )
    
    risky_df = df_work[risky_mask].copy()
    if risky_df.empty:
        return risky_df
    
    # Get latest record per farmer
    risky_df = risky_df.sort_values("Date", na_position="last")
    if "Farmer_ID" in risky_df.columns:
        risky_df = risky_df.groupby("Farmer_ID", as_index=False).tail(1)
    
    return risky_df

# Generate recommendation prompt and call AI
def create_recommendation(row: pd.Series) -> str:
    trend_days = int(pd.to_numeric(row.get("Trend_Days", 0), errors="coerce") or 0)
    fat_drop = float(row.get("Fat_Content_3DaysAgo", 0)) - float(row.get("Fat_Content", 0))
    snf_drop = float(row.get("SNF_Content_3DaysAgo", 0)) - float(row.get("SNF_Content", 0))
    
    prompt = (
        f"Farmer: {row.get('Farmer_Name', 'Unknown')} (ID: {row.get('Farmer_ID', '')})\n"
        f"Location: {row.get('Region', '')} / {row.get('Panchayat', '')}, {row.get('State', '')}\n"
        f"Trend Duration: {trend_days} days\n\n"
        f"Current Values:\n"
        f"• Fat: {float(row.get('Fat_Content', 0)):.2f}% (was {float(row.get('Fat_Content_3DaysAgo', 0)):.2f} → drop of {fat_drop:.2f})\n"
        f"• SNF: {float(row.get('SNF_Content', 0)):.2f}% (was {float(row.get('SNF_Content_3DaysAgo', 0)):.2f} → drop of {snf_drop:.2f})\n\n"
        "This farmer shows sustained low quality. Suggest 2–4 sentences of professional actions: "
        "field visit, cattle feed audit, water quality check, adulteration test, or temporary suspension."
    )
    return get_ai_recommendation(prompt)

# Send formatted HTML email alert
def dispatch_alert(row: pd.Series, recommendation: str) -> bool:
    recipients = [email.strip() for email in os.getenv("QUALITY_MANAGERS", "").split(",") if email.strip()]
    if not recipients:
        st.error("No recipients set in QUALITY_MANAGERS env var")
        return False
    
    # Build decline summary
    decline_items = []
    if float(row.get("Fat_Drop", 0)) > 0.4:
        decline_items.append(f"Fat ↓{float(row.get('Fat_Drop')):.2f}%")
    if float(row.get("SNF_Drop", 0)) > 0.6:
        decline_items.append(f"SNF ↓{float(row.get('SNF_Drop')):.2f}%")
    if int(pd.to_numeric(row.get("Flag", 0), errors="coerce") or 0) == 1:
        decline_items.append("System Flagged")
    
    decline_str = " | ".join(decline_items) if decline_items else "Decline detected"
    
    # Render HTML body
    html_body = template.render(
        farmer_name=row.get("Farmer_Name", ""),
        farmer_id=row.get("Farmer_ID", ""),
        region=row.get("Region", ""),
        panchayat=row.get("Panchayat", ""),
        state=row.get("State", ""),
        trend_days=int(pd.to_numeric(row.get("Trend_Days", 0), errors="coerce") or 0),
        decline_type=decline_str,
        recommendation=recommendation
    )
    
    # SMTP setup
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = os.getenv("SMTP_PORT")
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    if not all([smtp_host, smtp_port, smtp_user, smtp_pass]):
        st.error("Missing SMTP config (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS)")
        return False
    
    # Build and send message
    msg = MIMEMultipart()
    msg["From"] = f"{os.getenv('ALERT_FROM_NAME', 'Low-Quality Alert')} <{smtp_user}>"
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = "LOW-QUALITY MILK ALERT – IMMEDIATE ACTION REQUIRED"
    msg.attach(MIMEText(html_body, "html"))
    
    try:
        server = smtplib.SMTP(smtp_host, int(smtp_port))
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, recipients, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email dispatch failed: {e}")
        return False

# Main execution
data_df = fetch_domo_data()
if data_df.empty:
    st.stop()

alert_candidates = identify_risky_farmers(data_df)
st.write(f"**Detected {len(alert_candidates)} farmers with quality decline risks**")

if not alert_candidates.empty:
    for idx, farmer_row in alert_candidates.iterrows():
        with st.expander(
            f"{farmer_row.get('Farmer_Name', 'Unknown')} • {farmer_row.get('Farmer_ID', '')} • "
            f"Trend: {int(pd.to_numeric(farmer_row.get('Trend_Days', 0), errors='coerce') or 0)} days"
        ):
            cols = st.columns([2, 2, 1])
            with cols[0]:
                fat_drop = float(farmer_row.get("Fat_Content_3DaysAgo", 0)) - float(farmer_row.get("Fat_Content", 0))
                snf_drop = float(farmer_row.get("SNF_Content_3DaysAgo", 0)) - float(farmer_row.get("SNF_Content", 0))
                st.write(f"**Fat:** {float(farmer_row.get('Fat_Content', 0)):.2f}% (↓{fat_drop:.2f})")
                st.write(f"**SNF:** {float(farmer_row.get('SNF_Content', 0)):.2f}% (↓{snf_drop:.2f})")
            with cols[1]:
                st.write(f"**Region:** {farmer_row.get('Region', '')}")
                st.write(f"**Panchayat:** {farmer_row.get('Panchayat', '')}")
            with cols[2]:
                if st.button(f"Send Alert: {farmer_row.get('Farmer_ID', '')}", key=f"alert_{idx}"):
                    with st.spinner("Dispatching email..."):
                        rec_text = create_recommendation(farmer_row)
                        if dispatch_alert(farmer_row, rec_text):
                            st.success("Alert dispatched successfully!")
                        else:
                            st.error("Alert dispatch failed")
            
            st.markdown("**AI Recommendation:**")
            st.write(create_recommendation(farmer_row))
else:
    st.success("No quality decline patterns found in current data.")

# Sidebar: Bulk alert option
if st.sidebar.checkbox("Enable Bulk Alerts"):
    if st.sidebar.button("Dispatch All Alerts", type="primary"):
        dispatched = 0
        for _, farmer_row in alert_candidates.iterrows():
            rec_text = create_recommendation(farmer_row)
            if dispatch_alert(farmer_row, rec_text):
                dispatched += 1
        st.sidebar.success(f"Dispatched {dispatched} bulk alerts!")