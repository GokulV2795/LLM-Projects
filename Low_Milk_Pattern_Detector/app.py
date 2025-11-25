"""
Low-Quality Milk Alert System – FINAL EMAIL-ONLY VERSION
Zero crashes • Handles dirty data • Smart AI + Offline Fallback • Professional Email Alerts
"""

import os
import io
import base64
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from jinja2 import Template
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

# ============================= PAGE CONFIG =============================
st.set_page_config(page_title="Milk Quality Alert", layout="wide")
st.title("Low-Quality Milk Alert System")
st.markdown("**Detects Fat/SNF decline • Smart AI Actions • Professional Email Alerts**")

# ============================= EMAIL TEMPLATE =============================
EMAIL_TEMPLATE = """
<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
  <h2>Dear Quality & Procurement Team,</h2>
  <h1 style="color: #d32f2f;">LOW-QUALITY MILK ALERT – IMMEDIATE ACTION REQUIRED</h1>
  <h3>Farmer Details:</h3>
  <ul>
    <li><strong>Name:</strong> {{ farmer_name }}</li>
    <li><strong>ID:</strong> {{ farmer_id }}</li>
    <li><strong>Location:</strong> {{ region }} / {{ panchayat }}, {{ state }}</li>
    <li><strong>Decline Detected:</strong> {{ decline_type }}</li>
    <li><strong>Trend Duration:</strong> {{ trend_days }} days</li>
  </ul>
  <h3>Recommended Actions:</h3>
  <p><strong>{{ recommendation }}</strong></p>
  <hr>
  <p><em>This is an automated alert from the Low-Quality Milk Alert System.</em></p>
  <p>Please confirm receipt and planned action.</p>
  <br>
  <p>Thank you,<br><strong>Quality Assurance Team</strong></p>
</body>
</html>
"""
template = Template(EMAIL_TEMPLATE)

# ============================= DOMO DATA FETCH =============================
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
    if not dsid:
        st.error("DOMO_DATASET_ID missing in .env")
        return pd.DataFrame()

    try:
        token = get_domo_token()
        url = f"https://api.domo.com/v1/datasets/{dsid}/data"
        headers = {"Authorization": f"Bearer {token}", "Accept": "text/csv"}
        resp = requests.get(url, headers=headers, params={"includeHeader": "true"}, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = df.columns.str.strip()
        for c in ("Date", "Delivery_Date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        st.success(f"Loaded {len(df):,} records from Domo")
        return df
    except Exception as e:
        st.error(f"Domo connection failed: {e}")
        return pd.DataFrame()

# ============================= SAFE DATA HANDLING =============================
def safe_float(val, default=0.0):
    try:
        return float(str(val).strip()) if str(val).strip() not in ("", "None", "NaN") else default
    except:
        return default

def safe_int(val, default=0):
    try:
        return int(str(val).strip()) if str(val).strip() not in ("", "None", "NaN") else default
    except:
        return default

def is_flagged(flag_val):
    val = str(flag_val).strip().lower()
    return val in ("1", "yes", "true", "quality watch", "alert", "flagged", "high risk", "watch")

# ============================= SMART RECOMMENDATION ENGINE (NO CRASH) =============================
def get_recommendation(row: pd.Series) -> str:
    name = str(row.get("Farmer_Name", "Unknown Farmer")).strip()
    fid = str(row.get("Farmer_ID", "")).strip()
    loc = f"{row.get('Region','')} / {row.get('Panchayat','')}, {row.get('State','')}".strip(" ,/")

    trend_days = safe_int(row.get("Trend_Days", 0))
    fat_drop = safe_float(row.get("Fat_Drop", 0))
    snf_drop = safe_float(row.get("SNF_Drop", 0))
    current_fat = safe_float(row.get("Fat_Content", 0))
    current_snf = safe_float(row.get("SNF_Content", 0))
    flagged = is_flagged(row.get("Flag"))

    # Severity
    if flagged or fat_drop > 0.6 or snf_drop > 1.0 or current_fat < 3.2 or current_snf < 7.8:
        severity = "SEVERE"
    elif fat_drop > 0.4 or snf_drop > 0.6 or trend_days >= 7:
        severity = "MODERATE"
    else:
        severity = "MILD"

    prompt = f"""
Farmer: {name} (ID: {fid})
Location: {loc}
Trend: {trend_days} days
Current: Fat {current_fat:.2f}% (↓{fat_drop:.2f}%), SNF {current_snf:.2f}% (↓{snf_drop:.2f}%)
Severity: {severity}
{'SYSTEM FLAGGED — URGENT' if flagged else ''}

Provide 5–7 short, numbered, professional recommendations.
Prioritize field visit, adulteration test, feed/water audit, suspension if needed.
"""

    # Try AI models
    models = [
        ("meta-llama/llama-3.3-70b-instruct", "https://openrouter.ai/api/v1"),
        ("google/gemini-flash-1.5", "https://openrouter.ai/api/v1"),
        ("openai/gpt-4o-mini", "https://api.openai.com/v1")
    ]

    for model, url in models:
        try:
            key = os.getenv("OPENAI_API_KEY") if "openai.com" in url else os.getenv("OPENROUTER_API_KEY")
            if not key: continue
            resp = requests.post(
                f"{url}/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 450},
                headers={"Authorization": f"Bearer {key}"},
                timeout=20
            )
            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"].strip()
                if len(text.splitlines()) >= 3:
                    return text
        except:
            continue

    # 100% OFFLINE FALLBACK — ALWAYS WORKS
    if severity == "SEVERE":
        return (
            "1. **SUSPEND milk collection immediately**\n"
            "2. Urgent field visit within 24 hours (Quality Head required)\n"
            "3. Conduct full adulteration test panel\n"
            "4. Audit cattle feed and water source\n"
            "5. Collect next 3 samples under supervision\n"
            "6. Issue formal warning letter\n"
            "7. 7-day deadline for improvement"
        )
    elif severity == "MODERATE":
        return (
            "1. Schedule field visit within 3 working days\n"
            "2. Perform rapid adulteration screening\n"
            "3. Check feed quality and water hygiene\n"
            "4. Issue written warning to farmer\n"
            "5. Provide quality improvement guide\n"
            "6. Monitor next 7 collections\n"
            "7. Offer free nutrition consultation"
        )
    else:
        return (
            "1. Send advisory on maintaining Fat/SNF levels\n"
            "2. Include farmer in next training session\n"
            "3. Monitor trend for next 10 days\n"
            "4. Recommend mineral mixture in feed\n"
            "5. Share seasonal best practices\n"
            "6. Offer cattle health checkup"
        )

# ============================= SEND EMAIL =============================
def send_email(row: pd.Series, recommendation: str) -> bool:
    recipients = [e.strip() for e in os.getenv("QUALITY_MANAGERS", "").split(",") if e.strip()]
    if not recipients:
        st.error("No recipients in QUALITY_MANAGERS")
        return False

    decline = []
    if safe_float(row.get("Fat_Drop", 0)) > 0.4:
        decline.append(f"Fat ↓{safe_float(row.get('Fat_Drop')):.2f}%")
    if safe_float(row.get("SNF_Drop", 0)) > 0.6:
        decline.append(f"SNF ↓{safe_float(row.get('SNF_Drop')):.2f}%")
    if is_flagged(row.get("Flag")):
        decline.append("SYSTEM FLAGGED")

    html = template.render(
        farmer_name=row.get("Farmer_Name", "Unknown"),
        farmer_id=row.get("Farmer_ID", ""),
        region=row.get("Region", ""), panchayat=row.get("Panchayat", ""), state=row.get("State", ""),
        trend_days=safe_int(row.get("Trend_Days", 0)),
        decline_type=" | ".join(decline) if decline else "Quality decline",
        recommendation=recommendation.replace("\n", "<br>")
    )

    msg = MIMEMultipart()
    msg["From"] = f"{os.getenv('ALERT_FROM_NAME', 'Milk Alert')} <{os.getenv('SMTP_USER')}>"
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = "URGENT: Low-Quality Milk Alert"
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

# ============================= MAIN APP =============================
df = fetch_data()
if df.empty:
    st.stop()

# Ensure columns exist
for col in ["Fat_Content", "SNF_Content", "Fat_Content_3DaysAgo", "SNF_Content_3DaysAgo", "Trend_Days", "Flag"]:
    if col not in df.columns:
        df[col] = 0

df["Fat_Drop"] = df["Fat_Content_3DaysAgo"].apply(safe_float) - df["Fat_Content"].apply(safe_float)
df["SNF_Drop"] = df["SNF_Content_3DaysAgo"].apply(safe_float) - df["SNF_Content"].apply(safe_float)

# Detect risky farmers
risky = df[
    df["Flag"].apply(is_flagged) |
    (df["Fat_Drop"] > 0.4) |
    (df["SNF_Drop"] > 0.6) |
    ((df["Fat_Content"].apply(safe_float) < 3.3) & (df["Trend_Days"].apply(safe_int) >= 5)) |
    ((df["SNF_Content"].apply(safe_float) < 8.0) & (df["Trend_Days"].apply(safe_int) >= 5))
].copy()

if "Farmer_ID" in risky.columns:
    risky = risky.sort_values("Date", ascending=False).groupby("Farmer_ID").first().reset_index()

st.write(f"**{len(risky)} farmers require immediate attention**")

for _, row in risky.iterrows():
    with st.expander(
        f"{row.get('Farmer_Name','Unknown')} • {row.get('Farmer_ID','')} • Trend: {safe_int(row.get('Trend_Days',0))} days",
        expanded=True
    ):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**Fat:** {safe_float(row.get('Fat_Content')):.2f}% ↓{safe_float(row.get('Fat_Drop')):.2f}%")
            st.write(f"**SNF:** {safe_float(row.get('SNF_Content')):.2f}% ↓{safe_float(row.get('SNF_Drop')):.2f}%")
            st.write(f"**Location:** {row.get('Region','')} / {row.get('Panchayat','')}, {row.get('State','')}")

        with col2:
            if st.button("Send Email Alert", key=f"send_{row.get('Farmer_ID','')}", type="primary"):
                with st.spinner("Sending email..."):
                    rec = get_recommendation(row)
                    if send_email(row, rec):
                        st.success("Email alert sent successfully!")
                    else:
                        st.error("Failed to send email")

        st.markdown("### Recommended Actions")
        recommendation = get_recommendation(row)
        st.markdown(recommendation.replace("\n", "<br>"), unsafe_allow_html=True)

if risky.empty:
    st.success("No quality decline detected today")