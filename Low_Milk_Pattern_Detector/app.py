"""
Regional Milk Quality Dashboard + Charts + Email Report
Now with Trend Line & Risky Panchayat Bar Chart
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
from datetime import datetime, timedelta

load_dotenv()

st.set_page_config(page_title="Regional Milk Quality Dashboard", layout="wide")
st.title("Regional Milk Quality Dashboard")
st.markdown("**Select region → View stats, trends, top risky farmers & send report**")

# ============================= EMAIL TEMPLATE =============================
EMAIL_TEMPLATE = """
<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; color: #333; line-height: 1.8; margin: 20px;">
  <h1 style="color: #d32f2f; text-align: center;">REGIONAL MILK QUALITY REPORT</h1>
  <h2 style="color: #1976d2;">Region: {{ region }} | {{ state }}</h2>
  <p><strong>Report Date:</strong> {{ today }}</p>
  
  <h3>Key Statistics</h3>
  <ul style="font-size: 16px;">
    <li>Total Farmers: <strong>{{ total_farmers }}</strong></li>
    <li>Active Today: <strong>{{ active_today }}</strong></li>
    <li>Average Fat: <strong>{{ avg_fat }}%</strong> | Average SNF: <strong>{{ avg_snf }}%</strong></li>
    <li>Risky Farmers: <strong style="color:#d32f2f;">{{ risky_count }}</strong> ({{ risky_pct }}% of total)</li>
  </ul>

  <h3>Top 5 Risky Farmers – Immediate Action Required</h3>
  <table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%; font-size: 14px;">
    <tr style="background: #d32f2f; color: white; text-align: center;">
      <th>Rank</th><th>Farmer Name</th><th>ID</th><th>Panchayat</th><th>Fat %</th><th>SNF %</th><th>Drop</th><th>Status</th>
    </tr>
    {% for f in top5 %}
    <tr style="text-align: center;">
      <td><strong>{{ loop.index }}</strong></td>
      <td><strong>{{ f.name }}</strong></td>
      <td>{{ f.id }}</td>
      <td>{{ f.panchayat }}</td>
      <td>{{ f.fat }}</td>
      <td>{{ f.snf }}</td>
      <td>{{ f.drop }}</td>
      <td style="color:#d32f2f; font-weight:bold;">{{ f.status }}</td>
    </tr>
    {% endfor %}
  </table>

  <div style="background:#fff3e0; padding:15px; margin:20px 0; border-left:5px solid #ff9800; font-size:15px;">
    <p><strong>Urgent Action Required:</strong></p>
    <p>Immediate focus must be given to the top 5 flagged farmers and the 10 worst-performing panchayats. Conduct on-site visits within the next <strong>48–72 hours</strong> to audit cattle feed quality, water sources, storage practices, and test for adulteration.</p>
    <p>Farmers showing consistent decline over the past 4 months must receive written warnings, mandatory training on best feeding practices, and compulsory mineral mixture usage.</p>
    <p>Regions with more than 8% risky farmers require intensified weekly monitoring and supervised collection until average Fat ≥ 3.6% and SNF ≥ 8.5% is achieved.</p>
    <p><strong>Quick and decisive intervention now will prevent wider quality deterioration and protect overall collection standards.</strong></p>
  </div>

  <hr style="border: 1px solid #ddd; margin: 30px 0;">
  <p style="color:#777; font-size:13px;"><em>Automated Report • Low-Quality Milk Alert System</em></p>
</body>
</html>
"""
template = Template(EMAIL_TEMPLATE)

# ============================= DOMO FETCH =============================
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
        st.error("DOMO_DATASET_ID missing")
        return pd.DataFrame()

    try:
        token = get_domo_token()
        url = f"https://api.domo.com/v1/datasets/{dsid}/data"
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}", "Accept": "text/csv"}, params={"includeHeader": "true"}, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = df.columns.str.strip()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        st.success(f"Loaded {len(df):,} records")
        return df
    except Exception as e:
        st.error(f"Load failed: {e}")
        return pd.DataFrame()

# Safe helpers
def safe_float(x, default=0.0):
    try: return float(x) if pd.notna(x) else default
    except: return default
def safe_int(x, default=0):
    try: return int(x) if pd.notna(x) else default
    except: return default
def is_flagged(x):
    return str(x).strip().lower() in ("1","yes","true","quality watch","alert","flagged","high risk")

# ============================= LOAD DATA =============================
df = fetch_data()
if df.empty:
    st.stop()

# Prepare columns
for c in ["Fat_Content","SNF_Content","Fat_Content_3DaysAgo","SNF_Content_3DaysAgo","Trend_Days","Flag"]:
    if c not in df.columns: df[c] = 0

df["Fat_Drop"] = df["Fat_Content_3DaysAgo"].apply(safe_float) - df["Fat_Content"].apply(safe_float)
df["SNF_Drop"] = df["SNF_Content_3DaysAgo"].apply(safe_float) - df["SNF_Content"].apply(safe_float)
df["Risk_Score"] = df["Fat_Drop"] + df["SNF_Drop"] + (df["Flag"].apply(is_flagged).astype(int)*10)

# ============================= REGION SELECTOR =============================
regions = ["All Regions"] + sorted(df["Region"].dropna().unique().tolist())
selected_region = st.selectbox("Select Region", regions)

region_df = df if selected_region == "All Regions" else df[df["Region"] == selected_region].copy()

# ============================= STATS =============================
total_farmers = region_df["Farmer_ID"].nunique()
active_today = region_df[region_df["Date"] == region_df["Date"].max()]["Farmer_ID"].nunique() if "Date" in df.columns else 0
avg_fat = region_df["Fat_Content"].apply(safe_float).mean()
avg_snf = region_df["SNF_Content"].apply(safe_float).mean()

risky_df = region_df[
    region_df["Flag"].apply(is_flagged) |
    (region_df["Fat_Drop"] > 0.4) |
    (region_df["SNF_Drop"] > 0.6) |
    ((region_df["Fat_Content"].apply(safe_float) < 3.3) & (region_df["Trend_Days"].apply(safe_int) >= 5))
].copy()

risky_count = len(risky_df)
risky_pct = round(risky_count / total_farmers * 100, 1) if total_farmers > 0 else 0

top5 = risky_df.sort_values("Risk_Score", ascending=False).drop_duplicates("Farmer_ID").head(5)

# ============================= CHARTS =============================
st.subheader("Quality Trend (Last 3 days)")
last_30 = region_df[region_df["Date"] >= (datetime.now() - timedelta(days=120))]
trend = last_30.groupby(last_30["Date"].dt.date)[["Fat_Content","SNF_Content"]].mean().reset_index()
trend["Date"] = pd.to_datetime(trend["Date"])

if not trend.empty:
    st.line_chart(trend.set_index("Date")[["Fat_Content","SNF_Content"]], use_container_width=True)
else:
    st.info("No data in last 30 days")

st.subheader("Riskiest Panchayats (Bar Chart)")
panchayat_risk = risky_df["Panchayat"].value_counts().head(10)
if not panchayat_risk.empty:
    st.bar_chart(panchayat_risk, use_container_width=True)
else:
    st.info("No risky panchayats")

# ============================= DISPLAY STATS & TABLE =============================
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Total Farmers", total_farmers)
with c2: st.metric("Active Today", active_today)
with c3: st.metric("Avg Fat %", f"{avg_fat:.2f}")
with c4: st.metric("Avg SNF %", f"{avg_snf:.2f}")

st.error(f"Risky Farmers: {risky_count} ({risky_pct}%)")

if not top5.empty:
    st.subheader(f"Top 5 Risky Farmers – {selected_region}")
    disp = top5[["Farmer_Name","Farmer_ID","Panchayat","Fat_Content","SNF_Content","Fat_Drop","SNF_Drop","Flag"]].copy()
    disp["Fat"] = disp["Fat_Content"].apply(lambda x: f"{safe_float(x):.2f}")
    disp["SNF"] = disp["SNF_Content"].apply(lambda x: f"{safe_float(x):.2f}")
    disp["Drop"] = "Fat ↓" + disp["Fat_Drop"].apply(lambda x: f"{safe_float(x):.2f}") + " | SNF ↓" + disp["SNF_Drop"].apply(lambda x: f"{safe_float(x):.2f}")
    disp["Status"] = disp["Flag"].apply(lambda x: "FLAGGED" if is_flagged(x) else "Decline")
    st.dataframe(disp[["Farmer_Name","Farmer_ID","Panchayat","Fat","SNF","Drop","Status"]], use_container_width=True)
else:
    st.success("No risky farmers in this region")

# ============================= SEND EMAIL =============================
if st.button("Send Regional Summary Email", type="primary"):
    with st.spinner("Generating & sending report..."):
        top5_list = []
        for _, r in top5.iterrows():
            top5_list.append({
                "name": r.get("Farmer_Name","Unknown"),
                "id": r.get("Farmer_ID",""),
                "panchayat": r.get("Panchayat","N/A"),
                "fat": f"{safe_float(r['Fat_Content']):.2f}",
                "snf": f"{safe_float(r['SNF_Content']):.2f}",
                "drop": f"Fat ↓{safe_float(r['Fat_Drop']):.2f} | SNF ↓{safe_float(r['SNF_Drop']):.2f}",
                "status": "FLAGGED" if is_flagged(r["Flag"]) else "Decline"
            })

        state = region_df["State"].iloc[0] if not region_df.empty and "State" in region_df.columns else "N/A"

        html = template.render(
            region=selected_region,
            state=state,
            today=datetime.today().strftime("%d %B %Y"),
            total_farmers=total_farmers,
            active_today=active_today,
            avg_fat=f"{avg_fat:.2f}",
            avg_snf=f"{avg_snf:.2f}",
            risky_count=risky_count,
            risky_pct=risky_pct,
            top5=top5_list
        )

        msg = MIMEMultipart()
        msg["From"] = os.getenv("SMTP_USER")
        recipients = [e.strip() for e in os.getenv("QUALITY_MANAGERS","").split(",") if e.strip()]
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = f"Milk Quality Report: {selected_region} ({risky_count} Risky)"
        msg.attach(MIMEText(html, "html"))

        try:
            s = smtplib.SMTP(os.getenv("SMTP_HOST"), int(os.getenv("SMTP_PORT")))
            s.starttls()
            s.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
            s.sendmail(os.getenv("SMTP_USER"), recipients, msg.as_string())
            s.quit()
            st.success(f"Report sent for {selected_region}!")
        except Exception as e:
            st.error(f"Email failed: {e}")