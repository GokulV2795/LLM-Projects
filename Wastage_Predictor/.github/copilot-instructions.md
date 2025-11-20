# Wastage Predictor - AI Agent Instructions

## Project Overview
**Wastage Predictor** is a Streamlit-based inventory analytics dashboard that predicts product wastage risk and generates exportable visualizations. The system integrates with Snowflake for data retrieval and optionally uses OpenAI APIs for recommendations.

### Architecture
- **app.py**: Streamlit frontend with interactive charts (Plotly), data analysis, and export functionality
- **snowflake_read.py**: Database abstraction layer for Snowflake queries
- **email_sender.py**: SMTP email delivery with attachment support
- **requirements.txt**: Python dependencies (Streamlit, Plotly, Snowflake connector, etc.)

---

## Critical Patterns & Conventions

### 1. **Environment Configuration**
All external credentials load via `dotenv` in every module. **Never hardcode secrets.**

**Required env vars:**
- Snowflake: `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, `SNOWFLAKE_ROLE`
- Email (SMTP): `FROM_EMAIL`, `WAREHOUSE_MANAGER_EMAIL`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`
- AI: `OPENROUTER_API_KEY`

Missing credentials should trigger user-facing errors via `st.error()` and `st.stop()`.

### 2. **Data Flow & Prediction Logic**
1. **Fetch**: `fetch_recent_wastage(days)` returns pandas DataFrame from Snowflake `INVENTORY_DB.PUBLIC.WASTAGE_RISK_TABLE`
2. **Predict**: `generate_predictions(df)` applies lightweight heuristics (not expensive AI calls by default):
   - `RISK_SCORE`: 0–100 based on `DAYS_TO_EXPIRY_EST` (closer to expiry = higher risk)
   - `WASTAGE_PREDICTOR`: Categories ["HIGH" ≥70, "MEDIUM" ≥40, "LOW" <40]
   - `AI_RECOMMENDATION`: Template text per category
3. **Visualize**: `create_and_export_charts(df)` generates 4 charts and exports as PNG+ZIP

**Important**: The OpenAI/OpenRouter client is initialized but only used for future enhancements. Current predictions use deterministic heuristics for reliability.

### 3. **Chart Export Workflow**
- Charts created via Plotly (`px.pie`, `px.bar`, `px.histogram`)
- Each chart converted to PNG using `fig.to_image()` with `kaleido` backend (width=1200, height=600, scale=2)
- All PNGs zipped into single download buffer using `zipfile`
- Filename format: `wastage_charts_YYYYMMDD_HHMM.zip`

**When adding charts**: Always append tuple `(filename, fig)` to `figs` list AND `(filename, png_bytes)` to `png_bytes` list.

### 4. **Email Integration**
`send_email_smtp(subject, body, to=None, attachments=None)` sends via SMTP with optional file attachments.
- Defaults to `WAREHOUSE_MANAGER_EMAIL` if `to` not specified
- Silently skips failed attachments (logs nothing currently—consider adding logging)
- Raises `ValueError` if SMTP config incomplete

### 5. **UI/UX Conventions**
- **Sidebar controls**: `days` slider (1–30, default 7) + `model` dropdown (though model selection not used yet)
- **Primary action**: "Run Analysis Now" button triggers fetch → predict → visualize pipeline
- **Status**: Use `st.spinner()` for async operations, `st.success()`/`st.error()`/`st.info()` for messaging
- **Alerts**: `st.error()` highlights HIGH-risk items above charts

---

## Development & Testing

### Run Locally
```bash
uv pip install -r requirements.txt
# Or: pip install -r requirements.txt
streamlit run app.py
```

### Key Dependencies
- **Streamlit**: UI framework
- **Plotly**: Interactive charting
- **Kaleido**: PNG export (required for `fig.to_image()`)
- **Snowflake connector**: Database queries
- **python-dotenv**: Environment variable loading

---

## Integration Points & Future Extensions

1. **AI Model Selection**: The `model` dropdown selects OpenRouter models but isn't wired to predictions yet. Future: route to `generate_predictions()` or create `generate_ai_predictions(df, model)`.
2. **Email Alerts**: `send_email_smtp()` ready for HIGH-risk notifications; integrate with "Run Analysis" or schedule via external job.
3. **Export Formats**: Currently PNG+ZIP. Can extend to CSV, PDF, Excel via similar patterns.

---

## Common Pitfalls to Avoid

- **Missing env vars**: Always validate at startup; don't assume defaults
- **Empty DataFrames**: Check `df.empty` before charting; functions return early with `st.info()`
- **Attachment paths**: `email_sender.py` expects absolute or relative paths; use `os.path.abspath()` if needed
- **Plotly memory**: Large datasets (>50K rows) may slow PNG export; consider sampling for high-resolution exports

---

## When Extending

- **New data sources**: Follow `snowflake_read.py` pattern—create function, test with `if __name__ == '__main__'`, return DataFrame
- **New charts**: Add to `create_and_export_charts()` loop; ensure PNG export logic included
- **New email features**: Extend `send_email_smtp()` signature; document required env vars
- **New predictions**: Update `generate_predictions()` logic or create separate function if using external APIs
