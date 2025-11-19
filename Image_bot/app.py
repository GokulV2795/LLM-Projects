import streamlit as st
import pandas as pd
from openai import OpenAI
from PIL import Image
import io
import zipfile
from dotenv import load_dotenv
import os
from datetime import datetime
import json
import base64

load_dotenv()

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

if not os.getenv("OPENROUTER_API_KEY"):
    st.error("Add OPENROUTER_API_KEY to .env file!")
    st.stop()

st.title("Image Generation Bot...")
st.caption("Uses FLUX or SDXL Turbo via OpenRouter — no 405, no crashes")

# Load CSV
uploaded_file = st.file_uploader("Upload your CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("prompts.csv.csv")
        st.success("Loaded prompts.csv.csv")
    except:
        st.error("Please upload your CSV")
        st.stop()

# Filter image rows
image_df = df[df["generate_image"].astype(str).str.upper() == "TRUE"].copy()
if image_df.empty:
    st.warning("No rows with generate_image=TRUE")
    st.stop()

st.dataframe(image_df[["id", "prompt", "image_size", "metadata"]])

selected_ids = st.multiselect("Select IDs", options=image_df["id"].tolist(), default=[1001])
selected_rows = image_df[image_df["id"].isin(selected_ids)].reset_index(drop=True)

# Choose model
model_choice = st.radio(
    "Choose model (both free & instant)",
    ["openai/gpt-5-image","black-forest-labs/flux-schnell-dev", "stability-ai/sdxl-turbo"],
    index=0
)

if st.button("Generate Images", type="primary"):
    if selected_rows.empty:
        st.error("No rows selected!")
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()
    generated_images = []

    for idx, row in selected_rows.iterrows():
        current_progress = int((idx + 1) / len(selected_rows) * 100)
        progress_bar.progress(current_progress)
        status_text.text(f"Generating ID {row['id']}... ({current_progress}%)")

        # Build prompt
        prompt = str(row["prompt"])
        size = str(row.get("image_size", "1024x1024"))

        # Add metadata
        meta = row.get("metadata", "")
        if pd.notna(meta) and str(meta).strip():
            try:
                m = json.loads(meta)
                extras = [f"{k}: {v}" for k, v in m.items()]
                prompt += ". " + ", ".join(extras)
            except:
                prompt += f". {meta}"

        prompt += f", high quality, {size} resolution"

        try:
            response = client.images.generate(
                model=model_choice,
                prompt=prompt,
                n=1,
                size=size,
                response_format="b64_json"
            )

            b64 = response.data[0].b64_json
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes))

            st.image(img, caption=f"ID {row['id']} – {size} – {model_choice.split('/')[-1]}")

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            generated_images.append((f"ID_{row['id']}_{size}.png", buf.getvalue()))

        except Exception as e:
            st.error(f"ID {row['id']}: {str(e)}")

    # Final status
    progress_bar.progress(100)
    status_text.success("All images generated!")

    # ZIP download
    if generated_images:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in generated_images:
                zf.writestr(name, data)

        st.download_button(
            label="Download All Images as ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )

st.success("Bot ready! Select IDs and generate.")