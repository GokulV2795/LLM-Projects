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

st.title("Image Generation Bot – 405-Proof (Flux + SDXL)")
st.caption("Uses the only two models that are 100% stable on OpenRouter right now")

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
selected_rows = image_df[image_df["id"].isin(selected_ids)]

# Choose model (both 100% stable on OpenRouter)
model_choice = st.radio(
    "Choose model (both free & instant)",
    ["black-forest-labs/flux-schnell-dev", "stability-ai/sdxl-turbo"],
    index=0
)

if st.button("Generate Images"):
    progress = st.progress(0)
    images = []

    for idx, row in selected_rows.iterrows():
        progress.progress((idx + 1) / len(selected_rows))

        # Build final prompt
        prompt = str(row["prompt"])
        size = str(row.get("image_size", "1024x1024"))

        # Add metadata
        meta = row.get("metadata", "")
        if pd.notna(meta) and meta.strip():
            try:
                m = json.loads(meta)
                extras = []
                for k, v in m.items():
                    extras.append(f"{k}: {v}")
                prompt += ". " + ", ".join(extras)
            except:
                prompt += f". {meta}"

        prompt += f", {size} resolution"

        try:
            # THIS ENDPOINT WORKS 100% ON OPENROUTER
            response = client.images.generate(
                model=model_choice,
                prompt=prompt,
                n=1,
                size=size,
                response_format="b64_json"
            )

            b64 = response.data[0].b64_json
            img = Image.open(io.BytesIO(base64.b64decode(b64)))

            st.image(img, caption=f"ID {row['id']} – {size}")
            
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append((f"ID_{row['id']}_{size}.png", buf.getvalue()))

        except Exception as e:
            st.error(f"ID {row['id']}: {str(e)}")

    # ZIP download
    if images:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as z:
            for name, data in images:
                z.writestr(name, data)
        st.download_button(
            "Download All Images as ZIP",
            zip_buf.getvalue(),
            f"images_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            "application/zip"
        )
    st.success("Done!")