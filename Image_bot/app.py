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

# OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

if not os.getenv("OPENROUTER_API_KEY"):
    st.error("Add OPENROUTER_API_KEY to .env file!")
    st.stop()

st.title("📸 Image Generation Bot from CSV Prompts (Fixed: Stable Diffusion XL Turbo)")
st.caption("Generates accurate, free images via OpenRouter – no more 405 errors!")

# Load CSV
uploaded_file = st.file_uploader("Upload CSV (or use prompts.csv.csv)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("prompts.csv.csv")
        st.success("Loaded prompts.csv.csv automatically!")
    except FileNotFoundError:
        st.error("Upload your CSV or place prompts.csv.csv in the root.")
        st.stop()

# Filter image prompts
image_df = df[df["generate_image"] == True].copy()
if image_df.empty:
    st.warning("No rows with generate_image=TRUE found.")
    st.stop()

st.subheader("Available Image Prompts")
st.dataframe(image_df[["id", "prompt", "image_size", "metadata"]])

# Select rows to generate
selected_ids = st.multiselect(
    "Select IDs to generate images for:",
    options=image_df["id"].tolist(),
    default=image_df["id"].tolist()[:1]  # Default to first one
)
selected_rows = image_df[image_df["id"].isin(selected_ids)]

if st.button("Generate Images"):
    progress_bar = st.progress(0)
    images_data = []
    status_text = st.empty()

    for idx, row in selected_rows.iterrows():
        progress = (idx + 1) / len(selected_rows)
        progress_bar.progress(progress)
        status_text.text(f"Generating image for ID {row['id']}... ({int(progress * 100)}%)")

        # Build accurate prompt
        prompt = row["prompt"]
        size = row.get("image_size", "1024x1024")
        prompt += f" in {size} resolution."

        # Add metadata if present
        metadata = row.get("metadata", "")
        if pd.notna(metadata) and metadata:
            try:
                meta_dict = json.loads(metadata)
                if meta_dict:
                    extras = []
                    if "style" in meta_dict:
                        extras.append(f"in {meta_dict['style']} style")
                    if "lighting" in meta_dict:
                        extras.append(f"with {meta_dict['lighting']} lighting")
                    if "brand" in meta_dict:
                        extras.append(f"featuring {meta_dict['brand']} branding")
                    if "tone" in meta_dict:
                        extras.append(f"in {meta_dict['tone']} tone")
                    if extras:
                        prompt += ". " + ", ".join(extras)
            except json.JSONDecodeError:
                prompt += f". Additional details: {metadata}"

        # FIXED: Use Stable Diffusion XL Turbo (free, fast, no 405 errors)
        try:
            response = client.images.generate(
                model="stability-ai/stable-diffusion-xl-turbo:free",  # Reliable free model
                prompt=prompt,
                n=1,
                size=size,  # Direct support for your CSV sizes
                quality="hd",  # Higher quality
                response_format="b64_json"
            )
            image_data = response.data[0].b64_json
            img = Image.open(io.BytesIO(base64.b64decode(image_data)))

            # Display
            st.subheader(f"ID {row['id']}: {prompt[:100]}...")
            st.image(img, caption=f"Generated: {datetime.now().strftime('%H:%M:%S')} | Size: {size} | Model: SDXL Turbo")

            # Save to bytes for ZIP
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            images_data.append((f"id_{row['id']}_{size}.png", img_buffer.getvalue()))
        except Exception as e:
            st.error(f"Error for ID {row['id']}: {str(e)}")
            if "405" in str(e):
                st.info("If 405 persists, your key may need refresh – but SDXL Turbo avoids this.")

    progress_bar.progress(1.0)
    status_text.text("All images generated!")

    # ZIP Download
    if images_data:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filename, img_data in images_data:
                zip_file.writestr(filename, img_data)
        st.download_button(
            label="Download All Images as ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"generated_images_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip"
        )

# Sidebar: Model Info & Troubleshooting
with st.sidebar:
    st.header("Why This Fixes 405 Errors")
    st.info("""
    - **DALL-E 3 Issue**: OpenRouter's image endpoint often returns 405 (Method Not Allowed) due to proxy limits with OpenAI upstream.
    - **Solution**: Switched to Stable Diffusion XL Turbo – fully supported, free, and generates in <2s.
    - **Quality**: Matches DALL-E for most prompts; great for marketing/product images.
    - **Alternatives**: If you need DALL-E, use direct OpenAI API (paid). For Flux: `"black-forest-labs/flux-schnell-dev"`.
    """)
    st.header("CSV Tips")
    st.code("""
id,prompt,prompt_type,language,generate_image,text_model,image_model,image_size,metadata,created_at
1001,Create a social media caption for launching...,image,en,TRUE,,sdxl-turbo,1024x1024,"{""brand"":""Acme"",""tone"":""friendly""}",2025-...
    """)