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

st.title("📸 Image Generation Bot from CSV Prompts (OpenRouter + DALL-E 3)")
st.caption("Generates accurate images using DALL-E 3 via OpenRouter based on your CSV prompts.")

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
        # Map size to DALL-E supported aspect ratios (DALL-E 3 supports 1024x1024, 1792x1024, 1024x1792)
        size_map = {
            "1024x1024": "square",
            "1536x1536": "square",
            "2048x2048": "square",
            "1024x1792": "portrait",
            "1792x1024": "landscape"
        }
        aspect = size_map.get(size, "square")
        prompt += f" in {aspect} aspect ratio, high resolution."

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

        # Generate with DALL-E 3 via OpenRouter
        try:
            response = client.images.generate(
                model="openai/dall-e-3",  # High-quality model on OpenRouter
                prompt=prompt,
                n=1,
                size=size,  # Use exact size from CSV
                quality="standard",
                response_format="b64_json"  # For easy decoding
            )
            image_data = response.data[0].b64_json
            img = Image.open(io.BytesIO(base64.b64decode(image_data)))

            # Display
            st.subheader(f"ID {row['id']}: {prompt[:100]}...")
            st.image(img, caption=f"Generated: {datetime.now().strftime('%H:%M:%S')} | Size: {size}")

            # Save to bytes for ZIP
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            images_data.append((f"id_{row['id']}_{size}.png", img_buffer.getvalue()))
        except Exception as e:
            st.error(f"Error for ID {row['id']}: {str(e)}")

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

# Sidebar: Tips
with st.sidebar:
    st.header("Tips")
    st.info("""
    - **Model**: DALL-E 3 via OpenRouter – accurate, creative, and free for testing.
    - **Cost**: ~$0.04/image (free credits on signup).
    - **Supported Sizes**: 1024x1024 (square), 1792x1024 (landscape), 1024x1792 (portrait).
    - **Customization**: Use metadata JSON for styles (e.g., {"style": "minimal", "lighting": "natural"}).
    - **Rate Limits**: 50 images/hour on free tier.
    """)
    st.header("CSV Format Example")
    st.code("""
id,prompt,prompt_type,language,generate_image,text_model,image_model,image_size,metadata,created_at
1001,Create a social media caption...,image,en,TRUE,,dall-e-3,1024x1024,"{""style"":""modern"",""brand"":""Acme""}",2025-...
    """)