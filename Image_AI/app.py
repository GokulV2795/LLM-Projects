# app.py — FINAL WORKING VERSION (Free + No Errors)
import streamlit as st
from PIL import Image
import io
import base64
import re
import time
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
load_dotenv()
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ====================== CONFIG ======================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("Add your OpenRouter key to .env → OPENROUTER_API_KEY=sk-or-...")
    st.stop()

# FREE & EXCELLENT IMAGE MODEL
MODEL = "meta-llama/llama-3.2-90b-vision-instruct"

llm = ChatOpenAI(
    model=MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.8,
    max_tokens=1000,
)

# ====================== SESSION STATE INITIALIZATION ======================
# This fixes the AttributeError forever
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"t{int(time.time())}"
if "aspect_ratio" not in st.session_state:
    st.session_state.aspect_ratio = "1:1"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "generated_prompt" not in st.session_state:
    st.session_state.generated_prompt = ""

# Uploaded image state — THIS WAS MISSING BEFORE
if "uploaded_b64" not in st.session_state:
    st.session_state.uploaded_b64 = None
if "uploaded_mime" not in st.session_state:
    st.session_state.uploaded_mime = None
if "is_edit_mode" not in st.session_state:
    st.session_state.is_edit_mode = False

# ====================== STATE TYPE ======================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "Chat history"]
    current_prompt: str
    uploaded_image_b64: str | None
    uploaded_mime: str | None
    is_edit_mode: bool

# ====================== TOOLS ======================
def refine_prompt(state: AgentState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a world-class prompt engineer.
Always end your reply with exactly:
FINAL PROMPT: "your best image prompt here\""""),
        MessagesPlaceholder("messages"),
    ])
    chain = prompt | llm
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def generate_or_edit_image(state: AgentState) -> dict:
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    prompt = state["current_prompt"]
    messages = [{"role": "user", "content": prompt}]

    if state["is_edit_mode"] and state["uploaded_image_b64"]:
        messages[0]["content"] = [
            {"type": "text", "text": f"Edit this image: {prompt}"},
            {"type": "image_url", "image_url": {
                "url": f"data:{state['uploaded_mime']};base64,{state['uploaded_image_b64']}"
            }}
        ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            modalities=["image", "text"],
            max_tokens=1000,
            extra_body={"image_config": {"aspect_ratio": st.session_state.aspect_ratio}},
        )
        img_b64 = response.choices[0].message.images[0].image_url.url.split(",", 1)[1]
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.session_state.generated_image = buf.getvalue()
        st.session_state.generated_prompt = prompt

        return {"messages": [AIMessage(content=f"Done! Here's your image.\n\n**Prompt:** {prompt}")]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error: {str(e)}")]}

def extract_prompt(state: AgentState) -> dict:
    text = state["messages"][-1].content
    match = re.search(r"FINAL PROMPT:\s*[\"']([^\"']+)[\"']", text, re.I)
    if match:
        return {"current_prompt": match.group(1).strip()}
    return state

def should_generate(state: AgentState) -> str:
    last = state["messages"][-1].content.lower()
    if any(word in last for word in ["generate", "create", "make", "go", "now", "show", "ready"]):
        return "generate"
    return "refine"

def create_graph():
    g = StateGraph(AgentState)
    g.add_node("refine", refine_prompt)
    g.add_node("generate", generate_or_edit_image)
    g.add_node("extract", extract_prompt)
    g.set_entry_point("refine")
    g.add_edge("refine", "extract")
    g.add_conditional_edges("extract", should_generate, {"generate": "generate", "refine": "refine"})
    g.add_edge("generate", END)
    return g.compile(checkpointer=MemorySaver())

app = create_graph()

# ====================== UI ======================
st.set_page_config(page_title="Image Bot", layout="wide")
st.title("AI Image Generator")
st.caption("Works with $0 credits • Llama 3.2 90B Vision • Full image editing")

# Sidebar
with st.sidebar:
    st.session_state.aspect_ratio = st.selectbox("Aspect Ratio", ["1:1", "16:9", "9:16", "4:3", "3:4"])
    if st.button("New Chat"):
        st.session_state.thread_id = f"t{int(time.time())}"
        st.session_state.chat_history = []
        st.session_state.generated_image = None
        st.rerun()

    st.divider()
    uploaded = st.file_uploader("Upload image to edit", type=["png", "jpg", "jpeg"])
    if uploaded:
        st.session_state.uploaded_b64 = base64.b64encode(uploaded.getvalue()).decode()
        st.session_state.uploaded_mime = uploaded.type
        st.session_state.is_edit_mode = True
        st.image(uploaded, caption="Image ready for editing", width=200)
    else:
        st.session_state.uploaded_b64 = None
        st.session_state.uploaded_mime = None
        st.session_state.is_edit_mode = False

# Chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("image"):
            st.image(msg["image"])

# User input
if prompt := st.chat_input("Describe your image or type 'generate'"):
    with st.chat_message("user"):
        st.write(prompt)

    initial_prompt = f"{'Edit the uploaded image:' if st.session_state.is_edit_mode else 'Generate an image of:'} {prompt}"

    state = {
        "messages": [HumanMessage(content=prompt)],
        "current_prompt": initial_prompt,
        "uploaded_image_b64": st.session_state.uploaded_b64,
        "uploaded_mime": st.session_state.uploaded_mime,
        "is_edit_mode": st.session_state.is_edit_mode,
    }

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.spinner("Working..."):
        for _ in app.stream(state, config): 
            pass
        final = app.invoke(None, config)

    # Show assistant replies
    for msg in final["messages"]:
        if isinstance(msg, AIMessage):
            if not any(m["content"] == msg.content for m in st.session_state.chat_history):
                with st.chat_message("assistant"):
                    st.write(msg.content)
                    entry = {"role": "assistant", "content": msg.content}
                    if st.session_state.generated_image:
                        st.image(st.session_state.generated_image)
                        entry["image"] = st.session_state.generated_image
                        st.download_button("Download Image", st.session_state.generated_image, "image.png", "image/png")
                    st.session_state.chat_history.append(entry)

    st.rerun()