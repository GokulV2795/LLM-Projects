import streamlit as st
from PIL import Image
import io
import base64
import re
import time
from typing import TypedDict, Annotated, List

# Load .env file
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

if not OPENROUTER_API_KEY or OPENROUTER_API_KEY.strip() == "":
    st.error("OPENROUTER_API_KEY not found! Add it to your .env file.")
    st.info("Create a file named `.env` in your project folder with:\n`OPENROUTER_API_KEY=sk-or-v1-...`")
    st.stop()

MODEL = "meta-llama/llama-3.2-90b-vision-instruct"

llm = ChatOpenAI(
    model=MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.7,
)

# ====================== STATE ======================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "Chat history"]
    current_prompt: str
    uploaded_image_b64: str | None
    uploaded_mime: str | None
    is_edit_mode: bool

# ====================== TOOLS ======================
def refine_prompt(state: AgentState) -> dict:
    messages = state["messages"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert prompt engineer for image generation/editing.
Always respond conversationally and end with:
FINAL PROMPT: "your best refined prompt here"

Make it vivid, detailed, and optimized for Meta-llama image generation."""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain = prompt | llm
    response = chain.invoke({"messages": messages})
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
            extra_body={"image_config": {"aspect_ratio": st.session_state.aspect_ratio}},
        )
        img_url = response.choices[0].message.images[0].image_url.url
        img_data = base64.b64decode(img_url.split(",", 1)[1])
        img = Image.open(io.BytesIO(img_data))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.session_state.generated_image = buf.getvalue()
        st.session_state.generated_prompt = prompt

        return {
            "messages": [AIMessage(content=f"Image generated/edited successfully!\n\n**Final Prompt Used:**\n{prompt}")],
            "current_prompt": prompt
        }
    except Exception as e:
        error_msg = f"Image generation failed: {str(e)}"
        return {"messages": [AIMessage(content=error_msg)]}

# ====================== EXTRACT FINAL PROMPT ======================
def extract_final_prompt(state: AgentState) -> dict:
    last_msg = state["messages"][-1].content
    match = re.search(r"FINAL PROMPT:\s*[\"']([^\"']+)[\"']", last_msg, re.IGNORECASE)
    if match:
        new_prompt = match.group(1).strip()
        return {"current_prompt": new_prompt}
    return state

# ====================== BUILD GRAPH (FIXED!) ======================
def should_generate(state: AgentState) -> str:
    last_message = state["messages"][-1].content.lower()
    trigger_words = ["generate", "create", "make", "edit", "show", "now", "go", "ready"]
    if any(word in last_message for word in trigger_words):
        return "generate"
    return "refine"

def create_graph():
    graph = StateGraph(AgentState)

    graph.add_node("refine", refine_prompt)
    graph.add_node("generate", generate_or_edit_image)
    graph.add_node("extract", extract_final_prompt)

    graph.set_entry_point("refine")
    graph.add_edge("refine", "extract")
    graph.add_conditional_edges(
        "extract",
        should_generate,  # Fixed: proper function reference
        {
            "generate": "generate",
            "refine": "refine"
        }
    )
    graph.add_edge("generate", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

app = create_graph()

# ====================== STREAMLIT UI ======================
st.set_page_config(page_title="LangGraph Image Agent", layout="wide")
st.title("LangGraph + OpenRouter Image Agent")
st.caption("True stateful agent • Image generation & editing • Powered by Gemini 3 Pro")

# Initialize session state
defaults = {
    "thread_id": f"thread_{int(time.time())}",
    "aspect_ratio": "1:1",
    "generated_image": None,
    "chat_history": [],
    "uploaded_b64": None,
    "uploaded_mime": None,
    "is_edit_mode": False
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Sidebar
with st.sidebar:
    st.header("Controls")
    st.session_state.aspect_ratio = st.selectbox("Aspect Ratio", ["1:1", "16:9", "9:16", "4:3", "3:4"], index=0)
    
    if st.button("New Conversation"):
        st.session_state.thread_id = f"thread_{int(time.time())}"
        st.session_state.chat_history = []
        st.session_state.generated_image = None
        st.rerun()

    st.divider()
    uploaded = st.file_uploader("Upload image to edit (optional)", type=["png", "jpg", "jpeg"])
    if uploaded:
        bytes_data = uploaded.getvalue()
        b64 = base64.b64encode(bytes_data).decode()
        st.session_state.uploaded_b64 = b64
        st.session_state.uploaded_mime = uploaded.type
        st.session_state.is_edit_mode = True
        st.image(uploaded, caption="Ready for editing", width=200)
    else:
        st.session_state.uploaded_b64 = None
        st.session_state.uploaded_mime = None
        st.session_state.is_edit_mode = False

# Initial state
initial_state = {
    "messages": [],
    "current_prompt": "",
    "uploaded_image_b64": st.session_state.uploaded_b64,
    "uploaded_mime": st.session_state.uploaded_mime,
    "is_edit_mode": st.session_state.is_edit_mode,
}

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("image"):
            st.image(msg["image"])

# User input
if prompt := st.chat_input("Describe your image or type 'generate' to create it..."):
    with st.chat_message("user"):
        st.write(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    if not initial_state["current_prompt"]:
        base = "Edit the uploaded image:" if st.session_state.is_edit_mode else "Generate an image of:"
        initial_state["current_prompt"] = f"{base} {prompt}"

    initial_state["messages"].append(HumanMessage(content=prompt))

    with st.spinner("Thinking..."):
        for _ in app.stream(initial_state, config, stream_mode="values"):
            pass
        final_state = app.invoke(None, config)

    # Display new messages
    new_msgs = []
    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage):
            if not any(m["content"] == msg.content for m in st.session_state.chat_history):
                with st.chat_message("assistant"):
                    st.write(msg.content)
                    new_msgs.append({"role": "assistant", "content": msg.content})

    # Show image if generated
    if st.session_state.generated_image:
        with st.chat_message("assistant"):
            st.image(st.session_state.generated_image, caption=f"Generated: {st.session_state.generated_prompt}")
            if new_msgs:
                new_msgs[-1]["image"] = st.session_state.generated_image
            st.download_button(
                "Download Image",
                st.session_state.generated_image,
                "generated_image.png",
                "image/png"
            )

    st.session_state.chat_history.extend(new_msgs)
    st.rerun()