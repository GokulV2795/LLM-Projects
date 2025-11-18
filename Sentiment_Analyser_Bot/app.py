import streamlit as st
import pandas as pd
from textblob import TextBlob
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime
import base64

# NEW: PDF Generation Libraries
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

load_dotenv()

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

if not os.getenv("OPENROUTER_API_KEY"):
    st.error("OPENROUTER_API_KEY not found in .env!")
    st.stop()

st.title("Sentiment Analysis Bot + Reasoning → PDF Export")

# File upload & processing (same as before)
uploaded_file = st.file_uploader("Upload dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    if "text" not in df.columns:
        st.error("Need a 'text' column!")
        st.stop()

    with st.spinner("Analyzing sentiment..."):
        df["sentiment"] = df["text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    st.success("Sentiment analysis done!")
    st.dataframe(df)

    with st.spinner("Building RAG index..."):
        embeddings = OpenAIEmbeddings(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="nomic-ai/nomic-embed-text-v1.5"
        )
        docs = [Document(page_content=r["text"], metadata={"sentiment": r["sentiment"]}) for _, r in df.iterrows()]
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    st.success("Ready!")

    # RAG + Reasoning function
    def rag_with_reasoning(query: str):
        docs = retriever on.invoke(query)
        context = "\n\n".join(
            f"Text: {d.page_content}\nSentiment: {d.metadata['sentiment']:.3f}"
            for d in docs
        )

        messages = [
            {"role": "system", "content": "You are an expert sentiment analyst. Always think step-by-step in <thinking> tags before answering."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\n<thinking>Let me analyze this carefully...</thinking>\n\n<answer>"}
        ]

        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",  # or "openai/gpt-oss-20b:free"
            messages=messages,
            extra_body={"reasoning": {"enabled": True}},
            temperature=0.7,
            max_tokens=2000
        )

        msg = response.choices[0].message
        reasoning = getattr(msg, "reasoning_details", None)
        reasoning_text = reasoning.thinking if reasoning else "Reasoning not available (model may not support it)"

        return {
            "question": query,
            "answer": msg.content.strip(),
            "reasoning": reasoning_text,
            "retrieved_docs": docs,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    # PDF Generator
    def generate_pdf_report(data):
        buffer = bytes()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=1*inch, leftMargin=1*inch, rightMargin=1*inch)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Sentiment Analysis Report", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>Date:</b> {data['timestamp']}", styles['Normal']))
        story.append(Paragraph(f"<b>Question:</b> {data['question']}", styles['Normal']))
        story.append(Spacer(1, 20))

        # Retrieved Documents Table
        story.append(Paragraph("<b>Retrieved Evidence:</b>", styles['Heading3']))
        table_data = [["#", "Text", "Sentiment"]]
        for i, d in enumerate(data['retrieved_docs'], 1):
            sentiment = d.metadata['sentiment']
            label = "Positive" if sentiment > 0.1 else ("Negative" if sentiment < -0.1 else "Neutral")
            color = colors.green if sentiment > 0.1 else (colors.red if sentiment < -0.1 else colors.grey)
            table_data.append([i, d.page_content[:200] + "...", f"{sentiment:.3f} ({label})"])

        table = Table(table_data, colWidths=[30, 400, 100])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

        # Reasoning
        story.append(Paragraph("<b>Model Reasoning (Chain-of-Thought):</b>", styles['Heading3']))
        story.append(Paragraph(data['reasoning).replace("\n", "<br/>"), ParagraphStyle(name='Reasoning', fontName='Courier', fontSize=10)))
        story.append(Spacer(1, 20))

        # Final Answer
        story.append(Paragraph("<b>Final Answer:</b>", styles['Heading3']))
        story.append(Paragraph(data['answer'].replace("\n", "<br/>"), styles['Normal']))

        doc.build(story)
        return buffer.getvalue()

    # Query
    query = st.text_input("Ask about sentiments:", placeholder="What are the strongest complaints?")

    if query:
        with st.spinner("Analyzing with visible reasoning..."):
            result = rag_with_reasoning(query)

        st.subheader("Final Answer")
        st.markdown(result["answer"])

        if "thinking" in result["reasoning"].lower():
            with st.expander("Show Model Reasoning", expanded=True):
                st.code(result["reasoning"], language="markdown")

        # PDF Export Button
        pdf_bytes = generate_pdf_report(result)
        st.download_button(
            label="Download Full Report as PDF",
            data=pdf_bytes,
            file_name=f"sentiment_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

        with st.expander("Retrieved Documents"):
            for i, d in enumerate(result["retrieved_docs"], 1):
                st.write(f"**Doc {i}** | Sentiment: {d.metadata['sentiment']:.3f}")
                st.text(d.page_content)