import streamlit as st
import pandas as pd
from textblob import TextBlob
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime
from io import BytesIO

# PDF Libraries (already fixed)
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

load_dotenv()

# OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

if not os.getenv("OPENROUTER_API_KEY"):
    st.error("Add OPENROUTER_API_KEY to .env file!")
    st.stop()

st.title("Sentiment Analysis Bot...")

uploaded_file = st.file_uploader("Upload CSV/Excel with 'text' column", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    
    if "text" not in df.columns:
        st.error("Column 'text' not found!")
        st.stop()

    # Sentiment Analysis
    with st.spinner("Analyzing sentiment..."):
        df["sentiment"] = df["text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    st.success("Sentiment analysis complete!")
    st.dataframe(df)

    # Build Chroma Vector DB (auto-saved to ./chroma_db)
    with st.spinner("Building Chroma vector database..."):
        embeddings = OpenAIEmbeddings(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="openai/text-embedding-3-small"
        )

        documents = [
            Document(page_content=row["text"], metadata={"sentiment": row["sentiment"]})
            for _, row in df.iterrows()
        ]

        # Chromadb persists automatically!
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./chroma_db"  # Saves to disk
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    st.success("Chroma DB ready & saved to disk!")

    # RAG + Reasoning Function
    def rag_with_reasoning(query: str):
        docs = retriever.invoke(query)
        context = "\n\n".join(
            f"Text: {d.page_content}\nSentiment: {d.metadata['sentiment']:.3f}"
            for d in docs
        )

        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:turbo",
            messages=[
                {"role": "system", "content": "You are an expert sentiment analyst. Always think step-by-step before answering."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            extra_body={"reasoning": {"enabled": True}},
            temperature=0.7,
            max_tokens=2000
        )

        msg = response.choices[0].message
        reasoning = getattr(msg, "reasoning_details", None)
        reasoning_text = reasoning.thinking if reasoning else "No visible reasoning."

        return {
            "question": query,
            "answer": msg.content.strip(),
            "reasoning": reasoning_text,
            "retrieved_docs": docs,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    # PDF Generator (same as before, fixed)
    def generate_pdf_report(data):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Reasoning', fontName='Courier', fontSize=10, leading=12))

        story = []
        story.append(Paragraph("Sentiment Analysis Report", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Date: {data['timestamp']}", styles['Normal']))
        story.append(Paragraph(f"Question: {data['question']}", styles['Normal']))
        story.append(Spacer(1, 20))

        # Table of retrieved docs
        table_data = [["#", "Sentiment", "Text"]]
        for i, d in enumerate(data['retrieved_docs'], 1):
            s = d.metadata['sentiment']
            table_data.append([i, f"{s:+.3f}", d.page_content[:200] + "..."])

        table = Table(table_data, colWidths=[30, 80, 400])
        table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
        story.append(table)
        story.append(Spacer(1, 20))

        story.append(Paragraph("<b>Reasoning:</b>", styles['Heading3']))
        story.append(Paragraph(data['reasoning'].replace("\n", "<br/>"), styles['Reasoning']))
        story.append(Spacer(1, 20))

        story.append(Paragraph("<b>Final Answer:</b>", styles['Heading3']))
        story.append(Paragraph(data['answer'].replace("\n", "<br/>"), styles['Normal']))

        doc.build(story)
        return buffer.getvalue()

    # Query
    query = st.text_input("Ask about the sentiments:", placeholder="What are the main complaints?")

    if query:
        with st.spinner("Thinking..."):
            result = rag_with_reasoning(query)

        st.subheader("Answer")
        st.markdown(result["answer"])

        if "thinking" in result["reasoning"].lower():
            with st.expander("Show Reasoning", expanded=True):
                st.code(result["reasoning"])

        # PDF Export Button
        pdf = generate_pdf_report(result)
        st.download_button(
            "Download Full Report as PDF",
            data=pdf,
            file_name=f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )