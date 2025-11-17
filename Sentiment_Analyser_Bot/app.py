import streamlit as st
import pandas as pd
from textblob import TextBlob
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import importlib

# Try community vectorstore first, fall back to core langchain, and give an actionable error.
FAISS = None
for _mod in ("langchain_community.vectorstores", "langchain.vectorstores"):
    try:
        mod = importlib.import_module(_mod)
        FAISS = getattr(mod, "FAISS")
        break
    except Exception:
        FAISS = None

if FAISS is None:
    raise ImportError(
        "Could not import FAISS vectorstore. Install dependencies:\n"
        "  pip install langchain-community faiss-cpu\n"
        "or, if using core langchain:\n"
        "  pip install langchain faiss-cpu\n"
    )
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found in .env file. Please add it and restart the app.")
    st.stop()

st.title("Sentiment Analysis Bot with Advanced RAG")

uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    if "text" not in df.columns:
        st.error("Dataset must have a 'text' column.")
        st.stop()
    
    # Perform sentiment analysis
    st.write("Analyzing sentiment...")
    df["sentiment"] = df["text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # Display results
    st.subheader("Analyzed Dataset")
    st.dataframe(df)
    
    # Build vector store
    st.write("Building RAG vector store...")
    embeddings = OpenAIEmbeddings()
    documents = [
        Document(page_content=row["text"], metadata={"sentiment": row["sentiment"]})
        for _, row in df.iterrows()
    ]
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Set up retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Define advanced prompt with system message, few-shot examples, and CoT instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert sentiment analyst. Analyze the provided context, which includes texts and their sentiment polarities (-1 very negative, 0 neutral, +1 very positive). 
Provide thoughtful answers by:
1. Summarizing the sentiment distribution (e.g., mostly positive/negative).
2. Quoting relevant texts with their sentiments.
3. Drawing insights or patterns.
4. Directly answering the question.

Few-shot examples:
- Question: Summarize positive sentiments about AI.
  Context: Text: AI is amazing. Sentiment: 0.8\nText: AI scares me. Sentiment: -0.5
  Answer: Positive sentiments include "AI is amazing" (0.8). Overall, mixed but leaning positive on innovation.

- Question: What are negative opinions?
  Context: Text: Product is bad. Sentiment: -0.7\nText: Love it! Sentiment: 0.9
  Answer: Negative opinions like "Product is bad" (-0.7) highlight quality issues, contrasting with positives."""),
        ("human", """Context:
{context}

Question: {question}

Answer:"""),
    ])
    
    # Set up LLM and chain
    llm = ChatOpenAI(temperature=0.7)
    chain = {"context": RunnablePassthrough(), "question": RunnablePassthrough()} | prompt | llm
    
    # Query input
    query = st.text_input("Ask a question about the dataset (Advanced RAG will retrieve, analyze sentiments, and generate a nuanced response):")
    
    if query:
        # Retrieve and format documents
        docs = retriever.get_relevant_documents(query)
        formatted_context = "\n\n".join(
            f"Text: {doc.page_content}\nSentiment: {doc.metadata['sentiment']:.2f}" 
            for doc in docs
        )
        
        # Run the chain
        response = chain.invoke({"context": formatted_context, "question": query}).content
        st.subheader("Advanced RAG Response")
        st.write(response)