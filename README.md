# PDF_Answering_AI
# 1) PDF to Dataset in Chunks
This project demonstrates how to load a PDF document, split its content into manageable chunks, and save these chunks into a Chroma vector store using embeddings. The code uses the langchain library for handling the PDF and text processing, and supports both HuggingFace and OpenAI embeddings.

- Prerequisites
Python 3.7+,
- Required Python packages:
langchain, 
PyPDFLoader, 
RecursiveCharacterTextSplitter, 
Chroma,  
HuggingFaceInferenceAPIEmbeddings, and
OpenAIEmbeddings
 
# Code Explanation
load_document(pdf_file)
- This function uses PyPDFLoader to load the PDF document and return its content.

split_text(document)
- This function splits the loaded document into smaller chunks using RecursiveCharacterTextSplitter. Each chunk has a size of 1000 characters with an overlap of 100 characters to ensure context continuity.

save_to_chroma(chunks)
 - This function: Clears any existing Chroma data store if it exists.
Creates a new Chroma data store from the document chunks using OpenAIEmbeddings.
Persists the Chroma data store to disk.

# 2) Streamlit App for Earnings Call Transcript Analysis
- This Streamlit application is designed to process and analyze earnings call transcripts using LangChain and OpenAI models. The app allows users to summarize transcripts, perform question-answer analysis, and interact with a chatbot based on the transcript content.
- # Features
1. Welcome Page
- Summarize Transcript and Download: Upload an earnings call transcript in PDF format, process it to extract information such as company name, quarter, conference call date, number of pages, and management info. Summarized content is displayed, and users can interact with a conversational AI for additional insights.
2. ChatBot
  - Interactive Chatbot: Allows users to ask questions based on the entire earnings call transcript. The chatbot utilizes conversational memory and retrieval mechanisms to maintain context and provide accurate responses.
