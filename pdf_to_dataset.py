from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma 
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings,OpenAIEmbeddings
from secret_key import hugging_face_api_key,openai_api_key
import os
import shutil

CHROMA_PATH = "chroma"

def main():
    generate_data_store(pdf_file)

def generate_data_store(pdf_file):
    document = load_document(pdf_file)
    chunks = split_text(document)
    save_to_chroma(chunks)

def load_document(pdf_file):
    loader = PyPDFLoader(pdf_file)
    document = loader.load()

    return document 

def split_text(document):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,  
    length_function = len,
    add_start_index=True, 
    )
    chunks = text_splitter.split_documents(document)
    # print(f"Split {len(document)} documents into {len(chunks)} chunks.")

    # document = chunks[10]
    # print(document.page_content)
    # print(document.metadata)

    return chunks 

def save_to_chroma(chunks):
    #clear out database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    #create a new db from doc
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(api_key=openai_api_key), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} to {CHROMA_PATH}")



if __name__ =="__main__":
    main()