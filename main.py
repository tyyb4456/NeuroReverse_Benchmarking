from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI, HTTPException, Query, UploadFile,File
from pydantic import BaseModel
from typing import Dict,Optional,Any, List
from langchain_community.document_loaders import UnstructuredExcelLoader
import uvicorn
import os
import shutil

from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists

class ResponseRequest(BaseModel):
    docs_choice: int

# Store the last response in memory
latest_response: Optional[Dict[str, Any]] = None

@app.get("/")
def read_root():
    if latest_response is None:
        return {"message": "FastAPI is running! No response available yet."}
    return latest_response

@app.post("/upload/")
async def upload_file(files: List[UploadFile] = File(...)):
    """
    Endpoint to upload multiple PDF files.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files selected.")

    uploaded_files = []

    for file in files:
        if file.filename is None or not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Invalid file format: {file.filename}. Only PDFs are allowed.")

        if not file.filename:
            raise HTTPException(status_code=400, detail="File name is missing.")
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        # Save file to server
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        uploaded_files.append(file.filename)

    return {"uploaded_files": uploaded_files, "message": "Files uploaded successfully."}

def data():
    uploaded_files = os.listdir(UPLOAD_FOLDER)
    if not uploaded_files:
        raise HTTPException(status_code=400, detail="No files found. Please upload files first.")

    all_splits = []

    # Supported file extensions and their respective loaders
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".docx": Docx2txtLoader,
        ".xlsx": UnstructuredExcelLoader
    }

    # Process uploaded files
    for file_name in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        ext = os.path.splitext(file_name)[1].lower()

        if ext not in loaders:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        loader = loaders[ext](file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        all_splits.extend(splits)

    if not all_splits:
        raise HTTPException(status_code=400, detail="No valid documents processed.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)

    return vectorstore.as_retriever()

@app.post("/response")
def init_session(request: ResponseRequest):
    try:
        docs_choice = request.docs_choice

        global response
        response = None

        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)


        user_data_retriever = data()
        competitor_data_retriever = data()

        # System prompt for LLM
        system_prompt = (
            "You are an AI-powered competitor benchmarking assistant, designed to analyze and compare different products, "
            "whether hardware (e.g., smartphones, refrigerators, cars) or software (e.g., SaaS tools, AI applications, operating systems). "
            "Your goal is to provide objective, data-driven insights by comparing the user's product with a competitor's product "
            "based on retrieved specifications, features, performance metrics, and market positioning.\n\n"

            "🔹Instructions for Analysis:\n"
            "1. Extract relevant details from the retrieved product data, including technical specifications, features, performance metrics, pricing, and customer reviews.\n"
            "2. Compare the user's product with the competitor's, highlighting key **strengths and weaknesses**.\n"
            "3. Identify specific areas where the competitor's product outperforms the user's product and provide **actionable improvement suggestions**.\n"
            "4. Ensure responses are **precise, objective, and based strictly on retrieved data**—do not assume missing details.\n"
            "5. If sufficient data is not available, respond with 'I don't have enough details to compare this aspect.'\n\n"

            "🔹Response Format:\n"
            "Competitor vs. Your Product:\n"
            "Feature 1: Competitor (Value) vs. Your Product (Value) → (Strength/Weakness)\n"
            "Feature 2: Competitor (Value) vs. Your Product (Value) → (Strength/Weakness)\n"
            "Feature 3: Competitor (Value) vs. Your Product (Value) → (Strength/Weakness)\n\n"

            "📊 Innovation & Improvement Suggestions:\n"
            "- (Bullet points suggesting specific improvements based on weaknesses)\n\n"

            "Ensure responses remain structured, concise, and data-driven. If the retrieved context lacks information, state 'I don't have sufficient data to compare this feature.'\n\n"
            f"User Product Data:\n{user_data_retriever}\n\nCompetitor Product Data:\n{competitor_data_retriever}"
        )

        response = model.invoke(system_prompt)

        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)