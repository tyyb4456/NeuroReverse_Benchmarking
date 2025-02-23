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
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict,Optional,Any, List
from langchain_community.document_loaders import UnstructuredExcelLoader
import uvicorn
import os
import shutil

from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define upload directories
BASE_UPLOAD_FOLDER = "uploads"
USER_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, "user")
COMPETITOR_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, "competitor")

os.makedirs(USER_FOLDER, exist_ok=True)
os.makedirs(COMPETITOR_FOLDER, exist_ok=True)

response: Optional[Dict[str, Any]] = None  # Store last response in memory

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!", "last_response": response}

@app.post("/upload/")
async def upload_files(
    user_files: List[UploadFile] = File(None), 
    competitor_files: List[UploadFile] = File(None)
):
    """
    Uploads new files and deletes previous files before saving the new ones.
    """
    if not user_files and not competitor_files:
        raise HTTPException(status_code=400, detail="No files provided.")

    uploaded_files = {"user": [], "competitor": []}

    def clear_and_save_files(files, folder, category):
        # Remove previous files
        for existing_file in os.listdir(folder):
            os.remove(os.path.join(folder, existing_file))

        # Save new files
        for file in files:
            if file.filename:
                file_path = os.path.join(folder, os.path.basename(file.filename))  # Secure filename usage
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                uploaded_files[category].append(file.filename)

    if user_files:
        clear_and_save_files(user_files, USER_FOLDER, "user")
    if competitor_files:
        clear_and_save_files(competitor_files, COMPETITOR_FOLDER, "competitor")

    return {
        "uploaded_files": uploaded_files,
        "message": "Previous files removed. New files uploaded successfully."
    }

def load_and_process_files(data_type: str):
    """
    Process uploaded files separately for user and competitor.
    """
    folder = USER_FOLDER if data_type == "user" else COMPETITOR_FOLDER
    uploaded_files = os.listdir(folder)

    if not uploaded_files:
        raise HTTPException(status_code=400, detail=f"No {data_type} files found. Please upload files first.")

    all_splits = []
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".docx": Docx2txtLoader,
        ".xlsx": UnstructuredExcelLoader
    }

    for file_name in uploaded_files:
        file_path = os.path.join(folder, file_name)
        ext = os.path.splitext(file_name)[1].lower()

        if ext not in loaders:
            print(f"Skipping unsupported file type: {file_name}")  # Logging unsupported files
            continue

        try:
            loader = loaders[ext](file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")  # Debugging output

    if not all_splits:
        raise HTTPException(status_code=400, detail=f"No valid documents processed for {data_type}.")

    # Generate embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)

    return vectorstore.as_retriever()

@app.post("/response")
def init_session():
    global response
    response = None

    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        # Load data separately for user and competitor
        user_data_retriever = load_and_process_files("user")
        competitor_data_retriever = load_and_process_files("competitor")

        # Retrieve relevant documents
        user_docs = user_data_retriever.get_relevant_documents("Extract key product features and specifications.")
        competitor_docs = competitor_data_retriever.get_relevant_documents("Extract key product features and specifications.")

        if not user_docs or not competitor_docs:
            raise HTTPException(status_code=400, detail="Insufficient data for comparison.")

        # Construct system prompt with retrieved data
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
            f"User Product Data:\n{user_docs}\n\nCompetitor Product Data:\n{competitor_docs}"
        )

        response_message = model.invoke(system_prompt)
        response = {"message": response_message.content}

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=420)
