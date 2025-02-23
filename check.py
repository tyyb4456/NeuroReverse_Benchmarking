import pandas as pd
import docx
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import torch

# Upload file paths
users_file = "/content/googlepixel.xlsx"
competitors_file = "/content/samsung.xlsx"

# Function to extract text from different file formats
def extract_text(file_path):
    file_extension = file_path.split(".")[-1]
    if file_extension in ["csv", "xlsx"]:
        df = pd.read_csv(file_path) if file_extension == "csv" else pd.read_excel(file_path)
        return df.to_string(index=False)
    elif file_extension == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_extension == "docx":
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file_extension == "pdf":
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return ""

# Load and process data
def load_data():
    user_data = extract_text(users_file)
    competitor_data = extract_text(competitors_file)
    return f"User Product Data:\n{user_data}\n\nCompetitor Product Data:\n{competitor_data}"

# Split the loaded data into chunks
def split_data(_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
    return text_splitter.create_documents([_data])

# Create a vector store using FAISS
def create_vector_store(_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyDKFc4GFDf6LMEgkchugE70sF-fdjD1Y3k")
    return FAISS.from_documents(documents=_docs, embedding=embeddings)

# Process data
if users_file and competitors_file:
    data = load_data()
    docs = split_data(data)
    vectorstore = create_vector_store(docs)

# Setup retriever and LLM
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ibm-granite/granite-3.1-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

# System prompt
system_prompt = (
    "You are an AI-powered benchmarking assistant. Compare two products (hardware/software) based on specs, features, performance, pricing, and customer feedback, providing objective, data-driven insights.\n\n"
    "🔹 **Instructions:**\n"
    "- Extract key details (specs, features, pricing, reviews).\n"
    "- Compare strengths and weaknesses.\n"
    "- Highlight areas where the competitor outperforms.\n"
    "- Suggest improvements.\n"
    "- If data is insufficient, state: 'Insufficient data to compare this aspect.'\n\n"
    "🔹 **Response Format:**\n"
    "**Competitor vs. Your Product:**\n"
    "- **Feature:** Competitor (Value) vs. Your Product (Value) → (Strength/Weakness)\n"
    "- **Feature:** Competitor (Value) vs. Your Product (Value) → (Strength/Weakness)\n\n"
    "📊 **Improvement Suggestions:**\n"
    "- (Actionable recommendations based on weaknesses)\n\n"
    "Keep responses structured, concise, and data-driven.\n\n"
    f"{data}"
)

# Tokenize input and move to the same device as the model
input_ids = tokenizer(system_prompt, return_tensors="pt", truncation=True, max_length=500).input_ids.to(device)

# Generate response
output_ids = llm.generate(input_ids, max_new_tokens=250, eos_token_id=tokenizer.eos_token_id, temperature=0.3, top_p=0.9)

# Decode response
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("##📊 AI-Generated Product Comparison:")
print(response)