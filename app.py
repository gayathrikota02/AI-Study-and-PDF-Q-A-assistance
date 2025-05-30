import streamlit as st
from PIL import Image
from ocr_utils import extract_text_from_image
from ai_utils import get_solution
import speech_recognition as sr
import os
import logging
from langchain.llms.base import LLM
from typing import Optional, List
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import PrivateAttr
from dotenv import load_dotenv
import concurrent.futures
from concurrent.futures import TimeoutError as FuturesTimeoutError
from google.api_core.exceptions import ResourceExhausted

import google.generativeai as genai

genai.configure(api_key="AIzaSyACwPlnTCOgxDQQh80g_DNld7cj590bjW0")

import pytesseract

# Replace with your actual installed path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeminiLLM(LLM):
    model_name: str = "gemini-2.0-flash-lite"
    _model: any = PrivateAttr()
    timeout: int = 60

    def __init__(self, model_name="gemini-2.0-flash-lite", **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=os.getenv("AIzaSyACwPlnTCOgxDQQh80g_DNld7cj590bjW0"))
        self._model = genai.GenerativeModel(model_name)

    @property
    def _llm_type(self) -> str:
        return "custom_gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self._model.generate_content, prompt)
            try:
                response = future.result(timeout=self.timeout)
                return response.text
            except FuturesTimeoutError:
                return "Error: LLM call timed out."
            except ResourceExhausted:
                return "Error: Gemini API quota exhausted."


def load_and_split_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

def create_vectorstore(docs):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embedding_model)
def answer_question(gemini_llm, context, question: str):
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    return gemini_llm(prompt)

st.set_page_config(page_title="📚 AI Study & PDF Q&A Assistant", layout="wide")

st.markdown("""
    <style>
    html, body, .main {
        background-color: #f0f2f6;
        font-family: 'Poppins', sans-serif;
        color: #333;
    }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    h1 {
        background: linear-gradient(90deg, #7b4397, #dc2430);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 2rem;
    }
    .card {
        background: rgba(255, 255, 255, 0.85);
        padding: 1.8rem;
        border-radius: 1.2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.07);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid #e6eaf0;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #444;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📚 AI Study & PDF Q&A Assistant")


# Initialize session state variables
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "quiz_mode" not in st.session_state:
    st.session_state.quiz_mode = False

if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = {}

# --- Speech to text function ---
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak your question.")
        audio = r.listen(source, phrase_time_limit=10)
    try:
        text = r.recognize_google(audio)
        st.success(f"Recognized: {text}")
        return text
    except Exception as e:
        st.error(f"Speech recognition error: {e}")
        return ""

# --- Quiz Data ---
quiz_questions = [
    {
        "question": "What is the derivative of $x^2$?",
        "options": ["$2x$", "$x$", "$x^2$", "$1$"],
        "answer": "$2x$"
    },
    {
        "question": "Which element has the atomic number 1?",
        "options": ["Oxygen", "Hydrogen", "Carbon", "Helium"],
        "answer": "Hydrogen"
    },
    {
        "question": "What is the formula for force?",
        "options": ["$F=ma$", "$E=mc^2$", "$V=IR$", "$P=IV$"],
        "answer": "$F=ma$"
    },
]

# --- Streamlit UI ---


mode = st.sidebar.selectbox("Select Mode", ["Study Assistant", "Interactive Quiz" , "PDF Q&A Assistant"])

if mode == "Study Assistant":
    subject = st.selectbox("Select Subject", ["Math", "Physics", "Chemistry","Biology", "English",  "Computer Science", 
                                               "History", "Geography"])
    language = st.selectbox("Select Language", ["English", "Hindi", "Telugu","Spanish", "French"])
    input_mode = st.radio("Choose Input Mode", ["Text", "Image", "Speech"])

    question = ""

    if input_mode == "Text":
        question = st.text_area("Enter your problem statement")

    elif input_mode == "Image":
        uploaded_image = st.file_uploader("Upload a problem image", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            question = extract_text_from_image(image)
            st.write("📜 Extracted Text:")
            st.success(question)

    elif input_mode == "Speech":
        if st.button("Start Recording"):
            question = speech_to_text()
            if question:
                st.session_state.last_spoken_text = question
        question = st.session_state.get("last_spoken_text", "")

    if st.button("Get Solution"):
        if question:
            with st.spinner("Getting AI-powered step-by-step solution..."):
                solution = get_solution(question, language)
            st.markdown("## 📖 Solution")
            st.markdown(solution)
            st.session_state.qa_history.append({"question": question, "answer": solution})
        else:
            st.warning("Please enter a question, upload an image, or use speech input.")

    # Show Q&A History
    if st.session_state.qa_history:
        st.markdown("---")
        st.header("📚 Q&A History")
        for i, qa in enumerate(st.session_state.qa_history[::-1], 1):
            st.markdown(f"**Q{i}:** {qa['question']}")
            st.markdown(f"**A{i}:** {qa['answer']}")

elif mode == "Interactive Quiz":
    st.header("📝 Interactive Quiz Mode")

    if "quiz_index" not in st.session_state:
        st.session_state.quiz_index = 0
        st.session_state.score = 0
        st.session_state.quiz_finished = False

    if not st.session_state.quiz_finished:
        q = quiz_questions[st.session_state.quiz_index]
        st.markdown(f"### Q{st.session_state.quiz_index +1}: {q['question']}")
        option = st.radio("Choose your answer:", q["options"])
        if st.button("Submit Answer"):
            if option == q["answer"]:
                st.success("Correct!")
                st.session_state.score += 1
            else:
                st.error(f"Incorrect! Correct Answer: {q['answer']}")
            st.session_state.quiz_index += 1
            if st.session_state.quiz_index >= len(quiz_questions):
                st.session_state.quiz_finished = True
            st.rerun()

    else:
        st.markdown(f"### Quiz Completed! Your score: {st.session_state.score}/{len(quiz_questions)}")
        if st.button("Restart Quiz"):
            st.session_state.quiz_index = 0
            st.session_state.score = 0
            st.session_state.quiz_finished = False
            st.rerun()

elif mode == "PDF Q&A Assistant":
    st.header("📥 Upload PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.pdf_data:
                temp_dir = "temp"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    docs = load_and_split_pdf(temp_path)
                    vectorstore = create_vectorstore(docs)
                    gemini_llm = GeminiLLM()
                    st.session_state.pdf_data[uploaded_file.name] = {
                        "docs": docs,
                        "vectorstore": vectorstore,
                        "llm": gemini_llm,
                        "qa_answer": None
                    }

                    st.success(f"{uploaded_file.name} processed successfully!")

                except ResourceExhausted:
                    st.error("Gemini API quota exceeded while processing this PDF.")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

    if st.session_state.pdf_data:
        selected_pdf = st.sidebar.selectbox("📑 Select a processed PDF", list(st.session_state.pdf_data.keys()))
        data = st.session_state.pdf_data[selected_pdf]
        vectorstore = data["vectorstore"]
        gemini_llm = data["llm"]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">💬 Ask a Question from: {selected_pdf}</div>', unsafe_allow_html=True)
        question_input = st.text_input("Type your question:")

        @st.cache_data(show_spinner=False)
        def get_local_context(_vectorstore, query):
            return _vectorstore.similarity_search(query, k=3)

        if st.button("🎙 Get Answer") and question_input:
            with st.spinner("📝 Generating your answer..."):
                local_docs = get_local_context(vectorstore, question_input)
                context_text = "\n\n".join([doc.page_content for doc in local_docs])
                answer = answer_question(gemini_llm, context_text, question_input)
                data["qa_answer"] = answer
        if data.get("qa_answer"):
            st.success(data["qa_answer"])

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("📂 Upload PDFs from the sidebar to start.")