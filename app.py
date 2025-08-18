import os
import gradio as gr
import PyPDF2
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ use HuggingFace embeddings

GROQ_API_KEY = "your_groq_api_key_here"

# LLM from Groq (free & fast)
llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model="deepseek-r1-distill-llama-70b"
)




embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file.name)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


def build_vectorstore(files):
    all_text = ""
    for file in files:
        all_text += extract_text_from_pdf(file)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(all_text)

    vectorstore = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="chroma_db")
    return vectorstore, all_text


def create_qa_chain(vectorstore):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa_chain



chat_history = []
qa_chain = None
lecture_text = ""





def upload_pdfs(files):
    global qa_chain, chat_history, lecture_text
    vectorstore, all_text = build_vectorstore(files)
    qa_chain = create_qa_chain(vectorstore)
    chat_history = []
    lecture_text = all_text
    return "✅ Lecture notes processed! Use Chat Mode or Study Mode."

def chat_with_notes(query):
    global qa_chain, chat_history
    if qa_chain is None:
        return "Please upload PDFs first.", ""

    result = qa_chain({"question": query, "chat_history": chat_history})
    answer = result["answer"]
    chat_history.append((query, answer))

    formatted_history = "\n".join([f"👩‍🎓 {q}\n🤖 {a}" for q, a in chat_history])
    return answer, formatted_history

def generate_flashcards():
    global lecture_text
    if not lecture_text:
        return "Upload PDFs first!"
    prompt = f"Create 5 flashcards (Q&A format) and 5 multiple-choice quiz questions from this text:\n\n{lecture_text[:2000]}"
    response = llm.invoke(prompt).content
    return response




with gr.Blocks() as demo:
    gr.Markdown("# 📘 GenAI Study Assistant (Groq + Chroma + Hugging Face Embeddings)")

    with gr.Row():
        pdf_input = gr.File(file_types=[".pdf"], label="Upload Lecture Notes", file_count="multiple")
        upload_btn = gr.Button("Process PDFs")
        status_output = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("💬 Chat Mode"):
        query_input = gr.Textbox(label="Ask a Question")
        ask_btn = gr.Button("Ask")
        answer_output = gr.Textbox(label="Answer", lines=5)
        history_output = gr.Textbox(label="Chat History", lines=15)

    with gr.Tab("🃏 Study Mode"):
        flash_btn = gr.Button("Generate Flashcards & Quizzes")
        flash_output = gr.Textbox(label="Flashcards & Quizzes", lines=15)

    upload_btn.click(upload_pdfs, inputs=[pdf_input], outputs=[status_output])
    ask_btn.click(chat_with_notes, inputs=[query_input], outputs=[answer_output, history_output])
    flash_btn.click(generate_flashcards, outputs=[flash_output])

demo.launch()
