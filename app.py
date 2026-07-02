import os
import shutil
import gradio as gr
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
# HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
def get_llm(api_key, model_name, temperature):
    # Fallback order: user key -> env variable -> default key
    key = api_key.strip() if api_key and api_key.strip() else os.environ.get(
        "GROQ_API_KEY", 
        "SECRET_KEY"
    )
    return ChatGroq(
        temperature=temperature,
        groq_api_key=key,
        model=model_name
    )
def extract_text_from_pdf(file):
    pdf_reader = pypdf.PdfReader(file.name)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text
def build_vectorstore(files):
    # Clear old chroma_db files to prevent mixing collections
    if os.path.exists("chroma_db"):
        try:
            shutil.rmtree("chroma_db")
        except Exception as e:
            print(f"Warning: Failed to clear old database folder: {e}")
    all_text = ""
    for file in files:
        all_text += extract_text_from_pdf(file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(all_text)
    vectorstore = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="chroma_db")
    return vectorstore, all_text
def create_qa_chain(vectorstore, llm):
    # 1. Define prompt for history-aware retriever
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # 2. Create history-aware retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # 3. Define QA prompt with system prompt context and history placeholder
    qa_system_prompt = (
        "You are an expert study assistant. Answer the user's question using the provided context. "
        "If you do not know the answer based on the context, say that you don't know.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # 4. Create document combination chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 5. Create final retrieval chain
    qa_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return qa_chain
def upload_pdfs_handler(files, api_key, model_name, temp_val):
    if not files:
        return "⚠️ No files uploaded.", None, "", []
    
    try:
        # Build vector store
        vectorstore, all_text = build_vectorstore(files)
        
        # Instantiate LLM
        llm = get_llm(api_key, model_name, temp_val)
        
        # Create QA Chain
        qa_chain = create_qa_chain(vectorstore, llm)
        
        return "✅ Lecture notes processed! Use Chat Mode or Study Mode.", qa_chain, all_text, []
    except Exception as e:
        return f"❌ Processing failed: {str(e)}", None, "", []
def user_msg(user_message, history):
    if not user_message:
        return "", history
    new_history = history + [{"role": "user", "content": user_message}]
    return "", new_history
def extract_text_from_gradio_content(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
        return "\n".join(text_parts)
    elif isinstance(content, dict) and "text" in content:
        return content["text"]
    return str(content)
def bot_msg(history, qa_chain):
    if not history:
        return history
    
    raw_user_message = history[-1]["content"]
    user_message = extract_text_from_gradio_content(raw_user_message)
    
    if qa_chain is None:
        new_history = history + [{"role": "assistant", "content": "⚠️ Please upload and process lecture notes in the sidebar first!"}]
        return new_history
    # Convert history (excluding the current user message) to message format
    langchain_history = []
    for msg in history[:-1]:
        role = msg["role"]
        content = msg["content"]
        text_content = extract_text_from_gradio_content(content)
        if role == "user":
            langchain_history.append(HumanMessage(content=text_content))
        elif role == "assistant":
            langchain_history.append(AIMessage(content=text_content))
    try:
        result = qa_chain.invoke({
            "input": user_message,
            "chat_history": langchain_history
        })
        answer = result["answer"]
        new_history = history + [{"role": "assistant", "content": answer}]
    except Exception as e:
        import traceback
        traceback.print_exc()
        new_history = history + [{"role": "assistant", "content": f"❌ Error: {str(e)}"}]
        
    return new_history
def generate_flashcards_handler(lecture_text, api_key, model_name, temp_val):
    if not lecture_text:
        return "⚠️ Please upload and process PDFs first!"
        
    try:
        llm = get_llm(api_key, model_name, temp_val)
        
        prompt = (
            "You are an expert study assistant. Based on the following lecture text, create:\n"
            "1. 5 detailed Flashcards (Question & Answer format).\n"
            "2. 5 Multiple-Choice Quiz questions with options (A, B, C, D) and specify the correct answer with a short explanation.\n\n"
            "Format the entire output as clean, beautiful Markdown. Use headers, bold text, blockquotes, or lists to make it highly readable and visually polished.\n\n"
            f"Lecture Text:\n{lecture_text[:3000]}"
        )
        
        response = llm.invoke(prompt).content
        return response
    except Exception as e:
        return f"❌ Generation failed: {str(e)}"
# Custom CSS for modern dark-themed look
custom_css = """
body {
    background-color: #0f172a !important;
}
.gradio-container {
    max-width: 1300px !important;
}
h1 {
    font-size: 2.2rem !important;
    background: linear-gradient(to right, #38bdf8, #818cf8) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 800 !important;
    margin-bottom: 0.2rem !important;
}
.subtitle {
    color: #94a3b8;
    font-size: 1.0rem;
    margin-bottom: 1.5rem;
}
"""
with gr.Blocks(head="""
<script>
document.documentElement.classList.add('dark');
document.addEventListener("DOMContentLoaded", function() {
    document.documentElement.classList.add('dark');
});
</script>
""") as demo:
    gr.HTML("<div style='text-align: center;'><h1>📘 GenAI Study Assistant</h1><p class='subtitle'>Your premium AI study companion for lecture notes and exam prep.</p></div>")
    # Stateful session variables
    qa_chain_state = gr.State(None)
    lecture_text_state = gr.State("")
    with gr.Row():
        # Sidebar config
        with gr.Column(scale=1, min_width=320):
            with gr.Group():
                gr.Markdown("### ⚙️ Settings")
                api_key_input = gr.Textbox(
                    label="Groq API Key (Optional)",
                    placeholder="Enter your key or leave blank for default",
                    type="password"
                )
                model_dropdown = gr.Dropdown(
                    choices=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
                    value="llama-3.3-70b-versatile",
                    label="Groq Model"
                )
                temp_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                    label="Temperature"
                )
            
            with gr.Group():
                gr.Markdown("### 📂 Upload Notes")
                pdf_input = gr.File(
                    file_types=[".pdf"],
                    label="Select PDF files",
                    file_count="multiple"
                )
                upload_btn = gr.Button("🚀 Process PDFs", variant="primary")
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready. Please upload notes."
                )
        # Main tabs
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("💬 Interactive Chat"):
                    chatbot = gr.Chatbot(label="Chat History", height=550)
                    with gr.Row():
                        query_input = gr.Textbox(
                            show_label=False,
                            placeholder="Ask a question about your uploaded lecture notes...",
                            scale=4
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    clear_btn = gr.ClearButton([query_input, chatbot], value="🗑️ Clear Chat")
                with gr.Tab("🃏 Study Center"):
                    flash_btn = gr.Button("✨ Generate Flashcards & Quiz", variant="primary")
                    flash_output = gr.Markdown(
                        value="*Upload PDFs in the sidebar and click the button above to generate your study material.*"
                    )
    # Event handlers
    upload_btn.click(
        fn=upload_pdfs_handler,
        inputs=[pdf_input, api_key_input, model_dropdown, temp_slider],
        outputs=[status_output, qa_chain_state, lecture_text_state, chatbot]
    )
    submit_event = query_input.submit(
        fn=user_msg,
        inputs=[query_input, chatbot],
        outputs=[query_input, chatbot],
        queue=False
    ).then(
        fn=bot_msg,
        inputs=[chatbot, qa_chain_state],
        outputs=[chatbot]
    )
    click_event = submit_btn.click(
        fn=user_msg,
        inputs=[query_input, chatbot],
        outputs=[query_input, chatbot],
        queue=False
    ).then(
        fn=bot_msg,
        inputs=[chatbot, qa_chain_state],
        outputs=[chatbot]
    )
    flash_btn.click(
        fn=generate_flashcards_handler,
        inputs=[lecture_text_state, api_key_input, model_dropdown, temp_slider],
        outputs=[flash_output]
    )
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"), css=custom_css)
