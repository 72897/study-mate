# 🎓 GenAI-Powered Study Assistant

A student-focused Generative AI tool that helps in learning and interview preparation.  
Built with **Gradio**, **Groq API**, **ChromaDB**, and **Hugging Face free embeddings**.

---

## 🚀 Features

- 📄 **Lecture Summarization**  
  Upload lecture PDFs/slides and get concise summaries.

- 🧠 **Flashcards & Quizzes**  
  Automatically generate flashcards and quizzes for revision.

- 🎯 **Placement Q&A Generator**  
  Create practice interview questions & answers for placement preparation.

- 💬 **Chat with Context Memory**  
  Ask questions with **chat history saved in ChromaDB** for contextual answers.

- 🆓 **Free Embeddings**  
  Uses Hugging Face embedding models for vector storage.

---

## 🛠️ Tech Stack

- **Frontend/UI**: [Gradio](https://www.gradio.app/)  
- **LLM API**: [Groq API](https://groq.com/) (free tier model)  
- **Embeddings**: Hugging Face sentence-transformers  
- **Database**: [ChromaDB](https://www.trychroma.com/) for vector storage & chat history  
- **Framework**: LangChain (for chaining tasks and RAG pipeline)  
- **Hosting**: Hugging Face Spaces (free deployment option)  

---

## 📂 Project Structure

genai-study-assistant/
│── app.py # Main Gradio app
│── requirements.txt # Dependencies
│── utils/
│ ├── pdf_reader.py # Extract text from PDF/slides
│ ├── qa_generator.py # Placement Q&A logic
│ ├── quiz_generator.py # Flashcards & quiz creation
│── db/
│ └── chroma_store/ # ChromaDB storage
│── README.md # Documentation

yaml
Copy
Edit

---

## ⚡ Installation & Usage

1. **Clone this repository**
   ```bash
   git clone <repo-name>
   cd genai-study-assistant
Create virtual environment & install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set environment variables

bash
Copy
Edit
export GROQ_API_KEY="your_groq_api_key"
Run the app

bash
Copy
Edit
python app.py
Open in browser: http://localhost:7860

📌 Future Improvements
Add support for voice-based queries

Multi-language summarization

Export flashcards to Anki or CSV

Integration with Google Drive / Notion

🙌 Contribution
Pull requests are welcome!
For major changes, please open an issue first to discuss what you would like to change.

📜 License
This project is licensed under the MIT License.
