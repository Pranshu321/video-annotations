# 🧠 Surveillance Event Chat Assistant (Streamlit + LangChain + Groq)

This project is an **AI-powered assistant** that summarizes and analyzes surveillance events detected from video or JSON input. Built using **Streamlit**, **LangChain**, and **Groq API**, it supports interactive chat with memory, event summarization, and file uploads.

### Demo Video

[![asciicast](https://github.com/user-attachments/assets/5a102cef-e032-4c61-9869-0622cf07542c)](https://github.com/user-attachments/assets/b94c11fc-5b65-43ed-b3f6-3fb6965e7ced)



---

## 🚀 Features

✅ Upload surveillance videos or JSON event logs  
✅ Automatically detect and summarize violations  
✅ Ask follow-up questions using a natural language chatbot  
✅ Built-in memory using FAISS and LangChain  
✅ Uses Groq’s blazing-fast LLaMA 3 model  
✅ Works with sample/demo data for testing

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/) — UI framework  
- [LangChain](https://www.langchain.com/) — LLM orchestration  
- [Groq](https://groq.com/) — LLM inference API (LLaMA 3, Gemma)  
- [FAISS](https://github.com/facebookresearch/faiss) — Vector memory  
- [Hugging Face Transformers](https://huggingface.co/) — Embeddings  
- Python 3.10+

---

## App Interface

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/5a102cef-e032-4c61-9869-0622cf07542c" />

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/ac0cb9b4-c59a-4060-8923-0690649fd9e1" />

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/c8088387-032a-4ea5-9005-332c8071046a" />


---

## 🛠️ Local Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Set Your Groq API Key

1) Go to https://console.groq.com/keys
2) Copy your API key
Then in your terminal:
```bash
export OPENAI_API_KEY=your-groq-key
export OPENAI_API_BASE=https://api.groq.com/openai/v1
```
Alternatively, you can paste your key directly into main.py (not recommended for production).

### 5. Run the App

```bash
streamlit run main.py
```
Then open http://localhost:8501 in your browser.

## 🧪 How to Test

You can test the app using any of the following options:

### ✅ Option 1: Upload a JSON File

Click the “Upload surveillance events JSON” uploader in the app and upload a .json file with a structure like:
```bash
[
  {
    "timestamp": "2024-01-01 10:30:00",
    "event_type": "Unauthorized Access",
    "location": "Main Entrance",
    "confidence": 0.95
  },
  {
    "timestamp": "2024-01-01 11:15:00",
    "event_type": "Suspicious Activity",
    "location": "Parking Lot",
    "confidence": 0.78
  }
]
```

🔁 The assistant will summarize these and let you ask follow-up questions via chat.

### ✅ Option 2: Upload a Surveillance Video

Click the “Upload surveillance video” button and upload a .mp4, .avi, .mov, or .mkv file.
The app uses a VoilationDetection class to process the video and extract events. If this function is implemented properly, it will generate detected_violations.json internally.

### ✅ Option 3: Use a Default File

Place a detected_violations.json file in the root directory of the project.

The app will automatically load it if no file is uploaded manually.

### ✅ Option 4: Use Built-in Sample Data

Click the “Use Sample Data” button on the UI.

This loads demo events like:

```bash
[
  {
    "timestamp": "2024-01-01 10:30:00",
    "event_type": "Unauthorized Access",
    "location": "Main Entrance",
    "confidence": 0.95
  },
  {
    "timestamp": "2024-01-01 11:15:00",
    "event_type": "Suspicious Activity",
    "location": "Parking Lot",
    "confidence": 0.78
  },
  {
    "timestamp": "2024-01-01 12:00:00",
    "event_type": "Person Detection",
    "location": "Corridor A",
    "confidence": 0.92
  }
]
```

Once loaded, the app will:

● ✅ Summarize all events using the selected LLM (e.g. LLaMA 3 via Groq)

● ✅ Allow interactive chat about the events using LangChain memory

## 🧠 Example Questions You Can Ask

"What happened at the main entrance?"

"How many events were detected in total?"

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/edef5a43-279f-4762-8624-72c4c7af750e" />


"Which event had the highest confidence?"

"Was there any unauthorized access?"

The assistant will respond using your uploaded/simulated data and memory context.
