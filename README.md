# SurveilSense

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/github/actions/workflow/status/Pranshu321/SurveilSense/main.yml?branch=main)]()



<!-- TODO: Add a brief description of the project here -->

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack / Key Dependencies](#tech-stack--key-dependencies)
- [File Structure Overview](#file-structure-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage / Getting Started](#usage--getting-started)
- [Configuration](#configuration)
- [Performance Benchmarks](#performance-benchmarks)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

<!-- TODO: Add screenshots if applicable -->

## Features

🎥 Video Question Answering API
A FastAPI-based system that processes videos, extracts visual events, summarizes them using an LLM, and answers natural language questions (including multiple-choice) about the video content.

📌 Features
- Upload a video and ask any question about its content
- Multiprocessing-based fast frame extraction (OpenCV)
- Transformer models for action recognition and image captioning
- LLM-based summarization (Groq / LangChain)
- buffer Memory-powered QA Agent with LLM
- Optimized for low latency and high accuracy

## Architecture

<img width="1377" height="597" alt="EdrawMax-AI-diagram" src="https://github.com/user-attachments/assets/5aa92503-afe3-41b7-815a-201d65ecd054" />

```text
[ Client (curl/HTTP) ]
           ↓ (multipart POST: video + prompt)
/infer (FastAPI)
           ↓
[ Frame Extraction ] ← multiprocess_video.py
           ↓
[ Video Model (Actions + Captions) ] ← model.py → inference.py
           ↓
[ Summarizer (Groq) ] ← summerizer.py
           ↓
[ QA Agent (Groq + FAISS) ] ← qa_agent.py
           ↓
Plain-text answer → Client
```

## Tech Stack / Key Dependencies

| Layer              | Tools & Libraries                                | Reason                                          |
| ------------------ | ------------------------------------------------ | ----------------------------------------------- |
| Web server         | `FastAPI`, `uvicorn`                             | Ultra-fast, lightweight API                     |
| Video processing   | `opencv-python`, `multiprocessing`               | Efficient frame extraction                      |
| Vision models      | `torch`, `transformers`, `AutoImageProcessor`    | State-of-the-art vision inference               |
| Summarization + QA | `langchain-openai`, Groq LLM via OpenAI API      | Low-latency summarization and conversational QA |
| Retrieval          |  `sentence-transformers` embeddings              | Accurate context retrieval                      |
| Deployment & env   | `python-dotenv`, `Render`                        | Config management & GPU acceleration            |


## File Structure Overview

```text
.
├── .gitignore
├── fi.py
├── inference.py
├── main.py
├── model.py
├── multiprocess_video.py
├── qa_agent.py
├── requirements.txt
└── summerizer.py
```

## Prerequisites

- Python 3.10+
- Groq API key
- GPU recommended for faster inference

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Pranshu321/SurveilSense.git
   cd SurveilSense
   ```
2. Create Virtual Envionment:
   ```bash
   python -m venv .venv
   ```
3. Activate Virtual Envionment:
   ```bash
   source .venv/bin/activate
   ```
4. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage / Getting Started

1.  To run the main application:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
Example API request:
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "accept: text/plain" \
  -F "video=@path/to/video.mp4" \
  -F "prompt=What action is happening?"
```

## Configuration

Create a .env file:
```text
GROQ_API_KEY=your_api_key_here
```

## Performance Benchmarks

| Task                     | Metric              | Value                           |
| ------------------------ | ------------------- | ------------------------------- |
| Frame extraction         | FPS (8-core CPU)    | \~25 fps                        |
| Vision inference (batch) | Clip latency        | \~50 ms / clip (A100 GPU)       |
| Summarization + QA       | Latency per request | \~350 ms (Groq)                 |
| End-to-end `/infer`      | Total latency       | \~1.3 sec (per 60-sec video)    |
| Accuracy                 | Action+Cations      | \~82%; QA prompt accuracy \~88% |

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

- Fork this repo
- Create a new branch (feature/awesome-feature)
- Commit your changes
- Push to your branch
- Create a Pull Request

## License

Distributed under the MIT License. See `LICENSE` file for more information.

## Team Members

Pranshu Jain - [SurveilSense Project](https://github.com/Pranshu321/SurveilSense) - pranshujain0111@gmail.com
Riya Dubey - riyadubey.seya25@gmail.com
Aviansh Kuamr - avinash@gmail.com
