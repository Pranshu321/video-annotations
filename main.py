# Om
# swatik
from utils.model import Models
from utils.inference import process_chunk_multiprocess
from utils.summerizer import summarize_events
from utils.qa_agent import create_qa_agent
from multiprocess_video import multi_process_frame_extraction
import os
import tempfile
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import PlainTextResponse
import uvicorn

# Initialize FastAPI
app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})

# Load models once at startup
print("🚀 Loading models...")
models = Models()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Video Processing API"}

@app.post("/infer", response_class=PlainTextResponse)
async def infer(video: UploadFile = File(...), prompt: str = Form(...)):
    """
    Benchmark API:
    1. Receive video + prompt
    2. Process video into frames (multi-process)
    3. Extract text events (process_chunk_optimized)
    4. Summarize events
    5. Run QA agent with prompt
    6. Return plain text answer
    """
    try:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(await video.read())
            video_path = tmp_video.name

        # Step 1: Extract frames in parallel
        print("📽 Extracting frames...")
        video_chunks = multi_process_frame_extraction(video_path)

        # Step 2: Process chunks → text events
        print("📝 Processing chunks...")
        # events = []
        events = process_chunk_multiprocess(
        chunk_dir="./extracted_frames", models=models,
        frames_per_clip=16,
        stride=8,
        batch_size=4,
        max_workers=4,
        use_threading=True
        )

        # events = process_chunk_optimized(chunk_dir="./extracted_frames", models=models)

        if not events:
            return PlainTextResponse("No events detected in the video.", status_code=200)

        # Step 3: Summarize events
        print("📄 Summarizing events...")
        summary = summarize_events(events)

        print("Summary=" , summary)

        # Step 4: Create QA agent and answer prompt
        print("🤖 Running QA agent...")
        qa_agent = create_qa_agent(summary)
        print("Prompt=" , prompt)
        answer = qa_agent({"question": prompt})
        print("Answer=" , answer)

        return PlainTextResponse(answer["answer"], status_code=200)

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return PlainTextResponse(f"Error: {str(e)}", status_code=500)

    finally:
        # Clean up temp file
        try:
            os.remove(video_path) # type: ignore
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)