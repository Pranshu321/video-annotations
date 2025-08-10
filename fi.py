# from model import Models
# from inference import process_chunk_optimized

# models = Models()
# print(models.caption_model_name)
# events = process_chunk_optimized(models, "./extracted_frames")
# print(events)
from model import Models
from inference import process_chunk_optimized
from summerizer import summarize_events
from qa_agent import create_qa_agent

# Step 1: Load models and process video frames
models = Models()
events = process_chunk_optimized(models, "./extracted_frames")
print("Detected events:", events)

# Step 2: Summarize video
summary = summarize_events(events)
print("\n--- Video Summary ---\n", summary)

# Step 3: Build QA Agent
qa = create_qa_agent(summary)

# Ask some questions
print("\n--- Chat with Video ---")
while True:
    q = input("You: ")
    if q.lower() in ["exit", "quit"]:
        break
    answer = qa({"question": q})
    print("Bot:", answer["answer"])
