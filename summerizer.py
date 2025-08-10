# summerizer.py
import os
import time
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import dotenv
dotenv.load_dotenv()

def summarize_events(events, retries=5, delay=5):
    """
    Summarize extracted video events using Groq API with retry logic.
    """
    template = """
    You are a video summarization assistant.
    Below is a list of detected events from a video, each with an action label and caption.
    Summarize the overall content of the video in a concise paragraph.

    Events:
    {events}

    Summary:
    """
    prompt = PromptTemplate(input_variables=["events"], template=template)

    event_text = "\n".join(
        [f"{e['label']} ({e['score']:.2f}): {e['caption']}" for e in events]
    )

    llm = ChatOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model="llama3-70b-8192",  # You can change to llama3-70b if you have access
        temperature=0.3,
        max_tokens=512
    )

    chain = prompt | llm

    for attempt in range(1, retries + 1):
        try:
            print(f"🔄 Attempt {attempt} to summarize with Groq...")
            result = chain.invoke({"events": event_text})
            return result.content
        except Exception as e:
            print(f"⚠️ Groq API call failed: {e}")
            if attempt < retries:
                print(f"⏳ Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise RuntimeError("❌ Failed to get summary from Groq after multiple retries.")
