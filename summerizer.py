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
    PROMPT_TEMPLATE = """
    You are a detailed video summarization assistant.
    You are given a list of detected events from a video, each containing action labels, captions, and any available metadata.
    Write a concise but information-rich paragraph summarizing the entire video.
    Your summary should integrate all key details from every frame or event, including:

    Objects and entities: type of vehicles, people, animals, objects.
    Colors: describe the main visible colors of vehicles, clothing, or objects.
    Directions and movement: direction of travel (e.g., northbound, left-to-right), speed, acceleration.
    Actions and interactions: what each object is doing (e.g., overtaking, turning, braking, stopping, interacting).
    Scene context: location type (e.g., highway, city street, parking lot), weather, lighting, background elements.
    Special events: violations, unusual behavior, notable incidents.
    Temporal flow: describe the sequence of events in the order they occur.

    Keep the summary concise (5–7 sentences) but ensure no important visual or contextual detail is omitted.

    Events:
    {events}

    Summary:
    """
    prompt = PromptTemplate(input_variables=["events"], template=PROMPT_TEMPLATE)

    event_text = "\n".join(
        [f"{e['label']} ({e['score']:.2f}): {e['caption']}" for e in events]
    )

    llm = ChatOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model="llama3-70b-8192",  # You can change to llama3-70b if you have access
        temperature=0.3,
        max_tokens=512 # type: ignore
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
