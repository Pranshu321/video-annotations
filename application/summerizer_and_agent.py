import streamlit as st
import os
import json
import traceback
from voilation_detection import VoilationDetection

# --- Streamlit UI Setup First ---
st.set_page_config(page_title="Event Chat", page_icon="🧠")
st.title("🧠 Surveillance Event Chat Assistant")

# Debug info
st.write("App started successfully!")

try:
    # --- Import Libraries ---
    st.write("Loading libraries...")
    from langchain_openai import ChatOpenAI
    from langchain.memory import ConversationSummaryMemory, VectorStoreRetrieverMemory
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.schema import HumanMessage, AIMessage
    st.success("Libraries loaded successfully!")

    # --- Setup LLM ---
    st.write("Setting up LLM...")
    os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
    # Replace with your actual key
    os.environ["OPENAI_API_KEY"] = "your-groq-token"

    llm = ChatOpenAI(
        model="llama3-70b-8192",
        temperature=0.3
    )
    st.success("LLM setup complete!")

    # --- Load Embeddings & Memory ---
    @st.cache_resource
    def init_embeddings_and_memory():
        st.write("Initializing embeddings and memory...")
        try:
            embedding = HuggingFaceEmbeddings()
            vector_store = FAISS.from_texts(["example event"], embedding)

            event_memory = VectorStoreRetrieverMemory(
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                memory_key="events"
            )

            chat_memory = ConversationSummaryMemory(
                llm=llm,
                memory_key="chat_history",
                return_messages=True
            )

            return embedding, event_memory, chat_memory
        except Exception as e:
            st.error(f"Error initializing embeddings/memory: {e}")
            st.error(traceback.format_exc())
            return None, None, None

    embedding, event_memory, chat_memory = init_embeddings_and_memory()

    if embedding is None:
        st.error("Failed to initialize embeddings and memory. Cannot continue.")
        st.stop()

    st.success("Embeddings and memory initialized!")

    # --- Helper Functions ---
    def summarize_events(events):
        if not events:
            return type('Response', (), {'content': "No events detected."})()

        flat = "\n".join(
            f"{e.get('timestamp', 'Unknown time')}: {e.get('event_type', 'Unknown event')} at {e.get('location', 'Unknown location')} (confidence: {e.get('confidence', 'N/A')})"
            for e in events
        )
        prompt = f"Summarize the following surveillance events clearly and concisely:\n\n{flat}"

        try:
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            st.error(f"Error summarizing events: {e}")
            return type('Response', (), {'content': f"Error summarizing events: {e}"})()

    def handle_chat(user_input, summary_text):
        try:
            # Save event context
            event_memory.save_context(
                {"input": "event_summary"},
                {"output": summary_text}
            )

            # Add to chat memory
            chat_memory.chat_memory.add_user_message(user_input)

            # Retrieve relevant events
            retrieved = event_memory.load_memory_variables(
                {"input": user_input})
            chat_history = chat_memory.load_memory_variables(
                {"input": user_input}
            ).get("chat_history", [])

            # Format chat history
            formatted = "\n".join([
                f"User: {msg.content}" if isinstance(msg, HumanMessage)
                else f"Assistant: {msg.content}"
                for msg in chat_history
            ])

            prompt = f"""
You are an intelligent assistant helping summarize and analyze security surveillance data.

Context:
{retrieved.get("events", "")}

Conversation History:
{formatted}

User: {user_input}
Assistant:"""

            response = llm.invoke(prompt)
            chat_memory.chat_memory.add_ai_message(response.content)
            return response

        except Exception as e:
            st.error(f"Error handling chat: {e}")
            return type('Response', (), {'content': f"Sorry, I encountered an error: {e}"})()

    # --- File Upload Section ---
    st.header("📁 Load Event Data")

    # Initialize events variable
    events = []

    # Upload video file for processing
    video_file = st.file_uploader(
        "Upload surveillance video", type=["mp4", "avi", "mov", "mkv"])

    # Upload JSON file as alternative
    uploaded_file = st.file_uploader(
        "Or upload surveillance events JSON", type=["json"])

    # Process video file if uploaded
    if video_file:
        st.write("Processing uploaded video...")
        try:
            # Save uploaded video temporarily
            with open(f"temp_{video_file.name}", "wb") as f:
                f.write(video_file.getbuffer())

            # Process video through violation detection
            with st.spinner("Analyzing video for violations..."):
                event_json = VoilationDetection(f"temp_{video_file.name}")
                if event_json and os.path.exists(event_json):
                    with open(event_json, "r") as f:
                        events = json.load(f)
                    st.success(f"Detected {len(events)} events from video")
                else:
                    st.warning("No violations detected in the video")
                    events = []

            # Clean up temporary file
            if os.path.exists(f"temp_{video_file.name}"):
                os.remove(f"temp_{video_file.name}")

        except Exception as e:
            st.error(f"Error processing video: {e}")
            st.error(traceback.format_exc())
            events = []

    # Load events from uploaded JSON file
    elif uploaded_file:
        st.write("Processing uploaded JSON file...")
        try:
            events = json.load(uploaded_file)
            if not isinstance(events, list):
                st.error("JSON file should contain a list of events")
                st.stop()
            st.success(f"Loaded {len(events)} events from uploaded file")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.write("No file uploaded. Checking for default file...")
        # Try loading default file
        if os.path.exists("detected_violations.json"):
            try:
                with open("detected_violations.json", "r") as f:
                    events = json.load(f)
                    if not isinstance(events, list):
                        st.error("JSON file should contain a list of events")
                        st.stop()
                st.success(f"Loaded {len(events)} events from default file")
            except Exception as e:
                st.error(f"Error reading default file: {e}")
                st.write("Continuing without events for demonstration...")
                events = []
        else:
            st.info("No default file found. You can either:")
            st.write("1. Upload a video file for processing, OR")
            st.write("2. Upload a JSON file with events, OR")
            st.write(
                "3. Place a 'detected_violations.json' file in your working directory, OR")
            st.write("4. Use sample data for demonstration")

            # Provide sample data option
            if st.button("Use Sample Data"):
                events = [
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
                st.rerun()

    # --- Main Application ---
    if events:
        st.header("📊 Event Analysis")

        # Display raw events in expandable section
        with st.expander("📋 View Raw Events"):
            st.json(events)

        # Summarize once and cache
        @st.cache_data
        def get_event_summary(events_data):
            # Convert to string to make it hashable for caching
            events_str = json.dumps(events_data, sort_keys=True)
            return summarize_events(events_data)

        with st.spinner("Analyzing events..."):
            summary = get_event_summary(events)

        # Display summary
        with st.expander("📄 View Event Summary", expanded=True):
            st.markdown(summary.content)

        # --- Chat Interface ---
        st.header("💬 Chat Interface")

        # Initialize session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Handle user input
        user_input = st.chat_input("Ask about the events...")
        if user_input:
            # Show user message
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input})

            # Get assistant response
            with st.spinner("Thinking..."):
                reply = handle_chat(user_input, summary.content)

            with st.chat_message("assistant"):
                st.markdown(reply.content)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": reply.content})

        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    else:
        st.info(
            "No events loaded. Please upload a video file, JSON file, use the default file, or try sample data.")

except Exception as e:
    st.error("Critical error occurred:")
    st.error(str(e))
    st.error("Full traceback:")
    st.code(traceback.format_exc())
    st.write("Please check your environment and dependencies.")
