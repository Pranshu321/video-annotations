# qa_agent.py
import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def create_qa_agent(summary_text):
    # Split summary into chunks for retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([summary_text])

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()

    # Groq LLM
    llm = ChatOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model="llama3-70b-8192",
        temperature=0
    )

    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # qa_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=retriever,
    #     memory=memory
    # )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        verbose=False
    )

    return qa_chain
