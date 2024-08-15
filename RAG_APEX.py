# V_1

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores.faiss import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage  # Import these classes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Streamlit app with custom title and layout
st.set_page_config(
    page_title="RAG APEX",
    page_icon="💎",
    layout="wide",
)

# Title and description
st.markdown(
    """
    <style>
    .container {
        background-image: url("https://cdn.pixabay.com/animation/2023/06/26/03/02/03-02-03-917_512.gif");
        background-size: cover;
        margin: 0;
        padding: 50px;
        border-radius: 5px;
        border: 1px solid #ddd;
        position: relative;
        overflow: hidden;
    }

    .container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 0;
        height: 100%;
        background-color: #f90;
        transition: width 0.5s ease;
        z-index: 0;
    }

    .container:hover::before {
        width: 100%;
    }

    .container h4,
    .container p {
        position: relative;
        z-index: 1;
        color: #fff;
        transition: color 0.5s ease;
    }

    .container:hover h4,
    .container:hover p {
        color: #000;
    }
    </style>
    
    <div class="container">
        <h4>❄️ RAG APEX </h4>
        <p>Upload PDFs and chat with their content using state-of-the-art language models.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for settings and API key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Session ID input
    session_id = st.sidebar.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Function to get or create chat history for a session
    def get_session_history(session: str) -> ChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    # PDF uploader in the settings panel
    uploaded_files = st.sidebar.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Input for user questions
        user_prompt = st.chat_input('Ask APEX...')
        if user_prompt:
            st.chat_message('user').markdown(user_prompt)
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_prompt},
                config={
                    "configurable": {"session_id": session_id}
                },
            )
            st.chat_message('assistant').markdown(response['answer'])

            # Display chat history in the right-hand sidebar
            with st.sidebar.expander("Chat History", expanded=False):
                for message in session_history.messages:
                    if isinstance(message, HumanMessage):
                        st.write(f"**You:** {message.content}")
                    elif isinstance(message, AIMessage):
                        st.write(f"**Assistant:** {message.content}")

else:
    st.warning("Please enter the Groq API Key")
