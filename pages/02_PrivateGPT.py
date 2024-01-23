import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.embeddings import OpenAIEmbeddings,OllamaEmbeddings,CacheBackedEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from operator import itemgetter


st.set_page_config(page_title="Private GPT", page_icon="🔒")


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, model_name):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{model_name}/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    
    if model_name == "GPT-4":
        embeddings = OpenAIEmbeddings()
    elif model_name == "Phi2":
        embeddings = OllamaEmbeddings(model="phi")

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    return st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def save_memory(input, output):
    st.session_state["memory"].save_context({"input": input}, {"output": output})


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer, just say you don't know. DON'T make anything up. 

            Context
            {context}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

st.title("Private GPT")

st.markdown(
    """
Welcome!

Use this chatbot to ask question to an AI about your file

Upload your file on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"]
    )

    model_name = st.selectbox("Select Model", ["GPT-4", "Phi2"])

    if model_name == "GPT-4":
        llm = ChatOpenAI(
            temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()]
        )
    elif model_name == "Phi2":
        llm = ChatOllama(
            model="phi",
            temperature=0.1,
            streaming=True,
            callbacks=[ChatCallbackHandler()],
        )
    st.session_state["model_name"] = model_name


if file:
    retriever = embed_file(file, model_name)
    send_message("I'm ready! Ask anyway!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                # Add memory
                "chat_history": RunnableLambda(
                    st.session_state["memory"].load_memory_variables
                )
                | itemgetter("chat_history"),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            if model_name == "Phi2":
                result = chain.invoke("Insturct: " + message + "\nOutput:")
            else:
                result = chain.invoke(message)
            save_memory(message, result.content)

else:
    st.session_state["messages"] = []
    # Add memory
    st.session_state["memory"] = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,
    )
