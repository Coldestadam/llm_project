import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from rag import get_embedding_model, get_history_aware_retriever, get_llm_model, get_rag_chain_v0, get_rag_chain_v1, get_chromadb_client, stream_wrapper

import nltk
import os

# Set the NLTK data directory to a writable path
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

# Download NLTK data during the build process
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', download_dir="nltk_data")
        nltk.download('averaged_perceptron_tagger_eng', download_dir="nltk_data")
    except Exception as e:
        st.error(f"Failed to download NLTK data: {e}")

download_nltk_data()


# Setting page config
st.set_page_config(page_title="Adam's Assistant",
                   page_icon=":robot_face:"
)

# Getting all necessary objects for langchain app
embedding_model = get_embedding_model()
llm = get_llm_model()
chromadb_client = get_chromadb_client(embedding_model)

history_aware_retriever = get_history_aware_retriever(chromadb_client, embedding_model, llm)

# Function to delete the chat history in the session state when changing the RAG chain version
def reset_chat_history():
    if "chat_history" in st.session_state:
        del st.session_state["chat_history"]


st.title("Adam's Assistant")
st.image("info/headshot.jpeg", caption="Top tower of La Sagrada Famiília in Barcelona")

# Displaying a welcome message for the user
welcome_message = \
"""
Hi there! This app is built with [Langchain](https://www.langchain.com/) using the latest OpenAI products. It is meant to answer any personal or professional questions about myself!
It was a blast to learn these technologies and it has helped me grow technically by learning the latest engineering applications of LLMs. As I have more free time, I will be expanding the
knowledge base for this app to include more information about my work experience, travel experience, and personal hobbies. I hope you have fun with it:smile:

[Link for code](https://github.com/Coldestadam/llm_project)
"""
st.write(welcome_message)

# Add a streamlit drop down to select the RAG chain version
rag_chain_version = st.selectbox("Select RAG Chain Version", ["v0", "v1"], on_change=reset_chat_history)

# If the user selects v0, we will use the v0 RAG chain, otherwise we will use the v1 RAG chain
if rag_chain_version == "v0":
    rag_chain = get_rag_chain_v0(llm, history_aware_retriever)
   
else:
    rag_chain = get_rag_chain_v1(llm, history_aware_retriever)

# Storing messages in the session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
    # Writing the AI welcome message for the user in the app
    ai_intro = "Hello, I am Adam's personal assistant and will answer questions about his personal and professional life. Ask me a question!"
    st.session_state["chat_history"].append(AIMessage(content=ai_intro))

# Everytime the app reruns the script, we need to display chat_history
for message in st.session_state["chat_history"]:
    role = "human" if type(message) == HumanMessage else "ai"
    st.chat_message(name=role).write(message.content)


user_prompt = st.chat_input("Type here...")

if user_prompt:
    user_message = st.chat_message(name="human").write(user_prompt)

    with st.chat_message(name="ai"):
        ai_response_stream = rag_chain.stream({"input":user_prompt, "chat_history":st.session_state["chat_history"]})
        ai_response = st.write_stream(stream_wrapper(ai_response_stream))

    # # Adding the user and ai messages as a Lanchain Messages in chat_history
    st.session_state["chat_history"].extend(
        [
            HumanMessage(content=user_prompt),
            AIMessage(content=ai_response)
        ]
    )