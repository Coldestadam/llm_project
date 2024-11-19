import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from rag import get_embedding_model, get_history_aware_retriever, get_llm_model, get_rag_chain, get_chromadb_client, stream_wrapper

# Setting page config
st.set_page_config(page_title="Adam's Assistant",
                   page_icon=":robot_face:"
)

# Getting all necessary objects for langchain app
embedding_model = get_embedding_model()
llm = get_llm_model()
chromadb_client = get_chromadb_client(embedding_model)

history_aware_retriever = get_history_aware_retriever(chromadb_client, embedding_model, llm)
rag_chain = get_rag_chain(llm, history_aware_retriever)

st.title("Adam's Assistant")
st.caption("THIS IS IN BETA")
st.image("info/headshot.jpeg", caption="Top tower of La Sagrada Famiília in Barcelona")

welcome_message = \
"""
Hi there! This app is built with [Langchain](https://www.langchain.com/) using the latest OpenAI products. It is meant to answer any personal or professional questions about myself!
It was a blast to learn these technologies and it has helped me grow technically by learning the latest engineering applications of LLMs. As I have more free time, I will be expanding the
knowledge base for this app to include more information about my work experience, travel experience, and personal hobbies. I hope you have fun with it:smile:

[Link for code](https://github.com/Coldestadam/llm_project)
"""
st.write(welcome_message)


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