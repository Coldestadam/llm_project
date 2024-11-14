import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from rag import get_embedding_model, get_history_aware_retriever, get_llm_model, get_rag_chain, get_chromadb_client

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

# Getting all necessary objects for langchain app
embedding_model = get_embedding_model()
llm = get_llm_model()
chromadb_client = get_chromadb_client(embedding_model)

history_aware_retriever = get_history_aware_retriever(chromadb_client, embedding_model, llm)
rag_chain = get_rag_chain(llm, history_aware_retriever)

#st.write("Number of documents in DB:", chromadb_client.get_collection("adam_rag_txt").count())
# vector_store._collection.count()
# st.write("Number of documents in DB:", chromadb_client.get_collection("adam_rag_txt").count())

st.title("Adam's Assistant")
st.caption("THIS IS IN BETA")


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

    # Calling rag_chain on the user_promt
    ai_response = rag_chain.invoke({"input":user_prompt, "chat_history":st.session_state["chat_history"]})

    # Displaying AI's response to the app
    ai_message = st.chat_message(name="ai")
    ai_message.write(ai_response["answer"])
    #ai_message.write(ai_response['context'])

    # Adding the user and ai messages as a Lanchain Messages in chat_history
    #st.session_state["chat_history"].extending(HumanMessage(content=user_prompt))
    st.session_state["chat_history"].extend(
        [
            HumanMessage(content=user_prompt),
            AIMessage(content=ai_response["answer"])
        ]
    )
    # Simulating ai_response
    # ai_response = "This is just a simulated response"

    # ai_message = st.chat_message(name="ai").write(ai_response)

    # st.session_state["chat_history"].append(AIMessage(content=ai_response))