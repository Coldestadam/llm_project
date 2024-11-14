import os
import streamlit as st
import chromadb
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Initializing Langchain API

os.environ['LANGCHAIN_TRACING_V2'] = st.secrets['LANGCHAIN_TRACING_V2']
os.environ['LANGCHAIN_ENDPOINT'] = st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_API_KEY'] = st.secrets['LANGCHAIN_API_KEY']

# # Initializing OpenAI API
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

@st.cache_resource
def get_embedding_model():
    return OpenAIEmbeddings(model="text-embedding-3-small")

@st.cache_resource
def get_llm_model():
    return ChatOpenAI(model="gpt-4o-mini")
@st.cache_resource
def get_chromadb_client(_embedding_model):
    # Initializing chroma database client
    persistent_client = chromadb.PersistentClient()
    
    # Loading documents
    loader = TextLoader(file_path="info/Personal RAG Details.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                chunk_overlap=200,
                                                add_start_index=True)

    # Split documents to store in Vector Store
    splits = text_splitter.split_documents(documents)

    # if "adam_rag_txt" collection already exists, delete in to avoid duplicate embeddings
    try:
        if persistent_client.get_collection("adam_rag_txt"):
            persistent_client.delete_collection("adam_rag_txt")

        collection_name = "adam_rag_txt"
    except:
        collection_name = "adam_rag_txt"

    # Using langchain Chroma wrapper to store embeddings of the documents in the database
    vector_store = Chroma(client=persistent_client, collection_name=collection_name, embedding_function=_embedding_model)

    # Storing embeddings of the splits in vector store
    vector_store.add_documents(documents=splits)

    return persistent_client

# @st.cache_resource
# def get_vector_store(_embedding_model):
#     # Loading documents
#     loader = TextLoader(file_path="info/Personal RAG Details.txt")
#     documents = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
#                                                 chunk_overlap=200,
#                                                 add_start_index=True)

#     # Split documents to store in Vector Store
#     splits = text_splitter.split_documents(documents)

#     # Creating Chroma Vector Store DB to store using OpenAI's embedding model
#     # vector_store = Chroma(collection_name="adam_rag_txt",
#     #                       embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
#     vector_store = Chroma(collection_name="adam_rag_txt",
#                           embedding_function=_embedding_model)

#     # Storing embeddings of the splits in vector store
#     vector_store.add_documents(documents=splits)

#     return vector_store


#@st.cache_resource
def get_history_aware_retriever(chromadb_client, embedding_model, llm):

    # Loading vector store from chromadb client
    vector_store = Chroma(client=chromadb_client,
                          collection_name="adam_rag_txt",
                          embedding_function=embedding_model)
    
    # Creating a retriever object from vector store
    retriever = vector_store.as_retriever(search_type = "similarity",
                                           search_kwargs={"k":5}) # Getting top 5 stored documents in vector store

    # Creating a history aware retriever runnable object
    rephrase_prompt_template = hub.pull("langchain-ai/chat-langchain-rephrase").template

    rephrase_prompt = ChatPromptTemplate(
        [
            ("system", rephrase_prompt_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # Creating chain that returns context based on history
    history_aware_retriever = create_history_aware_retriever(llm=llm,
                                                            retriever=retriever,
                                                            prompt=rephrase_prompt)
    
    return history_aware_retriever

#@st.cache_resource
def get_rag_chain(_llm, _history_aware_retriever):
    # system_prompt = """You are Adam's personal assistant, and users are interested in learning more about him. It is your responsibility to
    #     answer questions about Adam's professional or personal life based on the given context below. If the question has nothing to do with Adam but is relevant to 
    #     the chat history, please expand on your knowledge. Otherwise, if the question has nothing to do with Adam and the chat history tell the user that you 
    #     cannot answer the question since you are restricted to answering questions about Adam only. If the question is about Adam but the context is not helpful or unrelated to the question, 
    #     tell the user that you do not have the necessary information to answer the question but insinuate that Adam is willing to answer that question directly. 
    #     If the question is inappropriate about Adam or something else, do not answer the question and tell the user that the question is inappropriate for this application.


    #     {context}
    #     """

    system_prompt = """You are Adam's personal assistant, and users are interested in learning more about him. It is your responsibility to
        answer questions about Adam's professional or personal life based on the given context below. If the question has nothing to do with Adam but is relevant to 
        the chat history, please answer the question with your knowledge and ignore the context. Otherwise, if the question has nothing to do with Adam and the chat history tell the user that you 
        cannot answer the question since you are restricted to answering questions about Adam only. If the question is about Adam but the context is not helpful or unrelated to the question, 
        tell the user that you do not have the necessary information to answer the question but insinuate that Adam is willing to answer that question directly. 
        If the question is inappropriate about Adam or something else, do not answer the question and tell the user that the question is inappropriate for this application.


        {context}
        """

    # Input into qa_prompt is context, chat_history, and input
    qa_prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm=_llm, prompt=qa_prompt)
    rag_chain = create_retrieval_chain(_history_aware_retriever, question_answer_chain)
    return rag_chain