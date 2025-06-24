import os
import glob
import streamlit as st
import chromadb
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader

# Initializing Langchain API
os.environ['LANGCHAIN_TRACING_V2'] = st.secrets['LANGCHAIN_TRACING_V2']
os.environ['LANGCHAIN_ENDPOINT'] = st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_API_KEY'] = st.secrets['LANGCHAIN_API_KEY']
os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']

# # Initializing OpenAI API
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

@st.cache_resource
def get_embedding_model():
    return OpenAIEmbeddings(model="text-embedding-3-small")

@st.cache_resource
def get_llm_model():
    return ChatOpenAI(model="gpt-4.1-mini")
@st.cache_resource
def get_chromadb_client(_embedding_model):
    """
    Initializes a persistent ChromaDB client and loads various documents into a vector store.
    These are the files that are loaded:
        1. All text files in the `info/` directory
            1a. Personal_RAG_Details.txt: Contains personal details about Adam.
            1b. RAG_Career.txt: Contains career details about Adam.
            1c. RAG_Fun_and Hobbies.txt: Contains fun and hobbies details about Adam.
        2. `.info/Academic Transcript.html`: Academic Transcript from a local HTML file.
        3. `.info/README.md`: The README file of this repository, which contains information about this project.
        4. `info/AV_MLE_Resume.pdf`: Adam's Machine Learning Engineer Resume in PDF format.
    Args:
        _embedding_model: The embedding model used to encode the documents for storage in the vector store
    Returns:
        A persistent ChromaDB client connected to the vector store containing the loaded documents.
    """
    # Initializing chroma database client
    persistent_client = chromadb.PersistentClient()
    
    # Loading documents from txt, academic transcript, resume (pdf) files, and README files
    documents = []

    # Loading all text files in directory info/
    txt_files = glob.glob("./info/*.txt")

    for txt_file in txt_files:
        loader = TextLoader(file_path=txt_file)
        documents.extend(loader.load())

    # Loading Academic Transcript from local html file
    transcript_path = "info/Academic Transcript.html"
    loader = UnstructuredHTMLLoader(transcript_path)
    documents.extend(loader.load())

    # Loading readme of the repo
    readme_loader = UnstructuredMarkdownLoader(file_path="README.md")
    readme_docs = readme_loader.load()
    documents.extend(readme_docs)

    # Loading Machine Learning Engineer Resume
    pdf_loader = PyPDFLoader("info/AV_MLE_Resume.pdf")
    pdf_docs = pdf_loader.load()
    documents.extend(pdf_docs)

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

def get_history_aware_retriever(chromadb_client, embedding_model, llm):
    """
    Creates a history-aware retriever for conversational retrieval-augmented generation (RAG).

    This function loads a vector store from a ChromaDB client, sets up a retriever to fetch the top 5 most similar documents,
    and wraps it with a prompt that rephrases user queries based on chat history. The result is a retriever that can 
    consider previous conversation turns when retrieving relevant context for the LLM.

    Args:
        chromadb_client: The ChromaDB client instance connected to the persistent vector store.
        embedding_model: The embedding model used to encode queries and documents.
        llm: The language model used for rephrasing and downstream tasks.

    Returns:
        A history-aware retriever object that can be used in a RAG pipeline.
    """

    # Load the vector store containing document embeddings
    vector_store = Chroma(client=chromadb_client,
                          collection_name="adam_rag_txt",
                          embedding_function=embedding_model)
    
    # Create a retriever to fetch top 5 similar documents from the vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Load a prompt template for rephrasing user queries based on chat history
    rephrase_prompt_template = hub.pull("langchain-ai/chat-langchain-rephrase").template

    # Build the chat prompt with system instructions, chat history, and user input
    rephrase_prompt = ChatPromptTemplate(
        [
            ("system", rephrase_prompt_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # Create a history-aware retriever chain that uses the LLM to rephrase queries
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=rephrase_prompt
    )
    
    return history_aware_retriever

def get_rag_chain_v0(_llm, _history_aware_retriever):
    """
    Creates a Retrieval-Augmented Generation (RAG) chain for answering questions about Adam.
    This function sets up a system prompt that guides the assistant's behavior, defines how to handle chat history,
    and configures the question-answering chain using the provided language model and history-aware retriever.

    This is versioned 0 (v0) of the RAG Chain, which is the first implemntation of this app.
    Args:
        _llm: The language model used for generating responses.
        _history_aware_retriever: The history-aware retriever that fetches relevant documents based on chat history.

    Returns:
        A RAG chain that can answer questions about Adam based on the provided context and chat history.
    """

    # Define the system prompt that guides the assistant's behavior, and uses {context} to provide context to the chain created with function create_stuff_documents_chain
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

    # Create a question-answering chain using the LLM, the defined prompt above, and the docs retrieved by the history-aware retriever for {context}
    question_answer_chain = create_stuff_documents_chain(llm=_llm, prompt=qa_prompt)

    # Create a RAG chain that combines the history-aware retriever and the question-answering chain that 
    rag_chain = create_retrieval_chain(_history_aware_retriever, question_answer_chain)
    return rag_chain

def get_rag_chain_v1(_llm, _history_aware_retriever):
    """
    Creates a Retrieval-Augmented Generation (RAG) chain for answering questions about Adam.
    This function sets up a system prompt that guides the assistant's behavior, defines how to handle chat history,
    and configures the question-answering chain using the provided language model and history-aware retriever.

    This is versioned 1 (v1) of the RAG Chain, which is the second implemntation of this app with a better system prompt.
    Args:
        _llm: The language model used for generating responses.
        _history_aware_retriever: The history-aware retriever that fetches relevant documents based on chat history.

    Returns:
        A RAG chain that can answer questions about Adam based on the provided context and chat history.
    """

    # Define the system prompt that guides the assistant's behavior, and uses {context} to provide context to the chain created with function create_stuff_documents_chain
    
    system_prompt = """You are Adam's personal assistant, and users are interested in learning more about him. It is your responsibility to
        answer questions about Adam's professional or personal life based on the given context and chat history.
        
        Please follow these guidelines:
        1. If the question's subject is Adam and the question is relevant to the chat history, provide a direct answer using the context
            1a. If the context is not helpful or unrelated, inform the user that you do not have the necessary information, but Adam is willing to answer directly.
            ex.
                Human: Where is Adam from?
                AI: Adam is originally from Fresno County, specifically from a small town called Reedley, California.
                Human: Did Adam go to High School in Reedley? (Context is not helpful from documents)
                AI: Yes, Adam attended Reedley High School

        2. If the question's subject is Adam but the question is not relevant to the chat history, provide a direct answer using the context
            2a. If the context is not helpful or unrelated, inform the user that you do not have the necessary information, but Adam is willing to answer directly.
            ex.
                Human: Where is Adam from?
                AI: Adam is originally from Fresno County, specifically from a small town called Reedley, California.
                Human: What is Adam's favorite color? (Context is not helpful from documents)
                AI: I do not have the necessary information to answer that question, but Adam may be willing to share his favorite color directly if you ask him!

        3. If the question's subject is not Adam and the question is relevant to the chat history, answer the question based on your knowledge and ignore the context.
            3a. If the context is not helpful or unrelated, answer the question based on your knowledge and ignore the context.
            ex.
                Human: Where is Adam from?
                AI: Adam is originally from Fresno County, specifically from a small town called Reedley, California.
                Human: What is the nickname of Reedley, California? (Context is not helpful from documents)
                AI: The nickname of Reedley, California is "The World's Fruit Basket" due to its rich agricultural heritage and production of various fruits.

        4. If the question's subject is not Adam and the question is not relevant chat history, inform the user that you cannot answer the question since you are restricted to answering questions about Adam only.

        5. If the question is inappropriate about Adam or something else, do not answer the question and tell the user that the question is inappropriate for this application.
        
        This is the context:
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

    # Create a question-answering chain using the LLM, the defined prompt above, and the docs retrieved by the history-aware retriever for {context}
    question_answer_chain = create_stuff_documents_chain(llm=_llm, prompt=qa_prompt)

    # Create a RAG chain that combines the history-aware retriever and the question-answering chain that 
    rag_chain = create_retrieval_chain(_history_aware_retriever, question_answer_chain)
    return rag_chain




def stream_wrapper(rag_chain_stream):
    for chunk in rag_chain_stream:
        if "answer" in chunk:
            yield chunk["answer"]