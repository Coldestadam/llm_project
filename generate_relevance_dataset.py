from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
import glob
import streamlit as st

from langsmith import Client

# Initializing Langchain API
os.environ['LANGCHAIN_TRACING_V2'] = st.secrets['LANGCHAIN_TRACING_V2']
os.environ['LANGCHAIN_ENDPOINT'] = st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_API_KEY'] = st.secrets['LANGCHAIN_API_KEY']
os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']

# # Initializing OpenAI API
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Function to convert roles in chat history to lowercase that Langsmith expects
def convert_roles_to_lowercase(chat_history):
    """Convert all roles in the chat history to lowercase."""
    return [
        {"role": entry["role"].lower(), "content": entry["content"]}
        for entry in chat_history
    ]

# Initializing LLM Client for generating questions and chat history
question_gen_llm = ChatOpenAI(model="gpt-4.1")

# Intializing Langsmith Client to run prompts for dataset generation
client = Client()

# Get the Langsmith dataset
dataset = client.read_dataset(dataset_id="8f7618bc-bfc3-4407-953e-183cea4d3267")

# Pulling the question generator prompt from Langsmith and creating a chain with GPT-4.1
question_generator_prompt = client.pull_prompt("question-gen-from-doc")
question_generator_chain = question_generator_prompt | question_gen_llm


# Pulling the relevance chat history prompt from Langsmith and creating a chain with GPT-4.1
revelant_chat_history_prompt = client.pull_prompt("rel_chat_history_gen_from_question_and_doc")
relevant_chat_history_chain = revelant_chat_history_prompt | question_gen_llm

# Pulling the irrelevant chat history prompt from Langsmith and creating a chain with GPT-4.1
irrelevant_chat_history_prompt = client.pull_prompt("irrel_chat_history_gen_from_question_and_doc")
irrelevant_chat_history_chain = irrelevant_chat_history_prompt | question_gen_llm

# Loading Machine Learning Engineer Resume
documents = []
pdf_loader = PyPDFLoader("info/AV_MLE_Resume.pdf")
pdf_docs = pdf_loader.load()
documents.extend(pdf_docs)

#Loading all text files in directory info/
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

for document in documents:
    print(f"Processing document: {document.metadata['source']}")

    # Creating a list to store the examples before pushing to Langsmith dataset
    examples = []

    # Generating questions from the document using the question generator prompt
    generated_questions = question_generator_chain.invoke({"document": document.page_content})
    
    # Get the questions related to adam
    adam_questions = generated_questions['adam_generated_questions']

    # Get the other questions
    other_questions = generated_questions['other_generated_questions']

    # Generating examples for each question related to Adam
    for question in adam_questions:

        try:
            # Generating relevant chat history for the question and document
            relevant_chat_history = convert_roles_to_lowercase(relevant_chat_history_chain.invoke({"question": question, "document": document})['relevant_chat_history'])

            
            # Create an example dictionary for the Langsmith dataset for the relevant chat history
            rel_example = {
                "inputs":{
                    "last_question": question,
                    "previous_chat_history": relevant_chat_history[:-1],  # Exclude the last entry which is the question itself
                },
                "outputs":{
                    "valid_question_to_answer": True
                },
                "metadata": {
                    "is_chat_history_relevant": True,
                    "is_question_related_to_adam": True,
                    "document_source": document.metadata['source']
                }
            }
            # Append the example to the list
            examples.append(rel_example)
        except:
            print(f"Error extracting relevant chat history json for question: {question}")

        try:
            # Generating irrelevant chat history for the question and document
            irrelevant_chat_history = convert_roles_to_lowercase(irrelevant_chat_history_chain.invoke({"question": question, "document": document})["irrelevant_chat_history"])

            # Create an example dictionary for the Langsmith dataset for the irrelevant chat history
            irrel_example = {
                "inputs":{
                    "last_question": question,
                    "previous_chat_history": irrelevant_chat_history[:-1],  # Exclude the last entry which is the question itself
                },
                "outputs":{
                    "valid_question_to_answer": True
                },
                "metadata": {
                    "is_chat_history_relevant": False,
                    "is_question_related_to_adam": True,
                    "document_source": document.metadata['source']
                }
            }

            # Append the examples to the list
            examples.append(irrel_example)

        except:
            print(f"Error extracting irrelevant chat history json for question: {question}")

    # Generating examples for each question not related to Adam
    for question in other_questions:

        try:
            # Generating relevant chat history for the question and document
            relevant_chat_history = convert_roles_to_lowercase(relevant_chat_history_chain.invoke({"question": question, "document": document})['relevant_chat_history'])

            
            # Create an example dictionary for the Langsmith dataset for the relevant chat history
            rel_example = {
                "inputs":{
                    "last_question": question,
                    "previous_chat_history": relevant_chat_history[:-1],  # Exclude the last entry which is the question itself
                },
                "outputs":{
                    "valid_question_to_answer": True
                },
                "metadata": {
                    "is_chat_history_relevant": True,
                    "is_question_related_to_adam": False,
                    "document_source": document.metadata['source']
                }
            }
            # Append the example to the list
            examples.append(rel_example)
        except:
            print(f"Error extracting relevant chat history json for question: {question}")

        try:
            # Generating irrelevant chat history for the question and document
            irrelevant_chat_history = convert_roles_to_lowercase(irrelevant_chat_history_chain.invoke({"question": question, "document": document})["irrelevant_chat_history"])

            # Create an example dictionary for the Langsmith dataset for the irrelevant chat history

            irrel_example = {
                "inputs":{
                    "last_question": question,
                    "previous_chat_history": irrelevant_chat_history[:-1],  # Exclude the last entry which is the question itself
                },
                "outputs":{
                    "valid_question_to_answer": False
                },
                "metadata": {
                    "is_chat_history_relevant": False,
                    "is_question_related_to_adam": False,
                    "document_source": document.metadata['source']
                }
            }

            # Append the examples to the list
            examples.append(irrel_example)

        except:
            print(f"Error extracting irrelevant chat history json for question: {question}")

    client.create_examples(dataset_id=dataset.id, examples=examples)