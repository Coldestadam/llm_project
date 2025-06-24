import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client, traceable
from typing_extensions import Annotated, TypedDict
from rag import get_rag_chain_v0, get_history_aware_retriever, get_chromadb_client, get_rag_chain_v1

# Initializing Langchain API
os.environ['LANGCHAIN_TRACING_V2'] = st.secrets['LANGCHAIN_TRACING_V2']
os.environ['LANGCHAIN_ENDPOINT'] = st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_API_KEY'] = st.secrets['LANGCHAIN_API_KEY']
os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']

# # Initializing OpenAI API
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Intializing Langsmith Client to run prompts for dataset generation
client = Client()

# Initializing LLM Clients for RAG chains versions
v0_llm = ChatOpenAI(model="gpt-4o-mini")
v1_llm = ChatOpenAI(model="gpt-4.1-mini")


# Intializing Embedding Model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Intializing ChromaDB Client
chromadb_client = get_chromadb_client(embedding_model)

# Initializing History Aware Retriever
history_aware_retriever = get_history_aware_retriever(chromadb_client, embedding_model, v0_llm)

# Intializing RAG Chain v0
rag_chain_v0 = get_rag_chain_v0(v0_llm, history_aware_retriever) 

# Intializing RAG Chain v1
rag_chain_v1 = get_rag_chain_v1(v0_llm, history_aware_retriever) # Using gpt-4o-mini for v1 chain as well Change this after

# Grade output schema
class ValidationGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    predicted_validation: Annotated[
        bool, ..., "True if the response meets the validate criteria, False otherwise"
    ]


# Grade prompt
validate_instructions = """You are an assistant that grades if the response of another assistant is valid based on the given previous chat history and question. 

You will be given a PREVIOUS CHAT HISTORY, QUESTION and a RESPONSE.

Here is how to determine if the RESPONSE is NOT valid:
1. If the QUESTION is not relevant to the PREVIOUS CHAT HISTORY and not related to the subject Adam, then the RESPONSE MUST ignore the QUESTION and describe that the QUESTION is unrelated to Adam.

Here is how to determine if the RESPONSE is valid:
2. If the QUESTION is either relevant to the PREVIOUS CHAT HISTORY or related to the subject Adam, then the RESPONSE MUST address the QUESTION.

Predicted Validation:
A predicted validation value of False means that the RESPONSE is not valid based on the (1).
A predicted validation value of True means that the RESPONSE is valid based on the (2)

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Get the Langsmith example to test
example = client.read_example(example_id="05e22b3e-b587-4b16-82d8-2a2f48504b9a")



# Grader LLM, using gpt-4.1 with structured output for behavior grading
behavior_llm = ChatOpenAI(model="gpt-4.1", temperature=0).with_structured_output(
    ValidationGrade, method="json_schema", strict=True
)


def get_prediction(inputs: dict):
    """Get the prediction from the behavior LLM."""
    # Extracting inputs from the dataset_inputs
    prev_chat_history = inputs['previous_chat_history']
    question = inputs['last_question']

    # Call RAG chain to get the response of the question
    rag_output = rag_chain_v1.invoke({'input':question, 
                               'chat_history':prev_chat_history})
    
    response = rag_output['answer']
    
    answer = f"PREVIOUS CHAT HISTORY: {prev_chat_history}\nQUESTION: {question}\nRESPONSE: {response}"
    
    pred_outputs = behavior_llm.invoke(
        [
            {"role": "system", "content": validate_instructions},
            {"role": "user", "content": answer},
        ]
    )
    
    return pred_outputs['predicted_validation']

def accuracy(outputs: dict, reference_outputs: dict) -> dict:
    # Row-level evaluator for accuracy.
    pred = outputs['output']
    expected = reference_outputs["valid_question_to_answer"]
    return {"score": expected == pred}

experiment_results = client.evaluate(
    get_prediction,
    data=client.list_examples(dataset_name="Behavior Dataset w/ Chat History", splits=["balanced_split"]),
    evaluators=[accuracy],
    experiment_prefix="rag-behavior-eval-v1-test-w-gpt-4o-mini",
    metadata={"split": "balanced_split", "rag_chain_version": "v1", "llm_model": "gpt-4o-mini", "embedding_model": "text-embedding-3-small"}
)