from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, CSVLoader, Docx2txtLoader
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from itertools import combinations
import numpy as np
from langchain.memory import ConversationSummaryBufferMemory,ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain, RetrievalQA, ConversationalRetrievalChain, RetrievalQAWithSourcesChain

from langchain_community.llms import HuggingFaceHub, HuggingFaceEndpoint
import gradio as gr

import os
from dotenv import load_dotenv
# from llama.api import HuggingFaceEndpoint
load_dotenv()


LOCAL_VECTOR_STORE_DIR = Path('./data')


def langchain_document_loader(TMP_DIR):
    """
    Load documents from the temporary directory (TMP_DIR). 
    Files can be in txt, pdf, CSV or docx format.
    """

    documents = []

    # txt_loader = DirectoryLoader(
    #     TMP_DIR.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    # )
    # documents.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())

    # csv_loader = DirectoryLoader(
    #     TMP_DIR.as_posix(), glob="**/*.csv", loader_cls=CSVLoader, show_progress=True,
    #     loader_kwargs={"encoding":"utf8"}
    # )
    # documents.extend(csv_loader.load())

    doc_loader = DirectoryLoader(
        TMP_DIR.as_posix(),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )
    documents.extend(doc_loader.load())
    return documents


directory_path = 'course reviews'
TMP_DIR = Path(directory_path)
documents = langchain_document_loader(TMP_DIR)

def select_embedding_model():
    embedding = OllamaEmbeddings(model='nomic-embed-text')
    return embedding

embeddings_nomic = select_embedding_model()

def create_vectorstore(embeddings,documents,vectorstore_name):
    """Create a Chroma vector database."""
    persist_directory = (LOCAL_VECTOR_STORE_DIR.as_posix() + "/" + vectorstore_name)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_store


create_vectorstores = True # change to True to create vectorstores

if create_vectorstores:
    vector_store_nomic = create_vectorstore(embeddings_nomic,documents,"vector_store_nomic")
    print("Vector store created")
    print("")
    

    
vector_store_nomic = Chroma(persist_directory = LOCAL_VECTOR_STORE_DIR.as_posix() + "/vector_store_nomic", 
                            embedding_function=embeddings_nomic)
print("vector_store_Ollama:",vector_store_nomic._collection.count(),"chunks.")

        
def Vectorstore_backed_retriever(vectorstore,search_type="similarity",k=4,score_threshold=None):
    """create a vectorsore-backed retriever
    Parameters: 
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold"
        k: number of documents to return (Default: 4) 
        score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
    """
    search_kwargs={}
    if k is not None:
        search_kwargs['k'] = k
    if score_threshold is not None:
        search_kwargs['score_threshold'] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    return retriever


# Similarity search
retriever = Vectorstore_backed_retriever(vector_store_nomic,search_type="similarity",k=4)



def instantiate_LLM(api_key,temperature=0.5,top_p=0.95,model_name=None):
    """Instantiate LLM in Langchain.
    Parameters:
        LLM_provider (str): the LLM provider; in ["OpenAI","Google","HuggingFace"]
        model_name (str): in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4-turbo-preview", 
            "gemini-pro", "mistralai/Mistral-7B-Instruct-v0.2"].            
        api_key (str): google_api_key or openai_api_key or huggingfacehub_api_token 
        temperature (float): Range: 0.0 - 1.0; default = 0.5
        top_p (float): : Range: 0.0 - 1.0; default = 1.
    """
    
  
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",          # working
        # repo_id="apple/OpenELM-3B-Instruct",                 # erros: remote trust something
        # repo_id="meta-llama/Meta-Llama-3-8B-Instruct",       # Takes too long
        # repo_id="mistralai/Mixtral-8x22B-Instruct-v0.1",     # RAM insufficient
        # repo_id=model_name,
        huggingfacehub_api_token=api_key,
        # model_kwargs={
        #     "temperature":temperature,
        #     "top_p": top_p,
        #     "do_sample": True,
        #     "max_new_tokens":1024
        # },
        # model_kwargs={"stop": "Human:"},
        
        stop_sequences = ["Human:"],
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        max_new_tokens=1024,
        trust_remote_code=True
    )
    return llm

# get the API key from .env file
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
llm = instantiate_LLM(api_key=HUGGING_FACE_API_KEY)



def create_memory():
    """Creates a ConversationSummaryBufferMemory for our model
    Creates a ConversationBufferWindowMemory for our models."""
    
    memory = ConversationBufferWindowMemory(
        memory_key="history",
        input_key="question",
        return_messages=True,
        k=3
    )

    return memory

memory = create_memory()


memory.save_context(
    {"question": "What can you do?"},
    {"output": "I can answer queries based on the past reviews and course outlines of various courses offered at LUMS."}
)

context_qa = """
You are a professional chatbot assistant for helping students at LUMS regarding course selection.

Please follow the following rules:

1. Answer the question in your own words from the context given to you.
2. If you don't know the answer, don't try to make up an answer.
3. If you don't have a course's review or outline, just say that you do not know about this course.
4. If a user enters a course code (e.g. ECON100 or CS370), match it with reviews with that course code. If the user enters a course name (e.g. Introduction to Economics or Database Systems), match it with reviews with that course name.
5. If you do not have information of a course, do not make up a course or suggest courses from universities other than LUMS.

Context: {context}

You are having a converation with a student at LUMS.

Chat History: {history}

Human: {question}

Assistant:
"""

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=context_qa
)


qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    verbose=False,
    return_source_documents=False,
    chain_type_kwargs={
        "prompt": prompt,
        "memory": memory
    },
)


# def rag_model(query):
#     # Your RAG model code here
#     result = qa({'query': query})

#     # Extract the answer from the result
#     answer = result['result']

#     # Extract the response from the answer (if needed)
#     # response = answer.split('Assistant123:')[-1]

#     # return response
#     return answer


# #This is for Gradio interface
# iface = gr.Interface(fn=rag_model, inputs="text", outputs="text", title="RAGs to Riches", theme=gr.themes.Soft(), description="This is a RAG model that can answer queries based on the past reviews and course outlines of various courses offered at LUMS.")
# iface.launch(share=True)


# Global list to store chat history
chat_history = []

def rag_model(query):
    # Your RAG model code here
    result = qa({'query': query})

    # Extract the answer from the result
    answer = result['result']

    # Append the query and answer to the chat history
    chat_history.append(f'User: {query}\nAssistant: {answer}\n')

    # Join the chat history into a string
    chat_string = '\n'.join(chat_history)

    return chat_string

# This is for Gradio interface
iface = gr.Interface(fn=rag_model, inputs="text", outputs="text", title="RAGs to Riches", theme=gr.themes.Soft(), description="This is a RAG model that can answer queries based on the past reviews and course outlines of various courses offered at LUMS.")
iface.launch(share=True)

#This is for Streamlit interface
# import streamlit as st

# st.title("RAG Model")

# # Text input box for user query
# query = st.text_input("Enter your query:")

# # Button to trigger model inference
# if st.button("Generate Response"):
#     # Call your RAG model function
#     response = rag_model(query)
#     st.write("Response:", response)