import streamlit as st
from os import listdir
import pandas as pd
from functools import lru_cache
from os.path import isfile, join
import random
import time
import uuid
import numpy as np
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex, get_response_synthesizer, StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.response.pprint_utils import pprint_source_node
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter, TokenTextSplitter, TextSplitter
from llama_index.core.ingestion import IngestionPipeline

DATA_PATH = "/Users/mohamedadelabdelhady/workspace/kaggle-sandbox/llm-playground/cv-data"
STORAGE_PATH = "/Users/mohamedadelabdelhady/workspace/kaggle-sandbox/llm-playground/cv-storage"


documents = SimpleDirectoryReader(DATA_PATH).load_data()
files = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
display_files = ''
for file in files:
    display_files += "- " + file.title() + "\n"


st.header('CV Search 🔎', divider='gray')

st.subheader("Find the right candidates!")

# st.text(f"Found {len(documents)} CVs")

@lru_cache
def get_response(prompt):
    result = []

    for i, doc in enumerate(documents):
        if prompt.lower() in doc.text.lower():
            result.append(doc.metadata['file_name'])
    df = pd.DataFrame(result, columns=["File Name"])

    return result, df.to_markdown()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []



# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("Search"):
    # Add user message to chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    result, response = get_response(prompt)


    response_message = response
    with st.chat_message("assistant"):
        st.write(f"Found {len(result)} CVs matching your search:")
        st.markdown(response_message)
    # st.write(response_message)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_message})