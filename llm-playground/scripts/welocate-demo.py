import streamlit as st
from os import listdir
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

STORAGE_PATH = "/Users/mohamedadelabdelhady/workspace/kaggle-sandbox/llm-playground/welocate-storage"


st.title("💬 Welocate Enhanced Chat: ")

st.text("Using all documents and information available online to generate accurate responses.")


def create_query_engine(index):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT,)

    # assemble the query engine
    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer,
                                        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])
    return query_engine


def load_index():
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_PATH)
    index = load_index_from_storage(storage_context)
    query_engine = create_query_engine(index)
    st.session_state.query_engine = query_engine
    st.toast('Query Engine created successfully!', icon='✅')



@lru_cache
def get_response(prompt):
    return st.session_state.query_engine.query(prompt)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_engine" not in st.session_state:
    load_index()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_response(prompt)

    response_message = ""

    if response.source_nodes:
        response_message += response.response
        response_message += "\n\n"
        response_message += """
| Source | Similarity Score |
|-----------|-------|
"""
        seen_access_links = {}
        for node in response.source_nodes:

            metadata = node.to_dict()["node"]['metadata']
            if metadata["access_link"] in seen_access_links:
                continue
            else:
                seen_access_links[metadata["access_link"]] = 1
            response_message += f"| {metadata['access_link']} | {round(node.score, 2)} |\n"
    else:
        response_message = "No relevant documents found."

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_message)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_message})