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

from llama_index.llms.openai import OpenAI

import os
from pinecone import Pinecone, ServerlessSpec, PodSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.response.pprint_utils import pprint_source_node
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)


st.title("💬 Find your perfect wine pairing: ")

st.text("Indexing all product information found at https://wijnopdronk.nl/")
pinecone_index = pc.Index("wine-index")

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)
custom_system_prompt = """You are a knowledgeable sommelier assistant. Your task is to provide 
expert advice on wine pairings based on the given context. Always strive to give specific 
recommendations and explain your choices. If no context is provided please reply by saying that no context is provided.
"""

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1, system_prompt=custom_system_prompt)
query_engine = index.as_query_engine(llm=llm)
st.session_state.query_engine = query_engine
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

@lru_cache
def get_response(prompt):
    return st.session_state.query_engine.query(prompt)


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
| Source |
|-----------|
"""
        seen_links = {}
        for node in response.source_nodes:
            link = node.to_dict()["node"]["relationships"]["1"]["node_id"]
            if link in seen_links:
                continue
            else:
                seen_links[link] = 1
            response_message += f"| {link} | {round(node.score, 2)} |\n"
    else:
        response_message = "No relevant documents found."

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_message)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_message})