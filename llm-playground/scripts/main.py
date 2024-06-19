import streamlit as st
from os import listdir
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

DATA_PATH = "/Users/mohamedadelabdelhady/workspace/kaggle-sandbox/llm-playground/data"
STORAGE_PATH = "/Users/mohamedadelabdelhady/workspace/kaggle-sandbox/llm-playground/storage"


documents = SimpleDirectoryReader(DATA_PATH).load_data()
files = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
display_files = ''
for file in files:
    display_files += "- " + file.title() + "\n"


st.title("Retrieval Augmented Generation App")
st.write("Found the following files: ")
st.markdown(display_files)


# creating text nodes
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)
st.write(f"We have created: {len(nodes)} nodes")

def create_query_engine(index):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)

    # assemble the query engine
    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer,
                                        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.8)])
    return query_engine


def generate_embeddings():
    print("generating embeddings")
    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(persist_dir=STORAGE_PATH)
    query_engine = create_query_engine(index)
    st.session_state.query_engine = query_engine
    st.toast('Query Engine created successfully!', icon='✅')


def load_index():
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_PATH)
    index = load_index_from_storage(storage_context)
    query_engine = create_query_engine(index)
    st.session_state.query_engine = query_engine
    st.toast('Query Engine created successfully!', icon='✅')


st.button("Index the data", on_click=generate_embeddings)
st.button("Load index", on_click=load_index)

# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    response = st.session_state.query_engine.query(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
