# Ingest documents from multiple sources
import uuid
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader

documents = SimpleDirectoryReader("./data").load_data()
documents += [
    Document(
        text="The simplest way to store your indexed data is to use the built-in .persist() method of every Index, "
        "which writes all the data to disk at the location specified. This works for any type of index.",
        doc_id=str(uuid.uuid4()),
        metadata={
            "foo": "bar",
            "category": "documentation",
        },  # metadata will propagate to the nodes
        excluded_llm_metadata_keys=[
            "foo"
        ],  # some keys could be excluded from the text_content()
    )
]
documents += SimpleWebPageReader(html_to_text=True).load_data(
    urls=["https://docs.pinecone.io/home"]
)


# Creating nodes/chunks
from llama_index.core.node_parser import (
    SimpleNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
    TextSplitter,
)
from llama_index.core.ingestion import IngestionPipeline

# creating text nodes
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)
print(len(nodes))

# using a different splitter -> this will create different number of nodes
text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
pipeline = IngestionPipeline(transformations=[text_splitter])
nodes = pipeline.run(documents=documents)
print(len(nodes))


# creating nodes with automatic metadata extraction
# here we need to start making API requests to an LLM
# you NEED to set the OPENAI_API_KEY env variable
import nest_asyncio

nest_asyncio.apply()

from llama_index.core.extractors import TitleExtractor, KeywordExtractor
from llama_index.core.schema import MetadataMode
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

enrich_metadata_pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
        TitleExtractor(llm=llm, metadata_mode=MetadataMode.EMBED),
        KeywordExtractor(llm=llm, metadata_mode=MetadataMode.EMBED),
    ]
)
nodes = enrich_metadata_pipeline.run(documents=documents)


from llama_index.core import VectorStoreIndex

# On a high-level, index can be created from documents directly, this will use a default node parser
# index = VectorStoreIndex.from_documents(documents, show_progress=True)

index = VectorStoreIndex(nodes, show_progress=True)


# this will overwrite all the json files in storage
index.storage_context.persist(persist_dir="./storage")
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)


from llama_index.core import (
    VectorStoreIndex,
    get_response_synthesizer,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    KeywordNodePostprocessor,
)
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.response.pprint_utils import pprint_source_node

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)

# assemble the query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.8)],
)


response = query_engine.query("Where did paul graham study?")
print(response)

for node in response.source_nodes:
    pprint_source_node(node)
