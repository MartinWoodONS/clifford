from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, ServiceContext, set_global_service_context

from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    KeywordExtractor,
    MetadataFeatureExtractor,
)
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter

# Apply default settings to use across funcs
llm = OpenAI(temperature=0, model="text-davinci-003", max_tokens=512)
service_context = ServiceContext.from_defaults(embed_model="local", llm=llm)
set_global_service_context(service_context)


# TODO needs to be configured to meet our use cases
# Customer metadata extractor to be called by node parser
class CustomExtractor(MetadataFeatureExtractor):
    def extract(self, nodes):
        metadata_list = [
            {
                "custom": node.metadata["document_title"]
                + "\n"
                + node.metadata["excerpt_keywords"]
            }
            for node in nodes
        ]
        return metadata_list

def parse_nodes_from_docs(docs: list):
    """
    Parse llama_index Documents into nodes
    Apply metadata extractor funcs to generate metadata from content
    """

    # Configure llm and text_splitter (can be altered to use clause or similar)
    # llm = OpenAI(temperature=0, model="text-davinci-003", max_tokens=512)
    text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)

    # Define which extractor will be called and args to pass
    metadata_extractor = MetadataExtractor(
        extractors=[
            QuestionsAnsweredExtractor(questions=1),
            SummaryExtractor(summaries=["prev", "self"]),
            KeywordExtractor(keywords=3),
            # CustomExtractor()
        ],
    )

    # Config parser with size of each node and metadata extraction funcs
    node_parser = SimpleNodeParser(
        text_splitter=text_splitter,
        metadata_extractor=metadata_extractor,
    )

    # Create nodes from docs
    nodes = node_parser.get_nodes_from_documents(docs, show_progress=True)
    return nodes

def create_index(nodes):
    # Create vector store index from nodes
    index = VectorStoreIndex(nodes, show_progress=True)
    return index

def save_index(index):
    # Persist index to disk so it doesn't need to be re-built
    index.storage_context.persist(persist_dir="../storage")

def load_index():
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="../storage")
    # load index
    index = load_index_from_storage(storage_context)