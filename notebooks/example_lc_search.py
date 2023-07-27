# %%
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, JSONLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Our own modules
from example_lc_enrich import LLMWrapper


EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
DB_DIR = "data/db"
SOURCE_DIR = "texts"


# Set smaller number of tokens to limit how long it spends hallucinating if it does
llm = LLMWrapper(max_new_tokens=20)


# %%
# LOAD DOCS INCLUDING ADDING METADATA
# HIDDEN DEPENDENCY, REQUIRES PACKAGE jq FOR READING JSON
# HIDDEN BUG, UNLESS YOU SPECIFY content_key, medatada doesn't load

def metadata_func(record: dict, metadata: dict) -> dict:
    """ Helper, instructs on how to fetch metadata """
    metadata["title"] = record.get("title")
    metadata["date"] = record.get("date")
    metadata["people"] = llm.query(
        question="List all the people mentioned in this document.",
        context=record.get("text")[:300]
    )
    return metadata

json_loader_kwargs = {"jq_schema": ".",
                      "content_key": "text",
                      "metadata_func": metadata_func}

loader = DirectoryLoader(
    SOURCE_DIR,
    glob="**/*.json",
    loader_cls=JSONLoader,
    loader_kwargs=json_loader_kwargs,
    show_progress=True)

# loader=TextLoader("texts/simples.txt")
docs = loader.load()

len(docs)

# %%
# OPTIONAL, SPLIT DOCUMENT
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
        )

docs = text_splitter.split_documents(docs)

# %%
# CREATE THE DOCUMENT DATABASE WITH EMBEDDINGS
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

if not os.path.exists(DB_DIR):
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_DIR)
else:
    print("Vector store already exists")
    db = FAISS.load_local(DB_DIR, embeddings)

# %%
# SEARCH FOR A RELEVANT DOCUMENT
# Lower "distance" == greater similarity
query = "Who referred to the weather as Global Boiling?"
result = db.similarity_search_with_score(query=query, k=5)

