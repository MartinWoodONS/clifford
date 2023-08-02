"""Streamlit app using LlamaIndex.

Examples:
    $ streamlit run clifford/llama_store/retrieve_docs.py 
"""
import streamlit as st

from llama_index.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from create_vector_store import load_index

index = load_index(dir="clifford/storage")

def search_docstore(index, term):
    vi_retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
    return vi_retriever.retrieve(term)

search_term = st.text_input(
    "Enter Search Term:"
)

if search_term:
    ret_list = search_docstore(index=index, term=search_term)
    st.write(ret_list)