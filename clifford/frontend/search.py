"""Streamlit app using LangChain + claude.

Examples:
    $ streamlit run clifford/frontend/search.py 
"""
import os
import streamlit as st
from joblib import Memory
import pandas as pd

from streamlit_chat import message

from clifford.engine.claude_llm import ClaudeLLM

LOCATION = "./cachedir"
MEMORY = Memory(LOCATION, verbose=0)

# @MEMORY.cache
def generate_response(human_input):
    """Prompt LangChain for a chat completion response."""
    chain = st.session_state["chain"]
    response = chain.predict(human_input=human_input)
    st.session_state["chain"] = chain

    return response

llm_runner = ClaudeLLM()

# Initialize session state variables
if "chain" not in st.session_state:
    st.session_state["chain"] = llm_runner.get_chain()

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

# Containers
# response_container = st.container()
# chat_container = st.container()


# with chat_container:
#     with st.form(key="my_form", clear_on_submit=True):
#         user_input = st.text_area("You:", key="input", height=100)
#         submit_button = st.form_submit_button(label="Send")

#     if submit_button and user_input:
#         output = generate_response(user_input).strip()
#         st.session_state["past"].append(user_input)
#         st.session_state["generated"].append(output)


df = pd.DataFrame({
    'source': [
        'tim.belden@someorg.com',
        'phillip.screwdriver@someorg.com',
    ],
    'date': [
        '14 May 2001',
        '23 Oct 2001',
    ],
    'text_summary': [
        'Related financial forecasts.',
        'Details of staff and positions in the office that made the forecasts.'
    ]
})
search_term = st.text_input(
    "Enter Search Term:"
)

if search_term:
    df = df[df['col1']].str.contains(search_term)

st.table(df)
# INITIAL_MESSAGE = """
# 🧞‍♂ I am a mind reader with magical abilities! 🔮
# 🤔 Give me a category e.g. animal, or a famous person.
# 💬 I will ask questions and guess what you are thinking of!
# """

# with response_container:
#     message(INITIAL_MESSAGE)
#     if st.session_state["generated"]:
#         for i in range(len(st.session_state["generated"])):
#             message(st.session_state["past"][i], is_user=True, key=f"{i}_user")
#             message(st.session_state["generated"][i], key=f"{i}")
