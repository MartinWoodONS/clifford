import json
import os
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridUpdateMode, ColumnsAutoSizeMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

from clifford.engine.claude_llm import ClaudeLLM

llm_runner = ClaudeLLM()
summarised_emails = 'data/summarised/example_emails.json'
# Initialize session state variables
if "summary_chain" not in st.session_state:
    st.session_state["summary_chain"] = llm_runner.get_chain("summarise")

def generate_summary(email):
    """Prompt LangChain for a chat completion response."""
    summary_chain = st.session_state["summary_chain"]
    response = summary_chain.predict(email=email)
    st.session_state["summary_chain"] = summary_chain

    return response

if st.button('summarise emails'):
    with open('data/raw/email.txt', 'r') as f:
        data = f.read()
    summary = generate_summary(data)
    if not os.path.isfile(summarised_emails):
        with open(summarised_emails,'w') as f:
            json.dump([
                {
                    "summerisation": summary,
                    ## TODO fill in the rest of the data
                    "tags": {
                        "topic" : "Charity",
                        "urgency" : "Green",
                        "action": 0,
                        "response": 0
                    },
                     "header": "Email header 1"
                }
            ], f)
    else:
        with open(summarised_emails,'r') as f:
            current_data = json.load(f)
        current_data.append(
            {
                "summerisation": summary,
                ## TODO fill in the rest of the data
                "tags": {
                    "topic" : "Charity",
                    "urgency" : "Green",
                    "action": 0,
                    "response": 0
                },
                    "header": "Email header 1"
            }
        )
        with open(summarised_emails,'w') as f:
            json.dump(current_data , f)

        st.write(f"Data was appended to {summarised_emails}")

def display_email_summarisation_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    df = pd.json_normalize(data)
    
    gd = GridOptionsBuilder.from_dataframe(df)
    gd.configure_selection(selection_mode='multiple', use_checkbox=True)
    gd.configure_columns("summerisation",wrapText = True)
    gd.configure_columns("summerisation",autoHeight = True)
    gridoptions = gd.build()

    MIN_HEIGHT = 50
    MAX_HEIGHT = 800
    ROW_HEIGHT = 60

    MIN_WIDTH = 50
    MAX_WIDTH = 800
    ROW_WIDTH = 60

    grid_table = AgGrid(
        df, 
        height=min(MIN_HEIGHT + len(df) * ROW_HEIGHT, MAX_HEIGHT), 
        width=min(MIN_WIDTH + df.shape[1] * MAX_WIDTH, ROW_WIDTH), 
        gridOptions=gridoptions,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW
        )

    st.write('## Selected')
    selected_row = grid_table["selected_rows"]
    
    if st.button('Send selected'):
        for row in selected_row:
            print(row)


display_email_summarisation_from_json(summarised_emails)
