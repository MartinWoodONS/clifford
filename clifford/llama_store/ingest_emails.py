import re

import pandas as pd

from pathlib import Path
from llama_index import Document

def cleanse_txt(text):
    """
    Func to perform simple cleansing steps on incoming text data
    """
    if(type(text) != str):
        text = ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_emails(limit=None):
    """
    Load email.csv file from data folder into DataFrame
    """
    # TODO remove hardcoding
    email_path = Path().absolute().parent.parent.joinpath('data', 'raw', 'emails.csv')

    df = pd.read_csv(email_path)
    # limit rows when testing
    if limit is None:
        return df
    else:
        return df.head(limit)
    

def rows_to_documents(df):
    """
    Loop over DataFrame rows and push fields into LlamaIndex Documents
    Uses email content as text and other fields for metadata
    """
    all_docs = []
    # Loop over df rows and put info into llamaindex Document
    for index, row in df.iterrows():
        doc = Document(
            text=cleanse_txt(row['content']),
            metadata={
                "msg_id" : row['Message-ID'],
                "date" : row['Date'],
                "from" : row['From'],
                "to" : row['To'],
                "subject" : cleanse_txt(row['Subject'])
            },
            excluded_llm_metadata_keys=['msg_id'],
            metadata_seperator="::",
            metadata_template="{key}=>{value}",
            text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
        )    
        all_docs.append(doc)
    return all_docs

def emails_to_documents(row_limit=20):
    """
    Load email data in and push it into LlamaIndex Documents
    """
    df = load_emails(limit=row_limit)
    docs = rows_to_documents(df)
    return docs