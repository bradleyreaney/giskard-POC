import os

import openai
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
# from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from giskard import Model, scan, GiskardClient

from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


# Reference Docs - https://docs.giskard.ai/en/latest/reference/notebooks/LLM_QA_Documentation.html

# Set the OpenAI API Key environment variable.
# OPENAI_API_KEY = ""
# openai.api_key = os.environ['OPENAI_API_KEY']
# os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
openai.api_key = os.environ['OPENAI_API_KEY']


# Display options.
pd.set_option("display.max_colwidth", None)

# DATA_URL = "https://drive.google.com/file/d/1bXvNo_k-vFA9Wmjbu5eq5S_J8J6wElMp/view?usp=drive_link"
PDF_LOCATION = "exampleHRDoc/Generic-HR-Policy.pdf"

LLM_NAME = "text-ada-001"

def get_context_storage() -> FAISS:
    """Initialize a vector storage with the context."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # docs = OnlinePDFLoader(DATA_URL).load_and_split(text_splitter)
    docs = PyPDFLoader(PDF_LOCATION).load_and_split(text_splitter)
    db = FAISS.from_documents(docs, embeddings)
    return db


# Create the chain.
llm = OpenAI(
    openai_api_key=OPENAI_API_KEY,
    request_timeout=20,
    max_retries=100,
    temperature=0.2,
    model_name=LLM_NAME,
)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=get_context_storage().as_retriever())

def save_local(persist_directory):
    get_context_storage().save_local(persist_directory)


def load_retriever(persist_directory):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(persist_directory, embeddings)
    return vectorstore.as_retriever()


giskard_model = Model(
    model=qa,  # A prediction function that encapsulates all the data pre-processing steps and that could be executed with the dataset used by the scan.
    model_type='text_generation',  # Either regression, classification or text_generation.
    # name="GNU sed, a stream editor", # Optional.
    description="A model that can answer any information found inside the HR policy document.",  # Is used to generate prompts during the scan.
    feature_names=['query'],  # Default: all columns of your dataset.
    loader_fn=load_retriever,
    save_db=save_local
)

results = scan(giskard_model)

# Generate the output path if it doesn't exist
isExist = os.path.exists("../output")

if not isExist:
    os.makedirs("../output")

now = datetime.now()

dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")

# Save it to a file
# results.to_html("../output/scan_report.html")
results.to_html(f'../output/scan_report_{dt_string}.html')