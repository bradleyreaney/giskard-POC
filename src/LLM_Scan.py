import os
import giskard
import requests
import pandas as pd
from datetime import datetime
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

# Reference Docs - https://docs.giskard.ai/en/latest/open_source/scan/scan_llm/index.html

nimble_api_url = os.environ["NIMBLE_API_URL"]
nimble_api_key = os.environ["NIMBLE_API_KEY"]
output_path = os.environ["OUTPUT_PATH"]

llm_description = "This is an internal RAG based LLM Chatbot. It will be used to answer low value questions staff may have around HR polices. It will have access to our internal HR policy documentation and will use these documents alone for it's context when answering questions. Please note that we're a UK based company so questions will be based on UK terminology. For instance we'd say holiday or annual leave instead of vacation."

def log_helper(message, logEntry):
    print(f"INFO - {datetime.now()} - {message} - {logEntry}")

def model_predict(df: pd.DataFrame):
    def llm_api(input_questions):
        log_helper("LLM Called with the following question", input_questions)
        response = requests.get(f"{nimble_api_url}/prod/answer?question={quote(input_questions)}", headers={"x-api-key": nimble_api_key}, timeout=30)
        log_helper("LLM Response", response.content.decode('utf-8'))
        return response.content.decode('utf-8')
    return [llm_api(question) for question in df["question"].values]

giskard_model = giskard.Model(
    model=model_predict,
    model_type="text_generation",
    name="Internal AI LLM Chatbot",
    description=llm_description,
    feature_names=["question"],
)

scan_results = giskard.scan(giskard_model)

isExist = os.path.exists(output_path)

if not isExist:
    os.makedirs(output_path)

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")

scan_results.to_html(f'{output_path}/scan_report_{dt_string}.html')


